"""The step-12 pilot (documentation/step-12-multistep-plan.md step 3): does each new family and level need sub-calls?

Checkpoint 20 runs each pilot task twice with prompt v4 with sub-calls (`pilot_on`) and twice without (`pilot_off`:
the same prompt with a budget of 0, the RLM paper's no-sub-call ablation). The base model with sub-calls
(`pilot_on_c0`, one sample) is optional (`--arms`): it shows how much of the behaviour checkpoint 20 brings. A level
stays in training only if sub-calls clearly help there and checkpoint 20's pass rate with them lands inside the band
below. The defaults (levels 1 and 3, 3 tasks each, the two checkpoint-20 arms) fit the $8 the plan gives the pilot at
~$0.10 an episode. Run inside WSL2 from the repo root, with Docker Desktop running:

    .venv-linux/bin/python scripts/pilot.py build pilot-v4 --families research,aggregation,edit_spec,edit_bugs
    .venv-linux/bin/python scripts/pilot.py run pilot-v4 --live --ceiling 8
    .venv-linux/bin/python scripts/pilot.py summary pilot-v4

`run` spends W&B inference credit (the ceiling is a hard stop, as in scripts/baselines.py). The W&B key is read from
.env and never printed.
"""

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import baselines, config, tasks as rtasks  # noqa: E402

RESULTS = ROOT / "results"
ARMS = ("pilot_on", "pilot_off", "pilot_on_c0")
DEFAULT_ARMS = ("pilot_on", "pilot_off")
BAND = (0.2, 0.8)  # checkpoint 20's pass rate with sub-calls must land inside this
MIN_GAIN = 0.2  # and beat the no-sub-call arm by at least this much


def run_dir(name: str) -> Path:
    return RESULTS / name


def build(args) -> None:
    out = run_dir(args.run)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "tasks.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists; pass --force to draw new tasks")
    plan = []
    for family in args.families.split(","):
        for level in (int(x) for x in args.levels.split(",")):
            top = rtasks.generator(family).MAX_LEVEL
            if level > top:
                continue
            for i in range(args.tasks):
                plan.append({"family": family, "level": level, "seed": rtasks.new_seed(),
                             "task_id": f"pilot-{family}-L{level}-{i:02d}"})
    path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(f"{len(plan)} pilot tasks -> {path}")


def load_tasks(name: str) -> list:
    plan = json.loads((run_dir(name) / "tasks.json").read_text(encoding="utf-8"))
    return [rtasks.make_task(p["family"], p["level"], seed=p["seed"], task_id=p["task_id"]) for p in plan]


async def run_live(args) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY; never printed
    from rlm_rl.backend import ArtBackend

    state = json.loads((RESULTS / f"{args.source_run}.json").read_text(encoding="utf-8"))
    ckpt = state["checkpoints"][str(args.checkpoint)]
    model = next(b["model"] for b in state["branches"] if b["branch"] == ckpt["branch"])
    tasks = load_tasks(args.run)
    backend = ArtBackend(project=config.PROJECT, base_model=config.BASE_MODEL)
    try:
        # the base model is the source run's step 0: a freshly registered model's step 0 answered only HTTP 500s
        # for minutes (2026-10-18), while prod-v1's answered at once
        await backend.open(args.source_run)
        await backend.open(model)
        base = backend.inference_name(args.source_run, 0)
        models = {"ckpt20": backend.inference_name(model, ckpt["backend_step"])}
        run = baselines.BaselineRun(args.run, RESULTS, openai_client=backend.openai_client(), base_model=base,
                                    ceiling=args.ceiling, eval_set=args.run, models=models, max_edit=args.max_edit)
        arms = args.arms.split(",")
        proj = baselines.project(run.run_dir, tasks, [baselines.ARMS[a] for a in arms])
        print(f"projected ${proj['todo_usd']:.2f} more; spent ${run.budget.spent():.2f}; ceiling ${args.ceiling:g}")
        await run.run(tasks, arms, retry_infra=1)
    finally:
        await backend.close()


def summary(args) -> dict:
    out = run_dir(args.run)
    plan = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    cells: dict = {}
    for arm in ARMS:
        for rec in baselines.arm_log(out, arm).records():
            if rec.get("infra_error"):
                continue
            key = (rec["family"], rec["level"])
            row = cells.setdefault(key, {}).setdefault(arm, {"n": 0, "correct": 0, "subcalls": [], "usd": []})
            row["n"] += 1
            row["correct"] += bool(rec["correct"])
            row["subcalls"].append(rec.get("subcalls_used") or 0)
            row["usd"].append((rec.get("usage") or {}).get("est_cost_usd") or 0.0)
    table = []
    for (family, level), arms in sorted(cells.items()):
        line = {"family": family, "level": level}
        for arm, row in arms.items():
            line[arm] = {"n": row["n"], "pass": round(row["correct"] / row["n"], 3) if row["n"] else None,
                         "subcalls": round(statistics.mean(row["subcalls"]), 1) if row["subcalls"] else None,
                         "usd": round(statistics.mean(row["usd"]), 4) if row["usd"] else None}
        on, off = (line.get("pilot_on") or {}).get("pass"), (line.get("pilot_off") or {}).get("pass")
        line["keep"] = (on is not None and off is not None and BAND[0] <= on <= BAND[1] and on - off >= MIN_GAIN)
        table.append(line)
        print(f"{family:12} L{level}  " + "  ".join(
            f"{a}: {v['pass']} pass, {v['subcalls']} sub-calls, ${v['usd']}" for a, v in line.items()
            if isinstance(v, dict)) + f"  -> {'keep' if line['keep'] else 'drop'}")
    result = {"tasks": len(plan), "band": BAND, "min_gain": MIN_GAIN, "levels": table}
    (out / "summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("run")
    p.add_argument("--families", required=True)
    p.add_argument("--levels", default="1,3")
    p.add_argument("--tasks", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("run")
    p.add_argument("--live", action="store_true", required=True, help="spends W&B inference credit")
    p.add_argument("--ceiling", type=float, required=True, help="hard stop, $")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--source-run", default="prod-v1")
    p.add_argument("--checkpoint", type=int, default=20)
    p.add_argument("--max-edit", type=int, default=baselines.MAX_EDIT_AT_ONCE,
                   help="edit episodes at once (training runs ~20; the baselines' default is 8)")
    p = sub.add_parser("summary")
    p.add_argument("run")
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "run":
        asyncio.run(run_live(args))
    else:
        summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
