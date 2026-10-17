"""Plan step 11's baselines on the selection set (rlm_rl/baselines.py): no training, checkpoint 0 only.

Run inside WSL2 (Ubuntu) from the repo root, with Docker Desktop running:

    .venv-linux/bin/python scripts/baselines.py project step11-v1 --arms rlm,direct,rlm_medium     # $0: what's left, priced
    .venv-linux/bin/python scripts/baselines.py run step11-v1 --arms rlm,direct --live --ceiling 12
    .venv-linux/bin/python scripts/baselines.py run step11-v1 --arms rlm_medium --live --ceiling 12
    .venv-linux/bin/python scripts/baselines.py status step11-v1

Arms: `rlm` (base model as an RLM, prompt v2, thinking off; edit tasks twice), `direct` (the context read directly, read
tasks only) and `rlm_medium` (decision 11b: root thinking at medium). Arms given together run in one process, sharing the
limiter and the sub-call spacing. `--ceiling` is a hard stop over every arm and every run of this command: an episode
that could take spend over it never starts, and `run` refuses to start when its projection is already over it.

A live run spends W&B inference credit, so `run` needs --live or --dry-run. --dry-run uses the fake model server and the
in-process sandboxes (tests/dryrun.py), with no W&B call and no credit, and writes to results/dry/; --real-edit runs its
edit episodes in the real edit sandbox and grader (Docker). The W&B key is read from .env and never printed.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import baselines, config, evals  # noqa: E402

RESULTS = ROOT / "results"
DRY_RESULTS = RESULTS / "dry"


def results_dir(args) -> Path:
    if getattr(args, "results", None):
        return Path(args.results)
    return DRY_RESULTS if getattr(args, "dry_run", False) else RESULTS


def arm_list(text: str) -> list[str]:
    names = [a.strip() for a in text.split(",") if a.strip()]
    unknown = [a for a in names if a not in baselines.ARMS]
    if unknown or not names:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {list(baselines.ARMS)}")
    return names


def selection_tasks():
    tasks, samples = evals.eval_set(baselines.EVAL_SET)
    assert samples == 1
    return tasks


def print_projection(spent: float, ceiling: float, proj: dict) -> None:
    print(f"spent so far ${spent:.2f}; ceiling ${ceiling:g}")
    for name, arm in proj["arms"].items():
        parts = [f"{k} {v['todo']} x ${v['per_episode_usd']:.3f}{'' if v['measured'] else ' (plan estimate)'}"
                 for k, v in arm["kinds"].items() if v["todo"]]
        print(f"  {name}: {arm['done']} done; to run: {', '.join(parts) or 'nothing'} = ${arm['todo_usd']:.2f}")
    print(f"projected total ${spent + proj['todo_usd']:.2f}")


class DirectOracle:
    """The dry run's answers to direct-reading prompts, which the fake server serves as plain calls: the gold answer
    with the oracle's probability, else a near miss."""

    def __init__(self, oracle):
        self.oracle = oracle
        self.gold = {}
        self.fallback = oracle.answer_for

    def record(self, tasks):
        for t in tasks:
            if not t.family.startswith("edit_"):
                self.gold[baselines.direct_messages(t.question, t.context)[1]["content"]] = (t.answer, t.level)
        return tasks

    def answer_for(self, model, prompt):
        if prompt not in self.gold:
            return self.fallback(model, prompt)
        answer, level = self.gold[prompt]
        value = answer if self.oracle.rng.random() < self.oracle.p_correct(0, level) else self.oracle.wrong(answer)
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        return f"Counting through the text.\nFINAL ANSWER: {value}"


async def run_dry(args, arms: list[str]) -> dict:
    from tests.dryrun import DryRun

    results = results_dir(args)
    async with DryRun(results, seed=args.seed, persist=True, real_edit=args.real_edit) as dry:
        if args.latency:
            dry.server.latency = (args.latency * 0.5, args.latency * 1.5)
        direct = DirectOracle(dry.oracle)
        dry.server.answer_for = direct.answer_for
        tasks = direct.record(dry.oracle.eval_set(baselines.EVAL_SET)[0])
        await dry.backend.open(args.run)
        base = dry.backend.inference_name(args.run, 0)
        kwargs = {}
        if not args.real_edit:
            from tests.fakes import FakeEditSandbox, FakeSandbox, local_grade

            kwargs = {"sandbox_factory": FakeSandbox,
                      "sandbox_kwargs": {"edit_sandbox_factory": FakeEditSandbox, "grade": local_grade}}
        else:
            from tests.fakes import FakeSandbox

            kwargs = {"sandbox_factory": FakeSandbox}
        from rlm_rl.client import Pacer

        run = baselines.BaselineRun(args.run, results, openai_client=dry.backend.openai_client(), base_model=base,
                                    ceiling=args.ceiling, pacer=Pacer(), **kwargs)
        return await go(run, tasks, arms, args)


async def run_live(args, arms: list[str]) -> dict:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY; never printed
    from rlm_rl.backend import ArtBackend

    tasks = selection_tasks()
    backend = ArtBackend(project=config.PROJECT, base_model=config.BASE_MODEL)
    try:
        await backend.open(args.run)  # a model of its own, so its step 0 is the base model (backend.py)
        base = backend.inference_name(args.run, 0)
        run = baselines.BaselineRun(args.run, results_dir(args), openai_client=backend.openai_client(),
                                    base_model=base, ceiling=args.ceiling)
        return await go(run, tasks, arms, args)
    finally:
        await backend.close()


async def go(run: baselines.BaselineRun, tasks, arms: list[str], args) -> dict:
    proj = baselines.project(run.run_dir, tasks, [baselines.ARMS[a] for a in arms])
    spent = run.budget.spent()
    print_projection(spent, run.ceiling, proj)
    if spent + proj["todo_usd"] > run.ceiling and not args.allow_partial:
        raise SystemExit(f"not started: the projection (${spent + proj['todo_usd']:.2f}) is over the ${run.ceiling:g} "
                         "ceiling. Run fewer arms, or pass --allow-partial to run until the ceiling stops it.")
    summary = await run.run(tasks, arms, retry_infra=args.retry_infra)
    if run.session["skipped_budget"]:
        print(f"STOPPED AT THE CEILING: {run.session['skipped_budget']} episode(s) were not started")
    return summary


def print_summary(summary: dict) -> None:
    spend = summary.get("spend") or {}
    print(f"{summary['run']} on {summary['eval_set']}: spent ${spend.get('total_usd') or 0:.2f}"
          f" of ${spend.get('ceiling_usd') or 0:g}")
    for name, arm in summary["arms"].items():
        if not arm.get("logged"):
            print(f"  {name}: nothing logged yet ({arm['planned']} planned)")
            continue
        acc, se = arm.get("accuracy"), arm.get("accuracy_se")
        acc_text = "n/a" if acc is None else f"{acc:.1%} ± {se:.1%}"
        print(f"  {name}: {arm['n']} scored of {arm['planned']} planned, accuracy {acc_text}, infra attempts "
              f"{arm['attempts']['infra']} ({arm['attempts']['infra_kinds']}), est ${arm['est_cost_usd']:.2f}")
        for kind, k in arm["kinds"].items():
            if k["n"]:
                print(f"    {kind}: {k['accuracy']['rate']:.1%} over {k['n']}, ${k['cost_mean_usd']:.3f} an episode,"
                      f" median {k['wall_median_s']:.0f} s, {k['turns_mean']:.1f} turns, {k['subcalls_mean']:.1f} sub-calls")
        if arm.get("edit"):
            e = arm["edit"]
            print(f"    edit turn cap: {e['turn_cap_hits']} hits ({e['turn_cap_unresolved_rate']:.0%} of edit episodes"
                  f" unresolved at the cap)")
            for tier, t in e["tiers"].items():
                print(f"      {tier}: {t['rate']:.0%} of {t['n']}, ${t['cost_mean_usd']:.3f}, {t['turns_mean']:.1f} turns")
    for name, c in (summary.get("comparisons") or {}).items():
        if c.get("diff") is not None:
            print(f"  {name}: {c['b']:.1%} vs {c['a']:.1%} over {c['tasks']} tasks, diff {c['diff']:+.1%}"
                  + (f" ± {c['se']:.1%}" if c.get("se") is not None else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("run")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="the base model on W&B serverless; spends inference credit")
    mode.add_argument("--dry-run", action="store_true", help="fake model server; no W&B, no credit")
    p.add_argument("--arms", required=True, help=f"comma-separated, from {list(baselines.ARMS)}")
    p.add_argument("--ceiling", type=float, required=True, help="hard inference ceiling over every arm, $")
    p.add_argument("--allow-partial", action="store_true", help="start even if the projection is over the ceiling")
    p.add_argument("--retry-infra", type=int, default=1, help="extra passes over infra episodes at the end")
    p.add_argument("--results", help="results folder (default results/, or results/dry/ for a dry run)")
    p.add_argument("--seed", type=int, default=0, help="dry run only: the fake policy's seed")
    p.add_argument("--latency", type=float, default=0.0, help="dry run only: mean seconds per fake request")
    p.add_argument("--real-edit", action="store_true", help="dry run only: edit episodes in Docker, not in process")
    for name in ("project", "status"):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.add_argument("--arms", default=",".join(baselines.ARMS))
        p.add_argument("--results")
        p.add_argument("--dry-run", action="store_true", help="a run under results/dry/")
        p.add_argument("--json", action="store_true")
    args = parser.parse_args()
    arms = arm_list(args.arms)

    if args.command in ("project", "status"):
        if args.dry_run:  # the dry run's eval seeds, never private/
            from rlm_rl import tasks as rtasks

            rtasks.PRIVATE_DIR = DRY_RESULTS / "private"
        tasks = selection_tasks()
        run_dir = results_dir(args) / args.run
        if args.command == "project":
            spent = float(baselines.read_spend(run_dir).get("total_usd") or 0.0)
            proj = baselines.project(run_dir, tasks, [baselines.ARMS[a] for a in arms])
            print(json.dumps(proj, indent=1) if args.json else "", end="")
            if not args.json:
                print_projection(spent, float(baselines.read_spend(run_dir).get("ceiling_usd") or 0.0), proj)
            return 0
        summary = baselines.write_summary(run_dir, tasks)
        print(json.dumps(summary, indent=1, default=str) if args.json else "", end="")
        if not args.json:
            print_summary(summary)
        return 0

    if args.real_edit and not args.dry_run:
        raise SystemExit("--real-edit is for dry runs; a live run always uses the real edit sandbox")
    summary = asyncio.run((run_dry if args.dry_run else run_live)(args, arms))
    print_summary(summary)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
