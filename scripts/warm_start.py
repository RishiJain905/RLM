"""The coding warm start for a production run (rlm_rl/warmstart.py; prod-v2, Rishi 2026-10-19).

Run inside WSL from the repo root, with Docker running, while the run is stopped between steps:

    .venv-linux/bin/python scripts/warm_start.py sample prod-v2 --budget 10 --live     # base model + teacher hint
    .venv-linux/bin/python scripts/warm_start.py report prod-v2                        # what was sampled and kept
    .venv-linux/bin/python scripts/warm_start.py train prod-v2 --lr 2e-5 --live        # one SFT job, then resume

`sample` draws coding tasks from the training pools (skipping retired ones) at level 1, runs `--rollouts` episodes of
each on the run's base model with the teacher hint, the run's prompt and limits and the pacing nudge, in rounds of
`--tasks` tasks, until the next round would pass `--budget` dollars. Episodes go to results/<run>/warmstart/samples.jsonl
(local only, never committed). `train` keeps the right, delegating episodes, turns each kept turn into one SFT sample
and runs TrainingRun.warm_start, which adds the sampling cost to the run's step spend. The W&B key is read from .env
and never printed.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import tasks as rtasks, training, warmstart  # noqa: E402
from rlm_rl.config import RunConfig  # noqa: E402
from rlm_rl.transcripts import TranscriptLog, episode_record  # noqa: E402

FAMILIES = ("edit_spec", "edit_bugs")


def samples_path(results: Path, run: str) -> Path:
    return results / run / "warmstart" / "samples.jsonl"


def records(results: Path, run: str) -> list[dict]:
    path = samples_path(results, run)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def report(results: Path, run: str) -> dict:
    out = warmstart.summary(records(results, run))
    for family, s in out.items():
        print(f"{family}: {s['sampled']} sampled, {s['right']} right, {s['kept']} kept (right and delegating), "
              f"${s['cost_usd']:.2f}")
    kept = [r for r in records(results, run) if warmstart.keep(r)]
    print(f"kept {len(kept)} episodes -> {sum(len(warmstart.sft_samples(r)) for r in kept)} SFT samples")
    return out


async def sample(args) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY; never printed
    from rlm_rl.backend import ArtBackend
    from rlm_rl.client import ModelClient
    from rlm_rl.env import run_groups

    results = Path(args.results)
    state = training.load_state(results, args.run)
    if state.get("status") == "running":
        sys.exit(f"{args.run} is running: stop it before sampling")
    cfg = RunConfig.from_dict(state["config"])
    base_name = cfg.base_inference or args.run
    backend = ArtBackend(project=cfg.project, base_model=cfg.base_model)
    log = TranscriptLog(samples_path(results, args.run))
    spent_before = sum(float((r.get("usage") or {}).get("est_cost_usd") or 0.0) for r in records(results, args.run))
    try:
        await backend.open(base_name)
        base = backend.inference_name(base_name, 0)
        client = ModelClient(backend.openai_client(), subcall_model=base, root_thinking=cfg.root_thinking)
        retired = rtasks.Retirement(state.get("retirement")).ids()
        families = [f for f in args.families.split(",") if f]
        round_cost, spent, n = None, spent_before, 0
        while spent + (round_cost or 0.0) <= args.budget:
            batch = [rtasks.make_task(families[(n + i) % len(families)], args.level, exclude=retired)
                     for i in range(args.tasks)]
            n += args.tasks
            mark = client.spent_usd()
            groups = await run_groups(batch, args.rollouts, client=client, root_model=base, run_id=f"{args.run}-warm",
                                      prompt_version=cfg.prompt_version, resample_broken=cfg.resample_broken,
                                      pace_nudge=True, system_extra=warmstart.TEACHER_EDIT)
            log.extend([episode_record(ep, split="warm", run=args.run, step=0, sample=s,
                                       key=f"warm/{ep.task_id}/{ep.episode_id}")
                        for group in groups for s, ep in enumerate(group)])
            round_cost = client.spent_usd() - mark
            spent += round_cost
            right = sum(bool(ep.correct) for g in groups for ep in g)
            print(f"round of {len(batch)} tasks x {args.rollouts}: {right} right, ${round_cost:.2f} "
                  f"(${spent:.2f} of ${args.budget:g})", flush=True)
    finally:
        await backend.close()
    report(results, args.run)


async def train(args) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from rlm_rl.backend import ArtBackend

    results = Path(args.results)
    rows = records(results, args.run)
    kept = [r for r in rows if warmstart.keep(r)]
    trajectories = warmstart.sft_trajectories(kept)
    if len(kept) < args.min_episodes:
        sys.exit(f"only {len(kept)} right, delegating episodes (need {args.min_episodes}): sample more first")
    cost = round(sum(float((r.get("usage") or {}).get("est_cost_usd") or 0.0) for r in rows), 4)
    state = training.load_state(results, args.run)
    backend = ArtBackend(project=state["config"]["project"], base_model=state["config"]["base_model"])
    try:
        run = training.TrainingRun(args.run, backend=backend, results=results)
        st = await run.warm_start(trajectories, learning_rate=args.lr, sampling_usd=cost,
                                  source=f"{len(kept)} of {len(rows)} sampled episodes",
                                  counts=warmstart.summary(rows))
    finally:
        await backend.close()
    print(st["notices"][-1]["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sample", "report", "train"):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.add_argument("--results", default=str(training.RESULTS))
        if name == "sample":
            p.add_argument("--budget", type=float, required=True, help="dollars of inference to spend at most")
            p.add_argument("--families", default=",".join(FAMILIES))
            p.add_argument("--level", type=int, default=1)
            p.add_argument("--tasks", type=int, default=6, help="tasks a round")
            p.add_argument("--rollouts", type=int, default=4)
        if name == "train":
            p.add_argument("--lr", type=float, required=True, help="SFT learning rate")
            p.add_argument("--min-episodes", type=int, default=8)
        if name != "report":
            p.add_argument("--live", action="store_true", required=True, help="spends W&B credit")
    args = parser.parse_args()
    if args.command == "report":
        report(Path(args.results), args.run)
    else:
        asyncio.run(sample(args) if args.command == "sample" else train(args))


if __name__ == "__main__":
    main()
