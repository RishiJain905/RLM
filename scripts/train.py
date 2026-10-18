"""The resumable GRPO training run on W&B serverless RL (plan step 6; the loop is rlm_rl/training.py).

Run inside WSL2 (Ubuntu) from the repo root, with Docker Desktop running:

    .venv-linux/bin/python scripts/train.py start --kind second_seed --run second-seed-v1 --live
    .venv-linux/bin/python scripts/train.py start --kind production --run prod-v1 --lr 2e-5 --live
    .venv-linux/bin/python scripts/train.py stop prod-v1 --reason "checking a transcript"
    .venv-linux/bin/python scripts/train.py resume prod-v1 --live                 # from the latest checkpoint
    .venv-linux/bin/python scripts/train.py resume prod-v1 --from best --live     # or --from 40, any kept checkpoint
    .venv-linux/bin/python scripts/train.py status prod-v1

A live run spends W&B inference credit, so `start` and `resume` need either --live or --dry-run.
--dry-run runs the same loop against a local fake model server and a fake backend, with no W&B
call and no credit, writing to results/dry/ (git-ignored). The fake backend's checkpoints persist
in results/dry/fake-backend.json, so a dry run can be stopped and resumed like a live one.

`start` and `resume` first remove containers a crashed process of the run left behind, and any sandbox
or grader container that finished long ago (plan step 12). A resume refuses while the run's state says
it's running and was written in the last two minutes, since another process may still be running it.

`stop` sets the stop switch: the run finishes the step it's on, and that step's eval if one is
due, then stops. Ctrl-C does the same; a second Ctrl-C stops at once, and `resume` picks up
from the last recorded step. The W&B key is read from .env and never printed.
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import config, sandbox, training  # noqa: E402

DRY_RESULTS = ROOT / "results" / "dry"


def results_dir(args) -> Path:
    if args.results:
        return Path(args.results)
    return DRY_RESULTS if getattr(args, "dry_run", False) else training.RESULTS


def install_stop_handler(run: training.TrainingRun) -> None:
    """First Ctrl-C: stop after this step. Second: stop now."""
    def handler(signum, frame):
        if run.stop_requested:
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt
        run.stop_requested = "Ctrl-C: stop after this step"
        print("\nstopping after this step (Ctrl-C again to stop now)", flush=True)

    signal.signal(signal.SIGINT, handler)


async def run_live(args, cfg) -> dict:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY; never printed
    from rlm_rl.backend import ArtBackend

    state = training.load_state(results_dir(args), args.run) if cfg is None else None
    project = (cfg.project if cfg else state["config"]["project"])
    base_model = (cfg.base_model if cfg else state["config"]["base_model"])
    backend = ArtBackend(project=project, base_model=base_model)
    try:
        run = training.TrainingRun(args.run, backend=backend, cfg=cfg, results=results_dir(args),
                                   prune=not args.no_prune, stop_after=args.stop_after,
                                   clear_containers=sandbox.clear_for_run)
        install_stop_handler(run)
        return await go(run, args, cfg)
    finally:
        await backend.close()


async def run_dry(args, cfg) -> dict:
    from tests.dryrun import DryRun

    async with DryRun(results_dir(args), seed=args.seed, persist=True, real_edit=args.real_edit) as dry:
        if args.latency:  # slows the fake server to a live run's pace, e.g. for the dashboard (plan step 9)
            dry.server.latency = (args.latency * 0.5, args.latency * 1.5)
        # with --real-edit, edit episodes run in real containers, so leftovers are cleared as in a live run
        clear = {"clear_containers": sandbox.clear_for_run} if args.real_edit else {}
        run = dry.training_run(args.run, cfg=cfg, prune=not args.no_prune, stop_after=args.stop_after, **clear)
        install_stop_handler(run)
        return await go(run, args, cfg)


async def go(run: training.TrainingRun, args, cfg) -> dict:
    if args.command == "start":
        return await run.start()
    ceilings = {k: v for k, v in (("train", args.ceiling_train), ("eval", args.ceiling_eval)) if v is not None}
    return await run.resume(from_checkpoint=args.from_checkpoint, lr=args.lr, format_penalty=args.format_penalty,
                            mask_broken=args.mask_broken, resample_broken=args.resample_broken,
                            retrain_pending=args.retrain_pending,
                            discard_pending=args.discard_pending, ceilings=ceilings or None)


def build_config(args) -> config.RunConfig:
    if args.kind == "second_seed":
        cfg = config.second_seed(args.run)
    else:
        if args.lr is None:
            raise SystemExit("production needs --lr: 2e-5 if the second seed's checks passed, else 1e-5 (spec §5.2)")
        cfg = config.production(args.run, args.lr, tasks_from="tickets" if args.ticket_tasks else "7a")
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.ceiling_train is not None:
        cfg.ceilings["train"] = args.ceiling_train
    if args.ceiling_eval is not None:
        cfg.ceilings["eval"] = args.ceiling_eval
    if args.capability_set:
        cfg.capability_set = args.capability_set
    return cfg


def print_status(state: dict) -> None:
    spend, rows = state["spend"], state["steps"]
    print(f"{state['run']} ({state['kind']}): {state['status']}"
          + (f" - {state['stop']['rule']}: {state['stop']['reason']}" if state.get("stop") else ""))
    print(f"latest checkpoint {state['latest']} on branch {state['branch']}, learning rate {state['lr']['current']:g}")
    print(f"spend: train ${spend['train_usd']:.2f} of ${spend['ceilings'].get('train', 0):g}, eval ${spend['eval_usd']:.2f}"
          f" of ${spend['ceilings'].get('eval', 0):g}, training GPU ~${spend['training_gpu_usd_est']:.2f}")
    if state.get("best"):
        b = state["best"]
        print(f"best: checkpoint {b['checkpoint']}, {b['accuracy']:.1%} on {b['eval']}")
    for row in rows[-5:]:
        p, pol = row["pass"], row["policy"]
        print(f"  step {row['step']:>3} {row['status']:>8}: pass {p['rate'] if p['rate'] is not None else float('nan'):.2f}"
              f" trainable {p['trainable_groups']}/{p['groups']} entropy {pol['entropy'] or float('nan'):.3f}"
              f" KL {pol['kl_to_base'] if pol['kl_to_base'] is not None else float('nan'):.3f} ${row['cost']['step_usd']:.3f}")
    for name, rule in state.get("stop_rules", {}).items():
        print(f"  rule {name}: {rule['state']} - {rule['text']}")
    for notice in state["notices"][-5:]:
        print(f"  [{time.strftime('%m-%d %H:%M', time.localtime(notice['at']))}] {notice['kind']}: {notice['text']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "resume"):
        p = sub.add_parser(name)
        if name == "start":
            p.add_argument("--run", required=True)
        else:
            p.add_argument("run")
        mode = p.add_mutually_exclusive_group(required=True)
        mode.add_argument("--live", action="store_true", help="train on W&B serverless; spends inference credit")
        mode.add_argument("--dry-run", action="store_true", help="fake model server and backend; no W&B, no credit")
        p.add_argument("--lr", type=float, help="learning rate (production: 2e-5 or 1e-5)")
        p.add_argument("--stop-after", type=int, help="stop once this step is done")
        p.add_argument("--ceiling-train", type=float, help="inference ceiling for steps, $")
        p.add_argument("--ceiling-eval", type=float, help="inference ceiling for evals, $")
        p.add_argument("--no-prune", action="store_true", help="keep every checkpoint")
        p.add_argument("--results", help="results folder (default results/, or results/dry/ for a dry run)")
        p.add_argument("--seed", type=int, default=0, help="dry run only: the fake policy's seed")
        p.add_argument("--latency", type=float, default=0.0,
                       help="dry run only: mean seconds the fake server takes per request, to pace the run")
        p.add_argument("--real-edit", action="store_true",
                       help="dry run only: edit episodes in the real edit sandbox and grader (Docker), not the in-process fakes")
        if name == "start":
            p.add_argument("--kind", choices=config.RUN_KINDS, required=True)
            p.add_argument("--max-steps", type=int, help="override the run kind's step limit")
            p.add_argument("--capability-set", help="capability prompts (JSONL), relative to the repo, or "
                                                    "data:<path> relative to the data root")
            p.add_argument("--ticket-tasks", action="store_true",
                           help="production: step 6's ticket stand-in instead of the production families (dry runs)")
        else:
            p.add_argument("--from", dest="from_checkpoint", help="latest (default), best, or a kept checkpoint")
            p.add_argument("--retrain-pending", action="store_true", help="train a step's saved batch again")
            p.add_argument("--discard-pending", action="store_true", help="redo a step whose training never finished")
            p.add_argument("--format-penalty", type=float,
                           help="reward taken off per broken reply in the rollouts from here on (0 is off)")
            p.add_argument("--mask-broken", action=argparse.BooleanOptionalAction, default=None,
                           help="broken replies take no gradient from here on (production's default)")
            p.add_argument("--resample-broken", type=int,
                           help="times a broken reply is drawn again in the rollouts from here on (production: 2; 0 is off)")
    p = sub.add_parser("stop", help="stop a running run after the step it's on")
    p.add_argument("run")
    p.add_argument("--reason", default="")
    p.add_argument("--results")
    p.add_argument("--dry-run", action="store_true", help="a run under results/dry/")
    p = sub.add_parser("status")
    p.add_argument("run")
    p.add_argument("--results")
    p.add_argument("--dry-run", action="store_true", help="a run under results/dry/")
    p.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.command == "stop":
        path = training.request_stop(results_dir(args), args.run, args.reason)
        print(f"stop switch set ({path}); the run stops after the step it's on")
        return 0
    if args.command == "status":
        state = training.load_state(results_dir(args), args.run)
        print(json.dumps(state, indent=1, default=str) if args.json else "", end="")
        if not args.json:
            print_status(state)
        return 0

    cfg = build_config(args) if args.command == "start" else None
    try:
        state = asyncio.run((run_dry if args.dry_run else run_live)(args, cfg))
    except training.ResumeBlocked as exc:
        print(f"not resumed: {exc}", file=sys.stderr)
        return 2
    print_status(state)
    return 0 if state["status"] in ("finished", "stopped") else 1


if __name__ == "__main__":
    sys.exit(main())
