"""RLSD vs GRPO for 20 steps on the sub-call-heavy ticket tasks (Qwen3.8-27B, W&B serverless).

Resumable: progress is saved to results/rlsd_vs_grpo_20.json after every step and eval,
and a rerun picks up where it stopped.

    .venv/Scripts/python.exe scripts/run_rlsd_vs_grpo.py
"""

import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY
assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY missing from .env"

import art
from art.serverless.backend import ServerlessBackend

from rlm_rl.rlm_env import SubcallBridge, evaluate_rlm, gather_rlm_groups
from rlm_rl.shaping import shape_groups, train_kwargs
from rlm_rl.ticket_tasks import make_ticket_tasks

BASE_MODEL = "Qwen/Qwen3.8-27B"
PROJECT = "rlm-rlsd-vs-grpo"
RUN_TAG = "v1"
ARMS = {"grpo": None, "rlsd": 0.1}  # arm -> tau for the advantage channel
TRAIN_STEPS = 20
TASKS_PER_STEP = 8
ROLLOUTS_PER_TASK = 4
LEARNING_RATE = 1e-5
RLSD_LAMBDA_0, RLSD_DECAY_STEPS = 0.5, 50  # paper: lambda decays linearly from 0.5 to 0 over 50 steps
EVAL_TASKS = make_ticket_tasks(50, seed=99_999)
EVAL_SAMPLES = 2
RESULTS_PATH = ROOT / "results" / "rlsd_vs_grpo_20.json"


def group_stats(groups):
    trajs = [t for g in groups for t in g.trajectories]
    mean = lambda k: sum(t.metrics[k] for t in trajs) / len(trajs)
    return {
        "train_reward": sum(t.reward for t in trajs) / len(trajs),
        "train_correct": mean("correct"),
        "turns": mean("turns"),
        "subcalls": mean("subcalls"),
        "used_subcalls": mean("used_subcalls"),
        "submitted": mean("submitted"),
        "first_turn_submit": mean("first_turn_submit"),
        "mixed_groups": sum(1 for g in groups if len({t.reward for t in g.trajectories}) > 1),
        "trajectories": len(trajs),
    }


async def main():
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {
        "config": {"base_model": BASE_MODEL, "arms": ARMS, "train_steps": TRAIN_STEPS, "tasks_per_step": TASKS_PER_STEP,
                   "rollouts_per_task": ROLLOUTS_PER_TASK, "learning_rate": LEARNING_RATE,
                   "rlsd_lambda": [RLSD_LAMBDA_0, RLSD_DECAY_STEPS], "eval_tasks": len(EVAL_TASKS),
                   "eval_samples": EVAL_SAMPLES},
        "steps": {arm: [] for arm in ARMS},
        "eval": {},
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    save = lambda: RESULTS_PATH.write_text(json.dumps(results, indent=2))

    backend = ServerlessBackend()
    models = {}
    for arm in ARMS:
        models[arm] = art.TrainableModel(name=f"qwen3-8-27b-{arm}", project=PROJECT,
                                         run_name=f"qwen3-8-27b-{arm}-{RUN_TAG}", base_model=BASE_MODEL)
        await models[arm].register(backend)
    anchor = models["grpo"]
    bridge = SubcallBridge(anchor.inference_base_url, anchor.inference_api_key, anchor.get_inference_name(step=0))

    if "base" not in results["eval"]:
        results["eval"]["base"] = await evaluate_rlm(anchor, 0, EVAL_TASKS, bridge, EVAL_SAMPLES)
        save()
        print("base eval:", results["eval"]["base"], flush=True)

    async def run_arm(arm, tau):
        model = models[arm]
        done = results["steps"][arm]
        step = done[-1]["trained_to"] if done else 0
        for i in range(len(done), TRAIN_STEPS):
            started = time.time()
            groups = await gather_rlm_groups(model, step, make_ticket_tasks(TASKS_PER_STEP, seed=1000 + i),
                                             ROLLOUTS_PER_TASK, bridge)
            row = {"step": step, **group_stats(groups)}
            await model.log(groups, split="train", step=step)
            if arm == "rlsd":
                lam = RLSD_LAMBDA_0 * max(0.0, 1.0 - i / RLSD_DECAY_STEPS)
                groups, stats = await shape_groups(model, step, groups, "rlsd", tau=tau, rlsd_lambda=lam, seed=i)
                row["shaping"] = {"lambda": lam, **stats.summary()}
            result = await backend.train(model, groups, learning_rate=LEARNING_RATE, **train_kwargs(arm, tau))
            await model.log(metrics=result.metrics, step=result.step)
            step = result.step
            row.update(trained_to=step, seconds=round(time.time() - started),
                       loss={k: v for k, v in result.metrics.items() if k.startswith("loss/")})
            done.append(row)
            save()
            print(f"[{arm}] step {i + 1}/{TRAIN_STEPS} reward={row['train_reward']:.3f} "
                  f"subcalls={row['subcalls']:.2f} mixed={row['mixed_groups']} ({row['seconds']}s)", flush=True)
        if arm not in results["eval"]:
            results["eval"][arm] = await evaluate_rlm(model, step, EVAL_TASKS, bridge, EVAL_SAMPLES)
            save()
            print(f"[{arm}] eval:", results["eval"][arm], flush=True)

    await asyncio.gather(*(run_arm(arm, tau) for arm, tau in ARMS.items()))


if __name__ == "__main__":
    asyncio.run(main())
