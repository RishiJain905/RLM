"""Live probe (needs Rishi's OK): does a fresh model train on from another checkpoint's weights?

1. Register a new model (the service makes its step 0, with no optimizer state yet).
2. Log `second-seed-v1:step15`'s LoRA into the new model's collection under the alias `step0`.
3. Train one step on the batch saved from the crashed throwaway-branch step (results/second-seed-v1/pending/
   step0017.pkl): no rollouts, so $0 of inference; the training call uses the GPU pot.
4. Print the new checkpoint; compare_weights.py then shows whether it sits next to checkpoint 15.

The WANDB_API_KEY comes from .env through load_dotenv and is never printed.
"""
import asyncio
import pickle
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import wandb  # noqa: E402

from rlm_rl import config  # noqa: E402
from rlm_rl.backend import ArtBackend  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "rewind-probe-1016"
SOURCE = "second-seed-v1:step15"
PROJECT = "rlm-production"


async def main():
    backend = ArtBackend(project=PROJECT, base_model=config.BASE_MODEL)
    await backend.open(NAME)
    print(NAME, "service checkpoints after register:", sorted(await backend.list_steps(NAME)))
    api = wandb.Api(timeout=120)
    ent = backend.models[NAME].entity
    print("artifacts before:", {a.version: a.aliases for a in api.artifacts("lora", f"{ent}/{PROJECT}/{NAME}")})
    with tempfile.TemporaryDirectory() as tmp:
        api.artifact(f"{ent}/{PROJECT}/{SOURCE}").download(root=tmp)
        run = wandb.init(project=PROJECT, entity=ent, job_type="checkpoint-seed", name=f"seed-{NAME}-from-step15",
                         settings=wandb.Settings(silent=True))
        art = wandb.Artifact(name=NAME, type="lora", metadata={"copied_from": SOURCE})
        art.add_dir(tmp)
        run.log_artifact(art, aliases=["step0", "latest"])
        run.finish()
    api = wandb.Api(timeout=120)
    print("artifacts after:", {a.version: a.aliases for a in api.artifacts("lora", f"{ent}/{PROJECT}/{NAME}")})
    groups = pickle.loads((ROOT / "results" / "second-seed-v1" / "pending" / "step0017.pkl").read_bytes())
    print("training on", len(groups), "saved groups")
    step, metrics = await backend.train(NAME, groups, **config.second_seed().train_args(1e-5))
    print("trained to step", step, {k: metrics.get(k) for k in ("loss/train", "loss/grad_norm", "loss/probs_corr")})
    print(NAME, "service checkpoints now:", sorted(await backend.list_steps(NAME)))
    await backend.close()


asyncio.run(main())
