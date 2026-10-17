"""$0 probe: does W&B serverless pick up a forked checkpoint if the fork happens BEFORE the model is registered?

No training or inference call is made. It lists the service's checkpoint table for second-seed-v1 and its
branch b1, then forks second-seed-v1:step15 into a fresh model name before registering it, and lists again.
The WANDB_API_KEY comes from .env through load_dotenv and is never printed.
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import art  # noqa: E402
import wandb  # noqa: E402
from art.serverless.backend import ServerlessBackend  # noqa: E402

PROJECT, BASE = "rlm-production", "Qwen/Qwen3.8-27B"


async def steps(backend, model):
    return [c.step async for c in backend._client.models.checkpoints.list(model_id=model.id)]


async def main():
    backend = ServerlessBackend()
    for name in ("second-seed-v1", "second-seed-v1-b1"):
        m = art.TrainableModel(name=name, project=PROJECT, run_name=name, base_model=BASE)
        await m.register(backend)  # return_existing=True: the model already exists, nothing new is made
        print(name, "service checkpoints:", sorted(await steps(backend, m)))
    probe = art.TrainableModel(name="fork-probe-1016", project=PROJECT, run_name="fork-probe-1016", base_model=BASE)
    probe.entity = wandb.Api().default_entity
    await backend._experimental_fork_checkpoint(probe, from_model="second-seed-v1", from_project=PROJECT,
                                                not_after_step=15)
    print("forked second-seed-v1:step15 into fork-probe-1016 before registering it")
    await probe.register(backend)
    print("fork-probe-1016 service checkpoints after register:", sorted(await steps(backend, probe)))
    print("latest step the next train call would start from:", await backend._get_step(probe))
    await backend.close()


asyncio.run(main())
