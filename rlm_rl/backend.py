"""The training loop's view of W&B serverless RL through ART 0.5.20 (spec §5.1, plan step 6).

Each branch of a run is one ART TrainableModel: "<run>" for branch 0, "<run>-b<n>" for the n-th
resume from an earlier checkpoint. Rollouts name a checkpoint as
`wandb-artifact:///<entity>/<project>/<model>:step<N>`, and sub-calls always go to branch 0's
step 0, the frozen base model.

A resume from an earlier checkpoint starts a new branch from it (`branch_from`), tested live on
2026-10-16 (step 10 results §8). W&B's training service starts each train call from the newest step
in its own checkpoint list, loading that step's LoRA from the artifact alias `step<N>`, and checks
it against the optimizer state it saved with that step. So a new model, whose only step is 0 and
which has no optimizer state yet, gets the source checkpoint's LoRA logged under its `step0` alias;
its first train call then continues from those weights with a fresh optimizer. What doesn't work:
ART's `_experimental_fork_checkpoint` (it logs the copy as `step<k>`, which the service never reads,
so training started from a fresh LoRA), deleting a model's newer checkpoints ("Cannot delete the
last checkpoint"), and swapping the weights under an existing model's newest step ("Optimizer
checkpoint and loaded LoRA weights do not match"). Nothing is deleted: every branch's checkpoints stay.

Pruning (spec §8) uses the serverless API's checkpoint list and delete calls, the same ones
ART's `delete_checkpoints()` uses.
"""

from __future__ import annotations

from typing import Any


class ArtBackend:
    def __init__(self, *, project: str, base_model: str):
        from art.serverless.backend import ServerlessBackend

        self.project = project
        self.base_model = base_model
        self.backend = ServerlessBackend()  # reads WANDB_API_KEY from the environment
        self.models: dict[str, Any] = {}
        self._root: str | None = None

    async def open(self, name: str) -> None:
        import art

        if name in self.models:
            return
        model = art.TrainableModel(name=name, project=self.project, run_name=name, base_model=self.base_model)
        await model.register(self.backend)
        self.models[name] = model
        self._root = self._root or name

    def inference_name(self, name: str, step: int) -> str:
        return self.models[name].get_inference_name(step=step)

    def openai_client(self) -> Any:
        return self.models[self._root].openai_client()

    async def train(self, name: str, groups: list, **kwargs) -> tuple[int, dict]:
        result = await self.backend.train(self.models[name], groups, **kwargs)
        return result.step, dict(result.metrics)

    async def train_sft(self, name: str, trajectories: list, *, learning_rate: float, batch_size="auto") -> int:
        """One SFT job on model `name` (the warm start, rlm_rl/warmstart.py), training only each trajectory's last
        assistant turn; returns the model's new latest step. It bills training GPU-hours, not inference."""
        import art

        config = art.TrainSFTConfig(learning_rate=learning_rate, batch_size=batch_size, assistant_turns="last")
        await self.models[name].train_sft(trajectories, config=config)
        return await self.latest_step(name)

    async def latest_step(self, name: str) -> int:
        return await self.backend._get_step(self.models[name])

    async def list_steps(self, name: str) -> list[int]:
        model = self.models[name]
        return [c.step async for c in self.backend._client.models.checkpoints.list(model_id=model.id)]

    async def delete_steps(self, name: str, steps: list[int]) -> None:
        if steps:
            await self.backend._client.models.checkpoints.delete(model_id=self.models[name].id, steps=list(steps))

    async def branch_from(self, name: str, source: str, step: int) -> None:
        """Start model `name` from checkpoint `step` of model `source`: register it, then log the source's LoRA as its
        `step0`, so its first train call continues from those weights (see the module docstring).

        Raises if `name` already has checkpoints past step 0, or if its `step0` doesn't end up holding the copy.
        """
        import tempfile

        import wandb

        await self.open(name)
        steps = sorted(await self.list_steps(name))
        if steps != [0]:
            raise RuntimeError(f"{name} isn't a fresh model (checkpoints {steps}); a branch needs a new name")
        model = self.models[name]
        api = wandb.Api(api_key=self.backend._client.api_key)
        ref = f"{model.entity}/{self.project}/{source}:step{step}"
        with tempfile.TemporaryDirectory() as tmp:
            api.artifact(ref).download(root=tmp)
            run = wandb.init(project=self.project, entity=model.entity, job_type="branch-seed",
                             name=f"seed-{name}-from-{source}-step{step}", settings=wandb.Settings(silent=True))
            copy = wandb.Artifact(name=name, type="lora", metadata={"copied_from": f"{source}:step{step}"})
            copy.add_dir(tmp)
            run.log_artifact(copy, aliases=["step0", "latest"])
            run.finish()
        seeded = wandb.Api(api_key=self.backend._client.api_key).artifact(f"{model.entity}/{self.project}/{name}:step0")
        if (seeded.metadata or {}).get("copied_from") != f"{source}:step{step}":
            raise RuntimeError(f"{name}:step0 doesn't hold {source}:step{step} after seeding it")

    async def log(self, name: str, groups: list | None, *, split: str, step: int, metrics: dict) -> None:
        await self.models[name].log(groups, split=split, step=step, metrics=metrics)

    def art_model(self, name: str) -> Any:
        return self.models[name]

    async def close(self) -> None:
        await self.backend.close()
