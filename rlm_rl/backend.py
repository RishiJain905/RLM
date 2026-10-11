"""The training loop's view of W&B serverless RL through ART 0.5.20 (spec §5.1, plan step 6).

Each branch of a run is one ART TrainableModel, named after the run ("<run>" for branch 0,
"<run>-b1" after the first resume from an earlier checkpoint). Rollouts name a checkpoint as
`wandb-artifact:///<entity>/<project>/<model>:step<N>`, and sub-calls always go to branch 0's
step 0, the frozen base model.

A resume from an earlier checkpoint forks it into a new branch with ART's
`_experimental_fork_checkpoint`, which copies the LoRA artifact into the new model's collection,
so nothing is deleted. That serverless training then continues from the forked weights is
*(inferred)* from ART's code; checking it is the first thing a live run should do (step 6 results).

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

    async def latest_step(self, name: str) -> int:
        return await self.backend._get_step(self.models[name])

    async def list_steps(self, name: str) -> list[int]:
        model = self.models[name]
        return [c.step async for c in self.backend._client.models.checkpoints.list(model_id=model.id)]

    async def delete_steps(self, name: str, steps: list[int]) -> None:
        if steps:
            await self.backend._client.models.checkpoints.delete(model_id=self.models[name].id, steps=list(steps))

    async def fork(self, name: str, from_name: str, step: int) -> None:
        await self.backend._experimental_fork_checkpoint(self.models[name], from_model=from_name,
                                                         from_project=self.project, not_after_step=step)

    async def log(self, name: str, groups: list | None, *, split: str, step: int, metrics: dict) -> None:
        await self.models[name].log(groups, split=split, step=step, metrics=metrics)

    def art_model(self, name: str) -> Any:
        return self.models[name]

    async def close(self) -> None:
        await self.backend.close()
