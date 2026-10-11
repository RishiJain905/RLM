"""An in-process stand-in for Sandbox, for testing the episode loop without Docker.

It runs cells with exec() in a worker thread and serves `llm` / `llm_batch` through the same
`complete` function a real Sandbox gets, with the same budget, error and infra rules as
EpisodeBridge. It isolates nothing: the real sandbox is tested in test_sandbox.py, and the
Docker cases in test_env.py and test_evals.py repeat the key checks through it.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from rlm_rl import bridge
from rlm_rl.bridge import InfraError, SubcallError, SubcallRecord
from rlm_rl.sandbox import CellResult

_POOL = ThreadPoolExecutor(max_workers=128, thread_name_prefix="fake-repl")


class FakeBridge:
    def __init__(self, complete, max_subcalls: int = bridge.MAX_SUBCALLS):
        self.complete = complete
        self.max_subcalls = max_subcalls
        self.subcalls_used = 0
        self.records: list[SubcallRecord] = []
        self.rejected: list[str] = []
        self.infra_error: str | None = None

    async def serve(self, prompts: list[str]) -> list[str]:
        if self.infra_error:
            raise RuntimeError("sub-call failed for an infrastructure reason; this episode will be discarded")
        if self.subcalls_used + len(prompts) > self.max_subcalls:
            raise RuntimeError(bridge.budget_exhausted_message(self.max_subcalls))
        self.subcalls_used += len(prompts)
        results = await asyncio.gather(*(self._one(bridge.truncate_prompt(p)) for p in prompts), return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException) and not isinstance(r, SubcallError):
                if self.infra_error is None:
                    self.infra_error = str(r) if isinstance(r, InfraError) else f"sub-call: {r!r}"
                raise RuntimeError("sub-call failed for an infrastructure reason; this episode will be discarded")
        for r in results:
            if isinstance(r, SubcallError):
                raise RuntimeError(f"sub-call failed: {r}")
        return results

    async def _one(self, prompt: str) -> str:
        started = time.monotonic()
        try:
            out = await self.complete(prompt)
        except BaseException as exc:
            self.records.append(SubcallRecord(prompt, None, repr(exc), time.monotonic() - started))
            raise
        self.records.append(SubcallRecord(prompt, out, None, time.monotonic() - started))
        return out


FAKE_TIMEOUT = "# fake: timeout"  # a cell containing this times out, as a real one would after 120 s
FAKE_EVENT = "# fake: audit "  # followed by a kind: the cell logs that sandbox audit event, then runs


class FakeSandbox:
    """`kill_after_cells=n` makes the container "stop from outside" once n cells have run.

    Two markers stand in for what only a real container does: a cell containing FAKE_TIMEOUT
    times out (the REPL restarts and the model is told), and FAKE_EVENT + kind logs an audit
    event, as the real REPL's audit hook does for a blocked attempt.
    """

    def __init__(self, context: str, complete, *, run_id: str = "adhoc", kill_after_cells: int | None = None,
                 max_subcalls: int = bridge.MAX_SUBCALLS):
        self.context = context
        self.bridge = FakeBridge(complete, max_subcalls)
        self.events: list[dict] = []
        self.cells = self.timeouts = self.crashes = self.restarts = 0
        self.kill_after_cells = kill_after_cells
        self._namespace: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def run_cell(self, code: str) -> CellResult:
        if self.kill_after_cells is not None and self.cells >= self.kill_after_cells:
            raise InfraError("the container stopped from outside the REPL (SIGKILL)")
        self.cells += 1
        for line in code.splitlines():
            if line.strip().startswith(FAKE_EVENT):
                self.events.append({"kind": line.strip()[len(FAKE_EVENT):].strip(), "event": "fake", "detail": "",
                                    "blocked": True})
        if FAKE_TIMEOUT in code:
            self.timeouts += 1
            self.restarts += 1
            self._namespace = {}
            return CellResult("", 0, bridge.CELL_TIMEOUT_ERROR, restarted=True, crash="timeout")
        loop = asyncio.get_running_loop()
        out = io.StringIO()
        state = {"submitted": False, "answer": None}

        def call(prompts):
            return asyncio.run_coroutine_threadsafe(self.bridge.serve(prompts), loop).result()

        def submit(answer):
            state["submitted"], state["answer"] = True, answer

        ns = self._namespace
        ns.update(context=self.context, submit=submit, llm=lambda p: call([p])[0], llm_batch=lambda ps: call(list(ps)),
                  print=lambda *a, sep=" ", end="\n", **k: out.write(sep.join(map(str, a)) + end))

        def run():
            try:
                exec(compile(code, "<cell>", "exec"), ns)
            except Exception:
                return "\n".join(traceback.format_exc().strip().splitlines()[-bridge.TRACEBACK_LINES:])
            return None

        started = time.monotonic()
        error = await loop.run_in_executor(_POOL, run)
        if self.bridge.infra_error:
            raise InfraError(self.bridge.infra_error)
        stdout = out.getvalue()
        return CellResult(stdout[:bridge.OUTPUT_MAX_CHARS], len(stdout), error, submitted=state["submitted"],
                          answer=state["answer"], wall_s=time.monotonic() - started,
                          compute_s=time.monotonic() - started)


class FakeBackend:
    """An in-memory stand-in for backend.ArtBackend: no W&B, no credit.

    Each model keeps a list of checkpoint steps. `train` checks the groups are trainable ART groups
    (rewards, and root choices carrying token ids) and adds the next step; `fork` starts a model from
    another's checkpoint; `train_metrics(name, step)` makes the metrics a test needs. Set
    `fail_next_train` to an exception to make the next train call raise it before training, or
    `fail_after_next_train` to raise it after the checkpoint exists. `fork_numbering="restart"`
    numbers a fork's checkpoints from 1, one of the two ways the serverless backend might *(inferred)*.
    """

    def __init__(self, base_url: str, *, api_key: str = "test-key", fork_numbering: str = "continue",
                 path: str | None = None):
        import openai

        self.path = path  # where the checkpoint lists persist between processes (the CLI dry run)
        self.base_url = base_url
        self.api_key = api_key
        self.fork_numbering = fork_numbering  # "continue": a fork at k trains to k+1; "restart": to 1
        self.models: dict[str, dict] = {}
        self.trained: list[dict] = []
        self.logged: list[dict] = []
        self.deleted: list[tuple[str, list[int]]] = []
        self.forks: list[tuple[str, str, int]] = []
        self.fail_next_train: BaseException | None = None
        self.fail_after_next_train: BaseException | None = None  # the job finishes, then the client loses it
        self.train_metrics = lambda name, step: {"loss/grad_norm": 0.7, "loss/probs_corr": 0.997,
                                                 "loss/clipped_token_fraction": 0.001,
                                                 "loss/importance_ratio_mean": 1.0, "loss/train": -0.05}
        self._openai = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.models = json.load(f)

    def _persist(self) -> None:
        if self.path:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.models, f)

    async def open(self, name: str) -> None:
        self.models.setdefault(name, {"steps": [], "next": 1})
        self._persist()

    def inference_name(self, name: str, step: int) -> str:
        return f"{name}:step{step}"

    def openai_client(self):
        return self._openai

    async def train(self, name: str, groups: list, **kwargs) -> tuple[int, dict]:
        if self.fail_next_train is not None:
            exc, self.fail_next_train = self.fail_next_train, None
            raise exc
        for group in groups:
            assert len(group.trajectories) >= 2, "a trained group needs at least 2 trajectories"
            assert len({t.reward for t in group.trajectories}) > 1, "an all-equal group reached training"
            for traj in group.trajectories:
                choices = [c for c in traj.messages_and_choices if not isinstance(c, dict)]
                assert choices and all((c.model_extra or {}).get("token_ids") for c in choices)
        model = self.models[name]
        step = model["next"]
        model["steps"].append(step)
        model["next"] = step + 1
        self._persist()
        self.trained.append({"model": name, "step": step, "groups": len(groups), "kwargs": kwargs})
        if self.fail_after_next_train is not None:
            exc, self.fail_after_next_train = self.fail_after_next_train, None
            raise exc
        return step, self.train_metrics(name, step)

    async def latest_step(self, name: str) -> int:
        return max(self.models[name]["steps"], default=0)

    async def list_steps(self, name: str) -> list[int]:
        return list(self.models[name]["steps"])

    async def delete_steps(self, name: str, steps: list[int]) -> None:
        self.deleted.append((name, list(steps)))
        self.models[name]["steps"] = [s for s in self.models[name]["steps"] if s not in steps]
        self._persist()

    async def fork(self, name: str, from_name: str, step: int) -> None:
        if step not in self.models[from_name]["steps"]:
            raise ValueError(f"{from_name} has no checkpoint {step}")
        self.forks.append((name, from_name, step))
        self.models[name] = {"steps": [step], "next": step + 1} if self.fork_numbering == "continue" else \
            {"steps": [], "next": 1}
        self._persist()

    async def log(self, name: str, groups, *, split: str, step: int, metrics: dict) -> None:
        self.logged.append({"model": name, "split": split, "step": step, "groups": len(groups or ()),
                            "metrics": dict(metrics)})

    def art_model(self, name: str):
        return None

    async def close(self) -> None:
        await self._openai.close()
