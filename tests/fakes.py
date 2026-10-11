"""An in-process stand-in for Sandbox, for testing the episode loop without Docker.

It runs cells with exec() in a worker thread and serves `llm` / `llm_batch` through the same
`complete` function a real Sandbox gets, with the same budget, error and infra rules as
EpisodeBridge. It isolates nothing: the real sandbox is tested in test_sandbox.py, and the
Docker cases in test_env.py and test_evals.py repeat the key checks through it.
"""

from __future__ import annotations

import asyncio
import io
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
