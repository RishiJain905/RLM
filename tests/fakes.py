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
    def __init__(self, complete, max_subcalls: int = bridge.MAX_SUBCALLS, *, max_chars: int = bridge.SUBCALL_MAX_CHARS,
                 oversize: str = "truncate"):
        self.complete = complete
        self.max_subcalls = max_subcalls
        self.max_chars = max_chars
        self.oversize = oversize
        self.subcalls_used = 0
        self.records: list[SubcallRecord] = []
        self.rejected: list[str] = []
        self.infra_error: str | None = None

    async def serve(self, prompts: list[str]) -> list[str]:
        if self.infra_error:
            raise RuntimeError("sub-call failed for an infrastructure reason; this episode will be discarded")
        if self.subcalls_used + len(prompts) > self.max_subcalls:
            raise RuntimeError(bridge.budget_exhausted_message(self.max_subcalls))
        longest = max((len(p) for p in prompts), default=0)
        if self.oversize == "error" and longest > self.max_chars:
            raise ValueError(bridge.oversize_message(longest, self.max_chars))
        self.subcalls_used += len(prompts)
        results = await asyncio.gather(*(self._one(bridge.truncate_prompt(p, self.max_chars)) for p in prompts),
                                       return_exceptions=True)
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
                 max_subcalls: int = bridge.MAX_SUBCALLS, messages: bridge.CellMessages | None = None,
                 subcall_max_chars: int = bridge.SUBCALL_MAX_CHARS, subcall_oversize: str = "truncate"):
        self.context = context
        self.messages = messages or bridge.cell_messages(bridge.PROMPT_VERSION)
        self.bridge = FakeBridge(complete, max_subcalls, max_chars=subcall_max_chars, oversize=subcall_oversize)
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
            return CellResult("", 0, self.messages.timeout, restarted=True, crash="timeout")
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
    (rewards, and root choices carrying token ids) and adds the next step; `branch_from` starts a new
    model from another's checkpoint, numbering its own checkpoints from 1, as W&B does (step 10
    results §8); `train_metrics(name, step)` makes the metrics a test needs. Set `fail_next_train` to an
    exception to make the next train call raise it before training, or `fail_after_next_train` to raise
    it after the checkpoint exists.
    """

    def __init__(self, base_url: str, *, api_key: str = "test-key", path: str | None = None):
        import openai

        self.path = path  # where the checkpoint lists persist between processes (the CLI dry run)
        self.base_url = base_url
        self.api_key = api_key
        self.models: dict[str, dict] = {}
        self.trained: list[dict] = []
        self.logged: list[dict] = []
        self.deleted: list[tuple[str, list[int]]] = []
        self.branched: list[tuple[str, str, int]] = []  # (new model, source model, source step)
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

    async def branch_from(self, name: str, source: str, step: int) -> None:
        if step not in self.models[source]["steps"]:
            raise ValueError(f"{source} has no checkpoint {step}")
        self.branched.append((name, source, step))
        self.models.setdefault(name, {"steps": [], "next": 1})  # a name already used keeps its steps: the loop refuses it
        self._persist()

    async def log(self, name: str, groups, *, split: str, step: int, metrics: dict) -> None:
        self.logged.append({"model": name, "split": split, "step": step, "groups": len(groups or ()),
                            "metrics": dict(metrics)})

    def art_model(self, name: str):
        return None

    async def close(self) -> None:
        await self._openai.close()


# ----------------------------------------------------------------------------- the edit families (spec §12)

import ast
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_EXEC_CALLS = {"run", "call", "check_call", "check_output", "Popen", "system", "getoutput", "getstatusoutput"}
_READ_CALLS = {"open", "listdir", "walk", "scandir", "glob", "rglob", "read_text", "iterdir", "exists", "isfile"}


def code_events(code: str) -> list[dict]:
    """What the real edit sandbox's audit hook would log for a cell, read from its code: each program it starts (with
    the literal arguments) and whether it reads /work/repo."""
    events, read = [], False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return events
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name in _EXEC_CALLS and node.args:
            try:
                value = ast.literal_eval(node.args[0])
            except (ValueError, SyntaxError):
                value = ["<computed>"]
            argv = value if isinstance(value, list) else str(value).split()
            events.append({"kind": "exec", "event": "subprocess.Popen", "detail": " ".join(map(str, argv)),
                           "argv": [str(a) for a in argv], "blocked": False})
        if name in _READ_CALLS:
            read = True
    if read:
        events.insert(0, {"kind": "repo_read", "event": "open", "detail": "/work/repo", "blocked": False})
    return events


def private_tools(root: Path, on_call):
    """A copy of rlm_rl/repl_tools.py with its own settings, pointed at `root`: in-process episodes run at once, and
    the module keeps its settings in globals, as it can in the real REPL (one per container)."""
    import importlib.util

    from rlm_rl import editbox

    spec = importlib.util.spec_from_file_location(f"repl_tools_{secrets.token_hex(6)}", editbox.TOOLS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure(root=str(root), on_call=on_call)
    return module


class FakeEditSandbox(FakeSandbox):
    """An in-process edit sandbox (Linux): the task's repository in a temporary folder, its diff taken the real way
    (editbox.diff), and each cell's changes and program runs recorded as the real one does. Cells see /work/repo as
    that folder and have the coding tools (repl_tools.py), whose calls are logged as the real REPL logs them. It
    isolates nothing; the real edit sandbox is tested in test_editbox.py."""

    def __init__(self, context: str, complete, *, task, run_id: str = "adhoc", max_subcalls: int = 30,
                 messages=None, cache: Path | None = None, kill_after_cells: int | None = None, limits=None):
        super().__init__(context, complete, run_id=run_id, kill_after_cells=kill_after_cells,
                         max_subcalls=max_subcalls, messages=messages or bridge.cell_messages(edit=True,
                                                                                               limits=bridge.EDIT_LIMITS))
        self.task = task
        self.limits = limits  # a task's own cell limit (spec §12.3); the fake has no clock to enforce it
        self.cache = cache
        self.root = Path(tempfile.mkdtemp(prefix="rlm-fake-edit-"))
        self.repo_dir = self.root / "repo"
        self.ws_dir = None
        self.edit_image = None
        self.docker = "docker"
        self.runtime = self.bundle = None
        self._fp = None
        self._cell_io = dict.fromkeys(("tool_writes", "raw_writes", "tool_execs", "raw_execs"), 0)

    async def __aenter__(self):
        from rlm_rl import editbox

        self.ws_dir = await asyncio.to_thread(self.task.workspace.ensure, self.cache)
        shutil.copytree(self.ws_dir / "tree", self.repo_dir, symlinks=True)
        shutil.copytree(self.ws_dir / "git", self.repo_dir / ".git", symlinks=True)
        self._fp = editbox.fingerprint(self.repo_dir)
        self._tools = private_tools(self.repo_dir, self._tool_call)
        self._namespace.update(self._tools.namespace())
        return self

    def _tool_call(self, tool: str, ok: bool, seconds: float, target: str) -> None:
        """What the real REPL logs for a tool call: the call, and the program shell() starts."""
        self.events.append({"kind": "tool", "event": tool, "detail": repr(target), "blocked": False, "ok": ok,
                            "seconds": seconds, "cell": self.cells})
        if tool == "shell":
            self._cell_io["tool_execs"] += 1
            self.events.append({"kind": "exec", "event": "subprocess.Popen", "detail": f"sh -c {target}",
                                "argv": ["sh", "-c", target], "blocked": False, "tool": True, "cell": self.cells})
        elif tool in ("edit_file", "write_file") and ok:
            self._cell_io["tool_writes"] += 1

    async def __aexit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    async def run_cell(self, code: str) -> CellResult:
        from rlm_rl import editbox

        raw = code_events(code)
        for event in raw:
            self.events.append({**event, "cell": self.cells + 1, **({"tool": False} if event["kind"] == "exec" else {})})
        self._cell_io = dict.fromkeys(self._cell_io, 0)
        result = await super().run_cell(code.replace(bridge.EDIT_REPO_PATH, str(self.repo_dir)))
        if result.restarted:  # a restarted REPL imports the tools again
            self._namespace.update(self._tools.namespace())
        now = editbox.fingerprint(self.repo_dir)
        result.repo_changed, self._fp = now != self._fp, now
        io_counts = dict(self._cell_io, raw_execs=sum(e["kind"] == "exec" for e in raw))
        if result.repo_changed and not io_counts["tool_writes"]:
            io_counts["raw_writes"] = 1  # plain open() made the change: the fake can't count each write
        if any(io_counts.values()):
            self.events.append({"kind": "io", "event": "cell", "detail": "''", "blocked": False, **io_counts,
                                "cell": self.cells})
        return result

    async def diff(self) -> str:
        from rlm_rl import editbox

        scratch = Path(tempfile.mkdtemp(prefix="rlm-fake-diff-"))
        try:
            return editbox.diff(self.ws_dir, self.repo_dir, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


async def local_grade(task, ep, box, run_id):
    """The grader without a container, for dry runs and tests (Linux). edit_synth's tests really run, with this
    machine's Python and the grader's own diff checks, canaries and report parser; a task whose runner needs its
    image (edit_real) is resolved when the patched files equal the gold patch's."""
    from rlm_rl import grader

    spec = grader.TestSpec.from_dict(task.grading["tests"])
    checked = grader.check_diff(ep.diff or "", spec.language, box.ws_dir / "tree")
    result = grader.GradeResult(applied=True, empty=checked.empty, resolved=False, f2p_total=len(spec.f2p),
                                p2p_total=len(spec.p2p), hacks=list(checked.hacks), audit=list(checked.audit),
                                stripped=checked.stripped, diff_lines=checked.diff_lines,
                                files_changed=checked.files_changed, oversize=checked.oversize)
    if checked.empty:
        return result
    started = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix="rlm-fake-grade-"))
    try:
        canary = grader.RUNNER_CLASSES[spec.runner].canary(spec) if spec.runner == "unittest" else \
            grader.Canary({}, "", "", [])
        applied, why = await asyncio.to_thread(grader.prepare, box.ws_dir, checked, spec, canary, root / "repo", root)
        if not applied:
            result.applied = False
            return result
        if spec.runner == "unittest":
            report = root / "report.json"
            driver = Path(grader.__file__).with_name("grader_driver.py")
            proc = await asyncio.create_subprocess_exec(sys.executable, "-I", "-B", str(driver), "unittest", "--report",
                                                        str(report), "--top", ".", *spec.scope, *canary.scope,
                                                        cwd=root / "repo", stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
            await asyncio.wait_for(proc.communicate(), spec.timeout_s)
            fake_report = "/fake/report.json"
            text = report.read_text(encoding="utf-8") if report.exists() else None
            driver_out = {"commands": [{"exit": proc.returncode}],
                          "reports": [{"path": fake_report, "found": text is not None, "text": text}]}
            results, build_failed, malformed = grader.RUNNER_CLASSES["unittest"].parse(spec, driver_out, fake_report)
            verdict = malformed or grader.canary_verdict(results, canary, driver_out["commands"], False, build_failed)
            if verdict:
                result.hacks.append({"rule": "grader_tamper", "detail": verdict})
            result.results = {k: v for k, v in results.items() if k in set(spec.f2p) | set(spec.p2p)}
            result.f2p_passed, result.p2p_failed, result.resolved = grader.score(results, spec)
        else:
            gold = await asyncio.to_thread(gold_files, task, box.ws_dir)
            same = all((root / "repo" / p).is_file() and (root / "repo" / p).read_text(encoding="utf-8") == t
                       for p, t in gold.items())
            result.resolved = same
            result.f2p_passed = len(spec.f2p) if same else 0
            result.p2p_failed = 0 if same else len(spec.p2p) // 2
    finally:
        shutil.rmtree(root, ignore_errors=True)
    result.seconds = round(time.monotonic() - started, 3)
    return result


def gold_files(task, ws_dir: Path | None = None) -> dict:
    """{path: text} of every file the task's gold change writes."""
    if task.grading.get("gold_files") is not None:
        return dict(task.grading["gold_files"])
    from rlm_rl import editbox, grader

    ws_dir = ws_dir or task.workspace.ensure()
    root = Path(tempfile.mkdtemp(prefix="rlm-gold-"))
    try:
        subprocess.run(["cp", "-a", str(ws_dir / "tree"), str(root / "repo")], check=True)
        checked = grader.check_diff(task.grading["gold_patch"], task.grading["language"], ws_dir / "tree")
        ok, why = grader._safe_apply(root / "repo", task.grading["gold_patch"], root)
        if not ok:
            raise RuntimeError(f"gold patch doesn't apply: {why}")
        return {fp.path: (root / "repo" / fp.path).read_text(encoding="utf-8") for fp in grader.parse_patch(
            task.grading["gold_patch"]) if (root / "repo" / fp.path).is_file()}
    finally:
        shutil.rmtree(root, ignore_errors=True)