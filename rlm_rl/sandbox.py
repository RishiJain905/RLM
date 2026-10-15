"""One Docker container per episode: the REPL sandbox (spec §1 and §4, holes H2-H4, decisions 3a-3c).

The environment runs inside WSL2 (3b). Each episode gets a fresh container (3a) from a
Python-only image (3c): no network, a read-only root, one tmpfs scratch dir, no
capabilities, CPU, memory and pids limits, and an empty environment. The episode directory
(context.txt and the bridge socket) is the only thing mounted, read-only, and the socket is
the only way out. Generators, seeds, gold answers and graders never enter: a Sandbox is given
the context text and nothing else.

Failures (H5):
- A cell that runs out of compute time, or a REPL death (an exit, a fatal signal, running out
  of memory), restarts the container and tells the model. That is the model's outcome and
  gets scored. The container's PID 1 supervisor reports every REPL death, and model code
  can't signal it, so the attribution can't be faked.
- Anything else raises InfraError: the container can't start or restart, Docker fails, a
  sub-call fails on the host, or the container stops without the supervisor's report (it was
  stopped from outside). The caller discards the episode.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import datetime
import hashlib
import io
import json
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import bridge
from .bridge import Complete, EpisodeBridge, InfraError

# Python 3.12.15 on Debian, pulled 2026-10-11. The image build never touches the network.
BASE_IMAGE = "python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1"
IMAGE_REPO = "rlm-sandbox"
SANDBOX_UID = 10001
LABEL = "rlm.sandbox"
WORKER_PATH = Path(__file__).with_name("repl_worker.py")

# Mirrors repl_worker.py.
AUDIT_PREFIX = "@@rlm-audit "
TAMPER_EXIT_CODE = 86

RESULT_LINE_LIMIT = 4 * 1024 * 1024
MAX_EVENTS = 1000
START_ATTEMPTS = 2  # container launches before a sandbox that won't start is an infra error
LOG_LINES = 200

DOCKERFILE = """\
FROM {base}
RUN rm -rf /usr/local/lib/python3.12/site-packages/pip /usr/local/lib/python3.12/site-packages/pip-* \\
           /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.12 /usr/local/lib/python3.12/ensurepip/_bundled \\
 && groupadd --gid {uid} sandbox \\
 && useradd --uid {uid} --gid {uid} --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin sandbox \\
 && mkdir -p /opt/rlm /episode /scratch /work/repo
COPY repl_worker.py /opt/rlm/repl_worker.py
USER {uid}:{uid}
WORKDIR /scratch
ENTRYPOINT ["/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8", "HOME=/scratch", "TMPDIR=/scratch", \\
            "/usr/local/bin/python3", "-I", "-B", "/opt/rlm/repl_worker.py"]
"""


@dataclass(frozen=True)
class SandboxLimits:
    cpus: float = 1.0
    memory_mb: int = 512
    pids: int = 64
    scratch_mb: int = 64
    cell_timeout_s: float = bridge.CELL_TIMEOUT_S  # compute only: time blocked on the bridge doesn't count
    cell_wall_cap_s: float = 1800.0  # a cell stuck this long on sub-calls is an infra failure
    startup_timeout_s: float = 300.0  # generous: creating a container on the USB data drive can take minutes under load
    require_landlock: bool = True


@dataclass
class CellResult:
    stdout: str
    stdout_chars: int
    error: str | None
    submitted: bool = False
    answer: Any = None  # a JSON value: int, float, str, or a list of those
    answer_valid: bool = True
    stderr: str = ""
    restarted: bool = False
    crash: str | None = None  # "timeout", or why the REPL crashed
    repo_changed: bool | None = None  # edit episodes: the cell changed a file in /work/repo
    compute_s: float = 0.0
    blocked_s: float = 0.0
    wall_s: float = 0.0


# ----------------------------------------------------------------------------- image


def dockerfile(base: str = BASE_IMAGE) -> str:
    return DOCKERFILE.format(base=base, uid=SANDBOX_UID)


def image_tag(base: str = BASE_IMAGE) -> str:
    digest = hashlib.sha256(dockerfile(base).encode() + WORKER_PATH.read_bytes()).hexdigest()
    return f"{IMAGE_REPO}:{digest[:12]}"


def build_image(base: str = BASE_IMAGE, docker: str = "docker", force: bool = False) -> str:
    """Build the sandbox image from the Dockerfile and the worker alone, with no build network. Returns its tag."""
    tag = image_tag(base)
    exists = subprocess.run([docker, "image", "inspect", tag], capture_output=True, env=docker_cli_env()).returncode == 0
    if exists and not force:
        return tag
    context = io.BytesIO()
    with tarfile.open(fileobj=context, mode="w") as tar:
        for name, data in (("Dockerfile", dockerfile(base).encode()), ("repl_worker.py", WORKER_PATH.read_bytes())):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    subprocess.run([docker, "build", "--network", "none", "--pull=false", "-t", tag, "-"],
                   input=context.getvalue(), check=True, env=docker_cli_env())
    return tag


def docker_cli_env() -> dict:
    """The Docker CLI gets only what it needs to reach the daemon, never the trainer's keys."""
    keep = ("PATH", "HOME", "LANG", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "XDG_RUNTIME_DIR")
    return {k: os.environ[k] for k in keep if k in os.environ}


def default_root() -> Path:
    return Path(os.environ.get("RLM_SANDBOX_DIR") or f"/tmp/rlm-sandbox-{os.getuid()}")


def _private_root(root: Path) -> Path:
    """The parent of every episode directory: owned by us, not a symlink, mode 0700."""
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"{root} is not a directory owned by this user")
    os.chmod(root, 0o700)
    return root


def cleanup_stale(run_id: str | None = None, docker: str = "docker", root: Path | None = None) -> int:
    """Remove sandbox containers (all, or one run's) and episode directories left by a crashed process."""
    label = f"{LABEL}=1" if run_id is None else f"rlm.run={run_id}"
    ids = subprocess.run([docker, "ps", "-aq", "--filter", f"label={label}"], capture_output=True, text=True,
                         check=True, env=docker_cli_env()).stdout.split()
    if ids:
        subprocess.run([docker, "rm", "-f", *ids], capture_output=True, env=docker_cli_env())
    root = root or default_root()
    if run_id is None and root.is_dir():
        for child in root.glob("ep-*"):
            shutil.rmtree(child, ignore_errors=True)
    return len(ids)


def cleanup_finished(docker: str = "docker", older_than_s: float = 600) -> int:
    """Remove sandbox and grader containers that exited more than `older_than_s` ago (or were created and never
    started, an hour ago): a process killed outright never removes its own. Unlike cleanup_stale, safe beside running
    processes: their containers are running, or have just exited and are about to be removed."""
    ids = subprocess.run([docker, "ps", "-aq", "--filter", f"label={LABEL}=1", "--filter", "status=exited",
                          "--filter", "status=dead", "--filter", "status=created"], capture_output=True, text=True,
                         env=docker_cli_env()).stdout.split()
    if not ids:
        return 0
    out = subprocess.run([docker, "inspect", "--format", "{{.Id}} {{.State.Status}} {{.State.FinishedAt}} {{.Created}}",
                          *ids], capture_output=True, text=True, env=docker_cli_env()).stdout
    now, old = time.time(), []
    for line in out.splitlines():
        cid, status, finished, created = (line.split() + ["", "", "", ""])[:4]
        when, limit = (created, 3600.0) if status == "created" else (finished, older_than_s)
        try:
            t = datetime.datetime.strptime(when[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
        if now - t.timestamp() > limit:
            old.append(cid)
    if old:
        subprocess.run([docker, "rm", "-f", *old], capture_output=True, env=docker_cli_env())
    return len(old)


def run_args(docker: str, name: str, image: str, episode_dir: Path, limits: SandboxLimits, run_id: str,
             entrypoint: list[str] | None = None) -> list[str]:
    """The `docker run` command for one episode container. `entrypoint` replaces the REPL (tests only)."""
    lim = limits
    args = [
        docker, "run", "-i", "--pull", "never", "--name", name,
        "--label", f"{LABEL}=1", "--label", f"rlm.run={run_id}",
        "--network", "none", "--ipc", "none", "--read-only",
        "--tmpfs", f"/scratch:rw,nosuid,nodev,noexec,size={lim.scratch_mb}m,uid={SANDBOX_UID},gid={SANDBOX_UID},mode=0700",
        "--user", f"{SANDBOX_UID}:{SANDBOX_UID}", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--cpus", str(lim.cpus), "--memory", f"{lim.memory_mb}m", "--memory-swap", f"{lim.memory_mb}m",
        "--pids-limit", str(lim.pids), "--ulimit", "core=0", "--hostname", "sandbox", "--log-driver", "none",
        "--mount", f"type=bind,source={episode_dir},target=/episode,readonly",
    ]
    if entrypoint:
        return args + ["--entrypoint", entrypoint[0], image, *entrypoint[1:]]
    return args + [image]


def _signal_name(code: int) -> str:
    try:
        return signal.Signals(code - 128).name if code > 128 else f"exit code {code}"
    except ValueError:
        return f"exit code {code}"


# ----------------------------------------------------------------------------- sandbox

_NOTHING = object()


class Sandbox:
    """One episode's REPL: a fresh container whose only way out is the episode's bridge socket.

        async with Sandbox(context, complete, run_id="run-1") as box:
            result = await box.run_cell("print(len(context))")
    """

    def __init__(self, context: str, complete: Complete, *, limits: SandboxLimits = SandboxLimits(),
                 image: str | None = None, run_id: str = "adhoc", docker: str = "docker", root: Path | None = None,
                 max_subcalls: int = bridge.MAX_SUBCALLS, messages: bridge.CellMessages | None = None):
        self.limits = limits
        self.image = image or image_tag()
        self.run_id = run_id
        self.docker = docker
        # what the model is told after a timeout or a crash (spec §4, §12.2)
        self.messages = messages or bridge.cell_messages(limits=dataclasses.replace(
            bridge.READ_LIMITS, cell_timeout_s=int(limits.cell_timeout_s)))
        self.episode_id = secrets.token_hex(8)
        self.dir = _private_root(root or default_root()) / f"ep-{self.episode_id}"
        self.bridge = EpisodeBridge(str(self.dir / "bridge.sock"), complete, max_subcalls)
        self.events: list[dict] = []  # blocked or logged attempts, from every container of this episode
        self.events_dropped = 0
        self.log: collections.deque[str] = collections.deque(maxlen=LOG_LINES)  # other container output
        self.cells = 0
        self.timeouts = 0
        self.crashes = 0
        self.restarts = 0
        self.start_retries = 0  # launches that didn't come up in time and were tried again
        self.landlock_abi: int | None = None
        self.python_version: str | None = None
        self._context = context
        self._names: list[str] = []
        self._proc: asyncio.subprocess.Process | None = None
        self._messages: asyncio.Queue | None = None
        self._pumps: list[asyncio.Task] = []
        self._closed = False

    async def __aenter__(self) -> Sandbox:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    @property
    def infra_error(self) -> str | None:
        return self.bridge.infra_error

    async def start(self) -> None:
        if sys.platform != "linux":
            raise RuntimeError("the sandbox runs inside WSL2 (decision 3b); start the environment from Ubuntu")
        self.dir.mkdir(mode=0o755)
        os.chmod(self.dir, 0o755)  # the container user has to reach the context and the socket
        context_file = self.dir / "context.txt"
        context_file.write_bytes(self._context.encode("utf-8"))
        os.chmod(context_file, 0o644)
        await self.bridge.start()
        try:
            await self._launch()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._names:
            await self._docker("rm", "-f", *self._names)
        if self._proc is not None and self._proc.returncode is None:
            self._proc.kill()
            await self._proc.wait()
            if self._names:  # as in _kill: a container the daemon was still creating
                await self._docker("rm", "-f", *self._names)
        await asyncio.gather(*self._pumps, return_exceptions=True)
        await self.bridge.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    # ------------------------------------------------------------------ cells

    async def run_cell(self, code: str) -> CellResult:
        """Run one cell. Returns the model's outcome, or raises InfraError if the episode must be discarded."""
        if self._closed:
            raise RuntimeError("sandbox is closed")
        self._raise_if_infra()
        self.cells += 1
        pending = self._pending_death()
        if pending is not _NOTHING:  # the REPL died after the last cell, e.g. from a thread the model left running
            return await self._recover(pending, during_cell=False)

        started, blocked_start = time.monotonic(), self.bridge.blocked_seconds()
        try:
            self._proc.stdin.write(json.dumps({"code": code}).encode() + b"\n")
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the container is gone; its report or EOF is on the way

        infra = asyncio.ensure_future(self.bridge.infra_event.wait())
        reply = None
        try:
            while True:
                message = asyncio.ensure_future(self._messages.get())
                outcome = await self._wait(message, infra, started, blocked_start)
                if outcome != "message":
                    message.cancel()
                    break
                reply = message.result()
                if reply is None or "stdout" in reply or "repl_died" in reply:
                    break
                self.log.append(f"[ignored protocol message] {str(reply)[:200]}")
        finally:
            infra.cancel()
        wall = time.monotonic() - started
        blocked = self.bridge.blocked_seconds() - blocked_start
        timing = {"compute_s": wall - blocked, "blocked_s": blocked, "wall_s": wall}

        if outcome == "infra":
            await self._kill()
            self._raise_if_infra()
        if outcome == "wall":
            await self._kill()
            self.bridge.flag_infra(f"a cell waited over {self.limits.cell_wall_cap_s:.0f} s on sub-calls")
            self._raise_if_infra()
        if outcome == "timeout":
            self.timeouts += 1
            await self._kill()
            await self._restart()
            return CellResult("", 0, self.messages.timeout, restarted=True, crash="timeout", **timing)
        if reply is None or "repl_died" in reply:
            return await self._recover(reply, during_cell=True, timing=timing)

        result = self._parse(reply)
        self._raise_if_infra()  # a sub-call failed on the host while the cell was finishing
        if result is None:
            self.crashes += 1
            await self._kill()
            await self._restart()
            return CellResult("", 0, self.messages.crash.format(reason="it returned an unreadable result"),
                              restarted=True, crash="unreadable result", **timing)
        result.compute_s, result.blocked_s, result.wall_s = timing["compute_s"], timing["blocked_s"], timing["wall_s"]
        return result

    async def _wait(self, message: asyncio.Future, infra: asyncio.Future, started: float, blocked_start: float) -> str:
        while True:
            wall = time.monotonic() - started
            compute = wall - (self.bridge.blocked_seconds() - blocked_start)
            if wall >= self.limits.cell_wall_cap_s:
                return "wall"
            remaining = min(self.limits.cell_timeout_s - compute, self.limits.cell_wall_cap_s - wall)
            if remaining <= 0:
                return "timeout"
            # Compute time can't grow faster than wall time, so waiting `remaining` never overshoots.
            done, _ = await asyncio.wait({message, infra}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if infra in done:
                return "infra"
            if message in done:
                return "message"

    def _parse(self, raw: dict) -> CellResult | None:
        try:
            stdout, error, stderr = raw["stdout"], raw["error"], raw.get("stderr", "")
            if not isinstance(stdout, str) or not (error is None or isinstance(error, str)) or not isinstance(stderr, str):
                return None
            stdout_chars = int(raw["stdout_chars"])
            submitted = raw["submitted"] is True
            answer = raw.get("answer")
        except (KeyError, TypeError, ValueError):
            return None
        answer_valid = not submitted or bridge.valid_answer(answer)
        if isinstance(answer, str):
            answer = bridge.clean_text(answer)
        elif isinstance(answer, list):
            answer = [bridge.clean_text(a) if isinstance(a, str) else a for a in answer]
        return CellResult(
            stdout=bridge.clean_text(stdout[: bridge.OUTPUT_MAX_CHARS]),
            stdout_chars=max(stdout_chars, len(stdout)),
            error=None if error is None else bridge.clean_text(error[: 2 * bridge.OUTPUT_MAX_CHARS]),
            submitted=submitted,
            answer=answer if submitted and answer_valid else None,
            answer_valid=answer_valid,
            stderr=bridge.clean_text(stderr[:4000]),
        )

    # ------------------------------------------------------------------ crashes and restarts

    def _pending_death(self):
        """A death report or EOF that arrived between cells, if any. Anything else waiting is stale and dropped."""
        while not self._messages.empty():
            message = self._messages.get_nowait()
            if message is None or "repl_died" in message:
                return message
        return _NOTHING

    async def _recover(self, message: dict | None, during_cell: bool, timing: dict | None = None) -> CellResult:
        """The REPL died. Restart it and tell the model, or raise InfraError if the container was stopped from outside."""
        if message is None:
            reason = await self._container_death_reason()
        else:
            reason = self._repl_death_reason(message["repl_died"])
        self.crashes += 1
        await self._kill()
        await self._restart()
        error = self.messages.crash.format(reason=reason)
        if not during_cell:
            error += self.messages.not_run
        return CellResult("", 0, error, restarted=True, crash=reason, **(timing or {}))

    def _repl_death_reason(self, died) -> str:
        """The supervisor's report. Model code can't signal the supervisor, so every REPL death is the model's."""
        died = died if isinstance(died, dict) else {}
        if died.get("oom"):
            return f"out of memory: the {self.limits.memory_mb} MB limit was reached"
        if "signal" in died:
            return f"killed by {str(died['signal'])[:20]}"
        code = died.get("exit_code")
        if code == TAMPER_EXIT_CODE:
            self._add_event({"kind": "tamper", "event": "stderr", "detail": "the audit log was closed or redirected",
                             "blocked": True})
            return "the sandbox log was tampered with"
        return f"exit code {str(code)[:20]}"

    async def _container_death_reason(self) -> str:
        """The container ended without the supervisor's report: out of memory, or stopped from outside (infra)."""
        try:
            await asyncio.wait_for(self._proc.wait(), 10)
        except asyncio.TimeoutError:
            pass
        rc, out, _ = await self._docker("inspect", "--format", "{{.State.OOMKilled}} {{.State.ExitCode}}", self._names[-1])
        fields = out.split()
        if rc == 0 and fields and fields[0] == "true":
            return f"out of memory: the {self.limits.memory_mb} MB limit was reached"
        exit_code = int(fields[1]) if rc == 0 and len(fields) > 1 and fields[1].lstrip("-").isdigit() else None
        detail = _signal_name(exit_code) if exit_code is not None else "state unavailable"
        self.bridge.flag_infra(f"the container stopped from outside the REPL ({detail})")
        self._raise_if_infra()

    async def _restart(self) -> None:
        self.restarts += 1
        try:
            await self._launch()
        except InfraError:
            raise
        except Exception as exc:
            self.bridge.flag_infra(f"sandbox restart failed: {exc!r}")
            self._raise_if_infra()

    async def _kill(self) -> None:
        await self._docker("rm", "-f", self._names[-1])
        if self._proc.returncode is None:
            try:
                await asyncio.wait_for(self._proc.wait(), 10)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
            # A launch that timed out may not have its container yet: on a slow drive the daemon finishes creating it
            # after the first rm, and it stays "Created" for good. Once the client is gone, remove it again.
            await self._docker("rm", "-f", self._names[-1])
        await asyncio.gather(*self._pumps, return_exceptions=True)

    # ------------------------------------------------------------------ docker

    async def _launch(self) -> None:
        for attempt in range(START_ATTEMPTS):  # a container that doesn't come up is tried again before it's infra
            name = f"rlm-ep-{self.episode_id}-{len(self._names)}"
            self._names.append(name)
            args = self._run_args(name)
            self._proc = await asyncio.create_subprocess_exec(
                *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=docker_cli_env(), limit=RESULT_LINE_LIMIT)
            self._messages = asyncio.Queue()
            self._pumps = [asyncio.create_task(self._pump_stdout(self._proc.stdout, self._messages)),
                           asyncio.create_task(self._pump_stderr(self._proc.stderr))]
            try:
                ready = await asyncio.wait_for(self._messages.get(), self.limits.startup_timeout_s)
            except asyncio.TimeoutError:
                ready = None
            if ready and ready.get("ready") is True:
                break
            await self._kill()
            if attempt + 1 == START_ATTEMPTS:
                detail = " | ".join(list(self.log)[-5:]) or "no output"
                raise InfraError(f"sandbox failed to start ({START_ATTEMPTS} attempts): {detail}")
            self.start_retries += 1
        self.landlock_abi = int(ready.get("landlock") or 0)
        self.python_version = ready.get("python")
        if self.limits.require_landlock and self.landlock_abi < 1:
            await self._kill()
            raise InfraError(f"Landlock is unavailable in the sandbox: {ready.get('landlock_error') or 'kernel ABI 0'}")

    async def _pump_stdout(self, stream: asyncio.StreamReader, queue: asyncio.Queue) -> None:
        """Protocol messages, one JSON object per line, then None at EOF. Other lines are logged and skipped."""
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                self.log.append("[over-long protocol line dropped]")
                continue
            if not line:
                queue.put_nowait(None)
                return
            try:
                message = json.loads(line)
            except ValueError:
                message = None
            if isinstance(message, dict):
                queue.put_nowait(message)
            elif line.strip():
                self.log.append(f"[unreadable protocol line] {line[:200]!r}")

    async def _pump_stderr(self, stream: asyncio.StreamReader) -> None:
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                self.log.append("[over-long stderr line dropped]")
                continue
            if not line:
                return
            text = line.decode("utf-8", "replace").rstrip("\n")
            if text.startswith(AUDIT_PREFIX):
                try:
                    event = json.loads(text[len(AUDIT_PREFIX):])
                except ValueError:
                    event = None
                self._add_event(event if isinstance(event, dict) else
                                {"kind": "malformed", "event": "audit", "detail": text[:300], "blocked": False})
            else:
                self.log.append(text[:500])

    def _run_args(self, name: str) -> list[str]:
        """The `docker run` command for this episode's next container. The edit sandbox overrides it."""
        return run_args(self.docker, name, self.image, self.dir, self.limits, self.run_id)

    def _add_event(self, event: dict) -> None:
        event["cell"] = self.cells  # the cell running when it arrived (the host's count, not the REPL's)
        if len(self.events) < MAX_EVENTS:
            self.events.append(event)
        else:
            self.events_dropped += 1

    async def _docker(self, *args: str, timeout: float = 60.0) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(self.docker, *args, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, env=docker_cli_env())
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, "", "timed out"
        return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

    def _raise_if_infra(self) -> None:
        if self.bridge.infra_error:
            raise InfraError(self.bridge.infra_error)
