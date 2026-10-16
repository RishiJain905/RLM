"""The REPL that runs inside the sandbox container, one per episode (spec §1, H2-H4).

Standard library only, and it imports nothing from the project: the image holds Python and
this file, nothing else. The host mounts the episode directory read-only at /episode
(context.txt and bridge.sock) and talks to the REPL over stdin and stdout: one JSON cell per
line in ({"code": ...}), one JSON result per line out.

Two processes run in the container. PID 1 is a small supervisor that ignores every signal it
can, forks the REPL and waits. If the REPL dies, the supervisor reports how (an exit code, a
signal, the out-of-memory killer) as `{"repl_died": ...}` and exits. Model code can't signal
PID 1, so any REPL death is the model's doing, and a container that dies without that report
was stopped from outside: an infra error (spec §1.2).

Before any model code runs, the REPL locks itself down in two layers on top of the
container:
- Landlock: no file reads outside /usr, /etc, /opt/rlm, /episode and /scratch, no writes
  outside /scratch, and no execve at all. /proc, /sys and /dev become unreadable.
- An audit hook (PEP 578) that blocks network, file, process, signal and ctypes attempts and
  logs each one to stderr as `@@rlm-audit {json}`, for the escape_attempt detector (spec §7).
The container itself (--network none, read-only root, no host files, empty environment) is
the real boundary. The hook is a tripwire and gives the model readable errors.

Edit mode (`--edit`, spec §12.4) is for the edit families. The REPL runs on our own Python,
mounted read-only at /opt/rlm/python, inside the repository's image, with the repository at
/work/repo. It differs from read mode in four ways:
- Landlock lets every file in the container be read and every program be run, so the image's
  own toolchain and test runner work through subprocess. Writes stay limited to /work/repo,
  /scratch, /tmp and the image's cache folders given with `--writable`.
- The audit hook still blocks network sockets, ctypes and writes elsewhere, but lets programs
  run and signals be sent. It logs, without blocking, each program started (`exec`, with its
  arguments), the first read of /work/repo in each cell (`repo_read`) and writes to it
  (`repo_write`), which the host turns into the edit metrics (spec §12.9). A blocked network
  attempt is logged as `network`; the host counts it as `network_try` (spec §12.6).
- During a cell, file descriptors 1 and 2 point at a pipe that feeds the cell's output, so
  what a program the model runs prints reaches the model. The audit log moves to a private
  descriptor.
- The REPL starts in /work/repo, with the seven coding tools of repl_tools.py (mounted next to
  this file) in its namespace. Each tool call is logged as `tool`, and writes and programs are
  marked as made by a tool or not, for the edit metrics (`raw_io_share`, spec §12.9).
"""

import contextlib
import io
import json
import math
import os
import signal
import socket
import sys
import threading
import time
import traceback

# Must match rlm_rl/bridge.py; tests/test_contract.py checks.
API_NAMES = ("context", "llm", "llm_batch", "submit")
WIRE_OPS = ("llm", "llm_batch")
OUTPUT_MAX_CHARS = 8000  # the largest cap; the host cuts each episode to its own (prompt v1: 2,000)
TRACEBACK_LINES = 6
SUBCALL_MAX_CHARS = 24_000
SUBCALL_TRUNCATION_MARK = "[prompt truncated]"
SUBMIT_MAX_BYTES = 100_000

EPISODE_DIR = "/episode"
CONTEXT_PATH = "/episode/context.txt"
SOCKET_PATH = "/episode/bridge.sock"
SCRATCH_DIR = "/scratch"
WORKER_DIR = "/opt/rlm"
EDIT_REPO = "/work/repo"
EDIT_TMP = "/tmp"
REPO_WRITES_LOGGED = 20  # repo_write events per cell; enough to order edits and test runs

AUDIT_PREFIX = "@@rlm-audit "
TAMPER_EXIT_CODE = 86  # stderr was closed or redirected, so blocked attempts could no longer be logged
STDERR_KEEP_CHARS = 4000

# Names that only exist on the host. Importing them fails anyway; trying is logged.
HOST_MODULES = frozenset({
    "rlm_rl", "scripts", "repl_worker", "art", "wandb", "openai", "dotenv", "httpx",
    "ticket_tasks", "rlm_env", "rollout", "sdpo", "shaping",
})
# Private modules the standard library itself tries to import (sysconfig, platform), which
# sys.stdlib_module_names doesn't list. Not worth a log line.
STDLIB_EXTRAS = ("_sysconfigdata", "_wmi")

_state = {"submitted": False, "answer": None}

# ----------------------------------------------------------------------------- bridge client

# Time spent waiting on the bridge, counted once however many threads wait: the host's cell limit counts compute
# only, so the coding tools' time_left leaves it out too.
_waiting = {"calls": 0, "since": 0.0, "total": 0.0}
_waiting_lock = threading.Lock()


def _bridge_seconds() -> float:
    with _waiting_lock:
        return _waiting["total"] + (time.monotonic() - _waiting["since"] if _waiting["calls"] else 0.0)


def _bridge(op: str, prompts: list) -> list:
    """One request over the episode's Unix socket. A fresh connection per call, so threads can call in parallel."""
    with _waiting_lock:
        if not _waiting["calls"]:
            _waiting["since"] = time.monotonic()
        _waiting["calls"] += 1
    try:
        return _bridge_call(op, prompts)
    finally:
        with _waiting_lock:
            _waiting["calls"] -= 1
            if not _waiting["calls"]:
                _waiting["total"] += time.monotonic() - _waiting["since"]


def _bridge_call(op: str, prompts: list) -> list:
    request = (json.dumps({"op": op, "prompts": prompts}) + "\n").encode()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.connect(SOCKET_PATH)
            conn.sendall(request)
            with conn.makefile("rb") as stream:
                line = stream.readline()
    except OSError as exc:
        raise RuntimeError(f"sub-call bridge unreachable: {exc}") from None
    if not line:
        raise RuntimeError("sub-call bridge closed the connection")
    reply = json.loads(line)
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["outputs"]


def _truncate(prompt: str) -> str:
    if len(prompt) > SUBCALL_MAX_CHARS:
        return prompt[:SUBCALL_MAX_CHARS] + "\n" + SUBCALL_TRUNCATION_MARK
    return prompt


def llm(prompt: str) -> str:
    """Ask a language model one question. It can read about 20,000 characters per call."""
    return _bridge("llm", [_truncate(str(prompt))])[0]


def llm_batch(prompts: list[str]) -> list[str]:
    """Ask several prompts in parallel; each prompt counts as one sub-call. Outputs come back in input order."""
    if isinstance(prompts, (str, bytes)):
        raise TypeError("llm_batch expects a list of prompts, not a single string")
    return _bridge("llm_batch", [_truncate(str(p)) for p in prompts])


def _plain(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return str(value)
    return None


def normalize_answer(answer):
    """int, float and str pass through; a list, tuple or set becomes a list of those; anything else becomes str()."""
    plain = _plain(answer)
    if plain is not None:
        return plain
    if isinstance(answer, (list, tuple, set, frozenset)):
        items = [str(v) if _plain(v) is None else _plain(v) for v in answer]
        if isinstance(answer, (set, frozenset)):
            items.sort(key=lambda v: (type(v).__name__, str(v)))
        return items
    return str(answer)


def submit(answer) -> None:
    """Give your final answer and end the episode."""
    value = normalize_answer(answer)
    size = len(json.dumps(value))
    if size > SUBMIT_MAX_BYTES:
        raise ValueError(f"answer too large ({size:,} characters as JSON; the limit is {SUBMIT_MAX_BYTES:,})")
    _state["submitted"] = True
    _state["answer"] = value


def make_namespace(context: str) -> dict:
    return {"context": context, "llm": llm, "llm_batch": llm_batch, "submit": submit, "__name__": "__repl__"}


# ----------------------------------------------------------------------------- Landlock

_FS_RIGHTS = ("EXECUTE", "WRITE_FILE", "READ_FILE", "READ_DIR", "REMOVE_DIR", "REMOVE_FILE", "MAKE_CHAR",
              "MAKE_DIR", "MAKE_REG", "MAKE_SOCK", "MAKE_FIFO", "MAKE_BLOCK", "MAKE_SYM", "REFER", "TRUNCATE")
FS = {name: 1 << i for i, name in enumerate(_FS_RIGHTS)}
_READ_TREE = FS["READ_FILE"] | FS["READ_DIR"]
_WRITE_TREE = (_READ_TREE | FS["WRITE_FILE"] | FS["REMOVE_DIR"] | FS["REMOVE_FILE"] | FS["MAKE_DIR"]
               | FS["MAKE_REG"] | FS["REFER"] | FS["TRUNCATE"])
# Edit mode: build tools make symlinks, sockets and fifos, and run what they build (spec §12.4).
_EDIT_WRITE_TREE = _WRITE_TREE | FS["MAKE_SYM"] | FS["MAKE_SOCK"] | FS["MAKE_FIFO"] | FS["EXECUTE"]

# Path -> rights. Anything not listed (/proc, /sys, /dev, /tmp, /var, /run, /home, /root, /mnt) is denied,
# and no path gets EXECUTE, so execve fails everywhere.
LANDLOCK_RULES = {
    "/usr": _READ_TREE,  # Python, its shared libraries and zoneinfo (/bin and /lib link here)
    "/etc": _READ_TREE,  # the dynamic loader and libc read ld.so.cache, localtime and nsswitch.conf
    WORKER_DIR: _READ_TREE,
    EPISODE_DIR: _READ_TREE,
    SCRATCH_DIR: _WRITE_TREE,
    "/dev/null": FS["READ_FILE"] | FS["WRITE_FILE"] | FS["TRUNCATE"],
    "/dev/zero": FS["READ_FILE"],
    "/dev/urandom": FS["READ_FILE"],
}


def edit_landlock_rules(writable) -> dict:
    """Edit mode: read and run anything in the container, write only under `writable` (spec §12.4)."""
    rules = {"/": _READ_TREE | FS["EXECUTE"],
             "/dev/null": FS["READ_FILE"] | FS["WRITE_FILE"] | FS["TRUNCATE"]}  # a file: no directory rights
    for path in writable:
        rules[path] = _EDIT_WRITE_TREE
    return rules


def apply_landlock(rules: dict | None = None) -> int:
    """Restrict this process and every child to `rules` (LANDLOCK_RULES by default). Returns the Landlock ABI applied,
    or 0 if the kernel has none."""
    import ctypes

    rules = LANDLOCK_RULES if rules is None else rules

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    create_ruleset, add_rule, restrict_self = 444, 445, 446  # the same numbers on every architecture

    def call(*args):
        result = libc.syscall(*[a if isinstance(a, ctypes._SimpleCData) or not isinstance(a, int) else ctypes.c_long(a)
                                for a in args])
        if result < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"landlock syscall {args[0]}: {os.strerror(errno)}")
        return result

    try:
        abi = call(create_ruleset, None, 0, 1)  # LANDLOCK_CREATE_RULESET_VERSION
    except OSError:
        return 0
    handled = (1 << 13) - 1  # ABI 1: EXECUTE .. MAKE_SYM
    if abi >= 2:
        handled |= FS["REFER"]
    if abi >= 3:
        handled |= FS["TRUNCATE"]

    class RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathBeneathAttr(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    attr = RulesetAttr(handled)
    ruleset = call(create_ruleset, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    try:
        for path, rights in rules.items():
            try:
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except FileNotFoundError:
                continue
            try:
                rule = PathBeneathAttr(rights & handled, fd)
                call(add_rule, ruleset, 1, ctypes.byref(rule), 0)  # LANDLOCK_RULE_PATH_BENEATH
            finally:
                os.close(fd)
        if libc.prctl(38, ctypes.c_ulong(1), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS)")
        call(restrict_self, ruleset, 0)
    finally:
        os.close(ruleset)
    return abi


# ----------------------------------------------------------------------------- audit hook

_PY_DIR = os.path.dirname(os.__file__)
READ_PREFIXES = (SCRATCH_DIR, EPISODE_DIR, WORKER_DIR, _PY_DIR, "/usr/share/zoneinfo")
READ_FILES = frozenset({"/dev/null", "/dev/zero", "/dev/urandom"})
WRITE_PREFIXES = (SCRATCH_DIR,)
WRITE_FILES = frozenset({"/dev/null"})
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

_stderr_id = None
_audit_fd = 2  # edit mode moves the audit log to a private descriptor (fds 1 and 2 feed the cell output)
# Per-cell markers, reset by run_cell. `io` counts /work/repo writes and programs started, by a tool or not.
_edit = {"on": False, "repo_read": False, "repo_writes": 0, "last_popen": None,
         "io": {"tool_writes": 0, "raw_writes": 0, "tool_execs": 0, "raw_execs": 0}}
_tools = None  # edit mode: the repl_tools module
_cell = {"timeout": None, "start": 0.0, "bridge_at_start": 0.0}


def _log(kind: str, event: str, detail, blocked: bool = True, **extra) -> None:
    """Write one audit line to the audit descriptor. If it was tampered with, the REPL exits instead (TAMPER_EXIT_CODE)."""
    try:
        st = os.fstat(_audit_fd)
        if (st.st_dev, st.st_ino) != _stderr_id:
            os._exit(TAMPER_EXIT_CODE)
        data = (AUDIT_PREFIX + json.dumps({"kind": kind, "event": event, "detail": repr(detail)[:300],
                                           "blocked": blocked, **extra}) + "\n").encode()
        while data:
            data = data[os.write(_audit_fd, data):]
    except Exception:
        os._exit(TAMPER_EXIT_CODE)


def _block(kind: str, event: str, detail, exc: type = PermissionError, shown=None) -> None:
    """Log the attempt, then fail it with an error the model can read."""
    _log(kind, event, detail)
    shown = str(detail if shown is None else shown)[:300]
    if exc is PermissionError:
        raise PermissionError(13, "Permission denied (sandbox)", shown)
    raise exc(f"{shown} is not available in the sandbox")


def _abspath(path) -> str | None:
    if path is None or isinstance(path, int):
        return None  # an already-open descriptor
    try:
        path = os.fsdecode(path)
        return os.path.normpath(os.path.join(os.getcwd(), path))
    except Exception:
        return "<unresolvable path>"


def _under(path: str, prefixes) -> bool:
    return any(path == p or path.startswith(p + "/") for p in prefixes)


def _check_read(event: str, path) -> None:
    p = _abspath(path)
    if p is not None and p not in READ_FILES and not _under(p, READ_PREFIXES):
        _block("file", event, p)


def _check_write(event: str, path) -> None:
    p = _abspath(path)
    if p is not None and p not in WRITE_FILES and not _under(p, WRITE_PREFIXES):
        _block("file", event, p)


def _on_open(args) -> None:
    path, mode, flags = (tuple(args) + (None, None, None))[:3]
    writing = isinstance(flags, int) and flags & _WRITE_FLAGS
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        writing = True
    (_check_write if writing else _check_read)("open", path)


def _on_socket_new(args) -> None:
    family = args[1] if len(args) > 1 else None
    if family != socket.AF_UNIX:
        _block("network", "socket.__new__", f"family={family}", shown="network sockets")


def _on_connect(args) -> None:
    address = args[1] if len(args) > 1 else None
    if address != SOCKET_PATH:
        _block("network", "socket.connect", address)


def _on_import(args) -> None:
    name = args[0] if args else ""
    if str(name).split(".")[0] in ("ctypes", "_ctypes"):
        _block("ctypes", "import", name, ImportError)


class _ImportWatch:
    """First on sys.meta_path: logs attempts to import host or non-stdlib modules, however the import is
    spelled (importlib.import_module raises no "import" audit event). It never finds anything itself."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        top = name.partition(".")[0]
        if top in HOST_MODULES:
            _log("host_import", "import", name, blocked=False)
        elif top not in sys.stdlib_module_names and not top.startswith(STDLIB_EXTRAS):
            _log("nonstdlib_import", "import", name, blocked=False)
        return None


def _reader(event):
    return lambda args: _check_read(event, args[0] if args else None)


def _writer(event, index=0):
    return lambda args: _check_write(event, args[index] if len(args) > index else None)


def _blocker(kind, event):
    return lambda args: _block(kind, event, args, shown=event)


def _on_addaudithook(args) -> None:
    _log("tamper", "sys.addaudithook", "")
    raise RuntimeError("sys.addaudithook is not available in the sandbox")  # RuntimeError makes Python skip the new hook


_HANDLERS = {
    "open": _on_open,
    "os.listdir": _reader("os.listdir"),
    "os.scandir": _reader("os.scandir"),
    "os.listxattr": _reader("os.listxattr"),
    "os.getxattr": _reader("os.getxattr"),
    "os.mkdir": _writer("os.mkdir"),
    "os.remove": _writer("os.remove"),
    "os.rmdir": _writer("os.rmdir"),
    "os.rename": lambda args: (_check_write("os.rename", args[0]), _check_write("os.rename", args[1])),
    "os.truncate": _writer("os.truncate"),
    "os.chmod": _writer("os.chmod"),
    "os.chown": _writer("os.chown"),
    "os.utime": _writer("os.utime"),
    "os.setxattr": _writer("os.setxattr"),
    "os.removexattr": _writer("os.removexattr"),
    "os.link": _blocker("file", "os.link"),
    "os.symlink": _blocker("file", "os.symlink"),
    "socket.__new__": _on_socket_new,
    "socket.connect": _on_connect,
    "socket.bind": _blocker("network", "socket.bind"),
    "socket.sendto": _blocker("network", "socket.sendto"),
    "socket.getaddrinfo": _blocker("network", "socket.getaddrinfo"),
    "socket.gethostbyname": _blocker("network", "socket.gethostbyname"),
    "socket.gethostbyaddr": _blocker("network", "socket.gethostbyaddr"),
    "socket.getnameinfo": _blocker("network", "socket.getnameinfo"),
    "socket.getservbyname": _blocker("network", "socket.getservbyname"),
    "socket.getservbyport": _blocker("network", "socket.getservbyport"),
    "socket.sethostname": _blocker("network", "socket.sethostname"),
    "subprocess.Popen": _blocker("process", "subprocess.Popen"),
    "os.system": _blocker("process", "os.system"),
    "os.exec": _blocker("process", "os.exec"),
    "os.posix_spawn": _blocker("process", "os.posix_spawn"),
    "os.spawn": _blocker("process", "os.spawn"),
    "os.fork": _blocker("process", "os.fork"),
    "os.forkpty": _blocker("process", "os.forkpty"),
    "os.kill": _blocker("signal", "os.kill"),
    "os.killpg": _blocker("signal", "os.killpg"),
    "signal.pthread_kill": _blocker("signal", "signal.pthread_kill"),
    "resource.setrlimit": _blocker("signal", "resource.setrlimit"),
    "resource.prlimit": _blocker("signal", "resource.prlimit"),
    "gc.get_objects": _blocker("tamper", "gc.get_objects"),
    "gc.get_referrers": _blocker("tamper", "gc.get_referrers"),
    "gc.get_referents": _blocker("tamper", "gc.get_referents"),
    "sys.addaudithook": _on_addaudithook,
    "import": _on_import,
}


# ----------------------------------------------------------------------------- edit mode (spec §12.4)


def _argv(value) -> list:
    """A program's arguments as short strings, for the exec log."""
    if isinstance(value, (str, bytes, os.PathLike)):
        value = [value]
    try:
        items = list(value)[:40]
    except TypeError:
        items = [value]
    out = []
    for a in items:
        try:
            out.append((os.fsdecode(a) if isinstance(a, (str, bytes, os.PathLike)) else str(a))[:200])
        except Exception:
            out.append("?")
    return out


def _by_tool() -> bool:
    """Whether a coding tool (repl_tools) is running on this thread right now."""
    return _tools is not None and _tools.active()


def _edit_exec(event: str, index: int):
    """Log a program start with its arguments; never block it (the image's toolchain runs through subprocess)."""
    def handler(args):
        args = tuple(args)
        value = args[index] if len(args) > index and args[index] is not None else (args[0] if args else "")
        argv = _argv(value)
        if event == "subprocess.Popen":
            _edit["last_popen"] = argv
        elif event == "os.posix_spawn" and argv == _edit["last_popen"]:
            _edit["last_popen"] = None  # subprocess.Popen starting it through posix_spawn: already logged
            return
        tool = _by_tool()
        _edit["io"]["tool_execs" if tool else "raw_execs"] += 1
        _log("exec", event, " ".join(argv)[:300], blocked=False, argv=argv, tool=tool)
    return handler


def _in_repo(path) -> bool:
    return isinstance(path, str) and _under(path, (EDIT_REPO,))


def _note_repo_read(event: str, path) -> None:
    p = _abspath(path)
    if _in_repo(p) and not _edit["repo_read"]:
        _edit["repo_read"] = True
        _log("repo_read", event, p, blocked=False)


def _note_write(event: str, path) -> None:
    """Writes outside the writable folders are blocked as in read mode; writes to /work/repo are logged."""
    _check_write(event, path)
    p = _abspath(path)
    if not _in_repo(p):
        return
    tool = _by_tool()
    _edit["io"]["tool_writes" if tool else "raw_writes"] += 1
    if _edit["repo_writes"] < REPO_WRITES_LOGGED:
        _edit["repo_writes"] += 1
        _log("repo_write", event, p, blocked=False, tool=tool)


def _edit_on_open(args) -> None:
    path, mode, flags = (tuple(args) + (None, None, None))[:3]
    writing = isinstance(flags, int) and flags & _WRITE_FLAGS
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        writing = True
    if writing:
        _note_write("open", path)
    else:
        _note_repo_read("open", path)


def _edit_reader(event):
    return lambda args: _note_repo_read(event, args[0] if args else None)


def _edit_writer(event, index=0):
    return lambda args: _note_write(event, args[index] if len(args) > index else None)


def _allow(args) -> None:
    return None


_EDIT_HANDLERS = {
    **_HANDLERS,
    "open": _edit_on_open,
    "os.listdir": _edit_reader("os.listdir"),
    "os.scandir": _edit_reader("os.scandir"),
    "os.listxattr": _edit_reader("os.listxattr"),
    "os.getxattr": _edit_reader("os.getxattr"),
    "os.mkdir": _edit_writer("os.mkdir"),
    "os.remove": _edit_writer("os.remove"),
    "os.rmdir": _edit_writer("os.rmdir"),
    "os.rename": lambda args: (_note_write("os.rename", args[0]), _note_write("os.rename", args[1])),
    "os.truncate": _edit_writer("os.truncate"),
    "os.chmod": _edit_writer("os.chmod"),
    "os.chown": _edit_writer("os.chown"),
    "os.utime": _edit_writer("os.utime"),
    "os.setxattr": _edit_writer("os.setxattr"),
    "os.removexattr": _edit_writer("os.removexattr"),
    "os.link": _edit_writer("os.link", 1),
    "os.symlink": _edit_writer("os.symlink", 1),
    "subprocess.Popen": _edit_exec("subprocess.Popen", 1),
    "os.system": _edit_exec("os.system", 0),
    "os.exec": _edit_exec("os.exec", 1),
    "os.posix_spawn": _edit_exec("os.posix_spawn", 1),
    "os.spawn": _edit_exec("os.spawn", 2),
    "os.fork": _allow,
    "os.forkpty": _allow,
    "os.kill": _allow,
    "os.killpg": _allow,
    "signal.pthread_kill": _allow,
    "resource.setrlimit": _allow,
    "resource.prlimit": _allow,
}

_handlers = _HANDLERS


def _audit_hook(event: str, args) -> None:
    handler = _handlers.get(event)
    if handler is not None:
        handler(args)
    elif event.startswith("ctypes."):
        _block("ctypes", event, args, shown=event)


def install_audit_hook(edit: bool = False) -> None:
    global _stderr_id, _handlers
    st = os.fstat(_audit_fd)
    _stderr_id = (st.st_dev, st.st_ino)
    _handlers = _EDIT_HANDLERS if edit else _HANDLERS
    sys.meta_path.insert(0, _ImportWatch)
    sys.addaudithook(_audit_hook)


# ----------------------------------------------------------------------------- cells


class _Capped(io.TextIOBase):
    """A stdout stand-in that keeps the first `cap` characters and counts the rest."""

    def __init__(self, cap: int):
        self.cap, self.total, self._parts, self._kept = cap, 0, [], 0
        self._lock = threading.Lock()  # edit mode: the fd pump writes from another thread

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def write(self, s):
        if not isinstance(s, str):
            raise TypeError(f"write() argument must be str, not {type(s).__name__}")
        with self._lock:
            self.total += len(s)
            if self._kept < self.cap:
                piece = s[: self.cap - self._kept]
                self._parts.append(piece)
                self._kept += len(piece)
        return len(s)

    def getvalue(self) -> str:
        with self._lock:
            return "".join(self._parts)


class _FdCapture:
    """Edit mode: during a cell, fds 1 and 2 are a pipe whose output goes to the cell's stdout, so programs the model
    runs without capturing their output still show it. A program left running in the background keeps the pipe
    open; its later output is dropped, since the cell is over."""

    DRAIN_S = 0.5

    def __init__(self, sink: "_Capped"):
        self._sink = sink
        read_fd, write_fd = os.pipe()
        self._read_fd = read_fd
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        self._thread = threading.Thread(target=self._pump, name="rlm-fd-capture", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        import codecs

        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                data = os.read(self._read_fd, 65536)
                if not data:
                    break
                text = decoder.decode(data)
                sink = self._sink
                if sink is not None and text:
                    sink.write(text)
        except OSError:
            pass
        finally:
            try:
                os.close(self._read_fd)
            except OSError:
                pass

    def stop(self) -> None:
        devnull = os.open("/dev/null", os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        self._thread.join(self.DRAIN_S)  # every writer that exited has closed its end; a background one may not have
        self._sink = None


def _format_error(exc: BaseException) -> str:
    tb = exc.__traceback__.tb_next if exc.__traceback__ is not None else None  # drop the worker's own frame
    lines = "".join(traceback.format_exception(type(exc), exc, tb)).splitlines()
    text = "\n".join(lines[-TRACEBACK_LINES:])
    if len(text) > OUTPUT_MAX_CHARS:
        text = text[:OUTPUT_MAX_CHARS] + f"\n[... {len(text) - OUTPUT_MAX_CHARS} more characters truncated]"
    return text


def cell_seconds_left() -> float:
    """Compute seconds left in the running cell (time waiting on the bridge doesn't count, as on the host)."""
    if not _cell["timeout"]:
        return float("inf")
    used = time.monotonic() - _cell["start"] - (_bridge_seconds() - _cell["bridge_at_start"])
    return _cell["timeout"] - used


def _log_tool_call(tool: str, ok: bool, seconds: float, target: str) -> None:
    """repl_tools' on_call: one `tool` event per call, for the edit metrics (spec §12.9)."""
    _log("tool", tool, target, blocked=False, ok=bool(ok), seconds=seconds)


def run_cell(code: str, namespace: dict) -> dict:
    _state["submitted"], _state["answer"] = False, None
    _edit["repo_read"], _edit["repo_writes"], _edit["last_popen"] = False, 0, None
    _edit["io"] = dict.fromkeys(_edit["io"], 0)
    _cell["start"], _cell["bridge_at_start"] = time.monotonic(), _bridge_seconds()
    out, err = _Capped(OUTPUT_MAX_CHARS), _Capped(STDERR_KEEP_CHARS)
    capture = _FdCapture(out) if _edit["on"] else None
    error = None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(code, "<repl>", "exec"), namespace)
    except BaseException as exc:
        error = _format_error(exc)
    finally:
        if capture is not None:
            capture.stop()
    if _edit["on"] and any(_edit["io"].values()):  # exact counts; the repo_write events stop at REPO_WRITES_LOGGED
        _log("io", "cell", "", blocked=False, **_edit["io"])
    return {
        "stdout": out.getvalue(),
        "stdout_chars": out.total,
        "stderr": err.getvalue(),
        "error": error,
        "submitted": _state["submitted"],
        "answer": _state["answer"],
        "threads": threading.active_count() - 1,
    }


def _send(fd: int, message: dict) -> None:
    data = (json.dumps(message) + "\n").encode()
    while data:
        data = data[os.write(fd, data):]


def parse_mode(argv: list) -> tuple[bool, tuple]:
    """`--edit [--writable /a:/b]` -> (True, writable folders); no arguments -> read mode."""
    edit, writable = "--edit" in argv, ()
    if "--writable" in argv:
        i = argv.index("--writable")
        writable = tuple(p for p in (argv[i + 1] if i + 1 < len(argv) else "").split(":") if p.startswith("/"))
    if edit:
        writable = tuple(dict.fromkeys((EDIT_REPO, SCRATCH_DIR, EDIT_TMP) + writable))
    return edit, writable


def _option(argv: list, name: str) -> str | None:
    """The value after `name` in argv (`--base <sha>`, `--cell-timeout <seconds>`), or None."""
    if name in argv:
        i = argv.index(name)
        return argv[i + 1] if i + 1 < len(argv) else None
    return None


def load_tools(base: str | None, cell_timeout: float | None):
    """Edit mode: import the coding tools mounted next to this file, before lockdown, and point them at the
    repository. Returns the module."""
    global _tools
    sys.path.insert(0, WORKER_DIR)
    try:
        import repl_tools
    finally:
        sys.path.remove(WORKER_DIR)
    _cell["timeout"] = cell_timeout
    repl_tools.configure(root=EDIT_REPO, base=base, time_left=cell_seconds_left, on_call=_log_tool_call)
    _tools = repl_tools
    return repl_tools


def repl_main(argv: list | None = None) -> None:
    """The REPL process: lock down, then run cells until the host closes stdin."""
    global _audit_fd, WRITE_PREFIXES
    argv = sys.argv[1:] if argv is None else argv
    edit, writable = parse_mode(argv)
    # The protocol moves to private descriptors. Model code that reads stdin gets EOF, and raw writes
    # to fd 1 land in the container log instead of corrupting a result.
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.dup(1)
    devnull = os.open("/dev/null", os.O_RDWR)
    os.dup2(devnull, 0)
    os.close(devnull)
    if edit:
        # The audit log moves to a private descriptor that programs the model runs don't inherit; fds 1 and 2 then
        # feed each cell's output (_FdCapture) and point at /dev/null between cells.
        _audit_fd = os.dup(2)
        devnull = os.open("/dev/null", os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        WRITE_PREFIXES = writable
        _edit["on"] = True
        os.umask(0o022)
    else:
        os.dup2(2, 1)
    sys.stdin = io.StringIO()

    with open(CONTEXT_PATH, encoding="utf-8", newline="") as f:  # keep line endings exactly as the host wrote them
        namespace = make_namespace(f.read())
    if edit:
        os.chdir(EDIT_REPO)
        timeout = _option(argv, "--cell-timeout")
        tools = load_tools(_option(argv, "--base"), float(timeout) if timeout else None)
        namespace.update(tools.namespace())
    try:
        landlock, landlock_error = apply_landlock(edit_landlock_rules(writable) if edit else None), None
    except OSError as exc:
        landlock, landlock_error = 0, str(exc)
    for name in [m for m in sys.modules if m == "_ctypes" or m == "ctypes" or m.startswith("ctypes.")]:
        del sys.modules[name]
    install_audit_hook(edit)
    _send(proto_out, {"ready": True, "landlock": landlock, "landlock_error": landlock_error,
                      "python": sys.version.split()[0], "edit": edit,
                      "tools": _tools.TOOLS_VERSION if _tools is not None else None})

    for line in proto_in:
        try:
            code = json.loads(line)["code"]
        except (ValueError, KeyError, TypeError):
            _send(proto_out, {"protocol_error": "expected {\"code\": ...}"})
            continue
        _send(proto_out, run_cell(code, namespace))


# ----------------------------------------------------------------------------- supervisor

CGROUP_MEMORY_EVENTS = "/sys/fs/cgroup/memory.events"


def _oom_kills() -> int | None:
    try:
        with open(CGROUP_MEMORY_EVENTS) as f:
            for line in f:
                key, _, value = line.partition(" ")
                if key == "oom_kill":
                    return int(value)
    except (OSError, ValueError):
        pass
    return None


def main() -> None:
    """PID 1: fork the REPL, wait for it, and report how it died. The supervisor never runs model code."""
    handlers = {}
    for sig in signal.valid_signals():
        if sig in (signal.SIGKILL, signal.SIGSTOP, signal.SIGCHLD):
            continue
        try:
            handlers[sig] = signal.signal(sig, signal.SIG_IGN)
        except (OSError, ValueError, RuntimeError):
            pass
    oom_before = _oom_kills()

    pid = os.fork()
    if pid == 0:
        for sig, handler in handlers.items():
            if handler is not None:
                signal.signal(sig, handler)
        code = 0
        try:
            repl_main()
        except BaseException:
            traceback.print_exc()
            code = 70
        os._exit(code)

    while True:
        done, status = os.waitpid(-1, 0)  # also reaps anything the REPL left behind
        if done == pid:
            break
    if os.WIFSIGNALED(status):
        number, oom_after = os.WTERMSIG(status), _oom_kills()
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = str(number)
        died = {"signal": name, "oom": None if oom_before is None or oom_after is None else oom_after > oom_before}
    else:
        died = {"exit_code": os.WEXITSTATUS(status)}
    # The leading newline ends any half-written line the REPL left on the shared stdout.
    data = ("\n" + json.dumps({"repl_died": died}) + "\n").encode()
    while data:
        data = data[os.write(1, data):]


if __name__ == "__main__":
    main()
