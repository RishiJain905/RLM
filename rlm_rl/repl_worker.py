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
import traceback

# Must match rlm_rl/bridge.py; tests/test_contract.py checks.
API_NAMES = ("context", "llm", "llm_batch", "submit")
WIRE_OPS = ("llm", "llm_batch")
OUTPUT_MAX_CHARS = 2000
TRACEBACK_LINES = 6
SUBCALL_MAX_CHARS = 24_000
SUBCALL_TRUNCATION_MARK = "[prompt truncated]"
SUBMIT_MAX_BYTES = 100_000

EPISODE_DIR = "/episode"
CONTEXT_PATH = "/episode/context.txt"
SOCKET_PATH = "/episode/bridge.sock"
SCRATCH_DIR = "/scratch"
WORKER_DIR = "/opt/rlm"

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


def _bridge(op: str, prompts: list) -> list:
    """One request over the episode's Unix socket. A fresh connection per call, so threads can call in parallel."""
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


def apply_landlock() -> int:
    """Restrict this process and every child to LANDLOCK_RULES. Returns the Landlock ABI applied, or 0 if the kernel has none."""
    import ctypes

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
        for path, rights in LANDLOCK_RULES.items():
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


def _log(kind: str, event: str, detail, blocked: bool = True) -> None:
    """Write one audit line to stderr. If stderr was tampered with, the REPL exits instead (TAMPER_EXIT_CODE)."""
    try:
        st = os.fstat(2)
        if (st.st_dev, st.st_ino) != _stderr_id:
            os._exit(TAMPER_EXIT_CODE)
        data = (AUDIT_PREFIX + json.dumps({"kind": kind, "event": event, "detail": repr(detail)[:300],
                                           "blocked": blocked}) + "\n").encode()
        while data:
            data = data[os.write(2, data):]
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


def _audit_hook(event: str, args) -> None:
    handler = _HANDLERS.get(event)
    if handler is not None:
        handler(args)
    elif event.startswith("ctypes."):
        _block("ctypes", event, args, shown=event)


def install_audit_hook() -> None:
    global _stderr_id
    st = os.fstat(2)
    _stderr_id = (st.st_dev, st.st_ino)
    sys.meta_path.insert(0, _ImportWatch)
    sys.addaudithook(_audit_hook)


# ----------------------------------------------------------------------------- cells


class _Capped(io.TextIOBase):
    """A stdout stand-in that keeps the first `cap` characters and counts the rest."""

    def __init__(self, cap: int):
        self.cap, self.total, self._parts, self._kept = cap, 0, [], 0

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def write(self, s):
        if not isinstance(s, str):
            raise TypeError(f"write() argument must be str, not {type(s).__name__}")
        self.total += len(s)
        if self._kept < self.cap:
            piece = s[: self.cap - self._kept]
            self._parts.append(piece)
            self._kept += len(piece)
        return len(s)

    def getvalue(self) -> str:
        return "".join(self._parts)


def _format_error(exc: BaseException) -> str:
    tb = exc.__traceback__.tb_next if exc.__traceback__ is not None else None  # drop the worker's own frame
    lines = "".join(traceback.format_exception(type(exc), exc, tb)).splitlines()
    text = "\n".join(lines[-TRACEBACK_LINES:])
    if len(text) > OUTPUT_MAX_CHARS:
        text = text[:OUTPUT_MAX_CHARS] + f"\n[... {len(text) - OUTPUT_MAX_CHARS} more characters truncated]"
    return text


def run_cell(code: str, namespace: dict) -> dict:
    _state["submitted"], _state["answer"] = False, None
    out, err = _Capped(OUTPUT_MAX_CHARS), _Capped(STDERR_KEEP_CHARS)
    error = None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(code, "<repl>", "exec"), namespace)
    except BaseException as exc:
        error = _format_error(exc)
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


def repl_main() -> None:
    """The REPL process: lock down, then run cells until the host closes stdin."""
    # The protocol moves to private descriptors. Model code that reads stdin gets EOF, and raw writes
    # to fd 1 land in the container log instead of corrupting a result.
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.dup(1)
    devnull = os.open("/dev/null", os.O_RDWR)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdin = io.StringIO()

    with open(CONTEXT_PATH, encoding="utf-8", newline="") as f:  # keep line endings exactly as the host wrote them
        namespace = make_namespace(f.read())
    try:
        landlock, landlock_error = apply_landlock(), None
    except OSError as exc:
        landlock, landlock_error = 0, str(exc)
    for name in [m for m in sys.modules if m == "_ctypes" or m == "ctypes" or m.startswith("ctypes.")]:
        del sys.modules[name]
    install_audit_hook()
    _send(proto_out, {"ready": True, "landlock": landlock, "landlock_error": landlock_error,
                      "python": sys.version.split()[0]})

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
