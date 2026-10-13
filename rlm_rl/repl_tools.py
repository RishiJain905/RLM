"""The coding tools the edit REPL gives the root (spec §12.1, code-editing-plan.md §5.6).

Seven functions modelled on what Claude Code, Codex CLI, DeepSeek Harness and Qwen Code give
their models: read_file, find_files, grep, edit_file, write_file, shell, diff. They run inside
the episode container, in the REPL process, so nothing new crosses the bridge, and Landlock
and --network none bound them as they bound plain `open` and `subprocess`.

Standard library only, and it imports nothing from the project: the container mounts this file
read-only at /opt/rlm next to repl_worker.py, and the Part 2 harness imports the same file, so
the deployed model calls the functions it trained on. Names, signatures and docstrings are
frozen with TOOLS_VERSION; tests/test_repl_tools.py pins them (H10).

The REPL wires them in with:

    repl_tools.configure(root="/work/repo", time_left=cell_seconds_left, on_call=log_event)
    namespace.update(repl_tools.namespace())
"""

import functools
import os
import re
import shutil
import subprocess
import threading
import time
from typing import NamedTuple

TOOLS_VERSION = "t1"
TOOL_NAMES = ("read_file", "find_files", "grep", "edit_file", "write_file", "shell", "diff")

REPO_DIR = "/work/repo"
READ_LINES = 40  # one printed turn (2,000 characters) holds about this many numbered lines
LINE_MAX_CHARS = 300
BINARY_SNIFF_BYTES = 8192
GREP_MAX_FILE_BYTES = 2_000_000
GREP_MAX_MATCHES = 10_000
SHELL_TIMEOUT_S = 120
SHELL_TIMEOUT_CODE = 124  # what coreutils `timeout` returns
SHELL_KEEP_CHARS = 1_000_000  # the end of the output is kept; the start of a huge log is dropped
SHELL_SHOW_CHARS = 1500  # what printing a result shows: the end, where failures are
# Searched only when the repository has no git to read .gitignore from.
FALLBACK_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache",
                                ".pytest_cache", ".gradle"})

_config = {"root": REPO_DIR, "base": None, "time_left": None, "on_call": None}
_local = threading.local()


def configure(root: str = REPO_DIR, base: str | None = None, time_left=None, on_call=None) -> None:
    """Set the repository the tools work in.

    base: the commit diff() compares against; None means the repository's first commit.
    time_left: a function returning the seconds left in the current cell; shell() never runs past it.
    on_call: called as on_call(tool, ok, seconds, target) after every tool call, for the edit metrics.
    """
    _config.update(root=os.path.abspath(root), base=base, time_left=time_left, on_call=on_call)


def namespace() -> dict:
    """The tools by name, for the REPL's namespace."""
    return {name: globals()[name] for name in TOOL_NAMES}


def active() -> bool:
    """True while a tool call runs on this thread, so the REPL's audit hook can tell tool I/O from raw open/subprocess."""
    return getattr(_local, "depth", 0) > 0


def _logged(target_arg: str):
    """Report each call to on_call with whether it raised, how long it took and its path or command."""
    def wrap(fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            bound = {**dict(zip(names, args)), **kwargs}
            target = str(bound.get(target_arg, ""))[:200] if target_arg else ""
            _local.depth = getattr(_local, "depth", 0) + 1
            start, ok = time.monotonic(), False
            try:
                result = fn(*args, **kwargs)
                ok = True
                return result
            finally:
                _local.depth -= 1
                if _config["on_call"]:
                    _config["on_call"](fn.__name__, ok, round(time.monotonic() - start, 3), target)
        return call
    return wrap


# ----------------------------------------------------------------------------- paths and files


def _abs(path: str) -> str:
    return os.path.normpath(os.path.join(_config["root"], os.fspath(path)))


def _show(path: str) -> str:
    """A path as the tools report it: relative to the repository when inside it, so it can go straight back into read_file."""
    rel = os.path.relpath(path, _config["root"])
    return path if rel == ".." or rel.startswith("../") else rel


def _is_binary(path: str) -> bool:
    with open(path, "rb") as f:
        return b"\0" in f.read(BINARY_SNIFF_BYTES)


def _cut(line: str) -> str:
    return line if len(line) <= LINE_MAX_CHARS else line[:LINE_MAX_CHARS] + f" [... {len(line) - LINE_MAX_CHARS} more characters]"


def _git(args: list, cwd: str, env: dict | None = None) -> str:
    if shutil.which("git") is None:
        raise RuntimeError("git is not installed in this environment")
    done = subprocess.run(["git", "-c", "safe.directory=*", "-c", "core.quotepath=off", *args], cwd=cwd,
                          capture_output=True, env=env)
    if done.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {done.stderr.decode(errors='replace').strip()[:500]}")
    return done.stdout.decode(errors="replace")


def _files_under(base: str) -> list[str]:
    """Absolute paths of the files under base, sorted. Inside a git work tree, git's own list (tracked plus
    untracked, minus what .gitignore excludes), as ripgrep does; elsewhere a walk that skips the usual
    dependency and cache folders."""
    if os.path.isfile(base):
        return [base]
    if not os.path.isdir(base):
        raise FileNotFoundError(f"no such directory: {_show(base)}")
    try:
        listed = _git(["ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=base)
        paths = {os.path.join(base, p) for p in listed.split("\0") if p}
        return sorted(p for p in paths if os.path.isfile(p))
    except RuntimeError:
        pass
    found = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in FALLBACK_SKIP_DIRS)
        found += [os.path.join(dirpath, f) for f in filenames]
    return sorted(found)


def _glob_regex(pattern: str) -> re.Pattern:
    """A glob as a regex over '/'-separated paths: * and ? stay within one folder, ** crosses folders, {a,b} picks one."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "{" and "}" in pattern[i:]:
            end = pattern.index("}", i)
            out.append("(?:" + "|".join(re.escape(p) for p in pattern[i + 1:end].split(",")) + ")")
            i = end + 1
        elif pattern[i] == "[" and "]" in pattern[i + 1:]:
            end = pattern.index("]", i + 1)
            body = pattern[i + 1:end]
            out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


# ----------------------------------------------------------------------------- the tools


@_logged("path")
def read_file(path: str, start: int = 1, n: int = READ_LINES) -> str:
    """Return n lines of a file from line `start` (1-based), numbered, with long lines cut.

    Ends with how many lines follow and the call that reads them. Raises for a missing,
    binary or empty-range read. For a file's raw text, use open().
    """
    full = _abs(path)
    if _is_binary(full):
        raise ValueError(f"{_show(full)} is a binary file")
    with open(full, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if not lines:
        return f"[{_show(full)} is empty]"
    start = max(1, int(start))
    if start > len(lines):
        raise ValueError(f"start={start} is past the end of {_show(full)} ({len(lines)} lines)")
    end = min(len(lines), start - 1 + max(1, int(n)))
    text = "\n".join(f"{i:6}\t{_cut(lines[i - 1])}" for i in range(start, end + 1))
    if end < len(lines):
        text += f"\n[{len(lines) - end} more lines; next: read_file({_show(full)!r}, start={end + 1})]"
    return text


@_logged("pattern")
def find_files(pattern: str, path: str = ".") -> list[str]:
    """Return the files under `path` whose path matches a glob, sorted.

    "*.py" matches only directly under `path`; "**/*.py" matches at any depth; "src/**/*.{ts,tsx}"
    also works. Skips what .gitignore excludes. Paths come back relative to the repository.
    """
    base = _abs(path)
    regex = _glob_regex(pattern)
    return [_show(p) for p in _files_under(base) if regex.match(os.path.relpath(p, base))]


@_logged("pattern")
def grep(pattern: str, path: str = ".", glob: str | None = None, ignore_case: bool = False,
         files_only: bool = False) -> list[str]:
    """Search file contents with a Python regex; return matches as "path:line: text".

    path: a folder or one file. glob: only files matching it ("*.go" matches by file name at any
    depth, "src/**/*.go" by path). files_only: return each matching file's path once instead.
    Skips binary files, files over 2 MB and what .gitignore excludes; stops after 10,000 matches,
    with a last entry saying so.
    """
    regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    base = _abs(path)
    keep = _glob_regex(glob) if glob else None
    results = []
    for full in _files_under(base):
        if keep and not keep.match(os.path.basename(full) if "/" not in glob else os.path.relpath(full, base)):
            continue
        if os.path.getsize(full) > GREP_MAX_FILE_BYTES or _is_binary(full):
            continue
        with open(full, encoding="utf-8", errors="replace") as f:
            for number, line in enumerate(f, 1):
                if not regex.search(line):
                    continue
                if files_only:
                    results.append(_show(full))
                    break
                text = _cut(line.rstrip("\r\n"))
                results.append(f"{_show(full)}:{number}: {text}")
                if len(results) >= GREP_MAX_MATCHES:
                    results.append(f"[stopped at {GREP_MAX_MATCHES:,} matches; narrow the pattern, path or glob]")
                    return results
    return results


@_logged("path")
def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
    """Replace the exact text `old` with `new` in a file and say where.

    `old` must occur exactly once (copy enough surrounding lines to make it unique), or pass
    replace_all=True to replace every occurrence. Otherwise it raises and leaves the file untouched.
    Keeps the file's line endings.
    """
    if not old:
        raise ValueError("old must not be empty; use write_file to create a file")
    if old == new:
        raise ValueError("old and new are the same")
    full = _abs(path)
    try:
        with open(full, encoding="utf-8", newline="") as f:
            text = f.read()
    except UnicodeDecodeError:
        raise ValueError(f"{_show(full)} is not UTF-8 text; change it with shell() instead") from None
    if "\r\n" in text and "\r\n" not in old:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    count = text.count(old)
    if count == 0:
        raise ValueError(f"old text not found in {_show(full)}; check whitespace and indentation against read_file")
    if count > 1 and not replace_all:
        raise ValueError(f"old text occurs {count} times in {_show(full)}; include more surrounding lines, "
                         "or pass replace_all=True to replace every one")
    line = text[:text.index(old)].count("\n") + 1
    with open(full, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace(old, new))
    done = count if replace_all else 1
    return f"edited {_show(full)}: {done} replacement{'s' if done > 1 else ''}, first at line {line}"


@_logged("path")
def write_file(path: str, text: str) -> str:
    """Create or overwrite a file with `text`, making its folders as needed."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")
    full = _abs(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return f"wrote {_show(full)} ({len(text.splitlines())} lines)"


class ShellResult(NamedTuple):
    """What shell() returns: `code` is the exit code, `out` holds stdout and stderr together."""
    code: int
    out: str

    def __str__(self) -> str:
        tail = self.out
        if len(tail) > SHELL_SHOW_CHARS:
            tail = f"[... {len(tail) - SHELL_SHOW_CHARS:,} earlier characters]\n" + tail[-SHELL_SHOW_CHARS:]
        return f"exit {self.code}" + (f"\n{tail}" if tail else "")

    __repr__ = __str__


@_logged("cmd")
def shell(cmd: str, timeout: float = SHELL_TIMEOUT_S) -> ShellResult:
    """Run a shell command (sh -c) in the repository and return its exit code and output.

    Printing the result shows the exit code and the last 1,500 characters; `.out` holds the
    whole output (its last 1,000,000 characters). On timeout the command is killed and the
    result has code 124 and what it printed. Never runs past the cell's time limit. No network.
    """
    limit = float(timeout)
    if _config["time_left"]:
        limit = max(1.0, min(limit, _config["time_left"]() - 2.0))
    proc = subprocess.Popen(["sh", "-c", cmd], cwd=_config["root"], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    kept, size = [], [0]

    def drain():
        for chunk in iter(lambda: proc.stdout.read1(65536), b""):
            kept.append(chunk)
            size[0] += len(chunk)
            while size[0] - len(kept[0]) > SHELL_KEEP_CHARS:  # stay near the cap without ever splitting a chunk
                size[0] -= len(kept.pop(0))

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, 9)  # the whole process group: test runners leave children behind
        except ProcessLookupError:
            pass
        proc.wait()
    reader.join(timeout=5)  # a process that left the group can hold the pipe open; don't wait on it
    if not reader.is_alive():
        proc.stdout.close()
    out = b"".join(list(kept)).decode(errors="replace")[-SHELL_KEEP_CHARS:]
    if timed_out:
        return ShellResult(SHELL_TIMEOUT_CODE, out + f"\n[timed out after {limit:g} s and killed]")
    return ShellResult(proc.returncode, out)


@_logged("")
def diff() -> str:
    """Return your change so far as a unified diff against the task's starting commit, new files
    included: what gets graded. Empty if nothing has changed."""
    root = _config["root"]
    base = _config["base"] or _git(["rev-list", "--max-parents=0", "HEAD"], cwd=root).split()[-1]
    # Stage into a copy of the index, so the repository's own index is never touched. The copy lives in
    # .git, the one place sure to be writable: the sandbox's Landlock rules leave /tmp out.
    git_dir = _git(["rev-parse", "--absolute-git-dir"], cwd=root).strip()
    real, index = os.path.join(git_dir, "index"), os.path.join(git_dir, "repl_tools_diff_index")
    try:
        if os.path.exists(real):
            shutil.copyfile(real, index)
        env = {**os.environ, "GIT_INDEX_FILE": index}
        _git(["add", "-A"], cwd=root, env=env)
        return _git(["diff", "--cached", "--no-color", "--no-ext-diff", base], cwd=root, env=env)
    finally:
        if os.path.exists(index):
            os.remove(index)
