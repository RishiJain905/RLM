"""The grader's driver: runs a task's tests inside the grader container and reports back (spec §12.5).

Standard library only, and it imports nothing from the project except the REPL worker's Landlock
code, loaded by path. The host mounts the runtime folder read-only at /opt/rlm and starts:

    python grader_driver.py run        # reads its job as one JSON object on stdin

The job lists the commands to run (argv, cwd, env overrides, timeout), the report files to read
afterwards, and a nonce. The driver
- makes itself non-dumpable, so the tests, which run as the same user, can't open its stdin or
  stdout through /proc and forge the result;
- locks itself and every test process down with the edit sandbox's Landlock rules;
- runs each command in its own process group with stdin from /dev/null and its output on pipes,
  under one deadline for the whole job (600 s by default), and kills the group when the command
  ends, so nothing the tests start keeps running to rewrite a report;
- prints one line, `@@rlm-grade <nonce> {json}`, with each command's exit code, timing and output,
  and the report files' contents.

A command that runs past the deadline is the model's outcome (every task's gold patch was
validated well inside it); the host scores its missing tests as failed. Anything else that goes
wrong here makes the host's grade an infra error.

`python grader_driver.py unittest --report PATH --top DIR module ...` is a small unittest runner
that writes per-test results as JSON. It runs on the repository's own Python, for edit_synth.
"""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

OUTPUT_CAP = 32 * 1024 * 1024  # bytes kept per stream; go test -json can be large
REPORT_CAP = 32 * 1024 * 1024
PREFIX = "@@rlm-grade "


def _non_dumpable() -> bool:
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(4, ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) == 0
    except Exception:
        return False


def _landlock(writable) -> int:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repl_worker.py")
    spec = importlib.util.spec_from_file_location("rlm_repl_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker.apply_landlock(worker.edit_landlock_rules(writable))


def _read_capped(stream, cap: int) -> tuple[bytes, int]:
    data = stream.read(cap)
    total = len(data)
    while True:  # keep draining so the writer never blocks, but keep only the first `cap` bytes
        more = stream.read(1 << 20)
        if not more:
            break
        total += len(more)
    return data, total


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_command(cmd: dict, deadline: float) -> dict:
    env = dict(os.environ)
    env.update(cmd.get("env") or {})
    timeout = max(1.0, min(float(cmd.get("timeout") or 600), deadline - time.monotonic()))
    started = time.monotonic()
    try:
        proc = subprocess.Popen(cmd["argv"], cwd=cmd.get("cwd") or None, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT if cmd.get("merge") else subprocess.PIPE,
                                start_new_session=True)
    except OSError as exc:
        return {"argv": cmd["argv"], "exit": None, "error": f"could not start: {exc}", "timed_out": False,
                "seconds": 0.0, "stdout": "", "stderr": "", "stdout_bytes": 0, "stderr_bytes": 0}
    import threading

    out, err = {}, {}
    readers = [threading.Thread(target=lambda: out.update(zip(("data", "total"), _read_capped(proc.stdout, OUTPUT_CAP))))]
    if proc.stderr is not None:
        readers.append(threading.Thread(
            target=lambda: err.update(zip(("data", "total"), _read_capped(proc.stderr, OUTPUT_CAP)))))
    for r in readers:
        r.daemon = True
        r.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    _kill_group(proc.pid)  # the command's own leftovers too, so nothing rewrites a report after it ends
    proc.wait()
    for r in readers:
        r.join(10)
    return {"argv": cmd["argv"], "exit": None if timed_out else proc.returncode, "timed_out": timed_out,
            "seconds": round(time.monotonic() - started, 3),
            "stdout": out.get("data", b"").decode("utf-8", "replace"), "stdout_bytes": out.get("total", 0),
            "stderr": err.get("data", b"").decode("utf-8", "replace"), "stderr_bytes": err.get("total", 0)}


def read_report(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            data = f.read(REPORT_CAP + 1)
    except FileNotFoundError:
        return {"path": path, "found": False}
    except OSError as exc:
        return {"path": path, "found": False, "error": str(exc)}
    return {"path": path, "found": True, "truncated": len(data) > REPORT_CAP,
            "text": data[:REPORT_CAP].decode("utf-8", "replace")}


def expand_reports(spec: dict) -> list[str]:
    """Report paths, with a trailing '/*.xml' style glob expanded (Maven and Gradle write one file per class)."""
    import glob

    out = []
    for path in spec.get("reports") or []:
        if any(c in path for c in "*?["):
            out.extend(sorted(glob.glob(path, recursive=True)))
        else:
            out.append(path)
    return out


def main_run() -> int:
    raw = sys.stdin.buffer.read()
    job = json.loads(raw)
    nonce = str(job["nonce"])
    result = {"non_dumpable": _non_dumpable(), "commands": [], "reports": [], "deadline_hit": False}
    try:
        result["landlock"] = _landlock(job.get("writable") or ["/work/repo", "/scratch", "/tmp"])
    except OSError as exc:
        result["landlock"], result["landlock_error"] = 0, str(exc)
    if job.get("require_landlock", True) and not result["landlock"]:
        result["fatal"] = "Landlock is unavailable"
    else:
        if job.get("report_dir"):
            os.makedirs(job["report_dir"], exist_ok=True)
        deadline = time.monotonic() + float(job.get("deadline_s") or 600)
        for cmd in job.get("commands") or []:
            if time.monotonic() >= deadline:
                result["deadline_hit"] = True
                result["commands"].append({"argv": cmd["argv"], "exit": None, "timed_out": True, "skipped": True,
                                           "seconds": 0.0, "stdout": "", "stderr": ""})
                continue
            res = run_command(cmd, deadline)
            result["commands"].append(res)
            if res["timed_out"]:
                result["deadline_hit"] = True
        result["reports"] = [read_report(p) for p in expand_reports(job)]
    line = (PREFIX + nonce + " " + json.dumps(result) + "\n").encode()
    while line:
        line = line[os.write(1, line):]
    return 0


# ----------------------------------------------------------------------------- the unittest runner (edit_synth)


def main_unittest(argv: list) -> int:
    import unittest

    report, top, modules = None, ".", []
    it = iter(argv)
    for arg in it:
        if arg == "--report":
            report = next(it)
        elif arg == "--top":
            top = next(it)
        else:
            modules.append(arg)
    sys.path.insert(0, os.path.abspath(top))
    outcomes = {}

    class Result(unittest.TestResult):
        def addSuccess(self, test):
            outcomes[test.id()] = "passed"

        def addFailure(self, test, err):
            outcomes[test.id()] = "failed"

        def addError(self, test, err):
            outcomes[getattr(test, "id", lambda: str(test))()] = "error"

        def addSkip(self, test, reason):
            outcomes[test.id()] = "skipped"

        def addExpectedFailure(self, test, err):
            outcomes[test.id()] = "skipped"

        def addUnexpectedSuccess(self, test):
            outcomes[test.id()] = "failed"

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    errors = {}
    for name in modules:
        try:
            suite.addTests(loader.loadTestsFromName(name))
        except Exception as exc:  # a module that doesn't import: its tests are missing, so they count as failed
            errors[name] = f"{type(exc).__name__}: {exc}"[:500]
    suite.run(Result())
    with open(report, "w", encoding="utf-8") as f:
        json.dump({"tests": outcomes, "import_errors": errors}, f)
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "run":
        sys.exit(main_run())
    if command == "unittest":
        sys.exit(main_unittest(sys.argv[2:]))
    print("usage: grader_driver.py run | unittest --report PATH --top DIR module ...", file=sys.stderr)
    sys.exit(2)
