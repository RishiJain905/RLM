"""Smoke run (rlm-harness v0, step 9): scripts/harness_smoke.py, offline.

No network, no key and no provider. `rlmh` is a stub, a small Python script the tests write into a temp folder whose
name holds a space: it answers `--version` and `config get`, and for `batch` it reads the tasks file and prints canned
`--json` lines (architecture 5.3, 7.8). The failures to simulate come from options placed before `batch` (the tests'
`run_slot`/`run_batch` calls) or from a stub_config.json next to it (the `main()` calls, where `--rlmh` can carry no
options). Every module path `main()` writes to is patched into the temp folder and the tasks are hand-made, so nothing
lands in the real results/, documentation/, private/ or %TEMP%/rlmh-smoke. Cases that need the HuffPost data root skip
without it (`data.available()`).
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlm_rl import bridge, data, reward, tasks
from rlm_rl.env import Task

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("harness_smoke", ROOT / "scripts" / "harness_smoke.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses use `from __future__ import annotations` and look the module up
    spec.loader.exec_module(module)
    return module


hs = _load_script()
REAL = SimpleNamespace(check_ignored=hs.check_ignored, task_folder=hs.task_folder, build_tasks=hs.build_tasks,
                       rlm_commit=hs.rlm_commit)  # the real functions, before any test patches the module's names
_WRITE_TEXT = Path.write_text

DATE = "2026-01-02"
FAMILY = "aggregation_heldout"
KEY = "sk-sentinel-123"
MODEL_A, MODEL_B = "deepseek/FAKE-chat", "openrouter/FAKE-no-tools"
PRICE_A = {"input": 1.0, "output": 4.0, "cache_read": 0.25, "cache_write": 1.5}
PRICE_B = {"input": 0.5, "output": 2.0, "cache_read": 0.05, "cache_write": 0.6}
# What the synthetic reference runs used, per run (roots and sub-calls).
REF_TOKENS = {"root_in": 100_000, "root_out": 10_000, "sub_in": 200_000, "sub_out": 20_000}
CEILING_IN = 80_000 + 190 * (4096 + 8000 / 3) + 80 * 60_000 / 3  # 2,964,906.67: READ_LIMITS_V4 by hand
CEILING_OUT = 20 * 4096 + 80 * 4096  # 409,600


def model(provider, name, price, tool_call):
    return {"provider": provider, "id": name, "name": name, "context_window": 128000, "max_output": 8192,
            "price": dict(price), "tool_call": tool_call, "reasoning": False}


CATALOG = {"version": 1, "source": {"repo": "FAKE/none", "commit": "0"},
           "models": [model("deepseek", "FAKE-chat", PRICE_A, True),
                      model("openrouter", "FAKE-no-tools", PRICE_B, False),
                      model("anthropic", "FAKE-claude", {"input": 3.0, "output": 15.0, "cache_read": 0.3,
                                                         "cache_write": 3.75}, True)]}

# ----------------------------------------------------------------------------- the stub rlmh

STUB = r'''
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = {"call_format": "auto", "permissions.mode": "accept-edits", "root.thinking": "off", "sub.thinking": "off",
          "profile": "default"}
USAGE = {"root": {"input_tokens": 1000, "output_tokens": 100, "cache_read_tokens": 400, "cache_write_tokens": 0},
         "sub": {"input_tokens": 2000, "output_tokens": 200, "cache_read_tokens": 0, "cache_write_tokens": 0}}
# the engine stand-in: it blocks on its stdin pipe and exits on EOF, as rlmh's engine does
CHILD = ("import os, sys\nopen(sys.argv[1], 'w').write(str(os.getpid()))\nsys.stdin.read()\n"
         "open(sys.argv[2], 'w').write('eof')\n")


def options(argv):
    cfg = {"fail": [], "fail_both": [], "garbage": False, "child": False, "ignore_term": False, "sha_lie": False,
           "cost": 0.125, "calls": 2, "slow": 0.0, "exit": 0, "die_after": 0, "turns": 3, "subcalls": 4, "wall": 1.5,
           "chars_delta": 0, "format_lie": None, "profile_lie": None, "usage": None, "answers": {}, "messages": {},
           "no_submit": [], "invalid": [], "denied": {}, "allowed": {}, "config": {}, "resolve": {}, "by_ref": {}}
    path = HERE / "stub_config.json"
    if path.is_file():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    i = 0
    while i < len(argv) and argv[i].startswith("--") and argv[i] != "--version":
        name = argv[i][2:].replace("-", "_")
        if isinstance(cfg.get(name), bool):
            cfg[name] = True
            i += 1
            continue
        value = argv[i + 1]
        i += 2
        if isinstance(cfg.get(name), list):
            cfg[name].append(value)
        elif isinstance(cfg.get(name), (int, float)):
            cfg[name] = type(cfg[name])(value)
        else:
            cfg[name] = value
    return cfg, argv[i:]


def log(entry):
    with open(HERE / "calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def emit(cfg, obj):
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    try:
        sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
        sys.stdout.buffer.flush()
    except OSError:
        pass  # the reader is gone (the cost cap): carry on, so that only terminate() or kill() stops this process
    if cfg["slow"]:
        time.sleep(cfg["slow"])


def start_child():
    pid_file, eof_file = HERE / "child.pid", HERE / "child.eof"
    for f in (pid_file, eof_file):
        f.unlink(missing_ok=True)
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(pid_file), str(eof_file)], stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while time.time() < deadline and not (pid_file.is_file() and pid_file.read_text().strip()):
        time.sleep(0.02)
    return child


def batch(args, cfg, entry):
    tasks_file = Path(args[0])
    flags, i = {}, 1
    while i < len(args):
        if args[i] == "--json":
            flags["--json"] = True
            i += 1
        else:
            flags[args[i]] = args[i + 1]
            i += 2
    out = Path(flags["--out"])
    out.mkdir(parents=True, exist_ok=True)
    rerun = out.name == "rerun"
    root = flags["--root"]
    cfg.update(cfg["by_ref"].get(root, {}))
    tasks = [json.loads(line) for line in tasks_file.read_text(encoding="utf-8").split("\n") if line.strip()]
    entry.update(tasks=[{"id": t["id"], "question": t["question"], "input": t["input"]} for t in tasks], flags=flags,
                 rerun=rerun, tasks_file=str(tasks_file))
    if os.environ.get("RLMH_FAKE_SCRIPT"):
        entry["fake_script_json"] = json.loads(Path(os.environ["RLMH_FAKE_SCRIPT"]).read_text(encoding="utf-8"))
    log(entry)
    if cfg["ignore_term"] and hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if cfg["exit"] == 2:
        sys.stderr.write("rlmh: usage error (the stub was told to exit 2)\n")
        return 2
    child = start_child() if cfg["child"] else None
    fail = {}
    for spec in cfg["fail_both"] + ([] if rerun else cfg["fail"]):
        kind, _, tid = spec.partition(":")
        fail[tid] = kind
    fmt = flags.get("--call-format") or cfg["resolve"].get(root, "native")
    free = root.startswith("fake/")
    prefix = "rr" if rerun else "r"
    finished_n = failed = 0
    total = 0.0
    for idx, task in enumerate(tasks):
        if cfg["die_after"] and not rerun and idx >= cfg["die_after"]:
            return 1  # the batch died: no more events, no batch.finished
        tid, run_id, kind = task["id"], f"{prefix}-{idx + 1}", fail.get(task["id"])
        text = (tasks_file.parent / task["input"]).read_bytes().decode("utf-8", "replace")
        sha = "0" * 64 if cfg["sha_lie"] else hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        emit(cfg, {"type": "run.started", "run_id": run_id, "tag": tid, "mode": "read",
                   "profile": cfg["profile_lie"] or flags["--profile"], "call_format": cfg["format_lie"] or fmt,
                   "root": root, "sub": flags["--sub"],
                   "input": {"chars": len(text) + cfg["chars_delta"], "lines": len(text.splitlines()), "sha256": sha,
                             "files": [task["input"]]}})
        cost = 0.0 if free else cfg["cost"]
        for c in range(cfg["calls"]):
            error = None
            if kind and kind != "crash" and c == cfg["calls"] - 1:
                error = {"kind": kind, "message": "stub " + kind}
            emit(cfg, {"type": "gateway.call", "run_id": run_id, "role": "root", "subcall_id": None, "model": root,
                       "usage": {"input_tokens": 500, "output_tokens": 50, "cache_read_tokens": 0,
                                 "cache_write_tokens": 0, "cost_usd": cost}, "seconds": 0.01, "error": error})
            if cfg["garbage"] and idx == 0 and c == 0:
                emit(cfg, "WARNING stub garbage line, not json: { caf\u00e9")
                sys.stderr.write("stub warning on stderr\n")
        for _ in range(cfg["denied"].get(tid, 0)):
            emit(cfg, {"type": "run.permission", "run_id": run_id, "turn": 1, "kind": "shell", "detail": "ls",
                       "allowed": False, "asked": False, "remembered": False})
        for _ in range(cfg["allowed"].get(tid, 0)):
            emit(cfg, {"type": "run.permission", "run_id": run_id, "turn": 1, "kind": "fetch", "detail": "x",
                       "allowed": True, "asked": True, "remembered": False})
        total += cost * cfg["calls"]
        if kind == "crash":
            failed += 1
            continue  # run.started but never run.finished
        usage = cfg["usage"] or USAGE
        record = {"type": "run.finished", "run_id": run_id, "tag": tid,
                  "answer": None if kind else cfg["answers"].get(tid, "0"),
                  "submitted": not kind and tid not in cfg["no_submit"],
                  "answer_valid": not kind and tid not in cfg["invalid"], "turns": cfg["turns"],
                  "subcalls": cfg["subcalls"], "usage": usage, "cost_usd": cost * cfg["calls"],
                  "infra_error": cfg["messages"].get(tid, "stub " + kind + " failure") if kind else None,
                  "infra_kind": kind, "wall_s": cfg["wall"], "log_path": str(out / "runs.jsonl")}
        emit(cfg, record)
        with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": run_id, "tag": tid, "root": {"provider": root.split("/")[0]},
                                "usage": usage, "turns": [{}] * cfg["turns"], "subcalls": [{}] * cfg["subcalls"],
                                "infra_error": record["infra_error"]}) + "\n")
        finished_n += 1
        failed += bool(kind)
    emit(cfg, {"type": "batch.finished", "runs": finished_n, "failed": failed, "cost_usd": total})
    if child is not None:
        child.stdin.close()
        child.wait()
    with open(HERE / "stub.finished", "a", encoding="utf-8") as f:
        f.write(str(out) + "\n")
    return 1 if failed else 0


def main():
    argv = sys.argv[1:]
    cfg, rest = options(argv)
    entry = {"argv": argv, "rest": rest, "has_fake_key": bool(os.environ.get("FAKE_API_KEY")),
             "fake_script": os.environ.get("RLMH_FAKE_SCRIPT")}
    if rest == ["--version"]:
        log(entry)
        print("rlmh 0.1.0")
        return 0
    if rest[:2] == ["config", "get"]:
        log(entry)
        print(cfg["config"].get(rest[2], CONFIG[rest[2]]))
        return 0
    if rest and rest[0] == "batch":
        return batch(rest[1:], cfg, entry)
    log(entry)
    sys.stderr.write("rlmh: unknown command\n")
    return 2


sys.exit(main())
'''


def write_stub(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "rlmh_stub.py"
    path.write_bytes(STUB.encode("utf-8"))
    return path


# ----------------------------------------------------------------------------- hand-made tasks and helpers


def huffpost_present():
    """The data root exists and holds the HuffPost headlines (the held-out source of the set)."""
    return data.available() and data.path("hf/heegyu__news-category-dataset/data.json").is_file()


def make_tasks(question=None, answer=0, grading=None):
    """Five tasks at levels 1, 1, 2, 2, 3, with accents in the questions and contexts. The gold is 0, which the
    stub's default answer ("0") hits exactly."""
    out = []
    for i, level in enumerate((1, 1, 2, 2, 3)):
        tid = f"smoke-L{level}-{i:02d}"
        context = f"Context of {tid}: caf\u00e9 \u00fcn\u00efcode\nsecond line {'x' * (50 * (i + 1))}\nthird line\n"
        out.append(Task(tid, FAMILY, question or f"How many entries mention topic {i}? (caf\u00e9, na\u00efve)",
                        context, answer=answer, level=level, grading=dict(grading or {"type": "int", "blind": 7})))
    return out


def slot_a():
    return hs.Slot("a", MODEL_A, "default", "auto", dict(PRICE_A), True, "native")


def slot_b():
    return hs.Slot("b", MODEL_B, "default", "fenced", dict(PRICE_B), False, "fenced")


def read(path):
    return Path(path).read_bytes().decode("utf-8")


def jsonl(path):
    return [json.loads(line) for line in read(path).split("\n") if line.strip()]


def cells(report, label):
    """The cells of a `| label | ... |` row of a report table."""
    for line in report.split("\n"):
        if line.startswith(f"| {label} |"):
            return [c.strip() for c in line.strip().strip("|").split("|")][1:]
    raise AssertionError(f"no row {label!r} in the report")


def wait_for(path, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if Path(path).is_file():
            return True
        time.sleep(0.02)
    return Path(path).is_file()


def pid_gone(pid, timeout=10.0):
    """No live process has this pid (a zombie counts as gone). Windows has no cheap check, and os.kill would kill."""
    if os.name == "nt":
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            state = ""
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


def windows_write_text(self, data, encoding=None, errors=None, newline=None):
    """Path.write_text as Windows does it: \n becomes \r\n. Patched in, so that a report or a task file written in text
    mode, where the script means LF on every OS (D-10), fails here on Linux and macOS too."""
    if newline is None:
        data = data.replace("\n", "\r\n")
    return _WRITE_TEXT(self, data, encoding=encoding, errors=errors, newline="")


class SmokeCase(unittest.TestCase):
    """A temp world: the stub, a catalog, reference tokens, hand-made tasks and every module path patched into it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="smoke run ", ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.stub = write_stub(self.tmp / "rlmh stub")
        self.results = self.tmp / "results" / "harness-smoke"
        self.summary = self.tmp / "documentation" / "harness-smoke-v0.md"
        self.scratch = self.tmp / "temp dir" / "rlmh-smoke"
        self.tasks = make_tasks()
        self.catalog = self.tmp / "catalog.json"
        self.catalog.write_text(json.dumps(CATALOG), encoding="utf-8")
        self.tokens = self.tmp / "tokens.jsonl"
        self.tokens.write_text("\n".join(json.dumps(
            {"run_id": f"ref{i}", "root": {"provider": "deepseek"}, "infra_error": None, "turns": [{}] * 3,
             "subcalls": [{}] * 4,
             "usage": {"root": {"input_tokens": REF_TOKENS["root_in"], "output_tokens": REF_TOKENS["root_out"]},
                       "sub": {"input_tokens": REF_TOKENS["sub_in"], "output_tokens": REF_TOKENS["sub_out"]}}})
            for i in range(2)) + "\n", encoding="utf-8")
        for name, value in (("RESULTS", self.results), ("SUMMARY", self.summary),
                            ("DEFAULT_TOKENS", self.tmp / "no-transcripts")):
            self.patch(hs, name, value)
        self.patch(hs, "task_folder", lambda date: self.scratch / date)
        self.patch(hs, "build_tasks", lambda: list(self.tasks))
        self.patch(hs, "check_ignored", lambda: None)
        self.patch(hs, "rlm_commit", lambda: "0123456789abcdef")
        # `rlmh --version` and the five `config get` reads are six process starts a mode: the tests that are about
        # them use real_info(), the rest read the same values from the stub's config without starting it
        self.info_patcher = mock.patch.object(hs, "rlmh_info", self.fast_info)
        self.info_patcher.start()
        self.addCleanup(self.info_patcher.stop)

        def no_network(*args, **kwargs):
            raise AssertionError("the smoke tests make no network call")

        self.patch(Path, "write_text", windows_write_text)
        self.patch(socket.socket, "connect", no_network)
        self.patch(socket.socket, "connect_ex", no_network)

    def patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    @contextlib.contextmanager
    def real_info(self):
        """Inside, rlmh_info starts the stub for `--version` and each `config get`, as the real one starts rlmh."""
        self.info_patcher.stop()
        try:
            yield
        finally:
            self.info_patcher.start()

    def fast_info(self, cmd):
        config = {"call_format": "auto", "permissions.mode": "accept-edits", "root.thinking": "off",
                  "sub.thinking": "off", "profile": "default"}
        path = self.stub.parent / "stub_config.json"
        if path.is_file():
            config.update(json.loads(read(path)).get("config", {}))
        return {"version": "rlmh 0.1.0", "config": config}

    # -- the stub
    def stub_config(self, **kw):
        path = self.stub.parent / "stub_config.json"
        cfg = json.loads(read(path)) if path.is_file() else {}
        cfg.update(kw)
        path.write_bytes(json.dumps(cfg).encode("utf-8"))

    def calls(self):
        path = self.stub.parent / "calls.jsonl"
        return jsonl(path) if path.is_file() else []

    def batch_calls(self):
        return [c for c in self.calls() if c["rest"][:1] == ["batch"]]

    def cmd(self, *opts):
        """The command run_slot/run_batch get: a .py path runs under this Python, with the stub's options after."""
        return [*hs.rlmh_command(str(self.stub)), *opts]

    # -- main() and the pieces under it
    def main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = hs.main(list(argv))
            except SystemExit as e:  # argparse
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def cli(self, mode, *extra, models=(MODEL_A, MODEL_B), date=DATE):
        argv = [mode, "--rlmh", str(self.stub), "--catalog", str(self.catalog), "--tokens-from", str(self.tokens),
                "--date", date, "--harness", str(self.tmp / "no-harness")]
        if models:
            argv += ["--model-a", models[0], "--model-b", models[1]]
        return self.main(*argv, *extra)

    def go(self, *extra, date=DATE, **kw):
        """The real flow: --estimate writes the task files, --go refuses to start unless they still match."""
        code, _, err = self.cli("--estimate", *extra, date=date)
        self.assertEqual(code, 0, err)
        return self.cli("--go", *extra, date=date, **kw)

    def run_dir(self, date=DATE):
        return self.results / date

    def slot_dir(self, name, date=DATE):
        return {"a": slot_a(), "b": slot_b()}[name].dir(self.run_dir(date))

    # -- run_batch without main()
    def batch(self, *opts, slots=None, cap=None, jobs=2, out=None):
        folder = self.tmp / "tasks folder"
        hs.write_task_files(self.tasks, folder)
        out = out or self.results / "batch"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stop = hs.run_batch(slots or [slot_a(), slot_b()], self.tasks, folder, out, self.cmd(*opts), jobs, cap)
        return SimpleNamespace(stop=stop, out=out, folder=folder, printed=buf.getvalue())

    def run_json(self, out, stop=None):
        hs.write_json(out / "run.json", {
            "date": DATE, "mode": "go", "rlmh_version": "rlmh 0.1.0", "config": {}, "rlm_commit": "0123abc",
            "generator": "aggregation-v4", "platform": "test", "estimate": None, "reference_notes": [],
            "stopped": stop, "tasks": [{"task_id": t.task_id, "level": t.level, "chars": len(t.context)}
                                       for t in self.tasks]})


# ----------------------------------------------------------------------------- the task set and its files


class TaskTests(SmokeCase):
    @unittest.skipUnless(huffpost_present(), "needs the HuffPost data root (rlm_rl.data.available())")
    def test_plan_and_task_files(self):
        real_seeds = tasks.PRIVATE_DIR / "seeds" / f"{hs.SET_NAME}.json"  # the real file must stay as it is
        before = real_seeds.read_bytes() if real_seeds.exists() else None
        private = self.tmp / "private"
        with mock.patch.object(tasks, "PRIVATE_DIR", private):
            first = REAL.build_tasks()
            seed_file = private / "seeds" / "harness-smoke-v1.json"
            self.assertTrue(seed_file.is_file(), "the seeds are drawn once into private/seeds/")
            seeds = json.loads(read(seed_file))
            second = REAL.build_tasks()
            self.assertEqual(json.loads(read(seed_file)), seeds, "the second build reuses the seeds")
        self.assertEqual(real_seeds.read_bytes() if real_seeds.exists() else None, before)

        self.assertEqual(len(first), 5)
        self.assertEqual([t.level for t in first], [1, 1, 2, 2, 3])
        self.assertEqual([t.task_id for t in first], [f"harness-smoke-v1-{FAMILY}-L{lv}-{i:02d}"
                                                      for lv, i in ((1, 0), (1, 1), (2, 0), (2, 1), (3, 0))])
        self.assertTrue(all(t.family == FAMILY for t in first))
        for t in first:
            self.assertFalse(any(str(s) in t.task_id for s in seeds), "a task id never holds a seed")
        self.assertEqual([hs.context_sha(t.context) for t in first], [hs.context_sha(t.context) for t in second])
        self.assertEqual([t.question for t in first], [t.question for t in second])

        folder = self.tmp / "ctx folder"
        path = hs.write_task_files(first, folder)
        self.assertEqual(path, folder / "tasks.jsonl")
        lines = read(path).split("\n")
        self.assertEqual(lines[-1], "", "one JSON object per line, the file ends in a newline")
        lines = lines[:-1]
        self.assertEqual(len(lines), 5)
        for line, t in zip(lines, first):
            obj = json.loads(line)
            self.assertEqual(set(obj), {"id", "question", "input"})
            self.assertEqual(obj, {"id": t.task_id, "question": t.question, "input": f"ctx/{t.task_id}.txt"})
            self.assertNotIn("\\", obj["input"])
            raw = (folder / "ctx" / f"{t.task_id}.txt").read_bytes()
            self.assertEqual(raw, t.context.encode("utf-8"))
            self.assertEqual(raw.count(b"\r"), t.context.count("\r"), "text mode would have added a \\r per line")
            golds = t.answer if isinstance(t.answer, list) else [t.answer]
            for g in map(str, golds):  # a gold that the question itself holds can't be told apart in the line
                if len(g) >= 3 and g not in t.question:
                    self.assertNotIn(g, line)
        self.assertEqual(hs.check_task_files(first, folder), [])

    def test_task_files_offline(self):
        """The same file rules on hand-made tasks, so they are checked without the data root."""
        text = "Pergunta \u00e1\u00e9\u00ed: quantos?"
        built = make_tasks(question=text, answer="SENTINEL-GOLD-1234567890")
        built[0].context = "line one\r\nline two\n"  # a \r in a context must survive byte for byte
        folder = self.tmp / "task files"
        path = hs.write_task_files(built, folder)
        raw = path.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw.count(b"\r"), 0)
        lines = raw.decode("utf-8").split("\n")[:-1]
        self.assertEqual(len(lines), 5)
        for line, t in zip(lines, built):
            self.assertEqual(json.loads(line), {"id": t.task_id, "question": text, "input": f"ctx/{t.task_id}.txt"})
            self.assertNotIn("SENTINEL-GOLD", line)
            self.assertNotIn("blind", line)
            self.assertEqual((folder / "ctx" / f"{t.task_id}.txt").read_bytes(), t.context.encode("utf-8"))
        self.assertEqual((folder / "ctx" / f"{built[0].task_id}.txt").read_bytes(), b"line one\r\nline two\n")
        self.assertEqual((folder / "ctx" / f"{built[1].task_id}.txt").read_bytes().count(b"\r"), 0)
        self.assertEqual(hs.check_task_files(built, folder), [])
        # a rerun file holds only the tasks it is given, in the same shape, next to ctx/
        again = hs.write_task_files(built[1:3], folder, "tasks-rerun.jsonl")
        self.assertEqual(again, folder / "tasks-rerun.jsonl")
        self.assertEqual([json.loads(x)["id"] for x in read(again).split("\n") if x], [t.task_id for t in built[1:3]])

    def test_check_task_files(self):
        folder = self.tmp / "task files"
        hs.write_task_files(self.tasks, folder)
        self.assertEqual(hs.check_task_files(self.tasks, folder), [])
        victim = self.tasks[2].task_id
        (folder / "ctx" / f"{victim}.txt").write_bytes(self.tasks[2].context.replace("x", "y", 1).encode("utf-8"))
        self.assertEqual(hs.check_task_files(self.tasks, folder), [f"ctx/{victim}.txt"])
        (folder / "ctx" / f"{victim}.txt").write_bytes(self.tasks[2].context.encode("utf-8"))
        (folder / "tasks.jsonl").write_bytes(b"{}\n")
        self.assertEqual(hs.check_task_files(self.tasks, folder), ["tasks.jsonl"])
        (folder / "ctx" / f"{self.tasks[0].task_id}.txt").unlink()
        self.assertEqual(hs.check_task_files(self.tasks, folder), [f"ctx/{self.tasks[0].task_id}.txt", "tasks.jsonl"])
        self.assertEqual(hs.check_task_files(self.tasks, self.tmp / "nowhere")[-1], "tasks.jsonl")

    def test_build_tasks_checks_the_plan(self):
        """build_tasks asks fixed_set for the set the spec names, and refuses anything but levels 1, 1, 2, 2, 3."""
        self.assertEqual(hs.SET_NAME, "harness-smoke-v1")
        self.assertEqual(hs.FAMILY, FAMILY)
        self.assertEqual(hs.PLAN, {(FAMILY, 1): 2, (FAMILY, 2): 2, (FAMILY, 3): 1})
        good = make_tasks()
        with mock.patch.object(tasks, "fixed_set", return_value=good) as fixed:
            self.assertEqual(REAL.build_tasks(), good)
        fixed.assert_called_once_with("harness-smoke-v1", {(FAMILY, 1): 2, (FAMILY, 2): 2, (FAMILY, 3): 1})
        wrong_levels = make_tasks()
        wrong_levels[4].level = 4
        with mock.patch.object(tasks, "fixed_set", return_value=wrong_levels):
            with self.assertRaises(hs.SetupError):
                REAL.build_tasks()
        wrong_family = make_tasks()
        wrong_family[0].family = "aggregation"
        with mock.patch.object(tasks, "fixed_set", return_value=wrong_family):
            with self.assertRaises(hs.SetupError):
                REAL.build_tasks()
        with mock.patch.object(tasks, "fixed_set", return_value=good[:4]):
            with self.assertRaises(hs.SetupError):
                REAL.build_tasks()

    def test_task_folder_is_outside_the_repo(self):
        self.assertEqual(REAL.task_folder(DATE), Path(tempfile.gettempdir()) / "rlmh-smoke" / DATE)

    def test_context_sha_is_the_run_started_hash(self):
        self.assertEqual(hs.context_sha("caf\u00e9\n"), hashlib.sha256("caf\u00e9\n".encode("utf-8")).hexdigest())



# ----------------------------------------------------------------------------- slots, prices, tokens, estimate


class EstimateTests(SmokeCase):
    def args(self, **over):
        base = dict(model_a=MODEL_A, model_b=MODEL_B, profile_a=None, profile_b=None, format_a="auto",
                    format_b="fenced", price_a=None, price_b=None)
        return SimpleNamespace(**{**base, **over})

    def test_no_default_models(self):
        for argv in ([], ["--estimate"], ["--go"], ["--rehearse"], ["--estimate", "--model-a", MODEL_A],
                     ["--estimate", "--model-b", MODEL_B]):
            with self.subTest(argv=argv):
                code, out, err = self.main(*argv, "--catalog", str(self.catalog), "--date", DATE)
                self.assertEqual(code, 2)
                self.assertIn("9a has no default", err)
                self.assertIn("--model-a", err)
                self.assertIn("--model-b", err)
                self.assertEqual(out, "")
        self.assertFalse(self.results.exists())
        self.assertFalse(self.scratch.exists())
        with self.assertRaises(hs.SetupError):
            hs.make_slots(self.args(model_b=None), CATALOG)

    def test_bad_flags(self):
        for argv, needle in ((["--date", "2026-1-2"], "--date"), (["--jobs", "0"], "positive"),
                             (["--margin", "0"], "positive"), (["--cost-cap", "-1"], "positive"),
                             (["--model-a", "nomodel"], "--model-a"), (["--price-a", "1,2,3"], "--price-a"),
                             (["--price-a", "1,2,x,4"], "--price-a"), (["--price-b", "1,-2,3,4"], "--price-b")):
            with self.subTest(argv=argv):
                argv = [*argv]
                if "--date" not in argv:
                    argv += ["--date", DATE]
                code, _, err = self.main("--estimate", "--rlmh", str(self.stub), "--catalog", str(self.catalog),
                                         "--model-a", MODEL_A, "--model-b", MODEL_B, *argv)
                self.assertEqual(code, 2)
                self.assertIn(needle, err)
        for argv in (["--format-a", "bogus"], ["--only", "c"], ["--estimate", "--go"]):
            with self.subTest(argv=argv):
                self.assertEqual(self.main(*argv)[0], 2)  # argparse's own exit 2

    def test_slot_formats(self):
        a, b = hs.make_slots(self.args(), hs.load_catalog(self.catalog))
        self.assertEqual((a.fmt, b.fmt), ("auto", "fenced"), "S-32: slot b is forced to fenced, slot a follows auto")
        self.assertEqual((a.profile, b.profile), ("default", "default"))
        cmd, tasks_file, out = ["rlmh"], Path("tasks.jsonl"), Path("out")
        ca, cb = (hs.batch_command(cmd, tasks_file, s, out / s.name, 5) for s in (a, b))
        self.assertNotIn("--call-format", ca)
        self.assertEqual(cb[cb.index("--call-format") + 1], "fenced")
        self.assertEqual(ca, ["rlmh", "batch", "tasks.jsonl", "--json", "--jobs", "5", "--out", str(out / "a"),
                              "--root", MODEL_A, "--sub", MODEL_A, "--profile", "default"])
        self.assertEqual(cb, [*ca[:7], str(out / "b"), "--root", MODEL_B, "--sub", MODEL_B, "--profile", "default",
                              "--call-format", "fenced"])
        native = hs.make_slots(self.args(format_a="native"), hs.load_catalog(self.catalog))[0]
        self.assertEqual(hs.batch_command(cmd, tasks_file, native, out, 5)[-2:], ["--call-format", "native"])
        for c in (ca, cb):
            self.assertNotIn("--permission-mode", c)

        # the same through main(): the argument lists the stub is really started with
        code, _, err = self.go("--jobs", "3")
        self.assertEqual(code, 0, err)
        call_a, call_b = self.batch_calls()
        self.assertNotIn("--call-format", call_a["rest"])
        self.assertEqual(call_b["flags"]["--call-format"], "fenced")
        for call, slot in ((call_a, slot_a()), (call_b, slot_b())):
            self.assertEqual(call["rest"][0], "batch")
            self.assertEqual(call["flags"]["--root"], slot.ref)
            self.assertEqual(call["flags"]["--sub"], slot.ref)
            self.assertEqual(call["flags"]["--profile"], "default")
            self.assertEqual(call["flags"]["--jobs"], "3")
            self.assertEqual(call["flags"]["--out"], str(slot.dir(self.run_dir())))
            self.assertTrue(call["flags"]["--json"])
            self.assertNotIn("--permission-mode", call["rest"])
        # --format-a native adds the flag to slot a; --format-b auto takes it off slot b
        before = len(self.batch_calls())
        code, _, err = self.go("--format-a", "native", "--format-b", "auto", date="2026-01-03")
        self.assertEqual(code, 0, err)
        call_a, call_b = self.batch_calls()[before:]
        self.assertEqual(call_a["flags"]["--call-format"], "native")
        self.assertNotIn("--call-format", call_b["rest"])
        for call in (call_a, call_b):
            self.assertNotIn("--permission-mode", call["rest"])
        # --only runs one slot
        before = len(self.batch_calls())
        code, _, err = self.go("--only", "b", date="2026-01-04")
        self.assertEqual(code, 0, err)
        only = self.batch_calls()[before:]
        self.assertEqual([c["flags"]["--root"] for c in only], [MODEL_B])
        self.assertEqual(len(jsonl(self.run_dir("2026-01-04") / "grades.jsonl")), 5)

    def test_make_slots_defaults(self):
        catalog = hs.load_catalog(self.catalog)
        local = hs.make_slots(self.args(model_a="local/qwen", model_b="anthropic/FAKE-claude"), catalog)
        self.assertEqual([s.profile for s in local], ["training-v4", "default"])
        self.assertEqual(local[0].price, dict.fromkeys(PRICE_A, 0.0))
        custom = hs.make_slots(self.args(profile_a="training-v4", profile_b="default", format_b="native"), catalog)
        self.assertEqual([(s.profile, s.fmt) for s in custom], [("training-v4", "auto"), ("default", "native")])
        self.assertEqual([s.tool_call for s in custom], [True, False])
        self.assertEqual(custom[0].slug, "deepseek-FAKE-chat")
        self.assertEqual(hs.Slot("a", "custom/My Model:7b/", "p", "auto", {}).slug, "custom-My-Model-7b")
        self.assertEqual(hs.Slot("b", "openrouter/anthropic/claude-haiku-5.5", "p", "auto", {}).slug,
                         "openrouter-anthropic-claude-haiku-5.5")
        self.assertEqual(custom[0].provider, "deepseek")
        self.assertEqual(custom[1].dir(Path("out")), Path("out") / "b-openrouter-FAKE-no-tools")

    def test_expected_format(self):
        """architecture 7.4: auto is native for an API provider's tool_call model, fenced for everything else."""
        def slot(ref, tool_call, fmt="auto"):
            return hs.Slot("a", ref, "default", fmt, {}, tool_call)

        cases = [(slot("anthropic/x", True), "auto", "native"), (slot("openai/x", True), "auto", "native"),
                 (slot("deepseek/x", True), None, "native"), (slot("openrouter/x", True), "auto", "native"),
                 (slot("gemini/x", True), "auto", "native"), (slot("fake/scripted", True), "auto", "native"),
                 (slot("deepseek/x", False), "auto", "fenced"), (slot("deepseek/x", None), "auto", "fenced"),
                 (slot("local/x", True), "auto", "fenced"), (slot("custom/x", True), "auto", "fenced"),
                 (slot("deepseek/x", True), "fenced", "fenced"), (slot("local/x", False), "native", "native"),
                 (slot("deepseek/x", True, "fenced"), "native", "fenced"),
                 (slot("deepseek/x", False, "native"), "auto", "native")]
        for s, setting, want in cases:
            with self.subTest(ref=s.ref, tool_call=s.tool_call, fmt=s.fmt, setting=setting):
                self.assertEqual(hs.expected_format(s, setting), want)
        self.assertIn("S-32", hs.format_why("fenced", "fenced"))
        self.assertIn("native path goes untested", hs.format_why("auto", "fenced"))
        self.assertIn("own profile", hs.format_why("auto", "native"))

    def test_load_prices_layouts(self):
        # step 5's layout loads, keyed by (provider, id)
        catalog = hs.load_catalog(self.catalog)
        self.assertEqual(set(catalog), {("deepseek", "FAKE-chat"), ("openrouter", "FAKE-no-tools"),
                                        ("anthropic", "FAKE-claude")})
        self.assertEqual(hs.load_prices(MODEL_A, catalog, None, "a"), (PRICE_A, True))
        self.assertEqual(hs.load_prices(MODEL_B, catalog, None, "b"), (PRICE_B, False))
        self.assertEqual(hs.load_prices(MODEL_A, catalog, "5,6,7,8", "a")[0],
                         {"input": 5.0, "output": 6.0, "cache_read": 7.0, "cache_write": 8.0})
        # any other layout exits 2 naming the file
        good = CATALOG["models"][0]
        without = lambda key: {**CATALOG, "models": [{k: v for k, v in good.items() if k != key}]}  # noqa: E731
        bad_docs = {"a bare list": [good], "no models": {"version": 1, "source": {}},
                    "version 2": {**CATALOG, "version": 2}, "no source": {"version": 1, "models": [good]},
                    "models not a list": {**CATALOG, "models": {}}, "entry without price": without("price"),
                    "entry without tool_call": without("tool_call"),
                    "entry not an object": {**CATALOG, "models": [good, "x"]}, "not an object": 7}
        for name, doc in bad_docs.items():
            with self.subTest(layout=name):
                path = self.tmp / f"bad {name}.json"
                path.write_text(json.dumps(doc), encoding="utf-8")
                with self.assertRaises(hs.SetupError) as caught:
                    hs.load_catalog(path)
                self.assertIn(str(path), str(caught.exception))
                code, _, err = self.cli("--estimate", "--catalog", str(path))
                self.assertEqual(code, 2)
                self.assertIn(str(path), err)
        broken = self.tmp / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        for path in (broken, self.tmp / "missing.json"):
            with self.subTest(file=path.name), self.assertRaises(hs.SetupError) as caught:
                hs.load_catalog(path)
            self.assertIn(str(path), str(caught.exception))

    @unittest.skipUnless((ROOT.parent / "rlm-harness" / "app" / "catalog" / "models.json").is_file(),
                         "needs ../rlm-harness (step 5's bundled catalog)")
    def test_real_catalog_layout(self):
        catalog = hs.load_catalog(ROOT.parent / "rlm-harness" / "app" / "catalog" / "models.json")
        self.assertGreater(len(catalog), 100)
        for (provider, name), entry in catalog.items():
            price, tool_call = hs.load_prices(f"{provider}/{name}", catalog, None, "a")
            self.assertEqual(set(price), {"input", "output", "cache_read", "cache_write"})
            self.assertIsInstance(tool_call, bool)

    def test_profile_limits(self):
        # rlm-harness D-152: a profile's [read] keys go on top of READ_LIMITS_V4, and the ceiling follows them
        with tempfile.TemporaryDirectory() as tmp:
            harness, home = Path(tmp) / "harness", Path(tmp) / "home"
            built_in = harness / "engine" / "rlm_engine" / "profiles" / "default"
            built_in.mkdir(parents=True)
            (built_in / "profile.toml").write_text('contract = "v4"\n[read]\nmax_turns = 50\nroot_max_tokens = 16384\n'
                                                   'subcall_max_tokens = 16384\n[repo]\nmax_turns = 150\n', encoding="utf-8")
            mine = home / "profiles" / "mine"
            mine.mkdir(parents=True)
            (mine / "profile.toml").write_text('contract = "v4"\n[read]\nmax_turns = 7\n', encoding="utf-8")
            lim, source = hs.profile_limits("default", harness, home)
            self.assertEqual((lim.max_turns, lim.root_max_tokens, lim.subcall_max_tokens, lim.max_subcalls),
                             (50, 16384, 16384, bridge.READ_LIMITS_V4.max_subcalls))
            self.assertEqual(source, "the `default` profile's read limits")
            self.assertEqual(hs.profile_limits("mine", harness, home)[0].max_turns, 7)
            lim0, source0 = hs.profile_limits("nowhere", harness, home)
            self.assertEqual(lim0, bridge.READ_LIMITS_V4)
            self.assertIn("no `nowhere` profile.toml", source0)
            # each slot's ceiling uses its own limits; the report's line shows the largest
            a = hs.Slot("a", "openrouter/x", "default", "auto", PRICE_A, True, limits=lim, limits_source=source)
            b = hs.Slot("b", "openrouter/y", "training-v4", "fenced", PRICE_A, True)
            est = hs.estimate([a, b], hs.Reference(), 5, 1.5, None)
            self.assertEqual(est["ceiling_tokens"], hs.ceiling_tokens(lim))
            self.assertEqual(est["ceiling_source"], source)
            self.assertGreater(est["slots"]["a"]["ceiling_run"], 4 * est["slots"]["b"]["ceiling_run"])
            self.assertAlmostEqual(est["slots"]["b"]["ceiling_run"], hs.run_cost(hs.ceiling_tokens(), PRICE_A), places=9)
            # each slot keeps its own source, so a slot that fell back to training's limits is named
            self.assertEqual((est["slots"]["a"]["ceiling_source"], est["slots"]["b"]["ceiling_source"]),
                             (source, "`READ_LIMITS_V4`"))
            self.assertEqual(est["slots"]["b"]["ceiling_tokens"], hs.ceiling_tokens())

    def test_estimate_math(self):
        ref = hs.Reference(n=2, root_in=REF_TOKENS["root_in"], root_out=REF_TOKENS["root_out"],
                           sub_in=REF_TOKENS["sub_in"], sub_out=REF_TOKENS["sub_out"])
        # the ceiling, by hand, for READ_LIMITS_V4: 20 turns, 80 sub-calls, 4096-token replies, 8000-char outputs
        lim = bridge.READ_LIMITS_V4
        self.assertEqual((lim.max_turns, lim.max_subcalls, lim.root_max_tokens, lim.output_max_chars,
                          lim.subcall_max_chars, lim.subcall_max_tokens), (20, 80, 4096, 8000, 60_000, 4096))
        ceiling = hs.ceiling_tokens()
        self.assertAlmostEqual(ceiling["root_in"], 80_000 + 190 * (4096 + 8000 / 3))
        self.assertEqual(ceiling["sub_in"], 1_600_000)
        self.assertEqual(ceiling["root_out"], 81_920)
        self.assertEqual(ceiling["sub_out"], 327_680)
        self.assertEqual(round(ceiling["root_in"] + ceiling["sub_in"]), 2_964_907)
        self.assertEqual(ceiling["root_out"] + ceiling["sub_out"], 409_600)
        self.assertAlmostEqual(CEILING_IN, 2_964_906.6667, places=3)

        a, b = slot_a(), slot_b()
        est = hs.estimate([a, b], ref, 5, 1.5, None)
        # slot a: (100k x 1 + 10k x 4 + 200k x 1 + 20k x 4) / 1e6 = 0.42 a run; slot b is half the input price
        want = {"a": (0.42, 0.63, (CEILING_IN * 1.0 + CEILING_OUT * 4.0) / 1e6),
                "b": (0.21, 0.315, (CEILING_IN * 0.5 + CEILING_OUT * 2.0) / 1e6)}
        for name, (expected, planned, ceil_run) in want.items():
            s = est["slots"][name]
            self.assertAlmostEqual(s["expected_run"], expected)
            self.assertAlmostEqual(s["planned_run"], planned)
            self.assertAlmostEqual(s["ceiling_run"], ceil_run, places=6)
            self.assertAlmostEqual(s["expected"], expected * 5)
            self.assertAlmostEqual(s["planned"], planned * 5)
            self.assertAlmostEqual(s["ceiling"], ceil_run * 5, places=5)
        self.assertAlmostEqual(est["planned_total"], 3.15 + 1.575)
        self.assertAlmostEqual(est["cap"], 2 * 4.725, msg="the cap defaults to 2 x the planned total over both slots")
        self.assertEqual((est["runs"], est["margin"], est["reference_runs"]), (5, 1.5, 2))
        self.assertEqual(hs.estimate([a, b], ref, 5, 2.0, 3.5)["cap"], 3.5)
        self.assertAlmostEqual(hs.estimate([a], ref, 5, 2.0, None)["slots"]["a"]["planned_run"], 0.84)

        # no reference runs: expected is unknown and the ceiling stands in for planned (D-115)
        bare = hs.estimate([a], hs.Reference(), 5, 1.5, None)["slots"]["a"]
        self.assertIsNone(bare["expected"])
        self.assertIsNone(bare["expected_run"])
        self.assertEqual(bare["planned_run"], bare["ceiling_run"])

        # cache reads lower the cost: 1000 in at $1, 600 of them read at $0.25, 200 written at $1.5, 100 out at $4
        self.assertAlmostEqual(hs.role_cost(1000, 100, PRICE_A), (1000 + 400) / 1e6)
        self.assertAlmostEqual(hs.role_cost(1000, 100, PRICE_A, cache_read=600), (400 * 1 + 600 * 0.25 + 400) / 1e6)
        self.assertAlmostEqual(hs.role_cost(1000, 100, PRICE_A, cache_read=600, cache_write=200),
                               (200 * 1 + 600 * 0.25 + 200 * 1.5 + 400) / 1e6)
        self.assertLess(hs.role_cost(1000, 100, PRICE_A, cache_read=600), hs.role_cost(1000, 100, PRICE_A))
        self.assertAlmostEqual(hs.role_cost(1000, 0, PRICE_A, cache_read=1000), 1000 * 0.25 / 1e6)  # all of it cached
        self.assertAlmostEqual(hs.role_cost(1000, 0, PRICE_A, cache_read=700, cache_write=700),
                               (700 * 0.25 + 700 * 1.5) / 1e6, msg="the uncached part is never negative")
        self.assertAlmostEqual(hs.run_cost(REF_TOKENS, PRICE_A), 0.42)

        # provider local is $0 without a catalog lookup, and so is the fake provider
        for ref_name in ("local/anything", "fake/scripted"):
            price, _ = hs.load_prices(ref_name, None, None, "a")
            self.assertEqual(price, dict.fromkeys(PRICE_A, 0.0))
        free = hs.Slot("a", "local/qwen", "training-v4", "auto", hs.load_prices("local/qwen", None, None, "a")[0], None)
        e = hs.estimate([free], ref, 5, 1.5, None)["slots"]["a"]
        self.assertEqual((e["expected"], e["planned"], e["ceiling"]), (0.0, 0.0, 0.0))

        # through main(): the printed numbers, and nothing else happens
        code, out, err = self.cli("--estimate")
        self.assertEqual(code, 0, err)
        self.assertIn("per run: expected $0.4200, planned $0.6300, ceiling $4.6033", out)
        self.assertIn("5 runs: expected $2.1000, planned $3.1500, ceiling $23.0165", out)
        self.assertIn("per run: expected $0.2100, planned $0.3150, ceiling $2.3017", out)
        self.assertIn("planned total $4.7250, cost cap $9.4500", out)
        self.assertIn("root 1,364,907 in / 81,920 out, sub 1,600,000 in / 327,680 out", out)
        self.assertIn("nothing runs until --go", out)
        self.assertIn(f"L1 {len(self.tasks[0].context):,}", out)
        code, out, _ = self.cli("--estimate", "--cost-cap", "1.25", "--margin", "2")
        self.assertEqual(code, 0)
        self.assertIn("cost cap $1.2500", out)
        self.assertIn("planned $0.8400", out)
        code, out, err = self.cli("--estimate", models=("local/qwen", "local/llama"))
        self.assertEqual(code, 0, err)
        self.assertIn("per run: expected $0.0000, planned $0.0000, ceiling $0.0000", out)
        self.assertIn("profile training-v4, call format fenced (auto)", out)
        self.assertIn("note: slot a resolves to fenced, so the native path goes untested", out)
        # a missing catalog entry exits 2 and names the flag; --price-a/-b stand in for it
        for models, flag in (((MODEL_A, "deepseek/not-there"), "--price-b"),
                             (("openai/not-there", MODEL_B), "--price-a")):
            with self.subTest(flag=flag):
                code, out, err = self.cli("--estimate", models=models)
                self.assertEqual(code, 2)
                self.assertIn(flag, err)
                self.assertIn("not-there", err)
                self.assertEqual(out, "")
        code, out, err = self.cli("--estimate", "--price-b", "1,2,0.1,1.25", models=(MODEL_A, "deepseek/not-there"))
        self.assertEqual(code, 0, err)
        self.assertIn("price per 1M: in $1.0, out $2.0, cache read $0.1, cache write $1.25", out)

    def test_estimate_mode_runs_nothing(self):
        with self.real_info():
            code, out, err = self.cli("--estimate")
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        self.assertEqual(self.batch_calls(), [], "only --go spends")
        self.assertFalse(self.results.exists(), "no output folder before --go")
        self.assertFalse(self.summary.exists())
        names = [c["rest"] for c in self.calls()]
        self.assertEqual(names[0], ["--version"])
        self.assertEqual([n[2] for n in names[1:]], ["call_format", "permissions.mode", "root.thinking", "sub.thinking",
                                                     "profile"])
        self.assertIn("rlmh 0.1.0", out)
        self.assertIn("call_format = auto", out)
        # the task files are written (outside both repos), with the contexts as bytes
        folder = self.scratch / DATE
        self.assertEqual(hs.check_task_files(self.tasks, folder), [])
        self.assertIn(str(folder), out)

    def test_load_tokens(self):
        def harness(i, infra=None, provider="deepseek", **over):
            rec = {"run_id": f"r{i}", "root": {"provider": provider}, "infra_error": infra,
                   "usage": {"root": {"input_tokens": 1000 * i, "output_tokens": 100 * i},
                             "sub": {"input_tokens": 3000 * i, "output_tokens": 300 * i}},
                   "turns": [{}] * (2 + i), "subcalls": [{}] * (3 + i)}
            rec.update(over)
            return rec

        def transcript(i, family="aggregation_heldout", infra=None):
            return {"family": family, "infra_error": infra,
                    "usage": {"root_prompt_tokens": 1000 * i, "root_completion_tokens": 100 * i,
                              "subcall_prompt_tokens": 3000 * i, "subcall_completion_tokens": 300 * i,
                              "root_calls": 2 + i, "subcalls": 3 + i}}

        runs = self.tmp / "runs.jsonl"
        runs.write_text("\n".join(json.dumps(r) for r in [harness(1), harness(3), harness(2, infra="boom"),
                                                          harness(9, provider="fake"), harness(5, usage=None)])
                        + "\n{torn line\n\n", encoding="utf-8")
        folder = self.tmp / "transcripts" / "nested"
        folder.mkdir(parents=True)
        (folder / "ep.jsonl").write_text("\n".join(json.dumps(r) for r in [
            transcript(1), transcript(3, family="aggregation"), transcript(2, infra="oom"),
            transcript(7, family="callgraph"), {"family": "aggregation", "usage": {}}]) + "\n", encoding="utf-8")
        (folder.parent / "other.txt").write_text("not jsonl", encoding="utf-8")

        want = dict(root_in=2000.0, root_out=200.0, sub_in=6000.0, sub_out=600.0, turns=4.0, subcalls=5.0)
        for label, paths in (("a run log", [runs]), ("a transcripts folder", [folder.parent])):
            with self.subTest(label):
                ref = hs.load_tokens(paths)
                self.assertEqual(ref.n, 2, "infra, fake-provider, foreign-family and unusable records are skipped")
                for key, value in want.items():
                    self.assertAlmostEqual(getattr(ref, key), value, msg=key)
                self.assertEqual(ref.tokens(), {k: want[k] for k in ("root_in", "root_out", "sub_in", "sub_out")})
                self.assertEqual(ref.notes, [f"{paths[0]}: 2 runs"], "n and the source are printed")
        ref = hs.load_tokens([runs, folder / "ep.jsonl", self.tmp / "gone"])
        self.assertEqual(ref.n, 4)
        self.assertAlmostEqual(ref.root_in, 2000.0)
        self.assertAlmostEqual(ref.turns, 4.0)
        self.assertEqual(ref.notes, [f"{runs}: 2 runs", f"{folder / 'ep.jsonl'}: 2 runs",
                                     f"{self.tmp / 'gone'}: missing"])
        empty = hs.load_tokens([self.tmp / "gone"])
        self.assertEqual((empty.n, empty.root_in), (0, 0.0))
        # unequal records average, they are not summed or taken from the first
        runs2 = self.tmp / "runs2.jsonl"
        runs2.write_text(json.dumps(harness(2)) + "\n" + json.dumps(harness(6)) + "\n" + json.dumps(harness(4)) + "\n",
                         encoding="utf-8")
        self.assertAlmostEqual(hs.load_tokens([runs2]).sub_in, 12000.0)
        # the row rules themselves
        self.assertIsNone(hs.token_row("x"))
        self.assertIsNone(hs.token_row({"usage": "x"}))
        self.assertEqual(hs.token_row(harness(1)), (1000, 100, 3000, 300, 3, 4))
        self.assertEqual(hs.token_row(transcript(1)), (1000, 100, 3000, 300, 3, 4))
        self.assertEqual(hs.token_row(transcript(1, family="aggregation")), (1000, 100, 3000, 300, 3, 4))

    def test_pure_formatting_helpers(self):
        self.assertEqual((hs._usd(None), hs._usd(1.23456), hs._usd(1234.5, 2)), ("n/a", "$1.2346", "$1,234.50"))
        self.assertEqual((hs._pct(None), hs._pct(0.12345), hs._num(None), hs._num(1234.56)), ("n/a", "12.3%", "n/a",
                                                                                               "1,234.6"))
        self.assertEqual(hs.rlmh_command("C:/x y/rlmh.exe"), ["C:/x y/rlmh.exe"])
        self.assertEqual(hs.rlmh_command("/tmp/s p/stub.py"), [sys.executable, "/tmp/s p/stub.py"])
        with mock.patch.object(hs.shutil, "which", return_value=None):
            with self.assertRaises(hs.SetupError) as caught:
                hs.rlmh_command(None)
        self.assertIn("--rlmh", str(caught.exception))
        with mock.patch.object(hs.shutil, "which", return_value="/usr/bin/rlmh"):
            self.assertEqual(hs.rlmh_command(None), ["/usr/bin/rlmh"])


# ----------------------------------------------------------------------------- the event stream


def event(kind, **kw):
    return json.dumps({"type": kind, **kw})


class StreamTests(SmokeCase):
    def test_parse_stream(self):
        self.stub_config(garbage=True, denied={self.tasks[0].task_id: 2}, allowed={self.tasks[0].task_id: 3})
        folder = self.tmp / "tasks folder"
        tasks_file = hs.write_task_files(self.tasks, folder)
        a = slot_a()
        out = self.results / "slot"
        args = hs.batch_command(self.cmd(), tasks_file, a, out, 2)
        self.assertIsInstance(args, list)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            attempt = hs.run_slot(args, out, label="a")
        stream = attempt.stream
        self.assertEqual((attempt.exit_code, attempt.capped), (0, False))
        ids = [t.task_id for t in self.tasks]
        self.assertEqual(list(stream.started), ids)
        self.assertEqual(list(stream.finished), ids)
        self.assertEqual(stream.tags, {f"r-{i + 1}": tid for i, tid in enumerate(ids)})
        for t in self.tasks:
            started = stream.started[t.task_id]
            self.assertEqual(started["profile"], "default")
            self.assertEqual(started["call_format"], "native")
            self.assertEqual(started["input"]["chars"], len(t.context))
            self.assertEqual(started["input"]["sha256"], hs.context_sha(t.context))
            f = stream.finished[t.task_id]
            self.assertEqual((f["tag"], f["answer"], f["submitted"], f["answer_valid"]), (t.task_id, "0", True, True))
            self.assertEqual((f["turns"], f["subcalls"], f["cost_usd"]), (3, 4, 0.25))
        self.assertEqual(stream.calls, 10)
        self.assertAlmostEqual(stream.cost, 1.25, msg="every gateway.call's usage.cost_usd")
        self.assertEqual(stream.denied, {ids[0]: 2}, "denied actions are counted per task; allowed ones are not")
        self.assertEqual(stream.batch["type"], "batch.finished")
        self.assertEqual((stream.batch["runs"], stream.batch["failed"]), (5, 0))
        # a garbage line is reported by number and length, never by its text
        self.assertEqual(len(stream.bad_lines), 1)
        self.assertRegex(stream.bad_lines[0], r"^line 3: not a JSON event \(\d+ chars\)$")
        self.assertNotIn("garbage", stream.bad_lines[0])
        self.assertEqual(stream.lines, len(read(out / "events.jsonl").split("\n")) - 1)
        # events.jsonl keeps every line verbatim, the garbage line too: what the stub prints, byte for byte
        direct = subprocess.run(args, capture_output=True, stdin=subprocess.DEVNULL)
        self.assertEqual(direct.returncode, 0)
        raw = (out / "events.jsonl").read_bytes()
        self.assertIn("WARNING stub garbage line, not json: { caf\u00e9\n".encode("utf-8"), raw)
        self.assertEqual(raw, direct.stdout)
        self.assertEqual(read(out / "stderr.log").strip(), "stub warning on stderr")
        # the progress lines: task id, turns, sub-calls, cost so far
        lines = [ln for ln in buf.getvalue().split("\n") if ln]
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], f"[a] {ids[0]}: submitted, 3 turns, 4 sub-calls, $0.2500; so far $0.2500")
        self.assertTrue(lines[-1].endswith("so far $1.2500"), lines[-1])
        # a second batch into the same folder appends to events.jsonl, it does not truncate it
        with contextlib.redirect_stdout(io.StringIO()):
            hs.run_slot(args, out, label="a")
        self.assertEqual((out / "events.jsonl").read_bytes(), raw + direct.stdout)  # the same bytes again
        # read_stream reads the saved file back the way the live stream was read
        again = hs.read_stream(out / "events.jsonl")
        self.assertEqual(list(again.finished), ids)
        self.assertEqual(len(again.bad_lines), 2)
        self.assertEqual(hs.read_stream(out / "nothing.jsonl").lines, 0)

    def test_parse_line_cases(self):
        s = hs.Stream()
        self.assertIsNone(hs.parse_line("\n", s))
        self.assertEqual(s.bad_lines, [], "a blank line is no error")
        for text in ("plain text", "[1, 2]", '{"no": "type"}', '{"type": 7}', '"str"', "{torn"):
            self.assertIsNone(hs.parse_line(text + "\n", s), text)
        self.assertEqual(len(s.bad_lines), 6)
        self.assertEqual(s.lines, 7)
        unknown = hs.parse_line(event("run.cell", run_id="x"), s)
        self.assertEqual(unknown["type"], "run.cell")
        self.assertEqual(len(s.bad_lines), 6)
        # a permission before its run.started counts under the run id; allowed ones count nowhere
        hs.parse_line(event("run.permission", run_id="early", allowed=False), s)
        hs.parse_line(event("run.started", run_id="r1", tag="t1"), s)
        hs.parse_line(event("run.permission", run_id="r1", allowed=False), s)
        hs.parse_line(event("run.permission", run_id="r1", allowed=True), s)
        hs.parse_line(event("run.permission", run_id="r1"), s)
        self.assertEqual(dict(s.denied), {"early": 1, "t1": 2})
        # gateway.call: a missing or null cost is $0, an error is counted by kind, a kindless one as "other"
        hs.parse_line(event("gateway.call", usage={"cost_usd": 0.5}), s)
        hs.parse_line(event("gateway.call", usage=None), s)
        hs.parse_line(event("gateway.call", usage={"cost_usd": None}, error={"kind": "rate_limit"}), s)
        hs.parse_line(event("gateway.call", usage={"cost_usd": 0.25}, error={"message": "no kind"}), s)
        hs.parse_line(event("gateway.call"), s)
        self.assertEqual((s.calls, s.cost), (5, 0.75))
        self.assertEqual(dict(s.call_errors), {"rate_limit": 1, "other": 1})

    def test_gateway_costs_are_summed_from_the_stream_not_run_finished(self):
        s = hs.Stream()
        for cost in (0.5, 0.25):
            hs.parse_line(event("gateway.call", usage={"cost_usd": cost}), s)
        hs.parse_line(event("run.finished", tag="t", cost_usd=99.0), s)
        self.assertEqual(s.cost, 0.75)

    def test_run_slot_pipes_utf8_and_closes_stdin(self):
        """The stub's accents survive the pipe, and the child gets no stdin to wait on."""
        script = self.tmp / "echo stdin.py"
        script.write_bytes(
            b"import sys\n"
            b"data = sys.stdin.read()\n"
            b"line = '{\"type\": \"gateway.call\", \"note\": \"caf\\u00e9 \\u65e5\\u672c\", \"n\": %d}\\n'\n"
            b"sys.stdout.buffer.write((line % len(data)).encode('utf-8'))\n")
        out = self.tmp / "echo out"
        with contextlib.redirect_stdout(io.StringIO()):
            attempt = hs.run_slot([sys.executable, str(script)], out)
        self.assertEqual(attempt.exit_code, 0)
        self.assertEqual(attempt.stream.calls, 1)
        self.assertEqual(jsonl(out / "events.jsonl")[0]["n"], 0)


# ----------------------------------------------------------------------------- grading


class GradingTests(SmokeCase):
    def finished(self, **kw):
        return {"type": "run.finished", "tag": "t", "answer": None, "submitted": True, "answer_valid": True,
                "turns": 2, "subcalls": 1, "infra_error": None, "infra_kind": None, "wall_s": 1.0, "cost_usd": 0.0,
                "usage": {}, **kw}

    def cases(self):
        def task(tid, answer, grading):
            return Task(tid, FAMILY, "question", "context", answer=answer, level=1, grading=grading)

        ids = {"type": "list", "item": "entity", "blind": ["1", "2"]}
        return {
            "int": (task("g-int", 42, {"type": "int", "blind": 10}), "42"),
            "count exact": (task("g-count", 17, {"type": "count", "blind": 9, "scope_lines": 100}), "17"),
            "count near miss": (task("g-near", 17, {"type": "count", "blind": 9, "scope_lines": 100}), "15"),
            "list exact": (task("g-list", ["alpha", "beta"], {"type": "list", "item": "entity", "blind": ["gamma"]}),
                           ["Beta", "alpha"]),
            "list with labels": (task("g-ids", ["11952", "62016"], ids), "user 11952, user 62016"),
            "list wrong": (task("g-wrong", ["alpha", "beta"], {"type": "list", "item": "entity", "blind": ["gamma"]}),
                           ["alpha", "gamma"]),
        }

    def test_grading(self):
        for name, (task, answer) in self.cases().items():
            with self.subTest(name):
                got = hs.grade(self.finished(answer=answer), task)
                want = reward.score_answer(answer, task.answer, task.grading)  # the numbers come from the scorer
                self.assertEqual((got.reward, got.correct), (want.reward, want.correct))
                self.assertEqual(got.components, want.components)
        by = {name: hs.grade(self.finished(answer=a), t) for name, (t, a) in self.cases().items()}
        self.assertTrue(by["int"].correct and by["count exact"].correct and by["list exact"].correct)
        self.assertTrue(by["list with labels"].correct, "'user 11952' scores as '11952' (strip_id_labels)")
        self.assertFalse(by["count near miss"].correct)
        self.assertTrue(0 < by["count near miss"].reward < by["count exact"].reward, "partial credit")
        self.assertFalse(by["list wrong"].correct)
        # no submit, and answer_valid false, score as no answer, whatever the answer text says
        task, answer = self.cases()["int"]
        nothing = reward.score_answer(None, task.answer, task.grading)
        self.assertEqual((nothing.reward, nothing.correct), (0.0, False))
        for label, fields in (("no submit", {"submitted": False}), ("answer_valid false", {"answer_valid": False}),
                              ("neither", {"submitted": False, "answer_valid": False}),
                              ("no answer", {"answer": None})):
            with self.subTest(label):
                got = hs.grade(self.finished(**{"answer": answer, **fields}), task)
                self.assertEqual((got.reward, got.correct, got.components), (nothing.reward, False, nothing.components))
        # the answer is JSON, not always a string (D-89)
        task, _ = self.cases()["list exact"]
        self.assertTrue(hs.grade(self.finished(answer=["beta", "alpha"]), task).correct)
        self.assertTrue(hs.grade(self.finished(answer="alpha, beta"), task).correct)

    def test_final_rows_grade_each_task(self):
        """grade() through final_rows and grade_line: the rows and the grades.jsonl lines the report is built from."""
        cases = self.cases()
        built = [t for t, _ in cases.values()]
        answers = {t.task_id: a for t, a in cases.values()}
        stream = hs.Stream()
        for i, t in enumerate(built[:-1]):
            hs.parse_line(event("run.started", run_id=f"r{i}", tag=t.task_id, call_format="native"), stream)
            fin = self.finished(tag=t.task_id, answer=answers[t.task_id], run_id=f"r{i}")
            if t.task_id == "g-count":
                fin.update(submitted=False)
            if t.task_id == "g-near":
                fin.update(answer_valid=False)
            hs.parse_line(json.dumps(fin), stream)
        meta = {"slot": "a", "model": MODEL_A, "capped": False, "reruns": [], "not_run": None}
        rows = hs.final_rows(meta, stream, None, built)
        by = {r["task_id"]: r for r in rows}
        for t, answer in cases.values():
            row = by[t.task_id]
            if t.task_id == "g-wrong":
                self.assertEqual(row["status"], hs.INFRA, "no run.finished and no cap: batch_crashed")
                self.assertEqual(row["infra_kind"], "batch_crashed")
                self.assertIsNone(row["exact"])
                continue
            sent = None if t.task_id in ("g-count", "g-near") else answer
            want = reward.score_answer(sent, t.answer, t.grading)
            self.assertEqual((row["status"], row["exact"], row["reward"]), (hs.GRADED, want.correct, want.reward),
                             t.task_id)
            line = hs.grade_line(row, t)
            self.assertEqual((line["gold"], line["answer"], line["slot"], line["level"]), (t.answer, answer, "a", 1))
            self.assertEqual((line["exact"], line["reward"], line["components"]), (want.correct, want.reward,
                                                                                    want.components))
        self.assertEqual(by["g-count"]["finished"]["submitted"], False)
        self.assertEqual(hs.grade_line(by["g-count"], built[1])["submitted"], False)
        self.assertEqual(hs.grade_line(by["g-near"], built[2])["answer_valid"], False)

    def test_rerun_kinds(self):
        """Rerun for rate_limit, server, timeout, network, worker, other and batch_crashed; never for setup kinds."""
        rerun = ("rate_limit", "server", "timeout", "network", "worker", "other", "batch_crashed")
        setup = ("auth", "no_key", "unknown_model", "context_window", "bad_request", "cancelled")
        self.assertEqual(set(hs.RERUN_KINDS), set(rerun))
        self.assertEqual(set(hs.SETUP_KINDS), set(setup))
        for kind in rerun + setup:
            with self.subTest(kind):
                s = hs.Stream()
                if kind != "batch_crashed":
                    done = self.finished(tag="t1", infra_error=f"{kind} happened", infra_kind=kind)
                    hs.parse_line(json.dumps(done), s)
                reruns, failures = hs.plan_reruns(["t1"], hs.Attempt(s, 1, False))
                if kind in rerun:
                    self.assertEqual(failures, [])
                    self.assertEqual([(r["task_id"], r["kind"]) for r in reruns], [("t1", kind)])
                else:
                    self.assertEqual((len(reruns), [f["kind"] for f in failures]), (0, [kind]))
        s = hs.Stream()
        hs.parse_line(json.dumps(self.finished(tag="graded")), s)
        hs.parse_line(json.dumps(self.finished(tag="odd", infra_error="no kind given", infra_kind=None)), s)
        reruns, failures = hs.plan_reruns(["graded", "odd", "silent"], hs.Attempt(s, 1, False))
        self.assertEqual([(r["task_id"], r["kind"]) for r in reruns], [("odd", "other"), ("silent", "batch_crashed")])
        self.assertEqual(failures, [])
        self.assertEqual(reruns[1]["message"], "no run.finished: the batch ended without one")
        # a run the cap stopped is not an infra failure: it is not run
        capped = hs.Attempt(s, -15, True)
        self.assertEqual(hs.plan_reruns(["silent"], capped), ([], []))
        self.assertEqual(hs.outcome(s, True, "silent")[0], "not run: cost cap")
        self.assertEqual(hs.outcome(s, False, "graded")[0], hs.GRADED)
        self.assertEqual(hs.outcome(s, True, "graded")[0], hs.GRADED, "a run that finished before the cap is graded")

    def test_rerun_once(self):
        ids = [t.task_id for t in self.tasks]
        t0, t1, t2, t3, t4 = ids

        with self.subTest("a server failure is rerun once, and the rerun's result is graded"):
            before = len(self.batch_calls())
            run = self.batch("--fail", f"server:{t0}", slots=[slot_a()], out=self.results / "server")
            self.assertIsNone(run.stop)
            first, second = self.batch_calls()[before:]
            self.assertFalse(first["rerun"])
            self.assertTrue(second["rerun"])
            sdir = slot_a().dir(run.out)
            self.assertEqual(second["flags"]["--out"], str(sdir / "rerun"))
            self.assertEqual(second["tasks_file"], str(run.folder / "tasks-rerun.jsonl"))
            self.assertEqual([t["id"] for t in second["tasks"]], [t0], "only the failed task goes again")
            rerun_file = (run.folder / "tasks-rerun.jsonl").read_bytes()
            self.assertEqual(rerun_file, hs.tasks_jsonl(self.tasks[:1]))
            self.assertTrue(rerun_file.endswith(b"\n") and b"\r" not in rerun_file, "LF, like tasks.jsonl")
            self.assertEqual(second["flags"]["--root"], MODEL_A)
            self.assertEqual(len(self.batch_calls()[before:]), 2, "once")
            meta, rows, _, rerun = hs.load_slot(sdir, self.tasks)
            self.assertEqual(meta["reruns"], [{"task_id": t0, "kind": "server", "message": "stub server failure"}])
            self.assertEqual(meta["setup_failures"], [])
            by = {r["task_id"]: r for r in rows}
            self.assertEqual({r["status"] for r in rows}, {hs.GRADED})
            self.assertTrue(by[t0]["reran"])
            self.assertEqual((by[t0]["first_kind"], by[t0]["first_message"]), ("server", "stub server failure"))
            self.assertTrue(by[t0]["run_id"].startswith("rr-"), "the rerun's record replaces the first")
            self.assertTrue(by[t1]["run_id"].startswith("r-") and not by[t1]["reran"])
            self.assertIsNotNone(rerun)
            self.run_json(run.out)
            hs.write_report(run.out, self.tasks, None)
            report = read(run.out / "report.md")
            self.assertIn(f"| a | `{t0}` | server | stub server failure | graded |", report)
            self.assertEqual(cells(report, "Accuracy (exact / graded)")[0], "5 / 5")
            self.assertEqual(cells(report, "Runs graded / infra-failed / not run")[0], "5 / 0 / 0")
            # five final runs of $0.25; the failed first attempt cost $0.25 too, so every call adds up to $1.50
            self.assertEqual(cells(report, "Cost of the final runs")[0], "$1.2500")
            self.assertEqual(cells(report, "Spend, every call (reruns included)")[0], "$1.5000")
            grades = {g["task_id"]: g for g in jsonl(run.out / "grades.jsonl") if g["slot"] == "a"}
            self.assertTrue(grades[t0]["reran"] and grades[t0]["status"] == hs.GRADED)
            self.assertTrue(grades[t0]["run_id"].startswith("rr-"))
            self.assertFalse(grades[t1]["reran"])

        with self.subTest("an auth failure is not rerun, and the later slot does not start"):
            before = len(self.batch_calls())
            run = self.batch("--fail", f"auth:{t1}", "--fail", f"server:{t2}", out=self.results / "auth")
            calls = self.batch_calls()[before:]
            self.assertEqual([c["flags"]["--root"] for c in calls], [MODEL_A, MODEL_A],
                             "slot a and its rerun; no slot b")
            self.assertEqual([t["id"] for t in calls[1]["tasks"]], [t2], "the auth failure is not in the rerun file")
            self.assertEqual(run.stop, "not run: setup failure in slot a")
            self.assertIn(f"{t1} (auth)", run.printed)
            meta, rows, _, _ = hs.load_slot(slot_a().dir(run.out), self.tasks)
            self.assertEqual([(s["task_id"], s["kind"]) for s in meta["setup_failures"]], [(t1, "auth")])
            self.assertEqual([r["task_id"] for r in meta["reruns"]], [t2])
            by = {r["task_id"]: r for r in rows}
            self.assertEqual((by[t1]["status"], by[t1]["infra_kind"], by[t1]["reran"]), (hs.INFRA, "auth", False))
            self.assertEqual(by[t2]["status"], hs.GRADED)
            meta_b, rows_b, _, _ = hs.load_slot(slot_b().dir(run.out), self.tasks)
            self.assertEqual(meta_b["not_run"], "not run: setup failure in slot a")
            self.assertEqual({r["status"] for r in rows_b}, {"not run: setup failure in slot a"})
            self.assertFalse((slot_b().dir(run.out) / "events.jsonl").exists())
            self.run_json(run.out, run.stop)
            hs.write_report(run.out, self.tasks, None)
            report = read(run.out / "report.md")
            self.assertIn("**Stopped: not run: setup failure in slot a.**", report)
            self.assertIn("## Setup failures (not rerun; for Rishi)", report)
            self.assertIn(f"- slot a `{t1}`: auth", report)
            self.assertIn("auth: stub auth failure", report)

        with self.subTest("a second failure is listed and excluded from accuracy"):
            before = len(self.batch_calls())
            run = self.batch("--fail-both", f"server:{t3}", slots=[slot_a()], out=self.results / "twice")
            self.assertIsNone(run.stop)
            self.assertEqual(len(self.batch_calls()[before:]), 2, "no third attempt")
            meta, rows, _, _ = hs.load_slot(slot_a().dir(run.out), self.tasks)
            by = {r["task_id"]: r for r in rows}
            self.assertEqual((by[t3]["status"], by[t3]["infra_kind"], by[t3]["reran"]), (hs.INFRA, "server", True))
            self.assertEqual(by[t3]["first_kind"], "server")
            self.assertEqual((by[t3]["exact"], by[t3]["reward"]), (None, None))
            self.run_json(run.out)
            hs.write_report(run.out, self.tasks, self.summary)
            report = read(run.out / "report.md")
            self.assertEqual(cells(report, "Accuracy (exact / graded)")[0], "4 / 4")
            self.assertEqual(cells(report, "Mean reward (partial credit)")[0], "1.000", "graded runs only")
            self.assertEqual(cells(report, "Mean turns")[0], "3.0")
            self.assertEqual(cells(report, "Runs graded / infra-failed / not run")[0], "4 / 1 / 0")
            self.assertIn(f"| a | `{t3}` | server | stub server failure | failed again: server |", report)
            self.assertIn(f"- slot a `{t3}`: server: stub server failure", report)
            self.assertIn("server: 2", cells(report, "Failed model calls (kind: n)")[0])
            summary = read(self.summary)
            self.assertIn(f"| a | `{t3}` | server | failed again: server |", summary)
            self.assertNotIn("stub server failure", summary, "the summary holds no message text")
            grade = [g for g in jsonl(run.out / "grades.jsonl") if g["task_id"] == t3][0]
            self.assertEqual((grade["status"], grade["infra_kind"], grade["exact"], grade["reward"], grade["reran"]),
                             (hs.INFRA, "server", None, None, True))

        with self.subTest("a missing run.finished is batch_crashed, and rerun"):
            before = len(self.batch_calls())
            run = self.batch("--fail", f"crash:{t4}", slots=[slot_a()], out=self.results / "crash")
            first, second = self.batch_calls()[before:]
            self.assertEqual([t["id"] for t in second["tasks"]], [t4])
            meta, rows, first_stream, _ = hs.load_slot(slot_a().dir(run.out), self.tasks)
            self.assertIn(t4, first_stream.started)
            self.assertNotIn(t4, first_stream.finished)
            self.assertEqual(meta["reruns"], [{"task_id": t4, "kind": "batch_crashed",
                                               "message": "no run.finished: the batch ended without one"}])
            by = {r["task_id"]: r for r in rows}
            self.assertEqual((by[t4]["status"], by[t4]["reran"], by[t4]["first_kind"]),
                             (hs.GRADED, True, "batch_crashed"))
            # the batch itself died after three tasks: the two it never started are rerun together
            before = len(self.batch_calls())
            run = self.batch("--die-after", "3", slots=[slot_a()], out=self.results / "died")
            first, second = self.batch_calls()[before:]
            self.assertEqual([t["id"] for t in second["tasks"]], [t3, t4])
            meta, rows, first_stream, _ = hs.load_slot(slot_a().dir(run.out), self.tasks)
            self.assertIsNone(first_stream.batch, "the dead batch printed no batch.finished")
            self.assertEqual({r["status"] for r in rows}, {hs.GRADED})
            self.assertEqual([r["reran"] for r in rows], [False, False, False, True, True])
            # and a second crash stays batch_crashed
            run = self.batch("--fail-both", f"crash:{t4}", slots=[slot_a()], out=self.results / "crash twice")
            _, rows, _, _ = hs.load_slot(slot_a().dir(run.out), self.tasks)
            self.assertEqual((rows[4]["status"], rows[4]["infra_kind"], rows[4]["reran"]),
                             (hs.INFRA, "batch_crashed", True))

    def test_usage_error_stops_with_the_stderr_tail(self):
        self.stub_config(exit=2)
        with self.assertRaises(hs.SetupError) as caught:
            self.batch(slots=[slot_a()], out=self.results / "usage")
        self.assertIn("exited 2", str(caught.exception))
        self.assertIn("usage error (the stub was told to exit 2)", str(caught.exception))
        self.assertEqual(len(self.batch_calls()), 1, "a usage error is not rerun")
        meta = json.loads(read(slot_a().dir(self.results / "usage") / "slot.json"))
        self.assertEqual(meta["exit_code"], 2)
        # and through main(): exit 2 with the tail on stderr
        code, _, err = self.go()
        self.assertEqual(code, 2)
        self.assertIn("usage error (the stub was told to exit 2)", err)
        self.assertIn("slot a", err)

    def test_rerun_hits_the_cap_too(self):
        """The rerun's spend counts toward the cap, and a rerun stopped by it leaves its task not run."""
        self.stub_config(cost=0.5, calls=2)  # 1.0 a task: five tasks spend 5.0, the rerun's first call reaches 5.5
        t0 = self.tasks[0].task_id
        run = self.batch("--fail", f"server:{t0}", cap=5.5, slots=[slot_a(), slot_b()], out=self.results / "rcap")
        self.assertEqual(run.stop, hs.CAP)
        self.assertEqual(len(self.batch_calls()), 2, "slot b never starts")
        meta, rows, _, rerun = hs.load_slot(slot_a().dir(run.out), self.tasks)
        self.assertEqual((meta["capped"], meta["rerun_capped"]), (False, True))
        self.assertEqual(rows[0]["status"], hs.CAP)
        self.assertTrue(rows[0]["reran"])
        self.assertEqual({r["status"] for r in rows[1:]}, {hs.GRADED})
        self.assertEqual(json.loads(read(slot_b().dir(run.out) / "slot.json"))["not_run"], hs.CAP)


# ----------------------------------------------------------------------------- the cost cap


class CapTests(SmokeCase):
    def tmp_args(self, slot=None):
        folder = self.tmp / "tasks folder"
        tasks_file = hs.write_task_files(self.tasks, folder)
        return hs.batch_command(self.cmd(), tasks_file, slot or slot_a(), self.results / "cap", 2)

    def test_cost_cap(self):
        # 0.5 a gateway.call, three calls a task, a slow stub with an engine stand-in on its stdin pipe
        self.stub_config(child=True, slow=0.1, cost=0.5, calls=3)
        started = time.monotonic()
        run = self.batch(cap=2.0)
        self.assertLess(time.monotonic() - started, 15, "terminate(), not a wait for the stub to finish")
        self.assertEqual(run.stop, "not run: cost cap")
        self.assertEqual(len(self.batch_calls()), 1, "no rerun, and slot b never started")
        stub_dir = self.stub.parent
        self.assertFalse((stub_dir / "stub.finished").exists(), "the stub was terminated before it ended")
        # no process is left: the engine stand-in saw its stdin close and exited
        self.assertTrue(wait_for(stub_dir / "child.eof"), "the engine stand-in is still running")
        self.assertTrue(pid_gone(int(read(stub_dir / "child.pid"))))

        ids = [t.task_id for t in self.tasks]
        sdir_a, sdir_b = slot_a().dir(run.out), slot_b().dir(run.out)
        meta, rows, first, rerun = hs.load_slot(sdir_a, self.tasks)
        self.assertTrue(meta["capped"])
        self.assertEqual(meta["reruns"], [], "a capped slot is not rerun")
        self.assertEqual(rows[0]["status"], hs.GRADED, "the run that finished before the cap")
        self.assertEqual([r["status"] for r in rows[1:]], ["not run: cost cap"] * 4,
                         "the unfinished runs and the unstarted")
        self.assertIn(ids[1], first.started)
        self.assertNotIn(ids[1], first.finished, "started, then cut off")
        self.assertNotIn(ids[2], first.started)
        meta_b, rows_b, _, _ = hs.load_slot(sdir_b, self.tasks)
        self.assertEqual(meta_b["not_run"], "not run: cost cap")
        self.assertEqual([r["status"] for r in rows_b], ["not run: cost cap"] * 5)
        self.assertTrue((sdir_b / "slot.json").is_file())
        self.assertFalse((sdir_b / "events.jsonl").exists())

        self.run_json(run.out, run.stop)
        rows, _ = hs.write_report(run.out, self.tasks, self.summary)
        report = read(run.out / "report.md")
        title, blank, stopped = report.split("\n")[:3]
        self.assertEqual(stopped, STOPPED_CAP, "said first")
        self.assertEqual(cells(report, "Runs graded / infra-failed / not run"), ["1 / 0 / 4", "0 / 0 / 5"])
        self.assertEqual(cells(report, "Accuracy (exact / graded)"), ["1 / 1", "0 / 0"])
        self.assertIn("not run: cost cap", read(self.summary))
        self.assertEqual(len(jsonl(run.out / "grades.jsonl")), 10)

    def test_cost_cap_boundary(self):
        """The cap is passed when spent + the batch's costs is above it, not at it. A stub that ignores terminate() is
        killed."""
        self.stub_config(slow=0.1, cost=0.5, calls=3)
        args = self.tmp_args()
        out = self.results / "cap"
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            attempt = hs.run_slot(args, out, cap=2.0, label="a")
        self.assertTrue(attempt.capped)
        self.assertEqual(attempt.stream.cost, 2.5, "cut at the first call above 2.0, not at 2.0")
        self.assertEqual(len(attempt.stream.finished), 1)
        self.assertNotEqual(attempt.exit_code, 0)
        self.assertIn("cost cap $2.0000 passed at $2.5000: stopping", printed.getvalue())
        lines = jsonl(out / "events.jsonl")
        self.assertEqual(len(lines), 8, "the line that crossed the cap is kept")
        self.assertEqual(lines[-1]["type"], "gateway.call")
        self.assertFalse((self.stub.parent / "stub.finished").exists())
        # exactly at the cap is not over it
        self.stub_config(slow=0)
        out2 = self.results / "cap exact"
        with contextlib.redirect_stdout(io.StringIO()):
            attempt = hs.run_slot(args, out2, cap=7.5, label="a")  # five tasks of 1.5 spend exactly 7.5
        self.assertFalse(attempt.capped)
        self.assertEqual(attempt.stream.cost, 7.5)
        self.assertEqual((attempt.exit_code, len(attempt.stream.finished)), (0, 5))
        # what earlier slots spent counts: 7.5 spent, 0.5 more is over a cap of 7.9
        out3 = self.results / "cap spent"
        with contextlib.redirect_stdout(io.StringIO()):
            attempt = hs.run_slot(args, out3, spent=7.5, cap=7.9, label="b")
        self.assertTrue(attempt.capped)
        self.assertEqual(attempt.stream.cost, 0.5, "7.5 + 0.5 is over 7.9")
        # no cap, no stop
        with contextlib.redirect_stdout(io.StringIO()):
            attempt = hs.run_slot(args, self.results / "cap none", spent=1e9, label="a")
        self.assertFalse(attempt.capped)

    def test_cost_cap_counts_every_slot(self):
        """Each slot spends 1.25: slot a ends under a cap of 2.0, and slot b alone would too, but not both together."""
        self.stub_config(cost=0.125, calls=2)
        run = self.batch(cap=2.0)
        self.assertEqual(run.stop, hs.CAP)
        meta_a = json.loads(read(slot_a().dir(run.out) / "slot.json"))
        meta_b = json.loads(read(slot_b().dir(run.out) / "slot.json"))
        self.assertEqual((meta_a["capped"], meta_a["not_run"]), (False, None))
        self.assertEqual((meta_b["capped"], meta_b["not_run"]), (True, None))
        _, rows_a, _, _ = hs.load_slot(slot_a().dir(run.out), self.tasks)
        _, rows_b, _, _ = hs.load_slot(slot_b().dir(run.out), self.tasks)
        self.assertEqual({r["status"] for r in rows_a}, {hs.GRADED})
        # 1.25 + 7 calls of 0.125 is 2.125, over 2.0: three of slot b's tasks finished, the fourth was cut off
        self.assertEqual([r["status"] for r in rows_b], [hs.GRADED] * 3 + [hs.CAP] * 2)

    def test_cost_cap_through_main(self):
        self.stub_config(slow=0.05, cost=0.5, calls=3)
        code, out, err = self.go("--cost-cap", "2")
        self.assertEqual(code, 1, err)
        self.assertIn("cost cap $2.0000 passed at $2.5000: stopping", out)
        self.assertIn("stopped: not run: cost cap", out)
        self.assertEqual(len(self.batch_calls()), 1)
        report = read(self.run_dir() / "report.md")
        self.assertEqual(report.split("\n")[2], STOPPED_CAP)
        self.assertEqual(json.loads(read(self.run_dir() / "run.json"))["stopped"], "not run: cost cap")
        self.assertIn("slot a: 1/1 exact, 0 infra-failed, 4 not run", out)
        self.assertIn("slot b: 0/0 exact, 0 infra-failed, 5 not run", out)

    @unittest.skipIf(os.name == "nt", "TerminateProcess can't be ignored")
    def test_terminate_ignored_is_killed(self):
        self.stub_config(ignore_term=True, slow=0.1, cost=0.5, calls=3)
        with mock.patch.object(hs, "STOP_WAIT_S", 0.5):
            started = time.monotonic()
            with contextlib.redirect_stdout(io.StringIO()):
                attempt = hs.run_slot(self.tmp_args(), self.results / "ignore", cap=2.0, label="a")
        self.assertTrue(attempt.capped)
        self.assertEqual(attempt.exit_code, -9)
        self.assertLess(time.monotonic() - started, 15)
        self.assertFalse((self.stub.parent / "stub.finished").exists())


# ----------------------------------------------------------------------------- the report and its guard


STOPPED_CAP = "**Stopped: not run: cost cap.** Later runs were not run; see the tables."
SENTINEL_QUESTION = "SENTINEL-QUESTION-3c9d2e71b8 which carrier had the most delays?"
SENTINEL_GOLD = "SENTINEL-GOLD-ANSWER-5e1f0c3b9a"
SENTINEL_ANSWER = "SENTINEL-MODEL-ANSWER-b84e10d5c2"


class ReportTests(SmokeCase):
    def test_report_has_no_task_text(self):
        self.tasks = make_tasks(question=SENTINEL_QUESTION, answer=SENTINEL_GOLD, grading={"type": "entity"})
        self.stub_config(answers={t.task_id: SENTINEL_ANSWER for t in self.tasks}, denied={self.tasks[0].task_id: 1})
        code, out, err = self.go()
        self.assertEqual(code, 0, err)
        grades = jsonl(self.run_dir() / "grades.jsonl")
        self.assertEqual(len(grades), 10)
        for g in grades:
            self.assertEqual((g["answer"], g["gold"]), (SENTINEL_ANSWER, SENTINEL_GOLD))
            self.assertEqual((g["exact"], g["reward"]), (False, 0.0))
        report, summary = read(self.run_dir() / "report.md"), read(self.summary)
        for text in (report, summary, out, err):
            for sentinel in (SENTINEL_QUESTION, SENTINEL_GOLD, SENTINEL_ANSWER, "SENTINEL"):
                self.assertNotIn(sentinel, text)
        # the question is in the tasks file the stub read and the answer is in the raw event log: both stay local
        self.assertEqual(self.batch_calls()[0]["tasks"][0]["question"], SENTINEL_QUESTION)
        self.assertIn(SENTINEL_ANSWER, read(self.slot_dir("a") / "events.jsonl"))
        self.assertIn("task ids and levels", summary)
        for t in self.tasks:
            self.assertIn(t.task_id, summary)
        self.assertNotIn("\r", report + summary + read(self.run_dir() / "grades.jsonl"), "LF on every OS")

    def test_guard_refuses_task_text(self):
        """A report that would hold a question, a gold or an answer (here, an error message that echoes the prompt) is
        refused whole: nothing is written, and the exit is 2."""
        self.tasks = make_tasks(question=SENTINEL_QUESTION, answer=SENTINEL_GOLD, grading={"type": "entity"})
        t0 = self.tasks[0].task_id
        self.stub_config(fail_both=[f"other:{t0}"],
                         messages={t0: f"bad request, the provider echoed: {SENTINEL_QUESTION}"})
        code, out, err = self.go()
        self.assertEqual(code, 2)
        self.assertIn("report guard", err)
        self.assertNotIn("SENTINEL", err + out)
        self.assertFalse((self.run_dir() / "report.md").exists())
        self.assertFalse(self.summary.exists())
        self.assertTrue((self.run_dir() / "grades.jsonl").is_file(), "grades.jsonl is where answers and gold belong")
        # the same guard, on its own
        built = make_tasks(question=SENTINEL_QUESTION, answer=[SENTINEL_GOLD, "second-gold-item"],
                           grading={"type": "list"})
        rows = [{"finished": {"answer": SENTINEL_ANSWER}}, {"finished": {"answer": ["list-answer-item-1", "x"]}},
                {"finished": {"answer": 12345678901234}}, {"finished": None}, {"finished": {"answer": None}}]
        path = Path("report.md")
        hs.guard({path: "numbers only: 12, 3.5, [] and task ids"}, built, rows)
        for needle in (SENTINEL_QUESTION, json.dumps([SENTINEL_GOLD, "second-gold-item"]), SENTINEL_ANSWER,
                       json.dumps(["list-answer-item-1", "x"]), "12345678901234"):
            with self.subTest(needle=needle), self.assertRaises(hs.SetupError) as caught:
                hs.guard({path: f"prefix {needle} suffix"}, built, rows)
            self.assertIn("report.md", str(caught.exception))
            self.assertNotIn(needle, str(caught.exception))
        with self.assertRaises(hs.SetupError):
            hs.guard({Path("ok.md"): "fine", Path("summary.md"): f"x {SENTINEL_ANSWER}"}, built, rows)
        # short answers are numbers a table may hold anyway
        short = [Task("s", FAMILY, "How many?", "c", answer=12, grading={"type": "int", "blind": 1})]
        hs.guard({path: "12 and How many and 5"}, short, [{"finished": {"answer": "12"}}])

    def test_report_numbers(self):
        """The report's numbers from the stub's canned events, slot a and slot b apart, by hand."""
        t = [x.task_id for x in self.tasks]
        wrong = reward.score_answer("5", 0, self.tasks[2].grading)
        self.stub_config(
            denied={t[0]: 2, t[1]: 1}, allowed={t[0]: 3},
            by_ref={MODEL_B: {"answers": {t[2]: "5"}, "turns": 5, "subcalls": 7, "wall": 2.5, "cost": 0.25,
                              "denied": {},
                              "usage": {"root": {"input_tokens": 3000, "output_tokens": 300, "cache_read_tokens": 0,
                                                 "cache_write_tokens": 0},
                                        "sub": {"input_tokens": 1000, "output_tokens": 100, "cache_read_tokens": 0,
                                                "cache_write_tokens": 0}}}})
        code, out, err = self.go()
        self.assertEqual(code, 0, err)
        report = read(self.run_dir() / "report.md")
        row = lambda label: cells(report, label)  # noqa: E731
        self.assertEqual(row("Model"), [f"`{MODEL_A}`", f"`{MODEL_B}`"])
        self.assertEqual(row("Runs graded / infra-failed / not run"), ["5 / 0 / 0", "5 / 0 / 0"])
        self.assertEqual(row("Accuracy (exact / graded)"), ["5 / 5", "4 / 5"])
        self.assertEqual(row("Mean reward (partial credit)"), ["1.000", f"{(4 * 1.0 + wrong.reward) / 5:.3f}"])
        self.assertEqual(row("Mean turns"), ["3.0", "5.0"])
        self.assertEqual(row("Mean sub-calls"), ["4.0", "7.0"])
        self.assertEqual(row("Cost of the final runs"), ["$1.2500", "$2.5000"])
        self.assertEqual(row("Cost per final run"), ["$0.2500", "$0.5000"])
        self.assertEqual(row("Spend, every call (reruns included)"), ["$1.2500", "$2.5000"])
        # cache reads over input tokens (D-22): slot a 2000 of 5000 root, 0 of 10000 sub, 2000 of 15000 all
        self.assertEqual(row("Cache hit rate, root"), ["40.0%", "none reported"])
        self.assertEqual(row("Cache hit rate, sub-calls"), ["0.0%", "none reported"])
        self.assertEqual(row("Cache hit rate, overall"), ["13.3%", "none reported"])
        self.assertEqual(row("Mean wall time (s)"), ["1.5", "2.5"])
        self.assertEqual(row("Denied actions"), ["3", "0"])
        self.assertEqual(row("Failed model calls (kind: n)"), ["none", "none"])
        # slots: the resolved profile and call format, what run.started showed, and why
        slot_rows = [ln for ln in report.split("\n")
                     if ln.startswith("| a | `deepseek") or ln.startswith("| b | `openrouter")]
        self.assertEqual(slot_rows, [
            f"| a | `{MODEL_A}` | `default` | native (auto) | native | the product's own profile and path (auto) |",
            f"| b | `{MODEL_B}` | `default` | fenced (--call-format fenced) | fenced | forced fenced: the LoRA's "
            "fenced format meets a real model (S-32) |"])
        self.assertIn("Slot b is forced to fenced text (S-32).", report)
        # the estimate against the actual cost
        self.assertIn("| a | $2.1000 | $3.1500 | $23.0165 | $1.2500 |", report)
        self.assertIn("| b | $1.0500 | $1.5750 | $11.5083 | $2.5000 |", report)
        # no profile.toml under the test's --harness: training's limits, and the report says why (rlm-harness D-152)
        self.assertIn("Ceiling per run: 2,964,907 tokens in, 409,600 out (`READ_LIMITS_V4`: no `default` profile.toml "
                      "under --harness or the harness home). Cost cap $9.4500.", report)
        self.assertIn("Reference runs: 2", report)
        self.assertIn("RLM commit `0123456789abcdef`", report)
        self.assertIn("`rlmh 0.1.0`", report)
        self.assertIn("`call_format = auto`", report)
        # per-run rows: ids, levels, status, exact, reward
        line = [ln for ln in report.split("\n") if ln.startswith(f"| b | `{t[2]}`")][0]
        self.assertEqual([c.strip() for c in line.strip("|").split("|")][:8],
                         ["b", f"`{t[2]}`", "2", "graded", "no", f"{wrong.reward:.3f}", "5", "7"])
        self.assertIn("## Reruns\n\nNone.", report)
        self.assertIn("## Failure notes\n\nNone.", report)
        summary = read(self.summary)
        self.assertTrue(summary.startswith("# Harness smoke run (rlm-harness v0, step 9), 2026-01-02"))
        self.assertNotIn("## Failure notes", summary)
        self.assertEqual(cells(summary, "Accuracy (exact / graded)"), ["5 / 5", "4 / 5"])
        self.assertIn("Set `harness-smoke-v1`: 5 tasks of `aggregation_heldout`, levels 1, 1, 2, 2, 3, generator "
                      "`aggregation-v4`", report)
        grades = jsonl(self.run_dir() / "grades.jsonl")
        self.assertEqual([(g["slot"], g["task_id"]) for g in grades], [(s, x) for s in "ab" for x in t])
        self.assertEqual({k for k in grades[0]}, {"slot", "model", "task_id", "level", "run_id", "status", "infra_kind",
                                                  "submitted", "answer_valid", "answer", "gold", "exact", "reward",
                                                  "components", "reran"})
        self.assertEqual([g["level"] for g in grades[:5]], [1, 1, 2, 2, 3])
        self.assertEqual(grades[7]["answer"], "5")
        self.assertEqual(grades[7]["gold"], 0)
        self.assertIn("slot a: 5/5 exact, 0 infra-failed, 0 not run", out)
        self.assertIn("slot b: 4/5 exact, 0 infra-failed, 0 not run", out)
        # each slot folder holds the run log, the raw events, stderr and slot.json
        for name in "ab":
            names = {p.name for p in self.slot_dir(name).iterdir()}
            self.assertEqual(names, {"runs.jsonl", "events.jsonl", "stderr.log", "slot.json"})

    def test_report_marks_a_call_format_mismatch(self):
        self.stub_config(format_lie="fenced")  # slot a was to be native
        code, out, err = self.go()
        self.assertEqual(code, 0, err)
        report = read(self.run_dir() / "report.md")
        self.assertRegex(report, r"\| a \| `deepseek/FAKE-chat` \| `default` \| native \(auto\) \| "
                                 r"fenced \*\*mismatch\*\* \|")
        self.assertNotRegex(report, r"\| b \|[^\n]*\*\*mismatch\*\*")
        self.assertIn(f"slot a {self.tasks[0].task_id}: call_format fenced, expected native (Task 4's table)", report)

    def test_report_rebuilds_from_the_saved_files(self):
        code, _, err = self.go()
        self.assertEqual(code, 0, err)
        names = ("report.md", "grades.jsonl")
        saved = {n: (self.run_dir() / n).read_bytes() for n in names}
        summary = self.summary.read_bytes()
        for n in names:
            (self.run_dir() / n).unlink()
        self.summary.unlink()
        code, out, err = self.cli("--report", models=None)
        self.assertEqual(code, 0, err)
        self.assertIn(str(self.run_dir() / "report.md"), out)
        self.assertEqual({n: (self.run_dir() / n).read_bytes() for n in names}, saved)
        self.assertEqual(self.summary.read_bytes(), summary)
        self.assertEqual(len(self.batch_calls()), 2, "--report starts no batch")
        # the events on disk are the source: drop one run.finished and that run is a batch_crashed
        events = self.slot_dir("a") / "events.jsonl"
        victim = self.tasks[1].task_id
        kept = [ln for ln in read(events).split("\n") if '"type": "run.finished"' not in ln or victim not in ln]
        events.write_bytes("\n".join(kept).encode("utf-8"))
        code, _, err = self.cli("--report", models=None)
        self.assertEqual(code, 0, err)
        report = read(self.run_dir() / "report.md")
        self.assertEqual(cells(report, "Runs graded / infra-failed / not run")[0], "4 / 1 / 0")
        self.assertIn(f"- slot a `{victim}`: batch_crashed: no run.finished", report)
        g = [x for x in jsonl(self.run_dir() / "grades.jsonl") if x["task_id"] == victim and x["slot"] == "a"][0]
        self.assertEqual((g["status"], g["infra_kind"]), (hs.INFRA, "batch_crashed"))
        # without a run there is nothing to rebuild
        code, _, err = self.main("--report", "--date", "2026-02-02")
        self.assertEqual(code, 2)
        self.assertIn("run.json is missing", err)


# ----------------------------------------------------------------------------- the modes through main()


class ModeTests(SmokeCase):
    def test_refuses_existing_output(self):
        code, _, err = self.cli("--estimate")
        self.assertEqual(code, 0, err)
        runs = self.slot_dir("a") / "runs.jsonl"
        runs.parent.mkdir(parents=True)
        runs.write_bytes(b"KEEP\n")
        code, out, err = self.cli("--go")
        self.assertEqual(code, 2)
        self.assertIn("output already exists", err)
        self.assertIn(str(runs.parent), err)
        self.assertEqual(runs.read_bytes(), b"KEEP\n", "an earlier run is never overwritten")
        self.assertEqual(self.batch_calls(), [])
        self.assertFalse((self.run_dir() / "run.json").exists())
        self.assertEqual(sorted(p.name for p in self.run_dir().iterdir()), [runs.parent.name])
        # slot b's folder is checked too, and --only leaves the other slot's alone
        runs.unlink()
        other = self.slot_dir("b") / "runs.jsonl"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"KEEP B\n")
        self.assertEqual(self.cli("--go")[0], 2)
        self.assertEqual(self.batch_calls(), [])
        code, _, err = self.cli("--go", "--only", "a")
        self.assertEqual(code, 0, err)
        self.assertEqual(other.read_bytes(), b"KEEP B\n")
        # a finished run is refused the same way on a second --go
        self.assertEqual(self.cli("--go", "--only", "a")[0], 2)
        # only a rehearsal clears its folder
        self.assertEqual(self.cli("--rehearse")[0], 0)

    def test_refuses_leftover_events(self):
        """A crashed attempt leaves events.jsonl without runs.jsonl: --go must not append a second attempt to it, but
        an empty one (rlmh exited 2 before printing anything) doesn't block a retry."""
        self.assertEqual(self.cli("--estimate")[0], 0)
        events = self.slot_dir("a") / "events.jsonl"
        events.parent.mkdir(parents=True)
        events.write_bytes(b'{"type": "run.started"}\n')
        code, _, err = self.cli("--go")
        self.assertEqual(code, 2)
        self.assertIn("output already exists", err)
        self.assertEqual(self.batch_calls(), [])
        events.write_bytes(b"")
        code, _, err = self.cli("--go")
        self.assertEqual(code, 0, err)

    def test_go_needs_matching_task_files(self):
        code, _, err = self.cli("--go")
        self.assertEqual(code, 2, "no --estimate first, so no task files")
        self.assertIn("tasks.jsonl", err)
        self.assertEqual(self.batch_calls(), [])
        code, _, err = self.cli("--estimate")
        self.assertEqual(code, 0, err)
        victim = self.tasks[3]
        ctx = self.scratch / DATE / "ctx" / f"{victim.task_id}.txt"
        ctx.write_bytes(ctx.read_bytes().replace(b"x", b"y", 1))
        code, _, err = self.cli("--go")
        self.assertEqual((code, self.batch_calls()), (2, []))
        self.assertIn(f"ctx/{victim.task_id}.txt", err)
        self.assertIn("run --estimate again and get a new go", err)
        self.assertFalse(self.results.exists())
        # the rebuilt tasks changed instead (a different seed): the same refusal
        self.assertEqual(self.cli("--estimate")[0], 0)
        self.tasks = make_tasks()
        self.tasks[0].context += "one more line\n"
        code, _, err = self.cli("--go")
        self.assertEqual(code, 2)
        self.assertIn(f"ctx/{self.tasks[0].task_id}.txt", err)
        self.assertEqual(self.batch_calls(), [])

    def test_windows_paths(self):
        self.assertIn(" ", str(self.stub))
        self.assertIn(" ", str(self.scratch))
        self.assertIn(" ", str(self.results))
        question = ("Qu\u00e9 transportista tuvo m\u00e1s retrasos en d\u00edas lluviosos \u2014 na\u00efve caf\u00e9 "
                    "\u201cquoted\u201d \u00fc\u00f1 \u65e5\u672c?")
        self.tasks = make_tasks(question=question)
        self.tasks[0].context = "R\u00e9sum\u00e9 \u00e0 la carte\nligne deux \u2014 \u65e5\u672c\nfin\n"
        with self.real_info():  # `--version` and `config get` run through a stub path with spaces too
            self.assertEqual(hs.rlmh_info(self.cmd())["version"], "rlmh 0.1.0")
        code, out, err = self.go()
        self.assertEqual(code, 0, err)
        self.assertNotIn("check:", out, "every context reached the stub as bytes: chars and sha256 match")
        folder = self.scratch / DATE
        for call in self.batch_calls():
            self.assertIsInstance(call["argv"], list)
            self.assertTrue(all(isinstance(a, str) for a in call["argv"]))
            self.assertEqual(call["rest"][0], "batch")
            self.assertEqual(call["rest"][1], str(folder / "tasks.jsonl"), "one argument, spaces and all")
            self.assertEqual(call["tasks_file"], str(folder / "tasks.jsonl"))
            self.assertEqual([t["question"] for t in call["tasks"]], [question] * 5, "accents survive the file")
            self.assertTrue(call["flags"]["--out"].startswith(str(self.run_dir())))
            self.assertIn(" ", call["flags"]["--out"])
            for t in call["tasks"]:
                self.assertEqual(t["input"], f"ctx/{t['id']}.txt", "forward slashes in the file")
        self.assertEqual({tuple(c["argv"][:1]) for c in self.calls()}, {("--version",), ("config",), ("batch",)})
        self.assertEqual(self.calls()[0]["argv"], ["--version"])
        for t in self.tasks:
            self.assertEqual((folder / "ctx" / f"{t.task_id}.txt").read_bytes(), t.context.encode("utf-8"))
        started = jsonl(self.slot_dir("a") / "events.jsonl")[0]
        self.assertEqual(started["input"]["chars"], len(self.tasks[0].context))
        # the rerun file and its path with spaces work the same way
        self.stub_config(fail=[f"server:{self.tasks[0].task_id}"])
        before = len(self.batch_calls())
        code, out, err = self.go(date="2026-01-06")
        self.assertEqual(code, 0, err)
        rerun = [c for c in self.batch_calls()[before:] if c["rerun"]]
        self.assertEqual(len(rerun), 2)
        for call in rerun:
            self.assertEqual(call["tasks_file"], str(self.scratch / "2026-01-06" / "tasks-rerun.jsonl"))
            self.assertEqual(call["tasks"][0]["question"], question)

    def test_no_key_output(self):
        with mock.patch.dict(os.environ, {"FAKE_API_KEY": KEY, "OPENAI_API_KEY": KEY + "-b"}):
            seen = []
            for mode in ("--estimate", "--rehearse", "--go", "--report"):
                with self.subTest(mode):
                    if mode == "--go":
                        code, out, err = self.go()
                    elif mode == "--estimate":
                        with self.real_info():  # rlmh --version and config get get the environment too
                            code, out, err = self.cli(mode)
                    else:
                        code, out, err = self.cli(mode, models=None if mode == "--report" else (MODEL_A, MODEL_B))
                    if mode == "--report":
                        self.assertEqual(code, 0, err)
                    self.assertIn(code, (0, 1))
                    seen += [out, err]
            batch_calls = self.batch_calls()
            self.assertEqual(len(batch_calls), 4, "a rehearsal and a go, two slots each")
            self.assertTrue(all(c["has_fake_key"] for c in batch_calls), "rlmh gets the environment unchanged")
            self.assertTrue(all(c["has_fake_key"] for c in self.calls()))
        files = [p for p in self.tmp.rglob("*") if p.is_file()]
        self.assertGreater(len(files), 20)
        for text in seen:
            self.assertNotIn(KEY, text)
        for path in files:
            self.assertNotIn(KEY.encode(), path.read_bytes(), str(path))
            self.assertNotIn(KEY.encode(), str(path).encode())

    def test_rehearse(self):
        stale = self.run_dir(f"{DATE}-rehearsal") / "stale.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("from an earlier rehearsal", encoding="utf-8")
        code, out, err = self.cli("--rehearse")
        self.assertEqual(code, 0, err)
        out_dir = self.run_dir(f"{DATE}-rehearsal")
        self.assertFalse(stale.exists(), "a rehearsal starts clean")
        self.assertFalse(self.run_dir().exists(), "and never writes where --go will")
        self.assertNotIn("Smoke run estimate", out)
        self.assertEqual(len(jsonl(out_dir / "grades.jsonl")), 10)
        report = read(out_dir / "report.md")
        self.assertEqual(cells(report, "Spend, every call (reruns included)"), ["$0.0000", "$0.0000"])
        self.assertEqual(cells(report, "Cost of the final runs"), ["$0.0000", "$0.0000"])
        self.assertIn("No estimate was recorded with this run.", report)
        self.assertEqual(cells(report, "Model"), [f"`fake/scripted` (stand-in for `{MODEL_A}`)",
                                                  f"`fake/scripted` (stand-in for `{MODEL_B}`)"])
        self.assertNotIn("mismatch", report)
        self.assertFalse(self.summary.exists(), "a rehearsal writes no summary")
        call_a, call_b = self.batch_calls()
        for call, fmt in ((call_a, None), (call_b, "fenced")):
            self.assertEqual((call["flags"]["--root"], call["flags"]["--sub"]), ("fake/scripted", "fake/scripted"))
            self.assertEqual(call["flags"].get("--call-format"), fmt)
            self.assertEqual(call["fake_script"], str(out_dir / f"fake-{'a' if call is call_a else 'b'}.json"))
        # the scripts: native submits through the tool call, fenced through the text alone
        self.assertEqual(call_a["fake_script_json"], {"root": [{"text": "", "tool_call": {"code": "submit('0')"}}],
                                                      "sub": [{"text": "0"}]})
        self.assertEqual(call_b["fake_script_json"], {"root": [{"text": "```python\nsubmit('0')\n```"}],
                                                      "sub": [{"text": "0"}]})
        self.assertEqual(hs.fake_script("native"), call_a["fake_script_json"])
        self.assertEqual(hs.fake_script("fenced"), call_b["fake_script_json"])
        self.assertEqual(hs.fake_script("native")["root"][0]["tool_call"], {"code": "submit('0')"})
        self.assertNotIn("tool_call", hs.fake_script("fenced")["root"][0])
        self.assertNotIn("```", hs.fake_script("native")["root"][0]["text"])
        self.assertIn("submit('0')", hs.fake_script("fenced")["root"][0]["text"])
        self.assertTrue(hs.fake_script("fenced")["root"][0]["text"].startswith("```python\n"))

    def test_rehearsal_checks(self):
        """A rehearsal exits 1 when a run didn't see its whole context, or ran in another format or profile."""
        for label, cfg, needle in (
                ("chars", {"chars_delta": 1}, "run.started.input is "),
                ("sha256", {"sha_lie": True}, "(sha256 differs)"),
                ("format", {"format_lie": "native"}, "call_format native, expected fenced"),
                ("profile", {"profile_lie": "training-v4"}, "profile training-v4, asked default")):
            with self.subTest(label):
                self.stub_config(**{"chars_delta": 0, "sha_lie": False, "format_lie": None, "profile_lie": None, **cfg})
                code, out, err = self.cli("--rehearse")
                self.assertEqual(code, 1, err)
                self.assertIn("check: ", out)
                self.assertIn(needle, out)
                self.assertIn(needle, read(self.run_dir(f"{DATE}-rehearsal") / "report.md"))
        self.stub_config(chars_delta=0, sha_lie=False, format_lie=None, profile_lie=None, garbage=True)
        code, out, _ = self.cli("--rehearse")
        self.assertEqual(code, 1, "a line that is not an event is a problem too")
        self.assertIn("not a JSON event", out)
        # in a real run a mismatch is reported but doesn't stop it
        self.stub_config(garbage=False, chars_delta=1)
        code, out, err = self.go()
        self.assertEqual(code, 0, err)
        self.assertIn("check: ", out)

    def test_exit_codes(self):
        # 0: every slot ran to its end, even with a failure that is listed
        self.stub_config(fail_both=[f"server:{self.tasks[0].task_id}"])
        code, out, err = self.go()
        self.assertEqual(code, 0, err)
        self.assertIn("slot a: 4/4 exact, 1 infra-failed, 0 not run", out)
        # 1: a slot was stopped by a setup failure
        self.stub_config(fail_both=[], fail=[f"auth:{self.tasks[1].task_id}"])
        code, out, err = self.go(date="2026-01-07")
        self.assertEqual(code, 1, err)
        self.assertIn("stopped: not run: setup failure in slot a", out)
        self.assertIn(f"{self.tasks[1].task_id} (auth)", out)
        self.assertIn("slot b: 0/0 exact, 0 infra-failed, 5 not run", out)

    def test_gitignore_rules(self):
        """check_ignored asks git about the report folder and the seeds folder before anything is written."""
        calls = []

        def fake_git(*args):
            calls.append(args)
            return subprocess.CompletedProcess(args, 1, "", "")

        with mock.patch.object(hs, "git", fake_git):
            with self.assertRaises(hs.SetupError) as caught:
                REAL.check_ignored()
        self.assertIn("results/harness-smoke/2026-01-01/a-x/runs.jsonl", str(caught.exception))
        self.assertEqual(calls, [("check-ignore", "-q", "results/harness-smoke/2026-01-01/a-x/runs.jsonl")])
        # both the results folder and the seeds folder must be ignored (D-46, private/)
        answers = iter([0, 1])
        with mock.patch.object(hs, "git", lambda *a: subprocess.CompletedProcess(a, next(answers), "", "")):
            with self.assertRaises(hs.SetupError) as caught:
                REAL.check_ignored()
        self.assertIn("private/seeds/harness-smoke-v1.json", str(caught.exception))
        with mock.patch.object(hs, "git", lambda *a: subprocess.CompletedProcess(a, 0, "", "")):
            REAL.check_ignored()
        self.patch(hs, "check_ignored", REAL.check_ignored)
        with mock.patch.object(hs, "git", fake_git):
            for argv in (["--estimate"], ["--go"], ["--rehearse"], ["--report"]):
                code, out, err = self.cli(*argv)
                self.assertEqual(code, 2, argv)
                self.assertIn("not git-ignored", err)
        self.assertFalse(self.scratch.exists(), "nothing is written before the rule is there")
        self.assertFalse(self.results.exists())

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs git in the repo")
    def test_repo_gitignore_has_the_rules(self):
        """Done when: `git check-ignore -v results/harness-smoke/2026-01-01/a-x/runs.jsonl` prints the rule."""
        done = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-v", hs.IGNORE_PROBES[0]], capture_output=True,
                              text=True)
        self.assertEqual(done.returncode, 0, "results/harness-*/ is not in .gitignore (D-46)")
        self.assertIn("results/harness-*/", done.stdout)
        REAL.check_ignored()

    def test_rlm_commit_and_progress_lines(self):
        def fake_git(head, dirty, ok=True):
            def run(*args):
                text = head if args[0] == "rev-parse" else dirty
                return subprocess.CompletedProcess(args, 0 if ok else 128, text if ok or args[0] != "rev-parse" else "",
                                                   "")
            return run

        for head, dirty, ok, want in (("abc123\n", "", True, "abc123"),
                                      ("abc123\n", " M scripts/x.py\n", True, "abc123 (uncommitted changes)"),
                                      ("", "", False, "unknown")):
            with self.subTest(want), mock.patch.object(hs, "git", fake_git(head, dirty, ok)):
                self.assertEqual(REAL.rlm_commit(), want)
        base = {"tag": "t1", "turns": 4, "subcalls": 9, "cost_usd": 0.5, "submitted": True}
        lines = []
        for f in (base, {**base, "submitted": False}, {**base, "infra_error": "x", "infra_kind": "server"}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                hs.progress("a rerun", f, 1.5)
            lines.append(buf.getvalue())
        self.assertEqual(lines, ["[a rerun] t1: submitted, 4 turns, 9 sub-calls, $0.5000; so far $1.5000\n",
                                 "[a rerun] t1: no answer, 4 turns, 9 sub-calls, $0.5000; so far $1.5000\n",
                                 "[a rerun] t1: infra server, 4 turns, 9 sub-calls, $0.5000; so far $1.5000\n"])

    def test_rlmh_info_reads_local_values_only(self):
        # the setting steers the expected format for the report (architecture 7.4)
        code, out, err = self.cli("--estimate")
        self.assertEqual(code, 0, err)
        self.assertIn("call format native (auto)", out)
        self.stub_config(config={"call_format": "fenced", "permissions.mode": "bypass"})
        with self.real_info():
            info = hs.rlmh_info(self.cmd())
            code, out, err = self.cli("--estimate")
        self.assertEqual(code, 0, err)
        self.assertIn("call format fenced (auto)", out)
        self.assertIn("permissions.mode = bypass", out)
        self.assertEqual(info["version"], "rlmh 0.1.0")
        self.assertEqual(info["config"], {"call_format": "fenced", "permissions.mode": "bypass", "root.thinking": "off",
                                          "sub.thinking": "off", "profile": "default"})
        self.assertEqual(self.calls()[0]["rest"], ["--version"])
        self.assertEqual([c["rest"][-1] for c in self.calls()[:6]],
                         ["--version", "call_format", "permissions.mode", "root.thinking", "sub.thinking", "profile"])
        self.assertEqual(self.batch_calls(), [])
        with self.assertRaises(hs.SetupError):
            hs.local_read([sys.executable, str(self.tmp / "no such stub.py")], ["--version"])
        with self.assertRaises(hs.SetupError):
            hs.local_read(["definitely-not-a-program-xyz"], ["--version"])


if __name__ == "__main__":
    unittest.main()
