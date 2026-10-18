"""Build the edit_spec and edit_bugs task pools (rlm_rl/tasks/editspec.py, editbugs.py). Run inside WSL2 with Docker.

    .venv-linux/bin/python scripts/build_edit_tasks_v2.py profile                # baseline run + per-test call map
    .venv-linux/bin/python scripts/build_edit_tasks_v2.py spec                   # clusters, stubs, documents, lab check
    .venv-linux/bin/python scripts/build_edit_tasks_v2.py screen                 # single candidate bugs, one by one
    .venv-linux/bin/python scripts/build_edit_tasks_v2.py bugs                   # linked bug sets, lab check
    .venv-linux/bin/python scripts/build_edit_tasks_v2.py grade spec bugs --jobs 6   # the real grader on every task
    .venv-linux/bin/python scripts/build_edit_tasks_v2.py index                  # edit/v2/{spec,bugs}_index.json
    .venv-linux/bin/python scripts/build_edit_tasks_v2.py report

Each stage takes repository names (substrings) to limit it; with none it runs every repository in REPOS.

Inputs are what edit_real's pipeline already built and validated (scripts/build_edit_data.py): each
repository's derived image `rlm-edit-<key>:<hash>`, its snapshot (edit/repos/<key>.tar) and its
registry entry (edit/repos.json). Nothing is pulled or downloaded. Outputs go under the data root's
edit/v2/ folder (D:\\RLM-data\\edit\\v2), never into the repository.

**The lab.** Validation runs the repository's tests in its own derived image, in one container per
repository and stage: the snapshot is unpacked into a tmpfs at /work/repo, the edit sandbox's Landlock
rules apply, and each run writes its files, runs pytest exactly as the grader does (grader.PYTEST_BOOT,
JUnit XML, the same flags), reads the report and puts the files back. That makes each task's check one
run of the original and one of the modified repository, as the brief asks, without a container start
per run on the slow data drive. The `profile` run adds a pytest plugin that records, for every test, the
repository functions it ran: which tests cover which function, and which functions run at import.

**Checks per task.** Fail-to-pass: passes in the baseline run and in the task's original run, fails
(or is missing) in its modified run. Pass-to-pass: passes in all three. Hidden files: the test files
with a fail-to-pass test (scope = hidden). The visible tests must still collect with those files gone
(as edit_real's visible-test check). edit_spec: every stubbed function is run by at least one
fail-to-pass test. edit_bugs: every bug, alone, fails 1-40 tests (`screen`), and every bug still shows
in the combined run. Then `grade` runs the real grader (grader.grade, fresh container, the task's
workspace built by editreal's builder) on every task: the gold patch must resolve it and the base must
fail every fail-to-pass test while passing every pass-to-pass one.
"""

import argparse
import asyncio
import collections
import concurrent.futures
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import editbox, editdata, grader  # noqa: E402
from rlm_rl.tasks import editbugs, editspec  # noqa: E402

V2 = "v2"  # under edit/
# Repositories (edit_real's registry keys) and the split their tasks go to. The selection and test repositories are
# edit_real's own held-out ones, which training never sees.
REPOS = {
    "andialbrecht__sqlparse.e57923b3": "train", "pallets__jinja.ada0a9a6": "train",
    "marshmallow-code__marshmallow.9716fc62": "train", "arrow-py__arrow.1d70d009": "train",
    "seperman__deepdiff.ed252022": "train", "tkrajina__gpxpy.09fc46b3": "train",
    "mahmoud__glom.fb3c4e76": "select", "lepture__mistune.bf54ef67": "test", "pyparsing__pyparsing.533adf47": "test",
}
# Candidates generated per repository and level, before the checks drop some, and tasks kept per split and level.
CANDIDATES = {"train": 9, "select": 6, "test": 4}
KEEP = {"train": 30, "select": 3, "test": 3}  # per level, across the split's repositories
SCREEN_PER_FUNCTION = 2
SCREEN_MAX_TESTS = 300  # a function more tests run is too central for a subtle bug
SCREEN_MAX = {"train": 420, "select": 200, "test": 100}  # candidate bugs screened per repository
SCREEN_TIMEOUT_S = 120  # a bug whose tests run longer than this probably hangs them: not kept
MAX_HIDDEN_FILES = 8
REPORT = "/tmp/rlm-lab-report.xml"
PROFILE_OUT = "/tmp/rlm-lab-profile.json"
TEST_HINT = "python -m pytest -q"
COLLECT = ["python", "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider"]  # editdata.visible_tests_run


def v2_path(*parts: str) -> Path:
    return editdata.edit_path(V2, *parts)


def registry() -> dict:
    return editdata.read_json(editdata.edit_path("repos.json"), {})


def pick_repos(names) -> list[str]:
    keys = list(REPOS)
    if names:
        keys = [k for k in keys if any(n.lower() in k.lower() for n in names)]
    return keys


# ----------------------------------------------------------------------------- the lab (runs inside the container)

LAB_SCRIPT = r'''
"""The v2 task lab (scripts/build_edit_tasks_v2.py): a job's pytest runs, one after another, in one container."""
import importlib.util
import json
import os
import signal
import subprocess
import tarfile
import time

REPO = "/work/repo"
OUT = "/lab/out/results.jsonl"


def landlock(writable):
    spec = importlib.util.spec_from_file_location("rlm_repl_worker", "/opt/rlm/repl_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker.apply_landlock(worker.edit_landlock_rules(writable))


def run(argv, env, timeout):
    started = time.monotonic()
    proc = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, _ = proc.communicate()
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    return proc.returncode, timed_out, time.monotonic() - started, out.decode("utf-8", "replace")


def read(path):
    with open(path, "rb") as f:
        return f.read()


def main():
    with open("/lab/job.json", encoding="utf-8") as f:
        job = json.load(f)
    with tarfile.open("/lab/snapshot.tar") as tar:
        tar.extractall(REPO, filter="tar")
    ll = landlock(job["writable"])
    with open(OUT, "a", encoding="utf-8") as out:
        out.write(json.dumps({"id": "@start", "landlock": ll}) + "\n")
        out.flush()
        for r in job["runs"]:
            saved = {}
            for rel, text in (r.get("write") or {}).items():
                p = os.path.join(REPO, rel)
                saved[rel] = read(p) if os.path.exists(p) else None
                with open(p, "w", encoding="utf-8", newline="") as f:
                    f.write(text)
            for rel in r.get("remove") or []:
                p = os.path.join(REPO, rel)
                if os.path.exists(p):
                    saved.setdefault(rel, read(p))
                    os.remove(p)
            for p in (job["report"], job["profile_out"]):
                if os.path.exists(p):
                    os.remove(p)
            env = dict(os.environ)
            env.update(job.get("env") or {})
            if r.get("collect"):
                argv = list(job["collect"])
            else:
                argv = list(job["pytest"])
                if r.get("profile"):
                    env["PYTHONPATH"] = "/lab/plugin" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                    env["RLM_LAB_PROFILE_OUT"] = job["profile_out"]
                    argv += ["-p", "rlm_lab_profile"]
                argv += list(r.get("args") or [])
            code, timed_out, secs, text = run(argv, env, r.get("timeout") or 900)
            junit = None
            if os.path.exists(job["report"]):
                junit = read(job["report"]).decode("utf-8", "replace")
            prof = None
            if r.get("profile") and os.path.exists(job["profile_out"]):
                with open(job["profile_out"], encoding="utf-8") as f:
                    prof = json.load(f)
            for rel, data in saved.items():
                p = os.path.join(REPO, rel)
                if data is None:
                    if os.path.exists(p):
                        os.remove(p)
                else:
                    with open(p, "wb") as f:
                        f.write(data)
            out.write(json.dumps({"id": r["id"], "exit": code, "timed_out": timed_out, "seconds": round(secs, 2),
                                  "out": text[-4000:], "junit": junit, "profile": prof}) + "\n")
            out.flush()


main()
'''

# A pytest plugin: per test, the repository functions it ran (a global trace function that only sees `call` events),
# and what ran while the tests were collected (module imports).
PROFILE_PLUGIN = r'''
import json
import os
import sys
import threading

import pytest

_REPO = os.path.realpath(os.environ.get("RLM_LAB_REPO", "/work/repo"))
_OUT = os.environ.get("RLM_LAB_PROFILE_OUT")
_codes = set()
_rel = {}
_data = {}


def _tracer(frame, event, arg):
    if event == "call":
        _codes.add(frame.f_code)
    return None


def _on():
    _codes.clear()
    threading.settrace(_tracer)
    sys.settrace(_tracer)


def _off():
    sys.settrace(None)
    threading.settrace(None)
    out = set()
    for c in list(_codes):
        f = c.co_filename
        rel = _rel.get(f)
        if rel is None:
            try:
                real = os.path.realpath(f)
            except Exception:
                real = f
            rel = real[len(_REPO) + 1:] if real.startswith(_REPO + os.sep) else ""
            _rel[f] = rel
        if rel:
            out.add("%s:%d:%s" % (rel, c.co_firstlineno, c.co_name))
    _codes.clear()
    return sorted(out)


# -p plugins load before the conftest files, which may import the package: trace from here, so what runs at import
# is recorded with the collection
_on()


def pytest_collection_finish(session):
    _data["<collect>"] = _off()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    _on()
    try:
        yield
    finally:
        _data[item.nodeid] = _off()
        _loc[item.nodeid] = item.location[0]  # the file JUnit's `file` attribute names (where the test is defined)


_loc = {}


def pytest_sessionfinish(session, exitstatus):
    sys.settrace(None)
    if _OUT:
        with open(_OUT, "w", encoding="utf-8") as f:
            json.dump({"calls": _data, "locations": _loc}, f)
'''


def junit_key(nodeid: str, location: str) -> str | None:
    """The grader's result key for a pytest node id: JUnit's classname and name as pytest's junitxml derives them
    (mangle_test_address), its `file` attribute (the item's location), then grader.pytest_key. Tests generated into
    another module (mistune's, defined in tests/__init__.py) get keys that differ from their node ids."""
    path, bracket, params = nodeid.partition("[")
    names = path.split("::")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    names[-1] += bracket + params
    return grader.pytest_key(location, ".".join(names[:-1]), names[-1])


def to_nodes(results: dict, nodeof: dict) -> dict:
    """Results keyed by grader key, re-keyed by pytest node id where the profile knows the test."""
    return {nodeof.get(k, k): v for k, v in results.items()}


def pytest_argv() -> list[str]:
    """The grader's pytest command (grader.Pytest.commands), with the lab's report path."""
    return ["python", "-c", grader.PYTEST_BOOT, "-p", "no:cacheprovider", "-o", "junit_family=xunit1",
            f"--junitxml={REPORT}", "--tb=short", "-q"]


def run_lab(key: str, entry: dict, runs: list[dict], label: str, timeout_s: float = 6 * 3600) -> dict[str, dict]:
    """Run `runs` in one lab container of the repository's derived image. Returns each run's result by id."""
    image = editbox.derived_image(entry["tag"])
    uid, gid = editbox.host_user()
    lab = editbox.cache_root() / "lab" / f"{key}-{label}-{secrets.token_hex(4)}"
    (lab / "out").mkdir(parents=True)
    (lab / "plugin").mkdir()
    (lab / "lab.py").write_text(LAB_SCRIPT, encoding="utf-8")
    (lab / "plugin" / "rlm_lab_profile.py").write_text(PROFILE_PLUGIN, encoding="utf-8")
    shutil.copyfile(editdata.data.path(entry["snapshot"]), lab / "snapshot.tar")
    writable = ["/work/repo", "/scratch", "/tmp", *image.writable, "/lab/out"]
    job = {"runs": runs, "writable": writable, "report": REPORT, "profile_out": PROFILE_OUT, "pytest": pytest_argv(),
           "collect": COLLECT, "env": {"PYTHONDONTWRITEBYTECODE": "1"}}
    (lab / "job.json").write_text(json.dumps(job), encoding="utf-8")
    for p in [lab, *lab.rglob("*")]:
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    os.chmod(lab / "out", 0o777)
    runtime, bundle = editbox.runtime_dir(), editbox.python_bundle_dir()
    name = f"rlm-lab-{secrets.token_hex(6)}"
    args = ["docker", "run", "--rm", "-i", "--pull", "never", "--name", name, "--label", "rlm.lab=1",
            "--network", "none", "--ipc", "none",
            "--tmpfs", f"/scratch:rw,nosuid,nodev,exec,size=256m,uid={uid},gid={gid},mode=0700",
            "--tmpfs", "/tmp:rw,nosuid,nodev,exec,size=1024m,mode=1777",
            "--tmpfs", f"/work/repo:rw,nosuid,nodev,exec,size=1536m,uid={uid},gid={gid},mode=0755",
            "--user", f"{uid}:{gid}", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--cpus", "2", "--memory", "4096m", "--memory-swap", "4096m", "--pids-limit", "256", "--ulimit", "core=0",
            "--hostname", "sandbox", "--workdir", "/work/repo",
            "--mount", f"type=bind,source={runtime},target=/opt/rlm,readonly",
            "--mount", f"type=bind,source={bundle},target=/opt/rlm/python,readonly",
            "--mount", f"type=bind,source={lab},target=/lab,readonly",
            "--mount", f"type=bind,source={lab / 'out'},target=/lab/out",
            "-e", "HOME=/scratch", "-e", "TMPDIR=/tmp"]
    for k, v in image.env:
        args += ["-e", f"{k}={v}"]
    entry_point, rest = editbox.python_command(image, "/lab/lab.py")
    args += ["--entrypoint", entry_point, image.tag, *rest]
    t = time.time()
    try:
        out = subprocess.run(args, capture_output=True, timeout=timeout_s, env=editbox.docker_cli_env())
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise
    results = {}
    path = lab / "out" / "results.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            results[r["id"]] = r
    start = results.pop("@start", None)
    if out.returncode or start is None or not start.get("landlock"):
        tail = (out.stderr or out.stdout).decode("utf-8", "replace")[-800:]
        if out.returncode or start is None:
            shutil.rmtree(lab, ignore_errors=True)
            raise RuntimeError(f"{key}: the lab failed (exit {out.returncode}, landlock {start}): {tail}")
    print(f"  {key} {label}: {len(results)} of {len(runs)} runs in {time.time() - t:.0f} s", flush=True)
    shutil.rmtree(lab, ignore_errors=True)
    return results


def parse_run(r: dict | None) -> tuple[dict, dict]:
    """(results by grader key, failure messages by key) from one lab run's JUnit report."""
    if not r or r.get("junit") is None:
        return {}, {}
    driver = {"commands": [{"exit": r["exit"], "timed_out": r["timed_out"]}],
              "reports": [{"path": REPORT, "found": True, "text": r["junit"]}]}
    spec = grader.TestSpec(runner="pytest", scope=[], f2p=[], p2p=[])
    results, _, _ = grader.RUNNER_CLASSES["pytest"].parse(spec, driver, REPORT)
    messages = {}
    try:
        import xml.etree.ElementTree as ET

        for case in ET.fromstring(r["junit"]).iter("testcase"):
            bad = case.find("failure")
            if bad is None:
                bad = case.find("error")
            if bad is None:
                continue
            key = grader.pytest_key(case.get("file"), case.get("classname") or "", case.get("name") or "")
            if key:
                messages[key] = bad.get("message") or (bad.text or "")[:500]
    except Exception:
        pass
    return results, messages


def collect_ok(r: dict | None, baseline: dict | None = None) -> bool:
    """Visible tests still collect (editdata.visible_tests_run's Python check), or show no more errors than the
    original did."""
    if not r or r.get("timed_out"):
        return False
    errors = lambda o: sum(int(n) for n in re.findall(r"(\d+) errors? (?:during collection|in )", o))  # noqa: E731
    if r["exit"] in (0, 5) and not errors(r["out"]):
        return True
    return baseline is not None and r["exit"] == baseline.get("exit") and errors(r["out"]) <= errors(baseline["out"])


# ----------------------------------------------------------------------------- repository data


def snapshot_texts(entry: dict) -> tuple[dict[str, str], dict[str, int]]:
    """The snapshot's Python files (text) and every file's size."""
    snap = editdata.data.path(entry["snapshot"])
    texts, sizes = {}, {}
    with tarfile.open(snap) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = m.name.removeprefix("./")
            sizes[name] = m.size
            if name.endswith(".py"):
                raw = tar.extractfile(m).read()
                try:
                    texts[name] = raw.decode("utf-8")
                except UnicodeDecodeError:
                    pass
    return texts, sizes


def test_files(texts: dict[str, str]) -> list[str]:
    out = []
    for p in texts:
        parts = p.split("/")
        name = parts[-1]
        if any(d in editspec.SKIP_DIRS - {"tests", "test"} for d in parts[:-1]):
            continue
        if (name.startswith("test_") or name.endswith("_test.py") or (p == "test.py")) and name != "conftest.py":
            if len(parts) == 1 or any(d in ("tests", "test") for d in parts[:-1]):
                out.append(p)
    return sorted(out)


def load_profile(key: str) -> dict:
    path = v2_path("profile", f"{key}.json")
    if not path.exists():
        raise SystemExit(f"{key}: no profile yet (run the profile stage)")
    return json.loads(path.read_text(encoding="utf-8"))


def repo_limits(entry: dict) -> tuple[int, int | None]:
    """(grader cap, cell limit) for the repository's tasks: the largest edit_real measured for its own tasks (its test
    command timed under load), so the slow drive costs time, never reward."""
    index = editdata.read_json(editdata.edit_path("index.json"), {"tasks": []})["tasks"]
    mine = [t for t in index if t.get("key") == entry_key(entry)]
    cap = max([int(t["tests"].get("timeout_s") or 600) for t in mine] + [600])
    cells = [int(t["cell_timeout_s"]) for t in mine if t.get("cell_timeout_s")]
    return cap, (max(cells) if cells else None)


def entry_key(entry: dict) -> str:
    return Path(entry["snapshot"]).stem


# ----------------------------------------------------------------------------- stage: profile


COLLECT_ERROR = re.compile(r"^ERROR ([^\s:]+\.py)", re.M)


def stage_profile(key: str, entry: dict) -> None:
    """The original repository's tests twice (once traced), and its collection: per test, the functions it runs and
    its grader key; the tests that pass both times are the baseline. A test file that errors at collection (marshmallow's
    mypy cases) stops pytest's whole run, so such files are left out and the runs repeated."""
    texts, _ = snapshot_texts(entry)
    tests = test_files(texts)
    dropped: list[str] = []
    for _ in range(3):
        runs = [{"id": "profile", "args": tests, "profile": True, "timeout": 3600},
                {"id": "plain", "args": tests, "timeout": 3600},
                {"id": "collect", "collect": True, "timeout": 900}]
        res = run_lab(key, entry, runs, "profile")
        bad = sorted(set(COLLECT_ERROR.findall(res["profile"]["out"])) & set(tests))
        if not bad:
            break
        dropped += bad
        tests = [t for t in tests if t not in bad]
    prof = res["profile"]
    data = prof.get("profile") or {}
    calls, locations = data.get("calls") or {}, data.get("locations") or {}
    keymap = {n: junit_key(n, locations.get(n) or n.split("::")[0]) for n in calls if n != "<collect>"}
    nodeof = {v: k for k, v in keymap.items()}
    results = to_nodes(parse_run(prof)[0], nodeof)
    plain = to_nodes(parse_run(res.get("plain"))[0], nodeof)
    passed = {n for n, v in results.items() if v == grader.PASSED and plain.get(n) == grader.PASSED}
    unmatched = sorted(n for n in results if n not in keymap)
    out = {"key": key, "tests": tests, "dropped_files": dropped, "results": results, "plain": plain,
           "passed_both": sorted(passed), "profile": calls, "keymap": keymap, "seconds": prof["seconds"],
           "plain_seconds": res["plain"]["seconds"],
           "collect": {k: res["collect"][k] for k in ("exit", "out", "timed_out")},
           "unmatched": unmatched[:50], "at": time.time()}
    editdata.write_json(v2_path("profile", f"{key}.json"), out)
    print(f"{key}: {len(results)} tests ({len(passed)} pass twice), {len(keymap)} profiled, {len(unmatched)} report "
          f"keys with no profiled test, files left out {dropped}, profile {prof['seconds']:.0f} s, plain "
          f"{res['plain']['seconds']:.0f} s, collect exit {res['collect']['exit']}", flush=True)


class Context:
    """What every later stage reads for one repository: its Python files, the analysis, the baseline and call map
    (tests by pytest node id), and node id <-> grader key."""

    def __init__(self, key: str, entry: dict):
        self.key, self.entry = key, entry
        self.texts, self.sizes = snapshot_texts(entry)
        self.prof = load_profile(key)
        self.repo = editspec.analyze(self.texts)
        self.calls, tests_of, self.at_import = editspec.map_profile(self.repo, self.prof["profile"])
        self.passed = set(self.prof["passed_both"])
        self.tests_of = {k: {t for t in v if t in self.passed} for k, v in tests_of.items()}
        self.keymap = self.prof["keymap"]
        self.nodeof = {v: k for k, v in self.keymap.items()}

    def results(self, run: dict | None) -> tuple[dict, dict]:
        results, messages = parse_run(run)
        return to_nodes(results, self.nodeof), to_nodes(messages, self.nodeof)

    def keys(self, nodes: list[str]) -> list[str]:
        return sorted(self.keymap.get(n, n) for n in nodes)


# ----------------------------------------------------------------------------- shared record building


def file_patch(changes: dict[str, tuple[str, str]]) -> str:
    """A git diff from each file's old text to its new one (git diff --no-index, bytes kept as they are)."""
    tmp = Path(tempfile.mkdtemp(prefix="rlm-v2-patch-"))
    try:
        out = []
        for rel, (old, new) in sorted(changes.items()):
            for side, text in (("a", old), ("b", new)):
                p = tmp / side / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(text.encode("utf-8"))
            r = subprocess.run(["git", "-c", "core.quotePath=false", "diff", "--no-index", "--no-color", "--full-index",
                                "--", f"a/{rel}", f"b/{rel}"], cwd=tmp, capture_output=True,
                               env=editbox.git_env(tmp))
            out.append(r.stdout.decode("utf-8").replace("a/a/", "a/").replace("b/b/", "b/"))
        return "".join(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def judge(passed: set, orig: dict, mod: dict) -> tuple[list[str], list[str]]:
    """(fail-to-pass, pass-to-pass) node ids from the baseline, the task's original run and its modified run."""
    nodes = set(orig) | set(mod)
    f2p = sorted(n for n in nodes if n in passed and orig.get(n) == grader.PASSED
                 and mod.get(n) in (grader.FAILED, grader.ERROR, None))
    p2p = sorted(n for n in nodes if n in passed and orig.get(n) == grader.PASSED and mod.get(n) == grader.PASSED)
    return f2p, p2p


def base_record(ctx: "Context", kind: str, level: int, ident: str, changes: dict, hidden: list, f2p: list, p2p: list,
                issue: str, targets: list, lab: dict) -> dict:
    """An index record in edit_real's format (tests by grader key), so editreal's workspace builder and the grader
    take it as they are."""
    entry = ctx.entry
    cap, cell = repo_limits(entry)
    break_patch = file_patch({p: (old, new) for p, (old, new) in changes.items()})
    gold = file_patch({p: (new, old) for p, (old, new) in changes.items()})
    nfiles, nlines = editdata.patch_size(gold)
    rec = {"id": ident, "split": REPOS[ctx.key], "language": "python", "repo": entry["repo"], "key": ctx.key,
           "kind": kind, "source": "generated", "instance_id": None, "licence": entry["licence"], "image": entry["tag"],
           "snapshot": entry["snapshot"], "snapshot_sha256": entry["sha256"], "break_patch": break_patch,
           "gold_patch": gold, "remove": sorted(hidden), "replace": {}, "issue": issue, "node_modules": [],
           "tests": grader.TestSpec(runner="pytest", scope=sorted(hidden), f2p=ctx.keys(f2p), p2p=ctx.keys(p2p),
                                    language="python", hidden={p: ctx.texts[p] for p in sorted(hidden)}, cwd=".",
                                    bin=[], args=[], timeout_s=cap).as_dict(),
           "test_hint": TEST_HINT, "level": level, "gold_files": nfiles, "gold_lines": nlines, "files": [],
           "targets": targets, "lab": lab,
           "generator": editspec.GENERATOR_VERSION if kind == "spec" else editbugs.GENERATOR_VERSION}
    if cell:
        rec["cell_timeout_s"] = cell
    return rec


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    by_id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by_id[r["id"]] = r  # a later line for the same id replaces an earlier one
    return list(by_id.values())


def raw_text(texts: dict, path: str) -> tuple[str, str]:
    split = editspec.split_newlines(texts[path])
    if split is None:
        raise ValueError(f"{path} mixes line ends")
    return split


def check_runs(ident: str, write: dict, scope: list[str]) -> list[dict]:
    """A task's three lab runs: its tests on the original, on the modified repository, and the visible tests'
    collection with the candidate hidden files gone."""
    return [{"id": f"{ident}:orig", "args": scope}, {"id": f"{ident}:mod", "args": scope, "write": write},
            {"id": f"{ident}:collect", "collect": True, "write": write, "remove": scope}]


def outcome(ctx: Context, res: dict, ident: str) -> tuple[list, list, list, dict]:
    """(fail-to-pass, pass-to-pass, hidden files, failure messages) of one task's lab runs."""
    orig, _ = ctx.results(res.get(f"{ident}:orig"))
    mod, messages = ctx.results(res.get(f"{ident}:mod"))
    f2p, p2p = judge(ctx.passed, orig, mod)
    hidden = sorted({editspec.test_file(t) for t in f2p})
    p2p = [t for t in p2p if editspec.test_file(t) in hidden]
    return f2p, p2p, hidden, messages


def common_why(ctx: Context, res: dict, ident: str, f2p: list, hidden: list) -> str:
    if len(hidden) > MAX_HIDDEN_FILES:
        return f"{len(hidden)} test files would be hidden"
    if not collect_ok(res.get(f"{ident}:collect"), ctx.prof.get("collect")):
        return "visible tests don't collect without the hidden files: " + (
            res.get(f"{ident}:collect") or {}).get("out", "")[-200:]
    return ""


def lab_info(res: dict, ident: str, why: str, f2p: list, p2p: list, **extra) -> dict:
    return {"ok": not why, "why": why, "f2p": len(f2p), "p2p": len(p2p),
            "orig_s": (res.get(f"{ident}:orig") or {}).get("seconds"),
            "mod_s": (res.get(f"{ident}:mod") or {}).get("seconds"), **extra}


def changed_texts(ctx: Context, by_file: dict, change) -> dict[str, tuple[str, str]]:
    """{path: (original text, changed text)}, each written back with the file's own line ends."""
    out = {}
    for path, items in by_file.items():
        text, newline = raw_text(ctx.texts, path)
        new = change(text, items)
        out[path] = (ctx.texts[path], new.replace("\n", newline) if newline != "\n" else new)
    return out


# ----------------------------------------------------------------------------- stage: spec


def stage_spec(key: str, entry: dict, seed: int) -> None:
    ctx = Context(key, entry)
    repo, split = ctx.repo, REPOS[key]
    rng = random.Random(f"{seed}-spec-{key}")
    done = {r["id"] for r in read_jsonl(v2_path("candidates", "spec.jsonl"))}
    total: dict = {}
    plans = []
    for level in range(1, editspec.MAX_LEVEL + 1):
        for cluster in editspec.spec_clusters(repo, ctx.tests_of, ctx.at_import, level, rng, CANDIDATES[split], {},
                                              total=total, max_files=MAX_HIDDEN_FILES):
            ident = "spec-" + entry["repo"].split("/")[1].lower() + f"-l{level}-" + hashlib.sha256(
                json.dumps([key, sorted(cluster)]).encode()).hexdigest()[:10]
            if ident in done:
                continue
            by_file: dict[str, list] = collections.defaultdict(list)
            for c in cluster:
                by_file[repo.fns[c].path].append(repo.fns[c])
            try:
                changes = changed_texts(ctx, by_file, editspec.stub_text)
            except (ValueError, SyntaxError):
                continue
            scope = sorted({editspec.test_file(t) for c in cluster for t in ctx.tests_of[c]})
            if len(scope) > MAX_HIDDEN_FILES * 2:
                continue
            plans.append((level, ident, cluster, changes, scope))
    if not plans:
        print(f"{key}: no new spec candidates")
        return
    runs = [r for _, ident, _, changes, scope in plans
            for r in check_runs(ident, {p: new for p, (old, new) in changes.items()}, scope)]
    res = run_lab(key, entry, runs, "spec")
    rows = []
    for level, ident, cluster, changes, scope in plans:
        f2p, p2p, hidden, _ = outcome(ctx, res, ident)
        uncovered = [c for c in cluster if not (ctx.tests_of[c] & set(f2p))]
        why = ("no test fails with the stubs" if not f2p else
               "a stub no fail-to-pass test runs: " + ", ".join(repo.fns[c].qualname for c in uncovered) if uncovered
               else common_why(ctx, res, ident, f2p, hidden))
        doc = editspec.requirements(entry["repo"], repo, cluster)
        leak = editdata.leaks(doc, file_patch({p: (n, o) for p, (o, n) in changes.items()}), [])
        if not why and leak:
            why = "; ".join(leak)
        targets = [{"path": repo.fns[c].path, "qualname": repo.fns[c].qualname, "kind": repo.fns[c].kind,
                    "body_lines": repo.fns[c].body_lines, "doc": bool(repo.fns[c].doc)} for c in cluster]
        rows.append(base_record(ctx, "spec", level, ident, changes, hidden, f2p, p2p, doc, targets,
                                lab_info(res, ident, why, f2p, p2p)))
    append_jsonl(v2_path("candidates", "spec.jsonl"), rows)
    ok = sum(r["lab"]["ok"] for r in rows)
    drops = collections.Counter(r["lab"]["why"].split(":")[0][:50] for r in rows if not r["lab"]["ok"])
    print(f"{key}: spec {ok} of {len(rows)} pass the lab; drops {dict(drops)}", flush=True)


# ----------------------------------------------------------------------------- stage: screen (edit_bugs)


def stage_screen(key: str, entry: dict, seed: int) -> None:
    """Each candidate bug alone, on the tests that run its function: kept if 1-40 of them fail (editbugs)."""
    ctx = Context(key, entry)
    repo = ctx.repo
    rng = random.Random(f"{seed}-screen-{key}")
    path = v2_path("screen", f"{key}.json")
    have = editdata.read_json(path, {})
    cand = sorted(k for k in editspec.eligible(repo, ctx.tests_of, ctx.at_import)
                  if len(ctx.tests_of[k]) <= SCREEN_MAX_TESTS)
    rng.shuffle(cand)
    runs, muts = [], {}
    for k in cand:
        for m in editbugs.mutations(repo, repo.fns[k], rng, SCREEN_PER_FUNCTION):
            if m.id in have or len(runs) >= SCREEN_MAX[REPOS[key]]:
                continue
            changes = changed_texts(ctx, {m.path: [m]}, editbugs.apply)
            runs.append({"id": m.id, "write": {m.path: changes[m.path][1]}, "args": sorted(ctx.tests_of[k]),
                         "timeout": SCREEN_TIMEOUT_S})
            muts[m.id] = m
    if not runs:
        print(f"{key}: nothing new to screen")
        return
    res = run_lab(key, entry, runs, "screen")
    kept = 0
    for mid, m in muts.items():
        r = res.get(mid)
        results, messages = ctx.results(r)
        ran = sorted(ctx.tests_of[m.fn])
        fails = [t for t in ran if results.get(t) in (grader.FAILED, grader.ERROR, None)]
        good = (r is not None and not r["timed_out"] and r["junit"] is not None
                and 1 <= len(fails) <= min(editbugs.MAX_SINGLE_FAILS, max(3, int(0.8 * len(ran)))))
        have[mid] = {"mutation": m.as_dict(), "fails": fails, "ran": len(ran), "good": good,
                     "messages": {t: messages.get(t, "") for t in fails[:60]},
                     "seconds": r["seconds"] if r else None}
        kept += good
    editdata.write_json(path, have)
    print(f"{key}: screened {len(muts)} bugs, {kept} kept ({sum(v['good'] for v in have.values())} in all)", flush=True)


# ----------------------------------------------------------------------------- stage: bugs


def stage_bugs(key: str, entry: dict, seed: int) -> None:
    ctx = Context(key, entry)
    repo, split = ctx.repo, REPOS[key]
    screened = editdata.read_json(v2_path("screen", f"{key}.json"), {})
    good: dict[str, list] = collections.defaultdict(list)
    for v in screened.values():
        m = v["mutation"]
        if v["good"] and m["fn"] in repo.fns and \
                ctx.texts[m["path"]].replace("\r\n", "\n")[m["start"]:m["end"]] == m["old"]:
            good[m["fn"]].append(v)
    rng = random.Random(f"{seed}-bugs-{key}")
    done = {r["id"] for r in read_jsonl(v2_path("candidates", "bugs.jsonl"))}
    total: dict = {}
    plans = []
    for level in range(1, editbugs.MAX_LEVEL + 1):
        for bugs in editbugs.bug_sets(repo, ctx.tests_of, good, level, rng, CANDIDATES[split], {}, total=total,
                                      max_files=MAX_HIDDEN_FILES):
            ids = [b["mutation"]["id"] for b in bugs]
            ident = "bugs-" + entry["repo"].split("/")[1].lower() + f"-l{level}-" + hashlib.sha256(
                json.dumps([key, sorted(ids)]).encode()).hexdigest()[:10]
            if ident in done:
                continue
            by_file = collections.defaultdict(list)
            for b in bugs:
                by_file[b["mutation"]["path"]].append(editbugs.Mutation(**b["mutation"]))
            changes = changed_texts(ctx, by_file, editbugs.apply)
            scope = sorted({editspec.test_file(t) for b in bugs for t in ctx.tests_of[b["mutation"]["fn"]]})
            plans.append((level, ident, bugs, changes, scope))
    if not plans:
        print(f"{key}: no new bug candidates")
        return
    runs = [r for _, ident, _, changes, scope in plans
            for r in check_runs(ident, {p: new for p, (old, new) in changes.items()}, scope)]
    res = run_lab(key, entry, runs, "bugs")
    rows = []
    for level, ident, bugs, changes, scope in plans:
        f2p, p2p, hidden, messages = outcome(ctx, res, ident)
        listed = editbugs.listed_failures(f2p, bugs, ctx.calls, random.Random(ident)) if f2p else None
        why = ("no test fails with the bugs" if not f2p else
               "a bug shows no failure of its own with the others in" if listed is None
               else common_why(ctx, res, ident, f2p, hidden))
        issue = editbugs.issue_text(entry["repo"], len(bugs), [(t, messages.get(t, "")) for t in listed or []])
        leak = editdata.leaks(issue, file_patch({p: (n, o) for p, (o, n) in changes.items()}), [])
        if not why and leak:
            why = "; ".join(leak)
        if not why and any(b["mutation"]["path"] in issue for b in bugs):
            why = "the text names a bugged file"
        targets = [{"path": b["mutation"]["path"], "qualname": b["mutation"]["fn"].split("::", 1)[1],
                    "mutation": b["mutation"]["kind"], "line": b["mutation"]["line"], "single_fails": len(b["fails"])}
                   for b in bugs]
        rows.append(base_record(ctx, "bugs", level, ident, changes, hidden, f2p, p2p, issue, targets,
                                lab_info(res, ident, why, f2p, p2p, listed=listed or [])))
    append_jsonl(v2_path("candidates", "bugs.jsonl"), rows)
    ok = sum(r["lab"]["ok"] for r in rows)
    drops = collections.Counter(r["lab"]["why"].split(":")[0][:50] for r in rows if not r["lab"]["ok"])
    print(f"{key}: bugs {ok} of {len(rows)} pass the lab; drops {dict(drops)}", flush=True)


# ----------------------------------------------------------------------------- stage: grade (the real grader)


async def _grade_all(records: list[dict], out_path: Path, jobs: int) -> None:
    gate = asyncio.Semaphore(jobs)
    images: dict = {}

    async def one(rec):
        async with gate:
            t = time.time()
            row = {"id": rec["id"], "at": time.time()}
            try:
                image = images.setdefault(rec["image"], editbox.EditImage.derived(rec["image"]))
                task = editspec.to_task(rec, "edit_spec") if rec["kind"] == "spec" else editbugs.to_task(rec, "edit_bugs")
                ws = await asyncio.to_thread(task.workspace.ensure)
                spec = grader.TestSpec.from_dict(rec["tests"])
                gold = await grader.grade(rec["gold_patch"], spec, ws_dir=ws, image=image, run_id="edit-v2-grade")
                base = await grader.grade("", spec, ws_dir=ws, image=image, run_id="edit-v2-grade", keep_all=True)
                stubs_in = all(target_in_base(rec, ws))
                row.update(gold_resolved=gold.resolved, gold_f2p=gold.f2p_passed, gold_p2p_failed=gold.p2p_failed,
                           gold_hacks=gold.hacks, base_f2p_passed=base.f2p_passed, base_p2p_failed=base.p2p_failed,
                           base_hacks=[h for h in base.hacks if h["rule"] == "grader_tamper"], gold_s=gold.seconds,
                           base_s=base.seconds, base_in_tree=stubs_in, retries=gold.retries + base.retries)
                row["ok"] = bool(gold.resolved and not gold.hacks and base.f2p_passed == 0 and base.p2p_failed == 0
                                 and not row["base_hacks"] and stubs_in)
                if not row["ok"]:
                    row["why"] = (f"gold resolved {gold.resolved} ({gold.f2p_passed}/{gold.f2p_total} f2p, "
                                  f"{gold.p2p_failed} p2p failed, hacks {gold.hacks}); base passes {base.f2p_passed} "
                                  f"f2p, fails {base.p2p_failed} p2p; base tree has the change: {stubs_in}")
            except Exception as exc:  # recorded; a rerun tries again
                row.update(ok=False, why=f"error: {exc!r}"[:400], error=True)
            append_jsonl(out_path, [row])
            print(f"{'OK ' if row['ok'] else 'no '} {rec['id']:45s} {time.time() - t:5.0f}s {row.get('why', '')[:150]}",
                  flush=True)

    await asyncio.gather(*(one(r) for r in records))


def target_in_base(rec: dict, ws: Path):
    """Each changed file in the task's base workspace differs from the snapshot (the break patch applied), and the
    hidden test files are gone."""
    for fp in grader.parse_patch(rec["break_patch"]):
        yield (ws / "tree" / fp.path).is_file()
    for p in rec["remove"]:
        yield not (ws / "tree" / p).exists()


def stage_grade(families: list[str], names, jobs: int, everything: bool = False) -> None:
    keys = set(pick_repos(names))
    for fam in families:
        cands = [r for r in read_jsonl(v2_path("candidates", f"{fam}.jsonl")) if r["lab"]["ok"] and r["key"] in keys]
        out = v2_path("graded", f"{fam}.jsonl")
        graded = {r["id"]: r for r in read_jsonl(out) if not r.get("error")}
        todo = grading_plan(cands, graded) if not everything else [r for r in cands if r["id"] not in graded]
        print(f"{fam}: grading {len(todo)} tasks ({len(graded)} done)", flush=True)
        asyncio.run(_grade_all(todo, out, jobs))


GRADE_SPARE = {"train": 4, "select": 1, "test": 1}  # graded beyond KEEP per level, for the ones the grader drops


def grading_plan(cands: list[dict], graded: dict) -> list[dict]:
    """The candidates worth a grader run: per split and level, round-robin over repositories (as the index takes
    them), the ones already graded and passed first, up to KEEP plus a few spares; returns those not graded yet."""
    rng = random.Random(11)
    todo = []
    for split, per_level in KEEP.items():
        want = per_level + GRADE_SPARE[split]
        for level in range(1, 6):
            pools = collections.defaultdict(list)
            for r in sorted((r for r in cands if r["split"] == split and r["level"] == level), key=lambda r: r["id"]):
                if r["id"] in graded and not graded[r["id"]].get("ok"):
                    continue
                pools[r["key"]].append(r)
            for items in pools.values():
                rng.shuffle(items)
                items.sort(key=lambda r: r["id"] not in graded)  # passed ones first
            picked = 0
            while picked < want and any(pools.values()):
                for k in sorted(pools):
                    if pools[k] and picked < want:
                        r = pools[k].pop(0)
                        picked += 1
                        if r["id"] not in graded:
                            todo.append(r)
    return todo


# ----------------------------------------------------------------------------- stage: index


def balance(records: list[dict], per_level: int, seed: int) -> list[dict]:
    """Up to `per_level` tasks per level, round-robin over repositories, a seeded shuffle within each."""
    rng = random.Random(seed)
    out = []
    for level in range(1, 6):
        pools = collections.defaultdict(list)
        for r in sorted((r for r in records if r["level"] == level), key=lambda r: r["id"]):
            pools[r["key"]].append(r)
        for items in pools.values():
            rng.shuffle(items)
        picked = []
        while len(picked) < per_level and any(pools.values()):
            for k in sorted(pools):
                if pools[k] and len(picked) < per_level:
                    picked.append(pools[k].pop())
        out += picked
    return out


def stage_index(families: list[str]) -> None:
    for fam in families:
        cands = {r["id"]: r for r in read_jsonl(v2_path("candidates", f"{fam}.jsonl")) if r["lab"]["ok"]}
        graded = {r["id"]: r for r in read_jsonl(v2_path("graded", f"{fam}.jsonl"))}
        ok = [dict(cands[i], graded=graded[i]) for i in cands if graded.get(i, {}).get("ok")]
        tasks = []
        for split, per_level in KEEP.items():
            tasks += balance([r for r in ok if r["split"] == split], per_level, seed=7)
        mod = editspec if fam == "spec" else editbugs
        for rec in tasks:
            ws = mod.to_task(rec, f"edit_{fam}").workspace.ensure()
            rec["files"] = sorted([str(p.relative_to(ws / "tree")), p.stat().st_size]
                                  for p in (ws / "tree").rglob("*") if p.is_file() and not p.is_symlink())
            g = rec["graded"]
            slowest = max(g.get("gold_s") or 0.0, g.get("base_s") or 0.0)
            rec["tests"] = dict(rec["tests"], timeout_s=max(rec["tests"]["timeout_s"], editdata.grader_cap(slowest)))
        path = editdata.data.path(mod.INDEX)
        editdata.write_json(path, {"format": 1, "built_at": time.time(), "generator": mod.GENERATOR_VERSION,
                                   "tasks": tasks})
        counts = collections.Counter((r["split"], r["level"]) for r in tasks)
        print(f"edit_{fam}: {len(tasks)} tasks -> {path}")
        for (split, level), n in sorted(counts.items()):
            print(f"  {split:6s} L{level} {n:3d}")


def stage_report(families: list[str]) -> None:
    for fam in families:
        cands = read_jsonl(v2_path("candidates", f"{fam}.jsonl"))
        graded = {r["id"]: r for r in read_jsonl(v2_path("graded", f"{fam}.jsonl"))}
        rows = collections.Counter()
        for r in cands:
            g = graded.get(r["id"], {})
            rows[(r["split"], r["key"].split(".")[0], r["level"], "lab_ok" if r["lab"]["ok"] else "lab_no",
                  "graded_ok" if g.get("ok") else "graded_no" if g else "ungraded")] += 1
        print(f"== {fam}")
        for k, n in sorted(rows.items()):
            print(" ", *k, n)
        drops = collections.Counter(r["lab"]["why"].split(":")[0][:60] for r in cands if not r["lab"]["ok"])
        print("  lab drops:", dict(drops))
        gdrops = collections.Counter((g.get("why") or "")[:80] for g in graded.values() if not g.get("ok"))
        print("  grade drops:", dict(gdrops))


# ----------------------------------------------------------------------------- main


def for_repos(names, fn, jobs: int, *args) -> None:
    reg = registry()
    keys = pick_repos(names)
    missing = [k for k in keys if k not in reg]
    if missing:
        raise SystemExit(f"not in edit/repos.json: {missing}")

    def one(key):
        try:
            fn(key, reg[key], *args)
        except Exception as exc:
            print(f"{key}: FAILED {exc!r}"[:600], flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, keys))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("profile", "spec", "screen", "bugs"):
        p = sub.add_parser(name)
        p.add_argument("repos", nargs="*")
        p.add_argument("--jobs", type=int, default=5, help="repositories at once (one lab container each)")
        p.add_argument("--seed", type=int, default=20261018)
    p = sub.add_parser("grade")
    p.add_argument("families", nargs="+", choices=("spec", "bugs"))
    p.add_argument("--repos", nargs="*")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--all", action="store_true", help="grade every candidate, not only what the index can take")
    for name in ("index", "report"):
        p = sub.add_parser(name)
        p.add_argument("families", nargs="*", default=["spec", "bugs"])
    args = parser.parse_args()
    if args.command == "profile":
        for_repos(args.repos, stage_profile, args.jobs)
    elif args.command in ("spec", "screen", "bugs"):
        stage = {"spec": stage_spec, "screen": stage_screen, "bugs": stage_bugs}[args.command]
        for_repos(args.repos, stage, args.jobs, args.seed)
    elif args.command == "grade":
        os.environ.setdefault(grader.GLOBAL_SLOTS_ENV, str(args.jobs))
        stage_grade(args.families, args.repos, args.jobs, args.all)
    elif args.command == "index":
        stage_index(args.families)
    else:
        stage_report(args.families)
    return 0


if __name__ == "__main__":
    sys.exit(main())
