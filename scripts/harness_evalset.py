"""Eval set `harness-eval-v1` for rlm-harness v1 (step 1): 48 tasks in five tracks, built, validated and checked.

    .venv\\Scripts\\python.exe scripts\\harness_evalset.py build [--only TRACKS] [--jobs N] [--no-validate] [--force]
    .venv\\Scripts\\python.exe scripts\\harness_evalset.py check
    .venv\\Scripts\\python.exe scripts\\harness_evalset.py summary

`--set NAME` (default harness-eval-v1), `--data DIR` (default data.root()) and `--harness DIR` (default ../rlm-harness)
apply to all three. The set goes to <data>/harness-eval/<set>/ in rlm-harness v1 architecture §6.1's layout, and the
build cache (candidates, gold, build venvs, the Web-Bench clone, the pnpm store, Chromium, the kernel sources and the
validation results) to <data>/harness-eval/build/<set>/, so a rerun resumes. Both must sit outside the RLM repo and the
rlm-harness checkout: task data never enters git.

Tracks (D-176): `python` (training's edit tasks, plus the `edit_large` family built here, D-181), `web` (Web-Bench
projects, D-222), `kernel` (GPU kernels in one CUDA/HIP dialect, written for this set and kept in the build cache,
D-223), `research` (training's research_test, D-224) and `search` (callgraph_heldout and needle_heldout). Splits are
disjoint by repository, project and kernel group (D-180). Every repo task is validated natively with the commands
`rlmh eval` grades with (D-183, D-184): base fails every fail-to-pass test, gold passes it three times out of three.

Nothing here calls a model. `build` downloads from PyPI (uv), the npm registry (pnpm), github.com (the Web-Bench clone)
and Playwright's browser host; the Docker fallback for a venv runs locally. Exit codes: 0 done; 1 `check` found
problems, or a cell could not fill (the build stops for Rishi's call); 2 setup (paths inside a repo, an existing set
without --force, a missing tool or input).
"""

from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures
import configparser
import contextlib
import datetime
import gzip
import hashlib
import io
import json
import os
import random
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rlm_rl import data, editdata, grader, tasks  # noqa: E402
from rlm_rl.tasks import editreal, editspec  # noqa: E402

EXIT_OK, EXIT_PROBLEMS, EXIT_SETUP = 0, 1, 2

# ----------------------------------------------------------------------------- the plan (D-176)

SET = "harness-eval-v1"
READ_PLAN = {("research_test", 2): 4, ("research_test", 3): 4,       # track "research"
             ("callgraph_heldout", 2): 2, ("needle_heldout", 3): 2}   # track "search"; 12 in all, size "R"
PYTHON_PLAN = {"S": 6, "M": 6, "L": 4}                                 # 16
WEB_PLAN = {"S": 4, "M": 4, "L": 2}                                    # 10
KERNEL_PLAN = {"S": 4, "M": 4, "L": 2}                                 # 10
SPLIT_REPOS = {"tune": ("andialbrecht__sqlparse.e57923b3", "pallets__jinja.ada0a9a6", "arrow-py__arrow.1d70d009",
                        "tkrajina__gpxpy.09fc46b3", "mahmoud__glom.fb3c4e76"),
               "heldout": ("marshmallow-code__marshmallow.9716fc62", "seperman__deepdiff.ed252022",
                           "lepture__mistune.bf54ef67", "pyparsing__pyparsing.533adf47")}   # D-180, python
WEBBENCH = ("https://github.com/bytedance/web-bench", "7b31ca2b786eef120dd49ce63dd03d0c0006046d")
WEB_SPLIT = {"tune": ("react", "svelte", "tailwind", "canvas", "form", "sass", "dom", "calculator"),
             "heldout": ("vue", "angular", "unocss", "table", "stylus", "survey", "bom", "grid")}   # D-180, preference order
WEB_REQUESTS = {"S": 3, "M": 6, "L": 12}                               # requests 1..n
NVRTC = 'nvidia-cuda-nvrtc-cu12==12.9.86; sys_platform != "darwin"'   # win_amd64 and manylinux wheels; unused on HIP machines
PNPM = "10.34.7"  # the newest 10.x at least two weeks old on 2026-11-02 (released 2026-10-12), fixed once

SPLITS = ("tune", "heldout")
TRACKS = ("python", "web", "kernel", "research", "search")
SIZES = ("R", "S", "M", "L")
READ_TRACK = {"research_test": "research", "callgraph_heldout": "search", "needle_heldout": "search"}
PYTHON_FAMILIES = {"S": (("edit_real", (1, 2)), ("edit_bugs", (1, 2))),
                   "M": (("edit_spec", (3, 5)), ("edit_bugs", (3, 5)), ("edit_real", (3, 5)))}
KERNEL_OPS = {  # (split, size): ops in table order; spares follow the split's ops (swapped in only by hand, and logged)
    ("tune", "S"): ("softmax_rows", "swiglu"), ("tune", "M"): ("dequant_gemv", "topk_topp"),
    ("tune", "L"): ("paged_attention",),
    ("heldout", "S"): ("rmsnorm", "rope"), ("heldout", "M"): ("tiled_matmul", "reductions"),
    ("heldout", "L"): ("flash_attention",)}
KERNEL_SPARES = {"tune": ("gelu_tanh", "int4_pack"), "heldout": ("layernorm", "transpose_tiled")}
WEB_VARIANTS = ("dom1", "calculator-files", "svg-solar", "react-no-ts")  # kept out of both splits (D-180)
WEB_SERVER_PACKAGES = ("sqlite3", "better-sqlite3", "mongoose", "mongodb", "prisma", "@prisma/client", "sequelize",
                       "typeorm", "mysql", "mysql2", "pg", "redis", "lowdb", "nedb")
WEB_BUILT = ("esbuild", "@swc/core", "@parcel/watcher", "lightningcss", "sharp")  # packages whose install script runs
MAX_PER_REPO = 2  # tasks from one repository (or project) in one cell (D-180)
LARGE = {"min_fns": 3, "max_modules": 8, "min_modules": 4, "min_lines": 300, "max_lines": 1200, "max_hidden": 16,
         "per_repo": 2}
GOLD_RUNS = 3
P2P_DROP_SHARE = 0.10
RESEARCH_REDRAWS = 20
PYTHONS = ("3.12", "3.11", "3.10")
TEST_EXTRAS = ("test", "tests", "testing", "dev")

LARGE_TEXT = ("Every function and method listed in REQUIREMENTS.md, at the root of this repository, has had its body "
              "replaced by `raise NotImplementedError`. Their signatures and docstrings are unchanged. Implement all of "
              "them so the package behaves as REQUIREMENTS.md and the docstrings describe, and keep the repository's "
              "other tests passing.")
WEB_TEXT = ("This is a web project: its source is in `src/`, and `package.json` and `playwright.config.js` set up its "
            "dev server and browser tests. FEATURES.md, at the root of this repository, lists feature requests. "
            "Implement them in order, each on top of the ones before it, so the site does everything the requests "
            "describe.")
WEB_TEXT_S = WEB_TEXT.replace("FEATURES.md, at the root of this repository, lists feature requests.",
                              "The feature requests are:")
KERNEL_TEXT = ("This repository is part of the GPU backend of an LLM inference engine. The kernels in "
               "kernels/*.gpu.cu have empty bodies. They are written in one dialect that compiles for both CUDA and "
               "HIP: gpu_compat.h, which the loader puts in front of each file, maps the differences, and a warp isn't "
               "always 32 threads. Implement the kernels as REQUIREMENTS.md and the doc comments describe, including "
               "every edge case they name, and set each launch configuration in launch.py. Test on the GPU with the "
               "command below, and add tests of your own.")
TEST_HINT = {"python": "{python} -m pytest -q", "kernel": "{python} -m pytest -q tests",
             "web": "{pnpm} exec playwright test"}
ID_RE = re.compile(r"^harness-eval-v1-(python|web|kernel|research|search)-(R|S|M|L)-\d{2}$")
KEY_LIKE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH|HEADERS", re.I)
SUMMARY = ROOT / "documentation" / "harness-eval-v1.md"
SEED_PRIVATE = ("private/seeds/harness-eval-v1-read.json", "private/seeds/harness-eval-v1-repo.json")


class SetupError(Exception):
    """Exit 2: something outside the build is wrong (a path, a tool, an input)."""


class Short(Exception):
    """Exit 1: a cell could not fill. Rishi decides (3 per repository, or a smaller set), logged as a D-number."""


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------- paths and the outside-the-repo guard


@dataclass
class Paths:
    data: Path
    name: str
    harness: Path

    @property
    def set_dir(self) -> Path:
        return self.data / "harness-eval" / self.name

    @property
    def cache(self) -> Path:
        return self.data / "harness-eval" / "build" / self.name


def _inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def check_outside(paths: Paths) -> None:
    """The set and the build cache resolve outside the RLM repo and the harness checkout (task data never enters git)."""
    for target in (paths.set_dir, paths.cache):
        for root in (ROOT, paths.harness):
            if _inside(target, root):
                raise SetupError(f"{target} is inside {root}: task data never enters git; pass --data outside both "
                                 "repositories")


def git_rlm(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def check_ignored() -> None:
    for probe in SEED_PRIVATE:
        if git_rlm("check-ignore", "-q", probe).returncode != 0:
            raise SetupError(f"{probe} is not git-ignored: `private/` must be in .gitignore before seeds are drawn")


def rlm_commit() -> dict:
    head = git_rlm("rev-parse", "HEAD").stdout.strip() or "unknown"
    dirty = bool(git_rlm("status", "--porcelain", "--untracked-files=no").stdout.strip())
    return {"repo": "RishiJain905/rlm", "commit": head, "dirty": dirty}


# ----------------------------------------------------------------------------- bytes, JSON lines and tars


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jline(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    tmp.write_bytes((json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(tmp, path)


def tree_files(folder: Path) -> dict[str, bytes]:
    """Every regular file under folder, by its relative POSIX path; skips .git and node_modules."""
    out = {}
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder).as_posix()
        parts = rel.split("/")
        if {".git", "node_modules", "__pycache__", ".pytest_cache"} & set(parts) or p.is_symlink() or not p.is_file():
            continue
        out[rel] = p.read_bytes()
    return out


def tar_bytes(files: dict[str, bytes], gz: bool = False) -> bytes:
    """A ustar archive of files: members sorted, mtime 0, uid and gid 0, no user or group names, mode 0644. The same
    files give the same bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(files[name]))
    raw = buf.getvalue()
    if not gz:
        return raw
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, filename="") as z:
        z.write(raw)
    return out.getvalue()


def untar(blob: bytes, gz: bool = False) -> dict[str, bytes]:
    if gz:
        blob = gzip.decompress(blob)
    out = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
        for m in tar.getmembers():
            if m.isfile():
                out[m.name] = tar.extractfile(m).read()
    return out


def tar_names(blob: bytes, gz: bool = False) -> list[tuple[str, bool]]:
    """(member name, is a regular file) for every member."""
    if gz:
        blob = gzip.decompress(blob)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
        return [(m.name, m.isfile()) for m in tar.getmembers()]


def put_files(folder: Path, files: dict[str, bytes]) -> None:
    for rel, b in files.items():
        p = folder / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)


def rmtree(path: Path) -> None:
    """Remove a folder, clearing read-only bits (Windows leaves them on git objects and node_modules)."""

    def onexc(func, target, exc):
        if isinstance(exc, FileNotFoundError):
            return
        os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
        func(target)

    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=onexc)
        else:
            shutil.rmtree(path, onerror=lambda func, target, info: onexc(func, target, info[1]))


def clean_env(extra: dict | None = None, drop: tuple = ()) -> dict:
    """The environment for a test run: key-like variables stripped (rlm-harness D-29, D-133), plus extra."""
    env = {k: v for k, v in os.environ.items() if not KEY_LIKE.search(k) and k not in drop}
    env = {k: v for k, v in env.items() if not k.startswith("GIT_CONFIG_")}
    env.update(extra or {})
    return env


def run(argv: list[str], cwd: Path | None = None, env: dict | None = None, timeout: float = 1800,
        input: bytes | None = None, stdout_only: bool = False) -> tuple[int, str, float]:
    """(exit code, output, seconds): the tail of stdout and stderr, or all of stdout alone when it is parsed. A timeout
    returns code -9."""
    started = time.monotonic()
    try:
        r = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=timeout, input=input,
                           stdin=None if input is not None else subprocess.DEVNULL)
        if stdout_only:
            return r.returncode, r.stdout.decode("utf-8", "replace"), time.monotonic() - started
        out = (r.stdout + r.stderr).decode("utf-8", "replace")
        return r.returncode, out[-4000:], time.monotonic() - started
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"") + (e.stderr or b"")).decode("utf-8", "replace")
        return -9, out[-4000:], time.monotonic() - started
    except FileNotFoundError as e:
        return -2, str(e), time.monotonic() - started


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------------- read tasks (research and search, D-224)


@dataclass
class ReadTask:
    track: str
    split: str
    family: str
    level: int
    task: object  # rlm_rl.env.Task

    @property
    def pass_score(self) -> float:
        return pass_score(self.task.grading)


def pass_score(grading: dict) -> float:
    """0.8 for a research list question (training's open-book bar), 1.0 for every other read task (D-170, D-224)."""
    return 0.8 if grading.get("type") == "list" and grading.get("source") == "qampari" else 1.0


GRADABLE = ("int", "float", "count", "entity", "list", "text")


def read_seeds(n: int) -> tuple[list[int], list[int]]:
    """(the plan's n seeds, replacement seeds appended after them), from private/seeds/harness-eval-v1-read.json."""
    name = "harness-eval-v1-read"
    try:
        seeds = tasks.load_private_seeds(name)
    except FileNotFoundError:
        seeds = tasks.private_seed_list(name, n)
    if len(seeds) < n:
        raise SetupError(f"private/seeds/{name}.json holds {len(seeds)} seeds, but the set needs at least {n}")
    return seeds[:n], seeds[n:]


def build_read(plan: dict | None = None, make=None) -> list[ReadTask]:
    """tasks.fixed_set's draw, with the list and type guards: each split's research tasks hold a list question, and
    every task grades with a type the `answer` grader ports (D-189). Replacement seeds are appended to the seed file.
    The seed file always holds the full READ_PLAN's seeds; a subset (`--only search`) takes its entries' positions in
    it, so a later full build draws the same tasks."""
    make = make or tasks.make_task
    plan = plan or READ_PLAN
    full = dict(READ_PLAN)
    for k, n in plan.items():  # a plan outside READ_PLAN (tests) gets positions after it
        full.setdefault(k, n)
    base, extra = read_seeds(sum(full.values()))
    positions, at = {}, 0
    for k, n in full.items():
        positions[k] = range(at, at + n)
        at += n
    queue = collections.deque(extra)
    added: list[int] = []

    def next_seed() -> int:
        if queue:
            return queue.popleft()
        s = tasks.new_seed()
        added.append(s)
        return s

    out = []
    for (family, level), count in plan.items():
        for i in range(count):
            seed = base[positions[(family, level)][i]]
            out.append(ReadTask(READ_TRACK[family], SPLITS[i % 2], family, level,
                                make(family, level, seed, f"{SET}-{family}-L{level}-{i:02d}")))
    for rt in out:  # before any redraw: a missing check file stops the build with nothing appended
        g = rt.task.grading
        if rt.track == "research" and (g.get("closed_book") is None or g.get("open_book") is None):
            raise SetupError("a research task's closed-book or open-book check is None: those checks must exist "
                             "(scripts/closed_book.py, scripts/open_book.py make model calls; ask Rishi)")
    try:
        for k, rt in enumerate(out):  # the type guard
            draws = 0
            while rt.task.grading.get("type") not in GRADABLE:
                if draws >= RESEARCH_REDRAWS:
                    raise Short(f"{rt.family} L{rt.level}: {RESEARCH_REDRAWS} draws gave no gradable type")
                draws += 1
                out[k] = rt = ReadTask(rt.track, rt.split, rt.family, rt.level,
                                       make(rt.family, rt.level, next_seed(), rt.task.task_id))
        for split in SPLITS:  # the list guard
            idx = [i for i, rt in enumerate(out) if rt.track == "research" and rt.split == split]
            draws = 0
            while idx and not any(out[i].task.grading.get("kind") == "list" for i in idx):
                if draws >= RESEARCH_REDRAWS:
                    raise Short(f"research {split}: {RESEARCH_REDRAWS} draws gave no list question")
                draws += 1
                old = out[idx[-1]]  # the split's last multi-hop task gets a new seed
                out[idx[-1]] = ReadTask(old.track, old.split, old.family, old.level,
                                        make(old.family, old.level, next_seed(), old.task.task_id))
    finally:
        if added:
            name = "harness-eval-v1-read"
            tasks.save_private_seeds(name, tasks.load_private_seeds(name) + added)
    for rt in out:
        g = rt.task.grading
        if rt.track == "research" and (g.get("closed_book") is None or g.get("open_book") is None):
            raise SetupError("a research task's closed-book or open-book check is None: those checks must exist "
                             "(scripts/closed_book.py, scripts/open_book.py make model calls; ask Rishi)")
    return out


def read_leak(question: str, gold) -> str:
    items = gold if isinstance(gold, list) else [gold]
    low = question.lower()
    for item in items:
        s = str(item).strip()
        if len(s) >= 4 and s.lower() in low:
            return "the question contains its gold"
    return ""


# ----------------------------------------------------------------------------- repo candidates


@dataclass
class Cand:
    """One repo-task candidate. `key` is stable across builds, so validation results resume from the cache."""
    key: str
    track: str
    split: str
    size: str
    source: dict  # family, record_id, repo, project, op, level: build cache only (sources.jsonl)
    task: str
    runner: str  # pytest | playwright
    env: str  # venv or node env key
    files: list[str]  # the hidden test files to run
    base: Callable[[Path], None]  # writes the base tree, hidden files left out
    hidden: dict[str, bytes]
    gold: Callable[[Path], None]  # turns a base tree into gold
    f2p: list[str] = field(default_factory=list)  # starting lists (python); web and kernel derive them from gold
    p2p: list[str] = field(default_factory=list)
    leak_texts: list[str] = field(default_factory=list)  # task text plus REQUIREMENTS.md / FEATURES.md / stubs
    gold_patch: str = ""
    mutants: list[Callable[[Path], None]] = field(default_factory=list)
    cover: dict = field(default_factory=dict)  # python L: stubbed function -> the tests that run it
    spec_titles: list[str] = field(default_factory=list)  # web: titles of the hidden specs (the leak rule)
    gold_lines: int = 0
    gold_files: int = 0

    def base_files(self) -> dict[str, bytes]:
        """The base exactly as its tar holds it (tree_files drops links, caches and empty folders), built once."""
        if not hasattr(self, "_base_files"):
            with tempfile.TemporaryDirectory(prefix="base-") as t:
                work = Path(t) / "repo"
                work.mkdir()
                self.base(work)
                self._base_files = tree_files(work)
        return self._base_files

    @property
    def group(self) -> str:
        """The repository, project or op: at most MAX_PER_REPO per cell."""
        return self.source.get("repo") or self.source.get("project") or self.source.get("op") or self.key


VALIDATOR = 2  # bump when validation rules change, so cached verdicts are redone


def cand_key(*parts) -> str:
    return sha256(json.dumps([VALIDATOR, *parts], sort_keys=True, default=str).encode())[:20]


# ---- python S and M (training's three edit indexes)


def edit_records() -> dict[str, list[dict]]:
    """Every Python, permissively licensed record of the three edit indexes, by family."""
    out = {}
    for family, rel in (("edit_real", editreal.INDEX), ("edit_spec", "edit/v2/spec_index.json"),
                        ("edit_bugs", "edit/v2/bugs_index.json")):
        path = data.path(rel)
        if not path.exists():
            raise SetupError(f"no edit index at {path}")
        recs = json.loads(path.read_text(encoding="utf-8"))["tasks"]
        out[family] = [r for r in recs if r.get("language") == "python" and r.get("licence") in editreal.PERMISSIVE]
    return out


def repo_key(record: dict) -> str:
    return record.get("key") or Path(record["snapshot"]).stem


def size_of(family: str, record: dict) -> str | None:
    level, files = int(record["level"]), int(record.get("gold_files") or 0)
    for size, fams in PYTHON_FAMILIES.items():
        for fam, (lo, hi) in fams:
            if fam == family and lo <= level <= hi and (files == 1 if size == "S" else files >= 2):
                return size
    return None


def assign_repos(recs: dict[str, list[dict]]) -> dict[str, str]:
    """Repository -> split (D-180): the nine v2 repositories as SPLIT_REPOS fixes them; any other one, in sorted order,
    to the split with fewer S and M candidates so far."""
    fixed = {k: s for s, keys in SPLIT_REPOS.items() for k in keys}
    counts = collections.Counter()
    per_repo = collections.Counter()
    for family, rs in recs.items():
        for r in rs:
            if size_of(family, r):
                per_repo[repo_key(r)] += 1
    for k, s in fixed.items():
        counts[s] += per_repo.get(k, 0)
    out = dict(fixed)
    for k in sorted(per_repo):
        if k in out:
            continue
        s = min(SPLITS, key=lambda sp: (counts[sp], SPLITS.index(sp)))
        out[k] = s
        counts[s] += per_repo[k]
    return out


def apply_patch(folder: Path, patch: str) -> None:
    from rlm_rl.editbox import git, git_env

    if not patch:
        return
    scratch = folder.parent / f"{folder.name}.git-{os.getpid()}-{threading.get_ident()}"
    env = git_env(folder.parent)
    try:
        git("init", "-q", "--bare", str(scratch), env=env)
        out = git("apply", "--whitespace=nowarn", "-", env=dict(env, GIT_DIR=str(scratch), GIT_WORK_TREE=str(folder)),
                  cwd=folder, check=False, input=patch.encode("utf-8"))
        if out.returncode:
            raise RuntimeError(f"the gold patch doesn't apply: {out.stderr.decode()[-300:]}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def python_cand(record: dict, family: str, split: str, size: str, extra_files: dict[str, bytes] | None = None,
                task: str | None = None, cover: dict | None = None) -> Cand:
    t = record["tests"]
    hidden = {p: txt.encode("utf-8") for p, txt in sorted((t.get("hidden") or {}).items())}
    extra_files = extra_files or {}

    def base(folder: Path) -> None:
        editreal._build_workspace(record, folder)
        put_files(folder, extra_files)

    def gold(folder: Path) -> None:
        apply_patch(folder, record.get("gold_patch") or "")
        for rel in extra_files:  # REQUIREMENTS.md goes with the gold (python L)
            (folder / rel).unlink(missing_ok=True)

    nfiles, nlines = editdata.patch_size(record.get("gold_patch") or "")
    text = task or record["issue"]
    return Cand(key=cand_key("python", record["id"], record.get("gold_patch"), record.get("break_patch"), size, text,
                             {p: sha256(b) for p, b in hidden.items()}, t.get("f2p"), t.get("p2p"), t.get("scope"),
                             {p: sha256(b) for p, b in extra_files.items()}),
                track="python", split=split, size=size,
                source={"family": family, "record_id": record["id"], "repo": repo_key(record),
                        "level": record["level"]},
                task=text, runner="pytest", env="", files=sorted(t.get("scope") or hidden),
                base=base, hidden=hidden, gold=gold, f2p=list(t.get("f2p") or []), p2p=list(t.get("p2p") or []),
                leak_texts=[text, *[b.decode("utf-8") for b in extra_files.values()]],
                gold_patch=record.get("gold_patch") or "", cover=cover or {}, gold_files=nfiles, gold_lines=nlines)


def python_sm_candidates(recs: dict[str, list[dict]], splits: dict[str, str]) -> dict[tuple, dict[str, list[Cand]]]:
    """(split, size) -> family -> candidates, families in PYTHON_FAMILIES order."""
    out: dict[tuple, dict[str, list[Cand]]] = {}
    for size, fams in PYTHON_FAMILIES.items():
        for split in SPLITS:
            out[(split, size)] = {}
            for family, _ in fams:
                out[(split, size)][family] = [python_cand(r, family, split, size) for r in
                                              sorted(recs[family], key=lambda r: r["id"])
                                              if size_of(family, r) == size and splits.get(repo_key(r)) == split]
    return out


# ---- python L: edit_large (D-181)


def large_clusters(ctx, rng: random.Random, want: int = LARGE["per_repo"]) -> list[list[str]]:
    """Up to `want` module clusters of one repository, no two sharing a module (the spec's edit_large rules)."""
    repo = ctx.repo
    cand = editspec.eligible(repo, ctx.tests_of, ctx.at_import)
    by_mod: dict[str, list[str]] = collections.defaultdict(list)
    for k in cand:
        by_mod[repo.fns[k].path].append(k)
    qualifying = {m: sorted(ks) for m, ks in by_mod.items() if len(ks) >= LARGE["min_fns"]}
    if len(qualifying) < LARGE["min_modules"]:
        return []
    static, _ = editspec.graph(repo, cand, ctx.tests_of)
    seeds = sorted(qualifying)
    rng.shuffle(seeds)
    used: set[str] = set()
    out = []
    for seed in seeds:
        if len(out) >= want:
            break
        if seed in used:
            continue
        chosen = [seed]
        lines = sum(repo.fns[k].body_lines for k in qualifying[seed])
        while len(chosen) < LARGE["max_modules"] and not (len(chosen) >= LARGE["min_modules"]
                                                           and lines >= LARGE["min_lines"]):
            members = {k for m in chosen for k in qualifying[m]}
            options = []
            for m in sorted(qualifying):
                if m in chosen or m in used:
                    continue
                edges = sum(len(static[k] & members) for k in qualifying[m])
                options.append((-edges, m))
            if not options:
                break
            m = min(options)[1]
            chosen.append(m)
            lines += sum(repo.fns[k].body_lines for k in qualifying[m])
        cluster = sorted((k for m in chosen for k in qualifying[m]), key=lambda k: (repo.fns[k].path, repo.fns[k].def_line))
        hidden = editspec.test_files_of(cluster, ctx.tests_of)
        if (len(chosen) < LARGE["min_modules"] or not LARGE["min_lines"] <= lines <= LARGE["max_lines"]
                or len(hidden) > LARGE["max_hidden"] or not any(repo.fns[k].doc for k in cluster)):
            continue
        used.update(chosen)
        out.append(cluster)
    return out


def large_candidates(seed: int, split: str, entries: dict) -> list[Cand]:
    """edit_large candidates of one split, built from build_edit_tasks_v2.Context (no Docker lab needed)."""
    import build_edit_tasks_v2 as v2

    out = []
    for key in SPLIT_REPOS[split]:
        if key not in entries:
            log(f"  python L: {key} has no registry entry; skipped")
            continue
        try:
            ctx = v2.Context(key, entries[key])
        except SystemExit as e:
            log(f"  python L: {key}: {e}; skipped")
            continue
        rng = random.Random(f"{seed}-{split}/python/L-{key}")
        for i, cluster in enumerate(large_clusters(ctx, rng)):
            fns = [ctx.repo.fns[k] for k in cluster]
            by_file = collections.defaultdict(list)
            for fn in fns:
                by_file[fn.path].append(fn)
            try:
                changes = v2.changed_texts(ctx, by_file, editspec.stub_text)
            except (ValueError, SyntaxError) as e:  # mixed line ends, or a stub that doesn't parse
                log(f"  python L {split}: {key} cluster {i} skipped: {e}")
                continue
            hidden = sorted(editspec.test_files_of(cluster, ctx.tests_of))
            tests = set().union(*(ctx.tests_of.get(k, set()) for k in cluster))
            f2p = sorted(tests)
            p2p = sorted(t for t in ctx.passed if editspec.test_file(t) in hidden and t not in tests)
            ident = f"large-{key}-{i}-" + sha256(json.dumps(cluster).encode())[:8]
            rec = v2.base_record(ctx, "large", "L", ident, changes, hidden, f2p, p2p, LARGE_TEXT, cluster, {})
            rec = dict(rec, split=split, kind="large", generator="evalset-large-v1")
            reqs = editspec.requirements(entries[key]["repo"], ctx.repo, cluster)
            cover = {k: ctx.keys(sorted(ctx.tests_of.get(k, set()))) for k in cluster}
            lines = sum(f.body_lines for f in fns)
            log(f"  python L {split}: {key} cluster {i}: {len({f.path for f in fns})} modules, {len(fns)} functions, "
                f"{lines} body lines, {len(hidden)} hidden files")
            out.append(python_cand(rec, "edit_large", split, "L", {"REQUIREMENTS.md": reqs.encode("utf-8")},
                                   LARGE_TEXT, cover))
    return out


# ---- web (D-222)


def lock_importers(lock_text: str) -> dict[str, dict[str, str]]:
    """Rush's common lockfile (lockfileVersion 6): importer path -> {package: resolved version, peer suffix dropped}."""
    out: dict[str, dict[str, str]] = {}
    section = cur = pkg = None
    for line in lock_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        text = line.strip()
        if indent == 0:
            section = text.rstrip(":")
            continue
        if section != "importers":
            continue
        if indent == 2:
            cur = text.rstrip(":").strip("'\"")
            out[cur] = {}
        elif indent == 6 and cur is not None:
            pkg = text.rstrip(":").strip("'\"")
        elif indent == 8 and cur is not None and pkg and text.startswith("version:"):
            out[cur][pkg] = re.sub(r"\(.*$", "", text.split(":", 1)[1].strip().strip("'\""))
    return out


REQUIRE_RE = re.compile(r"""(?:require\(\s*|from\s+|import\s+)['"]([^'"]+)['"]""")


def requires_of(text: str) -> list[str]:
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    return REQUIRE_RE.findall(body)


def web_skip_reason(proj: Path, name: str) -> str:
    if name in WEB_VARIANTS:
        return "a variant kept out of both splits"
    pkg = json.loads((proj / "package.json").read_text(encoding="utf-8"))
    if (pkg.get("eval") or {}).get("stable") is not True:
        return "not eval.stable"
    ids = {json.loads(line).get("id") for line in (proj / "tasks.jsonl").read_text(encoding="utf-8").split("\n")
           if line.strip()} if (proj / "tasks.jsonl").exists() else set()
    have = sum(1 for i in range(1, max(WEB_REQUESTS.values()) + 1) if f"task-{i}" in ids)
    if have < max(WEB_REQUESTS.values()):
        return f"only {have} requests"
    scripts = pkg.get("scripts") or {}
    if "scripts/dev.js" in (scripts.get("start") or ""):
        return "runs a server of its own (scripts/dev.js)"
    deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
    native = sorted(d for d in deps if d in WEB_SERVER_PACKAGES)
    if native:
        return f"depends on a database or native package ({native[0]})"
    for f in [proj / "playwright.config.js", *sorted((proj / "test").rglob("*.js"))]:
        if not f.exists():
            continue
        for req in requires_of(f.read_text(encoding="utf-8")):
            if req.startswith("."):
                target = (f.parent / req).resolve()
                if not _inside(target, proj):
                    return f"requires a path outside the project ({req})"
    return ""


def web_requests(proj: Path, n: int) -> list[str]:
    """Requests task-1 .. task-n by id: most projects start with an `init` line that src-init/ already satisfies."""
    rows = {r.get("id"): r for r in (json.loads(line) for line in
                                     (proj / "tasks.jsonl").read_text(encoding="utf-8").split("\n") if line.strip())}
    missing = [i for i in range(1, n + 1) if f"task-{i}" not in rows]
    if missing:
        raise SetupError(f"{proj.name}: tasks.jsonl has no task-{missing[0]}")
    return [rows[f"task-{i}"]["description"].strip() for i in range(1, n + 1)]


def spec_titles(texts: list[str]) -> list[str]:
    titles = []
    for t in texts:
        titles += re.findall(r"""\btest(?:\.only)?\(\s*(['"`])(.+?)\1""", t)
    return [x[1] for x in titles]


class WebBench:
    def __init__(self, cache: Path):
        self.dir = cache / "web-bench"

    def ensure(self) -> None:
        url, pin = WEBBENCH
        if not (self.dir / ".git").exists():
            self.dir.parent.mkdir(parents=True, exist_ok=True)
            code, out, _ = run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "clone", "-q",
                                "--filter=blob:none", "--no-checkout", url, str(self.dir)], timeout=1800)
            subprocess.run(["git", "-C", str(self.dir), "config", "core.autocrlf", "false"], capture_output=True)
            if code:
                raise SetupError(f"can't clone Web-Bench: {out[-300:]}")
        head = subprocess.run(["git", "-C", str(self.dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        if head.strip() != pin or not (self.dir / "rush.json").exists():  # a --no-checkout clone has no files yet
            code, out, _ = run(["git", "-C", str(self.dir), "-c", "advice.detachedHead=false", "-c", "core.autocrlf=false",
                                "-c", "core.eol=lf", "checkout", "-q", pin], timeout=1800)
            if code:
                raise SetupError(f"can't check out Web-Bench {pin}: {out[-300:]}")

    def importers(self) -> dict[str, dict[str, str]]:
        return lock_importers((self.dir / "common" / "config" / "rush" / "pnpm-lock.yaml").read_text(encoding="utf-8"))


def standalone(wb: Path, project: str, size: str, importers: dict) -> tuple[dict[str, bytes], dict[str, bytes],
                                                                              dict[str, bytes], list[str]]:
    """(base files, hidden files, gold src/ files, the requests) of one project as a standalone pnpm project, before
    its lockfile. Paths are POSIX, relative to the project."""
    proj = wb / "projects" / project
    n = WEB_REQUESTS[size]
    skip_top = {"src", "src-init", "test", "tasks.jsonl", "readme.md", "node_modules"}
    base: dict[str, bytes] = {}
    for p in sorted(proj.rglob("*")):
        rel = p.relative_to(proj).as_posix()
        if rel.split("/")[0] in skip_top or not p.is_file():
            continue
        base[rel] = p.read_bytes()
    for p in sorted((proj / "src-init").rglob("*")):
        if p.is_file():
            base["src/" + p.relative_to(proj / "src-init").as_posix()] = p.read_bytes()
    lib = wb / "libraries" / "test-util"
    for p in sorted(lib.rglob("*")):
        rel = p.relative_to(lib).as_posix()
        if p.is_file() and "node_modules" not in rel.split("/"):
            base["vendor/test-util/" + rel] = p.read_bytes()
    base["LICENSE.md"] = (wb / "LICENSE.md").read_bytes()
    base["NOTICE"] = (f"This project is adapted from Web-Bench (https://github.com/bytedance/web-bench), commit "
                      f"{WEBBENCH[1]}, project {project}, licensed under the Apache License 2.0 (LICENSE.md).\n"
                      ).encode("utf-8")
    hidden: dict[str, bytes] = {}
    for i in range(1, n + 1):
        spec = proj / "test" / f"task-{i}.spec.js"
        if spec.exists():
            hidden[f"test/task-{i}.spec.js"] = spec.read_bytes()
    if any(r.startswith("./util") or r.startswith("../util") for t in hidden.values()
           for r in requires_of(t.decode("utf-8"))) and (proj / "test" / "util").is_dir():
        for p in sorted((proj / "test" / "util").rglob("*")):
            if p.is_file():
                hidden["test/util/" + p.relative_to(proj / "test" / "util").as_posix()] = p.read_bytes()
    gold = {"src/" + p.relative_to(proj / "src").as_posix(): p.read_bytes()
            for p in sorted((proj / "src").rglob("*")) if p.is_file()}
    pkg = json.loads(base["package.json"])
    pins = importers.get(f"../../projects/{project}", {})
    root_pins = importers.get("../../common/temp/default", {}) | importers.get(".", {})
    for section in ("dependencies", "devDependencies"):
        for name, spec in list((pkg.get(section) or {}).items()):
            if name == "@web-bench/test-util":
                pkg[section][name] = "file:vendor/test-util"
            elif name in pins:
                pkg[section][name] = pins[name]
            elif str(spec).startswith("workspace:"):
                raise SetupError(f"{project}: {name} is a workspace package with no pin")
    declared = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
    texts = [base.get("playwright.config.js", b"").decode("utf-8"), *[b.decode("utf-8") for b in hidden.values()]]
    for req in sorted({r for t in texts for r in requires_of(t)}):
        if req.startswith(".") or req.startswith("node:") or req in sys_node_modules():
            continue
        name = "/".join(req.split("/")[:2]) if req.startswith("@") else req.split("/")[0]
        if name not in declared:
            version = pins.get(name) or root_pins.get(name)
            if not version:
                raise SetupError(f"{project}: the config or specs require {name}, which nothing pins")
            pkg.setdefault("devDependencies", {})[name] = version
            declared[name] = version
    pkg.setdefault("scripts", {})["test"] = "playwright test"
    built = sorted(n for n in WEB_BUILT if n in declared or n == "esbuild")
    pkg.setdefault("pnpm", {})["onlyBuiltDependencies"] = built
    base["package.json"] = (json.dumps(pkg, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    base[".gitignore"] = b"node_modules/\ntest-results/\nplaywright-report/\n"
    return base, hidden, gold, web_requests(proj, n)


def sys_node_modules() -> set[str]:
    return {"fs", "path", "os", "url", "child_process", "http", "https", "http2", "util", "crypto", "stream", "events",
            "assert", "buffer", "net", "zlib", "readline", "process", "module", "worker_threads", "perf_hooks",
            "constants", "timers", "dns", "tls", "dgram", "vm", "querystring", "string_decoder", "punycode", "tty",
            "v8", "cluster", "inspector", "async_hooks", "diagnostics_channel", "console", "fs/promises"}


def features_md(requests: list[str]) -> str:
    return "# Feature requests\n\n" + "\n\n".join(f"{i}. {r}" for i, r in enumerate(requests, 1)) + "\n"


def web_task_text(size: str, requests: list[str]) -> tuple[str, dict[str, bytes]]:
    if size == "S":
        listing = "\n".join(f"{i}. {r}" for i, r in enumerate(requests, 1))
        return f"{WEB_TEXT_S}\n\n{listing}", {}
    return WEB_TEXT, {"FEATURES.md": features_md(requests).encode("utf-8")}


class Pnpm:
    """The pinned pnpm, the shared store and Playwright's Chromium in the build cache."""

    def __init__(self, cache: Path):
        self.cache = cache
        self.store = cache / "pnpm-store"
        self.browsers = cache / "ms-playwright"
        self.exe = shutil.which("pnpm") or "pnpm"

    def check(self) -> None:
        node = shutil.which("node")
        if not node:
            raise SetupError("node 22 or later is not on PATH (installing it is Rishi's call)")
        code, out, _ = run([node, "--version"])
        if code or int(out.strip().lstrip("v").split(".")[0]) < 22:
            raise SetupError(f"node {out.strip()} is older than 22")
        code, out, _ = run([self.exe, "--version"])
        if code or out.strip() != PNPM:
            raise SetupError(f"pnpm {out.strip() or 'missing'} on PATH, but the set pins {PNPM} (installing it is "
                             "Rishi's call)")

    def env(self, extra: dict | None = None) -> dict:
        return clean_env({"PLAYWRIGHT_BROWSERS_PATH": str(self.browsers), "CI": "", **(extra or {})},
                         drop=("CI", "EVAL_PROJECT_ROOT", "npm_config_store_dir"))

    def cmd(self, folder: Path, *args: str, timeout: float = 1800, extra: dict | None = None):
        env = self.env(extra)
        env.pop("CI", None)
        return run([self.exe, *args], cwd=folder, env=env, timeout=timeout)

    def lockfile(self, folder: Path) -> bytes:
        code, out, _ = self.cmd(folder, "install", "--lockfile-only", "--ignore-scripts", "--store-dir", str(self.store))
        if code:
            raise RuntimeError(f"pnpm install --lockfile-only failed: {out[-500:]}")
        return (folder / "pnpm-lock.yaml").read_bytes()

    def fetch(self, folder: Path) -> None:
        code, out, _ = self.cmd(folder, "fetch", "--store-dir", str(self.store))
        if code:
            raise RuntimeError(f"pnpm fetch failed: {out[-500:]}")

    def install(self, folder: Path) -> tuple[int, str]:
        code, out, _ = self.cmd(folder, "install", "--offline", "--frozen-lockfile", "--store-dir", str(self.store))
        return code, out

    def browser(self, folder: Path, version: str) -> None:
        marker = self.browsers / f".rlmh-{version}"
        if marker.exists():
            return
        code, out, _ = self.cmd(folder, "exec", "playwright", "install", "chromium", timeout=3600)
        if code:
            raise RuntimeError(f"playwright install chromium failed: {out[-500:]}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"")


def web_cand(wb: WebBench, cache: Path, project: str, split: str, size: str, importers: dict, pn: Pnpm,
             nodeenvs: dict, validate: bool) -> Cand:
    base, hidden, gold_src, requests = standalone(wb.dir, project, size, importers)
    text, extra = web_task_text(size, requests)
    base.update(extra)
    folder = cache / "web" / f"{project}-{size}"
    lock_path = folder / "pnpm-lock.yaml"
    if not lock_path.exists() or tree_files(folder).get("package.json") != base["package.json"]:
        rmtree(folder)
        folder.mkdir(parents=True)
        put_files(folder, base)
        if validate:
            pn.lockfile(folder)
    if lock_path.exists():
        base["pnpm-lock.yaml"] = lock_path.read_bytes()
    pkg = json.loads(base["package.json"])
    pw = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}.get("@playwright/test")
    key = "web-" + sha256(base.get("pnpm-lock.yaml") or base["package.json"])[:12]
    nodeenvs[key] = {"node": "22", "pnpm": PNPM, "playwright": pw}

    def write_base(target: Path) -> None:
        put_files(target, base)

    def gold(target: Path) -> None:
        rmtree(target / "src")
        put_files(target, gold_src)

    files = sorted(p for p in hidden if p.endswith(".spec.js"))
    return Cand(key=cand_key("web", project, size, sha256(json.dumps({k: sha256(v) for k, v in base.items()},
                                                                      sort_keys=True).encode())),
                track="web", split=split, size=size, source={"family": "webbench", "project": project, "level": size},
                task=text, runner="playwright", env=key, files=files, base=write_base, hidden=hidden, gold=gold,
                leak_texts=[text, *[b.decode("utf-8") for b in extra.values()]],
                spec_titles=spec_titles([hidden[f].decode("utf-8") for f in files]),
                gold_files=len(gold_src), gold_lines=sum(b.count(b"\n") for b in gold_src.values()))


# ---- kernel (D-223)


def kernel_files(folder: Path) -> dict[str, bytes]:
    return tree_files(folder) if folder.is_dir() else {}


def kernel_cand(cache: Path, op: str) -> Cand:
    src = cache / "kernels" / op
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    base = kernel_files(src / "base")
    shared = kernel_files(cache / "kernels" / "_shared")
    for name in ("gpu.py", "gpu_compat.h"):
        if base.get(name) != shared.get(name):
            raise SetupError(f"kernel {op}: base/{name} differs from kernels/_shared/{name}")
    hidden = {"tests/" + k: v for k, v in kernel_files(src / "hidden" / "tests").items()}
    gold_files = kernel_files(src / "gold")
    mutants = []
    for d in sorted((src / "mutants").iterdir(), key=lambda p: p.name) if (src / "mutants").is_dir() else []:
        mfiles = kernel_files(d)
        mutants.append(lambda target, m=mfiles: put_files(target, m))
    changes = {rel: (base.get(rel, b"").decode("utf-8"), b.decode("utf-8")) for rel, b in gold_files.items()}
    gold_patch = file_patch(changes)
    stubs = [b.decode("utf-8") for rel, b in base.items() if rel.startswith("kernels/") and rel.endswith(".gpu.cu")]
    nfiles, nlines = editdata.patch_size(gold_patch)
    return Cand(key=cand_key("kernel", op, {k: sha256(v) for k, v in {**base, **hidden, **gold_files}.items()},
                             [sha256(json.dumps({k: sha256(v) for k, v in kernel_files(d).items()}).encode())
                              for d in sorted((src / "mutants").iterdir())] if (src / "mutants").is_dir() else []),
                track="kernel", split=meta["split"], size=meta["size"], source={"family": "kernel", "op": op,
                                                                                "level": meta["size"]},
                task=KERNEL_TEXT, runner="pytest", env="", files=sorted(hidden),
                base=lambda target: put_files(target, base), hidden=hidden,
                gold=lambda target: put_files(target, gold_files), mutants=mutants,
                leak_texts=[KERNEL_TEXT, base.get("REQUIREMENTS.md", b"").decode("utf-8"), *stubs],
                gold_patch=gold_patch, gold_files=nfiles, gold_lines=nlines)


def file_patch(changes: dict[str, tuple[str, str]]) -> str:
    import build_edit_tasks_v2 as v2

    return v2.file_patch(changes)


def copy_shared(paths: Paths) -> None:
    """Task 9: step 2's gpu.py and gpu_compat.h from the harness checkout into kernels/_shared/."""
    src = paths.harness / "app" / "test" / "fixtures" / "evalset" / "gpu"
    dst = paths.cache / "kernels" / "_shared"
    for name in ("gpu.py", "gpu_compat.h"):
        if not (src / name).exists():
            raise SetupError(f"{src / name} is missing: step 2 owns the GPU shim")
        dst.mkdir(parents=True, exist_ok=True)
        if not (dst / name).exists() or (dst / name).read_bytes() != (src / name).read_bytes():
            shutil.copyfile(src / name, dst / name)


# ----------------------------------------------------------------------------- environments


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def uv() -> str:
    found = shutil.which("uv")
    if not found:
        raise SetupError("uv is not on PATH")
    return found


def env_key(prefix: str, spec: dict) -> str:
    return f"{prefix}-" + sha256(json.dumps(spec, sort_keys=True).encode())[:12]


class Envs:
    """Build venvs, made once from their frozen spec ({python, requirements, import_root}), in <cache>/envs/<key>/."""

    def __init__(self, cache: Path):
        self.dir = cache / "envs"
        self.specs: dict[str, dict] = {}
        self.locks: dict[str, threading.Lock] = collections.defaultdict(threading.Lock)
        self.made: set[str] = set()
        self.log_path = self.dir / "resolved.json"

    def add(self, prefix: str, spec: dict) -> str:
        key = env_key(prefix, spec)
        self.specs[key] = spec
        return key

    def python(self, key: str) -> Path:
        with self.locks["make-" + key]:
            venv = self.dir / key
            py = venv_python(venv)
            if key in self.made or (py.exists() and (venv / ".rlmh-ok").exists()):
                self.made.add(key)
                return py
            rmtree(venv)
            spec = self.specs[key]
            code, out, _ = run([uv(), "venv", "-q", "--python", spec["python"], str(venv)], env=clean_env(), timeout=1800)
            if code:
                raise RuntimeError(f"uv venv {spec['python']} failed: {out[-300:]}")
            reqs = venv / "requirements.txt"
            reqs.write_bytes("".join(r + "\n" for r in spec["requirements"]).encode("utf-8"))
            code, out, _ = run([uv(), "pip", "install", "-q", "--python", str(py), "-r", str(reqs)], env=clean_env(),
                               timeout=3600)
            if code:
                raise RuntimeError(f"uv pip install failed: {out[-500:]}")
            (venv / ".rlmh-ok").write_bytes(b"")
            self.made.add(key)
            return py

    @contextlib.contextmanager
    def src_root(self, key: str, work: Path):
        """For import_root "src": a .pth naming the work copy's src/ for this run only (D-185); one run at a time."""
        spec = self.specs[key]
        if spec.get("import_root") != "src":
            yield
            return
        py = self.python(key)
        with self.locks["src-" + key]:
            code, out, _ = run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               stdout_only=True)
            pth = Path(out.strip().splitlines()[-1]) / "rlmh_src.pth"
            pth.write_bytes((str((work / "src").resolve()) + "\n").encode("utf-8"))
            try:
                yield
            finally:
                pth.unlink(missing_ok=True)


def _req_name(line: str) -> str:
    m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line)
    return re.sub(r"[-_.]+", "-", m.group(1)).lower() if m else ""


def _clean_reqs(lines, own: str) -> list[str]:
    out = []
    for line in lines:
        line = str(line).split("#", 1)[0].strip()
        if not line or line.startswith(("-", ".", "{", "/")) or "{" in line or "://" in line or line.startswith("git+"):
            continue
        if re.match(r"^[\w.,!-]+:\s", line):  # a tox factor condition ("py312: pytest-asyncio")
            continue
        if _req_name(line) and _req_name(line) != own:
            out.append(line)
    return out


def _setup_py_kwargs(text: str) -> dict:
    """Literal install_requires and extras_require from a setup() call; non-literal values are skipped."""
    out = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "setup":
            for kw in node.keywords:
                if kw.arg in ("install_requires", "extras_require", "name"):
                    try:
                        out[kw.arg] = ast.literal_eval(kw.value)
                    except ValueError:
                        pass
    return out


def project_requirements(tree: Path) -> tuple[list[str], list[str], str]:
    """(test requirements, runtime requirements, the package's own normalized name) from the snapshot's metadata."""
    test: list = []
    runtime: list = []
    own = ""
    pp = tree / "pyproject.toml"
    if pp.exists():
        try:
            meta = tomllib.loads(pp.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            meta = {}
        proj = meta.get("project") or {}
        own = own or _req_name(proj.get("name") or "")
        extras = proj.get("optional-dependencies") or {}
        groups = meta.get("dependency-groups") or {}
        for name in TEST_EXTRAS:
            test += [x for x in extras.get(name, []) if isinstance(x, str)]
            test += [x for x in groups.get(name, []) if isinstance(x, str)]
        runtime += proj.get("dependencies") or []
    cfg = tree / "setup.cfg"
    if cfg.exists():
        cp = configparser.ConfigParser(interpolation=None)
        try:
            cp.read_string(cfg.read_text(encoding="utf-8"))
        except (configparser.Error, UnicodeDecodeError):
            cp = configparser.ConfigParser(interpolation=None)
        own = own or _req_name(cp.get("metadata", "name", fallback=""))
        for name in TEST_EXTRAS:
            test += cp.get("options.extras_require", name, fallback="").splitlines()
        runtime += cp.get("options", "install_requires", fallback="").splitlines()
    sp = tree / "setup.py"
    if sp.exists():
        kw = _setup_py_kwargs(sp.read_text(encoding="utf-8", errors="replace"))
        own = own or _req_name(str(kw.get("name") or ""))
        extras = kw.get("extras_require") if isinstance(kw.get("extras_require"), dict) else {}
        for name in TEST_EXTRAS:
            v = extras.get(name) or []
            test += [v] if isinstance(v, str) else list(v)
        v = kw.get("install_requires") or []
        runtime += [v] if isinstance(v, str) else list(v)
    for pattern in ("requirements*test*.txt", "requirements-dev.txt"):
        for f in sorted(tree.glob(pattern)):
            test += f.read_text(encoding="utf-8", errors="replace").splitlines()
    tox = tree / "tox.ini"
    if tox.exists():
        cp = configparser.ConfigParser(interpolation=None)
        try:
            cp.read_string(tox.read_text(encoding="utf-8"))
            test += cp.get("testenv", "deps", fallback="").splitlines()
        except (configparser.Error, UnicodeDecodeError):
            pass
    test, runtime = _clean_reqs(test, own), _clean_reqs(runtime, own)
    if not any(_req_name(t) == "pytest" for t in test):
        test.append("pytest")
    dedup = lambda xs: list(dict.fromkeys(xs))  # noqa: E731
    return dedup(test), dedup(runtime), own


def import_root(tree: Path) -> str:
    """"src" when the package's import directory sits under src/ (D-185)."""
    src_pkgs = [p.parent.name for p in tree.glob("src/*/__init__.py")]
    return "src" if src_pkgs and not any((tree / name / "__init__.py").exists() for name in src_pkgs) else "."


def commit_date(entry: dict, tree: Path) -> tuple[str, bool]:
    """(the commit's date as YYYY-MM-DD, approximate?): the registry's date, else the newest member mtime of the
    snapshot tar (never the work tree: applying the break patch stamps files with today), else of the tree."""
    for k in ("commit_date", "date", "created_at", "committed"):
        if entry.get(k):
            return str(entry[k])[:10], False
    newest = 0.0
    snap = data.path(entry["snapshot"]) if entry.get("snapshot") else None
    if snap and snap.exists():
        with tarfile.open(snap) as t:
            newest = max((m.mtime for m in t.getmembers() if m.isfile()), default=0)
    if not newest:
        newest = max((p.stat().st_mtime for p in tree.rglob("*") if p.is_file()), default=time.time())
    return datetime.datetime.fromtimestamp(newest, datetime.timezone.utc).strftime("%Y-%m-%d"), True


RESOLVED_LOCK = threading.Lock()
DOCKER = (["docker"], ["wsl", "-d", "Ubuntu", "docker"])


def _resolved(cache: Path) -> dict:
    path = cache / "envs" / "resolved.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def docker_freeze(image: str, own: str) -> tuple[str, list[str], str]:
    """(the image's Python as X.Y, its frozen packages, why it failed) from `pip freeze` in the image, through Docker
    directly or through WSL's Ubuntu."""
    why = ""
    for docker in DOCKER:
        script = ("import sys, subprocess; print('%d.%d' % sys.version_info[:2]); "
                  "sys.stdout.flush(); subprocess.run([sys.executable, '-m', 'pip', 'freeze'])")
        code, out, _ = run([*docker, "run", "--rm", image, "python", "-c", script], timeout=1800, stdout_only=True)
        lines = out.split("\n")
        if code == 0 and lines and re.match(r"^3\.\d+$", lines[0].strip()):
            frozen = [x.strip() for x in lines[1:] if "==" in x and _req_name(x) != own and " @ " not in x]
            return lines[0].strip(), frozen, ""
        why += f" | {' '.join(docker)}: exit {code}"
    return "", [], why


def resolve_python_env(cache: Path, key: str, entry: dict, tree: Path, python: str) -> dict:
    """Resolve a repository's test venv at the commit's date under one Python, then freeze it. Falls back to the
    Docker image's own freeze (and its own Python) when resolution fails. Cached in envs/resolved.json, which every
    thread reads and writes under one lock."""
    rk = f"{key}@{python}"
    with RESOLVED_LOCK:
        done = _resolved(cache)
    if rk in done:
        return done[rk]
    test, runtime, own = project_requirements(tree)
    date, approx = commit_date(entry, tree)
    venv = cache / "envs" / "resolve" / rk.replace("@", "-")
    rmtree(venv)
    method, frozen, why, py = "uv", [], "", python
    code, out, _ = run([uv(), "venv", "-q", "--python", python, str(venv)], env=clean_env(), timeout=1800)
    if code == 0:
        reqs = venv / "requirements.in"
        reqs.write_bytes("".join(r + "\n" for r in test + runtime).encode("utf-8"))
        code, out, _ = run([uv(), "pip", "install", "-q", "--python", str(venv_python(venv)), "--exclude-newer",
                            f"{date}T23:59:59Z", "-r", str(reqs)], env=clean_env(), timeout=3600)
    if code == 0:
        code, out, _ = run([uv(), "pip", "freeze", "--python", str(venv_python(venv))], env=clean_env(),
                           stdout_only=True)
        frozen = [x for x in out.split("\n") if "==" in x and _req_name(x) != own and " @ " not in x]
    else:
        why = out[-300:]
        py, frozen, dwhy = docker_freeze(entry.get("tag") or entry.get("image") or "", own)
        method = "docker" if py else "failed"
        why += dwhy
    rmtree(venv)
    result = {"method": method, "python": py, "date": date, "date_approx": approx, "why": why.strip(),
              "spec": {"python": py, "requirements": frozen, "import_root": import_root(tree)}}
    with RESOLVED_LOCK:
        done = _resolved(cache)
        done[rk] = result
        write_json(cache / "envs" / "resolved.json", done)
    log(f"  venv {key} @ {python}: {method}{' (date approximate)' if approx else ''}, {len(frozen)} pins")
    return result


def kernel_env_spec(cache: Path) -> dict:
    """NVRTC plus numpy and pytest at their newest releases at least two weeks old on build day, frozen once."""
    path = cache / "envs" / "kernel.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    day = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=14)).strftime("%Y-%m-%dT00:00:00Z")
    venv = cache / "envs" / "resolve" / "kernel"
    rmtree(venv)
    code, out, _ = run([uv(), "venv", "-q", "--python", "3.12", str(venv)], env=clean_env())
    if code == 0:
        code, out, _ = run([uv(), "pip", "install", "-q", "--python", str(venv_python(venv)), "--exclude-newer", day,
                            "--only-binary", ":all:", "numpy", "pytest"], env=clean_env(), timeout=1800)
    if code:
        raise SetupError(f"can't resolve the kernel venv: {out[-300:]}")
    code, out, _ = run([uv(), "pip", "freeze", "--python", str(venv_python(venv))], env=clean_env(), stdout_only=True)
    rmtree(venv)
    spec = {"python": "3.12", "requirements": sorted([x for x in out.split("\n") if "==" in x]) + [NVRTC],
            "import_root": "."}
    write_json(path, spec)
    return spec


# ----------------------------------------------------------------------------- validation (D-183, D-184)


@dataclass
class Verdict:
    kept: bool
    why: str = ""
    f2p: list = field(default_factory=list)
    p2p: list = field(default_factory=list)
    dropped_p2p: list = field(default_factory=list)
    gold_seconds: float = 0.0
    mutants_caught: int | None = None
    timeout_s: int = 120
    python: str | None = None


def junit_results(path: Path, runner: str) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        cases = grader._junit_cases(path.read_text(encoding="utf-8", errors="replace"))
    except ET.ParseError:
        return {}
    out: dict[str, str] = {}
    for case, status in cases:
        if runner == "pytest":
            key = grader.pytest_key(case.get("file"), case.get("classname") or "", case.get("name") or "")
        else:
            key = playwright_key(case.get("classname") or "", case.get("name") or "")
        if key:
            grader._merge(out, key, status)
    return out


def playwright_key(classname: str, name: str) -> str | None:
    """D-184: the testcase's classname (the spec file, `/` separators), `::`, its name."""
    return f"{classname.replace(chr(92), '/')}::{name}" if name else None


class Runner:
    """Runs one candidate's tests in a fresh work folder, exactly as rlmh eval grades (architecture §6.4)."""

    def __init__(self, cache: Path, envs: Envs, pn: Pnpm | None):
        self.cache, self.envs, self.pn = cache, envs, pn
        self.work = cache / "work"
        self.gpu = threading.Lock()

    @contextlib.contextmanager
    def folder(self):
        self.work.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="v-", dir=self.work))
        try:
            yield tmp
        finally:
            try:
                rmtree(tmp)
            except OSError as e:  # a file still held (antivirus, an indexer): leave it, never lose the verdict
                log(f"    note: could not remove {tmp}: {e}")

    def tree(self, c: Cand, tmp: Path, gold: bool, hidden: bool = True, mutant: int | None = None) -> Path:
        work = tmp / "repo"
        rmtree(work)
        work.mkdir(parents=True)
        put_files(work, c.base_files())  # the bytes the tar ships, not the builder's tree
        if gold:
            c.gold(work)
        if mutant is not None:
            c.gold(work)
            c.mutants[mutant](work)
        if hidden:
            put_files(work, c.hidden)
        return work

    def pytest(self, c: Cand, work: Path, tmp: Path, files: list[str], timeout: float = 1800):
        py = self.envs.python(c.env)
        junit = tmp / "junit.xml"
        junit.unlink(missing_ok=True)
        argv = [str(py), "-c", grader.PYTEST_BOOT_TEMPLATE % str(work.resolve()), "-p", "no:cacheprovider",
                "-o", "junit_family=xunit1", f"--junitxml={junit}", "--tb=short", "-q", *files]
        with self.envs.src_root(c.env, work):
            code, out, secs = run(argv, cwd=work, env=clean_env({"PYTHONDONTWRITEBYTECODE": "1"}), timeout=timeout)
        return junit_results(junit, "pytest"), code, out, secs

    def playwright(self, c: Cand, work: Path, tmp: Path, files: list[str], timeout: float = 1800):
        code, out = self.pn.install(work)
        if code:
            return {}, code, "pnpm install: " + out, 0.0
        junit = tmp / "junit.xml"
        junit.unlink(missing_ok=True)
        extra = {"PLAYWRIGHT_JUNIT_OUTPUT_NAME": str(junit), "IS_EVAL_PRODUCTION": "1",
                 "EVAL_PROJECT_PORT": str(free_port())}
        code, out, secs = self.pn.cmd(work, "exec", "playwright", "test", "--reporter=junit", "--retries=0", *files,
                                      timeout=timeout, extra=extra)
        return junit_results(junit, "playwright"), code, out, secs

    def tests(self, c: Cand, work: Path, tmp: Path, files: list[str] | None = None, timeout: float = 1800):
        files = c.files if files is None else files
        if c.runner == "pytest":
            return self.pytest(c, work, tmp, files, timeout)
        return self.playwright(c, work, tmp, files, timeout)

    def validate(self, c: Cand) -> Verdict:
        lock = self.gpu if c.track == "kernel" else contextlib.nullcontext()
        with lock, self.folder() as tmp:
            return self._validate(c, tmp)

    def _validate(self, c: Cand, tmp: Path) -> Verdict:
        why = leak_reason(c)
        if why:
            return Verdict(False, why)
        # 1. base: every f2p test fails or is missing
        work = self.tree(c, tmp, gold=False)
        base, code, out, _ = self.tests(c, work, tmp)
        if c.runner == "playwright" and not base and code not in (0, 1):
            return Verdict(False, f"base run failed to start: {out[-200:]}")
        # 2. gold, three times
        golds, secs = [], []
        for _ in range(GOLD_RUNS):
            work = self.tree(c, tmp, gold=True)
            res, code, out, s = self.tests(c, work, tmp)
            if not res:
                return Verdict(False, f"gold run produced no report (exit {code}): {out[-300:]}")
            golds.append(res)
            secs.append(s)
        passed_all = set.intersection(*[{k for k, v in g.items() if v == grader.PASSED} for g in golds])
        seen = set().union(*[set(g) for g in golds])
        if c.f2p:
            f2p_start, p2p_start = list(c.f2p), list(c.p2p)
        else:  # web and kernel: every test the hidden files hold
            f2p_start = sorted(seen - {k for k, v in base.items() if v == grader.PASSED})
            p2p_start = sorted(k for k, v in base.items() if v == grader.PASSED)
        f2p, p2p, dropped = [], [], []
        for t in f2p_start:
            if base.get(t) == grader.PASSED:  # passes on base: p2p when it also passes on gold, else dropped
                (p2p if t in passed_all else dropped).append(t)
            elif t in passed_all:
                f2p.append(t)
            else:
                msg = "a fail-to-pass test does not pass the gold three times"
                if c.track == "kernel":
                    msg += " (a kernel task bug: fix the test or the gold)"
                return Verdict(False, f"{msg}: {t}")
        for t in p2p_start:
            (p2p if t in passed_all and base.get(t) == grader.PASSED else dropped).append(t)
        if not f2p:
            return Verdict(False, "no fail-to-pass test")
        if p2p_start and len([t for t in dropped if t in p2p_start]) > P2P_DROP_SHARE * len(p2p_start):
            return Verdict(False, f"{len(dropped)} of {len(p2p_start)} pass-to-pass tests are flaky or fail")
        timeout_s = max(120, int(editdata.CAP_FACTOR * max(secs) + 0.999))
        # 3. visible collection, without the hidden files
        work = self.tree(c, tmp, gold=False, hidden=False)
        why = self.visible(c, work, tmp)
        if why:
            return Verdict(False, why)
        # 4. python L: every stubbed function is run by a surviving f2p test
        if c.cover:
            alive = set(f2p)
            orphans = [k for k, tests_ in c.cover.items() if not alive & set(tests_)]
            if orphans:
                return Verdict(False, f"{len(orphans)} stubbed functions are run by no surviving fail-to-pass test")
        # 5. kernel: the mutant rule, and a gold that compiles with no warnings
        caught = None
        if c.track == "kernel":
            caught = 0
            for k in range(len(c.mutants)):
                work = self.tree(c, tmp, gold=False, mutant=k)
                res, _, _, _ = self.tests(c, work, tmp, timeout=timeout_s)
                # caught: the mutant compiled and ran, and an f2p test failed (not an error, timeout or empty report)
                if res and any(res.get(t) == grader.FAILED for t in f2p):
                    caught += 1
            if caught < 3:
                return Verdict(False, f"the hidden tests are too weak: they catch {caught} of {len(c.mutants)} "
                                      "mutants", mutants_caught=caught)
            warn = self.gold_warnings(c, tmp)
            if warn:
                return Verdict(False, f"the gold compiles with warnings: {warn[:200]}", mutants_caught=caught)
        if c.runner == "pytest" and leak_names(c, f2p + p2p):
            return Verdict(False, leak_names(c, f2p + p2p))
        return Verdict(True, "", sorted(f2p), sorted(p2p), sorted(dropped), round(max(secs), 2), caught, timeout_s)

    def visible(self, c: Cand, work: Path, tmp: Path) -> str:
        if c.track == "python":
            py = self.envs.python(c.env)
            argv = [str(py), "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider"]
            with self.envs.src_root(c.env, work):
                code, out, _ = run(argv, cwd=work, env=clean_env({"PYTHONDONTWRITEBYTECODE": "1"}), timeout=900)
            return "" if code in (0, 5) else f"the visible tests don't collect (exit {code}): {out[-200:]}"
        if c.track == "kernel":
            _, code, out, _ = self.pytest(c, work, tmp, ["tests/test_smoke.py"], timeout=900)
            return "" if code in (0, 1) else f"test_smoke.py doesn't import and compile (exit {code}): {out[-200:]}"
        code, out = self.pn.install(work)
        if code:
            return f"pnpm install failed: {out[-200:]}"
        (work / "test").mkdir(exist_ok=True)
        # Playwright exits 1 on "No tests found" unless told otherwise (D-227)
        code, out, _ = self.pn.cmd(work, "exec", "playwright", "test", "--list", "--pass-with-no-tests", timeout=600,
                                   extra={"IS_EVAL_PRODUCTION": "1", "EVAL_PROJECT_PORT": str(free_port())})
        return "" if code == 0 else f"playwright test --list fails with an empty test/ (exit {code}): {out[-200:]}"

    def gold_warnings(self, c: Cand, tmp: Path) -> str:
        work = self.tree(c, tmp, gold=True, hidden=False)
        py = self.envs.python(c.env)
        code_ = ("import glob, json, sys; sys.path.insert(0, '.'); import gpu; "
                 "m = gpu.load(sorted(glob.glob('kernels/*.gpu.cu'))); print(json.dumps(m.logs))")
        code, out, _ = run([str(py), "-c", code_], cwd=work, env=clean_env(), timeout=600, stdout_only=True)
        if code:
            return f"compile failed: {out[-300:]}"
        logs = json.loads(out.strip().splitlines()[-1])
        return "; ".join(f"{Path(k).name}: {v}" for k, v in logs.items() if v.strip())


FILE_TITLE = re.compile(r"/|\.[A-Za-z]{1,5}$")


def leak_reason(c: Cand) -> str:
    """The leak rules that need no test run."""
    if c.track == "web":
        texts = "\n".join(c.leak_texts)
        for name in c.files:  # the spec files; test/util/ helpers are not specs
            if Path(name).name in texts:
                return f"the task text names a hidden spec file ({Path(name).name})"
        for title in c.spec_titles:
            # a title that is a file path names a file the request asks for, not a test (D-229)
            if len(title) >= 8 and title in texts and not FILE_TITLE.search(title):
                return "the task text names a spec title"
    return ""


def leak_names(c: Cand, tests_: list[str]) -> str:
    for text in c.leak_texts:
        found = editdata.leaks(text, c.gold_patch, tests_)
        if found:
            return "leak: " + found[0]
    return ""


# ----------------------------------------------------------------------------- the cache of validation results


class Results:
    """validation.jsonl in the build cache: one line per candidate key; the last line for a key wins."""

    def __init__(self, cache: Path):
        self.path = cache / "validation.jsonl"
        self.rows = {r["key"]: r for r in read_jsonl(self.path)}
        self.lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def put(self, c: Cand, v: Verdict, env: str) -> dict:
        row = {"key": c.key, "id": None, "cell": f"{c.split}/{c.track}/{c.size}", "track": c.track, "source": c.source, "kept": v.kept, "why": v.why,
               "f2p": v.f2p, "p2p": v.p2p, "dropped_p2p": v.dropped_p2p, "gold_seconds": v.gold_seconds, "env": env,
               "mutants_caught": v.mutants_caught, "timeout_s": v.timeout_s, "python": v.python}
        with self.lock:
            self.rows[c.key] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "ab") as f:
                f.write(jline(row))
        return row


def unvalidated(c: Cand) -> dict:
    return {"key": c.key, "kept": True, "why": "", "f2p": list(c.f2p), "p2p": list(c.p2p), "dropped_p2p": [],
            "gold_seconds": 0.0, "env": c.env, "mutants_caught": None, "timeout_s": 120, "track": c.track,
            "source": c.source}


class Validator:
    def __init__(self, runner: Runner | None, results: Results, jobs: int, validate: bool):
        self.runner, self.results, self.jobs, self.validate_on = runner, results, max(1, jobs), validate

    def one(self, c: Cand) -> dict:
        if not self.validate_on:
            return unvalidated(c)
        row = self.results.get(c.key)
        if row is not None and row.get("env") == c.env:
            return row
        try:
            v = self.runner.validate(c)
        except Exception as e:  # infrastructure (a download, a disk, a lock): dropped this build, retried on the next
            log(f"    {c.track} {c.split} {c.size} {c.key[:8]}: error, not cached: {type(e).__name__}: {str(e)[:200]}")
            return {**unvalidated(c), "kept": False, "why": f"error: {type(e).__name__}: {str(e)[:300]}"}
        row = self.results.put(c, v, c.env)
        log(f"    {c.track} {c.split} {c.size} {c.key[:8]}: {'kept' if v.kept else 'dropped, ' + v.why[:120]}")
        return row

    def many(self, cands: list[Cand]) -> list[dict]:
        if self.jobs == 1 or len(cands) <= 1:
            return [self.one(c) for c in cands]
        with concurrent.futures.ThreadPoolExecutor(self.jobs) as pool:
            return list(pool.map(self.one, cands))


def draw_cell(cell: str, families: list[list[Cand]], need: int, seed: int, val: Validator) -> list[tuple[Cand, dict]]:
    """Round-robin over the cell's families, each in a seeded shuffle, skipping candidates that fail validation and
    repositories that already gave MAX_PER_REPO, until the cell holds `need` tasks. Validation runs ahead in batches of
    --jobs; acceptance follows the fixed order, so the result doesn't depend on --jobs."""
    rng = random.Random(f"{seed}-{cell}")
    lists = []
    for fam in families:
        fam = list(fam)
        rng.shuffle(fam)
        lists.append(collections.deque(fam))
    order = []
    while any(lists):
        for q in lists:
            if q:
                order.append(q.popleft())
    kept: list[tuple[Cand, dict]] = []
    per_group = collections.Counter()
    i = 0
    while len(kept) < need and i < len(order):
        batch, j = [], i
        while j < len(order) and len(batch) < val.jobs:
            if per_group[order[j].group] < MAX_PER_REPO:
                batch.append(order[j])
            j += 1
        val.many(batch)
        while i < j and len(kept) < need:
            c = order[i]
            i += 1
            if per_group[c.group] >= MAX_PER_REPO:
                continue
            row = val.one(c)  # cached by now
            if row["kept"]:
                kept.append((c, row))
                per_group[c.group] += 1
    if len(kept) < need:
        raise Short(f"cell {cell} is short by {need - len(kept)}: {len(order)} candidates, {len(kept)} kept")
    return kept


# ----------------------------------------------------------------------------- build


@dataclass
class Built:
    read: list = field(default_factory=list)  # ReadTask
    repo: list = field(default_factory=list)  # (Cand, row)
    venvs: dict = field(default_factory=dict)  # key -> spec
    nodeenvs: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def build_python(paths: Paths, val: Validator, envs: Envs, built: Built, seed: int) -> None:
    recs = edit_records()
    splits = assign_repos(recs)
    reg = editdata.read_json(editdata.edit_path("repos.json"), {})
    sm = python_sm_candidates(recs, splits)
    large = {split: large_candidates(seed, split, reg) for split in SPLITS}
    # One venv per repository: Python 3.12, else 3.11, else 3.10. A task whose gold fails under one Python tries the
    # next; afterwards every kept task of a repository moves to the lowest Python any of them needed.
    locks: dict[str, threading.Lock] = collections.defaultdict(threading.Lock)
    resolved: dict[tuple, str] = {}

    def env_for(c: Cand, python: str) -> str:
        repo = c.group
        with locks[repo]:
            if (repo, python) not in resolved:
                tree = Path(tempfile.mkdtemp(prefix="snap-", dir=paths.cache))
                try:
                    c.base(tree)
                    if val.validate_on:
                        res = resolve_python_env(paths.cache, repo, reg.get(repo) or {}, tree, python)
                        if res["method"] == "failed":
                            raise RuntimeError(f"under Python {python}: {res['why']}")
                        spec = res["spec"]
                    else:  # a --no-validate test build: the unresolved requirements
                        test, runtime, _ = project_requirements(tree)
                        spec = {"python": python, "requirements": test + runtime, "import_root": import_root(tree)}
                finally:
                    rmtree(tree)
                resolved[(repo, python)] = envs.add("py", spec)
            return resolved[(repo, python)]

    orig_one = val.one

    def one(c: Cand) -> dict:
        if c.track != "python":
            return orig_one(c)
        k = 0
        while True:
            try:
                if not c.env:
                    c.env = env_for(c, PYTHONS[k])
            except (RuntimeError, OSError, ValueError, SyntaxError) as e:  # no venv: dropped, never cached
                row = {**unvalidated(c), "kept": False, "why": f"error: no venv: {str(e)[:200]}"}
            else:
                row = orig_one(c)
                k = PYTHONS.index(envs.specs[c.env]["python"]) if envs.specs[c.env]["python"] in PYTHONS else k
            gold_failed = row["why"].startswith(("a fail-to-pass test does not pass the gold", "gold run produced",
                                                 "error: no venv"))
            if row["kept"] or not val.validate_on or not gold_failed or k + 1 >= len(PYTHONS):
                return row
            k += 1  # the gold failed (or no venv): try the next Python for this task
            c.env = ""

    val.one = one  # type: ignore[method-assign]
    try:
        for split in SPLITS:
            for size in PYTHON_FAMILIES:
                fams = [sm[(split, size)][f] for f, _ in PYTHON_FAMILIES[size]]
                built.repo += draw_cell(f"{split}/python/{size}", fams, PYTHON_PLAN.get(size, 0) // 2, seed, val)
            built.repo += draw_cell(f"{split}/python/L", [large[split]], PYTHON_PLAN.get("L", 0) // 2, seed, val)
    finally:
        val.one = orig_one  # type: ignore[method-assign]
    by_group: dict[str, list[int]] = collections.defaultdict(list)
    for k, (c, _) in enumerate(built.repo):
        if c.track == "python":
            by_group[c.group].append(k)
    for ks in by_group.values():
        if len({built.repo[k][0].env for k in ks}) > 1:
            target = max((built.repo[k][0].env for k in ks), key=lambda e: PYTHONS.index(envs.specs[e]["python"]))
            for k in ks:
                c = built.repo[k][0]
                if c.env != target:
                    c.env = target
                    row = val.one(c)
                    if not row["kept"]:
                        raise Short("a python repository's tasks need different Pythons (see validation.jsonl); "
                                    "Rishi's call")
                    built.repo[k] = (c, row)
    for c, _ in built.repo:
        if c.track == "python":
            built.venvs[c.env] = envs.specs[c.env]


def build_web(paths: Paths, val: Validator, built: Built, validate: bool) -> None:
    """Per split, walk the projects in preference order: the first that validates takes L (requests 1-12), the next
    two M (1-6), the next two S (1-3). A project gives at most one task."""
    pn = val.runner.pn if val.runner else Pnpm(paths.cache)
    wb = WebBench(paths.cache)
    wb.ensure()
    importers = wb.importers()
    order = {}
    for split in SPLITS:
        order[split] = []
        for project in WEB_SPLIT[split]:
            proj = wb.dir / "projects" / project
            why = web_skip_reason(proj, project) if proj.is_dir() else "missing"
            if why:
                built.notes.append(f"web {split}: {project} skipped, {why}")
            else:
                order[split].append(project)
        log(f"  web {split}: projects in order: {', '.join(order[split])}")
    for split in SPLITS:
        used: set[str] = set()
        for size in ("L", "M", "S"):
            need, got = WEB_PLAN[size] // 2, 0
            for project in order[split]:
                if got >= need:
                    break
                if project in used:
                    continue
                try:
                    c = web_cand(wb, paths.cache, project, split, size, importers, pn, built.nodeenvs, validate)
                    if validate:
                        folder = paths.cache / "web" / f"{project}-{size}"
                        pn.fetch(folder)
                        pn.browser(folder, built.nodeenvs[c.env]["playwright"])
                except (SetupError, RuntimeError, OSError) as e:
                    built.notes.append(f"web {split}: {project} at {size} skipped: {str(e)[:200]}")
                    continue
                row = val.one(c)
                if row["kept"]:
                    built.repo.append((c, row))
                    used.add(project)
                    got += 1
            if got < need:
                raise Short(f"web {split} {size}: short by {need - got} (the split's projects ran out)")
    built.nodeenvs = {k: v for k, v in built.nodeenvs.items() if any(c.env == k for c, _ in built.repo)}


def build_kernel(paths: Paths, val: Validator, envs: Envs, built: Built, validate: bool) -> None:
    copy_shared(paths)
    spec = kernel_env_spec(paths.cache) if validate else {"python": "3.12", "requirements": [NVRTC], "import_root": "."}
    key = envs.add("kernel", spec)
    if validate:
        check_gpu(envs, key, paths)
    for (split, size), ops in KERNEL_OPS.items():
        for op in ops:
            src = paths.cache / "kernels" / op
            if not (src / "meta.json").exists():
                raise SetupError(f"kernel task {op} is missing from {src.parent} (task 9 writes it)")
            c = kernel_cand(paths.cache, op)
            if (c.split, c.size) != (split, size):
                raise SetupError(f"kernel {op}: meta.json says {c.split}/{c.size}, the plan {split}/{size}")
            c.env = key
            row = val.one(c)
            if not row["kept"]:
                raise Short(f"kernel {op} failed validation: {row['why']} (fix the task, or swap in a spare from "
                            f"{', '.join(KERNEL_SPARES[split])} and log it)")
            built.repo.append((c, row))
    built.venvs[key] = spec


def check_gpu(envs: Envs, key: str, paths: Paths) -> None:
    py = envs.python(key)
    shared = paths.cache / "kernels" / "_shared"
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import gpu, numpy as np; print(gpu.backend()); "
             "import pathlib, tempfile; d = pathlib.Path(tempfile.mkdtemp()); f = d / 'k.gpu.cu'; "
             "f.write_text('GPU_KERNEL void k(float* o) { o[threadIdx.x] = 1.0f; }'); m = gpu.load([f]); "
             "o = gpu.empty(4, np.float32); m.launch('k', 1, 4, [o]); assert (o.to_host() == 1).all()")
    code, out, _ = run([str(py), "-c", probe, str(shared)], env=clean_env(), timeout=600)
    if code:
        raise SetupError(f"the kernel venv's GPU check failed (a GPU and its runtime are needed): {out[-400:]}")
    backend = next((line for line in out.splitlines() if line.strip() in ("cuda", "hip")), "?")
    log(f"  kernel venv: gpu.backend() = {backend.strip()}")


def kernel_backend(envs: Envs, key: str, paths: Paths) -> str:
    py = envs.python(key)
    code, out, _ = run([str(py), str(paths.cache / "kernels" / "_shared" / "gpu.py"), "--probe"], env=clean_env(),
                       stdout_only=True)
    m = re.match(r"backend=(\w+) devices=(\d+)", out.strip())
    return m.group(1) if m and m.group(1) != "none" else ""


def assemble(paths: Paths, built: Built, validate: bool, kernel_on: list[str]) -> dict:
    """Write the set folder from the built tasks; ids by track and size, tune before heldout."""
    out = paths.set_dir
    tmp = out.with_name(out.name + ".tmp")
    rmtree(tmp)
    tmp.mkdir(parents=True)
    task_lines, grade_lines, sources = [], [], []
    counts: dict[str, int] = collections.Counter()
    order = {t: i for i, t in enumerate(TRACKS)}
    numbered: dict[tuple, int] = collections.Counter()
    rows = sorted(built.read, key=lambda r: (order[r.track], SPLITS.index(r.split)))
    for rt in rows:
        numbered[(rt.track, "R")] += 1
        tid = f"{paths.name}-{rt.track}-R-{numbered[(rt.track, 'R')]:02d}"
        rel = f"ctx/{tid}/context.txt"
        put_files(tmp, {rel: rt.task.context.encode("utf-8")})
        task_lines.append({"id": tid, "kind": "read", "track": rt.track, "split": rt.split, "size": "R",
                           "question": rt.task.question, "input": [rel]})
        grade_lines.append({"id": tid, "grader": "answer", "gold": rt.task.answer, "grading": rt.task.grading,
                            "pass_score": rt.pass_score})
        sources.append({"id": tid, "family": rt.family, "level": rt.level, "record_id": rt.task.task_id})
        counts[f"{rt.split}/{rt.track}/R"] += 1
    repo = sorted(built.repo, key=lambda cr: (order[cr[0].track], SIZES.index(cr[0].size),
                                              SPLITS.index(cr[0].split)))
    results = Results(paths.cache)
    for c, row in repo:
        numbered[(c.track, c.size)] += 1
        tid = f"{paths.name}-{c.track}-{c.size}-{numbered[(c.track, c.size)]:02d}"
        put_files(tmp, {f"repos/{tid}.tar": tar_bytes(c.base_files()), f"hidden/{tid}.tar.gz": tar_bytes(c.hidden, gz=True)})
        line = {"id": tid, "kind": "repo", "track": c.track, "split": c.split, "size": c.size, "task": c.task,
                "base": f"repos/{tid}.tar"}
        line["node_env" if c.track == "web" else "venv"] = c.env
        line["test_hint"] = TEST_HINT[c.track]
        if c.track == "kernel":
            line["requires"] = ["gpu"]
        task_lines.append(line)
        grade_lines.append({"id": tid, "grader": "tests", "runner": c.runner, "hidden": f"hidden/{tid}.tar.gz",
                            "f2p": row["f2p"], "p2p": row["p2p"], "files": c.files, "timeout_s": int(row["timeout_s"])})
        sources.append({"id": tid, "key": c.key, **c.source, "gold_files": c.gold_files, "gold_lines": c.gold_lines,
                        "gold_seconds": row["gold_seconds"], "mutants_caught": row.get("mutants_caught"),
                        "f2p": len(row["f2p"]), "p2p": len(row["p2p"]), "dropped_p2p": len(row["dropped_p2p"])})
        counts[f"{c.split}/{c.track}/{c.size}"] += 1
        if c.key in results.rows and results.rows[c.key].get("id") != tid:
            results.rows[c.key]["id"] = tid
            with open(results.path, "ab") as f:
                f.write(jline(results.rows[c.key]))
    for key, spec in sorted(built.venvs.items()):
        put_files(tmp, {f"venvs/{key}.json": (json.dumps(spec, indent=1) + "\n").encode("utf-8")})
    for key, spec in sorted(built.nodeenvs.items()):
        put_files(tmp, {f"nodeenvs/{key}.json": (json.dumps(spec, indent=1) + "\n").encode("utf-8")})
    tasks_b = b"".join(jline(t) for t in task_lines)
    grades_b = b"".join(jline(g) for g in grade_lines)
    put_files(tmp, {"tasks.jsonl": tasks_b, "grades.jsonl": grades_b})
    manifest = {"format": 1, "name": paths.name,
                "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "builder": rlm_commit(), "permission_mode": "bypass", "validated": validate,
                "kernel_validated_on": kernel_on,
                "counts": {k: counts[k] for k in sorted(counts)}, "tasks_sha256": sha256(tasks_b),
                "grades_sha256": sha256(grades_b)}
    write_json(tmp / "manifest.json", manifest)
    (paths.cache).mkdir(parents=True, exist_ok=True)
    (paths.cache / "sources.jsonl").write_bytes(b"".join(jline(s) for s in sources))
    rmtree(out)
    tmp.rename(out)
    return manifest


def cmd_build(paths: Paths, only: list[str], jobs: int, validate: bool, force: bool) -> int:
    check_outside(paths)
    if paths.set_dir.exists() and not force:
        raise SetupError(f"{paths.set_dir} exists: pass --force to replace it")
    check_ignored()
    paths.cache.mkdir(parents=True, exist_ok=True)
    envs = Envs(paths.cache)
    pn = Pnpm(paths.cache) if "web" in only else None
    if pn and validate:
        pn.check()
    runner = Runner(paths.cache, envs, pn) if validate else None
    val = Validator(runner, Results(paths.cache), jobs, validate)
    built = Built()
    seed = tasks.private_seed_list("harness-eval-v1-repo", 1)[0] if {"python"} & set(only) else 0
    kernel_on: list[str] = []
    if {"research", "search"} & set(only):
        log("read tasks (research, search)")
        plan = {fl: n for fl, n in READ_PLAN.items() if READ_TRACK[fl[0]] in only}
        built.read = build_read(plan)
        for rt in built.read:
            why = read_leak(rt.task.question, rt.task.answer)
            if why:
                raise Short(f"a {rt.track} task is unusable: {why}")
    if "python" in only:
        log("python tasks")
        build_python(paths, val, envs, built, seed)
    if "web" in only:
        log("web tasks")
        build_web(paths, val, built, validate)
    if "kernel" in only:
        log("kernel tasks")
        build_kernel(paths, val, envs, built, validate)
        if validate:
            key = next(c.env for c, _ in built.repo if c.track == "kernel")
            backend = kernel_backend(envs, key, paths)
            kernel_on = [backend] if backend else []
    manifest = assemble(paths, built, validate, kernel_on)
    print_counts(paths, built, manifest)
    for n in built.notes:
        log(f"  note: {n}")
    return EXIT_OK


def print_counts(paths: Paths, built: Built, manifest: dict) -> None:
    rows = read_jsonl(paths.cache / "validation.jsonl")
    last = {r["key"]: r for r in rows}
    dropped = collections.Counter(re.sub(r"[:(].*", "", r["why"]).strip() for r in last.values() if not r["kept"])
    log(f"set {paths.set_dir}")
    for k, n in manifest["counts"].items():
        log(f"  {k}: {n}")
    log(f"  candidates validated: {len(last)}, kept {sum(r['kept'] for r in last.values())}")
    cells = collections.defaultdict(lambda: [0, 0])
    for r in last.values():
        cells[r.get("cell") or r["track"]][0] += 1
        cells[r.get("cell") or r["track"]][1] += bool(r["kept"])
    for cell, (n, k) in sorted(cells.items()):
        log(f"  {cell}: {n} candidates validated, {k} kept")
    resolved = _resolved(paths.cache)
    kinds = collections.Counter(f"python venv by {r['method']}, Python {r['python']}" for r in resolved.values())
    kinds.update(f"{'kernel venv' if k.startswith('kernel') else 'python venv'} in the set" for k in built.venvs)
    kinds.update("node env" for _ in built.nodeenvs)
    for what, n in sorted(kinds.items()):
        log(f"  environments: {n} {what}")
    for why, n in dropped.most_common(8):
        log(f"  dropped {n}: {why}")
    secs = sorted(r["gold_seconds"] for _, r in built.repo if r.get("gold_seconds"))
    if secs:
        log(f"  gold test seconds: median {secs[len(secs) // 2]:.1f}, max {secs[-1]:.1f}")
    for c, r in built.repo:
        if c.track == "kernel":
            log(f"  {c.split} {c.size} kernel: {r.get('mutants_caught')} of {len(c.mutants)} mutants caught")
    size = sum(p.stat().st_size for p in paths.set_dir.rglob("*") if p.is_file())
    log(f"  set size on disk: {size / 1e6:.1f} MB; tasks_sha256 {manifest['tasks_sha256'][:16]}…, "
        f"grades_sha256 {manifest['grades_sha256'][:16]}…")


# ----------------------------------------------------------------------------- check


def plan_counts() -> dict[str, int]:
    out = collections.Counter()
    for (family, _), n in READ_PLAN.items():
        for i in range(n):
            out[f"{SPLITS[i % 2]}/{READ_TRACK[family]}/R"] += 1
    for track, plan in (("python", PYTHON_PLAN), ("web", WEB_PLAN), ("kernel", KERNEL_PLAN)):
        for size, n in plan.items():
            for split in SPLITS:
                out[f"{split}/{track}/{size}"] += n // 2
    return dict(out)


TASK_KEYS = {"read": {"id", "kind", "track", "split", "size", "question", "input"},
             "repo": {"id", "kind", "track", "split", "size", "task", "base", "test_hint"}}
GRADE_KEYS = {"answer": {"id", "grader", "gold", "grading", "pass_score"},
              "tests": {"id", "grader", "runner", "hidden", "f2p", "p2p", "files", "timeout_s"}}


def cmd_check(paths: Paths) -> list[str]:
    """Every problem with the set folder (offline, $0)."""
    d = paths.set_dir
    problems: list[str] = []
    if not d.is_dir():
        return [f"no set at {d}"]
    try:
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"manifest.json: {e}"]
    try:
        tb, gb = (d / "tasks.jsonl").read_bytes(), (d / "grades.jsonl").read_bytes()
    except OSError as e:
        return [f"missing {Path(e.filename or '').name}"]
    if sha256(tb) != manifest.get("tasks_sha256"):
        problems.append("tasks.jsonl: sha256 differs from the manifest")
    if sha256(gb) != manifest.get("grades_sha256"):
        problems.append("grades.jsonl: sha256 differs from the manifest")
    if b"\r" in tb or b"\r" in gb:
        problems.append("tasks.jsonl or grades.jsonl holds a \\r")
    tl, gl = [], []
    for name, blob, rows in (("tasks.jsonl", tb, tl), ("grades.jsonl", gb, gl)):
        for n, x in enumerate(blob.decode("utf-8", "replace").split("\n"), 1):
            if not x.strip():
                continue
            try:
                row = json.loads(x)
            except ValueError as e:
                problems.append(f"{name} line {n}: not JSON ({e})")
                continue
            if not isinstance(row, dict) or "id" not in row:
                problems.append(f"{name} line {n}: not a task object")
                continue
            rows.append(row)
    tid, gid = {t["id"]: t for t in tl}, {g["id"]: g for g in gl}
    for i in sorted(set(tid) ^ set(gid)):
        problems.append(f"{i}: in only one of tasks.jsonl and grades.jsonl")
    shared = paths.cache / "kernels" / "_shared"
    shared_files = kernel_files(shared)
    sources = {s["id"]: s for s in read_jsonl(paths.cache / "sources.jsonl")}
    counts = collections.Counter()
    for t in tl:
        i = t["id"]
        if not ID_RE.match(i) or not i.startswith(paths.name):
            problems.append(f"{i}: id doesn't match the pattern")
        counts[f"{t['split']}/{t['track']}/{t['size']}"] += 1
        want = TASK_KEYS.get(t.get("kind"), set())
        if t.get("kind") == "repo":
            want = want | ({"node_env"} if t["track"] == "web" else {"venv"}) | (
                {"requires"} if t["track"] == "kernel" else set())
        if set(t) != want:
            problems.append(f"{i}: task keys {sorted(set(t) ^ want)} differ from §6.2")
        if t["track"] == "kernel" and t.get("requires") != ["gpu"]:
            problems.append(f"{i}: a kernel task without requires [gpu]")
        if t["track"] != "kernel" and "requires" in t:
            problems.append(f"{i}: requires on a {t['track']} task")
        if t.get("kind") == "read":
            for rel in t["input"]:
                if not (d / rel).is_file():
                    problems.append(f"{i}: missing {rel}")
                elif b"\r" in (d / rel).read_bytes():
                    problems.append(f"{i}: {rel} holds a \\r")
            continue
        key = t.get("venv") or t.get("node_env")
        folder = "venvs" if "venv" in t else "nodeenvs"
        if not (d / folder / f"{key}.json").is_file():
            problems.append(f"{i}: missing {folder}/{key}.json")
        g = gid.get(i, {})
        base_p, hid_p = d / t["base"], d / (g.get("hidden") or f"hidden/{i}.tar.gz")
        if not base_p.is_file() or not hid_p.is_file():
            problems.append(f"{i}: missing {'base tar' if not base_p.is_file() else 'hidden tar'}")
            continue
        try:
            base_names = tar_names(base_p.read_bytes())
            hid_names = tar_names(hid_p.read_bytes(), gz=True)
        except (tarfile.TarError, OSError, EOFError) as e:
            problems.append(f"{i}: a tar doesn't read: {e}")
            continue
        for label, names in (("base", base_names), ("hidden", hid_names)):
            for name, is_file in names:
                parts = name.split("/")
                if not grader.safe_path(name) or not is_file:
                    problems.append(f"{i}: {label} tar member {name!r} is unsafe or not a regular file")
                if ".git" in parts or "node_modules" in parts:
                    problems.append(f"{i}: {label} tar holds {name}")
        hidden_set = {n for n, _ in hid_names}
        for name, _ in base_names:
            if name in hidden_set:
                problems.append(f"{i}: hidden file {name} is in the base tar")
        if t["track"] == "kernel":
            base_files = untar(base_p.read_bytes())
            for name in ("gpu.py", "gpu_compat.h"):
                if not shared_files:
                    problems.append(f"{i}: no kernels/_shared/ in the build cache to compare {name} with")
                elif base_files.get(name) != shared_files.get(name):
                    problems.append(f"{i}: base {name} differs from the shared copy")
            op = (sources.get(i) or {}).get("op")
            forbidden = {}
            if op:
                src = paths.cache / "kernels" / op
                stubs = kernel_files(src / "base")
                for rel, b in kernel_files(src / "gold").items():
                    if stubs.get(rel) != b:
                        forbidden[b] = f"gold {rel}"
                for md in sorted((src / "mutants").iterdir()) if (src / "mutants").is_dir() else []:
                    for rel, b in kernel_files(md).items():
                        if stubs.get(rel) != b:
                            forbidden[b] = f"mutant {md.name}/{rel}"
            for blob in (base_p.read_bytes(), gzip.decompress(hid_p.read_bytes())):
                for name, _ in tar_names(blob):
                    if name.split("/")[0] in ("gold", "mutants"):
                        problems.append(f"{i}: a kernel gold or mutant file {name} is in a tar")
                for name, b in untar(blob).items():
                    if b in forbidden:
                        problems.append(f"{i}: {name} in a tar is the kernel {forbidden[b]}")
    for g in gl:
        want = GRADE_KEYS.get(g.get("grader"), set())
        if set(g) != want:
            problems.append(f"{g['id']}: grade keys {sorted(set(g) ^ want)} differ from §6.3")
    for g in gl:  # a gold string or a test name in the task's own line
        text = json.dumps(tid.get(g["id"], {}), ensure_ascii=False)
        if g.get("grader") == "answer":
            golds = g["gold"] if isinstance(g["gold"], list) else [g["gold"]]
            if any(len(str(x).strip()) >= 4 and str(x).strip().lower() in text.lower() for x in golds):
                problems.append(f"{g['id']}: a gold string is in tasks.jsonl")
        else:
            for name in g.get("f2p", []) + g.get("p2p", []):
                short = name.split("::")[-1].split(".")[-1].split("/")[0]
                if len(short) >= 8 and re.search(rf"(?<![\w]){re.escape(short)}(?![\w])", text):
                    problems.append(f"{g['id']}: a test name is in tasks.jsonl")
                    break
    plan = plan_counts()
    for k in sorted(set(plan) | set(counts)):
        if plan.get(k, 0) != counts.get(k, 0):
            problems.append(f"count {k}: {counts.get(k, 0)}, the plan says {plan.get(k, 0)}")
    if dict(manifest.get("counts") or {}) != {k: v for k, v in counts.items()}:
        problems.append("manifest counts differ from tasks.jsonl")
    return problems


# ----------------------------------------------------------------------------- summary


def cmd_summary(paths: Paths, out: Path = SUMMARY) -> Path:
    """documentation/harness-eval-v1.md: counts, ids, gold sizes and test timings. No task text, gold, test name, diff,
    or the name of a repository, Web-Bench project or kernel op."""
    d = paths.set_dir
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    tl = read_jsonl(d / "tasks.jsonl")
    gl = {g["id"]: g for g in read_jsonl(d / "grades.jsonl")}
    sources = {s["id"]: s for s in read_jsonl(paths.cache / "sources.jsonl")}
    lines = [f"# Eval set {manifest['name']}", "",
             "Built by `scripts/harness_evalset.py` for rlm-harness v1 (step 1). The set lives outside git, in "
             "`RLM_DATA_DIR\\harness-eval\\`; this page holds numbers and ids only.", "",
             f"- created {manifest['created']}, builder commit `{manifest['builder']['commit'][:12]}`"
             f"{' (dirty)' if manifest['builder']['dirty'] else ''}",
             f"- validated: {str(manifest['validated']).lower()}; permission mode `{manifest['permission_mode']}`; "
             f"kernel track validated on: {', '.join(manifest['kernel_validated_on']) or 'none'}",
             f"- tasks_sha256 `{manifest['tasks_sha256']}`", f"- grades_sha256 `{manifest['grades_sha256']}`", "",
             "## Counts", "", "| Track | Split | R | S | M | L |", "|---|---|---|---|---|---|"]
    for track in TRACKS:
        for split in SPLITS:
            row = [str(manifest["counts"].get(f"{split}/{track}/{s}", 0)) for s in SIZES]
            if any(x != "0" for x in row):
                lines.append(f"| {track} | {split} | " + " | ".join(row) + " |")
    lines += ["", "## Tasks", "", "| Id | Split | f2p | p2p | Gold files | Gold lines | Gold test s | timeout s | "
                                  "Mutants caught |", "|---|---|---|---|---|---|---|---|---|"]
    for t in tl:
        s, g = sources.get(t["id"], {}), gl.get(t["id"], {})
        if t["kind"] == "read":
            lines.append(f"| {t['id']} | {t['split']} | | | | | | | |")
            continue
        mc = s.get("mutants_caught")
        lines.append(f"| {t['id']} | {t['split']} | {len(g.get('f2p', []))} | {len(g.get('p2p', []))} | "
                     f"{s.get('gold_files', '')} | {s.get('gold_lines', '')} | {s.get('gold_seconds', '')} | "
                     f"{g.get('timeout_s', '')} | {'' if mc is None else f'{mc} of 4'} |")
    secs = sorted(float(s["gold_seconds"]) for s in sources.values() if s.get("gold_seconds"))
    if secs:
        lines += ["", f"Gold test runs: median {secs[len(secs) // 2]:.1f} s, max {secs[-1]:.1f} s."]
    text = "\n".join(lines) + "\n"
    summary_guard(text, tl, list(gl.values()), list(sources.values()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text.encode("utf-8"))
    return out


def summary_guard(text: str, tl: list, gl: list, sources: list) -> None:
    needles: list[str] = []
    for t in tl:
        needles += [t.get("question") or "", t.get("task") or ""]
    for g in gl:
        golds = g.get("gold") if isinstance(g.get("gold"), list) else [g.get("gold")]
        needles += [str(x) for x in golds if x is not None]
        needles += [n.split("::")[-1] for n in g.get("f2p", []) + g.get("p2p", [])]
    for s in sources:
        needles += [str(s.get(k) or "") for k in ("repo", "project", "op", "record_id")]
    for n in needles:
        n = n.strip()
        if len(n) >= 4 and n in text:
            raise SetupError("summary guard: task text, a gold answer, a test name or a source name would be written; "
                             "nothing written")


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("build", "check", "summary"))
    ap.add_argument("--set", default=SET)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--harness", type=Path, default=ROOT.parent / "rlm-harness")
    ap.add_argument("--only", default=",".join(TRACKS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="replace")
    try:
        only = [t.strip() for t in a.only.split(",") if t.strip()]
        bad = [t for t in only if t not in TRACKS]
        if bad:
            raise SetupError(f"--only takes {', '.join(TRACKS)}; not {', '.join(bad)}")
        root = a.data or data.root()
        if a.data:
            os.environ[data.DATA_ENV] = str(a.data)
        paths = Paths(Path(root).resolve(), a.set, a.harness.resolve())
        if a.command == "build":
            return cmd_build(paths, only, a.jobs, not a.no_validate, a.force)
        check_outside(paths)
        if a.command == "check":
            problems = cmd_check(paths)
            for p in problems:
                print(p)
            print(f"check: {len(problems)} problem{'s' if len(problems) != 1 else ''}")
            return EXIT_PROBLEMS if problems else EXIT_OK
        print(f"summary: {cmd_summary(paths)}")
        return EXIT_OK
    except SetupError as e:
        print(f"setup: {e}", file=sys.stderr)
        return EXIT_SETUP
    except Short as e:
        print(f"stopped: {e}", file=sys.stderr)
        return EXIT_PROBLEMS
    except FileNotFoundError as e:
        print(f"setup: {e}", file=sys.stderr)
        return EXIT_SETUP


if __name__ == "__main__":
    sys.exit(main())
