"""edit_real: code-editing tasks in real repositories, from the validated task index (spec §12.7-12.8).

scripts/build_edit_data.py builds everything this module reads, on the data drive (never in the
repository or git): each repository's derived image (`rlm-edit-<repo>:<hash>`, editbox.py), a
snapshot of its checkout at the image's commit, and `edit/index.json`, the tasks that passed
validation in our grader (spec §12.8). Three kinds:

- `bug`: a SWE-smith instance: a bug introduced into the source, an issue text describing the
  symptom, and the tests it breaks (moved into the hidden set).
- `feature`: generated here: a public function that existing tests cover is removed, and the issue
  asks for it back, from its docstring and what the tests check (decision E-f).
- `real_pr`: a real issue and its merged fix, from SWE-rebench V2, with the fix's tests hidden.

Splits: `edit_real` trains; `edit_real_select` (selection-v2) and `edit_real_test`
(heldout-test-v2) come from repositories training never sees, and are eval-only (H7).

Levels follow the gold patch: 1 = one file, up to ~10 changed lines; 2 = one file, up to ~40;
3 = two files; 4 = three or more files; 5 = multi-file changes of ~100+ lines. A level with no
task left falls back to the nearest one that has some *(mine)*.

A task's repository at its base is the snapshot with the task's break patch applied (the bug
introduced or the feature removed), the hidden test files taken out, and Rust's inline test
modules cut; the grader puts the tests back. Task retirement (spec §12.7) passes `exclude` to
`make_task`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import secrets
import shutil
import subprocess
import tarfile
from pathlib import Path

from ..env import Task

GENERATOR_VERSION = "editreal-v1"
FAMILIES = ("edit_real", "edit_real_select", "edit_real_test")
SPLIT = {"edit_real": "train", "edit_real_select": "select", "edit_real_test": "test"}
MAX_LEVEL = 5
INDEX = "edit/index.json"  # relative to the data root
KINDS = ("bug", "feature", "real_pr")
PERMISSIVE = ("MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "0BSD", "Unlicense", "Zlib", "MIT-0",
              "BSL-1.0", "PSF-2.0", "Python-2.0", "CC0-1.0")


def level_of(files: int, lines: int, feature: bool = False) -> int:
    """A task's level from its gold patch (spec §12.7)."""
    if files >= 2 and lines >= 100:
        return 5
    if files >= 3:
        return 4
    if files == 2:
        return 3
    return 1 if lines <= 10 else 2


def index_path() -> Path:
    from .. import data

    return data.path(INDEX)


@functools.lru_cache(maxsize=2)
def _load(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index() -> dict:
    path = index_path()
    if not path.exists():
        raise FileNotFoundError(f"no edit_real task index at {path}: build it with scripts/build_edit_data.py")
    return _load(str(path), path.stat().st_mtime)


def records(split: str) -> list[dict]:
    return [r for r in index()["tasks"] if r["split"] == split]


def training_sources() -> tuple[str, ...]:
    """The data sources the training split reads (rlm_rl/data.py SOURCES), for the licence bar (7b)."""
    try:
        return tuple(sorted({r["source"] for r in records("train")}))
    except FileNotFoundError:
        return ()


def check_licences() -> None:
    """Every training repository's own licence file is permissive (7b, spec §12.7)."""
    for r in records("train"):
        if r.get("licence") not in PERMISSIVE:
            raise ValueError(f"edit_real: {r['repo']} is licensed {r.get('licence')!r}, which the licence bar keeps "
                             "out of training (7b)")


def _pick(pool: list[dict], level: int, rng: random.Random) -> dict:
    by_level: dict[int, list] = {}
    for r in pool:
        by_level.setdefault(int(r["level"]), []).append(r)
    if not by_level:
        raise ValueError("no edit_real task left to draw")
    best = min(by_level, key=lambda lv: (abs(lv - level), lv > level))  # the nearest level, a lower one on a tie
    return rng.choice(sorted(by_level[best], key=lambda r: r["id"]))


def make_task(seed: int, family: str, level: int = 1, task_id: str | None = None, exclude=()) -> Task:
    """A task from the family's split at (or nearest) `level`, chosen by the seed, skipping retired ids."""
    if family not in FAMILIES:
        raise ValueError(f"unknown edit_real family {family!r}")
    excluded = set(exclude or ())
    pool = [r for r in records(SPLIT[family]) if r["id"] not in excluded]
    return to_task(_pick(pool, level, random.Random(seed)), family, task_id)


def eval_tasks(split: str, family: str) -> list[Task]:
    """Every task of an eval-only split, in a fixed order (selection-v2, heldout-test-v2)."""
    return [to_task(r, family, f"{family}-{i:03d}") for i, r in enumerate(sorted(records(split), key=lambda r: r["id"]))]


def snapshot_path(record: dict) -> Path:
    from .. import data

    return data.path(record["snapshot"])


def _build_workspace(record: dict, folder: Path) -> None:
    """The task's repository at its base: snapshot, break patch, hidden tests out, inline test modules cut."""
    from ..editbox import git, git_env
    from ..grader import safe_path

    with tarfile.open(snapshot_path(record)) as tar:
        members = [m for m in tar.getmembers() if safe_path(m.name.lstrip("./") or "x") and not m.isdev()
                   and (not m.issym() or not Path(m.linkname).is_absolute())]
        tar.extractall(folder, members=members, filter="tar")
    if record.get("break_patch"):
        scratch = folder.parent / f"{folder.name}.git-{secrets.token_hex(4)}"
        env = git_env(folder.parent)
        try:
            git("init", "-q", "--bare", str(scratch), env=env)
            out = git("apply", "--whitespace=nowarn", "-", env=dict(env, GIT_DIR=str(scratch), GIT_WORK_TREE=str(folder)),
                      cwd=folder, check=False, input=record["break_patch"].encode())
            if out.returncode:
                raise RuntimeError(f"{record['id']}: the break patch doesn't apply: {out.stderr.decode()[-300:]}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    for rel in record.get("node_modules") or []:  # JavaScript: the installed dependencies live in the image
        link = folder / rel
        if not link.exists() and not link.is_symlink():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(f"/opt/rlm-deps/{rel}")
    for rel in record.get("remove") or []:
        target = folder / rel
        if target.is_file() or target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    for rel, text in (record.get("replace") or {}).items():  # Go: the test file without its hidden test functions
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    for rel, block in (record.get("tests") or {}).get("append", {}).items():
        target = folder / rel
        text = target.read_text(encoding="utf-8")
        if not text.endswith(block):
            raise RuntimeError(f"{record['id']}: {rel} doesn't end with its inline test module")
        target.write_text(text[: len(text) - len(block)], encoding="utf-8")


def _context(record: dict) -> str:
    listing = "\n".join(f"{p}  {s:,}" for p, s in record["files"])
    return f"{record['issue']}\n\nRepository files (path, size in bytes):\n{listing}\n"


def to_task(record: dict, family: str, task_id: str | None = None) -> Task:
    from ..editbox import Workspace

    key_src = json.dumps({k: record.get(k) for k in ("id", "image", "snapshot", "snapshot_sha256", "break_patch",
                                                     "remove", "replace")}, sort_keys=True)
    key = "real-" + hashlib.sha256((GENERATOR_VERSION + key_src).encode()).hexdigest()[:24]
    grading = {"type": "edit", "kind": record["kind"], "language": record["language"], "repo": record["repo"],
               "tests": record["tests"], "gold_patch": record.get("gold_patch"), "source": record.get("source"),
               "instance_id": record.get("instance_id"), "record_id": record["id"],
               "files_to_change": record.get("gold_files"), "gold_lines": record.get("gold_lines")}
    edit = {"language": record["language"], "n_files": len(record["files"]), "test_hint": record["test_hint"],
            "image": record["image"], "repo": record["repo"], "kind": record["kind"]}
    if record.get("cell_timeout_s"):  # this task's own tests need longer cells (measured under load, spec §12.3)
        edit["cell_timeout_s"] = int(record["cell_timeout_s"])
    return Task(task_id or record["id"], family, record["issue"], _context(record), answer=None,
                level=int(record["level"]), grading=grading, edit=edit,
                workspace=Workspace(key, functools.partial(_build_workspace, record)))
