"""Repository bug localization from Long Code Arena (plan step 7, the 15% repo share of decision 7a).

Each task is a real GitHub issue and the Python files of its repository at the commit the issue was filed
against. The answer is the set of files the fix changed, scored by set-F1 (research §3.3). LCA's dataset is
Apache-2.0 and every repo in it is MIT or Apache-2.0, so it passes the licence bar (7b).

LCA ships repository archives (with full git history) only for its 48 test repos. 40 of them also have items
in LCA's Python train split: 698 issues, which are the ones usable here. `scripts/build_data.py lca-git`
unpacks each archive's git data once, and `lca-index` lists every usable item with the .py files at its base
commit. Git history never reaches the sandbox: only the rendered context does, and it holds files at the base
commit and nothing from the fix.

A repository is usually bigger than a level allows (the median is ~0.9M characters), so a context is the gold
files plus other files from the same repository up to the level's size. Each level up is longer, and draws
more of its filler from the gold files' own directories, which look most like the answer.

Families:
- `repo_loc`: training, from the train split, minus the selection repos.
- `repo_loc_select`: the selection set's repos (keras, modin, sanic), never trained on.
- `repo_loc_test`: LCA's Python test split, held-out test only (H7), with a separately written question.

Entities can't be renamed per seed here, because the code is real. The base model may have seen these
repositories, or even the fixes, in pretraining *(inferred)*; the held-out test reports which test repos also
appear in training.
"""

from __future__ import annotations

import ast
import functools
import gzip
import json
import random
import re
import secrets
import subprocess
from pathlib import Path

from .. import data
from ..env import Task

GENERATOR_VERSION = "repoloc-v1"
FAMILIES = ("repo_loc", "repo_loc_select", "repo_loc_test")
TRAIN_SOURCES = ("lca-bug-localization",)
LEVELS = {1: (32_000, 40_000), 2: (70_000, 90_000), 3: (130_000, 170_000), 4: (230_000, 290_000),
          5: (360_000, 420_000)}
MAX_LEVEL = max(LEVELS)
NEAR_SHARE = {1: 0.2, 2: 0.35, 3: 0.5, 4: 0.6, 5: 0.7}  # filler drawn from the gold files' directories
GOLD_SHARE = 0.5  # the gold files may fill at most half of a level's smallest context
SELECTION_REPOS = ("keras-team__keras", "modin-project__modin", "sanic-org__sanic")
REPO_CAP = 60  # a repository's weight when drawing an item is its item count, capped, so jax doesn't dominate
MAX_GOLD = 5
MAX_FILE_CHARS = 120_000  # bigger non-gold files (generated code, vendored data) are left out
ISSUE_MAX_CHARS = 4_000

TRAIN_QUESTIONS = ("Here is a bug report for the code base in `context`:\n\n<issue>\n{issue}\n</issue>\n\nWhich files "
                   "have to change to fix it? Answer with a list of file paths, written exactly as in the "
                   "`# ===== file: ... =====` headers.",
                   "A user filed this issue against the repository in `context`:\n\n<issue>\n{issue}\n</issue>\n\nList "
                   "the files a fix would modify. Answer with a list of file paths, written exactly as in the "
                   "`# ===== file: ... =====` headers.")
HELDOUT_QUESTIONS = ("Bug report:\n\n{issue}\n\nName the source files that a patch resolving this report would edit. "
                     "Reply with their paths, comma separated, copied from the file header lines.",)


# ----------------------------------------------------------------------------- git


def _git_dir(repo: str) -> Path:
    return data.derived("lca-git", f"{repo}.git")


def _git(repo: str, *args: str, stdin: bytes | None = None) -> bytes:
    cmd = ["git", "-c", "safe.directory=*", "-c", "core.quotepath=off", "--git-dir", str(_git_dir(repo)), *args]
    return subprocess.run(cmd, input=stdin, capture_output=True, check=True, timeout=300).stdout


def ls_py(repo: str, sha: str) -> list[tuple[str, int, str]]:
    """(path, size, blob) for every .py file at a commit."""
    out = []
    for entry in _git(repo, "ls-tree", "-r", "-l", "-z", sha).split(b"\0"):
        if not entry:
            continue
        meta, _, path = entry.decode("utf-8", "replace").partition("\t")
        mode, kind, blob, size = meta.split()
        if kind == "blob" and path.endswith(".py") and size != "-":
            out.append((path, int(size), blob))
    return out


def read_blobs(repo: str, blobs: list[str]) -> dict[str, str]:
    """Blob contents by id, through one `git cat-file --batch`."""
    raw = _git(repo, "cat-file", "--batch", stdin="".join(b + "\n" for b in blobs).encode())
    out, pos = {}, 0
    for _ in blobs:
        end = raw.index(b"\n", pos)
        blob, _, size = raw[pos:end].decode().split()
        start = end + 1
        out[blob] = raw[start:start + int(size)].decode("utf-8", "replace")
        pos = start + int(size) + 1
    return out


# ----------------------------------------------------------------------------- the index (scripts/build_data.py)


def index_item(row: dict, repo: str, listings: dict) -> tuple[str | None, dict | None]:
    """An LCA row as an index entry, or the reason it can't be used."""
    if not (_git_dir(repo) / ".unpacked").exists():
        return "no repo archive", None
    try:
        gold = sorted(set(ast.literal_eval(row["changed_files"])))
    except (ValueError, SyntaxError):
        return "unreadable changed_files", None
    if not 1 <= len(gold) <= MAX_GOLD:
        return f"not 1-{MAX_GOLD} changed files", None
    if not all(p.endswith(".py") for p in gold):
        return "a changed file isn't Python", None
    key = (repo, row["base_sha"])
    if key not in listings:
        try:
            listings[key] = ls_py(repo, row["base_sha"])
        except subprocess.CalledProcessError:
            listings[key] = None
    files = listings[key]
    if files is None:
        return "base commit missing", None
    sizes = {path: size for path, size, _ in files}
    if any(g not in sizes for g in gold):
        return "a changed file is new in the fix", None
    gold_chars = sum(sizes[g] for g in gold)
    if gold_chars > GOLD_SHARE * LEVELS[MAX_LEVEL][0]:
        return "gold files too big for any level", None
    issue = f"{row['issue_title'].strip()}\n\n{(row['issue_body'] or '').strip()}".strip()
    if len(issue) < 40:
        return "issue too short", None
    py_chars = sum(s for p, s, _ in files if s <= MAX_FILE_CHARS or p in gold)
    return None, {"id": str(row["id"]), "repo": repo, "base_sha": row["base_sha"], "issue": issue, "gold": gold,
                  "gold_chars": gold_chars, "py_chars": py_chars, "licence": row["repo_license"],
                  "files": [[p, s, b] for p, s, b in files]}


@functools.lru_cache(maxsize=None)
def items(split: str) -> tuple[dict, ...]:
    with gzip.open(data.derived("lca", f"index-{split}.jsonl.gz"), "rt", encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f if line.strip())


def pool(family: str) -> tuple[dict, ...]:
    if family == "repo_loc":
        return tuple(it for it in items("train") if it["repo"] not in SELECTION_REPOS)
    if family == "repo_loc_select":
        return tuple(it for it in items("train") if it["repo"] in SELECTION_REPOS)
    if family == "repo_loc_test":
        return items("test")
    raise ValueError(f"unknown repo-localization family {family!r}")


def eligible(item: dict, level: int) -> bool:
    lo = LEVELS[level][0]
    return item["gold_chars"] <= GOLD_SHARE * lo and item["py_chars"] >= lo


# ----------------------------------------------------------------------------- tasks


def _draw(rng: random.Random, family: str, level: int) -> dict:
    candidates = [it for it in pool(family) if eligible(it, level)]
    if not candidates:  # a small held-out pool may have nothing big enough at this level
        candidates = [it for it in pool(family) if it["gold_chars"] <= GOLD_SHARE * LEVELS[level][1]]
    by_repo: dict[str, list[dict]] = {}
    for it in candidates:
        by_repo.setdefault(it["repo"], []).append(it)
    repos = sorted(by_repo)
    repo = rng.choices(repos, [min(len(by_repo[r]), REPO_CAP) for r in repos])[0]
    return rng.choice(sorted(by_repo[repo], key=lambda it: it["id"]))


def _select_files(rng: random.Random, item: dict, level: int) -> list[tuple[str, int, str]]:
    lo, hi = LEVELS[level]
    budget = rng.randint(lo, hi)
    files = {p: (p, s, b) for p, s, b in item["files"]}
    chosen = [files[g] for g in item["gold"]]
    total = sum(s for _, s, _ in chosen)
    gold_dirs = {g.rpartition("/")[0] for g in item["gold"]}
    parents = {d.rpartition("/")[0] for d in gold_dirs}
    rest = [f for p, f in files.items() if p not in item["gold"] and f[1] <= MAX_FILE_CHARS and f[1] > 0]
    near = [f for f in rest if f[0].rpartition("/")[0] in gold_dirs | parents]
    far = [f for f in rest if f not in near]
    rng.shuffle(near)
    rng.shuffle(far)
    near_budget = total + NEAR_SHARE[level] * (budget - total)
    for group, limit in ((near, near_budget), (far, budget), (near, budget)):
        for f in group:
            if total >= limit:
                break
            if f not in chosen and total + f[1] <= hi:
                chosen.append(f)
                total += f[1]
    return sorted(chosen)


def _blind(paths: list[str], issue: str) -> list[str]:
    """The cheap guess: every context file whose name the issue text mentions (spec §3's blind strategy)."""
    text = issue.lower()
    out = []
    for path in paths:
        stem = path.rpartition("/")[2].removesuffix(".py").lower()
        if path.lower() in text or (len(stem) >= 3 and stem != "__init__" and re.search(rf"\b{re.escape(stem)}\b", text)):
            out.append(path)
    return out


def _build(rng: random.Random, family: str, level: int, task_id: str, item: dict | None = None) -> Task:
    if family not in FAMILIES:
        raise ValueError(f"unknown repo-localization family {family!r}")
    if level not in LEVELS:
        raise ValueError(f"no repo-localization level {level} (levels 1-{MAX_LEVEL})")
    item = item or _draw(rng, family, level)
    chosen = _select_files(rng, item, level)
    blobs = read_blobs(item["repo"], sorted({b for _, _, b in chosen}))
    context = "\n".join(f"# ===== file: {p} =====\n{blobs[b].rstrip()}\n" for p, _, b in chosen)
    issue = item["issue"]
    if len(issue) > ISSUE_MAX_CHARS:
        issue = issue[:ISSUE_MAX_CHARS].rstrip() + "\n[issue text truncated]"
    templates = HELDOUT_QUESTIONS if family == "repo_loc_test" else TRAIN_QUESTIONS
    question = rng.choice(templates).format(issue=issue)
    paths = [p for p, _, _ in chosen]
    blind = _blind(paths, issue)
    # When the issue names exactly the files the fix changed, matching names solves the task without reading code.
    grading = {"type": "list", "item": "entity", "blind": blind, "kind": "repo_loc",
               "code_only": set(blind) == set(item["gold"]), "repo": item["repo"], "lca_id": item["id"],
               "files": len(paths)}
    return Task(task_id, family, question, context, answer=list(item["gold"]), level=level, grading=grading)


def make_task(seed: int, family: str, level: int = 1, task_id: str | None = None) -> Task:
    """One task from a seed. The task id never contains the seed (H3)."""
    task_id = task_id or f"{family}-L{level}-{secrets.token_hex(4)}"
    return _build(random.Random(seed), family, level, task_id)


def top_level(item: dict) -> int:
    """The highest level an item fills, or 1 if it fills none."""
    return max((level for level in LEVELS if eligible(item, level)), default=1)


def test_tasks(seeds: list[int], prefix: str) -> list[Task]:
    """LCA's Python test split, each item once, at the highest level it fills (the held-out test)."""
    test = sorted(items("test"), key=lambda it: it["id"])
    if len(seeds) != len(test):
        raise ValueError(f"need {len(test)} seeds, got {len(seeds)}")
    return [_build(random.Random(seed), "repo_loc_test", top_level(it), f"{prefix}-repo_loc_test-{i:02d}", item=it)
            for i, (it, seed) in enumerate(zip(test, seeds))]
