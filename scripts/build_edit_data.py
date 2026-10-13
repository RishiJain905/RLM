"""Build the edit_real task data on the data drive (spec §12.7-12.8, plan step E4). Run inside WSL2 with Docker.

    .venv-linux/bin/python scripts/build_edit_data.py runtime                # the standalone Python bundle
    .venv-linux/bin/python scripts/build_edit_data.py pull python go         # source images (network on)
    .venv-linux/bin/python scripts/build_edit_data.py derive python go       # derived images, snapshots, licences
    .venv-linux/bin/python scripts/build_edit_data.py candidates python go   # SWE-smith rows -> candidate tasks
    .venv-linux/bin/python scripts/build_edit_data.py rebench python go typescript   # SWE-rebench V2 real PRs
    .venv-linux/bin/python scripts/build_edit_data.py validate python --jobs 4
    .venv-linux/bin/python scripts/build_edit_data.py index                  # edit/index.json
    .venv-linux/bin/python scripts/build_edit_data.py manifest               # edit/manifest.json
    .venv-linux/bin/python scripts/build_edit_data.py report

Everything goes under the data root's edit/ folder (D:\\RLM-data\\edit), never into the repository. Each stage
is resumable: it skips what is already done. `pull --print` prints the docker pull commands instead of running
them, for when the permission check refuses a bulk pull.
"""

import argparse
import asyncio
import collections
import itertools
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import data, editbox, editdata, grader  # noqa: E402
from rlm_rl.tasks import editreal  # noqa: E402

# Which repositories go where (code-editing-plan §4.1; my picks, with the coordinator's list). Training repositories
# give training tasks; `select` and `test` are held-out repositories training never sees (selection-v2,
# heldout-test-v2). Keys are SWE-smith's repo names without the commit.
PLAN = {
    "python": {"train": ["andialbrecht__sqlparse", "pallets__jinja", "marshmallow-code__marshmallow",
                         "arrow-py__arrow", "seperman__deepdiff", "tkrajina__gpxpy"],
               "select": ["mahmoud__glom"], "test": ["lepture__mistune", "pyparsing__pyparsing"]},
    "go": {"train": ["go-chi__chi", "Masterminds__squirrel", "caarlos0__env", "allegro__bigcache",
                     "hypermodeinc__ristretto", "c-bata__go-prompt"],
           # go-critic was the second test repository, but DeepSWE v1.1 has a go-critic task, so bitset replaces it
           "select": ["hjson__hjson-go"], "test": ["RoaringBitmap__roaring", "bits-and-blooms__bitset"]},
    # JavaScript, TypeScript, Rust and Java train only on tasks we generate (Rishi, 2026-10-12): their SWE-smith rows
    # are eval-only, so only the select and test repositories use them.
    "javascript": {"train": ["caolan__async", "iamkun__dayjs", "svg__svgo"], "select": ["websockets__ws"],
                   "test": ["nock__nock", "josdejong__mathjs"]},
    "typescript": {"train": ["ReactiveX__rxjs", "react-hook-form__react-hook-form", "foambubble__foam"],
                   "select": ["Redocly__redoc"], "test": ["homebridge__homebridge", "jaredpalmer__formik"]},
    "rust": {"train": ["chronotope__chrono", "rust-itertools__itertools", "dtolnay__semver", "marshallpierce__rust-base64"],
             "select": ["uuid-rs__uuid"], "test": ["BurntSushi__rust-csv", "sharkdp__pastel"]},
    "java": {"train": ["json-path__JsonPath", "remkop__picocli", "dropwizard__metrics"],
             "select": ["LMAX-Exchange__disruptor"], "test": ["joelittlejohn__jsonschema2pojo", "jankotek__mapdb"]},
}
# Tasks wanted per repository and split, before validation drops some (about a third extra). Per language this aims
# at code-editing-plan §4.1's ~90 / ~60 / ~60 (JS and TS together) / ~45 / ~35 training tasks.
WANT = {"train": {"python": 15, "go": 10, "javascript": 10, "typescript": 10, "rust": 11, "java": 12},
        "select": {"python": 2, "go": 2, "javascript": 1, "typescript": 2, "rust": 1, "java": 1},
        "test": {"python": 5, "go": 4, "javascript": 3, "typescript": 6, "rust": 3, "java": 3}}
SMITH_TRAIN = ("python", "go")  # languages whose SWE-smith rows may train (their dataset cards are MIT)
# Real pull requests from SWE-rebench V2 (code-editing-plan §4.1: ~24, 8 each in Python, Go and TypeScript), training
# only. Candidates per repository, the most recent first; about 10 per language for 8 that validate. Each task has its
# own image (~0.8-1.3 GB compressed, ~0.5 GB of it its own), so the repositories are few and the candidates per
# repository several, to share layers. Picked from the rows that pass rebench_rows(): permissive per-task licence, no
# flagged problems, a high-confidence issue text, small fixes, simple test commands, and not in an eval we report.
REBENCH = {"python": {"pallets/click": 4, "Python-Markdown/markdown": 3, "python-attrs/attrs": 3},
           "go": {"getkin/kin-openapi": 4, "goccy/go-yaml": 3, "fxamacker/cbor": 3},
           "typescript": {"typescript-eslint/tslint-to-eslint-config": 4, "marpple/FxTS": 3,
                          "remarkablemark/html-react-parser": 3}}
REBENCH_LICENCES = {"MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "ISC", "0BSD", "Unlicense", "MIT-0", "BSL-1.0"}
OVERSAMPLE = 1.6
SOURCE_NAMES = {"python": "swe-smith-py", "go": "swe-smith-go", "javascript": "swe-smith-js", "typescript": "swe-smith-ts",
                "rust": "swe-smith-rs", "java": "swe-smith-java"}


def registry_path() -> Path:
    return editdata.edit_path("repos.json")


def update_registry(key: str, entry: dict) -> None:
    """Write one repository's entry. Stages run side by side (derive for two language groups at once), so each write
    re-reads the file under a lock and changes only its own key."""
    import fcntl

    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = editdata.read_json(path, {})
        reg[key] = entry
        editdata.write_json(path, reg)


# How a repository's tests run when the derive stage's guess (from the files and packages it finds) is wrong: monorepos
# whose tests run from one package with their own configuration. `test_cwd` is that package, relative to the root.
REPO_TESTS = {
    # Vitest is installed, but the library's specs (`*-spec.ts`) run with mocha from packages/rxjs; apps/rxjs.dev's
    # specs need Karma and a browser, so its functions aren't used.
    "ReactiveX/rxjs": {"runner": "mocha", "bin": ["mocha"], "args": ["--config", "spec/support/.mocharc.js"],
                       "env": {"TS_NODE_PROJECT": "tsconfig.mocha.json"}, "test_cwd": "packages/rxjs",
                       "test_hint": "cd packages/rxjs && TS_NODE_PROJECT=tsconfig.mocha.json npx mocha "
                                    "--config spec/support/.mocharc.js 'spec/**/*-spec.ts'"},
}


def registry() -> dict:
    """repos.json, with REPO_TESTS laid over the entries it names."""
    reg = editdata.read_json(registry_path(), {})
    for entry in reg.values():
        entry.update(REPO_TESTS.get(entry.get("repo"), {}))
    return reg


def plan_repos(languages):
    for language in languages:
        for split, repos in PLAN[language].items():
            for repo in repos:
                yield language, split, repo


def image_for(language: str, repo: str) -> str:
    rows = editdata.swe_smith_rows(language, repo + ".", columns=["repo", "image_name"])
    images = sorted({r["image_name"] for r in rows})
    if len(images) != 1:
        raise SystemExit(f"{repo}: expected one image, found {images}")
    return images[0].split(":")[0]


# ----------------------------------------------------------------------------- stages


def cmd_runtime(args) -> int:
    print("bundle:", editbox.build_python_bundle())
    print("runtime:", editbox.runtime_dir())
    return 0


def cmd_pull(args) -> int:
    for language, split, repo in plan_repos(args.languages):
        image = image_for(language, repo)
        if args.print:
            print(f"docker pull {image}")
            continue
        have = subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
        if have:
            print(f"have  {image}")
            continue
        t = time.time()
        out = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True)
        print(f"{'pulled' if out.returncode == 0 else 'FAILED'} {image} ({time.time() - t:.0f} s) {out.stderr[-200:]}",
              flush=True)
    return 0


def cmd_derive(args) -> int:
    reg = registry()
    for language, split, repo in plan_repos(args.languages):
        image = image_for(language, repo)
        key = editdata.repo_key(image)
        entry = reg.get(key, {})
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
            print(f"missing {image}: pull it first")
            continue
        snapshot = editdata.edit_path("repos", f"{key}.tar")
        tag = None
        for attempt in range(3):  # the data drive is slow under load: BuildKit and docker run can time out
            try:
                if not snapshot.exists() or args.force:
                    entry.update(editdata.export_snapshot(image, language, snapshot))
                elif "sha256" not in entry:  # a snapshot left by a run that stopped before recording it
                    entry.update(editdata.snapshot_stats(snapshot, image, language))
                tag = editdata.derive(image, language, warm=REPO_TESTS.get(repo.replace("__", "/", 1), {}).get("warm"))
                break
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                print(f"retry {key} ({attempt + 1}/3): {str(exc)[-200:]}", flush=True)
                time.sleep(30)
        if tag is None:
            print(f"FAILED {key}: not derived", flush=True)
            continue
        licence, files = editdata.snapshot_licence(snapshot)
        entry.update(editdata.detect_runner(snapshot, language))
        entry.update(language=language, split=split, repo=repo.replace("__", "/", 1), source=image, tag=tag,
                     snapshot=f"{editdata.EDIT_DIR}/repos/{key}.tar", licence=licence, licence_files=files,
                     permissive=editdata.permissive(licence))
        reg[key] = entry
        update_registry(key, entry)
        print(f"{key:45s} {split:6s} {licence:18s} {entry['files']:6d} files {entry['bytes'] / 1e6:7.1f} MB  {tag}", flush=True)
    return 0


def _scope_python(f2p: list[str]) -> list[str]:
    return sorted({t.split("::")[0] for t in f2p})


def candidates_for(language: str, key: str, entry: dict, want: int, rng: random.Random) -> list[dict]:
    """SWE-smith bug rows of one repository as candidate records, spread over levels."""
    repo = entry["repo"].replace("/", "__", 1)
    rows = editdata.swe_smith_rows(language, repo + ".")
    snapshot = data.path(entry["snapshot"])
    files = editdata.snapshot_files(snapshot)
    texts_cache: dict[str, str] = {}
    import tempfile

    tree = Path(tempfile.mkdtemp(prefix="rlm-cand-"))
    import tarfile
    with tarfile.open(snapshot) as tar:
        tar.extractall(tree, filter="tar")
    pool = []
    for r in rows:
        issue = (r.get("problem_statement") or "").strip()
        if not issue or not r["FAIL_TO_PASS"] or len(r["FAIL_TO_PASS"]) > 25:
            continue
        nfiles, nlines = editdata.patch_size(r["patch"])
        if nfiles == 0 or nlines > 400:
            continue
        if any(grader.is_test_path(fp.path, language) for fp in grader.parse_patch(r["patch"])):
            continue
        pool.append((editreal.level_of(nfiles, nlines), r))
    rng.shuffle(pool)
    by_level = collections.defaultdict(list)
    for level, r in pool:
        by_level[level].append(r)
    picked = []
    while len(picked) < want and any(by_level.values()):
        for level in sorted(by_level):
            if by_level[level] and len(picked) < want:
                picked.append((level, by_level[level].pop()))
    out = []
    for level, r in picked:
        f2p = list(r["FAIL_TO_PASS"])
        append: dict = {}
        if language == "python":
            scope = _scope_python(f2p)
            remove = [p for p in scope if p in files]
            if len(remove) != len(scope):
                continue
            cwd = "."
        elif language == "go":
            need = {p for p in files if p.endswith("_test.go")}
            if not texts_cache:
                texts_cache.update(editdata.snapshot_read(snapshot, need))
            where = editdata.go_test_files(f2p, texts_cache)
            if not where or len(where) < len({n.split("/")[0] for n in f2p}):
                continue
            remove = sorted({p for ps in where.values() for p in ps})
            scope = sorted({"./" + (str(Path(p).parent) if str(Path(p).parent) != "." else "") for p in remove})
            scope = [s.rstrip("/") if s != "./" else "." for s in scope]
            cwd = "."
        else:
            mapped = scope_other(language, f2p, files, snapshot, texts_cache)
            if mapped is None:
                continue
            scope, remove, append, cwd = mapped
        try:
            gold = editdata.reverse_patch(r["patch"], tree)
        except ValueError:
            continue
        hidden = editdata.snapshot_read(snapshot, remove)
        nfiles, nlines = editdata.patch_size(gold)
        rid = f"{language[:2]}-{repo.split('__')[1]}-{r['instance_id'].rsplit('.', 1)[-1]}".lower()
        out.append({
            "id": rid, "split": entry["split"], "language": language, "repo": entry["repo"], "key": key,
            "kind": "bug", "source": SOURCE_NAMES[language], "instance_id": r["instance_id"],
            "licence": entry["licence"], "image": entry["tag"], "snapshot": entry["snapshot"],
            "snapshot_sha256": entry["sha256"], "break_patch": r["patch"], "gold_patch": gold, "remove": remove,
            "issue": r["problem_statement"].strip(), "node_modules": entry.get("node_modules", []),
            "tests": grader.TestSpec(runner=entry.get("runner") or editdata.LANGS[language]["runner"], scope=scope,
                                     f2p=[], p2p=[], language=language, hidden=hidden, append=append, cwd=cwd,
                                     bin=entry.get("bin") or [], args=entry.get("args") or [],
                                     junit=entry.get("junit")).as_dict(),
            "test_hint": entry.get("test_hint") or editdata.LANGS[language]["test_hint"], "level": level,
            "gold_files": nfiles, "gold_lines": nlines, "smith_f2p": f2p, "files": [],
        })
    import shutil
    shutil.rmtree(tree, ignore_errors=True)
    return out


def scope_other(language: str, f2p: list[str], files: dict, snapshot: Path, cache: dict):
    """Where an eval row's failing tests live, for Java, Rust, JavaScript and TypeScript:
    (scope, files to hide, inline test blocks to cut and re-append, cwd), or None when it can't be pinned down."""
    if language == "java":
        classes = sorted({t.rsplit(".", 1)[0] for t in f2p})
        remove, roots = [], set()
        for cls in classes:
            tail = "src/test/java/" + cls.replace(".", "/") + ".java"
            hits = [p for p in files if p == tail or p.endswith("/" + tail)]
            if len(hits) != 1:
                return None
            remove.append(hits[0])
            roots.add(hits[0][: -len(tail)].rstrip("/") or ".")
        if len(roots) != 1:
            return None
        return classes, sorted(remove), {}, roots.pop()
    if language == "rust":
        if not cache:
            cache.update(editdata.snapshot_read(snapshot, [p for p in files if p.endswith(".rs")]))
        append, remove, scope, crates = {}, [], set(), set()
        for t in f2p:
            name = t.split("::")[-1]
            hits = [p for p, text in cache.items() if re.search(rf"\bfn {re.escape(name)}\s*\(", text)]
            if not hits:
                return None
            path = hits[0] if len(hits) == 1 else next((h for h in hits if "::" in t and h.endswith(t.split("::")[0] + ".rs")), None)
            if path is None:
                return None
            crate = next((str(Path(path).parents[i]) for i in range(len(Path(path).parents))
                          if (str(Path(path).parents[i]) + "/Cargo.toml").lstrip("./") in files
                          or (str(Path(path).parents[i]) == "." and "Cargo.toml" in files)), None)
            if crate is None:
                return None
            crates.add(crate)
            rel = path[len(crate) + 1:] if crate != "." else path
            if rel.startswith("tests/"):
                remove.append(path)
                scope.update(["--test", Path(rel).stem])
            else:
                lines = cache[path].split("\n")
                blocks = grader._rust_test_blocks(lines)
                last = [b for b in blocks if b[1] >= len([l for l in lines if l.strip()]) - 1 or b[1] >= len(lines) - 2]
                if not last:
                    return None
                start = last[-1][0] - 1
                block = "\n".join(lines[start:])
                if not cache[path].endswith(block):
                    return None
                append[path] = "\n" + block if not block.startswith("\n") else block
                scope.add("--lib")
        if len(crates) != 1:
            return None
        flat = ["--lib"] if "--lib" in scope else []
        for i, s in enumerate(sorted(x for x in scope if x not in ("--lib", "--test"))):
            flat += ["--test", s]
        return flat, sorted(set(remove)), append, crates.pop()
    if language in ("javascript", "typescript"):
        tests = [p for p in files if grader.is_test_path(p, language) and p.endswith((".js", ".ts", ".tsx", ".mjs", ".cjs", ".jsx"))
                 and "/node_modules/" not in p]
        if not cache:
            cache.update(editdata.snapshot_read(snapshot, tests))
        remove = set()
        for t in f2p:
            leaf = t.split(" > ")[-1]
            hits = [p for p, text in cache.items() if leaf in text]
            if not hits or len(hits) > 3:
                return None
            remove.update(hits)
        return sorted(remove), sorted(remove), {}, "."
    return None


def rebench_rows(language: str) -> list[dict]:
    """SWE-rebench V2 rows of REBENCH's repositories that pass the bar (code-editing-plan §4.1)."""
    import pyarrow.parquet as pq

    lang = {"typescript": "ts"}.get(language, language)
    cols = ["instance_id", "repo", "language", "license", "image_name", "meta", "install_config", "FAIL_TO_PASS",
            "PASS_TO_PASS", "created_at", "problem_statement", "patch", "test_patch"]
    rows = []
    for f in sorted(data.path("hf", "nebius__SWE-rebench-V2", "data").glob("*.parquet")):
        for r in pq.read_table(f, columns=cols).to_pylist():
            if (r["language"] or "").lower() != lang or r["repo"] not in REBENCH[language]:
                continue
            meta = r["meta"] or {}
            llm = meta.get("llm_metadata") or {}
            flagged = llm.get("detected_issues") or {}
            if r["license"] not in REBENCH_LICENCES or any(flagged.values()) or (llm.get("confidence") or 0) < 0.9:
                continue
            if (meta.get("num_modified_files") or 0) > 4 or (meta.get("num_modified_lines") or 0) > 200:
                continue
            if not r["FAIL_TO_PASS"] or len(r["FAIL_TO_PASS"]) > 30 or len((r["problem_statement"] or "").strip()) < 200:
                continue
            rows.append(r)
    return rows


def rebench_key(row: dict) -> str:
    return f"rb-{row['repo'].replace('/', '__')}-{row['instance_id'].rsplit('-', 1)[-1]}".lower()


def _applies(tree: Path, patch: str, scratch: Path) -> tuple[bool, str]:
    """Whether a patch applies to `tree` (git apply --check, reading no config)."""
    gitdir = scratch / "checkgit"
    env = editbox.git_env(scratch)
    editbox.git("init", "-q", "--bare", str(gitdir), env=env)
    out = editbox.git("apply", "--check", "--whitespace=nowarn", "--recount", "-",
                      env=dict(env, GIT_DIR=str(gitdir), GIT_WORK_TREE=str(tree)), cwd=tree, check=False,
                      input=patch.encode("utf-8", "surrogateescape"))
    return out.returncode == 0, out.stderr.decode("utf-8", "replace")[-300:]


def _patched_files(tree: Path, patch: str) -> dict[str, str]:
    """{path: text} of the files a patch leaves, applied to a copy of `tree` (deleted files left out)."""
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="rlm-rb-")) / "t"
    shutil.copytree(tree, work, symlinks=True)
    try:
        ok, why = grader._safe_apply(work, patch, work.parent)
        if not ok:
            raise ValueError(f"the test patch doesn't apply: {why}")
        out = {}
        for fp in grader.parse_patch(patch):
            target = work / fp.path
            if target.is_file() and not fp.binary:
                out[fp.path] = target.read_text(encoding="utf-8", errors="replace")
        return out
    finally:
        shutil.rmtree(work.parent, ignore_errors=True)


def rebench_record(language: str, row: dict, key: str, entry: dict) -> dict | None:
    """A candidate record for one real PR: the image's checkout is the base, the merged fix is the gold patch, and the
    files its test patch leaves are the hidden tests (taken out of the base where they exist)."""
    import shutil
    import tarfile
    import tempfile

    snapshot = data.path(entry["snapshot"])
    tree = Path(tempfile.mkdtemp(prefix="rlm-rb-tree-"))
    try:
        with tarfile.open(snapshot) as tar:
            tar.extractall(tree / "t", filter="tar")
        base = tree / "t"
        ok, why = _applies(base, row["patch"], tree)
        if not ok:
            print(f"  skip {key}: the fix doesn't apply to the image's checkout ({why[:100]})")
            return None
        try:
            hidden = _patched_files(base, row["test_patch"])
        except ValueError as exc:
            print(f"  skip {key}: {exc}")
            return None
        if any(grader.is_test_path(fp.path, language) for fp in grader.parse_patch(row["patch"])):
            print(f"  skip {key}: the fix touches test files")
            return None
        tests = [p for p in hidden if grader.is_test_path(p, language)]
        if language == "go":
            scope = sorted({"./" + str(Path(p).parent) if str(Path(p).parent) != "." else "." for p in tests
                            if p.endswith("_test.go")})
        elif language == "python":
            scope = sorted(p for p in tests if p.endswith(".py"))
        else:
            scope = sorted(p for p in tests if p.endswith((".js", ".ts", ".tsx", ".mjs", ".cjs", ".jsx")))
        if not scope:
            print(f"  skip {key}: no test file to run in the test patch")
            return None
        remove = sorted(p for p in hidden if (base / p).is_file())
    finally:
        shutil.rmtree(tree, ignore_errors=True)
    nfiles, nlines = editdata.patch_size(row["patch"])
    return {
        "id": key, "split": "train", "language": language, "repo": row["repo"], "key": key, "kind": "real_pr",
        "source": "swe-rebench-v2", "instance_id": row["instance_id"], "licence": entry["licence"],
        "task_licence": row["license"], "image": entry["tag"], "snapshot": entry["snapshot"],
        "snapshot_sha256": entry["sha256"], "break_patch": "", "gold_patch": row["patch"], "remove": remove,
        "issue": row["problem_statement"].strip(), "node_modules": entry.get("node_modules", []),
        "tests": grader.TestSpec(runner=entry.get("runner") or editdata.LANGS[language]["runner"], scope=scope, f2p=[],
                                 p2p=[], language=language, hidden=hidden, cwd=".", bin=entry.get("bin") or [],
                                 junit=entry.get("junit")).as_dict(),
        "test_hint": entry.get("test_hint") or editdata.LANGS[language]["test_hint"],
        "level": editreal.level_of(nfiles, nlines), "gold_files": nfiles, "gold_lines": nlines,
        "rebench_f2p": list(row["FAIL_TO_PASS"]), "created_at": str(row["created_at"]), "files": [],
    }


def cmd_rebench(args) -> int:
    """Pull, snapshot and derive each picked real PR's image, and write its candidate record."""
    reg = registry()
    for language in args.languages:
        if language not in REBENCH:
            print(f"{language}: no real PRs planned")
            continue
        rows = rebench_rows(language)
        path = editdata.edit_path("candidates", f"{language}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        have = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        for repo, want in REBENCH[language].items():
            picked = sorted((r for r in rows if r["repo"] == repo), key=lambda r: str(r["created_at"]), reverse=True)
            added = 0
            for row in picked:
                if added >= want:
                    break
                key = rebench_key(row)
                if key in have:
                    added += 1
                    continue
                image = row["image_name"]
                if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
                    if args.print:
                        print(f"docker pull {image}")
                        continue
                    t = time.time()
                    out = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True)
                    print(f"{'pulled' if out.returncode == 0 else 'FAILED'} {image} ({time.time() - t:.0f} s)", flush=True)
                    if out.returncode:
                        continue
                wd = subprocess.run(["docker", "image", "inspect", "--format", "{{.Config.WorkingDir}}", image],
                                    capture_output=True, text=True).stdout.strip()
                if not wd.startswith("/") or wd == "/":
                    print(f"  skip {key}: no repository folder in the image ({wd!r})")
                    continue
                entry = reg.get(key, {})
                snapshot = editdata.edit_path("repos", f"{key}.tar")
                try:
                    if not snapshot.exists():
                        entry.update(editdata.export_snapshot(image, language, snapshot, repo_dir=wd))
                    elif "sha256" not in entry:
                        entry.update(editdata.snapshot_stats(snapshot, image, language, wd))
                    licence, files = editdata.snapshot_licence(snapshot)
                    if not editdata.permissive(licence):
                        print(f"  skip {key}: the repository's licence file reads {licence}")
                        continue
                    entry.update(editdata.detect_runner(snapshot, language))
                    if entry.get("runner") == "unsupported":
                        print(f"  skip {key}: no supported test runner")
                        continue
                    tag = editdata.derive(image, language, repo_dir=wd, name=key, rebench=True)
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    print(f"  skip {key}: {str(exc)[-200:]}")
                    continue
                entry.update(language=language, split="train", repo=repo, source=image, tag=tag, repo_dir=wd,
                             snapshot=f"{editdata.EDIT_DIR}/repos/{key}.tar", licence=licence, licence_files=files,
                             permissive=True, kind="real_pr", origin="swe-rebench-v2", task_licence=row["license"])
                reg[key] = entry
                update_registry(key, entry)
                rec = rebench_record(language, row, key, entry)
                if rec is None:
                    continue
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                have.add(key)
                added += 1
                print(f"{key:55s} {licence:14s} level {rec['level']}  {tag}", flush=True)
            print(f"{language} {repo}: {added} of {want} candidates")
    return 0


def cmd_prune(args) -> int:
    """Remove the images of real-PR candidates that failed validation (each has its own image; the data drive is
    limited). Only images this script pulled and derived for those tasks."""
    reg = registry()
    removed = 0
    for language in args.languages:
        vpath = editdata.edit_path("validated", f"{language}.jsonl")
        if not vpath.exists():
            continue
        results = {}
        for line in vpath.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            results[r["id"]] = r
        for key, e in sorted(reg.items()):
            if e.get("kind") != "real_pr" or e["language"] != language or key not in results or results[key]["ok"]:
                continue
            for image in (e["tag"], e["source"]):
                out = subprocess.run(["docker", "image", "rm", image], capture_output=True, text=True)
                print(f"{'removed' if out.returncode == 0 else 'kept   '} {image} {out.stderr.strip()[-120:]}")
                removed += out.returncode == 0
            e["pruned"] = True
            update_registry(key, e)
    print(f"{removed} images removed")
    return 0


def cmd_candidates(args) -> int:
    reg = registry()
    rng = random.Random(args.seed)
    for language in args.languages:
        path = editdata.edit_path("candidates", f"{language}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        have = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        done = generated_keys(path, "bug", source=SOURCE_NAMES[language])
        added = 0
        with open(path, "a", encoding="utf-8") as f:
            for key, entry in sorted(reg.items()):
                if (entry["language"] != language or not entry.get("permissive") or entry.get("kind") == "real_pr"
                        or key in done):
                    continue
                if entry["split"] == "train" and language not in SMITH_TRAIN:
                    continue  # this dataset's rows are eval-only (no licence on its card)
                want = round(WANT[entry["split"]][language] * OVERSAMPLE)
                for rec in candidates_for(language, key, entry, want, rng):
                    if rec["id"] not in have:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        added += 1
        print(f"{language}: {added} new candidates ({len(have)} before)")
    return 0


FEATURE_SHARE = 0.34  # of each repository's tasks; at least a quarter survive validation (code-editing-plan §4.1)


def features_for(language: str, key: str, entry: dict, want: int, rng: random.Random) -> list[dict]:
    """Feature-removal candidates in one repository (editfeatures.py). Task texts are written later."""
    import shutil
    import tarfile
    import tempfile

    from rlm_rl import editfeatures

    snapshot = data.path(entry["snapshot"])
    tree = Path(tempfile.mkdtemp(prefix="rlm-feat-"))
    try:
        with tarfile.open(snapshot) as tar:
            tar.extractall(tree / "clean", filter="tar")
        clean = tree / "clean"
        fns = editfeatures.functions(clean, language, grader.is_test_path)
        mentions = editfeatures.test_mentions(clean, language, grader.is_test_path, {f.name for f in fns})
        ranked = []
        for f in fns:
            tests = mentions.get(f.name) or []
            if not 1 <= len(tests) <= 4 or len(f.name) < 5:
                continue
            score = (bool(f.doc), -abs((f.body_end - f.body_start) - 14), rng.random())
            ranked.append((score, f, tests))
        ranked.sort(key=lambda x: x[0], reverse=True)
        out, used_files = [], set()
        for _, f, tests in ranked:
            if len(out) >= want:
                break
            if f.path in used_files:
                continue
            source = _source_text(clean / f.path)
            if source is None:
                continue
            text, newline = source
            stubbed = editfeatures.stub(text, f, language)
            work = tree / f"w{len(out)}"
            shutil.copytree(clean, work, symlinks=True)
            _write_source(work / f.path, stubbed, newline)
            break_patch = _tree_diff(clean, work, [f.path])
            shutil.rmtree(work, ignore_errors=True)
            if not break_patch.strip():
                continue
            gold = editdata.reverse_patch(break_patch, clean)
            scope, cwd, tests = generated_scope(language, tests, _test_texts(clean, tests, language), entry.get("test_cwd"))
            if not scope:
                continue
            hidden = editdata.snapshot_read(snapshot, tests)
            nfiles, nlines = editdata.patch_size(gold)
            used_files.add(f.path)
            rid = f"{language[:2]}-{entry['repo'].split('/')[1]}-feat-{f.name}".lower()[:80]
            out.append({
                "id": rid, "split": entry["split"], "language": language, "repo": entry["repo"], "key": key,
                "kind": "feature", "source": "generated", "instance_id": None, "licence": entry["licence"],
                "image": entry["tag"], "snapshot": entry["snapshot"], "snapshot_sha256": entry["sha256"],
                "break_patch": break_patch, "gold_patch": gold, "remove": hide_plan(language, tests)[0],
                "replace": hide_plan(language, tests)[1], "issue": "",
                "node_modules": entry.get("node_modules", []),
                "tests": grader.TestSpec(runner=entry.get("runner") or editdata.LANGS[language]["runner"], scope=scope,
                                         f2p=[], p2p=[], language=language, hidden=hidden, cwd=cwd,
                                         bin=entry.get("bin") or [], junit=entry.get("junit"),
                                         args=entry.get("args") or [], env=entry.get("env") or {}).as_dict(),
                "test_hint": entry.get("test_hint") or editdata.LANGS[language]["test_hint"],
                "level": editreal.level_of(nfiles, nlines),
                "gold_files": nfiles, "gold_lines": nlines, "files": [],
                "feature": {"path": f.path, "name": f.name, "qualname": f.qualname, "signature": f.signature,
                            "doc": f.doc[:2000], "tests": tests},
            })
            if language == "go":  # hide the test functions, not the files (shared test helpers)
                out[-1] = go_strip(out[-1], {f.name}, set()) or out[-1]
                out[-1]["id"] = rid
        return out
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def _tree_diff(a: Path, b: Path, paths: list[str]) -> str:
    """A git diff of `paths` between two sibling trees, with a/ and b/ prefixes relative to the trees. Bytes decoded
    as they are: text mode would turn a CRLF file's \r\n into \n and the patch would no longer apply."""
    out = []
    for p in paths:
        r = subprocess.run(["git", "-c", "core.quotePath=false", "diff", "--no-index", "--no-color", "--full-index",
                            "--", f"{a.name}/{p}", f"{b.name}/{p}"], cwd=a.parent, capture_output=True,
                           env=editbox.git_env(a.parent))
        out.append(r.stdout.decode("utf-8").replace(f"a/{a.name}/", "a/").replace(f"b/{b.name}/", "b/"))
    return "".join(out)


def _test_texts(tree: Path, tests: list[str], language: str) -> dict | None:
    """The test files' contents, where generated_scope needs them (Java, JavaScript, TypeScript)."""
    if language not in ("java", "javascript", "typescript"):
        return None
    return {t: (tree / t).read_text(encoding="utf-8", errors="replace") for t in tests}


def _source_text(path: Path) -> tuple[str, str] | None:
    """A source file's text with \n line ends, and the line end to write it back with. None for a file that mixes
    line ends (writing it back would change lines we didn't touch)."""
    raw = path.read_bytes().decode("utf-8")
    crlf = raw.count("\r\n")
    if 0 < crlf < raw.count("\n"):
        return None
    return (raw.replace("\r\n", "\n"), "\r\n") if crlf else (raw, "\n")


def _write_source(path: Path, text: str, newline: str) -> None:
    path.write_bytes((text.replace("\n", newline) if newline != "\n" else text).encode("utf-8"))

BUGS_PER_FUNCTION = 2  # injected bugs per function, on different lines (the first ones spread over functions first)
BUG_MAX_TEST_FILES = 8  # a function that more test files mention is too central to make a fair single bug


def bugs_for(language: str, key: str, entry: dict, want: int, rng: random.Random) -> list[dict]:
    """Our own injected bugs in one repository (editfeatures.mutations): one mutant per function. Texts come later."""
    import shutil
    import tarfile
    import tempfile

    from rlm_rl import editfeatures

    snapshot = data.path(entry["snapshot"])
    tree = Path(tempfile.mkdtemp(prefix="rlm-bug-"))
    try:
        with tarfile.open(snapshot) as tar:
            tar.extractall(tree / "clean", filter="tar")
        clean = tree / "clean"
        fns = editfeatures.functions(clean, language, grader.is_test_path)
        mentions = editfeatures.test_mentions(clean, language, grader.is_test_path, {f.name for f in fns})
        pool = [(f, mentions[f.name]) for f in fns if 1 <= len(mentions.get(f.name) or []) <= BUG_MAX_TEST_FILES]
        rng.shuffle(pool)
        out, used = [], set()
        jobs = []  # (function, tests, mutated text, kind, line, line end): up to BUGS_PER_FUNCTION per function
        for f, tests in pool:
            source = _source_text(clean / f.path)
            if source is None:
                continue
            text, newline = source
            for mutated, kind, line in editfeatures.mutations(text, f, language, rng, limit=BUGS_PER_FUNCTION):
                jobs.append((f, tests, mutated, kind, line, newline))
        # one mutant per function first, then the second ones, so the bugs spread over as many functions as possible
        firsts, seconds, seen = [], [], set()
        for job in jobs:
            (seconds if (job[0].path, job[0].name) in seen else firsts).append(job)
            seen.add((job[0].path, job[0].name))
        for f, tests, mutated, kind, line, newline in firsts + seconds:
            if len(out) >= want:
                break
            if (f.path, f.name, line) in used:
                continue
            work = tree / f"w{len(out)}"
            shutil.copytree(clean, work, symlinks=True)
            _write_source(work / f.path, mutated, newline)
            break_patch = _tree_diff(clean, work, [f.path])
            shutil.rmtree(work, ignore_errors=True)
            gold = editdata.reverse_patch(break_patch, clean)
            scope, cwd, tests = generated_scope(language, tests, _test_texts(clean, tests, language), entry.get("test_cwd"))
            if not scope:
                continue
            used.add((f.path, f.name, line))
            rid = f"{language[:2]}-{entry['repo'].split('/')[1]}-bug-{f.name}-{line}".lower()[:80]
            out.append({
                "id": rid, "split": entry["split"], "language": language, "repo": entry["repo"], "key": key,
                "kind": "bug", "source": "generated", "instance_id": None, "licence": entry["licence"],
                "image": entry["tag"], "snapshot": entry["snapshot"], "snapshot_sha256": entry["sha256"],
                "break_patch": break_patch, "gold_patch": gold, "remove": hide_plan(language, tests)[0],
                "replace": hide_plan(language, tests)[1], "issue": "",
                "node_modules": entry.get("node_modules", []),
                "tests": grader.TestSpec(runner=entry.get("runner") or editdata.LANGS[language]["runner"], scope=scope,
                                         f2p=[], p2p=[], language=language, hidden=editdata.snapshot_read(snapshot, tests),
                                         cwd=cwd, bin=entry.get("bin") or [], junit=entry.get("junit"),
                                         args=entry.get("args") or [], env=entry.get("env") or {}).as_dict(),
                "test_hint": entry.get("test_hint") or editdata.LANGS[language]["test_hint"], "level": 1,
                "gold_files": 1, "gold_lines": 2,
                "files": [], "feature": {"path": f.path, "name": f.name, "qualname": f.qualname,
                                          "signature": f.signature, "doc": f.doc[:2000], "tests": tests,
                                          "mutation": kind, "line": line},
            })
            if language == "go":  # hide the test functions, not the files (shared test helpers)
                out[-1] = go_strip(out[-1], {f.name}, set()) or out[-1]
                out[-1]["id"] = rid
        return out
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def generated_keys(path: Path, kind: str, source: str = "generated") -> set[str]:
    """Repositories that already have candidates of this kind from this source. The stages that write them skip those,
    so a rerun after more repositories are derived doesn't add a second, differently drawn set (the random draws are
    shared across repositories in order)."""
    if not path.exists():
        return set()
    recs = (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())
    return {r["key"] for r in recs if r["source"] == source and r["kind"] == kind}


def cmd_bugs(args) -> int:
    reg = registry()
    rng = random.Random(args.seed + 1)
    for language in args.languages:
        path = editdata.edit_path("candidates", f"{language}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        have = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        done = generated_keys(path, "bug")
        added = 0
        with open(path, "a", encoding="utf-8") as f:
            for key, entry in sorted(reg.items()):
                if (entry["language"] != language or not entry.get("permissive") or entry["split"] != "train"
                        or entry.get("kind") == "real_pr" or key in done):
                    continue
                want = round(WANT["train"][language] * (1 - FEATURE_SHARE) * OVERSAMPLE * 1.3)
                for rec in bugs_for(language, key, entry, want, rng):
                    if rec["id"] not in have:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        added += 1
                        have.add(rec["id"])
        print(f"{language}: {added} new bug candidates of our own")
    return 0


def cmd_briefs(args) -> int:
    """What I need to write each generated task's text: one Markdown file per language, under edit/issues/."""
    for language in args.languages:
        cands = {json.loads(l)["id"]: json.loads(l) for l in editdata.edit_path("candidates", f"{language}.jsonl").read_text(
            encoding="utf-8").splitlines()}
        valid = {}
        vpath = editdata.edit_path("validated", f"{language}.jsonl")
        for line in vpath.read_text(encoding="utf-8").splitlines() if vpath.exists() else []:
            r = json.loads(line)
            valid[r["id"]] = r
        texts = editdata.read_json(editdata.edit_path("issues", f"{language}.json"), {})
        out = []
        for rid, rec in sorted(cands.items()):
            if rec["issue"] or rid in texts or rec.get("kind") != "feature":
                continue
            if not args.all and not valid.get(rid, {}).get("ok"):
                continue  # --all: also tasks still being validated, so the texts can be written meanwhile
            f = rec["feature"]
            uses = []
            for path, text in rec["tests"]["hidden"].items():
                for line in text.splitlines():
                    if f["name"] in line and not re.match(r"\s*(def |func |fn |public void |it\(|test\(|describe\()", line):
                        uses.append(line.strip()[:160])
            out.append(f"## {rid}\n- kind: {rec['kind']} ({f.get('mutation', 'removed body')}), repo {rec['repo']}, "
                       f"file `{f['path']}`, function `{f['qualname']}`\n- signature: `{f['signature'][:200]}`\n"
                       f"- doc: {f['doc'][:600]!r}\n- how tests use it:\n" + "\n".join(f"    {u}" for u in uses[:8]) +
                       f"\n- gold patch (do not quote):\n```diff\n{rec['gold_patch'][:2500]}\n```\n")
        path = editdata.edit_path("issues", f"briefs-{language}.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(out), encoding="utf-8")
        print(f"{language}: {len(out)} briefs -> {path}")
    return 0


# Our own injected bugs' task texts (`bugtexts`), written from a template of mine: the function, its file, what its
# documentation says, and the symptom a mutation of that kind shows, never the line or the change. Each text says the
# function misbehaves, as a bug report naming the broken API would.
BUG_SYMPTOMS = {
    "flipped equality": "it treats values that match as if they differed, and the other way round, in some cases",
    "flipped inequality": "it treats values that match as if they differed, and the other way round, in some cases",
    "flipped strict equality": "it treats values that match as if they differed, and the other way round, in some cases",
    "flipped strict inequality": "it treats values that match as if they differed, and the other way round, in some cases",
    "boundary": "it gets the edge of a range wrong: inputs right at a limit are handled as if they were on the other side",
    "flipped logic": "one of its conditions is combined wrongly, so some inputs take the wrong path",
    "off by one": "a count or index in it is off by one",
    "flipped flag": "it behaves as if one of its options or flags were set the other way",
}
FIRST_SENTENCE = re.compile(r"^(.+?[.!?])(\s|$)", re.S)


def bug_text(rec: dict) -> str:
    f = rec["feature"]
    doc = re.sub(r"^\s*(///?|/\*\*?|\*/?|#)\s?", "", f.get("doc") or "", flags=re.M)
    doc = " ".join(doc.split())
    first = FIRST_SENTENCE.match(doc)
    lines = [f"`{f['qualname']}` in `{f['path']}` has a regression: {BUG_SYMPTOMS.get(f.get('mutation'), 'it returns wrong results for some inputs')}."
             " Code that relies on it now misbehaves, and the existing tests that exercise it fail."]
    if first and len(first.group(1)) >= 12:
        lines.append(f"What it should do, from its documentation: {first.group(1)}")
    lines.append(f"Find the bug and fix it, so that `{f['name']}` behaves as documented and as the rest of the code "
                 "expects. Don't change the tests.")
    return "\n\n".join(lines)


def cmd_bugtexts(args) -> int:
    for language in args.languages:
        cands = [json.loads(l) for l in editdata.edit_path("candidates", f"{language}.jsonl").read_text(
            encoding="utf-8").splitlines()]
        path = editdata.edit_path("issues", f"{language}.json")
        texts = editdata.read_json(path, {})
        added = 0
        for rec in cands:
            if rec["source"] == "generated" and rec["kind"] == "bug" and not rec["issue"] and rec["id"] not in texts:
                text = bug_text(rec)
                if editdata.leaks(text, rec["gold_patch"], []):
                    continue
                texts[rec["id"]] = text
                added += 1
        editdata.write_json(path, texts)
        print(f"{language}: {added} bug texts written ({len(texts)} texts in all)")
    return 0


JAVA_TEST_MARK = re.compile(r"@(?:Test|ParameterizedTest|RepeatedTest|TestFactory|TestTemplate)\b|\bextends\s+TestCase\b")
JS_TEST_MARK = re.compile(r"(?<![\w.$])(?:it|test|describe)(?:\.\w+)?\s*\(")


def generated_scope(language: str, tests: list[str], texts: dict | None = None,
                    test_cwd: str | None = None) -> tuple[list, str, list[str]]:
    """The runner scope and working folder for a generated task's hidden test files: (scope, cwd, the test files used).
    Go runs packages; Rust `--test <name>` per integration test file (or `--lib` for a src/**/tests.rs module) in the
    crate that holds them; Java test classes in their module; Python, JavaScript and TypeScript the files themselves.
    `texts` (path -> content), for Java, JavaScript and TypeScript: a file without a test in it is a helper the
    visible tests share, so it stays visible (hiding it would break them) and out of the scope. `test_cwd` (REPO_TESTS, JavaScript and
    TypeScript): the tests run from that folder, so only test files under it are used, named from it."""
    if language == "go":
        return (sorted({"./" + str(Path(t).parent).replace("\\", "/") if str(Path(t).parent) != "." else "."
                        for t in tests}), ".", sorted(tests))
    if language == "rust":
        by_crate: dict[str, list] = collections.defaultdict(list)
        for t in tests:
            parts = t.split("/")
            if "tests" in parts[:-1]:  # <crate>/tests/<name>.rs or <crate>/tests/<name>/main.rs
                i = parts.index("tests")
                crate = "/".join(parts[:i]) or "."
                name = parts[i + 1].removesuffix(".rs") if len(parts) > i + 2 else Path(parts[-1]).stem
                by_crate[crate].append((["--test", name], t))
            elif "src" in parts:  # an inline test module file, src/**/tests.rs
                crate = "/".join(parts[:parts.index("src")]) or "."
                by_crate[crate].append((["--lib"], t))
        if not by_crate:
            return [], ".", []
        crate, items = max(by_crate.items(), key=lambda kv: len(kv[1]))  # the crate with the most of them
        scope, seen = [], set()
        for args, _ in items:
            if tuple(args) not in seen:
                seen.add(tuple(args))
                scope += args
        return scope, crate, sorted(t for _, t in items)
    if language == "java":
        by_module: dict[str, list] = collections.defaultdict(list)
        for t in tests:
            if "src/test/java/" in t and t.endswith(".java"):
                if texts is not None and not JAVA_TEST_MARK.search(texts.get(t, "")):
                    continue
                module, cls = t.split("src/test/java/", 1)
                by_module[module.rstrip("/") or "."].append((cls[:-5].replace("/", "."), t))
        if not by_module:
            return [], ".", []
        module, items = max(by_module.items(), key=lambda kv: len(kv[1]))
        return sorted(c for c, _ in items), module, sorted(t for _, t in items)
    if texts is not None and language in ("javascript", "typescript"):
        tests = [t for t in tests if JS_TEST_MARK.search(texts.get(t, ""))]
    if test_cwd and language in ("javascript", "typescript"):
        under = sorted(t for t in tests if t.startswith(test_cwd.rstrip("/") + "/"))
        return [t[len(test_cwd.rstrip("/")) + 1:] for t in under], test_cwd, under
    return sorted(tests), ".", sorted(tests)


def hide_plan(language: str, tests: list[str]) -> tuple[list[str], dict[str, str]]:
    """How a task's hidden test files leave its repository: (files removed, files replaced by a placeholder). A Rust
    test module in its own file under src/ (src/**/tests.rs) is named by a `mod tests;` line in the code, so removing
    the file breaks the crate's build; it stays as an empty module instead. The grader writes the full file back."""
    if language != "rust":
        return sorted(tests), {}
    replace = {t: "" for t in tests if "src" in t.split("/")[:-1] and "tests" not in t.split("/")[:-1][:1]}
    return sorted(t for t in tests if t not in replace), replace


def go_strip(rec: dict, mentions: set[str], names: set[str]) -> dict | None:
    """A Go candidate that hides test functions instead of whole files: `replace` holds each hidden test file
    without those functions (helpers kept), `remove` keeps only files left with nothing to show. The hidden texts stay
    the full originals, which the grader writes back. None when no test function matched."""
    from rlm_rl import editfeatures

    replace, remove = {}, []
    for path in rec["remove"]:
        text = rec["tests"]["hidden"].get(path)
        if text is None or not path.endswith("_test.go"):
            remove.append(path)
            continue
        stripped = editfeatures.strip_go_tests(text, mentions, names)
        if stripped is None:
            remove.append(path)
        else:
            replace[path] = stripped
    if not replace:
        return None
    return dict(rec, id=rec["id"][:77] + "-gs", remove=sorted(remove), replace=replace)


def cmd_fixscope(args) -> int:
    """Generated Rust and Java candidates written with file paths as their scope (Cargo wants `--test <name>`, Maven
    test classes), hiding a test helper with the tests (Java, JavaScript, TypeScript), or removing a Rust test module
    file the code names (hide_plan): rewritten in place, and the results the old version caused forgotten so they're
    validated again."""
    for language in args.languages:
        path = editdata.edit_path("candidates", f"{language}.jsonl")
        cands = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        changed, ids = 0, set()
        for rec in cands:
            if rec["source"] != "generated":
                continue
            hidden = rec["tests"]["hidden"]
            test_cwd = rec["tests"].get("cwd") if rec["tests"].get("cwd", ".") != "." else None
            scope, cwd, used = generated_scope(language, sorted(hidden), hidden, test_cwd)
            helpers = language in ("java", "javascript", "typescript") and used != sorted(hidden)
            remove, replace = hide_plan(language, rec["remove"])
            if replace and (rec.get("replace") or {}) != replace:  # Rust: src/**/tests.rs stays as an empty module
                rec["remove"], rec["replace"] = remove, replace
                changed += 1
                ids.add(rec["id"])
            if scope and (scope != rec["tests"]["scope"] or cwd != rec["tests"].get("cwd", ".") or helpers):
                rec["tests"] = dict(rec["tests"], scope=scope, cwd=cwd)
                if helpers:  # the helpers go back into the repository
                    rec["tests"]["hidden"] = {t: hidden[t] for t in used}
                    rec["remove"] = used
                    if rec.get("feature"):
                        rec["feature"]["tests"] = [t for t in rec["feature"]["tests"] if t in used]
                changed += 1
                ids.add(rec["id"])
        editdata.write_lines(path, [json.dumps(c, ensure_ascii=False) for c in cands])
        vpath = editdata.edit_path("validated", f"{language}.jsonl")
        dropped = 0
        if vpath.exists():
            rows = vpath.read_text(encoding="utf-8").splitlines()
            keep = [l for l in rows if json.loads(l)["id"] not in ids]
            dropped = len(rows) - len(keep)
            editdata.write_lines(vpath, keep)
        print(f"{language}: {changed} generated candidates rescoped; {dropped} results forgotten")
    return 0


def cmd_gostrip(args) -> int:
    """Go candidates the visible-test check dropped (a hidden test file defined helpers the visible tests use),
    rewritten to hide only their test functions, as new candidates (id + "-gs") for the next validation."""
    path = editdata.edit_path("candidates", "go.jsonl")
    cands = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    have = {c["id"] for c in cands}
    vpath = editdata.edit_path("validated", "go.jsonl")
    dropped = {r["id"] for r in (json.loads(l) for l in vpath.read_text(encoding="utf-8").splitlines())
               if not r["ok"] and r["why"].startswith("visible tests break")}
    added = 0
    with open(path, "a", encoding="utf-8") as f:
        for rec in cands:
            if rec["id"] not in dropped or rec.get("replace"):
                continue
            if rec["kind"] == "feature" or rec["source"] == "generated":
                new = go_strip(rec, {rec["feature"]["name"]}, set())
            else:  # a SWE-smith bug: hide the failing tests by name
                new = go_strip(rec, set(), {n.split("/")[0] for n in rec.get("smith_f2p") or []})
            if new and new["id"] not in have:
                f.write(json.dumps(new, ensure_ascii=False) + "\n")
                have.add(new["id"])
                added += 1
    print(f"go: {added} candidates rewritten to hide test functions instead of files")
    if args.pending:  # candidates with no validation result yet: rewritten in place, same ids
        done = {json.loads(l)["id"] for l in vpath.read_text(encoding="utf-8").splitlines()}
        cands = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        changed = 0
        for i, rec in enumerate(cands):
            if rec["id"] in done or rec.get("replace"):
                continue
            if rec["kind"] == "feature" or rec["source"] == "generated":
                new = go_strip(rec, {rec["feature"]["name"]}, set())
            elif rec["kind"] == "real_pr":
                new = None  # a real PR's hidden files are whole new versions of its test files
            else:
                new = go_strip(rec, set(), {n.split("/")[0] for n in rec.get("smith_f2p") or []})
            if new:
                new["id"] = rec["id"]
                cands[i] = new
                changed += 1
        editdata.write_lines(path, [json.dumps(c, ensure_ascii=False) for c in cands])
        print(f"go: {changed} pending candidates now hide test functions instead of files")
    return 0


def cmd_features(args) -> int:
    reg = registry()
    rng = random.Random(args.seed)
    for language in args.languages:
        path = editdata.edit_path("candidates", f"{language}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        have = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        done = generated_keys(path, "feature")
        added = 0
        with open(path, "a", encoding="utf-8") as f:
            for key, entry in sorted(reg.items()):
                if (entry["language"] != language or not entry.get("permissive") or entry.get("kind") == "real_pr"
                        or key in done):
                    continue
                want = max(2, round(WANT[entry["split"]][language] * FEATURE_SHARE * OVERSAMPLE))
                for rec in features_for(language, key, entry, want, rng):
                    if rec["id"] not in have:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        added += 1
                        have.add(rec["id"])
        print(f"{language}: {added} new feature candidates")
    return 0


def read_results(path: Path) -> list[dict]:
    """A validated/<language>.jsonl file, one result per task: a later line for a task (`revisible`) replaces an
    earlier one."""
    if not path.exists():
        return []
    by_id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by_id[r["id"]] = r
    return list(by_id.values())


async def _revisible_all(records: list[tuple[dict, dict]], out_path: Path, jobs: int) -> None:
    gate = asyncio.Semaphore(jobs)
    images: dict[str, editbox.EditImage] = {}

    async def one(rec, old):
        async with gate:
            image = images.setdefault(rec["image"], editbox.EditImage.derived(rec["image"]))
            t = time.time()
            try:
                vis_ok, vis_out = await editdata.visible_tests_run(rec, image)
            except Exception as exc:  # left as it was; the next run tries again
                print(f"err {rec['id']:55s} {exc!r}"[:200], flush=True)
                return
            result = dict(old, ok=vis_ok, why="" if vis_ok else f"visible tests break without the hidden files: "
                          f"{vis_out[-200:]}", at=time.time(), visible_check=editdata.VISIBLE_CHECK_VERSION)
            with open(out_path, "a", encoding="utf-8") as f:  # appended: a validation may be writing this file too
                f.write(json.dumps(result) + "\n")
            print(f"{'OK ' if vis_ok else 'no '} {rec['id']:55s} {time.time() - t:5.0f}s {result['why'][:120]}",
                  flush=True)

    await asyncio.gather(*(one(rec, old) for rec, old in records))


def cmd_revisible(args) -> int:
    """Tasks an older visible-test check dropped (a 300 s cap the slow drive ran past, or an older Go check) get the
    current check again, alone: their base and gold runs passed, so nothing else is repeated."""
    for language in args.languages:
        cands = {json.loads(l)["id"]: json.loads(l) for l in editdata.edit_path("candidates", f"{language}.jsonl")
                 .read_text(encoding="utf-8").splitlines()}
        out = editdata.edit_path("validated", f"{language}.jsonl")
        reg = registry()
        todo = []
        for r in read_results(out):
            if (not r["ok"] and r["why"].startswith("visible tests break") and r["id"] in cands
                    and r.get("visible_check", 1) < editdata.VISIBLE_CHECK_VERSION):
                rec = dict(cands[r["id"]], image=reg[cands[r["id"]]["key"]]["tag"])
                todo.append((rec, r))
        print(f"{language}: checking the visible tests of {len(todo)} tasks again", flush=True)
        asyncio.run(_revisible_all(todo, out, args.jobs))
    return 0


EVAL_SPARE = 1  # per language and eval split beyond its even share, so a split still fills if a language comes up short


def eval_need(split: str) -> tuple[int, int]:
    """(tasks, feature tasks) one language must pass for an eval split to be full: its even share of the split
    (balance() takes them round-robin over the languages) and a spare."""
    total, features = EVAL_SIZES[split]
    n = len(editdata.LANGS)
    return math.ceil(total / n) + EVAL_SPARE, math.ceil(features / n)


class EvalQuota:
    """A language's passing selection and test tasks. Once a split is full, its remaining candidates are spare and not
    validated (Rishi, 2026-10-13: skip the spare evals); training, selection and test tasks keep every check."""

    def __init__(self, results: list[dict], cands: dict[str, dict]):
        self.passed = collections.defaultdict(lambda: {"all": 0, "feature": 0})
        for r in results:
            if r["ok"] and r["id"] in cands:
                self.add(cands[r["id"]])

    def add(self, rec: dict) -> None:
        if rec["split"] in EVAL_SIZES:
            self.passed[rec["split"]]["all"] += 1
            self.passed[rec["split"]]["feature"] += rec["kind"] == "feature"

    def full(self, rec: dict) -> bool:
        if rec["split"] not in EVAL_SIZES:
            return False
        total, features = eval_need(rec["split"])
        p = self.passed[rec["split"]]
        if rec["kind"] == "feature":
            return p["feature"] >= features and p["all"] >= total
        return p["all"] - min(p["feature"], features) >= total - features  # the other slots are taken


def write_spare(language: str, skipped: list[str]) -> None:
    """validated/<language>.spare.json: eval candidates left unvalidated because their split was full. They stay in
    candidates/ and can still be validated later (drop them from this file)."""
    path = editdata.edit_path("validated", f"{language}.spare.json")
    done = {r["id"] for r in read_results(editdata.edit_path("validated", f"{language}.jsonl"))}
    ids = sorted((set(editdata.read_json(path, [])) | set(skipped)) - done)
    editdata.write_json(path, ids)


async def _validate_all(records: list[dict], out_path: Path, jobs: int, quota: EvalQuota | None = None,
                        skipped: list[str] | None = None) -> None:
    gate = asyncio.Semaphore(jobs)
    images: dict[str, editbox.EditImage] = {}

    async def one(rec):
        async with gate:
            if quota is not None and quota.full(rec):  # decided when its turn comes, with the results so far
                skipped.append(rec["id"])
                print(f"--  {rec['id']:55s} spare: its {rec['split']} split is full", flush=True)
                return
            image = images.setdefault(rec["image"], editbox.EditImage.derived(rec["image"]))
            t = time.time()
            try:
                v = await editdata.validate(rec, image)
                vis_ok, vis_out = (await editdata.visible_tests_run(rec, image)) if v.ok else (True, "")
                if v.ok and not vis_ok:
                    v.ok, v.why = False, f"visible tests break without the hidden files: {vis_out[-200:]}"
            except Exception as exc:  # recorded, so a resume doesn't retry it forever
                v = editdata.Validation(False, f"error: {exc!r}"[:400])
            result = {"id": rec["id"], "ok": v.ok, "why": v.why, "f2p": v.f2p, "p2p": v.p2p, "flaky": v.flaky,
                      "gold_s": round(v.gold_s, 1), "base_s": round(v.base_s, 1), "at": time.time(),
                      "visible_check": editdata.VISIBLE_CHECK_VERSION}
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            if v.ok and quota is not None:
                quota.add(rec)
            print(f"{'OK ' if v.ok else 'no '} {rec['id']:55s} f2p={len(v.f2p):3d} p2p={len(v.p2p):4d} "
                  f"{time.time() - t:5.0f}s {v.why[:120]}", flush=True)

    await asyncio.gather(*(one(r) for r in records))


def cmd_validate(args) -> int:
    for language in args.languages:
        cands = [json.loads(l) for l in editdata.edit_path("candidates", f"{language}.jsonl").read_text(
            encoding="utf-8").splitlines()]
        out = editdata.edit_path("validated", f"{language}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        results = read_results(out)
        stale = {r["id"] for r in results if language == "go" and not r["ok"]  # only Go's check changed
                 and r["why"].startswith("visible tests break")
                 and r.get("visible_check", 1) < 3}  # `go vet` dropped them (later versions: `revisible`)
        infra = {r["id"] for r in results if not r["ok"] and r["why"].startswith("error: InfraError")}
        results = [r for r in results if r["id"] not in stale | infra]
        done = {r["id"] for r in results}
        if stale:
            print(f"{language}: {len(stale)} tasks dropped by an older visible-test check are validated again")
        if infra:
            print(f"{language}: {len(infra)} tasks that hit an infra error are validated again")
        reg = registry()
        for c in cands:  # the derived image may have been rebuilt since the candidate was written
            c["image"] = reg[c["key"]]["tag"]
        # a Go candidate rewritten by gostrip (id + "-gs") replaces its original, which is never validated again
        replaced = {c["id"][:-3] for c in cands if c["id"].endswith("-gs")}
        todo = [c for c in cands if c["id"] not in done and (not args.split or c["split"] == args.split)
                and not (not c["id"].endswith("-gs") and c["id"][:77] in replaced)]
        if args.per_repo:
            # Training bug tasks beyond what the index can take are not worth a validation on a slow drive: per
            # repository, stop once `per_repo` have passed (counting earlier runs). Features, real PRs and the eval
            # splits are always validated. Repositories are interleaved so each gets covered early.
            by_id = {c["id"]: c for c in cands}
            passed = collections.Counter(by_id[r["id"]]["key"] for r in results if r["ok"] and r["id"] in by_id
                                         and by_id[r["id"]]["kind"] == "bug" and by_id[r["id"]]["split"] == "train")
            capped = [c for c in todo if c["kind"] == "bug" and c["split"] == "train"]
            rest = [c for c in todo if not (c["kind"] == "bug" and c["split"] == "train")]
            queues = collections.defaultdict(list)
            for c in capped:
                queues[c["key"]].append(c)
            # each repository gets per_repo minus its passes, plus a sixth spare for failures
            picked = []
            for key, q in sorted(queues.items()):
                need = max(0, args.per_repo - passed[key])
                picked.append(q[:need + (need + 5) // 6 if need else 0])
            interleaved = [c for row in itertools.zip_longest(*picked) for c in row if c is not None]
            todo = rest + interleaved
        # training first, then the eval splits' features (they count toward both of a split's needs), then their bugs
        todo.sort(key=lambda c: (c["split"] in EVAL_SIZES, c["split"] in EVAL_SIZES and c["kind"] != "feature"))
        if args.limit:
            todo = todo[:args.limit]
        quota = None if args.all_evals else EvalQuota(results, {c["id"]: c for c in cands})
        skipped: list[str] = []
        print(f"{language}: validating {len(todo)} ({len(done)} done)", flush=True)
        asyncio.run(_validate_all(todo, out, args.jobs, quota, skipped))
        write_spare(language, skipped)
        if skipped:
            print(f"{language}: {len(skipped)} spare eval candidates left unvalidated (validated/{language}.spare.json)")
    return 0


# The eval splits' sizes (spec §12.10): selection-v2 holds the selection split's 10 tasks; heldout-test-v2 the test
# split's 40, at least 15 of them feature tasks. (total, minimum feature tasks)
EVAL_SIZES = {"select": (10, 3), "test": (40, 15)}
# Training tasks per language (code-editing-plan §4.1: ~90 / ~60 / ~60 for JS and TS together / ~45 / ~35), at least a
# quarter of them feature tasks, spread over the language's repositories; plus up to 8 real PRs each in Python, Go and
# TypeScript. A language with fewer validated tasks keeps what it has (edit-build-results.md reports the shortfall).
TRAIN_TARGET = {"python": 90, "go": 60, "javascript": 30, "typescript": 30, "rust": 45, "java": 35}
FEATURE_MIN_SHARE = 0.25
REAL_PR_TARGET = 8


def balance(records: list[dict], total: int, min_features: int, seed: int = 7, group: str = "language") -> list[dict]:
    """Up to `total` records, spread evenly over `group` (languages, or repositories with group="key"): feature tasks
    first up to `min_features`, then the rest round-robin, a seeded shuffle within each group. Fixed for a given set
    of records."""
    rng = random.Random(seed)
    pools: dict[str, dict[str, list]] = {}
    for r in sorted(records, key=lambda r: r["id"]):
        pools.setdefault(r[group], {"feature": [], "other": []})[
            "feature" if r["kind"] == "feature" else "other"].append(r)
    for lang in pools.values():
        for items in lang.values():
            rng.shuffle(items)
    picked: list[dict] = []

    def take(kinds, limit):
        while len(picked) < limit and any(pools[l][k] for l in pools for k in kinds):
            for lang in sorted(pools):
                for k in kinds:
                    if pools[lang][k] and len(picked) < limit:
                        picked.append(pools[lang][k].pop())
                        break

    take(("feature",), min(min_features, total))
    take(("other", "feature"), total)
    return picked


def _gold_writes(rec: dict, ws_dir: Path) -> str:
    """REPL code that writes the task's gold change into /work/repo (files computed on the host)."""
    import base64
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="rlm-gold-"))
    try:
        subprocess.run(["cp", "-a", str(ws_dir / "tree"), str(root / "repo")], check=True)
        ok, why = grader._safe_apply(root / "repo", rec["gold_patch"], root)
        if not ok:
            raise RuntimeError(f"{rec['id']}: the gold patch doesn't apply: {why[:200]}")
        lines = ["import base64, os"]
        for fp in grader.parse_patch(rec["gold_patch"]):
            target = root / "repo" / fp.path
            if target.is_file():
                blob = base64.b64encode(target.read_bytes()).decode()
                lines.append(f"os.makedirs(os.path.dirname({fp.path!r}) or '.', exist_ok=True)")
                lines.append(f"open({fp.path!r}, 'wb').write(base64.b64decode({blob!r}))")
            elif fp.deleted:
                lines.append(f"os.path.exists({fp.path!r}) and os.remove({fp.path!r})")
        return "\n".join(lines)
    finally:
        shutil.rmtree(root, ignore_errors=True)


async def _measure_all(records: list[dict], jobs: int, cell_s: int, out: dict, out_path: Path) -> None:
    """Each task once, `jobs` at a time (training-like load): container start, repo copy, the gold change's test
    command through shell(), and the grader on the gold patch."""
    import dataclasses
    import secrets

    from rlm_rl import bridge

    gate = asyncio.Semaphore(jobs)
    images: dict[str, editbox.EditImage] = {}
    limits = dataclasses.replace(editbox.edit_sandbox_limits(), cell_timeout_s=float(cell_s))
    messages = bridge.cell_messages(limits=dataclasses.replace(bridge.EDIT_LIMITS, cell_timeout_s=cell_s), edit=True)

    async def echo(prompt: str) -> str:
        return "ok"

    async def one(rec):
        async with gate:
            row = {"language": rec["language"], "repo": rec["repo"], "split": rec["split"], "kind": rec["kind"]}
            try:
                image = images.setdefault(rec["image"], editbox.EditImage.derived(rec["image"]))
                task = editreal.to_task(rec, "edit_real")
                ws = await asyncio.to_thread(task.workspace.ensure)
                tmp = editbox.cache_root() / f"measure-{secrets.token_hex(6)}"
                t = time.monotonic()
                await asyncio.to_thread(editbox.materialize, ws, tmp)
                row["copy_s"] = round(time.monotonic() - t, 2)
                editbox.remove_tree(tmp)
                code = await asyncio.to_thread(_gold_writes, rec, ws)
                box = editbox.EditSandbox(task.context, echo, workspace=task.workspace, image=image, limits=limits,
                                          run_id="edit-measure", messages=messages)
                t = time.monotonic()
                await box.start()
                row["start_s"] = round(time.monotonic() - t - row["copy_s"], 2)
                try:
                    res = await box.run_cell(code + f"\nr = shell({rec['test_hint']!r}, timeout={cell_s - 30})\n"
                                             "print('exit', r.code)")
                    shells = [e for e in box.events if e.get("kind") == "tool" and e.get("event") == "shell"]
                    row["test_s"] = round(float(shells[-1]["seconds"]), 2) if shells else None
                    row["test_exit"] = (res.stdout.split("exit ")[-1].strip()[:6] if "exit " in res.stdout
                                        else res.error or res.crash)
                finally:
                    await box.close()
                spec = grader.TestSpec.from_dict(rec["tests"])
                g = await grader.grade(rec["gold_patch"], spec, ws_dir=ws, image=image, run_id="edit-measure")
                row.update(grade_s=g.seconds, grade_queued_s=g.queued_s, grade_retries=g.retries,
                           grade_resolved=g.resolved)
            except Exception as exc:  # recorded; the summary counts them
                row["error"] = repr(exc)[:300]
            out[rec["id"]] = row
            editdata.write_json(out_path, out)
            print(f"{rec['id']:58s} start {row.get('start_s')} copy {row.get('copy_s')} test {row.get('test_s')} "
                  f"({row.get('test_exit')}) grade {row.get('grade_s')} {row.get('error', '')[:80]}", flush=True)

    await asyncio.gather(*(one(r) for r in records))


def _percentiles(values: list[float]) -> dict:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {}
    pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]  # noqa: E731
    return {"n": len(values), "p50": pick(0.5), "p95": pick(0.95), "max": values[-1]}


def cmd_measure(args) -> int:
    """Spec §12.3/§12.5 under load: time every indexed task (or --limit per language) `--jobs` at a time, write
    edit/measured.json, and print per-language p50/p95/max. `index` then sets caps and cell limits from it."""
    index = editdata.read_json(editdata.edit_path("index.json"), {"tasks": []})["tasks"]
    out_path = editdata.edit_path("measured.json")
    out = editdata.read_json(out_path, {})
    todo, per_lang, per_repo = [], collections.Counter(), collections.Counter()
    for rec in out.values():
        if "error" not in rec:
            per_repo[rec.get("repo")] += 1
    for rec in sorted(index, key=lambda r: r["id"]):
        if rec["id"] in out and "error" not in out[rec["id"]]:
            continue
        if args.limit and per_lang[rec["language"]] >= args.limit:
            continue
        if args.per_repo and per_repo[rec["repo"]] >= args.per_repo:
            continue  # the test command is the repository's own: a few tasks stand for the rest
        per_lang[rec["language"]] += 1
        per_repo[rec["repo"]] += 1
        todo.append(rec)
    random.Random(5).shuffle(todo)  # languages mixed, so the load is like training's
    print(f"measuring {len(todo)} tasks, {args.jobs} at a time", flush=True)
    asyncio.run(_measure_all(todo, args.jobs, args.cell, out, out_path))
    summary = {}
    for language in sorted({r["language"] for r in out.values()}):
        rows = [r for r in out.values() if r["language"] == language and "error" not in r]
        summary[language] = {k: _percentiles([r.get(k) for r in rows]) for k in ("start_s", "copy_s", "test_s", "grade_s")}
        summary[language]["errors"] = sum(1 for r in out.values() if r["language"] == language and "error" in r)
        summary[language]["raised_cell_limit"] = sum(1 for r in rows if editdata.cell_limit(r.get("test_s")))
    editdata.write_json(editdata.edit_path("measured-summary.json"), {"jobs": args.jobs, "summary": summary})
    for language, stats in summary.items():
        print(language, json.dumps(stats))
    return 0


def cmd_index(args) -> int:
    """Assemble edit/index.json from validated records, capped per repository and split."""
    tasks, counts = [], collections.Counter()
    for language in sorted(editdata.LANGS):
        cpath = editdata.edit_path("candidates", f"{language}.jsonl")
        vpath = editdata.edit_path("validated", f"{language}.jsonl")
        if not cpath.exists() or not vpath.exists():
            continue
        cands = {json.loads(l)["id"]: json.loads(l) for l in cpath.read_text(encoding="utf-8").splitlines()}
        reg = registry()
        for c in cands.values():
            c["image"] = reg[c["key"]]["tag"]
        results = {}
        for line in vpath.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            results[r["id"]] = r
        texts = editdata.read_json(editdata.edit_path("issues", f"{language}.json"), {})
        measured = editdata.read_json(editdata.edit_path("measured.json"), {})  # `measure`: timings under load
        repo_test: dict[str, float] = {}  # the slowest measured test command per repository, for unmeasured tasks
        for m in measured.values():
            if m.get("test_s") is not None and "error" not in m:
                repo_test[m["repo"]] = max(repo_test.get(m["repo"], 0.0), float(m["test_s"]))
        per_repo = collections.Counter()
        for rid, rec in sorted(cands.items()):
            v = results.get(rid)
            if not v or not v["ok"]:
                continue
            if not rec["issue"]:
                if not texts.get(rid):
                    continue  # a generated task without its text yet
                rec = dict(rec, issue=texts[rid].strip())
            bad = editdata.leaks(rec["issue"], rec["gold_patch"], v["f2p"])
            if bad:
                print(f"left out {rid}: {bad}")
                continue
            cap = WANT.get(rec["split"], {}).get(language)
            if cap and per_repo[(rec["key"], rec["split"])] >= cap and not rec.get("kind") == "feature":
                continue
            per_repo[(rec["key"], rec["split"])] += 1
            loaded = dict(measured.get(rid) or {})
            if loaded.get("test_s") is None and rec["repo"] in repo_test:
                loaded["test_s"], loaded["test_from_repo"] = repo_test[rec["repo"]], True
            slowest = max(v.get("gold_s") or 0.0, loaded.get("grade_s") or 0.0)
            rec = dict(rec, tests=dict(rec["tests"], f2p=v["f2p"], p2p=v["p2p"],
                                       timeout_s=editdata.grader_cap(slowest)), validated=v)
            if editdata.cell_limit(loaded.get("test_s")):
                rec["cell_timeout_s"] = editdata.cell_limit(loaded.get("test_s"))
            if loaded:
                rec["measured"] = loaded
            rec["level"] = editdata.level_for(rec)
            tasks.append(rec)
    train = []
    for language, target in TRAIN_TARGET.items():
        mine = [t for t in tasks if t["split"] == "train" and t["language"] == language]
        train += balance([t for t in mine if t["kind"] != "real_pr"], target, math.ceil(target * FEATURE_MIN_SHARE),
                         group="key")
        train += balance([t for t in mine if t["kind"] == "real_pr"], REAL_PR_TARGET, 0, group="repo")
    tasks = train + [t for split, (total, features) in EVAL_SIZES.items()
                     for t in balance([t for t in tasks if t["split"] == split], total, features)]
    for rec in tasks:
        if not rec.get("files"):
            ws = editreal.to_task(rec, "edit_real").workspace.ensure()
            rec["files"] = sorted([str(p.relative_to(ws / "tree")), p.stat().st_size]
                                  for p in (ws / "tree").rglob("*") if p.is_file() and not p.is_symlink())
        counts[(rec["language"], rec["split"], rec["kind"])] += 1
    editdata.write_json(editdata.edit_path("index.json"), {"format": 1, "built_at": time.time(), "tasks": tasks})
    for (language, split, kind), n in sorted(counts.items()):
        print(f"{language:11s} {split:6s} {kind:8s} {n:4d}")
    print(f"{len(tasks)} tasks in {editdata.edit_path('index.json')}")
    return 0


def cmd_manifest(args) -> int:
    """edit/manifest.json: sources and their licence status, images and digests, repositories and their licences,
    snapshot hashes, task counts and the index's hash (spec §8: config.manifest() pins its hash)."""
    import hashlib

    reg = registry()
    index_path = editdata.edit_path("index.json")
    index = editdata.read_json(index_path, {"tasks": []})
    counts = collections.Counter((t["language"], t["split"], t["kind"], t["source"]) for t in index["tasks"])
    repos = {}
    for key, e in sorted(reg.items()):
        digest = subprocess.run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", e["source"]],
                                capture_output=True, text=True).stdout.strip()
        repos[key] = {"repo": e["repo"], "language": e["language"], "split": e["split"], "licence": e["licence"],
                      "licence_files": e.get("licence_files"), "permissive": e.get("permissive"),
                      "source_image": e["source"], "source_digests": json.loads(digest or "[]"), "derived": e["tag"],
                      "snapshot": e["snapshot"], "snapshot_sha256": e.get("sha256"), "snapshot_bytes": e.get("bytes"),
                      "tasks": sum(1 for t in index["tasks"] if t["key"] == key)}
    manifest = {
        "format": 1, "built_at": time.time(),
        "decisions": {"E-b": "SWE-smith per-repository images, permissive repositories only, plus SWE-rebench V2 real PRs",
                      "licences": "JavaScript, TypeScript, Rust and Java train only on tasks we generate; their SWE-smith "
                                  "rows are eval-only (Rishi, 2026-10-12)"},
        "sources": {name: {k: data.SOURCES[name][k] for k in ("origin", "licence", "permissive", "role")}
                    for name in data.SOURCES if name.startswith(("swe-", "generated"))},
        "repositories": repos,
        "tasks": [{"language": l, "split": s, "kind": k, "source": src, "n": n} for (l, s, k, src), n in sorted(counts.items())],
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None,
        "runtime": {"python_bundle": editbox.python_bundle_id(), "runtime_dir": editbox.runtime_dir().name},
    }
    editdata.write_json(editdata.edit_path("manifest.json"), manifest)
    print(f"manifest: {len(repos)} repositories, {len(index['tasks'])} tasks -> {editdata.edit_path('manifest.json')}")
    return 0


def cmd_report(args) -> int:
    reg = registry()
    for key, e in sorted(reg.items()):
        print(f"{key:45s} {e['language']:10s} {e['split']:6s} {e['licence']:18s} {e['tag']}")
    for language in sorted(editdata.LANGS):
        vpath = editdata.edit_path("validated", f"{language}.jsonl")
        if vpath.exists():
            rows = read_results(vpath)
            why = collections.Counter(r["why"].split(":")[0][:60] for r in rows if not r["ok"])
            print(f"{language}: {sum(r['ok'] for r in rows)} ok of {len(rows)}; drops: {dict(why.most_common(6))}")
    out = subprocess.run(["docker", "system", "df"], capture_output=True, text=True)
    print(out.stdout)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("runtime")
    for name in ("pull", "derive", "candidates", "features", "bugs", "bugtexts", "briefs", "validate", "rebench", "prune",
                 "fixscope", "revisible"):
        p = sub.add_parser(name)
        p.add_argument("languages", nargs="+", choices=sorted(editdata.LANGS))
        if name in ("pull", "rebench"):
            p.add_argument("--print", action="store_true", help="print the docker pull commands instead")
        if name == "derive":
            p.add_argument("--force", action="store_true")
        if name in ("candidates", "features", "bugs"):
            p.add_argument("--seed", type=int, default=12)
        if name == "briefs":
            p.add_argument("--all", action="store_true", help="include candidates not validated yet")
        if name == "revisible":
            p.add_argument("--jobs", type=int, default=2)
        if name == "validate":
            p.add_argument("--jobs", type=int, default=4)
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--split", choices=("train", "select", "test"))
            p.add_argument("--per-repo", type=int, default=0,
                           help="validate training bug tasks only until this many per repository have passed")
            p.add_argument("--all-evals", action="store_true",
                           help="also validate eval candidates beyond what the selection and test splits need")
    p = sub.add_parser("measure")
    p.add_argument("--jobs", type=int, default=30, help="tasks at once (training runs ~30 episodes in flight)")
    p.add_argument("--limit", type=int, default=0, help="tasks per language (0: every indexed task)")
    p.add_argument("--cell", type=int, default=1800, help="the cell limit while measuring, s")
    p.add_argument("--per-repo", type=int, default=0,
                   help="tasks per repository (0: all); index gives the rest the repository's slowest test command")
    p = sub.add_parser("gostrip")
    p.add_argument("--pending", action="store_true", help="also rewrite, in place, candidates not validated yet")
    sub.add_parser("index")
    sub.add_parser("manifest")
    sub.add_parser("report")
    args = parser.parse_args()
    return {"runtime": cmd_runtime, "pull": cmd_pull, "derive": cmd_derive, "candidates": cmd_candidates, "features": cmd_features, "bugs": cmd_bugs, "briefs": cmd_briefs, "bugtexts": cmd_bugtexts,
            "rebench": cmd_rebench, "measure": cmd_measure, "gostrip": cmd_gostrip, "fixscope": cmd_fixscope, "revisible": cmd_revisible, "prune": cmd_prune, "validate": cmd_validate, "index": cmd_index, "manifest": cmd_manifest, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
