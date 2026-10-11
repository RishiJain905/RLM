"""Write manifests/dataset-manifest.json: every data file production reads, pinned by sha256 (plan step 7, spec §8).

    python scripts/build_manifest.py

The manifest records, for each source in rlm_rl/data.py, its origin and revision, its licence and whether the
licence bar (7b) lets it train, and the size and sha256 of every file used; the derived files built from them;
each task family's generator version, role and levels; decision 7a's mix with its unbuilt cyber share; and the
eval sets. Its own hash goes into every checkpoint's config.json (`config.manifest`). Hashing ~20 GB takes a few
minutes on the D drive. Nothing here calls a model or spends credit.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import capability_set, config, data, evals, tasks  # noqa: E402

OUT = ROOT / "manifests" / "dataset-manifest.json"
DERIVED = ["derived/lca/index-train.jsonl.gz", "derived/lca/index-test.jsonl.gz",
           f"derived/capability/{capability_set.VERSION}.jsonl"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def file_entry(rel: str) -> dict:
    path = data.root() / rel
    return {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def git_entry(spec: dict, pinned: str) -> dict:
    """A cloned repo is pinned by its commit; hashing every file would add nothing to that."""
    path = data.root() / spec["dir"]
    if not (path / ".git").exists():
        return {"status": "not downloaded", "dir": spec["dir"]}

    def git(*args):
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(path), *args], capture_output=True,
                              text=True).stdout
    head = git("rev-parse", "HEAD").strip()
    files = [f for f in git("ls-files", "-z").split("\0") if f]
    missing = [f for f in git("ls-files", "-z", "--deleted").split("\0") if f]
    size = sum((path / f).stat().st_size for f in files if (path / f).is_file())
    return {"status": "ok" if head == pinned and not missing else "check", "dir": spec["dir"], "commit": head,
            "commit_is_pinned": head == pinned, "files": len(files), "missing_files": len(missing), "bytes": size}


def zenodo_entry(spec: dict) -> dict:
    """Each zip is checked against the md5 Zenodo publishes."""
    out = {}
    for name, want in spec["md5"].items():
        path = data.root() / spec["dir"] / name
        if not path.exists():
            out[name] = {"status": "not downloaded", "md5": want}
            continue
        got = md5(path)
        out[name] = {"status": "ok" if got == want else "incomplete", "bytes": path.stat().st_size, "md5": want}
        print(f"  {name}: {out[name]['status']}", flush=True)
    return {"record": spec["record"], "dir": spec["dir"], "files": out}


def family_entry(family: str) -> dict:
    module = tasks.generator(family)
    levels = getattr(module, "LEVELS", {})
    return {"module": f"rlm_rl.tasks.{tasks.GENERATORS[family]}", "generator_version": module.GENERATOR_VERSION,
            "role": "eval" if tasks.eval_only(family) or family.startswith("tickets") else "train",
            "sources": list(tasks.training_sources(family)), "max_level": module.MAX_LEVEL,
            "levels": {str(k): list(v) for k, v in levels.items()},
            "level_unit": "tickets per context" if family.startswith("tickets") else "characters per context"}


def main() -> int:
    sources = {}
    for name, info in data.SOURCES.items():
        entry = {k: v for k, v in info.items() if k not in ("files", "dirs", "git", "zenodo")}
        entry["files"] = [file_entry(rel) for rel in info.get("files", [])]
        if "git" in info:
            entry["git"] = {"url": info["git"]["url"], **git_entry(info["git"], info["revision"])}
        if "zenodo" in info:
            entry["zenodo"] = zenodo_entry(info["zenodo"])
        for rel in info.get("dirs", []):
            for path in sorted((data.root() / rel).iterdir()):
                if path.is_file():
                    entry["files"].append(file_entry(f"{rel}/{path.name}"))
        sources[name] = entry
        print(f"{name}: {len(entry['files'])} files" + (f", git {entry['git']['status']}" if "git" in entry else ""),
              flush=True)
    manifest = {
        "version": "dataset-manifest-v1",
        "built": dt.date.today().isoformat(),
        "data_root": "D:\\RLM-data (Windows) or /mnt/d/RLM-data (WSL); override with $RLM_DATA_DIR",
        "decisions": {"7a": tasks.DOMAIN_MIX, "7b": "permissive licences only in training; the rest eval-only"},
        "domains": {d: list(f) for d, f in tasks.DOMAIN_FAMILIES.items()},
        "missing_domains": tasks.missing_domains(),
        "mix_without_missing": tasks.production_mix(leave_out_missing=True),
        "families": {family: family_entry(family) for family in sorted(tasks.GENERATORS)},
        "eval_sets": {
            "selection-v1": {f"{f}/L{lv}": n for (f, lv), n in evals.SELECTION_V1.items()},
            "heldout-test-v1": {**{f"{f}/L{lv}": n for (f, lv), n in evals.HELDOUT_TEST_V1.items()},
                                "repo_loc_test": "every usable LCA Python test item, once",
                                "oolong_synth_test": f"{evals.HELDOUT_OOLONG_PER_BUCKET} per length bucket"},
            "capability": config.CAPABILITY_SET,
        },
        "sources": sources,
        "derived": [file_entry(rel) for rel in DERIVED],
        "lca_git": "derived/lca-git/<repo>.git, unpacked from the LCA repo archives listed under sources",
        "not_used": data.NOT_USED,
        "deferred": data.DEFERRED,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({hashlib.sha256(OUT.read_bytes()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
