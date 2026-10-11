"""Write manifests/dataset-manifest.json: every data file production reads, pinned by sha256 (plan step 7, spec §8).

    python scripts/build_manifest.py

The manifest records, for each source in rlm_rl/data.py, its origin and revision, its licence and whether the
licence bar (7b) lets it train, and the size and sha256 of every file used; the derived files built from them;
each task family's generator version, role and levels; the domain mix (decisions 7a and 7c); and the
eval sets. Its own hash goes into every checkpoint's config.json (`config.manifest`). Hashing ~20 GB takes a few
minutes on the D drive. Nothing here calls a model or spends credit.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import capability_set, config, data, evals, tasks  # noqa: E402

OUT = ROOT / "manifests" / "dataset-manifest.json"
DERIVED = ["derived/lca/index-train.jsonl.gz", "derived/lca/index-test.jsonl.gz", f"derived/{data.ARXIV_POOL}",
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
        entry = {k: v for k, v in info.items() if k not in ("files", "dirs")}
        entry["files"] = [file_entry(rel) for rel in info.get("files", [])]
        for rel in info.get("dirs", []):
            for path in sorted((data.root() / rel).iterdir()):
                if path.is_file():
                    entry["files"].append(file_entry(f"{rel}/{path.name}"))
        sources[name] = entry
        print(f"{name}: {len(entry['files'])} files", flush=True)
    manifest = {
        "version": "dataset-manifest-v1",
        "built": dt.date.today().isoformat(),
        "data_root": "D:\\RLM-data (Windows) or /mnt/d/RLM-data (WSL); override with $RLM_DATA_DIR",
        "decisions": {"7a": {"cyber": 0.35, "aggregation": 0.30, "code": 0.20, "repo": 0.15},
                      "7b": "permissive licences only in training; the rest eval-only",
                      "7c": "cyber dropped; its 35% split between code (+20) and aggregation (+15)"},
        "domain_mix": tasks.DOMAIN_MIX,
        "domains": {d: list(f) for d, f in tasks.DOMAIN_FAMILIES.items()},
        "mix": tasks.production_mix(),
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
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({hashlib.sha256(OUT.read_bytes()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
