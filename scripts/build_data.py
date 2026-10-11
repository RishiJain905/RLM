"""Build the derived data that plan step 7's task families read, under the data root's derived/ folder.

    python scripts/build_data.py lca-git          # unpack each LCA repo archive's git data (once, ~10 GB)
    python scripts/build_data.py lca-index        # index the LCA Python items the repo-localization family uses
    python scripts/build_data.py aggregation-pools  # the fixed sample of arXiv titles the aggregation family uses
    python scripts/build_data.py capability       # the ~100-prompt capability check (rlm_rl/capability.py format)
    python scripts/build_data.py all

Nothing here calls a model or spends credit. The data root is D:\\RLM-data (or $RLM_DATA_DIR); see rlm_rl/data.py.
Every input is a file the Hugging Face CLI fetched at a pinned revision, and every output is pinned by hash in
manifests/dataset-manifest.json (scripts/build_manifest.py).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import data  # noqa: E402
from rlm_rl.tasks import repoloc  # noqa: E402

LCA = "hf/JetBrains-Research__lca-bug-localization"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- LCA


def lca_git(args) -> None:
    """Unpack the .git folder of every LCA Python repo archive into derived/lca-git/<repo>.git (a bare repo)."""
    src = data.path(LCA, "repos", "py")
    out_dir = data.derived("lca-git")
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(src.glob("*.zip"), key=lambda p: p.stat().st_size)
    for n, archive in enumerate(zips, 1):
        repo = archive.stem
        dest = out_dir / f"{repo}.git"
        done = dest / ".unpacked"
        if done.exists():
            continue
        log(f"{n}/{len(zips)} {repo} ({archive.stat().st_size / 1e6:.0f} MB)")
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if not info.filename.startswith(".git/") or info.is_dir():
                    continue
                target = dest / info.filename[len(".git/"):]
                if ".." in Path(info.filename).parts:  # an archive is data: never write outside dest
                    raise ValueError(f"unsafe path in {archive.name}: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as fin, open(target, "wb") as fout:
                    while chunk := fin.read(1 << 20):
                        fout.write(chunk)
        done.write_text(archive.name, encoding="utf-8")
    log("lca-git done")


def lca_index(args) -> None:
    """One gzipped JSON line per usable LCA Python item, with the .py files at its base commit."""
    import pyarrow.parquet as pq

    cols = ["id", "repo_owner", "repo_name", "base_sha", "issue_title", "issue_body", "changed_files",
            "repo_license"]
    for split in ("train", "test"):
        rows = pq.read_table(data.path(LCA, "py", f"{split}-00000-of-00001.parquet"), columns=cols).to_pylist()
        out = data.derived("lca", f"index-{split}.jsonl.gz")
        out.parent.mkdir(parents=True, exist_ok=True)
        kept, skipped = 0, {}
        listings: dict[tuple[str, str], list | None] = {}
        with gzip.open(out, "wt", encoding="utf-8") as f:
            for row in rows:
                repo = f"{row['repo_owner']}__{row['repo_name']}"
                reason, item = repoloc.index_item(row, repo, listings)
                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                f.write(json.dumps(item) + "\n")
                kept += 1
        log(f"lca-index {split}: kept {kept} of {len(rows)}; skipped {skipped}")


# ----------------------------------------------------------------------------- aggregation pools


def aggregation_pools(args) -> None:
    out = data.build_arxiv_pool()
    sizes = {label: len(texts) for label, texts in data.arxiv_title_pools().items()}
    log(f"aggregation-pools: arXiv titles {sizes} -> {out}")


# ----------------------------------------------------------------------------- capability check


def capability(args) -> None:
    from rlm_rl import capability_set

    items = capability_set.build(random.Random(args.seed))
    out = data.derived("capability", f"{capability_set.VERSION}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for item in items:
        counts[item["source"]] = counts.get(item["source"], 0) + 1
    log(f"capability: {len(items)} prompts {counts} -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["lca-git", "lca-index", "aggregation-pools", "capability", "all"])
    parser.add_argument("--seed", type=int, default=7, help="the capability check's fixed sample")
    args = parser.parse_args()
    steps = {"lca-git": [lca_git], "lca-index": [lca_index], "aggregation-pools": [aggregation_pools],
             "capability": [capability], "all": [lca_git, lca_index, aggregation_pools, capability]}[args.what]
    for step in steps:
        step(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
