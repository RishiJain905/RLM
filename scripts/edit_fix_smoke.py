"""Smoke checks after the step 9.5 red-team fixes (F1 and F3), real containers, no model.

  .venv-linux/bin/python scripts/edit_fix_smoke.py go       # each re-derived Go image: cache clean, one gold resolves
  .venv-linux/bin/python scripts/edit_fix_smoke.py python   # two golds per Python repository, with the new pytest command
  .venv-linux/bin/python scripts/edit_fix_smoke.py plant    # F1 live: a pytest.py in the base no longer runs as the runner

Writes results/red-team-fixes/smoke-<mode>.json (git-ignored).
"""
import asyncio
import collections
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("RLM_GRADER_SLOTS", "5")

import build_edit_data as b  # noqa: E402
from rlm_rl import editbox, editdata, grader  # noqa: E402
from rlm_rl.tasks import editreal  # noqa: E402

GO_KEYS = ("go-chi__chi", "masterminds__squirrel", "allegro__bigcache", "c-bata__go-prompt", "hjson__hjson-go",
           "roaringbitmap__roaring", "bits-and-blooms__bitset", "hypermodeinc__ristretto")
CLEAN_GO = ("rb-getkin__kin-openapi-1064", "rb-getkin__kin-openapi-1070")


def cache_check(tag: str, module: str, repo_dir: str) -> dict:
    uid = os.getuid()
    script = (f'echo hits $(grep -rlaF -e "{module}" -e {repo_dir}/ /opt/rlm-cache/go-build 2>/dev/null | wc -l); '
              'echo files $(find /opt/rlm-cache/go-build -type f | wc -l)')
    out = subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", str(uid), "--entrypoint", "sh", tag,
                          "-c", script], capture_output=True, text=True, timeout=1800)
    got = dict(line.split() for line in out.stdout.splitlines() if len(line.split()) == 2)
    return {"hits": int(got.get("hits", -1)), "files": int(got.get("files", -1)), "err": out.stderr[-200:]}


def module_of(rec: dict) -> str:
    ws = editreal.to_task(rec, "edit_real").workspace.ensure()
    gomod = ws / "tree" / (rec["tests"].get("cwd") or ".") / "go.mod"
    if not gomod.exists():
        gomod = ws / "tree" / "go.mod"
    for line in gomod.read_text(encoding="utf-8").splitlines():
        if line.startswith("module "):
            return line.split("//")[0].split()[1].strip('"')
    raise ValueError("no module line")


async def gold(rec: dict, tag: str, ws: Path | None = None) -> dict:
    spec = grader.TestSpec.from_dict(rec["tests"])
    ws = ws or await asyncio.to_thread(editreal.to_task(rec, "edit_real").workspace.ensure)
    t = time.monotonic()
    try:
        g = await grader.grade(rec["gold_patch"], spec, ws_dir=ws, image=editbox.EditImage.derived(tag),
                               run_id="red-team-smoke")
    except Exception as exc:
        return {"id": rec["id"], "error": repr(exc)[:300]}
    return {"id": rec["id"], "resolved": g.resolved, "f2p": f"{g.f2p_passed}/{g.f2p_total}",
            "p2p_failed": g.p2p_failed, "hacks": g.hacks, "build_failed": g.build_failed, "seconds": round(g.seconds),
            "wall_s": round(time.monotonic() - t), "retries": g.retries, "gold_s_at_validation":
            (rec.get("validated") or {}).get("gold_s"), "cap_s": rec["tests"].get("timeout_s"),
            "measured_grade_s": (rec.get("measured") or {}).get("grade_s")}


def tasks() -> list[dict]:
    return editdata.read_json(editdata.edit_path("index.json"), {})["tasks"]


async def main(mode: str) -> dict:
    reg = b.registry()
    out: dict = {"mode": mode, "started": time.time()}
    if mode == "go":
        by_key = collections.defaultdict(list)
        for rec in tasks():
            if rec["language"] == "go":
                by_key[rec["key"]].append(rec)
        picks = []
        for key, recs in sorted(by_key.items()):
            tag = reg[key]["tag"]
            rec = sorted(recs, key=lambda r: (r.get("measured") is None, r["id"]))[0]
            check = await asyncio.to_thread(cache_check, tag, module_of(rec), reg[key].get("repo_dir") or "/testbed")
            out.setdefault("cache", {})[key] = dict(check, tag=tag, rederived=key.startswith(GO_KEYS))
            print(key, tag, check, flush=True)
            if key.startswith(GO_KEYS):
                picks.append((rec, tag))
        out["gold"] = await asyncio.gather(*(gold(r, t) for r, t in picks))
    elif mode == "python":
        by_key = collections.defaultdict(list)
        for rec in tasks():
            if rec["language"] == "python":
                by_key[rec["key"]].append(rec)
        picks = [(r, reg[k]["tag"]) for k, recs in sorted(by_key.items()) for r in sorted(recs, key=lambda r: r["id"])[:2]]
        # F2's task: its gold adds sys.exit(...) calls, which the old skip marker took for mocha's xit(
        picks += [(r, reg[r["key"]]["tag"]) for recs in by_key.values() for r in recs
                  if r["id"] == "py-deepdiff-feat-extract" and all(p[0]["id"] != r["id"] for p in picks)]
        out["gold"] = await asyncio.gather(*(gold(r, t) for r, t in picks))
    elif mode == "plant":
        rec = next(r for r in sorted(tasks(), key=lambda r: r["id"]) if r["repo"] == "andialbrecht/sqlparse")
        ws = await asyncio.to_thread(editreal.to_task(rec, "edit_real").workspace.ensure)
        planted = editbox.cache_root() / "red-team-plant"
        shutil.rmtree(planted, ignore_errors=True)
        shutil.copytree(ws, planted, symlinks=True)
        (planted / "tree" / "pytest.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
        out["plain"] = await gold(rec, reg[rec["key"]]["tag"])
        out["planted"] = await gold(rec, reg[rec["key"]]["tag"], ws=planted)
        shutil.rmtree(planted, ignore_errors=True)
    out["finished"] = time.time()
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    result = asyncio.run(main(mode))
    path = ROOT / "results" / "red-team-fixes" / f"smoke-{mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    for row in result.get("gold", []) + [result[k] for k in ("plain", "planted") if k in result]:
        print(json.dumps(row), flush=True)
