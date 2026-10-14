"""Read-only red-team scan of the derived edit images (documentation/step-9.5-red-team.md, B2).

    .venv-linux/bin/python scripts/edit_image_scan.py [image ...]

For each image the edit task index references, starts one throwaway container (--rm --network
none, this user's uid, at most one at a time) and searches the file system, outside /work/repo,
for files and folders named after the repository's own package or module: an installed release
in site-packages, a copy in node_modules pulled in as a tool's dependency, the same module in
the Go module cache, or built output. Hits are printed with sizes; nothing is written inside
the container or on the data drive. Output goes to results/red-team/transcripts/b2-scan.log.

A hit is only a finding once compared against the task's gold: see compare_gold() at the
bottom, which reads a found copy's file and hashes it against the gold version reconstructed
from the index (base tree + gold patch). Run it by hand for the hits worth checking.
"""

import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl.tasks import editreal  # noqa: E402

LOG = ROOT / "results" / "red-team" / "transcripts" / "b2-scan.log"

# repo -> names worth searching for: the import name (Python), the module's last element (Go),
# the package name (JavaScript/TypeScript), plus their packaging spellings.
def patterns(language: str, repo: str) -> list[str]:
    pkg = {"pallets/jinja": "jinja2"}.get(repo, repo.split("/")[-1].lower())
    names = [pkg, pkg.replace("-", "_")]
    if language == "go":
        return [f"{pkg}@*"]
    if language == "python":
        return [pkg, f"{pkg}-*.dist-info", f"{pkg}-*.egg-info", f"{pkg}-*.egg-link", f"__editable__*{pkg}*",
                pkg.replace("-", "_")]
    return [pkg]  # javascript/typescript: node_modules/<pkg>, and anything else named after it


SCAN = r"""
set -e
echo "== $IMAGE"
(go env GOMODCACHE 2>/dev/null && go env GOCACHE 2>/dev/null) | sed 's/^/go /' || true
for p in /opt/rlm-cache /opt/rlm-deps /root/go /go; do [ -e "$p" ] && echo "top $p $(du -sh "$p" 2>/dev/null | cut -f1)"; done
python3 -c 'import site,sys; [print("site", p) for p in site.getsitepackages() + [site.getusersitepackages()]]' 2>/dev/null || true
find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o \( -path /work -o -path /work/repo \) -prune -o \( PREDICATE \) -print 2>/dev/null | head -200
grep -rl -a -m1 MODULEPATH /opt/rlm-cache/go-build 2>/dev/null | head -20 | sed 's/^/buildcache /'
find /opt/rlm-cache /opt/rlm-deps -name '*.a' -o -name '*.test' -o -name '*.jar' 2>/dev/null | head -20 | sed 's/^/artifact /'
"""


def scan(image: str, language: str, repo: str) -> list[str]:
    names = patterns(language, repo)
    predicate = " -o ".join(f"-name {n!r}" for n in names)
    module = "github.com/" + repo
    script = (SCAN.replace("PREDICATE", predicate).replace("MODULEPATH", module)
              .replace("$IMAGE", image))
    name = "rt-b2-scan-" + re.sub(r"[^a-z0-9]+", "-", image.split("/")[-1].split(":")[0])[-40:]
    out = subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", "1000", "--name", name,
                          "--entrypoint", "/bin/sh", image, "-c", script], capture_output=True, text=True,
                         timeout=900)
    return out.stdout.splitlines() + ([f"exit={out.returncode} stderr={out.stderr.strip()[:300]}"]
                                       if out.returncode else [])


def main() -> int:
    which = set(sys.argv[1:])
    repos = collections.defaultdict(lambda: {"images": set(), "n": 0, "language": None})
    for t in editreal.index()["tasks"]:
        e = repos[t["repo"]]
        e["images"].add(t["image"])
        e["n"] += 1
        e["language"] = t["language"]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        for repo, e in sorted(repos.items()):
            for image in sorted(e["images"]):
                if which and image not in which:
                    continue
                print(f"scanning {repo} {image} ...", flush=True)
                lines = scan(image, e["language"], repo)
                log.write(f"### {repo} ({e['language']}, {e['n']} tasks) {image}\n")
                log.write("\n".join(lines) + "\n\n")
                log.flush()
    print(f"done; see {LOG}")
    return 0


def compare_gold(repo: str, rel: str, roots: list[str]) -> None:
    """Print whether any found copy of `rel` (repo-relative) matches the gold text of a task."""
    from rlm_rl import grader

    for t in editreal.index()["tasks"]:
        if t["repo"] != repo or t["kind"] != "bug":
            continue
        golds = {fp.path: grader._apply_to_text(fp) for fp in grader.parse_patch(t.get("gold_patch") or "")
                 if fp.path == rel}
        if not golds:
            continue
        task = editreal.to_task(t, "edit_real")
        base = (task.workspace.ensure() / "tree" / rel).read_text(encoding="utf-8", errors="replace")
        gold = golds[rel]
        print(f"{t['id']}: {rel} gold-changed={gold != base}")
        for root in roots:
            out = subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", "1000",
                                  "--entrypoint", "/bin/sh", t["image"], "-c",
                                  f"sha256sum {root}/{rel} 2>&1 | head -2; wc -c < {root}/{rel} 2>/dev/null"],
                                 capture_output=True, text=True, timeout=900)
            print(f"  {root}/{rel}: {out.stdout.strip()[:200]}")
        import hashlib

        print(f"  gold sha256 {hashlib.sha256(gold.encode()).hexdigest()}  base {hashlib.sha256(base.encode()).hexdigest()}")
        return
    print(f"no bug task in {repo} touches {rel}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "gold":
        compare_gold(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        sys.exit(main())
