"""Read-only red-team check over the edit task index (documentation/step-9.5-red-team.md, B1).

    .venv-linux/bin/python scripts/edit_index_check.py

Reads edit/index.json (D:\\RLM-data, or $RLM_DATA_DIR) and never writes anything, on the data
drive or anywhere else. For every indexed task: at least one real fail-to-pass test, no
canary-shaped result key (the bug fixed in 3555b2b), an issue text that leaks neither a hidden
test's name nor a gold-patch line (editdata.leaks, as at build time), no hidden path in the
episode's file listing, permissive training licences and training-allowed sources, and
training repositories disjoint from the eval splits. Prints a summary and every failure, and
exits non-zero if any check fails.
"""

import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rlm_rl import data, editdata  # noqa: E402
from rlm_rl.tasks import editreal  # noqa: E402
from build_edit_data import CANARY_KEY  # noqa: E402  the validator's own canary shape

CANARY_NAME = re.compile(r"^c[0-9a-f]{10}$", re.IGNORECASE)


def main() -> int:
    index = editreal.index()
    tasks = index["tasks"]
    counts = collections.Counter((t["language"], t["split"], t["kind"]) for t in tasks)
    print(f"{len(tasks)} tasks, format {index.get('format')}, built {index.get('built_at')}")
    for (language, split, kind), n in sorted(counts.items()):
        print(f"  {language:10s} {split:6s} {kind:8s} {n:3d}")

    failures = []

    def fail(task, check, detail):
        failures.append((task["id"], check, detail))

    repos = collections.defaultdict(set)
    for task in tasks:
        tid, tests = task["id"], task["tests"]
        f2p, p2p = tests.get("f2p") or [], tests.get("p2p") or []
        hidden = (tests.get("hidden") or {}).keys()
        listed = {p for p, _ in (task.get("files") or [])}
        if not f2p:
            fail(task, "no fail-to-pass test", f2p)
        for key in f2p + p2p:
            if CANARY_KEY.search(key) or CANARY_NAME.match(key.rsplit("::", 1)[-1].rsplit(".", 1)[-1]):
                fail(task, "canary-shaped result key", key)
        leaks = editdata.leaks(task.get("issue") or "", task.get("gold_patch") or "", f2p + p2p)
        for why in leaks:
            fail(task, "issue text leaks", why)
        # Go hides test *functions*: a hidden test file that stays as a stripped `replace` record is
        # visible by design (results doc §6 item 10), so only whole-file removals must not be listed.
        gone = set(task.get("remove") or []) | {p for p in hidden if p not in (task.get("replace") or {})}
        for rel in sorted(gone):
            if rel in listed:
                fail(task, "removed hidden path in the episode's file listing", rel)
        repos[task["split"]].add(task["key"])
        if task["split"] == "train":
            if task.get("licence") not in editreal.PERMISSIVE:
                fail(task, "training licence not permissive", str(task.get("licence")))
            source = task.get("source")
            info = data.SOURCES.get(source)
            if not info or info.get("role") != data.TRAIN:
                fail(task, "training source not cleared for training", str(source))

    train, select, test = repos["train"], repos["select"], repos["test"]
    for split, names in (("select", select), ("test", test)):
        for leak in sorted(train & names):
            failures.append(("*", "training repository in the eval split", leak))
    for overlap in sorted(select & test):
        failures.append(("*", "repository in both eval splits", overlap))

    try:
        data.check_training_sources("edit_real", sorted({r["source"] for r in tasks if r["split"] == "train"}))
    except ValueError as exc:
        failures.append(("*", "training source bar", str(exc)))

    print(f"\n{len(failures)} failure(s)")
    for tid, check, detail in failures:
        print(f"  {tid}: {check}: {str(detail)[:160]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())