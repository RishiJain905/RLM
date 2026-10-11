"""Replay the 20-step run's training episodes through the hack detectors (spec §7), to measure false alarms.

    python3 scripts/backtest_detectors.py [.art/archive/rlm-rlsd-vs-grpo]

The 20-step run (GRPO and RLSD arms, 20 steps x 32 episodes each) had no detectors, no sandbox
and no reason to hack, and its audit found none, so every detector hit on it is a false alarm.
Its ART trajectory files (git-ignored, local only) hold each episode's conversation and code,
but not the last cell's output, the sub-call outputs or sandbox events. So `copied_answer` is
checked against fewer outputs than it will see in production (it can only fire more often here),
and `escape_attempt` and `timeout_loop` have nothing to read. Nothing here calls a model.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import detectors  # noqa: E402
from rlm_rl.env import extract_code  # noqa: E402

_COUNTERS = re.compile(r"\n\(\d+ turns? left, \d+ sub-calls? left\)(\nYour next turn is the last one.*)?$", re.S)


def episode(row: dict) -> SimpleNamespace:
    """An Episode-shaped view of one archived trajectory: enough for the detectors."""
    messages = [{"role": m["role"], "content": m["content"] or ""} for m in row["messages"]]
    meta, metrics = json.loads(row["metadata"]), json.loads(row["metrics"])
    turns = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        shown = messages[i + 1]["content"] if i + 1 < len(messages) else ""
        stdout = _COUNTERS.sub("", shown)
        turns.append(SimpleNamespace(code=extract_code(m["content"]), cell={"stdout": stdout, "error": None}))
    submitted = metrics.get("submitted") == 1.0
    answer = meta.get("submitted_answer") if submitted else None
    return SimpleNamespace(messages=messages, turns=turns, submitted=submitted, answer=answer, subcalls=[],
                           events=[], rejected=[], timeouts=0, family=meta["family"],
                           correct=metrics.get("correct") == 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=str(ROOT / ".art" / "archive" / "rlm-rlsd-vs-grpo"))
    args = parser.parse_args()
    import pyarrow.parquet as pq

    files = sorted(Path(args.root).glob("models/*/trajectories/train/*.parquet"))
    if not files:
        print(f"no trajectory files under {args.root}", file=sys.stderr)
        return 1
    totals, hits, wins_hit = collections.Counter(), collections.Counter(), collections.Counter()
    examples = []
    for path in files:
        arm = path.parts[-4]
        for row in pq.read_table(path).to_pylist():
            ep = episode(row)
            key = (arm, ep.family)
            totals[key] += 1
            totals[key + ("submitted",)] += ep.submitted
            for hit in detectors.check(None, ep):
                hits[key + (hit["rule"],)] += 1
                wins_hit[key + (hit["rule"],)] += ep.correct
                if len(examples) < 10:
                    examples.append(f"{arm} {ep.family}: {hit['rule']}: {hit['detail']}")
    print(f"{len(files)} trajectory files\n")
    print("| Arm | Family | Episodes | Submitted | " + " | ".join(detectors.RULES) + " |")
    print("|---|---|---|---|" + "---|" * len(detectors.RULES))
    for key in sorted(k for k in totals if len(k) == 2):
        cells = [f"{hits[key + (r,)]} ({wins_hit[key + (r,)]} wins)" for r in detectors.RULES]
        print(f"| {key[0]} | {key[1]} | {totals[key]} | {totals[key + ('submitted',)]} | " + " | ".join(cells) + " |")
    episodes = sum(v for k, v in totals.items() if len(k) == 2)
    print(f"\n{sum(hits.values())} detector hits over {episodes} episodes")
    for line in examples:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
