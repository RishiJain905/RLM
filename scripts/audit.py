"""Read episode transcripts, training and eval alike, for the hack audit (H9, H11, decision 5a).

    python3 scripts/audit.py summary results/<run>                  # every log under the run
    python3 scripts/audit.py sample results/<run> --split val -n 5  # sampled wins, as Markdown for a Claude audit
    python3 scripts/audit.py show <log.jsonl> <episode_id>          # one transcript in full

`summary` prints one row per log: episodes, infra errors and their kinds, accuracy, and the
things a reviewer looks at first (first-turn submits, sandbox audit events, refused bridge
requests, cell timeouts). The rule-based detectors that zero rewards arrive in step 5.
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl.transcripts import TranscriptLog, find_logs  # noqa: E402

CLIP = 1500


def clip(text, n=CLIP) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= n else text[:n] + f"\n[... {len(text) - n} more characters]"


def flags(r: dict) -> list[str]:
    """What a reviewer should look at first. Not the step 5 detectors: nothing here changes a reward."""
    out = []
    if r["metrics"].get("first_turn_submit"):
        out.append("submitted on turn 1")
    if r["events"]:
        kinds = collections.Counter(e.get("kind") for e in r["events"])
        out.append("sandbox events: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items(), key=str)))
    if r["rejected"]:
        out.append(f"{len(r['rejected'])} refused bridge requests")
    if r["timeouts"]:
        out.append(f"{r['timeouts']} cell timeouts")
    if r["submitted"] and not r["answer_valid"]:
        out.append("invalid submit")
    return out


def summary(paths: list[Path], root: Path) -> int:
    print(f"{'log':<46} {'episodes':>8} {'infra':>6} {'acc':>6} {'turn-1':>6} {'events':>6} {'refused':>7} {'t/o':>4}")
    for path in paths:
        log = TranscriptLog(path)
        records = log.records()
        clean = [r for r in records if not r.get("infra_error")]
        infra = collections.Counter(r.get("infra_kind") for r in records if r.get("infra_error"))
        acc = sum(bool(r["correct"]) for r in clean) / len(clean) if clean else float("nan")
        name = str(path.relative_to(root) if root.is_dir() else path.name)
        print(f"{name:<46} {len(records):>8} {sum(infra.values()):>6} {acc:>6.2f} "
              f"{sum(bool(r['metrics'].get('first_turn_submit')) for r in clean):>6} "
              f"{sum(len(r['events']) for r in records):>6} {sum(len(r['rejected']) for r in records):>7} "
              f"{sum(r['timeouts'] for r in records):>4}")
        if infra:
            print(f"{'':<4}infra: " + ", ".join(f"{k} {n}" for k, n in sorted(infra.items(), key=str)))
        if log.torn_lines:
            print(f"{'':<4}{log.torn_lines} torn line(s) skipped")
        duplicates = len(log.all_records()) - len(records)
        if duplicates:
            print(f"{'':<4}{duplicates} earlier attempt(s) replaced by a rerun")
    return 0


def render(r: dict) -> str:
    lines = [f"### {r['split']} {r.get('eval') or ''} step {r['step']} · task {r['task_id']} ({r['family']}, level "
             f"{r['level']}) · episode {r['episode_id']}",
             f"- Gold: `{r['gold']}` · submitted: `{r['answer']}` · reward {r['reward']} · correct {r['correct']}",
             f"- Turns {len(r['turns'])}, sub-calls {r['subcalls_used']}, root model `{r['root_model']}`"]
    for flag in flags(r):
        lines.append(f"- **Flag:** {flag}")
    lines.append(f"\n**Question:** {r['question']}\n")
    for i, turn in enumerate(r["turns"], 1):
        lines.append(f"#### Turn {i}\n\n{clip(turn['reply'], 4000)}\n")
        cell = turn["cell"]
        if cell:
            lines.append("REPL output:\n```text\n" + clip(cell["stdout"]) + "\n```")
            if cell["error"]:
                lines.append("Error:\n```text\n" + clip(cell["error"], 600) + "\n```")
    if r["subcalls"]:
        lines.append(f"\n#### Sub-calls ({len(r['subcalls'])})\n")
        for j, call in enumerate(r["subcalls"], 1):
            lines.append(f"{j}. prompt ({len(call['prompt'])} chars): `{clip(call['prompt'], 200)!s}`\n"
                         f"   output: `{clip(call['output'] if call['output'] is not None else call['error'], 300)}`")
    return "\n".join(lines) + "\n"


def sample(paths: list[Path], split: str | None, n: int, seed: int, wins_only: bool) -> int:
    records = [r for p in paths for r in TranscriptLog(p).records()
               if not r.get("infra_error") and (split is None or r["split"] == split)]
    pool = [r for r in records if r["correct"]] if wins_only else records
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    print(f"# Audit sample: {len(picked)} of {len(pool)} {'wins' if wins_only else 'episodes'} (seed {seed})\n")
    for r in picked:
        print(render(r))
    return 0


def show(path: Path, episode_id: str) -> int:
    for r in TranscriptLog(path).all_records():
        if r["episode_id"] == episode_id:
            print(render(r))
            print("Full record:\n```json\n" + json.dumps(r, indent=1, ensure_ascii=False)[:200_000] + "\n```")
            return 0
    print(f"no episode {episode_id} in {path}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("summary")
    p.add_argument("root")
    p = sub.add_parser("sample")
    p.add_argument("root")
    p.add_argument("--split", choices=["train", "val"])
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true", help="sample from every episode, not just wins")
    p = sub.add_parser("show")
    p.add_argument("log")
    p.add_argument("episode_id")
    args = parser.parse_args()

    if args.command == "show":
        return show(Path(args.log), args.episode_id)
    paths = find_logs(args.root)
    if not paths:
        print(f"no transcript logs under {args.root}", file=sys.stderr)
        return 1
    if args.command == "summary":
        return summary(paths, Path(args.root))
    return sample(paths, args.split, args.n, args.seed, not args.all)


if __name__ == "__main__":
    sys.exit(main())
