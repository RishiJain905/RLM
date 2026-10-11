"""Episode transcripts: one JSON line per finished episode, for training and eval alike (H9).

Every episode is written in full: the conversation exactly as the model saw it, each cell's
result, every sub-call, the reward and its components, the hack detectors that fired, the infra
flag, the sandbox's audit events, token use, and the checkpoint that produced it. The hack audit (scripts/audit.py) reads
training and eval logs the same way.

    results/<run>/transcripts/train/step0007.jsonl
    results/<run>/transcripts/val/<eval name>/step0010.jsonl

A record never holds the task's seed (H3). The context is identified by its size and hash, and
the first user message already shows its first 3 lines. Records do hold gold answers, so
results/*/transcripts/ is git-ignored and stays on this PC.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from . import bridge

FORMAT_VERSION = 2  # 2 adds "hacks", the detectors that fired (step 5)


def transcript_path(run_dir: str | Path, split: str, step: int, eval_name: str | None = None) -> Path:
    base = Path(run_dir) / "transcripts" / split
    if eval_name is not None:
        if not eval_name or any(c in eval_name for c in "/\\.") or eval_name != eval_name.strip():
            raise ValueError(f"bad eval name {eval_name!r}")
        base = base / eval_name
    return base / f"step{step:04d}.jsonl"


def episode_record(ep: Any, *, split: str, run: str, step: int, key: str, eval_name: str | None = None,
                   sample: int = 0) -> dict:
    """Everything about one finished episode, as plain JSON. `ep` is an env.Episode."""
    return {
        "format": FORMAT_VERSION,
        "key": key,
        "split": split,
        "run": run,
        "step": step,
        "eval": eval_name,
        "sample": sample,
        "episode_id": ep.episode_id,
        "task_id": ep.task_id,
        "family": ep.family,
        "level": ep.level,
        "question": ep.question,
        "gold": ep.gold,
        "root_model": ep.root_model,
        "subcall_model": ep.subcall_model,
        "api_version": bridge.API_VERSION,
        "prompt_version": bridge.PROMPT_VERSION,
        "prompt_hash": bridge.PROMPT_HASH,
        "context_chars": ep.context_chars,
        "context_lines": ep.context_lines,
        "context_sha256": ep.context_sha256,
        "messages": ep.messages,
        "turns": [dataclasses.asdict(t) for t in ep.turns],
        "subcalls": ep.subcalls,
        "subcalls_used": ep.subcalls_used,
        "submitted": ep.submitted,
        "answer": ep.answer,
        "answer_valid": ep.answer_valid,
        "reward": ep.reward,
        "correct": ep.correct,
        "reward_components": ep.reward_components,
        "hacks": ep.hacks,
        "infra_error": ep.infra_error,
        "infra_kind": ep.infra_kind,
        "host_traceback": ep.host_traceback,
        "events": ep.events,
        "rejected": ep.rejected,
        "timeouts": ep.timeouts,
        "crashes": ep.crashes,
        "restarts": ep.restarts,
        "usage": ep.usage.as_dict(),
        "metrics": ep.metrics(),
        "started_at": ep.started_at,
        "wall_s": ep.wall_s,
        "logged_at": time.time(),
    }


class TranscriptLog:
    """An append-only JSONL file of episode records. Each line is flushed to disk as it is written,
    so a crash loses at most the line being written, and the reader skips a torn last line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.torn_lines = 0

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def all_records(self) -> list[dict]:
        """Every line, in the order written, including infra attempts a rerun replaced."""
        if not self.path.exists():
            return []
        records, self.torn_lines = [], 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    self.torn_lines += 1
        return records

    def records(self) -> list[dict]:
        """One record per episode key: the newest. A rerun's line replaces an earlier one for the same key."""
        latest: dict[str, dict] = {}
        for record in self.all_records():
            latest.pop(record["key"], None)  # ordered by when each key's newest line was written
            latest[record["key"]] = record
        return list(latest.values())

    def done_keys(self) -> set[str]:
        """Keys with a finished episode. An infra episode doesn't count, so a rerun runs it again."""
        return {r["key"] for r in self.records() if not r.get("infra_error")}


def find_logs(root: str | Path) -> list[Path]:
    """Every transcript log under a run directory, or the one file given."""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.rglob("step*.jsonl"))


def load(paths: Iterable[str | Path]) -> list[dict]:
    return [r for p in paths for r in TranscriptLog(p).records()]
