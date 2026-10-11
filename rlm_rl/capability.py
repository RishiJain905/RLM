"""The capability check: ~100 short prompts that catch frying (research §8.5, stop rule 3).

Plan step 7 builds the prompt set (IFEval, an MMLU-Pro subset, a short coding set and chat
prompts). This module runs it: one plain chat call per prompt to a checkpoint, thinking off and
fixed decoding, each reply graded by checks that need no judge. The run compares every
checkpoint's accuracy with checkpoint 0's.

A prompt set is a JSONL file, one item per line:

    {"id": "mmlu-0007", "source": "mmlu_pro", "prompt": "...", "checks": [{"kind": "choice", "value": "C"}]}
    {"id": "ifeval-0003", "source": "ifeval", "prompt": "...",
     "checks": [{"kind": "max_words", "value": 50}, {"kind": "lowercase"}]}

A reply passes when every check passes. A prompt whose call fails after retries is an infra
failure: it is left out of the accuracy and counted, and a check that is less than 90% complete
isn't compared (stoprules.CAPABILITY_MIN_COMPLETE).
"""

from __future__ import annotations

import asyncio
import collections
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .bridge import InfraError
from .client import ModelClient

CHECK_KINDS = ("exact", "choice", "number", "contains", "not_contains", "regex", "max_words", "min_words",
               "lowercase", "json")
MAX_AT_ONCE = 32

_CHOICE_RES = [re.compile(r"answer is\s*\(?([A-J])\)?", re.I), re.compile(r"answer:\s*\(?([A-J])\)?", re.I),
               re.compile(r"^\s*\(?([A-J])\)?[.)]?\s*$", re.M)]
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Item:
    id: str
    prompt: str
    checks: list[dict]
    source: str = ""
    system: str | None = None

    def messages(self) -> list[dict]:
        return ([{"role": "system", "content": self.system}] if self.system else []) + \
               [{"role": "user", "content": self.prompt}]


def resolve(path: str | Path) -> Path:
    """A prompt-set path: `data:<rel>` is relative to the data root (rlm_rl/data.py), anything else to the repo."""
    text = str(path)
    if text.startswith("data:"):
        from . import data

        return data.root() / text.removeprefix("data:")
    return Path(__file__).resolve().parents[1] / text


def load(path: str | Path) -> list[Item]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        data = json.loads(line)
        item = Item(str(data["id"]), data["prompt"], list(data["checks"]), data.get("source", ""), data.get("system"))
        for check in item.checks:
            if check.get("kind") not in CHECK_KINDS:
                raise ValueError(f"{path}:{n}: unknown check kind {check.get('kind')!r}")
        items.append(item)
    if len({i.id for i in items}) != len(items):
        raise ValueError(f"{path}: item ids must be unique")
    return items


def _choice(text: str) -> str | None:
    for pattern in _CHOICE_RES:
        found = pattern.findall(text)
        if found:
            return found[-1].upper()
    return None


def passes(check: dict, text: str) -> bool:
    kind, value = check["kind"], check.get("value")
    stripped = text.strip()
    if kind == "exact":
        return stripped.strip("\"'`.").lower() == str(value).strip().lower()
    if kind == "choice":
        return _choice(text) == str(value).upper()
    if kind == "number":
        found = _NUMBER_RE.findall(text.replace(",", ""))
        return bool(found) and abs(float(found[-1]) - float(value)) <= float(check.get("tolerance", 1e-6))
    if kind == "contains":
        return str(value).lower() in text.lower()
    if kind == "not_contains":
        return str(value).lower() not in text.lower()
    if kind == "regex":
        return re.search(value, text, re.S) is not None
    if kind == "max_words":
        return len(text.split()) <= int(value)
    if kind == "min_words":
        return len(text.split()) >= int(value)
    if kind == "lowercase":
        return text == text.lower()
    if kind == "json":
        try:
            json.loads(stripped.removeprefix("```json").removeprefix("```").removesuffix("```"))
            return True
        except ValueError:
            return False
    raise ValueError(f"unknown check kind {kind!r}")


def grade(item: Item, text: str) -> bool:
    return all(passes(check, text) for check in item.checks)


@dataclass
class CheckResult:
    summary: dict
    records: list[dict] = field(default_factory=list)


async def run(items: list[Item], *, client: ModelClient, model: str, max_at_once: int = MAX_AT_ONCE) -> CheckResult:
    """Ask every prompt once and grade the replies."""
    gate = asyncio.Semaphore(max_at_once)
    cost_before = client.spend["chat"].cost_usd()

    async def one(item: Item) -> dict:
        async with gate:
            try:
                text = await client.chat(item.messages(), model=model)
            except InfraError as exc:
                return {"id": item.id, "source": item.source, "infra_error": str(exc), "passed": None, "reply": None}
        return {"id": item.id, "source": item.source, "infra_error": None, "passed": grade(item, text), "reply": text}

    records = await asyncio.gather(*(one(item) for item in items))
    graded = [r for r in records if r["passed"] is not None]
    by_source: dict[str, list[bool]] = collections.defaultdict(list)
    for r in graded:
        by_source[r["source"]].append(r["passed"])
    summary = {
        "items": len(items), "graded": len(graded), "infra": len(records) - len(graded),
        "complete": len(graded) / len(items) if items else 0.0,
        "accuracy": sum(r["passed"] for r in graded) / len(graded) if graded else None,
        "by_source": {s: {"n": len(v), "accuracy": sum(v) / len(v)} for s, v in sorted(by_source.items())},
        "est_cost_usd": round(client.spend["chat"].cost_usd() - cost_before, 6),
    }
    return CheckResult(summary, list(records))
