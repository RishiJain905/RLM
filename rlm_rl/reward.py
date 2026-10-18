"""The reward: exact answers, calibrated partial credit for near misses (spec §3, H1, decision 1b).

Every rollout gets one scalar, computed on the host in this order:
1. No submit, or an answer that can't be parsed: 0.
2. Exact answer: 1. Exact accuracy is the headline metric.
3. Near miss: partial credit calibrated against the blind guess, the answer a task's metadata
   alone gives without reading. It pays only for beating the blind guess:
   - numbers:  0.8 x max(0, 1 - err / e_blind), where e_blind is the blind guess's error
   - lists:    0.8 x max(0, (F1 - F1_blind) / (1 - F1_blind)), by set-F1, which is the same
               formula with err = 1 - F1 and e_blind = 1 - F1_blind
   - entities: no partial credit
   So the blind guess itself earns only its lucky exact hits.
4. A run that turns the format penalty on (off in production: prod-v1's branch 1 got worse under it) takes FORMAT_PENALTY off for each
   broken reply, up to FORMAT_PENALTY_MAX_REPLIES of them (`broken_replies`, applied in run_episode).
5. A hack detector fired: 0 (detectors.py; run_episode applies it before the group mean).
6. infra_error: not scored at all (env.py).

The task's generator supplies the blind guess in `task.grading["blind"]`. It is computed on
the host and never enters the sandbox.

    {"type": "int", "blind": 71}
    {"type": "float", "blind": 1234.5}
    {"type": "entity"}
    {"type": "list", "item": "entity", "blind": ["host-a", "host-b"]}
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

ANSWER_TYPES = ("int", "float", "entity", "list")
LIST_ITEM_TYPES = ("int", "float", "entity")

EXACT_REWARD = 1.0
NEAR_MISS_MAX = 0.8
# How fast near-miss credit falls, in units of the blind guess's error. 1 is the spec's formula: credit
# reaches 0 exactly at the blind guess's error (decision 1b, kept by decision 5b on 2026-10-11).
NEAR_MISS_SLOPE = 1.0
DECIMAL_TOLERANCE = 0.005
# A process penalty for broken replies (2026-10-18). In prod-v1 replies that end on their own mid-code, empty or as a
# bare tag grew from ~1% of turns to 25% by step 16, and they cascade within an episode. The outcome reward never told
# a win with broken replies from a clean one, so wins kept reinforcing them (research §4.1: small process penalties
# only once the logs show the need).
FORMAT_PENALTY = 0.1  # per broken reply *(mine)*
FORMAT_PENALTY_MAX_REPLIES = 3  # so the most it takes is 0.3 *(mine)*

# A minus sign counts only when it isn't joined to a word, so "host-42" parses as 42.
_NUMBER_RE = re.compile(r"(?:(?<!\w)-)?\d+(?:\.\d+)?")
_QUOTES = "\"'`‘’“”"
_LIST_SPLIT_RE = re.compile(r"[,;\n]")


@dataclass
class Score:
    reward: float
    correct: bool  # exact, and later False if a hack detector fired; the headline metric
    components: dict = field(default_factory=dict)


def settings() -> dict:
    """The pinned reward constants, for the run's config.json (spec §8)."""
    return {"exact_reward": EXACT_REWARD, "near_miss_max": NEAR_MISS_MAX, "near_miss_slope": NEAR_MISS_SLOPE,
            "decimal_tolerance": DECIMAL_TOLERANCE, "answer_types": list(ANSWER_TYPES),
            "edit": "binary: 1 when every fail-to-pass and pass-to-pass test passes (spec §12.6, decision E-c)",
            "format_penalty_max_replies": FORMAT_PENALTY_MAX_REPLIES}


def broken_replies(turns: Any) -> int:
    """Turns that ended on their own without a runnable code block: empty, prose only, a bare tag or a fence left
    open. A reply cut at the token cap isn't counted: that's the length limit, not the format."""
    return sum(t.code is None and t.finish_reason != "length" for t in turns)


def format_penalty(broken: int, per_reply: float) -> float:
    """What `broken` broken replies take off the reward, at `per_reply` each (0 turns it off)."""
    return round(per_reply * min(broken, FORMAT_PENALTY_MAX_REPLIES), 4)


# ----------------------------------------------------------------------------- parsing


def parse_number(value: Any) -> float | None:
    """The first number in an answer, after removing `$` and `,` (spec §3). None if there is none."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    match = _NUMBER_RE.search(value.replace("$", "").replace(",", ""))
    return float(match.group()) if match else None


def normalize_entity(value: Any) -> str | None:
    """Trim whitespace and quotes, ignore case. None for an empty answer or a list."""
    if isinstance(value, (bool, list)) or value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip().strip(_QUOTES).strip().casefold()
    return text or None


def _normalize_item(value: Any, item: str) -> Any:
    if item == "entity":
        return normalize_entity(value)
    number = parse_number(value)
    if number is None:
        return None
    return round(number, 2) if item == "float" else number


def normalize_set(value: Any, item: str) -> set:
    """A list answer as a set of normalized items. A string is split on commas, semicolons and newlines."""
    if isinstance(value, str):
        value = [part for part in _LIST_SPLIT_RE.split(value) if part.strip()]
    elif not isinstance(value, list):
        value = [value]
    items = {_normalize_item(v, item) for v in value}
    items.discard(None)
    return items


def set_f1(predicted: set, gold: set) -> float:
    if not predicted and not gold:
        return 1.0
    hits = len(predicted & gold)
    if not hits:
        return 0.0
    precision, recall = hits / len(predicted), hits / len(gold)
    return 2 * precision * recall / (precision + recall)


# ----------------------------------------------------------------------------- scoring


def check_grading(grading: dict, gold: Any) -> None:
    """Raise ValueError if a task can't be scored. Every number or list task must carry its blind guess."""
    kind = grading.get("type")
    if kind not in ANSWER_TYPES:
        raise ValueError(f"unknown answer type {kind!r}")
    if kind in ("int", "float") and (parse_number(gold) is None or parse_number(grading.get("blind")) is None):
        raise ValueError(f"a {kind} task needs a numeric gold answer and a numeric blind guess")
    if kind == "list":
        if grading.get("item") not in LIST_ITEM_TYPES:
            raise ValueError(f"a list task needs an item type in {LIST_ITEM_TYPES}")
        if not isinstance(gold, list) or not isinstance(grading.get("blind"), list):
            raise ValueError("a list task needs a list gold answer and a list blind guess")
    if kind == "entity" and normalize_entity(gold) is None:
        raise ValueError("an entity task needs a non-empty gold answer")


def near_miss_credit(err: float, e_blind: float) -> float:
    """0.8 x max(0, 1 - err / e_blind). If the blind guess was exact (e_blind = 0), a near miss earns 0."""
    if e_blind <= 0:
        return 0.0
    return NEAR_MISS_MAX * min(1.0, max(0.0, 1.0 - NEAR_MISS_SLOPE * err / e_blind))


_UNPARSED = {"parsed": 0.0, "exact": 0.0, "partial": 0.0}


def score_answer(answer: Any, gold: Any, grading: dict) -> Score:
    """Score one answer (spec §3 steps 1-3). `answer` is None when nothing valid was submitted."""
    check_grading(grading, gold)
    kind = grading["type"]
    if answer is None:
        return Score(0.0, False, dict(_UNPARSED))

    if kind in ("int", "float"):
        value, target, blind = parse_number(answer), parse_number(gold), parse_number(grading["blind"])
        if value is None:
            return Score(0.0, False, dict(_UNPARSED))
        err, e_blind = abs(value - target), abs(blind - target)
        exact = value == target if kind == "int" else err <= DECIMAL_TOLERANCE + 1e-9
        partial = 0.0 if exact else round(near_miss_credit(err, e_blind), 4)
        # A wrong answer nearer the blind guess than the gold looks like guessing from metadata. Logged only,
        # never part of the reward: if its rate climbs, the policy is drifting toward the guess.
        near_blind = float(not exact and abs(value - blind) < err)
        return Score(EXACT_REWARD if exact else partial, exact,
                     {"parsed": 1.0, "exact": float(exact), "partial": partial, "err": err, "e_blind": e_blind,
                      "near_blind": near_blind})

    if kind == "entity":
        value = normalize_entity(answer)
        if value is None:
            return Score(0.0, False, dict(_UNPARSED))
        exact = value == normalize_entity(gold)
        return Score(EXACT_REWARD if exact else 0.0, exact, {"parsed": 1.0, "exact": float(exact), "partial": 0.0})

    item = grading["item"]
    predicted, target = normalize_set(answer, item), normalize_set(gold, item)
    if not predicted and target:
        return Score(0.0, False, dict(_UNPARSED))
    exact = predicted == target
    f1, f1_blind = set_f1(predicted, target), set_f1(normalize_set(grading["blind"], item), target)
    partial = 0.0 if exact else round(near_miss_credit(1 - f1, 1 - f1_blind), 4)
    return Score(EXACT_REWARD if exact else partial, exact,
                 {"parsed": 1.0, "exact": float(exact), "partial": partial, "f1": round(f1, 4),
                  "f1_blind": round(f1_blind, 4)})


def score(task: Any, ep: Any) -> Score:
    """The scorer run_episode uses: spec §3 steps 1-3 for a finished episode. Hack zeroing comes after."""
    if task.grading.get("type") == "edit":
        return score_edit(task, ep)
    answer = ep.answer if ep.submitted and ep.answer_valid else None
    return score_answer(answer, task.answer, task.grading)


def score_edit(task: Any, ep: Any) -> Score:
    """An edit episode's reward (spec §12.6), before the detectors:
    1. no submit, an empty diff, or a diff that doesn't apply: 0
    3. every fail-to-pass and pass-to-pass test passed: 1 (`resolved`, the edit families' headline)
    4. anything else: 0 (binary, decision E-c)
    Step 2, a detector, comes after (run_episode); a grader failure never reaches here (infra_error).
    The components carry spec §12.9's per-episode numbers.
    """
    grade = ep.grade or {}
    empty = not (ep.diff or "").strip()
    graded = bool(grade.get("graded"))
    resolved = bool(ep.submitted and not empty and graded and grade.get("applied") and not grade.get("empty")
                    and grade.get("resolved"))
    f2p_total = grade.get("f2p_total") or 0
    components = {
        "resolved": float(resolved), "exact": float(resolved), "submitted": float(bool(ep.submitted)),
        "empty_diff": float(empty), "applied": float(bool(grade.get("applied", True))),
        "f2p_pass": round((grade.get("f2p_passed") or 0) / f2p_total, 4) if graded and f2p_total else (0.0 if graded else None),
        "p2p_broken": float((grade.get("p2p_failed") or 0) > 0) if graded else None,
        "build_failed": float(bool(grade.get("build_failed"))),
        **(getattr(ep, "edit_stats", None) or {}),
    }
    return Score(EXACT_REWARD if resolved else 0.0, resolved, components)


def blind_score(task: Any) -> Score:
    """What the task's own blind guess earns. The H1 check: never more than its exact hit."""
    return score_answer(task.grading.get("blind"), task.answer, task.grading)
