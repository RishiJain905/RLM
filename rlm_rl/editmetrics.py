"""What edit episodes did, beyond the reward (spec §12.9, code-editing-plan §8.3). Logged, never rewarded.

Per episode (`episode_stats`), from what the host recorded: each cell's code, output and whether it
changed /work/repo (the edit sandbox fingerprints the repository after every cell), the sandbox's
audit events (programs started, with their arguments; reads and writes of /work/repo; blocked
network attempts; each coding-tool call; each cell's count of writes and programs made by a tool
or not), the sub-calls, the diff and the grade. Per step (`step_block`), the `edit` block every
step record carries, with exactly §12.9's keys; the dashboard reads them.

Test commands count however they ran: through `shell()` (logged as `sh -c <command>`) or
`subprocess`. Edits count whether a tool or plain `open` made them (the repository's fingerprint).
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Iterable

from .repl_tools import TOOL_NAMES

# A program run that is a build or test command (from the exec event's arguments).
TEST_COMMAND = re.compile(
    r"(^|[\s/;&|])(pytest|py\.test|nosetests|tox|nox|jest|vitest|mocha|ava|tap|karma|ctest|phpunit|rspec)(\s|$)"
    r"|-m\s+(pytest|unittest|doctest)\b|\bunittest\s+discover\b"
    r"|(^|[\s/;&|])go\s+(test|vet|build)\b|(^|[\s/;&|])cargo\s+(test|nextest|build|check)\b"
    r"|(^|[\s/;&|])(npm|pnpm|yarn|bun)\s+(run\s+)?(test|build)\b|(^|[\s/;&|])npx\s+(jest|vitest|mocha|tsc)\b"
    r"|(^|[\s/;&|])(mvn|mvnw|\./mvnw)\b.*\b(test|verify|compile)\b|(^|[\s/;&|])(gradle|gradlew|\./gradlew)\b.*\b(test|check|build)\b"
    r"|(^|[\s/;&|])make\s+(test|check)\b|(^|[\s/;&|])tsc\b")
# A program run that fetches from the network: it fails with no network, and is logged as network_try (spec §12.6).
NETWORK_COMMAND = re.compile(
    r"\bpip3?\s+install\b|-m\s+pip\s+install\b|\b(npm|pnpm|yarn|bun)\s+(install|i|ci|add)\b|\bgo\s+(get|mod\s+download|install)\b"
    r"|\bcargo\s+(install|fetch|update|add)\b|(^|[\s/;&|])(curl|wget)\s|\bgit\s+(clone|fetch|pull|submodule\s+update)\b"
    r"|\bapt(-get)?\s+install\b|\bconda\s+install\b|\bgem\s+install\b|\bmvn\b.*\bdependency:|\buv\s+(pip\s+)?(add|install|sync)\b"
    r"|\bpoetry\s+(add|install)\b")
# A sub-call whose prompt holds a diff: drafting or reviewing a change (code-editing-plan §8.3).
DIFF_IN_PROMPT = re.compile(r"(^|\n)(diff --git |@@ -\d+(,\d+)? \+\d+(,\d+)? @@|--- a/\S+\n\+\+\+ b/)")

EDIT_KEYS = ("episodes", "resolved", "f2p_pass", "p2p_broken", "empty_diff", "submitted", "tested_before_submit",
             "test_runs", "edit_cycles", "turns_to_first_edit", "diff_lines", "files_changed", "subcall_read_share",
             "review_subcalls", "repeated_cells", "network_tries", "grader_s", "edit_errors", "raw_io_share", "grader_retries")
# Read separately, like by_language: {tool: mean calls per episode} over repl_tools.TOOL_NAMES.
TOOL_CALLS_KEY = "tool_calls"
IO_KEYS = ("tool_writes", "raw_writes", "tool_execs", "raw_execs")


def _argv_text(event: dict) -> str:
    argv = event.get("argv")
    if isinstance(argv, list):
        return " ".join(str(a) for a in argv)
    return str(event.get("detail") or "").strip("'\"")


def is_test_command(text: str) -> bool:
    return bool(TEST_COMMAND.search(text or ""))


def is_network_command(text: str) -> bool:
    return bool(NETWORK_COMMAND.search(text or ""))


def _cell_turns(ep: Any) -> dict[int, int]:
    """The sandbox counts cells; the k-th cell ran in the k-th turn that had code (1-based turn numbers)."""
    out, k = {}, 0
    for i, turn in enumerate(ep.turns, 1):
        if turn.code is not None:
            k += 1
            out[k] = i
    return out


def episode_stats(ep: Any) -> dict:
    """Spec §12.9's numbers for one edit episode."""
    turns = ep.turns
    cell_turn = _cell_turns(ep)
    events_by_turn: dict[int, list] = {}
    for ev in ep.events:
        t = cell_turn.get(int(ev.get("cell") or 0))
        if t is not None:
            events_by_turn.setdefault(t, []).append(ev)
    edits = [i for i, t in enumerate(turns, 1) if t.cell and t.cell.get("repo_changed")]
    tests_at: list[tuple[int, int]] = []  # (turn, position in the turn's events) of each test command
    network = 0
    for t, evs in events_by_turn.items():
        for pos, ev in enumerate(evs):
            kind = ev.get("kind")
            if kind == "exec":
                text = _argv_text(ev)
                if is_test_command(text):
                    tests_at.append((t, pos))
                if is_network_command(text):
                    network += 1
            elif kind == "network":
                network += 1
    tests_at.sort()
    last_edit = edits[-1] if edits else None
    tested = False
    if ep.submitted and last_edit is not None:
        last_writes = [pos for pos, ev in enumerate(events_by_turn.get(last_edit, [])) if ev.get("kind") == "repo_write"]
        cutoff = last_writes[-1] if last_writes else -1  # a program may have made the edit: count tests after it then
        tested = any(t > last_edit or (t == last_edit and pos > cutoff) for t, pos in tests_at)
    cycles, pending = 0, False
    test_turns = {t for t, _ in tests_at}
    for i in range(1, len(turns) + 1):
        if i in edits:
            pending = True
        if i in test_turns and pending:
            cycles += 1
            pending = False
    tool_calls = dict.fromkeys(TOOL_NAMES, 0)
    edit_errors = 0
    io_counts = dict.fromkeys(IO_KEYS, 0)
    for ev in ep.events:
        if ev.get("kind") == "tool" and ev.get("event") in tool_calls:
            tool_calls[ev["event"]] += 1
            edit_errors += ev["event"] == "edit_file" and ev.get("ok") is False
        elif ev.get("kind") == "io":
            for key in IO_KEYS:
                io_counts[key] += int(ev.get(key) or 0)
    io_total = sum(io_counts.values())
    raw_share = (io_counts["raw_writes"] + io_counts["raw_execs"]) / io_total if io_total else None
    seen, repeated = set(), 0
    for t in turns:
        if t.code is not None:
            key = t.code.strip()
            repeated += key in seen
            seen.add(key)
    sent = sum(len(c.get("prompt") or "") for c in ep.subcalls)
    printed = sum(min(int(t.cell.get("stdout_chars") or 0), 2000) + len(t.cell.get("error") or "")
                  for t in turns if t.cell)
    grade = ep.grade or {}
    graded = bool(grade.get("graded"))
    diff_lines, files = _diff_size(ep.diff or "")
    return {
        "tested_before_submit": float(tested),
        "test_runs": float(len(tests_at)),
        "edit_cycles": float(cycles),
        "turns_to_first_edit": float(edits[0]) if edits else None,
        "diff_lines": float(diff_lines),
        "files_changed": float(files),
        "subcall_read_share": round(sent / (sent + printed), 4) if sent + printed else 0.0,
        "review_subcalls": float(sum(bool(DIFF_IN_PROMPT.search(c.get("prompt") or "")) for c in ep.subcalls)),
        "repeated_cells": float(repeated),
        "network_tries": float(network),
        "grader_s": float(grade.get("seconds") or 0.0) if graded else None,
        "graded": float(graded),
        TOOL_CALLS_KEY: tool_calls,
        "edit_errors": float(edit_errors),
        "raw_io_share": round(raw_share, 4) if raw_share is not None else None,
        "grader_retries": float(grade.get("retries") or 0),  # graders rerun alone after hitting the cap
    }


def _diff_size(diff: str) -> tuple[int, int]:
    """Changed lines and files in a diff, binary files and symlinks left out (the grader strips them too)."""
    from .grader import parse_patch

    files = [fp for fp in parse_patch(diff) if not fp.binary and not fp.special]
    return sum(fp.added + fp.removed for fp in files), len(files)


def _mean(values: Iterable) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def step_block(episodes: list[Any]) -> dict | None:
    """A step's `edit` block (spec §12.9) over its scored edit episodes, or None when the step drew none."""
    from .bridge import is_edit

    drawn = [ep for ep in episodes if is_edit(ep.family)]
    if not drawn:
        return None
    eps = [ep for ep in drawn if ep.infra_error is None]
    comp = [ep.reward_components or {} for ep in eps]
    graded = [c for c in comp if c.get("graded")]
    with_diff = [c for c in comp if not c.get("empty_diff")]
    by_language: dict[str, dict] = {}
    for ep in eps:
        lang = str(ep.language or "unknown").lower()
        row = by_language.setdefault(lang, {"episodes": 0, "resolved": 0})
        row["episodes"] += 1
        row["resolved"] += bool(ep.correct)
    return {
        "episodes": len(eps),
        "resolved": _mean(float(bool(ep.correct)) for ep in eps),
        "f2p_pass": _mean(c.get("f2p_pass") for c in graded),
        "p2p_broken": _mean(c.get("p2p_broken") for c in graded),
        "empty_diff": _mean(c.get("empty_diff") for c in comp),
        "submitted": _mean(c.get("submitted") for c in comp),
        "tested_before_submit": _mean(c.get("tested_before_submit") for c in comp),
        "test_runs": _mean(c.get("test_runs") for c in comp),
        "edit_cycles": _mean(c.get("edit_cycles") for c in comp),
        "turns_to_first_edit": _mean(c.get("turns_to_first_edit") for c in comp),
        "diff_lines": _mean(c.get("diff_lines") for c in with_diff),
        "files_changed": _mean(c.get("files_changed") for c in with_diff),
        "subcall_read_share": _mean(c.get("subcall_read_share") for c in comp),
        "review_subcalls": _mean(c.get("review_subcalls") for c in comp),
        "repeated_cells": _mean(c.get("repeated_cells") for c in comp),
        "network_tries": _mean(c.get("network_tries") for c in comp),
        "grader_s": _mean(c.get("grader_s") for c in graded),
        "edit_errors": _mean(c.get("edit_errors") for c in comp),
        "raw_io_share": _mean(c.get("raw_io_share") for c in comp),
        "grader_retries": _mean(c.get("grader_retries") for c in comp),
        TOOL_CALLS_KEY: {name: round(statistics.fmean(float((c.get(TOOL_CALLS_KEY) or {}).get(name) or 0) for c in comp), 4)
                         for name in TOOL_NAMES} if comp else {},
        "by_language": {lang: {"episodes": r["episodes"], "resolved": round(r["resolved"] / r["episodes"], 4)}
                        for lang, r in sorted(by_language.items())},
    }
