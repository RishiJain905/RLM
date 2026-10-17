"""The step 11 dashboard: plan step 11's baselines (rlm_rl/baselines.py) on a page of their own (Rishi, 2026-10-16).

A baseline run has no training steps, so the training dashboard (dashboard.py) doesn't fit it. This page reads the run's
summary.json, progress.json, spend.json and transcripts, and shows, per arm: pass rates by family and level, the edit
tiers, cost and time per episode, infra and rate limiting, and every episode as one row. It answers the plan's step 11
gates as the numbers come in.

The same rules as the training dashboard: an allowlist, never a copy, and `dashboard.check()` scans every build. Every
episode here is a selection-set task, which is an eval item, so no task text reaches the page: no question, context,
answer, gold, task id, reply, code or sub-call, and for edit tasks no repository. An episode row is its arm, family,
level, language (edit tasks), outcome and its numbers. The transcripts themselves stay on this PC.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from . import dashboard
from .dashboard import LANGUAGES, _num, scrub

TEMPLATE = Path(__file__).with_name("baselines_page.html")
SCHEMA = 1

ARM_LABELS = {"rlm": "Base model as an RLM", "direct": "Base model reading directly",
              "rlm_medium": "Base model as an RLM, root thinking medium"}
GATES = {"turn_cap_unresolved": 0.20, "edit_real_l1": 0.15, "medium_gain": 0.12}
# Production's mix (spec §12.7) and plan figures, for the cost projection (code-editing-plan.md §10.2).
PRODUCTION_MIX = {"aggregation": 0.25, "callgraph": 0.10, "needle": 0.10, "repo_loc": 0.15, "edit_synth": 0.15,
                  "edit_real": 0.25}
EPISODES_PER_STEP = 32
PRODUCTION_CEILING_USD = 157.0
PLAN_STEP_USD = 2.10
PLAN_EDIT_USD = {"edit_synth": 0.06, "edit_real": 0.15}
FAMILY_TO_MIX = {"aggregation": "aggregation", "callgraph": "callgraph", "needle": "needle",
                 "repo_loc_select": "repo_loc", "edit_synth": "edit_synth", "edit_real_select": "edit_real"}


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _name_ok(name: str) -> bool:
    return isinstance(name, str) and 0 < len(name) <= 60 and all(c.isalnum() or c in "_-/" for c in name)


def _rate(g: Any) -> dict:
    g = g if isinstance(g, dict) else {}
    return {"n": _num(g.get("n")), "rate": _num(g.get("rate", g.get("accuracy"))), "se": _num(g.get("se"))}


def _tier_key_ok(key: str) -> bool:
    """edit_real, edit_real/L1 or edit_real/python: a tier, a level or an allowed language, never anything else."""
    head, _, tail = key.partition("/")
    if head not in ("edit_real", "edit_synth"):
        return False
    return not tail or (tail[:1] == "L" and tail[1:].isdigit()) or tail in LANGUAGES


def _kind(k: Any) -> dict:
    k = k if isinstance(k, dict) else {}
    keys = ("n", "cost_mean_usd", "cost_total_usd", "cost_p90_usd", "cost_max_usd", "wall_median_s", "wall_p90_s",
            "wall_max_s", "turns_mean", "subcalls_mean", "root_prompt_tokens_mean", "root_completion_tokens_mean",
            "reasoning_tokens_mean", "truncated_turns_mean")
    return {**{key: _num(k.get(key)) for key in keys}, "accuracy": _rate(k.get("accuracy"))}


def _arm(name: str, s: dict, prog: dict, reqs: dict) -> dict:
    edit = s.get("edit") or {}
    att = s.get("attempts") or {}
    return {
        "name": name, "label": ARM_LABELS.get(name, name),
        "planned": _num(s.get("planned")), "logged": _num(s.get("logged")),
        "n": _num(s.get("n")), "accuracy": _num(s.get("accuracy")), "accuracy_se": _num(s.get("accuracy_se")),
        "submitted": _num(s.get("submitted")), "turns": _num(s.get("turns")), "subcalls": _num(s.get("subcalls")),
        "budget_hit": _num(s.get("budget_hit")), "hacked": _num(s.get("hacked")),
        "hacks": {k: v for k, v in (s.get("hacks") or {}).items() if _name_ok(k) and _num(v)},
        "est_cost_usd": _num(s.get("est_cost_usd")), "span_s": _num(s.get("span_s")),
        "by_family": {k: _rate(v) for k, v in (s.get("by_family") or {}).items() if _name_ok(k)},
        "by_level": {k: _rate(v) for k, v in (s.get("by_level") or {}).items() if _name_ok(k)},
        "by_language": {k: _rate(v) for k, v in (s.get("by_language") or {}).items() if k in LANGUAGES},
        "kinds": {k: _kind((s.get("kinds") or {}).get(k)) for k in ("read", "edit")},
        "tool_call": {k: {**{f: _num(((s.get("tool_call") or {}).get(k) or {}).get(f))
                             for f in ("episodes", "episodes_with", "turns", "turns_with", "first_turn")},
                          "accuracy_with": _rate(((s.get("tool_call") or {}).get(k) or {}).get("accuracy_with")),
                          "accuracy_without": _rate(((s.get("tool_call") or {}).get(k) or {}).get("accuracy_without"))}
                      for k in ("read", "edit")},
        "edit": None if not edit else {
            "tiers": {k: {**_rate(v), "cost_mean_usd": _num(v.get("cost_mean_usd")),
                          "wall_median_s": _num(v.get("wall_median_s")), "turns_mean": _num(v.get("turns_mean"))}
                      for k, v in (edit.get("tiers") or {}).items() if _tier_key_ok(k)},
            **{k: _num(edit.get(k)) for k in ("turn_cap_hits", "turn_cap_hits_unresolved", "turn_cap_rate",
                                              "turn_cap_unresolved_rate", "submitted", "empty_diff", "f2p_pass_mean",
                                              "tested_before_submit", "grader_s_median")}},
        "attempts": {"total": _num(att.get("total")), "infra": _num(att.get("infra")),
                     "kinds": dashboard._counts(att.get("infra_kinds")), "infra_cost_usd": _num(att.get("infra_cost_usd"))},
        "progress": {k: _num((prog or {}).get(k)) for k in ("planned", "done", "running", "infra", "skipped_budget")},
        "requests": {**{k: _num((reqs or {}).get(k)) for k in ("attempts", "retries", "rate_limited", "server_errors",
                                                                "timeouts", "connection_errors", "rejected", "paced_s")},
                     "statuses": dashboard._counts((reqs or {}).get("statuses"))},
    }


def _outcome(r: dict) -> str:
    if r.get("infra_error"):
        return "infra"
    if r.get("correct"):
        return "correct"
    return "wrong" if r.get("submitted") else "no answer"


def episode_rows(run_dir: Path, arms: list[str]) -> list[dict]:
    """One row per episode attempt, infra attempts included: numbers only, no task text."""
    rows = []
    for arm in arms:
        for path in sorted((Path(run_dir) / "transcripts" / "val" / arm).glob("step*.jsonl")):
            for r in dashboard._read_jsonl(path):
                edit = dashboard.is_edit_family(r.get("family"))
                lang = str(r.get("language") or "").lower()
                usage = r.get("usage") or {}
                turns = r.get("turns") or []
                rows.append({
                    "arm": arm if _name_ok(arm) else "arm", "family": r.get("family") if _name_ok(r.get("family")) else "?",
                    "level": _num(r.get("level")), "sample": _num(r.get("sample")),
                    "language": lang if edit and lang in LANGUAGES else None, "edit": edit,
                    "outcome": _outcome(r), "infra_kind": scrub(r.get("infra_kind"), 30) if r.get("infra_error") else None,
                    "turns": len(turns), "max_turns": _num((r.get("limits") or {}).get("max_turns")),
                    "subcalls": _num(r.get("subcalls_used")),
                    "root_tokens": _num(usage.get("root_completion_tokens")),
                    "reasoning_tokens": _num(usage.get("root_reasoning_tokens")),
                    "truncated_turns": sum(t.get("finish_reason") == "length" for t in turns),
                    "tool_call_turns": sum("<tool_call>" in (t.get("reply") or "") for t in turns),
                    "cost_usd": _num(usage.get("est_cost_usd")), "wall_s": _num(r.get("wall_s")),
                    "context_chars": _num(r.get("context_chars")), "finished_at": _num(r.get("logged_at")),
                    "f2p_pass": _num((r.get("reward_components") or {}).get("f2p_pass")) if edit else None,
                })
    rows.sort(key=lambda x: -(x["finished_at"] or 0))
    return rows


def projection(arms: dict) -> dict | None:
    """Production's cost per step from the rlm arm's measured cost per family, at spec §12.7's mix *(inferred: the
    selection set spreads levels 1-5, while production starts every family at level 1)*."""
    rlm = arms.get("rlm")
    if not rlm or not rlm.get("n"):
        return None
    run_dir_costs = rlm.get("_family_costs") or {}
    per_mix: dict[str, float] = {}
    for family, mix in FAMILY_TO_MIX.items():
        if family in run_dir_costs:
            per_mix[mix] = run_dir_costs[family]
    if set(per_mix) != set(PRODUCTION_MIX):
        return {"complete": False, "measured": sorted(per_mix)}
    step = EPISODES_PER_STEP * sum(PRODUCTION_MIX[m] * per_mix[m] for m in PRODUCTION_MIX)
    out = {"complete": True, "per_family_usd": per_mix, "step_usd": step,
           "steps_for_ceiling": PRODUCTION_CEILING_USD / step if step else None,
           "plan_step_usd": PLAN_STEP_USD, "plan_steps": round(PRODUCTION_CEILING_USD / PLAN_STEP_USD)}
    medium = arms.get("rlm_medium")
    if medium and medium.get("est_cost_usd") and medium.get("_matched_rlm_usd"):
        ratio = medium["est_cost_usd"] / medium["_matched_rlm_usd"]
        out.update(medium_ratio=ratio, medium_step_usd=step * ratio,
                   medium_steps_for_ceiling=PRODUCTION_CEILING_USD / (step * ratio))
    return out


def _family_costs(run_dir: Path, arm: str) -> dict:
    costs: dict[str, list] = {}
    for path in (Path(run_dir) / "transcripts" / "val" / arm).glob("step*.jsonl"):
        latest = {}
        for r in dashboard._read_jsonl(path):
            latest[r.get("key")] = r
        for r in latest.values():
            if not r.get("infra_error"):
                costs.setdefault(r.get("family"), []).append((r.get("usage") or {}).get("est_cost_usd") or 0.0)
    return {f: statistics.fmean(v) for f, v in costs.items() if v}


def _matched_cost(run_dir: Path, arm: str, other: str) -> tuple[float, float] | None:
    """The two arms' total cost over the task samples both finished, for a like-for-like cost ratio."""
    def latest(a):
        out = {}
        for path in (Path(run_dir) / "transcripts" / "val" / a).glob("step*.jsonl"):
            for r in dashboard._read_jsonl(path):
                if not r.get("infra_error"):
                    out[(r.get("task_id"), r.get("sample"))] = (r.get("usage") or {}).get("est_cost_usd") or 0.0
        return out

    a, b = latest(arm), latest(other)
    common = set(a) & set(b)
    if not common:
        return None
    return sum(a[k] for k in common), sum(b[k] for k in common)


def page_data(run_dir: Path, *, pot: dict | None = None, dry_run: bool = False, now: float | None = None) -> dict:
    now = now or time.time()
    run_dir = Path(run_dir)
    summary = _load(run_dir / "summary.json", {})
    progress = _load(run_dir / "progress.json", {})
    spend = _load(run_dir / "spend.json", {})
    arm_names = [a for a in ("rlm", "direct", "rlm_medium") if a in (summary.get("arms") or {})]
    arms = {}
    for a in arm_names:
        arms[a] = _arm(a, summary["arms"][a], (progress.get("arms") or {}).get(a, {}),
                       (progress.get("requests") or {}).get(a, {}))
    family_costs = {a: _family_costs(run_dir, a) for a in arm_names}
    proj_input = {a: dict(arms[a]) for a in arms}
    if "rlm" in proj_input:
        proj_input["rlm"]["_family_costs"] = family_costs.get("rlm", {})
    if "rlm_medium" in proj_input:
        matched = _matched_cost(run_dir, "rlm_medium", "rlm")
        if matched:
            proj_input["rlm_medium"]["est_cost_usd"], proj_input["rlm_medium"]["_matched_rlm_usd"] = matched
    comparisons = {}
    for k, c in (summary.get("comparisons") or {}).items():
        if _name_ok(k) and isinstance(c, dict):
            comparisons[k] = {f: _num(c.get(f)) for f in ("tasks", "a", "b", "diff", "se", "b_only", "a_only")}
    running = any((arms[a]["progress"].get("running") or 0) > 0 for a in arms)
    updated = _num(progress.get("updated_at")) or _num(summary.get("built_at"))
    stale = running and updated is not None and now - updated > dashboard.STALE_AFTER_S
    status = "stalled" if stale else "running" if running else ("finished" if arms and all(
        (arms[a]["progress"].get("done") or 0) >= (arms[a]["planned"] or 0) for a in arms) else "paused")
    return {
        "schema": SCHEMA, "kind": "baselines", "built_at": now, "dry_run": bool(dry_run),
        "run": {"name": scrub(run_dir.name, 60), "eval_set": scrub(summary.get("eval_set"), 30), "status": status,
                "updated_at": updated},
        "spend": {"total_usd": _num(spend.get("total_usd")), "ceiling_usd": _num(spend.get("ceiling_usd")),
                  "by_arm": {k: _num(v) for k, v in (spend.get("by_arm") or {}).items() if _name_ok(k)},
                  "reserved_usd": _num(progress.get("reserved_usd"))},
        "pot": {k: v for k, v in (pot or {}).items() if k in ("inference_left_usd", "as_of")},
        "arms": arms,
        "comparisons": comparisons,
        "gates": dict(GATES),
        "production": projection(proj_input),
        "family_costs": {a: {f: _num(v) for f, v in fc.items() if _name_ok(f)} for a, fc in family_costs.items()},
        "plan_edit_usd": dict(PLAN_EDIT_USD),
        "episodes": episode_rows(run_dir, arm_names),
    }


def render(data: dict) -> str:
    return dashboard.render(data, TEMPLATE.read_text(encoding="utf-8"))


def build(run_dir: Path, out: Path, *, seed_roots: list[Path], pot: dict | None = None, dry_run: bool = False) -> dict:
    """Build the page and scan it (dashboard.check); the page is written only if the scan passes."""
    page = render(page_data(run_dir, pot=pot, dry_run=dry_run))
    items, _ = dashboard.eval_items(run_dir)
    task_ids, repos = dashboard.eval_names(run_dir)
    result = dashboard.check(page, seeds=dashboard.private_seeds(seed_roots), items=items, task_ids=task_ids,
                             repos=repos)
    if result["ok"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(page, encoding="utf-8")
        tmp.replace(out)
        result["page"] = str(out)
    return result


def _fmt(x: float | None) -> str:  # for tests and the command line
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
