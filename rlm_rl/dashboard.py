"""The live dashboard's data (plan step 9): results/<run>.json in, one self-contained HTML page out.

A Claude artifact page can't fetch from W&B or from this PC, so the dashboard is rebuilt here and
republished by a Claude session after each step (decision 9a). `scripts/dashboard.py` is the
command line; this module only reads files and writes the page. It uses the standard library
alone, so it runs fast from Windows while the training loop runs in WSL.

What goes on the page is an allowlist, never a copy of the state:
- each step's metrics (spec §9, and §12.9's `edit` block for the edit families), evals, stop rules,
  notices, spend and the controller's levels
- a few sampled training rollouts (decision 9c), cut short. Eval transcripts are never read.

What never goes on it: keys, task seeds and eval seed lists (H3), eval and benchmark items,
model and checkpoint names (they carry the W&B entity), local file paths, and for the edit
families the gold patch, hidden test names, sandbox paths, eval task ids and the selection set's
repositories. `check()` scans every built page for all of these before it can be published, and
reports counts only, so a finding never prints the secret it found.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from .repl_tools import TOOL_NAMES

SCHEMA = 1
TEMPLATE = Path(__file__).with_name("dashboard_page.html")
TITLES = {"production": "RLM Production Run"}  # the page's <title> by run kind; others keep the template's
DATA_MARK = "/*__DASHBOARD_DATA__*/null"

# The two W&B pots (plan "Where we start"). W&B's billing page is the truth for both. Before production
# (Rishi, 2026-10-17) $191.01 was left, split like config.PRODUCTION_CEILINGS: production ceiling 146, evals 34,
# reserve 10, and ~1 of margin for the run's estimates against the bill. The second seed and the baselines are spent.
POT = {"inference_usd": 250.0, "inference_left_usd": 191.01, "production_usd": 146.0, "evals_usd": 34.0,
       "reserve_usd": 10.0, "margin_usd": 1.01, "gpu_usd": 1000.0, "gpu_left_usd": 992.0, "as_of": "2026-10-17",
       # W&B's training pot read during prod-v1 (Rishi, 2026-10-18: $12.43 used after 6 trained steps, when the run's own
       # estimate was 6 x $0.16): the meter shows that read plus the run's estimate for the steps since
       "gpu_used_usd": 12.43, "gpu_used_at_est_usd": 0.96, "gpu_as_of": "2026-10-18"}

# The 20-step run (documentation/archive/rlsd-vs-grpo-20-step-results.md): held-out accuracy over
# 100 episodes on the ticket tasks, the second seed's comparison.
BASELINES = [
    {"label": "Base model as an RLM (20-step run)", "eval": "heldout-seed1", "accuracy": 0.50, "n": 100},
    {"label": "GRPO, first seed (20-step run)", "eval": "heldout-seed1", "accuracy": 0.70, "n": 100},
    {"label": "RLSD (20-step run)", "eval": "heldout-seed1", "accuracy": 0.59, "n": 100},
]

# Sampled transcripts (decision 9c): from training rollouts only.
SAMPLES_PER_STEP = {"win": 1, "fail": 1, "flagged": 2}
KEEP_RECENT_STEPS = 6      # every sample from the newest steps
KEEP_FLAGGED = 24          # detector-flagged rollouts kept from any step
EVAL_EVERY_FALLBACK = 10   # samples at eval checkpoints are kept for the whole run
CUT = {"question": 900, "context_head": 600, "plan": 600, "code": 1000, "observation": 600,
       "answer": 400, "subcall": 220, "turns_head": 3, "turns_tail": 3, "subcalls_shown": 3}

STALE_AFTER_S = 300.0      # a running loop rewrites its state every 15 s (training.HEARTBEAT_S)

# The edit families (spec §12). Their gold is the gold patch and the hidden tests, which never reach the page.
EDIT_FAMILIES = ("edit_real", "edit_synth", "edit_synth_heldout")


def is_edit_family(family: Any) -> bool:
    """Any edit family, the eval-only splits included (edit_real_select, edit_real_test): bridge.EDIT_PREFIX. The
    page check keys and names eval tasks with this, since EDIT_FAMILIES alone missed edit_real's eval splits."""
    return str(family or "").startswith("edit_")
# Languages a by_language table may name (spec §12.9). Any other key, which could be a repository or a
# task id, is pooled into "other" so its episodes still count but its name never reaches the page.
LANGUAGES = ("python", "javascript", "typescript", "go", "rust", "java", "kotlin", "scala", "c", "cpp", "csharp",
             "ruby", "php", "swift", "shell")


# ----------------------------------------------------------------------------- reading the run


def read_state(path: Path, *, tries: int = 10) -> dict:
    """Read results/<run>.json. The loop swaps it in atomically, but a reader can still catch the
    swap on Windows (PermissionError) or a half-synced file over WSL's mount, so retry briefly."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            last = exc
            time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"could not read {path}: {last!r}")


def signature(state: dict) -> dict:
    """What a viewer would notice changing. A heartbeat alone changes nothing here, so a quiet
    run isn't republished every 15 s; a new step, eval, notice, audit verdict or status is."""
    audits = state.get("audits") or {}
    return {
        "run": state.get("run"),
        "steps": len(state.get("steps") or []),
        "last_step": (state.get("steps") or [{}])[-1].get("step"),
        "branch": state.get("branch"),
        "evals": len(state.get("evals") or []),
        "notices": len(state.get("notices") or []),
        "status": state.get("status"),
        "stop": (state.get("stop") or {}).get("rule"),
        "audited": sum(int((a or {}).get("audited") or 0) for a in audits.values()),
        "phase": ((state.get("live") or {}).get("phase") if (state.get("live") or {}).get("phase")
                  in ("eval", "capability") else None),
    }


def heartbeat_age(state: dict, now: float | None = None) -> float | None:
    """Seconds since the loop last wrote its state, or None when it isn't running."""
    if state.get("status") != "running":
        return None
    return (now or time.time()) - float(state.get("updated_at") or 0)


# ----------------------------------------------------------------------------- sanitizing text

_URL_RE = re.compile(r"\b(?:https?|wandb-artifact|s3|gs|file)://\S+", re.I)
_WIN_PATH_RE = re.compile(r"\b[A-Za-z]:[\\/][^\s'\"<>|]*")
_NIX_PATH_RE = re.compile(r"(?<![\w.])/(?:mnt|home|tmp|root|usr|opt|var|etc|proc|Users)/[^\s'\"<>|]*")
# the edit sandbox's repository and the images' own checkouts (spec §12.4)
_SANDBOX_PATH_RE = re.compile(r"(?<![\w.])/(?:work/repo|testbed|scratch)\b[^\s'\"<>|]*")
# SWE-bench and SWE-smith instance ids, owner__repo-1234 or owner__repo.abc123.kind__x: they name the repository
_TASK_ID_RE = re.compile(r"(?<![\w.\-])[A-Za-z0-9][A-Za-z0-9.\-]*[A-Za-z0-9]__[A-Za-z0-9][\w.\-]*")
_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{40,}\b")
_MODEL_RE = re.compile(r"\b[\w.\-]+/[\w.\-]+/[\w.\-]+:step\d+\b")


def scrub(text: Any, limit: int | None = None) -> str:
    """Remove URLs, local and sandbox paths, checkpoint names, task ids and long tokens from free text, then cut it."""
    s = "" if text is None else str(text)
    s = _URL_RE.sub("[link removed]", s)
    s = _MODEL_RE.sub("[checkpoint]", s)
    s = _WIN_PATH_RE.sub("[path]", s)
    s = _NIX_PATH_RE.sub("[path]", s)
    s = _SANDBOX_PATH_RE.sub("[path]", s)
    s = _TASK_ID_RE.sub("[task id]", s)
    s = _TOKEN_RE.sub("[token removed]", s)
    if limit is not None and len(s) > limit:
        s = s[:limit].rstrip() + f" [... {len(s) - limit:,} more characters]"
    return s


def _num(x: Any) -> Any:
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _pick(src: dict | None, keys: Iterable[str]) -> dict:
    src = src or {}
    return {k: _num(src.get(k)) for k in keys}


def _counts(src: dict | None) -> dict:
    """A {name: count} map with short, plain names (detector rules, infra kinds)."""
    out = {}
    for k, v in (src or {}).items():
        if isinstance(k, str) and re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", k) and _num(v) is not None:
            out[k] = v
    return out


def _by_language(src: dict | None, count: str, rate: str) -> dict:
    """{language: {count, rate}} with only LANGUAGES named. Any other key is pooled into "other", its rate
    weighted by its count, so the totals still add up without the name reaching the page."""
    pooled: dict[str, list] = {}
    for name, g in (src or {}).items():
        n = _num(g.get(count)) if isinstance(g, dict) else None
        if not n:
            continue
        lang = str(name).lower() if str(name).lower() in LANGUAGES else "other"
        r = _num(g.get(rate))
        acc = pooled.setdefault(lang, [0, 0.0, True])
        acc[0] += n
        acc[1] += (r or 0) * n
        acc[2] = acc[2] and r is not None
    return {lang: {count: n, rate: round(total / n, 4) if known else None}
            for lang, (n, total, known) in sorted(pooled.items())}


# ----------------------------------------------------------------------------- the page's data

PASS_KEYS = ("rate", "rate_with_infra", "groups", "trainable_groups", "all_zero_groups", "all_one_groups",
             "other_equal_groups", "small_groups")
REWARD_KEYS = ("mean", "exact", "partial", "near_blind", "submitted", "hacked")
POLICY_KEYS = ("entropy", "entropy_tokens", "reward_std", "kl_to_base", "kl_k3", "grad_norm", "probs_corr",
               "clip_fraction", "importance_ratio_mean", "importance_ratio_p95", "importance_ratio_p99", "loss")
SHAPE_KEYS = ("subcalls", "budget_hit", "turns", "first_turn_submit", "no_code_turns", "broken_replies", "format_penalty",
              "resampled_replies",
              "repl_errors", "timeouts",
              "root_completion_tokens", "subcall_tokens", "truncated_turns", "episodes_with_truncated_turn",
              "output_truncations", "context_chars", "wall_s")
REQUEST_KEYS = ("attempts", "retries", "rate_limited", "server_errors", "timeouts", "connection_errors", "rejected",
                "paced_s")
TOKEN_KEYS = ("root_prompt_tokens", "root_completion_tokens", "subcall_prompt_tokens", "subcall_completion_tokens",
              "root_calls", "subcalls")
FAMILY_KEYS = ("level", "episodes", "scored", "pass_rate", "mean_reward", "window_pass_rate")
# spec §12.9, over the step's edit episodes; by_language is read separately
EDIT_KEYS = ("episodes", "resolved", "f2p_pass", "p2p_broken", "empty_diff", "submitted", "tested_before_submit",
             "test_runs", "edit_cycles", "turns_to_first_edit", "diff_lines", "files_changed", "subcall_read_share",
             "review_subcalls", "repeated_cells", "network_tries", "grader_s", "edit_errors", "raw_io_share",
             "grader_retries")


def edit_block(edit: Any) -> dict | None:
    """A step's `edit` block (spec §12.9), allowlisted. None for old runs and steps without one."""
    if not isinstance(edit, dict):
        return None
    calls = edit.get("tool_calls")
    return {**_pick(edit, EDIT_KEYS), "tool_calls": _pick(calls if isinstance(calls, dict) else None, TOOL_NAMES),
            "by_language": _by_language(edit.get("by_language"), "episodes", "resolved")}


def step_row(r: dict) -> dict:
    """One step's metrics, allowlisted (spec §9 and step 6 §2.3)."""
    pol, shape, infra = r.get("policy") or {}, r.get("shape") or {}, r.get("infra") or {}
    reqs = infra.get("requests") or {}
    kl = pol.get("kl") or {}
    rtt = shape.get("root_transcript_tokens") or {}
    edit = edit_block(r.get("edit"))
    return {
        "step": r.get("step"), "status": r.get("status") if r.get("status") in ("trained", "skipped", "rolled_out")
        else None, "branch": _num(r.get("branch")), "lr": _num(r.get("lr")),
        "started_at": _num(r.get("started_at")), "finished_at": _num(r.get("finished_at")),
        "pass": {**_pick(r.get("pass"), PASS_KEYS),
                 "histogram": {k: _num(v) for k, v in ((r.get("pass") or {}).get("histogram") or {}).items()
                               if re.fullmatch(r"\d/4", k)}},
        "families": {name: {**_pick(f, FAMILY_KEYS), "saturated": bool((f or {}).get("saturated"))}
                     for name, f in (r.get("families") or {}).items() if re.fullmatch(r"[a-z0-9_]{1,40}", name)},
        "moves": [{"family": m.get("family"), "from": _num(m.get("from")), "to": _num(m.get("to")),
                   "pass_rate": _num(m.get("pass_rate")), "applied": bool(m.get("applied"))}
                  for m in (r.get("moves") or []) if re.fullmatch(r"[a-z0-9_]{1,40}", str(m.get("family")))],
        "reward": {**_pick(r.get("reward"), REWARD_KEYS), "hacks": _counts((r.get("reward") or {}).get("hacks"))},
        "policy": {**_pick(pol, POLICY_KEYS), "kl_tokens": _num(kl.get("tokens")), "kl_failed": _num(kl.get("failed")),
                   "kl_misaligned": _num(kl.get("misaligned_turns")), "kl_requests": _num(kl.get("requests")),
                   "kl_policy_nll": _num(kl.get("policy_nll")), "kl_base_nll": _num(kl.get("base_nll"))},
        "shape": {**_pick(shape, SHAPE_KEYS), "transcript_tokens_mean": _num(rtt.get("mean")),
                  "transcript_tokens_p95": _num(rtt.get("p95")), "transcript_tokens_max": _num(rtt.get("max"))},
        "infra": {**_pick(infra, ("episodes", "infra", "rate")), "kinds": _counts(infra.get("kinds")),
                  **_pick(reqs, REQUEST_KEYS), "peak_in_flight": _num((reqs.get("peaks") or {}).get("total")),
                  "limiter_wait_s": _num((reqs.get("peaks") or {}).get("waited_s"))},
        "infra_pause": scrub(r.get("infra_pause"), 300) if r.get("infra_pause") else None,
        "skip_train": scrub((r.get("filter") or {}).get("skip_train"), 200) or None,
        "cost": _pick(r.get("cost"), ("step_usd", "kl_usd", "spent_train_usd")),
        "time": _pick(r.get("time"), ("rollouts_s", "kl_s", "train_s", "total_s")),
        "tokens": _pick(r.get("tokens"), TOKEN_KEYS),
        **({"edit": edit} if edit is not None else {}),  # absent, not null, so old steps add no CSV column
        # the edit_real pool (12b): counts only, never a task id
        **({"pool": {**_pick(r["retired"], ("total", "left")),
                     "moved": _num(sum(_num((m or {}).get("slots")) or 0 for m in (r.get("fallback") or {}).values()))}}
           if isinstance(r.get("retired"), dict) and r["retired"].get("left") is not None else {}),
    }


EVAL_KEYS = ("episodes", "n", "accuracy", "accuracy_se", "mean_reward", "submitted", "first_turn_submit", "turns",
             "subcalls", "budget_hit", "hacked", "infra_episodes", "items", "graded", "infra", "complete")


def eval_entry(e: dict) -> dict:
    s = e.get("summary") or {}

    def groups(key):
        return {name: _pick(g, ("n", "accuracy")) for name, g in (s.get(key) or {}).items()
                if re.fullmatch(r"[A-Za-z0-9_\-/]{1,60}", name)}

    name = str(e.get("name") or "")
    return {"checkpoint": _num(e.get("checkpoint")), "name": name if re.fullmatch(r"[a-z0-9_\-]{1,40}", name) else "eval",
            "branch": _num(e.get("branch")), "at": _num(e.get("at")), "cost_usd": _num(e.get("cost_usd")),
            "ran": _num(e.get("ran")), "summary": {**_pick(s, EVAL_KEYS), "hacks": _counts(s.get("hacks")),
                                                   "by_family": groups("by_family"), "by_level": groups("by_level"),
                                                   "by_source": groups("by_source"),
                                                   "by_language": _by_language(s.get("by_language"), "n", "accuracy")}}


def _config(cfg: dict) -> dict:
    keep = ("kind", "learning_rate", "lr_rule", "tasks_per_step", "rollouts", "max_steps", "eval_every",
            "eval_at_start", "selection", "final_eval", "keep_every", "kl_samples", "base_model", "frozen_controller")
    out = {k: cfg.get(k) for k in keep if isinstance(cfg.get(k), (int, float, str, bool)) or cfg.get(k) is None}
    out["mix"] = {k: _num(v) for k, v in (cfg.get("mix") or {}).items() if re.fullmatch(r"[a-z0-9_]{1,40}", k)}
    out["ceilings"] = {k: _num(v) for k, v in (cfg.get("ceilings") or {}).items() if k in ("train", "eval")}
    out["has_capability_set"] = bool(cfg.get("capability_set"))
    return out


def _controller(ctl: dict) -> dict:
    fams = {}
    for name, f in (ctl.get("families") or {}).items():
        if re.fullmatch(r"[a-z0-9_]{1,40}", name):
            fams[name] = {"level": _num(f.get("level")), "max_level": _num(f.get("max_level")),
                          "moved_at": _num(f.get("moved_at")), "saturated": bool(f.get("saturated"))}
    history = [{"step": _num(m.get("step")), "family": m.get("family"), "from": _num(m.get("from_level")),
                "to": _num(m.get("to_level")), "pass_rate": _num(m.get("pass_rate")), "applied": bool(m.get("applied"))}
               for m in (ctl.get("history") or []) if m.get("family") in fams]
    return {"frozen": bool(ctl.get("frozen")), "families": fams, "history": history}


def _lr(lr: dict, decision: dict | None) -> dict:
    out = {k: lr.get(k) for k in ("current", "start", "floor", "rule", "fallback_used")}
    fb = lr.get("fallback") or None
    out["fallback"] = None if not fb else {**_pick(fb, ("at_step", "to_checkpoint", "from_lr", "to_lr")),
                                           "signal": scrub(fb.get("signal"), 40)}
    if decision:
        checks = {}
        for name, c in (decision.get("checks") or {}).items():
            checks[name] = {"passed": c.get("passed"), "need": scrub(c.get("need"), 80),
                            **{k: _num(v) for k, v in c.items() if k not in ("passed", "need")}}
        out["decision"] = {"production_lr": _num(decision.get("production_lr")), "passed": decision.get("passed"),
                           "note": scrub(decision.get("note"), 200), "checks": checks}
    return out


def page_data(state: dict, *, run_dir: Path | None = None, samples: list | None = None, dry_run: bool = False,
              pot: dict | None = None, baselines: list | None = None, now: float | None = None) -> dict:
    """Everything the page shows, built from an allowlist of the run's state."""
    now = now or time.time()
    spend = state.get("spend") or {}
    stop = state.get("stop") or None
    live = state.get("live") or None
    rules = {}
    for name, rule in (state.get("stop_rules") or {}).items():
        if re.fullmatch(r"[a-z0-9_]{1,30}", name):
            rules[name] = {"state": rule.get("state") if rule.get("state") in ("ok", "watch", "fired", "n/a") else "n/a",
                           "text": scrub(rule.get("text"), 300), "action": "pause" if rule.get("action") == "pause" else "stop"}
    audits = {}
    for c, a in (state.get("audits") or {}).items():
        if str(c).isdigit():
            audits[str(c)] = {"status": a.get("status") if a.get("status") in ("pending", "audited", "no_sample") else None,
                              **_pick(a, ("audited", "sampled", "clean_share"))}
    abandoned = [{"branch": _num(a.get("branch")), "after": _num(a.get("after")), "at": _num(a.get("at")),
                  "reason": scrub(a.get("reason"), 200), "steps": [step_row(r) for r in a.get("steps") or []],
                  "evals": [eval_entry(e) for e in a.get("evals") or []]}
                 for a in state.get("abandoned") or []]
    best = state.get("best") or None
    return {
        "schema": SCHEMA,
        "built_at": now,
        "dry_run": bool(dry_run),
        "run": {
            "name": scrub(state.get("run"), 60), "kind": state.get("kind"), "status": state.get("status"),
            "stop": None if not stop else {"rule": scrub(stop.get("rule"), 40), "reason": scrub(stop.get("reason"), 400),
                                           "at_step": _num(stop.get("at_step")), "at": _num(stop.get("at"))},
            "created_at": _num(state.get("created_at")), "updated_at": _num(state.get("updated_at")),
            "latest": _num(state.get("latest")), "branch": _num(state.get("branch")),
            "live": None if not live else {"step": _num(live.get("step")), "phase": scrub(live.get("phase"), 20),
                                           "at": _num(live.get("at")), "uncommitted_usd": _num(live.get("uncommitted_usd"))},
            "heartbeat_age_s": heartbeat_age(state, now),
            "best": None if not best else {"checkpoint": _num(best.get("checkpoint")),
                                           "accuracy": _num(best.get("accuracy")), "eval": scrub(best.get("eval"), 40)},
            "kept": [k for k in state.get("kept") or [] if isinstance(k, int)],
            "config": _config(state.get("config") or {}),
            "branches": [{"branch": _num(b.get("branch")), "at": _num(b.get("at")), "reason": scrub(b.get("reason"), 200),
                          "from_checkpoint": _num((b.get("from") or {}).get("checkpoint"))}
                         for b in state.get("branches") or []],
        },
        "lr": _lr(state.get("lr") or {}, state.get("lr_decision")),
        "spend": {**_pick(spend, ("train_usd", "kl_usd", "eval_usd", "interrupted_usd", "total_usd",
                                  "training_gpu_usd_est")),
                  "ceilings": {k: _num(v) for k, v in (spend.get("ceilings") or {}).items() if k in ("train", "eval")}},
        "pot": dict(POT, **(pot or {})),
        "controller": _controller(state.get("controller") or {}),
        "steps": [step_row(r) for r in state.get("steps") or []],
        "evals": [eval_entry(e) for e in state.get("evals") or []],
        "abandoned": abandoned,
        "stop_rules": rules,
        "audits": audits,
        "notices": [{"at": _num(n.get("at")), "step": _num(n.get("step")), "kind": scrub(n.get("kind"), 30),
                     "text": scrub(n.get("text"), 600)} for n in state.get("notices") or []],
        "baselines": list(BASELINES if baselines is None else baselines),
        "samples": samples or [],
    }


# ----------------------------------------------------------------------------- sampled transcripts

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n.*?(?:```|$)", re.S)


def _turn(t: dict) -> dict:
    cell = t.get("cell") or {}
    plan = _CODE_BLOCK_RE.sub("", t.get("reply") or "").strip()
    out = scrub(cell.get("stdout"), CUT["observation"]) if cell else ""
    return {"plan": scrub(plan, CUT["plan"]), "code": scrub(t.get("code"), CUT["code"]) if t.get("code") else "",
            "output": out, "error": scrub(cell.get("error"), 300) if cell.get("error") else "",
            "stdout_chars": _num(cell.get("stdout_chars")), "submitted": bool(cell.get("submitted")),
            "restarted": bool(cell.get("restarted")), "finish": scrub(t.get("finish_reason"), 20),
            "tokens": _num(t.get("completion_tokens"))}


def sample_record(rec: dict, why: str) -> dict:
    """One training rollout for the page: what it was asked, what it did, and how it was scored."""
    turns = rec.get("turns") or []
    head, tail = CUT["turns_head"], CUT["turns_tail"]
    shown = turns if len(turns) <= head + tail else turns[:head] + turns[-tail:]
    first_user = next((m.get("content") for m in rec.get("messages") or [] if m.get("role") == "user"), "") or ""
    context_head = first_user.split("The first 3 lines are:", 1)[1].strip() if "The first 3 lines are:" in first_user else ""
    subcalls = rec.get("subcalls") or []
    edit = rec.get("family") in EDIT_FAMILIES
    language = str(rec.get("language") or "").lower()
    return {
        "key": hashlib.sha1(str(rec.get("key")).encode()).hexdigest()[:12],
        "why": why, "step": _num(rec.get("step")), "family": scrub(rec.get("family"), 40), "level": _num(rec.get("level")),
        "edit": edit, "language": language if edit and language in LANGUAGES else None,
        "question": scrub(rec.get("question"), CUT["question"]), "context_chars": _num(rec.get("context_chars")),
        "context_lines": _num(rec.get("context_lines")), "context_head": scrub(context_head, CUT["context_head"]),
        # an edit task's gold is its patch and hidden tests (spec §12.5): never shown
        "gold": "(hidden tests)" if edit else scrub(json.dumps(rec.get("gold"), ensure_ascii=False)
                                                    if not isinstance(rec.get("gold"), str) else rec.get("gold"),
                                                    CUT["answer"]),
        "answer": scrub(json.dumps(rec.get("answer"), ensure_ascii=False) if not isinstance(rec.get("answer"), str)
                        else rec.get("answer"), CUT["answer"]) if rec.get("submitted") else None,
        "submitted": bool(rec.get("submitted")), "reward": _num(rec.get("reward")), "correct": bool(rec.get("correct")),
        # edit components are limited to §12.9's names, so a per-test result keyed by a hidden test's name can't pass
        "components": {k: _num(v) for k, v in (rec.get("reward_components") or {}).items()
                       if re.fullmatch(r"[a-z][a-z0-9_]{0,29}", k) and (not edit or k in EDIT_KEYS)},
        # rule names only: a detector's detail can quote a test file, a canary or a path (spec §12.6)
        "hacks": [r for r in (h.get("rule") if isinstance(h, dict) else h for h in rec.get("hacks") or [])
                  if isinstance(r, str) and re.fullmatch(r"[a-z_]{1,40}", r)],
        "infra_kind": scrub(rec.get("infra_kind"), 40) if rec.get("infra_error") else None,
        "turn_count": len(turns), "turns_omitted": max(0, len(turns) - len(shown)), "turns": [_turn(t) for t in shown],
        "subcalls_used": _num(rec.get("subcalls_used")), "timeouts": _num(rec.get("timeouts")),
        "events": len(rec.get("events") or []), "rejected": len(rec.get("rejected") or []),
        "subcalls": [{"prompt": scrub(s.get("prompt"), CUT["subcall"]), "output": scrub(s.get("output"), CUT["subcall"]),
                      "error": bool(s.get("error"))} for s in subcalls[:CUT["subcalls_shown"]]],
        "wall_s": _num(rec.get("wall_s")),
    }


def _rank(rec: dict) -> str:
    return hashlib.sha1(f"{rec.get('key')}".encode()).hexdigest()


def pick_samples(records: list[dict]) -> list[tuple[dict, str]]:
    """A step's sample: one win, one failure, and up to two detector-flagged or infra rollouts."""
    ordered = sorted(records, key=_rank)  # stable and unbiased, without a random generator
    flagged = [r for r in ordered if r.get("hacks") or r.get("infra_error")][:SAMPLES_PER_STEP["flagged"]]
    wins = [r for r in ordered if r.get("correct") and r not in flagged][:SAMPLES_PER_STEP["win"]]
    fails = [r for r in ordered if not r.get("correct") and not r.get("infra_error") and r not in flagged]
    fails = fails[:SAMPLES_PER_STEP["fail"]]
    return [(r, "flagged" if r.get("hacks") else "infra") for r in flagged] + [(r, "win") for r in wins] + \
           [(r, "fail") for r in fails]


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:  # a torn last line from a crash
                        pass
    except OSError:
        pass
    return out


def update_samples(cache: dict, state: dict, run_dir: Path) -> list[dict]:
    """Sample each new step's training rollouts once, and keep a bounded set (decision 9c).

    Kept: every sample from the newest steps, the samples at each eval checkpoint (every
    `eval_every` steps, so readers can compare across the run), and detector-flagged rollouts.
    Samples from steps a rollback abandoned are dropped.
    """
    rows = {(r["step"], r.get("branch", 0)) for r in state.get("steps") or []}
    sampled = {tuple(x) for x in cache.get("sampled", [])}
    samples = [s for s in cache.get("samples", []) if (s["step"], s.get("branch", 0)) in rows]
    for step, branch in sorted(rows):
        if (step, branch) in sampled:
            continue
        path = Path(run_dir) / "transcripts" / "train" / f"step{step:04d}.jsonl"
        records = [r for r in _read_jsonl(path) if r.get("split") == "train"]
        for rec, why in pick_samples(records):
            samples.append({**sample_record(rec, why), "branch": branch})
        sampled.add((step, branch))
    every = (state.get("config") or {}).get("eval_every") or EVAL_EVERY_FALLBACK
    latest = max((s for s, _ in rows), default=0)
    recent = {s for s in range(latest - KEEP_RECENT_STEPS + 1, latest + 1)}
    flagged = [s for s in samples if s["why"] in ("flagged", "infra")][-KEEP_FLAGGED:]
    keep = [s for s in samples if s["step"] in recent or s["step"] % every == 0 or s["step"] == 1 or s in flagged]
    cache["samples"], cache["sampled"] = keep, sorted([list(x) for x in sampled if x in rows])
    return sorted(keep, key=lambda s: (-s["step"], s["why"]))


# ----------------------------------------------------------------------------- rendering


def render(data: dict, template: str | None = None) -> str:
    """The page: the template with the data embedded as JSON. `</` is escaped so no string in the
    data can close the script element."""
    template = TEMPLATE.read_text(encoding="utf-8") if template is None else template
    if DATA_MARK not in template:
        raise ValueError("the template has no data mark")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # "<" only occurs inside JSON strings, where \u003c is the same character to JSON.parse
    blob = blob.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    title = TITLES.get((data.get("run") or {}).get("kind"))
    if title:  # production's page is its own artifact (step 12), so its tab and gallery card say which run it is
        template = template.replace("<title>RLM Live Run</title>", f"<title>{title}</title>", 1)
    return template.replace(DATA_MARK, blob)


def embedded_data(page: str) -> Any:
    """The data a built page carries, for the check and the tests."""
    m = re.search(r'<script type="application/json" id="dash-data">(.*?)</script>', page, re.S)
    if not m:
        raise ValueError("no embedded data in the page")
    return json.loads(m.group(1))


# ----------------------------------------------------------------------------- the publish check

KEY_PATTERNS = {
    "long hex string (W&B keys are 40 hex characters)": re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{32,}(?![0-9A-Fa-f])"),
    "key or token name": re.compile(r"WANDB_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|HF_TOKEN|api[_\-]?key|"
                                    r"secret[_\-]?key|access[_\-]?token", re.I),
    "token prefix": re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|sk-ant-|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|"
                               r"github_pat_|wandb_v\d+_[A-Za-z0-9]{8,}|xox[bp]-|AKIA[0-9A-Z]{16})"),
    "bearer header": re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "checkpoint or artifact name": re.compile(r"wandb-artifact://|[\w.\-]+/[\w.\-]+/[\w.\-]+:step\d+"),
    "local file path": re.compile(r"\b[A-Za-z]:\\\\?(?:Users|Personal|RLM)|/mnt/[a-z]/|\.env\b"),
    "sandbox or repository path": _SANDBOX_PATH_RE,
    "task id (owner__repo style)": _TASK_ID_RE,
    "seed-sized integer (13+ digits)": re.compile(r"(?<![\d.\w])\d{13,}(?![\d.])"),
}


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _ints_in(value: Any) -> Iterable[int]:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, str) and value.strip().isdigit():
        yield int(value.strip())
    elif isinstance(value, dict):
        for v in value.values():
            yield from _ints_in(v)
    elif isinstance(value, list):
        for v in value:
            yield from _ints_in(v)


def private_seeds(roots: Iterable[Path]) -> set[int]:
    """Every seed in the private seed files (eval seed lists, H3). Read here, never printed."""
    seeds: set[int] = set()
    for root in roots:
        for path in Path(root).glob("**/*.json"):
            try:
                seeds |= {s for s in _ints_in(json.loads(path.read_text(encoding="utf-8"))) if abs(s) >= 10 ** 6}
            except (OSError, ValueError):
                continue
    return seeds


def item_key(question: str, context_head: str) -> tuple[str, str] | None:
    """What identifies one task: its question and the first lines of its context, as the page would
    show them. Questions alone are templated (two ticket tasks can ask the same thing of different
    logs), so a task is only on the page when both appear."""
    q, h = scrub(question)[:150], scrub(context_head)[:150]
    return (q, h) if len(q) >= 20 and len(h) >= 20 else None


def _context_head(messages: list | None) -> str:
    first_user = next((m.get("content") for m in messages or [] if isinstance(m, dict) and m.get("role") == "user"), "")
    marker = "The first 3 lines are:"
    return first_user.split(marker, 1)[1].strip() if first_user and marker in first_user else ""


def eval_items(run_dir: Path) -> tuple[set[tuple[str, str]], set[str]]:
    """The eval tasks this run has logged (question and context head), and nothing else from them;
    and nothing from the capability check, whose records don't carry prompts (deep_items does).

    An edit task's first message has no context lines, and its task text isn't templated, so it is
    keyed by the task text alone (an empty head is in every page)."""
    items: set[tuple[str, str]] = set()
    unkeyed = 0
    for path in Path(run_dir).glob("transcripts/val/**/*.jsonl"):
        for rec in _read_jsonl(path):
            if is_edit_family(rec.get("family")):
                q = scrub(rec.get("question") or "")[:150]
                key = (q, "") if len(q) >= 20 else None
            else:
                key = item_key(rec.get("question") or "", _context_head(rec.get("messages")))
            if key:
                items.add(key)
            else:
                unkeyed += 1
    return items, ({f"{unkeyed} eval records had too short a question or context to key"} if unkeyed else set())


def eval_names(run_dir: Path) -> tuple[set[str], set[str]]:
    """The task ids and the repositories of the edit eval tasks this run has logged. A selection or
    held-out repository is one training never sees, so its name, like its task ids, must never reach
    the page."""
    ids: set[str] = set()
    repos: set[str] = set()
    for path in Path(run_dir).glob("transcripts/val/**/*.jsonl"):
        for rec in _read_jsonl(path):
            if is_edit_family(rec.get("family")):
                if isinstance(rec.get("task_id"), str):
                    ids.add(rec["task_id"])
                if isinstance(rec.get("repo"), str):
                    repos.add(rec["repo"])
    return ids, repos


def secret_env_values() -> list[str]:
    """Values of key-like variables already in this process's environment (the .env file is never read)."""
    names = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.I)
    return [v for k, v in os.environ.items() if names.search(k) and len(v) >= 16]


def _word_hits(words: list[str], text: str) -> int:
    """How many of `words` appear in `text` as whole words, case-sensitively."""
    return sum(1 for w in words if re.search(r"(?<![\w\-])" + re.escape(w) + r"(?![\w\-])", text))


def check(page: str, *, seeds: set[int] = frozenset(), items: Iterable[tuple[str, str]] = (),
          prompts: Iterable[str] = (), task_ids: Iterable[str] = (), repos: Iterable[str] = ()) -> dict:
    """Scan a built page for anything secret. Returns counts per finding, never the matches.

    The raw HTML and the decoded strings of its embedded data are both scanned, so JSON escaping
    can't hide a match. `task_ids` and `repos` are the edit eval tasks' (eval_names), matched as whole
    words and case-sensitively. Repositories are looked for everywhere but the sampled training
    transcripts: a training rollout can import a library that shares a held-out repository's name.
    """
    items, prompts = list(items), [p for p in prompts if p and len(p) >= 30]
    task_ids = sorted({n for n in task_ids if n and len(n) >= 4})
    repos = sorted({n for n in repos if n and len(n) >= 3})
    data = embedded_data(page)
    texts = [page] + list(_strings(data))
    joined = "\n".join(texts)
    findings: dict[str, int] = {}
    for label, pattern in KEY_PATTERNS.items():
        n = sum(len(pattern.findall(t)) for t in texts)
        if n:
            findings[label] = n
    env = [v for v in secret_env_values() if v in joined]
    if env:
        findings["value of a key variable in the environment"] = len(env)
    if seeds:
        ints = set(_ints_in(data)) | {int(m) for m in re.findall(r"(?<![\d.])\d{7,}(?![\d.])", joined)}
        hit = len(seeds & ints)
        if hit:
            findings["private seed"] = hit
    hit = sum(1 for q, h in items if q in joined and h in joined)
    if hit:
        findings["eval task (its question and context)"] = hit
    hit = sum(1 for p in prompts if scrub(p)[:150] in joined)
    if hit:
        findings["capability or benchmark prompt"] = hit
    hit = _word_hits(task_ids, joined)
    if hit:
        findings["eval task id"] = hit
    outside_samples = "\n".join(_strings({k: v for k, v in (data or {}).items() if k != "samples"}))
    hit = _word_hits(repos, outside_samples)
    if hit:
        findings["eval repository name"] = hit
    return {"ok": not findings, "findings": findings, "bytes": len(page.encode("utf-8")),
            "checked": {"seeds": len(seeds), "eval_tasks": len(items), "prompts": len(prompts),
                        "task_ids": len(task_ids), "repos": len(repos)}}
