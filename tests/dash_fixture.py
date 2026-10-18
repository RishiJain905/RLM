"""A synthetic production run for the dashboard: no model, no sandbox, no W&B, just plausible numbers.

The dry run in tests/dryrun.py drives the real loop, but on the ticket stand-in and without the edit
families. This writes what the loop will write once spec §12 is built: read and edit episodes mixed
per the §12.7 mix, each step's `edit` block (§12.9), evals with `by_language`, the new detectors,
training transcripts (an edit win and a flagged edit rollout among them) and eval transcripts.

    python3 -m tests.dash_fixture OUT_DIR [--steps 30]   # writes OUT_DIR/results/<run>.json and its files
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

RUN = "dash-edit"
MIX = {"aggregation": 0.25, "callgraph": 0.10, "needle": 0.10, "repo_loc": 0.15, "edit_real": 0.25, "edit_synth": 0.15}
READ_MIX = {"aggregation": 0.45, "callgraph": 0.20, "needle": 0.20, "repo_loc": 0.15}
RULES = ("early_submit", "copied_answer", "escape_attempt", "timeout_loop")
EDIT_RULES = ("test_tamper", "grader_tamper", "test_aware", "network_try")
LANGS = {"python": 0.45, "go": 0.15, "javascript": 0.12, "typescript": 0.1, "java": 0.1, "rust": 0.08}
LANG_SKILL = {"python": 1.0, "go": 0.8, "javascript": 0.85, "typescript": 0.75, "java": 0.7, "rust": 0.55}
# what an eval transcript and a private seed file hold, so the build's check has something to find
EVAL_TASK_ID = "selection-v2-edit_real-L2-03"
EVAL_REPO = "quillmark"
EVAL_QUESTION = "Parsing a quoted header with an escaped comma drops everything after the comma; keep the full value."
SEEDS = [8301746629481730551, 1945502286610338411]


def _r(x: float, nd: int = 4) -> float:
    return round(x, nd)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _tasks(step: int, mix: dict) -> dict:
    """8 tasks a step spread across families by the mix, carrying the rounding across steps."""
    out = {}
    for f, w in mix.items():
        out[f] = int(step * 8 * w + 0.5) - int((step - 1) * 8 * w + 0.5)
    while sum(out.values()) > 8:
        out[max(out, key=out.get)] -= 1
    while sum(out.values()) < 8:
        out[min(out, key=lambda f: out[f] - 8 * mix[f])] += 1
    return out


def edit_block(rng: random.Random, t: float, episodes: int) -> dict:
    """One step's spec §12.9 block; t runs from 0 (start) to 1 (end of the run)."""
    resolved = _clip(0.16 + 0.3 * t + rng.gauss(0, 0.05))
    by_language, left = {}, episodes
    for i, (lang, share) in enumerate(LANGS.items()):
        n = left if i == len(LANGS) - 1 else min(left, int(round(episodes * share + rng.uniform(-1, 1))))
        left -= n
        if n > 0:
            by_language[lang] = {"episodes": n, "resolved": _r(round(_clip(resolved * LANG_SKILL[lang] + rng.gauss(0, 0.06)) * n) / n)}
    return {
        "episodes": episodes, "resolved": _r(resolved),
        "f2p_pass": _r(_clip(resolved + 0.17 + rng.gauss(0, 0.04))),
        "p2p_broken": _r(_clip(0.24 - 0.12 * t + rng.gauss(0, 0.03))),
        "empty_diff": _r(_clip(0.14 - 0.1 * t + rng.gauss(0, 0.02))),
        "submitted": _r(_clip(0.84 + 0.13 * t + rng.gauss(0, 0.02))),
        "tested_before_submit": _r(_clip(0.28 + 0.42 * t + rng.gauss(0, 0.04))),
        "test_runs": _r(1.4 + 1.9 * t + rng.gauss(0, 0.2), 2), "edit_cycles": _r(0.8 + 1.2 * t + rng.gauss(0, 0.12), 2),
        "turns_to_first_edit": _r(5.6 - 1.4 * t + rng.gauss(0, 0.3), 2),
        "diff_lines": _r(34 - 9 * t + rng.gauss(0, 3), 1), "files_changed": _r(1.3 + 0.5 * t + rng.gauss(0, 0.1), 2),
        "subcall_read_share": _r(_clip(0.34 + 0.27 * t + rng.gauss(0, 0.03))),
        "review_subcalls": _r(max(0, 0.15 + 0.9 * t + rng.gauss(0, 0.1)), 2),
        "repeated_cells": _r(max(0, 0.7 - 0.4 * t + rng.gauss(0, 0.08)), 2),
        "network_tries": _r(max(0, 0.32 - 0.25 * t + rng.gauss(0, 0.04)), 2),
        "grader_s": _r(52 + 18 * t + rng.gauss(0, 5), 1),
        "edit_errors": _r(max(0, 0.9 - 0.6 * t + rng.gauss(0, 0.08)), 2),
        "raw_io_share": _r(_clip(0.3 - 0.2 * t + rng.gauss(0, 0.03))),
        "grader_retries": _r(max(0, 0.02 + rng.gauss(0, 0.01)), 3),
        "tool_calls": {"read_file": _r(6.5 - 2 * t + rng.gauss(0, 0.4), 2), "find_files": _r(1.6 + rng.gauss(0, 0.2), 2),
                       "grep": _r(3.1 + 1.2 * t + rng.gauss(0, 0.3), 2), "edit_file": _r(2.2 + 0.6 * t + rng.gauss(0, 0.2), 2),
                       "write_file": _r(0.5 + rng.gauss(0, 0.08), 2), "shell": _r(1.9 + 1.8 * t + rng.gauss(0, 0.2), 2),
                       "diff": _r(0.3 + 0.9 * t + rng.gauss(0, 0.1), 2)},
        "by_language": by_language,
    }


def step_row(rng: random.Random, n: int, steps: int, levels: dict, mix: dict, now: float, spent: float,
             edit: bool) -> dict:
    t = (n - 1) / max(1, steps - 1)
    tasks = _tasks(n, mix)
    families, correct, eps = {}, 0, 0
    for f, k in tasks.items():
        base = {"aggregation": 0.55, "callgraph": 0.45, "needle": 0.6, "repo_loc": 0.5,
                "edit_real": 0.18, "edit_synth": 0.32}[f] + 0.25 * t - 0.08 * (levels[f] - 1)
        rate = _clip(base + rng.gauss(0, 0.08)) if k else None
        c = round((rate or 0) * k * 4)
        correct, eps = correct + c, eps + 4 * k
        families[f] = {"level": levels[f], "levels_drawn": [levels[f]] if k else [], "episodes": 4 * k, "scored": 4 * k,
                       "pass_rate": _r(c / (4 * k)) if k else None, "mean_reward": _r(c / (4 * k)) if k else None,
                       "window_pass_rate": _r(_clip((rate or base) + rng.gauss(0, 0.03))), "saturated": False}
    edit_eps = 4 * sum(k for f, k in tasks.items() if f.startswith("edit_"))
    hacks = dict.fromkeys(RULES + (EDIT_RULES if edit else ()), 0)
    if n % 7 == 3:
        hacks["early_submit"] = 1
    if edit and n % 9 == 4:
        hacks["test_tamper"] = 1
    if edit and n % 11 == 6:
        hacks["grader_tamper"] = 1
    if edit:
        hacks["network_try"] = max(0, round((0.3 - 0.25 * t) * edit_eps / 2 + rng.uniform(-1, 1)))
    hacked = sum(v for k, v in hacks.items() if k != "network_try")
    cost = _r((1.85 if edit else 0.9) + 0.4 * t + rng.gauss(0, 0.08), 4)
    rollouts_s, train_s = 520 + 160 * t + rng.gauss(0, 30), 110 + rng.gauss(0, 10)
    started = now - (steps - n + 1) * 900
    row = {
        "step": n, "branch": 0, "status": "trained", "lr": 2e-5, "started_at": started, "finished_at": started + 820,
        "pass": {"rate": _r(correct / eps), "rate_with_infra": _r(correct / eps),
                 "histogram": {"0/4": 2, "1/4": 1, "2/4": 2, "3/4": 2, "4/4": 1}, "groups": 8, "trainable_groups": 5,
                 "all_zero_groups": 2, "all_one_groups": 1, "other_equal_groups": 0, "small_groups": 0},
        "families": families, "moves": [],
        "reward": {"mean": _r(correct / eps + 0.03), "exact": _r(correct / eps), "partial": _r(0.04 + rng.uniform(0, 0.02)),
                   "near_blind": _r(0.02 + rng.uniform(0, 0.01)), "submitted": _r(0.93 + 0.04 * t), "hacked": hacked,
                   "hacks": hacks},
        "policy": {"entropy": _r(0.52 - 0.08 * t + rng.gauss(0, 0.01)), "entropy_tokens": 180000,
                   "reward_std": _r(0.41 + rng.gauss(0, 0.03)), "kl_to_base": _r(0.002 + 0.01 * t, 5),
                   "kl_k3": _r(0.0018 + 0.009 * t, 5), "grad_norm": _r(0.32 + rng.gauss(0, 0.04)),
                   "probs_corr": _r(0.997 + rng.gauss(0, 0.0008), 5), "clip_fraction": _r(0.0006 + rng.uniform(0, 0.0004), 6),
                   "importance_ratio_mean": _r(1.0 + rng.gauss(0, 0.004)), "importance_ratio_p95": _r(1.12 + rng.gauss(0, 0.02)),
                   "importance_ratio_p99": _r(1.31 + rng.gauss(0, 0.05)), "loss": _r(rng.gauss(0, 0.02)),
                   "kl": {"tokens": 30000, "failed": 0, "misaligned_turns": 0}},
        "shape": {"subcalls": _r(7 + 3 * t, 2), "budget_hit": _r(0.02, 3), "turns": _r((8.5 if edit else 5.6) + 1.2 * t, 2),
                  "first_turn_submit": 0.0, "no_code_turns": _r(0.08, 2), "repl_errors": _r(0.9 - 0.3 * t, 2),
                  "timeouts": _r(0.06, 2), "root_completion_tokens": round(5200 + 2400 * t), "subcall_tokens": round(2600 + 900 * t),
                  "truncated_turns": 0.04, "episodes_with_truncated_turn": _r(0.03 + rng.uniform(0, 0.02), 3),
                  "output_truncations": _r(0.5, 2), "context_chars": round(52000 + 30000 * t),
                  "root_transcript_tokens": {"mean": round(13000 + 6000 * t), "p95": round(31000 + 6000 * t), "max": round(42000 + 8000 * t)},
                  "wall_s": _r(210 + 40 * t, 1)},
        "infra": {"episodes": 32, "infra": 0, "rate": 0.0, "kinds": {},
                  "requests": {"attempts": 900, "retries": 2, "rate_limited": 1, "server_errors": 0, "timeouts": 0,
                               "connection_errors": 0, "rejected": 0, "peaks": {"total": 26, "waited_s": 3.1}}},
        "filter": {"skip_train": None}, "infra_pause": None,
        "cost": {"step_usd": cost, "kl_usd": 0.011, "spent_train_usd": _r(spent + cost)},
        "time": {"rollouts_s": _r(rollouts_s, 2), "kl_s": 21.0, "train_s": _r(train_s, 2), "total_s": _r(rollouts_s + 21 + train_s, 2)},
        "tokens": {"root_prompt_tokens": round(2.1e6 + 1.2e6 * t), "root_completion_tokens": round(170000 + 70000 * t),
                   "subcall_prompt_tokens": round(1.4e6 + 0.5e6 * t), "subcall_completion_tokens": 90000,
                   "root_calls": 260, "subcalls": 240},
    }
    if edit:
        row["edit"] = edit_block(rng, t, edit_eps)
    return row, cost


def eval_entry(rng: random.Random, c: int, steps: int, now: float, edit: bool) -> dict:
    t = c / max(1, steps)
    acc = _clip(0.38 + 0.12 * t + rng.gauss(0, 0.01))
    by_family = {"aggregation": {"n": 12, "accuracy": _r(_clip(0.5 + 0.12 * t))},
                 "callgraph": {"n": 6, "accuracy": _r(_clip(0.33 + 0.17 * t))},
                 "needle": {"n": 6, "accuracy": _r(_clip(0.5 + 0.17 * t))},
                 "repo_loc": {"n": 9, "accuracy": _r(_clip(0.44 + 0.11 * t))},
                 "tickets_all": {"n": 15, "accuracy": _r(_clip(0.47 + 0.13 * t))}}
    by_language = {}
    if edit:
        by_family.update({"edit_real": {"n": 10, "accuracy": _r(_clip(0.1 + 0.2 * t))},
                          "edit_synth_heldout": {"n": 6, "accuracy": _r(_clip(0.17 + 0.33 * t))}})
        by_language = {"python": {"n": 10, "accuracy": _r(_clip(0.2 + 0.2 * t))}, "go": {"n": 2, "accuracy": _r(0.5 * t)},
                       "typescript": {"n": 2, "accuracy": 0.0 if t < 0.6 else 0.5}, "rust": {"n": 2, "accuracy": 0.0}}
    summary = {"episodes": 64, "n": 64, "accuracy": _r(acc), "accuracy_se": _r((acc * (1 - acc) / 64) ** 0.5),
               "mean_reward": _r(acc + 0.02), "submitted": 0.95, "first_turn_submit": 0.0, "turns": 7.4, "subcalls": 8.1,
               "budget_hit": 0.02, "hacked": 0, "hacks": dict.fromkeys(RULES + (EDIT_RULES if edit else ()), 0),
               "infra_episodes": 0, "by_family": by_family,
               "by_level": {f"{f}/L1": v for f, v in by_family.items()}, "by_language": by_language}
    return {"checkpoint": c, "name": "selection-v2", "summary": summary, "ran": 64, "branch": 0,
            "model": f"ent/proj/{RUN}:step{c}", "at": now - (steps - c) * 900 + 300, "cost_usd": 3.4}


def state(steps: int = 30, seed: int = 0, edit: bool = True, now: float | None = None) -> dict:
    rng, now = random.Random(seed), now or time.time()
    mix = MIX if edit else READ_MIX
    levels = dict.fromkeys(mix, 1)
    rows, evals, notices, history, spent = [], [], [], [], 0.0
    notices.append({"at": now - steps * 900 - 60, "step": 0, "kind": "start", "text": "production started at lr 2e-05"})
    evals.append(eval_entry(rng, 0, steps, now, edit))  # the base model, before step 1
    for n in range(1, steps + 1):
        row, cost = step_row(rng, n, steps, levels, mix, now, spent, edit)
        spent += cost
        rows.append(row)
        for f in ("aggregation", "needle") if n in (9, 21) else ():
            history.append({"step": n, "family": f, "from_level": levels[f], "to_level": levels[f] + 1, "pass_rate": 0.74,
                            "applied": True})
            row["moves"].append({"family": f, "from": levels[f], "to": levels[f] + 1, "pass_rate": 0.74, "applied": True})
            notices.append({"at": row["finished_at"], "step": n, "kind": "level", "text": f"{f} moved up to level {levels[f] + 1}"})
            levels[f] += 1
        if n % 10 == 0:
            e = eval_entry(rng, n, steps, now, edit)
            evals.append(e)
            notices.append({"at": e["at"], "step": n, "kind": "eval",
                            "text": f"selection-v2 at checkpoint {n}: accuracy {e['summary']['accuracy']:.1%} over 64 ($3.400)"})
    best = max(evals, key=lambda e: e["summary"]["accuracy"])
    eval_usd = sum(e["cost_usd"] for e in evals)
    return {
        "format": 1, "run": RUN, "kind": "production", "status": "running", "stop": None,
        "created_at": now - steps * 900 - 60, "updated_at": now, "latest": steps, "branch": 0,
        "config": {"kind": "production", "learning_rate": 2e-5, "lr_rule": "fallback", "tasks_per_step": 8, "rollouts": 4,
                   "max_steps": 200, "eval_every": 10, "eval_at_start": True, "selection": "selection-v2",
                   "final_eval": None, "keep_every": 5, "kl_samples": 4, "base_model": "Qwen/Qwen3.8-27B",
                   "frozen_controller": False, "mix": mix, "ceilings": {"train": 146.0, "eval": 34.0},
                   "capability_set": "capability-v1"},
        "lr": {"current": 2e-5, "start": 2e-5, "floor": 1e-5, "rule": "fallback", "fallback_used": False, "fallback": None},
        "controller": {"frozen": False, "families": {f: {"level": levels[f], "max_level": 5, "moved_at": None,
                                                         "saturated": False} for f in mix},
                       "history": history},
        "steps": rows, "evals": evals, "best": {"checkpoint": best["checkpoint"], "accuracy": best["summary"]["accuracy"],
                                                "eval": "selection-v2"},
        "kept": list(range(0, steps + 1, 5)),
        "spend": {"train_usd": _r(spent), "kl_usd": _r(0.011 * steps), "eval_usd": _r(eval_usd), "interrupted_usd": 0.0,
                  "total_usd": _r(spent + eval_usd), "training_gpu_usd_est": _r(0.16 * steps),
                  "ceilings": {"train": 146.0, "eval": 34.0}},
        "stop_rules": {"plateau": {"state": "ok", "text": "selection accuracy still rising over the last 3 evals", "action": "stop"},
                       "overfitting": {"state": "ok", "text": "train and selection move together", "action": "stop"},
                       "frying": {"state": "ok", "text": "capability check within 2 points of checkpoint 0", "action": "stop"},
                       "saturation": {"state": "n/a", "text": "no family at its top level yet", "action": "stop"},
                       "hacking": {"state": "watch", "text": "2 detector hits in the last 10 steps", "action": "pause"}},
        "notices": notices, "abandoned": [],
        "live": {"step": steps + 1, "phase": "rollouts", "at": now - 120}, "audits": {},
    }


def _turn(reply: str, code: str, stdout: str, submitted: bool = False) -> dict:
    return {"reply": reply + "\n```python\n" + code + "\n```", "finish_reason": "stop", "completion_tokens": 420, "code": code,
            "cell": {"stdout": stdout, "stdout_chars": len(stdout), "error": None, "submitted": submitted, "restarted": False},
            "observation": None, "root_s": 4.0, "cell_s": 1.0}


def _edit_record(step: int, why: str) -> dict:
    turns = [_turn("Survey the repository and find where headers are parsed.",
                   "import os\nfiles=[os.path.join(d,f) for d,_,fs in os.walk('/work/repo') for f in fs if f.endswith('.go')]\n"
                   "print(len(files)); print([f for f in files if 'header' in f])",
                   "118\n['/work/repo/internal/header/parse.go', '/work/repo/internal/header/parse_test.go']"),
             _turn("Ask sub-calls which functions split on commas.",
                   "ans=llm_batch([f'Which function splits values on commas?\\n'+open(p).read() for p in files[:12]])\nprint(ans[:2])",
                   "['splitValues in parse.go ignores quotes', 'none']"),
             _turn("Patch splitValues to respect quotes, then run the tests.",
                   "import subprocess\nprint(subprocess.run(['go','test','./...'],cwd='/work/repo',capture_output=True,text=True).stdout[-300:])",
                   "ok  \tinternal/header\t0.41s"),
             _turn("Tests pass; submit.", "submit('splitValues keeps quoted commas')", "", True)]
    flagged = why == "flagged"
    return {"format": 2, "key": f"{RUN}/step{step}/edit_real-{step}/1", "split": "train", "step": step, "task_id": f"acme__headers-{100 + step}",
            "family": "edit_real", "level": 2, "language": "go",
            "question": "Task: header values with a quoted comma are split in two. Repository acme__headers-1234 at /work/repo.",
            "gold": {"patch": "diff --git a/internal/header/parse.go", "f2p": ["internal/header/parse_test.go::TestQuotedComma"]},
            "context_chars": 4200, "context_lines": 130, "messages": [{"role": "user", "content": "Task: header values ..."}],
            "turns": turns, "subcalls": [{"prompt": "Which function splits values on commas?", "output": "splitValues", "error": False}],
            "subcalls_used": 12, "submitted": True, "answer": "splitValues keeps quoted commas", "reward": 0.0 if flagged else 1.0,
            "correct": not flagged, "reward_components": {"resolved": 0.0 if flagged else 1.0, "f2p_pass": 1.0,
                                                          "TestQuotedComma": 1.0, "test_hidden_case": 1.0},
            "hacks": [{"rule": "test_tamper", "detail": "changed internal/header/parse_test.go::TestQuotedComma"}] if flagged else [],
            "infra_error": None, "events": [], "rejected": [], "timeouts": 0, "wall_s": 412.0}


def _read_record(step: int, correct: bool) -> dict:
    content = ("Question: How many tickets mention a refund?\n\n`context` holds 41,220 characters in 812 lines. The first 3 lines are:\n"
               "ticket 1 | billing | refund requested\nticket 2 | login | cannot sign in\nticket 3 | billing | charged twice")
    return {"format": 2, "key": f"{RUN}/step{step}/aggregation-{step}/{int(correct)}", "split": "train", "step": step,
            "task_id": f"aggregation-L1-{step}", "family": "aggregation", "level": 1, "question": "How many tickets mention a refund?",
            "gold": 37, "context_chars": 41220, "context_lines": 812, "messages": [{"role": "user", "content": content}],
            "turns": [_turn("Count lines with refund.", "print(sum('refund' in l for l in context.splitlines()))", "37"),
                      _turn("Submit.", "submit(37)" if correct else "submit(35)", "", True)],
            "subcalls": [], "subcalls_used": 0, "submitted": True, "answer": 37 if correct else 35,
            "reward": 1.0 if correct else 0.0, "correct": correct, "reward_components": {"exact": 1.0 if correct else 0.0},
            "hacks": [], "infra_error": None, "events": [], "rejected": [], "timeouts": 0, "wall_s": 61.0}


def write_run(results: Path, run: str = RUN, steps: int = 30, seed: int = 0, edit: bool = True) -> dict:
    """Write results/<run>.json, its training and eval transcripts, and a private seed file."""
    results = Path(results)
    st = state(steps, seed, edit)
    st["run"] = run
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{run}.json").write_text(json.dumps(st), encoding="utf-8")
    train = results / run / "transcripts" / "train"
    train.mkdir(parents=True, exist_ok=True)
    for n in range(1, steps + 1):
        recs = [_read_record(n, True), _read_record(n, False)]
        if edit:
            recs += [_edit_record(n, "win")] + ([_edit_record(n, "flagged")] if n % 9 == 4 else [])
        (train / f"step{n:04d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    val = results / run / "transcripts" / "val" / "selection-v2"
    val.mkdir(parents=True, exist_ok=True)
    recs = [{**_read_record(0, True), "split": "val", "key": f"selection-v2/step0/agg-00/0",
             "question": "How many tickets in the held-out log mention a chargeback dispute?"}]
    if edit:
        recs.append({**_edit_record(0, "win"), "split": "val", "key": "selection-v2/step0/edit-03/0", "task_id": EVAL_TASK_ID,
                     "repo": EVAL_REPO, "question": EVAL_QUESTION})
    (val / "step0000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    seeds = results / "private" / "seeds"
    seeds.mkdir(parents=True, exist_ok=True)
    (seeds / "selection-v2.json").write_text(json.dumps(SEEDS), encoding="utf-8")
    return st


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--run", default=RUN)
    args = parser.parse_args()
    write_run(Path(args.out) / "results", args.run, args.steps)
    print(f"wrote {Path(args.out) / 'results' / (args.run + '.json')}")


if __name__ == "__main__":
    main()
