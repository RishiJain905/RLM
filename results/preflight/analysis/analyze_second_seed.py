"""Analysis of the second-seed-v1 run: per-step metrics and transcripts.

Prints aggregate numbers only; never prints task text, questions or gold answers.
"""
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"F:\Personal\RLM\RLM\results")
RUN = json.loads((ROOT / "second-seed-v1.json").read_text(encoding="utf-8"))
ARCH = json.loads((ROOT / "archive" / "rlsd_vs_grpo_20.json").read_text(encoding="utf-8"))
TR = ROOT / "second-seed-v1" / "transcripts"

steps = RUN["steps"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


print("=== per-step table ===")
hdr = ("st pass  all   cust  trn  ent    kl     gn    pc     clip      rstd  sub  turn bud  rtt_mean rtt_p95 "
       "rootcomp  cost   roll_s train_s tot_s  429 retr att  peak waited_s")
print(hdr)
for s in steps:
    p, f, pol, sh, inf, tok, t = s["pass"], s["families"], s["policy"], s["shape"], s["infra"], s["tokens"], s["time"]
    rq = inf["requests"]
    print(f"{s['step']:2d} {p['rate']:.3f} {f['tickets_all']['pass_rate']:.3f} {f['tickets_customer']['pass_rate']:.3f} "
          f"{p['trainable_groups']:3d}  {pol['entropy']:.3f} {pol['kl_to_base']:.4f} {pol['grad_norm']:.3f} "
          f"{pol['probs_corr']:.4f} {pol['clip_fraction']:.2e} {pol['reward_std']:.3f} {sh['subcalls']:.2f} {sh['turns']:.2f} "
          f"{sh['budget_hit']:.2f} {sh['root_transcript_tokens']['mean']:8.0f} {sh['root_transcript_tokens']['p95']:7.0f} "
          f"{sh['root_completion_tokens']:7.0f} {s['cost']['step_usd']:.3f} {t['rollouts_s']:6.0f} {t['train_s']:6.0f} "
          f"{t['total_s']:6.0f} {rq['rate_limited']:4d} {rq['retries']:4d} {rq['attempts']:4d} {rq['peaks']['total']:3d} "
          f"{rq['peaks']['waited_s']:8.0f}")


def avg(rng, fn):
    return mean([fn(s) for s in steps if s["step"] in rng])


A, B = range(1, 6), range(16, 21)
print("\n=== steps 1-5 vs 16-20 ===")
for name, fn in [
    ("pass", lambda s: s["pass"]["rate"]),
    ("pass_all", lambda s: s["families"]["tickets_all"]["pass_rate"]),
    ("pass_cust", lambda s: s["families"]["tickets_customer"]["pass_rate"]),
    ("reward_mean", lambda s: s["reward"]["mean"]),
    ("submitted", lambda s: s["reward"]["submitted"]),
    ("entropy", lambda s: s["policy"]["entropy"]),
    ("kl", lambda s: s["policy"]["kl_to_base"]),
    ("kl_k3", lambda s: s["policy"]["kl_k3"]),
    ("grad_norm", lambda s: s["policy"]["grad_norm"]),
    ("probs_corr", lambda s: s["policy"]["probs_corr"]),
    ("clip", lambda s: s["policy"]["clip_fraction"]),
    ("ir_p99", lambda s: s["policy"]["importance_ratio_p99"]),
    ("reward_std", lambda s: s["policy"]["reward_std"]),
    ("trainable", lambda s: s["pass"]["trainable_groups"]),
    ("all_one", lambda s: s["pass"]["all_one_groups"]),
    ("all_zero", lambda s: s["pass"]["all_zero_groups"]),
    ("subcalls", lambda s: s["shape"]["subcalls"]),
    ("turns", lambda s: s["shape"]["turns"]),
    ("first_turn_submit", lambda s: s["shape"]["first_turn_submit"]),
    ("budget_hit", lambda s: s["shape"]["budget_hit"]),
    ("repl_errors", lambda s: s["shape"]["repl_errors"]),
    ("no_code_turns", lambda s: s["shape"]["no_code_turns"]),
    ("output_trunc", lambda s: s["shape"]["output_truncations"]),
    ("rtt_mean", lambda s: s["shape"]["root_transcript_tokens"]["mean"]),
    ("rtt_max", lambda s: s["shape"]["root_transcript_tokens"]["max"]),
    ("root_comp_tok", lambda s: s["shape"]["root_completion_tokens"]),
    ("subcall_tok", lambda s: s["shape"]["subcall_tokens"]),
    ("wall_s", lambda s: s["shape"]["wall_s"]),
    ("cost", lambda s: s["cost"]["step_usd"]),
    ("rollouts_s", lambda s: s["time"]["rollouts_s"]),
    ("train_s", lambda s: s["time"]["train_s"]),
    ("total_s", lambda s: s["time"]["total_s"]),
    ("root_prompt_tok", lambda s: s["tokens"]["root_prompt_tokens"]),
    ("rate_limited", lambda s: s["infra"]["requests"]["rate_limited"]),
    ("attempts", lambda s: s["infra"]["requests"]["attempts"]),
]:
    print(f"{name:18s} {avg(A, fn):10.4f} {avg(B, fn):10.4f}")

print("\n=== totals ===")
tot = defaultdict(float)
for s in steps:
    for k, v in s["tokens"].items():
        tot[k] += v
    tot["rollouts_s"] += s["time"]["rollouts_s"]
    tot["train_s"] += s["time"]["train_s"]
    tot["kl_s"] += s["time"]["kl_s"]
    tot["total_s"] += s["time"]["total_s"]
    tot["rate_limited"] += s["infra"]["requests"]["rate_limited"]
    tot["retries"] += s["infra"]["requests"]["retries"]
    tot["attempts"] += s["infra"]["requests"]["attempts"]
    tot["infra_eps"] += s["infra"]["infra"]
    tot["step_usd"] += s["cost"]["step_usd"]
    tot["server_errors"] += s["infra"]["requests"]["server_errors"]
    tot["timeouts"] += s["infra"]["requests"]["timeouts"]
for k, v in tot.items():
    print(f"{k:28s} {v:,.2f}")
print("pass rate all steps:", mean([s["pass"]["rate"] for s in steps]))
print("pass all fam all steps:", mean([s["families"]["tickets_all"]["pass_rate"] for s in steps]),
      mean([s["families"]["tickets_customer"]["pass_rate"] for s in steps]))

# cost split estimate: root prompt vs completion vs subcall, using the per-step est_cost
print("\n=== hacks / notices / infra kinds per step ===")
for s in steps:
    h = {k: v for k, v in s["reward"]["hacks"].items() if v}
    ik = s["infra"]["kinds"]
    fail = s["infra"]["requests"]["failed"]
    if h or ik or fail or s["infra"]["requests"]["rate_limited"]:
        print(s["step"], "hacks", h, "infra_kinds", ik, "failed", fail, "429", s["infra"]["requests"]["rate_limited"],
              "filter hacked", s["filter"]["hacked_episodes"])
for n in RUN["notices"]:
    if n["kind"] not in ("level",):
        print("NOTICE", n["step"], n["kind"], n["text"][:200])

print("\n=== first seed (archive grpo) ===")
g = ARCH["steps"]["grpo"]
for r in g:
    print(r["step"], r["train_correct"], round(r["turns"], 2), round(r["subcalls"], 2), r.get("used_subcalls"),
          r["submitted"], r["seconds"], r["mixed_groups"], round(r["loss"]["loss/grad_norm"], 3),
          round(r["loss"]["loss/probs_corr"], 4), round(r["loss"]["loss/clipped_token_fraction"], 3))
print("seed1 train_correct 0-4", mean([r["train_correct"] for r in g[:5]]), "15-19",
      mean([r["train_correct"] for r in g[15:]]))
print("seed1 turns", mean([r["turns"] for r in g]), "subcalls", mean([r["subcalls"] for r in g]),
      "first5 sub", mean([r["subcalls"] for r in g[:5]]), "last5 sub", mean([r["subcalls"] for r in g[15:]]))
print("seed1 grad_norm median", st.median([r["loss"]["loss/grad_norm"] for r in g]), "max",
      max(r["loss"]["loss/grad_norm"] for r in g))
print("seed1 probs_corr min", min(r["loss"]["loss/probs_corr"] for r in g))
print("seed1 seconds mean", mean([r["seconds"] for r in g]))
print("seed1 eval", ARCH["eval"])
