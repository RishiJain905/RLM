"""Extra checks on the original 20-step branch of second-seed-v1.

Steps 1-15 live in the run state's `steps`; steps 16-20 moved to `abandoned[0]` when the operator
resumed from checkpoint 15 (plan step 10d). Transcripts for 16-20 and the held-out eval moved to
second-seed-v1/abandoned/b0-after15/. Aggregates only.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

R = Path(r"F:\Personal\RLM\RLM\results")
RUN = json.loads((R / "second-seed-v1.json").read_text(encoding="utf-8"))
ab = [a for a in RUN["abandoned"] if a.get("after") == 15][0]
steps = [s for s in RUN["steps"] if s["step"] <= 15 and s["branch"] == 0] + ab["steps"]
steps.sort(key=lambda s: s["step"])
assert [s["step"] for s in steps] == list(range(1, 21)), [s["step"] for s in steps]


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


TRAIN = {}
for n in range(1, 21):
    base = R / "second-seed-v1" / ("transcripts" if n <= 15 else "abandoned/b0-after15/transcripts") / "train"
    TRAIN[n] = load(base / f"step{n:04d}.jsonl")
VAL = load(R / "second-seed-v1/abandoned/b0-after15/transcripts/val/heldout-seed1/step0020.jsonl")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def corr(x, y):
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


print("=== spec 5.2 fallback rules replayed over 20 steps ===")
rs = [s["policy"]["reward_std"] for s in steps]
ent = [s["policy"]["entropy"] for s in steps]
below = []
for i in range(20):
    w = rs[max(0, i - 10):i]
    if w:
        below.append(rs[i] < mean(w) / 2)
    else:
        below.append(False)
print("reward_std below half its trailing 10-step mean:", [s["step"] for s, b in zip(steps, below) if b])
print("entropy vs trailing-10 peak (min ratio):", min(ent[i] / max(ent[max(0, i - 9):i + 1]) for i in range(20)))
print("entropy fall within 10 steps (max drop from any earlier step within 10):",
      max((ent[j] - ent[i]) / ent[j] for i in range(20) for j in range(max(0, i - 10), i)) if True else None)
print("entropy step1 (base)", ent[0], "steps 2-5", mean(ent[1:5]), "steps 6-15", mean(ent[5:15]), "16-20", mean(ent[15:]))
kl = [s["policy"]["kl_to_base"] for s in steps]
xs = list(range(1, 21))
print("KL slope per step (OLS):", sum((a - mean(xs)) * (b - mean(kl)) for a, b in zip(xs, kl)) / sum((a - mean(xs)) ** 2 for a in xs))
print("entropy corr with step", corr(xs, ent), "KL corr with step", corr(xs, kl))
gn = [s["policy"]["grad_norm"] for s in steps]
print("grad_norm median", sorted(gn)[9:11], "max", max(gn))

print("\n=== 429s vs load ===")
r429 = [s["infra"]["requests"]["rate_limited"] for s in steps]
att = [s["infra"]["requests"]["attempts"] for s in steps]
sub = [s["infra"]["requests"]["calls"]["subcall"] for s in steps]
root = [s["infra"]["requests"]["calls"]["root"] for s in steps]
rate = [a / b for a, b in zip(r429, att)]
print("429 share of attempts per step:", [round(x, 2) for x in rate])
print("steps 2-20 429 share: mean", mean(rate[1:]), "overall", sum(r429) / sum(att), "step1", rate[0])
print("corr(429, subcall calls) steps 2-20:", corr(r429[1:], sub[1:]), " corr(429, root calls):", corr(r429[1:], root[1:]))
# requests: first-attempt (granted minus retries) -> retry ratio
print("retries per original request (2-20):", sum(s["infra"]["requests"]["retries"] for s in steps[1:]) /
      sum(sum(s["infra"]["requests"]["calls"].values()) for s in steps[1:]))

print("\n=== request concurrency demand (Little's law from transcripts) ===")
print("step  rollouts_s  root_req_s/rollout  sub_req_s/rollout  root_calls  sub_calls  root_lat_mean sub_lat_mean")
for s in steps:
    eps = TRAIN[s["step"]]
    root_s = sum(t.get("root_s") or 0 for e in eps for t in e["turns"])
    sub_s = sum(x.get("seconds") or 0 for e in eps for x in e["subcalls"])
    nr = sum(len(e["turns"]) for e in eps)
    ns = sum(len(e["subcalls"]) for e in eps)
    ro = s["time"]["rollouts_s"]
    print(f"{s['step']:3d} {ro:8.0f} {root_s / ro:10.1f} {sub_s / ro:10.1f} {nr:6d} {ns:6d} {root_s / nr:8.1f} {sub_s / max(ns, 1):8.1f}")
# how long until the first 20 episodes finish vs last: tail share of rollout time
print("\n=== tail: rollout time vs median episode wall ===")
for s in steps:
    w = sorted(e["wall_s"] for e in TRAIN[s["step"]])
    print(s["step"], "rollouts_s", round(s["time"]["rollouts_s"]), "median ep", round(w[16]), "p75", round(w[24]),
          "max", round(w[-1]), "2nd", round(w[-2]), "tail share (max-p75)/max", round((w[-1] - w[24]) / w[-1], 2))

print("\n=== turn cap (12) and no-submit ===")
for label, rng in (("1-5", range(1, 6)), ("16-20", range(16, 21)), ("all", range(1, 21))):
    for fam in ("tickets_all", "tickets_customer"):
        eps = [e for n in rng for e in TRAIN[n] if e["family"] == fam and not e.get("infra_kind")]
        cap = [e for e in eps if len(e["turns"]) >= 12]
        print(f" {label:5s} {fam:17s} at 12 turns {len(cap)}/{len(eps)} = {len(cap) / len(eps):.2f}; "
              f"acc at cap {mean(1.0 if e['correct'] else 0 for e in cap):.2f}; "
              f"no-submit {sum(1 for e in eps if not e['submitted'])} (of which at cap {sum(1 for e in cap if not e['submitted'])}); "
              f"output truncations/ep {mean(sum(1 for t in e['turns'] if (t.get('cell') or {}).get('stdout_chars', 0) > 2000) for e in eps):.2f}; "
              f"repl errors/ep {mean(sum(1 for t in e['turns'] if (t.get('cell') or {}).get('error')) for e in eps):.2f}")
for fam in ("tickets_all", "tickets_customer"):
    eps = [e for e in VAL if e["family"] == fam and not e.get("infra_kind")]
    cap = [e for e in eps if len(e["turns"]) >= 12]
    print(f" VAL   {fam:17s} at 12 turns {len(cap)}/{len(eps)}; acc at cap {mean(1.0 if e['correct'] else 0 for e in cap):.2f}; "
          f"no-submit {sum(1 for e in eps if not e['submitted'])}")

print("\n=== per-family training trend by step (scored) ===")
for n in range(1, 21):
    row = []
    for fam in ("tickets_all", "tickets_customer"):
        eps = [e for e in TRAIN[n] if e["family"] == fam and not e.get("infra_kind")]
        row.append(f"{sum(1 for e in eps if e['correct'])}/{len(eps)}")
    print(n, row)
for fam in ("tickets_all", "tickets_customer"):
    a = [e for n in range(1, 6) for e in TRAIN[n] if e["family"] == fam and not e.get("infra_kind")]
    b = [e for n in range(16, 21) for e in TRAIN[n] if e["family"] == fam and not e.get("infra_kind")]
    pa, pb = mean(1.0 if e["correct"] else 0 for e in a), mean(1.0 if e["correct"] else 0 for e in b)
    se = math.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))
    print(f"{fam}: {pa:.3f} -> {pb:.3f}, diff {pb - pa:+.3f} (SE {se:.3f}, z {(pb - pa) / se:.2f})")
a = [e for n in range(1, 6) for e in TRAIN[n] if not e.get("infra_kind")]
b = [e for n in range(16, 21) for e in TRAIN[n] if not e.get("infra_kind")]
pa, pb = mean(1.0 if e["correct"] else 0 for e in a), mean(1.0 if e["correct"] else 0 for e in b)
se = math.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))
print(f"overall: {pa:.3f} -> {pb:.3f}, z {(pb - pa) / se:.2f}")
# held-out vs first seed
acc, n = 80 / 96, 96
se_new = math.sqrt(acc * (1 - acc) / n)
se_old = math.sqrt(0.7 * 0.3 / 100)
print(f"held-out 0.833 (SE {se_new:.3f}) vs first seed 0.70 (SE {se_old:.3f}): gap {acc - 0.7:+.3f} = "
      f"{(acc - 0.7) / se_new:.1f} of our SE, {(acc - 0.7) / math.hypot(se_new, se_old):.1f} SE of the difference")
# worst case: count 4 infra as wrong
print("held-out if 4 infra counted wrong:", 80 / 100)
# held-out task difficulty proxies vs train: context chars, and step-20 checkpoint training pass on same families
for fam in ("tickets_all", "tickets_customer"):
    v = [e for e in VAL if e["family"] == fam]
    t = [e for n in range(16, 21) for e in TRAIN[n] if e["family"] == fam]
    print(fam, "val ctx", round(mean(e["context_chars"] for e in v)), "lines", round(mean(e["context_lines"] for e in v)),
          "| train16-20 ctx", round(mean(e["context_chars"] for e in t)), "lines", round(mean(e["context_lines"] for e in t)),
          "| val gold mean", round(mean(float(e["gold"]) for e in v), 1), "train gold mean", round(mean(float(e["gold"]) for e in t), 1))

print("\n=== train backend time vs GPU time, 16-20 ===")
for s in steps:
    tm = s.get("train_metrics") or {}
    print(s["step"], "train_s", s["time"]["train_s"], "backend", round(tm.get("time/step_backend_train_s", 0)),
          "gpu step", round(tm.get("time/step_train_s", 0)), "grad steps", tm.get("data/step_num_gradient_steps"),
          "trained eps", s["filter"]["trained_episodes"])
print("total wall h:", (steps[-1]["finished_at"] - steps[0]["started_at"]) / 3600,
      "eval end - step20 end (min):", (RUN["abandoned"][0]["evals"][0]["at"] - steps[-1]["finished_at"]) / 60)
print("sum rollouts_s", sum(s["time"]["rollouts_s"] for s in steps), "sum train_s", sum(s["time"]["train_s"] for s in steps),
      "sum total_s", sum(s["time"]["total_s"] for s in steps))
print("cost per step min/max", min(s["cost"]["step_usd"] for s in steps), max(s["cost"]["step_usd"] for s in steps))
print("est cost per eval episode", 1.96581 / 100)
print("root prompt share of tokens", sum(s["tokens"]["root_prompt_tokens"] for s in steps),
      "cached", sum(s["tokens"]["root_cached_tokens"] for s in steps), sum(s["tokens"]["subcall_cached_tokens"] for s in steps))
# if prefix caching at 0.15 applied to all but the newest turn's tokens: rough upper-bound saving
P_IN, P_C, P_OUT = 0.40, 0.15, 3.00
save = 0.0
tot = 0.0
for n in TRAIN:
    for e in TRAIN[n]:
        u = e["usage"]
        tot += (u["root_prompt_tokens"] + u["subcall_prompt_tokens"]) * P_IN + (u["root_completion_tokens"] + u["subcall_completion_tokens"]) * P_OUT
        # cacheable: each root call's prompt minus the newest observation and reply; approximate as prompt of previous call
        k = len(e["turns"])
        if k > 1:
            # prompt tokens grow ~linearly with turn index; cached prefix ≈ prompt_total * (k-1)/(k+1)
            cache = u["root_prompt_tokens"] * (k - 1) / (k + 1)
            save += cache * (P_IN - P_C)
print(f"inferred prefix-caching saving if W&B cached root prefixes: ~{save / tot:.0%} of train inference cost")
