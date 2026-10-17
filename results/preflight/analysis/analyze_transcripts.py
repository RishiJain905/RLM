"""Episode-level analysis of second-seed-v1 transcripts. Aggregates only: no task text, questions or gold."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

TR = Path(r"F:\Personal\RLM\RLM\results\second-seed-v1\transcripts")


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def outcome(e):
    if e.get("infra_error") or e.get("infra_kind"):
        return "infra:" + str(e.get("infra_kind"))
    if e.get("hacks"):
        return "hack"
    if e.get("correct"):
        return "correct"
    if not e.get("submitted"):
        return "no_submit"
    if not e.get("answer_valid", True):
        return "invalid_answer"
    return "wrong_answer"


def err_bucket(e):
    """Relative error bucket for wrong answers, without revealing values."""
    try:
        a, g = float(e["answer"]), float(e["gold"])
    except (TypeError, ValueError, KeyError):
        return "n/a"
    d = a - g
    if g == 0:
        return "gold0"
    r = abs(d) / abs(g)
    sign = "over" if d > 0 else "under"
    if abs(d) <= 1:
        return f"off_by_1_{sign}"
    if r <= 0.25:
        return f"within25%_{sign}"
    return f">25%_{sign}"


train = {}
for f in sorted((TR / "train").glob("step*.jsonl")):
    train[int(f.stem[4:])] = load(f)
val = load(TR / "val" / "heldout-seed1" / "step0020.jsonl")


def summarize(eps, label):
    sc = [e for e in eps if not (e.get("infra_error") or e.get("infra_kind"))]
    oc = Counter(outcome(e) for e in eps)
    print(f"\n--- {label}: n={len(eps)} scored={len(sc)} outcomes={dict(oc)}")
    for fam in sorted({e["family"] for e in eps}):
        fe = [e for e in sc if e["family"] == fam]
        fa = [e for e in eps if e["family"] == fam]
        acc = mean([1.0 if e["correct"] else 0.0 for e in fe])
        se = math.sqrt(acc * (1 - acc) / len(fe)) if fe else float("nan")
        print(f"  {fam:17s} n={len(fe):3d}/{len(fa):3d} acc={acc:.3f}±{se:.3f} turns={mean([len(e['turns']) for e in fe]):.2f} "
              f"subcalls={mean([e['subcalls_used'] for e in fe]):.2f} used_sub={mean([1.0 if e['subcalls_used'] else 0.0 for e in fe]):.2f} "
              f"reward={mean([e['reward'] for e in fe]):.3f} submitted={mean([1.0 if e['submitted'] else 0.0 for e in fe]):.2f} "
              f"wall_p50={q([e['wall_s'] for e in fe], .5):.0f} p95={q([e['wall_s'] for e in fe], .95):.0f} max={max([e['wall_s'] for e in fe] or [0]):.0f} "
              f"cost/ep={mean([e['usage']['est_cost_usd'] for e in fe]):.4f} ctx={mean([e['context_chars'] for e in fe]):.0f}")
        print("     outcomes", dict(Counter(outcome(e) for e in fa)),
              "wrong-answer error buckets", dict(Counter(err_bucket(e) for e in fe if outcome(e) == "wrong_answer")))
        # correct vs incorrect behaviour
        for c in (True, False):
            ce = [e for e in fe if bool(e["correct"]) == c]
            if ce:
                print(f"     correct={c}: n={len(ce)} turns={mean([len(e['turns']) for e in ce]):.2f} "
                      f"sub={mean([e['subcalls_used'] for e in ce]):.2f} used_sub={mean([1.0 if e['subcalls_used'] else 0 for e in ce]):.2f}")
    # subcall use vs correctness overall
    for k in ("0 subcalls", ">=1 subcalls"):
        ce = [e for e in sc if (e["subcalls_used"] == 0) == (k == "0 subcalls")]
        print(f"  {k}: n={len(ce)} acc={mean([1.0 if e['correct'] else 0 for e in ce]):.3f}")


summarize(val, "HELD-OUT heldout-seed1 @ ckpt 20")
# held-out infra episodes: which family, how far they got
for e in val:
    if e.get("infra_kind"):
        print("  infra ep:", e["family"], "turns", len(e["turns"]), "subcalls", e["subcalls_used"], "wall", e["wall_s"],
              "task", e["task_id"][-3:], "sample", e["sample"])
# held-out tasks: two samples each? per-task agreement
bytask = defaultdict(list)
for e in val:
    bytask[e["task_id"]].append(e)
print("  heldout tasks:", len(bytask), "samples/task", Counter(len(v) for v in bytask.values()))
both = Counter()
for t, v in bytask.items():
    sv = [e for e in v if not e.get("infra_kind")]
    if len(sv) == 2:
        both[(sv[0]["family"], sum(1 for e in sv if e["correct"]))] += 1
print("  per-task correct count (2 samples)", dict(both))

summarize(train[1], "TRAIN step 1 (base model)")
summarize([e for s in range(1, 6) for e in train[s]], "TRAIN steps 1-5")
summarize([e for s in range(16, 21) for e in train[s]], "TRAIN steps 16-20")
summarize([e for s in train for e in train[s]], "TRAIN all")

print("\n=== per-step behaviour from transcripts ===")
print("st  n  sub/ep used_sub turns ftsub budget  rtok_comp  rprompt/ep  sub_prompt/ep  cost/ep  wall_p50 p90  max  rollouts_s_est  slowest_fam  ctx")
for s, eps in train.items():
    w = [e["wall_s"] for e in eps]
    slow = max(eps, key=lambda e: e["wall_s"])
    print(f"{s:2d} {len(eps):2d} {mean([e['subcalls_used'] for e in eps]):5.2f} {mean([1.0 if e['subcalls_used'] else 0 for e in eps]):5.2f} "
          f"{mean([len(e['turns']) for e in eps]):5.2f} {mean([e['metrics']['first_turn_submit'] for e in eps]):4.2f} "
          f"{mean([e['metrics']['budget_hit'] for e in eps]):4.2f} {mean([e['usage']['root_completion_tokens'] for e in eps]):8.0f} "
          f"{mean([e['usage']['root_prompt_tokens'] for e in eps]):10.0f} {mean([e['usage']['subcall_prompt_tokens'] for e in eps]):10.0f} "
          f"{mean([e['usage']['est_cost_usd'] for e in eps]):.4f} {q(w, .5):6.0f} {q(w, .9):5.0f} {max(w):5.0f} "
          f"{slow['family'][:10]:>10s} t={len(slow['turns'])} sc={slow['subcalls_used']}  {mean([e['context_chars'] for e in eps]):.0f}")

# cost split by component across all train episodes
P_IN, P_OUT = 0.40, 3.00
rp = sum(e["usage"]["root_prompt_tokens"] for s in train for e in train[s])
rc = sum(e["usage"]["root_completion_tokens"] for s in train for e in train[s])
sp = sum(e["usage"]["subcall_prompt_tokens"] for s in train for e in train[s])
scc = sum(e["usage"]["subcall_completion_tokens"] for s in train for e in train[s])
tot = (rp + sp) * P_IN + (rc + scc) * P_OUT
print(f"\ncost split (train, est): root prompt {rp*P_IN/tot:.1%} root completion {rc*P_OUT/tot:.1%} "
      f"subcall prompt {sp*P_IN/tot:.1%} subcall completion {scc*P_OUT/tot:.1%}; total ${tot/1e6:.2f}")
# root prompt per episode vs turns: quadratic growth
by_t = defaultdict(list)
for s in train:
    for e in train[s]:
        by_t[len(e["turns"])].append(e["usage"]["root_prompt_tokens"])
print("root prompt tokens/episode by turns:", {k: (len(v), round(mean(v))) for k, v in sorted(by_t.items())})

# wall time tails: per-step slowest vs median; what drives the slowest
allep = [e for s in train for e in train[s]]
w = sorted(e["wall_s"] for e in allep)
print(f"\nwall_s all train eps: p50={q(w,.5):.0f} p90={q(w,.9):.0f} p99={q(w,.99):.0f} max={max(w):.0f}")
top = sorted(allep, key=lambda e: -e["wall_s"])[:10]
for e in top:
    root_s = sum(t.get("root_s") or 0 for t in e["turns"])
    cell_s = sum(t.get("cell_s") or 0 for t in e["turns"])
    sub_s = sum(sc.get("seconds") or 0 for sc in e["subcalls"])
    print(f"  slow: step {e['step']} {e['family']} wall={e['wall_s']:.0f} turns={len(e['turns'])} sub={e['subcalls_used']} "
          f"root_s={root_s:.0f} cell_s={cell_s:.0f} sub_s_sum={sub_s:.0f} correct={e['correct']} infra={e.get('infra_kind')}")
# root_s per turn distribution (latency incl. retry waits)
rs = [t.get("root_s") for e in allep for t in e["turns"] if t.get("root_s") is not None]
print(f"root_s per turn: p50={q(rs,.5):.1f} p90={q(rs,.9):.1f} p99={q(rs,.99):.1f} max={max(rs):.1f} n={len(rs)}")
ss = [sc.get("seconds") for e in allep for sc in e["subcalls"] if sc.get("seconds") is not None]
print(f"subcall seconds: p50={q(ss,.5):.1f} p90={q(ss,.9):.1f} p99={q(ss,.99):.1f} max={max(ss):.1f} n={len(ss)}")
# per-step root_s p50 / p90
print("per-step root_s p50/p90 and subcall p50/p90:")
for s, eps in train.items():
    r = [t.get("root_s") for e in eps for t in e["turns"] if t.get("root_s") is not None]
    sc = [x.get("seconds") for e in eps for x in e["subcalls"] if x.get("seconds") is not None]
    print(f"  {s:2d} root p50={q(r,.5):5.1f} p90={q(r,.9):5.1f} max={max(r):5.1f} | sub p50={q(sc,.5):5.1f} p90={q(sc,.9):5.1f} max={max(sc) if sc else 0:5.1f}")

# hacks / flags / events
print("\nflagged train episodes:")
for e in allep:
    if e.get("hacks") or e.get("flags"):
        print("  step", e["step"], e["family"], "hacks", e["hacks"], "flags", e["flags"], "correct", e["correct"],
              "events", [ev.get("kind") if isinstance(ev, dict) else ev for ev in e.get("events", [])][:5])
print("train infra episodes:", [(e["step"], e["family"], e.get("infra_kind"), len(e["turns"]), e["wall_s"]) for e in allep if e.get("infra_kind")])
ev = Counter()
for e in allep + val:
    for x in e.get("events", []):
        ev[x.get("kind") if isinstance(x, dict) else str(x)] += 1
print("events kinds:", dict(ev))
print("finish reasons:", Counter(t.get("finish_reason") for e in allep for t in e["turns"]))
print("val finish reasons:", Counter(t.get("finish_reason") for e in val for t in e["turns"]))
