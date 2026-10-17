"""Watch helper for the second seed: one line per step against the plan's rules (spec §5.2, plan step 10).

    python results/preflight/step_check.py second-seed-v1

Reads results/<run>.json only. Flags are things to tell Rishi about; stopping stays his call.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
run = sys.argv[1] if len(sys.argv) > 1 else "second-seed-v1"
s = json.loads((ROOT / "results" / f"{run}.json").read_text(encoding="utf-8"))
steps = [r for r in s["steps"] if r.get("status") == "trained" or r.get("policy")]


def pol(r, k):
    return (r.get("policy") or {}).get(k)


flags = []
if steps:
    r = steps[-1]
    ent = [pol(x, "entropy") for x in steps if pol(x, "entropy") is not None]
    gn = [pol(x, "grad_norm") for x in steps if pol(x, "grad_norm") is not None]
    rs = [pol(x, "reward_std") for x in steps if pol(x, "reward_std") is not None]
    if ent and len(ent) >= 2 and ent[-1] < 0.7 * max(ent[-10:]):
        flags.append(f"entropy {ent[-1]:.3f} under 70% of its 10-step peak {max(ent[-10:]):.3f}")
    if len(ent) >= 6:
        first = statistics.mean(ent[:5])
        if statistics.mean(ent[-5:]) < 0.8 * first:
            flags.append(f"entropy last-5 mean {statistics.mean(ent[-5:]):.3f} under 80% of steps 1-5 ({first:.3f})")
    if len(gn) >= 3 and gn[-1] > 3 * statistics.median(gn):
        flags.append(f"grad norm {gn[-1]:.3f} over 3x its median {statistics.median(gn):.3f}")
    pc = pol(r, "probs_corr")
    if pc is not None and pc < 0.98:
        flags.append(f"probs_corr {pc:.4f} under 0.98")
    cf = pol(r, "clip_fraction")
    if cf is not None and cf > 0.05:
        flags.append(f"clip fraction {cf:.3f} (near 0 is normal with the clip restored)")
    if len(rs) >= 4:
        low = [rs[i] < 0.5 * statistics.mean(rs[max(0, i - 10):i]) for i in range(1, len(rs))]
        if len(low) >= 3 and all(low[-3:]):
            flags.append("reward std under half its 10-step mean for 3 steps")
    inf = (r.get("infra") or {})
    if inf.get("infra"):
        flags.append(f"infra episodes {inf.get('infra')} ({inf.get('kinds')})")
    hacks = {k: v for k, v in ((r.get("reward") or {}).get("hacks") or {}).items() if v}
    if hacks:
        flags.append(f"detector hits {hacks}")
    if r.get("skip_train"):
        flags.append(f"train call skipped: {r.get('skip_train')}")
    p = r.get("pass") or {}
    t = r.get("time") or {}
    print(f"step {r['step']}: pass {p.get('rate')} trainable {p.get('trainable_groups')}/{p.get('groups')} "
          f"entropy {pol(r, 'entropy')} reward_std {pol(r, 'reward_std')} KL {pol(r, 'kl_to_base')} "
          f"grad {pol(r, 'grad_norm')} probs_corr {pc} clip {cf} | {t.get('total_s')} s "
          f"(rollouts {t.get('rollouts_s')}, KL {t.get('kl_s')}, train {t.get('train_s')}) "
          f"| step ${(r.get('cost') or {}).get('step_usd')} total ${s['spend'].get('total_usd')} "
          f"| infra {inf.get('rate')} retries {inf.get('retries')} 429s {inf.get('rate_limited')}")
print(f"status {s['status']} live {s.get('live')} evals {len(s['evals'])} notices {len(s['notices'])}")
for n in s["notices"][-3:]:
    print(f"  notice {n['kind']}: {n['text'][:200]}")
print("FLAGS: " + ("; ".join(flags) if flags else "none"))
