"""H1: does a blind guesser earn more than its lucky exact hits? (plan step 5 done-when, spec §3)

    python3 scripts/reward_check.py                 # every family and level, 1,000 tasks each
    python3 scripts/reward_check.py -n 300 --slope 2

For each family and level it generates tasks on the host and scores answers that never read
the context:
- blind: the task's own blind guess (`task.grading["blind"]`). The done-when check is that its
  mean reward is no more than its exact-hit rate. It holds by construction for this guesser.
- best offset: the blind guess plus a fixed offset, the best of a range of offsets.
- noisy: the blind guess plus Gaussian noise. A policy that half-learns the guess looks like this.
and, for scale, a reader whose count is off by Gaussian noise of 1, 2 or 3.

`--slope` scores with a different NEAR_MISS_SLOPE (1 is the spec; 2 halves the credit window).
Nothing here calls a model or spends credit.
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import reward, tasks  # noqa: E402


def mean_reward(task_list, guess) -> float:
    return statistics.fmean(reward.score_answer(guess(t), t.answer, t.grading).reward for t in task_list)


def check_family(family: str, level: int, n: int, seed: int) -> dict:
    task_list = [tasks.make_task(family, level, seed=seed * 1000 + i) for i in range(n)]
    blind = lambda t: t.grading["blind"]  # noqa: E731
    exact_rate = statistics.fmean(t.grading["blind"] == t.answer for t in task_list)
    blind_reward = mean_reward(task_list, blind)
    spread = statistics.fmean(abs(t.grading["blind"] - t.answer) for t in task_list) or 1.0
    offsets = sorted({round(k * spread / 4) for k in range(-12, 13)} - {0})
    best = max((mean_reward(task_list, lambda t, d=d: t.grading["blind"] + d), d) for d in offsets)
    rng = random.Random(seed)
    noisy = {s: mean_reward(task_list, lambda t, s=s: t.grading["blind"] + round(rng.gauss(0, s * spread)))
             for s in (0.5, 1.0)}
    readers = {s: mean_reward(task_list, lambda t, s=s: t.answer + round(rng.gauss(0, s))) for s in (1, 2, 3)}
    return {"family": family, "level": level, "n": n, "mean_e_blind": spread, "exact_hit_rate": exact_rate,
            "blind_reward": blind_reward, "passes": blind_reward <= exact_rate + 1e-12,
            "best_offset": best[1], "best_offset_reward": best[0], "noisy_half": noisy[0.5], "noisy_one": noisy[1.0],
            "reader1": readers[1], "reader2": readers[2], "reader3": readers[3]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=1000, help="tasks per family and level")
    parser.add_argument("--slope", type=float, default=reward.NEAR_MISS_SLOPE)
    parser.add_argument("--families", nargs="*", default=sorted(tasks.GENERATORS))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    reward.NEAR_MISS_SLOPE = args.slope
    print(f"Near-miss slope {args.slope:g} ({'spec' if args.slope == 1 else 'not the spec'}), {args.n} tasks each.\n")
    print("| Family | Level | mean e_blind | Exact-hit rate | Blind guess | Check | Best offset | Noisy (0.5x) "
          "| Noisy (1x) | Reader off by 1 | Reader off by 2 | Reader off by 3 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    ok = True
    for family in args.families:
        for level in range(1, tasks.max_levels([family])[family] + 1):
            r = check_family(family, level, args.n, args.seed)
            ok &= r["passes"]
            print(f"| {family} | {level} | {r['mean_e_blind']:.1f} | {r['exact_hit_rate']:.3f} | {r['blind_reward']:.3f} "
                  f"| {'pass' if r['passes'] else 'FAIL'} | {r['best_offset_reward']:.3f} ({r['best_offset']:+d}) "
                  f"| {r['noisy_half']:.3f} | {r['noisy_one']:.3f} | {r['reader1']:.3f} | {r['reader2']:.3f} "
                  f"| {r['reader3']:.3f} |", flush=True)
    print(f"\nDone-when check (blind guess earns no more than its exact-hit rate): {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
