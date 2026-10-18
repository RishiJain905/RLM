"""H1: does a blind guesser earn more than its lucky exact hits? (plan step 5 done-when, spec §3)

    python3 scripts/reward_check.py                 # every training family and level, 1,000 tasks each
    python3 scripts/reward_check.py -n 300 --slope 2

For each family and level it generates tasks on the host and scores answers that never read
the context, through the real scorer (`reward.score_answer`). A family mixes answer types
(spec §3), so each task is scored by its own type:

- blind: the task's own blind guess (`task.grading["blind"]`). The done-when check is that its
  mean reward is no more than its exact-hit rate. It holds by construction for every type.
- numbers also get a best fixed offset and Gaussian noise on the blind guess, and honest
  readers whose count is off by Gaussian noise of 1, 2 or 3.
- lists also get subset guesses (the first k of the blind list, the best k) and a random
  subset, and honest readers missing 1, 2 or 3 items.
- names (entities) have no guess and no partial credit, so every guesser and reader earns 0.

`--slope` scores with a different NEAR_MISS_SLOPE (1 is the spec; 2 halves the credit window,
decision 5b). Nothing here calls a model or spends credit.
"""

import argparse
import collections
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import reward, tasks  # noqa: E402

# The training families (the plan's step 8 areas B and D) plus the held-out ticket generator.
# `--families` adds the held-out step 7 families.
DEFAULT_FAMILIES = ("tickets_all", "tickets_customer", "aggregation", "callgraph", "needle", "repo_loc")


def guess_rewards(t, rng: random.Random, spread: float = 0.0) -> dict:
    """The per-task stochastic guessers and honest readers, by answer type (spec §3). The "offset" slot is
    filled by best_fixed_strategy in check_family: a guesser picks one strategy before seeing tasks, so
    the best fixed one is measured over the whole set, not chosen per task. `spread` is the family's
    typical blind error, the noise scale step 5 documented (per-task noise would inflate it)."""
    kind, blind = t.grading["type"], t.grading.get("blind")

    if kind in ("int", "float", "count"):
        gold, guess = reward.parse_number(t.answer), reward.parse_number(blind)
        scale = spread or abs(guess - gold) or 1.0
        noisy = {s: reward.score_answer(guess + round(rng.gauss(0, s * scale)), t.answer, t.grading).reward
                 for s in (0.5, 1.0)}
        readers = {s: reward.score_answer(gold + round(rng.gauss(0, s)), t.answer, t.grading).reward
                   for s in (1, 2, 3)}
        return {"offset": 0.0, "noisy_half": noisy[0.5], "noisy_one": noisy[1.0],
                "reader1": readers[1], "reader2": readers[2], "reader3": readers[3]}
    if kind in ("entity", "text"):  # text (research): no blind guess either
        return {"offset": 0.0, "noisy_half": 0.0, "noisy_one": 0.0,
                "reader1": 0.0, "reader2": 0.0, "reader3": 0.0}  # no guess, no partial credit

    # lists: a noisy guesser submits a random-ish subset of the task's own blind list, readers miss items
    blind_list = list(blind) if isinstance(blind, list) else ([] if blind is None else [blind])
    noisy_k = max(0, min(len(blind_list), round(len(blind_list) / 2 + rng.gauss(0, max(1, len(blind_list) / 4)))))
    noisy = reward.score_answer(rng.sample(blind_list, noisy_k) if noisy_k else [], t.answer, t.grading).reward
    readers = {s: reward.score_answer(t.answer[s:], t.answer, t.grading).reward for s in (1, 2, 3)}
    return {"offset": 0.0, "noisy_half": noisy, "noisy_one": noisy,
            "reader1": readers[1], "reader2": readers[2], "reader3": readers[3]}


def best_fixed_strategy(tasks_of_type: list) -> float:
    """What the best FIXED guesser earns on one answer type's tasks: one offset (numbers) or one prefix
    fraction (lists), picked before seeing the tasks. Step 5's semantics, extended to lists."""
    kind = tasks_of_type[0].grading["type"]
    if kind in ("entity", "text"):
        return 0.0
    if kind in ("int", "float", "count"):
        spreads = [abs(reward.parse_number(t.grading["blind"]) - reward.parse_number(t.answer)) or 1.0
                   for t in tasks_of_type]
        mean_spread = statistics.fmean(spreads)
        offsets = sorted({round(k * mean_spread / 4) for k in range(-12, 13)} - {0})
        return max((statistics.fmean(reward.score_answer(t.grading["blind"] + d, t.answer, t.grading).reward
                                      for t in tasks_of_type), d) for d in offsets)[0] if offsets else 0.0
    fractions = (0.25, 0.5, 0.75, 1.0)
    return max(statistics.fmean(
        reward.score_answer(list(t.grading["blind"])[:round(f * len(t.grading["blind"]))],
                            t.answer, t.grading).reward for t in tasks_of_type) for f in fractions)


def check_family(family: str, level: int, n: int, seed: int) -> dict:
    task_list = [tasks.make_task(family, level, seed=seed * 1000 + i) for i in range(n)]
    rng = random.Random(seed)
    by_type = collections.defaultdict(list)
    for t in task_list:
        by_type[t.grading["type"]].append(t)
    spreads = {kind: (statistics.fmean(abs(reward.parse_number(t.grading["blind"]) - reward.parse_number(t.answer))
                                        or 1.0 for t in ts) if kind in ("int", "float", "count") else 0.0)
               for kind, ts in by_type.items()}
    blinds = [reward.blind_score(t) for t in task_list]
    rows = [guess_rewards(t, rng, spreads[t.grading["type"]]) for t in task_list]
    strategy = {kind: best_fixed_strategy(ts) for kind, ts in by_type.items()}
    for r, t in zip(rows, task_list):  # the best fixed guesser of this task's answer type
        r["offset"] = strategy[t.grading["type"]]
    exact_rate = statistics.fmean(float(s.correct) for s in blinds)
    blind_reward = statistics.fmean(s.reward for s in blinds)
    means = {k: statistics.fmean(r[k] for r in rows) for k in rows[0]}
    return {"family": family, "level": level, "n": n,
            "types": dict(collections.Counter(t.grading["type"] for t in task_list)),
            "exact_hit_rate": exact_rate, "blind_reward": blind_reward,
            "passes": blind_reward <= exact_rate + 1e-12,
            "best_offset": means["offset"], "noisy_half": means["noisy_half"], "noisy_one": means["noisy_one"],
            "reader1": means["reader1"], "reader2": means["reader2"], "reader3": means["reader3"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=1000, help="tasks per family and level")
    parser.add_argument("--slope", type=float, default=reward.NEAR_MISS_SLOPE)
    parser.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    reward.NEAR_MISS_SLOPE = args.slope
    print(f"Near-miss slope {args.slope:g} ({'spec' if args.slope == 1 else 'not the spec'}), {args.n} tasks each.\n")
    print("| Family | Level | answer types | Exact-hit rate | Blind guess | Check | Best offset/subset "
          "| Noisy (0.5x) | Noisy (1x) | Reader off by 1 | Reader off by 2 | Reader off by 3 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    ok = True
    for family in args.families:
        for level in range(1, tasks.max_levels([family])[family] + 1):
            r = check_family(family, level, args.n, args.seed)
            ok &= r["passes"]
            types = ", ".join(f"{k}:{v}" for k, v in sorted(r["types"].items()))
            print(f"| {family} | {level} | {types} | {r['exact_hit_rate']:.3f} | {r['blind_reward']:.3f} "
                  f"| {'pass' if r['passes'] else 'FAIL'} | {r['best_offset']:.3f} | {r['noisy_half']:.3f} "
                  f"| {r['noisy_one']:.3f} | {r['reader1']:.3f} | {r['reader2']:.3f} | {r['reader3']:.3f} |",
                  flush=True)
    print(f"\nDone-when check (blind guess earns no more than its exact-hit rate): {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())