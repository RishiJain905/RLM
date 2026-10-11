"""Plan step 7's done-when checks on the generated data (H7, H8, H1, the licence bar and the 7a mix).

    python scripts/data_check.py            # writes manifests/data-check.json and prints the tables
    python scripts/data_check.py -n 6       # fewer tasks per family and level, for a quick look

Checks:
- H8: each family's context sizes by level; every training family's top level has a median above 130K characters.
- H1: at every family and level, the task's own blind guess earns no more than its exact-hit rate.
- H7: held-out families share no question n-grams with training, held-out sources share no text with training
  sources, held-out behaviours are disjoint, and the selection and test LCA items never appear in training.
- 7b: the production mix passes the licence bar, and no eval-only family or source is in it.
- 7a: the mix, the unbuilt cyber share, and the share of training tasks that code alone can solve (~20% target).
- The selection set (64 tasks) and the held-out test build, with unique ids and their sizes.

Nothing here calls a model or spends credit. Building the selection set and the held-out test creates their
private seed lists in private/seeds/ (git-ignored) the first time, as the training loop would.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import capability, capability_set, data, evals, reward, tasks  # noqa: E402
from rlm_rl.tasks import aggregation, behaviours, oolong, repoloc, synthcode  # noqa: E402

TRAINING = ("aggregation", "callgraph", "needle", "repo_loc")
HELDOUT = ("aggregation_heldout", "callgraph_heldout", "needle_heldout", "repo_loc_select")
H8_MIN_MEDIAN = 130_000
NGRAM = 6


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", re.sub(r"\{\w+\}", " ", text.lower()))


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def overlap(heldout: list[str], train: list[str], n: int = NGRAM) -> dict:
    train_grams = set().union(*(ngrams(t, n) for t in train)) if train else set()
    held_grams = set().union(*(ngrams(t, n) for t in heldout)) if heldout else set()
    shared = held_grams & train_grams
    return {"n": n, "heldout_ngrams": len(held_grams), "shared": len(shared),
            "examples": [" ".join(g) for g in sorted(shared)[:5]]}


def sizes_and_blind(families, n: int, rng: random.Random) -> dict:
    out = {}
    for family in families:
        module = tasks.generator(family)
        for level in range(1, module.MAX_LEVEL + 1):
            t0 = time.time()
            batch = [tasks.make_task(family, level, seed=rng.getrandbits(64)) for _ in range(n)]
            lens = [len(t.context) for t in batch]
            blind_reward, blind_exact = [], []
            for t in batch:
                guess = t.grading.get("blind")
                score = reward.score_answer(guess, t.answer, t.grading)
                blind_reward.append(score.reward)
                blind_exact.append(float(score.correct))
            out[f"{family}/L{level}"] = {
                "family": family, "level": level, "n": n, "median_chars": statistics.median(lens), "min_chars": min(lens),
                "max_chars": max(lens), "blind_reward": statistics.fmean(blind_reward),
                "blind_exact_rate": statistics.fmean(blind_exact),
                "h1_pass": statistics.fmean(blind_reward) <= statistics.fmean(blind_exact) + 1e-12,
                "code_only": statistics.fmean(float(bool(t.grading.get("code_only"))) for t in batch),
                "kinds": dict(collections.Counter(t.grading.get("kind") for t in batch)),
                "seconds_per_task": round((time.time() - t0) / n, 3)}
            print(f"  {family} L{level}: median {out[f'{family}/L{level}']['median_chars']:,.0f} chars, blind "
                  f"{out[f'{family}/L{level}']['blind_reward']:.3f} vs exact {out[f'{family}/L{level}']['blind_exact_rate']:.3f}",
                  flush=True)
    return out


def h7_checks() -> dict:
    agg_train = [t for ts in aggregation.TRAIN_TEMPLATES.values() for t in ts] + list(aggregation.TRAIN_HEADERS)
    agg_held = [t for ts in aggregation.HELDOUT_TEMPLATES.values() for t in ts] + list(aggregation.HELDOUT_HEADERS)
    code_train = [t for ts in synthcode.TRAIN_QUESTIONS.values() for t in ts]
    code_held = [t for ts in synthcode.HELDOUT_QUESTIONS.values() for t in ts]
    desc_train = [d for b in behaviours.TRAIN for d in b.descriptions]
    desc_held = [d for b in behaviours.HELDOUT for d in b.descriptions]
    out = {
        "aggregation_templates": overlap(agg_held, agg_train),
        "code_templates": overlap(code_held, code_train),
        "repoloc_templates": overlap(list(repoloc.HELDOUT_QUESTIONS), list(repoloc.TRAIN_QUESTIONS)),
        "needle_descriptions": overlap(desc_held, desc_train),
        "needle_behaviours_shared": sorted({b.key for b in behaviours.HELDOUT} & {b.key for b in behaviours.TRAIN}),
        "line_formats_shared": sorted(set(aggregation.HELDOUT_FORMATS) & set(aggregation.TRAIN_FORMATS)),
        "call_styles_heldout_only": sorted(synthcode.HELDOUT_STYLES),
        "call_styles_in_training": sorted({s for styles in synthcode.CALL_STYLES.values() for s in styles}),
    }
    # Held-out source text against the training sources: exact duplicates and 8-gram overlap
    train_texts = set()
    for pools in (data.clinc_domain_pools(), data.massive_scenario_pools()):
        for texts in pools.values():
            train_texts.update(t.lower() for t in texts)
    held_texts = {t.lower() for name in aggregation.HELDOUT_LABELS for t in data.huffpost_pools()[name]}
    train8 = set().union(*(ngrams(t, 8) for t in train_texts))
    held8 = set().union(*(ngrams(t, 8) for t in held_texts))
    out["huffpost_vs_training"] = {"heldout_texts": len(held_texts), "exact_duplicates": len(held_texts & train_texts),
                                   "shared_8grams": len(held8 & train8), "heldout_8grams": len(held8)}
    # OOLONG-synth test: its source sets, and how many of its instance texts appear in our training pools
    sample = oolong.load(evals.HELDOUT_OOLONG_PER_BUCKET)
    lines = [ln.split("|| Instance:", 1)[1].strip().lower() for t in sample for ln in t.context.splitlines()
             if "|| Instance:" in ln]
    out["oolong_test"] = {"tasks": len(sample), "source_sets": sorted({t.grading["dataset"] for t in sample}),
                          "instances_checked": len(lines), "instances_in_training_pools": sum(l in train_texts for l in lines)}
    # LCA: no selection or test item in training
    train_items = repoloc.pool("repo_loc")
    train_ids = {it["id"] for it in train_items}
    train_bases = {(it["repo"], it["base_sha"]) for it in train_items}
    select, test = repoloc.pool("repo_loc_select"), repoloc.pool("repo_loc_test")
    train_repos = {it["repo"] for it in train_items}
    out["lca"] = {
        "train_items": len(train_items), "train_repos": len(train_repos),
        "selection_items": len(select), "selection_repos": sorted({it["repo"] for it in select}),
        "selection_repos_in_training": sorted({it["repo"] for it in select} & train_repos),
        "test_items": len(test), "test_ids_in_training": len({it["id"] for it in test} & train_ids),
        "test_base_commits_in_training": len({(it["repo"], it["base_sha"]) for it in test} & train_bases),
        "test_items_from_training_repos": sum(it["repo"] in train_repos for it in test),
        "test_items_from_unseen_repos": sum(it["repo"] not in train_repos for it in test),
        "licences": dict(collections.Counter(it["licence"] for it in train_items)),
    }
    return out


def licence_checks() -> dict:
    mix = tasks.production_mix(leave_out_missing=True)
    tasks.check_mix(mix)  # raises if an eval-only family or a non-permissive source is in it
    by_family = {f: list(tasks.training_sources(f)) for f in mix}
    used = sorted({s for srcs in by_family.values() for s in srcs})
    return {"mix_passes": True, "training_sources": used,
            "training_sources_permissive": all(data.SOURCES[s]["permissive"] and data.SOURCES[s]["role"] == data.TRAIN
                                               for s in used),
            "eval_only_sources": sorted(n for n, s in data.SOURCES.items() if s["role"] == data.EVAL),
            "sources": {n: {"licence": s["licence"], "role": s["role"]} for n, s in data.SOURCES.items()},
            "families_sources": by_family}


def code_only_share(rng: random.Random, n: int) -> dict:
    """The share of training tasks code alone can solve, drawn by the production mix at each level."""
    mix = tasks.production_mix(leave_out_missing=True)
    out = {}
    for level in range(1, 6):
        draws = tasks.allocate(n, mix)
        flags = []
        for family, count in draws.items():
            flags += [bool(tasks.make_task(family, level, seed=rng.getrandbits(64)).grading.get("code_only"))
                      for _ in range(count)]
        out[f"L{level}"] = statistics.fmean(flags)
    return out


def eval_sets() -> dict:
    out = {}
    for name in ("selection-v1", "heldout-test-v1"):
        t0 = time.time()
        built, samples = evals.eval_set(name)
        ids = [t.task_id for t in built]
        by_family = collections.defaultdict(list)
        for t in built:
            by_family[t.family].append(len(t.context))
        out[name] = {"tasks": len(built), "unique_ids": len(set(ids)) == len(ids), "samples": samples,
                     "in_training_mix": sorted({t.family for t in built} & set(tasks.production_mix(True))),
                     "by_family": {f: {"n": len(v), "median_chars": statistics.median(v), "max_chars": max(v)}
                                   for f, v in sorted(by_family.items())},
                     "seconds": round(time.time() - t0, 1)}
        print(f"  {name}: {len(built)} tasks in {out[name]['seconds']} s", flush=True)
    items = capability.load(capability_set.path())
    out["capability"] = {"items": len(items), "by_source": dict(collections.Counter(i.source for i in items))}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=20, help="tasks per family and level")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=str(ROOT / "manifests" / "data-check.json"))
    args = parser.parse_args()
    rng = random.Random(args.seed)

    print("H8 and H1: sizes and blind guesses", flush=True)
    sizes = sizes_and_blind(TRAINING + HELDOUT, args.n, rng)
    tops = {f: sizes[f"{f}/L{tasks.generator(f).MAX_LEVEL}"]["median_chars"] for f in TRAINING}
    report = {
        "h8": {"top_level_median_chars": tops, "pass": all(v > H8_MIN_MEDIAN for v in tops.values())},
        "h1": {"pass": all(r["h1_pass"] for r in sizes.values())},
        "families": sizes,
    }
    print("H7: overlap", flush=True)
    report["h7"] = h7_checks()
    print("7b: licences", flush=True)
    report["licence"] = licence_checks()
    print("7a: mix and code-only share", flush=True)
    report["mix"] = {"decision_7a": tasks.DOMAIN_MIX, "missing_domains": tasks.missing_domains(),
                     "mix_without_missing": tasks.production_mix(leave_out_missing=True),
                     "code_only_share": code_only_share(rng, 200)}
    print("Eval sets", flush=True)
    report["eval_sets"] = eval_sets()
    Path(args.out).write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    h7 = report["h7"]
    h7_pass = (all(h7[k]["shared"] == 0 for k in ("aggregation_templates", "code_templates", "repoloc_templates",
                                                  "needle_descriptions"))
               and not h7["needle_behaviours_shared"] and not h7["line_formats_shared"]
               and h7["huffpost_vs_training"]["exact_duplicates"] == 0
               and h7["lca"]["test_ids_in_training"] == 0 and not h7["lca"]["selection_repos_in_training"])
    print(f"\nH8 (top levels above {H8_MIN_MEDIAN:,} chars): {'pass' if report['h8']['pass'] else 'FAIL'} {tops}")
    print(f"H1 (blind guess earns no more than its exact hits): {'pass' if report['h1']['pass'] else 'FAIL'}")
    print(f"H7 (no shared templates, behaviours, sources or items): {'pass' if h7_pass else 'FAIL'}")
    print(f"7b (licence bar): {'pass' if report['licence']['training_sources_permissive'] else 'FAIL'}")
    print(f"code-only share by level: {report['mix']['code_only_share']}")
    print(f"wrote {args.out}")
    return 0 if report["h8"]["pass"] and report["h1"]["pass"] and h7_pass else 1


if __name__ == "__main__":
    sys.exit(main())
