"""Task generators. They run on the host only (H3): a generator, its seed and its gold answer never enter the sandbox.

Every task gets a fresh random 64-bit seed from `new_seed()`, and only the generated context
text is handed to the sandbox. Eval seed lists live in the git-ignored private/ folder, never
in the repo, a prompt or a log the model can see.

Each family maps to the module that generates it. A generator module provides
`make_task(seed, family, level, task_id)`, `MAX_LEVEL` and `GENERATOR_VERSION`, and every task
it makes carries its blind guess in `task.grading` (spec §3).

Plan step 7 adds the production families and decision 7a's domain mix (`DOMAIN_MIX`). The cyber
domain's 35% share is still unbuilt (Rishi cut it from step 7 on 2026-10-11), so `production_mix()`
refuses to make a mix without it unless told to leave it out. Families whose name ends in `_heldout`,
`_select` or `_test` are eval-only (H7), and `check_mix()` keeps them, and any family that reads a
source outside the licence bar (7b), out of training.
"""

import importlib
import json
import os
import secrets
from pathlib import Path

PRIVATE_DIR = Path(__file__).resolve().parents[2] / "private"

# family -> module in rlm_rl.tasks
GENERATORS = {
    "tickets_all": "tickets", "tickets_customer": "tickets",  # the second seed's families; held out of production
    "aggregation": "aggregation", "aggregation_heldout": "aggregation",
    "callgraph": "synthcode", "needle": "synthcode", "callgraph_heldout": "synthcode", "needle_heldout": "synthcode",
    "repo_loc": "repoloc", "repo_loc_select": "repoloc", "repo_loc_test": "repoloc",
}
EVAL_ONLY_SUFFIXES = ("_heldout", "_select", "_test")

# Decision 7a (Rishi, 2026-10-11): the production domain mix, and the training families that fill each domain.
DOMAIN_MIX = {"cyber": 0.35, "aggregation": 0.30, "code": 0.20, "repo": 0.15}
DOMAIN_FAMILIES = {"cyber": (), "aggregation": ("aggregation",), "code": ("callgraph", "needle"), "repo": ("repo_loc",)}


def new_seed() -> int:
    return secrets.randbits(64)


def save_private_seeds(name: str, seeds: list[int]) -> Path:
    path = PRIVATE_DIR / "seeds" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([int(s) for s in seeds]), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def load_private_seeds(name: str) -> list[int]:
    return [int(s) for s in json.loads((PRIVATE_DIR / "seeds" / f"{name}.json").read_text(encoding="utf-8"))]


def generator(family: str):
    if family not in GENERATORS:
        raise KeyError(f"no generator for family {family!r}")
    return importlib.import_module(f"{__name__}.{GENERATORS[family]}")


def max_levels(families) -> dict[str, int]:
    """Each family's top level, for the difficulty controller."""
    return {family: generator(family).MAX_LEVEL for family in families}


def versions(families) -> dict[str, str]:
    """Each family's generator version, saved with every checkpoint (spec §8)."""
    return {family: generator(family).GENERATOR_VERSION for family in families}


def make_task(family: str, level: int, seed: int | None = None, task_id: str | None = None):
    """One task at a level, from a fresh seed unless one is given (eval seed lists)."""
    return generator(family).make_task(new_seed() if seed is None else seed, family, level, task_id)


def eval_only(family: str) -> bool:
    return family.endswith(EVAL_ONLY_SUFFIXES)


def training_sources(family: str) -> tuple[str, ...]:
    """The data sources a family reads (rlm_rl/data.py SOURCES). Purely synthetic families read none."""
    return tuple(getattr(generator(family), "TRAIN_SOURCES", ()))


def missing_domains() -> list[str]:
    return [domain for domain, families in DOMAIN_FAMILIES.items() if not families]


def production_mix(leave_out_missing: bool = False) -> dict[str, float]:
    """Decision 7a as family weights: each domain's share split evenly over its families. A domain with no
    families yet (cyber) raises, unless `leave_out_missing`, which spreads the built domains over 100%."""
    missing = missing_domains()
    if missing and not leave_out_missing:
        raise ValueError(f"domain(s) {missing} have no task family yet (cyber was deferred from plan step 7); "
                         "pass leave_out_missing=True to train without them")
    built = {d: w for d, w in DOMAIN_MIX.items() if DOMAIN_FAMILIES[d]}
    total = sum(built.values())
    return {f: w / total / len(DOMAIN_FAMILIES[d]) for d, w in built.items() for f in DOMAIN_FAMILIES[d]}


def check_mix(mix: dict[str, float]) -> None:
    """Refuse a training mix with an eval-only family (H7) or a family reading a source the licence bar keeps
    out of training (7b)."""
    from .. import data

    for family in mix:
        if eval_only(family):
            raise ValueError(f"{family} is eval-only (H7) and can't be in a training mix")
        data.check_training_sources(family, list(training_sources(family)))


def allocate(n: int, mix: dict[str, float]) -> dict[str, int]:
    """Split n tasks across families in proportion to `mix`, largest remainder first; ties go to the first listed."""
    if n < 0 or not mix or any(w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError("need n >= 0 and non-negative weights that sum to more than 0")
    total = sum(mix.values())
    exact = {family: n * w / total for family, w in mix.items()}
    counts = {family: int(x) for family, x in exact.items()}
    order = sorted(mix, key=lambda f: (-(exact[f] - counts[f]), list(mix).index(f)))
    for family in order[: n - sum(counts.values())]:
        counts[family] += 1
    return counts


def draw(n: int, mix: dict[str, float], levels: dict[str, int]) -> list:
    """One step's tasks: n tasks split across families by `mix`, each at its family's current level, fresh seeds."""
    out = []
    for family, count in allocate(n, mix).items():
        out.extend(make_task(family, levels[family]) for _ in range(count))
    secrets.SystemRandom().shuffle(out)
    return out


def private_seed_list(name: str, n: int) -> list[int]:
    """n seeds kept in private/seeds/<name>.json, drawn once."""
    try:
        seeds = load_private_seeds(name)
    except FileNotFoundError:
        seeds = [new_seed() for _ in range(n)]
        save_private_seeds(name, seeds)
    if len(seeds) != n:
        raise ValueError(f"private/seeds/{name}.json holds {len(seeds)} seeds, but the set needs {n}")
    return seeds


def fixed_set(name: str, plan: dict) -> list:
    """A fixed eval set. `plan` maps (family, level) to a task count. The seeds are drawn once and kept in
    private/seeds/<name>.json, so every eval of the set sees the same tasks, and no seed reaches the repo or a log."""
    seeds = private_seed_list(name, sum(plan.values()))
    out, it = [], iter(seeds)
    for (family, level), count in plan.items():
        out.extend(make_task(family, level, next(it), f"{name}-{family}-L{level}-{i:02d}") for i in range(count))
    return out
