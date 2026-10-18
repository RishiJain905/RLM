"""Task generators. They run on the host only (H3): a generator, its seed and its gold answer never enter the sandbox.

Every task gets a fresh random 64-bit seed from `new_seed()`, and only the generated context
text is handed to the sandbox. Eval seed lists live in the git-ignored private/ folder, never
in the repo, a prompt or a log the model can see.

Each family maps to the module that generates it. A generator module provides
`make_task(seed, family, level, task_id)`, `MAX_LEVEL` and `GENERATOR_VERSION`, and every task
it makes carries its blind guess in `task.grading` (spec §3).

Plan step 7 adds the production families and the domain mix (`DOMAIN_MIX`, decisions 7a and 7c).
Families whose name ends in `_heldout`, `_select` or `_test` are eval-only (H7), and `check_mix()`
keeps them, and any family that reads a source outside the licence bar (7b), out of training.
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
    "edit_synth": "editsynth", "edit_synth_heldout": "editsynth",  # spec §12.7
    "edit_real": "editreal", "edit_real_select": "editreal", "edit_real_test": "editreal",
    # the step-12 task shift (documentation/step-12-multistep-plan.md, 2026-10-18)
    "research": "research", "research_select": "research", "research_test": "research",
    "edit_spec": "editspec", "edit_spec_select": "editspec", "edit_spec_test": "editspec",
    "edit_bugs": "editbugs", "edit_bugs_select": "editbugs", "edit_bugs_test": "editbugs",
}
EVAL_ONLY_SUFFIXES = ("_heldout", "_select", "_test")

# The production domain mix, and the training families that fill each domain. Decision 7a (Rishi, 2026-10-11) was
# 35% cyber, 30% aggregation, 20% code, 15% repo; decision 7c (2026-10-11) dropped cyber for 45/40/15. Spec v2 §12.7
# (decision E-a, Rishi 2026-10-12) gives editing 40%: aggregation 25%, code reading 20%, repo localization 15%.
DOMAIN_MIX = {"aggregation": 0.25, "code": 0.20, "repo": 0.15, "editing": 0.40}
DOMAIN_FAMILIES = {"aggregation": ("aggregation",), "code": ("callgraph", "needle"), "repo": ("repo_loc",),
                   "editing": ("edit_real", "edit_synth")}
# Families whose share of their domain isn't an even split (spec §12.7: editing's 25/15).
FAMILY_SHARES = {"edit_real": 0.25, "edit_synth": 0.15}


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


def make_task(family: str, level: int, seed: int | None = None, task_id: str | None = None, exclude=None):
    """One task at a level, from a fresh seed unless one is given (eval seed lists). `exclude` names retired tasks of
    a fixed pool (edit_real, spec §12.7)."""
    gen = generator(family)
    seed = new_seed() if seed is None else seed
    if exclude and family in RETIRABLE:
        return gen.make_task(seed, family, level, task_id, exclude=exclude)
    return gen.make_task(seed, family, level, task_id)


def eval_only(family: str) -> bool:
    return family.endswith(EVAL_ONLY_SUFFIXES)


def training_sources(family: str) -> tuple[str, ...]:
    """The data sources a family reads (rlm_rl/data.py SOURCES). Purely synthetic families read none."""
    gen = generator(family)
    if callable(getattr(gen, "training_sources", None)):
        return tuple(gen.training_sources())
    return tuple(getattr(gen, "TRAIN_SOURCES", ()))


def production_mix() -> dict[str, float]:
    """The domain mix as family weights: each domain's share split evenly over its families, except where
    FAMILY_SHARES sets a family's share directly (editing's 25/15, spec §12.7). The order is DOMAIN_FAMILIES'."""
    total = sum(DOMAIN_MIX.values())
    mix = {}
    for domain, share in DOMAIN_MIX.items():
        families = DOMAIN_FAMILIES[domain]
        direct = {f: FAMILY_SHARES[f] for f in families if f in FAMILY_SHARES}
        rest = [f for f in families if f not in direct]
        left = share - sum(direct.values())
        for f in families:
            mix[f] = (direct[f] if f in direct else left / len(rest)) / total
    return mix


def check_mix(mix: dict[str, float]) -> None:
    """Refuse a training mix with an eval-only family (H7) or a family reading a source the licence bar keeps
    out of training (7b). edit_real also needs every training repository's own licence to be permissive."""
    from .. import data

    for family in mix:
        if eval_only(family):
            raise ValueError(f"{family} is eval-only (H7) and can't be in a training mix")
        data.check_training_sources(family, list(training_sources(family)))
        if callable(getattr(generator(family), "check_licences", None)):
            generator(family).check_licences()


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


def allocate_step(n: int, mix: dict[str, float], step: int) -> dict[str, int]:
    """Training step `step`'s (1-based) n tasks, as its slice of one run-long sequence that tracks `mix`.

    `allocate(8, ...)` rounds the same way every step, so at 8 tasks the production mix 45/20/20/15
    would always draw 4/2/1/1: 50/25/12.5/12.5 over the run. Here task k of the run goes to the family
    furthest behind its share of the first k+1 tasks (ties to the first listed). Every family then
    stays within one task of its share at every step, and the run's mix is exact whenever the shares
    divide evenly (every 5 steps at 8 tasks: 18/8/8/6 of 40). It depends only on the step number, so a
    resume or a rollback that redoes a step draws the same families for it.
    """
    if n < 0 or step < 1 or not mix or any(w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError("need n >= 0, step >= 1 and non-negative weights that sum to more than 0")
    total = sum(mix.values())
    families = list(mix)
    share = {f: mix[f] / total for f in families}
    so_far = dict.fromkeys(families, 0)
    out = dict.fromkeys(families, 0)
    for k in range(step * n):
        family = max(families, key=lambda f: ((k + 1) * share[f] - so_far[f], -families.index(f)))
        so_far[family] += 1
        if k >= (step - 1) * n:
            out[family] += 1
    return out


def draw(n: int, mix: dict[str, float], levels: dict[str, int], step: int | None = None, exclude=None) -> list:
    """One step's tasks: n tasks split across families by `mix`, each at its family's current level, fresh seeds.

    With `step` (the training loop always passes it), the split carries its rounding across steps
    (`allocate_step`); without it, every call rounds the same way (`allocate`). `exclude` holds retired
    task ids (spec §12.7); a fixed pool also never repeats a task within one step. A slot of a fixed pool
    with no task left to draw goes to its FALLBACK family, if the run has it (decision 12b).
    """
    out = []
    counts = allocate(n, mix) if step is None else allocate_step(n, mix, step)
    for family, count in counts.items():
        taken = set(exclude or ())
        for _ in range(count):
            if family not in RETIRABLE:
                out.append(make_task(family, levels[family]))
                continue
            try:
                task = make_task(family, levels[family], exclude=taken)
            except PoolEmpty:
                sub = FALLBACK.get(family)
                if sub not in levels:
                    raise
                if sub in RETIRABLE:  # another pool: skip its retired tasks and this step's draws of it too
                    task = make_task(sub, levels[sub], exclude=taken | {t.task_id for t in out if t.family == sub})
                else:
                    task = make_task(sub, levels[sub])
                out.append(task)
                continue
            taken.add(task.task_id)
            out.append(task)
    secrets.SystemRandom().shuffle(out)
    return out


def pool_size(family: str) -> int | None:
    """How many training tasks a fixed pool holds (edit_real's index), or None when it can't be read."""
    size = getattr(generator(family), "pool_size", None)
    try:
        return size(family) if callable(size) else None
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- task retirement (spec §12.7)

RETIRABLE = ("edit_real", "edit_spec", "edit_bugs")  # fixed pools whose tasks come back; generators never repeat one
RETIRE_ALL_ZERO = 2  # a task whose group came back all-0 this many times isn't drawn again
RETIRE_ALL_ONE = 1  # nor one whose group came back all-1 once
# Families built to need sub-calls (step-12 task shift, 2026-10-18). A task of theirs that the policy solves without a
# single sub-call (at least half the group right, and none of the right ones delegated) teaches nothing about
# delegation, so it retires too (ASearcher's "solved in one tool turn" filter, arXiv 2508.07976).
SOLO_RETIRABLE: tuple = ("edit_spec", "edit_bugs")
RETIRE_SOLO = 1
# Decision 12b (Rishi, 2026-10-17): ~150 edit_real draws are wanted from 107 training tasks, so retirement can use the
# pool up. Its slots then go to edit_synth, which generates fresh tasks at its own level, rather than redrawing retired
# tasks: those came back all-1 (no signal left) or all-0 twice. A rollback restores the retired set, and the pool.
# The task shift's two coding pools (150 training tasks each) stand in for each other when one runs dry.
FALLBACK = {"edit_real": "edit_synth", "edit_spec": "edit_bugs", "edit_bugs": "edit_spec"}


class PoolEmpty(ValueError):
    """A fixed pool has no task left to draw: every one is retired, or already drawn this step."""


class Retirement:
    """Which tasks of a fixed pool keep coming back all-0 or all-1, so they stop wasting groups. Saved with every
    checkpoint, so a resume or rollback keeps it."""

    def __init__(self, state: dict | None = None):
        state = state or {}
        self.counts: dict[str, dict] = {k: dict(v) for k, v in (state.get("counts") or {}).items()}
        self.retired: dict[str, dict] = {k: dict(v) for k, v in (state.get("retired") or {}).items()}

    def update(self, step: int, groups: list) -> list[str]:
        """Record one step's groups. Returns the task ids retired by it."""
        newly = []
        for group in groups:
            clean = [ep for ep in group if ep.infra_error is None]
            if len(clean) < 2 or clean[0].family not in RETIRABLE:
                continue
            task_id, family = clean[0].task_id, clean[0].family
            rewards = {ep.reward for ep in clean}
            equal = len(rewards) == 1 and next(iter(rewards)) in (0.0, 1.0)
            right = [ep for ep in clean if ep.correct]
            solo = (family in SOLO_RETIRABLE and 2 * len(right) >= len(clean)
                    and not any(getattr(ep, "subcalls_used", 0) for ep in right))
            if not equal and not solo:
                continue
            row = self.counts.setdefault(task_id, {"zero": 0, "one": 0})
            if equal:
                row["one" if clean[0].reward == 1.0 else "zero"] += 1
            if solo:
                row["solo"] = row.get("solo", 0) + 1
            why = ("all-1" if row["one"] >= RETIRE_ALL_ONE else "all-0" if row["zero"] >= RETIRE_ALL_ZERO
                   else "solved without sub-calls" if row.get("solo", 0) >= RETIRE_SOLO else None)
            if task_id not in self.retired and why:
                self.retired[task_id] = {"step": step, "why": why, "family": family}
                newly.append(task_id)
        return newly

    def ids(self) -> set[str]:
        return set(self.retired)

    def state_dict(self) -> dict:
        return {"counts": self.counts, "retired": self.retired,
                "rule": {"all_zero": RETIRE_ALL_ZERO, "all_one": RETIRE_ALL_ONE, "families": list(RETIRABLE),
                         "solo": RETIRE_SOLO, "solo_families": list(SOLO_RETIRABLE)}}


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
