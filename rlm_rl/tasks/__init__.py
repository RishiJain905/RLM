"""Task generators. They run on the host only (H3): a generator, its seed and its gold answer never enter the sandbox.

Every task gets a fresh random 64-bit seed from `new_seed()`, and only the generated context
text is handed to the sandbox. Eval seed lists live in the git-ignored private/ folder, never
in the repo, a prompt or a log the model can see.

Each family maps to the module that generates it. A generator module provides
`make_task(seed, family, level, task_id)`, `MAX_LEVEL` and `GENERATOR_VERSION`, and every task
it makes carries its blind guess in `task.grading` (spec §3). Step 7 adds the cyber, code and
OOLONG families here.
"""

import importlib
import json
import os
import secrets
from pathlib import Path

PRIVATE_DIR = Path(__file__).resolve().parents[2] / "private"

GENERATORS = {"tickets_all": "tickets", "tickets_customer": "tickets"}  # family -> module in rlm_rl.tasks


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
