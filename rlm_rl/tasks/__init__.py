"""Task generators. They run on the host only (H3): a generator, its seed and its gold answer never enter the sandbox.

Every task gets a fresh random 64-bit seed from `new_seed()`, and only the generated context
text is handed to the sandbox. Eval seed lists live in the git-ignored private/ folder, never
in the repo, a prompt or a log the model can see.
"""

import json
import os
import secrets
from pathlib import Path

PRIVATE_DIR = Path(__file__).resolve().parents[2] / "private"


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
