"""Run configuration and the pinned values saved with every checkpoint (spec §5, §8).

Two kinds of run use the training loop (rlm_rl/training.py):
- `second_seed` (plan step 10): 20 steps at level 1 on the ticket tasks, learning rate 1e-5,
  the controller frozen, the infra pause counted per step, and one 100-episode held-out eval at
  the end on the first seed's exact eval tasks.
- `production` (plan step 12): runs until a stop rule fires or the inference ceiling is reached,
  with the selection set and the capability check every 10 steps and the infra pause counted over
  10 steps. It starts at 2e-5 only if the second seed's stability checks passed (spec §5.2), and
  falls back to 1e-5 on a collapse.

`pinned()` gathers what spec §8 says is saved with every checkpoint, as results/<run>/config.json.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import bridge, client, controller, detectors, reward, tasks

REPO = Path(__file__).resolve().parents[1]

RUN_KINDS = ("second_seed", "production")
BASE_MODEL = "Qwen/Qwen3.8-27B"
ART_VERSION = "0.5.20"
PROJECT = "rlm-production"

# Learning rate (spec §5.2, Rishi 2026-10-10)
LR_SECOND_SEED = 1e-5
LR_PRODUCTION_HIGH = 2e-5
LR_FLOOR = 1e-5

# The train call (spec §5.1, decision 1c). The learning rate and the packed sequence length are passed separately.
TRAIN_ARGS = {"loss_fn": "cispo", "epsilon": 1.0, "epsilon_high": 4.0, "scale_rewards": False, "kl_penalty_coef": 0.0}

TICKET_MIX = {"tickets_all": 0.5, "tickets_customer": 0.5}  # the second seed's mix

# Production's eval sets and data (plan step 7). The capability path is relative to the data root (rlm_rl/data.py).
PRODUCTION_SELECTION = "selection-v1"
CAPABILITY_SET = "data:derived/capability/capability-v1.jsonl"
DATASET_MANIFEST = REPO / "manifests" / "dataset-manifest.json"

# Inference ceilings, $ (plan step 10a). Each is a hard stop: the run never starts a step or an eval it can't afford.
SECOND_SEED_CEILINGS = {"train": 14.0, "eval": 3.0}  # ~$12 planned: ~$10 of steps and ~$1.6 of eval *(mine)*
PRODUCTION_CEILINGS = {"train": 150.0, "eval": 25.0}  # the $25 reserve is never spent without Rishi (12a)

TRAINING_GPU_USD_PER_STEP = 0.16  # the 20-step run's measured rate (research §7.1); an estimate for the dashboard


@dataclass
class RunConfig:
    run: str
    kind: str
    learning_rate: float
    lr_rule: str  # "fixed": never changes; "fallback": drops to LR_FLOOR on a collapse (spec §5.2 step 3)
    mix: dict = field(default_factory=lambda: dict(TICKET_MIX))
    frozen_controller: bool = False
    base_model: str = BASE_MODEL
    project: str = PROJECT
    tasks_per_step: int = 8
    rollouts: int = 4
    packed_sequence_length: int | None = None  # None lets the trainer fit the batch, as the 20-step run did
    max_steps: int | None = None
    eval_every: int | None = None  # the selection set and the capability check
    eval_at_start: bool = False  # evaluate checkpoint 0 too, for the baselines the stop rules compare against
    selection: str | None = None  # eval set name (rlm_rl/evals.py)
    final_eval: str | None = None  # an eval run once, at max_steps
    capability_set: str | None = None  # path to the capability prompts (plan step 7 builds them)
    ceilings: dict = field(default_factory=dict)  # inference $ by split: "train" (rollouts and KL), "eval"
    keep_every: int = 5  # checkpoints kept besides the best and the latest (spec §8)
    kl_samples: int = 4  # rollouts re-scored under the base model each step (spec §5.3)
    first_step_estimate_usd: float = 1.0  # used by the budget check until a step has been measured
    eval_episode_estimate_usd: float = 0.03  # before any step has been measured (research §7.2: ~$0.016 at level 1)

    def __post_init__(self):
        if self.kind not in RUN_KINDS:
            raise ValueError(f"run kind must be one of {RUN_KINDS}")
        if self.lr_rule not in ("fixed", "fallback"):
            raise ValueError("lr_rule must be 'fixed' or 'fallback'")
        if not self.run or any(c in self.run for c in "/\\. ") or self.run != self.run.strip():
            raise ValueError(f"bad run name {self.run!r}")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RunConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def pause_window_steps(self) -> int:
        from .env import PAUSE_WINDOW_STEPS

        return PAUSE_WINDOW_STEPS[self.kind]

    def train_args(self, learning_rate: float) -> dict:
        return {"learning_rate": learning_rate, **TRAIN_ARGS, "packed_sequence_length": self.packed_sequence_length}


def second_seed(run: str = "second-seed-v1") -> RunConfig:
    """Plan step 10: the 20-step GRPO run repeated on environment v2, plus the 100-episode held-out eval."""
    return RunConfig(run=run, kind="second_seed", learning_rate=LR_SECOND_SEED, lr_rule="fixed", frozen_controller=True,
                     max_steps=20, final_eval="heldout-seed1", ceilings=dict(SECOND_SEED_CEILINGS))


def production(run: str, learning_rate: float, tasks_from: str = "7a") -> RunConfig:
    """Plan step 12. `learning_rate` is the second seed's call (spec §5.2): 2e-5 if its checks passed, else 1e-5.

    The mix is `tasks.production_mix()` (decisions 7a and 7c). `tasks_from="tickets"` keeps step 6's ticket
    stand-in (the ticket mix, selection-tickets-v1 and no capability file), for the loop's tests and dry runs
    that don't need the datasets.
    """
    if learning_rate not in (LR_PRODUCTION_HIGH, LR_FLOOR):
        raise ValueError(f"production starts at {LR_PRODUCTION_HIGH} or {LR_FLOOR} (spec §5.2), not {learning_rate}")
    if tasks_from == "tickets":
        mix, selection, capability_set = dict(TICKET_MIX), "selection-tickets-v1", None
    elif tasks_from == "7a":
        mix = tasks.production_mix()
        tasks.check_mix(mix)
        selection, capability_set = PRODUCTION_SELECTION, CAPABILITY_SET
    else:
        raise ValueError(f"tasks_from must be '7a' or 'tickets', not {tasks_from!r}")
    return RunConfig(run=run, kind="production", learning_rate=learning_rate,
                     lr_rule="fallback" if learning_rate > LR_FLOOR else "fixed", mix=mix, eval_every=10,
                     eval_at_start=True, selection=selection, capability_set=capability_set,
                     ceilings=dict(PRODUCTION_CEILINGS))


# ----------------------------------------------------------------------------- pinned values (spec §8)


def chat_template_sha256(base_model: str = BASE_MODEL) -> str | None:
    """The chat template's hash from the local Hugging Face cache, or None when it isn't cached here."""
    folder = "models--" + base_model.replace("/", "--")
    for config in sorted(Path.home().glob(f".cache/huggingface/hub/{folder}/snapshots/*/tokenizer_config.json")):
        try:
            template = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
        except (OSError, ValueError):
            continue
        if template:
            text = template if isinstance(template, str) else json.dumps(template, sort_keys=True)
            return hashlib.sha256(text.encode()).hexdigest()
    return None


def template_fingerprint(prompt_ids: list[int]) -> str:
    """A hash of how the server rendered the pinned system prompt and the generation prompt, from a root call's
    rebuilt prompt ids. It changes if the server's chat template or tokenizer does, and costs nothing."""
    starts = [i for i, t in enumerate(prompt_ids) if t == client.IM_START_ID]
    system = prompt_ids[:starts[1]] if len(starts) > 1 else prompt_ids
    return hashlib.sha256(json.dumps([system, prompt_ids[-7:]]).encode()).hexdigest()


def git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def art_version() -> str | None:
    try:
        return importlib.metadata.version("openpipe-art")
    except importlib.metadata.PackageNotFoundError:
        return None


def manifest(cfg: RunConfig) -> dict:
    """The dataset manifest (spec §8, plan step 7): generator versions, the mix, the eval sets, and the hash of
    manifests/dataset-manifest.json, which pins every source file by sha256."""
    try:
        dataset = hashlib.sha256(DATASET_MANIFEST.read_bytes()).hexdigest()
    except OSError:
        dataset = None
    data = {"generators": tasks.versions(cfg.mix), "mix": cfg.mix, "dataset_manifest_sha256": dataset,
            "evals": {"selection": cfg.selection, "final": cfg.final_eval, "capability": cfg.capability_set}}
    return {**data, "sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()}


def pinned(cfg: RunConfig, *, lr: dict, controller_state: dict, spend: dict, template: str | None = None,
           git: dict | None = None) -> dict[str, Any]:
    """Everything spec §8 says is saved with each checkpoint."""
    return {
        "art_version": art_version(),
        "art_version_pinned": ART_VERSION,
        "base_model": cfg.base_model,
        "chat_template_sha256": chat_template_sha256(cfg.base_model),
        "template_fingerprint": template,
        "bridge": {"api_version": bridge.API_VERSION, "prompt_version": bridge.PROMPT_VERSION,
                   "prompt_hash": bridge.PROMPT_HASH},
        "limits": {"max_turns": bridge.MAX_TURNS, "max_subcalls": bridge.MAX_SUBCALLS,
                   "root_max_tokens": bridge.ROOT_MAX_TOKENS, "output_max_chars": bridge.OUTPUT_MAX_CHARS,
                   "traceback_lines": bridge.TRACEBACK_LINES, "cell_timeout_s": bridge.CELL_TIMEOUT_S,
                   "subcall_max_chars": bridge.SUBCALL_MAX_CHARS, "subcall_max_tokens": bridge.SUBCALL_MAX_TOKENS,
                   "subcall_temperature": bridge.SUBCALL_TEMPERATURE, "root_temperature": client.ROOT_TEMPERATURE,
                   "thinking": client.ENABLE_THINKING},
        "train_call": cfg.train_args(lr["current"]),
        "lr": lr,
        "episodes_per_step": {"tasks": cfg.tasks_per_step, "rollouts": cfg.rollouts},
        "reward": reward.settings(),
        "detectors": detectors.settings(),
        "controller": {"settings": controller.settings(), "state": controller_state},
        "generators": tasks.versions(cfg.mix),
        "manifest": manifest(cfg),
        "limiter": {"max_in_flight": client.MAX_IN_FLIGHT, "per_model": client.MAX_IN_FLIGHT_PER_MODEL,
                    "retry": dataclasses.asdict(client.RetryPolicy())},
        "git": git if git is not None else git_state(),
        "spend": spend,
        "run": cfg.as_dict(),
    }
