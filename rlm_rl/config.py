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

from . import bridge, client, controller, detectors, repl_tools, reward, tasks

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

# Production's eval sets and data (plan step 7, spec v2 §12.10). The capability path is relative to the data root
# (rlm_rl/data.py).
PRODUCTION_SELECTION = "selection-v3"  # selection-v2 with its ticket tasks replaced (Rishi, 2026-10-17; evals.py)
CAPABILITY_SET = "data:derived/capability/capability-v1.jsonl"
DATASET_MANIFEST = REPO / "manifests" / "dataset-manifest.json"
EDIT_MANIFEST = "edit/manifest.json"  # the edit task data's manifest, on the data drive (scripts/build_edit_data.py)

# Inference ceilings, $ (plan step 10a, spec v2 §12.11). Each is a hard stop: the run never starts a step or an eval it
# can't afford.
SECOND_SEED_CEILINGS = {"train": 14.0, "eval": 3.0}  # ~$12 planned: ~$10 of steps and ~$1.6 of eval *(mine)*
# Production, from W&B's billing page before the run (Rishi, 2026-10-17: $191.01 left): $146 of steps and $34 of evals,
# so ~$11 stays unspent: the $10 reserve, never spent without Rishi (12a), and ~$1 for the ~1% the run's own spend
# figures have differed from the bill (step 10, step 11's rerun). Evals get ~23% of steps' share: at step 11's prices
# (~$1.5 a step, ~$3 for the selection set and the capability check every 10 steps) both run out near step 100.
PRODUCTION_CEILINGS = {"train": 146.0, "eval": 34.0}  # spec §12.11 planned $157 and $30 of a larger pot

# Root thinking for production (Rishi, 2026-10-17, decision 11b): the base model with its root thinking at `medium` beat
# thinking off by 17 points in the step 11 baselines, at ~1.25x the inference cost a step. Sub-calls stay off.
PRODUCTION_ROOT_THINKING = "medium"
# Broken replies drawn again in training rollouts (2026-10-18): once one is in the conversation the model copies it,
# and one prod-v1 edit episode spent 23 of its 30 turns that way. Two more draws a turn; evals keep one draw.
RESAMPLE_BROKEN = 2

# The step-12 task shift (documentation/step-12-multistep-plan.md, Rishi 2026-10-18): at checkpoint 20 prod-v1 moves to
# families built to need sub-calls, prompt v4 (the orchestrator prompt and the bigger sub-call budget), 6 tasks x 6
# rollouts (more mixed groups from about the same episodes) and selection-v4. Applied at a resume with
# `train.py resume prod-v1 --task-shift v4`; the selection set then re-scores the checkpoint it resumes from.
TASK_SHIFTS = {
    # The pilot (2026-10-18): only context search rewarded delegating (0.34 vs 0.11 at L1, 0.35 vs 0 at L3, ~20
    # sub-calls an episode); research stayed grep-solvable and coding rarely delegated. So context search leads the
    # mix (40, from the plan's 20) and the bonus nudges the rest.
    "v4": {"mix": {"aggregation": 0.40, "research": 0.20, "edit_spec": 0.25, "edit_bugs": 0.15},
           "prompt_version": "v4", "tasks_per_step": 6, "rollouts": 6, "selection": "selection-v4",
           "delegation_bonus": 0.2, "delegation_bonus_every": 1, "delegation_bonus_until": 50},
}

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
    # The selection set by spend instead of every eval_every steps (prod-v2, Rishi 2026-10-19: two rounds fit the eval
    # budget): once as the train spend passes each of these shares of its ceiling, and on the last checkpoint when the
    # train ceiling stops the run if `selection_at_end`. eval_every then times only the capability check.
    selection_at_spend: list | None = None
    selection_at_end: bool = False
    selection: str | None = None  # eval set name (rlm_rl/evals.py)
    final_eval: str | None = None  # an eval run once, at max_steps
    capability_set: str | None = None  # path to the capability prompts (plan step 7 builds them)
    ceilings: dict = field(default_factory=dict)  # inference $ by split: "train" (rollouts and KL), "eval"
    keep_every: int = 5  # checkpoints kept besides the best and the latest (spec §8)
    kl_samples: int = 4  # rollouts re-scored under the base model each step (spec §5.3)
    first_step_estimate_usd: float = 1.0  # used by the budget check until a step has been measured
    eval_episode_estimate_usd: float = 0.03  # before any step has been measured (research §7.2: ~$0.016 at level 1)
    prompt_version: str = bridge.PROMPT_VERSION  # v3 (step 11 fixes on spec §12.2's v2); the second seed ran v1
    root_thinking: str | None = None  # client.ROOT_THINKING_EFFORTS or None (off); production thinks at medium
    format_penalty: float = 0.0  # reward off per broken reply in training rollouts (reward.FORMAT_PENALTY); 0 is off
    mask_broken: bool = False  # broken replies take no gradient (env.to_trajectory); production's fix since 2026-10-18
    resample_broken: int = 0  # times a training rollout draws a broken reply again before it enters the conversation
    # A correctness-gated delegation bonus (step-12 plan step 4; 0 is off): on every `delegation_bonus_every`-th step
    # up to `delegation_bonus_until`, a right answer from an episode that really delegated (training.delegated) gets up
    # to this much more reward, fading linearly to 0 at `delegation_bonus_until` (Parallel-R1 arXiv 2509.07980, RAO
    # arXiv 2605.06639, Kimi K2.5 PARL). The task shift turns it on: the pilot found delegating and solo rollouts
    # equally right, so without it GRPO has no reason to prefer delegating (Rishi, 2026-10-18: "reward answers that
    # used sub-calls and got it right").
    delegation_bonus: float = 0.0
    delegation_bonus_every: int = 5
    delegation_bonus_until: int = 50

    def __post_init__(self):
        if self.kind not in RUN_KINDS:
            raise ValueError(f"run kind must be one of {RUN_KINDS}")
        if self.lr_rule not in ("fixed", "fallback"):
            raise ValueError("lr_rule must be 'fixed' or 'fallback'")
        if not self.run or any(c in self.run for c in "/\\. ") or self.run != self.run.strip():
            raise ValueError(f"bad run name {self.run!r}")
        if self.prompt_version not in bridge.PROMPT_VERSIONS:
            raise ValueError(f"unknown prompt version {self.prompt_version!r}")
        if self.prompt_version == "v1" and any(bridge.is_edit(f) for f in self.mix):
            raise ValueError("edit families need prompt v2 (spec §12.2)")
        if self.root_thinking is not None and self.root_thinking not in client.ROOT_THINKING_EFFORTS:
            raise ValueError(f"root thinking must be None or one of {client.ROOT_THINKING_EFFORTS}")
        if not 0.0 <= self.format_penalty <= 0.2:
            raise ValueError("format penalty must be between 0 and 0.2 a reply")
        if not 0 <= self.resample_broken <= 3:
            raise ValueError("resample_broken must be between 0 and 3 draws a turn")
        if not 0.0 <= self.delegation_bonus <= 0.3 or self.delegation_bonus_every < 1:
            raise ValueError("delegation_bonus must be between 0 and 0.3, every 1 or more steps")
        if self.selection_at_spend is not None:
            self.selection_at_spend = sorted(float(f) for f in self.selection_at_spend)
            if not all(0.0 < f < 1.0 for f in self.selection_at_spend):
                raise ValueError("selection_at_spend shares must be between 0 and 1 of the train ceiling")
            if "train" not in self.ceilings:
                raise ValueError("selection_at_spend needs a train ceiling to take shares of")

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
                     max_steps=20, final_eval="heldout-seed1", ceilings=dict(SECOND_SEED_CEILINGS),
                     prompt_version="v1")  # decision E-g: it replicates the 20-step run, on prompt v1


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
                     ceilings=dict(PRODUCTION_CEILINGS), root_thinking=PRODUCTION_ROOT_THINKING, mask_broken=True,
                     resample_broken=RESAMPLE_BROKEN)


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
    edit = None
    if any(bridge.is_edit(f) for f in cfg.mix):
        try:
            from . import data as rdata

            edit = hashlib.sha256(rdata.path(EDIT_MANIFEST).read_bytes()).hexdigest()
        except (OSError, FileNotFoundError):
            edit = None
    data = {"generators": tasks.versions(cfg.mix), "mix": cfg.mix, "dataset_manifest_sha256": dataset,
            "edit_manifest_sha256": edit,
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
        "bridge": {"api_version": bridge.API_VERSION, "prompt_version": cfg.prompt_version,
                   "prompt_hash": bridge.PROMPT_HASHES[cfg.prompt_version],
                   "tools_version": repl_tools.TOOLS_VERSION},  # the edit REPL's coding tools (spec §12.1)
        "edit_limits": bridge.limits("edit_real", cfg.prompt_version).as_dict(),  # v3: 30 turns
        "limits": {"max_turns": bridge.MAX_TURNS, "max_subcalls": bridge.MAX_SUBCALLS,
                   "root_max_tokens": bridge.ROOT_MAX_TOKENS,
                   "output_max_chars": bridge.limits("read", cfg.prompt_version).output_max_chars,
                   "traceback_lines": bridge.TRACEBACK_LINES, "cell_timeout_s": bridge.CELL_TIMEOUT_S,
                   "subcall_max_chars": bridge.SUBCALL_MAX_CHARS, "subcall_max_tokens": bridge.SUBCALL_MAX_TOKENS,
                   "subcall_temperature": bridge.SUBCALL_TEMPERATURE, "root_temperature": client.ROOT_TEMPERATURE,
                   "thinking": cfg.root_thinking or False,  # the root's reasoning effort, or False for off
                   "subcall_thinking": bridge.SUBCALL_ENABLE_THINKING},
        "train_call": cfg.train_args(lr["current"]),
        "lr": lr,
        "episodes_per_step": {"tasks": cfg.tasks_per_step, "rollouts": cfg.rollouts},
        "reward": reward.settings(),
        "detectors": detectors.settings(),
        "controller": {"settings": controller.settings(), "state": controller_state},
        "generators": tasks.versions(cfg.mix),
        "manifest": manifest(cfg),
        "limiter": client.settings(),  # caps, the sub-call gap and the retry policy (client.py, step 10 §6)
        "git": git if git is not None else git_state(),
        "spend": spend,
        "run": cfg.as_dict(),
    }
