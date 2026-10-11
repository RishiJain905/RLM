"""Per-step metrics for W&B, results/<run>.json and the dashboard (spec §9, plan step 6).

Everything here is computed from what a step already has: its groups of episodes, what
filter_step() did to them, the controller, and the train call's metrics. Two training signals
need more than that:

- **Entropy** (spec §5.2). ART's serverless metrics don't report it, so it is estimated as the
  mean of -logprob over every sampled root token of the step's scored episodes. The root calls
  already ask for those logprobs, so it costs nothing. At temperature 1.0 this is an unbiased
  estimate of the policy's per-token entropy on its own samples.
- **KL to base** (spec §5.3). Each step, `kl_samples` scored rollouts are re-scored under the base
  model with `prompt_logprobs`, one request per rollout carrying its whole conversation. Qwen3.8's
  template renders every assistant turn with the same empty think block the policy generated
  after, so each turn's prompt plus its sampled tokens is a prefix of that conversation, and the
  policy's and the base model's logprobs line up token for token. The logged KL is the mean of
  (policy logprob - base logprob) over those tokens. A turn whose tokens don't line up is skipped
  and counted. It is logged, never put in the loss.
"""

from __future__ import annotations

import asyncio
import math
import random
import statistics
from typing import Any

from . import bridge, detectors
from .client import ModelClient, RequestFailed, Usage
from .env import Episode, StepFilter

TRAIN_METRIC_KEYS = {  # ART's serverless train metrics -> our names (spec §9)
    "loss/grad_norm": "grad_norm",
    "loss/probs_corr": "probs_corr",
    "loss/clipped_token_fraction": "clip_fraction",
    "loss/importance_ratio_mean": "importance_ratio_mean",
    "loss/importance_ratio_p95": "importance_ratio_p95",
    "loss/importance_ratio_p99": "importance_ratio_p99",
    "loss/train": "loss",
}


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _round(value, digits=6):
    return None if value is None else round(value, digits)


def scored(groups: list[list[Episode]]) -> list[Episode]:
    return [ep for group in groups for ep in group if ep.infra_error is None]


# ----------------------------------------------------------------------------- policy signals


def sampled_logprobs(ep: Episode) -> list[float]:
    """The logprob of every root token the policy sampled in this episode."""
    out = []
    for choice in ep.choices:
        content = getattr(getattr(choice, "logprobs", None), "content", None) or ()
        out.extend(float(t.logprob) for t in content if t.logprob is not None)
    return out


def entropy(episodes: list[Episode]) -> tuple[float | None, int]:
    """Mean -logprob over every sampled root token, and how many tokens that was."""
    total, n = 0.0, 0
    for ep in episodes:
        lps = sampled_logprobs(ep)
        total -= sum(lps)
        n += len(lps)
    return (total / n if n else None), n


def reward_std(groups: list[list[Episode]]) -> float | None:
    """Mean within-group reward std (population) over groups with at least 2 scored episodes: the Echo Trap signal."""
    stds = []
    for group in groups:
        rewards = [ep.reward for ep in group if ep.infra_error is None]
        if len(rewards) >= 2:
            stds.append(statistics.pstdev(rewards))
    return _mean(stds)


def align(ep: Episode, prompt_ids: list[int], base_logprobs: list[float | None]) -> tuple[list[tuple[float, float]], int]:
    """Pair each sampled root token's policy logprob with its base logprob in the re-scored conversation.

    Returns the pairs and the number of turns that didn't line up (skipped).
    """
    pairs, misaligned = [], 0
    for choice in ep.choices:
        extra = choice.model_extra or {}
        p_ids, c_ids = extra.get("prompt_token_ids") or [], extra.get("token_ids") or []
        content = getattr(getattr(choice, "logprobs", None), "content", None) or ()
        start, end = len(p_ids), len(p_ids) + len(c_ids)
        if (not c_ids or len(content) != len(c_ids) or end > len(prompt_ids) or prompt_ids[start:end] != c_ids
                or prompt_ids[1:start] != p_ids[1:start]):
            misaligned += 1
            continue
        for i, token in enumerate(content):
            base = base_logprobs[start + i]
            if token.logprob is not None and base is not None:
                pairs.append((float(token.logprob), base))
    return pairs, misaligned


async def kl_to_base(client: ModelClient, episodes: list[Episode], *, base_model: str, n: int,
                     rng: random.Random) -> dict[str, Any]:
    """KL to base from `n` re-scored rollouts (spec §5.3). Never raises: failures are counted."""
    picked = rng.sample(episodes, min(n, len(episodes))) if episodes else []
    usage_before = client.spend["kl"].cost_usd()

    async def one(ep: Episode):
        try:
            return ep, await client.prompt_logprobs(ep.messages, model=base_model)
        except RequestFailed as exc:
            return ep, exc

    results = await asyncio.gather(*(one(ep) for ep in picked))
    pairs, misaligned, failed = [], 0, 0
    for ep, result in results:
        if isinstance(result, Exception):
            failed += 1
            continue
        got, bad = align(ep, *result)
        pairs.extend(got)
        misaligned += bad
    diffs = [p - b for p, b in pairs]
    # k3 = E[exp(r) - 1 - r] with r = base - policy: never negative and lower variance than k1 (logged beside it)
    k3 = [math.exp(-d) - 1 + d for d in diffs]
    return {"kl": _round(_mean(diffs)), "kl_k3": _round(_mean(k3)), "tokens": len(diffs), "rollouts": len(picked),
            "failed": failed, "misaligned_turns": misaligned,
            "cost_usd": round(client.spend["kl"].cost_usd() - usage_before, 6)}


# ----------------------------------------------------------------------------- the step's metrics


def _transcript_tokens(ep: Episode) -> int:
    """The root transcript's length in tokens: the last turn's prompt plus its sampled tokens."""
    if not ep.choices:
        return 0
    extra = ep.choices[-1].model_extra or {}
    return len(extra.get("prompt_token_ids") or ()) + len(extra.get("token_ids") or ())


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def pass_histogram(groups: list[list[Episode]]) -> dict[str, int]:
    """How many groups had each share of correct rollouts, in quarters (spec §9)."""
    hist = {f"{k}/4": 0 for k in range(5)}
    for group in groups:
        clean = [ep for ep in group if ep.infra_error is None]
        if clean:
            share = sum(bool(ep.correct) for ep in clean) / len(clean)
            hist[f"{round(share * 4)}/4"] += 1
    return hist


def step_metrics(groups: list[list[Episode]], step_filter: StepFilter, *, controller: Any = None,
                 moves: list | None = None, kl: dict | None = None, train_metrics: dict | None = None,
                 request_stats: dict | None = None) -> dict[str, Any]:
    """Every per-step number in spec §9 that the step's own data gives."""
    eps = [ep for group in groups for ep in group]
    clean = scored(groups)
    n_all, n = len(eps), len(clean)
    correct = sum(bool(ep.correct) for ep in clean)

    def m(name):
        return _round(_mean([ep.metrics()[name] for ep in clean]))

    families: dict[str, dict] = {}
    for ep in eps:
        fam = families.setdefault(ep.family, {"episodes": 0, "scored": 0, "correct": 0, "rewards": [], "levels": set()})
        fam["episodes"] += 1
        fam["levels"].add(ep.level)
        if ep.infra_error is None:
            fam["scored"] += 1
            fam["correct"] += bool(ep.correct)
            fam["rewards"].append(ep.reward)
    ctl = controller.summary() if controller is not None else {}
    by_family = {}
    for name in sorted(set(families) | set(ctl)):
        fam = families.get(name, {"episodes": 0, "scored": 0, "correct": 0, "rewards": [], "levels": set()})
        state = ctl.get(name, {})
        by_family[name] = {"level": state.get("level"), "levels_drawn": sorted(fam["levels"]),
                           "episodes": fam["episodes"], "scored": fam["scored"],
                           "pass_rate": _round(fam["correct"] / fam["scored"] if fam["scored"] else None),
                           "mean_reward": _round(_mean(fam["rewards"])),
                           "window_pass_rate": _round(state.get("pass_rate")), "saturated": state.get("saturated")}

    ent, ent_tokens = entropy(clean)
    lengths = [_transcript_tokens(ep) for ep in clean]
    output_truncations = [sum(1 for t in ep.turns if t.cell and t.cell.get("stdout_chars", 0) > bridge.OUTPUT_MAX_CHARS)
                          for ep in clean]
    usage = Usage()
    for ep in eps:
        for key, value in ep.usage.__dict__.items():
            setattr(usage, key, getattr(usage, key) + value)
    train = {ours: _round((train_metrics or {}).get(theirs)) for theirs, ours in TRAIN_METRIC_KEYS.items()}

    return {
        "pass": {"rate": _round(correct / n if n else None), "rate_with_infra": _round(correct / n_all if n_all else None),
                 "histogram": pass_histogram(groups), "groups": len(groups),
                 "trainable_groups": len(step_filter.groups), "all_zero_groups": step_filter.dropped_all_zero,
                 "all_one_groups": step_filter.dropped_all_one, "other_equal_groups": step_filter.dropped_equal_other,
                 "small_groups": step_filter.dropped_small},
        "families": by_family,
        "moves": [{"family": mv.family, "from": mv.from_level, "to": mv.to_level, "pass_rate": mv.pass_rate,
                   "applied": mv.applied} for mv in (moves or [])],
        "reward": {"mean": _round(_mean([ep.reward for ep in clean])), "exact": _round(correct / n if n else None), "partial": m("partial"),
                   "near_blind": m("near_blind"), "submitted": m("submitted"), "hacked": step_filter.hacked_episodes,
                   "hacks": step_filter.hacks or dict.fromkeys(detectors.RULES, 0)},
        "policy": {"entropy": _round(ent), "entropy_tokens": ent_tokens, "reward_std": _round(reward_std(groups)),
                   "kl_to_base": (kl or {}).get("kl"), "kl_k3": (kl or {}).get("kl_k3"), "kl": kl or {}, **train},
        "shape": {"subcalls": m("subcalls"), "budget_hit": m("budget_hit"), "turns": m("turns"),
                  "first_turn_submit": m("first_turn_submit"), "no_code_turns": m("no_code_turns"),
                  "repl_errors": m("repl_errors"), "timeouts": m("timeouts"),
                  "root_completion_tokens": m("root_completion_tokens"), "subcall_tokens": m("subcall_tokens"),
                  "truncated_turns": m("truncated_turns"),
                  "episodes_with_truncated_turn": _round(_mean([float(ep.metrics()["truncated_turns"] > 0)
                                                                for ep in clean])),
                  "output_truncations": _round(_mean(output_truncations)),
                  "root_transcript_tokens": {"mean": _round(_mean(lengths), 1), "p95": _percentile(lengths, 0.95),
                                             "max": max(lengths) if lengths else None},
                  "context_chars": _round(_mean([ep.context_chars for ep in eps]), 1),
                  "wall_s": m("wall_s")},
        "infra": {"episodes": n_all, "infra": step_filter.infra, "rate": _round(step_filter.infra_rate),
                  "kinds": step_filter.infra_kinds,
                  "requests": {k: v for k, v in (request_stats or {}).items() if k != "events"}},
        "tokens": usage.as_dict(),
    }


def flat(row: dict, prefix: str = "rlm") -> dict[str, float]:
    """Numeric leaves of a metrics dict as flat W&B keys, e.g. rlm/policy/entropy."""
    out = {}
    for key, value in row.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            out.update(flat(value, name))
        elif isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, (int, float)) and value is not None and math.isfinite(value):
            out[name] = float(value)
    return out
