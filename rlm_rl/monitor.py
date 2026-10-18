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

  With root thinking (production, decision 11b) a turn's sampled tokens include its thinking, which
  the next turn's prompt drops, so the whole conversation no longer lines up. Each re-scored
  rollout then re-scores `KL_TURNS_PER_THINKING_EPISODE` of its turns, picked at random, one
  request each: the conversation up to that turn, with the turn itself as the assistant's
  unfinished reply, its thinking in the message's `reasoning_content`, under the same thinking setting. That
  renders exactly the prompt the turn was sampled from plus its sampled tokens, except the final
  end-of-turn token, which isn't scored. Two turns keep the cost near one whole-conversation request.
"""

from __future__ import annotations

import asyncio
import math
import random
import statistics
from typing import Any

from . import bridge, detectors, editmetrics
from .client import IM_END_ID, ModelClient, RequestFailed, Usage
from .env import Episode, StepFilter

KL_TURNS_PER_THINKING_EPISODE = 2  # turns re-scored per rollout when the root thinks *(mine)*

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


def align_turn(choice: Any, prompt_ids: list[int], base_logprobs: list[float | None]) -> list[tuple[float, float]] | None:
    """Pair one sampled turn's policy logprobs with its base logprobs, from a re-score of that turn alone (thinking
    mode). The re-score ends where the reply's text does, so a final end-of-turn token isn't paired. None if the
    tokens don't line up."""
    extra = choice.model_extra or {}
    p_ids, c_ids = extra.get("prompt_token_ids") or [], extra.get("token_ids") or []
    content = getattr(getattr(choice, "logprobs", None), "content", None) or ()
    expected = c_ids[:-1] if c_ids and c_ids[-1] == IM_END_ID else c_ids
    start, end = len(p_ids), len(p_ids) + len(expected)
    if (not expected or len(content) != len(c_ids) or end > len(prompt_ids) or prompt_ids[start:end] != expected
            or prompt_ids[1:start] != p_ids[1:start]):
        return None
    return [(float(token.logprob), base_logprobs[start + i]) for i, token in enumerate(content[:len(expected)])
            if token.logprob is not None and base_logprobs[start + i] is not None]


def thinking_turn_messages(ep: Episode, k: int) -> list[dict]:
    """The conversation up to turn k, with turn k as the assistant's unfinished reply and its thinking in the message's
    `reasoning_content`, which the template renders as the think block the model sampled.

    W&B's template doesn't parse <think> tags inside the content: it adds an empty think block before them, so in
    prod-v1's first step every re-scored turn misaligned. A probe on 2026-10-18 lined up token for token with the
    field (the template also evens out the newlines around the block)."""
    turn = ep.turns[k]
    return ep.messages[:2 + 2 * k] + [{"role": "assistant", "content": turn.reply,
                                       "reasoning_content": turn.reasoning or ""}]


async def kl_to_base(client: ModelClient, episodes: list[Episode], *, base_model: str, n: int,
                     rng: random.Random) -> dict[str, Any]:
    """KL to base from `n` re-scored rollouts (spec §5.3). Never raises: failures are counted."""
    picked = rng.sample(episodes, min(n, len(episodes))) if episodes else []
    usage_before = client.spend["kl"].cost_usd()
    thinking = client.root_thinking
    jobs = []  # (episode, turn index or None for the whole conversation)
    for ep in picked:
        if thinking:
            # a turn cut off at the token cap never closed its think block, so it can't be re-rendered as sampled
            whole = [k for k in range(min(len(ep.choices), len(ep.turns)))
                     if getattr(ep.choices[k], "finish_reason", None) != "length"]
            jobs += [(ep, k) for k in sorted(rng.sample(whole, min(KL_TURNS_PER_THINKING_EPISODE, len(whole))))]
        else:
            jobs.append((ep, None))

    async def one(ep: Episode, k: int | None):
        try:
            if k is None:
                return ep, k, await client.prompt_logprobs(ep.messages, model=base_model)
            return ep, k, await client.prompt_logprobs(thinking_turn_messages(ep, k), model=base_model,
                                                       thinking=thinking, continue_final=True)
        except RequestFailed as exc:
            return ep, k, exc

    results = await asyncio.gather(*(one(ep, k) for ep, k in jobs))
    pairs, misaligned, failed = [], 0, 0
    for ep, k, result in results:
        if isinstance(result, Exception):
            failed += 1
            continue
        if k is None:
            got, bad = align(ep, *result)
        else:
            got = align_turn(ep.choices[k], *result)
            got, bad = (got, 0) if got is not None else ([], 1)
        pairs.extend(got)
        misaligned += bad
    diffs = [p - b for p, b in pairs]
    # k3 = E[exp(r) - 1 - r] with r = base - policy: never negative and lower variance than k1 (logged beside it)
    k3 = [math.exp(-d) - 1 + d for d in diffs]
    # each model's mean surprise (-logprob) on the same sampled tokens, for the matched collapse check (stoprules)
    nll = {"policy_nll": _round(_mean([-p for p, _ in pairs])), "base_nll": _round(_mean([-b for _, b in pairs]))}
    return {"kl": _round(_mean(diffs)), "kl_k3": _round(_mean(k3)), **nll, "tokens": len(diffs), "rollouts": len(picked),
            "requests": len(jobs), "failed": failed, "misaligned_turns": misaligned,
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
        fam = families.setdefault(ep.family, {"episodes": 0, "scored": 0, "correct": 0, "rewards": [], "levels": set(),
                                              "subcalls": 0, "delegated": 0, "correct_delegated": 0})
        fam["episodes"] += 1
        fam["levels"].add(ep.level)
        if ep.infra_error is None:
            fam["scored"] += 1
            fam["correct"] += bool(ep.correct)
            fam["rewards"].append(ep.reward)
            used = int(getattr(ep, "subcalls_used", 0) or 0)
            fam["subcalls"] += used
            fam["delegated"] += used > 0
            fam["correct_delegated"] += bool(ep.correct) and used > 0
    # tasks solved without sub-calls (the solo retirement rule, tasks.Retirement): at least half the group right and
    # none of the right answers delegated
    solo: dict[str, list[int]] = {}
    for group in groups:
        clean_group = [ep for ep in group if ep.infra_error is None]
        if not clean_group:
            continue
        right = [ep for ep in clean_group if ep.correct]
        row = solo.setdefault(clean_group[0].family, [0, 0])
        row[1] += 1
        row[0] += (2 * len(right) >= len(clean_group) and bool(right)
                   and not any(getattr(ep, "subcalls_used", 0) for ep in right))
    ctl = controller.summary() if controller is not None else {}
    by_family = {}
    for name in sorted(set(families) | set(ctl)):
        fam = families.get(name, {"episodes": 0, "scored": 0, "correct": 0, "rewards": [], "levels": set(),
                                  "subcalls": 0, "delegated": 0, "correct_delegated": 0})
        state = ctl.get(name, {})
        by_family[name] = {"level": state.get("level"), "levels_drawn": sorted(fam["levels"]),
                           "episodes": fam["episodes"], "scored": fam["scored"],
                           # delegation (step-12 task shift): sub-calls per scored episode, the share of episodes that
                           # made any, and the share of right answers that came from an episode that delegated
                           "subcalls_per_episode": _round(fam["subcalls"] / fam["scored"] if fam["scored"] else None),
                           "delegated_share": _round(fam["delegated"] / fam["scored"] if fam["scored"] else None),
                           "correct_delegated_share": _round(fam["correct_delegated"] / fam["correct"]
                                                             if fam["correct"] else None),
                           "solo_task_share": _round(solo[name][0] / solo[name][1] if solo.get(name, [0, 0])[1]
                                                     else None),
                           "pass_rate": _round(fam["correct"] / fam["scored"] if fam["scored"] else None),
                           "mean_reward": _round(_mean(fam["rewards"])),
                           "window_pass_rate": _round(state.get("pass_rate")), "saturated": state.get("saturated")}

    ent, ent_tokens = entropy(clean)
    lengths = [_transcript_tokens(ep) for ep in clean]
    output_truncations = [sum(1 for t in ep.turns if t.cell and t.cell.get("stdout_chars", 0)
                              > (ep.limits or {}).get("output_max_chars", bridge.OUTPUT_MAX_CHARS))
                          for ep in clean]
    usage = Usage()
    for ep in eps:
        for key, value in ep.usage.__dict__.items():
            setattr(usage, key, getattr(usage, key) + value)
    train = {ours: _round((train_metrics or {}).get(theirs)) for theirs, ours in TRAIN_METRIC_KEYS.items()}
    edit = editmetrics.step_block(eps)  # spec §12.9; absent when the step drew no edit task

    return {
        **({"edit": edit} if edit is not None else {}),
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
                   "hacked_contained": step_filter.hacked_contained,
                   "hacks": step_filter.hacks or dict.fromkeys(detectors.RULES, 0)},
        "policy": {"entropy": _round(ent), "entropy_tokens": ent_tokens, "reward_std": _round(reward_std(groups)),
                   "kl_to_base": (kl or {}).get("kl"), "kl_k3": (kl or {}).get("kl_k3"), "kl": kl or {}, **train},
        "shape": {"subcalls": m("subcalls"), "budget_hit": m("budget_hit"), "turns": m("turns"),
                  "first_turn_submit": m("first_turn_submit"), "no_code_turns": m("no_code_turns"),
                  "broken_replies": m("broken_replies"), "format_penalty": m("format_penalty"),
                  "resampled_replies": m("resampled_replies"),
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
