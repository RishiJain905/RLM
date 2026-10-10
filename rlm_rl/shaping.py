"""Per-token advantage shaping for multi-turn RLM rollouts on W&B serverless ART.

Both self-distillation arms use the same teacher: the current checkpoint re-reading the
student's whole root conversation with privileged feedback added to the first user
message. They differ only in how the teacher's per-token log-ratio
`delta_t = log pi_teacher(y_t) - log pi_student(y_t)` becomes an advantage.

- SDPO hybrid (arXiv:2601.20802):  A_t = A_i + tau * delta_t
- RLSD (arXiv:2604.03128, Alg. 1): A_t = A_i * ((1 - lam) + lam * clip(exp(sign(A_i) * delta_t), 1 - eps_w, 1 + eps_w))

Both go through the same channel (see sdpo.py): ART trains with
`A_t = A_grpo - tau * (logp_new - logp_old)`, and we choose `logp_old` per token.
For SDPO we send the teacher logprob. For RLSD we send `logp_student + (A_t - A_i) / tau`.

Teacher context (privileged information), following documentation/rl-method-for-rlm.md:
a successful sibling's code turns with its submit lines removed, otherwise the
environment's feedback on the student's own failed attempt. Never the gold answer.
"""

import asyncio
import copy
import json
import math
import random
from dataclasses import dataclass, field

import art
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import Choice

from .rollout import _first_prompt_token_id, chat_with_retry

EOS_TOKEN_ID = 248046  # <|im_end|> for the Qwen3.6 / Qwen3.8 tokenizer


def _privileged_text(trajectory: art.Trajectory, siblings: list[art.Trajectory], rng: random.Random) -> str | None:
    winners = [t for t in siblings if t.reward >= 1.0]
    if winners:
        codes = json.loads(rng.choice(winners).metadata["codes"])
        code = "\n\n".join(
            "\n".join(line for line in c.splitlines() if "submit(" not in line) for c in codes
        ).strip()
        if code:
            return ("Hint from a successful attempt at this question. It used this code "
                    f"(final answer removed):\n```python\n{code}\n```")
    if trajectory.reward < 1.0 and trajectory.metadata.get("feedback"):
        return f"Feedback on a previous attempt at this question: {trajectory.metadata['feedback']}"
    return None


def _api_messages(trajectory: art.Trajectory, privileged: str) -> tuple[list[dict], list[Choice]]:
    messages, choices = [], []
    for item in trajectory.messages_and_choices:
        if isinstance(item, Choice):
            choices.append(item)
            messages.append({"role": "assistant", "content": item.message.content or ""})
        else:
            messages.append(dict(item))
    first_user = next(i for i, m in enumerate(messages) if m["role"] == "user")
    messages[first_user]["content"] += f"\n\n{privileged}"
    return messages, choices


def _choice_ids(choice: Choice) -> list[int]:
    ids = list(choice.model_extra["token_ids"])
    return ids[:-1] if ids and ids[-1] == EOS_TOKEN_ID else ids


async def teacher_logprobs(client, model_name: str, base_model: str, trajectory, privileged: str) -> list[list[float]] | None:
    """Teacher logprob of every root token, per choice (EOS excluded). None if alignment fails."""
    messages, choices = _api_messages(trajectory, privileged)
    response = await chat_with_retry(
        client,
        model=model_name,
        messages=messages,
        max_tokens=1,
        temperature=1.0,
        extra_body={
            "prompt_logprobs": 0,
            "add_generation_prompt": False,
            "continue_final_message": True,
            "chat_template_kwargs": {"preserve_thinking": True},
        },
    )
    entries = (response.model_extra or {}).get("prompt_logprobs") or []
    prompt_ids = [_first_prompt_token_id(base_model)] + [int(next(iter(e))) for e in entries[1:]]
    out, cursor = [], 0
    for choice in choices:
        needle = _choice_ids(choice)
        start = _find(prompt_ids, needle, cursor)
        if start is None:
            return None
        out.append([entries[start + k][str(tid)]["logprob"] for k, tid in enumerate(needle)])
        cursor = start + len(needle)
    return out


def _find(haystack: list[int], needle: list[int], start: int) -> int | None:
    if not needle:
        return start
    first, n = needle[0], len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i] == first and haystack[i : i + n] == needle:
            return i
    return None


@dataclass
class ShapingStats:
    rescored: int = 0
    no_feedback: int = 0
    misaligned: int = 0
    skipped_zero_advantage: int = 0
    deltas: list[float] = field(default_factory=list)
    shifts: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        d, s = self.deltas, self.shifts
        return {
            "rescored": self.rescored,
            "no_feedback": self.no_feedback,
            "misaligned": self.misaligned,
            "skipped_zero_advantage": self.skipped_zero_advantage,
            "mean_delta": sum(d) / len(d) if d else 0.0,
            "min_delta": min(d) if d else 0.0,
            "mean_abs_advantage_shift": sum(abs(x) for x in s) / len(s) if s else 0.0,
        }


async def shape_groups(
    model: art.TrainableModel,
    step: int,
    groups: list[art.TrajectoryGroup],
    method: str,
    *,
    tau: float,
    rlsd_lambda: float = 0.5,
    rlsd_eps: float = 0.2,
    seed: int = 0,
) -> tuple[list[art.TrajectoryGroup], ShapingStats]:
    """Return copies of `groups` with per-token logprobs rewritten for `method` ('sdpo' or 'rlsd')."""
    assert method in ("sdpo", "rlsd")
    rng = random.Random(seed)
    groups = copy.deepcopy(groups)
    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key, timeout=600)
    model_name = model.get_inference_name(step=step)
    stats = ShapingStats()
    jobs = []
    for group in groups:
        rewards = [t.reward for t in group.trajectories]
        mean_reward = sum(rewards) / len(rewards)
        for i, traj in enumerate(group.trajectories):
            advantage = traj.reward - mean_reward  # matches scale_rewards=False on the server
            if method == "rlsd" and abs(advantage) < 1e-9:
                stats.skipped_zero_advantage += 1
                continue
            privileged = _privileged_text(traj, group.trajectories[:i] + group.trajectories[i + 1 :], rng)
            if privileged is None:
                stats.no_feedback += 1
                continue
            jobs.append((traj, privileged, advantage))

    async def rescore(traj, privileged, advantage):
        lps = await teacher_logprobs(client, model_name, model.base_model, traj, privileged)
        if lps is None:
            stats.misaligned += 1
            return
        choices = [c for c in traj.messages_and_choices if isinstance(c, Choice)]
        for choice, teacher in zip(choices, lps):
            for token, t_lp in zip(choice.logprobs.content, teacher):
                delta = t_lp - token.logprob
                stats.deltas.append(delta)
                if method == "sdpo":
                    shift = tau * delta
                    token.logprob = t_lp
                else:
                    w = math.exp(math.copysign(1.0, advantage) * max(min(delta, 20.0), -20.0))
                    w = min(max(w, 1 - rlsd_eps), 1 + rlsd_eps)
                    shift = advantage * rlsd_lambda * (w - 1.0)
                    token.logprob = token.logprob + shift / tau
                stats.shifts.append(shift)
        stats.rescored += 1

    await asyncio.gather(*(rescore(*job) for job in jobs))
    await client.close()
    return groups, stats


def train_kwargs(method: str, tau: float) -> dict:
    """Identical loss settings for every arm; only the tau channel differs."""
    common = {"epsilon": 0.0, "epsilon_high": 0.0, "scale_rewards": False}
    if method == "grpo":
        return common
    return {**common, "kimi_k2_tau": tau}
