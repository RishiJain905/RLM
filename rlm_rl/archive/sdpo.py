"""SDPO (Self-Distillation Policy Optimization, arXiv:2601.20802) on W&B serverless ART.

The serverless training API only accepts a scalar reward per trajectory, so there
is no field for per-token advantages. We get SDPO's dense advantages in anyway by
reusing two things ART's loss already does:

1. The "old" logprobs used in the loss are the per-token logprobs we send on each
   Choice (precalculate_logprobs defaults to False).
2. With `kimi_k2_tau`, the loss sets
       A_t = A_grpo - tau * (logp_new(y_t) - logp_old(y_t)).

If we replace logp_old with the self-teacher's logprob logp(y_t | x, feedback, y_<t)
and train on-policy (so logp_new is the student's logp(y_t | x, y_<t)), this becomes
       A_t = A_grpo + tau * log(pi_teacher(y_t) / pi_student(y_t)),
which is the paper's hybrid SDPO+GRPO advantage with tau = (1 - lambda) / lambda.
Setting epsilon = epsilon_high = 0 pins the clipped importance weight to 1 so the
teacher logprobs do not leak into the IS ratio.

The teacher is the current checkpoint itself, scored with vLLM `prompt_logprobs`
on the chat endpoint, so no extra generation is needed.
"""

import asyncio
import copy
import random
from dataclasses import dataclass, field

import art
from openai import AsyncOpenAI

from .rollout import chat_with_retry
from .tasks import SYSTEM_PROMPT


def train_kwargs(tau: float) -> dict:
    """Extra ServerlessBackend.train kwargs that turn injected logprobs into SDPO advantages."""
    return {"kimi_k2_tau": tau, "epsilon": 0.0, "epsilon_high": 0.0}


def teacher_messages(question: str, solution: str) -> list[dict]:
    """Self-teacher context: the question plus a correct sibling solution (paper Table 2)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{question}\n\nCorrect solution:\n\n{solution}\n\n"
                "Correctly solve the original question."
            ),
        },
    ]


async def teacher_logprobs(
    client: AsyncOpenAI,
    model_name: str,
    context: list[dict],
    response_text: str,
    response_token_ids: list[int],
) -> list[float] | None:
    """Score `response_token_ids` under the teacher context with the policy checkpoint.

    Returns one logprob per response token, or None if the re-rendered response
    does not tokenize to the same ids (then the trajectory keeps plain GRPO).
    """
    response = await chat_with_retry(
        client,
        model=model_name,
        messages=[*context, {"role": "assistant", "content": response_text}],
        max_tokens=1,
        temperature=1.0,
        extra_body={
            "prompt_logprobs": 0,
            "add_generation_prompt": False,
            "continue_final_message": True,
            "chat_template_kwargs": {"preserve_thinking": True},
        },
    )
    # With prompt_logprobs=0 each entry holds exactly the actual prompt token.
    prompt_lps = (response.model_extra or {}).get("prompt_logprobs") or []
    n = len(response_token_ids)
    tail = prompt_lps[-n:] if n else []
    if len(tail) != n or any(not e or str(tid) not in e for tid, e in zip(response_token_ids, tail)):
        return None
    return [e[str(tid)]["logprob"] for tid, e in zip(response_token_ids, tail)]


@dataclass
class SDPOStats:
    rescored: int = 0
    skipped_no_solution: int = 0
    skipped_misaligned: int = 0
    log_ratios: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        lr = self.log_ratios
        return {
            "rescored": self.rescored,
            "skipped_no_solution": self.skipped_no_solution,
            "skipped_misaligned": self.skipped_misaligned,
            "mean_token_log_ratio": sum(lr) / len(lr) if lr else 0.0,
            "min_token_log_ratio": min(lr) if lr else 0.0,
        }


def _response_ids(choice) -> list[int]:
    # ART requests return_tokens_as_token_ids, so sampled tokens read "token_id:<id>".
    ids = [int(t.token.removeprefix("token_id:")) for t in choice.logprobs.content]
    # The sampled EOS token is not part of the re-rendered assistant message.
    if choice.finish_reason == "stop" and ids:
        ids = ids[:-1]
    return ids


async def inject_teacher_logprobs(
    model: art.TrainableModel,
    step: int,
    groups: list[art.TrajectoryGroup],
    seed: int = 0,
) -> tuple[list[art.TrajectoryGroup], SDPOStats]:
    """Return copies of `groups` whose Choice logprobs are self-teacher logprobs.

    A trajectory is re-scored when another rollout in its group got reward 1;
    that sibling's response becomes the teacher's reference solution. All other
    tokens keep their sampled logprob, which gives them zero SDPO advantage.
    """
    rng = random.Random(seed)
    groups = copy.deepcopy(groups)
    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)
    model_name = model.get_inference_name(step=step)
    stats = SDPOStats()
    jobs = []

    for group in groups:
        for i, traj in enumerate(group.trajectories):
            choice = traj.messages_and_choices[-1]
            siblings = [
                t for j, t in enumerate(group.trajectories) if j != i and t.reward >= 1.0
            ]
            if not siblings:
                stats.skipped_no_solution += 1
                continue
            solution = rng.choice(siblings).messages_and_choices[-1].message.content
            context = teacher_messages(traj.metadata["question"], solution)
            jobs.append((choice, context))

    async def rescore(choice, context):
        ids = _response_ids(choice)
        lps = await teacher_logprobs(client, model_name, context, choice.message.content or "", ids)
        if lps is None:
            stats.skipped_misaligned += 1
            return
        for token, teacher_lp in zip(choice.logprobs.content, lps):
            stats.log_ratios.append(teacher_lp - token.logprob)
            token.logprob = teacher_lp
        stats.rescored += 1

    await asyncio.gather(*(rescore(c, ctx) for c, ctx in jobs))
    await client.close()
    return groups, stats
