"""Single-turn rollouts and evals against an ART serverless model checkpoint."""

import asyncio
import functools
import json
from pathlib import Path

import art
import openai
from huggingface_hub import hf_hub_download

from .tasks import Task, messages_for, score

MAX_TOKENS = 1536

# W&B inference enforces a per-user, per-model concurrency limit (HTTP 429).
MAX_CONCURRENT_REQUESTS = 8
_limiter = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def chat_with_retry(client, retries: int = 6, **kwargs):
    """chat.completions.create under the shared concurrency cap, backing off on 429s."""
    for attempt in range(retries):
        try:
            async with _limiter:
                return await client.chat.completions.create(**kwargs)
        except openai.RateLimitError:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** (attempt + 1))


@functools.cache
def _first_prompt_token_id(base_model: str) -> int:
    """Id of <|im_start|>, the first token of every Qwen chat prompt (vLLM reports no logprob for it)."""
    config = json.loads(Path(hf_hub_download(base_model, "tokenizer_config.json")).read_text("utf-8"))
    for token_id, token in config["added_tokens_decoder"].items():
        if token["content"] == "<|im_start|>":
            return int(token_id)
    raise ValueError(f"{base_model} has no <|im_start|> token")


def attach_token_ids(choice, prompt_logprobs: list, base_model: str) -> None:
    """Attach the prompt/completion token ids ART's trainer requires.

    W&B's inference gateway drops vLLM's `return_token_ids` fields, but it does pass
    through `prompt_logprobs`, whose entries are keyed by the actual prompt token id,
    and ART's `return_tokens_as_token_ids` makes sampled tokens read "token_id:<id>".
    """
    prompt_ids = [_first_prompt_token_id(base_model)]
    prompt_ids += [int(next(iter(entry))) for entry in prompt_logprobs[1:]]
    completion_ids = [int(t.token.removeprefix("token_id:")) for t in choice.logprobs.content]
    choice.model_extra["prompt_token_ids"] = prompt_ids
    choice.model_extra["token_ids"] = completion_ids


async def rollout(model: art.TrainableModel, step: int, task: Task) -> art.Trajectory:
    """Sample one completion from checkpoint `step` and score it."""
    messages = messages_for(task)
    response = await chat_with_retry(
        model.openai_client(),
        model=model.get_inference_name(step=step),
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        logprobs=True,
        extra_body={"prompt_logprobs": 0},
    )
    choice = response.choices[0]
    attach_token_ids(choice, response.model_extra["prompt_logprobs"], model.base_model)
    reward = score(task, choice.message.content)
    return art.Trajectory(
        messages_and_choices=[*messages, choice],
        reward=reward,
        metrics={"correct": reward, "truncated": float(choice.finish_reason == "length")},
        metadata={"question": task.question, "answer": str(task.answer)},
    )


async def gather_groups(
    model: art.TrainableModel, step: int, tasks: list[Task], rollouts_per_task: int
) -> list[art.TrajectoryGroup]:
    return await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(rollout(model, step, task) for _ in range(rollouts_per_task))
            for task in tasks
        ),
        pbar_desc=f"rollouts@step{step}",
    )


async def evaluate(
    model: art.TrainableModel, step: int, tasks: list[Task], samples_per_task: int = 2
) -> float:
    """Mean accuracy of checkpoint `step` over `tasks`."""
    trajectories = await asyncio.gather(
        *(rollout(model, step, task) for task in tasks for _ in range(samples_per_task))
    )
    return sum(t.reward for t in trajectories) / len(trajectories)
