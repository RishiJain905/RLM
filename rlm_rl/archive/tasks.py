"""Toy verifiable tasks for smoke-testing RL pipelines.

Multi-digit multiplication with brief working allowed. Base models get it right
some of the time, which gives GRPO groups reward variance to learn from.
"""

import random
import re
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a careful calculator. Show brief working if you need it, then put "
    "the final answer on the last line as 'ANSWER: <number>'."
)


@dataclass
class Task:
    question: str
    answer: int


def make_tasks(n: int, digits: int = 4, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    lo, hi = 10 ** (digits - 1), 10**digits - 1
    tasks = []
    for _ in range(n):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        tasks.append(Task(question=f"What is {a} * {b}?", answer=a * b))
    return tasks


def messages_for(task: Task) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.question},
    ]


_ANSWER_RE = re.compile(r"ANSWER:\s*(-?[\d,]+)")


def parse_answer(text: str | None) -> int | None:
    matches = _ANSWER_RE.findall(text or "")
    if not matches:
        return None
    try:
        return int(matches[-1].replace(",", ""))
    except ValueError:
        return None


def score(task: Task, text: str | None) -> float:
    return 1.0 if parse_answer(text) == task.answer else 0.0
