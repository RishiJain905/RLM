"""Minimal RLM environment for RL: the root model writes Python against a REPL that
holds the long context, and can recurse through `llm_query` / `llm_batch`.

- Tasks are synthetic and verifiable: semantic counting over customer messages, and
  filtered aggregation over order records. Both need the REPL; the first rewards
  using sub-calls to classify text.
- Each episode gets its own REPL subprocess (`repl_worker.py`) with no secrets in its
  environment. Sub-calls come back to the host over a local HTTP bridge and are
  served by a fixed base-model checkpoint (same base model as the root).
- A rollout is the root conversation only. REPL output turns are masked by ART.
"""

import asyncio
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import art
import openai

from .rollout import attach_token_ids, chat_with_retry

MAX_TURNS = 6
MAX_SUBCALLS = 20
ROOT_MAX_TOKENS = 1536
SUBCALL_MAX_TOKENS = 512
SUBCALL_MAX_CHARS = 24_000
OUTPUT_MAX_CHARS = 2_000
CELL_TIMEOUT_S = 120
MAX_CONCURRENT_SUBCALLS = 6

SYSTEM_PROMPT = f"""You are a Recursive Language Model. You answer a question about a long text that you cannot see directly. The text is stored in a persistent Python REPL as the string variable `context`.

Each turn, write a short plan and then exactly one ```python code block. The REPL runs it and shows you what it prints (truncated to {OUTPUT_MAX_CHARS} characters). Variables persist between turns.

Available in the REPL:
- `context`: the full text (str)
- `llm_query(prompt: str) -> str`: ask a language model; it can read about 20,000 characters per call
- `llm_batch(prompts: list[str]) -> list[str]`: ask several prompts in parallel
- `submit(answer)`: give your final answer and end the episode

You have at most {MAX_TURNS} turns and {MAX_SUBCALLS} sub-calls in total. Look at the structure of `context` before processing it, decompose the work, use code for exact filtering and arithmetic, and use sub-calls for judgments that need reading comprehension. Call `submit()` once you are confident."""

# --------------------------------------------------------------------------- tasks

_CATEGORY_TEMPLATES = {
    "delivery": [
        "My package still hasn't arrived and it's been {n} days.",
        "The courier left my parcel at the wrong address again.",
        "Tracking says delivered but there was nothing at my door.",
        "The box showed up crushed and the {item} inside was broken.",
        "My order has been stuck at the sorting center for {n} days.",
        "Only half of my order arrived, the {item} is missing.",
        "The driver never came even though I waited home all day.",
    ],
    "billing": [
        "I was charged twice for the same {item}.",
        "There is a fee on my invoice that nobody explained.",
        "My refund for the {item} still hasn't reached my card after {n} days.",
        "Why did my subscription price go up this month?",
        "I cancelled but you billed me again anyway.",
        "The discount code was accepted but the total didn't change.",
    ],
    "technical": [
        "The app crashes every time I open my order history.",
        "The password reset email never arrives.",
        "The checkout button does nothing when I click it.",
        "Your tracking page just shows an error.",
        "I can't upload a photo for my {item} review, it times out.",
        "The website logs me out every {n} minutes.",
    ],
    "praise": [
        "Delivery was quick and the {item} arrived in perfect shape!",
        "The package came a day early, really impressed with the shipping.",
        "Support fixed my billing question in minutes, thank you.",
        "Love the {item}, already ordered a second one.",
        "The app update is great, checkout is much smoother now.",
        "No problems at all with my delivery this time, great job.",
    ],
    "other": [
        "Do you ship to Canada?",
        "What are your opening hours on weekends?",
        "Can I change the color of my {item} before it ships?",
        "Is there a student discount?",
        "Do you sell gift cards?",
        "Will the {item} come back in stock soon?",
    ],
}

_QUESTIONS = {
    "delivery": "How many messages report a problem with the delivery or shipping of an order (late, lost, misdelivered, damaged in transit, or incomplete)?",
    "billing": "How many messages report a problem with billing or payment (wrong charges, unexplained fees, missing refunds, price changes)?",
    "technical": "How many messages report a technical problem with the app or website?",
}

_NAMES = ["Maria Lopez", "Wei Chen", "Aisha Khan", "Tom Becker", "Priya Nair", "Lucas Silva",
          "Emma Brown", "Omar Haddad", "Sofia Rossi", "Kenji Sato", "Nina Ivanova", "Jack Wilson"]
_ITEMS = ["lamp", "kettle", "backpack", "headphones", "desk chair", "blender", "jacket", "monitor"]


@dataclass
class RLMTask:
    family: str
    question: str
    answer: float
    context: str


def _count_task(rng: random.Random) -> RLMTask:
    target = rng.choice(list(_QUESTIONS))
    n_lines = rng.randint(200, 320)
    lines, count = [], 0
    weights = {"delivery": 1, "billing": 1, "technical": 1, "praise": 1.2, "other": 1.2}
    for i in range(n_lines):
        cat = rng.choices(list(weights), weights=list(weights.values()))[0]
        text = rng.choice(_CATEGORY_TEMPLATES[cat]).format(n=rng.randint(2, 14), item=rng.choice(_ITEMS))
        lines.append(f"[msg {1000 + i}] {rng.choice(_NAMES)}: {text}")
        count += cat == target
    return RLMTask("count", _QUESTIONS[target] + " Answer with a single integer.", count, "\n".join(lines))


def _sum_task(rng: random.Random) -> RLMTask:
    who = rng.choice(_NAMES)
    n_lines = rng.randint(300, 450)
    lines, total = [], 0.0
    for i in range(n_lines):
        if rng.random() < 0.15:
            lines.append(f"[note] {rng.choice(['warehouse restock scheduled', 'price update pending', 'carrier delay reported', 'audit complete'])} ({rng.randint(1, 30)} Oct)")
            continue
        name = who if rng.random() < 0.12 else rng.choice(_NAMES)
        amount = round(rng.uniform(5, 250), 2)
        status = rng.choices(["shipped", "delivered", "refunded", "pending"], weights=[3, 3, 1.5, 1])[0]
        lines.append(f"[order {5000 + i}] customer={name} item={rng.choice(_ITEMS)} amount=${amount:.2f} status={status}")
        if name == who and status != "refunded":
            total += amount
    question = (f"What is the total amount in dollars of all orders by {who} that were NOT refunded? "
                "Answer with a number rounded to 2 decimal places.")
    return RLMTask("sum", question, round(total, 2), "\n".join(lines))


def make_rlm_tasks(n: int, seed: int) -> list[RLMTask]:
    rng = random.Random(seed)
    # Two counting tasks per aggregation task: counting is harder, so it keeps more groups mixed.
    return [(_sum_task if i % 3 == 2 else _count_task)(rng) for i in range(n)]


def task_prompt(task: RLMTask) -> str:
    lines = task.context.splitlines()
    preview = "\n".join(lines[:3])
    return (f"Question: {task.question}\n\n`context` holds {len(task.context):,} characters in {len(lines)} lines. "
            f"The first 3 lines are:\n{preview}")


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value).replace("$", ""))
    return float(match.group().replace(",", "")) if match else None


def score_answer(task: RLMTask, submitted: str | None) -> float:
    """1 for exact, up to 0.5 partial credit for being close, 0 if missing."""
    value = _parse_number(submitted)
    if value is None:
        return 0.0
    tolerance = 0.5 if task.family == "count" else 0.011
    if abs(value - task.answer) <= tolerance:
        return 1.0
    rel_err = abs(value - task.answer) / max(abs(task.answer), 1.0)
    return round(0.5 * max(0.0, 1.0 - 2.0 * rel_err), 4)


# ------------------------------------------------------------------------- bridge

class SubcallBridge:
    """Local HTTP endpoint that serves `llm_query` / `llm_batch` for every running episode."""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=600)
        self._model_name = model_name
        self._pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SUBCALLS)
        self._episodes: dict[str, dict] = {}
        self._lock = threading.Lock()
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                episode_id = self.path.rsplit("/", 1)[-1]
                prompts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["prompts"]
                reply = bridge._handle(episode_id, prompts)
                body = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def register(self) -> tuple[str, dict]:
        episode_id = uuid.uuid4().hex
        stats = {"subcalls": 0, "subcall_tokens": 0}
        with self._lock:
            self._episodes[episode_id] = stats
        return f"http://127.0.0.1:{self._server.server_address[1]}/llm/{episode_id}", stats

    def unregister(self, url: str) -> None:
        with self._lock:
            self._episodes.pop(url.rsplit("/", 1)[-1], None)

    def _handle(self, episode_id: str, prompts: list[str]) -> dict:
        with self._lock:
            stats = self._episodes.get(episode_id)
            if stats is None:
                return {"error": "unknown episode"}
            if stats["subcalls"] + len(prompts) > MAX_SUBCALLS:
                return {"error": f"sub-call budget exhausted ({MAX_SUBCALLS} total)"}
            stats["subcalls"] += len(prompts)
        outputs = list(self._pool.map(lambda p: self._complete(p, stats), prompts))
        return {"outputs": outputs}

    def _complete(self, prompt: str, stats: dict) -> str:
        if len(prompt) > SUBCALL_MAX_CHARS:
            prompt = prompt[:SUBCALL_MAX_CHARS] + "\n[prompt truncated]"
        for attempt in range(6):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=SUBCALL_MAX_TOKENS,
                    temperature=0.3,
                )
                with self._lock:
                    stats["subcall_tokens"] += response.usage.total_tokens if response.usage else 0
                return response.choices[0].message.content or ""
            except openai.RateLimitError:
                threading.Event().wait(2 ** (attempt + 1))
        return "[sub-call failed: rate limited]"


# ------------------------------------------------------------------------- REPL

_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "WANDB")


class ReplWorker:
    def __init__(self, context: str, bridge_url: str):
        self._dir = tempfile.TemporaryDirectory(prefix="rlm-repl-")
        self._context_file = Path(self._dir.name) / "context.txt"
        self._context_file.write_text(context, encoding="utf-8")
        self._bridge_url = bridge_url
        self._start()

    def _start(self) -> None:
        env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in _SECRET_HINTS)}
        env["PYTHONIOENCODING"] = "utf-8"
        self._proc = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).with_name("repl_worker.py")), str(self._context_file), self._bridge_url],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=self._dir.name, env=env, text=True, encoding="utf-8",
        )

    def _exchange(self, code: str) -> str:
        self._proc.stdin.write(json.dumps({"code": code}) + "\n")
        self._proc.stdin.flush()
        return self._proc.stdout.readline()

    async def run(self, code: str) -> dict:
        try:
            line = await asyncio.wait_for(asyncio.to_thread(self._exchange, code), CELL_TIMEOUT_S)
        except (asyncio.TimeoutError, OSError):
            line = ""
        if not line:
            self._proc.kill()
            self._start()
            return {"stdout": "", "error": f"Cell timed out or crashed after {CELL_TIMEOUT_S}s; the REPL was restarted and all variables were lost.",
                    "submitted": False, "answer": None}
        return json.loads(line)

    def close(self) -> None:
        self._proc.kill()
        self._dir.cleanup()


# ------------------------------------------------------------------------ episode

_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str | None) -> str | None:
    match = _CODE_RE.search(text or "")
    return match.group(1) if match else None


def _observation(result: dict, turns_left: int, subcalls_left: int) -> str:
    out = result["stdout"]
    if len(out) > OUTPUT_MAX_CHARS:
        out = out[:OUTPUT_MAX_CHARS] + f"\n[... {len(result['stdout']) - OUTPUT_MAX_CHARS} more characters truncated]"
    parts = [f"REPL output:\n{out}" if out.strip() else "REPL output: (nothing printed)"]
    if result["error"]:
        parts.append(f"Error:\n{result['error']}")
    parts.append(f"({turns_left} turns left, {subcalls_left} sub-calls left)")
    if turns_left == 1:
        parts.append("Your next turn is the last one: it must call submit() with your best answer.")
    return "\n".join(parts)


async def run_episode(model: art.TrainableModel, step: int, task: RLMTask, bridge: SubcallBridge) -> art.Trajectory:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task_prompt(task)}]
    trajectory_items: list = list(messages)
    url, stats = bridge.register()
    worker = ReplWorker(task.context, url)
    codes: list[str] = []
    errors, submitted, answer, turns = 0, False, None, 0
    last_error = None
    try:
        for turn in range(MAX_TURNS):
            turns = turn + 1
            response = await chat_with_retry(
                model.openai_client(),
                model=model.get_inference_name(step=step),
                messages=messages,
                max_tokens=ROOT_MAX_TOKENS,
                temperature=1.0,
                logprobs=True,
                extra_body={"prompt_logprobs": 0},
            )
            choice = response.choices[0]
            attach_token_ids(choice, response.model_extra["prompt_logprobs"], model.base_model)
            trajectory_items.append(choice)
            content = choice.message.content or ""
            messages.append({"role": "assistant", "content": content})

            code = extract_code(content)
            if code is None:
                result = {"stdout": "", "error": "No ```python code block found. Reply with exactly one ```python block.",
                          "submitted": False, "answer": None}
            else:
                codes.append(code)
                result = await worker.run(code)
            if result["error"]:
                errors += 1
                last_error = result["error"]
            if result["submitted"]:
                submitted, answer = True, result["answer"]
                break
            if turn < MAX_TURNS - 1:
                observation = _observation(result, MAX_TURNS - turn - 1, MAX_SUBCALLS - stats["subcalls"])
                messages.append({"role": "user", "content": observation})
                trajectory_items.append({"role": "user", "content": observation})
    finally:
        worker.close()
        bridge.unregister(url)

    reward = score_answer(task, answer) if submitted else 0.0
    if not submitted:
        feedback = "The previous attempt never called submit() before running out of turns."
    elif reward < 1.0:
        feedback = f"The previous attempt submitted {answer}, which is incorrect."
    else:
        feedback = ""
    if last_error and reward < 1.0:
        feedback += f" Its last REPL error was:\n{last_error}"
    return art.Trajectory(
        messages_and_choices=trajectory_items,
        reward=reward,
        metrics={
            "correct": float(reward >= 1.0),
            "submitted": float(submitted),
            "turns": float(turns),
            "subcalls": float(stats["subcalls"]),
            "subcall_tokens": float(stats["subcall_tokens"]),
            "repl_errors": float(errors),
            "first_turn_submit": float(submitted and turns == 1),
            "used_subcalls": float(stats["subcalls"] > 0),
        },
        metadata={
            "family": task.family,
            "question": task.question,
            "answer": str(task.answer),
            "submitted_answer": "" if answer is None else str(answer),
            "feedback": feedback.strip(),
            "codes": json.dumps(codes),
        },
    )


async def gather_rlm_groups(model, step, tasks, rollouts_per_task, bridge) -> list[art.TrajectoryGroup]:
    return await art.gather_trajectory_groups(
        (art.TrajectoryGroup(run_episode(model, step, task, bridge) for _ in range(rollouts_per_task)) for task in tasks),
        pbar_desc=f"{model.name}@step{step}",
        max_exceptions=max(1, len(tasks) * rollouts_per_task // 10),
    )


async def evaluate_rlm(model, step, tasks, bridge, samples_per_task: int = 1) -> dict:
    trajectories = await asyncio.gather(
        *(run_episode(model, step, task, bridge) for task in tasks for _ in range(samples_per_task)),
        return_exceptions=True,
    )
    good = [t for t in trajectories if isinstance(t, art.Trajectory)]
    def mean(key):
        return sum(t.metrics[key] for t in good) / max(len(good), 1)
    return {
        "accuracy": mean("correct"),
        "mean_reward": sum(t.reward for t in good) / max(len(good), 1),
        "submitted": mean("submitted"),
        "turns": mean("turns"),
        "subcalls": mean("subcalls"),
        "first_turn_submit": mean("first_turn_submit"),
        "n": len(good),
        "failed_episodes": len(trajectories) - len(good),
    }
