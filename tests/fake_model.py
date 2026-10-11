"""A fake OpenAI-compatible model server for tests: no W&B, no credit, standard library only.

It speaks enough of /v1/chat/completions for ModelClient: root calls get completion logprobs as
"token_id:<id>" tokens and vLLM-style `prompt_logprobs` ending with the thinking-off block, so the
token-id rebuild and the thinking checks run for real. Prompts are rendered the way Qwen3.8's
chat template renders them with thinking off: every assistant turn, past or being generated, starts
with the empty think block, so a turn's prompt plus its sampled tokens is a prefix of the whole
conversation (the KL-to-base rescoring in step 6 relies on this). It measures requests in flight on
the server side, enforces W&B's limit of 20 per model name with a 429 (step 2 §5), and can inject
faults. Root replies come from `policy(messages)`, sub-call replies from `answer(prompt)`; set
`policy_for(model, messages)` or `answer_for(model, prompt)` to make them depend on the checkpoint.

    server = FakeModelServer()
    await server.start()
    client = openai.AsyncOpenAI(base_url=server.base_url, api_key="test")
"""

from __future__ import annotations

import asyncio
import collections
import json
import random
import time
import unittest
from typing import Callable

from rlm_rl import client as rclient


class AsyncCase(unittest.IsolatedAsyncioTestCase):
    """IsolatedAsyncioTestCase with asyncio's debug mode off.

    Debug mode records a stack trace for every task and future. On WSL's mount of the Windows
    drive each trace stats every source file, which made the load tests about 20 times slower.
    """

    def _setupAsyncioRunner(self):
        super()._setupAsyncioRunner()
        self._asyncioRunner.get_loop().set_debug(False)

NEWLINE_ID, DOUBLE_NEWLINE_ID, ASSISTANT_ID = 198, 271, 77091
ROLE_IDS = {"system": 8948, "user": 872, "assistant": ASSISTANT_ID}
THINKING_OFF_TAIL = [rclient.IM_START_ID, ASSISTANT_ID, NEWLINE_ID, rclient.THINK_START_ID, DOUBLE_NEWLINE_ID,
                     rclient.THINK_END_ID, DOUBLE_NEWLINE_ID]
THINKING_ON_TAIL = [rclient.IM_START_ID, ASSISTANT_ID, NEWLINE_ID, rclient.THINK_START_ID, NEWLINE_ID]


def fake_ids(text: str) -> list[int]:
    """One fake token per 4 characters."""
    return [1000 + (sum(map(ord, text[i:i + 4])) % 90_000) for i in range(0, len(text), 4)] or [1000]


def render(messages: list[dict], thinking: bool = False) -> list[int]:
    """Prompt token ids as Qwen3.8's template lays them out, with fake ids for the text."""
    ids = []
    for m in messages:
        ids += [rclient.IM_START_ID, ROLE_IDS[m["role"]], NEWLINE_ID]
        if m["role"] == "assistant":
            ids += THINKING_OFF_TAIL[3:]
        ids += fake_ids(m["content"]) + [rclient.IM_END_ID, NEWLINE_ID]
    return ids + (THINKING_ON_TAIL if thinking else THINKING_OFF_TAIL)


def submit_policy(messages: list[dict]) -> str:
    """Turn 1 looks at the context and asks one sub-call; turn 2 submits the line count."""
    if sum(m["role"] == "assistant" for m in messages) == 0:
        return ("Look first.\n```python\nlines = context.splitlines()\nprint(len(lines))\n"
                "print(llm('Summarize: ' + lines[0]))\n```")
    return "Submit the count.\n```python\nsubmit(len(context.splitlines()))\n```"


class FakeModelServer:
    def __init__(self, *, per_model_limit: int | None = 20, latency: tuple[float, float] = (0.0, 0.0),
                 policy: Callable[[list[dict]], str] = submit_policy, answer: Callable[[str], str] = lambda p: "ok",
                 thinking: bool = False, seed: int = 0):
        self.per_model_limit = per_model_limit
        self.latency = latency
        self.policy = policy
        self.answer = answer
        self.policy_for: Callable[[str, list[dict]], str] | None = None
        self.answer_for: Callable[[str, str], str] | None = None
        # logprob of every sampled token, and of every prompt token when prompt_logprobs is asked for
        self.completion_logprob: Callable[[str], float] = lambda model: -0.5
        self.prompt_logprob: Callable[[str], float] = lambda model: -1.0
        self.thinking = thinking
        # fault(kind, body) -> HTTP status to fail with, "hang" to never answer, or None to serve normally
        self.fault: Callable[[str, dict], int | str | None] = lambda kind, body: None
        self.in_flight = 0
        self.in_flight_by_model: collections.Counter = collections.Counter()
        self.peak = 0
        self.peak_by_model: collections.Counter = collections.Counter()
        self.requests: collections.Counter = collections.Counter()  # by kind: root, subcall
        self.statuses: collections.Counter = collections.Counter()
        self.bodies: list[dict] = []
        self._rng = random.Random(seed)
        self._server: asyncio.AbstractServer | None = None
        self._handlers: set[asyncio.Task] = set()
        self._rays = 0

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._connection, "127.0.0.1", 0)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            for task in list(self._handlers):
                task.cancel()
            await asyncio.gather(*self._handlers, return_exceptions=True)
            await self._server.wait_closed()
            self._server = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}/v1"

    def reset_counts(self) -> None:
        self.peak, self.peak_by_model = 0, collections.Counter()
        self.requests, self.statuses = collections.Counter(), collections.Counter()

    # ------------------------------------------------------------------ HTTP

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        self._handlers.add(task)
        try:
            while True:
                request_line = await reader.readline()
                if not request_line:
                    break
                headers = {}
                while True:
                    line = await reader.readline()
                    if line in (b"\r\n", b"\n", b""):
                        break
                    name, _, value = line.decode("latin-1").partition(":")
                    headers[name.strip().lower()] = value.strip()
                body = await reader.readexactly(int(headers.get("content-length", "0")))
                status, payload, extra = await self._handle(request_line.decode("latin-1").split(" ")[1], body)
                data = json.dumps(payload).encode()
                head = [f"HTTP/1.1 {status} {'OK' if status == 200 else 'Error'}", "content-type: application/json",
                        f"content-length: {len(data)}", *(f"{k}: {v}" for k, v in extra.items())]
                writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + data)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self._handlers.discard(task)
            writer.close()

    async def _handle(self, path: str, raw: bytes) -> tuple[int, dict, dict]:
        if not path.endswith("/chat/completions"):
            return 404, {"error": {"message": "not found"}}, {}
        body = json.loads(raw)
        self.bodies.append(body)
        model = body["model"]
        kind = "root" if body.get("logprobs") else "kl" if body.get("prompt_logprobs") is not None else "subcall"
        self.requests[kind] += 1
        if self.per_model_limit is not None and self.in_flight_by_model[model] >= self.per_model_limit:
            return self._error(429, "Concurrency limit reached for requests")
        self.in_flight += 1
        self.in_flight_by_model[model] += 1
        self.peak = max(self.peak, self.in_flight)
        self.peak_by_model[model] = max(self.peak_by_model[model], self.in_flight_by_model[model])
        try:
            fault = self.fault(kind, body)
            if fault == "hang":
                await asyncio.sleep(3600)
            low, high = self.latency
            if high:
                await asyncio.sleep(self._rng.uniform(low, high))
            if isinstance(fault, int):
                return self._error(fault, f"injected HTTP {fault}")
            self.statuses[200] += 1
            return 200, self._completion(kind, body), {}
        finally:
            self.in_flight -= 1
            self.in_flight_by_model[model] -= 1

    def _error(self, status: int, message: str) -> tuple[int, dict, dict]:
        self.statuses[status] += 1
        self._rays += 1
        return status, {"error": {"message": message, "type": "error", "code": status}}, {"cf-ray": f"fake{self._rays:06d}-YYZ"}

    def _completion(self, kind: str, body: dict) -> dict:
        messages, model = body["messages"], body["model"]
        if kind == "root":
            text = self.policy_for(model, messages) if self.policy_for else self.policy(messages)
        else:
            prompt_text = messages[-1]["content"]
            text = self.answer_for(model, prompt_text) if self.answer_for else self.answer(prompt_text)
        completion = fake_ids(text) + [rclient.IM_END_ID]
        prompt = render(messages, self.thinking)
        choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        out = {"id": f"chatcmpl-{time.monotonic_ns()}", "object": "chat.completion", "created": int(time.time()),
               "model": body["model"], "choices": [choice],
               "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                         "total_tokens": len(prompt) + len(completion),
                         "prompt_tokens_details": {"cached_tokens": 0},
                         "completion_tokens_details": {"reasoning_tokens": 8 if self.thinking else 0}}}
        if body.get("logprobs"):
            lp = self.completion_logprob(model)
            choice["logprobs"] = {"content": [{"token": f"token_id:{i}", "logprob": lp, "bytes": None,
                                               "top_logprobs": []} for i in completion]}
        if body.get("prompt_logprobs") is not None:
            lp = self.prompt_logprob(model)
            out["prompt_logprobs"] = [None] + [{str(i): {"logprob": lp, "rank": 1, "decoded_token": "x"}}
                                               for i in prompt[1:]]
        return out
