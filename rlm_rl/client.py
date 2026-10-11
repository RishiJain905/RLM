"""Every request to the model goes through here: one shared limiter, retries and infra errors (spec §1.2-1.3, H5, H6).

- One Limiter caps requests in flight across root turns and sub-calls alike: 30 in total, and
  20 for any one model name, which is W&B's own limit per checkpoint (step 2 §5). Root calls go
  to the training checkpoint (`:stepN`) and sub-calls to the base model (`:step0`), so training
  can use all 30. A base-model eval sends everything to `:step0` and runs at 20.
- A 429, a 5xx, a timeout or a dropped connection is retried with jittered exponential backoff,
  up to 6 tries starting at 2 s. A request gives its slot back while it waits. When the tries run
  out it raises RequestFailed, an InfraError: the episode is flagged infra_error and never
  reaches training, and the model is never shown a fake answer.
- Every response is checked for thinking mode (step 2 §4): reasoning tokens must be 0, and a
  root call's rendered prompt must end with the empty think block. A failed check is infra.
- Root calls rebuild the token ids ART's trainer needs from prompt logprobs, which W&B's
  gateway strips.

    client = ModelClient(model.openai_client(), subcall_model=model.get_inference_name(step=0))
    choice = await client.root_turn(messages, model=model.get_inference_name(step=step), usage=usage)
    async with Sandbox(context, client.subcaller(usage)) as box: ...
"""

from __future__ import annotations

import asyncio
import collections
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import openai

from . import bridge
from .bridge import Complete, InfraError, SubcallError

# Requests in flight (step 2 §5, Rishi 2026-10-10). Replaces the cap of 8 in spec §1.3.
MAX_IN_FLIGHT = 30
MAX_IN_FLIGHT_PER_MODEL = 20

ROOT_TEMPERATURE = 1.0  # training and eval alike (spec §5.1)
ENABLE_THINKING = False  # root and sub-calls (spec §5.1)

# Qwen/Qwen3.8-27B special tokens, from its tokenizer_config.json. vLLM reports no logprob for a
# prompt's first token, which is always <|im_start|>.
IM_START_ID = 248045
IM_END_ID = 248046
THINK_START_ID = 248068
THINK_END_ID = 248069

# W&B inference prices for Qwen3.8-27B, $ per 1M tokens (research §6). For the per-episode estimate only.
PRICE_INPUT = 0.40
PRICE_CACHED_INPUT = 0.15
PRICE_OUTPUT = 3.00

CHAT_MAX_TOKENS = 1024  # one plain chat call: the capability check (step 6)
CHAT_TEMPERATURE = 0.0  # fixed decoding, so a capability drop between checkpoints isn't sampling noise *(mine)*

RETRY_STATUSES = frozenset({408, 409, 429})  # plus every 5xx
SUBCALL_REJECT_STATUSES = frozenset({400, 413, 422})  # the model's prompt was refused: shown to it as an error


@dataclass(frozen=True)
class RetryPolicy:
    tries: int = 6
    base_s: float = 2.0
    max_s: float = 60.0
    jitter: float = 0.5  # each wait is scaled by a random factor in [1 - jitter, 1 + jitter]
    timeout_s: float = 600.0  # one request, sent to the API as its timeout

    def delay(self, attempt: int, rng: random.Random, retry_after: float | None = None) -> float:
        """The wait after failed try number `attempt` (0-based)."""
        wait = min(self.max_s, self.base_s * 2 ** attempt) * rng.uniform(1 - self.jitter, 1 + self.jitter)
        if retry_after is not None:
            wait = max(wait, min(retry_after, self.max_s))
        return wait


class RequestFailed(InfraError):
    """A request that failed for a reason outside the model's control, after any retries."""

    def __init__(self, kind: str, message: str, *, model: str, status: int | None = None,
                 cf_ray: str | None = None, attempts: int = 1):
        super().__init__(f"{kind}: {message} (model {model}, {attempts} tries)")
        self.kind = kind
        self.model = model
        self.status = status
        self.cf_ray = cf_ray
        self.attempts = attempts


# ----------------------------------------------------------------------------- limiter


class Limiter:
    """Caps requests in flight, in total and per model name. Waiters are served first come, first served,
    except that one blocked only by its own model's cap doesn't hold up requests for another model."""

    def __init__(self, total: int = MAX_IN_FLIGHT, per_model: int = MAX_IN_FLIGHT_PER_MODEL):
        if per_model > total:
            raise ValueError("the per-model cap can't be above the total cap")
        self.total = total
        self.per_model = per_model
        self.in_flight = 0
        self.by_model: collections.Counter[str] = collections.Counter()
        self._waiters: collections.deque[tuple[str, asyncio.Future]] = collections.deque()
        self.reset_peaks()

    def reset_peaks(self) -> dict:
        """Return the peaks since the last reset, then start counting again (once per step)."""
        peaks = getattr(self, "_peaks", None)
        self._peaks = {"total": self.in_flight, "by_model": dict(self.by_model), "granted": 0, "waited_s": 0.0}
        return peaks or {}

    @property
    def peaks(self) -> dict:
        return self._peaks

    @property
    def waiting(self) -> int:
        return sum(1 for _, fut in self._waiters if not fut.done())

    @asynccontextmanager
    async def slot(self, model: str):
        await self.acquire(model)
        try:
            yield
        finally:
            self.release(model)

    async def acquire(self, model: str) -> None:
        fut = asyncio.get_running_loop().create_future()
        started = time.monotonic()
        self._waiters.append((model, fut))
        self._grant()
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():  # granted, then cancelled before it could run
                self.release(model)
            else:
                self._drop(fut)
            raise
        self._peaks["waited_s"] += time.monotonic() - started

    def release(self, model: str) -> None:
        self.in_flight -= 1
        self.by_model[model] -= 1
        if not self.by_model[model]:
            del self.by_model[model]
        self._grant()

    def _grant(self) -> None:
        for item in list(self._waiters):
            model, fut = item
            if fut.done():
                self._waiters.remove(item)
                continue
            if self.in_flight >= self.total:
                break
            if self.by_model[model] >= self.per_model:
                continue  # keeps its place; a later request for another model may go first
            self._waiters.remove(item)
            self.in_flight += 1
            self.by_model[model] += 1
            peaks = self._peaks
            peaks["granted"] += 1
            peaks["total"] = max(peaks["total"], self.in_flight)
            peaks["by_model"][model] = max(peaks["by_model"].get(model, 0), self.by_model[model])
            fut.set_result(None)

    def _drop(self, fut: asyncio.Future) -> None:
        for item in list(self._waiters):
            if item[1] is fut:
                self._waiters.remove(item)
        self._grant()


# ----------------------------------------------------------------------------- accounting


@dataclass
class Usage:
    """Tokens one episode used, root turns and sub-calls separately."""

    root_calls: int = 0
    root_prompt_tokens: int = 0
    root_cached_tokens: int = 0
    root_completion_tokens: int = 0
    subcalls: int = 0
    subcall_prompt_tokens: int = 0
    subcall_cached_tokens: int = 0
    subcall_completion_tokens: int = 0

    def add(self, kind: str, usage: Any) -> None:
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        if kind == "root":
            self.root_calls += 1
            self.root_prompt_tokens += prompt
            self.root_cached_tokens += cached
            self.root_completion_tokens += completion
        else:
            self.subcalls += 1
            self.subcall_prompt_tokens += prompt
            self.subcall_cached_tokens += cached
            self.subcall_completion_tokens += completion

    def cost_usd(self) -> float:
        """Estimated W&B inference cost at the research's prices. The billing page is the truth."""
        cached = self.root_cached_tokens + self.subcall_cached_tokens
        uncached = self.root_prompt_tokens + self.subcall_prompt_tokens - cached
        output = self.root_completion_tokens + self.subcall_completion_tokens
        return (uncached * PRICE_INPUT + cached * PRICE_CACHED_INPUT + output * PRICE_OUTPUT) / 1e6

    def as_dict(self) -> dict:
        return {**self.__dict__, "est_cost_usd": round(self.cost_usd(), 6)}


@dataclass
class RequestStats:
    """What the client saw since the last reset (once per step): logged as infra metrics."""

    calls: collections.Counter = field(default_factory=collections.Counter)  # by kind: root, subcall
    attempts: int = 0
    retries: int = 0
    rate_limited: int = 0  # 429 responses, including ones a retry recovered from
    server_errors: int = 0
    timeouts: int = 0
    connection_errors: int = 0
    rejected: int = 0  # non-retryable 4xx
    failed: collections.Counter = field(default_factory=collections.Counter)  # RequestFailed raised, by kind
    events: collections.deque = field(default_factory=lambda: collections.deque(maxlen=500))  # failures with cf-ray

    def as_dict(self) -> dict:
        return {"calls": dict(self.calls), "attempts": self.attempts, "retries": self.retries,
                "rate_limited": self.rate_limited, "server_errors": self.server_errors, "timeouts": self.timeouts,
                "connection_errors": self.connection_errors, "rejected": self.rejected, "failed": dict(self.failed),
                "events": list(self.events)}


# ----------------------------------------------------------------------------- token ids and thinking


def attach_token_ids(choice: Any, prompt_logprobs: list) -> list[int]:
    """Attach the prompt and completion token ids ART's trainer needs, and return the prompt ids.

    W&B's gateway drops vLLM's token id fields but passes `prompt_logprobs`, keyed by the actual
    prompt token id, and ART's `return_tokens_as_token_ids` makes sampled tokens read "token_id:<id>".
    """
    prompt_ids = [IM_START_ID] + [int(next(iter(entry))) for entry in prompt_logprobs[1:]]
    completion_ids = [int(t.token.removeprefix("token_id:")) for t in choice.logprobs.content]
    extra = choice.model_extra if choice.model_extra is not None else {}
    extra["prompt_token_ids"] = prompt_ids
    extra["token_ids"] = completion_ids
    return prompt_ids


def thinking_off_prompt(prompt_ids: list[int]) -> bool:
    """Whether the rendered prompt ends with the empty think block (`<think>\\n\\n</think>\\n\\n`).

    With thinking on, Qwen3.8's template ends the prompt with `<think>\\n` instead (step 2 §4).
    """
    tail = prompt_ids[-6:]
    if THINK_END_ID not in prompt_ids[-3:] or THINK_START_ID not in tail:
        return False
    return tail.index(THINK_START_ID) < max(i for i, t in enumerate(tail) if t == THINK_END_ID)


def reasoning_tokens(usage: Any) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", 0) or 0


# ----------------------------------------------------------------------------- client


def _header(exc: Exception, name: str) -> str | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    return headers.get(name) if headers is not None else None


def _retry_after(exc: Exception) -> float | None:
    value = _header(exc, "retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class ModelClient:
    """Root turns and sub-calls for one run, through one limiter.

    `openai_client` is any AsyncOpenAI-compatible client, usually ART's `model.openai_client()`.
    Its own retries are turned off, so every retry is counted and waits outside the limiter.
    """

    def __init__(self, openai_client: Any, *, subcall_model: str, limiter: Limiter | None = None,
                 retry: RetryPolicy = RetryPolicy(), rng: random.Random | None = None):
        self._client = openai_client.with_options(max_retries=0)
        self.subcall_model = subcall_model
        self.limiter = limiter or Limiter()
        self.retry = retry
        self.stats = RequestStats()
        self.spend: collections.defaultdict[str, Usage] = collections.defaultdict(Usage)  # every response, by kind
        self._rng = rng or random.Random()

    def spent_usd(self) -> float:
        """Estimated inference spend of every response this client has received (step 6's spend counter)."""
        return sum(usage.cost_usd() for usage in self.spend.values())

    def reset_stats(self) -> dict:
        """Return this step's request stats and limiter peaks, then start counting again."""
        stats, self.stats = self.stats.as_dict(), RequestStats()
        stats["peaks"] = self.limiter.reset_peaks()
        return stats

    async def root_turn(self, messages: list[dict], *, model: str, usage: Usage | None = None,
                        max_tokens: int = bridge.ROOT_MAX_TOKENS, temperature: float = ROOT_TEMPERATURE) -> Any:
        """One root turn. Returns the choice, with the token ids ART needs attached. Raises RequestFailed."""
        response = await self._request(
            "root", model, messages=messages, max_tokens=max_tokens, temperature=temperature, logprobs=True,
            extra_body={"prompt_logprobs": 0, "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}})
        choice = self._first_choice(response, model)
        prompt_logprobs = (response.model_extra or {}).get("prompt_logprobs")
        if not prompt_logprobs or choice.logprobs is None or not choice.logprobs.content:
            self._fail("bad_response", "the response has no prompt or completion logprobs", model)
        try:
            prompt_ids = attach_token_ids(choice, prompt_logprobs)
        except (AttributeError, StopIteration, TypeError, ValueError) as exc:
            self._fail("bad_response", f"token ids could not be rebuilt ({exc!r})", model)
        if reasoning_tokens(response.usage) or not thinking_off_prompt(prompt_ids):
            self._fail("thinking_on", "the server rendered a thinking prompt or spent reasoning tokens", model)
        if usage is not None:
            usage.add("root", response.usage)
        return choice

    def subcaller(self, usage: Usage | None = None, model: str | None = None) -> Complete:
        """The `complete` function a Sandbox serves sub-calls with, counting tokens into `usage`."""
        model = model or self.subcall_model

        async def complete(prompt: str) -> str:
            response = await self._request(
                "subcall", model, messages=[{"role": "user", "content": prompt}],
                max_tokens=bridge.SUBCALL_MAX_TOKENS, temperature=bridge.SUBCALL_TEMPERATURE,
                extra_body={"chat_template_kwargs": {"enable_thinking": bridge.SUBCALL_ENABLE_THINKING}})
            choice = self._first_choice(response, model)
            if reasoning_tokens(response.usage):
                self._fail("thinking_on", "a sub-call spent reasoning tokens", model)
            if usage is not None:
                usage.add("subcall", response.usage)
            return choice.message.content or ""

        return complete

    async def chat(self, messages: list[dict], *, model: str, max_tokens: int = CHAT_MAX_TOKENS,
                   temperature: float = CHAT_TEMPERATURE) -> str:
        """One plain chat call with thinking off, for the capability check. Raises RequestFailed."""
        response = await self._request(
            "chat", model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}})
        choice = self._first_choice(response, model)
        if reasoning_tokens(response.usage):
            self._fail("thinking_on", "a chat call spent reasoning tokens", model)
        return choice.message.content or ""

    async def prompt_logprobs(self, messages: list[dict], *, model: str) -> tuple[list[int], list[float | None]]:
        """Score a finished conversation under `model`: each prompt token's id and logprob (KL to base, spec §5.3).

        The conversation is sent as the prompt with `prompt_logprobs=0`, and one token is generated.
        The first token has no logprob, as in the token-id rebuild. Raises RequestFailed.
        """
        response = await self._request(
            "kl", model, messages=messages, max_tokens=1, temperature=ROOT_TEMPERATURE,
            extra_body={"prompt_logprobs": 0, "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}})
        entries = (response.model_extra or {}).get("prompt_logprobs")
        if not entries:
            self._fail("bad_response", "the response has no prompt logprobs", model)
        ids, logprobs = [IM_START_ID], [None]
        try:
            for entry in entries[1:]:
                token = next(iter(entry))
                ids.append(int(token))
                logprobs.append(float(entry[token]["logprob"]))
        except (AttributeError, KeyError, StopIteration, TypeError, ValueError) as exc:
            self._fail("bad_response", f"prompt logprobs could not be read ({exc!r})", model)
        return ids, logprobs

    def _first_choice(self, response: Any, model: str) -> Any:
        choices = getattr(response, "choices", None)
        if not choices or getattr(choices[0], "message", None) is None:
            self._fail("bad_response", "the response has no choices", model)
        return choices[0]

    def _fail(self, kind: str, message: str, model: str, **kw) -> None:
        self.stats.failed[kind] += 1
        self.stats.events.append({"kind": kind, "model": model, "status": kw.get("status"),
                                  "cf_ray": kw.get("cf_ray"), "at": time.time()})
        raise RequestFailed(kind, message, model=model, **kw)

    async def _request(self, kind: str, model: str, **kwargs) -> Any:
        """One chat completion under the limiter, retried on 429, 5xx, timeouts and dropped connections."""
        self.stats.calls[kind] += 1
        tries = self.retry.tries
        for attempt in range(tries):
            retry_after = failure = status = cf_ray = None
            self.stats.attempts += 1
            async with self.limiter.slot(model):
                try:
                    response = await self._client.chat.completions.create(model=model, timeout=self.retry.timeout_s,
                                                                          **kwargs)
                    self.spend[kind].add(kind, getattr(response, "usage", None))
                    return response
                except openai.APIStatusError as exc:
                    status, cf_ray = exc.status_code, _header(exc, "cf-ray")
                    if status == 429:
                        self.stats.rate_limited += 1
                        failure = "rate_limited"
                    elif status >= 500:
                        self.stats.server_errors += 1
                        failure = "server_error"
                    elif status in RETRY_STATUSES:
                        failure = f"http_{status}"
                    else:
                        self.stats.rejected += 1
                        if kind == "subcall" and status in SUBCALL_REJECT_STATUSES:
                            raise SubcallError(f"the API rejected the request (HTTP {status})") from exc
                        self._fail("rejected", f"HTTP {status}: {str(exc)[:300]}", model, status=status,
                                   cf_ray=cf_ray, attempts=attempt + 1)
                    retry_after = _retry_after(exc)
                except openai.APITimeoutError:
                    self.stats.timeouts += 1
                    failure = "timeout"
                except openai.APIConnectionError:
                    self.stats.connection_errors += 1
                    failure = "connection"
            self.stats.events.append({"kind": failure, "model": model, "status": status, "cf_ray": cf_ray,
                                      "attempt": attempt + 1, "at": time.time()})
            if attempt < tries - 1:
                self.stats.retries += 1
                await asyncio.sleep(self.retry.delay(attempt, self._rng, retry_after))
        self._fail(failure, f"still failing after {tries} tries", model, status=status, cf_ray=cf_ray,
                   attempts=tries)
