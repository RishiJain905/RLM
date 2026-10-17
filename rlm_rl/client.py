"""Every request to the model goes through here: limits, pacing, retries and infra errors (spec §1.2-1.3, H5, H6).

- One Limiter caps requests in flight across root turns and sub-calls alike: 24 in total, 12 for
  the sub-call model, and 20 for any other model name, which is W&B's own limit per checkpoint
  (step 2 §5). Root calls go to the training checkpoint (`:stepN`) and sub-calls to the base model
  (`:step0`), so training runs at most 20 root calls and 12 sub-calls, 24 together.
- A Pacer starts requests to the sub-call model at least 0.25 s apart, at most 4 new ones a
  second, so an `llm_batch` of 10 goes out over about 2.5 s instead of at once. Every try waits
  for its start time, retries too, before it asks the limiter for a slot, so pacing holds no slot.
  Other models aren't paced. The time spent pacing is logged as the step's `paced_s`.
- Caps and gaps go by model name, not by kind of call. A base-model eval sends its root calls to
  `:step0` too, so it runs at 12 in flight and 4 starts a second, and so does training's first
  step, which samples from `:step0`. The KL rescoring against base shares the sub-call model's 12
  and its gap.
- A 429, a 5xx, a timeout or a dropped connection is retried with jittered exponential backoff,
  up to 8 tries starting at 4 s and capped at 120 s, which rides out about 6 minutes of
  throttling. A Retry-After header is honoured up to the cap. A request gives its slot back while
  it waits. When the tries run out it raises RequestFailed, an InfraError: the episode is flagged
  infra_error and never reaches training, and the model is never shown a fake answer.
- Why the cap for sub-calls, the pacing and the longer retries (step 10 §6, 2026-10-16): at 30 in
  total and 20 per model, W&B answered 429 to 19-47% of tries from step 2 on, once root calls and
  sub-calls went to different models; step 1, all on one model, had none. The 429s tracked sub-call
  volume, which arrives in bursts from `llm_batch`. Six tries from 2 s covered about a minute, and
  in step 20 seven requests ran out of tries, which made 7 of its 32 episodes infra errors. Rishi
  decided to spread sub-calls out in production even if it is slower. The numbers are defaults to
  tune *(mine)*.
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
from dataclasses import asdict, dataclass, field
from typing import Any

import openai

from . import bridge
from .bridge import Complete, InfraError, SubcallError

# Requests in flight. Step 2 §5 (Rishi 2026-10-10) set 30 in total and 20 per model name, replacing
# the cap of 8 in spec §1.3. At those caps W&B answered 429 to 19-47% of tries once sub-calls had a
# model of their own (step 10 §6, 2026-10-16), so the total fell to 24 and the sub-call model got a
# cap of 12 (step 10 §10 #1 *(mine)*, defaults to tune). Average demand there was ~10-20 in flight.
MAX_IN_FLIGHT = 24
MAX_IN_FLIGHT_PER_MODEL = 20  # every model but the sub-call model: W&B's own limit per checkpoint (step 2 §5)
MAX_SUBCALLS_IN_FLIGHT = 12  # the sub-call model (`:step0`), set by the ModelClient that has it (step 10 §6)

# The least time between the starts of two requests to the sub-call model: at most 4 new ones a
# second (step 10 §6, Rishi 2026-10-16: spread sub-calls out even if it is slower). A default to tune *(mine)*.
SUBCALL_START_GAP_S = 0.25

ROOT_TEMPERATURE = 1.0  # training and eval alike (spec §5.1)
ENABLE_THINKING = False  # root and sub-calls (spec §5.1)
# Root thinking for one baseline arm only (plan step 11, decision 11b; step 2 §6): the base model as an RLM with its
# root at reasoning effort `medium`. Training keeps thinking off, sub-calls stay off whatever the root does, and
# `xhigh` is never used (it broke the turn cap and once never answered).
ROOT_THINKING_EFFORTS = ("low", "medium")

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
    """Jittered exponential backoff. The defaults wait 4, 8, 16, 32, 64, 120 and 120 s between 8 tries,
    364 s before jitter, so a request rides out about 6 minutes of throttling. They were 6 tries from
    2 s, capped at 60 s, which covered about a minute and ran out on 7 requests in step 10's last step
    (step 10 §6, 2026-10-16). Defaults to tune *(mine)*."""

    tries: int = 8
    base_s: float = 4.0
    max_s: float = 120.0  # also caps a Retry-After header
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


# ----------------------------------------------------------------------------- limiter and pacer


class Limiter:
    """Caps requests in flight, in total and per model name. Waiters are served first come, first served,
    except that one blocked only by its own model's cap doesn't hold up requests for another model.

    `model_caps` gives some models a cap of their own in place of `per_model`, such as the sub-call
    model's MAX_SUBCALLS_IN_FLIGHT, which ModelClient sets on the limiter it creates.
    """

    def __init__(self, total: int = MAX_IN_FLIGHT, per_model: int = MAX_IN_FLIGHT_PER_MODEL,
                 model_caps: dict[str, int] | None = None):
        if per_model > total:
            raise ValueError("the per-model cap can't be above the total cap")
        self.model_caps = dict(model_caps or {})
        for model, cap in self.model_caps.items():
            if cap > total:
                raise ValueError(f"the cap for {model} ({cap}) can't be above the total cap ({total})")
            if cap < 1:
                raise ValueError(f"the cap for {model} must be at least 1, not {cap}")
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

    def cap(self, model: str) -> int:
        """The most requests `model` may have in flight."""
        return self.model_caps.get(model, self.per_model)

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
            if self.by_model[model] >= self.cap(model):
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


class Pacer:
    """Spaces out the starts of requests to a model: each starts at least `gaps[model]` seconds after
    the one booked before it. Models without a gap start at once.

    A start time is booked without an await in between, so requests that arrive together under
    asyncio get distinct times: a burst of 10 with a 0.25 s gap goes out over 2.25 s. A request that
    wakes late, behind a busy event loop, waits on until a full gap has passed since the request
    before it started, so late wake-ups don't bunch either. A request cancelled while it waits leaves
    its time unused, which only widens the gap. ModelClient paces before it asks the limiter for a
    slot, so a request waiting here holds none. One that then waits in the limiter's queue starts
    when a slot frees, one per finished request, so that queue doesn't release a burst either.
    """

    def __init__(self, gaps: dict[str, float] | None = None):
        self.gaps = dict(gaps or {})
        for model, gap in self.gaps.items():
            if gap < 0:
                raise ValueError(f"the start gap for {model} can't be negative ({gap})")
        self._next: dict[str, float] = {}  # by model: the next start time free to book, on time.monotonic()
        self._last: dict[str, float] = {}  # by model: when the last request actually started

    def reserve(self, model: str) -> float:
        """Book `model`'s next start time and return how long to wait for it (0 for a model without a gap)."""
        gap = self.gaps.get(model, 0.0)
        if gap <= 0:
            return 0.0
        now = time.monotonic()
        start = max(now, self._next.get(model, now))
        self._next[model] = start + gap
        return start - now

    async def wait(self, model: str) -> float:
        """Wait for `model`'s next start time. Returns the seconds waited."""
        gap = self.gaps.get(model, 0.0)
        if gap <= 0:
            return 0.0
        began = time.monotonic()
        delay = self.reserve(model)
        while True:
            if delay > 0:
                await asyncio.sleep(delay)
            now = time.monotonic()
            last = self._last.get(model)
            delay = 0.0 if last is None else last + gap - now
            if delay <= 0:
                break
        self._last[model] = now
        self._next[model] = max(self._next[model], now + gap)  # later bookings follow the actual starts
        return now - began


def settings() -> dict:
    """The pinned request limits, for the run's config.json (spec §8)."""
    return {"max_in_flight": MAX_IN_FLIGHT, "per_model": MAX_IN_FLIGHT_PER_MODEL,
            "subcall_in_flight": MAX_SUBCALLS_IN_FLIGHT, "subcall_start_gap_s": SUBCALL_START_GAP_S,
            "retry": asdict(RetryPolicy())}


# ----------------------------------------------------------------------------- accounting


@dataclass
class Usage:
    """Tokens one episode used, root turns and sub-calls separately."""

    root_calls: int = 0
    root_prompt_tokens: int = 0
    root_cached_tokens: int = 0
    root_completion_tokens: int = 0
    root_reasoning_tokens: int = 0  # part of root_completion_tokens; 0 unless the root thinks (decision 11b's arm)
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
            self.root_reasoning_tokens += reasoning_tokens(usage)
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
    paced_s: float = 0.0  # waiting for a start time from the Pacer, summed over tries, like the limiter's waited_s
    failed: collections.Counter = field(default_factory=collections.Counter)  # RequestFailed raised, by kind
    statuses: collections.Counter = field(default_factory=collections.Counter)  # failed tries by HTTP status (step 11)
    events: collections.deque = field(default_factory=lambda: collections.deque(maxlen=500))  # failures with cf-ray

    def as_dict(self) -> dict:
        return {"calls": dict(self.calls), "attempts": self.attempts, "retries": self.retries,
                "rate_limited": self.rate_limited, "server_errors": self.server_errors, "timeouts": self.timeouts,
                "connection_errors": self.connection_errors, "rejected": self.rejected,
                "paced_s": round(self.paced_s, 3), "failed": dict(self.failed),
                "statuses": {str(k): v for k, v in self.statuses.items()}, "events": list(self.events)}


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


def thinking_on_prompt(prompt_ids: list[int]) -> bool:
    """Whether the rendered prompt ends with an open think block (`<think>\\n`), as Qwen3.8's template does with
    thinking on (step 2 §4)."""
    return THINK_START_ID in prompt_ids[-2:] and THINK_END_ID not in prompt_ids[-3:]


def reasoning_text(choice: Any) -> str | None:
    """The thinking text of a choice, which W&B's server returns beside the content (step 2 §6), if any."""
    message = getattr(choice, "message", None)
    for name in ("reasoning", "reasoning_content"):
        text = getattr(message, name, None) or ((getattr(message, "model_extra", None) or {}).get(name))
        if text:
            return str(text)
    return None


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
    """Root turns and sub-calls for one run, through one limiter and one pacer.

    `openai_client` is any AsyncOpenAI-compatible client, usually ART's `model.openai_client()`.
    Its own retries are turned off, so every retry is counted and waits outside the limiter.
    Without a `limiter` or `pacer` given, the client makes its own, giving `subcall_model` its cap
    of MAX_SUBCALLS_IN_FLIGHT and its start gap of SUBCALL_START_GAP_S (step 10 §6). One given is
    used as it is: pass the same ones to clients that should share them.

    `root_thinking` is None (thinking off, as in training) or a reasoning effort in ROOT_THINKING_EFFORTS, for
    decision 11b's baseline arm: root turns then think at that effort, and sub-calls stay off.
    """

    def __init__(self, openai_client: Any, *, subcall_model: str, limiter: Limiter | None = None,
                 pacer: Pacer | None = None, retry: RetryPolicy = RetryPolicy(), rng: random.Random | None = None,
                 root_thinking: str | None = None):
        if root_thinking is not None and root_thinking not in ROOT_THINKING_EFFORTS:
            raise ValueError(f"root thinking must be None or one of {ROOT_THINKING_EFFORTS}, not {root_thinking!r}")
        self._client = openai_client.with_options(max_retries=0)
        self.root_thinking = root_thinking
        self.subcall_model = subcall_model
        self.limiter = limiter if limiter is not None else Limiter(model_caps={subcall_model: MAX_SUBCALLS_IN_FLIGHT})
        self.pacer = pacer if pacer is not None else Pacer({subcall_model: SUBCALL_START_GAP_S})
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
        """One root turn. Returns the choice, with the token ids ART needs attached. Raises RequestFailed.

        With `root_thinking` set, the turn thinks at that effort: the rendered prompt must then end with an open
        think block, and the choice also carries its `reasoning_tokens` (in model_extra)."""
        effort = self.root_thinking
        template = {"enable_thinking": True, "reasoning_effort": effort} if effort else {"enable_thinking": ENABLE_THINKING}
        response = await self._request(
            "root", model, messages=messages, max_tokens=max_tokens, temperature=temperature, logprobs=True,
            extra_body={"prompt_logprobs": 0, "chat_template_kwargs": template})
        choice = self._first_choice(response, model)
        prompt_logprobs = (response.model_extra or {}).get("prompt_logprobs")
        if not prompt_logprobs or choice.logprobs is None or not choice.logprobs.content:
            self._fail("bad_response", "the response has no prompt or completion logprobs", model)
        try:
            prompt_ids = attach_token_ids(choice, prompt_logprobs)
        except (AttributeError, StopIteration, TypeError, ValueError) as exc:
            self._fail("bad_response", f"token ids could not be rebuilt ({exc!r})", model)
        if effort:
            if not thinking_on_prompt(prompt_ids):
                self._fail("thinking_off", "thinking was asked for, but the server rendered a prompt without it", model)
            if choice.model_extra is not None:
                choice.model_extra["reasoning_tokens"] = reasoning_tokens(response.usage)
        elif reasoning_tokens(response.usage) or not thinking_off_prompt(prompt_ids):
            self._fail("thinking_on", "the server rendered a thinking prompt or spent reasoning tokens", model)
        if usage is not None:
            usage.add("root", response.usage)
        return choice

    async def answer(self, messages: list[dict], *, model: str, usage: Usage | None = None, max_tokens: int,
                     temperature: float = ROOT_TEMPERATURE) -> Any:
        """One call with thinking off that returns the whole choice and counts its tokens as a root call: the
        direct-reading baseline (plan step 11), where the context is in the prompt instead of a REPL. No logprobs:
        nothing trains on it, and a prompt can be ~130K tokens. Raises RequestFailed."""
        response = await self._request(
            "direct", model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}})
        choice = self._first_choice(response, model)
        if reasoning_tokens(response.usage):
            self._fail("thinking_on", "a direct-reading call spent reasoning tokens", model)
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

    async def prompt_logprobs(self, messages: list[dict], *, model: str, thinking: str | None = None,
                              continue_final: bool = False) -> tuple[list[int], list[float | None]]:
        """Score a finished conversation under `model`: each prompt token's id and logprob (KL to base, spec §5.3).

        The conversation is sent as the prompt with `prompt_logprobs=0`, and one token is generated.
        The first token has no logprob, as in the token-id rebuild. Raises RequestFailed.

        For a turn sampled with root thinking (monitor.kl_to_base), `thinking` renders the prompt as that turn's was
        rendered, and `continue_final` renders the last message, the turn with its thinking, as the assistant's
        unfinished reply instead of adding a new one.
        """
        template = {"enable_thinking": True, "reasoning_effort": thinking} if thinking else {"enable_thinking": ENABLE_THINKING}
        extra = {"prompt_logprobs": 0, "chat_template_kwargs": template}
        if continue_final:
            extra.update(continue_final_message=True, add_generation_prompt=False)
        response = await self._request("kl", model, messages=messages, max_tokens=1, temperature=ROOT_TEMPERATURE,
                                       extra_body=extra)
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
        """One chat completion, paced and under the limiter, retried on 429, 5xx, timeouts and dropped connections."""
        self.stats.calls[kind] += 1
        tries = self.retry.tries
        for attempt in range(tries):
            retry_after = failure = status = cf_ray = None
            paced = await self.pacer.wait(model)  # every try, before it holds a slot
            self.stats.paced_s += paced  # after the await: `+= await` would read the total first and lose others' time
            self.stats.attempts += 1
            async with self.limiter.slot(model):
                try:
                    response = await self._client.chat.completions.create(model=model, timeout=self.retry.timeout_s,
                                                                          **kwargs)
                    self.spend[kind].add(kind, getattr(response, "usage", None))
                    return response
                except openai.APIStatusError as exc:
                    status, cf_ray = exc.status_code, _header(exc, "cf-ray")
                    self.stats.statuses[status] += 1
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
