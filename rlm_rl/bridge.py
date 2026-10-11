"""Bridge API v1: the one contract between the root model, the REPL sandbox and the host.

The training environment and the Part 2 harness both import this module (spec §1-2, H10).
It pins what the model can call in the REPL, the prompt text and its hash, the sub-call
rules and the wire protocol, and it serves one episode's sub-calls over a Unix socket.

The REPL side of the protocol lives in repl_worker.py, which runs inside the sandbox and
can't import this module. tests/test_contract.py checks that the two agree.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

API_VERSION = "v1"
PROMPT_VERSION = "v1"

# What the model sees in the REPL (spec §1.1), and the only requests the socket accepts (§1.4).
# `submit` never crosses the bridge: its value comes back with the cell result.
API_NAMES = ("context", "llm", "llm_batch", "submit")
WIRE_OPS = ("llm", "llm_batch")

# Episode limits (spec §4). The system prompt quotes several of them.
MAX_TURNS = 12
MAX_SUBCALLS = 30
ROOT_MAX_TOKENS = 3072
OUTPUT_MAX_CHARS = 2000
TRACEBACK_LINES = 6
CELL_TIMEOUT_S = 120

# Sub-calls (spec §1.2).
SUBCALL_MAX_CHARS = 24_000
SUBCALL_TRUNCATION_MARK = "[prompt truncated]"
SUBCALL_TEMPERATURE = 0.3
SUBCALL_MAX_TOKENS = 1024
SUBCALL_ENABLE_THINKING = False

# Wire limits (mine): a larger request line is refused, and a larger submitted answer is an error in the REPL.
MAX_REQUEST_BYTES = 8 * 1024 * 1024
SUBMIT_MAX_BYTES = 100_000

# ----------------------------------------------------------------------------- prompt text v1

SYSTEM_PROMPT = """You are a Recursive Language Model. You answer a question about a long text that you cannot see directly. The text is stored in a persistent Python REPL as the string variable `context`.

Each turn, write a short plan and then exactly one ```python code block. The REPL runs it and shows you what it prints (truncated to 2000 characters). Variables persist between turns.

Available in the REPL:
- `context`: the full text (str)
- `llm(prompt: str) -> str`: ask a language model; it can read about 20,000 characters per call
- `llm_batch(prompts: list[str]) -> list[str]`: ask several prompts in parallel; each prompt counts as one sub-call
- `submit(answer)`: give your final answer and end the episode

Only the Python standard library is available.

You have at most 12 turns and 30 sub-calls in total. Look at the structure of `context` before processing it, decompose the work, use code for exact filtering and arithmetic, and use sub-calls for judgments that need reading comprehension. Call `submit()` once you are confident."""

FIRST_USER_TEMPLATE = """Question: {question}

`context` holds {n_chars:,} characters in {n_lines} lines. The first 3 lines are:
{first_3_lines}"""

# Observation after each turn (spec §2.3), assembled by `observation()`.
OBS_OUTPUT = "REPL output:\n{stdout}"
OBS_NOTHING_PRINTED = "REPL output: (nothing printed)"
OBS_TRUNCATED = "\n[... {n} more characters truncated]"
OBS_ERROR = "Error:\n{error}"
OBS_COUNTERS = "({turns_left} turns left, {subcalls_left} sub-calls left)"
OBS_LAST_TURN = "Your next turn is the last one: it must call submit() with your best answer."
NO_CODE_BLOCK_ERROR = "No ```python code block found. Reply with exactly one ```python block."

# What the model is told after a timeout or crash (spec §4). New in step 3; not yet quoted in the spec.
CELL_TIMEOUT_ERROR = (
    "The cell timed out after 120 seconds of compute. The REPL was restarted and all variables were lost; "
    "`context` and your sub-call budget are unchanged."
)
CELL_CRASH_ERROR = (
    "The REPL crashed ({reason}). It was restarted and all variables were lost; "
    "`context` and your sub-call budget are unchanged."
)

PROMPT_TEXTS = {
    "system": SYSTEM_PROMPT,
    "first_user": FIRST_USER_TEMPLATE,
    "observation": "\n".join([OBS_OUTPUT, OBS_NOTHING_PRINTED, OBS_TRUNCATED, OBS_ERROR, OBS_COUNTERS, OBS_LAST_TURN,
                              NO_CODE_BLOCK_ERROR, CELL_TIMEOUT_ERROR, CELL_CRASH_ERROR]),
}
PROMPT_HASH = hashlib.sha256(json.dumps(PROMPT_TEXTS, sort_keys=True).encode()).hexdigest()


def first_user_message(question: str, context: str) -> str:
    lines = context.splitlines()
    return FIRST_USER_TEMPLATE.format(question=question, n_chars=len(context), n_lines=len(lines),
                                      first_3_lines="\n".join(lines[:3]))


def initial_messages(question: str, context: str) -> list[dict]:
    """The system prompt and first user message every episode starts from."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_user_message(question, context)}]


def observation(stdout: str, stdout_chars: int, error: str | None, turns_left: int, subcalls_left: int) -> str:
    """The user message after a turn. `stdout_chars` is how much the cell printed before truncation."""
    out = stdout[:OUTPUT_MAX_CHARS]
    if stdout_chars > len(out):
        out += OBS_TRUNCATED.format(n=stdout_chars - len(out))
    parts = [OBS_OUTPUT.format(stdout=out) if out.strip() else OBS_NOTHING_PRINTED]
    if error:
        parts.append(OBS_ERROR.format(error=error))
    parts.append(OBS_COUNTERS.format(turns_left=turns_left, subcalls_left=subcalls_left))
    if turns_left == 1:
        parts.append(OBS_LAST_TURN)
    return "\n".join(parts)


def truncate_prompt(prompt: str) -> str:
    if len(prompt) > SUBCALL_MAX_CHARS:
        return prompt[:SUBCALL_MAX_CHARS] + "\n" + SUBCALL_TRUNCATION_MARK
    return prompt


def budget_exhausted_message(max_subcalls: int = MAX_SUBCALLS) -> str:
    return f"sub-call budget exhausted ({max_subcalls} total)"


def clean_text(text: str) -> str:
    """Replace lone surrogates, which the model can print but no API request can encode."""
    return text.encode("utf-8", "replace").decode("utf-8")


def valid_answer(value: Any) -> bool:
    """Whether a submitted value has the shape spec §1.2 allows: int, float, str, or a list of those."""
    scalar = (int, float, str)
    if isinstance(value, bool):
        return False
    if isinstance(value, scalar):
        return True
    return isinstance(value, list) and all(isinstance(v, scalar) and not isinstance(v, bool) for v in value)


# ----------------------------------------------------------------------------- host side


class InfraError(Exception):
    """A failure outside the model's control. The episode stops, is flagged infra_error and never reaches training (H5)."""


class SubcallError(Exception):
    """A sub-call failure the model should see as a RuntimeError in the REPL, such as a request the API rejects."""


Complete = Callable[[str], Awaitable[str]]
"""Serves one sub-call: takes a prompt, returns the model's text. Raises SubcallError or InfraError."""


@dataclass
class SubcallRecord:
    prompt: str
    output: str | None
    error: str | None
    seconds: float


class EpisodeBridge:
    """Serves one episode's sub-calls on its own Unix socket (spec §1.2-1.4).

    The socket accepts `{"op": "llm" | "llm_batch", "prompts": [...]}` one JSON object per line
    and nothing else. The budget is enforced here, on the host, since the REPL is untrusted.
    Only host-side failures set `infra_error`; nothing the REPL sends can.
    """

    def __init__(self, socket_path: str, complete: Complete, max_subcalls: int = MAX_SUBCALLS):
        self.socket_path = socket_path
        self.max_subcalls = max_subcalls
        self.subcalls_used = 0
        self.records: list[SubcallRecord] = []
        self.rejected: list[str] = []
        self.infra_error: str | None = None
        self.infra_event = asyncio.Event()
        self._complete = complete
        self._server: asyncio.AbstractServer | None = None
        self._inflight = 0
        self._blocked_since = 0.0
        self._blocked_total = 0.0

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(self._serve, path=self.socket_path, limit=MAX_REQUEST_BYTES)
        os.chmod(self.socket_path, 0o666)  # the sandbox user is not the host user

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

    @property
    def serving(self) -> bool:
        return self._server is not None and self._server.is_serving()

    def blocked_seconds(self) -> float:
        """Wall time during which at least one sub-call was in flight. It doesn't count toward the cell timeout."""
        if self._inflight:
            return self._blocked_total + time.monotonic() - self._blocked_since
        return self._blocked_total

    def flag_infra(self, reason: str) -> None:
        if self.infra_error is None:
            self.infra_error = reason
            self.infra_event.set()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    line = await reader.readline()
                except ValueError:  # the line ran past MAX_REQUEST_BYTES
                    self.rejected.append("request too large")
                    await self._write(writer, {"error": f"request too large (over {MAX_REQUEST_BYTES:,} bytes)"})
                    break
                if not line:
                    break
                await self._write(writer, await self._handle(line))
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    @staticmethod
    async def _write(writer: asyncio.StreamWriter, reply: dict) -> None:
        writer.write(json.dumps(reply).encode() + b"\n")
        await writer.drain()

    def _reject(self, reason: str) -> dict:
        self.rejected.append(reason)
        return {"error": reason}

    async def _handle(self, line: bytes) -> dict:
        try:
            message = json.loads(line)
        except ValueError:
            return self._reject("malformed request")
        if not isinstance(message, dict) or message.get("op") not in WIRE_OPS:
            return self._reject("unknown request")
        prompts = message.get("prompts")
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            return self._reject("prompts must be a list of strings")
        if message["op"] == "llm" and len(prompts) != 1:
            return self._reject("llm takes exactly one prompt")
        if self.infra_error:
            return {"error": "sub-call failed for an infrastructure reason; this episode will be discarded"}
        if self.subcalls_used + len(prompts) > self.max_subcalls:
            return {"error": budget_exhausted_message(self.max_subcalls)}
        self.subcalls_used += len(prompts)
        if not prompts:
            return {"outputs": []}

        if self._inflight == 0:
            self._blocked_since = time.monotonic()
        self._inflight += 1
        try:
            results = await asyncio.gather(*(self._one(p) for p in prompts), return_exceptions=True)
        finally:
            self._inflight -= 1
            if self._inflight == 0:
                self._blocked_total += time.monotonic() - self._blocked_since

        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, SubcallError):
                self.flag_infra(f"sub-call: {result!r}" if not isinstance(result, InfraError) else str(result))
                return {"error": "sub-call failed for an infrastructure reason; this episode will be discarded"}
        for result in results:
            if isinstance(result, SubcallError):
                return {"error": f"sub-call failed: {result}"}
        return {"outputs": results}

    async def _one(self, prompt: str) -> str:
        prompt = truncate_prompt(clean_text(prompt))
        started = time.monotonic()
        try:
            output = await self._complete(prompt)
            if not isinstance(output, str):
                raise InfraError(f"sub-call returned {type(output).__name__}, not str")
        except BaseException as exc:
            self.records.append(SubcallRecord(prompt, None, repr(exc), time.monotonic() - started))
            raise
        self.records.append(SubcallRecord(prompt, output, None, time.monotonic() - started))
        return output
