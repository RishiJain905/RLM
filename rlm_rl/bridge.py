"""Bridge API v1: the one contract between the root model, the REPL sandbox and the host.

The training environment and the Part 2 harness both import this module (spec §1-2, H10).
It pins what the model can call in the REPL, the prompt text and its hash, the episode
limits per family (spec §12.3), the sub-call rules and the wire protocol, and it serves one
episode's sub-calls over a Unix socket.

Prompt v3 is the prompt for every family from production on: prompt v2 (spec §12.2, one system prompt
with each task's limits in its first message) plus Rishi's fixes after the step 11 baselines (2026-10-17):
code the model writes in its native tool-call syntax (`<function=python>...`) runs like a ```python
block, the "no code block" error says tool-call tags don't run here, and edit episodes get 30 turns
instead of 20. Prompt v2 is kept for the step 11 baselines, and prompt v1 (spec §2) for the second
seed, which replicates the 20-step run on it (decision E-g); every run records which version it used.

The REPL side of the protocol lives in repl_worker.py, which runs inside the sandbox and
can't import this module. tests/test_contract.py checks that the two agree.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

API_VERSION = "v1"
PROMPT_VERSION = "v3"
PROMPT_VERSIONS = ("v1", "v2", "v3")
# Versions that run code written in Qwen3.8's tool-call syntax (env.extract_code): in step 11 the base model wrote
# `<tool_call>` tags on 43% of its turns under prompt v2, whose system prompt lists the coding tools.
TOOL_CALL_CODE_VERSIONS = ("v3",)

# What the model sees in the REPL (spec §1.1), and the only requests the socket accepts (§1.4).
# `submit` never crosses the bridge: its value comes back with the cell result.
API_NAMES = ("context", "llm", "llm_batch", "submit")
WIRE_OPS = ("llm", "llm_batch")

# Episode limits for the read families (spec §4). Edit families use EDIT_LIMITS (spec §12.3); `limits(family)`
# picks. Prompt v1's system prompt quotes the read limits; prompt v2 puts them in each first message.
MAX_TURNS = 12
MAX_SUBCALLS = 30
ROOT_MAX_TOKENS = 3072
# What one cell's printed output shows the model. Raised from 2,000 for prompt v2 after the second seed
# (step 10 results §5, Rishi 2026-10-16): 40% of its cells were cut, the median cut cell printed 4,100
# characters, and a cut list led to the one suspicious audited win. 8,000 covers 78% of those cells. The
# REPL keeps this much; `limits(family, version)` gives each episode its own cap, which env.py applies.
OUTPUT_MAX_CHARS = 8000
OUTPUT_MAX_CHARS_V1 = 2000  # prompt v1 (the second seed) keeps the cap it ran with, and its system prompt says so
TRACEBACK_LINES = 6
CELL_TIMEOUT_S = 120

EDIT_PREFIX = "edit_"  # edit_real, edit_synth and their eval-only splits (spec §12)
EDIT_REPO_PATH = "/work/repo"


@dataclass(frozen=True)
class Limits:
    """One family's episode limits (spec §4, §12.3). Each episode records the limits it ran under."""

    max_turns: int
    max_subcalls: int
    root_max_tokens: int
    output_max_chars: int
    cell_timeout_s: int  # compute only: time blocked on the bridge doesn't count
    cpus: float
    memory_mb: int
    pids: int
    grader_timeout_s: int | None  # edit families only

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


READ_LIMITS = Limits(max_turns=MAX_TURNS, max_subcalls=MAX_SUBCALLS, root_max_tokens=ROOT_MAX_TOKENS,
                     output_max_chars=OUTPUT_MAX_CHARS, cell_timeout_s=CELL_TIMEOUT_S, cpus=1.0, memory_mb=512,
                     pids=64, grader_timeout_s=None)
EDIT_LIMITS = Limits(max_turns=20, max_subcalls=30, root_max_tokens=4096, output_max_chars=OUTPUT_MAX_CHARS,
                     cell_timeout_s=300, cpus=2.0, memory_mb=4096, pids=256, grader_timeout_s=600)
# Prompt v3's edit turns (Rishi, 2026-10-17): 56% of the base model's edit episodes ended unresolved at 20 turns in
# step 11, over code-editing-plan.md §9's 20% threshold.
EDIT_MAX_TURNS_V3 = 30


def is_edit(family: str) -> bool:
    return str(family).startswith(EDIT_PREFIX)


def limits(family: str, version: str = PROMPT_VERSION) -> Limits:
    """The limits an episode of this family runs under (spec §12.3): prompt v1's output cap for a v1 episode, and
    prompt v3's 30 edit turns."""
    lim = EDIT_LIMITS if is_edit(family) else READ_LIMITS
    if version == "v1":
        return dataclasses.replace(lim, output_max_chars=OUTPUT_MAX_CHARS_V1)
    if version == "v3" and is_edit(family):
        return dataclasses.replace(lim, max_turns=EDIT_MAX_TURNS_V3)
    return lim

# Sub-calls (spec §1.2).
SUBCALL_MAX_CHARS = 24_000
SUBCALL_TRUNCATION_MARK = "[prompt truncated]"
SUBCALL_TEMPERATURE = 0.3
SUBCALL_MAX_TOKENS = 1024
SUBCALL_ENABLE_THINKING = False

# Wire limits (mine): a larger request line is refused, and a larger submitted answer is an error in the REPL.
MAX_REQUEST_BYTES = 8 * 1024 * 1024
SUBMIT_MAX_BYTES = 100_000

# ----------------------------------------------------------------------------- prompt text v2 (spec §12.2)

SYSTEM_PROMPT = """You are a Recursive Language Model. You work on a task whose input is too large to read directly. The input is stored in a persistent Python REPL as the string variable `context`; for coding tasks, a repository is also checked out on disk.

Each turn, write a short plan and then exactly one ```python code block. The REPL runs it and shows you what it prints (truncated to 8000 characters). Variables persist between turns.

Available in the REPL:
- `context`: the task input (str)
- `llm(prompt: str) -> str`: ask a language model; it can read about 20,000 characters per call
- `llm_batch(prompts: list[str]) -> list[str]`: ask several prompts in parallel; each prompt counts as one sub-call
- `submit(answer)`: give your final answer and end the episode

The REPL has only the Python standard library.

Your turn and sub-call limits are given with the task. Look at the structure of the input before processing it, decompose the work, use code for exact filtering and arithmetic, and use sub-calls for judgments that need reading comprehension. Call `submit()` once you are confident.

For coding tasks, the REPL starts in the repository and also has these tools. They return values; print what you need to see.
- `read_file(path, start=1, n=40) -> str`: numbered lines of a file
- `find_files(pattern, path=".") -> list[str]`: files matching a glob such as "src/**/*.py"
- `grep(pattern, path=".", glob=None, ignore_case=False, files_only=False) -> list[str]`: regex matches as "path:line: text"
- `edit_file(path, old, new, replace_all=False) -> str`: replace exact text that occurs once (or every time with replace_all=True); raises otherwise
- `write_file(path, text) -> str`: create or overwrite a file
- `shell(cmd, timeout=120)`: run a shell command; the result has `.code` and `.out` and prints as the exit code and the end of the output
- `diff() -> str`: your changes so far, as they will be graded

Explore with code and print summaries, not whole files. Use `llm_batch` to read many files at once, and `llm` to draft or review a change. Run the relevant build or tests with `shell` before you submit. Your changes to the repository are what gets graded, by tests you cannot see, so do not change existing tests or test configuration. There is no network."""

FIRST_USER_TEMPLATE = """Question: {question}

`context` holds {n_chars:,} characters in {n_lines} lines. The first 3 lines are:
{first_3_lines}

You have at most {max_turns} turns and {max_subcalls} sub-calls."""

EDIT_FIRST_USER_TEMPLATE = """Task: {task}

The repository is at /work/repo ({language}, {n_files} files). `context` holds this task and the repository's file list with sizes. Run its tests with: {test_hint}

When your change is ready, call submit() with a one-line summary. You have at most {max_turns} turns and {max_subcalls} sub-calls, and each cell may run for {cell_timeout_s} seconds."""

# Observation after each turn (spec §2.3), assembled by `observation()`. The same in v1 and v2.
OBS_OUTPUT = "REPL output:\n{stdout}"
OBS_NOTHING_PRINTED = "REPL output: (nothing printed)"
OBS_TRUNCATED = "\n[... {n} more characters truncated]"
OBS_ERROR = "Error:\n{error}"
OBS_COUNTERS = "({turns_left} turns left, {subcalls_left} sub-calls left)"
OBS_LAST_TURN = "Your next turn is the last one: it must call submit() with your best answer."
NO_CODE_BLOCK_ERROR = "No ```python code block found. Reply with exactly one ```python block."
# Prompt v3 (step 11): under v2 the base model often answered in tool-call tags, then repeated them after the error.
NO_CODE_BLOCK_ERROR_V3 = ("No ```python code block found. Reply with exactly one ```python block. Tool-call tags "
                          "don't run here: call llm, llm_batch, submit and the tools as Python functions inside "
                          "the block.")

# What the model is told after a timeout or crash (spec §4, §12.2): v2 quotes the cell limit as a value, and edit
# episodes add one sentence.
CELL_TIMEOUT_TEMPLATE = (
    "The cell timed out after {cell_timeout_s} seconds of compute. The REPL was restarted and all variables were "
    "lost; `context` and your sub-call budget are unchanged."
)
CELL_CRASH_TEMPLATE = (
    "The REPL crashed ({{reason}}). It was restarted and all variables were lost; "
    "`context` and your sub-call budget are unchanged."
)
EDIT_FILES_UNCHANGED = " Your files in /work/repo are unchanged."
CELL_NOT_RUN = " This cell was not run."

# ----------------------------------------------------------------------------- prompt text v1 (spec §2), second seed only

SYSTEM_PROMPT_V1 = """You are a Recursive Language Model. You answer a question about a long text that you cannot see directly. The text is stored in a persistent Python REPL as the string variable `context`.

Each turn, write a short plan and then exactly one ```python code block. The REPL runs it and shows you what it prints (truncated to 2000 characters). Variables persist between turns.

Available in the REPL:
- `context`: the full text (str)
- `llm(prompt: str) -> str`: ask a language model; it can read about 20,000 characters per call
- `llm_batch(prompts: list[str]) -> list[str]`: ask several prompts in parallel; each prompt counts as one sub-call
- `submit(answer)`: give your final answer and end the episode

Only the Python standard library is available.

You have at most 12 turns and 30 sub-calls in total. Look at the structure of `context` before processing it, decompose the work, use code for exact filtering and arithmetic, and use sub-calls for judgments that need reading comprehension. Call `submit()` once you are confident."""

FIRST_USER_TEMPLATE_V1 = """Question: {question}

`context` holds {n_chars:,} characters in {n_lines} lines. The first 3 lines are:
{first_3_lines}"""

CELL_TIMEOUT_ERROR = (
    "The cell timed out after 120 seconds of compute. The REPL was restarted and all variables were lost; "
    "`context` and your sub-call budget are unchanged."
)
CELL_CRASH_ERROR = (
    "The REPL crashed ({reason}). It was restarted and all variables were lost; "
    "`context` and your sub-call budget are unchanged."
)

_OBSERVATION = [OBS_OUTPUT, OBS_NOTHING_PRINTED, OBS_TRUNCATED, OBS_ERROR, OBS_COUNTERS, OBS_LAST_TURN,
                NO_CODE_BLOCK_ERROR]
PROMPTS = {
    "v1": {"system": SYSTEM_PROMPT_V1, "first_user": FIRST_USER_TEMPLATE_V1,
           "observation": "\n".join(_OBSERVATION + [CELL_TIMEOUT_ERROR, CELL_CRASH_ERROR])},
    "v2": {"system": SYSTEM_PROMPT, "first_user": FIRST_USER_TEMPLATE + "\n\n" + EDIT_FIRST_USER_TEMPLATE,
           "observation": "\n".join(_OBSERVATION + [CELL_TIMEOUT_TEMPLATE, CELL_CRASH_TEMPLATE, EDIT_FILES_UNCHANGED,
                                                    CELL_NOT_RUN])},
    # v2's text with v3's "no code block" error; "code" and "edit_max_turns" put v3's other two changes in its hash
    "v3": {"system": SYSTEM_PROMPT, "first_user": FIRST_USER_TEMPLATE + "\n\n" + EDIT_FIRST_USER_TEMPLATE,
           "observation": "\n".join([NO_CODE_BLOCK_ERROR_V3 if t == NO_CODE_BLOCK_ERROR else t for t in _OBSERVATION]
                                    + [CELL_TIMEOUT_TEMPLATE, CELL_CRASH_TEMPLATE, EDIT_FILES_UNCHANGED, CELL_NOT_RUN]),
           "code": "```python blocks, and <function=python> tool-call blocks", "edit_max_turns": str(EDIT_MAX_TURNS_V3)},
}
PROMPT_HASHES = {v: hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest() for v, texts in PROMPTS.items()}
PROMPT_TEXTS = PROMPTS[PROMPT_VERSION]
PROMPT_HASH = PROMPT_HASHES[PROMPT_VERSION]


def _check_version(version: str) -> None:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version {version!r}")


def system_prompt(version: str = PROMPT_VERSION) -> str:
    _check_version(version)
    return PROMPTS[version]["system"]


def first_user_message(question: str, context: str, version: str = PROMPT_VERSION,
                       limits: Limits = READ_LIMITS) -> str:
    """A read family's first message (v1: spec §2.2; v2: the same plus one limits line)."""
    _check_version(version)
    lines = context.splitlines()
    fields = dict(question=question, n_chars=len(context), n_lines=len(lines), first_3_lines="\n".join(lines[:3]))
    if version == "v1":
        return FIRST_USER_TEMPLATE_V1.format(**fields)
    return FIRST_USER_TEMPLATE.format(**fields, max_turns=limits.max_turns, max_subcalls=limits.max_subcalls)


def edit_first_message(task: str, *, language: str, n_files: int, test_hint: str, limits: Limits = EDIT_LIMITS) -> str:
    """An edit family's first message (spec §12.2). `test_hint` runs the visible tests, never a hidden test's name."""
    return EDIT_FIRST_USER_TEMPLATE.format(task=task, language=language, n_files=n_files, test_hint=test_hint,
                                           max_turns=limits.max_turns, max_subcalls=limits.max_subcalls,
                                           cell_timeout_s=limits.cell_timeout_s)


def initial_messages(question: str, context: str, version: str = PROMPT_VERSION, limits: Limits = READ_LIMITS,
                     edit: dict | None = None) -> list[dict]:
    """The system prompt and first user message every episode starts from.

    `edit` makes it an edit episode's first message: {"language", "n_files", "test_hint"}, with `question` the task.
    """
    if edit is not None:
        if version == "v1":
            raise ValueError("edit tasks need prompt v2 (spec §12.2)")
        first = edit_first_message(question, language=edit["language"], n_files=edit["n_files"],
                                   test_hint=edit["test_hint"], limits=limits)
    else:
        first = first_user_message(question, context, version, limits)
    return [{"role": "system", "content": system_prompt(version)}, {"role": "user", "content": first}]


@dataclass(frozen=True)
class CellMessages:
    """What the sandbox tells the model after a timeout or a crash. `crash` has a {reason} field."""

    timeout: str
    crash: str
    not_run: str = CELL_NOT_RUN


def cell_messages(version: str = PROMPT_VERSION, limits: Limits = READ_LIMITS, edit: bool = False) -> CellMessages:
    _check_version(version)
    if version == "v1":
        return CellMessages(CELL_TIMEOUT_ERROR, CELL_CRASH_ERROR)
    tail = EDIT_FILES_UNCHANGED if edit else ""
    return CellMessages(CELL_TIMEOUT_TEMPLATE.format(cell_timeout_s=limits.cell_timeout_s) + tail,
                        CELL_CRASH_TEMPLATE.format() + tail)


def no_code_error(version: str = PROMPT_VERSION) -> str:
    """What a turn without code gets back (v3 adds that tool-call tags don't run)."""
    _check_version(version)
    return NO_CODE_BLOCK_ERROR_V3 if version == "v3" else NO_CODE_BLOCK_ERROR


def observation(stdout: str, stdout_chars: int, error: str | None, turns_left: int, subcalls_left: int,
                max_chars: int = OUTPUT_MAX_CHARS) -> str:
    """The user message after a turn. `stdout_chars` is how much the cell printed before truncation."""
    out = stdout[:max_chars]
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
