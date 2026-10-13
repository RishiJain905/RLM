"""Bridge API v1: the pinned prompt text, and the host side of the wire protocol (spec §1-2)."""

import asyncio
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

from rlm_rl import bridge
from rlm_rl.bridge import EpisodeBridge, InfraError, SubcallError

# Spec §12.2 (prompt v2), copied from the spec. A change here is a change to what the model is trained on.
SPEC_SYSTEM_PROMPT_V2 = """You are a Recursive Language Model. You work on a task whose input is too large to read directly. The input is stored in a persistent Python REPL as the string variable `context`; for coding tasks, a repository is also checked out on disk.

Each turn, write a short plan and then exactly one ```python code block. The REPL runs it and shows you what it prints (truncated to 2000 characters). Variables persist between turns.

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

SPEC_EDIT_FIRST_V2 = """Task: {task}

The repository is at /work/repo ({language}, {n_files} files). `context` holds this task and the repository's file list with sizes. Run its tests with: {test_hint}

When your change is ready, call submit() with a one-line summary. You have at most {max_turns} turns and {max_subcalls} sub-calls, and each cell may run for {cell_timeout_s} seconds."""

# Spec §2.1 (prompt v1), which the second seed still runs on (decision E-g).
SPEC_SYSTEM_PROMPT = """You are a Recursive Language Model. You answer a question about a long text that you cannot see directly. The text is stored in a persistent Python REPL as the string variable `context`.

Each turn, write a short plan and then exactly one ```python code block. The REPL runs it and shows you what it prints (truncated to 2000 characters). Variables persist between turns.

Available in the REPL:
- `context`: the full text (str)
- `llm(prompt: str) -> str`: ask a language model; it can read about 20,000 characters per call
- `llm_batch(prompts: list[str]) -> list[str]`: ask several prompts in parallel; each prompt counts as one sub-call
- `submit(answer)`: give your final answer and end the episode

Only the Python standard library is available.

You have at most 12 turns and 30 sub-calls in total. Look at the structure of `context` before processing it, decompose the work, use code for exact filtering and arithmetic, and use sub-calls for judgments that need reading comprehension. Call `submit()` once you are confident."""


class PromptTextTest(unittest.TestCase):
    def test_system_prompt_matches_spec(self):
        self.assertEqual(bridge.PROMPT_VERSION, "v2")
        self.assertEqual(bridge.SYSTEM_PROMPT, SPEC_SYSTEM_PROMPT_V2)
        self.assertEqual(bridge.SYSTEM_PROMPT_V1, SPEC_SYSTEM_PROMPT)
        self.assertEqual(bridge.system_prompt("v1"), SPEC_SYSTEM_PROMPT)
        self.assertEqual(bridge.EDIT_FIRST_USER_TEMPLATE, SPEC_EDIT_FIRST_V2)

    def test_prompt_quotes_the_pinned_limits(self):
        v1 = bridge.SYSTEM_PROMPT_V1
        self.assertIn(f"at most {bridge.MAX_TURNS} turns and {bridge.MAX_SUBCALLS} sub-calls", v1)
        self.assertIn(f"truncated to {bridge.OUTPUT_MAX_CHARS} characters", bridge.SYSTEM_PROMPT)
        self.assertNotIn("at most 12 turns", bridge.SYSTEM_PROMPT)  # v2 gives the limits with each task
        self.assertIn(f"{bridge.CELL_TIMEOUT_S} seconds", bridge.CELL_TIMEOUT_ERROR)
        self.assertEqual(bridge.budget_exhausted_message(), "sub-call budget exhausted (30 total)")

    def test_limits_per_family(self):
        """Spec §12.3."""
        read, edit = bridge.limits("aggregation"), bridge.limits("edit_real")
        self.assertEqual((read.max_turns, read.max_subcalls, read.root_max_tokens, read.cell_timeout_s, read.cpus,
                          read.memory_mb, read.pids, read.grader_timeout_s), (12, 30, 3072, 120, 1.0, 512, 64, None))
        self.assertEqual((edit.max_turns, edit.max_subcalls, edit.root_max_tokens, edit.cell_timeout_s, edit.cpus,
                          edit.memory_mb, edit.pids, edit.grader_timeout_s), (20, 30, 4096, 300, 2.0, 4096, 256, 600))
        for family in ("edit_synth", "edit_synth_heldout", "edit_real_select", "edit_real_test"):
            self.assertIs(bridge.limits(family), bridge.EDIT_LIMITS)
        self.assertEqual(read.output_max_chars, edit.output_max_chars)

    def test_cell_messages_quote_the_limit(self):
        """Spec §12.2: the timeout message quotes the cell limit as a value; edit episodes add one sentence."""
        read = bridge.cell_messages("v2", bridge.READ_LIMITS)
        edit = bridge.cell_messages("v2", bridge.EDIT_LIMITS, edit=True)
        self.assertIn("after 120 seconds of compute", read.timeout)
        self.assertIn("after 300 seconds of compute", edit.timeout)
        self.assertTrue(edit.timeout.endswith(" Your files in /work/repo are unchanged."))
        self.assertTrue(edit.crash.format(reason="exit code 1").endswith("Your files in /work/repo are unchanged."))
        self.assertNotIn("/work/repo", read.timeout + read.crash)
        self.assertEqual(bridge.cell_messages("v1"), bridge.CellMessages(bridge.CELL_TIMEOUT_ERROR, bridge.CELL_CRASH_ERROR))

    def test_every_api_name_is_in_the_prompt(self):
        self.assertEqual(bridge.API_NAMES, ("context", "llm", "llm_batch", "submit"))
        for prompt in (bridge.SYSTEM_PROMPT, bridge.SYSTEM_PROMPT_V1):
            for name in bridge.API_NAMES:
                self.assertIn(f"`{name}", prompt)
            self.assertNotIn("llm_query", prompt)

    def test_hash_covers_all_three_texts(self):
        import hashlib

        self.assertEqual(len(bridge.PROMPT_HASH), 64)
        self.assertEqual(bridge.PROMPT_HASH, bridge.PROMPT_HASHES["v2"])
        self.assertNotEqual(bridge.PROMPT_HASHES["v1"], bridge.PROMPT_HASHES["v2"])
        for version, texts in bridge.PROMPTS.items():
            for key in texts:
                changed = dict(texts, **{key: texts[key] + " "})
                self.assertNotEqual(hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).hexdigest(),
                                    bridge.PROMPT_HASHES[version])

    def test_first_user_message(self):
        context = "a,b\nline 2\nline 3\nline 4"
        head = (f"Question: How many?\n\n`context` holds {len(context):,} characters in 4 lines. The first 3 lines are:\n"
                "a,b\nline 2\nline 3")
        self.assertEqual(bridge.first_user_message("How many?", context, "v1"), head)
        self.assertEqual(bridge.first_user_message("How many?", context),
                         head + "\n\nYou have at most 12 turns and 30 sub-calls.")
        self.assertIn("12,345 characters", bridge.first_user_message("q", "x" * 12_345))
        messages = bridge.initial_messages("q", context)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIs(messages[0]["content"], bridge.SYSTEM_PROMPT)
        self.assertIs(bridge.initial_messages("q", context, "v1")[0]["content"], bridge.SYSTEM_PROMPT_V1)

    def test_edit_first_message(self):
        messages = bridge.initial_messages("Fix the parser.", "ignored", edit={"language": "go", "n_files": 41,
                                                                                "test_hint": "go test ./..."},
                                           limits=bridge.EDIT_LIMITS)
        self.assertEqual(messages[1]["content"], SPEC_EDIT_FIRST_V2.format(
            task="Fix the parser.", language="go", n_files=41, test_hint="go test ./...", max_turns=20,
            max_subcalls=30, cell_timeout_s=300))
        with self.assertRaises(ValueError):
            bridge.initial_messages("t", "c", "v1", edit={"language": "go", "n_files": 1, "test_hint": "x"})

    def test_observation(self):
        self.assertEqual(bridge.observation("", 0, None, 5, 30), "REPL output: (nothing printed)\n(5 turns left, 30 sub-calls left)")
        self.assertEqual(bridge.observation("  \n", 3, None, 5, 30).splitlines()[0], "REPL output: (nothing printed)")
        self.assertEqual(bridge.observation("hi\n", 3, "NameError: x", 1, 2),
                         "REPL output:\nhi\n\nError:\nNameError: x\n(1 turns left, 2 sub-calls left)\n"
                         "Your next turn is the last one: it must call submit() with your best answer.")
        long = bridge.observation("x" * 2000, 2500, None, 3, 30)
        self.assertTrue(long.startswith("REPL output:\n" + "x" * 2000 + "\n[... 500 more characters truncated]"))

    def test_truncate_prompt(self):
        self.assertEqual(bridge.truncate_prompt("abc"), "abc")
        self.assertEqual(bridge.truncate_prompt("x" * 24_000), "x" * 24_000)
        self.assertEqual(bridge.truncate_prompt("x" * 24_001), "x" * 24_000 + "\n[prompt truncated]")

    def test_valid_answer(self):
        for ok in (3, 2.5, "host-7", [], [1, "a", 2.0]):
            self.assertTrue(bridge.valid_answer(ok), ok)
        for bad in (None, True, {"a": 1}, [[1]], [None], [False]):
            self.assertFalse(bridge.valid_answer(bad), bad)


async def _echo(prompt: str) -> str:
    return f"echo:{prompt}"


@unittest.skipIf(sys.platform == "win32" or not hasattr(socket, "AF_UNIX"), "Unix sockets: run inside WSL2")
class EpisodeBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "bridge.sock")
        self.calls: list[str] = []
        self.behaviour = _echo
        self.bridge = EpisodeBridge(self.path, self._complete, max_subcalls=30)
        await self.bridge.start()

    async def asyncTearDown(self):
        await self.bridge.close()
        self._tmp.cleanup()

    async def _complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return await self.behaviour(prompt)

    async def send(self, payload) -> dict:
        reader, writer = await asyncio.open_unix_connection(self.path, limit=bridge.MAX_REQUEST_BYTES * 2)
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        writer.write(data + b"\n")
        try:
            await writer.drain()
        except ConnectionError:  # the host stops reading an over-long request; its reply is already queued
            pass
        line = await reader.readline()
        writer.close()
        return json.loads(line)

    async def test_llm(self):
        self.assertEqual(await self.send({"op": "llm", "prompts": ["hi"]}), {"outputs": ["echo:hi"]})
        self.assertEqual(self.bridge.subcalls_used, 1)
        self.assertEqual(self.bridge.records[0].output, "echo:hi")

    async def test_llm_batch_keeps_input_order(self):
        async def slow_first(prompt):
            await asyncio.sleep(0.2 if prompt == "a" else 0.0)
            return prompt.upper()
        self.behaviour = slow_first
        self.assertEqual(await self.send({"op": "llm_batch", "prompts": ["a", "b", "c"]}), {"outputs": ["A", "B", "C"]})
        self.assertEqual(self.bridge.subcalls_used, 3)

    async def test_budget_is_enforced_on_the_host(self):
        self.bridge.subcalls_used = 28
        reply = await self.send({"op": "llm_batch", "prompts": ["a", "b", "c"]})
        self.assertEqual(reply, {"error": "sub-call budget exhausted (30 total)"})
        self.assertEqual(self.bridge.subcalls_used, 28)  # an over-budget call uses nothing
        self.assertEqual(await self.send({"op": "llm_batch", "prompts": ["a", "b"]}), {"outputs": ["echo:a", "echo:b"]})
        self.assertEqual((await self.send({"op": "llm", "prompts": ["c"]}))["error"], "sub-call budget exhausted (30 total)")
        self.assertEqual(len(self.calls), 2)

    async def test_long_prompts_are_cut(self):
        await self.send({"op": "llm", "prompts": ["y" * 30_000]})
        self.assertEqual(self.calls[0], "y" * 24_000 + "\n[prompt truncated]")

    async def test_only_llm_and_llm_batch_are_accepted(self):
        for payload in ({"op": "submit", "prompts": ["1"]}, {"op": "shell", "prompts": ["ls"]}, {"prompts": ["x"]},
                        ["llm", "x"], {"op": "llm", "prompts": ["a", "b"]}, {"op": "llm", "prompts": [1]},
                        {"op": "llm_batch", "prompts": "abc"}):
            self.assertIn("error", await self.send(payload), payload)
        self.assertIn("error", await self.send(b"not json"))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.bridge.subcalls_used, 0)
        self.assertEqual(len(self.bridge.rejected), 8)
        self.assertIsNone(self.bridge.infra_error)

    async def test_oversized_request_is_refused(self):
        reply = await self.send({"op": "llm", "prompts": ["z" * (bridge.MAX_REQUEST_BYTES + 10)]})
        self.assertIn("request too large", reply["error"])
        self.assertEqual(self.calls, [])

    async def test_subcall_error_is_shown_to_the_model(self):
        async def rejected(prompt):
            raise SubcallError("the API rejected the request (400)")
        self.behaviour = rejected
        reply = await self.send({"op": "llm", "prompts": ["x"]})
        self.assertEqual(reply, {"error": "sub-call failed: the API rejected the request (400)"})
        self.assertEqual(self.bridge.subcalls_used, 1)
        self.assertIsNone(self.bridge.infra_error)

    async def test_infra_error_flags_the_episode(self):
        async def rate_limited(prompt):
            raise InfraError("429 after 6 retries")
        self.behaviour = rate_limited
        reply = await self.send({"op": "llm_batch", "prompts": ["x", "y"]})
        self.assertIn("infrastructure", reply["error"])
        self.assertNotIn("outputs", reply)  # never a fake answer (spec §1.2)
        self.assertEqual(self.bridge.infra_error, "429 after 6 retries")
        self.assertTrue(self.bridge.infra_event.is_set())
        self.behaviour = _echo
        self.assertIn("infrastructure", (await self.send({"op": "llm", "prompts": ["z"]}))["error"])

    async def test_unexpected_host_failure_is_infra(self):
        async def broken(prompt):
            raise ValueError("bug in the client")
        async def not_text(prompt):
            return None
        for behaviour in (broken, not_text):
            bridge_ = EpisodeBridge(self.path + "2", behaviour)
            await bridge_.start()
            reader, writer = await asyncio.open_unix_connection(self.path + "2")
            writer.write(b'{"op": "llm", "prompts": ["x"]}\n')
            reply = json.loads(await reader.readline())
            writer.close()
            await bridge_.close()
            self.assertIn("infrastructure", reply["error"])
            self.assertIsNotNone(bridge_.infra_error)

    async def test_time_blocked_on_subcalls_is_measured(self):
        async def slow(prompt):
            await asyncio.sleep(0.3)
            return "ok"
        self.behaviour = slow
        before = self.bridge.blocked_seconds()
        await asyncio.gather(self.send({"op": "llm", "prompts": ["a"]}), self.send({"op": "llm", "prompts": ["b"]}))
        blocked = self.bridge.blocked_seconds() - before
        self.assertGreaterEqual(blocked, 0.29)
        self.assertLess(blocked, 0.6)  # two overlapping calls block once, not twice

    async def test_lone_surrogates_cannot_cause_an_infra_error(self):
        reply = await self.send(b'{"op": "llm", "prompts": ["a\\ud800b"]}')
        self.assertEqual(reply, {"outputs": ["echo:a?b"]})
        self.assertIsNone(self.bridge.infra_error)


if __name__ == "__main__":
    unittest.main()
