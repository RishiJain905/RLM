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

# Spec §2.1, copied from the approved spec page. A change here is a change to what the model was trained on.
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
        self.assertEqual(bridge.SYSTEM_PROMPT, SPEC_SYSTEM_PROMPT)

    def test_prompt_quotes_the_pinned_limits(self):
        self.assertIn(f"at most {bridge.MAX_TURNS} turns and {bridge.MAX_SUBCALLS} sub-calls", bridge.SYSTEM_PROMPT)
        self.assertIn(f"truncated to {bridge.OUTPUT_MAX_CHARS} characters", bridge.SYSTEM_PROMPT)
        self.assertIn(f"{bridge.CELL_TIMEOUT_S} seconds", bridge.CELL_TIMEOUT_ERROR)
        self.assertEqual(bridge.budget_exhausted_message(), "sub-call budget exhausted (30 total)")

    def test_every_api_name_is_in_the_prompt(self):
        self.assertEqual(bridge.API_NAMES, ("context", "llm", "llm_batch", "submit"))
        for name in bridge.API_NAMES:
            self.assertIn(f"`{name}", bridge.SYSTEM_PROMPT)
        self.assertNotIn("llm_query", bridge.SYSTEM_PROMPT)

    def test_hash_covers_all_three_texts(self):
        self.assertEqual(len(bridge.PROMPT_HASH), 64)
        for key in bridge.PROMPT_TEXTS:
            changed = dict(bridge.PROMPT_TEXTS, **{key: bridge.PROMPT_TEXTS[key] + " "})
            import hashlib
            self.assertNotEqual(hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).hexdigest(), bridge.PROMPT_HASH)

    def test_first_user_message(self):
        context = "a,b\nline 2\nline 3\nline 4"
        self.assertEqual(
            bridge.first_user_message("How many?", context),
            f"Question: How many?\n\n`context` holds {len(context):,} characters in 4 lines. The first 3 lines are:\n"
            "a,b\nline 2\nline 3")
        self.assertIn("12,345 characters", bridge.first_user_message("q", "x" * 12_345))
        messages = bridge.initial_messages("q", context)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIs(messages[0]["content"], bridge.SYSTEM_PROMPT)

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
