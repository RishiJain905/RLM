"""H10: one bridge contract, shared by the sandbox worker, the training environment and the harness.

The worker runs inside the sandbox and can't import rlm_rl.bridge, so its copies of the
protocol constants are checked here. Every module that builds a root prompt must take it from
rlm_rl.bridge; PROMPT_CONSUMERS lists them, and the test diffs each one's system prompt.
"""

import ast
import difflib
import importlib
import importlib.util
import inspect
import math
import re
import unittest
from pathlib import Path

import openai

from rlm_rl import bridge, repl_tools, repl_worker, sandbox
from tests.fake_model import AsyncCase

ROOT = Path(__file__).resolve().parents[1]

# Modules that render the system prompt. rlm_rl.env and rlm_rl.evals arrived in step 4; the Part 2
# harness adds its entry point here when it exists.
PROMPT_CONSUMERS = ("rlm_rl.env", "rlm_rl.evals", "harness")


class WorkerMatchesBridgeTest(unittest.TestCase):
    def test_protocol_constants(self):
        for name in ("API_NAMES", "WIRE_OPS", "OUTPUT_MAX_CHARS", "TRACEBACK_LINES", "SUBCALL_MAX_CHARS",
                     "SUBCALL_TRUNCATION_MARK", "SUBMIT_MAX_BYTES"):
            self.assertEqual(getattr(repl_worker, name), getattr(bridge, name), name)
        self.assertEqual(repl_worker.AUDIT_PREFIX, sandbox.AUDIT_PREFIX)
        self.assertEqual(repl_worker.TAMPER_EXIT_CODE, sandbox.TAMPER_EXIT_CODE)

    def test_repl_exposes_exactly_the_api(self):
        names = set(repl_worker.make_namespace("text")) - {"__name__"}
        self.assertEqual(names, set(bridge.API_NAMES))
        self.assertEqual(repl_worker.make_namespace("text")["context"], "text")

    def test_truncation_agrees(self):
        for n in (0, 10, 24_000, 24_001, 50_000):
            self.assertEqual(repl_worker._truncate("q" * n), bridge.truncate_prompt("q" * n))

    def test_every_submitted_value_is_a_valid_answer(self):
        class Thing:
            def __str__(self):
                return "thing"
        samples = [3, -7, 2.5, math.nan, math.inf, "host-1", True, None, [1, "a", 2.0], (1, 2), {"b", "a"},
                   frozenset({3}), [[1, 2], None, True], {"k": 1}, Thing(), 10 ** 30]
        for value in samples:
            self.assertTrue(bridge.valid_answer(repl_worker.normalize_answer(value)), repr(value))
        self.assertEqual(repl_worker.normalize_answer({"b", "a"}), ["a", "b"])
        self.assertEqual(repl_worker.normalize_answer(True), "True")
        self.assertEqual(repl_worker.normalize_answer([[1, 2], None]), ["[1, 2]", "None"])
        self.assertEqual(repl_worker.normalize_answer(math.nan), "nan")

    def test_worker_imports_nothing_from_the_project(self):
        # repl_tools is the one exception: it is mounted next to the worker, and is standard library only itself
        # (tests/test_repl_tools.py checks).
        source = (ROOT / "rlm_rl" / "repl_worker.py").read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                module = stripped.split()[1].split(".")[0]
                self.assertIn(module, __import__("sys").stdlib_module_names | {"repl_tools"}, line)


class OneSystemPromptTest(unittest.TestCase):
    def _production_sources(self):
        for folder in ("rlm_rl", "scripts"):
            for path in (ROOT / folder).rglob("*.py"):
                if "archive" not in path.relative_to(ROOT).parts:
                    yield path

    def test_prompt_text_is_defined_only_in_the_bridge(self):
        for path in self._production_sources():
            text = path.read_text(encoding="utf-8")
            if path.name != "bridge.py":
                self.assertNotIn("You are a Recursive Language Model", text, path)
            self.assertNotIn("llm_query", text, path)

    def test_consumers_use_the_bridge_prompt(self):
        checked = []
        for module_name in PROMPT_CONSUMERS:
            if importlib.util.find_spec(module_name.split(".")[0]) is None:
                continue
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                if exc.name != module_name:  # the consumer exists but a dependency is missing: fail, don't skip
                    raise
                continue
            with self.subTest(module=module_name):
                self.assertEqual(getattr(module, "SYSTEM_PROMPT", None), bridge.SYSTEM_PROMPT)
            checked.append(module_name)
        print(f"\n  system prompt diffed for: {', '.join(checked) or 'no consumers yet'}")


class ToolsInPromptTest(unittest.TestCase):
    """Spec §12.1: the coding section of prompt v2 names each tool in repl_tools with its real parameters,
    defaults and return type, and no other."""

    def _listed(self) -> dict:
        section = bridge.SYSTEM_PROMPT.split("For coding tasks", 1)[1]
        return {m.group(1): (m.group(2), m.group(3)) for m in
                re.finditer(r"^- `(\w+)\(([^)]*)\)(?: -> ([^`]+))?`", section, re.M)}

    def test_each_tool_is_listed_with_its_signature(self):
        listed = self._listed()
        self.assertEqual(tuple(listed), repl_tools.TOOL_NAMES)
        for name, fn in repl_tools.namespace().items():
            params, returns = listed[name]
            sig = inspect.signature(fn)
            shown = [p.strip() for p in params.split(",") if p.strip()]
            self.assertEqual([p.split("=")[0] for p in shown], list(sig.parameters), name)
            for p in shown:
                key, _, default = p.partition("=")
                expected = sig.parameters[key].default
                if default:
                    self.assertEqual(ast.literal_eval(default), expected, f"{name}({key})")
                else:
                    self.assertIs(expected, inspect.Parameter.empty, f"{name}({key}) has a default the prompt omits")
            if returns:
                self.assertEqual(returns, inspect.formatannotation(sig.return_annotation), name)

    def test_v1_has_no_tools(self):
        self.assertNotIn("read_file", bridge.system_prompt("v1"))


def prompt_diff(sent: str, pinned: str, name: str) -> str:
    """A readable line diff between what a consumer sent and the bridge's pinned text."""
    return "\n".join(difflib.unified_diff(pinned.splitlines(), sent.splitlines(), f"bridge {name}", f"sent {name}",
                                          lineterm=""))


class WirePromptTest(AsyncCase):
    """H10 on the wire: what the environment actually sends the model is the bridge's text, word for word.

    The attribute check above catches a consumer that defines its own SYSTEM_PROMPT; this one
    catches a consumer that edits the messages on the way out. When the Part 2 harness exists, it
    gets the same test by driving one episode through it against the same fake server.
    """

    async def test_env_sends_the_bridge_prompt(self):
        from rlm_rl.env import Task, run_episode  # needs openai, like the consumers above
        from rlm_rl.client import ModelClient
        from tests.fakes import FakeSandbox
        from tests.fake_model import FakeModelServer

        server = FakeModelServer()
        await server.start()
        client = openai.AsyncOpenAI(base_url=server.base_url, api_key="test-key")
        try:
            task = Task("t", "f", "How many lines?", "alpha\nbeta\ngamma\ndelta", answer=4,
                        grading={"type": "int", "blind": 1})
            ep = await run_episode(task, client=ModelClient(client, subcall_model="fake-model:step0"),
                                   root_model="fake-model:step3", sandbox_factory=FakeSandbox)
        finally:
            await client.close()
            await server.close()
        roots = [b for b in server.bodies if b.get("logprobs")]
        self.assertEqual(len(roots), 2)
        sent, pinned = roots[0]["messages"], bridge.initial_messages(task.question, task.context)
        for role, i in (("system", 0), ("first user", 1)):
            self.assertEqual(sent[i]["content"], pinned[i]["content"],
                             "\n" + prompt_diff(sent[i]["content"], pinned[i]["content"], role))
        cell = ep.turns[0].cell
        observation = bridge.observation(cell["stdout"], cell["stdout_chars"], cell["error"], bridge.MAX_TURNS - 1,
                                         bridge.MAX_SUBCALLS - 1)
        self.assertEqual(roots[1]["messages"][3]["content"], observation,
                         "\n" + prompt_diff(roots[1]["messages"][3]["content"], observation, "observation"))
        self.assertEqual(roots[1]["messages"][:2], roots[0]["messages"])  # the prefix never changes mid-episode


if __name__ == "__main__":
    unittest.main()
