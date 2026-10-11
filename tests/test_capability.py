"""The capability check's grading and runner (research §8.5, stop rule 3).

    python3 -m unittest -v tests.test_capability
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import openai

from rlm_rl import capability
from rlm_rl.capability import Item, passes
from rlm_rl.client import ModelClient
from tests.fake_model import AsyncCase, FakeModelServer
from tests.test_env import BASE, FAST, ROOT


class ChecksTest(unittest.TestCase):
    def test_each_check_kind(self):
        cases = [
            ({"kind": "exact", "value": "Paris"}, "paris.", True), ({"kind": "exact", "value": "Paris"}, "Paris, France", False),
            ({"kind": "choice", "value": "C"}, "Let me think. The answer is (C).", True),
            ({"kind": "choice", "value": "C"}, "Answer: B", False), ({"kind": "choice", "value": "D"}, "D", True),
            ({"kind": "number", "value": 1234}, "It is 1,234.", True), ({"kind": "number", "value": 12}, "13", False),
            ({"kind": "contains", "value": "def "}, "```python\ndef f(): pass\n```", True),
            ({"kind": "not_contains", "value": "sorry"}, "Sorry, no.", False),
            ({"kind": "regex", "value": r"^\d{3}-\d{4}$"}, "555-1234", True),
            ({"kind": "max_words", "value": 3}, "one two three four", False), ({"kind": "min_words", "value": 2}, "a b", True),
            ({"kind": "lowercase"}, "all lower case", True), ({"kind": "lowercase"}, "Not", False),
            ({"kind": "json"}, '```json\n{"a": 1}\n```', True), ({"kind": "json"}, "{a: 1}", False),
        ]
        for check, text, expected in cases:
            self.assertEqual(passes(check, text), expected, (check, text))

    def test_every_check_must_pass(self):
        item = Item("x", "p", [{"kind": "contains", "value": "cat"}, {"kind": "max_words", "value": 2}])
        self.assertTrue(capability.grade(item, "a cat"))
        self.assertFalse(capability.grade(item, "a big cat"))

    def test_load_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "set.jsonl"
            path.write_text(json.dumps({"id": "a", "prompt": "p", "checks": [{"kind": "exact", "value": "1"}]}) + "\n\n",
                            encoding="utf-8")
            self.assertEqual(len(capability.load(path)), 1)
            path.write_text(json.dumps({"id": "a", "prompt": "p", "checks": [{"kind": "vibes"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                capability.load(path)


class RunTest(AsyncCase):
    async def asyncSetUp(self):
        self.server = FakeModelServer(answer=lambda prompt: "42" if "6 x 7" in prompt else "no idea")
        await self.server.start()
        self.openai = openai.AsyncOpenAI(base_url=self.server.base_url, api_key="test-key")
        self.client = ModelClient(self.openai, subcall_model=BASE, retry=FAST, rng=random.Random(0))

    async def asyncTearDown(self):
        await self.openai.close()
        await self.server.close()

    async def test_run_grades_and_counts_infra_apart(self):
        items = [Item("a", "What is 6 x 7?", [{"kind": "number", "value": 42}], "math"),
                 Item("b", "Capital of France?", [{"kind": "exact", "value": "Paris"}], "chat"),
                 Item("c", "FAIL this one", [{"kind": "exact", "value": "x"}], "chat")]
        self.server.fault = lambda kind, body: 503 if "FAIL" in body["messages"][-1]["content"] else None
        result = await capability.run(items, client=self.client, model=ROOT)
        s = result.summary
        self.assertEqual((s["items"], s["graded"], s["infra"]), (3, 2, 1))
        self.assertAlmostEqual(s["accuracy"], 0.5)
        self.assertAlmostEqual(s["complete"], 2 / 3)
        self.assertEqual(s["by_source"]["math"], {"n": 1, "accuracy": 1.0})
        body = next(b for b in self.server.bodies if b["messages"][-1]["content"].startswith("What is"))
        self.assertEqual((body["model"], body["temperature"]), (ROOT, 0.0))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
