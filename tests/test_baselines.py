"""Plan step 11's baselines (rlm_rl/baselines.py): root thinking for one arm, direct reading, the spend ceiling and
the summary. Runs against tests/fake_model.py; no W&B call is made.

    python3 -m unittest -v tests.test_baselines
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import openai

from rlm_rl import baselines, bridge, client as rclient
from rlm_rl.client import ModelClient, Pacer, RequestFailed, RetryPolicy
from rlm_rl.env import Task
from rlm_rl.tasks import tickets
from tests.fake_model import THINKING_OFF_TAIL, THINKING_ON_TAIL, AsyncCase, FakeModelServer
from tests.fakes import FakeSandbox

FAST = RetryPolicy(tries=3, base_s=0.01, max_s=0.05, timeout_s=10)
BASE = "fake-model:step0"
MESSAGES = [{"role": "system", "content": bridge.SYSTEM_PROMPT}, {"role": "user", "content": "Question: q"}]


def int_task(task_id="t-1", answer=7, family="tickets_all", context="a\nb\nc\n") -> Task:
    return Task(task_id, family, "How many?", context, answer, 1, {"type": "int", "blind": 3})


class ParseFinalTest(unittest.TestCase):
    def test_last_final_answer_line_wins(self):
        self.assertEqual(baselines.parse_final("FINAL ANSWER: 3\nno wait\nFinal answer: 5\nthanks"), (True, 5))

    def test_literals_and_strings(self):
        self.assertEqual(baselines.parse_final("FINAL ANSWER: [\"a\", \"b\"]"), (True, ["a", "b"]))
        self.assertEqual(baselines.parse_final("FINAL ANSWER: **42**"), (True, 42))
        self.assertEqual(baselines.parse_final("FINAL ANSWER: host-a, host-b"), (True, "host-a, host-b"))
        self.assertEqual(baselines.parse_final("FINAL ANSWER: `pkg.mod.func`"), (True, "pkg.mod.func"))

    def test_no_answer(self):
        self.assertEqual(baselines.parse_final("I think it is 4."), (False, None))
        self.assertEqual(baselines.parse_final("FINAL ANSWER:   "), (False, None))
        self.assertEqual(baselines.parse_final(None), (False, None))

    def test_true_is_not_an_answer_shape(self):
        self.assertEqual(baselines.parse_final("FINAL ANSWER: true"), (True, "true"))  # a string, scored as an entity

    def test_logged_messages_hide_the_context(self):
        context = "secret line one\nsecret line two\n" * 50
        logged = baselines.logged_direct_messages("How many?", context)[1]["content"]
        self.assertNotIn("secret line", logged)
        self.assertIn(f"{len(context):,} characters", logged)


class BudgetTest(unittest.TestCase):
    def test_refuses_past_the_ceiling_and_counts_reservations(self):
        class C:
            def __init__(self):
                self.usd = 0.0

            def spent_usd(self):
                return self.usd

        c = C()
        budget = baselines.Budget(1.0, 0.5, [c])
        self.assertTrue(budget.reserve(0.3))
        self.assertFalse(budget.reserve(0.3))  # 0.5 + 0.3 reserved + 0.3 > 1.0
        budget.release(0.3)
        c.usd = 0.4  # the finished episode cost 0.4
        self.assertFalse(budget.reserve(0.2))
        self.assertTrue(budget.reserve(0.1))
        self.assertEqual(budget.refused, 2)


class ServerCase(AsyncCase):
    async def asyncSetUp(self):
        self.server = FakeModelServer()
        await self.server.start()
        self.openai = openai.AsyncOpenAI(base_url=self.server.base_url, api_key="test-key")

    async def asyncTearDown(self):
        await self.openai.close()
        await self.server.close()


class RootThinkingTest(ServerCase):
    async def test_medium_thinking_asks_for_it_and_checks_the_prompt(self):
        client = ModelClient(self.openai, subcall_model=BASE, retry=FAST, root_thinking="medium")
        usage = rclient.Usage()
        choice = await client.root_turn(MESSAGES, model=BASE, usage=usage)
        body = self.server.bodies[-1]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True, "reasoning_effort": "medium"})
        self.assertTrue(rclient.thinking_on_prompt(choice.model_extra["prompt_token_ids"]))
        self.assertEqual(choice.model_extra["reasoning_tokens"], 8)
        self.assertEqual(usage.root_reasoning_tokens, 8)
        self.assertEqual(rclient.reasoning_text(choice), "fake thinking about the next step")
        await client.subcaller()("hello")  # sub-calls stay off
        self.assertEqual(self.server.bodies[-1]["chat_template_kwargs"], {"enable_thinking": False})

    async def test_a_server_that_ignores_thinking_is_infra(self):
        client = ModelClient(self.openai, subcall_model=BASE, retry=FAST, root_thinking="medium")
        original = self.server._completion

        def ignore_thinking(kind, body):
            body = dict(body, chat_template_kwargs={"enable_thinking": False})
            return original(kind, body)

        self.server._completion = ignore_thinking
        with self.assertRaises(RequestFailed) as caught:
            await client.root_turn(MESSAGES, model=BASE)
        self.assertEqual(caught.exception.kind, "thinking_off")

    def test_xhigh_is_refused(self):
        with self.assertRaises(ValueError):
            ModelClient(openai.AsyncOpenAI(base_url="http://127.0.0.1:9/v1", api_key="x"), subcall_model=BASE,
                        root_thinking="xhigh")

    def test_tail_checks(self):
        self.assertTrue(rclient.thinking_on_prompt([5, 6] + THINKING_ON_TAIL))
        self.assertFalse(rclient.thinking_on_prompt([5, 6] + THINKING_OFF_TAIL))


class DirectTest(ServerCase):
    async def test_scores_the_final_answer_and_counts_tokens(self):
        self.server.answer = lambda prompt: "Let me count.\nFINAL ANSWER: 7"
        client = ModelClient(self.openai, subcall_model=BASE, retry=FAST)
        task = int_task()
        ep = await baselines.run_direct(task, client=client, model=BASE)
        self.assertIsNone(ep.infra_error)
        self.assertTrue(ep.submitted and ep.correct)
        self.assertEqual(ep.reward, 1.0)
        body = self.server.bodies[-1]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn(task.context, body["messages"][1]["content"])  # sent in full
        self.assertNotIn(task.context, json.dumps(ep.messages))  # logged by size and hash
        self.assertEqual(ep.usage.root_calls, 1)
        self.assertGreater(ep.usage.cost_usd(), 0)

    async def test_no_final_answer_scores_zero_and_a_failed_request_is_infra(self):
        self.server.answer = lambda prompt: "It is probably 7."
        client = ModelClient(self.openai, subcall_model=BASE, retry=FAST)
        ep = await baselines.run_direct(int_task(), client=client, model=BASE)
        self.assertEqual((ep.submitted, ep.correct, ep.reward), (False, False, 0.0))
        self.server.fault = lambda kind, body: 429
        ep = await baselines.run_direct(int_task(), client=client, model=BASE)
        self.assertEqual(ep.infra_kind, "rate_limited")
        self.assertIsNone(ep.reward)


class StatusCountTest(ServerCase):
    async def test_failed_tries_are_counted_by_status(self):
        faults = iter([503, 503, 429, None])
        self.server.fault = lambda kind, body: next(faults, None)
        client = ModelClient(self.openai, subcall_model=BASE, retry=RetryPolicy(tries=5, base_s=0.01, max_s=0.02,
                                                                                 timeout_s=10))
        await client.root_turn(MESSAGES, model=BASE)
        stats = client.reset_stats()
        self.assertEqual(stats["statuses"], {"503": 2, "429": 1})
        self.assertEqual((stats["server_errors"], stats["rate_limited"]), (2, 1))


class RunTest(ServerCase):
    """A small run of two arms on ticket tasks, with the in-process sandbox."""

    def tasks(self, n=4):
        return tickets.legacy_tasks(n, 12_345)

    def baseline_run(self, results, ceiling=5.0, **kw):
        return baselines.BaselineRun("bl-test", results, openai_client=self.openai, base_model=BASE, ceiling=ceiling,
                                     sandbox_factory=FakeSandbox, pacer=Pacer(), retry=FAST, log=lambda s: None, **kw)

    async def test_runs_resumes_and_summarizes(self):
        self.server.answer = lambda prompt: "FINAL ANSWER: 3" if "<context>" in prompt else "ok"
        tasks = self.tasks()
        with tempfile.TemporaryDirectory() as tmp:
            run = self.baseline_run(Path(tmp))
            summary = await run.run(tasks, ["rlm", "direct"])
            self.assertEqual(summary["arms"]["rlm"]["n"], 4)
            self.assertEqual(summary["arms"]["direct"]["n"], 4)
            self.assertIn("rlm_vs_direct_read", summary["comparisons"])
            spend = json.loads((Path(tmp) / "bl-test" / "spend.json").read_text())
            self.assertAlmostEqual(spend["total_usd"], sum(spend["by_arm"].values()), places=5)
            first = spend["total_usd"]
            # a second run has nothing left to do and keeps the spend
            requests = sum(self.server.requests.values())
            run2 = self.baseline_run(Path(tmp))
            await run2.run(tasks, ["rlm", "direct"])
            self.assertEqual(sum(self.server.requests.values()), requests)
            self.assertAlmostEqual(json.loads((Path(tmp) / "bl-test" / "spend.json").read_text())["total_usd"], first)
            records = [json.loads(l) for l in (Path(tmp) / "bl-test/transcripts/val/direct/step0000.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["prompt_hash"] == baselines.DIRECT_PROMPT_HASH for r in records))
            self.assertTrue(all(r["arm"] == "direct" for r in records))

    async def test_the_ceiling_stops_new_episodes(self):
        tasks = self.tasks(6)
        with tempfile.TemporaryDirectory() as tmp:
            # the first reservation is the default $0.08, over a $0.05 ceiling: nothing starts, nothing is spent
            run = self.baseline_run(Path(tmp), ceiling=0.05)
            summary = await run.run(tasks, ["rlm"], retry_infra=3)
            self.assertEqual(run.session["skipped_budget"], 6)
            self.assertEqual(summary["arms"]["rlm"]["logged"], 0)
            self.assertEqual(sum(self.server.requests.values()), 0)
            # spend from an earlier session counts: with $0.97 of $1 spent, nothing starts either
            (Path(tmp) / "bl-test" / "spend.json").write_text(json.dumps({"total_usd": 0.97, "by_arm": {"rlm": 0.97}}))
            run = self.baseline_run(Path(tmp), ceiling=1.0)
            await run.run(tasks, ["rlm"])
            self.assertEqual(run.session["skipped_budget"], 6)
            self.assertAlmostEqual(json.loads((Path(tmp) / "bl-test" / "spend.json").read_text())["total_usd"], 0.97)

    async def test_a_tight_ceiling_waits_for_measured_costs_instead_of_dropping_episodes(self):
        tasks = self.tasks(6)
        with tempfile.TemporaryDirectory() as tmp:
            # one cautious default reservation ($0.08) fits under $0.10 at a time; once an episode has been measured
            # (a fraction of a cent here), the rest fit together, so all six run and none is refused
            run = self.baseline_run(Path(tmp), ceiling=0.10)
            summary = await run.run(tasks, ["rlm"])
            self.assertEqual(run.session["skipped_budget"], 0)
            self.assertEqual(summary["arms"]["rlm"]["n"], 6)
            self.assertLessEqual(run.budget.spent(), 0.10)

    async def test_infra_episodes_are_retried_and_counted(self):
        tasks = self.tasks(2)
        failing = {"n": 0}

        def fault(kind, body):
            if kind == "root" and failing["n"] < 3:
                failing["n"] += 1
                return 429
            return None

        self.server.fault = fault
        with tempfile.TemporaryDirectory() as tmp:
            run = self.baseline_run(Path(tmp), max_read=1)
            run.clients["rlm"].retry = RetryPolicy(tries=1, base_s=0.01, max_s=0.01, timeout_s=10)
            summary = await run.run(tasks, ["rlm"], retry_infra=2)
            arm = summary["arms"]["rlm"]
            self.assertEqual(arm["n"], 2)  # every episode finished in the end
            self.assertGreaterEqual(arm["attempts"]["infra"], 1)
            self.assertEqual(set(arm["attempts"]["infra_kinds"]), {"rate_limited"})


class PairedTest(unittest.TestCase):
    def test_per_task_means(self):
        a = [{"task_id": "x", "correct": True}, {"task_id": "x", "correct": False}, {"task_id": "y", "correct": False}]
        b = [{"task_id": "x", "correct": True}, {"task_id": "y", "correct": True}, {"task_id": "z", "correct": True}]
        out = baselines.paired(a, b)
        self.assertEqual(out["tasks"], 2)
        self.assertAlmostEqual(out["diff"], 0.75)
        self.assertEqual((out["b_only"], out["a_only"]), (2, 0))


if __name__ == "__main__":
    unittest.main()
