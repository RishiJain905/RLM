"""Per-step metrics: entropy, KL to base, within-group reward std and the rest of spec §9 (plan step 6).

    python3 -m unittest -v tests.test_monitor
"""

import random

import openai

from rlm_rl import monitor
from rlm_rl.client import ModelClient
from rlm_rl.env import Score, filter_step, run_groups
from tests.fake_model import AsyncCase, FakeModelServer
from tests.fakes import FakeSandbox
from tests.test_env import BASE, FAST, ROOT, coin_policy, exact, make_task


class MonitorCase(AsyncCase):
    async def asyncSetUp(self):
        self.server = FakeModelServer(policy=coin_policy(3))
        self.server.completion_logprob = lambda model: -0.4
        self.server.prompt_logprob = lambda model: -0.9 if model == BASE else -0.4
        await self.server.start()
        self.openai = openai.AsyncOpenAI(base_url=self.server.base_url, api_key="test-key")
        self.client = ModelClient(self.openai, subcall_model=BASE, retry=FAST, rng=random.Random(0))

    async def asyncTearDown(self):
        await self.openai.close()
        await self.server.close()

    async def groups(self, n_tasks=4, rollouts=4, scorer=exact, **kw):
        tasks = [make_task(5 + i, f"t{i}") for i in range(n_tasks)]
        return await run_groups(tasks, rollouts, client=self.client, root_model=ROOT, scorer=scorer,
                                sandbox_factory=FakeSandbox, **kw)


class SignalsTest(MonitorCase):
    async def test_entropy_is_the_mean_negative_logprob_of_sampled_tokens(self):
        groups = await self.groups()
        value, tokens = monitor.entropy(monitor.scored(groups))
        self.assertAlmostEqual(value, 0.4)
        self.assertEqual(tokens, sum(len(c.model_extra["token_ids"]) for g in groups for ep in g for c in ep.choices))

    async def test_kl_to_base_lines_up_every_sampled_token(self):
        groups = await self.groups()
        episodes = monitor.scored(groups)
        kl = await monitor.kl_to_base(self.client, episodes, base_model=BASE, n=4, rng=random.Random(0))
        self.assertAlmostEqual(kl["kl"], 0.5)  # policy -0.4, base -0.9
        self.assertEqual((kl["rollouts"], kl["failed"], kl["misaligned_turns"]), (4, 0, 0))
        self.assertGreater(kl["tokens"], 0)
        self.assertGreater(kl["kl_k3"], 0)
        self.assertGreater(kl["cost_usd"], 0)
        bodies = [b for b in self.server.bodies if b.get("prompt_logprobs") is not None and not b.get("logprobs")]
        self.assertEqual(len(bodies), 4)
        self.assertTrue(all(b["model"] == BASE and b["max_tokens"] == 1 for b in bodies))
        self.assertEqual(bodies[0]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(bodies[0]["messages"][-1]["role"], "assistant")  # the whole finished conversation

    async def test_tokens_that_do_not_line_up_are_skipped_and_counted(self):
        ep = monitor.scored(await self.groups(1, 1))[0]
        ids, lps = await self.client.prompt_logprobs(ep.messages, model=BASE)
        pairs, bad = monitor.align(ep, ids, lps)
        self.assertEqual(bad, 0)
        self.assertEqual(len(pairs), sum(len(c.model_extra["token_ids"]) for c in ep.choices))
        last = len(ep.choices[-1].model_extra["prompt_token_ids"])
        changed = list(ids)
        changed[last] += 1  # the server tokenized the last reply differently
        self.assertEqual(monitor.align(ep, changed, lps)[1], 1)
        first = len(ep.choices[0].model_extra["prompt_token_ids"])
        changed = list(ids)
        changed[first] += 1  # the first reply: the second turn's prompt no longer matches either
        self.assertEqual(monitor.align(ep, changed, lps)[1], 2)

    async def test_kl_with_root_thinking_rescores_turns_with_their_thinking(self):
        """Production thinks at medium (decision 11b): a turn's sampled tokens include its thinking, which the next
        turn's prompt drops, so each re-scored rollout re-scores two of its turns, each with its thinking."""
        self.client = ModelClient(self.openai, subcall_model=BASE, retry=FAST, rng=random.Random(0), root_thinking="medium")
        groups = await self.groups()
        episodes = monitor.scored(groups)
        self.assertTrue(all(t.reasoning for ep in episodes for t in ep.turns))
        start = len(self.server.bodies)
        kl = await monitor.kl_to_base(self.client, episodes, base_model=BASE, n=3, rng=random.Random(0))
        self.assertAlmostEqual(kl["kl"], 0.5)  # policy -0.4, base -0.9, thinking tokens included
        self.assertEqual((kl["rollouts"], kl["failed"], kl["misaligned_turns"]), (3, 0, 0))
        self.assertEqual(kl["requests"], 3 * monitor.KL_TURNS_PER_THINKING_EPISODE)
        bodies = [b for b in self.server.bodies[start:] if b.get("prompt_logprobs") is not None]
        self.assertEqual(len(bodies), kl["requests"])
        for body in bodies:
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True, "reasoning_effort": "medium"})
            self.assertTrue(body["continue_final_message"])
            self.assertFalse(body["add_generation_prompt"])
            self.assertNotIn("<think>", body["messages"][-1]["content"])
            self.assertTrue(body["messages"][-1]["reasoning_content"])  # the thinking in its own field
        # every scored token is a sampled one but the final end-of-turn token of each re-scored turn
        ep = episodes[0]
        pairs = monitor.align_turn(ep.choices[0], *await self.client.prompt_logprobs(
            monitor.thinking_turn_messages(ep, 0), model=BASE, thinking="medium", continue_final=True))
        self.assertEqual(len(pairs), len(ep.choices[0].model_extra["token_ids"]) - 1)
        # thinking left in <think> tags inside the content gets an empty think block in front, as on W&B (prod-v1
        # step 1): it doesn't line up
        turn = ep.turns[0]
        tagged = ep.messages[:2] + [{"role": "assistant", "content": f"<think>\n{turn.reasoning}\n</think>\n\n{turn.reply}"}]
        self.assertIsNone(monitor.align_turn(ep.choices[0], *await self.client.prompt_logprobs(
            tagged, model=BASE, thinking="medium", continue_final=True)))
        # a turn cut off at the token cap isn't picked: its think block never closed
        for e in episodes:
            e.choices[0].finish_reason = "length"
        kl = await monitor.kl_to_base(self.client, episodes, base_model=BASE, n=3, rng=random.Random(0))
        self.assertEqual((kl["requests"], kl["misaligned_turns"]), (3, 0))  # one whole turn left in each
        # a turn whose thinking was recorded wrongly no longer lines up, and is counted
        ep.turns[0].reasoning = "something else"
        self.assertIsNone(monitor.align_turn(ep.choices[0], *await self.client.prompt_logprobs(
            monitor.thinking_turn_messages(ep, 0), model=BASE, thinking="medium", continue_final=True)))

    async def test_a_failed_rescoring_is_counted_not_raised(self):
        self.server.fault = lambda kind, body: 500 if kind == "kl" else None
        episodes = monitor.scored(await self.groups(2, 2))
        kl = await monitor.kl_to_base(self.client, episodes, base_model=BASE, n=3, rng=random.Random(0))
        self.assertEqual((kl["failed"], kl["tokens"], kl["kl"]), (3, 0, None))

    async def test_reward_std_and_histogram(self):
        rewards = {"t0": [1, 0, 1, 0], "t1": [1, 1, 1, 1], "t2": [0, 0, 0, 0.5]}

        def scorer(task, ep):
            r = rewards[task.task_id].pop()
            return Score(float(r), r == 1, {"exact": float(r == 1)})

        groups = await self.groups(3, 4, scorer=scorer)
        self.assertAlmostEqual(monitor.reward_std(groups), (0.5 + 0 + 0.2165063509) / 3, places=6)
        self.assertEqual(monitor.pass_histogram(groups), {"0/4": 1, "1/4": 0, "2/4": 1, "3/4": 0, "4/4": 1})


class StepMetricsTest(MonitorCase):
    async def test_every_spec_9_metric_is_there(self):
        self.server.fault = lambda kind, body: 500 if kind == "root" and "FAIL" in body["messages"][1]["content"] else None
        tasks = [make_task(5, "t0"), make_task(6, "t1", first="FAIL")]
        groups = await run_groups(tasks, 4, client=self.client, root_model=ROOT, scorer=exact, sandbox_factory=FakeSandbox)
        filt = filter_step(groups)
        kl = await monitor.kl_to_base(self.client, monitor.scored(groups), base_model=BASE, n=2, rng=random.Random(0))
        m = monitor.step_metrics(groups, filt, kl=kl, request_stats=self.client.reset_stats(),
                                 train_metrics={"loss/grad_norm": 0.8, "loss/probs_corr": 0.99,
                                                "loss/clipped_token_fraction": 0.01})
        self.assertEqual(m["infra"]["infra"], 4)
        self.assertEqual(m["infra"]["kinds"], {"server_error": 4})
        self.assertEqual(m["infra"]["requests"]["server_errors"], 12)  # 4 episodes x 3 tries
        self.assertNotIn("events", m["infra"]["requests"])
        self.assertAlmostEqual(m["pass"]["rate_with_infra"], m["pass"]["rate"] / 2)
        for section, keys in {
            "pass": ["rate", "rate_with_infra", "histogram", "trainable_groups", "all_zero_groups", "all_one_groups"],
            "reward": ["mean", "exact", "partial", "near_blind", "hacked", "hacks"],
            "policy": ["entropy", "kl_to_base", "reward_std", "grad_norm", "probs_corr", "clip_fraction",
                       "importance_ratio_mean"],
            "shape": ["subcalls", "budget_hit", "turns", "root_completion_tokens", "subcall_tokens",
                      "truncated_turns", "output_truncations", "root_transcript_tokens"],
            "infra": ["rate", "kinds", "requests"],
        }.items():
            for key in keys:
                self.assertIn(key, m[section], f"{section}.{key}")
        self.assertEqual((m["policy"]["grad_norm"], m["policy"]["clip_fraction"]), (0.8, 0.01))
        self.assertEqual(m["families"]["lines"]["episodes"], 8)
        self.assertGreater(m["shape"]["root_transcript_tokens"]["max"], 0)
        flat = monitor.flat({"policy": m["policy"]})
        self.assertIn("rlm/policy/entropy", flat)
        self.assertNotIn("rlm/policy/importance_ratio_p95", flat)  # missing values are left out

    async def test_the_client_counts_every_response_it_pays_for(self):
        before = self.client.spent_usd()
        await self.groups(1, 2)
        after_rollouts = self.client.spent_usd()
        self.assertGreater(after_rollouts, before)
        self.assertEqual(set(self.client.spend), {"root", "subcall"})
        await self.client.chat([{"role": "user", "content": "hi"}], model=ROOT)
        self.assertGreater(self.client.spent_usd(), after_rollouts)
        self.assertIn("chat", self.client.spend)
