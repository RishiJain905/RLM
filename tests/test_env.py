"""The episode loop and H5: infra failures flag the episode, and flagged episodes never reach training.

The fast cases run the loop against tests/fake_model.py with an in-process stand-in for the
sandbox (tests/fakes.py), so they also run on Windows. DockerTest repeats the done-when checks
through real containers, inside WSL2:

    python3 -m unittest -v tests.test_env
"""

import asyncio
import random
import shutil
import subprocess
import sys
import unittest

import openai

from rlm_rl import bridge, env, sandbox
from rlm_rl.client import Limiter, ModelClient, RetryPolicy
from rlm_rl.env import Episode, InfraPause, Score, Task, filter_step, run_episode, run_groups, to_trajectory
from tests.fake_model import AsyncCase, FakeModelServer
from tests.fakes import FakeSandbox

FAST = RetryPolicy(tries=3, base_s=0.01, max_s=0.02, timeout_s=10)
ROOT, BASE = "fake-model:step3", "fake-model:step0"


def make_task(n_lines=5, task_id="t1", first="alpha", family="lines") -> Task:
    context = "\n".join([first] + [f"row {i}" for i in range(1, n_lines)])
    return Task(task_id, family, "How many lines?", context, answer=n_lines)


def exact(task: Task, ep: Episode) -> Score:
    ok = ep.submitted and ep.answer == task.answer
    return Score(float(ok), ok, {"exact": float(ok)})


def coin_policy(seed=0):
    """Turn 1 makes one sub-call, turn 2 submits the line count or one more, at random."""
    rng = random.Random(seed)

    def policy(messages):
        if sum(m["role"] == "assistant" for m in messages) == 0:
            return "```python\nlines = context.splitlines()\nprint(llm('Summarize: ' + lines[0]))\n```"
        return f"```python\nsubmit(len(context.splitlines()) + {rng.choice([0, 1])})\n```"

    return policy


class EnvCase(AsyncCase):
    sandbox_factory = FakeSandbox
    sandbox_kwargs: dict = {}

    async def asyncSetUp(self):
        self.server = FakeModelServer()
        await self.server.start()
        self.openai = openai.AsyncOpenAI(base_url=self.server.base_url, api_key="test-key")
        self.client = ModelClient(self.openai, subcall_model=BASE, retry=FAST, rng=random.Random(0))

    async def asyncTearDown(self):
        await self.openai.close()
        await self.server.close()

    async def episode(self, task=None, factory=None, scorer=exact, **kw) -> Episode:
        return await run_episode(task or make_task(), client=self.client, root_model=ROOT, scorer=scorer,
                                 sandbox_factory=factory or self.sandbox_factory, **{**self.sandbox_kwargs, **kw})


class EpisodeTest(EnvCase):
    async def test_runs_to_submit(self):
        ep = await self.episode()
        self.assertIsNone(ep.infra_error)
        self.assertEqual((ep.submitted, ep.answer, ep.reward, ep.correct), (True, 5, 1.0, True))
        self.assertEqual((len(ep.turns), ep.subcalls_used, len(ep.choices)), (2, 1, 2))
        self.assertEqual([m["role"] for m in ep.messages], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(ep.messages[:2], bridge.initial_messages(ep.question, make_task().context))
        self.assertEqual(ep.messages[3]["content"], bridge.observation("5\nok\n", 5, None, 11, 29))
        self.assertEqual(ep.subcalls[0]["prompt"], "Summarize: alpha")
        self.assertEqual((ep.usage.root_calls, ep.usage.subcalls), (2, 1))
        self.assertEqual(ep.metrics()["first_turn_submit"], 0.0)
        self.assertEqual(env.SYSTEM_PROMPT, bridge.SYSTEM_PROMPT)

    async def test_reply_without_code_uses_up_a_turn(self):
        replies = iter(["I will think about it.", "```python\nsubmit(len(context.splitlines()))\n```"])
        self.server.policy = lambda messages: next(replies)
        ep = await self.episode()
        self.assertIn(bridge.NO_CODE_BLOCK_ERROR, ep.messages[3]["content"])
        self.assertEqual((ep.turns[0].code, ep.turns[0].cell, ep.correct), (None, None, True))

    async def test_turn_limit(self):
        self.server.policy = lambda messages: "```python\nprint(1)\n```"
        ep = await self.episode()
        self.assertEqual((len(ep.turns), ep.submitted, ep.reward), (bridge.MAX_TURNS, False, 0.0))
        self.assertIsNone(ep.turns[-1].observation)
        self.assertIn(bridge.OBS_LAST_TURN, ep.turns[-2].observation)
        self.assertEqual(ep.messages[-1]["role"], "assistant")

    async def test_the_30_call_budget(self):
        """Spec §1.2: 30 sub-calls an episode, llm_batch uses one per prompt, and an over-budget call uses nothing."""
        self.assertEqual((bridge.MAX_SUBCALLS, bridge.MAX_TURNS), (30, 12))
        replies = iter(["```python\nouts = llm_batch(['q'] * 28)\nprint(len(outs))\n```",
                        "```python\nprint(llm_batch(['a', 'b', 'c']))\n```",  # 3 more would make 31: refused whole
                        "```python\nprint(llm('a'), llm('b'))\n```",  # exactly 30
                        "```python\nprint(llm('one too many'))\n```",
                        "```python\nsubmit(len(context.splitlines()))\n```"])
        self.server.policy = lambda messages: next(replies)
        ep = await self.episode()
        self.assertEqual((ep.subcalls_used, len(ep.subcalls), ep.budget_hit, ep.correct), (30, 30, True, True))
        self.assertIn("(10 turns left, 2 sub-calls left)", ep.messages[5]["content"])
        self.assertIn("RuntimeError: sub-call budget exhausted (30 total)", ep.messages[5]["content"])
        self.assertIn("(9 turns left, 0 sub-calls left)", ep.messages[7]["content"])
        self.assertIn("sub-call budget exhausted (30 total)", ep.messages[9]["content"])
        self.assertEqual(ep.metrics()["budget_hit"], 1.0)

    async def test_ticket_task_with_the_default_reward(self):
        """Env v2 end to end on a level-1 ticket task: calibrated reward, detectors, exact accuracy."""
        from rlm_rl.tasks import tickets

        task = tickets.make_task(2024, "tickets_all", 1, "t-1")
        off_by = iter([0, 2])

        def policy(messages):
            if sum(m["role"] == "assistant" for m in messages) == 0:
                return "```python\nprint(len(context.splitlines()))\n```"
            return f"```python\nsubmit({task.answer + next(off_by)})\n```"

        self.server.policy = policy
        exact = await run_episode(task, client=self.client, root_model=ROOT, sandbox_factory=FakeSandbox)
        near = await run_episode(task, client=self.client, root_model=ROOT, sandbox_factory=FakeSandbox)
        self.assertEqual((exact.reward, exact.correct, exact.hacks), (1.0, True, []))
        e_blind = abs(task.grading["blind"] - task.answer)
        expected = round(0.8 * max(0.0, 1 - 2 / e_blind), 4) if e_blind else 0.0
        self.assertEqual((near.reward, near.correct), (expected, False))
        self.assertEqual(near.reward_components["e_blind"], e_blind)


class InfraTest(EnvCase):
    """Done when: forced 429s and a killed kernel produce flagged episodes (H5)."""

    def assert_flagged(self, ep: Episode, kind: str):
        self.assertIsNotNone(ep.infra_error)
        self.assertEqual(ep.infra_kind, kind)
        self.assertIsNone(ep.reward)
        if ep.turns:  # the failed cell's output was never shown
            self.assertEqual(ep.messages[-1]["role"], "assistant")
        for m in ep.messages:
            self.assertNotIn("sub-call failed", m["content"])
        with self.assertRaises(ValueError):
            to_trajectory(ep)

    async def test_forced_429_on_subcalls(self):
        self.server.fault = lambda kind, body: 429 if kind == "subcall" else None
        ep = await self.episode()
        self.assert_flagged(ep, "rate_limited")
        self.assertEqual(self.server.requests["subcall"], FAST.tries)
        self.assertEqual(ep.subcalls[0]["output"], None)

    async def test_forced_429_on_root_turns(self):
        self.server.fault = lambda kind, body: 429 if kind == "root" else None
        ep = await self.episode()
        self.assert_flagged(ep, "rate_limited")
        self.assertEqual(len(ep.turns), 0)

    async def test_5xx(self):
        self.server.fault = lambda kind, body: 503
        self.assert_flagged(await self.episode(), "server_error")

    async def test_killed_sandbox(self):
        ep = await self.episode(factory=lambda *a, **k: FakeSandbox(*a, kill_after_cells=1, **k))
        self.assert_flagged(ep, "sandbox")
        self.assertEqual(len(ep.turns), 1)

    async def test_host_bug_is_infra_not_a_zero(self):
        def broken(task, ep):
            raise KeyError("grader bug")
        ep = await self.episode(scorer=broken)
        self.assertEqual((ep.infra_kind, ep.reward), ("host_bug", None))
        self.assertIn("KeyError", ep.host_traceback)

    async def test_none_reach_training(self):
        """A full step with forced 429s and killed sandboxes: every flagged episode is counted, none is trained on."""
        self.server.policy = coin_policy()
        self.server.fault = lambda kind, body: (429 if kind == "subcall" and "FAIL429" in body["messages"][-1]["content"]
                                                else None)
        kill = {"KILL"}

        def factory(context, complete, **kw):
            return FakeSandbox(context, complete, kill_after_cells=1 if context.split("\n")[0] in kill else None, **kw)

        tasks = [make_task(5 + i, f"t{i}", first) for i, first in
                 enumerate(["alpha", "FAIL429", "KILL", "beta", "gamma", "delta", "eps", "zeta"])]
        groups = await run_groups(tasks, 4, client=self.client, root_model=ROOT, scorer=exact,
                                  sandbox_factory=factory)
        flagged = {ep.episode_id for g in groups for ep in g if ep.infra_error}
        self.assertEqual(len(flagged), 8)
        result = filter_step(groups)
        self.assertEqual((result.episodes, result.infra), (32, 8))
        self.assertEqual(result.infra_kinds, {"rate_limited": 4, "sandbox": 4})
        self.assertEqual(result.dropped_small, 2)  # both groups lost every episode
        trained = []

        async def fake_train(groups_to_train):
            trained.extend(ep.episode_id for g in groups_to_train for ep in g)

        await fake_train(result.groups)
        self.assertTrue(trained)
        self.assertFalse(flagged & set(trained))
        self.assertIsNotNone(InfraPause().update(result))  # 25% > 1%: the run pauses
        try:
            import art  # noqa: F401
        except ImportError:
            return
        trajectories = [to_trajectory(ep) for g in result.groups for ep in g]
        self.assertTrue(all(len(t.messages_and_choices) == 5 for t in trajectories))


def fake_episode(reward=1.0, infra=None) -> Episode:
    ep = Episode("t", "f", 1, "q", 1, ROOT, BASE)
    ep.reward, ep.correct = (None, None) if infra else (reward, reward == 1)
    ep.infra_error, ep.infra_kind = (infra, "rate_limited") if infra else (None, None)
    return ep


class FilterTest(unittest.TestCase):
    def test_spec_5_4(self):
        groups = [
            [fake_episode(1), fake_episode(0), fake_episode(0.5), fake_episode(infra="429")],  # kept, 3 left
            [fake_episode(1), fake_episode(infra="x"), fake_episode(infra="y"), fake_episode(infra="z")],  # 1 left
            [fake_episode(0)] * 4,  # all 0
            [fake_episode(1)] * 4,  # all 1
            [fake_episode(0.4)] * 4,  # all equal, partial credit
            [fake_episode(1), fake_episode(0)],
        ]
        result = filter_step(groups)
        self.assertEqual([len(g) for g in result.groups], [3, 2])
        self.assertEqual((result.episodes, result.infra, result.infra_kinds), (22, 4, {"rate_limited": 4}))
        self.assertEqual((result.dropped_small, result.dropped_all_zero, result.dropped_all_one,
                          result.dropped_equal_other), (1, 1, 1, 1))
        self.assertIsNone(result.skip_train)
        self.assertTrue(all(ep.infra_error is None for g in result.groups for ep in g))

    def test_too_few_groups_skips_the_train_call(self):
        result = filter_step([[fake_episode(1), fake_episode(0)], [fake_episode(1)] * 4])
        self.assertEqual(result.skip_train, "only 1 trainable group (need 2)")

    def test_infra_pause(self):
        clean = filter_step([[fake_episode(1), fake_episode(0)] * 2] * 8)
        one_bad = filter_step([[fake_episode(1), fake_episode(0), fake_episode(1), fake_episode(infra="x")]]
                              + [[fake_episode(1), fake_episode(0)] * 2] * 7)
        self.assertIsNone(InfraPause().update(clean))
        self.assertIn("1 of 32 episodes", InfraPause().update(one_bad))  # the spec as written: per step
        windowed = InfraPause(window_steps=10)
        for _ in range(9):
            self.assertIsNone(windowed.update(clean))
        self.assertIsNone(windowed.update(one_bad))  # 1 of 320 is under 1%
        self.assertIsNone(windowed.update(one_bad))
        self.assertIsNone(windowed.update(one_bad))
        self.assertIsNotNone(windowed.update(one_bad))  # 4 of 320

    def test_pause_rule_per_run(self):
        """Rishi, 2026-10-11: per step for the second seed, a 10-step window in production."""
        self.assertEqual(InfraPause.for_run("second_seed").window.maxlen, 1)
        self.assertEqual(InfraPause.for_run("production").window.maxlen, 10)
        with self.assertRaises(KeyError):
            InfraPause.for_run("pilot")


def docker_ready() -> str | None:
    if sys.platform != "linux":
        return "the sandbox runs inside WSL2 (decision 3b)"
    if shutil.which("docker") is None:
        return "no docker CLI"
    if subprocess.run(["docker", "image", "inspect", sandbox.BASE_IMAGE], capture_output=True).returncode != 0:
        return "the pinned base image is not on this machine"
    return None


class DockerTest(EnvCase):
    """The done-when checks again, through real sandbox containers."""

    sandbox_factory = sandbox.Sandbox

    @classmethod
    def setUpClass(cls):
        reason = docker_ready()
        if reason:
            raise unittest.SkipTest(reason)
        cls.sandbox_kwargs = {"image": sandbox.build_image()}

    async def asyncTearDown(self):
        await super().asyncTearDown()
        left = subprocess.run(["docker", "ps", "-aq", "--filter", "label=rlm.run=test-step4"], capture_output=True,
                              text=True).stdout.split()
        self.assertEqual(left, [])

    async def test_forced_429_through_the_real_bridge(self):
        self.server.fault = lambda kind, body: 429 if kind == "subcall" else None
        ep = await self.episode(run_id="test-step4")
        self.assertEqual((ep.infra_kind, ep.reward), ("rate_limited", None))

    async def test_container_killed_mid_cell(self):
        self.server.policy = lambda messages: "```python\nimport time\ntime.sleep(20)\n```"

        def factory(context, complete, **kw):
            box = sandbox.Sandbox(context, complete, **kw)
            run_cell = box.run_cell

            async def killing_run_cell(code):
                async def kill():
                    await asyncio.sleep(2)
                    subprocess.run(["docker", "kill", box._names[-1]], capture_output=True)
                killer = asyncio.ensure_future(kill())
                try:
                    return await run_cell(code)
                finally:
                    killer.cancel()

            box.run_cell = killing_run_cell
            return box

        ep = await self.episode(factory=factory, run_id="test-step4")
        self.assertEqual(ep.infra_kind, "sandbox")
        self.assertIn("stopped from outside", ep.infra_error)
        self.assertIsNone(ep.reward)

    async def test_time_queued_for_subcalls_does_not_count_toward_the_cell_timeout(self):
        """Plan step 4: a cell whose sub-calls wait in the limiter's queue isn't timed out for it."""
        self.client = ModelClient(self.openai, subcall_model=BASE, retry=FAST, limiter=Limiter(total=1, per_model=1))
        self.server.latency = (0.5, 0.5)
        replies = iter(["```python\nprint(len(llm_batch(['q'] * 8)))\n```", "```python\nsubmit(5)\n```"])
        self.server.policy = lambda messages: next(replies)
        ep = await self.episode(run_id="test-step4", limits=sandbox.SandboxLimits(cell_timeout_s=2))
        self.assertIsNone(ep.infra_error)
        cell = ep.turns[0].cell
        self.assertEqual((cell["stdout"], cell["crash"]), ("8\n", None))
        self.assertGreater(cell["blocked_s"], 3.5)  # 8 sub-calls one at a time, 0.5 s each
        self.assertLess(cell["compute_s"], 2)

    async def test_load_through_real_sandboxes(self):
        """32 episodes at once, each asking 10 sub-calls in one llm_batch: never over the caps, no 429."""
        self.server.latency = (0.2, 0.5)

        def policy(messages):
            if sum(m["role"] == "assistant" for m in messages) == 0:
                return "```python\nouts = llm_batch([f'q{i}' for i in range(10)])\nprint(len(outs))\n```"
            return "```python\nsubmit(len(context.splitlines()))\n```"

        self.server.policy = policy
        tasks = [make_task(5, f"t{i}") for i in range(8)]
        groups = await run_groups(tasks, 4, client=self.client, root_model=ROOT, scorer=exact,
                                  run_id="test-step4", **self.sandbox_kwargs)
        episodes = [ep for g in groups for ep in g]
        self.assertEqual([ep.infra_error for ep in episodes], [None] * 32)
        self.assertTrue(all(ep.correct and ep.subcalls_used == 10 for ep in episodes))
        self.assertLessEqual(self.server.peak, 30)
        self.assertLessEqual(max(self.server.peak_by_model.values()), 20)
        self.assertEqual(self.server.statuses[429], 0)
        print(f"\n  load: server peak {self.server.peak} in flight "
              f"({dict(self.server.peak_by_model)}), client peak {self.client.limiter.peaks['total']}")


if __name__ == "__main__":
    unittest.main()
