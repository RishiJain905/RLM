"""The coding warm start (rlm_rl/warmstart.py, TrainingRun.warm_start; prod-v2, 2026-10-19)."""

import unittest

from rlm_rl import training, warmstart
from tests.test_training import LoopCase, second_seed


def record(correct=True, delegating=True, family="edit_bugs"):
    """A sampled edit episode as a transcript record: three turns, the second a broken reply."""
    system = "You are the root model." + warmstart.SEPARATOR + warmstart.TEACHER_EDIT
    big = "x" * (training.DELEGATED_EDIT_PROMPT_CHARS + 10)
    calls = [{"prompt": big, "output": "draft"}, {"prompt": big, "output": "fix"}] if delegating else []
    return {
        "family": family, "task_id": "t1", "correct": correct, "hacks": [], "infra_error": None, "context_chars": 10,
        "subcalls": calls, "usage": {"est_cost_usd": 0.25},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": "task"},
                     {"role": "assistant", "content": "```python\nx = llm(src)\n```"}, {"role": "user", "content": "ok"},
                     {"role": "assistant", "content": "I will now"}, {"role": "user", "content": "no code"},
                     {"role": "assistant", "content": "```python\nsubmit('done')\n```"}],
        "turns": [{"reply": "a", "finish_reason": "stop", "code": "x = llm(src)", "reasoning": "think 1"},
                  {"reply": "b", "finish_reason": "stop", "code": None, "reasoning": "think 2"},
                  {"reply": "c", "finish_reason": "stop", "code": "submit('done')", "reasoning": "think 3"}],
    }


class WarmStartDataTest(unittest.TestCase):
    def test_each_kept_turn_is_one_sample_ending_on_its_reply_with_its_thinking(self):
        samples = warmstart.sft_samples(record())
        self.assertEqual(len(samples), 2)  # the broken reply is never a target
        for s in samples:
            self.assertEqual(s[0]["content"], "You are the root model.")  # the teacher hint is gone
            self.assertEqual(s[-1]["role"], "assistant")
            self.assertTrue(all("reasoning_content" not in m for m in s[:-1]))  # history as it was sampled
        self.assertEqual((samples[0][-1]["reasoning_content"], len(samples[0])), ("think 1", 3))
        self.assertEqual((samples[1][-1]["reasoning_content"], len(samples[1])), ("think 3", 7))
        self.assertEqual(samples[1][4]["content"], "I will now")  # the broken reply stays as context
        with self.assertRaises(ValueError):
            warmstart.strip_teacher("a system prompt without the hint")

    def test_only_right_unhacked_delegating_episodes_are_kept(self):
        self.assertTrue(warmstart.keep(record()))
        self.assertFalse(warmstart.keep(record(correct=False)))
        self.assertFalse(warmstart.keep(record(delegating=False)))
        self.assertFalse(warmstart.keep({**record(), "hacks": [{"rule": "early_submit"}]}))
        self.assertFalse(warmstart.keep({**record(), "infra_error": "server_error"}))
        s = warmstart.summary([record(), record(correct=False), record(family="edit_spec")])
        self.assertEqual(s["edit_bugs"], {"sampled": 2, "right": 1, "kept": 1, "cost_usd": 0.5})
        self.assertEqual(len(warmstart.sft_trajectories([record()])), 2)


class WarmStartRunTest(LoopCase):
    async def test_the_latest_checkpoint_points_at_the_sft_result_and_the_next_step_samples_from_it(self):
        st = await self.run_("ws", cfg=second_seed("ws", 4), stop_after=1).start()
        before = st["checkpoints"]["1"]
        trajs = warmstart.sft_trajectories([record()])
        st = await self.run_("ws").warm_start(trajs, learning_rate=2e-5, sampling_usd=1.5, source="test")
        after = st["checkpoints"]["1"]
        self.assertEqual((after["branch"], after["warm_start"]["from_backend_step"]), (0, before["backend_step"]))
        self.assertGreater(after["backend_step"], before["backend_step"])
        self.assertEqual(self.dry.backend.sft[-1]["samples"], 2)
        self.assertEqual(st["spend"]["warm_start_usd"], 1.5)
        self.assertAlmostEqual(st["spend"]["train_usd"], st["steps"][-1]["cost"]["spent_train_usd"] + 1.5, places=5)
        self.assertEqual(st["warm_starts"][0]["checkpoint"], 1)
        self.assertTrue(any(n["kind"] == "warm_start" for n in st["notices"]))
        st = await self.run_("ws", stop_after=2).resume()
        self.assertEqual(st["steps"][-1]["sampled_from"]["model"], f"ws:step{after['backend_step']}")
        with self.assertRaises(ValueError):
            await self.run_("ws").warm_start([], learning_rate=2e-5)


if __name__ == "__main__":
    unittest.main()
