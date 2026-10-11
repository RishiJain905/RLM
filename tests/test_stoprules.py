"""The stop rules, the collapse check and the learning-rate rule on synthetic curves (research §8.6, spec §5.2).

    python3 -m unittest -v tests.test_stoprules
"""

import unittest

from rlm_rl import stoprules as sr


def row(step, *, entropy=0.6, reward_std=0.3, reward=0.5, hacked=0, episodes=32, infra=0, grad=0.7, corr=0.997):
    return {"step": step, "policy": {"entropy": entropy, "reward_std": reward_std, "grad_norm": grad, "probs_corr": corr},
            "reward": {"mean": reward, "hacked": hacked}, "infra": {"episodes": episodes, "infra": infra}}


def sel(checkpoint, accuracy, name="sel"):
    return {"checkpoint": checkpoint, "name": name, "summary": {"accuracy": accuracy}}


def cap(checkpoint, accuracy, complete=1.0):
    return {"checkpoint": checkpoint, "name": "capability", "summary": {"accuracy": accuracy, "complete": complete}}


class EvalRulesTest(unittest.TestCase):
    def test_new_bests_and_best_checkpoint(self):
        points = [(0, 0.5), (10, 0.6), (20, 0.6), (30, 0.55), (40, 0.7)]
        self.assertEqual(sr.new_bests(points), [True, True, False, False, True])
        self.assertEqual(sr.best_checkpoint(points), (40, 0.7))
        self.assertEqual(sr.best_checkpoint([(10, 0.6), (20, 0.6)]), (10, 0.6))  # a tie keeps the earlier one

    def test_plateau_fires_after_three_evals_without_a_best_and_no_level_up(self):
        points = [(0, 0.5), (10, 0.6), (20, 0.58), (30, 0.6), (40, 0.55)]
        status = sr.plateau(points, level_ups=[5])
        self.assertTrue(status.fired, status.text)
        self.assertIn("3 of 3", status.text)
        self.assertFalse(sr.plateau(points[:-1], level_ups=[]).fired)  # only 2 without a best
        self.assertEqual(sr.plateau(points[:-1], level_ups=[]).state, "watch")

    def test_a_recent_level_up_keeps_the_run_going(self):
        points = [(0, 0.5), (10, 0.6), (20, 0.58), (30, 0.6), (40, 0.55)]
        status = sr.plateau(points, level_ups=[12])  # within the last 30 steps of checkpoint 40
        self.assertFalse(status.fired)
        self.assertIn("moved up at step 12", status.text)
        self.assertTrue(sr.plateau(points, level_ups=[10]).fired)  # step 10 is 30 steps back: too old

    def test_overfitting_needs_two_evals_in_a_row(self):
        rows = [row(s, reward=0.3 + 0.01 * s) for s in range(1, 41)]  # training reward keeps rising
        points = [(0, 0.5), (10, 0.6), (20, 0.62), (30, 0.58), (40, 0.55)]
        self.assertTrue(sr.overfitting(points, rows).fired)
        self.assertFalse(sr.overfitting(points[:-1], rows).fired)
        flat = [row(s, reward=0.5) for s in range(1, 41)]  # selection falls but training reward doesn't rise
        self.assertFalse(sr.overfitting(points, flat).fired)

    def test_frying_on_a_capability_drop(self):
        self.assertTrue(sr.frying([cap(0, 0.80), cap(10, 0.79), cap(20, 0.77)]).fired)  # 3 points
        status = sr.frying([cap(0, 0.80), cap(10, 0.79)])
        self.assertFalse(status.fired)
        self.assertIn("1.0 points below", status.text)
        self.assertFalse(sr.frying([cap(0, 0.80), cap(10, 0.70, complete=0.5)]).fired)  # incomplete: not compared
        self.assertEqual(sr.frying([cap(10, 0.5)]).state, "n/a")  # no checkpoint 0 to compare with
        self.assertTrue(sr.frying([], collapse_after_rollback="entropy fell again").fired)

    def test_saturation(self):
        summary = {"a": {"level": 5, "max_level": 5, "saturated": True}, "b": {"level": 5, "max_level": 5, "saturated": True}}
        self.assertTrue(sr.saturation(summary).fired)
        summary["b"]["saturated"] = False
        self.assertFalse(sr.saturation(summary).fired)


class HackingTest(unittest.TestCase):
    def test_an_audit_hack_pauses(self):
        status = sr.hacking([row(1)], [{"checkpoint": 10, "episode_id": "e1", "key": "10/e1"}])
        self.assertTrue(status.fired)
        self.assertEqual(status.action, "pause")

    def test_a_rising_detector_rate_pauses(self):
        rows = [row(s, hacked=0) for s in range(1, 11)] + [row(s, hacked=3) for s in range(11, 21)]
        status = sr.hacking(rows, [])
        self.assertTrue(status.fired, status.text)  # 30 of 320 = 9.4%, up from 0%
        steady = [row(s, hacked=3) for s in range(1, 21)]
        self.assertFalse(sr.hacking(steady, []).fired)  # high but not rising
        self.assertFalse(sr.hacking(rows, [], since_step=18).fired)  # a resume restarts the window

    def test_infra_episodes_are_not_counted_as_scored(self):
        rows = [row(s, hacked=2, episodes=32, infra=30) for s in range(1, 6)]
        self.assertTrue(sr.hacking(rows, []).fired)  # 2 of 2 scored episodes each step


class CollapseTest(unittest.TestCase):
    def test_entropy_drop(self):
        rows = [row(s, entropy=0.6) for s in range(1, 14)] + [row(14, entropy=0.40)]
        signal = sr.collapse(rows)
        self.assertEqual((signal.kind, signal.step, signal.first_bad_step), ("entropy", 14, 14))
        self.assertIsNone(sr.collapse(rows[:-1] + [row(14, entropy=0.45)]))  # 25% is inside the limit
        self.assertIsNone(sr.collapse(rows, since_step=14))

    def test_entropy_needs_some_history(self):
        self.assertIsNone(sr.collapse([row(1, entropy=0.6), row(2, entropy=0.6), row(3, entropy=0.1)]))

    def test_reward_std_streak(self):
        rows = [row(s, reward_std=0.30) for s in range(1, 11)]
        rows += [row(11, reward_std=0.10), row(12, reward_std=0.10)]
        self.assertIsNone(sr.collapse(rows))  # 2 steps in a row is not 3
        signal = sr.collapse(rows + [row(13, reward_std=0.08)])
        self.assertEqual((signal.kind, signal.first_bad_step), ("reward_std", 11))
        self.assertIsNone(sr.collapse(rows + [row(13, reward_std=0.25)]))

    def test_last_good_checkpoint(self):
        self.assertEqual(sr.last_good_checkpoint(14, [0, 5, 10, 15]), 10)
        self.assertEqual(sr.last_good_checkpoint(12, [0, 5, 10]), 10)
        self.assertEqual(sr.last_good_checkpoint(11, [0, 5, 10]), 5)  # checkpoint 10 produced step 11's rollouts
        self.assertEqual(sr.last_good_checkpoint(3, [0, 5]), 0)


class LearningRateTest(unittest.TestCase):
    def test_stable_second_seed_allows_2e5(self):
        rows = [row(s, entropy=0.6 - 0.002 * s) for s in range(1, 21)]
        decision = sr.lr_decision(rows, high=2e-5, floor=1e-5)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["production_lr"], 2e-5)

    def test_any_failed_check_keeps_1e5(self):
        falling = [row(s, entropy=0.6 - 0.02 * s) for s in range(1, 21)]
        spiky = [row(s, grad=3.0 if s == 9 else 0.7) for s in range(1, 21)]
        drifting = [row(s, corr=0.97 if s == 15 else 0.997) for s in range(1, 21)]
        for rows, check in ((falling, "entropy_flat"), (spiky, "grad_norm_spikes"), (drifting, "probs_corr")):
            decision = sr.lr_decision(rows, high=2e-5, floor=1e-5)
            self.assertEqual(decision["production_lr"], 1e-5, check)
            self.assertFalse(decision["checks"][check]["passed"], check)

    def test_missing_data_keeps_1e5(self):
        rows = [row(s, grad=None) for s in range(1, 21)]
        for r in rows:
            r["policy"]["grad_norm"] = None
        decision = sr.lr_decision(rows, high=2e-5, floor=1e-5)
        self.assertEqual(decision["production_lr"], 1e-5)
        self.assertIn("missing data", decision["note"])


if __name__ == "__main__":
    unittest.main()
