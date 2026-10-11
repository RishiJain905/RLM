"""H1: the calibrated reward (spec §3, decision 1b). A blind guess earns only its lucky exact hits.

    python3 -m unittest -v tests.test_reward
"""

import random
import statistics
import unittest
from pathlib import Path
from types import SimpleNamespace


from rlm_rl import data, reward, tasks
from rlm_rl.reward import blind_score, near_miss_credit, normalize_entity, parse_number, score, score_answer, set_f1

INT = {"type": "int", "blind": 60}  # gold 70 below: the blind guess is 10 off


class ParseTest(unittest.TestCase):
    def test_first_number(self):
        cases = {"42": 42, "about 42 tickets (of 300)": 42, "$1,234.50": 1234.5, "-5": -5, "is -5": -5,
                 "host-42": 42, "1,234": 1234, 7: 7, 2.5: 2.5}
        for value, expected in cases.items():
            self.assertEqual(parse_number(value), expected, value)
        for value in ("none", "", None, True, [1], float("nan"), "nan", "inf"):
            self.assertIsNone(parse_number(value), value)

    def test_entities(self):
        self.assertEqual(normalize_entity('  "WS-Finance-07" '), "ws-finance-07")
        self.assertEqual(normalize_entity("`svc_backup`"), "svc_backup")
        self.assertEqual(normalize_entity(12.0), "12")
        for value in ("", "  ''  ", None, ["a"]):
            self.assertIsNone(normalize_entity(value))

    def test_set_f1(self):
        self.assertEqual(set_f1({1, 2}, {1, 2}), 1.0)
        self.assertEqual(set_f1(set(), set()), 1.0)
        self.assertEqual(set_f1({1}, {2}), 0.0)
        self.assertAlmostEqual(set_f1({1, 2, 3}, {1, 2}), 0.8)


class IntegerTest(unittest.TestCase):
    def test_exact(self):
        for answer in (70, 70.0, "70", "I count 70.", "$70"):
            s = score_answer(answer, 70, INT)
            self.assertEqual((s.reward, s.correct, s.components["exact"]), (1.0, True, 1.0), answer)

    def test_near_miss_is_calibrated(self):
        # 0.8 x max(0, 1 - err / e_blind) with e_blind = 10
        for answer, expected in ((69, 0.72), (72, 0.64), (65, 0.4), (61, 0.08), (60, 0.0), (55, 0.0), (90, 0.0)):
            s = score_answer(answer, 70, INT)
            self.assertAlmostEqual(s.reward, expected, places=4, msg=answer)
            self.assertFalse(s.correct)
        self.assertEqual(score_answer(70.4, 70, INT).correct, False)  # an integer must be equal

    def test_blind_guess_earns_nothing_but_its_exact_hits(self):
        self.assertEqual(score_answer(60, 70, INT).reward, 0.0)
        self.assertEqual(score_answer(70, 70, {"type": "int", "blind": 70}).reward, 1.0)
        # when the blind guess was exact, a near miss earns nothing
        self.assertEqual(score_answer(71, 70, {"type": "int", "blind": 70}).reward, 0.0)

    def test_unparseable_or_missing_is_zero(self):
        for answer in (None, "no idea", "", [70]):
            s = score_answer(answer, 70, INT)
            self.assertEqual((s.reward, s.correct, s.components["parsed"]), (0.0, False, 0.0), answer)

    def test_first_number_only(self):
        """Hedging doesn't pay: "69 or 70" is scored as 69."""
        self.assertAlmostEqual(score_answer("69 or 70", 70, INT).reward, 0.72)

    def test_near_blind_is_logged_not_rewarded(self):
        self.assertEqual(score_answer(62, 70, INT).components["near_blind"], 1.0)  # 2 from the guess, 8 from gold
        self.assertEqual(score_answer(68, 70, INT).components["near_blind"], 0.0)
        self.assertEqual(score_answer(70, 70, INT).components["near_blind"], 0.0)

    def test_slope(self):
        self.assertEqual(near_miss_credit(5, 10), 0.4)
        old = reward.NEAR_MISS_SLOPE
        try:
            reward.NEAR_MISS_SLOPE = 2
            self.assertEqual(near_miss_credit(5, 10), 0.0)
            self.assertAlmostEqual(near_miss_credit(2, 10), 0.48)
        finally:
            reward.NEAR_MISS_SLOPE = old


class OtherTypesTest(unittest.TestCase):
    def test_decimal(self):
        grading = {"type": "float", "blind": 100.0}
        self.assertTrue(score_answer("$1,234.504", 1234.5, grading).correct)
        self.assertFalse(score_answer(1234.51, 1234.5, grading).correct)
        self.assertGreater(score_answer(1234.51, 1234.5, grading).reward, 0.79)

    def test_entity_has_no_partial_credit(self):
        grading = {"type": "entity"}
        self.assertEqual(score_answer(" 'WS-07' ", "ws-07", grading).reward, 1.0)
        self.assertEqual(score_answer("ws-08", "ws-07", grading).reward, 0.0)
        self.assertEqual(score_answer(["ws-07"], "ws-07", grading).reward, 0.0)

    def test_list_by_set_f1_against_the_blind_strategy(self):
        gold = ["a", "b", "c", "d"]
        blind = ["a", "b", "c", "d", "e", "f", "g", "h"]  # e.g. every candidate the prompt shows: F1 = 2/3
        grading = {"type": "list", "item": "entity", "blind": blind}
        self.assertEqual(score_answer(["D", "c", "b", "a", "a"], gold, grading).reward, 1.0)
        self.assertEqual(score_answer(blind, gold, grading).reward, 0.0)
        s = score_answer(["a", "b", "c"], gold, grading)  # F1 = 6/7
        self.assertAlmostEqual(s.reward, round(0.8 * (6 / 7 - 2 / 3) / (1 - 2 / 3), 4))
        self.assertEqual(score_answer("a, b; c\nd", gold, grading).reward, 1.0)  # a string is split
        self.assertEqual(score_answer([], gold, grading).reward, 0.0)
        ints = {"type": "list", "item": "int", "blind": [1]}
        self.assertEqual(score_answer(["3", 2, "$1"], [1, 2, 3], ints).reward, 1.0)

    def test_grading_must_carry_a_blind_guess(self):
        for grading, gold in (({"type": "int"}, 3), ({"type": "list", "item": "entity"}, ["a"]),
                              ({"type": "list", "item": "thing", "blind": []}, ["a"]), ({"type": "count"}, 3),
                              ({"type": "entity"}, "")):
            with self.assertRaises(ValueError, msg=grading):
                score_answer(1, gold, grading)


class EpisodeScoreTest(unittest.TestCase):
    def test_uses_only_a_valid_submit(self):
        task = SimpleNamespace(answer=70, grading=INT)
        self.assertEqual(score(task, SimpleNamespace(submitted=True, answer=70, answer_valid=True)).reward, 1.0)
        self.assertEqual(score(task, SimpleNamespace(submitted=False, answer=70, answer_valid=True)).reward, 0.0)
        self.assertEqual(score(task, SimpleNamespace(submitted=True, answer=None, answer_valid=False)).reward, 0.0)


class BlindGuesserTest(unittest.TestCase):
    """Done when: a blind guesser scores no more than its exact-hit rate on every task family (H1).

    scripts/reward_check.py prints the full table, including guessers that add an offset or noise.
    """

    N = 300  # the ticket families; plan step 7's families are slower to make, and scripts/data_check.py runs 20 each
    N_DATA = 12

    def test_every_family_and_level(self):
        for family in sorted(tasks.GENERATORS):
            if tasks.training_sources(family) and not data.available() or                     family.startswith("repo_loc") and not data.derived("lca", "index-train.jsonl.gz").exists():
                continue  # needs the step 7 data root (rlm_rl/data.py)
            n = self.N if family.startswith("tickets") else self.N_DATA
            for level in range(1, tasks.max_levels([family])[family] + 1):
                task_list = [tasks.make_task(family, level) for _ in range(n)]
                rewards = [blind_score(t).reward for t in task_list]
                exact_rate = statistics.fmean(float(blind_score(t).correct) for t in task_list)
                with self.subTest(family=family, level=level):
                    self.assertLessEqual(statistics.fmean(rewards), exact_rate + 1e-12)
                    self.assertTrue(all(r in (0.0, 1.0) for r in rewards))


class RewardCheckScriptTest(unittest.TestCase):
    """Step 8: scripts/reward_check.py must score every family a generator can make, and a family mixes
    answer types (spec §3). The step 5 script read t.grading['blind'] and subtracted list from list, so it
    crashed (KeyError / TypeError) on the needle, aggregation, callgraph and repo_loc families -- the four
    that step 7 added. This pins the fix: each answer type dispatches through the real scorer."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        cls.script = importlib.util.module_from_spec(
            importlib.util.spec_from_file_location(
                "reward_check", Path(__file__).resolve().parents[1] / "scripts" / "reward_check.py"))
        cls.script.__loader__.exec_module(cls.script)

    def fake_task(self, kind: str):
        shapes = {
            "int": (SimpleNamespace(answer=70, grading={"type": "int", "blind": 60}),),
            "float": (SimpleNamespace(answer=1234.5, grading={"type": "float", "blind": 100.0}),),
            "entity": (SimpleNamespace(answer="ws-07", grading={"type": "entity"}),),  # no blind guess
            "list": (SimpleNamespace(answer=["a", "b", "c", "d"], grading={"type": "list", "item": "entity",
                                                                           "blind": ["a", "b", "c", "d", "e", "f"]}),),
        }
        return shapes[kind][0]

    def test_guess_rewards_dispatches_by_answer_type(self):
        rng = random.Random(0)
        for kind in ("int", "float", "entity", "list"):
            row = self.script.guess_rewards(self.fake_task(kind), rng)  # crashed here before the fix
            with self.subTest(kind=kind):
                self.assertEqual(sorted(row), sorted(["offset", "noisy_half", "noisy_one",
                                                      "reader1", "reader2", "reader3"]))
                self.assertTrue(all(0.0 <= v <= 1.0 for v in row.values()), row)
                if kind == "entity":
                    self.assertEqual(sorted(row.values()), [0.0] * 6)  # a name has no guess and no credit

    def test_blind_score_of_each_shape(self):
        for kind in ("int", "float", "entity", "list"):
            s = blind_score(self.fake_task(kind))  # the entity shape raised KeyError('blind') before
            with self.subTest(kind=kind):
                self.assertLessEqual(s.reward, float(s.correct))

    def test_check_family_runs_a_real_family(self):
        """End to end over the generator, on a family whose tasks the step 5 script could score."""
        r = self.script.check_family("tickets_customer", 1, 6, seed=1)
        self.assertEqual((r["n"], r["types"]), (6, {"int": 6}))
        self.assertTrue(r["passes"])


if __name__ == "__main__":
    unittest.main()

