"""H13: the difficulty controller moves levels correctly on synthetic pass rates (spec §6, plan step 5).

    python3 -m unittest -v tests.test_controller
"""

import json
import unittest
from types import SimpleNamespace

from rlm_rl import controller
from rlm_rl.controller import DifficultyController


def episodes(family: str, level: int, passes: int, total: int = 16, infra: int = 0) -> list:
    out = [SimpleNamespace(family=family, level=level, correct=i < passes, infra_error=None) for i in range(total)]
    out += [SimpleNamespace(family=family, level=level, correct=None, infra_error="429") for _ in range(infra)]
    return out


class ControllerTest(unittest.TestCase):
    def run_steps(self, ctl, rates: list[float], family="a", total=16, start=1):
        """Feed one step per rate at the family's current level. Returns the moves, step by step."""
        out = []
        for i, rate in enumerate(rates):
            out.append(ctl.update(start + i, episodes(family, ctl.level(family), round(rate * total), total)))
        return out

    def test_moves_up_above_70_percent_after_three_steps(self):
        ctl = DifficultyController({"a": 5})
        moves = self.run_steps(ctl, [0.9, 0.9, 0.9])
        self.assertEqual([len(m) for m in moves], [0, 0, 1])  # not before its window holds 3 steps
        move = moves[2][0]
        self.assertEqual((move.family, move.from_level, move.to_level, move.direction, move.applied),
                         ("a", 1, 2, "up", True))
        self.assertEqual(ctl.level("a"), 2)

    def test_moves_down_below_20_percent_but_never_below_level_1(self):
        ctl = DifficultyController({"a": 5}, levels={"a": 2})
        self.run_steps(ctl, [0.1, 0.1, 0.1])
        self.assertEqual(ctl.level("a"), 1)
        moves = self.run_steps(ctl, [0.0] * 6, start=4)
        self.assertEqual((sum(map(len, moves)), ctl.level("a")), (0, 1))

    def test_stays_inside_the_band_and_at_the_edges(self):
        ctl = DifficultyController({"a": 5}, levels={"a": 3})
        moves = self.run_steps(ctl, [0.7, 0.7, 0.7, 0.2, 0.2, 0.2, 0.5, 0.25, 0.65], total=10)
        self.assertEqual((sum(map(len, moves)), ctl.level("a")), (0, 3))  # 70% and 20% exactly don't move

    def test_the_rate_is_over_the_window_not_the_last_step(self):
        ctl = DifficultyController({"a": 5})
        self.run_steps(ctl, [0.94, 0.94, 0.0])  # 15/16, 15/16, 0/16: 63% over three steps (not unanimous)
        self.assertEqual(ctl.level("a"), 1)
        self.run_steps(ctl, [0.94], start=4)  # now 15, 0, 15: 63% again
        self.assertEqual(ctl.level("a"), 1)
        self.run_steps(ctl, [0.94], start=5)  # 0, 15, 15
        self.assertEqual(ctl.level("a"), 1)
        self.run_steps(ctl, [0.94], start=6)  # 15, 15, 15: 94%
        self.assertEqual(ctl.level("a"), 2)

    def test_a_unanimous_window_moves_at_once(self):
        """Every episode right (or wrong) on at least 8 episodes moves without waiting for 3 steps (prod-v1, edit synth)."""
        ctl = DifficultyController({"a": 5})
        self.assertEqual(len(ctl.update(1, episodes("a", 1, 4, total=4))), 0)  # 4 episodes: too few
        moves = ctl.update(2, episodes("a", 1, 4, total=4))  # 8 of 8
        self.assertEqual([(m.from_level, m.to_level) for m in moves], [(1, 2)])
        moves = ctl.update(3, episodes("a", 2, 16, total=16))  # 16 of 16 in one step
        self.assertEqual(ctl.level("a"), 3)
        moves = ctl.update(4, episodes("a", 3, 0, total=16))  # 0 of 16: down at once
        self.assertEqual((moves[0].direction, ctl.level("a")), ("down", 2))
        ctl.update(5, episodes("a", 2, 15, total=16))  # 94%: back to waiting for 3 steps
        self.assertEqual(ctl.level("a"), 2)
        self.assertTrue(controller.settings()["unanimous_moves_at_once"])

    def test_waits_three_steps_after_a_move(self):
        ctl = DifficultyController({"a": 5})
        moves = self.run_steps(ctl, [0.9] * 9)
        self.assertEqual([len(m) for m in moves], [0, 0, 1, 0, 0, 1, 0, 0, 1])
        self.assertEqual(ctl.level("a"), 4)
        self.assertEqual([m.step for m in ctl.history], [3, 6, 9])

    def test_only_episodes_at_the_current_level_count(self):
        ctl = DifficultyController({"a": 5})
        self.run_steps(ctl, [0.9, 0.9, 0.9])
        for step in (4, 5, 6):  # stragglers drawn at level 1 before the move don't count toward level 2
            self.assertEqual(ctl.update(step, episodes("a", 1, 16)), [])
        self.assertEqual((ctl.level("a"), ctl.summary()["a"]["window_steps"]), (2, 0))

    def test_top_level_is_saturation_not_a_move(self):
        ctl = DifficultyController({"a": 2, "b": 1}, levels={"a": 2})
        for step in (1, 2, 3):
            moves = ctl.update(step, episodes("a", 2, 15) + episodes("b", 1, 12))
        self.assertEqual(moves, [])
        self.assertTrue(ctl.saturated())  # stop rule 4: every family at its top with a pass rate above 70%
        ctl.update(4, episodes("a", 2, 0) + episodes("b", 1, 12))
        self.assertFalse(ctl.saturated())

    def test_infra_and_absent_steps_do_not_count(self):
        ctl = DifficultyController({"a": 5, "b": 5})
        for step in (1, 2, 3):
            ctl.update(step, episodes("a", 1, 15) + episodes("b", 1, 0, total=0, infra=4))
        self.assertEqual(ctl.levels(), {"a": 2, "b": 1})
        self.assertEqual(ctl.summary()["b"]["window_steps"], 0)
        # b only appears every other step, so its window spans more steps
        for step in range(4, 10):
            ctl.update(step, episodes("b", 1, 14) if step % 2 else [])
        self.assertEqual(ctl.level("b"), 2)

    def test_too_few_episodes_do_not_move(self):
        ctl = DifficultyController({"a": 5})
        for step in (1, 2, 3):
            ctl.update(step, episodes("a", 1, 2, total=2))  # 6 episodes in the window, all passing
        self.assertEqual(ctl.level("a"), 1)
        ctl.update(4, episodes("a", 1, 4, total=4))  # 8 now
        self.assertEqual(ctl.level("a"), 2)

    def test_frozen_records_but_never_moves(self):
        """The second seed runs at level 1 (plan step 10): the controller logs what it would do."""
        ctl = DifficultyController({"a": 5}, frozen=True)
        moves = self.run_steps(ctl, [0.9, 0.9, 0.9, 0.9])
        self.assertEqual(ctl.level("a"), 1)
        self.assertEqual([(m.to_level, m.applied) for ms in moves for m in ms], [(2, False), (2, False)])

    def test_state_round_trips_through_json(self):
        ctl = DifficultyController({"a": 5, "b": 3})
        self.run_steps(ctl, [0.9, 0.9, 0.9, 0.95])
        state = json.loads(json.dumps(ctl.state_dict()))
        again = DifficultyController.from_state(state)
        self.assertEqual(again.state_dict(), ctl.state_dict())
        self.assertEqual(state["settings"], controller.settings())
        self.assertEqual([len(again.update(5, episodes("a", 2, 16))), len(ctl.update(5, episodes("a", 2, 16)))],
                         [0, 0])
        self.assertEqual(again.state_dict(), ctl.state_dict())

    def test_bad_setup(self):
        for args in (({},), ({"a": 0},)):
            with self.assertRaises(ValueError):
                DifficultyController(*args)
        with self.assertRaises(ValueError):
            DifficultyController({"a": 3}, levels={"a": 4})
        with self.assertRaises(KeyError):
            DifficultyController({"a": 3}).update(1, episodes("z", 1, 1))


if __name__ == "__main__":
    unittest.main()
