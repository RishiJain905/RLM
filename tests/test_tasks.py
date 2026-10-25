"""H3 on the host: fresh 64-bit seeds, and eval seed lists kept out of git."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlm_rl import tasks

ROOT = Path(__file__).resolve().parents[1]


class SeedTest(unittest.TestCase):
    def test_seeds_are_fresh_64_bit_numbers(self):
        seeds = [tasks.new_seed() for _ in range(1000)]
        self.assertTrue(all(0 <= s < 2 ** 64 for s in seeds))
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertGreater(max(seeds), 2 ** 60)  # not the small fixed seeds of the 20-step run (1000 + i, 99_999)

    def test_private_seed_lists_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(tasks, "PRIVATE_DIR", Path(tmp)):
            seeds = [tasks.new_seed() for _ in range(64)]
            path = tasks.save_private_seeds("selection", seeds)
            self.assertEqual(tasks.load_private_seeds("selection"), seeds)
            self.assertEqual(path.parent, Path(tmp) / "seeds")

    @unittest.skipIf(shutil.which("git") is None, "git not installed")
    def test_private_folder_is_git_ignored(self):
        self.assertEqual(tasks.PRIVATE_DIR, ROOT / "private")
        result = subprocess.run(["git", "check-ignore", "-q", "private/seeds/selection.json"], cwd=ROOT)
        self.assertEqual(result.returncode, 0, "private/ must be in .gitignore")


if __name__ == "__main__":
    unittest.main()


class SoloRetirementTest(unittest.TestCase):
    """Step-12 task shift: a task of a sub-call family that's solved without any sub-call retires."""

    def test_solved_without_sub_calls_retires(self):
        from types import SimpleNamespace
        from unittest import mock

        from rlm_rl import tasks as rtasks

        def ep(correct, subcalls, task="t1", family="fam"):
            return SimpleNamespace(task_id=task, family=family, infra_error=None, reward=float(correct),
                                   correct=correct, subcalls_used=subcalls)

        with mock.patch.object(rtasks, "RETIRABLE", ("fam",)), mock.patch.object(rtasks, "SOLO_RETIRABLE", ("fam",)):
            r = rtasks.Retirement()
            self.assertEqual(r.update(1, [[ep(True, 3), ep(True, 0), ep(False, 0), ep(False, 2)]]), [])  # one delegated
            self.assertEqual(r.update(2, [[ep(True, 0, "t2"), ep(True, 0, "t2"), ep(False, 1, "t2"),
                                           ep(False, 0, "t2")]]), ["t2"])
            self.assertEqual(r.retired["t2"]["why"], "solved without sub-calls")
            self.assertEqual(r.update(3, [[ep(True, 0, "t3"), ep(False, 0, "t3"), ep(False, 0, "t3"),
                                           ep(False, 0, "t3")]]), [])  # under half right: not retired
            self.assertEqual(r.update(4, [[ep(True, 1, "t4"), ep(True, 1, "t4")]]), ["t4"])  # all-1 still retires
            self.assertEqual(r.retired["t4"]["why"], "all-1")
        with mock.patch.object(rtasks, "RETIRABLE", ("fam",)):
            r = rtasks.Retirement()  # outside SOLO_RETIRABLE nothing changes
            self.assertEqual(r.update(1, [[ep(True, 0), ep(True, 0), ep(False, 0)]]), [])


class PoolFallbackTest(unittest.TestCase):
    """Step-12 task shift: edit_spec's and edit_bugs's pools stand in for each other when one runs dry."""

    def test_a_dry_pool_draws_from_its_partner_without_retired_tasks(self):
        from types import SimpleNamespace
        from unittest import mock

        from rlm_rl import tasks as rtasks

        seen = []

        def fake(family, level, seed=None, task_id=None, exclude=None):
            seen.append((family, set(exclude or ())))
            if family == "edit_spec":
                raise rtasks.PoolEmpty("edit_spec has no task left")
            return SimpleNamespace(family=family, task_id=f"{family}-{len(seen)}")

        with mock.patch.object(rtasks, "make_task", fake):
            out = rtasks.draw(3, {"edit_spec": 1.0}, {"edit_spec": 2, "edit_bugs": 4}, exclude={"old"})
        self.assertEqual([t.family for t in out], ["edit_bugs"] * 3)
        bug_draws = [ex for fam, ex in seen if fam == "edit_bugs"]
        self.assertTrue(all("old" in ex for ex in bug_draws))  # retired tasks stay out
        self.assertEqual(len(bug_draws[-1] - {"old"}), 2)  # and so do this step's earlier draws
        with mock.patch.object(rtasks, "make_task", fake), self.assertRaises(rtasks.PoolEmpty):
            rtasks.draw(1, {"edit_spec": 1.0}, {"edit_spec": 2})  # no partner in the run


class ReleveledPoolTest(unittest.TestCase):
    def test_the_build_from_spec_training_split_is_levelled_by_code_size(self):
        """prod-v2 (2026-10-25): code size, not the function count, told solvable level-1 tasks from unsolvable ones."""
        from rlm_rl.tasks import editspec

        records = [{"id": f"t{i:02d}", "level": 1 + i % 5, "gold_lines": g} for i, g in
                   enumerate([40, 9, 120, 30, 30, 55, 12, 80, 200, 61])]
        out = editspec.relevel_by_rank(records, "gold_lines")
        by_id = {r["id"]: r for r in out}
        self.assertEqual([r["gold_lines"] for r in sorted(out, key=lambda r: (r["level"], r["gold_lines"]))],
                         sorted(r["gold_lines"] for r in records))
        self.assertEqual([by_id["t01"]["level"], by_id["t06"]["level"], by_id["t08"]["level"]], [1, 1, 5])
        self.assertEqual(sorted(r["level"] for r in out), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # equal bins
        self.assertEqual((by_id["t03"]["level"], by_id["t04"]["level"]), (2, 2))  # a tie stays in id order
        self.assertEqual(by_id["t01"]["stored_level"], 2)
        self.assertEqual(records[1]["level"], 2)  # the stored records are not changed
        self.assertEqual(editspec.POOL.relevel_train, "gold_lines")
