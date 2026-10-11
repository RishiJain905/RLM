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
