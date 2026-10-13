"""A slow disk costs time, never the reward (Rishi, 2026-10-13; spec §12.3, §12.5).

The grader reruns a timed-out run once, alone, at double the cap; only a second timeout is the model's. A container
that doesn't start in time is launched again, and one that never starts is an infra error, never a 0.
"""

import asyncio
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from rlm_rl import bridge, grader, reward
from rlm_rl.grader import TestSpec
from tests.fake_model import AsyncCase
from tests.test_grader import file_patch

SPEC = TestSpec(runner="pytest", scope=["tests/test_a.py"], f2p=["tests/test_a.py::test_new"],
                p2p=["tests/test_a.py::test_old"])
PASSED = {"tests/test_a.py::test_new": "passed", "tests/test_a.py::test_old": "passed"}
TIMEOUT = {"applied": True, "why": "", "deadline_hit": True, "build_failed": False, "verdict": None,
           "results": {"tests/test_a.py::test_old": "passed"}}  # the run was cut off before test_new reported
DONE = {"applied": True, "why": "", "deadline_hit": False, "build_failed": False, "verdict": None, "results": PASSED}


def edit_episode(result: grader.GradeResult, diff: str):
    return types.SimpleNamespace(grade={**result.as_dict(), "graded": True}, diff=diff, submitted=True, edit_stats={})


class GraderRetryTest(unittest.TestCase):
    def setUp(self):
        self.ws = Path(tempfile.mkdtemp())
        (self.ws / "tree" / "src").mkdir(parents=True)
        (self.ws / "tree" / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
        self.addCleanup(shutil.rmtree, self.ws, True)
        self.diff = file_patch("src/a.py", ["x = 1"], ["x = 2"])

    def grade_with(self, outcomes):
        caps, outcomes = [], list(outcomes)

        async def once(checked, spec, cap_s, **kw):
            caps.append(cap_s)
            return outcomes.pop(0)

        with mock.patch.object(grader, "_grade_once", once):
            result = asyncio.run(grader.grade(self.diff, SPEC, ws_dir=self.ws, image=None))
        return result, caps

    def test_a_timeout_then_a_pass_scores_1(self):
        result, caps = self.grade_with([TIMEOUT, DONE])
        self.assertEqual(caps, [SPEC.timeout_s, 2 * SPEC.timeout_s])  # the rerun gets double the cap
        self.assertEqual((result.retries, result.deadline_hit, result.resolved), (1, False, True))
        self.assertEqual(reward.score_edit(None, edit_episode(result, self.diff)).reward, 1.0)

    def test_two_timeouts_score_0(self):
        result, caps = self.grade_with([TIMEOUT, TIMEOUT])
        self.assertEqual(len(caps), 2)  # never a third run
        self.assertEqual((result.retries, result.deadline_hit, result.resolved), (1, True, False))
        self.assertEqual(reward.score_edit(None, edit_episode(result, self.diff)).reward, 0.0)

    def test_no_retry_without_a_timeout(self):
        result, caps = self.grade_with([DONE])
        self.assertEqual((caps, result.retries, result.resolved), ([SPEC.timeout_s], 0, True))

    def test_the_task_cap_and_cell_limit_follow_loaded_timings(self):
        from rlm_rl import editdata

        self.assertEqual(editdata.grader_cap(30.0), bridge.EDIT_LIMITS.grader_timeout_s)  # never below the spec's
        self.assertEqual(editdata.grader_cap(400.0), 1600)  # 4x the slowest gold run under load
        self.assertIsNone(editdata.cell_limit(None))
        self.assertIsNone(editdata.cell_limit(bridge.EDIT_LIMITS.cell_timeout_s / 2))
        self.assertEqual(editdata.cell_limit(200.0), 420)  # twice the gold test command, in 30 s steps


class GraderSlotsTest(unittest.TestCase):
    def test_a_retry_runs_alone(self):
        async def scenario():
            slots, log = grader.GraderSlots(limit=None), []

            async def shared(name, hold):
                async with slots.shared():
                    log.append(f"{name} in")
                    await asyncio.sleep(hold)
                    log.append(f"{name} out")

            async def alone():
                await asyncio.sleep(0.01)  # after a1 holds a slot
                async with slots.alone():
                    log.append("retry in")
                    await asyncio.sleep(0.05)
                    log.append("retry out")

            async def late():
                await asyncio.sleep(0.03)  # arrives while the retry is waiting or running
                async with slots.shared():
                    log.append("late in")

            await asyncio.gather(shared("a1", 0.05), alone(), late())
            return log

        log = asyncio.run(scenario())
        self.assertLess(log.index("a1 out"), log.index("retry in"))  # waits for running graders
        self.assertLess(log.index("retry out"), log.index("late in"))  # and keeps new ones waiting

    def test_a_concurrency_cap(self):
        async def scenario():
            slots, active, peak = grader.GraderSlots(limit=2), [0], [0]

            async def one():
                async with slots.shared():
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                    await asyncio.sleep(0.01)
                    active[0] -= 1

            await asyncio.gather(*(one() for _ in range(6)))
            return peak[0]

        self.assertEqual(asyncio.run(scenario()), 2)


@unittest.skipUnless(sys.platform == "linux", "the sandbox runs inside WSL2 (decision 3b)")
class SlowStartTest(AsyncCase):
    """A container that comes up late is launched again; one that never comes up is an infra error, never a 0."""

    def sandbox(self, delays):
        from rlm_rl import sandbox

        counter = Path(tempfile.mkdtemp()) / "n"
        counter.write_text("0")
        self.addCleanup(shutil.rmtree, counter.parent, True)
        script = ("import json, sys, time\n"
                  f"p = {str(counter)!r}\nn = int(open(p).read())\nopen(p, 'w').write(str(n + 1))\n"
                  f"delays = {list(delays)!r}\ntime.sleep(delays[min(n, len(delays) - 1)])\n"
                  "print(json.dumps({'ready': True, 'landlock': 3, 'python': '3.12'}), flush=True)\nsys.stdin.read()\n")

        class Slow(sandbox.Sandbox):
            def _run_args(self, name):
                return [sys.executable, "-c", script]

        async def complete(prompt):
            return "ok"

        limits = sandbox.SandboxLimits(startup_timeout_s=1.0)
        return Slow("ctx", complete, limits=limits, run_id="test-slow-start", docker="true")

    async def test_a_late_container_is_launched_again(self):
        box = self.sandbox([3.0, 0.0])
        async with box:
            self.assertEqual((box.start_retries, box.landlock_abi), (1, 3))

    async def test_a_container_that_never_starts_is_infra_not_a_zero(self):
        from rlm_rl import sandbox
        from rlm_rl.env import Task, run_episode

        box = self.sandbox([3.0, 3.0])
        with self.assertRaises(sandbox.InfraError):
            async with box:
                pass
        factory = lambda context, complete, **kw: self.sandbox([3.0, 3.0])  # noqa: E731
        client = mock.MagicMock()
        client.subcall_model = "fake"
        client.subcaller.return_value = None
        task = Task("t", "f", "How many lines?", "a\nb", answer=2, grading={"type": "int", "blind": 1})
        ep = await run_episode(task, client=client, root_model="fake:step0", sandbox_factory=factory)
        self.assertIsNotNone(ep.infra_error)
        self.assertIsNone(ep.reward)  # never trained on as a 0


if __name__ == "__main__":
    unittest.main()
