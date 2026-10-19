"""Edit episodes through the loop (spec §12): limits, prompt, grading, reward, detectors, metrics, retirement.

The fast cases run on Linux with the in-process edit sandbox and the local grader (tests/fakes.py),
which runs edit_synth's tests with this machine's Python. DockerEditTest repeats a full episode
through the real edit sandbox and grader.
"""

import base64
import sys
import unittest

import openai

from rlm_rl import bridge, detectors, editbox, editmetrics, env, monitor, repl_tools, sandbox, tasks
from rlm_rl.client import ModelClient
from rlm_rl.env import Episode, filter_step, run_episode
from rlm_rl.tasks import editsynth
from tests.fake_model import AsyncCase, FakeModelServer
from tests.test_env import BASE, FAST, ROOT, docker_ready

linux_only = unittest.skipUnless(sys.platform == "linux", "the edit sandbox and its fakes run on Linux (WSL2)")


def writes(files: dict) -> str:
    return "\n".join(f"__import__('pathlib').Path('/work/repo/{p}').write_bytes(__import__('base64').b64decode('{base64.b64encode(t.encode()).decode()}'))"
                     for p, t in files.items())


def script(*cells: str):
    """A policy that replies with these cells in order, one per turn."""
    def policy(messages):
        turn = sum(m["role"] == "assistant" for m in messages)
        return f"Next.\n```python\n{cells[min(turn, len(cells) - 1)]}\n```"
    return policy


LOOK = "import os\nprint(len(os.listdir('/work/repo')))"
TESTS = "import subprocess\nr = subprocess.run(['python3', '-m', 'unittest', 'discover', '-s', 'tests'], cwd='/work/repo', capture_output=True, text=True)\nprint(r.returncode)"
SUBMIT = "submit('fixed it')"


class EditEnvCase(AsyncCase):
    real = False

    async def asyncSetUp(self):
        self.server = FakeModelServer()
        await self.server.start()
        self.openai = openai.AsyncOpenAI(base_url=self.server.base_url, api_key="test-key")
        self.client = ModelClient(self.openai, subcall_model=BASE, retry=FAST)
        self.task = editsynth.make_task(31337, "edit_synth", 1, kind="fix")

    async def asyncTearDown(self):
        await self.openai.close()
        await self.server.close()

    async def episode(self, *cells, task=None, **kw) -> Episode:
        from tests.fakes import FakeEditSandbox, local_grade

        self.server.policy = script(*cells)
        if not self.real:
            kw.setdefault("edit_sandbox_factory", FakeEditSandbox)
            kw.setdefault("grade", local_grade)
        return await run_episode(task or self.task, client=self.client, root_model=ROOT, sandbox_factory=None, **kw)


@linux_only
class EditEpisodeTest(EditEnvCase):
    async def test_a_resolved_episode(self):
        gold = writes(self.task.grading["gold_files"])
        ep = await self.episode(LOOK, gold + "\n" + TESTS, "print(llm('Review this change:\\ndiff --git a/x b/x\\n@@ -1 +1 @@\\n-a\\n+b'))", SUBMIT)
        self.assertIsNone(ep.infra_error, ep.infra_error)
        self.assertEqual((ep.reward, ep.correct, ep.submitted), (1.0, True, True))
        self.assertEqual(ep.limits, bridge.limits(self.task.family).as_dict())  # prompt v3: 30 turns
        self.assertEqual(ep.limits["max_turns"], bridge.EDIT_MAX_TURNS_V3)
        self.assertEqual(ep.language, "python")
        self.assertEqual(ep.messages[1]["content"], bridge.edit_first_message(
            self.task.question, language="python", n_files=self.task.edit["n_files"], test_hint=editsynth.TEST_HINT,
            limits=bridge.limits(self.task.family)))
        self.assertIn("(29 turns left, 30 sub-calls left)", ep.messages[3]["content"])  # prompt v3: 30 turns
        roots = [b for b in self.server.bodies if b.get("logprobs")]
        self.assertEqual({b["max_tokens"] for b in roots}, {bridge.EDIT_LIMITS.root_max_tokens})
        c = ep.reward_components
        for key in set(editmetrics.EDIT_KEYS) - {"episodes", "resolved"}:
            self.assertIn(key, c, key)
        self.assertEqual((c["resolved"], c["f2p_pass"], c["p2p_broken"], c["empty_diff"]), (1.0, 1.0, 0.0, 0.0))
        self.assertEqual((c["tested_before_submit"], c["test_runs"], c["edit_cycles"], c["turns_to_first_edit"]),
                         (1.0, 1.0, 1.0, 2.0))
        self.assertEqual((c["review_subcalls"], c["repeated_cells"]), (1.0, 0.0))
        self.assertGreater(c["diff_lines"], 0)
        self.assertEqual(c["files_changed"], 1.0)
        self.assertGreater(c["subcall_read_share"], 0)
        self.assertTrue(ep.grade["graded"] and ep.grade["resolved"])
        self.assertEqual(ep.hacks, [])

    async def test_coding_tool_metrics(self):
        gold = self.task.grading["gold_files"]
        path, text = next(iter(gold.items()))
        write_all = "\n".join(f"write_file({p!r}, __import__('base64').b64decode({base64.b64encode(t.encode()).decode()!r}).decode())"
                              for p, t in gold.items())
        cells = [f"print(read_file({path!r}, n=5))",
                 f"edit_file({path!r}, 'no such text anywhere', 'x')",  # raises: one edit error
                 write_all, "print(shell('python3 -m unittest discover -s tests'))", "print(diff()[:200])", SUBMIT]
        ep = await self.episode(*cells)
        self.assertEqual((ep.reward, ep.correct), (1.0, True))
        c = ep.reward_components
        self.assertEqual(c["tool_calls"], {**dict.fromkeys(repl_tools.TOOL_NAMES, 0), "read_file": 1, "edit_file": 1,
                                           "write_file": len(gold), "shell": 1, "diff": 1})
        self.assertEqual((c["edit_errors"], c["raw_io_share"]), (1.0, 0.0))
        self.assertEqual((c["test_runs"], c["tested_before_submit"], c["edit_cycles"]), (1.0, 1.0, 1.0))
        raw = await self.episode(LOOK, writes(gold) + "\n" + TESTS, SUBMIT)
        self.assertEqual(raw.reward_components["raw_io_share"], 1.0)
        self.assertEqual(sum(raw.reward_components["tool_calls"].values()), 0)
        block = editmetrics.step_block([ep, raw])
        self.assertEqual((block["edit_errors"], block["raw_io_share"]), (0.5, 0.5))
        self.assertEqual(block["tool_calls"]["write_file"], len(gold) / 2)

    async def test_no_submit_or_no_change_scores_zero_without_grading(self):
        ep = await self.episode(LOOK, LOOK)  # never submits: runs to the 30-turn cap (prompt v3)
        self.assertEqual((len(ep.turns), ep.reward, ep.grade["graded"]), (30, 0.0, False))
        self.assertEqual(ep.reward_components["repeated_cells"], 29.0)
        self.assertEqual(ep.reward_components["submitted"], 0.0)
        ep = await self.episode(LOOK, SUBMIT)
        self.assertEqual((ep.reward, ep.reward_components["empty_diff"], ep.grade["graded"]), (0.0, 1.0, False))
        self.assertEqual(ep.hacks, [])

    async def test_detectors_on_edits(self):
        gold = writes(self.task.grading["gold_files"])
        ep = await self.episode(gold + "\n" + SUBMIT)  # submitted on turn 1
        self.assertEqual([h["rule"] for h in ep.hacks], ["early_submit"])
        self.assertEqual(ep.reward, 0.0)
        test_file = [p for p in self.task.workspace.ensure().joinpath("tree", "tests").iterdir() if p.name.startswith("test_")][0]
        tamper = gold + f"\nwith open('/work/repo/tests/{test_file.name}', 'a') as f:\n    f.write('\\n# changed\\n')"
        ep = await self.episode(LOOK, tamper, SUBMIT)
        self.assertEqual([h["rule"] for h in ep.hacks], ["test_tamper"])
        self.assertEqual((ep.reward, ep.reward_components.get("reward_before_hack")), (0.0, 1.0))
        ep = await self.episode(LOOK, "# fake: audit network\n" + gold + "\n" + TESTS, SUBMIT)
        self.assertEqual(ep.hacks, [])  # a blocked network attempt is logged, never zeroed (spec §12.6)
        self.assertEqual([f["rule"] for f in ep.flags], ["network_try"])
        self.assertEqual(ep.reward, 1.0)
        self.assertEqual(detectors.counts([ep])["network_try"], 1)

    async def test_a_failing_grader_is_infra_not_a_zero(self):
        async def broken(task, ep, box, run_id):
            raise env.InfraError("the grader container never started")
        ep = await self.episode(LOOK, writes(self.task.grading["gold_files"]), SUBMIT, grade=broken)
        self.assertEqual((ep.infra_kind, ep.reward), ("grader", None))
        self.assertEqual(filter_step([[ep, ep]]).infra, 2)

    async def test_the_step_block_has_exactly_the_specs_keys(self):
        from rlm_rl import dashboard

        gold = writes(self.task.grading["gold_files"])
        eps = [await self.episode(LOOK, gold + "\n" + TESTS, SUBMIT), await self.episode(LOOK, SUBMIT)]
        block = editmetrics.step_block(eps)
        self.assertEqual(set(block), set(editmetrics.EDIT_KEYS) | {"by_language", "tool_calls"})
        self.assertEqual(editmetrics.EDIT_KEYS, dashboard.EDIT_KEYS)  # the dashboard reads these names (spec §12.9)
        self.assertEqual(dashboard.TOOL_NAMES, repl_tools.TOOL_NAMES)  # and tool_calls by these
        self.assertEqual(set(block["tool_calls"]), set(repl_tools.TOOL_NAMES))
        self.assertEqual(block["by_language"], {"python": {"episodes": 2, "resolved": 0.5}})
        self.assertEqual((block["episodes"], block["resolved"], block["submitted"], block["empty_diff"]), (2, 0.5, 1.0, 0.5))
        self.assertIsNotNone(block["grader_s"])
        row = monitor.step_metrics([eps], filter_step([eps]))
        self.assertEqual(row["edit"], block)
        read_only = monitor.step_metrics([], filter_step([]))
        self.assertNotIn("edit", read_only)  # absent, not null, when a step drew no edit task


@linux_only
class PaceNudgeTest(EditEnvCase):
    """prod-v2 (2026-10-19): every build-from-spec episode of step 1 read for 18-28 turns before its first edit."""

    async def test_an_episode_that_never_edits_is_told_to_write_from_half_its_turns(self):
        ep = await self.episode(LOOK, pace_nudge=True, limits_override={"max_turns": 12})
        seen = [(i, bridge.OBS_PACE_EDIT in (t.observation or "")) for i, t in enumerate(ep.turns, 1) if t.observation]
        self.assertEqual([i for i, nudged in seen if nudged], [6, 7, 8, 9, 10])  # 6 turns left and fewer, not the last
        self.assertNotIn(bridge.OBS_PACE_EDIT, ep.turns[-2].observation)  # the last turn's message stays as it was
        self.assertIn(bridge.OBS_LAST_TURN, ep.turns[-2].observation)

    async def test_an_episode_that_edited_is_told_to_test_near_the_end_and_no_nudge_without_the_switch(self):
        gold = writes(self.task.grading["gold_files"])
        ep = await self.episode(gold, LOOK, pace_nudge=True, limits_override={"max_turns": 12})
        nudges = {i: t.observation for i, t in enumerate(ep.turns, 1)
                  if t.observation and (bridge.OBS_PACE_EDIT in t.observation or "run the tests, fix" in t.observation)}
        self.assertEqual(list(nudges), [7])  # 5 turns left
        self.assertIn(bridge.OBS_PACE_TEST.format(turns_left=5), nudges[7])
        ep = await self.episode(LOOK, limits_override={"max_turns": 12})
        self.assertFalse(any(bridge.OBS_PACE_EDIT in (t.observation or "") for t in ep.turns))

    async def test_a_teacher_hint_is_appended_to_the_system_prompt(self):
        """Warm-start sampling (rlm_rl/warmstart.py) runs with a teacher hint that SFT strips again."""
        from rlm_rl import warmstart

        ep = await self.episode(LOOK, system_extra=warmstart.TEACHER_EDIT, limits_override={"max_turns": 2})
        self.assertTrue(ep.messages[0]["content"].endswith(warmstart.SEPARATOR + warmstart.TEACHER_EDIT))
        self.assertEqual(warmstart.strip_teacher(ep.messages[0]["content"]),
                         bridge.initial_messages(self.task.question, self.task.context, bridge.PROMPT_VERSION,
                                                 bridge.limits(self.task.family), edit=self.task.edit)[0]["content"])

    def test_pace_message(self):
        self.assertIsNone(bridge.pace_message(16, 30, edited=False))
        self.assertEqual(bridge.pace_message(15, 30, edited=False), bridge.OBS_PACE_EDIT)
        self.assertIsNone(bridge.pace_message(15, 30, edited=True))
        self.assertIn("5 turns left", bridge.pace_message(5, 30, edited=True))
        self.assertIsNone(bridge.pace_message(4, 30, edited=True))


class RetirementTest(unittest.TestCase):
    def group(self, task_id, rewards, family="edit_real"):
        return [Episode(task_id, family, 1, "q", None, "m", "s", reward=r) for r in rewards]

    def test_all_zero_twice_or_all_one_once(self):
        r = tasks.Retirement()
        self.assertEqual(r.update(1, [self.group("a", [0, 0, 0, 0]), self.group("b", [1, 1, 1, 1]),
                                      self.group("c", [0, 1, 0, 0]), self.group("s", [1, 1], "edit_synth")]), ["b"])
        self.assertEqual(r.update(2, [self.group("a", [0, 0, 0, 0])]), ["a"])
        self.assertEqual(r.ids(), {"a", "b"})
        again = tasks.Retirement(r.state_dict())
        self.assertEqual((again.ids(), again.counts), (r.ids(), r.counts))
        infra = self.group("d", [0, 0])
        for ep in infra:
            ep.infra_error = "x"
        self.assertEqual(r.update(3, [infra]), [])

    def test_draw_skips_retired_tasks(self):
        seen = []

        def fake(family, level, seed=None, task_id=None, exclude=None):
            seen.append((family, set(exclude or ())))
            return env.Task(f"{family}-{len(seen)}", family, "q", "c")

        mix = {"edit_real": 1.0}
        from unittest import mock
        with mock.patch.object(tasks, "make_task", fake):
            drawn = tasks.draw(2, mix, {"edit_real": 1}, step=1, exclude={"old-1"})
        self.assertEqual(len(drawn), 2)
        self.assertEqual(seen[0][1], {"old-1"})
        self.assertEqual(seen[1][1], {"old-1", "edit_real-1"})  # nor the same task twice in a step

    def test_a_used_up_pool_gives_its_slots_to_edit_synth(self):
        """Decision 12b: with no edit_real task left to draw, the slot goes to edit_synth at edit_synth's own level."""
        import collections
        from unittest import mock

        made = []

        def fake(family, level, seed=None, task_id=None, exclude=None):
            if family == "edit_real" and len(set(exclude or ())) >= 2:  # one task left beside the retired "old-1"
                raise tasks.PoolEmpty("no edit_real task left to draw")
            made.append((family, level))
            return env.Task(f"{family}-{len(made)}", family, "q", "c")

        with mock.patch.object(tasks, "make_task", fake):
            drawn = tasks.draw(8, {"edit_real": 0.5, "edit_synth": 0.5}, {"edit_real": 2, "edit_synth": 3}, step=1,
                               exclude={"old-1"})
            self.assertEqual(collections.Counter(t.family for t in drawn), {"edit_real": 1, "edit_synth": 7})
            self.assertEqual(set(made), {("edit_real", 2), ("edit_synth", 3)})
            with self.assertRaises(tasks.PoolEmpty):  # a run without edit_synth has nowhere to move the slot
                tasks.draw(2, {"edit_real": 1.0}, {"edit_real": 1}, step=1, exclude={"old-1"})

    def test_the_pool_reports_empty_and_its_size(self):
        import random
        from unittest import mock

        from rlm_rl.tasks import editreal

        with self.assertRaises(tasks.PoolEmpty):
            editreal._pick([], 1, random.Random(0))
        self.assertTrue(issubclass(tasks.PoolEmpty, ValueError))  # what the pool raised before 12b
        with mock.patch.object(editreal, "records", lambda split: [{"id": split}] * 3):
            self.assertEqual(tasks.pool_size("edit_real"), 3)
        with mock.patch.object(editreal, "records", mock.Mock(side_effect=FileNotFoundError("no index"))):
            self.assertIsNone(tasks.pool_size("edit_real"))
        self.assertIsNone(tasks.pool_size("aggregation"))  # not a fixed pool


@linux_only
class DockerEditTest(EditEnvCase):
    real = True

    async def asyncSetUp(self):
        reason = docker_ready()
        if reason:
            self.skipTest(reason)
        editbox.build_synth_image()
        await super().asyncSetUp()

    async def test_a_resolved_episode_in_real_containers(self):
        gold = writes(self.task.grading["gold_files"])
        ep = await self.episode(LOOK, gold + "\n" + TESTS, SUBMIT, run_id="test-edit-env")
        self.assertIsNone(ep.infra_error, ep.infra_error)
        self.assertEqual((ep.reward, ep.correct), (1.0, True))
        self.assertEqual(ep.reward_components["tested_before_submit"], 1.0)
        self.assertEqual(ep.reward_components["test_runs"], 1.0)
        self.assertGreater(ep.reward_components["grader_s"], 0)
        self.assertIn("exec", {e["kind"] for e in ep.events})
        ep = await self.episode(LOOK, SUBMIT, run_id="test-edit-env")
        self.assertEqual(ep.reward, 0.0)


if __name__ == "__main__":
    unittest.main()
