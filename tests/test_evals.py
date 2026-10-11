"""H9: every eval episode is logged in full, a rerun leaves one transcript per episode, and the audit reads them.

    python3 -m unittest -v tests.test_evals
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import openai

from rlm_rl import bridge, evals, sandbox
from rlm_rl.client import ModelClient
from rlm_rl.evals import run_eval
from rlm_rl.transcripts import TranscriptLog, find_logs, transcript_path
from tests.fake_model import AsyncCase, FakeModelServer
from tests.fakes import FakeSandbox
from tests.test_env import BASE, FAST, ROOT, coin_policy, docker_ready, exact, make_task

REPO = Path(__file__).resolve().parents[1]
FIRSTS = ["alpha", "FAIL429", "beta", "gamma", "FAIL429", "delta"]


class EvalCase(AsyncCase):
    sandbox_factory = FakeSandbox
    sandbox_kwargs: dict = {}

    async def asyncSetUp(self):
        self.server = FakeModelServer(policy=coin_policy(1))
        await self.server.start()
        self.openai = openai.AsyncOpenAI(base_url=self.server.base_url, api_key="test-key")
        self.client = ModelClient(self.openai, subcall_model=BASE, retry=FAST, rng=random.Random(0))
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "results" / "run-a"
        self.tasks = [make_task(5 + i, f"sel-{i:02d}", first) for i, first in enumerate(FIRSTS)]

    async def asyncTearDown(self):
        await self.openai.close()
        await self.server.close()
        self.tmp.cleanup()

    async def run_selection(self, **kw):
        return await run_eval(self.tasks, name="selection", step=3, run_dir=self.run_dir, run="run-a",
                              client=self.client, root_model=ROOT, scorer=exact, samples=2,
                              sandbox_factory=self.sandbox_factory, **{**self.sandbox_kwargs, **kw})

    def audit(self, *args) -> str:
        out = subprocess.run([sys.executable, str(REPO / "scripts" / "audit.py"), *args], capture_output=True,
                             text=True, encoding="utf-8", cwd=REPO, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout


class EvalLoggingTest(EvalCase):
    async def test_every_episode_is_logged_in_full(self):
        result = await self.run_selection()
        self.assertEqual(result.path, self.run_dir / "transcripts" / "val" / "selection" / "step0003.jsonl")
        lines = result.path.read_text("utf-8").splitlines()
        self.assertEqual((len(lines), result.ran, len(result.records)), (12, 12, 12))
        record = json.loads(lines[0])
        for field in ("messages", "turns", "subcalls", "reward", "correct", "reward_components", "infra_error",
                      "events", "usage", "metrics", "root_model", "subcall_model", "prompt_hash", "gold", "answer"):
            self.assertIn(field, record)
        self.assertEqual((record["split"], record["eval"], record["step"], record["root_model"]),
                         ("val", "selection", 3, ROOT))
        self.assertEqual(record["prompt_hash"], bridge.PROMPT_HASH)
        self.assertEqual(record["messages"][0]["content"], evals.SYSTEM_PROMPT)
        clean = [r for r in result.records if not r["infra_error"]]
        self.assertTrue(all(r["turns"][0]["cell"]["stdout"] and r["subcalls"][0]["output"] == "ok" for r in clean))
        self.assertNotIn("context", record)  # the context is identified by size and hash, not copied
        self.assertEqual(len(record["context_sha256"]), 64)

    async def test_rerun_leaves_one_transcript_per_episode(self):
        """Done when: a rerun eval writes one transcript per episode, and the audit script reads them."""
        self.server.fault = lambda kind, body: (429 if kind == "subcall" and "FAIL429" in body["messages"][-1]["content"]
                                                else None)
        first = await self.run_selection()
        self.assertEqual((first.summary["infra_episodes"], first.summary["n"]), (4, 8))
        self.assertEqual(first.summary["infra_kinds"], {"rate_limited": 4})

        self.server.fault = lambda kind, body: None
        second = await self.run_selection()
        self.assertEqual(second.ran, 4)  # only the infra episodes ran again
        log = TranscriptLog(second.path)
        self.assertEqual((len(log.all_records()), len(log.records())), (16, 12))
        self.assertEqual(second.summary["infra_episodes"], 0)
        self.assertEqual(sorted(r["key"] for r in second.records),
                         sorted(f"selection/step3/{t.task_id}/{s}" for t in self.tasks for s in range(2)))

        third = await self.run_selection()
        self.assertEqual((third.ran, len(log.records())), (0, 12))

        with open(second.path, "a", encoding="utf-8") as f:  # a crash mid-write leaves a torn line
            f.write('{"key": "selection/step3/sel-00/0", "trunc')
        self.assertEqual(len(log.records()), 12)
        self.assertEqual(log.torn_lines, 1)

        summary = self.audit("summary", str(self.run_dir))
        self.assertIn("selection", summary)
        self.assertIn("4 earlier attempt(s) replaced by a rerun", summary)
        self.assertIn("1 torn line(s) skipped", summary)
        sample = self.audit("sample", str(self.run_dir), "--split", "val", "-n", "3")
        self.assertIn("# Audit sample: 3 of", sample)
        self.assertIn("#### Turn 1", sample)
        episode_id = second.records[0]["episode_id"]
        self.assertIn(episode_id, self.audit("show", str(second.path), episode_id))

    async def test_logged_through_art(self):
        try:
            import art
        except ImportError:
            self.skipTest("ART is not installed in this environment")
        import pyarrow.parquet as pq

        self.server.fault = lambda kind, body: (429 if kind == "subcall" and "FAIL429" in body["messages"][-1]["content"]
                                                else None)
        model = art.Model(name="eval-test", project="step4-test", base_path=str(Path(self.tmp.name) / ".art"),
                          report_metrics=[])  # local only: nothing goes to W&B
        result = await self.run_selection(art_model=model)
        out = Path(self.tmp.name) / ".art" / "step4-test" / "models" / "eval-test"
        table = pq.read_table(out / "trajectories" / "val" / "0003.parquet")
        self.assertGreaterEqual(table.num_rows, result.summary["n"])
        history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
        self.assertTrue(any(k.endswith("selection/accuracy") for row in history for k in row))
        self.assertTrue(any(row.get("val/exception_rate", 0) > 0 for row in history))  # infra episodes counted

    def test_paths(self):
        self.assertEqual(transcript_path("r", "train", 7), Path("r/transcripts/train/step0007.jsonl"))
        for bad in ("", "a/b", "..", " x"):
            with self.assertRaises(ValueError):
                transcript_path("r", "val", 1, bad)
        self.assertEqual(find_logs("does-not-exist"), [])

    def test_transcripts_are_git_ignored(self):
        check = subprocess.run(["git", "check-ignore", "-q", "results/run-a/transcripts/val/x/step0001.jsonl"],
                               cwd=REPO)
        self.assertEqual(check.returncode, 0)


class DockerEvalTest(EvalCase):
    """The rerun check again, through real sandbox containers."""

    sandbox_factory = sandbox.Sandbox

    @classmethod
    def setUpClass(cls):
        reason = docker_ready()
        if reason:
            raise unittest.SkipTest(reason)
        cls.sandbox_kwargs = {"image": sandbox.build_image()}

    async def test_rerun_through_real_sandboxes(self):
        self.server.fault = lambda kind, body: (429 if kind == "subcall" and "FAIL429" in body["messages"][-1]["content"]
                                                else None)
        first = await self.run_selection()
        self.assertEqual(first.summary["infra_episodes"], 4)
        self.server.fault = lambda kind, body: None
        second = await self.run_selection()
        self.assertEqual((second.ran, len(second.records), second.summary["infra_episodes"]), (4, 12, 0))
        self.assertIn("selection", self.audit("summary", str(self.run_dir)))


if __name__ == "__main__":
    unittest.main()
