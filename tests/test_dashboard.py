"""The live dashboard (plan step 9): the page's data, the secrets check and the publish loop, with no W&B call.

A short production dry run (ticket stand-in, fake model and backend) gives a real results file, real
training transcripts and real eval transcripts to build against.

    python3 -m unittest -v tests.test_dashboard
"""

import argparse
import asyncio
import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from rlm_rl import config, dashboard
from tests.dryrun import DryRun, Oracle

REPO = Path(__file__).resolve().parents[1]
RUN = "dash-test"


def _cli():
    spec = importlib.util.spec_from_file_location("rlm_dashboard_cli", REPO / "scripts" / "dashboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DryRunCase(unittest.TestCase):
    """One 12-step production dry run, shared by every test in the class."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.results = Path(cls.tmp.name) / "results"
        cls.out = Path(cls.tmp.name) / "dashboard"

        async def go():
            async with DryRun(cls.results, seed=3, oracle=Oracle(3)) as dry:
                cfg = config.production(RUN, 2e-5, tasks_from="tickets")
                with contextlib.redirect_stdout(io.StringIO()):
                    await dry.training_run(RUN, cfg=cfg, stop_after=12).start()

        asyncio.run(go())
        cls.cli = _cli()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def state(self) -> dict:
        return dashboard.read_state(self.results / f"{RUN}.json")

    def build(self, **kw) -> dict:
        return self.cli.build(RUN, self.results, dry=True, out_dir=self.out, **kw)


class PageTest(DryRunCase):
    def test_every_step_and_panel_has_its_data(self):
        state = self.state()
        data = dashboard.page_data(state, run_dir=self.results / RUN, dry_run=True)
        self.assertEqual([r["step"] for r in data["steps"]], [r["step"] for r in state["steps"]])
        row = data["steps"][-1]
        for section, keys in (("pass", ("rate", "trainable_groups", "all_zero_groups")),
                              ("policy", ("entropy", "kl_to_base", "reward_std", "grad_norm", "probs_corr", "clip_fraction")),
                              ("shape", ("subcalls", "turns", "transcript_tokens_max", "context_chars")),
                              ("reward", ("exact", "mean", "hacks")), ("infra", ("rate", "peak_in_flight")),
                              ("cost", ("step_usd",)), ("time", ("total_s",))):
            for key in keys:
                self.assertIn(key, row[section], f"{section}.{key}")
        self.assertTrue(row["families"])
        self.assertEqual({e["checkpoint"] for e in data["evals"]}, {0, 10})
        self.assertEqual(set(data["stop_rules"]), {"plateau", "overfitting", "frying", "saturation", "hacking"})
        self.assertTrue(data["notices"])
        self.assertEqual(data["run"]["status"], "stopped")
        self.assertTrue(data["dry_run"])

    def test_build_round_trips_and_passes_the_check(self):
        r = self.build()
        self.assertTrue(r["ok"], r["findings"])
        page = (self.out / "page.html").read_text(encoding="utf-8")
        data = dashboard.embedded_data(page)
        self.assertEqual(data["steps"][-1]["step"], 12)
        self.assertIn("<title>RLM Live Run</title>", page[:8192])
        self.assertLess(r["bytes"], 16 * 1024 * 1024)
        self.assertGreater(r["checked"]["seeds"], 0)          # the dry run's eval seeds were read
        self.assertGreater(r["checked"]["eval_tasks"], 0)  # and its eval transcripts

    def test_nothing_identifying_reaches_the_page(self):
        self.build()
        page = (self.out / "page.html").read_text(encoding="utf-8")
        for needle in (f"{RUN}:step", "wandb-artifact", str(self.results), self.tmp.name, "prompt_hash",
                       "context_sha256", "sample.md", "root_model"):
            self.assertNotIn(needle, page)
        seeds = dashboard.private_seeds([self.results / "private" / "seeds"])
        self.assertTrue(seeds)
        self.assertFalse({str(s) for s in seeds} & set(page.replace(",", " ").replace(":", " ").split()))

    def test_samples_are_training_rollouts_cut_short(self):
        cache: dict = {}
        samples = dashboard.update_samples(cache, self.state(), self.results / RUN)
        self.assertTrue(samples)
        steps = {r["step"] for r in self.state()["steps"]}
        items, _ = dashboard.eval_items(self.results / RUN)
        for s in samples:
            self.assertIn(s["step"], steps)
            self.assertIn(s["why"], ("win", "fail", "flagged", "infra"))
            self.assertLessEqual(len(s["question"]), dashboard.CUT["question"] + 40)
            self.assertLessEqual(len(s["turns"]), dashboard.CUT["turns_head"] + dashboard.CUT["turns_tail"])
            self.assertNotIn(dashboard.item_key(s["question"], s["context_head"]), items)
        per_step = {}
        for s in samples:
            per_step[s["step"]] = per_step.get(s["step"], 0) + 1
        self.assertLessEqual(max(per_step.values()), sum(dashboard.SAMPLES_PER_STEP.values()))
        # sampling again changes nothing: each step is read once
        again = dashboard.update_samples(cache, self.state(), self.results / RUN)
        self.assertEqual([s["key"] for s in again], [s["key"] for s in samples])

    def test_rolled_back_steps_drop_their_samples(self):
        cache: dict = {}
        state = self.state()
        dashboard.update_samples(cache, state, self.results / RUN)
        state["steps"] = [r for r in state["steps"] if r["step"] <= 5]
        kept = dashboard.update_samples(cache, state, self.results / RUN)
        self.assertTrue(kept)
        self.assertTrue(all(s["step"] <= 5 for s in kept))


class CheckTest(DryRunCase):
    def page_with(self, mutate) -> str:
        state = self.state()
        mutate(state)
        return dashboard.render(dashboard.page_data(state, run_dir=self.results / RUN, dry_run=True))

    def test_a_key_in_free_text_is_scrubbed_and_a_planted_one_is_caught(self):
        fake_key = "0123456789abcdef" * 2 + "01234567"  # 40 hex characters, like a W&B key
        page = self.page_with(lambda s: s["notices"].append(
            {"at": time.time(), "step": 1, "kind": "logging",
             "text": f"W&B failed: key {fake_key} at https://api.wandb.ai/x and F:\\Personal\\RLM\\RLM\\.env"}))
        self.assertNotIn(fake_key, page)
        self.assertNotIn("api.wandb.ai", page)
        self.assertNotIn("Personal\\RLM", page)
        self.assertTrue(dashboard.check(page)["ok"])
        planted = page.replace("RLM Live Run</title>", f"RLM Live Run</title><!-- {fake_key} -->")
        r = dashboard.check(planted)
        self.assertFalse(r["ok"])
        self.assertNotIn(fake_key, json.dumps(r))  # findings are counts, never the match

    def test_a_private_seed_is_caught(self):
        seeds = dashboard.private_seeds([self.results / "private" / "seeds"])
        seed = sorted(seeds)[0]
        page = self.page_with(lambda s: s["steps"][-1]["pass"].update(groups=seed))
        r = dashboard.check(page, seeds=seeds)
        self.assertFalse(r["ok"])
        self.assertIn("private seed", r["findings"])
        self.assertNotIn(str(seed), json.dumps(r))

    def test_an_eval_task_is_caught_but_a_shared_question_template_is_not(self):
        items, _ = dashboard.eval_items(self.results / RUN)
        question, head = sorted(items)[0]
        # ticket questions are templated: the same question over a different log is a different task
        page = self.page_with(lambda s: s["notices"].append({"at": 1.0, "step": 1, "kind": "x", "text": question}))
        self.assertTrue(dashboard.check(page, items=items)["ok"])
        page = self.page_with(lambda s: s["notices"].extend([{"at": 1.0, "step": 1, "kind": "x", "text": question},
                                                             {"at": 1.0, "step": 1, "kind": "x", "text": head}]))
        r = dashboard.check(page, items=items)
        self.assertFalse(r["ok"])
        self.assertIn("eval task (its question and context)", r["findings"])

    def test_text_cannot_break_out_of_the_data(self):
        evil = "</script><script>alert(1)</script><!--"
        page = self.page_with(lambda s: s["notices"].append({"at": 1.0, "step": 1, "kind": "x", "text": evil}))
        self.assertNotIn("</script><script>alert", page)
        self.assertIn(evil, [n["text"] for n in dashboard.embedded_data(page)["notices"]])

    def test_a_failed_check_writes_nothing(self):
        self.build()
        before = (self.out / "page.html").read_text(encoding="utf-8")
        real = dashboard.check
        try:
            dashboard.check = lambda page, **kw: {"ok": False, "findings": {"planted": 1}, "bytes": len(page),
                                                  "checked": {}}
            r = self.build()
        finally:
            dashboard.check = real
        self.assertFalse(r["ok"])
        self.assertEqual((self.out / "page.html").read_text(encoding="utf-8"), before)


class SignatureTest(unittest.TestCase):
    def test_a_heartbeat_alone_is_not_news(self):
        state = {"run": "r", "status": "running", "steps": [{"step": 3}], "evals": [], "notices": [], "audits": {},
                 "live": {"step": 4, "phase": "rollouts", "at": 1.0}, "updated_at": 1.0}
        sig = dashboard.signature(state)
        beat = dict(state, live={"step": 4, "phase": "training", "at": 9.0}, updated_at=9.0)
        self.assertEqual(dashboard.signature(beat), sig)
        self.assertNotEqual(dashboard.signature(dict(state, steps=[{"step": 3}, {"step": 4}])), sig)
        self.assertNotEqual(dashboard.signature(dict(state, notices=[{}])), sig)
        self.assertNotEqual(dashboard.signature(dict(state, status="paused")), sig)
        self.assertNotEqual(dashboard.signature(dict(state, live={"step": 4, "phase": "eval", "at": 2.0})), sig)

    def test_scrub(self):
        text = "see wandb-artifact:///ent/proj/run:step4 in /mnt/f/Personal/x and C:\\Users\\me\\a.txt " + "z" * 50
        out = dashboard.scrub(text)
        for bad in ("ent/proj", "/mnt/f", "C:\\Users", "z" * 50):
            self.assertNotIn(bad, out)
        self.assertTrue(dashboard.scrub("x" * 30, 10).startswith("x" * 10))


class WaitTest(DryRunCase):
    def args(self, **kw):
        base = dict(run=RUN, results=str(self.results), dry_run=True, pot_left=None, out_dir=str(self.out), ack=False,
                    poll=0.05, settle=0.0, min_interval=0.0, stale_after=dashboard.STALE_AFTER_S, timeout=2.0)
        base.update(kw)
        return argparse.Namespace(**base)

    def wait(self, **kw) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = self.cli.cmd_wait(self.args(**kw))
        return code, buf.getvalue()

    def setUp(self):
        shutil.rmtree(self.out, ignore_errors=True)
        self.state_file = self.results / f"{RUN}.json"
        self.original = self.state_file.read_text(encoding="utf-8")

    def tearDown(self):
        self.state_file.write_text(self.original, encoding="utf-8")

    def test_publish_ack_then_wait_for_the_next_change(self):
        code, out = self.wait()
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("PUBLISH"), out)
        self.assertTrue((self.out / "page.html").exists())
        code, out = self.wait(ack=True, timeout=0.6)  # nothing new since the publish
        self.assertEqual(code, 4, out)
        self.assertTrue(out.startswith("TIMEOUT"))

        def change():
            time.sleep(0.3)
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            state["notices"].append({"at": time.time(), "step": 12, "kind": "operator", "text": "a test notice"})
            self.state_file.write_text(json.dumps(state), encoding="utf-8")

        threading.Thread(target=change).start()
        code, out = self.wait(ack=True, timeout=5)
        self.assertEqual(code, 0, out)
        self.assertTrue(out.startswith("PUBLISH"), out)
        self.assertIn("notices=", out)
        page = dashboard.embedded_data((self.out / "page.html").read_text(encoding="utf-8"))
        self.assertEqual(page["notices"][-1]["text"], "a test notice")

    def test_a_silent_run_is_reported_once(self):
        state = json.loads(self.original)
        state.update(status="running", updated_at=time.time() - 3600, stop=None)
        self.state_file.write_text(json.dumps(state), encoding="utf-8")
        code, out = self.wait()
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("STALE"), out)  # built and reported, so the page can say so
        self.assertIn("heartbeat_age=", out)
        code, out = self.wait(ack=True, timeout=0.6)  # reported once, until the run writes again
        self.assertTrue(out.startswith("TIMEOUT"), out)


if __name__ == "__main__":
    unittest.main()
