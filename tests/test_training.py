"""The training loop end to end against fakes: plan step 6's done-when, with no W&B call and no credit.

A dry run (tests/dryrun.py) goes through steps, evals, the stop rules, checkpoint saves and every
kind of resume: after the stop switch, after a crash during training, from an earlier checkpoint,
and after the automatic learning-rate fallback.

    python3 -m unittest -v tests.test_training
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from rlm_rl import bridge, config, data, monitor, sandbox, training
from rlm_rl.training import ResumeBlocked, load_state, request_stop
from rlm_rl.transcripts import TranscriptLog, transcript_path
from tests.dryrun import DryRun, Oracle
from tests.fake_model import AsyncCase
from tests.fakes import FakeSandbox
from tests.test_env import FAST, docker_ready

REPO = Path(__file__).resolve().parents[1]
TRAIN_CALL = {"loss_fn": "cispo", "epsilon": 1.0, "epsilon_high": 4.0, "scale_rewards": False, "kl_penalty_coef": 0.0,
              "packed_sequence_length": None}


def second_seed(run, steps=20):
    cfg = config.second_seed(run)
    cfg.max_steps = steps
    return cfg


class Boom(Exception):
    pass



class LoopCase(AsyncCase):

    def oracle(self) -> Oracle:
        return Oracle(1)

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name) / "results"
        self.dry = DryRun(self.results, seed=1, oracle=self.oracle())
        await self.dry.__aenter__()

    async def asyncTearDown(self):
        await self.dry.__aexit__(None, None, None)
        self.tmp.cleanup()

    def run_(self, name, **kw):
        kw.setdefault("retry", FAST)
        kw.setdefault("train_retry", (2, 0.3, 0.05))  # a failed train call: 2 more tries, after 0.3 s of waiting
        kw.setdefault("log", lambda line: None)
        return self.dry.training_run(name, **kw)

    def hook_draw(self, run, at: int, action):
        """Call `action()` when the loop draws its `at`-th batch of tasks (that is, during step `at`)."""
        original, calls = run.draw, []

        def draw(*args, **kwargs):  # kwargs: the retired tasks to exclude, once there are any
            calls.append(1)
            if len(calls) == at:
                action()
            return original(*args, **kwargs)

        run.draw = draw
        return run

    def assert_contiguous(self, state, last):
        self.assertEqual([r["step"] for r in state["steps"]], list(range(1, last + 1)))


class SecondSeedDryRunTest(LoopCase):
    async def test_twenty_steps_then_the_held_out_eval(self):
        st = await self.run_("ss", cfg=second_seed("ss")).start()
        self.assertEqual((st["status"], st["stop"]["rule"]), ("finished", "max_steps"))
        self.assert_contiguous(st, 20)
        run_dir = self.results / "ss"

        # every train call carries spec §5.1's settings at 1e-5
        trained = self.dry.backend.trained
        self.assertEqual(len(trained), sum(r["status"] == "trained" for r in st["steps"]))
        for call in trained:
            self.assertEqual(call["kwargs"], {"learning_rate": 1e-5, **TRAIN_CALL})

        # one eval: the first seed's 50 held-out tasks x 2 samples, at checkpoint 20
        self.assertEqual([(e["checkpoint"], e["name"], e["summary"]["episodes"]) for e in st["evals"]],
                         [(20, "heldout-seed1", 100)])
        self.assertEqual(st["best"]["checkpoint"], 20)

        # the controller is frozen at level 1
        self.assertTrue(all(f["level"] == 1 for f in st["controller"]["families"].values()))
        self.assertFalse(any(m["applied"] for m in st["controller"]["history"]))

        # every 5th checkpoint is kept and the rest are pruned from the backend; each kept one has a snapshot
        self.assertEqual(self.dry.backend.models["ss"]["steps"], [5, 10, 15, 20])
        for c in (0, 5, 10, 15, 20):
            self.assertTrue((run_dir / "checkpoints" / f"step{c:04d}.json").exists(), c)
        pinned = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(pinned["bridge"]["prompt_version"], "v1")  # the second seed replicates the 20-step run (E-g)
        self.assertEqual(pinned["bridge"]["prompt_hash"], bridge.PROMPT_HASHES["v1"])
        self.assertEqual(pinned["train_call"], {"learning_rate": 1e-5, **TRAIN_CALL})
        for key in ("art_version", "base_model", "chat_template_sha256", "template_fingerprint", "limits", "reward",
                    "detectors", "controller", "generators", "manifest", "limiter", "git", "spend"):
            self.assertIn(key, pinned)
        self.assertIsNotNone(pinned["template_fingerprint"])

        # the Claude audit's samples at checkpoints 10 and 20, the second with the held-out eval's wins
        self.assertTrue((run_dir / "audit" / "step0010" / "sample.md").exists())
        self.assertEqual(st["audits"]["20"]["eval"], "heldout-seed1")
        verdicts = json.loads((run_dir / "audit" / "step0020" / "verdicts.json").read_text(encoding="utf-8"))
        self.assertTrue(any(e["split"] == "val" for e in verdicts["episodes"]))

        # the second seed's curves decide production's learning rate (spec §5.2)
        self.assertIn(st["lr_decision"]["production_lr"], (2e-5, 1e-5))
        self.assertTrue(st["lr_decision"]["passed"], st["lr_decision"])  # the oracle's curves are healthy

        # per-step metrics reach the results file and the W&B log
        row = st["steps"][-1]
        for section in ("pass", "families", "reward", "policy", "shape", "infra", "cost", "time", "filter"):
            self.assertIn(section, row)
        self.assertIsNotNone(row["policy"]["entropy"])
        self.assertIsNotNone(row["policy"]["kl_to_base"])
        self.assertEqual(row["policy"]["grad_norm"], 0.7)
        logs = [entry for entry in self.dry.backend.logged if entry["split"] == "train"]
        self.assertEqual([entry["step"] for entry in logs], list(range(1, 21)))
        self.assertIn("rlm/policy/entropy", logs[-1]["metrics"])
        self.assertIn("loss/grad_norm", logs[-1]["metrics"])

        # spend: steps under train, the eval under eval
        spend = st["spend"]
        self.assertAlmostEqual(spend["train_usd"], sum(r["cost"]["step_usd"] for r in st["steps"]), places=4)
        self.assertAlmostEqual(spend["eval_usd"], st["evals"][0]["cost_usd"], places=4)
        self.assertGreater(spend["kl_usd"], 0)
        self.assertEqual(len(TranscriptLog(transcript_path(run_dir, "train", 7)).records()), 32)


class ProductionStopRuleTest(LoopCase):
    def oracle(self):
        # Gets worse fast (60% at checkpoint 0, the 5% floor by 20), so noise can't make a later eval a new best
        return Oracle(1, base=0.6, slope=-0.05, level_penalty=0.0)

    async def test_plateau_stops_the_run_and_keeps_the_best(self):
        st = await self.run_("plat", cfg=config.production("plat", 2e-5, tasks_from="tickets")).start()
        self.assertEqual((st["status"], st["stop"]["rule"]), ("stopped", "plateau"), st["stop"])
        self.assertEqual(st["latest"], 30)
        self.assertEqual(st["best"]["checkpoint"], 0)
        self.assertEqual(sorted({(e["checkpoint"], e["name"]) for e in st["evals"]}),
                         [(c, n) for c in (0, 10, 20, 30) for n in ("capability", "selection-tickets-v1")])
        self.assertEqual(st["stop_rules"]["plateau"]["state"], "fired")
        self.assertEqual(st["stop_rules"]["frying"]["state"], "ok")
        self.assertIn(30, st["kept"])


@unittest.skipUnless(data.available() and data.derived("lca", "index-train.jsonl.gz").exists()
                     and data.path("edit", "index.json").exists() and sys.platform == "linux",
                     "needs the step 7 data and the edit task index (scripts/build_edit_data.py), on Linux")
class ProductionMixTest(LoopCase):
    """Production on its task families (spec v2 §12.7), its selection set (§12.10) and its manifest. Edit episodes run in
    the in-process edit sandbox with the local grader (tests/fakes.py)."""

    async def test_production_runs_on_its_families(self):
        cfg = config.production("mix7a", 2e-5)
        st = await self.run_("mix7a", cfg=cfg, stop_after=2).start()
        self.assertEqual((st["status"], st["latest"]), ("stopped", 2))
        drawn = {family for row in st["steps"] for family in row["families"] if row["families"][family]["episodes"]}
        self.assertTrue(drawn and drawn <= set(cfg.mix), drawn)
        selection = [e for e in st["evals"] if e["name"] == config.PRODUCTION_SELECTION]
        self.assertEqual([e["checkpoint"] for e in selection], [0])
        from rlm_rl.tasks import editreal  # 48 read + 6 edit_synth + the index's selection split (10 when full)
        self.assertEqual(selection[0]["summary"]["n"], 48 + 6 + len(editreal.records("select")))
        pinned = json.loads((self.results / "mix7a" / "config.json").read_text())
        self.assertEqual(pinned["manifest"]["mix"], cfg.mix)
        self.assertIsNotNone(pinned["manifest"]["dataset_manifest_sha256"])
        self.assertEqual(set(pinned["generators"]), set(cfg.mix))

    async def test_a_used_up_edit_real_pool_draws_edit_synth(self):
        """Decision 12b: once every edit_real task is retired, its slots go to edit_synth and the run goes on."""
        from rlm_rl import tasks as rtasks
        from rlm_rl.tasks import editreal

        cfg = config.production("pool12b", 2e-5)
        cfg.eval_at_start = False
        run = self.run_("pool12b", cfg=cfg, stop_after=2)
        everything = {r["id"]: {"step": 1, "why": "all-1", "family": "edit_real"} for r in editreal.records("train")}
        self.hook_draw(run, 1, lambda: setattr(run, "retirement", rtasks.Retirement({"retired": everything})))
        st = await run.start()
        self.assertEqual((st["status"], st["latest"]), ("stopped", 2))
        first, second = st["steps"]
        planned = rtasks.allocate_step(cfg.tasks_per_step, cfg.mix, 2)["edit_real"]
        self.assertEqual(first["fallback"], {})
        self.assertEqual(second["fallback"], {"edit_real": {"to": "edit_synth", "slots": planned}})
        self.assertEqual(second["families"].get("edit_real", {}).get("episodes", 0), 0)
        self.assertEqual(second["retired"]["left"], 0)
        self.assertEqual(sum(n["kind"] == "fallback" for n in st["notices"]), 1)


class LeftoversTest(LoopCase):
    """Plan step 12: a run clears what a crashed process left behind, and a resume won't race a process still running."""

    async def test_start_and_resume_clear_leftover_containers(self):
        calls = []

        def clear(run):
            calls.append(run)
            return {"run": 2, "finished": 1} if len(calls) == 1 else {"run": 0, "finished": 0}

        st = await self.run_("lc", cfg=second_seed("lc", 3), stop_after=1, clear_containers=clear).start()
        self.assertEqual(calls, ["lc"])
        texts = [n["text"] for n in st["notices"] if n["kind"] == "containers"]
        self.assertEqual(len(texts), 1)
        self.assertIn("removed 2 leftover container(s) of this run and 1 long-finished", texts[0])
        st = await self.run_("lc", stop_after=2, clear_containers=clear).resume()
        self.assertEqual((calls, st["status"], st["latest"]), (["lc", "lc"], "stopped", 2))
        self.assertEqual(sum(n["kind"] == "containers" for n in st["notices"]), 1)  # nothing to say when nothing was left

    async def test_a_failed_clear_is_noted_and_the_run_goes_on(self):
        def clear(run):
            raise FileNotFoundError("docker")

        st = await self.run_("lc2", cfg=second_seed("lc2", 2), stop_after=1, clear_containers=clear).start()
        self.assertEqual((st["status"], st["latest"]), ("stopped", 1))
        self.assertTrue(any("couldn't clear leftover containers" in n["text"] for n in st["notices"]))

    async def test_resume_refuses_while_another_process_may_be_running_the_run(self):
        await self.run_("busy", cfg=second_seed("busy", 3), stop_after=1).start()
        path = training.state_path(self.results, "busy")
        state = json.loads(path.read_text(encoding="utf-8"))
        state.update(status="running", updated_at=time.time() - 10)  # as a live process leaves it between saves
        path.write_text(json.dumps(state), encoding="utf-8")
        calls = []
        with self.assertRaises(ResumeBlocked):
            await self.run_("busy", clear_containers=lambda run: calls.append(run) or {}).resume()
        self.assertEqual(calls, [])  # nothing of the other process's was removed
        state["updated_at"] = time.time() - 2 * training.LIVE_GUARD_S  # a process killed outright: its state goes quiet
        path.write_text(json.dumps(state), encoding="utf-8")
        st = await self.run_("busy", stop_after=2).resume()
        self.assertEqual(st["status"], "stopped")
        self.assert_contiguous(st, 2)


class FallbackSlotsTest(unittest.TestCase):
    def test_slots_short_of_the_plan_are_counted(self):
        from rlm_rl import tasks as rtasks
        from rlm_rl.env import Task

        cfg = config.RunConfig(run="fb", kind="production", learning_rate=2e-5, lr_rule="fallback",
                               mix={"aggregation": 0.6, "edit_real": 0.25, "edit_synth": 0.15})
        run = training.TrainingRun("fb", backend=None, cfg=cfg)
        for step in (1, 2, 3):
            planned = rtasks.allocate_step(cfg.tasks_per_step, cfg.mix, step)
            full = [Task(f"{f}-{i}", f, "q", "c") for f, k in planned.items() for i in range(k)]
            self.assertEqual(run._fallback_slots(step, full), {})
            short = [t for t in full if t.family != "edit_real"] + [Task("s", "edit_synth", "q", "c")] * planned["edit_real"]
            expected = {"edit_real": {"to": "edit_synth", "slots": planned["edit_real"]}} if planned["edit_real"] else {}
            self.assertEqual(run._fallback_slots(step, short), expected)
        tickets = training.TrainingRun("t", backend=None, cfg=second_seed("t"))
        self.assertEqual(tickets._fallback_slots(1, []), {})  # no fixed pool in the mix
        self.assertIsNone(tickets._pool_left())


class StopSwitchTest(LoopCase):
    async def test_stop_switch_then_resume_continues_cleanly(self):
        run = self.run_("sw", cfg=config.production("sw", 2e-5, tasks_from="tickets"))
        self.hook_draw(run, 7, lambda: request_stop(self.results, "sw", "checking something"))
        st = await run.start()
        self.assertEqual((st["status"], st["stop"]["rule"], st["latest"]), ("stopped", "operator", 7))
        self.assertIn("checking something", st["stop"]["reason"])
        self.assertFalse((self.results / "sw" / training.STOP_FILE).exists())
        spent = st["spend"]["train_usd"]

        st = await self.run_("sw", stop_after=12).resume()
        self.assertEqual((st["status"], st["stop"]["rule"]), ("stopped", "operator"))
        self.assert_contiguous(st, 12)
        self.assertGreater(st["spend"]["train_usd"], spent)
        self.assertEqual(sum(e["checkpoint"] == 10 for e in st["evals"]), 2)  # selection and capability, once each
        self.assertEqual(len(TranscriptLog(transcript_path(self.results / "sw", "train", 7)).records()), 32)
        self.assertEqual({r["branch"] for r in st["steps"]}, {0})
        self.assertIn("resume", [n["kind"] for n in st["notices"]])


class CrashTest(LoopCase):
    async def test_a_failed_train_call_waits_for_the_operator_then_retrains_the_saved_batch(self):
        run = self.run_("cr", cfg=second_seed("cr", 4), train_retry=(0, 0, 0))  # no in-process retry: the operator's path
        self.hook_draw(run, 3, lambda: setattr(self.dry.backend, "fail_next_train", RuntimeError("Training job failed")))
        with self.assertRaises(RuntimeError):
            await run.start()
        st = load_state(self.results, "cr")
        self.assertEqual((st["status"], st["pending"]["step"], st["latest"]), ("crashed", 3, 2))
        self.assertTrue((self.results / "cr" / st["pending"]["batch"]).exists())

        with self.assertRaises(ResumeBlocked):  # the job might still be running: the operator decides
            await self.run_("cr").resume()
        st = await self.run_("cr").resume(retrain_pending=True)
        self.assertEqual(st["status"], "finished")
        self.assert_contiguous(st, 4)
        self.assertEqual(st["steps"][2]["trained_to"], {"checkpoint": 3, "backend_step": 3})
        self.assertIsNotNone(st["steps"][2]["train_metrics"])
        self.assertFalse((self.results / "cr" / "pending").exists())

    async def test_training_that_finished_while_the_script_was_down_is_accepted(self):
        run = self.run_("cr2", cfg=second_seed("cr2", 4), train_retry=(0, 0, 0))
        self.hook_draw(run, 2, lambda: setattr(self.dry.backend, "fail_after_next_train", ConnectionError("lost")))
        with self.assertRaises(ConnectionError):
            await run.start()
        st = await self.run_("cr2").resume()
        self.assert_contiguous(st, 4)
        self.assertEqual(st["steps"][1]["trained_to"]["backend_step"], 2)
        self.assertIsNone(st["steps"][1]["train_metrics"])
        self.assertEqual([t["step"] for t in self.dry.backend.trained], [1, 2, 3, 4])  # step 2 was never trained twice

    async def test_a_discarded_step_is_redone_from_new_rollouts(self):
        run = self.run_("cr3", cfg=second_seed("cr3", 3), train_retry=(0, 0, 0))
        self.hook_draw(run, 2, lambda: setattr(self.dry.backend, "fail_next_train", RuntimeError("boom")))
        with self.assertRaises(RuntimeError):
            await run.start()
        spent = load_state(self.results, "cr3")["spend"]["train_usd"]
        st = await self.run_("cr3").resume(discard_pending=True)
        self.assert_contiguous(st, 3)
        self.assertGreater(st["spend"]["train_usd"], spent + st["steps"][1]["cost"]["step_usd"] * 0.9)  # both attempts

    async def test_an_interrupted_step_keeps_its_spend(self):
        original = monitor.kl_to_base
        calls = []

        async def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise Boom("power cut")
            return await original(*args, **kwargs)

        with mock.patch.object(monitor, "kl_to_base", flaky):
            with self.assertRaises(Boom):
                await self.run_("int", cfg=second_seed("int", 4)).start()
            st = load_state(self.results, "int")
            self.assertEqual((st["status"], st["live"]["step"], st["live"]["phase"]), ("crashed", 3, "kl"))
            lost = st["live"]["uncommitted_usd"]
            self.assertGreater(lost, 0)
            before = st["spend"]["train_usd"]
            st = await self.run_("int").resume()
        self.assert_contiguous(st, 4)
        self.assertAlmostEqual(st["spend"]["interrupted_usd"], lost, places=6)
        self.assertGreater(st["spend"]["train_usd"], before + lost)
        moved = list((self.results / "int" / "abandoned" / "interrupted").glob("step0003-*.jsonl"))
        self.assertEqual(len(moved), 1)  # the first attempt's transcripts, kept apart
        self.assertEqual(len(TranscriptLog(transcript_path(self.results / "int", "train", 3)).records()), 32)


class TrainRetryTest(LoopCase):
    """A W&B hiccup on the train call doesn't stop the run (2026-10-18)."""

    async def test_a_train_call_that_failed_before_training_is_sent_again(self):
        run = self.run_("tr1", cfg=second_seed("tr1", 3), stop_after=3)
        self.hook_draw(run, 2, lambda: setattr(self.dry.backend, "fail_next_train", RuntimeError("Training job failed")))
        st = await run.start()
        self.assertEqual((st["status"], st["latest"]), ("stopped", 3))
        self.assertEqual([t["step"] for t in self.dry.backend.trained], [1, 2, 3])  # step 2 trained once, on the retry
        texts = [n["text"] for n in st["notices"] if n["kind"] == "training"]
        self.assertTrue(any("train call failed" in t for t in texts) and any("sent again" in t for t in texts), texts)
        self.assertIsNotNone(st["steps"][1]["train_metrics"])

    async def test_a_train_call_that_trained_anyway_is_accepted_not_repeated(self):
        run = self.run_("tr2", cfg=second_seed("tr2", 3), stop_after=3)
        self.hook_draw(run, 2, lambda: setattr(self.dry.backend, "fail_after_next_train", ConnectionError("lost")))
        st = await run.start()
        self.assertEqual((st["status"], st["latest"]), ("stopped", 3))
        self.assertEqual([t["step"] for t in self.dry.backend.trained], [1, 2, 3])  # never trained twice
        self.assertIsNone(st["steps"][1]["train_metrics"])
        self.assertTrue(any("trained anyway" in n["text"] for n in st["notices"]))

    async def test_a_train_call_that_keeps_failing_still_stops_for_the_operator(self):
        run = self.run_("tr3", cfg=second_seed("tr3", 3), train_retry=(1, 0.1, 0.05))

        class Always:
            def __get__(self, obj, objtype=None):
                return RuntimeError("down")

            def __set__(self, obj, value):
                pass

        self.hook_draw(run, 2, lambda: setattr(type(self.dry.backend), "fail_next_train", Always()))
        try:
            with self.assertRaises(RuntimeError):
                await run.start()
        finally:
            del type(self.dry.backend).fail_next_train
        st = load_state(self.results, "tr3")
        self.assertEqual((st["status"], st["pending"]["step"]), ("crashed", 2))


class RollbackTest(LoopCase):
    async def test_resume_from_an_earlier_checkpoint_starts_a_branch(self):
        st = await self.run_("rb", cfg=second_seed("rb"), stop_after=12).start()
        spent = st["spend"]["train_usd"]
        with self.assertRaises(ValueError):  # checkpoint 7 wasn't kept
            await self.run_("rb").resume(from_checkpoint=7)

        st = await self.run_("rb", stop_after=8).resume(from_checkpoint=5)
        self.assert_contiguous(st, 8)
        self.assertEqual(st["branch"], 1)
        self.assertEqual(st["branches"][1]["from"], {"checkpoint": 5, "branch": 0, "backend_step": 5})
        # a new model started from checkpoint 5's LoRA, numbering its own checkpoints from 1
        self.assertEqual(st["branches"][1]["model"], "rb-b1")
        self.assertEqual(self.dry.backend.branched, [("rb-b1", "rb", 5)])
        self.assertEqual([r["branch"] for r in st["steps"]], [0] * 5 + [1] * 3)
        self.assertEqual(st["steps"][5]["sampled_from"]["model"], "rb:step5")  # step 6 samples from checkpoint 5
        self.assertEqual(st["checkpoints"]["6"], {"branch": 1, "backend_step": 1})
        self.assertEqual(st["steps"][6]["sampled_from"]["model"], "rb-b1:step1")
        self.assertEqual([r["step"] for r in st["abandoned"][0]["steps"]], list(range(6, 13)))
        gone = self.results / "rb" / "abandoned" / "b0-after5" / "transcripts" / "train"
        self.assertEqual(sorted(p.name for p in gone.glob("*.jsonl")), [f"step{s:04d}.jsonl" for s in range(6, 13)])
        self.assertTrue((self.results / "rb" / "abandoned" / "b0-after5" / "audit" / "step0010").exists())
        self.assertEqual(len(TranscriptLog(transcript_path(self.results / "rb", "train", 6)).records()), 32)
        self.assertGreater(st["spend"]["train_usd"], spent)  # spend never rolls back
        self.assertIn(10, self.dry.backend.models["rb"]["steps"])  # the old branch is untouched
        self.assertNotIn("10", st["audits"])

    async def test_a_resume_turns_the_format_penalty_on_for_the_rollouts_after_it(self):
        """prod-v1 (2026-10-18): a run started before the format penalty existed gets it on resume, with the rollback."""
        st = await self.run_("fp", cfg=second_seed("fp"), stop_after=6).start()
        self.assertEqual(st["config"]["format_penalty"], 0.0)
        oracle = self.dry.server.policy_for

        def broken_first(model, messages):  # every episode opens with a reply cut off mid-code
            if sum(m["role"] == "assistant" for m in messages) == 0:
                return "```python\nprint("
            return oracle(model, messages)

        self.dry.server.policy_for = broken_first
        with self.assertRaises(ValueError):
            await self.run_("fp").resume(format_penalty=0.5)
        st = await self.run_("fp", stop_after=6).resume(from_checkpoint=5, format_penalty=0.1)
        self.assertEqual((st["latest"], st["branch"], st["config"]["format_penalty"]), (6, 1, 0.1))
        self.assertTrue(any(n["kind"] == "format" and "0 -> 0.1" in n["text"] for n in st["notices"]))
        before = TranscriptLog(transcript_path(self.results / "fp", "train", 5)).records()
        after = TranscriptLog(transcript_path(self.results / "fp", "train", 6)).records()
        self.assertFalse(any("format_penalty" in r["reward_components"] for r in before))
        self.assertTrue(after and all(r["reward_components"].get("format_penalty") == 0.1 for r in after
                                      if r["infra_error"] is None))
        self.assertGreaterEqual(st["steps"][-1]["shape"]["broken_replies"], 1.0)
        self.assertEqual(config.production("p", 2e-5).format_penalty, 0.1)

    async def test_a_branch_model_that_already_has_checkpoints_stops_the_resume(self):
        await self.run_("rw", cfg=second_seed("rw"), stop_after=6).start()
        self.dry.backend.models["rw-b1"] = {"steps": [1], "next": 2}  # e.g. left by an earlier attempt
        with self.assertRaises(RuntimeError):  # training would continue from its step 1, not from checkpoint 5
            await self.run_("rw").resume(from_checkpoint=5)
        self.assertEqual(self.dry.backend.trained[-1]["model"], "rw")  # nothing trained on the stale model


class LearningRateFallbackTest(LoopCase):
    async def test_a_collapse_at_2e5_rolls_back_to_1e5_and_a_second_one_stops_the_run(self):
        # entropy collapses from checkpoint 13 on, and again on the branch the rollback starts (its own steps 1, 2, ...)
        self.dry.server.completion_logprob = lambda model: -0.1 if Oracle.step_of(model) >= 13 or "-b1:" in model else -0.6
        st = await self.run_("lrf", cfg=config.production("lrf", 2e-5, tasks_from="tickets")).start()
        self.assertEqual((st["status"], st["stop"]["rule"]), ("stopped", "frying"), st["stop"])
        self.assertIn("came back", st["stop"]["reason"])
        fallback = st["lr"]["fallback"]
        # the fake re-scores 300 tokens a step, so the pool reaches back 5 steps and crosses 30% at step 15; the collapse
        # dates from step 14 (sampled from checkpoint 13), so the last good kept checkpoint is 10
        self.assertEqual((fallback["signal"], fallback["to_checkpoint"], fallback["at_step"]), ("entropy", 10, 15))
        self.assertEqual(st["lr"]["current"], 1e-5)
        rates = {(t["model"], t["kwargs"]["learning_rate"]) for t in self.dry.backend.trained}
        self.assertEqual(rates, {("lrf", 2e-5), ("lrf-b1", 1e-5)})
        self.assertEqual(st["abandoned"][0]["after"], 10)
        kinds = [n["kind"] for n in st["notices"]]
        self.assertLess(kinds.index("lr_fallback"), kinds.index("rollback"))


class PauseTest(LoopCase):
    async def test_an_infra_error_pauses_the_second_seed_until_resumed(self):
        made = []

        def factory(context, complete, **kw):
            made.append(1)
            return FakeSandbox(context, complete, kill_after_cells=0 if len(made) == 33 else None, **kw)

        st = await self.run_("inf", cfg=second_seed("inf", 4), sandbox_factory=factory).start()
        self.assertEqual((st["status"], st["stop"]["rule"], st["latest"]), ("paused", "infra", 2))
        self.assertEqual(st["steps"][1]["infra"]["kinds"], {"sandbox": 1})
        self.assertEqual(st["steps"][1]["status"], "trained")  # the step's clean groups still trained
        st = await self.run_("inf").resume()
        self.assertEqual(st["status"], "finished")
        self.assert_contiguous(st, 4)

    async def test_an_audit_hack_pauses_and_the_resume_acknowledges_it(self):
        def mark_hack():
            path = self.results / "hk" / "audit" / "step0010" / "verdicts.json"
            verdicts = json.loads(path.read_text(encoding="utf-8"))
            win = next(e for e in verdicts["episodes"] if e["kind"] == "win")
            win.update(verdict="hack", rule="guessed", note="turn 2 submits the size line's count")
            path.write_text(json.dumps(verdicts), encoding="utf-8")

        run = self.hook_draw(self.run_("hk", cfg=config.production("hk", 2e-5, tasks_from="tickets")), 12, mark_hack)
        st = await run.start()
        self.assertEqual((st["status"], st["stop"]["rule"], st["latest"]), ("paused", "hacking", 12))
        st = await self.run_("hk", stop_after=13).resume()
        self.assertEqual((st["status"], st["stop"]["rule"], st["latest"]), ("stopped", "operator", 13))
        self.assertEqual(len(st["acknowledged_audit_hacks"]), 1)
        self.assertEqual(st["audits"]["10"]["audited"], 1)

    async def test_the_ceiling_stops_the_run_before_it_overspends(self):
        cfg = second_seed("bud")
        cfg.ceilings = {"train": 0.10, "eval": 3.0}
        cfg.first_step_estimate_usd = 0.02
        st = await self.run_("bud", cfg=cfg).start()
        self.assertEqual((st["status"], st["stop"]["rule"]), ("stopped", "budget"))
        self.assertLessEqual(st["spend"]["train_usd"], 0.10)
        self.assertGreaterEqual(st["latest"], 3)
        self.assertIn("ceiling", st["stop"]["reason"])


class SecretsTest(LoopCase):
    async def test_results_hold_no_seed_and_no_key(self):
        os.environ["WANDB_API_KEY"] = "canary-key-123"
        try:
            await self.run_("sec", cfg=config.production("sec", 2e-5, tasks_from="tickets"), stop_after=10).start()
        finally:
            del os.environ["WANDB_API_KEY"]
        seeds = json.loads((self.results / "private" / "seeds" / "selection-tickets-v1.json").read_text())
        self.assertEqual(len(seeds), 64)
        files = [self.results / "sec.json", self.results / "sec" / "config.json",
                 *(self.results / "sec" / "checkpoints").glob("*.json"), *(self.results / "sec").rglob("*.jsonl")]
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("canary-key-123", text, path)
            for seed in seeds:
                self.assertNotIn(str(seed), text, path)

    def test_new_run_folders_are_git_ignored(self):
        for path in ("results/x/pending/step0003.pkl", "results/x/capability/step0010.jsonl",
                     "results/x/abandoned/b0-after5/transcripts/train/step0006.jsonl", "results/dry/x.json"):
            out = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
            self.assertEqual(out.returncode, 0, path)


class DockerLoopTest(LoopCase):
    """Two steps and a resume through real sandbox containers, inside WSL2."""

    @classmethod
    def setUpClass(cls):
        reason = docker_ready()
        if reason:
            raise unittest.SkipTest(reason)
        cls.image = sandbox.build_image()

    async def test_steps_and_a_resume_through_docker(self):
        kw = {"sandbox_factory": sandbox.Sandbox, "sandbox_kwargs": {"image": self.image}}
        st = await self.run_("dock", cfg=second_seed("dock", 3), stop_after=2, **kw).start()
        self.assertEqual((st["status"], st["latest"]), ("stopped", 2))
        st = await self.run_("dock", **kw).resume()
        self.assertEqual(st["status"], "finished")
        self.assert_contiguous(st, 3)
        self.assertEqual(sum(r["infra"]["infra"] for r in st["steps"]), 0)
        self.assertEqual(st["evals"][0]["summary"]["episodes"], 100)
        left = subprocess.run(["docker", "ps", "-aq", "--filter", "label=rlm.run=dock"], capture_output=True,
                              text=True).stdout.split()
        self.assertEqual(left, [])


class CliTest(unittest.TestCase):
    def cli(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(REPO / "scripts" / "train.py"), *args], capture_output=True,
                              text=True, encoding="utf-8", cwd=REPO, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                              timeout=600)

    def test_dry_run_start_stop_status_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.cli("start", "--dry-run", "--kind", "second_seed", "--run", "cli", "--max-steps", "4",
                           "--stop-after", "2", "--results", tmp)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("stopped - operator", out.stdout)
            out = self.cli("stop", "cli", "--results", tmp, "--reason", "test")
            self.assertTrue((Path(tmp) / "cli" / training.STOP_FILE).exists())
            out = self.cli("resume", "cli", "--dry-run", "--results", tmp)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("finished", out.stdout)  # a resume clears a leftover switch
            out = self.cli("status", "cli", "--results", tmp)
            self.assertIn("latest checkpoint 4", out.stdout)

    def test_a_live_or_dry_choice_is_required(self):
        out = self.cli("start", "--kind", "second_seed", "--run", "x")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("--live", out.stderr)
        out = self.cli("start", "--dry-run", "--kind", "production", "--run", "x")
        self.assertIn("needs --lr", out.stderr)


if __name__ == "__main__":
    unittest.main()
