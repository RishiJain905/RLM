"""The AA-LCR post-training benchmark: the loader (rlm_rl/tasks/aalcr.py) and the judge and summary helpers of
scripts/benchmark.py, on a tiny fake copy of the dataset."""

import csv
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlm_rl import baselines, data
from rlm_rl.tasks import aalcr

ROOT = Path(__file__).resolve().parents[1]

ROWS = [  # (category, set id, question id, answer, files)
    ("Academia", "ac_a", 1, "Airline (12)", "x.txt;y.txt"),
    ("Academia", "ac_a", 2, "Apple v. Samsung", "y.txt"),
    ("Company Documents", "cd_b", 3, "$901 million", "q1.txt;q2.txt"),
    ("Company Documents", "cd_b", 4, "Equinix", "q2.txt"),
    ("Legal", "lg_c", 5, "Section 4", "act.txt"),
]


def _benchmark_script():
    spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDataMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name) / aalcr.DIR
        base.mkdir(parents=True)
        with open(base / "AA-LCR_Dataset.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", "document_category", "document_set_id", "question_id", "question", "answer",
                        "data_source_filenames", "data_source_urls", "input_tokens"])
            for i, (cat, set_id, qid, answer, files) in enumerate(ROWS):
                w.writerow([i, cat, set_id, qid, f"Question {qid}?", answer, files, "", 90000])
                folder = base / "extracted" / "lcr" / cat.replace(" ", "_") / set_id
                folder.mkdir(parents=True, exist_ok=True)
                for name in files.split(";"):
                    (folder / name).write_text(f"text of {name}", encoding="utf-8")
        patcher = mock.patch.dict(os.environ, {data.DATA_ENV: self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        aalcr.load.cache_clear()
        self.addCleanup(aalcr.load.cache_clear)


class LoaderTest(FakeDataMixin, unittest.TestCase):
    def test_every_question_once_categories_spread(self):
        tasks = aalcr.load()
        self.assertEqual(sorted(t.grading["question_id"] for t in tasks), [1, 2, 3, 4, 5])
        firsts = [t.grading["category"] for t in tasks[:3]]
        self.assertEqual(sorted(firsts), ["Academia", "Company Documents", "Legal"])
        # each two-question category has one question in each half of the order
        for cat in ("Academia", "Company Documents"):
            spots = [i for i, t in enumerate(tasks) if t.grading["category"] == cat]
            self.assertLess(spots[0], 3)
            self.assertGreaterEqual(spots[1], 3)

    def test_task_ids_sort_in_the_fixed_order(self):
        tasks = aalcr.load()
        ids = [t.task_id for t in tasks]
        self.assertEqual(ids, sorted(ids))  # baselines.plan_jobs starts read episodes in task-id order
        jobs = baselines.plan_jobs(list(tasks), baselines.ARMS["bench_ckpt"])
        self.assertEqual([j.task.task_id for j in jobs], ids)

    def test_documents_in_the_listed_order_under_the_research_header(self):
        task = next(t for t in aalcr.load() if t.grading["question_id"] == 1)
        self.assertEqual(task.context, "# ===== doc 1: x\ntext of x.txt\n\n# ===== doc 2: y\ntext of y.txt")
        self.assertTrue(task.question.startswith("Question 1?"))
        self.assertEqual(task.answer, "Airline (12)")
        self.assertEqual(task.family, aalcr.FAMILY)

    def test_eval_only_and_scorable(self):
        from rlm_rl import reward

        self.assertEqual(data.SOURCES["aa-lcr"]["role"], data.EVAL)
        for task in aalcr.load():
            reward.check_grading(task.grading, task.answer)


class JudgeHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = _benchmark_script()

    def test_parse_verdict(self):
        self.assertEqual(self.b.parse_verdict('{"verdict": "CORRECT"}'), "CORRECT")
        self.assertEqual(self.b.parse_verdict('```json\n{"verdict":"incorrect"}\n```'), "INCORRECT")
        self.assertEqual(self.b.parse_verdict("The answer is INCORRECT."), "INCORRECT")
        self.assertIsNone(self.b.parse_verdict("no idea"))

    def test_answer_text(self):
        self.assertIsNone(self.b.answer_text(None))
        self.assertIsNone(self.b.answer_text("  "))
        self.assertEqual(self.b.answer_text(["Apple v. Samsung", "Crocs"]), "Apple v. Samsung, Crocs")
        self.assertEqual(self.b.answer_text(901), "901")

    def test_question_id(self):
        self.assertEqual(self.b.question_id("aa-lcr-007-q042"), 42)

    def test_mcnemar(self):
        self.assertIsNone(self.b.mcnemar_p(0, 0))
        self.assertEqual(self.b.mcnemar_p(3, 3), 1.0)
        self.assertAlmostEqual(self.b.mcnemar_p(8, 1), 0.0390625)

    def test_judge_prompt_matches_the_card(self):
        self.assertIn("Reply as JSON, with a verdict of CORRECT or INCORRECT.", self.b.JUDGE_USER)
        self.assertTrue(self.b.JUDGE_SYSTEM.startswith("Decide whether the CANDIDATE ANSWER is correct"))

    def test_oolong_score(self):
        self.assertAlmostEqual(self.b.oolong_score({"gold": 10, "answer": "12"}), 0.5625)
        self.assertEqual(self.b.oolong_score({"gold": 10, "answer": 10}), 1.0)
        self.assertEqual(self.b.oolong_score({"gold": 10, "answer": None}), 0.0)
        self.assertEqual(self.b.oolong_score({"gold": "spam", "answer": "spam", "correct": True}), 1.0)
        self.assertEqual(self.b.oolong_score({"gold": "spam", "answer": "ham", "correct": False}), 0.0)

    def test_attached_needs_a_real_difference(self):
        self.assertIsNone(self.b.mean_abs_diff([None], [None]))
        self.assertAlmostEqual(self.b.mean_abs_diff([None, -1.0, -2.0], [None, -1.5, -2.0]), 0.25)


if __name__ == "__main__":
    unittest.main()
