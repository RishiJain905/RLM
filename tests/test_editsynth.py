"""edit_synth (spec §12.7, plan step E3): three kinds, levels 1-5, hidden tests drawn on the host, a held-out variant.

The generated tests really run here, with this machine's Python and the grader's own unittest runner
(grader_driver.py), so no Docker is needed. test_editbox.py grades through the real container.
"""

import ast
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rlm_rl import bridge, grader, tasks
from rlm_rl.tasks import editsynth

DRIVER = Path(grader.__file__).with_name("grader_driver.py")


def base_files(task) -> dict:
    files = {}

    class Collect:
        def __truediv__(self, path):
            return _Writer(files, path)

    task.workspace.build(Collect())
    return files


class _Writer:
    def __init__(self, files, path):
        self.files, self.path = files, path

    @property
    def parent(self):
        return self

    def mkdir(self, **kw):
        pass

    def write_text(self, text, encoding=None):
        self.files[self.path] = text


def run_tests(files: dict, spec: dict) -> dict:
    """Run the task's hidden and visible tests on `files` the way the grader does. Returns {test key: outcome}."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for path, text in {**files, **spec["hidden"]}.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text, encoding="utf-8")
        report = root / "report.json"
        subprocess.run([sys.executable, "-I", "-B", str(DRIVER), "unittest", "--report", str(report), "--top", ".",
                        *spec["scope"]], cwd=root, capture_output=True, timeout=300)
        return json.loads(report.read_text())["tests"]


def resolved(results: dict, spec: dict) -> bool:
    return grader.score(results, grader.TestSpec.from_dict(spec))[2]


def strip_function(text: str, name: str, stub: str | None) -> str:
    """Delete a function, or replace its body with `stub` (e.g. "return None")."""
    tree = ast.parse(text)
    lines = text.split("\n")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start, end = node.lineno - 1, node.end_lineno
            head = lines[start]
            replacement = [head, f"    {stub}"] if stub else []
            return "\n".join(lines[:start] + replacement + lines[end:])
    raise AssertionError(f"{name} not found")


class KindsAndLevelsTest(unittest.TestCase):
    def test_every_kind_level_and_family(self):
        """The gold change resolves every task; the base fails every hidden fail-to-pass test and passes every
        pass-to-pass one, so each task has signal."""
        for family in editsynth.FAMILIES:
            for kind in editsynth.KINDS:
                for level in range(1, editsynth.MAX_LEVEL + 1):
                    with self.subTest(family=family, kind=kind, level=level):
                        task = editsynth.make_task(101 * level + len(kind), family, level, kind=kind)
                        spec, files = task.grading["tests"], base_files(task)
                        base = run_tests(files, spec)
                        self.assertTrue(all(base.get(k) != "passed" for k in spec["f2p"]), base)
                        self.assertTrue(all(base.get(k) == "passed" for k in spec["p2p"]), base)
                        gold = run_tests({**files, **task.grading["gold_files"]}, spec)
                        self.assertTrue(resolved(gold, spec), gold)
                        lo, hi = editsynth.LEVELS[level]
                        self.assertTrue(0.7 * lo <= task.grading["size_chars"] <= 1.2 * hi, task.grading["size_chars"])

    def test_blind_patches_score_zero(self):
        """Spec §12.6 H1 for edits: the empty patch, a patch that deletes the target code and one that stubs it to
        return a constant all score 0, on every kind and level."""
        for kind in editsynth.KINDS:
            for level in range(1, editsynth.MAX_LEVEL + 1):
                task = editsynth.make_task(7 * level + 3 * len(kind), "edit_synth", level, kind=kind)
                spec, files, target = task.grading["tests"], base_files(task), task.grading["target"]
                gold = {**files, **task.grading["gold_files"]}
                path = target["path"]
                for name, patched in (("empty", files),
                                      ("delete", {**gold, path: strip_function(gold[path], target.get("new") or target["function"], None)}),
                                      ("stub", {**gold, path: strip_function(gold[path], target.get("new") or target["function"], "return None")}),
                                      ("delete-base", {**files, path: strip_function(files[path], target["function"], None)}),
                                      ("stub-base", {**files, path: strip_function(files[path], target["function"], "return 0")})):
                    with self.subTest(kind=kind, level=level, patch=name):
                        self.assertFalse(resolved(run_tests(patched, spec), spec))

    def test_visible_tests_pass_on_the_base(self):
        task = editsynth.make_task(5, "edit_synth", 2, kind="feature")
        with tempfile.TemporaryDirectory() as tmp:
            for path, text in base_files(task).items():
                (Path(tmp) / path).parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / path).write_text(text, encoding="utf-8")
            out = subprocess.run([sys.executable, *editsynth.TEST_HINT.split()[1:]], cwd=tmp, capture_output=True,
                                 text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr[-500:])


class TaskShapeTest(unittest.TestCase):
    def test_what_the_model_sees(self):
        task = editsynth.make_task(11, "edit_synth", 3, kind="callers")
        files = base_files(task)
        self.assertTrue(bridge.is_edit(task.family))
        self.assertEqual(task.edit["n_files"], len(files))
        self.assertEqual(task.edit["test_hint"], editsynth.TEST_HINT)
        self.assertTrue(task.context.startswith(task.question))
        for path in files:
            self.assertIn(path, task.context)  # the file list, with sizes
        for path in task.grading["tests"]["hidden"]:
            self.assertNotIn(path, files)  # hidden tests are never in the repository
            self.assertNotIn(Path(path).stem, task.context)
        self.assertIsNone(task.answer)
        first = bridge.initial_messages(task.question, task.context, edit=task.edit, limits=bridge.limits(task.family))[1]
        self.assertIn("You have at most 20 turns and 30 sub-calls, and each cell may run for 300 seconds.", first["content"])

    def test_hidden_inputs_differ_from_the_task_s_example(self):
        for seed in range(6):
            task = editsynth.make_task(seed, "edit_synth", 2, kind="fix")
            entry, example = re.search(r"`(\w+)\((.*?)\)` (?:returns|comes back)", task.question).groups()
            hidden = "\n".join(task.grading["tests"]["hidden"].values())
            target = [line for line in hidden.splitlines() if f"api.{entry}(" in line]
            self.assertGreaterEqual(len(target), editsynth.HIDDEN_INPUTS)
            self.assertNotIn(f"api.{entry}({example})", "\n".join(target))

    def test_seeds_and_determinism(self):
        a = editsynth.make_task(123456789, "edit_synth", 2)
        b = editsynth.make_task(123456789, "edit_synth", 2)
        self.assertEqual((a.question, base_files(a)), (b.question, base_files(b)))
        self.assertEqual(a.workspace.key, b.workspace.key)
        self.assertNotIn("123456789", a.task_id + a.context + json.dumps(a.grading))
        c = editsynth.make_task(987654321, "edit_synth", 2)
        self.assertNotEqual(a.workspace.key, c.workspace.key)

    def test_registered_and_heldout_is_eval_only(self):
        self.assertEqual(tasks.max_levels(["edit_synth", "edit_synth_heldout"]), {"edit_synth": 5, "edit_synth_heldout": 5})
        self.assertTrue(tasks.eval_only("edit_synth_heldout"))
        self.assertFalse(tasks.eval_only("edit_synth"))
        with self.assertRaisesRegex(ValueError, "eval-only"):
            tasks.check_mix({"edit_synth_heldout": 1.0})
        tasks.check_mix({"edit_synth": 1.0})  # ours: no dataset, nothing for the licence bar

    def test_heldout_wording_and_mutations_never_appear_in_training(self):
        """H7: held-out templates share no 6-word sequence with training's; their mutation kinds and options differ."""
        def grams(text):
            words = re.findall(r"[a-z]+", text.lower())
            return {tuple(words[i:i + 6]) for i in range(len(words) - 5)}

        train = set().union(*(grams(t) for ts in editsynth.TRAIN_TEXTS.values() for t in ts))
        held = set().union(*(grams(t) for ts in editsynth.HELDOUT_TEXTS.values() for t in ts))
        self.assertFalse(train & held)
        self.assertFalse({p for p, _ in editsynth.TRAIN_MUTATIONS} & {p for p, _ in editsynth.HELDOUT_MUTATIONS})
        self.assertFalse(set(editsynth.TRAIN_OPTIONS) & set(editsynth.HELDOUT_OPTIONS))
        self.assertNotEqual(editsynth.TRAIN_PARAM[0], editsynth.HELDOUT_PARAM[0])


if __name__ == "__main__":
    unittest.main()
