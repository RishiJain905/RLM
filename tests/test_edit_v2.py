"""edit_spec and edit_bugs (edit tasks v2): the Python analysis, stubs, clusters, the requirements document, the
candidate bugs, the symptom-only task text, the fixed pools, the workspace a record builds, and the builder's
helpers. The last classes read the built indexes on the data drive and, with Docker, grade one task of each family in
the real grader; they skip when those are missing.
"""

import ast
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlm_rl import bridge, editdata, grader, tasks
from rlm_rl.tasks import PoolEmpty, editbugs, editspec

ROOT = Path(__file__).resolve().parents[1]
linux_only = unittest.skipUnless(sys.platform == "linux", "git workspaces and the grader run on Linux (WSL2)")


def builder():
    """scripts/build_edit_tasks_v2.py, loaded as a module."""
    spec = importlib.util.spec_from_file_location("build_edit_tasks_v2", ROOT / "scripts" / "build_edit_tasks_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CORE = '''"""Core helpers for demo."""

from .util import clamp


def total(values, limit=10):
    """Sum the values, each clamped to the limit."""
    out = 0
    for v in values:
        out += clamp(v, limit)
    return out


def average(values):
    """The mean of the clamped values, or 0 for none."""
    if not values:
        return 0
    return total(values) / len(values)


class Box:
    """A box of numbers."""

    def __init__(self, items):
        self.items = list(items)

    def size(self):
        """How many items the box holds."""
        n = len(self.items)
        return n

    def mean(self):
        """The box's average."""
        if self.size() == 0:
            return 0
        return average(self.items)

    @property
    def biggest(self):
        """The largest item."""
        best = None
        for x in self.items:
            if best is None or x > best:
                best = x
        return best
'''
UTIL = '''"""Utilities."""


def clamp(value, limit):
    """Limit value to the range -limit..limit."""
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def pairs(items, step):
    out = []
    for i in range(0, len(items) - 1, step):
        out.append((items[i], items[i + 1]))
    return out
'''
API = '''"""The public API."""

from . import core
from .util import pairs


def summary(values):
    """A summary dict."""
    box = core.Box(values)
    return {"mean": box.mean(), "pairs": pairs(values, 1), "max": box.biggest}
'''
TEST = 'from demo import api\n\n\ndef test_summary():\n    assert api.summary([1, 2])["mean"] == 1.5\n'
FILES = {"demo/__init__.py": '"""demo: a tiny package."""\n', "demo/core.py": CORE, "demo/util.py": UTIL,
         "demo/api.py": API, "tests/test_api.py": TEST, "tests/conftest.py": ""}


def demo_repo():
    return editspec.analyze(FILES)


def everything_tested(repo) -> dict:
    """Every function run by two tests, the way the lab's profile reports it."""
    return {k: {"tests/test_api.py::test_a", "tests/test_api.py::test_b"} for k in repo.fns}


class AnalysisTest(unittest.TestCase):
    def test_inventory(self):
        repo = demo_repo()
        self.assertEqual(sorted(repo.fns), sorted([
            "demo/api.py::summary", "demo/core.py::Box.__init__", "demo/core.py::Box.biggest", "demo/core.py::Box.mean",
            "demo/core.py::Box.size", "demo/core.py::average", "demo/core.py::total", "demo/util.py::clamp",
            "demo/util.py::pairs"]))  # tests and conftest.py are not source
        fns = repo.fns
        self.assertFalse(fns["demo/core.py::Box.__init__"].stubbable)  # dunder methods stay
        self.assertTrue(all(fns[k].stubbable for k in fns if not k.endswith("__init__")))
        self.assertEqual(fns["demo/core.py::Box.biggest"].kind, "property")
        self.assertEqual(fns["demo/core.py::Box.biggest"].first + 1, fns["demo/core.py::Box.biggest"].def_line)
        self.assertEqual(fns["demo/core.py::total"].doc, "Sum the values, each clamped to the limit.")
        self.assertEqual(fns["demo/core.py::total"].signature, "def total(values, limit=10):")
        self.assertEqual(fns["demo/core.py::total"].body_lines, 4)
        self.assertEqual(repo.modules["demo/core.py"].name, "demo.core")

    def test_references_resolve_through_imports_self_and_modules(self):
        refs = {k: fn.refs for k, fn in demo_repo().fns.items()}
        self.assertEqual(refs["demo/core.py::total"], {"demo/util.py::clamp"})  # from .util import clamp
        self.assertEqual(refs["demo/core.py::average"], {"demo/core.py::total"})
        self.assertEqual(refs["demo/core.py::Box.mean"], {"demo/core.py::Box.size", "demo/core.py::average"})  # self.
        self.assertEqual(refs["demo/api.py::summary"], {"demo/util.py::pairs"})  # box.mean() can't be typed statically
        usage = [u for u in demo_repo().usages if u.target == "demo/util.py::clamp"]
        self.assertEqual([(u.where, u.text) for u in usage], [("demo/core.py::total", "out += clamp(v, limit)")])

    def test_a_local_name_never_resolves_to_the_module_function(self):
        shadow = ('"""m."""\nfrom .util import clamp\n\n\ndef f(clamp, limit):\n    """Uses its own clamp."""\n'
                  '    value = clamp\n    return value + limit\n\n\ndef g(x):\n    """Calls util.clamp."""\n'
                  '    y = clamp(x, 3)\n    return y\n')
        repo = editspec.analyze({**FILES, "demo/shadow.py": shadow})
        self.assertEqual(repo.fns["demo/shadow.py::f"].refs, set())
        self.assertEqual(repo.fns["demo/shadow.py::g"].refs, {"demo/util.py::clamp"})

    def test_profile_entries_map_to_functions(self):
        repo = demo_repo()
        biggest = repo.fns["demo/core.py::Box.biggest"]
        profile = {"<collect>": ["demo/util.py:4:clamp"],
                   "tests/test_api.py::test_a": [f"demo/core.py:{biggest.first}:biggest", "demo/core.py:1:<module>",
                                                 f"demo/core.py:{repo.fns['demo/core.py::total'].first}:total"]}
        calls, tests_of, at_import = editspec.map_profile(repo, profile)
        self.assertEqual(calls["tests/test_api.py::test_a"], {"demo/core.py::Box.biggest", "demo/core.py::total"})
        self.assertEqual(tests_of["demo/core.py::total"], {"tests/test_api.py::test_a"})
        self.assertEqual(at_import, {"demo/util.py::clamp"})
        self.assertNotIn("demo/util.py::clamp", editspec.eligible(repo, tests_of, at_import))  # never stub an import

    def test_stubs_keep_signatures_and_docstrings(self):
        repo = demo_repo()
        fns = [repo.fns["demo/core.py::total"], repo.fns["demo/core.py::Box.biggest"]]
        out = editspec.stub_text(CORE, fns)
        self.assertEqual(out.count("raise NotImplementedError"), 2)
        self.assertIn('    """Sum the values, each clamped to the limit."""\n    raise NotImplementedError\n', out)
        self.assertIn('    @property\n    def biggest(self):\n        """The largest item."""\n'
                      '        raise NotImplementedError\n', out)
        self.assertNotIn("clamp(v, limit)", out)
        self.assertIn("return total(values) / len(values)", out)  # the rest is untouched
        ast.parse(out)

    def test_newlines_round_trip(self):
        self.assertEqual(editspec.split_newlines("a\r\nb\r\n"), ("a\nb\n", "\r\n"))
        self.assertEqual(editspec.split_newlines("a\nb\n"), ("a\nb\n", "\n"))
        self.assertIsNone(editspec.split_newlines("a\r\nb\n"))


class ClusterTest(unittest.TestCase):
    def test_levels_give_connected_clusters_of_their_size_and_spread(self):
        repo = demo_repo()
        tests_of = everything_tested(repo)
        for level, ((kmin, kmax), (fmin, fmax)) in editspec.LEVELS.items():
            if kmin > 4:
                continue  # the demo package has too few functions for levels 4 and 5
            got = editspec.spec_clusters(repo, tests_of, set(), level, random.Random(level), 5, {})
            for cluster in got:
                files = {repo.fns[c].path for c in cluster}
                self.assertTrue(kmin <= len(cluster) <= kmax, (level, cluster))
                self.assertTrue(fmin <= len(files) <= fmax, (level, cluster))
                self.assertTrue(any(repo.fns[c].doc for c in cluster))
            self.assertEqual(len({frozenset(c) for c in got}), len(got))
        one = editspec.spec_clusters(repo, tests_of, set(), 1, random.Random(0), 5, {})
        self.assertTrue(one)

    def test_reuse_caps(self):
        repo = demo_repo()
        tests_of = everything_tested(repo)
        total: dict = {}
        for level in (1, 2, 3):
            editspec.spec_clusters(repo, tests_of, set(), level, random.Random(level), 9, {}, total=total, max_total=2)
        self.assertLessEqual(max(total.values()), 2)

    def test_whole_module_and_build_order(self):
        repo = demo_repo()
        keys = editspec.whole_module(repo, "demo/core.py", set(repo.fns), 3, 8)
        self.assertEqual(len(keys), 5)  # every stubbable function of core.py (not __init__)
        order = editspec.build_order(repo, ["demo/core.py::Box.mean", "demo/core.py::average", "demo/core.py::total"])
        self.assertEqual(order, ["demo/core.py::total", "demo/core.py::average", "demo/core.py::Box.mean"])


class RequirementsTest(unittest.TestCase):
    def setUp(self):
        self.repo = demo_repo()
        self.cluster = ["demo/core.py::average", "demo/core.py::total", "demo/util.py::clamp"]
        self.doc = editspec.requirements("someone/demo", self.repo, self.cluster)

    def test_the_document_names_each_function_with_its_interface(self):
        doc = self.doc
        self.assertTrue(doc.startswith("Implement the missing code in demo (3 functions in 2 files)."))
        for sig in ("def total(values, limit=10):", "def average(values):", "def clamp(value, limit):"):
            self.assertIn(sig, doc)
        self.assertIn("Limit value to the range -limit..limit.", doc)
        self.assertIn("Module `demo.core`: Core helpers for demo.", doc)
        self.assertIn("It relies on these other missing functions: `clamp`.", doc)
        self.assertIn("It is used by these other missing functions: `average`.", doc)
        self.assertIn("A dependency order (each function after the missing ones it relies on): `clamp`, `total`, "
                      "`average`.", doc)

    def test_usage_comes_from_code_outside_the_cluster(self):
        # Box.mean (not in the cluster) calls average: quoted; total's call to clamp is in the cluster: not quoted
        self.assertIn("`demo/core.py`, `Box.mean`: `return average(self.items)`", self.doc)
        self.assertNotIn("out += clamp(v, limit)", self.doc)

    @linux_only
    def test_no_line_of_the_answer(self):
        b = builder()
        stubbed = editspec.stub_text(CORE, [self.repo.fns[k] for k in self.cluster[:2]])
        gold = b.file_patch({"demo/core.py": (stubbed, CORE)})
        self.assertIn("+        out += clamp(v, limit)", gold)
        self.assertEqual(editdata.leaks(self.doc, gold, []), [])


class BugsTest(unittest.TestCase):
    def setUp(self):
        self.repo = demo_repo()

    def kinds(self, key):
        return {c[0] for c in editbugs.candidates(self.repo, self.repo.fns[key])}

    def test_candidate_kinds(self):
        self.assertTrue({"boundary"} <= self.kinds("demo/util.py::clamp"))
        self.assertTrue({"off_by_one", "arithmetic"} <= self.kinds("demo/util.py::pairs"))
        self.assertTrue({"arithmetic", "swapped_arguments"} <= self.kinds("demo/core.py::total"))
        self.assertIn("wrong_branch", self.kinds("demo/core.py::average"))
        self.assertTrue({"boolean", "dropped_condition", "boundary"} <= self.kinds("demo/core.py::Box.biggest"))

    def test_each_mutation_parses_and_changes_the_program(self):
        for key, fn in self.repo.fns.items():
            for m in editbugs.mutations(self.repo, fn, random.Random(1), limit=10):
                text = self.repo.modules[fn.path].source.text
                out = editbugs.apply(text, [m])
                self.assertNotEqual(ast.dump(ast.parse(out)), ast.dump(ast.parse(text)), m.id)
                self.assertEqual(out.count("\n"), text.count("\n"), m.id)  # one site, no line added or lost
        rewritten = {editbugs.apply(UTIL, [m]).split("\n")[m.line - 1].strip()
                     for m in editbugs.mutations(self.repo, self.repo.fns["demo/util.py::clamp"], random.Random(0), 5)}
        self.assertTrue(rewritten & {"if value >= limit:", "if value <= -limit:"}, rewritten)

    def test_dropped_condition_keeps_the_other_operand(self):
        src = self.repo.modules["demo/core.py"].source
        drops = {src.text[:s] + new + src.text[e:] for kind, line, s, e, new in
                 editbugs.candidates(self.repo, self.repo.fns["demo/core.py::Box.biggest"]) if kind == "dropped_condition"}
        lines = {[ln for ln in t.split("\n") if "if best" in ln or "if x" in ln][0].strip() for t in drops}
        self.assertEqual(lines, {"if best is None:", "if x > best:"})

    def test_apply_refuses_a_stale_span(self):
        m = editbugs.mutations(self.repo, self.repo.fns["demo/util.py::clamp"], random.Random(0), 1)[0]
        with self.assertRaises(ValueError):
            editbugs.apply("x" + UTIL, [m])

    def test_bug_sets_pick_one_screened_bug_per_linked_function(self):
        tests_of = everything_tested(self.repo)
        good = {}
        for k, fn in self.repo.fns.items():
            ms = editbugs.mutations(self.repo, fn, random.Random(3), 2)
            good[k] = [{"mutation": m.as_dict(), "fails": [f"tests/test_api.py::t_{k}"]} for m in ms]
        for level in (1, 2, 3):
            (kmin, kmax), (fmin, fmax) = editbugs.LEVELS[level]
            for bugs in editbugs.bug_sets(self.repo, tests_of, good, level, random.Random(level), 4, {}):
                fns = [b["mutation"]["fn"] for b in bugs]
                self.assertEqual(len(fns), len(set(fns)))
                self.assertTrue(kmin <= len(fns) <= kmax)
                self.assertTrue(fmin <= len({b["mutation"]["path"] for b in bugs}) <= fmax)


class IssueTextTest(unittest.TestCase):
    def test_messages_lose_paths_and_line_numbers(self):
        msg = ("AssertionError: assert 3 == 4\n  /work/repo/sqlparse/engine/grouping.py:123: in group\n\x1b[31mred\x1b[0m"
               "\n  +  where 3 = len([1, 2, 3])")
        out = editbugs.clean_message(msg)
        self.assertNotIn("grouping.py", out)
        self.assertNotIn("123", out)
        self.assertNotIn("\x1b", out)
        self.assertIn("assert 3 == 4", out)
        self.assertLessEqual(len(editbugs.clean_message("x" * 2000)), 504)

    def test_the_text_gives_symptoms_and_the_number_of_faults(self):
        text = editbugs.issue_text("someone/demo", 3, [("tests/test_a.py::test_one", "AssertionError: assert 1 == 2"),
                                                        ("tests/test_b.py::TestB::test_two", "")])
        self.assertIn("1. `tests/test_a.py::test_one`\n   AssertionError: assert 1 == 2", text)
        self.assertIn("(the test failed without a message)", text)
        self.assertIn("3 separate faults", text)
        self.assertIn("one fault", editbugs.issue_text("x/demo", 1, [("t::a", "m")]))

    def test_each_bug_gets_a_listed_failure_of_its_own(self):
        bugs = [{"mutation": {"fn": "f"}, "fails": ["t::1", "t::2"]}, {"mutation": {"fn": "g"}, "fails": ["t::3"]}]
        calls = {"t::1": {"f", "g"}, "t::2": {"f"}, "t::3": {"g"}}
        listed = editbugs.listed_failures(["t::1", "t::2", "t::3", "t::4"], bugs, calls, random.Random(0))
        self.assertIn("t::2", listed)  # f's own failure that runs no other bugged function
        self.assertIn("t::3", listed)
        self.assertEqual(len(listed), 4)
        self.assertIsNone(editbugs.listed_failures(["t::1", "t::2"], bugs, calls, random.Random(0)))  # g is masked


def fake_record(i: int, split: str = "train", level: int = 1, kind: str = "spec") -> dict:
    spec = grader.TestSpec(runner="pytest", scope=["tests/test_api.py"], f2p=[f"tests/test_api.py::test_{i}"], p2p=[],
                           hidden={"tests/test_api.py": TEST}).as_dict()
    return {"id": f"{kind}-demo-l{level}-{i:04d}", "split": split, "language": "python", "repo": "someone/demo",
            "key": "someone__demo.0", "kind": kind, "source": "generated", "licence": "MIT", "image": "rlm-edit-demo:x",
            "snapshot": "edit/repos/demo.tar", "snapshot_sha256": "0", "break_patch": "", "gold_patch": "",
            "remove": ["tests/test_api.py"], "replace": {}, "issue": f"Do task {i}.", "node_modules": [],
            "tests": spec, "test_hint": "python -m pytest -q", "level": level, "gold_files": 1, "gold_lines": 3,
            "files": [["demo/core.py", 100]], "targets": [{"path": "demo/core.py", "qualname": "total"}]}


class PoolTest(unittest.TestCase):
    def setUp(self):
        self.recs = ([fake_record(i, level=1 + i % 3) for i in range(9)] + [fake_record(50, "select", 2)]
                     + [fake_record(60, "test", 4)])
        self.patch = mock.patch.object(editspec.POOL, "index", lambda: {"tasks": self.recs})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_the_interface_tasks_expects(self):
        for mod, fams in ((editspec, ("edit_spec", "edit_spec_select", "edit_spec_test")),
                          (editbugs, ("edit_bugs", "edit_bugs_select", "edit_bugs_test"))):
            self.assertEqual(mod.FAMILIES, fams)
            self.assertTrue(all(bridge.is_edit(f) for f in fams))
            self.assertEqual([tasks.eval_only(f) for f in fams], [False, True, True])
            self.assertEqual(mod.MAX_LEVEL, 5)
            for name in ("make_task", "pool_size", "training_sources", "check_licences", "eval_tasks", "GENERATOR_VERSION"):
                self.assertTrue(hasattr(mod, name), name)

    def test_draws_are_deterministic_nearest_level_and_skip_retired(self):
        a = editspec.make_task(5, "edit_spec", 2)
        self.assertEqual(a.task_id, editspec.make_task(5, "edit_spec", 2).task_id)
        self.assertEqual(a.level, 2)
        self.assertEqual(editspec.make_task(5, "edit_spec", 5).level, 3)  # nearest level that has tasks
        level2 = {r["id"] for r in self.recs if r["split"] == "train" and r["level"] == 2}
        b = editspec.make_task(5, "edit_spec", 2, exclude=level2 - {a.task_id})
        self.assertEqual(b.task_id, a.task_id)
        c = editspec.make_task(5, "edit_spec", 2, exclude=level2)
        self.assertNotEqual(c.level, 2)
        with self.assertRaises(PoolEmpty):
            editspec.make_task(5, "edit_spec", 2, exclude={r["id"] for r in self.recs})
        self.assertEqual(editspec.pool_size("edit_spec"), 9)
        self.assertEqual(editspec.make_task(1, "edit_spec_select", 1).level, 2)  # eval splits draw their own records
        with self.assertRaises(ValueError):
            editspec.make_task(1, "edit_bugs", 1)

    def test_the_task(self):
        t = editspec.make_task(5, "edit_spec", 1, task_id="t-1")
        self.assertEqual((t.task_id, t.family, t.answer), ("t-1", "edit_spec", None))
        self.assertEqual(t.grading["type"], "edit")
        self.assertEqual(grader.TestSpec.from_dict(t.grading["tests"]).f2p, t.grading["tests"]["f2p"])
        self.assertEqual(t.edit["image"], "rlm-edit-demo:x")
        self.assertIn("Repository files (path, size in bytes):\ndemo/core.py  100", t.context)
        self.assertTrue(t.workspace.key.startswith("spec-"))
        evals = editspec.eval_tasks("test", "edit_spec_test")
        self.assertEqual([e.task_id for e in evals], ["edit_spec_test-000"])
        self.assertEqual(editspec.training_sources(), ("generated",))
        editspec.check_licences()

    def test_through_the_task_registry_once_wired(self):
        """What the integrator adds to tasks/__init__.py: GENERATORS and RETIRABLE entries."""
        gens = dict(tasks.GENERATORS, edit_spec="editspec", edit_spec_select="editspec", edit_spec_test="editspec")
        with mock.patch.object(tasks, "GENERATORS", gens), \
                mock.patch.object(tasks, "RETIRABLE", tasks.RETIRABLE + ("edit_spec",)):
            first = tasks.make_task("edit_spec", 1, seed=3)
            again = tasks.make_task("edit_spec", 1, seed=3, exclude={first.task_id})
            self.assertNotEqual(first.task_id, again.task_id)
            self.assertEqual(tasks.max_levels(["edit_spec"]), {"edit_spec": 5})
            self.assertEqual(tasks.pool_size("edit_spec"), 9)
            tasks.check_mix({"edit_spec": 1.0})
            with self.assertRaises(ValueError):
                tasks.check_mix({"edit_spec_select": 1.0})


@linux_only
class WorkspaceTest(unittest.TestCase):
    """A record's workspace: the snapshot, the break patch applied, the hidden tests out; the gold patch undoes it."""

    def test_the_base_and_the_gold(self):
        b = builder()
        repo = demo_repo()
        stubbed = editspec.stub_text(CORE, [repo.fns["demo/core.py::total"], repo.fns["demo/core.py::average"]])
        tmp = Path(tempfile.mkdtemp(prefix="rlm-v2-test-"))
        try:
            src = tmp / "src"
            for p, text in FILES.items():
                (src / p).parent.mkdir(parents=True, exist_ok=True)
                (src / p).write_text(text, encoding="utf-8")
            (tmp / "data" / "edit" / "repos").mkdir(parents=True)
            with tarfile.open(tmp / "data" / "edit" / "repos" / "demo.tar", "w") as tar:
                tar.add(src, arcname=".")
            rec = fake_record(1)
            rec.update(break_patch=b.file_patch({"demo/core.py": (CORE, stubbed)}),
                       gold_patch=b.file_patch({"demo/core.py": (stubbed, CORE)}))
            with mock.patch.dict(os.environ, {"RLM_DATA_DIR": str(tmp / "data")}):
                folder = tmp / "ws"
                folder.mkdir()
                editspec.to_task(rec, "edit_spec").workspace.build(folder)
            self.assertEqual((folder / "demo" / "core.py").read_text(encoding="utf-8"), stubbed)
            self.assertFalse((folder / "tests" / "test_api.py").exists())
            self.assertTrue((folder / "tests" / "conftest.py").exists())
            ok, why = grader._safe_apply(folder, rec["gold_patch"], tmp)
            self.assertTrue(ok, why)
            self.assertEqual((folder / "demo" / "core.py").read_text(encoding="utf-8"), CORE)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class BuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = builder()

    def test_junit_keys_match_the_graders(self):
        b = self.b
        self.assertEqual(b.junit_key("tests/test_a.py::TestA::test_b[1-2]", "tests/test_a.py"),
                         "tests/test_a.py::TestA::test_b[1-2]")
        self.assertEqual(b.junit_key("tests/test_a.py::test_c", "tests/test_a.py"), "tests/test_a.py::test_c")
        # mistune: test methods generated from tests/__init__.py report that file
        self.assertEqual(b.junit_key("tests/test_commonmark.py::TestCommonMark::test_tabs_001", "tests/__init__.py"),
                         "tests/__init__.py::tests::test_commonmark::TestCommonMark::test_tabs_001")
        xml = ('<testsuites><testsuite><testcase classname="tests.test_commonmark.TestCommonMark" '
               'name="test_tabs_001" file="tests/__init__.py"/></testsuite></testsuites>')
        results, _ = b.parse_run({"exit": 0, "timed_out": False, "junit": xml})
        self.assertEqual(list(results), ["tests/__init__.py::tests::test_commonmark::TestCommonMark::test_tabs_001"])

    def test_judge(self):
        passed = {"a", "b", "c", "d"}
        orig = {"a": "passed", "b": "passed", "c": "passed", "d": "failed", "e": "passed"}
        mod = {"a": "failed", "b": "passed", "e": "failed"}  # c is missing: failed
        self.assertEqual(self.b.judge(passed, orig, mod), (["a", "c"], ["b"]))

    def test_collect_check(self):
        ok = {"exit": 0, "timed_out": False, "out": "12 tests collected in 0.1s"}
        bad = {"exit": 2, "timed_out": False, "out": "1 error during collection"}
        self.assertTrue(self.b.collect_ok(ok))
        self.assertFalse(self.b.collect_ok(bad))
        self.assertTrue(self.b.collect_ok(bad, baseline=bad))  # no worse than the original
        self.assertTrue(self.b.collect_ok({"exit": 5, "timed_out": False, "out": "no tests ran"}))

    def test_test_files(self):
        texts = {"tests/test_a.py": "", "tests/conftest.py": "", "test.py": "", "pkg/test/test_x.py": "",
                 "examples/test_demo.py": "", "pkg/mod.py": "", "tests/helpers.py": ""}
        self.assertEqual(self.b.test_files(texts), ["pkg/test/test_x.py", "test.py", "tests/test_a.py"])

    def test_lab_scripts_compile(self):
        compile(self.b.LAB_SCRIPT, "lab.py", "exec")
        compile(self.b.PROFILE_PLUGIN, "rlm_lab_profile.py", "exec")


def index_or_skip(mod):
    try:
        return mod.POOL.index()["tasks"]
    except (FileNotFoundError, OSError):
        raise unittest.SkipTest(f"no {mod.POOL.name} index on this machine")


class BuiltIndexTest(unittest.TestCase):
    """The pools scripts/build_edit_tasks_v2.py built (D:\\RLM-data\\edit\\v2)."""

    def test_the_pools(self):
        for mod in (editspec, editbugs):
            recs = index_or_skip(mod)
            ids = [r["id"] for r in recs]
            self.assertEqual(len(ids), len(set(ids)))
            by_split = {s: {r["repo"] for r in recs if r["split"] == s} for s in ("train", "select", "test")}
            self.assertFalse(by_split["train"] & (by_split["select"] | by_split["test"]))  # held-out repositories (H7)
            self.assertFalse(by_split["select"] & by_split["test"])
            for r in recs:
                self.assertEqual(r["kind"], mod.KIND)
                self.assertTrue(r["tests"]["f2p"], r["id"])
                self.assertTrue(set(r["tests"]["scope"]) == set(r["remove"]) == set(r["tests"]["hidden"]), r["id"])
                self.assertIn(r["level"], range(1, 6))
                self.assertTrue(r["graded"]["ok"], r["id"])
                self.assertEqual(editdata.leaks(r["issue"], r["gold_patch"], [] if mod is editbugs else r["tests"]["f2p"]),
                                 [], r["id"])
                if mod is editbugs:
                    self.assertFalse(any(t["path"] in r["issue"] for t in r["targets"]), r["id"])
                    self.assertNotIn("/work/repo", r["issue"])
                else:
                    self.assertEqual(r["issue"].count("### "), len(r["targets"]))
            mod.check_licences()
            self.assertEqual(mod.training_sources(), ("generated",))


@linux_only
class GraderTest(unittest.TestCase):
    """One level-1 task of each family in the real grader: the gold patch resolves it, the base doesn't."""

    def test_gold_resolves_and_base_fails(self):
        import asyncio

        from rlm_rl import editbox
        from tests.test_env import docker_ready

        reason = docker_ready()
        if reason:
            self.skipTest(reason)
        for mod, family in ((editspec, "edit_spec"), (editbugs, "edit_bugs")):
            recs = sorted((r for r in index_or_skip(mod) if r["split"] == "train"), key=lambda r: (r["level"], r["id"]))
            rec = recs[0]
            task = mod.to_task(rec, family)
            if subprocess.run(["docker", "image", "inspect", rec["image"]], capture_output=True).returncode:
                self.skipTest(f"{rec['image']} is not on this machine")
            ws = task.workspace.ensure()
            image = editbox.EditImage.derived(rec["image"])
            spec = grader.TestSpec.from_dict(task.grading["tests"])
            gold = asyncio.run(grader.grade(rec["gold_patch"], spec, ws_dir=ws, image=image, run_id="test-edit-v2"))
            self.assertTrue(gold.resolved, gold.as_dict())
            base = asyncio.run(grader.grade("", spec, ws_dir=ws, image=image, run_id="test-edit-v2", keep_all=True))
            self.assertFalse(base.resolved)
            self.assertEqual(base.f2p_passed, 0)


if __name__ == "__main__":
    unittest.main()
