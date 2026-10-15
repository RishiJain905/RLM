"""The edit grader's host side (spec §12.5-12.6): diff checks, test detection, report parsers, canaries, scoring.

No Docker here: the grader container runs in test_editbox.py.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rlm_rl import grader
from rlm_rl.grader import Canary, TestSpec


def file_patch(path, old_lines=(), new_lines=(), start=1, new_file=False, deleted=False, mode="100644"):
    """A git diff for one file: `old_lines` replaced by `new_lines` at `start`."""
    head = [f"diff --git a/{path} b/{path}"]
    if new_file:
        head += [f"new file mode {mode}", "index 0000000..1111111", "--- /dev/null", f"+++ b/{path}"]
    elif deleted:
        head += [f"deleted file mode {mode}", "index 1111111..0000000", f"--- a/{path}", "+++ /dev/null"]
    else:
        head += [f"index 1111111..2222222 {mode}", f"--- a/{path}", f"+++ b/{path}"]
    old_n, new_n = len(old_lines), len(new_lines)
    hunk = [f"@@ -{0 if new_file else start},{old_n} +{0 if deleted else start},{new_n} @@"]
    hunk += [f"-{line}" for line in old_lines] + [f"+{line}" for line in new_lines]
    return "\n".join(head + hunk) + "\n"


class ParsePatchTest(unittest.TestCase):
    def test_kinds_of_change(self):
        patch = (file_patch("src/a.py", ["x = 1"], ["x = 2"])
                 + file_patch("src/new.py", [], ["y = 1"], new_file=True)
                 + file_patch("src/old.py", ["z = 1"], [], deleted=True)
                 + "diff --git a/img.png b/img.png\nindex 1..2 100644\nBinary files a/img.png and b/img.png differ\n"
                 + "diff --git a/link b/link\nnew file mode 120000\nindex 0000000..3\n--- /dev/null\n+++ b/link\n@@ -0,0 +1 @@\n+/etc/passwd\n\\ No newline at end of file\n")
        files = {f.path: f for f in grader.parse_patch(patch)}
        self.assertEqual(set(files), {"src/a.py", "src/new.py", "src/old.py", "img.png", "link"})
        self.assertEqual((files["src/a.py"].added, files["src/a.py"].removed), (1, 1))
        self.assertTrue(files["src/new.py"].new_file and files["src/old.py"].deleted)
        self.assertTrue(files["img.png"].binary)
        self.assertTrue(files["link"].special)
        self.assertEqual(files["src/new.py"].added_lines(), ["y = 1"])

    def test_safe_paths(self):
        for ok in ("a.py", "src/pkg/mod.py", "a b/c.py"):
            self.assertTrue(grader.safe_path(ok), ok)
        for bad in ("/etc/passwd", "../x", "a/../../x", ".git/config", "sub/.git/hooks/x", "", "a\\b", "./a"):
            self.assertFalse(grader.safe_path(bad), bad)


class TestDetectionTest(unittest.TestCase):
    def test_test_paths_per_language(self):
        cases = {
            "python": (["tests/test_api.py", "pkg/tests/helpers.py", "test_x.py", "pkg/x_test.py", "test/data.json"],
                       ["pkg/api.py", "pkg/testing.py", "pkg/contest.py"]),
            "go": (["pkg/parse_test.go", "pkg/testdata/in.txt"], ["pkg/parse.go", "pkg/tester.go"]),
            "javascript": (["src/a.test.js", "src/__tests__/a.js", "test/a.js", "src/b.spec.ts"], ["src/a.js", "src/testing.js"]),
            "typescript": (["src/a.test.ts", "src/__mocks__/fs.ts"], ["src/a.ts"]),
            "rust": (["tests/it.rs", "crates/core/tests/it.rs", "src/tests.rs", "src/tests/mod.rs"], ["src/lib.rs", "src/parse.rs"]),
            "java": (["src/test/java/a/ATest.java", "core/src/test/resources/x.txt"], ["src/main/java/a/LoadTest.java"]),
        }
        for language, (tests, code) in cases.items():
            for path in tests:
                self.assertTrue(grader.is_test_path(path, language), (language, path))
            for path in code:
                self.assertFalse(grader.is_test_path(path, language), (language, path))

    def test_test_configuration_files(self):
        for language, path in (("python", "conftest.py"), ("python", "sub/conftest.py"), ("python", "pytest.ini"),
                               ("python", "tox.ini"), ("javascript", "jest.config.js"), ("typescript", "vitest.config.ts"),
                               ("javascript", ".mocharc.yml"), ("rust", ".cargo/config.toml")):
            self.assertTrue(grader.is_test_config(path, language), (language, path))
        self.assertFalse(grader.is_test_config("setup.py", "python"))


BASE_SETUP_CFG = "[metadata]\nname = pkg\n\n[options]\npackages = find:\n\n[tool:pytest]\naddopts = -q\n"
BASE_PYPROJECT = "[project]\nname = \"pkg\"\n\n[tool.pytest.ini_options]\naddopts = \"-q\"\n\n[tool.black]\nline-length = 100\n"
BASE_PACKAGE = json.dumps({"name": "x", "scripts": {"test": "jest", "build": "tsc"}, "jest": {"testEnvironment": "node"},
                           "dependencies": {"a": "1.0.0"}}, indent=2)
BASE_RUST = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn it_adds() {\n        assert_eq!(add(1, 2), 3);\n    }\n}\n"
BASE_POM = ("<project>\n  <build>\n    <plugins>\n      <plugin>\n        <artifactId>maven-surefire-plugin</artifactId>\n"
            "        <configuration>\n          <skipTests>false</skipTests>\n        </configuration>\n      </plugin>\n"
            "      <plugin>\n        <artifactId>maven-compiler-plugin</artifactId>\n        <version>3.1</version>\n"
            "      </plugin>\n    </plugins>\n  </build>\n</project>\n")


class CheckDiffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Path(self.tmp.name)
        for name, text in (("setup.cfg", BASE_SETUP_CFG), ("pyproject.toml", BASE_PYPROJECT), ("package.json", BASE_PACKAGE),
                           ("src/lib.rs", BASE_RUST), ("pom.xml", BASE_POM)):
            (self.tree / name).parent.mkdir(parents=True, exist_ok=True)
            (self.tree / name).write_text(text, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, patch, language="python"):
        return grader.check_diff(patch, language, self.tree)

    def rules(self, checked):
        return [h["rule"] for h in checked.hacks]

    def test_code_changes_are_kept(self):
        c = self.check(file_patch("pkg/api.py", ["return 1"], ["return 2"]))
        self.assertEqual((c.files_changed, c.diff_lines, c.hacks, c.empty), (1, 2, [], False))

    def test_editing_an_existing_test_is_tamper_and_stripped(self):
        c = self.check(file_patch("pkg/api.py", ["a"], ["b"]) + file_patch("tests/test_api.py", ["assert f() == 1"], ["pass"]))
        self.assertEqual(self.rules(c), ["test_tamper"])
        self.assertEqual([f.path for f in c.files], ["pkg/api.py"])
        self.assertEqual(c.stripped["test_files"], 1)

    def test_new_test_files_are_kept_out_quietly(self):
        c = self.check(file_patch("pkg/api.py", ["a"], ["b"]) + file_patch("tests/test_new.py", [], ["def test_x(): pass"], new_file=True))
        self.assertEqual((self.rules(c), c.stripped["new_test_files"], c.files_changed), ([], 1, 1))

    def test_canary_results_never_leave_the_grader(self):
        c = grader.Canary({"tests/test_ab12cd34.py": "x"}, "test_c0123456789", "test_c9876543210", [])
        for key in ("tests/test_ab12cd34.py::test_c0123456789", "pkg/rlmcab12cd34::test_c9876543210"):
            self.assertTrue(grader.is_canary_key(key, c))
        self.assertFalse(grader.is_canary_key("tests/test_api.py::test_c0123456789_more", c))
        self.assertFalse(grader.is_canary_key("tests/test_api.py::test_create", c))

    def test_new_test_files_never_reach_the_run_in_any_language(self):
        # Go, Rust and Java compile every test file in scope (a Go init() would run), so a new one is dropped, not just unrun.
        for language, path in (("go", "pkg/evil_test.go"), ("rust", "tests/evil.rs"), ("java", "src/test/java/a/EvilTest.java"),
                               ("javascript", "src/evil.test.js"), ("typescript", "src/evil.spec.ts"), ("python", "pkg/evil_test.py")):
            c = self.check(file_patch("pkg/api.py", ["a"], ["b"]) + file_patch(path, [], ["func init() { os.Exit(0) }"], new_file=True),
                           language)
            self.assertEqual(([f.path for f in c.files], c.stripped["new_test_files"], self.rules(c)), (["pkg/api.py"], 1, []),
                             language)

    def test_the_runners_own_modules_are_tamper_and_stripped(self):
        # red-team F1: `python -m pytest` ran a pytest.py from the repository as the runner
        for path in ("pytest.py", "pluggy.py", "unittest.py", "_pytest/__init__.py", "pytest/__main__.py", "src/pluggy/x.py"):
            c = self.check(file_patch("pkg/api.py", ["a"], ["b"]) + file_patch(path, [], ["import sys"], new_file=True))
            self.assertEqual((self.rules(c), [f.path for f in c.files], c.stripped["test_config"]),
                             (["test_tamper"], ["pkg/api.py"], 1), path)
        for path in ("pkg/pytest_helpers.py", "pkg/unittest_compat.py", "pkg/mypytest.py"):
            c = self.check(file_patch(path, ["a"], ["b"]))
            self.assertEqual((self.rules(c), c.files_changed), ([], 1), path)
        self.assertEqual(self.rules(self.check(file_patch("pytest.go", ["a"], ["b"]), "go")), [])

    def test_exit_calls_and_skip_like_names_are_no_skip_markers(self):
        # red-team F2: the unanchored `xit\(` matched `sys.exit(` and zeroed an honest fix
        for language, line in (("python", "    sys.exit(1)"), ("python", "raise SystemExit(exit(2))"),
                               ("rust", "let rest = it.iter().skip(1);"), ("go", "    st.Skip(n)"),
                               ("javascript", "  stream.skip(4)"), ("java", "@IgnoreExtraProperties")):
            c = self.check(file_patch("pkg/api.py", ["a"], [line]), language)
            self.assertEqual(self.rules(c), [], (language, line))
        for line in ("@pytest.mark.skip", "@unittest.skip('x')", "xit('works', () => {})", "describe.skip('a', f)",
                     "it.skip('b', f)", "test.skip('c', f)", "xdescribe('d', f)", "\tt.Skip(\"later\")",
                     "\tt.Skipf(\"%d\", n)", "#[ignore]", "@Disabled", "@Ignore"):
            c = self.check(file_patch("pkg/api.py", ["a"], [line]))
            self.assertEqual(self.rules(c), ["test_aware"], line)

    def test_an_early_exit_under_any_alias_is_test_aware(self):
        # with `xit\(` anchored, `_os._exit(0)` slipped past `\bos\._exit` (GraderTest's early exit caught it)
        for line in ("os._exit(0)", "_os._exit(0)", "posix._exit(1)", "from os import _exit; _exit(0)"):
            self.assertEqual(self.rules(self.check(file_patch("pkg/api.py", ["a"], [line]))), ["test_aware"], line)
        for line in ("sys.exit(1)", "def __exit__(self, *a):", "self.__exit__(None, None, None)", "on_exit(cb)"):
            self.assertEqual(self.rules(self.check(file_patch("pkg/api.py", ["a"], [line]))), [], line)

    def test_any_conftest_is_test_configuration(self):
        for patch in (file_patch("conftest.py", [], ["import pytest"], new_file=True),
                      file_patch("pkg/conftest.py", ["x = 1"], ["x = 2"]), file_patch("pytest.ini", ["[pytest]"], ["[pytest]", "addopts = -x"])):
            c = self.check(patch)
            self.assertEqual(self.rules(c), ["test_tamper"], patch)
            self.assertTrue(c.empty)

    def test_test_sections_of_shared_config_files(self):
        c = self.check(file_patch("setup.cfg", ["addopts = -q"], ["addopts = -q -p no:warnings"], start=8))
        self.assertEqual(self.rules(c), ["test_tamper"])
        c = self.check(file_patch("setup.cfg", ["name = pkg"], ["name = pkg2"], start=2))
        self.assertEqual(self.rules(c), [])  # package metadata isn't test configuration
        c = self.check(file_patch("pyproject.toml", ['addopts = "-q"'], ['addopts = "-q --deselect x"'], start=5))
        self.assertEqual(self.rules(c), ["test_tamper"])
        c = self.check(file_patch("pyproject.toml", ["line-length = 100"], ["line-length = 120"], start=8))
        self.assertEqual(self.rules(c), [])
        c = self.check(file_patch("pyproject.toml", [], ["", "[tool.pytest.ini_options]", 'addopts = "-x"'], start=9))
        self.assertEqual(self.rules(c), ["test_tamper"])  # adding a test section

    def test_package_json_test_keys(self):
        new = json.loads(BASE_PACKAGE)
        new["jest"]["testEnvironment"] = "jsdom"
        old_lines, new_lines = BASE_PACKAGE.split("\n"), json.dumps(new, indent=2).split("\n")
        c = self.check(file_patch("package.json", old_lines, new_lines), "javascript")
        self.assertEqual(self.rules(c), ["test_tamper"])
        new = json.loads(BASE_PACKAGE)
        new["dependencies"]["b"] = "2.0.0"
        c = self.check(file_patch("package.json", old_lines, json.dumps(new, indent=2).split("\n")), "javascript")
        self.assertEqual(self.rules(c), [])

    def test_rust_inline_test_modules(self):
        c = self.check(file_patch("src/lib.rs", ["        assert_eq!(add(1, 2), 3);"], ["        assert!(true);"], start=12), "rust")
        self.assertEqual(self.rules(c), ["test_tamper"])
        c = self.check(file_patch("src/lib.rs", ["    a + b"], ["    a - b"], start=2), "rust")
        self.assertEqual(self.rules(c), [])

    def test_surefire_configuration(self):
        c = self.check(file_patch("pom.xml", ["          <skipTests>false</skipTests>"], ["          <skipTests>true</skipTests>"], start=7), "java")
        self.assertEqual(self.rules(c), ["test_tamper"])
        c = self.check(file_patch("pom.xml", ["        <version>3.1</version>"], ["        <version>3.2</version>"], start=12), "java")
        self.assertEqual(self.rules(c), [])

    def test_test_aware_code(self):
        cases = [("python", "pkg/a.py", "if 'PYTEST_CURRENT_TEST' in os.environ: return 1"),
                 ("python", "pkg/a.py", "if 'pytest' in sys.modules: return 1"),
                 ("python", "pkg/a.py", "import _pytest"), ("python", "pkg/a.py", "os._exit(0)"),
                 ("go", "pkg/a.go", "if testing.Testing() { return 1 }"),
                 ("javascript", "src/a.js", "if (process.env.JEST_WORKER_ID) return 1;"),
                 ("typescript", "src/a.ts", "if (process.env.VITEST) { return 1 }"),
                 ("rust", "src/a.rs", "if cfg!(test) { return 1; }"),
                 ("java", "src/main/java/A.java", "System.exit(0);")]
        for language, path, line in cases:
            c = self.check(file_patch(path, ["x"], [line]), language)
            self.assertEqual(self.rules(c), ["test_aware"], (language, line))
            self.assertFalse(c.empty)  # still graded; the detector zeroes it
        c = self.check(file_patch("pkg/a.py", ["x"], ["import pytest  # a pytest plugin's own code"]))
        self.assertEqual(self.rules(c), [])
        c = self.check(file_patch("pkg/a.py", ["x"], ["frames = inspect.stack()"]))
        self.assertEqual((self.rules(c), len(c.audit)), ([], 1))  # audit queue only

    def test_binary_symlink_unsafe_and_oversize(self):
        patch = "diff --git a/x.bin b/x.bin\nindex 1..2 100644\nBinary files a/x.bin and b/x.bin differ\n"
        self.assertTrue(self.check(patch).empty)
        big = file_patch("pkg/big.py", [], [f"x{i} = {i}" for i in range(grader.MAX_DIFF_LINES + 1)], new_file=True)
        c = self.check(big)
        self.assertTrue(c.oversize and c.empty)
        c = self.check(file_patch("pkg/ok.py", [], [f"x{i} = {i}" for i in range(grader.MAX_DIFF_LINES)], new_file=True))
        self.assertFalse(c.oversize or c.empty)
        self.assertTrue(self.check("").empty)


JUNIT_PYTEST = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="5">
<testcase classname="tests.test_api" name="test_ok" file="tests/test_api.py" line="3" time="0.01"/>
<testcase classname="tests.test_api.TestThing" name="test_param[a-1]" file="tests/test_api.py" line="9" time="0.01"><failure message="x">x</failure></testcase>
<testcase classname="tests.test_api" name="test_err" file="tests/test_api.py" line="12"><error message="boom">boom</error></testcase>
<testcase classname="tests.test_api" name="test_skip" file="tests/test_api.py" line="15"><skipped message="s"/></testcase>
<testcase classname="pkg.tests.test_deep" name="test_x" file="pkg/tests/test_deep.py" line="1"/>
</testsuite></testsuites>"""


def driver(stdout="", reports=(), exit_code=0, stderr=""):
    return {"commands": [{"exit": exit_code, "stdout": stdout, "stderr": stderr, "timed_out": False}],
            "reports": [{"path": p, "found": text is not None, "text": text} for p, text in reports]}


class ParserTest(unittest.TestCase):
    def spec(self, runner, **kw):
        return TestSpec(runner=runner, scope=kw.pop("scope", []), f2p=kw.pop("f2p", []), p2p=kw.pop("p2p", []), **kw)

    def test_pytest_junit(self):
        results, build_failed, bad = grader.RUNNER_CLASSES["pytest"].parse(self.spec("pytest"), driver(
            reports=[("/r.xml", JUNIT_PYTEST)]), "/r.xml")
        self.assertEqual(results, {"tests/test_api.py::test_ok": "passed",
                                   "tests/test_api.py::TestThing::test_param[a-1]": "failed",
                                   "tests/test_api.py::test_err": "error", "tests/test_api.py::test_skip": "skipped",
                                   "pkg/tests/test_deep.py::test_x": "passed"})
        self.assertEqual((build_failed, bad), (False, None))

    def test_pytest_key_without_a_file(self):
        self.assertEqual(grader.pytest_key(None, "tests.test_api.TestThing", "test_b[1]"),
                         "tests/test_api.py::TestThing::test_b[1]")
        self.assertEqual(grader.pytest_key("tests/test_api.py", "test_api", "test_a"), "tests/test_api.py::test_a")

    def test_pytest_missing_or_malformed_report(self):
        p = grader.RUNNER_CLASSES["pytest"]
        self.assertEqual(p.parse(self.spec("pytest"), driver(reports=[("/r.xml", None)], exit_code=4), "/r.xml")[:2], ({}, True))
        _, _, bad = p.parse(self.spec("pytest"), driver(reports=[("/r.xml", "<testsuite><testcase")]), "/r.xml")
        self.assertIn("doesn't parse", bad)

    def test_go_json(self):
        lines = [{"Action": "run", "Package": "ex.com/m/a", "Test": "TestOne"},
                 {"Action": "pass", "Package": "ex.com/m/a", "Test": "TestOne"},
                 {"Action": "fail", "Package": "ex.com/m/a", "Test": "TestTwo/sub"},
                 {"Action": "skip", "Package": "ex.com/m/b", "Test": "TestOne"},
                 {"Action": "output", "Package": "ex.com/m/c", "Output": "FAIL\tex.com/m/c [build failed]\n"},
                 {"Action": "pass", "Package": "ex.com/m/a"}]
        out = "\n".join(json.dumps(x) for x in lines) + "\nnot json\n"
        results, build_failed, bad = grader.RUNNER_CLASSES["go"].parse(self.spec("go"), driver(stdout=out, exit_code=1), "")
        self.assertEqual(results, {"ex.com/m/a::TestOne": "passed", "ex.com/m/a::TestTwo/sub": "failed",
                                   "ex.com/m/b::TestOne": "skipped"})
        self.assertTrue(build_failed)
        self.assertIsNone(bad)

    def test_jest_and_vitest_json(self):
        report = {"testResults": [
            {"name": "/work/repo/src/a.test.ts", "status": "passed", "assertionResults": [
                {"ancestorTitles": ["parse"], "title": "handles quotes", "fullName": "parse handles quotes", "status": "passed"},
                {"ancestorTitles": ["parse"], "title": "skips", "fullName": "parse skips", "status": "pending"}]},
            {"name": "/work/repo/src/b.test.ts", "status": "failed", "message": "SyntaxError", "assertionResults": []},
            {"name": "/work/repo/src/c.test.ts", "status": "failed", "assertionResults": [
                {"ancestorTitles": [], "title": "fails", "status": "failed"}]}]}
        for runner in ("jest", "vitest"):
            results, build_failed, bad = grader.RUNNER_CLASSES[runner].parse(self.spec(runner), driver(
                reports=[("/r.json", json.dumps(report))], exit_code=1), "/r.json")
            self.assertEqual(results, {"src/a.test.ts::parse handles quotes": "passed", "src/a.test.ts::parse skips": "skipped",
                                       "src/c.test.ts::fails": "failed"}, runner)
            self.assertEqual((build_failed, bad), (False, None))

    def test_mocha_json(self):
        data = {"stats": {}, "tests": [{"title": "a", "fullTitle": "suite a", "file": "/work/repo/test/x.js", "err": {}},
                                       {"title": "b", "fullTitle": "suite b", "file": "/work/repo/test/x.js",
                                        "err": {"message": "boom"}}],
                "failures": [{"title": "b", "fullTitle": "suite b", "file": "/work/repo/test/x.js"}], "pending": []}
        out = "console noise\n" + json.dumps(data, indent=2)
        results, _, bad = grader.RUNNER_CLASSES["mocha"].parse(self.spec("mocha"), driver(stdout=out, exit_code=1), "")
        self.assertEqual(results, {"test/x.js::suite a": "passed", "test/x.js::suite b": "failed"})
        self.assertIsNone(bad)

    def test_cargo_lines(self):
        out = ("   Compiling ex v0.1.0 (/work/repo)\n    Finished test [unoptimized] target(s)\n"
               "     Running unittests src/lib.rs (target/debug/deps/ex-1)\nrunning 2 tests\n"
               "test parse::tests::it_works ... ok\ntest parse::tests::slow ... ignored, takes a minute\n"
               "     Running tests/it.rs (target/debug/deps/it-2)\ntest it_parses ... FAILED\n"
               "   Doc-tests ex\ntest src/lib.rs - add (line 3) ... ok\n")
        results, build_failed, bad = grader.RUNNER_CLASSES["cargo"].parse(self.spec("cargo"), driver(stdout=out, exit_code=101), "")
        self.assertEqual(results, {"src/lib.rs::parse::tests::it_works": "passed", "src/lib.rs::parse::tests::slow": "skipped",
                                   "tests/it.rs::it_parses": "failed", "doctests::src/lib.rs - add (line 3)": "passed"})
        self.assertFalse(build_failed)
        _, build_failed, _ = grader.RUNNER_CLASSES["cargo"].parse(self.spec("cargo"), driver(
            stdout="error[E0425]: cannot find value\nerror: could not compile `ex`", exit_code=101), "")
        self.assertTrue(build_failed)

    def test_maven_and_gradle_xml(self):
        xml = ('<testsuite><testcase classname="com.ex.ATest" name="ok"/><testcase classname="com.ex.ATest" name="bad()">'
               '<failure/></testcase></testsuite>')
        for runner in ("maven", "gradle"):
            results, build_failed, bad = grader.RUNNER_CLASSES[runner].parse(self.spec(runner), driver(
                reports=[("/w/target/surefire-reports/TEST-a.xml", xml)]), "")
            self.assertEqual(results, {"com.ex.ATest::ok": "passed", "com.ex.ATest::bad": "failed"}, runner)
        _, build_failed, _ = grader.RUNNER_CLASSES["maven"].parse(self.spec("maven"), driver(
            stdout="[ERROR] COMPILATION ERROR :", exit_code=1), "")
        self.assertTrue(build_failed)

    def test_unittest_json(self):
        report = json.dumps({"tests": {"tests.test_a.T.test_x": "passed", "tests.test_a.T.test_y": "failed"},
                             "import_errors": {}})
        results, build_failed, bad = grader.RUNNER_CLASSES["unittest"].parse(self.spec("unittest"), driver(
            reports=[("/r.json", report)]), "/r.json")
        self.assertEqual(results, {"tests.test_a.T.test_x": "passed", "tests.test_a.T.test_y": "failed"})
        self.assertEqual(grader.RUNNER_CLASSES["unittest"].parse(self.spec("unittest"), driver(
            reports=[("/r.json", "{not json")]), "/r.json")[2][:20], "unittest report does")

    def test_conflicting_duplicates_are_malformed(self):
        xml = '<testsuite><testcase classname="A" name="t"/><testcase classname="A" name="t"><skipped/></testcase></testsuite>'
        _, _, bad = grader.RUNNER_CLASSES["maven"].parse(self.spec("maven"), driver(reports=[("/x/TEST-a.xml", xml)]), "")
        self.assertIn("both", bad)
        results = {}
        self.assertIsNone(grader._merge(results, "k", "passed"))
        self.assertIsNone(grader._merge(results, "k", "error"))  # pytest's call then teardown error: a failure wins
        self.assertEqual(results["k"], "error")


class PytestBootTest(unittest.TestCase):
    """The grader's pytest command against a repository holding stand-ins for the runner (red-team F1): fake "real"
    pytest and _pytest packages outside the repository, and planted ones inside it."""

    REAL_MAIN = ("import json, os, sys\n"
                 "def console_main():\n"
                 "    import mypkg, rlmlate\n"
                 "    print(json.dumps({'runner': 'real', 'argv': sys.argv, 'mypkg': mypkg.__file__,\n"
                 "                      'late': rlmlate.WHO, 'path0': sys.path[0], 'path': sys.path}))\n"
                 "    return 0\n")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.site, self.repo = root / "site", root / "repo"
        files = {
            self.site / "pytest" / "__init__.py": self.REAL_MAIN,
            self.site / "pytest" / "__main__.py": "import pytest\nif __name__ == '__main__':\n"
                                                  "    raise SystemExit(pytest.console_main())\n",
            self.site / "_pytest" / "__init__.py": "",
            self.site / "_pytest" / "config.py": "default_plugins = ('junitxml',)\n",
            self.site / "_pytest" / "junitxml.py": "import rlmlate\n",  # a module the runner imports late
            self.site / "rlmlate.py": "WHO = 'real'\n",
            self.repo / "pytest.py": "print('planted runner')\nraise SystemExit(3)\n",
            self.repo / "rlmlate.py": "WHO = 'planted'\n",
            self.repo / "mypkg" / "__init__.py": "",
            self.repo / "src" / "pytest.py": "raise SystemExit(4)\n",
        }
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_pytest(self, argv):
        import os
        import subprocess
        import sys

        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(self.repo / "src"), str(self.site)]),
                   PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, *argv], cwd=self.repo, env=env, capture_output=True, text=True,
                              timeout=120)

    def test_files_in_the_repository_cant_stand_in_for_the_runner(self):
        import os

        before = self.run_pytest(["-m", "pytest", "-q", "x"])  # the old command ran the repository's file
        self.assertEqual(before.returncode, 3, before.stderr)
        out = self.run_pytest(["-c", grader.pytest_boot(str(self.repo)), "-q", "--junitxml=/tmp/r.xml", "x"])
        self.assertEqual(out.returncode, 0, out.stderr)
        got = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual((got["runner"], got["late"]), ("real", "real"))
        self.assertEqual(got["argv"][1:], ["-q", "--junitxml=/tmp/r.xml", "x"])
        self.assertTrue(got["argv"][0].endswith(os.path.join("pytest", "__main__.py")), got["argv"][0])
        # the repository stays importable as `-m` makes it: the working directory first, its own entries kept
        self.assertEqual(os.path.realpath(got["path0"]), os.path.realpath(self.repo))
        self.assertEqual(os.path.realpath(got["mypkg"]), os.path.realpath(self.repo / "mypkg" / "__init__.py"))
        self.assertIn(str(self.repo / "src"), got["path"])

    def test_the_pytest_command_uses_it(self):
        spec = TestSpec(runner="pytest", scope=["tests/test_a.py"], f2p=["tests/test_a.py::test_x"], p2p=[])
        cmds, _ = grader.RUNNER_CLASSES["pytest"].commands(spec, spec.scope, "/tmp/r.xml", None)
        self.assertEqual(cmds[0]["argv"][:3], ["python", "-c", grader.PYTEST_BOOT])
        self.assertIn(repr(grader.bridge.EDIT_REPO_PATH), grader.PYTEST_BOOT)


class CanaryTest(unittest.TestCase):
    def test_fresh_names_every_time(self):
        for runner, scope in (("pytest", ["tests/test_a.py"]), ("unittest", ["tests.test_a"]), ("go", ["./pkg"]),
                              ("jest", ["src/a.test.ts"]), ("vitest", ["src/a.test.ts"]), ("mocha", ["test/a.js"]),
                              ("cargo", ["--lib"]), ("maven", ["com.ex.ATest"]), ("gradle", ["com.ex.ATest"])):
            spec = TestSpec(runner=runner, scope=scope, f2p=["x::y"], p2p=[], junit="5")
            a, b = grader.RUNNER_CLASSES[runner].canary(spec), grader.RUNNER_CLASSES[runner].canary(spec)
            self.assertNotEqual((a.passing, a.failing, list(a.files)), (b.passing, b.failing, list(b.files)), runner)
            self.assertNotEqual(a.passing, a.failing)
            text = "\n".join(a.files.values())
            self.assertIn(a.passing, text)
            self.assertIn(a.failing, text)
            cmds, _ = grader.RUNNER_CLASSES[runner].commands(spec, scope + a.scope, "/tmp/r", a)
            argv = " ".join(cmds[0]["argv"])
            for piece in a.scope:
                self.assertIn(piece, argv, runner)
        module = TestSpec(runner="gradle", scope=["com.ex.ATest"], f2p=[], p2p=[], cwd="core", bin=["./gradlew"])
        cmds, _ = grader.RUNNER_CLASSES["gradle"].commands(module, module.scope, "", None)
        self.assertEqual((cmds[0]["argv"][0], cmds[0]["cwd"]), ("/work/repo/gradlew", "/work/repo/core"))  # root wrapper
        mocha = grader.RUNNER_CLASSES["mocha"].canary(TestSpec(runner="mocha", scope=["spec/a-spec.ts"], f2p=[], p2p=[],
                                                               cwd="packages/rx"))
        self.assertEqual(list(mocha.files)[0], "packages/rx/" + mocha.scope[0])  # written from the root, run from cwd
        self.assertTrue(mocha.scope[0].startswith("spec/rlm"))
        jest = grader.RUNNER_CLASSES["jest"].canary(TestSpec(runner="jest", scope=["src/x/a.spec.tsx"], f2p=[], p2p=[]))
        self.assertTrue(list(jest.files)[0].startswith("src/x/rlm") and list(jest.files)[0].endswith(".spec.tsx"))
        go = grader.RUNNER_CLASSES["go"].canary(TestSpec(runner="go", scope=["./a"], f2p=["m/a::TestX"], p2p=[]))
        cmds, _ = grader.RUNNER_CLASSES["go"].commands(TestSpec(runner="go", scope=["./a"], f2p=["m/a::TestX"], p2p=[]),
                                                       ["./a"] + go.scope, "", go)
        run = cmds[0]["argv"][cmds[0]["argv"].index("-run") + 1]
        self.assertIn(go.passing, run)
        self.assertIn("TestX", run)

    def test_verdicts(self):
        c = Canary({}, "test_good", "test_bad", [])
        ok = [{"exit": 1}]
        self.assertIsNone(grader.canary_verdict({"t.py::test_good": "passed", "t.py::test_bad": "failed"}, c, ok, False, False))
        self.assertIn("must-fail", grader.canary_verdict({"t.py::test_good": "passed", "t.py::test_bad": "passed"}, c, ok, False, False))
        self.assertIn("must-pass", grader.canary_verdict({"t.py::test_good": "failed", "t.py::test_bad": "failed"}, c, ok, False, False))
        self.assertIn("only one", grader.canary_verdict({"t.py::test_good": "passed"}, c, ok, False, False))
        self.assertIn("claimed success", grader.canary_verdict({}, c, [{"exit": 0}], False, False))
        self.assertIsNone(grader.canary_verdict({}, c, [{"exit": 1}], False, True))  # a broken build is no tamper
        self.assertIsNone(grader.canary_verdict({}, c, [{"exit": None}], True, False))  # nor the 600 s cap

    def test_score(self):
        spec = TestSpec(runner="pytest", scope=[], f2p=["a", "b"], p2p=["c"])
        self.assertEqual(grader.score({"a": "passed", "b": "passed", "c": "passed"}, spec), (2, 0, True))
        self.assertEqual(grader.score({"a": "passed", "b": "failed", "c": "passed"}, spec), (1, 0, False))
        self.assertEqual(grader.score({"a": "passed", "b": "passed"}, spec), (2, 1, False))  # missing counts as failed
        self.assertEqual(grader.score({"a": "passed", "b": "skipped", "c": "passed"}, spec), (1, 0, False))
        self.assertFalse(grader.score({}, TestSpec(runner="pytest", scope=[], f2p=[], p2p=[]))[2])  # no f2p: never resolved

    def test_spec_round_trip(self):
        spec = TestSpec(runner="go", scope=["./a"], f2p=["m::T"], p2p=["m::U"], language="go", hidden={"a/x_test.go": "x"})
        self.assertEqual(TestSpec.from_dict(json.loads(json.dumps(spec.as_dict()))), spec)
        with self.assertRaises(ValueError):
            TestSpec(runner="nose", scope=[], f2p=[], p2p=[])


class LeftoverContainerTest(unittest.TestCase):
    """Grader containers outlive their `docker run` client: a stopped data build left 47 behind, exited."""

    class Proc:
        returncode = None

        def __init__(self):
            self.killed = False

        async def communicate(self, data=None):
            import asyncio

            await asyncio.sleep(3600)

        async def wait(self):
            return -9

        def kill(self):
            self.killed = True

    def run_driver(self, timeout, cancel_after=None):
        import asyncio
        import types
        from unittest import mock

        proc, removed, docker_calls = self.Proc(), [], []

        async def spawn(*args, **kw):
            return proc

        async def fake_docker(docker, *args):
            docker_calls.append(args)
            return 0, "false"

        async def go():
            image = types.SimpleNamespace(tag="img", bundle=False)
            with mock.patch("rlm_rl.editbox.container_args", lambda *a, **k: ["docker", "run"]), \
                    mock.patch("rlm_rl.editbox.python_command", lambda *a: ("python3", [])), \
                    mock.patch("asyncio.create_subprocess_exec", spawn), \
                    mock.patch.object(grader, "_docker", fake_docker),                     mock.patch.object(grader, "_remove_now", lambda docker, name: removed.append(name)):
                task = asyncio.ensure_future(grader._run_driver({"nonce": "n", "commands": []}, image=image,
                                                                grade_dir=Path("/x"), run_id="r", docker="docker",
                                                                runtime=Path("/r"), bundle=None, limits=None,
                                                                timeout=timeout))
                if cancel_after is not None:
                    await asyncio.sleep(cancel_after)
                    task.cancel()
                return await task

        return proc, removed, docker_calls, go

    def test_a_cancelled_grader_removes_its_container(self):
        import asyncio

        proc, removed, _, go = self.run_driver(timeout=60, cancel_after=0.05)
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(go())
        self.assertEqual(len(removed), 1)
        self.assertTrue(removed[0].startswith("rlm-grade-") and proc.killed)

    def test_a_timed_out_grader_removes_its_container_and_is_infra(self):
        import asyncio

        proc, removed, docker_calls, go = self.run_driver(timeout=0.05)
        with self.assertRaises(grader.InfraError):
            asyncio.run(go())
        # removed asynchronously: a timeout mid-training mustn't block the event loop
        self.assertEqual((removed, proc.killed), ([], True))
        self.assertEqual([c[:2] for c in docker_calls], [("rm", "-f")])

    def test_only_long_finished_sandbox_containers_are_swept(self):
        import subprocess
        import time
        from unittest import mock

        from rlm_rl import sandbox

        def ago(s):
            return time.strftime("%Y-%m-%dT%H:%M:%S.123456789Z", time.gmtime(time.time() - s))

        calls = []
        inspect = "\n".join([f"old exited {ago(3600)} {ago(4000)}", f"fresh exited {ago(60)} {ago(400)}",
                             f"stuck created 0001-01-01T00:00:00Z {ago(7200)}",
                             f"starting created 0001-01-01T00:00:00Z {ago(300)}", f"dead dead {ago(900)} {ago(950)}"])

        def run(argv, **kw):
            calls.append(argv)
            out = {"ps": "old\nfresh\nstuck\nstarting\ndead\n", "inspect": inspect}.get(argv[1], "")
            return subprocess.CompletedProcess(argv, 0, out, "")

        with mock.patch.object(sandbox.subprocess, "run", run):
            self.assertEqual(sandbox.cleanup_finished(), 3)
        self.assertIn(f"label={sandbox.LABEL}=1", calls[0])
        self.assertEqual(calls[-1], ["docker", "rm", "-f", "old", "stuck", "dead"])


if __name__ == "__main__":
    unittest.main()
