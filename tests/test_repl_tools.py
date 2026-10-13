"""The edit REPL's coding tools (spec §12.1): behaviour, and the frozen API the Part 2 harness shares (H10)."""

import hashlib
import inspect
import os
import shutil
import subprocess
import tempfile
import time
import unittest

from rlm_rl import repl_tools as rt

HAS_GIT = shutil.which("git") is not None

# The trained API. Changing a name, signature or docstring changes what the model learned, so it
# needs a new TOOLS_VERSION and a new pin here.
FROZEN = {
    "read_file": "(path: str, start: int = 1, n: int = 40) -> str",
    "find_files": "(pattern: str, path: str = '.') -> list[str]",
    "grep": "(pattern: str, path: str = '.', glob: str | None = None, ignore_case: bool = False, "
            "files_only: bool = False) -> list[str]",
    "edit_file": "(path: str, old: str, new: str, replace_all: bool = False) -> str",
    "write_file": "(path: str, text: str) -> str",
    "shell": "(cmd: str, timeout: float = 120) -> rlm_rl.repl_tools.ShellResult",
    "diff": "() -> str",
}
FROZEN_SHA = {"t1": "816344133c226653"}


def api_fingerprint() -> str:
    parts = [f"{name}{inspect.signature(fn)}\n{inspect.getdoc(fn)}" for name, fn in rt.namespace().items()]
    return hashlib.sha256("\n\n".join(parts).encode()).hexdigest()[:16]


def git(root, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True,
                   capture_output=True)


class ToolCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        rt.configure(root=self.root)
        self.addCleanup(rt.configure)

    def put(self, rel, text, mode="w"):
        full = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, mode, **({} if "b" in mode else {"encoding": "utf-8", "newline": ""})) as f:
            f.write(text)

    def text(self, rel):
        with open(os.path.join(self.root, rel), encoding="utf-8", newline="") as f:
            return f.read()


class FrozenApiTest(unittest.TestCase):
    def test_names_and_signatures(self):
        self.assertEqual(tuple(rt.namespace()), rt.TOOL_NAMES)
        self.assertEqual({n: str(inspect.signature(f)) for n, f in rt.namespace().items()}, FROZEN)

    def test_docstrings_are_pinned_to_the_version(self):
        self.assertEqual(api_fingerprint(), FROZEN_SHA[rt.TOOLS_VERSION],
                         "the tool API changed: bump TOOLS_VERSION and pin the new fingerprint")

    def test_standard_library_only(self):
        source = inspect.getsource(rt)
        self.assertNotIn("rlm_rl", source.split('"""', 2)[2])  # no project imports after the docstring


class ReadFileTest(ToolCase):
    def test_numbered_window_and_next_hint(self):
        self.put("a.py", "".join(f"line {i}\n" for i in range(1, 101)))
        out = rt.read_file("a.py", start=11, n=5)
        self.assertEqual(out.splitlines()[0], "    11\tline 11")
        self.assertEqual(out.splitlines()[4], "    15\tline 15")
        self.assertIn("[85 more lines; next: read_file('a.py', start=16)]", out)
        self.assertNotIn("more lines", rt.read_file("a.py", start=61, n=40))

    def test_default_window_fits_one_printed_turn(self):
        self.put("a.py", "".join(f"    value_{i} = compute_something(value_{i - 1})\n" for i in range(200)))
        self.assertLess(len(rt.read_file("a.py")), 2000 + 200)

    def test_long_lines_are_cut(self):
        self.put("min.js", "x" * 1000 + "\n")
        self.assertIn("[... 700 more characters]", rt.read_file("min.js"))

    def test_refusals(self):
        self.put("b.bin", b"\x00\x01", mode="wb")
        self.put("short.txt", "one\n")
        with self.assertRaisesRegex(ValueError, "binary"):
            rt.read_file("b.bin")
        with self.assertRaisesRegex(ValueError, "past the end"):
            rt.read_file("short.txt", start=5)
        with self.assertRaises(FileNotFoundError):
            rt.read_file("missing.py")
        self.put("empty.txt", "")
        self.assertEqual(rt.read_file("empty.txt"), "[empty.txt is empty]")


class FindFilesTest(ToolCase):
    def setUp(self):
        super().setUp()
        for rel in ("setup.py", "pkg/core.py", "pkg/sub/deep.py", "web/app.ts", "web/view.tsx", "README.md",
                    "node_modules/dep/index.js", "pkg/__pycache__/core.cpython-312.pyc"):
            self.put(rel, "x\n")

    def test_glob_rules(self):
        self.assertEqual(rt.find_files("*.py"), ["setup.py"])
        self.assertEqual(rt.find_files("**/*.py"), ["pkg/core.py", "pkg/sub/deep.py", "setup.py"])
        self.assertEqual(rt.find_files("web/*.{ts,tsx}"), ["web/app.ts", "web/view.tsx"])
        self.assertEqual(rt.find_files("*.py", path="pkg"), ["pkg/core.py"])
        self.assertEqual(rt.find_files("**/*.js"), [], "dependency folders are skipped without git")

    @unittest.skipUnless(HAS_GIT, "needs git")
    def test_gitignore_is_respected_in_a_repository(self):
        git(self.root, "init", "-q")
        self.put(".gitignore", "build/\n")
        self.put("build/gen.py", "x\n")
        self.assertNotIn("build/gen.py", rt.find_files("**/*.py"))
        self.assertIn("pkg/core.py", rt.find_files("**/*.py"), "untracked files still count")


class GrepTest(ToolCase):
    def setUp(self):
        super().setUp()
        self.put("pkg/a.py", "def parse(x):\n    return Parse(x)\n")
        self.put("pkg/b.go", "func parse() {}\n")
        self.put("blob.bin", b"parse\x00", mode="wb")

    def test_matches_glob_case_and_files_only(self):
        self.assertEqual(rt.grep("parse"), ["pkg/a.py:1: def parse(x):", "pkg/b.go:1: func parse() {}"])
        self.assertEqual(rt.grep("parse", glob="*.py", ignore_case=True),
                         ["pkg/a.py:1: def parse(x):", "pkg/a.py:2:     return Parse(x)"])
        self.assertEqual(rt.grep("parse", glob="pkg/**/*.go", files_only=True), ["pkg/b.go"])
        self.assertEqual(rt.grep("parse", path="pkg/b.go"), ["pkg/b.go:1: func parse() {}"])

    def test_match_cap_is_explicit(self):
        self.put("many.txt", "hit\n" * (rt.GREP_MAX_MATCHES + 5))
        out = rt.grep("hit", path="many.txt")
        self.assertEqual(len(out), rt.GREP_MAX_MATCHES + 1)
        self.assertIn("stopped at 10,000 matches", out[-1])


class EditAndWriteTest(ToolCase):
    def test_unique_replacement(self):
        self.put("m.py", "a = 1\nb = 2\n")
        self.assertEqual(rt.edit_file("m.py", "b = 2", "b = 3"), "edited m.py: 1 replacement, first at line 2")
        self.assertEqual(self.text("m.py"), "a = 1\nb = 3\n")

    def test_failures_leave_the_file_untouched(self):
        self.put("m.py", "x = 1\nx = 1\n")
        with self.assertRaisesRegex(ValueError, "occurs 2 times"):
            rt.edit_file("m.py", "x = 1", "x = 2")
        with self.assertRaisesRegex(ValueError, "not found"):
            rt.edit_file("m.py", "y = 1", "y = 2")
        self.assertEqual(self.text("m.py"), "x = 1\nx = 1\n")
        self.assertIn("2 replacements", rt.edit_file("m.py", "x = 1", "x = 2", replace_all=True))

    def test_crlf_files_keep_their_line_endings(self):
        self.put("w.java", "int a;\r\nint b;\r\n")
        rt.edit_file("w.java", "int a;\nint b;", "int a;\nint c;")
        self.assertEqual(self.text("w.java"), "int a;\r\nint c;\r\n")

    def test_write_makes_folders(self):
        self.assertEqual(rt.write_file("new/dir/f.rs", "fn main() {}\n"), "wrote new/dir/f.rs (1 lines)")
        self.assertEqual(self.text("new/dir/f.rs"), "fn main() {}\n")


class ShellTest(ToolCase):
    def test_code_output_and_printed_tail(self):
        r = rt.shell("echo out; echo err >&2; exit 3")
        self.assertEqual((r.code, r.out), (3, "out\nerr\n"))
        long = rt.shell("python3 -c \"print('x' * 5000 + 'END')\"")
        self.assertTrue(str(long).startswith("exit 0\n[... 3,504 earlier characters]"))
        self.assertTrue(str(long).rstrip().endswith("END"))
        code, out = rt.shell("pwd")
        self.assertEqual(os.path.realpath(out.strip()), os.path.realpath(self.root))

    def test_timeout_kills_the_group_and_keeps_output(self):
        start = time.monotonic()
        r = rt.shell("echo started; sleep 30 & sleep 30", timeout=1)
        self.assertLess(time.monotonic() - start, 10)
        self.assertEqual(r.code, rt.SHELL_TIMEOUT_CODE)
        self.assertIn("started", r.out)
        self.assertIn("timed out after 1 s", r.out)

    def test_cell_time_left_caps_the_timeout(self):
        rt.configure(root=self.root, time_left=lambda: 3.0)
        start = time.monotonic()
        self.assertEqual(rt.shell("sleep 30", timeout=120).code, rt.SHELL_TIMEOUT_CODE)
        self.assertLess(time.monotonic() - start, 10)


@unittest.skipUnless(HAS_GIT, "needs git")
class DiffTest(ToolCase):
    def setUp(self):
        super().setUp()
        self.put("a.py", "x = 1\n")
        git(self.root, "init", "-q")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "base")

    def test_shows_edits_and_new_files_without_touching_the_index(self):
        self.assertEqual(rt.diff(), "")
        rt.edit_file("a.py", "x = 1", "x = 2")
        rt.write_file("b.py", "y = 1\n")
        d = rt.diff()
        self.assertIn("-x = 1\n+x = 2", d)
        self.assertIn("+++ b/b.py", d)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(status.stdout, " M a.py\n?? b.py\n", "the repository's own index is unchanged")
        self.assertNotIn("repl_tools_diff_index", os.listdir(os.path.join(self.root, ".git")))

    def test_a_commit_does_not_hide_the_change(self):
        rt.edit_file("a.py", "x = 1", "x = 2")
        git(self.root, "commit", "-qam", "mine")
        self.assertIn("+x = 2", rt.diff())


class LoggingTest(ToolCase):
    def test_every_call_is_reported(self):
        seen = []
        rt.configure(root=self.root, on_call=lambda *a: seen.append(a))
        self.put("a.py", "x\n")
        rt.read_file("a.py")
        with self.assertRaises(ValueError):
            rt.edit_file("a.py", "nope", "y")
        rt.shell("true")
        self.assertEqual([(t, ok, target) for t, ok, _, target in seen],
                         [("read_file", True, "a.py"), ("edit_file", False, "a.py"), ("shell", True, "true")])
        self.assertFalse(rt.active())

    def test_active_only_during_a_call(self):
        self.put("a.py", "x\n")
        during, real = [], rt._is_binary
        rt._is_binary = lambda path: during.append(rt.active()) or real(path)
        self.addCleanup(setattr, rt, "_is_binary", real)
        rt.read_file("a.py")
        self.assertEqual((during, rt.active()), ([True], False))


if __name__ == "__main__":
    unittest.main()
