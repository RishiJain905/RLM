"""The edit_real data builder's pure parts (plan step E4): licences, derived images, feature removal, mutations, leaks."""

import json
import random
import sys
import unittest

from rlm_rl import editdata, editfeatures, grader
from rlm_rl.tasks import editreal

PY = '''"""mod"""


def public_one(xs, n=2):
    """Return the first n even numbers of xs, doubled."""
    out = []
    for x in xs:
        if x % 2 == 0:
            out.append(x * 2)
    return out[:n]


def _private(x):
    return x + 1
    return x
    return x


class Thing:
    def method_two(self, a, b):
        total = a + b
        if total > 10:
            total -= 10
        return total
'''

GO = '''package demo

import (
	"fmt"
	"strings"
)

// Join joins parts with sep, skipping empty ones.
func Join(parts []string, sep string) string {
	keep := []string{}
	for _, p := range parts {
		if p != "" {
			keep = append(keep, p)
		}
	}
	return strings.Join(keep, sep)
}

func Show(x int) string {
	return fmt.Sprint(x)
}
'''

RUST = '''pub fn clamp_add(a: i32, b: i32) -> i32 {
    let s = a + b;
    if s > 100 {
        return 100;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn it_works() {
        let x = clamp_add(1, 2);
        assert_eq!(x, 3);
        assert!(x < 4);
    }
}
'''

JAVA = '''package a;

public class Util {
    /** Counts the vowels. */
    public static int countVowels(String s) {
        int n = 0;
        for (char c : s.toCharArray()) {
            if ("aeiou".indexOf(c) >= 0) {
                n++;
            }
        }
        return n;
    }
}
'''

TS = '''export function sumPositive(xs: number[]): number {
  let total = 0;
  for (const x of xs) {
    if (x > 0) {
      total += x;
    }
  }
  return total;
}
'''


class LicenceTest(unittest.TestCase):
    def test_classify(self):
        mit = "Permission is hereby granted, free of charge, to any person obtaining a copy"
        apache = "Apache License\nVersion 2.0, January 2004"
        bsd3 = ("Redistribution and use in source and binary forms, with or without modification ... Neither the name "
                "of the copyright holder")
        cases = {"MIT": {"LICENSE": mit}, "Apache-2.0": {"LICENSE": apache}, "BSD-3-Clause": {"LICENSE": bsd3},
                 "BSD-2-Clause": {"LICENSE": "Redistribution and use in source and binary forms, with or without"},
                 "MIT OR Apache-2.0": {"LICENSE-MIT": mit, "LICENSE-APACHE": apache},
                 "GPL": {"COPYING": "GNU GENERAL PUBLIC LICENSE Version 3"},
                 "LGPL": {"LICENSE": "GNU LESSER GENERAL PUBLIC LICENSE"},
                 "MPL-2.0": {"LICENSE": "Mozilla Public License Version 2.0"},
                 "unknown": {"LICENSE": "All rights reserved."}}
        for want, texts in cases.items():
            self.assertEqual(editdata.classify_licence(texts), want, want)
        for licence in ("MIT", "Apache-2.0", "BSD-3-Clause", "MIT OR Apache-2.0", "ISC"):
            self.assertTrue(editdata.permissive(licence), licence)
        for licence in ("GPL", "LGPL", "MPL-2.0", "AGPL-3.0", "unknown", "source-available"):
            self.assertFalse(editdata.permissive(licence), licence)

    def test_repo_key(self):
        self.assertEqual(editdata.repo_key("swebench/swesmith.x86_64.go-chi_1776_chi.23c395f8"), "go-chi__chi.23c395f8")
        self.assertEqual(editdata.repo_key("swebench/swesmith.x86_64.pdfminer_1776_pdfminer.six.1a8bd2f7:latest"),
                         "pdfminer__pdfminer.six.1a8bd2f7")


class DeriveTest(unittest.TestCase):
    def test_the_derived_image_drops_the_checkout_and_stays_offline(self):
        for language in editdata.LANGS:
            text = editdata.derive_dockerfile("src:img", language, 1000, 1000, path="/opt/env/bin:/usr/bin")
            self.assertIn("rm -rf /testbed", text)
            self.assertIn("ln -s /work/repo /testbed", text)
            self.assertIn("chown -R 1000:1000", text)
            self.assertIn('LABEL rlm.edit.uid="1000"', text)
            self.assertIn(f'LABEL rlm.edit.language="{language}"', text)
            self.assertIn('ENV HOME="/scratch"', text)
            self.assertIn("/opt/env/bin", text)
        go = editdata.derive_dockerfile("src:img", "go", 1000, 1000)
        for flag in ('GOFLAGS="-mod=mod"', 'GOPROXY="off"', 'GOCACHE="/opt/rlm-cache/go-build"'):
            self.assertIn(flag, go)
        rust = editdata.derive_dockerfile("s", "rust", 1, 1)
        self.assertIn('CARGO_NET_OFFLINE="true"', rust)
        self.assertLess(rust.index("mv /testbed/target /opt/rlm-cache/target"), rust.index("rm -rf /testbed"))
        self.assertIn('LABEL rlm.edit.writable="/opt/rlm-cache"', go)

    def test_go_images_keep_nothing_compiled_from_the_repository(self):
        # red-team F3: the warm build cache held the repository's own pre-break packages and hidden-test mains
        for go in (editdata.derive_dockerfile("src:img", "go", 1000, 1000),
                   editdata.derive_dockerfile("src:img", "go", 1000, 1000, repo_dir="/src/app",
                                              go={"gomodcache": "/go/pkg/mod", "gocache": "/root/.cache/go-build"})):
            repo = "/src/app" if "/src/app" in go else "/testbed"
            module, purge = editdata.go_cache_purge(repo)
            self.assertLess(go.index(module), go.index(f"rm -rf {repo} "))
            self.assertLess(go.index("mv "), go.index(purge))
            self.assertLess(go.index(purge), go.index("chown -R"))
        self.assertNotIn("go-build", editdata.derive_dockerfile("src:img", "python", 1000, 1000))

    @unittest.skipUnless(sys.platform == "linux", "GNU grep and xargs, as in the Go images")
    def test_the_go_cache_purge_keeps_dependencies_warm(self):
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            repo, cache = Path(tmp) / "testbed", Path(tmp) / "go-build"
            (repo).mkdir()
            (repo / "go.mod").write_text('module github.com/Masterminds/squirrel // a comment\n\ngo 1.21\n'
                                         "require github.com/lann/builder v0.0.0\n", encoding="utf-8")
            entries = {
                "1a/1a2b-d": b"!<arch>\n__.PKGDEF\x00github.com/Masterminds/squirrel.Select\x00",  # the package itself
                "2c/2c3d-d": b"\x7fELF\x00" + f"{repo}/select.go\x00".encode(),  # compiled from the checkout
                "3e/3e4f-d": b"!<arch>\n\x00github.com/lann/builder.Set\x00/root/go/pkg/mod/github.com/lann/b.go",
                "4a/4a5b-a": b"v1 4a5b 3e4f 120 1700000000\n",  # an action entry: hashes only
                "trim.txt": b"1700000000",
            }
            for name, body in entries.items():
                (cache / name).parent.mkdir(parents=True, exist_ok=True)
                (cache / name).write_bytes(body)
            module, purge = editdata.go_cache_purge(str(repo), str(cache))
            subprocess.run(["sh", "-c", f"set -e; {module}; {purge}"], check=True, timeout=60)
            left = sorted(str(p.relative_to(cache)).replace("\\", "/") for p in cache.rglob("*") if p.is_file())
        self.assertEqual(left, ["3e/3e4f-d", "4a/4a5b-a", "trim.txt"])

    @unittest.skipUnless(sys.platform == "linux", "GNU grep and xargs, as in the Go images")
    def test_without_a_go_mod_only_the_checkouts_paths_go(self):
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "go-build"
            cache.mkdir()
            (cache / "a-d").write_bytes(b"/testbed/x.go")
            (cache / "b-d").write_bytes(b"github.com/dep/x")
            module, purge = editdata.go_cache_purge(f"{tmp}/none", str(cache))
            purge = purge.replace(f"{tmp}/none/", "/testbed/")
            subprocess.run(["sh", "-c", f"set -e; {module}; {purge}"], check=True, timeout=60)
            self.assertEqual(sorted(p.name for p in cache.iterdir()), ["b-d"])

    def test_a_module_its_runner_cant_build_is_dropped_before_any_container(self):
        import asyncio
        import tempfile
        import types
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "tree" / "codegen").mkdir(parents=True)
            (Path(tmp) / "tree" / "codegen" / "build.gradle").write_text("", encoding="utf-8")
            task = types.SimpleNamespace(workspace=types.SimpleNamespace(ensure=lambda: Path(tmp)))
            record = {"tests": {"runner": "maven", "scope": ["a.BTest"], "cwd": "codegen", "language": "java"}}
            with mock.patch.object(editdata.editreal, "to_task", lambda rec, family: task), \
                    mock.patch.object(grader, "grade", side_effect=AssertionError("no container")):
                v = asyncio.run(editdata.validate(record, image=None))
        self.assertFalse(v.ok)
        self.assertIn("no pom.xml in codegen", v.why)

    def test_node_modules_stay_where_the_image_installed_them(self):
        text = editdata.derive_dockerfile("src:img", "typescript", 1000, 1000, deps=["node_modules", "pkg/a/node_modules"])
        self.assertIn("rm -rf /testbed/.git", text)
        self.assertIn("find /testbed -mindepth 1 \\( -type d -name node_modules -prune \\) -o \\( ! -type d -exec rm -f {} + \\)",
                      text)  # every checkout file goes, node_modules stay
        self.assertIn("ln -s /testbed /opt/rlm-deps", text)
        self.assertIn("chown 1000:1000 /testbed/node_modules/.cache /testbed/node_modules/.vite", text)  # caches only
        self.assertNotIn("mv /testbed", text)  # no copy of node_modules
        self.assertNotIn("ln -s /work/repo /testbed", text)
        self.assertNotIn("/opt/rlm-deps", text.split("chown -R 1000:1000")[1].split("\n")[0])  # never re-owned (a copy)

    def test_a_real_pr_image_keeps_its_go_cache_readable_and_moves_unwritable_caches(self):
        text = editdata.derive_dockerfile("rb:img", "go", 1000, 1000, repo_dir="/kin-openapi",
                                          go={"gomodcache": "/root/go/pkg/mod", "gocache": "/root/.cache/go-build"},
                                          extra_env={"XDG_CACHE_HOME": "/scratch/.cache/xdg_cache_home"})
        self.assertIn("rm -rf /kin-openapi", text)
        self.assertIn("ln -s /work/repo /kin-openapi", text)
        self.assertIn("chmod o+rx /root /root/go /root/go/pkg /root/go/pkg/mod", text)  # folders only: no copy-up
        self.assertIn('mv "/root/.cache/go-build" /opt/rlm-cache/go-build', text)
        self.assertIn('ENV GOMODCACHE="/root/go/pkg/mod"', text)
        self.assertIn('ENV XDG_CACHE_HOME="/scratch/.cache/xdg_cache_home"', text)
        self.assertNotIn("chown -R 1000:1000 /root", text)


class FeatureTest(unittest.TestCase):
    def test_python_functions_and_stub(self):
        fns = {f.qualname: f for f in editfeatures.python_functions("m.py", PY)}
        self.assertEqual(set(fns), {"public_one", "Thing.method_two"})
        stubbed = editfeatures.stub(PY, fns["public_one"], "python")
        self.assertIn('"""Return the first n even numbers of xs, doubled."""\n    raise NotImplementedError\n', stubbed)
        self.assertNotIn("out.append", stubbed)
        compile(stubbed, "m.py", "exec")
        self.assertEqual(fns["public_one"].doc, "Return the first n even numbers of xs, doubled.")

    def test_go_stub_drops_imports_it_leaves_unused(self):
        fns = {f.name: f for f in editfeatures.brace_functions("d.go", GO, "go")}
        self.assertEqual(set(fns), {"Join"})  # Show's body is one line: too short
        stubbed = editfeatures.stub(GO, fns["Join"], "go")
        self.assertIn('\tpanic("not implemented")\n}', stubbed)
        self.assertNotIn('"strings"', stubbed)
        self.assertIn('"fmt"', stubbed)
        self.assertIn("// Join joins parts", fns["Join"].doc)

    def test_rust_java_typescript(self):
        rust = editfeatures.brace_functions("src/lib.rs", RUST, "rust")
        self.assertEqual([f.name for f in rust], ["clamp_add"])  # the test module's functions are left alone
        self.assertIn("unimplemented!()", editfeatures.stub(RUST, rust[0], "rust"))
        java = editfeatures.brace_functions("src/main/java/a/Util.java", JAVA, "java")
        self.assertEqual([f.name for f in java], ["countVowels"])
        self.assertIn('throw new UnsupportedOperationException("not implemented");', editfeatures.stub(JAVA, java[0], "java"))
        ts = editfeatures.brace_functions("src/sum.ts", TS, "typescript")
        self.assertEqual([f.name for f in ts], ["sumPositive"])
        stubbed = editfeatures.stub(TS, ts[0], "typescript")
        self.assertIn('throw new Error("not implemented");', stubbed)
        self.assertTrue(stubbed.rstrip().endswith("}"))

    def test_go_hides_test_functions_not_files(self):
        test_file = ('package demo\n\nimport (\n\t"strings"\n\t"testing"\n)\n\n'
                     'func helper(t *testing.T) string { return "x" }\n\n'
                     '// TestJoin checks Join.\nfunc TestJoin(t *testing.T) {\n\tif Join([]string{"a"}, ",") != strings.ToUpper("a") {\n\t\tt.Fatal(helper(t))\n\t}\n}\n\n'
                     'func TestOther(t *testing.T) {\n\t_ = helper(t)\n}\n')
        stripped = editfeatures.strip_go_tests(test_file, mentions={"Join"})
        self.assertNotIn("TestJoin", stripped)
        self.assertNotIn("checks Join", stripped)  # its doc comment goes too
        self.assertIn("func helper", stripped)  # helpers stay for the other tests
        self.assertIn("func TestOther", stripped)
        self.assertNotIn('"strings"', stripped)  # an import only the hidden test used
        self.assertIn('"testing"', stripped)
        self.assertEqual(editfeatures.strip_go_tests(test_file, names={"TestOther"}).count("func Test"), 1)
        self.assertIsNone(editfeatures.strip_go_tests(test_file, mentions={"Nothing"}))

    def test_brace_scanner_skips_strings_and_comments(self):
        text = 'func F() {\n\ts := "}{"\n\t// } comment\n\tr := `}`\n\tc := \'}\'\n\treturn\n}\nafter'
        end = editfeatures._scan_braces(text, text.index("{"), "go")
        self.assertEqual(text[end:], "\nafter")

    def test_mutations_change_one_line(self):
        fn = editfeatures.brace_functions("src/sum.ts", TS, "typescript")[0]
        muts = editfeatures.mutations(TS, fn, "typescript", random.Random(1), limit=5)
        self.assertTrue(muts)
        for text, kind, line in muts:
            diff = [i for i, (a, b) in enumerate(zip(TS.split("\n"), text.split("\n"))) if a != b]
            self.assertEqual(diff, [line])
            self.assertTrue(fn.body_start <= line < fn.body_end)


class LeakAndLevelTest(unittest.TestCase):
    def test_leaks(self):
        gold = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-    return a\n"
                "+    return sorted(items, key=lambda item: item.priority)\n")
        self.assertEqual(editdata.leaks("Sorting is wrong.", gold, ["tests/test_x.py::test_sorts_by_priority"]), [])
        self.assertTrue(editdata.leaks("Use return sorted(items, key=lambda item: item.priority) here", gold, []))
        self.assertTrue(editdata.leaks("test_sorts_by_priority fails", gold, ["tests/test_x.py::test_sorts_by_priority"]))

    def test_levels(self):
        self.assertEqual(editreal.level_of(1, 8), 1)
        self.assertEqual(editreal.level_of(1, 30), 2)
        self.assertEqual(editreal.level_of(2, 30), 3)
        self.assertEqual(editreal.level_of(4, 60), 4)
        self.assertEqual(editreal.level_of(3, 150), 5)

    def test_rust_removed_files_named_by_what_is_left(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            for rel, text in {"Cargo.toml": '[package]\nname = "x"\n\n[[test]]\nname = "special"\npath = "tests/special.rs"\n',
                              "tests/a.rs": "mod common;\n#[test] fn a() { common::help(); }\n",
                              "tests/b.rs": "#[test] fn b() {}\n", "src/lib.rs": "pub fn f() {}\n"}.items():
                (tree / rel).parent.mkdir(parents=True, exist_ok=True)
                (tree / rel).write_text(text, encoding="utf-8")
            self.assertEqual(editdata.rust_hidden_named(tree, ["tests/b.rs"]), [])  # its own crate: nothing breaks
            self.assertEqual(len(editdata.rust_hidden_named(tree, ["tests/common/mod.rs"])), 1)  # `mod common;`
            self.assertEqual(len(editdata.rust_hidden_named(tree, ["tests/special.rs"])), 1)  # a Cargo.toml path

    def test_patches_without_git_headers_get_them(self):
        bare = "--- a/src/a.ts\n+++ b/src/a.ts\n@@ -1,2 +1,2 @@\n x\n-y\n+z\n--- /dev/null\n+++ b/src/n.ts\n@@ -0,0 +1 @@\n+w\n"
        fixed = editdata.with_git_headers(bare)
        self.assertEqual(editdata.patch_size(bare), (0, 0))
        self.assertEqual(editdata.patch_size(fixed), (2, 3))
        self.assertIn("diff --git a/src/n.ts b/src/n.ts\nnew file mode 100644\n", fixed)
        self.assertEqual(editdata.with_git_headers(fixed), fixed)

    def test_patch_size(self):
        patch = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x\n+y\n z\n"
                 "diff --git a/b.bin b/b.bin\nindex 1..2 100644\nBinary files a/b.bin and b/b.bin differ\n")
        self.assertEqual(editdata.patch_size(patch), (1, 2))


class EvalSplitTest(unittest.TestCase):
    def test_balance_spreads_languages_and_keeps_enough_features(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "build_edit_data", Path(__file__).resolve().parents[1] / "scripts" / "build_edit_data.py")
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        records = [{"id": f"{lang}-{kind}-{i}", "language": lang, "kind": kind}
                   for lang, n_bug, n_feat in (("python", 20, 3), ("go", 10, 8), ("rust", 4, 2), ("java", 2, 6))
                   for kind, n in (("bug", n_bug), ("feature", n_feat)) for i in range(n)]
        picked = build.balance(records, 40, 15)
        self.assertEqual(len(picked), 40)
        self.assertEqual(len({r["id"] for r in picked}), 40)
        self.assertGreaterEqual(sum(r["kind"] == "feature" for r in picked), 15)
        per_lang = {lang: sum(r["language"] == lang for r in picked) for lang in ("python", "go", "rust", "java")}
        self.assertEqual(per_lang["rust"], 6)  # a small language gives everything it has
        self.assertEqual(build.balance(records, 40, 15), picked)  # fixed for a given set
        self.assertEqual(len(build.balance(records[:5], 10, 3)), 5)

    def test_spare_eval_candidates_are_skipped_once_a_split_is_full(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "build_edit_data", Path(__file__).resolve().parents[1] / "scripts" / "build_edit_data.py")
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        self.assertEqual((build.eval_need("test"), build.eval_need("select")), ((8, 3), (3, 1)))  # 40/6 + 1, 15/6
        cands = {f"t{i}": {"id": f"t{i}", "split": "test", "kind": "bug"} for i in range(7)}
        cands.update({f"f{i}": {"id": f"f{i}", "split": "test", "kind": "feature"} for i in range(2)})
        q = build.EvalQuota([{"id": k, "ok": True} for k in cands], cands)
        bug, feature = {"split": "test", "kind": "bug"}, {"split": "test", "kind": "feature"}
        self.assertTrue(q.full(bug))  # 7 bugs fill the 5 non-feature slots
        self.assertFalse(q.full(feature))  # 2 of the 3 features
        q.add(feature)
        self.assertTrue(q.full(feature))
        self.assertFalse(q.full({"split": "train", "kind": "bug"}))  # training is never spare
        self.assertFalse(q.full({"split": "select", "kind": "bug"}))

    def test_reruns_skip_repositories_that_have_generated_tasks(self):
        import importlib.util
        import tempfile
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "build_edit_data", Path(__file__).resolve().parents[1] / "scripts" / "build_edit_data.py")
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        rows = [{"id": "a1", "key": "a", "kind": "feature", "source": "generated"},
                {"id": "b1", "key": "b", "kind": "bug", "source": "generated"},
                {"id": "c1", "key": "c", "kind": "bug", "source": "swe-smith-go"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "go.jsonl"
            self.assertEqual(build.generated_keys(path, "bug"), set())
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.assertEqual((build.generated_keys(path, "feature"), build.generated_keys(path, "bug")), ({"a"}, {"b"}))

    def test_a_crlf_file_keeps_its_line_ends_and_its_patch_applies(self):
        import importlib.util
        import shutil
        import tempfile
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "build_edit_data", Path(__file__).resolve().parents[1] / "scripts" / "build_edit_data.py")
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "clean"
            (clean / "src").mkdir(parents=True)
            (clean / "src" / "A.java").write_bytes(b"class A {\r\n  int f() {\r\n    return 1;\r\n  }\r\n}\r\n")
            (clean / "src" / "Mixed.java").write_bytes(b"class M {\r\n}\n")
            self.assertIsNone(build._source_text(clean / "src" / "Mixed.java"))
            text, newline = build._source_text(clean / "src" / "A.java")
            self.assertEqual((text.count("\r"), newline), (0, "\r\n"))
            work = Path(tmp) / "w0"
            shutil.copytree(clean, work)
            build._write_source(work / "src" / "A.java", text.replace("return 1", "return 2"), newline)
            patch = build._tree_diff(clean, work, ["src/A.java"])
            self.assertEqual(sum(l.startswith(("+", "-")) and not l.startswith(("+++", "---")) for l in patch.split("\n")), 2)
            gold = editdata.reverse_patch(patch, clean)
            self.assertIn("+    return 1;\r\n", gold)

    def test_java_helpers_stay_visible(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "build_edit_data", Path(__file__).resolve().parents[1] / "scripts" / "build_edit_data.py")
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        texts = {"core/src/test/java/a/ParseTest.java": "class ParseTest { @Test void p() {} }",
                 "core/src/test/java/a/TestUtils.java": "class TestUtils { static int f() { return 1; } }",
                 "core/src/test/java/a/OldTest.java": "class OldTest extends TestCase { void testX() {} }"}
        self.assertEqual(build.generated_scope("java", sorted(texts), texts),
                         (["a.OldTest", "a.ParseTest"], "core",
                          ["core/src/test/java/a/OldTest.java", "core/src/test/java/a/ParseTest.java"]))
        self.assertEqual(len(build.generated_scope("java", sorted(texts))[2]), 3)  # without texts: every file
        tests = ["packages/rx/spec/a-spec.ts", "apps/site/src/b.spec.ts"]  # a monorepo's tests run from one package
        self.assertEqual(build.generated_scope("typescript", tests, test_cwd="packages/rx"),
                         (["spec/a-spec.ts"], "packages/rx", ["packages/rx/spec/a-spec.ts"]))
        self.assertTrue(all(o.get("runner", "jest") in grader.RUNNERS for o in build.REPO_TESTS.values()))
        self.assertEqual(build.hide_plan("rust", ["src/datetime/tests.rs", "tests/it.rs", "crates/c/src/tests/mod.rs"]),
                         (["tests/it.rs"], {"src/datetime/tests.rs": "", "crates/c/src/tests/mod.rs": ""}))
        self.assertEqual(build.hide_plan("go", ["a_test.go"]), (["a_test.go"], {}))
        js = {"packages/rx/spec/a-spec.ts":"describe('a', () => { it('works', () => {}); });",
              "packages/rx/spec/helpers/marbles.ts": "export function hot(m: string) { return cold(m); }"}
        self.assertEqual(build.generated_scope("typescript", sorted(js), js, "packages/rx")[2], ["packages/rx/spec/a-spec.ts"])


if __name__ == "__main__":
    unittest.main()
