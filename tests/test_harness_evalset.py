"""Tests for scripts/harness_evalset.py (rlm-harness v1 step 1). Offline: every fixture is built in a temp folder.

    .venv\\Scripts\\python.exe -m unittest -v tests.test_harness_evalset

Cases that run pytest need uv; cases that run Playwright need Node, pnpm and a Chromium for the pinned Playwright in
PLAYWRIGHT_BROWSERS_PATH (HARNESS_EVALSET_PLAYWRIGHT names the version, default Web-Bench's 1.57.0); cases that run
kernels need a GPU. Each skips without them. No case reads D:, clones from the network, calls a model or reads a key.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import harness_evalset as he  # noqa: E402
from rlm_rl.env import Task  # noqa: E402
from rlm_rl.tasks import editspec  # noqa: E402

HARNESS = Path(os.environ.get("RLMH_HARNESS", ROOT.parent / "rlm-harness"))
GPU_DIR = HARNESS / "app" / "test" / "fixtures" / "evalset" / "gpu"
HAVE_UV = shutil.which("uv") is not None
HAVE_WEB = bool(shutil.which("node") and shutil.which("pnpm") and os.environ.get("PLAYWRIGHT_BROWSERS_PATH"))
PW_VERSION = os.environ.get("HARNESS_EVALSET_PLAYWRIGHT", "1.57.0")


def load_gpu():
    if not (GPU_DIR / "gpu.py").exists():
        raise unittest.SkipTest(f"no gpu.py at {GPU_DIR} (set RLMH_HARNESS to the rlm-harness checkout)")
    spec = importlib.util.spec_from_file_location(f"gpu_{random.randrange(10**9)}", GPU_DIR / "gpu.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def have_gpu() -> bool:
    try:
        return load_gpu().probe()[1] > 0
    except unittest.SkipTest:
        return False


SMALL_PLAN = {"READ_PLAN": {("research_test", 2): 2}, "PYTHON_PLAN": {"S": 2}, "WEB_PLAN": {"M": 2},
              "KERNEL_PLAN": {"S": 2}}


@contextlib.contextmanager
def small_plan():
    with contextlib.ExitStack() as stack:
        for k, v in SMALL_PLAN.items():
            stack.enter_context(mock.patch.object(he, k, v))
        yield


def put(folder: Path, files: dict[str, str | bytes]) -> None:
    for rel, v in files.items():
        p = folder / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(v.encode("utf-8") if isinstance(v, str) else v)


# ----------------------------------------------------------------------------- fixtures


def read_task(i: int, kind: str = "multihop", question: str = "Which river runs past the old mill?",
              answer="Severn", context: str = "doc one\ndoc two\n") -> he.ReadTask:
    grading = ({"type": "list", "item": "entity", "blind": [], "source": "qampari", "kind": "list"} if kind == "list"
               else {"type": "text", "aliases": [], "blind": None, "source": "musique", "kind": "multihop"})
    grading.update(closed_book="v1", open_book="v1")
    task = Task(f"{he.SET}-research_test-L2-{i:02d}", "research_test", question, context, answer=answer, level=2,
                grading=grading)
    return he.ReadTask("research", he.SPLITS[i % 2], "research_test", 2, task)


def simple_cand(track: str, split: str, size: str, base: dict, hidden: dict, gold: dict, source: dict, env: str,
                task: str = "Fix the parser so empty input works.", runner: str = "pytest") -> he.Cand:
    return he.Cand(key=he.cand_key(track, split, size, source), track=track, split=split, size=size, source=source,
                   task=task, runner=runner, env=env, files=sorted(hidden), leak_texts=[task],
                   base=lambda t: he.put_files(t, {k: v.encode() for k, v in base.items()}),
                   hidden={k: v.encode() for k, v in hidden.items()},
                   gold=lambda t: he.put_files(t, {k: v.encode() for k, v in gold.items()}))


def kernel_tree(cache: Path, op: str, split: str, size: str, shared: Path | None = None) -> Path:
    """A tiny kernel task (vec_add) in the build cache's layout."""
    shared = shared or cache / "kernels" / "_shared"
    if not (shared / "gpu.py").exists():
        put(shared, {"gpu.py": "# the shim\n", "gpu_compat.h": "// the prelude\n"})
    src = cache / "kernels" / op
    stub = ("// vec_add: out[i] = a[i] + b[i] for i < n, fp32.\nGPU_KERNEL void vec_add(const float* a, const float* b, "
            "float* out, int n) {\n    (void)a; (void)b; (void)out; (void)n;\n}\n")
    gold = ("// vec_add: out[i] = a[i] + b[i] for i < n, fp32.\nGPU_KERNEL void vec_add(const float* a, const float* b, "
            "float* out, int n) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) out[i] = a[i] + b[i];\n}\n")
    kernels_py = textwrap.dedent('''\
        import glob, os
        import numpy as np
        import gpu
        import launch
        HERE = os.path.dirname(os.path.abspath(__file__))
        MODULE = gpu.load(sorted(glob.glob(os.path.join(HERE, "kernels", "*.gpu.cu"))))


        def add(a, b):
            a = np.ascontiguousarray(a, np.float32)
            b = np.ascontiguousarray(b, np.float32)
            n = a.size
            if n == 0:
                return np.zeros(0, np.float32)
            cfg = launch.vec_add(n)
            out = gpu.empty(n, np.float32, fill=0xff)
            MODULE.launch("vec_add", cfg["grid"], cfg["block"], [gpu.to_device(a), gpu.to_device(b), out, np.int32(n)])
            return out.to_host()
        ''')
    put(src, {
        "meta.json": json.dumps({"op": op, "split": split, "size": size, "files": ["kernels/vec_add.gpu.cu"]}),
        "base/kernels/vec_add.gpu.cu": stub, "base/kernels.py": kernels_py,
        "base/launch.py": 'def vec_add(n):\n    """Grid and block for n elements."""\n    raise NotImplementedError("set the launch configuration")\n',
        "base/tests/conftest.py": "import pytest\n",
        "base/tests/test_smoke.py": "import kernels\n\n\ndef test_small():\n    assert (kernels.add([1.0], [2.0]) == 3).all()\n",
        "base/REQUIREMENTS.md": "# vec_add\n\nAdd two fp32 vectors of any length.\n",
        "base/gpu.py": (shared / "gpu.py").read_bytes(), "base/gpu_compat.h": (shared / "gpu_compat.h").read_bytes(),
        "gold/kernels/vec_add.gpu.cu": gold,
        "gold/launch.py": 'def vec_add(n):\n    return {"grid": (n + 255) // 256, "block": 256, "shared": 0}\n',
        "hidden/tests/test_vec_add_values.py": textwrap.dedent('''\
            import numpy as np
            import kernels


            def test_vec_add_many_blocks():
                rng = np.random.default_rng(1)
                a, b = rng.random(1000, dtype=np.float32), rng.random(1000, dtype=np.float32)
                np.testing.assert_allclose(kernels.add(a, b), a + b, rtol=1e-6)  # fp32: 1e-6 relative


            def test_vec_add_one():
                assert kernels.add(np.ones(1, np.float32), np.ones(1, np.float32))[0] == 2.0
            '''),
    })
    bad = {  # mutants: no bounds check, a subtraction, a constant, an index off by one block
        1: gold.replace("if (i < n) out[i]", "out[i]"),
        2: gold.replace("a[i] + b[i]", "a[i] - b[i]"),
        3: gold.replace("a[i] + b[i]", "0.0f"),
        4: gold.replace("int i = blockIdx.x * blockDim.x + threadIdx.x;", "int i = threadIdx.x;"),
    }
    for k, text in bad.items():
        put(src, {f"mutants/{k}/kernels/vec_add.gpu.cu": text})
    return src


class Fixture:
    """A temp data root, build cache and a small set assembled from fake candidates of every track."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="evalset-test-"))
        self.paths = he.Paths(self.tmp / "data", he.SET, self.tmp / "no-harness")
        self.paths.cache.mkdir(parents=True)

    def close(self):
        he.rmtree(self.tmp)

    def built(self) -> he.Built:
        b = he.Built()
        b.read = [read_task(0, "list", "List the rivers of the valley.", ["Severn", "Wye"]), read_task(1)]
        envs = he.Envs(self.paths.cache)
        py = envs.add("py", {"python": "3.12", "requirements": ["pytest==8.0.0"], "import_root": "."})
        for split in he.SPLITS:
            c = simple_cand("python", split, "S", {"pkg/mod.py": "def f():\n    raise NotImplementedError\n"},
                            {"tests/test_mod.py": "from pkg.mod import f\n\n\ndef test_f_returns_one():\n    assert f() == 1\n"},
                            {"pkg/mod.py": "def f():\n    return 1\n"},
                            {"family": "edit_real", "record_id": f"secret-record-{split}", "repo": f"acme__lib{split}.123",
                             "level": 2}, py)
            b.repo.append((c, {"f2p": ["tests/test_mod.py::test_f_returns_one"], "p2p": [], "dropped_p2p": [],
                               "gold_seconds": 1.5, "timeout_s": 120}))
            node_key = "web-" + "a" * 12
            b.nodeenvs[node_key] = {"node": "22", "pnpm": he.PNPM, "playwright": "1.57.0"}
            w = simple_cand("web", split, "M", {"src/index.html": "<h1></h1>\n", "package.json": "{}\n",
                                                "FEATURES.md": "1. Show a heading.\n"},
                            {"test/task-1.spec.js": "test('heading shows', async () => {})\n"},
                            {"src/index.html": "<h1>Hello</h1>\n"},
                            {"family": "webbench", "project": f"zebraproj{split}", "level": "M"}, node_key,
                            task=he.WEB_TEXT, runner="playwright")
            b.repo.append((w, {"f2p": ["task-1.spec.js::heading shows"], "p2p": [], "dropped_p2p": [],
                               "gold_seconds": 4.0, "timeout_s": 120}))
        kernel_tree(self.paths.cache, "opalpha", "tune", "S")
        kernel_tree(self.paths.cache, "opbeta", "heldout", "S")
        kkey = envs.add("kernel", {"python": "3.12", "requirements": [he.NVRTC], "import_root": "."})
        for op in ("opalpha", "opbeta"):
            c = he.kernel_cand(self.paths.cache, op)
            c.env = kkey
            b.repo.append((c, {"f2p": ["tests/test_vec_add_values.py::test_vec_add_many_blocks"], "p2p": [],
                               "dropped_p2p": [], "gold_seconds": 2.0, "timeout_s": 120, "mutants_caught": 4}))
        b.venvs = dict(envs.specs)
        return b

    def assemble(self) -> dict:
        with small_plan():
            return he.assemble(self.paths, self.built(), True, ["hip"])


class Base(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def set_lines(self, name: str) -> list[dict]:
        return he.read_jsonl(self.fx.paths.set_dir / name)


# ----------------------------------------------------------------------------- the plan and the files


class TestPlan(unittest.TestCase):
    def test_plan(self):
        self.assertEqual(sum(he.READ_PLAN.values()), 12)
        self.assertEqual((sum(he.PYTHON_PLAN.values()), sum(he.WEB_PLAN.values()), sum(he.KERNEL_PLAN.values())),
                         (16, 10, 10))
        counts = he.plan_counts()
        self.assertEqual(sum(counts.values()), 48)
        for plan in (he.PYTHON_PLAN, he.WEB_PLAN, he.KERNEL_PLAN):
            self.assertTrue(all(n % 2 == 0 for n in plan.values()))
        for split in he.SPLITS:
            by_size = {s: sum(n for k, n in counts.items() if k.startswith(split + "/") and k.endswith("/" + s))
                       for s in he.SIZES}
            self.assertEqual(by_size, {"R": 6, "S": 7, "M": 7, "L": 4})
        self.assertEqual(counts["tune/research/R"], 4)
        self.assertEqual(counts["heldout/search/R"], 2)

    def test_pass_score(self):
        self.assertEqual(he.pass_score({"type": "list", "source": "qampari", "kind": "list"}), 0.8)
        self.assertEqual(he.pass_score({"type": "text", "source": "musique"}), 1.0)
        self.assertEqual(he.pass_score({"type": "list", "item": "entity", "blind": []}), 1.0)  # callgraph
        self.assertEqual(he.pass_score({"type": "entity", "kind": "needle"}), 1.0)


class TestSetFiles(Base):
    def test_ids_reveal_nothing(self):
        self.fx.assemble()
        tl = self.set_lines("tasks.jsonl")
        secrets_ = ["secret-record", "acme", "lib", "zebraproj", "opalpha", "opbeta", "edit_real", "L2", "research_test"]
        for t in tl:
            self.assertRegex(t["id"], he.ID_RE)
            rest = t["id"][len(he.SET):]
            for s in secrets_:
                self.assertNotIn(s, rest)
            self.assertNotRegex(rest, r"-(L[1-5]|level)")

    def test_tasks_jsonl_keys(self):
        self.fx.assemble()
        tl, gl = self.set_lines("tasks.jsonl"), self.set_lines("grades.jsonl")
        for t in tl:
            if t["kind"] == "read":
                self.assertEqual(set(t), {"id", "kind", "track", "split", "size", "question", "input"})
                continue
            want = {"id", "kind", "track", "split", "size", "task", "base", "test_hint"}
            want |= {"node_env"} if t["track"] == "web" else {"venv"}
            want |= {"requires"} if t["track"] == "kernel" else set()
            self.assertEqual(set(t), want, t["id"])
            if t["track"] == "kernel":
                self.assertEqual(t["requires"], ["gpu"])
            self.assertEqual(t["test_hint"], he.TEST_HINT[t["track"]])
        for g in gl:
            if g["grader"] == "answer":
                self.assertEqual(set(g), {"id", "grader", "gold", "grading", "pass_score"})
            else:
                self.assertEqual(set(g), {"id", "grader", "runner", "hidden", "f2p", "p2p", "files", "timeout_s"})
        text = (self.fx.paths.set_dir / "tasks.jsonl").read_text(encoding="utf-8")
        for g in gl:
            for name in g.get("f2p", []):
                self.assertNotIn(name.split("::")[-1], text)
            if g["grader"] == "answer":
                for item in g["gold"] if isinstance(g["gold"], list) else [g["gold"]]:
                    self.assertNotIn(item, text)

    def test_tar_deterministic(self):
        self.fx.assemble()
        first = {p.name: p.read_bytes() for p in (self.fx.paths.set_dir / "repos").iterdir()}
        first |= {p.name: p.read_bytes() for p in (self.fx.paths.set_dir / "hidden").iterdir()}
        tasks1 = (self.fx.paths.set_dir / "tasks.jsonl").read_bytes()
        self.fx.assemble()
        second = {p.name: p.read_bytes() for p in (self.fx.paths.set_dir / "repos").iterdir()}
        second |= {p.name: p.read_bytes() for p in (self.fx.paths.set_dir / "hidden").iterdir()}
        self.assertEqual(first, second)
        self.assertEqual(tasks1, (self.fx.paths.set_dir / "tasks.jsonl").read_bytes())
        self.assertEqual(he.tar_bytes({"b": b"2", "a": b"1"}), he.tar_bytes({"a": b"1", "b": b"2"}))
        for t in self.set_lines("tasks.jsonl"):
            if t["kind"] != "repo":
                continue
            base = he.tar_names((self.fx.paths.set_dir / t["base"]).read_bytes())
            hidden = {n for n, _ in he.tar_names((self.fx.paths.set_dir / f"hidden/{t['id']}.tar.gz").read_bytes(), True)}
            for name, _ in base:
                self.assertNotIn(".git", name.split("/"))
                self.assertNotIn("node_modules", name.split("/"))
                self.assertNotIn(name, hidden)
            with tarfile.open(self.fx.paths.set_dir / t["base"]) as tar:
                for m in tar.getmembers():
                    self.assertEqual((m.mtime, m.uid, m.gid, m.uname, m.gname), (0, 0, 0, "", ""))

    def test_line_separators_stay_inside_lines(self):
        b = self.fx.built()
        b.read[1] = read_task(1, question="Which river\u2028runs past the\x85 old mill?")
        with small_plan():
            he.assemble(self.fx.paths, b, True, ["hip"])
            self.assertEqual(he.cmd_check(self.fx.paths), [])
        self.assertIn("\u2028", self.set_lines("tasks.jsonl")[1]["question"] + self.set_lines("tasks.jsonl")[0]["question"])

    def test_bytes_lf(self):
        self.fx.assemble()
        d = self.fx.paths.set_dir
        tl = self.set_lines("tasks.jsonl")
        for t in tl:
            if t["kind"] == "read":
                self.assertEqual((d / t["input"][0]).read_bytes(), "doc one\ndoc two\n".encode("utf-8"))
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in (".jsonl", ".json", ".txt"):
                self.assertNotIn(b"\r", p.read_bytes(), p.name)

    def test_kernel_layout(self):
        cache = self.fx.paths.cache
        kernel_tree(cache, "opgamma", "tune", "S")
        c = he.kernel_cand(cache, "opgamma")
        self.assertEqual(c.task, he.KERNEL_TEXT)
        self.assertTrue(he.KERNEL_TEXT.startswith("This repository is part of the GPU backend of an LLM inference"))
        with tempfile.TemporaryDirectory() as t:
            c.base(Path(t))
            files = he.tree_files(Path(t))
        self.assertEqual(set(files), {"kernels/vec_add.gpu.cu", "kernels.py", "launch.py", "tests/conftest.py",
                                      "tests/test_smoke.py", "REQUIREMENTS.md", "gpu.py", "gpu_compat.h"})
        shared = cache / "kernels" / "_shared"
        self.assertEqual(files["gpu.py"], (shared / "gpu.py").read_bytes())
        self.assertEqual(files["gpu_compat.h"], (shared / "gpu_compat.h").read_bytes())
        gold = (cache / "kernels" / "opgamma" / "gold" / "kernels" / "vec_add.gpu.cu").read_bytes()
        self.assertNotEqual(files["kernels/vec_add.gpu.cu"], gold)
        self.assertEqual(sorted(c.hidden), ["tests/test_vec_add_values.py"])
        self.assertEqual(len(c.mutants), 4)
        self.fx.assemble()
        kl = [t for t in self.set_lines("tasks.jsonl") if t["track"] == "kernel"]
        self.assertTrue(kl and all(t["requires"] == ["gpu"] for t in kl))

    def test_check_detects(self):
        self.fx.assemble()
        with small_plan():
            self.assertEqual(he.cmd_check(self.fx.paths), [])
        d = self.fx.paths.set_dir
        tl = self.set_lines("tasks.jsonl")
        kernel_id = next(t["id"] for t in tl if t["track"] == "kernel")
        python_id = next(t["id"] for t in tl if t["track"] == "python")

        def tamper_grades():
            b = bytearray((d / "grades.jsonl").read_bytes())
            b[10] = ord("X") if b[10] != ord("X") else ord("Y")
            (d / "grades.jsonl").write_bytes(bytes(b))

        def remove_tar():
            (d / "repos" / f"{python_id}.tar").unlink()

        def gold_in_tasks():
            lines = (d / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[0])
            row["question"] += " Severn"
            lines[0] = json.dumps(row, ensure_ascii=False)
            (d / "tasks.jsonl").write_bytes(("\n".join(lines) + "\n").encode())

        def git_member():
            p = d / "repos" / f"{python_id}.tar"
            files = he.untar(p.read_bytes())
            files[".git/config"] = b"[core]\n"
            p.write_bytes(he.tar_bytes(files))

        def kernel_gold():
            p = d / "repos" / f"{kernel_id}.tar"
            files = he.untar(p.read_bytes())
            src = self.fx.paths.cache / "kernels" / "opalpha" / "gold" / "kernels" / "vec_add.gpu.cu"
            files["kernels/vec_add.gpu.cu"] = src.read_bytes()
            p.write_bytes(he.tar_bytes(files))

        def break_json():
            (d / "grades.jsonl").write_bytes(b"{not json\n" + (d / "grades.jsonl").read_bytes())

        def drop_tasks():
            (d / "tasks.jsonl").unlink()

        for name, damage in (("broken JSON", break_json), ("missing tasks.jsonl", drop_tasks),
                             ("grades byte", tamper_grades), ("removed tar", remove_tar),
                             ("gold in tasks", gold_in_tasks), (".git member", git_member),
                             ("kernel gold", kernel_gold)):
            with self.subTest(name):
                self.fx.assemble()
                damage()
                with small_plan():
                    problems = he.cmd_check(self.fx.paths)
                self.assertTrue(problems, name)
        self.fx.assemble()
        self.assertTrue(he.cmd_check(self.fx.paths))  # the full plan: this small set is short

    def test_summary_no_text(self):
        self.fx.assemble()
        out = self.fx.tmp / "summary.md"
        he.cmd_summary(self.fx.paths, out)
        text = out.read_text(encoding="utf-8")
        for sentinel in ("old mill", "Severn", "Wye", "test_f_returns_one", "heading shows", "zebraproj", "acme__lib",
                         "opalpha", "opbeta", "secret-record", "Fix the parser", "web project"):
            self.assertNotIn(sentinel, text)
        self.assertIn("python", text)
        self.assertRegex(text, r"\| harness-eval-v1-python-S-01 \|")
        with mock.patch.object(he, "TRACKS", he.TRACKS + ("zebraprojtune",)), \
                self.assertRaises(he.SetupError):
            sources = he.read_jsonl(self.fx.paths.cache / "sources.jsonl")
            he.summary_guard("a line naming zebraprojtune", [], [], sources)


# ----------------------------------------------------------------------------- read tasks


class TestResearch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        p = mock.patch("rlm_rl.tasks.PRIVATE_DIR", self.tmp / "private")
        p.start()
        self.addCleanup(p.stop)

    def fake_make(self, lists_after: int | None, checks=("v1", "v1")):
        calls = []

        def make(family, level, seed, task_id):
            calls.append(seed)
            kind = "list" if lists_after is not None and len(calls) > lists_after else "multihop"
            grading = {"type": "list" if kind == "list" else "text", "kind": kind, "closed_book": checks[0],
                       "open_book": checks[1], "source": "qampari" if kind == "list" else "musique"}
            return Task(task_id, family, "Q?", "ctx", answer="A", level=level, grading=grading)

        return make, calls

    def test_research_guards(self):
        plan = {("research_test", 2): 4}
        make, calls = self.fake_make(lists_after=4)  # the first 4 tasks are multi-hop; the 5th draw is a list
        out = he.build_read(plan, make)
        seeds = json.loads((self.tmp / "private" / "seeds" / "harness-eval-v1-read.json").read_text())
        self.assertEqual(len(seeds), sum(he.READ_PLAN.values()) + 2)  # the full plan's seeds + one redraw per split
        for split in he.SPLITS:
            self.assertTrue(any(rt.task.grading["kind"] == "list" for rt in out if rt.split == split))
        # a rerun reads the appended seeds back and draws nothing new
        make2, _ = self.fake_make(lists_after=4)
        he.build_read(plan, make2)
        self.assertEqual(len(json.loads((self.tmp / "private" / "seeds" / "harness-eval-v1-read.json").read_text())),
                         sum(he.READ_PLAN.values()) + 2)
        never, _ = self.fake_make(lists_after=None)
        with self.assertRaises(he.Short):
            he.build_read({("research_test", 2): 2}, never)
        none_check, _ = self.fake_make(lists_after=0, checks=(None, "v1"))
        shutil.rmtree(self.tmp / "private")
        with self.assertRaises(he.SetupError):
            he.build_read({("research_test", 2): 2}, none_check)
        shutil.rmtree(self.tmp / "private")
        data = self.tmp / "data"
        data.mkdir()
        with mock.patch.object(he.tasks, "make_task", none_check), \
                mock.patch.object(he, "build_read", wraps=he.build_read) as spy:
            code = he.main(["build", "--only", "research", "--data", str(data), "--no-validate"])
        self.assertEqual(code, 2)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(len(json.loads((self.tmp / "private" / "seeds" / "harness-eval-v1-read.json").read_text())),
                         sum(he.READ_PLAN.values()))  # stopped before any redraw


# ----------------------------------------------------------------------------- python candidates


def edit_record(rid: str, repo: str, level: int, files: int, family_split: str = "test") -> dict:
    return {"id": rid, "split": family_split, "language": "python", "licence": "MIT", "repo": "owner/" + repo,
            "key": repo, "level": level, "gold_files": files, "gold_lines": 5 * files, "snapshot": f"edit/repos/{repo}.tar",
            "tests": {"f2p": [], "p2p": [], "hidden": {}, "scope": []}, "issue": "x", "gold_patch": ""}


class TestPython(unittest.TestCase):
    def test_repo_disjoint(self):
        repos = [f"r{i}__pkg.{i}" for i in range(6)]
        recs = {"edit_real": [], "edit_spec": [], "edit_bugs": []}
        n = 0
        for repo in repos:
            for _ in range(4):
                n += 1
                recs["edit_real"].append(edit_record(f"real-{n}", repo, 1, 1))
                recs["edit_bugs"].append(edit_record(f"bugs-{n}", repo, 2, 1))
                recs["edit_spec"].append(edit_record(f"spec-{n}", repo, 4, 2))
        splits = he.assign_repos(recs)
        self.assertEqual(set(splits.values()) & set(he.SPLITS), set(he.SPLITS))
        cands = he.python_sm_candidates(recs, splits)

        class AllKept:
            jobs, validate_on = 3, False

            def many(self, cs):
                return [self.one(c) for c in cs]

            def one(self, c):
                return {"kept": True}

        seen: dict[str, set] = {}
        for (split, size), fams in cands.items():
            kept = he.draw_cell(f"{split}/python/{size}", [fams[f] for f, _ in he.PYTHON_FAMILIES[size]], 3, 7,
                                AllKept())
            groups = [c.group for c, _ in kept]
            self.assertLessEqual(max(groups.count(g) for g in groups), he.MAX_PER_REPO)
            for g in groups:
                seen.setdefault(g, set()).add(split)
        self.assertTrue(all(len(s) == 1 for s in seen.values()))
        fixed = set(he.SPLIT_REPOS["tune"]) & set(he.SPLIT_REPOS["heldout"])
        self.assertFalse(fixed)
        web_t, web_h = set(he.WEB_SPLIT["tune"]), set(he.WEB_SPLIT["heldout"])
        self.assertFalse(web_t & web_h)
        for v in he.WEB_VARIANTS:
            self.assertNotIn(v, web_t | web_h)
        tune_ops = {op for (s, _), ops in he.KERNEL_OPS.items() if s == "tune" for op in ops}
        held_ops = {op for (s, _), ops in he.KERNEL_OPS.items() if s == "heldout" for op in ops}
        self.assertFalse(tune_ops & held_ops)

    def test_draw_is_independent_of_jobs(self):
        repos = [f"r{i}__p.{i}" for i in range(4)]
        recs = {"edit_real": [edit_record(f"a{i}{j}", r, 1, 1) for i, r in enumerate(repos) for j in range(3)],
                "edit_spec": [], "edit_bugs": []}
        cands = he.python_sm_candidates(recs, he.assign_repos(recs))

        class Some:
            validate_on = False

            def __init__(self, jobs):
                self.jobs = jobs

            def many(self, cs):
                return [self.one(c) for c in cs]

            def one(self, c):
                return {"kept": int(c.source["record_id"][-1]) != 0}

        fams = [cands[("tune", "S")]["edit_real"], cands[("tune", "S")]["edit_bugs"]]
        a = [c.key for c, _ in he.draw_cell("tune/python/S", fams, 2, 3, Some(1))]
        b = [c.key for c, _ in he.draw_cell("tune/python/S", fams, 2, 3, Some(4))]
        self.assertEqual(a, b)
        with self.assertRaises(he.Short):
            he.draw_cell("tune/python/S", fams, 50, 3, Some(2))

    def test_large_cluster(self):
        mods = {}
        names = [f"m{i}" for i in range(6)]
        for i, m in enumerate(names):
            nxt = names[(i + 1) % len(names)]
            body = []
            for j in range(4):
                lines = "\n".join(f"    v{k} = x + {k}" for k in range(14))
                body.append(f'def f{j}(x):\n    """Return x shifted, step {j}."""\n{lines}\n    return {nxt}.f{j}(v0) if x < 0 else v13\n')
            mods[f"pkg/{m}.py"] = "from pkg import " + nxt + "\n\n\n" + "\n\n".join(body)
        mods["pkg/__init__.py"] = ""
        repo = editspec.analyze(mods)
        tests_of = {k: {f"tests/test_{fn.path.split('/')[-1]}::test_{fn.name}"} for k, fn in repo.fns.items()}

        class Ctx:
            pass

        ctx = Ctx()
        ctx.repo, ctx.tests_of, ctx.at_import = repo, tests_of, set()
        clusters = he.large_clusters(ctx, random.Random(1))
        self.assertTrue(clusters)
        c = clusters[0]
        self.assertGreaterEqual(len({repo.fns[k].path for k in c}), 4)
        self.assertGreaterEqual(sum(repo.fns[k].body_lines for k in c), 300)
        small = {p: t for p, t in mods.items() if p in ("pkg/m0.py", "pkg/m1.py", "pkg/m2.py", "pkg/__init__.py")}
        repo3 = editspec.analyze(small)
        ctx.repo = repo3
        ctx.tests_of = {k: {"tests/test_x.py::t"} for k in repo3.fns}
        self.assertEqual(he.large_clusters(ctx, random.Random(1)), [])

    def test_large_task_text(self):
        self.assertEqual(he.LARGE_TEXT, (
            "Every function and method listed in REQUIREMENTS.md, at the root of this repository, has had its body "
            "replaced by `raise NotImplementedError`. Their signatures and docstrings are unchanged. Implement all of "
            "them so the package behaves as REQUIREMENTS.md and the docstrings describe, and keep the repository's "
            "other tests passing."))
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        data = tmp / "data"
        snap = he.tar_bytes({"pkg/mod.py": b"def f():\n    return 1\n"})
        put(data, {"edit/repos/acme.tar": snap})
        old, new = "def f():\n    return 1\n", "def f():\n    raise NotImplementedError\n"
        import build_edit_tasks_v2 as v2
        rec = {"id": "large-1", "snapshot": "edit/repos/acme.tar", "break_patch": v2.file_patch({"pkg/mod.py": (old, new)}),
               "gold_patch": v2.file_patch({"pkg/mod.py": (new, old)}), "remove": [], "replace": {}, "level": "L",
               "repo": "o/acme", "key": "acme", "tests": {"hidden": {"tests/test_mod.py": "def test_f():\n    pass\n"},
                                                         "scope": ["tests/test_mod.py"], "f2p": [], "p2p": []},
               "issue": he.LARGE_TEXT}
        with mock.patch.dict(os.environ, {"RLM_DATA_DIR": str(data)}):
            c = he.python_cand(rec, "edit_large", "tune", "L", {"REQUIREMENTS.md": b"# Requirements\n"}, he.LARGE_TEXT)
            work = tmp / "w"
            work.mkdir()
            c.base(work)
            self.assertEqual((work / "REQUIREMENTS.md").read_bytes(), b"# Requirements\n")
            self.assertIn("NotImplementedError", (work / "pkg" / "mod.py").read_text())
            self.assertNotIn("tests/test_mod.py", he.tree_files(work))
            c.gold(work)
            self.assertFalse((work / "REQUIREMENTS.md").exists())
            self.assertEqual((work / "pkg" / "mod.py").read_text(), old)

    def test_leaks(self):
        gold_line = "    return parse_header(stream, strict=True)"
        import build_edit_tasks_v2 as v2
        patch = v2.file_patch({"pkg/a.py": ("def f():\n    pass\n", f"def f():\n{gold_line}\n")})
        c = simple_cand("python", "tune", "S", {}, {}, {}, {"repo": "r"}, "py-x",
                        task=f"The reader breaks; it should do `{gold_line.strip()}` instead.")
        c.gold_patch = patch
        self.assertIn("gold patch", he.leak_names(c, []))
        c2 = simple_cand("python", "tune", "S", {}, {}, {}, {"repo": "r"}, "py-x",
                         task="Make test_reader_handles_empty pass again.")
        self.assertIn("hidden test", he.leak_names(c2, ["tests/test_r.py::test_reader_handles_empty"]))
        c3 = simple_cand("python", "tune", "S", {}, {}, {}, {"repo": "r"}, "py-x", task="Parsing breaks on tabs.")
        self.assertEqual(he.leak_names(c3, ["tests/test_r.py::test_reader_handles_empty"]), "")
        w = simple_cand("web", "tune", "S", {}, {"test/task-1.spec.js": ""}, {}, {"project": "p"}, "web-x",
                        task="1. Make sure Check Header is on the top of Main.", runner="playwright")
        w.spec_titles = ["Check Header is on the top of Main"]
        w.leak_texts = [w.task]
        self.assertIn("spec title", he.leak_reason(w))
        w.leak_texts = ["Build the page."]
        self.assertEqual(he.leak_reason(w), "")
        w.spec_titles, w.leak_texts = ["common/Question.js"], ["1. Create common/Question.js."]
        self.assertEqual(he.leak_reason(w), "")  # a title naming the file the request asks for (D-229)
        w.spec_titles, w.leak_texts = ["toolbar area"], ["2. Append a button in the toolbar area."]
        self.assertEqual(he.leak_reason(w), "")  # two words name a part of the page the request names (D-236)
        # check applies the same web rule, on whole titles (a dot in a title is not a pytest separator)
        line = '{"task": "2. In the toolbar area, add a button. Make sure Check Header is on the top of Main."}'
        self.assertFalse(he.names_test("task-3.spec.js::toolbar area", "playwright", line))
        self.assertTrue(he.names_test("task-1.spec.js::Check Header is on the top of Main", "playwright", line))
        self.assertFalse(he.names_test("task-1.spec.js::common/Question.js", "playwright", line + " common/Question.js"))
        self.assertTrue(he.names_test("tests/test_r.py::test_reader_handles_empty", "pytest",
                                      "Fix test_reader_handles_empty."))
        w.hidden["test/util/index.js"] = b""
        w.leak_texts = ["1. Export it from index.js."]
        self.assertEqual(he.leak_reason(w), "")  # a helper, not a spec
        w.leak_texts = ["1. Make task-1.spec.js pass."]
        self.assertIn("spec file", he.leak_reason(w))
        self.assertTrue(he.read_leak("Which city did the Severn Bridge connect?", "Severn Bridge"))
        self.assertFalse(he.read_leak("Which city did the bridge connect?", ["Bristol", "Chepstow"]))

    def test_requirements_and_import_root(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        put(tmp, {"pyproject.toml": '[project]\nname = "My_Pkg"\ndependencies = ["attrs>=20"]\n'
                                    '[project.optional-dependencies]\ntest = ["pytest>=7", "hypothesis", "my-pkg[x]"]\n',
                  "tox.ini": "[testenv]\ndeps =\n    coverage\n    -rrequirements.txt\n    {[base]deps}\n"
                             "    py312: pytest-asyncio\n",
                  "src/my_pkg/__init__.py": ""})
        test, runtime, own = he.project_requirements(tmp)
        self.assertEqual(own, "my-pkg")
        self.assertEqual(test, ["pytest>=7", "hypothesis", "coverage"])
        self.assertEqual(runtime, ["attrs>=20"])
        self.assertEqual(he.import_root(tmp), "src")
        put(tmp, {"my_pkg/__init__.py": ""})
        self.assertEqual(he.import_root(tmp), ".")

    def test_runtime_file_and_extras(self):
        # D-230: a setup.py that reads requirements.txt, and an extra a test imports
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        put(tmp, {"setup.py": "from setuptools import setup\nsetup(name='pkg', install_requires=read_reqs(),\n"
                              "      extras_require={'cli': ['click'], 'test': ['pytest'], 'all': ['pkg[cli]']})\n",
                  "requirements.txt": "orderly-set==5.2.2\n# a comment\n",
                  "setup.cfg": "[options.extras_require]\ndiagrams =\n    railroad-diagrams\n    jinja2\n"})
        test, runtime, own = he.project_requirements(tmp)
        self.assertEqual(runtime, ["orderly-set==5.2.2"])
        self.assertEqual(he.project_extras(tmp), ["railroad-diagrams", "jinja2", "click"])
        self.assertEqual(he.PYTEST_ENV["PYTHONUTF8"], "1")
        self.assertTrue(any(a.startswith("tzdata==") for a in he.ALWAYS))


# ----------------------------------------------------------------------------- web


def fake_webbench(root: Path, extra_projects: dict | None = None) -> Path:
    wb = root / "web-bench"
    reqs = json.dumps({"id": "init", "description": "Set up the page."}) + "\n" + "\n".join(
        json.dumps({"id": f"task-{i}", "description": f"Request number {i}: add block {i}."}) for i in range(1, 13)) + "\n"
    specs = {f"projects/demo/test/task-{i}.spec.js":
             f"const {{ test, expect }} = require('@playwright/test')\nconst {{ helper }} = require('./util')\n"
             f"const constants = require('constants')\n"
             f"test('block {i} shows', async ({{ page }}) => {{ await page.goto('/'); helper() }})\n"
             for i in range(1, 13)}
    put(wb, {
        "LICENSE.md": "Apache License 2.0\n",
        "libraries/test-util/package.json": '{"name": "@web-bench/test-util", "main": "src/index.js"}\n',
        "libraries/test-util/src/index.js": "module.exports = {}\n",
        "projects/demo/package.json": json.dumps({
            "name": "@web-bench/demo", "eval": {"stable": True}, "scripts": {"start": "vite src"},
            "devDependencies": {"@web-bench/test-util": "workspace:*", "@playwright/test": "1.57.0", "vite": "^7.1.5"}}),
        "projects/demo/playwright.config.js": "const { defineConfig } = require('@playwright/test')\n"
                                              "require('dotenv')\nmodule.exports = defineConfig({ testDir: './test' })\n",
        "projects/demo/tasks.jsonl": reqs, "projects/demo/readme.md": "# demo\n",
        "projects/demo/src-init/index.html": "<html></html>\n", "projects/demo/src/index.html": "<h1>done</h1>\n",
        "projects/demo/test/util/index.js": "module.exports = { helper() {} }\n",
        **specs,
        "common/config/rush/pnpm-lock.yaml": textwrap.dedent("""\
            lockfileVersion: '6.0'

            importers:

              .:
                dependencies:
                  dotenv:
                    specifier: ^16.0.0
                    version: 16.4.5

              ../../projects/demo:
                devDependencies:
                  '@playwright/test':
                    specifier: 1.57.0
                    version: 1.57.0
                  '@web-bench/test-util':
                    specifier: workspace:*
                    version: link:../../libraries/test-util
                  vite:
                    specifier: ^7.1.5
                    version: 7.1.5(@types/node@22.15.32)(sass@1.92.1)

            packages:

              /vite@7.1.5:
                resolution: {integrity: sha512-x}
            """),
    })
    for name, files in (extra_projects or {}).items():
        put(wb / "projects" / name, files)
    return wb


class TestWeb(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_lock_importers(self):
        wb = fake_webbench(self.tmp)
        imp = he.lock_importers((wb / "common/config/rush/pnpm-lock.yaml").read_text())
        self.assertEqual(imp["../../projects/demo"]["vite"], "7.1.5")
        self.assertEqual(imp["../../projects/demo"]["@playwright/test"], "1.57.0")
        self.assertEqual(imp["."]["dotenv"], "16.4.5")

    def test_web_standalone(self):
        wb = fake_webbench(self.tmp)
        imp = he.lock_importers((wb / "common/config/rush/pnpm-lock.yaml").read_text())
        base, hidden, gold, requests = he.standalone(wb, "demo", "M", imp)
        pkg = json.loads(base["package.json"])
        deps = {**pkg.get("dependencies", {}), **pkg["devDependencies"]}
        self.assertFalse([v for v in deps.values() if str(v).startswith("workspace:")])
        self.assertEqual(deps["@web-bench/test-util"], "file:vendor/test-util")
        self.assertEqual(deps["vite"], "7.1.5")
        self.assertEqual(deps["dotenv"], "16.4.5")  # required by the config, pinned from the monorepo root
        self.assertEqual(pkg["scripts"]["test"], "playwright test")
        self.assertIn("esbuild", pkg["pnpm"]["onlyBuiltDependencies"])
        self.assertIn("LICENSE.md", base)
        self.assertIn(b"web-bench", base["NOTICE"])
        self.assertEqual(base["src/index.html"], (wb / "projects/demo/src-init/index.html").read_bytes())
        self.assertIn("vendor/test-util/src/index.js", base)
        self.assertEqual(base[".gitignore"], b"node_modules/\ntest-results/\nplaywright-report/\n")
        for rel in base:
            self.assertFalse(rel.startswith(("src-init/", "test/")) or rel in ("tasks.jsonl", "readme.md"), rel)
        self.assertEqual(sorted(hidden), sorted([f"test/task-{i}.spec.js" for i in range(1, 7)] + ["test/util/index.js"]))
        self.assertEqual(gold, {"src/index.html": b"<h1>done</h1>\n"})
        text, extra = he.web_task_text("M", requests)
        self.assertEqual(text, he.WEB_TEXT)
        feats = extra["FEATURES.md"].decode()
        self.assertIn("1. Request number 1", feats)  # the init line is not a request
        self.assertNotIn("Set up the page", feats)
        self.assertIn("6. Request number 6", feats)
        self.assertNotIn("Request number 7", feats)
        _, _, _, req_s = he.standalone(wb, "demo", "S", imp)
        text_s, extra_s = he.web_task_text("S", req_s)
        self.assertEqual(extra_s, {})
        self.assertTrue(text_s.startswith(he.WEB_TEXT_S))
        self.assertIn("3. Request number 3", text_s)
        self.assertNotIn("Request number 4", text_s)
        tarred = he.untar(he.tar_bytes(base))
        self.assertFalse([n for n in tarred if n.startswith(("test/", "src-init/")) or n == "tasks.jsonl"])

    def test_web_filter(self):
        good_pkg = {"eval": {"stable": True}, "scripts": {"start": "vite"}, "devDependencies": {}}
        tasks_ = "\n".join(json.dumps({"id": f"task-{i}"}) for i in range(1, 13)) + "\n"
        cases = {
            "server": ({**good_pkg, "scripts": {"start": "node scripts/dev.js"}}, "", "server of its own"),
            "dbproj": ({**good_pkg, "dependencies": {"sqlite3": "5"}}, "", "database or native"),
            "outside": (good_pkg, "require('../../../libraries/other')\n", "outside the project"),
            "unstable": ({**good_pkg, "eval": {"stable": False}}, "", "stable"),
            "dom1": (good_pkg, "", "variant"),
        }
        extra = {}
        for name, (pkg, spec, _) in cases.items():
            extra[name] = {"package.json": json.dumps(pkg), "tasks.jsonl": tasks_,
                           "playwright.config.js": "module.exports = {}\n", "test/task-1.spec.js": spec}
        wb = fake_webbench(self.tmp, extra)
        for name, (_, _, why) in cases.items():
            self.assertIn(why, he.web_skip_reason(wb / "projects" / name, name), name)
        self.assertEqual(he.web_skip_reason(wb / "projects" / "demo", "demo"), "")

    @unittest.skipUnless(HAVE_WEB, "needs node, pnpm and PLAYWRIGHT_BROWSERS_PATH with a Chromium")
    def test_web_validation(self):
        cache = self.tmp / "cache"
        server = ("const http = require('http'); const fs = require('fs'); const port = +process.env.EVAL_PROJECT_PORT;"
                  " http.createServer((q, r) => { r.setHeader('content-type', 'text/html');"
                  " r.end(fs.readFileSync(__dirname + '/src/index.html')) }).listen(port)")
        config = textwrap.dedent("""\
            const { defineConfig, devices } = require('@playwright/test')
            const PORT = process.env.EVAL_PROJECT_PORT || 3211
            module.exports = defineConfig({
              testDir: './test',
              timeout: 30000,
              retries: process.env.CI ? 2 : 0,
              use: { baseURL: `http://localhost:${PORT}` },
              webServer: { command: `node serve.js`, url: `http://localhost:${PORT}`, reuseExistingServer: false },
              projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
            })
            """)
        base = {"package.json": json.dumps({"name": "demo", "private": True,
                                            "devDependencies": {"@playwright/test": PW_VERSION}}) + "\n",
                "playwright.config.js": config, "serve.js": server, "src/index.html": "<html><body></body></html>\n"}
        hidden = {"test/task-1.spec.js": "const { test, expect } = require('@playwright/test')\n"
                                         "test('heading reads hello', async ({ page }) => {\n  await page.goto('/')\n"
                                         "  await expect(page.locator('h1')).toHaveText('Hello')\n})\n"}
        folder = cache / "web" / "demo"
        put(folder, base)
        pn = he.Pnpm(cache)
        pn.browsers = Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"])
        base["pnpm-lock.yaml"] = pn.lockfile(folder).decode()
        pn.fetch(folder)
        c = simple_cand("web", "tune", "S", base, hidden, {"src/index.html": "<html><body><h1>Hello</h1></body></html>\n"},
                        {"project": "demo"}, "web-x", task="Show a heading.", runner="playwright")
        v = he.Runner(cache, he.Envs(cache), pn).validate(c)
        self.assertTrue(v.kept, v.why)
        self.assertEqual(v.f2p, ["task-1.spec.js::heading reads hello"])
        self.assertGreaterEqual(v.timeout_s, 120)


# ----------------------------------------------------------------------------- validation with uv


PKG_TEST = textwrap.dedent('''\
    from pkg.mod import add, other


    def test_add_two_numbers():
        assert add(2, 3) == 5


    def test_other_still_works():
        assert other() == "ok"
    ''')


@unittest.skipUnless(HAVE_UV, "needs uv")
class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(he.rmtree, self.tmp)
        self.cache = self.tmp / "cache"
        self.envs = he.Envs(self.cache)
        self.key = self.envs.add("py", {"python": "3.12", "requirements": ["pytest"], "import_root": "."})
        self.runner = he.Runner(self.cache, self.envs, None)

    def cand(self, gold_body="return a + b", hidden=None, f2p=None, p2p=None, base_extra=None) -> he.Cand:
        base = {"pkg/__init__.py": "", "pkg/mod.py": "def add(a, b):\n    raise NotImplementedError\n\n\n"
                                                     "def other():\n    return 'ok'\n", **(base_extra or {})}
        gold = {"pkg/mod.py": f"def add(a, b):\n    {gold_body}\n\n\ndef other():\n    return 'ok'\n"}
        c = simple_cand("python", "tune", "S", base, hidden or {"tests/test_mod.py": PKG_TEST}, gold, {"repo": "r"},
                        self.key)
        c.f2p = f2p if f2p is not None else ["tests/test_mod.py::test_add_two_numbers"]
        c.p2p = p2p if p2p is not None else ["tests/test_mod.py::test_other_still_works"]
        return c

    def test_validation_keeps_good(self):
        v = self.runner.validate(self.cand())
        self.assertTrue(v.kept, v.why)
        self.assertEqual(v.f2p, ["tests/test_mod.py::test_add_two_numbers"])
        self.assertEqual(v.p2p, ["tests/test_mod.py::test_other_still_works"])
        self.assertGreaterEqual(v.timeout_s, 120)
        self.assertGreater(v.gold_seconds, 0)

    def test_validation_drops_bad(self):
        v = self.runner.validate(self.cand(gold_body="return a - b"))
        self.assertFalse(v.kept)
        self.assertIn("does not pass the gold", v.why)
        # a p2p test that fails on the second gold run: removed from p2p
        counter = self.tmp / "count.txt"
        flaky = PKG_TEST + textwrap.dedent(f'''\


            def test_flaky_on_second_run():
                import pathlib
                p = pathlib.Path({str(counter)!r})
                n = int(p.read_text()) if p.exists() else 0
                p.write_text(str(n + 1))
                assert n != 1
            ''')
        many = flaky + "".join(f"\n\ndef test_stable_{i}():\n    assert True\n" for i in range(12))
        p2p = ["tests/test_mod.py::test_other_still_works", "tests/test_mod.py::test_flaky_on_second_run"] + [
            f"tests/test_mod.py::test_stable_{i}" for i in range(12)]
        v = self.runner.validate(self.cand(hidden={"tests/test_mod.py": many}, p2p=p2p))
        self.assertTrue(v.kept, v.why)
        self.assertIn("tests/test_mod.py::test_flaky_on_second_run", v.dropped_p2p)
        self.assertNotIn("tests/test_mod.py::test_flaky_on_second_run", v.p2p)
        counter.unlink()
        v = self.runner.validate(self.cand(hidden={"tests/test_mod.py": flaky}, p2p=p2p[:2]))
        self.assertFalse(v.kept)  # 1 of 2 p2p tests lost: more than 10%
        self.assertIn("pass-to-pass", v.why)

    def test_src_layout(self):
        tmp = self.tmp / "tree"
        put(tmp, {"src/pkg/__init__.py": ""})
        self.assertEqual(he.import_root(tmp), "src")
        key = self.envs.add("py", {"python": "3.12", "requirements": ["pytest"], "import_root": "src"})
        hidden = {"tests/test_where.py": "import os\nimport pkg\n\n\ndef test_imported_from_work_copy():\n"
                                         "    assert os.path.realpath(pkg.__file__).startswith(os.path.realpath(os.getcwd()))\n"
                                         "    assert pkg.VALUE == 2\n"}
        c = simple_cand("python", "tune", "S", {"src/pkg/__init__.py": "VALUE = 1\n"}, hidden,
                        {"src/pkg/__init__.py": "VALUE = 2\n"}, {"repo": "r"}, key)
        c.f2p, c.p2p = ["tests/test_where.py::test_imported_from_work_copy"], []
        v = self.runner.validate(c)
        self.assertTrue(v.kept, v.why)
        purelib = subprocess.run([str(self.envs.python(key)), "-c",
                                  "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                 capture_output=True, text=True).stdout.strip()
        self.assertFalse((Path(purelib) / "rlmh_src.pth").exists())  # removed after each run


# ----------------------------------------------------------------------------- GPU


class TestGpu(unittest.TestCase):
    def test_gpu_pick(self):
        gpu = load_gpu()

        class FakeRt:
            def __init__(self, n):
                self.n = n

            def devices(self):
                return self.n

        def rig(cuda, hip):  # None: the runtime doesn't load; else its device count
            loaders = {"cuda": (lambda: None) if cuda is None else (lambda: object()),
                       "hip": (lambda: None) if hip is None else (lambda: object())}
            runtimes = {"cuda": lambda lib: type("C", (FakeRt,), {"name": "cuda"})(cuda),
                        "hip": lambda lib: type("H", (FakeRt,), {"name": "hip"})(hip)}
            return mock.patch.multiple(gpu, LOADERS=loaders, RUNTIMES=runtimes)

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RLMH_GPU_BACKEND", None)
            with rig(1, 1):
                self.assertEqual(gpu.pick().name, "cuda")
                self.assertEqual(gpu.probe(), ("cuda", 1))
            with rig(None, 2):
                self.assertEqual(gpu.pick().name, "hip")
            with rig(0, 1):
                self.assertEqual(gpu.pick().name, "hip")
            with rig(0, None):
                self.assertIsNone(gpu.pick())
                self.assertEqual(gpu.probe(), ("none", 0))
                with self.assertRaisesRegex(gpu.GpuError, "no GPU found"):
                    gpu._current = None
                    gpu.backend()
            with rig(1, 1), mock.patch.dict(os.environ, {"RLMH_GPU_BACKEND": "hip"}):
                self.assertEqual(gpu.pick().name, "hip")
            with rig(1, None), mock.patch.dict(os.environ, {"RLMH_GPU_BACKEND": "hip"}):
                self.assertIsNone(gpu.pick())
        forbidden = ("Name", "Uuid", "UUID", "Serial", "PCIBus", "DriverGetVersion", "DeviceProperties",
                     "RuntimeGetVersion", "nvml")
        for group, names in gpu.CALLS.items():
            for n in names:
                self.assertFalse([f for f in forbidden if f in n and n not in ("cuGetErrorName", "hipGetErrorName")],
                                 n)
        with self.assertRaises(gpu.GpuError):
            gpu._bind(object(), "cuda", "cuDeviceGetName")
        src = (GPU_DIR / "gpu.py").read_text(encoding="utf-8")
        for f in ("cuDeviceGetName", "cuDeviceGetUuid", "cuDriverGetVersion", "hipGetDeviceProperties",
                  "hipDeviceGetName", "hipDriverGetVersion", "nvidia-smi"):
            self.assertNotIn(f, src)
        out = subprocess.run([sys.executable, "-I", str(GPU_DIR / "gpu.py"), "--probe", "--backend", "cuda"],
                             capture_output=True, text=True)
        self.assertRegex(out.stdout, r"^backend=(cuda|none) devices=\d+\n$")

    @unittest.skipUnless(HAVE_UV and have_gpu(), "needs uv and a GPU")
    def test_kernel_validation(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(he.rmtree, tmp)
        cache = tmp / "cache"
        put(cache / "kernels" / "_shared", {"gpu.py": (GPU_DIR / "gpu.py").read_bytes(),
                                            "gpu_compat.h": (GPU_DIR / "gpu_compat.h").read_bytes()})
        kernel_tree(cache, "vecadd", "tune", "S")
        envs = he.Envs(cache)
        key = envs.add("kernel", {"python": "3.12", "requirements": ["numpy", "pytest", he.NVRTC], "import_root": "."})
        c = he.kernel_cand(cache, "vecadd")
        c.env = key
        runner = he.Runner(cache, envs, None)
        v = runner.validate(c)
        self.assertTrue(v.kept, v.why)
        self.assertGreaterEqual(v.mutants_caught, 3)
        weak = kernel_tree(cache, "weak", "tune", "S")  # only the n = 1 test: misses most mutants
        (weak / "hidden" / "tests" / "test_vec_add_values.py").write_text(
            "import numpy as np\nimport kernels\n\n\ndef test_vec_add_one():\n"
            "    assert kernels.add(np.ones(1, np.float32), np.zeros(1, np.float32))[0] == 1.0\n")
        c = he.kernel_cand(cache, "weak")
        c.env = key
        v = runner.validate(c)
        self.assertFalse(v.kept)
        self.assertIn("too weak", v.why)


# ----------------------------------------------------------------------------- the guard


class TestGuard(unittest.TestCase):
    def test_outside_repo_guard(self):
        self.assertEqual(he.main(["check", "--data", str(ROOT / "scratch-data")]), 2)
        with tempfile.TemporaryDirectory() as t:
            harness = Path(t) / "harness"
            harness.mkdir()
            self.assertEqual(he.main(["check", "--data", str(harness / "data"), "--harness", str(harness)]), 2)
            self.assertEqual(he.main(["check", "--data", str(Path(t) / "data"), "--harness", str(harness)]), 1)

    def test_web_cache(self):
        # D-235: the web track can live on another drive; the choice is remembered, and it must be outside both repos
        with tempfile.TemporaryDirectory() as t:
            harness, data_ = Path(t) / "harness", Path(t) / "data"
            harness.mkdir()
            self.assertEqual(he.main(["check", "--data", str(data_), "--harness", str(harness),
                                      "--web-cache", str(harness / "web")]), 2)
            paths = he.Paths(data_.resolve(), he.SET, harness.resolve())
            self.assertEqual(paths.web_cache, paths.cache)
            he.resolve_web_root(paths, Path(t) / "ntfs")
            self.assertEqual(paths.web_cache, (Path(t) / "ntfs").resolve() / "build" / he.SET)
            again = he.Paths(data_.resolve(), he.SET, harness.resolve())
            he.resolve_web_root(again, None)
            self.assertEqual(again.web_cache, paths.web_cache)
            runner = he.Runner(paths.cache, he.Envs(paths.cache), None, paths.web_cache / "work")
            with runner.folder("web") as w, runner.folder("python") as p:
                self.assertEqual((w.parent, p.parent), (paths.web_cache / "work", paths.cache / "work"))


if __name__ == "__main__":
    unittest.main()
