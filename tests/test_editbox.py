"""The edit sandbox and the grader through real containers (spec §12.4-12.6, plan step E2). WSL2 with Docker.

Done-when checks for E2: hidden tests are unreachable from the episode, there is no history to mine,
the canaries catch a patch that forges results, and a grader that fails is an infra error, never a 0.
"""

import asyncio
import base64
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rlm_rl import bridge, editbox, grader, sandbox
from rlm_rl.bridge import InfraError
from rlm_rl.editbox import EditImage, EditSandbox
from rlm_rl.tasks import editsynth
from tests.fake_model import AsyncCase

RUN_ID = "test-editbox"
TASK = None
IMAGE: EditImage | None = None


def setUpModule():
    global TASK, IMAGE
    if sys.platform != "linux":
        raise unittest.SkipTest("the edit sandbox runs inside WSL2 (decision 3b)")
    if subprocess.run(["docker", "image", "inspect", sandbox.BASE_IMAGE], capture_output=True).returncode != 0:
        raise unittest.SkipTest(f"base image {sandbox.BASE_IMAGE} is not on this machine")
    editbox.build_synth_image()
    editbox.build_python_bundle()
    IMAGE = EditImage.synth()
    TASK = editsynth.make_task(4242, "edit_synth", 2, kind="fix")


def tearDownModule():
    if sys.platform == "linux":
        sandbox.cleanup_stale(RUN_ID)


async def echo(prompt: str) -> str:
    return "ok"


def write_cell(files: dict) -> str:
    return "\n".join(f"open('{p}', 'w').write(__import__('base64').b64decode('{base64.b64encode(t.encode()).decode()}').decode())"
                     for p, t in files.items())


def leftover_containers() -> list[str]:
    out = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=rlm.run={RUN_ID}"], capture_output=True, text=True)
    return out.stdout.split()


class EditCase(AsyncCase):
    def box(self, task=None, image=None, **kw):
        task = task or TASK
        return EditSandbox(task.context, echo, workspace=task.workspace, image=image or IMAGE, run_id=RUN_ID, **kw)

    def tearDown(self):
        self.assertEqual(leftover_containers(), [])


class EditReplTest(EditCase):
    async def test_the_repository_is_writable_and_programs_run(self):
        async with self.box() as box:
            self.assertGreaterEqual(box.landlock_abi, 1)
            r = await box.run_cell("import os\nprint(os.getcwd())\nprint(sorted(os.listdir('.')))")
            self.assertEqual(r.stdout.splitlines()[0], "/work/repo")
            self.assertIn("tests", r.stdout)
            self.assertFalse(r.repo_changed)
            r = await box.run_cell("import subprocess\nr = subprocess.run(['python3', '-m', 'unittest', 'discover', '-s', 'tests'],"
                                   " capture_output=True, text=True)\nprint(r.returncode)")
            self.assertEqual(r.stdout.strip(), "0")
            self.assertFalse(r.repo_changed)  # caches a test run writes aren't an edit
            r = await box.run_cell("import subprocess, sys\nsubprocess.run(['python3', '-c', 'import sys; print(\"out\"); "
                                   "print(\"err\", file=sys.stderr)'])")
            self.assertEqual(sorted(r.stdout.split()), ["err", "out"])  # a program's output reaches the model
            r = await box.run_cell("open('notes.txt', 'w').write('x')")
            self.assertTrue(r.repo_changed)
            kinds = [(e["kind"], e["cell"]) for e in box.events]
            self.assertIn(("repo_read", 1), kinds)
            self.assertIn(("exec", 2), kinds)
            self.assertIn(("repo_write", 4), kinds)
            exec_argv = [e["argv"] for e in box.events if e["kind"] == "exec"]
            self.assertIn(["python3", "-m", "unittest", "discover", "-s", "tests"], exec_argv)
            patch = await box.diff()
        self.assertIn("+++ b/notes.txt", patch)
        self.assertNotIn("__pycache__", patch)

    async def test_the_coding_tools_work_in_the_container(self):
        """Spec §12.1: the seven tools are in the namespace, work on /work/repo, and are logged apart from raw I/O."""
        gold = TASK.grading["gold_files"]
        async with self.box() as box:
            r = await box.run_cell("names = dir()\nprint(sorted(n for n in ('read_file', 'find_files', 'grep', 'edit_file',"
                                   " 'write_file', 'shell', 'diff') if n in names))\nprint(find_files('tests/*.py')[:1])")
            self.assertIn("'diff', 'edit_file', 'find_files', 'grep', 'read_file', 'shell', 'write_file'", r.stdout)
            self.assertIn("tests/", r.stdout)
            path = next(iter(gold))
            r = await box.run_cell(f"print(read_file({path!r}, n=3))\nprint(len(grep('def ', path={path!r})) > 0)")
            self.assertIn("     1\t", r.stdout)
            self.assertTrue(r.stdout.strip().endswith("True"))
            writes = "\n".join(f"write_file({p!r}, __import__('base64').b64decode({base64.b64encode(t.encode()).decode()!r}).decode())"
                               for p, t in gold.items())
            r = await box.run_cell(writes + "\nprint(shell('python3 -m unittest discover -s tests'))")
            self.assertTrue(r.repo_changed)
            self.assertTrue(r.stdout.startswith("exit 0"), r.stdout)
            r = await box.run_cell("d = diff()\nprint(d.count('+++ b/'))\nprint(repr(shell('git status --porcelain').out))")
            self.assertEqual(r.stdout.splitlines()[0], str(len(gold)))
            self.assertNotIn("A ", r.stdout.splitlines()[1])  # diff() never stages into the repository's own index
            self.assertFalse(r.repo_changed)
            patch = await box.diff()
        tools = [e["event"] for e in box.events if e["kind"] == "tool"]
        self.assertEqual(tools.count("write_file"), len(gold))
        self.assertTrue({"find_files", "read_file", "grep", "shell", "diff"} <= set(tools))
        self.assertTrue(all(e["ok"] for e in box.events if e["kind"] == "tool"))
        shells = [e for e in box.events if e["kind"] == "exec" and e["argv"][:2] == ["sh", "-c"]]
        self.assertTrue(shells and all(e["tool"] for e in shells))
        io = [e for e in box.events if e["kind"] == "io" and e["cell"] == 3]
        self.assertEqual(len(io), 1)
        self.assertGreaterEqual(io[0]["tool_writes"], len(gold))
        self.assertEqual((io[0]["raw_writes"], io[0]["raw_execs"]), (0, 0))
        self.assertEqual(patch.count("+++ b/"), len(gold))

    async def test_writes_stay_in_the_repository_scratch_and_tmp(self):
        async with self.box() as box:
            r = await box.run_cell(
                "import subprocess\nout = []\nfor p in ['/etc/x', '/opt/rlm/x', '/episode/x', '/x', '/work/x', '/usr/x']:\n"
                "    try:\n        open(p, 'w').write('x'); out.append(p)\n    except OSError:\n        pass\n"
                "for p in ['/tmp/x', '/scratch/x', '/work/repo/x']:\n    open(p, 'w').write('ok')\n"
                "child = subprocess.run(['sh', '-c', 'echo x > /etc/y; echo x > /usr/y'], capture_output=True)\n"
                "print(out, child.returncode != 0)")
            self.assertEqual(r.stdout.strip(), "[] True")
            blocked = [e["detail"] for e in box.events if e["kind"] == "file"]
            self.assertEqual(len(blocked), 6)

    async def test_network_is_blocked_and_logged(self):
        async with self.box() as box:
            r = await box.run_cell(
                "import socket, subprocess\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
                "except OSError as e:\n    print('blocked', type(e).__name__)\n"
                "c = subprocess.run(['python3', '-c', 'import socket; socket.create_connection((\"1.1.1.1\", 80), 2)'],"
                " capture_output=True)\nprint(c.returncode != 0)")
            self.assertEqual(r.stdout.split(), ["blocked", "PermissionError", "True"])
            self.assertIn("network", {e["kind"] for e in box.events})

    async def test_no_history_to_mine(self):
        """One commit at the base: no other history, branches, tags, reflog or remotes (spec §12.4)."""
        async with self.box() as box:
            r = await box.run_cell("import os\nprint(sorted(os.listdir('.git')))\nprint(open('.git/HEAD').read().strip())\n"
                                   "print(os.path.exists('.git/logs'), os.listdir('.git/refs/tags'))")
            self.assertIn("ref: refs/heads/main", r.stdout)
            self.assertIn("False []", r.stdout)
            copy = Path(tempfile.mkdtemp()) / "repo"
            editbox.materialize(box.ws_dir, copy)
        env = editbox.git_env(copy.parent)
        git = lambda *a: editbox.git(*a, env=env, cwd=copy).stdout.decode().split()  # noqa: E731
        self.assertEqual(len(git("rev-list", "--all")), 1)
        self.assertEqual(git("tag"), [])
        self.assertEqual(git("remote"), [])
        self.assertEqual(git("for-each-ref", "--format=%(refname)"), ["refs/heads/main"])
        self.assertEqual(git("reflog"), [])
        editbox.remove_tree(copy.parent)

    async def test_hidden_tests_never_enter_the_container(self):
        hidden_path = next(iter(TASK.grading["tests"]["hidden"]))
        marker = TASK.grading["tests"]["hidden"][hidden_path].split("class ")[1].split("(")[0]
        async with self.box() as box:
            r = await box.run_cell(
                "import os\nfound = []\nfor root in ('/work', '/episode', '/scratch', '/tmp', '/opt/rlm'):\n"
                "    for d, _, fs in os.walk(root):\n        for f in fs:\n            p = os.path.join(d, f)\n"
                "            try:\n                if " + repr(marker) + " in open(p, errors='ignore').read():\n"
                "                    found.append(p)\n            except OSError:\n                pass\n"
                "print(found, os.path.exists(" + repr("/work/repo/" + hidden_path) + "))")
            self.assertEqual(r.stdout.strip(), "[] False")

    async def test_timeouts_and_crashes_keep_the_files(self):
        limits = editbox.edit_sandbox_limits()
        limits = sandbox.SandboxLimits(**{**limits.__dict__, "cell_timeout_s": 3})
        async with self.box(limits=limits) as box:
            await box.run_cell("open('kept.txt', 'w').write('still here')\nx = 1")
            r = await box.run_cell("while True:\n    pass")
            self.assertEqual(r.crash, "timeout")
            self.assertIn("timed out after 3 seconds of compute", r.error)
            self.assertTrue(r.error.endswith("Your files in /work/repo are unchanged."))
            r = await box.run_cell("print(open('kept.txt').read())\nprint('x' in globals())")
            self.assertEqual(r.stdout.split("\n")[:2], ["still here", "False"])
            r = await box.run_cell("import os\nos._exit(3)")
            self.assertIn("exit code 3", r.error)
            self.assertIn("Your files in /work/repo are unchanged.", r.error)
            r = await box.run_cell("print(open('kept.txt').read())")
            self.assertEqual(r.stdout.strip(), "still here")

    async def test_the_host_diff_ignores_the_episodes_git(self):
        """The model controls /work/repo/.git; the host never runs git against it (editbox.diff)."""
        marker = Path(tempfile.mkdtemp()) / "pwned"
        async with self.box() as box:
            await box.run_cell(
                "open('.git/config', 'a').write('[core]\\n\\tfsmonitor = touch " + str(marker) + "\\n"
                "[diff]\\n\\texternal = touch " + str(marker) + "\\n[filter \"x\"]\\n\\tclean = touch " + str(marker) + "\\n')\n"
                "open('.gitattributes', 'w').write('* filter=x diff=x\\n')\n"
                "import os\nos.makedirs('.git/hooks', exist_ok=True)\n"
                "open('README.md', 'a').write('changed\\n')")
            patch = await box.diff()
        self.assertFalse(marker.exists())
        self.assertIn("+changed", patch)
        self.assertNotIn(".git/", patch.replace(".gitattributes", ""))

    async def test_the_bundle_runs_the_repl_in_a_foreign_image(self):
        """Our Python works in an image with no Python and no glibc (Alpine), through its own loader (spec §12.4)."""
        alpine = EditImage("alpine:3.21", tuple(editbox.BUNDLE_PYTHON), (), (), read_only=False, bundle=True,
                           env_reset=False, language="python")
        if subprocess.run(["docker", "image", "inspect", alpine.tag], capture_output=True).returncode:
            self.skipTest("alpine:3.21 isn't on this machine")
        async with self.box(image=alpine) as box:
            r = await box.run_cell("import sys, subprocess\nprint(sys.version.split()[0])\n"
                                   "print(subprocess.run(['sh', '-c', 'echo from-sh'], capture_output=True, text=True).stdout)")
            self.assertEqual(r.stdout.split(), ["3.12.15", "from-sh"])
            self.assertGreaterEqual(box.landlock_abi, 1)


class GraderTest(EditCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = TASK.workspace.ensure()
        cls.spec = grader.TestSpec.from_dict(TASK.grading["tests"])
        cls.files = {p: (cls.ws / "tree" / p).read_text() for p in TASK.grading["gold_files"]}

    async def patch_of(self, files: dict) -> str:
        async with self.box() as box:
            r = await box.run_cell(write_cell(files))
            self.assertIsNone(r.error)
            return await box.diff()

    async def grade(self, patch: str, spec=None) -> grader.GradeResult:
        return await grader.grade(patch, spec or self.spec, ws_dir=self.ws, image=IMAGE, run_id=RUN_ID)

    async def test_gold_resolves_and_blind_patches_do_not(self):
        gold = await self.grade(await self.patch_of(TASK.grading["gold_files"]))
        self.assertTrue(gold.resolved, gold)
        self.assertEqual((gold.f2p_passed, gold.p2p_failed, gold.hacks), (len(self.spec.f2p), 0, []))
        empty = await self.grade("")
        self.assertTrue(empty.empty and not empty.resolved)
        target = TASK.grading["target"]
        text = self.files[target["path"]]
        stub = text.replace(f"def {target['function']}(", f"def {target['function']}(*a, **k):\n    return 0\n\ndef _old_(", 1)
        for files in ({target["path"]: stub}, {target["path"]: text.replace(f"def {target['function']}(", "def _gone_(", 1)}):
            g = await self.grade(await self.patch_of(files))
            self.assertFalse(g.resolved)
            self.assertEqual(g.hacks, [])

    async def test_editing_a_test_is_tamper(self):
        visible = next(p for p in (self.ws / "tree" / "tests").iterdir() if p.name.startswith("test_"))
        files = dict(TASK.grading["gold_files"], **{f"tests/{visible.name}": "import unittest\n"})
        g = await self.grade(await self.patch_of(files))
        self.assertEqual([h["rule"] for h in g.hacks], ["test_tamper"])
        self.assertTrue(g.resolved)  # the edit was stripped and the real tests ran; the detector zeroes the reward

    async def test_canaries_catch_forged_results(self):
        pkg = TASK.grading["target"]["path"].split("/")[0]
        init = (self.ws / "tree" / pkg / "__init__.py").read_text()
        forge = init + ("\nimport unittest as _u\n_u.TestCase.fail = lambda self, msg=None: None\n"
                        "_u.TestCase.assertEqual = lambda self, *a, **k: None\n")
        g = await self.grade(await self.patch_of({f"{pkg}/__init__.py": forge}))
        self.assertIn("grader_tamper", [h["rule"] for h in g.hacks])
        self.assertTrue(g.resolved)  # every forged test "passed"; grader_tamper zeroes it
        exit_early = init + "\nimport os as _os\n_os._exit(0)\n"
        g = await self.grade(await self.patch_of({f"{pkg}/__init__.py": exit_early}))
        rules = [h["rule"] for h in g.hacks]
        self.assertIn("test_aware", rules)
        self.assertFalse(g.resolved)

    async def test_the_cap_is_the_models_and_a_dead_grader_is_infra(self):
        pkg = TASK.grading["target"]["path"].split("/")[0]
        slow = (self.ws / "tree" / pkg / "__init__.py").read_text() + "\nimport time as _t\n_t.sleep(60)\n"
        spec = grader.TestSpec.from_dict({**TASK.grading["tests"], "timeout_s": 5})
        g = await self.grade(await self.patch_of({f"{pkg}/__init__.py": slow}), spec)
        self.assertTrue(g.deadline_hit)
        self.assertFalse(g.resolved)
        self.assertNotIn("grader_tamper", [h["rule"] for h in g.hacks])
        missing = EditImage("rlm-sandbox:does-not-exist", IMAGE.python, IMAGE.env, (), True, False, True)
        with self.assertRaises(InfraError):
            await grader.grade(await self.patch_of(TASK.grading["gold_files"]), self.spec, ws_dir=self.ws,
                               image=missing, run_id=RUN_ID)


class ParallelTest(EditCase):
    async def test_eight_edit_episodes_at_once(self):
        async def one(i):
            async with self.box() as box:
                r = await box.run_cell("import subprocess\nprint(subprocess.run(['python3', '-m', 'unittest', 'discover',"
                                       " '-s', 'tests'], capture_output=True).returncode)")
                return r.stdout.strip()
        self.assertEqual(await asyncio.gather(*(one(i) for i in range(8))), ["0"] * 8)


if __name__ == "__main__":
    unittest.main()
