"""H2-H4 and the sandbox side of H5, checked against real containers.

Run inside WSL2 with Docker Desktop's WSL integration on, from the repo root:

    python3 -m unittest -v tests.test_sandbox

Each check runs model-style code through Sandbox.run_cell, the path every episode takes.
LayersTest also probes a container with the same flags but no audit hook, to show that the
container and Landlock hold on their own. Probes never print what they manage to read: an
attempt that works is recorded as "ALLOWED" and fails the test.
"""

import asyncio
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

from rlm_rl import bridge, sandbox
from rlm_rl.bridge import InfraError, SubcallError
from rlm_rl.sandbox import Sandbox, SandboxLimits
from rlm_rl.tasks import new_seed

REPO = Path(__file__).resolve().parents[1]
RUN_ID = f"test-{secrets.token_hex(4)}"
LIMITS = SandboxLimits(cell_timeout_s=20)
CONTEXT = "first line\r\nsecond line, café ✓\nthird line\n" + "\n".join(f"row {i}" for i in range(100))
IMAGE = ""

PRELUDE = r'''
import json as _json
_r = {}
def attempt(name, fn):
    try:
        fn()
        _r[name] = "ALLOWED"
    except BaseException as e:
        _r[name] = f"{type(e).__name__}: {e}"[:160]
'''
FINISH = "\nsubmit(_json.dumps(_r))\n"


def setUpModule():
    global IMAGE
    if sys.platform != "linux":
        raise unittest.SkipTest("the sandbox runs inside WSL2 (decision 3b)")
    if shutil.which("docker") is None:
        raise unittest.SkipTest("no docker CLI: turn on Docker Desktop's WSL integration for Ubuntu")
    if subprocess.run(["docker", "image", "inspect", sandbox.BASE_IMAGE], capture_output=True).returncode != 0:
        raise unittest.SkipTest(f"base image {sandbox.BASE_IMAGE} is not on this machine")
    IMAGE = sandbox.build_image()


def leftover_containers() -> list[str]:
    out = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=rlm.run={RUN_ID}"], capture_output=True, text=True)
    return out.stdout.split()


async def echo(prompt: str) -> str:
    return f"echo:{prompt}"


class SandboxCase(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        # Every episode, however it ended, leaves no container and no episode directory behind (3a).
        self.assertEqual(leftover_containers(), [])

    def box(self, context=CONTEXT, complete=echo, limits=LIMITS, **kw) -> Sandbox:
        return Sandbox(context, complete, limits=limits, image=IMAGE, run_id=RUN_ID, **kw)

    async def cells(self, *codes, **kw):
        box = self.box(**kw)
        async with box:
            results = [await box.run_cell(code) for code in codes]
        self.assertFalse(box.dir.exists())
        return results, box

    async def probe(self, body: str, **kw) -> tuple[dict, Sandbox]:
        (result,), box = await self.cells(PRELUDE + body + FINISH, **kw)
        self.assertIsNone(result.error, result.error)
        self.assertTrue(result.submitted)
        return json.loads(result.answer), box

    def assertAllBlocked(self, results: dict):
        allowed = [name for name, outcome in results.items() if outcome == "ALLOWED"]
        self.assertEqual(allowed, [], f"these should have failed: {allowed}")


class NetworkTest(SandboxCase):
    """H4: pip install, curl and DNS fail; the bridge socket is the only way out."""

    async def test_no_network(self):
        results, box = await self.probe(r'''
import os, runpy, socket, subprocess, urllib.request
def system(cmd):
    assert os.system(cmd) == 0
attempt("tcp 1.1.1.1:80", lambda: socket.create_connection(("1.1.1.1", 80), timeout=3))
attempt("tcp6", lambda: socket.create_connection(("2606:4700:4700::1111", 80), timeout=3))
attempt("udp 8.8.8.8:53", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("8.8.8.8", 53)))
attempt("dns", lambda: socket.getaddrinfo("example.com", 443))
attempt("dns gethostbyname", lambda: socket.gethostbyname("pypi.org"))
attempt("http", lambda: urllib.request.urlopen("http://example.com", timeout=3))
attempt("https pypi", lambda: urllib.request.urlopen("https://pypi.org/simple/", timeout=3))
attempt("import pip", lambda: runpy.run_module("pip", run_name="__main__"))
attempt("pip install", lambda: subprocess.run(["pip", "install", "requests"], check=True))
attempt("python -m pip install", lambda: subprocess.run(["python3", "-m", "pip", "install", "requests"], check=True))
attempt("curl", lambda: subprocess.run(["curl", "-sS", "https://example.com"], check=True))
attempt("os.system curl", lambda: system("curl -sS https://example.com"))
attempt("docker socket", lambda: socket.socket(socket.AF_UNIX).connect("/var/run/docker.sock"))
attempt("fork", lambda: os.fork())
''')
        self.assertAllBlocked(results)
        kinds = {e["kind"] for e in box.events}
        self.assertTrue({"network", "process"} <= kinds, kinds)

    async def test_bridge_is_the_way_out(self):
        (result,), box = await self.cells('print(llm("ping")); print(llm_batch(["a", "b"]))')
        self.assertIsNone(result.error, result.error)
        self.assertEqual(result.stdout, "echo:ping\n['echo:a', 'echo:b']\n")
        self.assertEqual(box.bridge.subcalls_used, 3)
        self.assertEqual(box.events, [])


class FilesTest(SandboxCase):
    """H2: .env, .art, the repo, home and /proc are unreadable; nothing outside /scratch is writable."""

    async def test_host_files_and_proc_are_unreadable(self):
        home = os.path.expanduser("~")
        reads = [f"{REPO}/.env", f"{REPO}/.art", str(REPO), f"{REPO}/rlm_rl/tasks/__init__.py", home, f"{home}/.ssh",
                 "/root", "/home", "/mnt/f", "/mnt/c/Users", "/mnt/wsl", "/run/desktop/mnt/host/f", "/host_mnt",
                 "F:/Personal/RLM/RLM/.env", "/proc", "/proc/self/environ", "/proc/1/environ", "/proc/self/mountinfo",
                 "/proc/1/cmdline", "/sys/class/net", "/etc/passwd", "/etc/hosts", "/var/run/docker.sock", "/dev",
                 "/tmp", "/scratch/../proc/self/status", "/episode/../proc/self/environ"]
        results, box = await self.probe(f"import os\nREADS = {reads!r}\n" + r'''
for p in READS:
    attempt("read " + p, lambda p=p: os.listdir(p) if os.path.isdir(p) else open(p, "rb").read(1))
attempt("os.open /proc/self/environ", lambda: os.open("/proc/self/environ", os.O_RDONLY))
attempt("symlink to /proc", lambda: os.symlink("/proc/self/environ", "/scratch/link"))
''')
        self.assertAllBlocked(results)
        self.assertIn("file", {e["kind"] for e in box.events})

    async def test_writes_only_in_scratch(self):
        writes = ["/x", "/tmp/x", "/etc/x", "/opt/rlm/x", "/episode/x", "/episode/context.txt", "/dev/shm/x",
                  "/var/tmp/x", "/home/x", "/usr/local/lib/python3.12/x.py", "/scratch/../tmp/x", "/dev/x"]
        results, _ = await self.probe(f"import os\nWRITES = {writes!r}\n" + r'''
for p in WRITES:
    attempt("write " + p, lambda p=p: open(p, "w").write("x"))
attempt("mkdir /tmp/d", lambda: os.mkdir("/tmp/d"))
attempt("remove context", lambda: os.remove("/episode/context.txt"))
attempt("rename context", lambda: os.rename("/episode/context.txt", "/scratch/c.txt"))
''')
        self.assertAllBlocked(results)
        (result,), _ = await self.cells(
            "import os, tempfile\n"
            "open('/scratch/a.txt', 'w').write('kept')\n"
            "os.mkdir('/scratch/d')\n"
            "with tempfile.NamedTemporaryFile('w', delete=False) as f: f.write('t')\n"
            "print(open('/scratch/a.txt').read(), f.name.startswith('/scratch/'), sorted(os.listdir('/episode')))")
        self.assertIsNone(result.error, result.error)
        self.assertEqual(result.stdout, "kept True ['bridge.sock', 'context.txt']\n")


class ImportsAndSecretsTest(SandboxCase):
    """H3 and keys: project modules can't be imported, the environment is empty, and no seed is visible."""

    async def test_project_modules_are_not_importable(self):
        modules = ["rlm_rl", "rlm_rl.bridge", "rlm_rl.tasks", "scripts", "art", "wandb", "openai", "dotenv", "httpx",
                   "ticket_tasks", "rlm_env", "repl_worker", "ctypes", "numpy", "pandas"]
        results, box = await self.probe(f"import importlib, sys\nMODULES = {modules!r}\n" + r'''
for m in MODULES:
    attempt("import " + m, lambda m=m: importlib.import_module(m))
    attempt("import statement " + m, lambda m=m: exec("import " + m, {}))
_r["sys.path"] = sys.path
''')
        sys_path = results.pop("sys.path")
        self.assertAllBlocked(results)
        self.assertFalse([p for p in sys_path if "rlm" in p.lower() or p in ("", ".", "/scratch")], sys_path)
        kinds = {e["kind"] for e in box.events}
        self.assertTrue({"host_import", "ctypes"} <= kinds, kinds)

    async def test_no_keys_or_seed_inside(self):
        canary = f"canary-{secrets.token_hex(8)}"
        saved = {k: os.environ.get(k) for k in ("WANDB_API_KEY", "OPENAI_API_KEY", "RLM_TEST_SECRET")}
        os.environ.update(WANDB_API_KEY=canary, OPENAI_API_KEY=canary, RLM_TEST_SECRET=canary)
        seed = new_seed()
        rng = random.Random(seed)  # a stand-in generator: only its output enters the sandbox
        context = "\n".join(f"[msg {i}] {rng.choice(['refund', 'late parcel', 'login bug', 'thanks'])}" for i in range(200))
        try:
            results, _ = await self.probe(r'''
import os, socket, sys
_r["environ"] = dict(os.environ)
_r["argv"] = [sys.argv, getattr(sys, "orig_argv", [])]
_r["host"] = [socket.gethostname(), os.uname().nodename, os.getcwd(), os.getuid()]
_r["episode"] = sorted(os.listdir("/episode"))
_r["globals"] = sorted(globals())
attempt("read /proc/self/environ", lambda: open("/proc/self/environ", "rb").read(1))
attempt("read /proc/1/environ", lambda: open("/proc/1/environ", "rb").read(1))
''', context=context)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        dump = json.dumps(results)
        self.assertNotIn(canary, dump)
        self.assertLessEqual(set(results["environ"]), {"PATH", "LANG", "HOME", "TMPDIR"})
        self.assertFalse([k for k in results["environ"] if any(h in k for h in ("KEY", "TOKEN", "SECRET", "WANDB"))])
        self.assertEqual(results["host"][3], sandbox.SANDBOX_UID)
        self.assertEqual(results["episode"], ["bridge.sock", "context.txt"])
        for form in (str(seed), format(seed, "x"), format(seed, "X")):
            self.assertNotIn(form, dump)
            self.assertNotIn(form, context)
        self.assertNotEqual(results["read /proc/self/environ"], "ALLOWED")
        self.assertNotEqual(results["read /proc/1/environ"], "ALLOWED")

    async def test_stdlib_works_without_tripping_the_audit_log(self):
        (result,), box = await self.cells(r'''
import re, json, collections, itertools, statistics, ast, decimal, fractions, datetime, zoneinfo, csv, io
import hashlib, gzip, zlib, bz2, lzma, textwrap, difflib, heapq, bisect, uuid, tempfile, sqlite3, html, xml.etree.ElementTree
from concurrent.futures import ThreadPoolExecutor
lines = context.splitlines()
counts = collections.Counter(w for line in lines for w in re.findall(r"[a-z]+", line))
when = datetime.datetime(2026, 10, 10, tzinfo=zoneinfo.ZoneInfo("Europe/London"))
db = sqlite3.connect(":memory:"); db.execute("create table t(x)"); db.execute("insert into t values (1)")
tree = ast.parse("def f(): return g()")
with ThreadPoolExecutor(4) as pool:
    answers = list(pool.map(llm, ["a", "b", "c"]))
print(len(lines), counts["row"], statistics.mean([1, 2, 3]), decimal.Decimal("1.10") + 1, when.utcoffset(),
      db.execute("select sum(x) from t").fetchone()[0], type(tree).__name__, answers, len(uuid.uuid4().hex))
''')
        self.assertIsNone(result.error, result.error)
        self.assertEqual(result.stdout, "103 100 2 2.10 1:00:00 1 Module ['echo:a', 'echo:b', 'echo:c'] 32\n")
        self.assertEqual(box.events, [], "normal standard-library use must not look like an escape attempt")


class ReplTest(SandboxCase):
    """The bridge API inside the sandbox (spec §1.1-1.2)."""

    async def test_context_arrives_exactly(self):
        (result,), _ = await self.cells("print(len(context), repr(context[:40]))")
        self.assertEqual(result.stdout, f"{len(CONTEXT)} {CONTEXT[:40]!r}\n")

    async def test_variables_persist_and_output_is_truncated(self):
        results, _ = await self.cells("x = 41", "x += 1; print(x)", "print('y' * 9000)")
        self.assertEqual(results[1].stdout, "42\n")
        self.assertEqual((len(results[2].stdout), results[2].stdout_chars), (bridge.OUTPUT_MAX_CHARS, 9001))

    async def test_budget_and_errors_reach_the_model(self):
        async def picky(prompt):
            if prompt == "bad":
                raise SubcallError("the API rejected the request (400)")
            return "ok"
        results, box = await self.cells('llm_batch(["a"] * 6)', 'llm("bad")', 'print(llm("c"))',
                                        'llm_batch(["d", "e", "f", "g"])', complete=picky, max_subcalls=5)
        self.assertIn("RuntimeError: sub-call budget exhausted (5 total)", results[0].error)
        self.assertIn("RuntimeError: sub-call failed: the API rejected the request (400)", results[1].error)
        self.assertEqual(results[2].stdout, "ok\n")
        self.assertIn("sub-call budget exhausted (5 total)", results[3].error)
        self.assertEqual(box.bridge.subcalls_used, 2)
        self.assertIsNone(box.infra_error)

    async def test_submit_keeps_types(self):
        cases = {"submit(42)": 42, "submit(2.5)": 2.5, "submit(' host-7 ')": " host-7 ", "submit([3, 'a', 1.5])": [3, "a", 1.5],
                 "submit({'b', 'a'})": ["a", "b"], "submit({'k': 1})": "{'k': 1}", "submit(1); submit(2)": 2}
        results, _ = await self.cells(*cases, "submit('x' * 200_000)")
        for (code, expected), result in zip(cases.items(), results):
            self.assertTrue(result.submitted, code)
            self.assertEqual(result.answer, expected, code)
        self.assertFalse(results[-1].submitted)
        self.assertIn("answer too large", results[-1].error)

    async def test_stdin_and_exit_do_not_hang_or_crash(self):
        results, box = await self.cells("input()", "import sys; sys.exit(3)", "print('alive')")
        self.assertIn("EOFError", results[0].error)
        self.assertIn("SystemExit: 3", results[1].error)
        self.assertEqual(results[2].stdout, "alive\n")
        self.assertEqual(box.restarts, 0)


class FailureTest(SandboxCase):
    """H5: model-caused failures are the model's outcome; host-side failures raise InfraError, never a wrong answer."""

    async def test_timeout_restarts_and_tells_the_model(self):
        results, box = await self.cells("x = 1", "while True: pass", "print(x)", "print(len(context))",
                                        limits=SandboxLimits(cell_timeout_s=2))
        self.assertEqual(results[1].crash, "timeout")
        self.assertEqual(results[1].error, box.messages.timeout)
        self.assertIn("timed out after 2 seconds of compute", results[1].error)  # prompt v2 quotes the limit (spec §12.2)
        self.assertIn("NameError", results[2].error)
        self.assertEqual(results[3].stdout, f"{len(CONTEXT)}\n")
        self.assertEqual((box.timeouts, box.restarts), (1, 1))

    async def test_time_blocked_on_subcalls_does_not_count(self):
        async def slow(prompt):
            await asyncio.sleep(3)
            return "late"
        (result,), _ = await self.cells('print(llm("x"))', complete=slow, limits=SandboxLimits(cell_timeout_s=2))
        self.assertIsNone(result.crash)
        self.assertEqual(result.stdout, "late\n")
        self.assertGreaterEqual(result.blocked_s, 2.9)
        self.assertLess(result.compute_s, 2)

    async def test_model_crashes_are_not_infra(self):
        results, box = await self.cells("import os; os._exit(3)", "print('back')",
                                        "b = b'x' * (400 * 1024 * 1024)", "print('back again')",
                                        limits=SandboxLimits(cell_timeout_s=20, memory_mb=128))
        self.assertEqual(results[0].crash, "exit code 3")
        self.assertEqual(results[1].stdout, "back\n")
        self.assertTrue((results[2].crash or "").startswith("out of memory") or "MemoryError" in (results[2].error or ""),
                        (results[2].crash, results[2].error))
        self.assertEqual(results[3].stdout, "back again\n")
        self.assertIsNone(box.infra_error)

    async def test_self_inflicted_kills_are_the_models(self):
        # A REPL that kills itself must not pass for an infra error, which would drop a failing episode from training.
        box = self.box()
        async with box:
            results = [await box.run_cell(code) for code in (
                "import os; os._exit(137)",
                "import signal; signal.raise_signal(signal.SIGKILL)",
                "import os, signal; os.kill(1, signal.SIGKILL)",
                "import os, threading; threading.Timer(0.5, lambda: os._exit(9)).start()")]
            await asyncio.sleep(2)
            results.append(await box.run_cell("print('after the timer')"))
        self.assertEqual(results[0].crash, "exit code 137")
        self.assertEqual(results[1].crash, "killed by SIGKILL")
        self.assertIn("Permission denied (sandbox)", results[2].error)  # the supervisor can't be signalled
        self.assertIsNone(results[3].crash)
        self.assertEqual(results[4].crash, "exit code 9")  # died between cells, from the thread it left running
        self.assertIn("This cell was not run.", results[4].error)
        self.assertIsNone(box.infra_error)

    async def test_tampering_with_the_audit_log_is_caught(self):
        results, box = await self.cells("import os; os.dup2(os.open('/dev/null', os.O_WRONLY), 2); open('/etc/passwd')",
                                        "print('ok')")
        self.assertEqual(results[0].crash, "the sandbox log was tampered with")
        self.assertIn("tamper", {e["kind"] for e in box.events})
        self.assertEqual(results[1].stdout, "ok\n")

    async def test_subcall_infra_error_discards_the_episode(self):
        async def exhausted(prompt):
            raise InfraError("429 after 6 retries")
        box = self.box(complete=exhausted)
        async with box:
            with self.assertRaises(InfraError):
                await box.run_cell('try:\n    llm("x")\nexcept Exception:\n    pass\nsubmit(7)')
            with self.assertRaises(InfraError):
                await box.run_cell("print(1)")
        self.assertEqual(box.infra_error, "429 after 6 retries")

    async def test_host_bug_in_a_subcall_is_infra(self):
        async def buggy(prompt):
            raise KeyError("choices")
        box = self.box(complete=buggy)
        async with box:
            with self.assertRaises(InfraError):
                await box.run_cell('llm("x")')

    async def test_container_that_cannot_start_is_infra(self):
        box = Sandbox(CONTEXT, echo, image="rlm-sandbox:does-not-exist", run_id=RUN_ID)
        with self.assertRaises(InfraError):
            await box.start()
        self.assertFalse(box.dir.exists())

    async def test_killed_from_outside_is_infra(self):
        box = self.box()
        async with box:
            await box.run_cell("x = 1")
            subprocess.run(["docker", "kill", box._names[-1]], capture_output=True, check=True)
            await asyncio.sleep(1)
            with self.assertRaises(InfraError):
                await box.run_cell("print(x)")
        box = self.box()
        async with box:
            async def kill_soon():
                await asyncio.sleep(2)
                subprocess.run(["docker", "kill", box._names[-1]], capture_output=True, check=True)
            killer = asyncio.create_task(kill_soon())
            with self.assertRaises(InfraError):
                await box.run_cell("import time; time.sleep(30)")
            await killer


class LayersTest(SandboxCase):
    """The container and Landlock on their own, without the audit hook, in a container with the episode's flags."""

    async def test_container_and_landlock_without_the_hook(self):
        code = r'''
import json, os, socket, subprocess, sys
r = {"uid": os.getuid(), "interfaces": sorted(os.listdir("/sys/class/net")), "env": sorted(os.environ)}
def t(name, fn):
    try:
        fn()
        r[name] = "ALLOWED"
    except BaseException as e:
        r[name] = f"{type(e).__name__}: {e}"[:120]
t("tcp", lambda: socket.create_connection(("1.1.1.1", 80), timeout=3))
t("dns", lambda: socket.getaddrinfo("example.com", 80))
t("write /", lambda: open("/x", "w"))
t("write /etc", lambda: open("/etc/x", "w"))
t("write /episode", lambda: open("/episode/x", "w"))
sys.path.insert(0, "/opt/rlm")
import repl_worker
r["landlock"] = repl_worker.apply_landlock()
t("read /proc/self/environ", lambda: os.open("/proc/self/environ", os.O_RDONLY))
t("list /proc", lambda: os.listdir("/proc"))
t("list /sys", lambda: os.listdir("/sys/class/net"))
t("list /dev", lambda: os.listdir("/dev"))
t("write /dev/shm", lambda: open("/dev/shm/x", "w"))
t("exec a program", lambda: subprocess.run(["/usr/local/bin/python3", "-c", "pass"], check=True))
t("read context", lambda: open("/episode/context.txt").read(1))
t("write scratch", lambda: open("/scratch/a", "w").write("x"))
t("write /dev/null", lambda: open("/dev/null", "w").write("x"))
t("read stdlib", lambda: open(os.__file__).read(1))
t("load an extension module", lambda: __import__("decimal").Decimal("1.5") + __import__("sqlite3").connect(":memory:").execute("select 1").fetchone()[0])
t("bridge socket", lambda: socket.socket(socket.AF_UNIX).connect("/episode/bridge.sock"))
print(json.dumps(r))
'''
        box = self.box()
        async with box:  # gives us an episode directory with a live bridge socket
            name = f"rlm-layers-{box.episode_id}"
            entry = ["/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8",
                     "/usr/local/bin/python3", "-I", "-B", "-c", code]
            args = sandbox.run_args("docker", name, IMAGE, box.dir, LIMITS, RUN_ID, entrypoint=entry)
            proc = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.DEVNULL,
                                                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await asyncio.wait_for(proc.communicate(), 60)
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        self.assertEqual(proc.returncode, 0, err.decode())
        r = json.loads(out)
        self.assertEqual(r["uid"], sandbox.SANDBOX_UID)
        self.assertEqual(r["interfaces"], ["lo"])  # --network none
        self.assertEqual(r["env"], ["LANG", "PATH"])
        self.assertGreaterEqual(r["landlock"], 1)
        blocked = ["tcp", "dns", "write /", "write /etc", "write /episode", "read /proc/self/environ", "list /proc",
                   "list /sys", "list /dev", "write /dev/shm", "exec a program"]
        allowed = ["read context", "write scratch", "write /dev/null", "read stdlib", "load an extension module",
                   "bridge socket"]
        self.assertEqual([k for k in blocked if r[k] == "ALLOWED"], [], r)
        self.assertEqual([k for k in allowed if r[k] != "ALLOWED"], [], r)


class ScaleTest(SandboxCase):
    """3a at training scale: one step's 32 episodes (8 tasks x 4 rollouts) in fresh containers at once, all cleaned up."""

    async def test_parallel_episodes(self):
        n = 32
        boxes = [self.box(context=f"episode {i}\n" * (i + 1)) for i in range(n)]
        started = time.monotonic()

        async def one(box):
            t0 = time.monotonic()
            async with box:
                startup = time.monotonic() - t0
                result = await box.run_cell("print(len(context.splitlines()), llm('hi'))")
            return startup, result

        outcomes = await asyncio.gather(*(one(b) for b in boxes))
        elapsed = time.monotonic() - started
        for i, (_, result) in enumerate(outcomes):
            self.assertEqual(result.stdout, f"{i + 1} echo:hi\n")
        startups = sorted(s for s, _ in outcomes)
        print(f"\n  {n} episodes in parallel: {elapsed:.1f} s total, container start median {startups[n // 2]:.2f} s, "
              f"max {startups[-1]:.2f} s")


if __name__ == "__main__":
    unittest.main()
