"""Parity with training (rlm-harness v0, step 8): replay recorded v4 episodes through the harness engine.

The script plays the app's part on the engine's JSON-RPC link (rlm-harness documentation/v0/architecture.md §3-§5).
Each root `model.complete` is answered with the reply the trained model gave at that turn, and each sub-call with its
recorded output, so the engine's REPL runs the same cells training ran. Every root request is compared with the
messages the model saw in training: message for message, byte for byte. A difference passes only when the masks of
one of the ACCEPTED classes make the two texts equal, and each class says why the model can't notice it.

    .venv\\Scripts\\python.exe scripts\\harness_parity.py --harness F:\\Personal\\RLM\\rlm-harness --write-summary

It imports nothing from rlm-harness, and imports rlm_rl only to read transcripts, rebuild tasks and workspaces, and
look up limits and hashes. The report holds task text, so it goes to results/harness-parity/<date>/, which the
`results/harness-*/` rule in .gitignore keeps out of git (D-46); the script refuses to run without that rule.
`--write-summary` writes documentation/harness-parity-v0.md from ids, families, counts and class names only, and
refuses (exit 2) if any episode's question or first context line appears in it.

Exit codes: 0 every selected episode ran to its end and every difference is in an accepted class; 1 an unexplained
difference, an unmatched or unused sub-call, an extra root call, an infra_error, or the engine died; 2 setup failed
(the engine didn't start, a hash or protocol mismatch, fewer eligible episodes than asked, bad flags, no .gitignore
rule, or the summary guard found task text). `--strict` makes accepted classes count as failures too.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import collections
import dataclasses
import datetime
import difflib
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import bridge, transcripts  # noqa: E402
from rlm_rl.client import ROOT_TEMPERATURE  # noqa: E402

EXIT_PASS, EXIT_FAIL, EXIT_SETUP = 0, 1, 2

# rlm-harness engine/rlm_engine/contract.py TRAINING_HASH; checked against engine.info and bridge.PROMPT_HASHES["v4"].
TRAINING_HASH = "a71401e2e3bb2e94a424309d2ac01b58ec53a3407a7d9053e589512742b8a1d6"
PROTOCOL = 2  # rlm-harness protocol 2 since its v1 step 3 (run.start.window, run.compacted)
PROFILE, PROFILE_NOPACE = "training-v4", "training-v4-nopace"
MODEL_REF = {"provider": "parity", "model": "recorded", "thinking": None}
ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
              "reasoning_tokens": 0, "cost_usd": 0.0}
MODEL_ERROR, INTERNAL_ERROR = -32001, -32603

# Stripped from the engine's environment, as the app does (D-29, D-133); git's env config goes as a whole (D-62).
KEY_LIKE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH|HEADERS", re.I)
GIT_ENV_CONFIG = re.compile(r"^GIT_CONFIG_(COUNT|KEY_\d+|VALUE_\d+)$", re.I)

ENGINE_STOP_S = 10
CANCEL_WAIT_S = 30
FIDELITY = 0.99  # share of the listed files a native workspace must reproduce with the listed size
# The listing at the end of an edit task's context: editreal._context gives bytes, editsynth characters.
LISTING_RE = re.compile(r"Repository files \(path, size in (bytes|characters)\):\n")
GUARD_MIN_CHARS = 10  # shorter needles ("", "{") would match any summary
# The PC lacks a repository's own toolchain or packages (the training images had them): not a parity difference.
TOOLCHAIN_MARKERS = ("command not found", "No module named", "is not recognized")

# The pacing line (bridge.pace_message) was on for prod-v2 from checkpoint 1's notice on: its train transcripts from
# step 2, and every val transcript. prod-v1 had no pace_nudge key: off. Anything else is unknown.
PACE_ON_FROM_STEP = {"prod-v2": 2}
PACE_OFF_RUNS = ("prod-v1",)
PACE_TEXTS = (bridge.OBS_PACE_EDIT, bridge.OBS_PACE_TEST.split("{turns_left}", 1)[1])

# Difference classes a run may pass with, and why the model can't notice each (step 8 spec, "Difference classes").
ACCEPTED = {
    "timing": "No. The same cell prints different times on two runs in training too, so no behaviour can depend on the "
              "value. Times are masked, and pytest's `=` padding, whose width follows the time's length; test counts "
              "stay.",
    "python-version": "No. Docker had 3.12.15 and uv's 3.12 is another patch release. Patch releases keep the language "
                      "the cells use, and the string shows only if a cell prints `sys.version` (its build string with "
                      "it).",
    "traceback-path": "Only inside the last lines of a traceback of an error raised in a harness function. Training showed "
                      "/opt/rlm/repl_worker.py and /opt/rlm/repl_tools.py. The file is not part of the task and the "
                      "name doesn't change what was raised.",
    "path-form": "Yes, if a cell prints os.getcwd() or an absolute path. Tools return relative paths, and /work/repo/... "
                 "still works through the alias (3a). Accepted only in text the cell's own code printed (it computes a "
                 "path), never in a tool's return value.",
}
# Engine fixes this step made (rlm-harness commit, what, the test that pins it). The summary names each.
ENGINE_FIXES: tuple[tuple[str, str, str], ...] = (
    ("d0e7a1c", "the worker names its own files in a traceback as training did, /opt/rlm/repl_worker.py and "
                "/opt/rlm/repl_tools.py, never the install folder (D-109); line numbers stay its own",
     "engine/tests/test_worker.py::test_traceback_names_training_files"),
    ("807eca5", "shell() maps /work/repo in the command to the workspace, as the tools do (D-150); training's "
                "sandbox had that folder, so `cd /work/repo && ...` failed natively",
     "engine/tests/test_tools.py::test_shell_maps_the_repo_alias"),
    ("807eca5", "on Windows /tmp is one folder (<scratch>/tmp) for the file tools and the shell, as training's one "
                "/tmp was (D-150)", "engine/tests/test_tools.py::test_tmp_is_one_folder_for_the_tools_and_the_shell"),
    ("807eca5", "on Windows the shell's \\r\\n reaches the model as \\n, as on training's Linux (D-150)",
     "engine/tests/test_tools.py::test_shell_lines_end_as_in_training"),
    ("807eca5", "the model's commands no longer inherit the engine's PYTHONUNBUFFERED, which put a child "
                "Python's traceback after its buffered prints (D-150)",
     "engine/tests/test_worker.py::test_env_strips_keys_and_temp_is_scratch"),
)

TIME_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|seconds)\b")
PYTEST_LINE_RE = re.compile(r"^=+ (.*?) =+$", re.M)
VERSION_RE = re.compile(r"(?<![\d.])3\.12\.\d+(?: \([^()\n]*\) ?\[[^\]\n]*\])?")
TRACEBACK_RE = re.compile(r'File "(?!<repl>)[^"]+", line \d+')
# A cell whose code computes a path itself: what path-form accepts.
PATH_CODE_RE = re.compile(r"getcwd|abspath|realpath|resolve\(|\.cwd\(|__file__|expanduser|gettempdir|tempfile|"
                          r"absolute\(|environ")


class SetupError(Exception):
    """Exit 2."""


class Skip(Exception):
    """A candidate that can't be replayed; `code` goes to the report's skipped counts."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


class RpcReply(Exception):
    """An error answer to one of the engine's requests."""

    def __init__(self, code: int, message: str, data: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data

    def to_json(self) -> dict:
        err = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


def model_error(kind: str, message: str, status: int | None = None) -> RpcReply:
    return RpcReply(MODEL_ERROR, message, {"kind": kind, "provider": MODEL_REF["provider"],
                                           "model": MODEL_REF["model"], "status": status, "message": message})


# ----------------------------------------------------------------------------- load, pace, eligibility


@dataclass
class Candidate:
    run: str  # the folder under results/
    record: dict
    pace: str  # "on" | "off" | "unknown"

    @property
    def id(self) -> str:
        return self.record["episode_id"]

    @property
    def family(self) -> str:
        return self.record["family"]

    @property
    def edit(self) -> bool:
        return bridge.is_edit(self.family)

    @property
    def order(self) -> tuple:
        return self.family, self.record["task_id"], self.id


def pace_state(run: str, record: dict) -> str:
    """Whether training added the pacing line to this episode's observations: "on", "off" or "unknown"."""
    if run in PACE_OFF_RUNS:
        return "off"
    if run in PACE_ON_FROM_STEP:
        if record.get("split") == "val":
            return "on"
        if record.get("split") == "train" and isinstance(record.get("step"), int):
            return "on" if record["step"] >= PACE_ON_FROM_STEP[run] else "unknown"
    return "unknown"


def check_pace_text(run: str, record: dict, state: str) -> None:
    """A record the table says is off must carry no pacing line; if one does, the table is wrong (exit 2)."""
    if state != "off":
        return
    for message in record.get("messages", [])[2:]:
        if message.get("role") == "user" and any(text in message.get("content", "") for text in PACE_TEXTS):
            raise SetupError(f"{run} episode {record.get('episode_id')} has a pacing line, but the pace table says off")


def ineligible(record: dict, pace: str) -> str | None:
    """The reason code a record can't be replayed for, from the record alone, or None."""
    if record.get("prompt_version") != "v4":
        return "prompt_version"
    if record.get("infra_error"):
        return "infra_error"
    for key in ("hacks", "flags"):
        if record.get(key):
            return key
    # Edit-mode REPLs log every tool call, program and repo access as an unblocked event; only a blocked attempt (a
    # write or program the harness would ask about and allow) can't replay. An event without the flag was blocked.
    if any(e.get("blocked", True) not in (False, "False") for e in record.get("events") or []):
        return "events"
    for key in ("timeouts", "crashes", "restarts"):  # a blocked write or restart can't replay: the harness asks instead
        if record.get(key):
            return key
    if not record.get("turns"):
        return "no_turns"
    family = record.get("family", "")
    if record.get("limits") != bridge.limits(family, "v4").as_dict():
        return "limits"  # the pilot arms, and tasks with their own cell_timeout_s
    messages = record.get("messages") or [{}]
    if messages[0].get("content") != bridge.system_prompt("v4"):
        return "system_prompt"  # warm-start episodes carry a hint
    if bridge.is_edit(family) and pace == "unknown":
        return "pace_unknown"
    return None


def load(runs: list[str], results: Path) -> tuple[list[Candidate], collections.Counter]:
    """Every record in the runs' transcripts, as eligible candidates plus counts of the skipped ones by reason."""
    eligible, skipped = [], collections.Counter()
    for run in runs:
        folder = results / run / "transcripts"
        if not folder.is_dir():
            print(f"note: no transcripts at {folder}", file=sys.stderr)
            continue
        for path in transcripts.find_logs(folder):
            for record in transcripts.TranscriptLog(path).records():
                state = pace_state(run, record)
                check_pace_text(run, record, state)
                reason = ineligible(record, state)
                if reason:
                    skipped[reason] += 1
                else:
                    eligible.append(Candidate(run, record, state))
    eligible.sort(key=lambda c: c.order)
    return eligible, skipped


# ----------------------------------------------------------------------------- rebuild a task


class Rebuilder:
    """A record's task, rebuilt and checked against the record (step 8 task 5)."""

    def __init__(self):
        self._eval_sets: dict[str, dict] = {}
        self._pools: dict[tuple[str, str], dict] = {}

    def task(self, record: dict):
        family = record["family"]
        if bridge.is_edit(family):
            task = self._edit(record)
        elif record.get("split") == "val":
            task = self._eval_task(record)
        else:
            raise Skip("seed_not_recorded")  # training-split read tasks use fresh seeds that were never recorded
        check_task(record, task)
        return task

    def _eval_task(self, record: dict):
        from rlm_rl import evals
        from rlm_rl.tasks import PRIVATE_DIR

        name = record.get("eval") or ""
        if name not in self._eval_sets:
            # eval_set draws and saves new seeds when the file is missing; that must never happen here
            if not (PRIVATE_DIR / "seeds" / f"{name}.json").is_file():
                raise Skip("seed_not_recorded", f"no private/seeds/{name}.json")
            try:
                tasks, _ = evals.eval_set(name)
            except Exception as exc:
                raise Skip("context_not_rebuildable", f"eval_set({name!r}): {exc!r}") from None
            self._eval_sets[name] = {t.task_id: t for t in tasks}
        task = self._eval_sets[name].get(record["task_id"])
        if task is None:
            raise Skip("context_not_rebuildable", f"{record['task_id']} is not in {name}")
        return task

    def _edit(self, record: dict):
        from rlm_rl import tasks as task_sets

        family = record["family"]
        gen = task_sets.generator(family)
        if not all(hasattr(gen, name) for name in ("SPLIT", "records", "to_task", "eval_tasks")):
            raise Skip("context_not_rebuildable", f"{family} is not a fixed pool")
        split = gen.SPLIT[family]
        key = (family, record["split"])
        try:
            if key not in self._pools:
                if record["split"] == "train":  # the task id is the pool record's own id
                    self._pools[key] = {r["id"]: r for r in gen.records(split)}
                else:
                    self._pools[key] = {t.task_id: t for t in gen.eval_tasks(split, family)}
            found = self._pools[key].get(record["task_id"])
            if found is None:
                raise Skip("context_not_rebuildable", f"{record['task_id']} is not in {family}'s {split} split")
            return gen.to_task(found, family) if record["split"] == "train" else found
        except Skip:
            raise
        except Exception as exc:
            raise Skip("context_not_rebuildable", repr(exc)) from None


def check_task(record: dict, task) -> None:
    context = task.context
    if (hashlib.sha256(context.encode("utf-8", "replace")).hexdigest() != record.get("context_sha256")
            or len(context) != record.get("context_chars")):
        raise Skip("context_not_rebuildable", "context hash or size differs")
    edit = bridge.is_edit(record["family"])
    first = bridge.initial_messages(task.question, context, "v4", bridge.limits(record["family"], "v4"),
                                    edit=(task.edit or None) if edit else None)
    if first != record["messages"][:2]:
        raise Skip("first_message_differs")


# ----------------------------------------------------------------------------- selection


@dataclass
class Slot:
    name: str
    pick: Callable[[list[Candidate]], list[Candidate]]  # the candidates for this slot, in the order to try them


def _family_slot(name: str, edit: bool, test: Callable[[str], bool]) -> Slot:
    def pick(pool):
        found = [c for c in pool if c.edit == edit and test(c.family)]
        if edit:  # python first: the PC is likeliest to have its toolchain
            found.sort(key=lambda c: (c.record.get("language") != "python", c.order))
        return found
    return Slot(name, pick)


def _most_subcalls(pool):
    return sorted((c for c in pool if not c.edit), key=lambda c: (-int(c.record.get("subcalls_used") or 0), c.order))


def slots(n_read: int, n_edit: int, pinned: list[str] | None = None) -> list[Slot]:
    """The quotas: one aggregation, one research_select and the read episode with the most sub-calls (so an llm_batch
    is covered); one edit_spec* and one edit_bugs*. More asked than that takes the next eligible of the mode."""
    if pinned:
        return [Slot(eid, lambda pool, eid=eid: [c for c in pool if c.id == eid]) for eid in pinned]
    read = [_family_slot("aggregation", False, lambda f: f == "aggregation"),
            _family_slot("research_select", False, lambda f: f == "research_select"),
            Slot("read with the most sub-calls", _most_subcalls)]
    edit = [_family_slot("edit_spec*", True, lambda f: f.startswith("edit_spec")),
            _family_slot("edit_bugs*", True, lambda f: f.startswith("edit_bugs"))]
    read += [_family_slot("any read", False, lambda f: True)] * max(0, n_read - len(read))
    edit += [_family_slot("any edit", True, lambda f: True)] * max(0, n_edit - len(edit))
    return read[:n_read] + edit[:n_edit]


# ----------------------------------------------------------------------------- native workspaces


def git_env() -> dict:
    """No system or global config, and training's author, committer and date (editbox.git_env, GIT_DATE)."""
    from rlm_rl.editbox import GIT_DATE

    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0",
               GIT_AUTHOR_NAME="rlm", GIT_AUTHOR_EMAIL="rlm@localhost", GIT_COMMITTER_NAME="rlm",
               GIT_COMMITTER_EMAIL="rlm@localhost", GIT_AUTHOR_DATE=GIT_DATE, GIT_COMMITTER_DATE=GIT_DATE)
    return env


GIT_CONFIG = ["-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-c", "core.longpaths=true",
              "-c", "init.defaultBranch=main"]


def commit_base(folder: Path) -> None:
    """The base commit training's _commit_base made, so the engine's snapshot and diff() work unchanged."""
    env = git_env()
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "--allow-empty", "--no-verify", "-m", "base"]):
        done = subprocess.run(["git", *GIT_CONFIG, *args], cwd=folder, env=env, capture_output=True)
        if done.returncode:
            raise Skip("workspace_mismatch", f"git {args[0]}: {done.stderr.decode(errors='replace').strip()[:300]}")


def listed_files(context: str) -> tuple[list[tuple[str, int]], str]:
    """The `path  size` listing at the end of an edit task's context, and its unit ("bytes" or "characters")."""
    heads = list(LISTING_RE.finditer(context))
    if not heads:
        return [], "bytes"
    out = []
    for line in context[heads[-1].end():].splitlines():
        path, _, size = line.rpartition("  ")
        if path and size.replace(",", "").isdigit():
            out.append((path, int(size.replace(",", ""))))
    return out, heads[-1].group(1)


def fidelity(context: str, folder: Path) -> tuple[int, int]:
    """(listed files that exist with the listed size, listed files)."""
    listed, unit = listed_files(context)
    ok = 0
    for path, size in listed:
        target = folder / path
        try:
            if unit == "bytes":
                ok += target.is_file() and target.stat().st_size == size
            else:
                ok += target.is_file() and len(target.read_bytes().decode("utf-8", "replace")) == size
        except OSError:
            pass
    return ok, len(listed)


def materialise(task, folder: Path) -> None:
    """Build the task's workspace natively (never Workspace.ensure or materialize: they need fcntl and cp), check it
    against the context's listing, and make training's base commit. Raises Skip("workspace_mismatch")."""
    remove_tree(folder)
    folder.mkdir(parents=True)
    try:
        task.workspace.build(folder)
    except Exception as exc:
        raise Skip("workspace_mismatch", f"build: {exc!r}") from None
    ok, total = fidelity(task.context, folder)
    if not total or ok < FIDELITY * total:  # CRLF, symlinks or case-folding the OS can't reproduce
        raise Skip("workspace_mismatch", f"{ok} of {total} listed files match")
    commit_base(folder)


def remove_tree(path: Path) -> None:
    """rmtree that clears read-only bits (git objects on Windows), tried 4 times for antivirus locks."""
    def clear(func, target, _exc):
        os.chmod(target, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        func(target)

    handler = {"onexc": clear} if sys.version_info >= (3, 12) else {"onerror": clear}
    for attempt in range(4):
        if not os.path.lexists(path):
            return
        try:
            shutil.rmtree(path, **handler)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if attempt == 3:
                print(f"note: could not remove {path}: {exc}", file=sys.stderr)
                return
            time.sleep(1)


# ----------------------------------------------------------------------------- the engine


def split_command(line: str) -> list[str]:
    """--engine-cmd split as the app splits RLMH_ENGINE: whitespace, "…" and '…' group, backslashes are plain."""
    out, word, in_word, quote = [], "", False, None
    for ch in line:
        if quote is not None:
            if ch == quote:
                quote = None
            else:
                word += ch
        elif ch in "\"'":
            quote, in_word = ch, True
        elif ch.isspace():
            if in_word:
                out.append(word)
            word, in_word = "", False
        else:
            word, in_word = word + ch, True
    if quote is not None:
        raise SetupError(f"--engine-cmd has an unbalanced {quote} quote")
    if in_word:
        out.append(word)
    return out


def engine_env(home: Path, base: dict | None = None) -> dict:
    """The parent's environment minus key-like names and git's env config, plus RLMH_HOME (D-49) and UTF-8 mode, as the
    app starts it."""
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if not KEY_LIKE.search(k) and not GIT_ENV_CONFIG.match(k)}
    env.update(RLMH_HOME=str(home), PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    return env


class EngineDied(Exception):
    pass


class Engine:
    """The engine process and its link: binary pipes, a reader thread feeding an asyncio queue (as D-30 does for the
    engine), and a lock that serialises writes. `handler` answers the engine's requests and takes its events."""

    def __init__(self, cmd: list[str], cwd: Path, env: dict, stderr_path: Path):
        self.cmd, self.cwd, self.env, self.stderr_path = cmd, cwd, env, stderr_path
        self.handler: Any = None
        self.dead: str | None = None
        self.died = asyncio.Event()
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self.proc: subprocess.Popen | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        self._stderr = open(self.stderr_path, "wb")
        extra = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                 else {"start_new_session": True})
        try:
            self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=self._stderr, **extra)
        except OSError as exc:
            raise SetupError(f"could not start the engine ({' '.join(self.cmd)}): {exc}") from None

        def read() -> None:
            for line in iter(self.proc.stdout.readline, b""):
                loop.call_soon_threadsafe(queue.put_nowait, line)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=read, daemon=True, name="engine-stdout").start()
        self._spawn(self._dispatch(queue))

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _dispatch(self, queue: asyncio.Queue) -> None:
        while True:
            line = await queue.get()
            if line is None:
                self._die("the engine closed its stdout")
                return
            try:
                msg = json.loads(line.decode("utf-8"))
            except ValueError:
                print(f"note: a line from the engine is not JSON: {line[:200]!r}", file=sys.stderr)
                continue
            if "method" in msg and "id" in msg:
                self._spawn(self._serve(msg))
            elif "method" in msg:
                if self.handler is not None:
                    self.handler.event(msg["method"], msg.get("params") or {})
            else:
                future = self._pending.pop(msg.get("id"), None)
                if future is not None and not future.done():
                    if "error" in msg:
                        err = msg["error"] or {}
                        future.set_exception(RpcReply(err.get("code", 0), err.get("message", ""), err.get("data")))
                    else:
                        future.set_result(msg.get("result"))

    async def _serve(self, msg: dict) -> None:
        method, params = msg["method"], msg.get("params") or {}
        try:
            if self.handler is None:
                raise RpcReply(INTERNAL_ERROR, "no episode is running")
            if method == "model.complete":
                reply = {"result": await self.handler.complete(params)}
            elif method == "approval.request":
                reply = {"result": await self.handler.approve(params)}
            else:
                raise RpcReply(-32601, f"unknown method {method}")
        except RpcReply as err:
            reply = {"error": err.to_json()}
        except Exception as exc:  # a bug here: tell the engine, and the episode fails on what follows
            reply = {"error": {"code": INTERNAL_ERROR, "message": repr(exc)}}
        await self.send({"jsonrpc": "2.0", "id": msg["id"], **reply})

    async def send(self, msg: dict) -> None:
        # ensure_ascii: a recorded text may hold lone surrogates, which UTF-8 can't carry; the engine decodes \ud8xx back
        data = (json.dumps(msg) + "\n").encode("ascii")
        async with self._lock:
            try:
                await asyncio.to_thread(self._write, data)
            except OSError as exc:
                self._die(f"writing to the engine failed: {exc}")

    def _write(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    async def request(self, method: str, params: dict, timeout: float | None = None) -> Any:
        if self.dead:
            raise EngineDied(self.dead)
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[self._next_id] = future
        await self.send({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params})
        return await asyncio.wait_for(future, timeout)

    def _die(self, why: str) -> None:
        if self.dead is None:
            self.dead = why
            self.died.set()
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(EngineDied(why))
            self._pending.clear()

    async def stop(self) -> None:
        """Close stdin, wait 10 s, then kill the whole tree."""
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            await asyncio.to_thread(self.proc.wait, ENGINE_STOP_S)
        except subprocess.TimeoutExpired:
            kill_tree(self.proc)
        if os.name != "nt":
            kill_tree(self.proc)  # anything uv or the engine left in its session
        self._stderr.close()


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        pass


def check_info(info: dict, profiles: tuple[str, ...]) -> None:
    """engine.info: protocol 2, training's hash, and each profile hashing to it. Raises SetupError."""
    if info.get("protocol") != PROTOCOL:
        raise SetupError(f"engine protocol {info.get('protocol')!r}, this script speaks {PROTOCOL}")
    if not (info.get("training_hash") == TRAINING_HASH == bridge.PROMPT_HASHES["v4"]):
        raise SetupError(f"training hash mismatch: engine {info.get('training_hash')}, script {TRAINING_HASH}, "
                         f"rlm_rl {bridge.PROMPT_HASHES['v4']}")
    listed = {p.get("name"): p for p in info.get("profiles") or []}
    for name in profiles:
        if name not in listed:
            raise SetupError(f"the engine does not list profile {name}")
        if listed[name].get("prompt_hash") != TRAINING_HASH:
            raise SetupError(f"profile {name} hashes to {listed[name].get('prompt_hash')}, not training's")


def write_nopace_profile(harness: Path, home: Path) -> None:
    """training-v4 with pace_nudge off, as a user profile (D-34 forbids reusing the built-in name). The pace flag is not
    in the prompt hash, so it still hashes to training's."""
    source = harness / "engine" / "rlm_engine" / "profiles" / PROFILE / "profile.toml"
    try:
        text = source.read_bytes().decode("utf-8")
    except OSError as exc:
        raise SetupError(f"no {PROFILE} profile at {source}: {exc}") from None
    text, n = re.subn(r"(?m)^pace_nudge = true", "pace_nudge = false", text)
    if n != 1:
        raise SetupError(f"{source} has {n} `pace_nudge = true` lines, not 1")
    folder = home / "profiles" / PROFILE_NOPACE
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "profile.toml").write_bytes(text.encode("utf-8"))


# ----------------------------------------------------------------------------- sub-calls


def subcall_error_message(error: str) -> str:
    """What the model saw for a recorded sub-call error. Training logged repr(SubcallError(msg)) and showed
    "sub-call failed: msg"; the engine adds the prefix to the message it gets, so this is msg."""
    match = re.fullmatch(r"SubcallError\((.*)\)", error.strip(), re.S)
    if match:
        try:
            value = ast.literal_eval(match.group(1))
            if isinstance(value, str):
                return value
        except (ValueError, SyntaxError):
            pass
    return error


class SubcallBook:
    """Recorded sub-calls by exact prompt. Duplicates are consumed in the order recorded, whatever order the engine's
    concurrent calls arrive in."""

    def __init__(self, subcalls: list[dict]):
        self.prompts = [s["prompt"] for s in subcalls]
        self._queues: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        for s in subcalls:
            self._queues[s["prompt"]].append(s)
        self.unmatched: list[dict] = []

    def answer(self, prompt: str) -> dict:
        """The model.complete result for a sub-call, or RpcReply: a recorded error as bad_request (400), and an
        unmatched prompt as `other`, which stops the run (D-06)."""
        queue = self._queues.get(prompt)
        if not queue:
            self.unmatched.append(self.describe(prompt))
            raise model_error("other", "parity: no recorded sub-call has this prompt")
        record = queue.popleft()
        if record.get("error"):
            message = subcall_error_message(record["error"])
            raise model_error("bad_request", message, 400)
        return {"text": record.get("output") or "", "tool_call": None, "reasoning": None, "usage": dict(ZERO_USAGE),
                "finish_reason": "stop", "served_model": MODEL_REF["model"], "seconds": 0}

    def describe(self, prompt: str) -> dict:
        """An unmatched prompt for the report: its length and hash, and the longest prefix it shares with a recorded
        prompt (no text)."""
        best, closest = 0, None
        for other in self.prompts:
            n = len(os.path.commonprefix([prompt, other]))
            if n > best or closest is None:
                best, closest = n, other
        return {"chars": len(prompt), "sha256": hashlib.sha256(prompt.encode("utf-8", "replace")).hexdigest(),
                "common_prefix": best, "closest_chars": len(closest) if closest is not None else None}

    def unused(self) -> int:
        return sum(len(q) for q in self._queues.values())


# ----------------------------------------------------------------------------- classify


def mask_timing(text: str) -> str:
    text = PYTEST_LINE_RE.sub(lambda m: f"= {m.group(1)} =" if TIME_RE.search(m.group(1)) else m.group(0), text)
    return TIME_RE.sub("<T>", text)


def mask_version(text: str) -> str:
    return VERSION_RE.sub("3.12.x", text)


def mask_traceback(text: str) -> str:
    return TRACEBACK_RE.sub('File "<F>", line <N>', text)


def path_spellings(path: str) -> list[str]:
    """A native folder as a cell may print it: as given and resolved, with \\, / or doubled \\ (a repr)."""
    out = []
    for base in dict.fromkeys([path, os.path.realpath(path)]):
        out += [base, base.replace("\\", "/"), base.replace("\\", "\\\\")]
    return sorted({s.rstrip("/\\") for s in out if s}, key=len, reverse=True)


def mask_paths(text: str, roots: list[tuple[str, str]]) -> str:
    """Native workspace and scratch folders become /work/repo and /scratch, with / inside the rest of the path."""
    for native, alias in roots:
        for spelling in path_spellings(native):
            pattern = re.escape(spelling) + r"((?:\\\\|[\\/])[^\s'\"<>|]*)?"
            text = re.sub(pattern, lambda m: alias + re.sub(r"\\\\|\\", "/", m.group(1) or ""), text)
    return text


def classify(recorded: str, got: str, *, roots: list[tuple[str, str]] = (), code: str | None = None
             ) -> tuple[list[str], bool]:
    """The classes a difference belongs to, and whether all are accepted. Masks apply to both texts; if the masked
    texts differ it is ["unexplained"]. Otherwise it is every class whose mask the match needed (or, when none alone
    was, every class whose mask changed something)."""
    masks: list[tuple[str, Callable[[str], str]]] = [("timing", mask_timing), ("python-version", mask_version)]
    if roots:
        masks.append(("path-form", lambda t: mask_paths(t, list(roots))))
    masks.append(("traceback-path", mask_traceback))

    def apply(text: str, skip: str | None = None) -> str:
        for name, fn in masks:
            if name != skip:
                text = fn(text)
        return text

    if apply(recorded) != apply(got):
        return ["unexplained"], False
    changed = [name for name, fn in masks if fn(recorded) != recorded or fn(got) != got]
    needed = [name for name in changed if apply(recorded, skip=name) != apply(got, skip=name)]
    classes = needed or changed
    if "path-form" in classes and not PATH_CODE_RE.search(code or ""):
        classes = ["path-form (tool)" if c == "path-form" else c for c in classes]
    return classes, all(c in ACCEPTED for c in classes)


def toolchain_gap(recorded: str, got: str) -> bool:
    return any(marker in got and marker not in recorded for marker in TOOLCHAIN_MARKERS)


# ----------------------------------------------------------------------------- compare


@dataclass
class Difference:
    request: int  # the first root request it shows in
    index: int  # the message's index in the conversation
    role: str
    recorded: str
    got: str
    classes: list[str] = field(default_factory=list)
    accepted: bool = False

    def as_dict(self, text: bool = True) -> dict:
        out = {"request": self.request, "index": self.index, "role": self.role, "classes": self.classes,
               "accepted": self.accepted}
        if text:
            out["diff"] = unified(self.recorded, self.got)
        return out


def unified(recorded: str, got: str) -> str:
    """A unified diff in which a line ending other than \\n shows as its escape and a missing final newline is marked,
    so a CRLF or newline difference is visible."""
    def lines(text: str) -> list[str]:
        out = []
        for line in text.splitlines(keepends=True):
            body = line.rstrip("\r\n\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")
            end = line[len(body):]
            out.append(body if end == "\n" else body + (repr(end)[1:-1] if end else " [no newline at end]"))
        return out

    return "\n".join(difflib.unified_diff(lines(recorded), lines(got), "training", "harness", lineterm="", n=2))


def compare_requests(recorded: list[dict], requests: list[dict], limits: dict
                     ) -> tuple[list[Difference], list[str], int]:
    """Root request k against the recorded messages[:2 + 2*(k-1)]: (differences, failures, byte-for-byte requests).
    A message that differs is reported once, at the first request it differs in; a request is byte-for-byte only when
    none of its messages differ."""
    diffs: list[Difference] = []
    failures: list[str] = []
    reported: set[int] = set()
    matched = 0
    for k, params in enumerate(requests, 1):
        want = recorded[:2 + 2 * (k - 1)]
        got = params.get("messages")
        got = got if isinstance(got, list) else []
        same = len(got) == len(want)
        if not same:
            failures.append(f"message_count: request {k} has {len(got)} messages, training had {len(want)}")
        for i, (w, g) in enumerate(zip(want, got)):
            if not (isinstance(g, dict) and set(g) == {"role", "content"} and g["role"] == w["role"]
                    and isinstance(g["content"], str)):
                same = False
                if i not in reported:
                    reported.add(i)
                    failures.append(f"message_shape: request {k}, message {i} is not {{role: {w['role']}, content}}")
                continue
            if g["content"] != w["content"]:
                same = False
                if i not in reported:
                    reported.add(i)
                    diffs.append(Difference(k, i, w["role"], w["content"], g["content"]))
        matched += same
        expected = {"tools": [], "max_tokens": limits["root_max_tokens"], "temperature": ROOT_TEMPERATURE,
                    "thinking": "off"}
        for key, value in expected.items():
            if params.get(key) != value:
                failures.append(f"request_param: request {k} has {key} {params.get(key)!r}, expected {value!r}")
    return diffs, failures, matched


def check_subcall_params(requests: list[dict], limits: dict) -> list[str]:
    expected = {"temperature": bridge.SUBCALL_TEMPERATURE, "max_tokens": limits["subcall_max_tokens"],
                "thinking": limits.get("subcall_thinking") or "off", "tools": []}
    bad = collections.Counter()
    for params in requests:
        for key, value in expected.items():
            if params.get(key) != value:
                bad[f"subcall_param: {key} {params.get(key)!r}, expected {value!r}"] += 1
        messages = params.get("messages")
        if not (isinstance(messages, list) and len(messages) == 1 and isinstance(messages[0], dict)
                and set(messages[0]) == {"role", "content"} and messages[0]["role"] == "user"):
            bad["subcall_param: messages are not one user message"] += 1
    return [f"{what} (x{n})" for what, n in bad.items()]


def cell_locators(turns: list[dict], cells: list[dict]) -> list[dict]:
    """Per cell, which of code, error, observation and restarted differ from training's turn: to locate a difference."""
    out = []
    for event in cells:
        t = event.get("turn")
        if not isinstance(t, int) or not 1 <= t <= len(turns):
            out.append({"turn": t, "fields": ["no such recorded turn"]})
            continue
        turn = turns[t - 1]
        cell = turn.get("cell") or {}
        recorded = {"code": turn.get("code"),
                    "error": cell.get("error") if turn.get("code") is not None else bridge.no_code_error("v4"),
                    "observation": turn.get("observation"), "restarted": bool(cell.get("restarted"))}
        got = {"code": event.get("code"), "error": event.get("error"), "observation": event.get("observation"),
               "restarted": bool(event.get("restarted"))}
        fields = [k for k in recorded if recorded[k] != got[k]]
        if fields:
            out.append({"turn": t, "fields": fields})
    return out


def check_end(record: dict, context: str, started: dict | None, finished: dict | None) -> list[str]:
    """run.finished and run.started against the record."""
    if finished is None:
        return ["no_run_finished: the run never finished"]
    failures = []
    if finished.get("infra_error"):
        failures.append(f"infra_error: {finished['infra_error']}")
    checks = {"turns": len(record["turns"]), "submitted": record.get("submitted"),
              "answer": record.get("answer"), "subcalls": record.get("subcalls_used")}
    for key, value in checks.items():
        if json.dumps(finished.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
            failures.append(f"end_mismatch: run.finished {key} {finished.get(key)!r}, training {value!r}")
    if started is None:
        return failures + ["no_run_started: the run never started"]
    limits = record.get("limits") or {}
    for key, value in (started.get("limits") or {}).items():
        if key in limits and limits[key] != value:
            failures.append(f"end_mismatch: run.started limits.{key} {value!r}, training {limits[key]!r}")
    inp = started.get("input") or {}
    want = {"chars": record.get("context_chars"), "sha256": record.get("context_sha256"),
            "lines": len(context.splitlines())}
    for key, value in want.items():
        if inp.get(key) != value:
            failures.append(f"end_mismatch: run.started input.{key} {inp.get(key)!r}, training {value!r}")
    return failures


# ----------------------------------------------------------------------------- one episode


@dataclass
class Result:
    id: str
    run: str
    family: str
    mode: str
    profile: str
    turns: int
    subcalls: int
    requests: int = 0
    matched: int = 0
    subcalls_served: int = 0
    subcalls_unmatched: list[dict] = field(default_factory=list)
    subcalls_unused: int = 0
    approvals: dict = field(default_factory=dict)
    differences: list[Difference] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    cells: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    toolchain: bool = False
    seconds: float = 0.0

    @property
    def unexplained(self) -> int:
        return sum(not d.accepted for d in self.differences)

    def verdict(self, strict: bool = False) -> str:
        if self.toolchain:
            return "replaced(toolchain)"
        bad = self.failures or self.unexplained or self.subcalls_unmatched or self.subcalls_unused
        if strict and any(d.accepted for d in self.differences):
            bad = True
        return "FAIL" if bad else "PASS"

    def as_dict(self, strict: bool = False) -> dict:
        out = dataclasses.asdict(self)
        out["differences"] = [d.as_dict() for d in self.differences]
        out["verdict"] = self.verdict(strict)
        return out


class Replay:
    """Answers one run's requests and keeps its events (step 8 task 9)."""

    def __init__(self, record: dict):
        self.record = record
        self.turns = record["turns"]
        self.book = SubcallBook(record.get("subcalls") or [])
        self.root: list[dict] = []
        self.sub: list[dict] = []
        self.failures: list[str] = []
        self.approvals: collections.Counter = collections.Counter()
        self.events: list[tuple[str, dict]] = []
        self.finished = asyncio.Event()

    async def complete(self, params: dict) -> dict:
        if params.get("role") == "root":
            self.root.append(params)
            k = len(self.root)
            if params.get("turn") != k:
                self.failures.append(f"turn_number: root call {k} has turn {params.get('turn')!r}")
            if k > len(self.turns):
                self.failures.append(f"extra_root_call: root call {k}, training had {len(self.turns)} turns")
                raise model_error("other", "parity: no recorded turn for this root call")
            turn = self.turns[k - 1]
            return {"text": turn["reply"], "tool_call": None, "reasoning": None,
                    "finish_reason": turn.get("finish_reason"), "usage": dict(ZERO_USAGE),
                    "served_model": MODEL_REF["model"], "seconds": 0}
        self.sub.append(params)
        messages = params.get("messages") or [{}]
        prompt = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        return self.book.answer(prompt)

    async def approve(self, params: dict) -> dict:
        self.approvals[str(params.get("kind"))] += 1
        return {"allow": True, "remember_for_run": False}

    def event(self, method: str, params: dict) -> None:
        if method.startswith("run."):
            self.events.append((method, params))
            if method == "run.finished":
                self.finished.set()

    def last(self, method: str) -> dict | None:
        found = [p for m, p in self.events if m == method]
        return found[-1] if found else None


async def replay(engine: Engine, cand: Candidate, task, args, paths: dict) -> Result:
    record = cand.record
    profile = PROFILE_NOPACE if cand.edit and cand.pace == "off" else PROFILE
    result = Result(cand.id, cand.run, cand.family, "repo" if cand.edit else "read", profile, len(record["turns"]),
                    int(record.get("subcalls_used") or 0))
    params = {"profile": profile, "root": MODEL_REF, "sub": MODEL_REF, "permission_mode": args.permission_mode,
              "call_format": "fenced", "interactive": True, "tag": cand.id, "batch": "parity",
              "log_dir": str(paths["runlogs"])}
    workspace = None
    if cand.edit:
        workspace = paths["work"] / "ws" / cand.id[:8]
        try:
            materialise(task, workspace)
        except Skip:
            if not args.keep_work:
                remove_tree(workspace)
            raise
        edit = task.edit or {}
        params.update(mode="repo", task=record["question"], repo_path=str(workspace), test_hint=edit["test_hint"],
                      language=edit["language"], n_files=int(edit["n_files"]), context=task.context)
    else:
        source = paths["work"] / "inputs" / f"{cand.id}.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(task.context.encode("utf-8", "replace"))
        params.update(mode="read", question=record["question"], inputs=[str(source)])

    run = Replay(record)
    engine.handler = run
    started = time.monotonic()
    try:
        try:
            reply = await engine.request("run.start", params, timeout=args.episode_timeout_s)
        except RpcReply as err:
            result.failures.append(f"run_start: {err.code} {err.message}")
            return result
        except asyncio.TimeoutError:
            result.failures.append(f"run_start: no answer in {args.episode_timeout_s:g} s")
            return result
        run_id = reply["run_id"]
        waiter = asyncio.ensure_future(run.finished.wait())
        dead = asyncio.ensure_future(engine.died.wait())
        done, _ = await asyncio.wait({waiter, dead}, timeout=args.episode_timeout_s,
                                     return_when=asyncio.FIRST_COMPLETED)
        waiter.cancel()
        dead.cancel()
        if engine.dead:
            raise EngineDied(engine.dead)
        if not done:
            result.failures.append(f"timeout: no run.finished after {args.episode_timeout_s:g} s")
            try:
                await engine.request("run.cancel", {"run_id": run_id}, timeout=CANCEL_WAIT_S)
                await asyncio.wait_for(run.finished.wait(), CANCEL_WAIT_S)
            except (RpcReply, asyncio.TimeoutError):
                pass
    finally:
        engine.handler = None
        result.seconds = round(time.monotonic() - started, 1)
        if workspace is not None and not args.keep_work:
            remove_tree(workspace)

    limits = record["limits"]
    result.requests = min(len(run.root), len(record["turns"]))
    diffs, failures, result.matched = compare_requests(record["messages"], run.root[:len(record["turns"])], limits)
    native = native_roots(cand.edit, workspace, paths["home"], run_id)
    for d in diffs:
        d.classes, d.accepted = classify(d.recorded, d.got, roots=native, code=turn_code(record["turns"], d.index))
        result.toolchain = result.toolchain or (not d.accepted and toolchain_gap(d.recorded, d.got))
    result.differences = diffs
    result.failures += run.failures + failures + check_subcall_params(run.sub, limits)
    result.failures += check_end(record, task.context, run.last("run.started"), run.last("run.finished"))
    result.subcalls_served = len(run.sub) - len(run.book.unmatched)
    result.subcalls_unmatched = run.book.unmatched
    result.subcalls_unused = run.book.unused()
    result.approvals = dict(sorted(run.approvals.items()))
    result.cells = cell_locators(record["turns"], [p for m, p in run.events if m == "run.cell"])
    result.info = runlog_info(run.last("run.finished"), run_id, record)
    return result


def turn_code(turns: list[dict], index: int) -> str | None:
    """The code of the turn whose observation is message `index` (3 is turn 1's, 5 turn 2's), or None."""
    turn = (index - 1) // 2
    return turns[turn - 1].get("code") if index >= 3 and index % 2 and turn <= len(turns) else None


def native_roots(edit: bool, workspace: Path | None, home: Path, run_id: str) -> list[tuple[str, str]]:
    scratch = str(home / "scratch" / run_id)
    return [(str(workspace), bridge.EDIT_REPO_PATH), (scratch, "/scratch")] if edit else [(scratch, "/scratch")]


def runlog_info(finished: dict | None, run_id: str, record: dict) -> dict:
    """Not gated: the run log's messages against training's (as a prefix), and its diff against the recorded diff."""
    try:
        lines = Path(finished["log_path"]).read_text(encoding="utf-8").splitlines()
        logged = next(json.loads(line) for line in reversed(lines) if f'"{run_id}"' in line
                      and json.loads(line).get("run_id") == run_id)
    except Exception as exc:
        return {"runlog": f"unreadable: {exc!r}"}
    messages = logged.get("messages") or []
    info = {"runlog_messages": "prefix" if messages == record["messages"][:len(messages)] else "differs"}
    if bridge.is_edit(record["family"]):
        got, want = logged.get("diff") or "", record.get("diff") or ""
        # training graded a --full-index diff; the run log keeps diff()'s own form, with abbreviated index lines
        info["diff"] = ("equal" if got == want else "equal apart from index abbreviation"
                        if short_index(got) == short_index(want) else "differs")
    return info


def short_index(diff: str) -> str:
    return re.sub(r"(?m)^index ([0-9a-f]{7})[0-9a-f]*\.\.([0-9a-f]{7})[0-9a-f]*", r"index \1..\2", diff)


# ----------------------------------------------------------------------------- the report


def exit_code(results: list[Result], asked: int, *, strict: bool, engine_died: bool = False) -> int:
    kept = [r for r in results if not r.toolchain]
    if engine_died or any(r.verdict(strict) == "FAIL" for r in kept):
        return EXIT_FAIL
    if len(kept) < asked:
        return EXIT_SETUP
    return EXIT_PASS


def counters(results: list[Result], replaced: dict, skipped: collections.Counter, *, strict: bool) -> dict:
    kept = [r for r in results if not r.toolchain]
    classes = collections.Counter(c for r in kept for d in r.differences if d.accepted for c in d.classes)
    failures = collections.Counter(f.split(":")[0] for r in kept for f in r.failures)
    for r in kept:
        failures["subcall_unmatched"] += len(r.subcalls_unmatched)
        failures["subcall_unused"] += r.subcalls_unused
    return {
        "episodes": len(kept), "read": sum(r.mode == "read" for r in kept), "edit": sum(r.mode == "repo" for r in kept),
        "requests": sum(r.requests for r in kept), "matched": sum(r.matched for r in kept),
        "accepted": dict(sorted(classes.items())), "unexplained": sum(r.unexplained for r in kept),
        "failures": {k: v for k, v in sorted(failures.items()) if v}, "replaced": replaced,
        "skipped": dict(sorted(skipped.items())),
    }


def digest(results: list[Result], strict: bool) -> str:
    """What two runs must agree on, hashed, so a second run is checked at a glance: each episode's verdict, requests
    compared, unexplained differences and failures. Not the byte-for-byte count or the accepted classes: a cell that
    prints a time near zero can match training's on one run and not the next."""
    rows = [(r.id, r.verdict(strict), r.requests, r.unexplained, sorted(f.split(":")[0] for f in r.failures),
             len(r.subcalls_unmatched), r.subcalls_unused) for r in results]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:12]


def _joined(items: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in items.items()) or "none"


def header_lines(header: dict, c: dict, verdict: str) -> list[str]:
    return [
        f"date: {header['date']}",
        f"harness commit: {header['harness_commit']}",
        f"RLM commit: {header['rlm_commit']}",
        f"engine: {header['engine']}",
        f"OS: {header['os']}",
        f"git: {header['git']}",
        f"shell: {header['shell']}",
        f"permission mode: {header['permission_mode']}",
        "",
        f"episodes: {c['episodes']} (read {c['read']}, edit {c['edit']})",
        f"requests compared: {c['requests']}",
        f"byte-for-byte requests: {c['matched']} of {c['requests']}",
        f"accepted differences: {_joined(c['accepted'])}",
        f"unexplained differences: {c['unexplained']}",
        f"failures: {_joined(c['failures'])}",
        f"replaced: {_joined(c['replaced'])}",
        f"skipped: {_joined(c['skipped'])}",
        f"digest: {header['digest']}",
        f"verdict: {verdict}",
    ]


def report_md(header: dict, c: dict, results: list[Result], verdict: str, strict: bool) -> str:
    out = ["# Harness parity report", "", *[f"- {line}" if line else "" for line in header_lines(header, c, verdict)],
           "", "## Episodes", "",
           "| id | run | family | mode | profile | turns | sub-calls | requests matched | sub-calls served / unmatched / unused | approvals | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        out.append(f"| {r.id} | {r.run} | {r.family} | {r.mode} | {r.profile} | {r.turns} | {r.subcalls} | "
                   f"{r.matched} of {r.requests} | {r.subcalls_served} / {len(r.subcalls_unmatched)} / "
                   f"{r.subcalls_unused} | {_joined(r.approvals)} | {r.verdict(strict)} |")
    for r in results:
        out += ["", f"## {r.id} ({r.family})", ""]
        out += [f"- failure: {f}" for f in r.failures]
        out += [f"- unmatched sub-call: {u}" for u in r.subcalls_unmatched]
        out += [f"- cells that differ: " + ", ".join(f"turn {x['turn']} ({', '.join(x['fields'])})" for x in r.cells)
                ] if r.cells else []
        out += [f"- {k}: {v}" for k, v in r.info.items()]
        for d in r.differences:
            out += ["", f"### message {d.index} ({d.role}), first in request {d.request}: {', '.join(d.classes)}, "
                        f"{'accepted' if d.accepted else 'NOT accepted'}", "", "```diff", unified(d.recorded, d.got),
                    "```"]
    seen = sorted({c for r in results for d in r.differences for c in d.classes})
    out += ["", "## Classes seen", "", "| class | accepted | reason |", "|---|---|---|"]
    out += [f"| {c} | {'yes' if c in ACCEPTED else 'no'} | {ACCEPTED.get(c, 'not in the ACCEPTED table')} |"
            for c in seen] or ["| none | | |"]
    return "\n".join(out) + "\n"


def summary_md(header: dict, c: dict, results: list[Result], verdict: str, strict: bool) -> str:
    """documentation/harness-parity-v0.md: counters, ids, families and class names only, never task text."""
    out = ["# Harness parity v0", "",
           f"Written by `scripts/harness_parity.py --write-summary` on {header['date']}. It replays recorded v4 "
           "episodes through the rlm-harness engine (step 8 of its v0 build) and compares every root request with the "
           "messages the trained model saw. The full report, with the diffs, is in "
           f"`results/harness-parity/{header['date']}/` (git-ignored: it holds task text).", ""]
    out += [f"- {line}" if line else "" for line in header_lines(header, c, verdict)]
    rest = ("every other one differs only in the accepted classes below" if not c["unexplained"] and not c["failures"]
            else f"{c['unexplained']} differences are unexplained and the failures above stand, so this run failed")
    out += ["", f"{c['matched']} of {c['requests']} root requests matched byte for byte; {rest}. A `--strict` run exits "
                "1 whenever that count is short of the total.",
            "", "## Episodes", "", "| episode | run | family | mode | profile | turns | sub-calls | requests matched | verdict |",
            "|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {r.id} | {r.run} | {r.family} | {r.mode} | {r.profile} | {r.turns} | {r.subcalls} | "
            f"{r.matched} of {r.requests} | {r.verdict(strict)} |" for r in results]
    seen = collections.Counter(c for r in results for d in r.differences for c in d.classes)
    out += ["", "## Difference classes", "", "| class | differences | accepted | reason |", "|---|---|---|---|"]
    out += [f"| {k} | {n} | {'yes' if k in ACCEPTED else 'no'} | {ACCEPTED.get(k, 'not accepted')} |"
            for k, n in sorted(seen.items())] or ["| none | 0 | | |"]
    out += ["", "## Engine fixes", ""]
    out += [f"- {what} (rlm-harness {commit}; {test})" for commit, what, test in ENGINE_FIXES] or ["- none"]
    return "\n".join(out) + "\n"


def guard(summary: str, candidates: list[tuple[str, str]]) -> list[str]:
    """Which (question, context) pairs leak into the summary: a question's or first context line's first 60
    characters."""
    leaks = []
    for question, context in candidates:
        first_line = context.splitlines()[0] if context else ""
        for label, text in (("question", question), ("first context line", first_line)):
            needle = (text or "")[:60].strip()
            if len(needle) >= GUARD_MIN_CHARS and needle in summary:
                leaks.append(f"{label}: {len(needle)} characters")
    return leaks


def commit_of(folder: Path) -> str:
    def git(*args):
        return subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "HEAD") or "unknown"
    return sha + (" dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def shell_kind() -> str:
    """The shell the engine picks (rlm-harness workspace.find_shell): sh on POSIX; on Windows Git Bash next to git or
    under Program Files, else PowerShell."""
    if os.name != "nt":
        return "sh"
    candidates = []
    git = shutil.which("git")
    if git:
        for folder in list(Path(git).parents)[:3]:
            candidates += [folder / "bin" / "bash.exe", folder / "usr" / "bin" / "bash.exe"]
    for var, sub in (("PROGRAMFILES", ()), ("PROGRAMFILES(X86)", ()), ("LOCALAPPDATA", ("Programs",))):
        if os.environ.get(var):
            candidates.append(Path(os.environ[var], *sub, "Git", "bin", "bash.exe"))
    for bash in candidates:
        if not {"system32", "windowsapps"} & {p.lower() for p in bash.parts} and bash.is_file():
            return f"Git Bash ({bash})"
    return "PowerShell"


# ----------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--harness", type=Path, required=True, help="the rlm-harness checkout")
    p.add_argument("--read", type=int, default=3)
    p.add_argument("--edit", type=int, default=2)
    p.add_argument("--episodes", default="", help="pin these episode ids, comma-separated, instead of the quotas")
    p.add_argument("--runs", default="prod-v2,prod-v1", help="folders under results/ whose transcripts to read")
    p.add_argument("--permission-mode", default="accept-edits", choices=("accept-edits", "default", "bypass"))
    p.add_argument("--engine-cmd", default="", help="replaces `uv run --project <harness>/engine python -X utf8 -m "
                                                    "rlm_engine serve`")
    p.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()) / "rlm-parity")
    p.add_argument("--keep-work", action="store_true", help="keep each edit episode's native workspace")
    p.add_argument("--strict", action="store_true", help="accepted classes count as failures")
    p.add_argument("--write-summary", action="store_true", help="write documentation/harness-parity-v0.md")
    p.add_argument("--date", default=datetime.date.today().isoformat())
    p.add_argument("--episode-timeout-s", type=float, default=1800)
    args = p.parse_args(argv)
    try:
        datetime.date.fromisoformat(args.date)
    except ValueError:
        p.error(f"--date must be YYYY-MM-DD, not {args.date!r}")
    if args.read < 0 or args.edit < 0 or args.episode_timeout_s <= 0:
        p.error("--read and --edit must be 0 or more, and --episode-timeout-s above 0")
    return args


async def run(args: argparse.Namespace, rebuilder: Rebuilder | None = None) -> int:
    harness = args.harness.resolve()
    report_dir = ROOT / "results" / "harness-parity" / args.date
    rel = report_dir.relative_to(ROOT).as_posix() + "/report.md"
    if subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", rel]).returncode != 0:
        raise SetupError(f"{rel} is not git-ignored: add the line results/harness-*/ to .gitignore first (D-46)")
    if not args.engine_cmd and not (harness / "engine" / "pyproject.toml").is_file():
        raise SetupError(f"no engine at {harness / 'engine'}")
    work = args.work_dir.resolve()
    home = work / "home"
    paths = {"work": work, "home": home, "runlogs": report_dir / "runlogs"}
    report_dir.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    write_nopace_profile(harness, home)

    pinned = [e.strip() for e in args.episodes.split(",") if e.strip()]
    plan = slots(args.read, args.edit, pinned)
    candidates, skipped = load([r.strip() for r in args.runs.split(",") if r.strip()], ROOT / "results")
    rebuilder = rebuilder or Rebuilder()

    cmd = (split_command(args.engine_cmd) if args.engine_cmd else
           [shutil.which("uv") or "uv", "run", "--project", str(harness / "engine"), "python", "-X", "utf8", "-m",
            "rlm_engine", "serve"])
    engine = Engine(cmd, harness, engine_env(home), report_dir / "engine-stderr.log")
    results: list[Result] = []
    replaced: dict[str, str] = {}
    leaks_from: list[tuple[str, str]] = []
    info: dict = {}
    died = False
    await engine.start()
    try:
        try:
            info = await engine.request("engine.info", {}, timeout=120)
        except (EngineDied, RpcReply, asyncio.TimeoutError) as exc:
            raise SetupError(f"engine.info failed: {exc!r}; see {report_dir / 'engine-stderr.log'}") from None
        check_info(info, (PROFILE, PROFILE_NOPACE))
        tried: set[str] = set()
        for slot in plan:
            for cand in slot.pick([c for c in candidates if c.id not in tried]):
                tried.add(cand.id)
                try:
                    task = rebuilder.task(cand.record)
                    leaks_from.append((cand.record.get("question") or "", task.context))
                    print(f"{cand.id} {cand.family}: replaying", file=sys.stderr)
                    result = await replay(engine, cand, task, args, paths)
                except Skip as skip:
                    skipped[skip.code] += 1
                    print(f"{cand.id} {cand.family}: skipped, {skip}", file=sys.stderr)
                    continue
                results.append(result)
                if result.toolchain:
                    replaced[cand.id] = "toolchain"
                    print(f"{cand.id}: replaced (toolchain)", file=sys.stderr)
                    continue
                print(f"{cand.id}: {result.verdict(args.strict)}, {result.matched} of {result.requests} requests",
                      file=sys.stderr)
                break
            else:
                print(f"note: no eligible episode left for {slot.name}", file=sys.stderr)
    except EngineDied as exc:
        died = True
        print(f"the engine died: {exc}; see {report_dir / 'engine-stderr.log'}", file=sys.stderr)
    finally:
        await engine.stop()

    asked = len(plan)
    code = exit_code(results, asked, strict=args.strict, engine_died=died)
    verdict = "PASS" if code == EXIT_PASS else "FAIL"
    git_version = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
    header = {"date": args.date, "harness_commit": commit_of(harness), "rlm_commit": commit_of(ROOT),
              "engine": f"{info.get('version')}, python {info.get('python')}",
              "os": platform.platform(), "git": git_version, "shell": shell_kind(),
              "permission_mode": args.permission_mode, "digest": digest(results, args.strict)}
    c = counters(results, replaced, skipped, strict=args.strict)
    (report_dir / "report.md").write_text(report_md(header, c, results, verdict, args.strict), encoding="utf-8",
                                          errors="replace")
    (report_dir / "report.json").write_text(json.dumps(
        {"header": header, "counters": c, "verdict": verdict, "exit": code, "asked": asked,
         "episodes": [r.as_dict(args.strict) for r in results], "accepted_classes": ACCEPTED},
        indent=1, ensure_ascii=False), encoding="utf-8", errors="replace")
    for line in header_lines(header, c, verdict):
        if line:
            print(line)
    if args.write_summary:
        summary = summary_md(header, c, results, verdict, args.strict)
        leaks = guard(summary, leaks_from)
        if leaks:
            print(f"summary guard: refused ({'; '.join(leaks)})")
            return EXIT_SETUP
        (ROOT / "documentation" / "harness-parity-v0.md").write_text(summary, encoding="utf-8", newline="\n")
        print("summary guard: ok")
    kept = sum(not r.toolchain for r in results)
    if kept < asked:
        print(f"fewer eligible episodes than asked: {kept} of {asked}")
    print(f"report: {report_dir / 'report.md'}")
    return code


def main(argv: list[str] | None = None, rebuilder: Rebuilder | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run(args, rebuilder))
    except SetupError as exc:
        print(f"setup failed: {exc}", file=sys.stderr)
        return EXIT_SETUP


if __name__ == "__main__":
    sys.exit(main())
