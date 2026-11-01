"""Smoke run (rlm-harness v0, step 9): `rlmh batch --json` on 5 held-out context-search tasks with each of two models.

The first end-to-end run of the finished `rlmh` with real models. Each slot is one API model, used as both the root
and the sub-call model: slot a follows `call_format = auto` (the native `python` tool for a `tool_call` model), and
slot b is forced to fenced text, so the LoRA's format meets a real model (S-32). Runs are graded with this repo's own
scorer (`reward.score_answer`, as `reward.score` does for a training episode), and the report gives accuracy, turns
and sub-calls per run, cost and cache hit rate. Run from the repo root:

    .venv\\Scripts\\python.exe scripts\\harness_smoke.py --model-a P/M --model-b P/M              # estimate, $0
    .venv\\Scripts\\python.exe scripts\\harness_smoke.py --model-a P/M --model-b P/M --rehearse   # fake provider, $0
    .venv\\Scripts\\python.exe scripts\\harness_smoke.py --model-a P/M --model-b P/M --go         # spends
    .venv\\Scripts\\python.exe scripts\\harness_smoke.py --report --date YYYY-MM-DD               # rebuild the report

The set is `harness-smoke-v1`: `aggregation_heldout` at levels 1, 1, 2, 2 and 3 from `tasks.fixed_set`, its seeds in
the git-ignored private/seeds/. Each context goes to `<temp>/rlmh-smoke/<date>/ctx/<task id>.txt` as UTF-8 bytes, next
to a `tasks.jsonl` with no gold (D-28); `--go` rebuilds the tasks and refuses to start if a file differs from what the
estimate wrote. Outputs go to results/harness-smoke/<date>/, which the `results/harness-*/` rule keeps out of git
(D-46): per slot `events.jsonl` (every `--json` line, verbatim), rlmh's `runs.jsonl`, `stderr.log`, `slot.json` and
a `rerun/` folder; then `grades.jsonl` (answers and gold) and `report.md` (numbers and failure notes, no task text).
`--go` and `--report` also write documentation/harness-smoke-v0.md, numbers, task ids and levels only.

Only `--go` spends, and only through `rlmh`: the script makes no network call, never reads, validates or prints a key,
passes its own environment to `rlmh` unchanged, and never prints it. Batch runs are non-interactive (D-15), so a
guarded action is denied and the model sees a `PermissionError`. The cost cap (default 2 x the planned total) stops
the batch when the running total of `gateway.call` costs passes it.

Exit codes: 0 every slot ran to its end; 1 a slot was stopped (the cost cap, or a setup failure such as `auth`), or a
rehearsal check failed; 2 setup failed (no model flags, bad flags, no .gitignore rule, a task file differs, existing
output, a catalog problem, or `rlmh` exited 2).
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import bridge, reward, tasks  # noqa: E402

EXIT_OK, EXIT_STOPPED, EXIT_SETUP = 0, 1, 2

SET_NAME = "harness-smoke-v1"
FAMILY = "aggregation_heldout"
PLAN = {(FAMILY, 1): 2, (FAMILY, 2): 2, (FAMILY, 3): 1}
RESULTS = ROOT / "results" / "harness-smoke"
SUMMARY = ROOT / "documentation" / "harness-smoke-v0.md"
# Paths that must be git-ignored before anything is written (D-46; private/ holds the seeds).
IGNORE_PROBES = ("results/harness-smoke/2026-01-01/a-x/runs.jsonl", "private/seeds/harness-smoke-v1.json")
DEFAULT_TOKENS = ROOT / "results" / "prod-v2" / "transcripts"
TOKEN_FAMILIES = ("aggregation", "aggregation_heldout")

SLOTS = ("a", "b")
FORMATS = ("auto", "fenced", "native")
DEFAULT_FORMAT = {"a": "auto", "b": "fenced"}  # S-32: slot b is forced to fenced
FAKE_REF = "fake/scripted"
# rlm-harness app/src/gateway/providers.ts: the providers with `api: true`, whose `tool_call` models `auto` makes
# native (architecture §7.4). `local` and `custom` always resolve to fenced; `fake` is tool_call true.
API_PROVIDERS = ("anthropic", "openai", "deepseek", "openrouter", "gemini", "fake")
FREE_PROVIDERS = ("local", "fake")
CONFIG_KEYS = ("call_format", "permissions.mode", "root.thinking", "sub.thinking", "profile")

# infra kinds (architecture §7.8) a single rerun can cure; the setup kinds fail the same way again.
RERUN_KINDS = ("rate_limit", "server", "timeout", "network", "worker", "other", "batch_crashed")
SETUP_KINDS = ("auth", "no_key", "unknown_model", "context_window", "bad_request", "cancelled")
CAP = "not run: cost cap"
GRADED, INFRA = "graded", "infra_error"

FIRST_MESSAGE_TOKENS = 4000  # the system prompt plus the first message, for the ceiling *(mine)*
CHARS_PER_TOKEN = 3  # the gateway's own window estimate (D-25)
STOP_WAIT_S = 30
GUARD_MIN_CHARS = 10  # shorter answers ("12", "[]") are numbers a table may hold anyway
STDERR_TAIL = 20


class SetupError(Exception):
    """Exit 2."""


# ----------------------------------------------------------------------------- slots and prices


@dataclass
class Slot:
    name: str  # "a" or "b"
    ref: str  # provider/model, the root and the sub-call model
    profile: str
    fmt: str  # what the script passes: auto (no --call-format), fenced or native
    price: dict  # USD per 1M tokens: input, output, cache_read, cache_write
    tool_call: bool | None = None  # the catalog's flag; None when the catalog has no entry
    expected: str = ""  # the call format run.started should report
    limits: Any = None  # the profile's read-mode limits, for the ceiling; None is training's READ_LIMITS_V4
    limits_source: str = "`READ_LIMITS_V4`"  # where `limits` came from, as the report names it

    @property
    def provider(self) -> str:
        return self.ref.split("/", 1)[0]

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", self.ref).strip("-")

    def dir(self, out: Path) -> Path:
        return out / f"{self.name}-{self.slug}"


def format_why(fmt: str, expected: str) -> str:
    """Task 4's table: why a slot runs in its call format."""
    if fmt == "fenced":
        return "forced fenced: the LoRA's fenced format meets a real model (S-32)"
    if fmt == "native":
        return "forced native"
    if expected == "native":
        return "the product's own profile and path (auto)"
    return "auto falls back to fenced: the native path goes untested"


def split_ref(ref: str, flag: str) -> tuple[str, str]:
    provider, _, model = (ref or "").partition("/")
    if not provider or not model:
        raise SetupError(f"{flag} needs provider/model, got {ref!r}")
    return provider, model


def load_catalog(path: Path) -> dict[tuple[str, str], dict]:
    """Step 5's layout, {"version": 1, "source": ..., "models": [ModelInfo...]} (architecture §7.3), by (provider, id)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SetupError(f"can't read the catalog {path}: {e}") from e
    models = doc.get("models") if isinstance(doc, dict) else None
    if not (isinstance(doc, dict) and doc.get("version") == 1 and "source" in doc and isinstance(models, list)
            and all(isinstance(m, dict) and {"provider", "id", "price", "tool_call"} <= m.keys() for m in models)):
        raise SetupError(f"{path} is not a catalog in step 5's layout {{version: 1, source, models: [ModelInfo]}}")
    return {(m["provider"], m["id"]): m for m in models}


def parse_price(text: str, flag: str) -> dict:
    parts = [p.strip() for p in text.split(",")]
    try:
        values = [float(p) for p in parts]
    except ValueError:
        values = []
    if len(values) != 4 or any(v < 0 for v in values):
        raise SetupError(f"{flag} takes in,out,cache_read,cache_write in USD per 1M tokens, got {text!r}")
    return dict(zip(("input", "output", "cache_read", "cache_write"), values))


def load_prices(ref: str, catalog: dict | None, override: str | None, slot: str) -> tuple[dict, bool | None]:
    """A slot's prices and the catalog's tool_call flag. `local` (and the fake) cost 0 without a lookup."""
    provider, model = split_ref(ref, f"--model-{slot}")
    entry = (catalog or {}).get((provider, model))
    tool_call = entry["tool_call"] if entry else (True if provider == "fake" else None)
    if override:
        return parse_price(override, f"--price-{slot}"), tool_call
    if provider in FREE_PROVIDERS:
        return dict.fromkeys(("input", "output", "cache_read", "cache_write"), 0.0), tool_call
    if entry is None:
        raise SetupError(f"{ref} is not in the catalog: pass --price-{slot} in,out,cache_read,cache_write")
    return {k: float(entry["price"][k]) for k in ("input", "output", "cache_read", "cache_write")}, tool_call


def expected_format(slot: Slot, setting: str | None) -> str:
    """What `rlmh` resolves the slot's call format to (architecture §7.4)."""
    if slot.fmt != "auto":
        return slot.fmt
    if setting in ("fenced", "native"):
        return setting
    return "native" if slot.provider in API_PROVIDERS and slot.tool_call else "fenced"


def make_slots(args, catalog: dict | None) -> list[Slot]:
    if not (args.model_a and args.model_b):
        raise SetupError("9a has no default: pass --model-a and --model-b")
    slots = []
    for name in SLOTS:
        ref = getattr(args, f"model_{name}")
        provider, _ = split_ref(ref, f"--model-{name}")
        price, tool_call = load_prices(ref, catalog, getattr(args, f"price_{name}"), name)
        profile = getattr(args, f"profile_{name}") or ("training-v4" if provider == "local" else "default")
        fmt = getattr(args, f"format_{name}") or DEFAULT_FORMAT[name]
        harness = getattr(args, "harness", None)
        limits, source = profile_limits(profile, Path(harness)) if harness else (None, "`READ_LIMITS_V4`")
        slots.append(Slot(name, ref, profile, fmt, price, tool_call, limits=limits, limits_source=source))
    return slots


CEILING_KEYS = ("max_turns", "max_subcalls", "root_max_tokens", "output_max_chars", "subcall_max_chars",
                "subcall_max_tokens")


def profile_limits(profile: str, harness: Path, home: Path | None = None) -> tuple[Any, str]:
    """The read-mode limits `profile` runs under, read as rlm-harness reads them: training's READ_LIMITS_V4 with the
    profile's `[read]` keys on top. rlm-harness's `default` profile has more turns and output tokens than training's
    (its D-152), so its ceiling is several times training's. A built-in profile is read from the harness checkout
    (a built-in shadows a user's of the same name), a user's from the harness home. A profile in neither gets
    READ_LIMITS_V4, and the source says so, since that ceiling may be low."""
    home = home or Path(os.environ.get("RLMH_HOME") or Path.home() / ".rlm-harness")
    for path in (harness / "engine" / "rlm_engine" / "profiles" / profile / "profile.toml",
                 home / "profiles" / profile / "profile.toml"):
        if path.is_file():
            read = tomllib.loads(path.read_text(encoding="utf-8")).get("read", {})
            changes = {k: read[k] for k in CEILING_KEYS if k in read}
            return dataclasses.replace(bridge.READ_LIMITS_V4, **changes), f"the `{profile}` profile's read limits"
    return bridge.READ_LIMITS_V4, f"`READ_LIMITS_V4`: no `{profile}` profile.toml under --harness or the harness home"


# ----------------------------------------------------------------------------- reference tokens and the estimate


@dataclass
class Reference:
    """Mean tokens per run of earlier runs, from harness run logs or training transcripts."""

    n: int = 0
    notes: list = field(default_factory=list)
    root_in: float = 0.0
    root_out: float = 0.0
    sub_in: float = 0.0
    sub_out: float = 0.0
    turns: float = 0.0
    subcalls: float = 0.0

    def tokens(self) -> dict:
        return {"root_in": self.root_in, "root_out": self.root_out, "sub_in": self.sub_in, "sub_out": self.sub_out}


def read_jsonl(path: Path) -> list:
    """Every parsable line; a torn or foreign line is skipped."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def token_row(rec: Any) -> tuple | None:
    """(root in, root out, sub in, sub out, turns, sub-calls) of one finished run, or None."""
    if not isinstance(rec, dict) or rec.get("infra_error"):
        return None
    usage = rec.get("usage")
    if not isinstance(usage, dict):
        return None
    if isinstance(usage.get("root"), dict):  # an rlmh run-log record (architecture §7.8)
        if (rec.get("root") or {}).get("provider") == "fake":
            return None  # a rehearsal's scripted tokens say nothing about a real model
        root, sub = usage["root"], usage.get("sub") or {}
        return (root.get("input_tokens", 0), root.get("output_tokens", 0), sub.get("input_tokens", 0),
                sub.get("output_tokens", 0), len(rec.get("turns") or []), len(rec.get("subcalls") or []))
    if "root_prompt_tokens" in usage:  # a training transcript (rlm_rl.client.Usage)
        if rec.get("family") not in TOKEN_FAMILIES:
            return None
        return (usage["root_prompt_tokens"], usage.get("root_completion_tokens", 0),
                usage.get("subcall_prompt_tokens", 0), usage.get("subcall_completion_tokens", 0),
                usage.get("root_calls", 0), usage.get("subcalls", 0))
    return None


def load_tokens(paths: list) -> Reference:
    ref, rows = Reference(), []
    for path in map(Path, paths):
        if not path.exists():
            ref.notes.append(f"{path}: missing")
            continue
        files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
        got = [row for f in files for row in map(token_row, read_jsonl(f)) if row is not None]
        rows.extend(got)
        ref.notes.append(f"{path}: {len(got)} runs")
    if rows:
        ref.n = len(rows)
        means = [sum(col) / len(rows) for col in zip(*rows)]
        ref.root_in, ref.root_out, ref.sub_in, ref.sub_out, ref.turns, ref.subcalls = means
    return ref


def role_cost(tokens_in: float, tokens_out: float, price: dict, cache_read: float = 0.0,
              cache_write: float = 0.0) -> float:
    """D-22: uncached input, cache reads, cache writes and output, each at its own price per 1M tokens."""
    uncached = max(0.0, tokens_in - cache_read - cache_write)
    return (uncached * price["input"] + cache_read * price["cache_read"] + cache_write * price["cache_write"]
            + tokens_out * price["output"]) / 1e6


def run_cost(tok: dict, price: dict) -> float:
    return role_cost(tok["root_in"], tok["root_out"], price) + role_cost(tok["sub_in"], tok["sub_out"], price)


def ceiling_tokens(lim: Any = bridge.READ_LIMITS_V4) -> dict:
    """One run with every limit used: chars / 3 per prompt token (D-25), each turn's prompt growing by a full reply
    and a full output."""
    per_turn = lim.root_max_tokens + lim.output_max_chars / CHARS_PER_TOKEN
    return {"root_in": sum(FIRST_MESSAGE_TOKENS + (t - 1) * per_turn for t in range(1, lim.max_turns + 1)),
            "root_out": float(lim.max_turns * lim.root_max_tokens),
            "sub_in": lim.max_subcalls * lim.subcall_max_chars / CHARS_PER_TOKEN,
            "sub_out": float(lim.max_subcalls * lim.subcall_max_tokens)}


def estimate(slots: list[Slot], ref: Reference, runs: int, margin: float, cap: float | None) -> dict:
    """Expected (reference tokens, no cache reads), planned (x margin) and ceiling cost per slot. With no reference
    runs the ceiling stands in for planned (D-115)."""
    ceilings = {s.name: ceiling_tokens(s.limits or bridge.READ_LIMITS_V4) for s in slots}
    # the report's one ceiling line shows the largest; each slot's cost uses its own profile's
    widest = max(slots, key=lambda s: sum(ceilings[s.name].values()), default=None)
    out = {"runs": runs, "margin": margin, "reference_runs": ref.n,
           "ceiling_tokens": ceilings[widest.name] if widest else ceiling_tokens(),
           "ceiling_source": widest.limits_source if widest else "`READ_LIMITS_V4`", "slots": {}}
    for s in slots:
        ceil = ceilings[s.name]
        ceiling_run = run_cost(ceil, s.price)
        expected_run = run_cost(ref.tokens(), s.price) if ref.n else None
        planned_run = expected_run * margin if expected_run is not None else ceiling_run
        out["slots"][s.name] = {"ceiling_tokens": ceil, "ceiling_source": s.limits_source,
                                "expected_run": expected_run, "planned_run": planned_run, "ceiling_run": ceiling_run,
                                "expected": None if expected_run is None else expected_run * runs,
                                "planned": planned_run * runs, "ceiling": ceiling_run * runs}
    out["planned_total"] = sum(v["planned"] for v in out["slots"].values())
    out["cap"] = cap if cap is not None else 2 * out["planned_total"]
    return out


# ----------------------------------------------------------------------------- tasks


def build_tasks() -> list:
    built = tasks.fixed_set(SET_NAME, PLAN)
    levels = sorted(t.level for t in built)
    want = sorted(level for (_, level), n in PLAN.items() for _ in range(n))
    if levels != want or any(t.family != FAMILY for t in built):
        raise SetupError(f"{SET_NAME} built levels {levels}, expected {want}")
    return built


def context_sha(text: str) -> str:
    """As run.started.input.sha256 defines it (architecture §5.3)."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def tasks_jsonl(task_list: list) -> bytes:
    """D-28 read-mode lines with `/` paths, relative to the file's folder. No gold."""
    lines = [json.dumps({"id": t.task_id, "question": t.question, "input": f"ctx/{t.task_id}.txt"}, ensure_ascii=False)
             for t in task_list]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_task_files(task_list: list, folder: Path, name: str = "tasks.jsonl") -> Path:
    """Contexts as bytes: text mode on Windows would turn \\n into \\r\\n and change what the model reads."""
    (folder / "ctx").mkdir(parents=True, exist_ok=True)
    for t in task_list:
        (folder / "ctx" / f"{t.task_id}.txt").write_bytes(t.context.encode("utf-8"))
    path = folder / name
    path.write_bytes(tasks_jsonl(task_list))
    return path


def check_task_files(task_list: list, folder: Path) -> list[str]:
    """What differs between the rebuilt tasks and the files the estimate wrote."""
    bad = []
    for t in task_list:
        p = folder / "ctx" / f"{t.task_id}.txt"
        if not p.is_file() or p.read_bytes() != t.context.encode("utf-8"):
            bad.append(f"ctx/{t.task_id}.txt")
    p = folder / "tasks.jsonl"
    if not p.is_file() or p.read_bytes() != tasks_jsonl(task_list):
        bad.append("tasks.jsonl")
    return bad


def task_folder(date: str) -> Path:
    return Path(tempfile.gettempdir()) / "rlmh-smoke" / date


# ----------------------------------------------------------------------------- rlmh and git


def rlmh_command(path: str | None) -> list[str]:
    """The binary as an argument list. A .py path runs under this Python (the tests' stub)."""
    found = path or shutil.which("rlmh")
    if not found:
        raise SetupError("rlmh is not on PATH: pass --rlmh PATH")
    return [sys.executable, found] if found.endswith(".py") else [found]


def local_read(cmd: list[str], args: list[str]) -> str:
    """`rlmh --version` or `rlmh config get KEY`: local reads that start no engine and touch no network."""
    try:
        done = subprocess.run([*cmd, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              stdin=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SetupError(f"can't run {cmd[-1]} {' '.join(args)}: {e}") from e
    if done.returncode != 0:
        raise SetupError(f"{cmd[-1]} {' '.join(args)} exited {done.returncode}: {done.stderr.strip()[-500:]}")
    return done.stdout.strip()


def rlmh_info(cmd: list[str]) -> dict:
    return {"version": local_read(cmd, ["--version"]),
            "config": {k: local_read(cmd, ["config", "get", k]) for k in CONFIG_KEYS}}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def check_ignored() -> None:
    for probe in IGNORE_PROBES:
        if git("check-ignore", "-q", probe).returncode != 0:
            raise SetupError(f"{probe} is not git-ignored: add `results/harness-*/` (D-46) and `private/` to "
                             ".gitignore before anything is written")


def rlm_commit() -> str:
    head = git("rev-parse", "HEAD").stdout.strip() or "unknown"
    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    return f"{head} (uncommitted changes)" if dirty else head


# ----------------------------------------------------------------------------- one rlmh batch


@dataclass
class Stream:
    """What one `rlmh batch --json` printed, by task id (the run's tag)."""

    started: dict = field(default_factory=dict)
    finished: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)  # run_id -> task id
    denied: collections.Counter = field(default_factory=collections.Counter)  # task id -> denied actions
    cost: float = 0.0  # every gateway.call's usage.cost_usd
    calls: int = 0
    call_errors: collections.Counter = field(default_factory=collections.Counter)  # error kind -> calls
    batch: dict | None = None
    bad_lines: list = field(default_factory=list)
    lines: int = 0


def parse_line(line: str, stream: Stream) -> dict | None:
    """Fold one --json line into `stream` and return the event. A line that isn't a JSON object is noted by number
    and length only, so no task text reaches the report; events.jsonl keeps it verbatim."""
    stream.lines += 1
    try:
        obj = json.loads(line)
    except ValueError:
        obj = None
    if not isinstance(obj, dict) or not isinstance(obj.get("type"), str):
        if line.strip():
            stream.bad_lines.append(f"line {stream.lines}: not a JSON event ({len(line.rstrip())} chars)")
        return None
    kind = obj["type"]
    if kind == "run.started":
        stream.tags[obj.get("run_id")] = obj.get("tag")
        stream.started[obj.get("tag")] = obj
    elif kind == "run.finished":
        stream.finished[obj.get("tag")] = obj
    elif kind == "run.permission" and not obj.get("allowed"):
        stream.denied[stream.tags.get(obj.get("run_id"), obj.get("run_id"))] += 1
    elif kind == "gateway.call":
        stream.calls += 1
        stream.cost += float((obj.get("usage") or {}).get("cost_usd") or 0.0)
        if obj.get("error"):
            stream.call_errors[(obj["error"] or {}).get("kind", "other")] += 1
    elif kind == "batch.finished":
        stream.batch = obj
    return obj


def read_stream(path: Path) -> Stream:
    stream = Stream()
    if path.is_file():
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for line in f:
                parse_line(line, stream)
    return stream


def batch_command(cmd: list[str], tasks_file: Path, slot: Slot, out: Path, jobs: int, ref: str | None = None) -> list:
    """No shell, no --permission-mode (D-15 denies what would ask). `auto` passes no --call-format (§7.4)."""
    ref = ref or slot.ref
    args = [*cmd, "batch", str(tasks_file), "--json", "--jobs", str(jobs), "--out", str(out),
            "--root", ref, "--sub", ref, "--profile", slot.profile]
    if slot.fmt != "auto":
        args += ["--call-format", slot.fmt]
    return args


@dataclass
class Attempt:
    stream: Stream
    exit_code: int | None
    capped: bool


def run_slot(args: list, out: Path, *, env: dict | None = None, spent: float = 0.0, cap: float | None = None,
             label: str = "") -> Attempt:
    """Run one batch, appending each stdout line to events.jsonl before parsing it. When `spent` plus this batch's
    gateway costs passes `cap`, the process is terminated (rlmh's engine then sees its stdin close and exits)."""
    out.mkdir(parents=True, exist_ok=True)
    stream, capped, done = Stream(), False, False
    with open(out / "stderr.log", "w", encoding="utf-8", errors="replace") as err, \
            open(out / "events.jsonl", "a", encoding="utf-8", newline="") as events:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, text=True,
                                encoding="utf-8", errors="replace", env=env)
        try:
            for line in proc.stdout:
                events.write(line)
                events.flush()
                event = parse_line(line, stream)
                if event and event["type"] == "run.finished":
                    progress(label, event, spent + stream.cost)
                if cap is not None and spent + stream.cost > cap:
                    capped = True
                    print(f"[{label}] cost cap ${cap:.4f} passed at ${spent + stream.cost:.4f}: stopping", flush=True)
                    break
            done = not capped
        finally:
            if not done and proc.poll() is None:
                proc.terminate()
            proc.stdout.close()
            try:
                code = proc.wait(timeout=STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
    return Attempt(stream, code, capped)


def progress(label: str, f: dict, total: float) -> None:
    status = f"infra {f.get('infra_kind')}" if f.get("infra_error") else ("submitted" if f.get("submitted")
                                                                          else "no answer")
    print(f"[{label}] {f.get('tag')}: {status}, {f.get('turns')} turns, {f.get('subcalls')} sub-calls, "
          f"${(f.get('cost_usd') or 0):.4f}; so far ${total:.4f}", flush=True)


def tail(path: Path, n: int = STDERR_TAIL) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


# ----------------------------------------------------------------------------- outcomes, reruns and the slot record


def outcome(stream: Stream, capped: bool, task_id: str) -> tuple[str, dict | None, str | None, str | None]:
    """(status, run.finished, infra kind, message) of one task in one attempt."""
    f = stream.finished.get(task_id)
    if f is not None and not f.get("infra_error"):
        return GRADED, f, None, None
    if f is not None:
        return INFRA, f, f.get("infra_kind") or "other", f.get("infra_error")
    if capped:
        return CAP, None, None, None
    return INFRA, None, "batch_crashed", "no run.finished: the batch ended without one"


def plan_reruns(task_ids: list[str], attempt: Attempt) -> tuple[list[dict], list[dict]]:
    """(reruns, setup failures). One rerun for a kind a repeat can cure; never for a setup kind."""
    reruns, setup = [], []
    for tid in task_ids:
        status, _, kind, message = outcome(attempt.stream, attempt.capped, tid)
        if status == INFRA:
            (reruns if kind in RERUN_KINDS else setup).append({"task_id": tid, "kind": kind, "message": message})
    return reruns, setup


def write_json(path: Path, obj: Any) -> None:
    path.write_bytes((json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))


def run_batch(slots: list[Slot], task_list: list, folder: Path, out: Path, cmd: list[str], jobs: int,
              cap: float | None, env_for=lambda slot: None, ref_for=lambda slot: None) -> str | None:
    """Slots in order, each with its single rerun. Returns why later slots stopped, if they did. Each slot's
    slot.json records what the report needs besides the event streams."""
    ids = [t.task_id for t in task_list]
    spent, stop = 0.0, None
    for slot in slots:
        sdir = slot.dir(out)
        sdir.mkdir(parents=True, exist_ok=True)
        meta = {"slot": slot.name, "ref": ref_for(slot) or slot.ref, "model": slot.ref,
                **{k: getattr(slot, k) for k in ("profile", "fmt", "price", "tool_call", "expected")},
                "not_run": stop, "exit_code": None, "capped": False, "reruns": [], "setup_failures": [],
                "rerun_exit_code": None, "rerun_capped": False}
        if stop:
            write_json(sdir / "slot.json", meta)
            continue
        args = batch_command(cmd, folder / "tasks.jsonl", slot, sdir, jobs, ref_for(slot))
        print(f"[{slot.name}] {' '.join(args)}", flush=True)
        first = run_slot(args, sdir, env=env_for(slot), spent=spent, cap=cap, label=slot.name)
        spent += first.stream.cost
        meta.update(exit_code=first.exit_code, capped=first.capped)
        if first.exit_code == EXIT_SETUP:
            write_json(sdir / "slot.json", meta)
            raise SetupError(f"rlmh batch exited 2 (a usage error) for slot {slot.name}:\n{tail(sdir / 'stderr.log')}")
        if first.capped:
            stop = CAP
        else:
            reruns, setup = plan_reruns(ids, first)
            meta.update(reruns=reruns, setup_failures=setup)
            if reruns:
                again = [t for t in task_list if t.task_id in {r["task_id"] for r in reruns}]
                rfile = folder / "tasks-rerun.jsonl"
                rfile.write_bytes(tasks_jsonl(again))
                print(f"[{slot.name}] rerunning {len(again)}: " + ", ".join(f"{r['task_id']} ({r['kind']})"
                                                                           for r in reruns), flush=True)
                rargs = batch_command(cmd, rfile, slot, sdir / "rerun", jobs, ref_for(slot))
                second = run_slot(rargs, sdir / "rerun", env=env_for(slot), spent=spent, cap=cap,
                                  label=f"{slot.name} rerun")
                spent += second.stream.cost
                meta.update(rerun_exit_code=second.exit_code, rerun_capped=second.capped)
                if second.exit_code == EXIT_SETUP:
                    write_json(sdir / "slot.json", meta)
                    raise SetupError(f"rlmh batch exited 2 on slot {slot.name}'s rerun:\n"
                                     f"{tail(sdir / 'rerun' / 'stderr.log')}")
                if second.capped:
                    stop = CAP
            if setup and not stop:
                stop = f"not run: setup failure in slot {slot.name}"
                print(f"[{slot.name}] setup failures, not rerun; tell Rishi: " + ", ".join(
                    f"{s['task_id']} ({s['kind']})" for s in setup), flush=True)
        write_json(sdir / "slot.json", meta)
    return stop


# ----------------------------------------------------------------------------- grading and the report


def grade(f: dict, task: Any) -> reward.Score:
    """As reward.score does for a training episode: no valid submit scores as no answer."""
    answer = f.get("answer") if f.get("submitted") and f.get("answer_valid") else None
    return reward.score_answer(answer, task.answer, task.grading)


def final_rows(meta: dict, first: Stream, rerun: Stream | None, task_list: list) -> list[dict]:
    """One row per task: the first attempt, or its rerun when it had one (the rerun's result replaces it)."""
    rows = []
    rerun_ids = {r["task_id"]: r for r in meta.get("reruns") or []}
    for t in task_list:
        row = {"slot": meta["slot"], "model": meta["model"], "task_id": t.task_id, "level": t.level,
               "reran": t.task_id in rerun_ids, "first_kind": None, "first_message": None}
        if meta.get("not_run"):
            status, f, kind, message = meta["not_run"], None, None, None
        else:
            status, f, kind, message = outcome(first, meta.get("capped", False), t.task_id)
            src = first
            if row["reran"]:
                row["first_kind"], row["first_message"] = kind, message
                status, f, kind, message = outcome(rerun or Stream(), meta.get("rerun_capped", False), t.task_id)
                src = rerun or Stream()
            row["started"] = src.started.get(t.task_id)
            row["denied"] = src.denied.get(t.task_id, 0)
        row.update(status=status, finished=f, infra_kind=kind, infra_error=message,
                   run_id=(f or {}).get("run_id"))
        row.setdefault("started", None)
        row.setdefault("denied", 0)
        if status == GRADED:
            score = grade(f, t)
            row.update(exact=score.correct, reward=score.reward, components=score.components)
        else:
            row.update(exact=None, reward=None, components=None)
        rows.append(row)
    return rows


def load_slot(sdir: Path, task_list: list) -> tuple[dict, list[dict], Stream, Stream | None]:
    meta = json.loads((sdir / "slot.json").read_text(encoding="utf-8"))
    first = read_stream(sdir / "events.jsonl")
    rerun = read_stream(sdir / "rerun" / "events.jsonl") if (sdir / "rerun" / "events.jsonl").is_file() else None
    return meta, final_rows(meta, first, rerun, task_list), first, rerun


def grade_line(row: dict, task: Any) -> dict:
    f = row["finished"] or {}
    return {"slot": row["slot"], "model": row["model"], "task_id": row["task_id"], "level": row["level"],
            "run_id": row["run_id"], "status": row["status"], "infra_kind": row["infra_kind"],
            "submitted": f.get("submitted"), "answer_valid": f.get("answer_valid"), "answer": f.get("answer"),
            "gold": task.answer, "exact": row["exact"], "reward": row["reward"], "components": row["components"],
            "reran": row["reran"]}


def _mean(xs: list) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _usage_sum(rows: list, role: str) -> dict:
    keys = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    out = dict.fromkeys(keys, 0)
    for r in rows:
        u = ((r["finished"] or {}).get("usage") or {}).get(role) or {}
        for k in keys:
            out[k] += u.get(k) or 0
    return out


def slot_stats(meta: dict, rows: list, first: Stream, rerun: Stream | None) -> dict:
    graded = [r for r in rows if r["status"] == GRADED]
    with_record = [r for r in rows if r["finished"]]
    root, sub = _usage_sum(with_record, "root"), _usage_sum(with_record, "sub")
    reads = root["cache_read_tokens"] + sub["cache_read_tokens"]

    def rate(u: dict) -> float | None:
        return u["cache_read_tokens"] / u["input_tokens"] if u["input_tokens"] else None

    total_in = root["input_tokens"] + sub["input_tokens"]
    return {
        "graded": len(graded), "infra": sum(r["status"] == INFRA for r in rows),
        "not_run": sum(r["status"].startswith("not run") for r in rows),
        "exact": sum(bool(r["exact"]) for r in graded), "mean_reward": _mean([r["reward"] for r in graded]),
        "mean_turns": _mean([r["finished"]["turns"] for r in graded]),
        "mean_subcalls": _mean([r["finished"]["subcalls"] for r in graded]),
        "cost": sum((r["finished"].get("cost_usd") or 0.0) for r in with_record),
        "spend": first.cost + (rerun.cost if rerun else 0.0),
        "cache_root": rate(root), "cache_sub": rate(sub),
        "cache_all": reads / total_in if total_in else None, "cache_reported": reads > 0,
        "mean_wall": _mean([r["finished"]["wall_s"] for r in graded]),
        "denied": sum(r["denied"] for r in rows),
        "formats": sorted({(r["started"] or {}).get("call_format") for r in rows if r["started"]} - {None}),
        "profiles": sorted({(r["started"] or {}).get("profile") for r in rows if r["started"]} - {None}),
        "call_errors": dict(first.call_errors + (rerun.call_errors if rerun else collections.Counter())),
    }


def checks(rows: list, task_list: list, first: Stream, rerun: Stream | None, meta: dict) -> list[str]:
    """What a rehearsal must find clean: every event parses, each run saw its whole context, the format matches."""
    by_id = {t.task_id: t for t in task_list}
    problems = [f"slot {meta['slot']}: {b}" for b in first.bad_lines]
    problems += [f"slot {meta['slot']} rerun: {b}" for b in (rerun.bad_lines if rerun else [])]
    for r in rows:
        s = r["started"]
        if not s:
            continue
        t = by_id[r["task_id"]]
        inp = s.get("input") or {}
        if inp.get("chars") != len(t.context) or inp.get("sha256") != context_sha(t.context):
            problems.append(f"slot {meta['slot']} {r['task_id']}: run.started.input is {inp.get('chars')} chars, "
                            f"the context is {len(t.context)} (sha256 {'equal' if inp.get('sha256') == context_sha(t.context) else 'differs'})")
        if s.get("call_format") != meta.get("expected"):
            problems.append(f"slot {meta['slot']} {r['task_id']}: call_format {s.get('call_format')}, expected "
                            f"{meta.get('expected')} (Task 4's table)")
        if s.get("profile") != meta.get("profile"):
            problems.append(f"slot {meta['slot']} {r['task_id']}: profile {s.get('profile')}, asked {meta.get('profile')}")
    return problems


def _usd(x: float | None, places: int = 4) -> str:
    return "n/a" if x is None else f"${x:,.{places}f}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None, places: int = 1) -> str:
    return "n/a" if x is None else f"{x:,.{places}f}"


def render(run: dict, slots: list[tuple[dict, list, dict]], problems: list[str], notes: bool) -> str:
    """report.md (notes=True) or the summary doc: numbers, task ids and levels only, never task text."""
    est = run.get("estimate") or {}
    stopped = run.get("stopped")
    lines = []
    title = "Harness smoke run" if notes else "Harness smoke run (rlm-harness v0, step 9)"
    lines += [f"# {title}, {run['date']}", ""]
    if stopped:
        lines += [f"**Stopped: {stopped}.** Later runs were not run; see the tables.", ""]
    if not notes:
        lines += ["Generated by `scripts/harness_smoke.py` from the run's own records. It holds numbers, task ids and "
                  "levels only: the questions, answers and gold stay in the git-ignored "
                  f"`results/harness-smoke/{run['date']}/`.", ""]
    levels = ", ".join(str(t["level"]) for t in run["tasks"])
    lines += [f"- Set `{SET_NAME}`: {len(run['tasks'])} tasks of `{FAMILY}`, levels {levels}, generator "
              f"`{run.get('generator')}`.",
              f"- `{run.get('rlmh_version')}`, RLM commit `{run.get('rlm_commit')}`, {run.get('platform')}.",
              f"- rlmh config: " + ", ".join(f"`{k} = {v}`" for k, v in (run.get("config") or {}).items()) + ".",
              f"- Each slot's model is both the root and the sub-call model. Batch runs are non-interactive (D-15): "
              f"a guarded action is denied.", ""]
    lines += ["## Slots", "", "| Slot | Model | Profile | Call format | Seen in run.started | Why |",
              "|---|---|---|---|---|---|"]
    for meta, _, st in slots:
        seen = ", ".join(st["formats"]) or "none"
        flag = "" if not st["formats"] or st["formats"] == [meta["expected"]] else " **mismatch**"
        lines.append(f"| {meta['slot']} | `{meta['model']}` | `{meta['profile']}` | {meta['expected']} "
                     f"({'--call-format ' + meta['fmt'] if meta['fmt'] != 'auto' else 'auto'}) | {seen}{flag} | "
                     f"{format_why(meta['fmt'], meta['expected'])} |")
    lines += ["", "Slot b is forced to fenced text (S-32). Both slots use the same profile, so they differ only in model "
              "and call format. Five tasks per model rank nothing.", ""]

    lines += ["## Results", "", "| | " + " | ".join(f"slot {m['slot']}" for m, _, _ in slots) + " |",
              "|---|" + "---|" * len(slots)]

    def row(label: str, fn) -> None:
        lines.append(f"| {label} | " + " | ".join(fn(m, r, s) for m, r, s in slots) + " |")

    row("Model", lambda m, r, s: f"`{m['ref']}`" + (f" (stand-in for `{m['model']}`)" if m["ref"] != m["model"] else ""))
    row("Runs graded / infra-failed / not run", lambda m, r, s: f"{s['graded']} / {s['infra']} / {s['not_run']}")
    row("Accuracy (exact / graded)", lambda m, r, s: f"{s['exact']} / {s['graded']}")
    row("Mean reward (partial credit)", lambda m, r, s: _num(s["mean_reward"], 3))
    row("Mean turns", lambda m, r, s: _num(s["mean_turns"]))
    row("Mean sub-calls", lambda m, r, s: _num(s["mean_subcalls"]))
    row("Cost of the final runs", lambda m, r, s: _usd(s["cost"]))
    row("Cost per final run", lambda m, r, s: _usd(s["cost"] / len([x for x in r if x["finished"]]))
        if any(x["finished"] for x in r) else "n/a")
    row("Spend, every call (reruns included)", lambda m, r, s: _usd(s["spend"]))
    row("Cache hit rate, root", lambda m, r, s: _pct(s["cache_root"]) if s["cache_reported"] else "none reported")
    row("Cache hit rate, sub-calls", lambda m, r, s: _pct(s["cache_sub"]) if s["cache_reported"] else "none reported")
    row("Cache hit rate, overall", lambda m, r, s: _pct(s["cache_all"]) if s["cache_reported"] else "none reported")
    row("Mean wall time (s)", lambda m, r, s: _num(s["mean_wall"]))
    row("Denied actions", lambda m, r, s: str(s["denied"]))
    row("Failed model calls (kind: n)", lambda m, r, s: ", ".join(f"{k}: {v}" for k, v in sorted(s["call_errors"].items()))
        or "none")
    lines += ["", "Cache hit rate is cache reads over input tokens (D-22). A provider may report no cache tokens at all, "
              "and then the rate says nothing about caching.", ""]

    lines += ["## Per run", "", "| Slot | Task | Level | Status | Exact | Reward | Turns | Sub-calls | Cost | Wall (s) "
              "| Denied | Format | Reran |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for meta, rows, _ in slots:
        for r in rows:
            f = r["finished"] or {}
            status = r["status"] if r["status"] != INFRA else f"infra: {r['infra_kind']}"
            exact = "" if r["exact"] is None else ("yes" if r["exact"] else "no")
            score = "" if r["reward"] is None else f"{r['reward']:.3f}"
            lines.append(
                f"| {r['slot']} | `{r['task_id']}` | {r['level']} | {status} | {exact} | {score} | "
                f"{f.get('turns', '')} | {f.get('subcalls', '')} | {_usd(f.get('cost_usd')) if f else ''} | "
                f"{_num(f.get('wall_s')) if f else ''} | {r['denied']} | "
                f"{(r['started'] or {}).get('call_format', '')} | {'yes' if r['reran'] else ''} |")
    lines.append("")

    lines += ["## Estimate against actual", ""]
    if est:
        lines += [f"Reference runs: {est.get('reference_runs')} ({'; '.join(run.get('reference_notes') or [])}). "
                  f"Margin {est.get('margin')}. Ceiling per run: {est['ceiling_tokens']['root_in'] + est['ceiling_tokens']['sub_in']:,.0f} "
                  f"tokens in, {est['ceiling_tokens']['root_out'] + est['ceiling_tokens']['sub_out']:,.0f} out "
                  f"({est.get('ceiling_source', '`READ_LIMITS_V4`')}). Cost cap {_usd(est.get('cap'))}.", ""]
        sources = {name: e.get("ceiling_source") for name, e in est["slots"].items() if e.get("ceiling_source")}
        if len(set(sources.values())) > 1:  # a slot priced differently, a fallback to training's limits included
            lines += ["Each slot's ceiling: " + "; ".join(f"slot {name} from {src}" for name, src in sorted(sources.items()))
                      + ".", ""]
        lines += ["| Slot | Expected | Planned | Ceiling | Actual spend |", "|---|---|---|---|---|"]
        for meta, _, st in slots:
            e = est["slots"].get(meta["slot"], {})
            lines.append(f"| {meta['slot']} | {_usd(e.get('expected'))} | {_usd(e.get('planned'))} | "
                         f"{_usd(e.get('ceiling'))} | {_usd(st['spend'])} |")
        lines.append("")
    else:
        lines += ["No estimate was recorded with this run.", ""]

    lines += ["## Reruns", ""]
    reran = [(meta, r) for meta, rows, _ in slots for r in rows if r["reran"]]
    if reran:
        head = "| Slot | Task | First failure | Message | Outcome |" if notes else "| Slot | Task | First failure | Outcome |"
        lines += [head, "|---|---|---|---|---|" if notes else "|---|---|---|---|"]
        for meta, r in reran:
            out_text = "graded" if r["status"] == GRADED else (f"failed again: {r['infra_kind']}"
                                                               if r["status"] == INFRA else r["status"])
            msg = f" {(r['first_message'] or '')[:200]} |" if notes else ""
            lines.append(f"| {r['slot']} | `{r['task_id']}` | {r['first_kind']} |{msg} {out_text} |")
    else:
        lines.append("None.")
    lines.append("")
    setup = [(m["slot"], s) for m, _, _ in slots for s in (m.get("setup_failures") or [])]
    if setup:
        lines += ["## Setup failures (not rerun; for Rishi)", ""]
        lines += [f"- slot {sl} `{s['task_id']}`: {s['kind']}" + (f": {(s['message'] or '')[:300]}" if notes else "")
                  for sl, s in setup]
        lines.append("")
    if notes:
        lines += ["## Failure notes", ""]
        infra = [r for _, rows, _ in slots for r in rows if r["status"] == INFRA]
        lines += [f"- slot {r['slot']} `{r['task_id']}`: {r['infra_kind']}: {(r['infra_error'] or '')[:300]}"
                  for r in infra]
        lines += [f"- {p}" for p in problems]
        for meta, _, _ in slots:
            if meta.get("exit_code") not in (None, 0, 1):
                lines.append(f"- slot {meta['slot']}: rlmh batch exited {meta['exit_code']}")
        if not infra and not problems:
            lines.append("None.")
        lines.append("")
    return "\n".join(lines)


def guard(texts: dict[Path, str], task_list: list, rows: list) -> None:
    """Refuse a report or summary that holds a question, a gold answer or a model's answer."""
    needles = [t.question for t in task_list]
    needles += [a if isinstance(a, str) else json.dumps(a, ensure_ascii=False) for a in [t.answer for t in task_list]]
    for r in rows:
        a = (r["finished"] or {}).get("answer")
        if a is not None:
            needles.append(a if isinstance(a, str) else json.dumps(a, ensure_ascii=False))
    needles = [n for n in needles if len(n.strip()) >= GUARD_MIN_CHARS]
    for path, text in texts.items():
        if any(n in text for n in needles):
            raise SetupError(f"report guard: task text found in {path.name}; nothing written")


def write_report(out: Path, task_list: list, summary: Path | None) -> tuple[list, list[str]]:
    """grades.jsonl, report.md and (for a real run) the summary, all rebuilt from the files in `out`."""
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    by_id = {t.task_id: t for t in task_list}
    slots, all_rows, problems = [], [], []
    for sdir in sorted(p for p in out.iterdir() if p.is_dir() and re.match(r"^[ab]-", p.name)):
        if not (sdir / "slot.json").is_file():
            continue
        meta, rows, first, rerun = load_slot(sdir, task_list)
        slots.append((meta, rows, slot_stats(meta, rows, first, rerun)))
        all_rows += rows
        problems += checks(rows, task_list, first, rerun, meta)
    if not slots:
        raise SetupError(f"no slot folders with slot.json under {out}")
    grades = "".join(json.dumps(grade_line(r, by_id[r["task_id"]]), ensure_ascii=False) + "\n" for r in all_rows)
    (out / "grades.jsonl").write_bytes(grades.encode("utf-8"))
    texts = {out / "report.md": render(run, slots, problems, notes=True)}
    if summary is not None:
        texts[summary] = render(run, slots, [], notes=False)
    guard(texts, task_list, all_rows)
    for path, text in texts.items():  # LF on every OS
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return all_rows, problems


# ----------------------------------------------------------------------------- modes


def fake_script(fmt: str) -> dict:
    """The rehearsal's RLMH_FAKE_SCRIPT (architecture §7.5): the root submits '0' at once, in the slot's format.
    Fenced format reads only the text, native only the tool call."""
    root = {"text": "", "tool_call": {"code": "submit('0')"}} if fmt == "native" else \
        {"text": "```python\nsubmit('0')\n```"}
    return {"root": [root], "sub": [{"text": "0"}]}


def print_estimate(slots: list[Slot], est: dict, ref: Reference, task_list: list, folder: Path, info: dict,
                   cmd: list[str]) -> None:
    sizes = ", ".join(f"L{t.level} {len(t.context):,}" for t in task_list)
    ceil = est["ceiling_tokens"]
    print(f"Smoke run estimate: {SET_NAME}, {len(task_list)} tasks of {FAMILY} per slot")
    print(f"  context chars: {sizes}")
    print(f"  task files (no gold): {folder}")
    print(f"  rlmh: {info['version']} ({cmd[-1]})")
    print("  rlmh config: " + ", ".join(f"{k} = {v}" for k, v in info["config"].items()))
    print(f"  reference: n = {ref.n} ({'; '.join(ref.notes) or 'no source'})")
    if ref.n:
        print(f"    per run: root {ref.root_in:,.0f} in / {ref.root_out:,.0f} out, sub {ref.sub_in:,.0f} in / "
              f"{ref.sub_out:,.0f} out, {ref.turns:.1f} turns, {ref.subcalls:.1f} sub-calls")
    else:
        print("    no reference runs: expected is unknown, and the ceiling stands in for planned (D-115)")
    print(f"  ceiling per run ({est['ceiling_source'].replace('`', '')}): root {ceil['root_in']:,.0f} in / "
          f"{ceil['root_out']:,.0f} out, "
          f"sub {ceil['sub_in']:,.0f} in / {ceil['sub_out']:,.0f} out")
    for s in slots:
        e = est["slots"][s.name]
        p = s.price
        print(f"  slot {s.name}: {s.ref}, profile {s.profile}, call format {s.expected} "
              f"({'auto' if s.fmt == 'auto' else '--call-format ' + s.fmt}); catalog tool_call {s.tool_call}")
        print(f"    price per 1M: in ${p['input']}, out ${p['output']}, cache read ${p['cache_read']}, "
              f"cache write ${p['cache_write']}")
        print(f"    ceiling from {e['ceiling_source'].replace('`', '')}")
        print(f"    per run: expected {_usd(e['expected_run'])}, planned {_usd(e['planned_run'])}, "
              f"ceiling {_usd(e['ceiling_run'])}")
        print(f"    {est['runs']} runs: expected {_usd(e['expected'])}, planned {_usd(e['planned'])}, "
              f"ceiling {_usd(e['ceiling'])}")
        if s.fmt == "auto" and s.expected == "fenced":
            print(f"    note: slot {s.name} resolves to fenced, so the native path goes untested (Task 4)")
    print(f"  planned total {_usd(est['planned_total'])}, cost cap {_usd(est['cap'])} over every slot")
    print("  nothing runs until --go")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="rlm-harness v0 step 9: the smoke run")
    for s in SLOTS:
        p.add_argument(f"--model-{s}", help="provider/model, the root and the sub-call model (9a: no default)")
        p.add_argument(f"--profile-{s}", help="default: training-v4 for provider local, else default")
        p.add_argument(f"--format-{s}", choices=FORMATS, default=DEFAULT_FORMAT[s])
        p.add_argument(f"--price-{s}", help="in,out,cache_read,cache_write in USD per 1M tokens")
    mode = p.add_mutually_exclusive_group()
    for m in ("estimate", "rehearse", "go", "report"):
        mode.add_argument(f"--{m}", dest="mode", action="store_const", const=m)
    p.set_defaults(mode="estimate")
    p.add_argument("--rlmh", help="the rlmh binary (default: rlmh on PATH)")
    p.add_argument("--harness", default=str(ROOT.parent / "rlm-harness"), help="finds app/catalog/models.json")
    p.add_argument("--catalog")
    p.add_argument("--tokens-from", action="append", help="a runs.jsonl or a transcripts folder (repeatable)")
    p.add_argument("--margin", type=float, default=1.5)
    p.add_argument("--cost-cap", type=float)
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--only", choices=SLOTS)
    p.add_argument("--date", default=datetime.date.today().isoformat())
    args = p.parse_args(argv)
    try:
        return run_mode(args)
    except SetupError as e:
        print(f"harness_smoke: {e}", file=sys.stderr)
        return EXIT_SETUP


def run_mode(args) -> int:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        raise SetupError(f"--date takes YYYY-MM-DD, got {args.date!r}")
    if args.jobs < 1 or args.margin <= 0 or (args.cost_cap is not None and args.cost_cap <= 0):
        raise SetupError("--jobs, --margin and --cost-cap must be positive")
    if args.mode == "report":
        check_ignored()
        out = RESULTS / args.date
        if not (out / "run.json").is_file():
            raise SetupError(f"no run under {out} (run.json is missing)")
        _, problems = write_report(out, build_tasks(), SUMMARY)
        print(f"report: {out / 'report.md'}\nsummary: {SUMMARY}")
        return EXIT_OK

    if not (args.model_a and args.model_b):
        raise SetupError("9a has no default: pass --model-a and --model-b")
    catalog = load_catalog(Path(args.catalog) if args.catalog else Path(args.harness) / "app" / "catalog" / "models.json")
    slots = make_slots(args, catalog)
    if args.only:
        slots = [s for s in slots if s.name == args.only]
    cmd = rlmh_command(args.rlmh)
    info = rlmh_info(cmd)
    for s in slots:
        s.expected = expected_format(s, info["config"].get("call_format"))
    check_ignored()
    task_list = build_tasks()
    folder = task_folder(args.date)

    rehearse = args.mode == "rehearse"
    out = RESULTS / (f"{args.date}-rehearsal" if rehearse else args.date)
    if args.mode == "go":
        # a crashed attempt leaves events.jsonl and no runs.jsonl; appending to it would mix two attempts
        taken = [s.dir(out) for s in slots if (s.dir(out) / "runs.jsonl").exists()
                 or ((s.dir(out) / "events.jsonl").is_file() and (s.dir(out) / "events.jsonl").stat().st_size > 0)]
        if taken:
            raise SetupError(f"output already exists: {', '.join(map(str, taken))}; pick another --date")
        bad = check_task_files(task_list, folder)
        if bad:
            raise SetupError(f"the task files in {folder} differ from the rebuilt tasks ({', '.join(bad)}): run "
                             "--estimate again and get a new go")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        write_task_files(task_list, folder)
    ref = load_tokens(args.tokens_from or [DEFAULT_TOKENS])
    est = estimate(slots, ref, len(task_list), args.margin, args.cost_cap)
    if args.mode != "rehearse":
        print_estimate(slots, est, ref, task_list, folder, info, cmd)
    if args.mode == "estimate":
        return EXIT_OK

    if rehearse:
        for s in slots:
            s.expected = expected_format(dataclasses.replace(s, ref=FAKE_REF, tool_call=True),
                                         info["config"].get("call_format"))
        if out.exists():
            shutil.rmtree(out)  # only rehearsals write here
    out.mkdir(parents=True, exist_ok=True)
    run = {"date": args.date, "mode": args.mode, "rlmh_version": info["version"], "config": info["config"],
           "rlm_commit": rlm_commit(), "generator": tasks.generator(FAMILY).GENERATOR_VERSION,
           "platform": f"{sys.platform}, Python {sys.version.split()[0]}",
           "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "tasks": [{"task_id": t.task_id, "level": t.level, "chars": len(t.context)} for t in task_list],
           "estimate": None if rehearse else est, "reference_notes": ref.notes, "stopped": None}
    write_json(out / "run.json", run)

    env_for, ref_for = (lambda slot: None), (lambda slot: None)
    if rehearse:
        scripts = {}
        for s in slots:
            scripts[s.name] = out / f"fake-{s.name}.json"
            write_json(scripts[s.name], fake_script(s.expected))
        env_for = lambda slot: {**os.environ, "RLMH_FAKE_SCRIPT": str(scripts[slot.name])}  # noqa: E731
        ref_for = lambda slot: FAKE_REF  # noqa: E731
    started = time.monotonic()
    stop = run_batch(slots, task_list, folder, out, cmd, args.jobs, None if rehearse else est["cap"], env_for, ref_for)
    run["stopped"] = stop
    run["wall_s"] = round(time.monotonic() - started, 1)
    write_json(out / "run.json", run)
    rows, problems = write_report(out, task_list, None if rehearse else SUMMARY)

    print(f"\nreport: {out / 'report.md'}")
    for meta in sorted({r["slot"] for r in rows}):
        mine = [r for r in rows if r["slot"] == meta]
        graded = [r for r in mine if r["status"] == GRADED]
        print(f"slot {meta}: {sum(bool(r['exact']) for r in graded)}/{len(graded)} exact, "
              f"{sum(r['status'] == INFRA for r in mine)} infra-failed, "
              f"{sum(r['status'].startswith('not run') for r in mine)} not run")
    for prob in problems:
        print(f"check: {prob}")
    if stop:
        print(f"stopped: {stop}")
        return EXIT_STOPPED
    return EXIT_STOPPED if rehearse and problems else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
