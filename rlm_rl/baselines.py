"""Plan step 11's baselines: the base model on the selection set, with no training (research §8.2, spec §12.10).

Three arms, each an eval of checkpoint 0, logged like any other eval (evals.py) under results/<run>/transcripts/val/<arm>/:
- `rlm`: the base model as an RLM in environment v2, prompt v2, thinking off. It is the step-0 number production's stop
  rules compare against. Edit tasks run twice (`EDIT_SAMPLES`), which also measures, before production, the cost of an
  edit episode, the turn-cap hits and the base pass rate per tier (code-editing-plan.md §9-§11, §16).
- `direct`: the base model reading the context directly, in one call with thinking off (read tasks only; every
  selection-v2 context fits in the 262K-token window). Without it, no claim that the RLM setup helps holds.
- `rlm_medium` (decision 11b): as `rlm`, with root thinking at `medium`; sub-calls stay off. Production thinks at
  `medium` only if it beats `rlm` by about 12 points (plan step 11).
- `prod_v3`: the base model in production's set-up as Rishi chose it after the first three arms (2026-10-17): prompt v3
  (tool-call code runs, 30 edit turns) and root thinking at `medium`, on selection-v3. It is the clean step-0 number.

The first three arms ran prompt v2 on selection-v2 (run step11-v1); `prod_v3` runs on selection-v3 (run step11-v3).

Sub-calls go to the base model, as in training (decision 11a). Arms run in one process share one limiter and one pacer
(client.py), so the caps and the sub-call spacing hold across them.

Spend has a hard ceiling over every arm and every run of the command. An episode starts only if what is spent, plus the
reservations of the episodes in flight, plus its own reservation, stays under the ceiling. A reservation is the arm's
measured mean cost for its kind of task (read or edit) times RESERVE_MARGIN, or a cautious default until one has
finished. spend.json records the clients' spend every few seconds, so an interrupted run's spend still counts.

Rerunning resumes: finished episodes are skipped and infra episodes run again (evals.py). Infra episodes are never
scored as failures; every attempt stays in the log, so the write-up can count episodes lost to rate limiting.
"""

from __future__ import annotations

import ast
import asyncio
import collections
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import bridge, client as rclient, reward
from .bridge import InfraError
from .client import ModelClient, Usage
from .env import Episode, Task, Turn, run_episode
from .evals import episode_key, summarize
from .sandbox import Sandbox
from .transcripts import TranscriptLog, episode_record, transcript_path

EVAL_SET = "selection-v2"  # the first three arms; prod_v3 runs on selection-v3 (scripts/baselines.py --eval-set)
STEP = 0  # every arm evaluates checkpoint 0, the base model
EDIT_SAMPLES = {"rlm": 2, "rlm_medium": 1}  # plan step 11: the rlm arm runs the 16 edit tasks twice

# How many episodes run at once. Read episodes barely touch the data drive; edit episodes start a container and run
# real test suites on it, and past about 8-10 at once the drive timed out (edit-build-results.md §5a) *(mine)*.
MAX_READ_AT_ONCE = 32
MAX_EDIT_AT_ONCE = 8
MAX_DIRECT_AT_ONCE = 6  # ~130K-token prompts; the limiter's cap on the base model bounds them anyway

# A reservation is an arm's measured mean cost per episode for its kind, times this margin; before an arm has a
# measured episode of a kind, the default below *(mine: step 10 measured ~$0.02 a read episode at level 1;
# code-editing-plan.md §10 guessed ~$0.06-0.15 an edit episode; thinking at medium ~1.6x, step 2 §6)*.
RESERVE_MARGIN = 1.5
DEFAULT_RESERVE_USD = {("rlm", "read"): 0.08, ("rlm", "edit"): 0.40, ("direct", "read"): 0.08,
                       ("rlm_medium", "read"): 0.15, ("rlm_medium", "edit"): 0.60,
                       ("prod_v3", "read"): 0.10, ("prod_v3", "edit"): 0.20}  # from rlm_medium's measured costs
# What the projection before a run assumes for an episode not yet measured: the plan's estimates.
DEFAULT_PROJECT_USD = {("rlm", "read"): 0.04, ("rlm", "edit"): 0.11, ("direct", "read"): 0.03,
                       ("rlm_medium", "read"): 0.065, ("rlm_medium", "edit"): 0.18,
                       ("prod_v3", "read"): 0.04, ("prod_v3", "edit"): 0.06}  # step 11: medium $0.036 / $0.042

SPEND_EVERY_S = 5.0
TOOL_CALL_MARK = "<tool_call>"  # Qwen3.8's native tool-call tag, which the REPL can't run (step 11 results)


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str  # "rlm" or "direct"
    root_thinking: str | None = None  # client.ROOT_THINKING_EFFORTS, or None for off
    edit_samples: int = 1  # samples per edit task; read tasks run once
    read_only: bool = False
    prompt_version: str = "v2"  # step 11's first arms ran v2; bridge.PROMPT_VERSION is v3 since


ARMS = {
    "rlm": Arm("rlm", "rlm", None, EDIT_SAMPLES["rlm"]),
    "direct": Arm("direct", "direct", None, 0, read_only=True),
    "rlm_medium": Arm("rlm_medium", "rlm", "medium", EDIT_SAMPLES["rlm_medium"]),
    "prod_v3": Arm("prod_v3", "rlm", "medium", 1, prompt_version="v3"),
}


def kind_of(task: Task) -> str:
    return "edit" if bridge.is_edit(task.family) else "read"


# ----------------------------------------------------------------------------- direct reading

DIRECT_VERSION = "direct-v1"
DIRECT_MAX_TOKENS = 6144  # room to work through a question in writing before the answer *(mine)*
DIRECT_SYSTEM = ("You answer questions about long texts. The whole text is in the user's message, between <context> "
                 "and </context>. Read it carefully and answer the question.")
DIRECT_TEMPLATE = """You will be asked a question about the text below.

Question: {question}

<context>
{context}
</context>

Question (repeated): {question}

Work through it in writing if that helps. Then end your reply with exactly one line of this form:
FINAL ANSWER: <your answer>
Write a number as digits only, and a list as its items separated by commas."""
DIRECT_PROMPT_HASH = hashlib.sha256(json.dumps({"system": DIRECT_SYSTEM, "user": DIRECT_TEMPLATE},
                                               sort_keys=True).encode()).hexdigest()
FINAL_MARK = "final answer:"


def direct_messages(question: str, context: str) -> list[dict]:
    return [{"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": DIRECT_TEMPLATE.format(question=question, context=context)}]


def logged_direct_messages(question: str, context: str) -> list[dict]:
    """The messages as logged: the context replaced by its size and hash, as every transcript identifies it (H3)."""
    digest = hashlib.sha256(context.encode("utf-8", "replace")).hexdigest()[:16]
    return direct_messages(question, f"[the context: {len(context):,} characters, sha256 {digest}...]")


def parse_final(text: str | None) -> tuple[bool, Any]:
    """(answered, answer) from a direct-reading reply: the value after the last `FINAL ANSWER:`. A JSON or Python
    literal of an allowed shape (bridge.valid_answer) is taken as is; anything else stays a string, which the scorer
    parses like any submitted string (a number in it, a comma-separated list, an entity)."""
    text = text or ""
    at = text.lower().rfind(FINAL_MARK)
    if at < 0:
        return False, None
    value = text[at + len(FINAL_MARK):].strip().splitlines()[0].strip() if text[at + len(FINAL_MARK):].strip() else ""
    value = value.strip("*_` ").strip()
    if not value:
        return False, None
    for parse in (json.loads, ast.literal_eval):
        try:
            parsed = parse(value)
        except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
            continue
        if bridge.valid_answer(parsed):
            return True, parsed
    return True, value


async def run_direct(task: Task, *, client: ModelClient, model: str, scorer=reward.score) -> Episode:
    """One direct-reading episode: the whole context in the prompt, one answer. Never raises, except on cancellation.
    A request that fails after retries is an infra error, as in an RLM episode."""
    ep = Episode(task.task_id, task.family, task.level, task.question, task.answer, model, "none",
                 context_chars=len(task.context), context_lines=len(task.context.splitlines()),
                 context_sha256=hashlib.sha256(task.context.encode("utf-8", "replace")).hexdigest(),
                 prompt_version=DIRECT_VERSION, limits={"mode": "direct", "max_turns": 1,
                                                        "root_max_tokens": DIRECT_MAX_TOKENS})
    ep.messages = logged_direct_messages(task.question, task.context)
    started = time.monotonic()
    try:
        choice = await client.answer(direct_messages(task.question, task.context), model=model, usage=ep.usage,
                                     max_tokens=DIRECT_MAX_TOKENS)
        reply = choice.message.content or ""
        ep.messages.append({"role": "assistant", "content": reply})
        ep.turns.append(Turn(reply, choice.finish_reason, ep.usage.root_completion_tokens, None, None, None,
                             round(time.monotonic() - started, 3), 0.0))
        ep.submitted, ep.answer = parse_final(reply)
        ep.answer_valid = ep.submitted
    except InfraError as exc:
        ep.infra_kind, ep.infra_error = getattr(exc, "kind", "request"), str(exc)
    except Exception as exc:  # a bug on the host: counted, never scored
        import traceback

        ep.infra_kind, ep.infra_error = "host_bug", f"host error: {exc!r}"
        ep.host_traceback = traceback.format_exc()
    ep.wall_s = round(time.monotonic() - started, 3)
    if ep.infra_error is None:
        try:
            score = scorer(task, ep)
            ep.reward, ep.correct, ep.reward_components = float(score.reward), bool(score.correct), dict(score.components)
        except Exception as exc:
            ep.infra_kind, ep.infra_error = "host_bug", f"scorer error: {exc!r}"
    return ep


# ----------------------------------------------------------------------------- spend


class Budget:
    """The hard ceiling over every arm (module docstring). Single event loop, so no lock: nothing awaits between the
    check and the reservation.

    An episode that doesn't fit while others are in flight waits for them (`acquire`): their reservations are
    released as they finish, and the estimates behind new reservations improve as costs are measured. It is refused
    only when nothing is in flight and it still doesn't fit, so a tight ceiling never drops episodes that would fit
    once the cautious defaults give way to measured costs (step 11: 34 episodes were refused at the start of the
    medium-thinking arm before this)."""

    def __init__(self, ceiling: float, spent_before: float, clients: list[ModelClient]):
        self.ceiling = ceiling
        self.spent_before = spent_before
        self.clients = clients
        self.reserved = 0.0
        self.refused = 0
        self._released: asyncio.Condition | None = None

    def spent(self) -> float:
        return self.spent_before + sum(c.spent_usd() for c in self.clients)

    def fits(self, amount: float) -> bool:
        return self.spent() + self.reserved + amount <= self.ceiling

    def reserve(self, amount: float) -> bool:
        if not self.fits(amount):
            self.refused += 1
            return False
        self.reserved += amount
        return True

    async def acquire(self, amount: Callable[[], float]) -> float | None:
        """Reserve `amount()` (recomputed after every wait), waiting while other episodes hold reservations. Returns
        the amount reserved, or None if it can't fit even with nothing in flight."""
        if self._released is None:
            self._released = asyncio.Condition()
        async with self._released:
            while True:
                want = amount()
                if self.fits(want):
                    self.reserved += want
                    return want
                if self.reserved <= 1e-12:
                    self.refused += 1
                    return None
                await self._released.wait()

    def release(self, amount: float) -> None:
        self.reserved = max(0.0, self.reserved - amount)
        if self._released is not None:
            async def notify():
                async with self._released:
                    self._released.notify_all()
            asyncio.get_running_loop().create_task(notify())


def read_spend(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "spend.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"total_usd": 0.0, "by_arm": {}, "sessions": []}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------------- the plan of episodes


@dataclass
class Job:
    arm: Arm
    task: Task
    sample: int

    @property
    def key(self) -> str:
        return episode_key(self.arm.name, STEP, self.task.task_id, self.sample)

    @property
    def kind(self) -> str:
        return kind_of(self.task)


# Edit languages in the order their episodes start: the slowest first, so the long pole starts early
# (edit-build-results.md §5a: Go's test runs take ~15-30 minutes under load, JavaScript's ~4-8, Python's ~2-5).
LANGUAGE_ORDER = {"go": 0, "javascript": 1, "typescript": 1, "rust": 0, "java": 0, "python": 2}


def plan_jobs(tasks: list[Task], arm: Arm) -> list[Job]:
    jobs = []
    for task in tasks:
        if bridge.is_edit(task.family):
            jobs += [Job(arm, task, s) for s in range(arm.edit_samples)]
        else:
            jobs.append(Job(arm, task, 0))

    def order(job: Job):
        lang = str((job.task.edit or {}).get("language") or "").lower()
        return (0 if job.kind == "edit" else 1, LANGUAGE_ORDER.get(lang, 1), job.sample, job.task.task_id)

    return sorted(jobs, key=order)


def arm_log(run_dir: Path, arm: str) -> TranscriptLog:
    return TranscriptLog(transcript_path(run_dir, "val", STEP, arm))


def measured_mean(records: list[dict], kind: str) -> float | None:
    costs = [r["usage"]["est_cost_usd"] for r in records if not r.get("infra_error")
             and ("edit" if bridge.is_edit(r["family"]) else "read") == kind]
    return statistics.fmean(costs) if costs else None


def project(run_dir: Path, tasks: list[Task], arms: list[Arm]) -> dict:
    """What the arms still have to run and what it should cost: the arm's measured mean per kind where there is one,
    else the plan's estimate. Infra episodes count as still to run."""
    out = {"arms": {}, "todo_usd": 0.0}
    for arm in arms:
        records = arm_log(run_dir, arm.name).records()
        done = {r["key"] for r in records if not r.get("infra_error")}
        jobs = [j for j in plan_jobs(tasks, arm) if j.key not in done and not (arm.read_only and j.kind == "edit")]
        per = {}
        for kind in ("read", "edit"):
            mean = measured_mean(records, kind)
            per[kind] = {"todo": sum(j.kind == kind for j in jobs),
                         "per_episode_usd": mean if mean is not None else DEFAULT_PROJECT_USD.get((arm.name, kind), 0.0),
                         "measured": mean is not None}
        todo = sum(v["todo"] * v["per_episode_usd"] for v in per.values())
        out["arms"][arm.name] = {"kinds": per, "todo_usd": round(todo, 4), "done": len(done)}
        out["todo_usd"] += todo
    out["todo_usd"] = round(out["todo_usd"], 4)
    return out


# ----------------------------------------------------------------------------- running


class BaselineRun:
    """Runs (or resumes) some arms of one baseline run in one process.

        run = BaselineRun("step11-v1", results_dir, openai_client=..., base_model=..., ceiling=12.0)
        await run.run(tasks, ["rlm", "direct"])
    """

    def __init__(self, run: str, results: Path, *, openai_client: Any, base_model: str, ceiling: float,
                 sandbox_factory: Callable[..., Any] = Sandbox, sandbox_kwargs: dict | None = None,
                 pacer: Any = None, retry: rclient.RetryPolicy | None = None, log: Callable[[str], None] = print,
                 max_read: int = MAX_READ_AT_ONCE, max_edit: int = MAX_EDIT_AT_ONCE,
                 max_direct: int = MAX_DIRECT_AT_ONCE, eval_set: str = EVAL_SET):
        self.name = run
        self.eval_set = eval_set
        self.run_dir = Path(results) / run
        self.base_model = base_model
        self.ceiling = ceiling
        self.sandbox_factory = sandbox_factory
        self.sandbox_kwargs = sandbox_kwargs or {}
        self.say = log
        self.limiter = rclient.Limiter(model_caps={base_model: rclient.MAX_SUBCALLS_IN_FLIGHT})
        self.pacer = pacer if pacer is not None else rclient.Pacer({base_model: rclient.SUBCALL_START_GAP_S})
        retry = retry or rclient.RetryPolicy()
        self.clients = {
            name: ModelClient(openai_client, subcall_model=base_model, limiter=self.limiter, pacer=self.pacer,
                              retry=retry, root_thinking=arm.root_thinking)
            for name, arm in ARMS.items()}
        self.gates = {"read": asyncio.Semaphore(max_read), "edit": asyncio.Semaphore(max_edit),
                      "direct": asyncio.Semaphore(max_direct)}
        ledger = read_spend(self.run_dir)  # earlier sessions of this run
        self._before_by_arm = {k: float(v) for k, v in (ledger.get("by_arm") or {}).items()}
        self._before_sessions = list(ledger.get("sessions") or [])
        # spend outside this script's clients, added by hand and counted against the ceiling: probes, and an upper
        # bound for failed tries W&B may have billed ({"what": usd}); kept across sessions
        self._adjustments = dict(ledger.get("adjustments") or {})
        self._adjustments_note = ledger.get("adjustments_note")
        self.budget = Budget(ceiling, float(ledger.get("total_usd") or 0.0), list(self.clients.values()))
        self.session = {"started_at": time.time(), "arms": [], "skipped_budget": 0, "finished": 0, "infra": 0}
        self.progress: dict[str, dict] = {}

    # ------------------------------------------------------------------ spend ledger

    def _save_spend(self) -> None:
        names = set(self._before_by_arm) | {n for n, c in self.clients.items() if c.spent_usd()}
        by_arm = {n: round(self._before_by_arm.get(n, 0.0) + self.clients[n].spent_usd(), 6) for n in sorted(names)}
        record = {**self.session, "spent_usd": round(sum(c.spent_usd() for c in self.clients.values()), 6),
                  "updated_at": time.time()}
        write_json(self.run_dir / "spend.json", {
            "total_usd": round(self.budget.spent(), 6), "ceiling_usd": self.ceiling, "by_arm": by_arm,
            "adjustments": self._adjustments, "adjustments_note": self._adjustments_note,
            "sessions": self._before_sessions + [record], "updated_at": time.time()})

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(SPEND_EVERY_S)
            self._save_spend()
            self._save_progress()

    # ------------------------------------------------------------------ episodes

    def _reservation(self, arm: Arm, kind: str, records: list[dict]) -> float:
        mean = measured_mean(records, kind)
        if mean is None:
            return DEFAULT_RESERVE_USD.get((arm.name, kind), 0.5)
        return max(mean * RESERVE_MARGIN, 0.01)

    async def _one(self, job: Job, log: TranscriptLog, finished: list[dict]) -> None:
        arm, kind = job.arm, job.kind
        gate = self.gates["direct" if arm.mode == "direct" else kind]
        prog = self.progress[arm.name]
        async with gate:
            prog["waiting_budget"] = prog.get("waiting_budget", 0) + 1
            try:
                amount = await self.budget.acquire(lambda: self._reservation(arm, kind, finished))
            finally:
                prog["waiting_budget"] -= 1
            if amount is None:
                prog["skipped_budget"] += 1
                self.session["skipped_budget"] += 1
                return
            prog["running"] += 1
            try:
                client = self.clients[arm.name]
                if arm.mode == "direct":
                    ep = await run_direct(job.task, client=client, model=self.base_model)
                else:
                    ep = await run_episode(job.task, client=client, root_model=self.base_model,
                                           run_id=f"{self.name}-{arm.name}", sandbox_factory=self.sandbox_factory,
                                           prompt_version=arm.prompt_version, **self.sandbox_kwargs)
            finally:
                prog["running"] -= 1
                self.budget.release(amount)
        record = episode_record(ep, split="val", run=self.name, step=STEP, key=job.key, eval_name=arm.name,
                                sample=job.sample)
        record["arm"] = arm.name
        record["root_thinking"] = arm.root_thinking
        if arm.mode == "direct":
            record["prompt_hash"] = DIRECT_PROMPT_HASH
        log.append(record)
        self.session["finished"] += 1
        if record.get("infra_error"):
            prog["infra"] += 1
            self.session["infra"] += 1
        else:
            finished.append(record)
            prog["done"] += 1
        self.say(f"[{arm.name}] {prog['done']}/{prog['planned']} {job.task.family} L{job.task.level}"
                 f"{' s' + str(job.sample) if job.sample else ''}: "
                 + ("infra " + str(record.get("infra_kind")) if record.get("infra_error")
                    else f"{'correct' if record['correct'] else 'wrong'}, {len(record['turns'])} turns,"
                         f" ${record['usage']['est_cost_usd']:.3f}, {record['wall_s']:.0f} s")
                 + f" | spent ${self.budget.spent():.2f} of ${self.ceiling:g}")

    def _save_progress(self) -> None:
        write_json(self.run_dir / "progress.json", {
            "run": self.name, "updated_at": time.time(), "arms": self.progress, "spent_usd": round(self.budget.spent(), 6),
            "ceiling_usd": self.ceiling, "reserved_usd": round(self.budget.reserved, 6),
            "requests": {name: {k: v for k, v in c.stats.as_dict().items() if k != "events"}
                         for name, c in self.clients.items() if name in self.progress},
            "limiter_peaks": self.limiter.peaks})

    async def run(self, tasks: list[Task], arm_names: list[str], *, retry_infra: int = 1) -> dict:
        """Run (or resume) the arms together. Infra episodes get `retry_infra` more passes at the end."""
        arms = [ARMS[name] for name in arm_names]
        self.session["arms"] = arm_names
        self.run_dir.mkdir(parents=True, exist_ok=True)
        heartbeat = asyncio.create_task(self._beat())
        started = time.time()
        try:
            for attempt in range(1 + retry_infra):
                batches = []
                for arm in arms:
                    log = arm_log(self.run_dir, arm.name)
                    records = log.records()
                    finished = [r for r in records if not r.get("infra_error")]
                    done = {r["key"] for r in finished}
                    planned = [j for j in plan_jobs(tasks, arm) if not (arm.read_only and j.kind == "edit")]
                    jobs = [j for j in planned if j.key not in done]
                    prog = self.progress.setdefault(arm.name, {"planned": len(planned), "done": 0, "running": 0,
                                                               "infra": 0, "skipped_budget": 0, "started_at": started})
                    prog["done"] = len(done & {j.key for j in planned})
                    batches += [(job, log, finished) for job in jobs]
                if not batches:
                    break
                if attempt:
                    self.say(f"retrying {len(batches)} infra episode(s), pass {attempt + 1}")
                # arms interleaved so each makes progress; within an arm, edit episodes (the long pole) start first
                await asyncio.gather(*(self._one(*b) for b in _interleave(batches)))
                if self.session["skipped_budget"]:
                    break  # the ceiling is reached: no retry pass would start either
            return self.summary(tasks, arm_names)
        finally:
            heartbeat.cancel()
            self._save_progress()
            self._save_spend()

    def summary(self, tasks: list[Task], arm_names: list[str] | None = None) -> dict:
        return write_summary(self.run_dir, tasks, arm_names, ceiling=self.ceiling, eval_set=self.eval_set)


def _interleave(batches: list[tuple]) -> list[tuple]:
    by_arm: dict[str, list] = collections.defaultdict(list)
    for b in batches:
        by_arm[b[0].arm.name].append(b)
    out, queues = [], list(by_arm.values())
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


# ----------------------------------------------------------------------------- the summary


def _q(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(math.ceil(q * len(values))) - 1)]


def _rate(xs: list[bool]) -> dict:
    n = len(xs)
    p = sum(xs) / n if n else None
    return {"n": n, "rate": p, "se": math.sqrt(p * (1 - p) / n) if n else None}


def edit_tier(family: str) -> str:
    return "edit_real" if family.startswith("edit_real") else "edit_synth" if family.startswith("edit_synth") else family


def arm_summary(records: list[dict], all_attempts: list[dict]) -> dict:
    """evals.summarize plus what step 11's gates read: per kind costs and times, the edit tiers, the turn cap."""
    out = summarize(records)
    clean = [r for r in records if not r.get("infra_error")]
    kinds = {}
    for kind in ("read", "edit"):
        rs = [r for r in clean if ("edit" if bridge.is_edit(r["family"]) else "read") == kind]
        costs = [r["usage"]["est_cost_usd"] for r in rs]
        walls = [r["wall_s"] for r in rs]
        kinds[kind] = {
            "n": len(rs), "accuracy": _rate([bool(r["correct"]) for r in rs]),
            "cost_mean_usd": statistics.fmean(costs) if costs else None, "cost_total_usd": sum(costs),
            "cost_p90_usd": _q(costs, 0.9), "cost_max_usd": max(costs) if costs else None,
            "wall_median_s": statistics.median(walls) if walls else None, "wall_p90_s": _q(walls, 0.9),
            "wall_max_s": max(walls) if walls else None,
            "turns_mean": statistics.fmean([len(r["turns"]) for r in rs]) if rs else None,
            "subcalls_mean": statistics.fmean([r["subcalls_used"] for r in rs]) if rs else None,
            "root_prompt_tokens_mean": statistics.fmean([r["usage"]["root_prompt_tokens"] for r in rs]) if rs else None,
            "root_completion_tokens_mean": statistics.fmean([r["usage"]["root_completion_tokens"] for r in rs]) if rs else None,
            "reasoning_tokens_mean": statistics.fmean([r["usage"].get("root_reasoning_tokens", 0) for r in rs]) if rs else None,
            "truncated_turns_mean": statistics.fmean([r["metrics"]["truncated_turns"] for r in rs]) if rs else None,
        }
    out["kinds"] = kinds
    # Step 11 finding: with prompt v2 the base model often answers a turn in its native tool-call syntax instead of a
    # ```python block, which the REPL can't run (the turn is a "no code block" error)
    tool = {}
    for kind in ("read", "edit"):
        rs = [r for r in clean if ("edit" if bridge.is_edit(r["family"]) else "read") == kind]
        turns = [t for r in rs for t in r["turns"]]
        hit = [r for r in rs if any(TOOL_CALL_MARK in (t.get("reply") or "") for t in r["turns"])]
        miss = [r for r in rs if r not in hit]
        tool[kind] = {"episodes": len(rs), "episodes_with": len(hit),
                      "turns": len(turns), "turns_with": sum(TOOL_CALL_MARK in (t.get("reply") or "") for t in turns),
                      "first_turn": sum(TOOL_CALL_MARK in (r["turns"][0].get("reply") or "") for r in rs if r["turns"]),
                      "accuracy_with": _rate([bool(r["correct"]) for r in hit]),
                      "accuracy_without": _rate([bool(r["correct"]) for r in miss])}
    out["tool_call"] = tool
    edit = [r for r in clean if bridge.is_edit(r["family"])]
    if edit:
        tiers = collections.defaultdict(list)
        for r in edit:
            tiers[edit_tier(r["family"])].append(r)
            tiers[f"{edit_tier(r['family'])}/L{r['level']}"].append(r)
            if r.get("language"):
                tiers[f"{edit_tier(r['family'])}/{str(r['language']).lower()}"].append(r)
        cap_hits = [r for r in edit if len(r["turns"]) >= (r.get("limits") or {}).get("max_turns", 20)]
        out["edit"] = {
            "tiers": {k: {**_rate([bool(r["correct"]) for r in v]),
                          "cost_mean_usd": statistics.fmean([r["usage"]["est_cost_usd"] for r in v]),
                          "wall_median_s": statistics.median([r["wall_s"] for r in v]),
                          "turns_mean": statistics.fmean([len(r["turns"]) for r in v])}
                      for k, v in sorted(tiers.items())},
            "turn_cap_hits": len(cap_hits), "turn_cap_hits_unresolved": sum(not r["correct"] for r in cap_hits),
            "turn_cap_rate": len(cap_hits) / len(edit), "turn_cap_unresolved_rate": sum(not r["correct"] for r in cap_hits) / len(edit),
            "submitted": statistics.fmean([float(bool(r["submitted"])) for r in edit]),
            "empty_diff": statistics.fmean([float(bool((r.get("reward_components") or {}).get("empty_diff"))) for r in edit]),
            "f2p_pass_mean": _mean_known([(r.get("reward_components") or {}).get("f2p_pass") for r in edit]),
            "tested_before_submit": _mean_known([(r.get("edit_stats") or {}).get("tested_before_submit") for r in edit]),
            "grader_s_median": _median_known([(r.get("edit_stats") or {}).get("grader_s") for r in edit]),
        }
    infra_attempts = [r for r in all_attempts if r.get("infra_error")]
    out["attempts"] = {"total": len(all_attempts), "infra": len(infra_attempts),
                       "infra_kinds": dict(collections.Counter(r.get("infra_kind") for r in infra_attempts)),
                       "infra_cost_usd": round(sum(r["usage"]["est_cost_usd"] for r in infra_attempts), 4)}
    starts = [r["started_at"] for r in all_attempts]
    ends = [r["started_at"] + r["wall_s"] for r in all_attempts]
    out["span_s"] = (max(ends) - min(starts)) if starts else None
    return out


def _mean_known(values) -> float | None:
    known = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.fmean(known) if known else None


def _median_known(values) -> float | None:
    known = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.median(known) if known else None


def paired(records_a: list[dict], records_b: list[dict]) -> dict:
    """Arm b against arm a on the tasks both finished, per task (an arm's samples of a task averaged): the mean
    difference in accuracy, its standard error over tasks, and the tasks only one got right."""
    def by_task(records):
        out = collections.defaultdict(list)
        for r in records:
            if not r.get("infra_error"):
                out[r["task_id"]].append(float(bool(r["correct"])))
        return {k: statistics.fmean(v) for k, v in out.items()}

    a, b = by_task(records_a), by_task(records_b)
    common = sorted(set(a) & set(b))
    diffs = [b[t] - a[t] for t in common]
    n = len(diffs)
    mean = statistics.fmean(diffs) if diffs else None
    se = statistics.stdev(diffs) / math.sqrt(n) if n > 1 else None
    return {"tasks": n, "a": statistics.fmean([a[t] for t in common]) if common else None,
            "b": statistics.fmean([b[t] for t in common]) if common else None, "diff": mean, "se": se,
            "b_only": sum(1 for d in diffs if d > 0), "a_only": sum(1 for d in diffs if d < 0)}


def write_summary(run_dir: Path, tasks: list[Task], arm_names: list[str] | None = None, *,
                  ceiling: float | None = None, eval_set: str = EVAL_SET) -> dict:
    """results/<run>/summary.json: every arm that has a log (or the arms named), and the comparisons between them."""
    arm_names = [n for n in ARMS if arm_log(run_dir, n).path.exists() or n in (arm_names or ())]
    planned = {name: plan_jobs(tasks, ARMS[name]) for name in arm_names}
    arms, recs = {}, {}
    for name in arm_names:
        log = arm_log(run_dir, name)
        keys = {j.key for j in planned[name] if not (ARMS[name].read_only and j.kind == "edit")}
        records = [r for r in log.records() if r["key"] in keys]
        attempts = [r for r in log.all_records() if r["key"] in keys]
        recs[name] = records
        arms[name] = {"planned": len(keys), "logged": len(records), **arm_summary(records, attempts)} if records else \
            {"planned": len(keys), "logged": 0}
    comparisons = {}
    if recs.get("rlm") and recs.get("direct"):
        read = [r for r in recs["rlm"] if not bridge.is_edit(r["family"])]
        comparisons["rlm_vs_direct_read"] = paired(recs["direct"], read)
    if recs.get("rlm") and recs.get("rlm_medium"):
        comparisons["medium_vs_rlm"] = paired(recs["rlm"], recs["rlm_medium"])
        for kind in ("read", "edit"):
            pick = (lambda r: bridge.is_edit(r["family"])) if kind == "edit" else (lambda r: not bridge.is_edit(r["family"]))
            comparisons[f"medium_vs_rlm_{kind}"] = paired([r for r in recs["rlm"] if pick(r)],
                                                          [r for r in recs["rlm_medium"] if pick(r)])
    spend = read_spend(run_dir)
    summary = {"run": run_dir.name, "eval_set": eval_set, "built_at": time.time(), "arms": arms,
               "comparisons": comparisons, "spend": {"total_usd": spend.get("total_usd"), "by_arm": spend.get("by_arm"),
                                                     "ceiling_usd": ceiling or spend.get("ceiling_usd")}}
    write_json(run_dir / "summary.json", summary)
    return summary
