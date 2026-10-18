"""The resumable GRPO training loop (plan step 6, spec §5): steps, evals, stop rules, checkpoints and resume.

One step n:
1. Draw 8 tasks at the controller's levels and run 4 rollouts each from checkpoint n-1.
   Every episode is logged to results/<run>/transcripts/train/step<n>.jsonl.
2. Filter (spec §5.4): infra episodes out, then small and all-equal groups. Update the
   difficulty controller. Re-score 4 rollouts under the base model for KL (monitor.py).
3. Commit the step's inference spend, then train on what's left, giving checkpoint n. Fewer
   than 2 trainable groups skips the train call, and checkpoint n is then checkpoint n-1.
4. Log the step's metrics (spec §9) to results/<run>.json and W&B.

Between steps the loop runs any eval that is due at the latest checkpoint (the selection set and
the capability check every `eval_every` steps, the final eval at `max_steps`), writes the Claude
audit's sample at every 10th checkpoint, saves a snapshot of every kept checkpoint, prunes the
rest, and checks, in this order:
- the infra pause (spec §5.4 step 4) and the collapse check behind the LR rule (spec §5.2)
- the five stop rules (stoprules.py)
- `max_steps`, the operator's stop switch, and whether the next step fits under the ceiling

Files, under results/:
- <run>.json: the run's state and every step's metrics, rewritten atomically. The dashboard reads it.
- <run>/config.json: the pinned values (spec §8), rewritten at each kept checkpoint.
- <run>/checkpoints/step<NNNN>.json: what resuming from that checkpoint needs.
- <run>/transcripts/, <run>/capability/, <run>/audit/: episode logs, capability replies and audit samples.
- <run>/STOP: the stop switch. The run finishes the step it's on, and that step's eval if one is
  due, then stops. `scripts/train.py stop <run>` writes it.
- <run>/pending/: the batch sent for training, until its checkpoint is recorded.
- <run>/abandoned/: logs of steps a resume from an earlier checkpoint left behind.

Checkpoints. Logical checkpoint n is the weights after step n, held by a model on the backend at
some backend step; each branch records its model. Checkpoint 0 is the base model. Every 5th
checkpoint is kept, plus every evaluated one, the best and the latest; the rest are deleted from
the backend (spec §8).

Resume. `resume()` picks up from the latest checkpoint. With `from_checkpoint`, it continues
from an earlier kept checkpoint instead, on a new branch: a new model on the backend that starts
from that checkpoint's LoRA (backend.branch_from), so nothing is deleted. The later steps' logs move
to abandoned/, and the controller and learning rate come back from that checkpoint's snapshot. The
new branch's optimizer starts fresh. Spend never rolls back. The automatic LR fallback (spec §5.2
step 3) uses the same mechanism. The step 10 tests (results §8) showed why it works this way: W&B
trains from the newest step it has saved for a model, so a fork of the artifact alone, a deletion of
newer steps, or new weights under an old step all failed.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import importlib.util
import io
import json
import os
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from . import capability, config as rconfig, monitor, stoprules, tasks as rtasks
from .bridge import InfraError
from .client import ModelClient, RetryPolicy
from .config import RunConfig
from .controller import DifficultyController
from .env import Episode, InfraPause, filter_step, run_groups, to_trajectory
from .evals import eval_set, run_eval
from .sandbox import Sandbox
from .transcripts import TranscriptLog, episode_record, transcript_path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
STATE_FORMAT = 1
HEARTBEAT_S = 15.0
# A failed train call (a dropped connection, a W&B hiccup) may still train on the server, so the loop waits up to
# TRAIN_SETTLE_S for a new checkpoint, polling every TRAIN_POLL_S, before sending the batch again, at most
# TRAIN_RETRIES times; only then does the failure stop the run (2026-10-18: one hiccup mustn't stop a 2-day run).
TRAIN_RETRIES = 2
TRAIN_SETTLE_S = 900.0
TRAIN_POLL_S = 30.0
AUDIT_EVERY = 10  # the Claude audit's checkpoints (decision 5a)
STOP_FILE = "STOP"
LIVE_GUARD_S = 120.0  # a "running" state written this recently may belong to a process that is still running it
CAPABILITY_ITEM_ESTIMATE_USD = 0.005  # one short chat call, before the first check has been measured


class Backend(Protocol):
    """What the loop needs from the trainer. backend.ArtBackend is W&B serverless; tests/fakes.FakeBackend is local."""

    async def open(self, name: str) -> None: ...
    def inference_name(self, name: str, step: int) -> str: ...
    def openai_client(self) -> Any: ...
    async def train(self, name: str, groups: list, **kwargs) -> tuple[int, dict]: ...
    async def latest_step(self, name: str) -> int: ...
    async def list_steps(self, name: str) -> list[int]: ...
    async def delete_steps(self, name: str, steps: list[int]) -> None: ...
    async def branch_from(self, name: str, source: str, step: int) -> None: ...
    async def log(self, name: str, groups: list | None, *, split: str, step: int, metrics: dict) -> None: ...
    def art_model(self, name: str) -> Any: ...


class ResumeBlocked(Exception):
    """Resuming can't go ahead safely without the operator's call; the message says what to do."""


def branch_name(run: str, branch: int) -> str:
    return run if branch == 0 else f"{run}-b{branch}"


def state_path(results: Path, run: str) -> Path:
    return Path(results) / f"{run}.json"


def request_stop(results: Path, run: str, reason: str = "") -> Path:
    """The stop switch: the run stops after the step it's on."""
    path = Path(results) / run / STOP_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reason or "stop requested", encoding="utf-8")
    return path


def load_state(results: Path, run: str) -> dict:
    return json.loads(state_path(results, run).read_text(encoding="utf-8"))


class TrainingRun:
    def __init__(self, run: str, *, backend: Backend, cfg: RunConfig | None = None, results: Path = RESULTS,
                 sandbox_factory: Callable[..., Any] = Sandbox, sandbox_kwargs: dict | None = None,
                 draw: Callable[..., list] = rtasks.draw, eval_sets: Callable[[str], tuple] = eval_set,
                 capability_items: list | None = None, prune: bool = True, stop_after: int | None = None,
                 rng: random.Random | None = None, retry: RetryPolicy | None = None, pacer: Any = None,
                 clear_containers: Callable[[str], dict] | None = None,
                 train_retry: tuple[int, float, float] = (TRAIN_RETRIES, TRAIN_SETTLE_S, TRAIN_POLL_S),
                 log: Callable[[str], None] = print):
        self.run = run
        self.cfg = cfg
        self.backend = backend
        self.results = Path(results)
        self.run_dir = self.results / run
        self.sandbox_factory = sandbox_factory
        self.sandbox_kwargs = sandbox_kwargs or {}
        self.draw = draw
        self.eval_sets = eval_sets
        self.capability_items = capability_items
        self.prune_enabled = prune
        self.stop_after = stop_after
        self.rng = rng or random.Random()
        self.retry = retry or RetryPolicy()
        self.pacer = pacer
        self.clear_containers = clear_containers  # sandbox.clear_for_run on a real run (scripts/train.py)
        self.train_retry = train_retry  # (retries, settle seconds, poll seconds) for a failed train call
        self.say = log
        self.state: dict = {}
        self.ctl: DifficultyController | None = None
        self.retirement = rtasks.Retirement()  # edit_real tasks that keep coming back all-0 or all-1 (spec §12.7)
        self.client: ModelClient | None = None
        self.pause: InfraPause | None = None
        self.stop_requested: str | None = None  # set by the operator (Ctrl-C) while running
        self._mark = 0.0  # client spend already committed to the state
        self._git: dict | None = None
        self._heartbeat: asyncio.Task | None = None

    # ------------------------------------------------------------------ entry points

    async def start(self) -> dict:
        if self.cfg is None:
            raise ValueError("start() needs a RunConfig")
        if state_path(self.results, self.run).exists():
            raise FileExistsError(f"{state_path(self.results, self.run)} exists; resume the run instead")
        if self.cfg.run != self.run:
            raise ValueError("the config is for another run")
        self.ctl = DifficultyController(rtasks.max_levels(self.cfg.mix), frozen=self.cfg.frozen_controller)
        self.retirement = rtasks.Retirement()
        self.state = self._new_state()
        self._clear_leftovers()
        self._load_capability()
        await self._open()
        self._notice("start", f"{self.cfg.kind} run started at learning rate {self.cfg.learning_rate:g}, "
                              f"ceilings {self._ceilings_text()}")
        self._save()
        return await self._loop()

    async def resume(self, *, from_checkpoint: str | int | None = None, lr: float | None = None,
                     retrain_pending: bool = False, discard_pending: bool = False,
                     ceilings: dict | None = None) -> dict:
        """Continue the run: from the latest checkpoint, or from `from_checkpoint` ("best" or a kept checkpoint)."""
        self.state = load_state(self.results, self.run)
        if self.state.get("format") != STATE_FORMAT:
            raise ValueError(f"unknown state format {self.state.get('format')}")
        age = time.time() - float(self.state.get("updated_at") or 0)
        if self.state.get("status") == "running" and age < LIVE_GUARD_S:
            raise ResumeBlocked(f"the run's state says it is running and was written {age:.0f} s ago, so another "
                                f"process may still be running it. Check with `train.py status {self.run}` and `ps`; "
                                f"if nothing is running it, resume again after {LIVE_GUARD_S:.0f} s")
        self.cfg = RunConfig.from_dict(self.state["config"])
        self.ctl = DifficultyController.from_state(self.state["controller"])
        self.retirement = rtasks.Retirement(self.state.get("retirement"))
        self._clear_leftovers()
        self._load_capability()
        await self._open()
        self._settle_interrupted()
        await self._settle_pending(retrain=retrain_pending, discard=discard_pending)
        if ceilings:
            self.cfg.ceilings.update(ceilings)
            self.state["config"]["ceilings"] = dict(self.cfg.ceilings)
            self.state["spend"]["ceilings"] = dict(self.cfg.ceilings)
            self._notice("ceiling", f"inference ceilings set by the operator: {self._ceilings_text()}")
        target = self._resolve_checkpoint(from_checkpoint)
        if target is not None and target != self._latest():
            await self._rollback(target, reason="resumed by the operator from an earlier checkpoint", new_lr=lr)
        elif lr is not None and lr != self.state["lr"]["current"]:
            self._notice("lr", f"learning rate set by the operator: {self.state['lr']['current']:g} -> {lr:g}")
            self.state["lr"]["current"] = lr
        # Resuming acknowledges whatever paused the run: audit findings so far, and the infra and hack windows.
        last = self._latest_step()
        self.state["acknowledged_audit_hacks"] = sorted(
            set(self.state["acknowledged_audit_hacks"]) | {h["key"] for h in self._audit_hacks(all_=True)})
        self.state["acknowledged_through"] = last
        self.state["rule5_since"] = last
        self._reset_pause_window()
        stop_file = self.run_dir / STOP_FILE
        if stop_file.exists():
            stop_file.unlink()
            self._notice("operator", "a leftover stop switch was cleared by the resume")
        previous = self.state["status"]
        self.state.update(status="running", stop=None)
        self._notice("resume", f"resumed from checkpoint {self._latest()} (was {previous}) on branch "
                               f"{self.state['branch']} at learning rate {self.state['lr']['current']:g}")
        self._save()
        return await self._loop()

    # ------------------------------------------------------------------ state

    def _new_state(self) -> dict:
        cfg, now = self.cfg, time.time()
        return {
            "format": STATE_FORMAT, "run": cfg.run, "kind": cfg.kind, "status": "running", "stop": None,
            "created_at": now, "updated_at": now, "config": cfg.as_dict(), "latest": 0,
            "branch": 0, "branches": [{"branch": 0, "model": branch_name(cfg.run, 0), "from": None, "at": now}],
            "checkpoints": {"0": {"branch": 0, "backend_step": 0}},
            "lr": {"current": cfg.learning_rate, "start": cfg.learning_rate, "floor": rconfig.LR_FLOOR,
                   "rule": cfg.lr_rule, "fallback_used": False, "fallback": None},
            "controller": self.ctl.state_dict(), "retirement": self.retirement.state_dict(),
            "steps": [], "evals": [], "best": None, "kept": [0], "pruned": [],
            "spend": {"train_usd": 0.0, "kl_usd": 0.0, "eval_usd": 0.0, "interrupted_usd": 0.0, "total_usd": 0.0,
                      "training_gpu_usd_est": 0.0, "ceilings": dict(cfg.ceilings)},
            "stop_rules": {}, "notices": [], "abandoned": [], "pending": None, "live": None, "audits": {},
            "acknowledged_audit_hacks": [], "acknowledged_through": 0, "rule5_since": 0, "collapse_since": 0,
            "collapse_noted": [], "template_fingerprint": None, "lr_decision": None,
        }

    def _save(self) -> None:
        self.state["updated_at"] = time.time()
        self.state["latest"] = self._latest()
        path = state_path(self.results, self.run)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str), encoding="utf-8")
        for attempt in range(5):  # a reader holding the file open on Windows can block the swap for a moment
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.2 * (attempt + 1))
        os.replace(tmp, path)

    def _notice(self, kind: str, text: str) -> None:
        """The operator log (plan step 9's Notices panel)."""
        self.state["notices"].append({"at": time.time(), "step": self._latest_step(), "kind": kind, "text": text})
        self.say(f"[{self.run}] {kind}: {text}")

    def _latest_step(self) -> int:
        steps = self.state.get("steps") or []
        return steps[-1]["step"] if steps else 0

    def _latest(self) -> int:
        """The latest checkpoint: the number of the last step done (0 before any)."""
        return self._latest_step()

    def _ckpt(self, c: int) -> dict:
        return self.state["checkpoints"][str(c)]

    def _model_for(self, c: int) -> str:
        ckpt = self._ckpt(c)
        return self.backend.inference_name(self._model_of(ckpt["branch"]), ckpt["backend_step"])

    def _model_of(self, branch: int) -> str:
        """The backend model a branch trains on, as recorded when the branch began."""
        entry = next((b for b in self.state["branches"] if b["branch"] == branch), None)
        return (entry or {}).get("model") or branch_name(self.run, branch)

    @property
    def _branch(self) -> str:
        return self._model_of(self.state["branch"])

    def _ceilings_text(self) -> str:
        return ", ".join(f"{k} ${v:g}" for k, v in self.cfg.ceilings.items()) or "none"

    def _clear_leftovers(self) -> None:
        """Remove containers a crashed or killed process left behind (plan step 12, before-start item 2): this run's own,
        whatever their state, and every sandbox or grader container that finished long ago. A failure is only noted:
        the episodes themselves will show whether Docker works."""
        if self.clear_containers is None:
            return
        try:
            removed = self.clear_containers(self.run)
        except Exception as exc:
            self._notice("containers", f"couldn't clear leftover containers: {exc!r}"[:300])
            return
        if any(removed.values()):
            self._notice("containers", f"removed {removed.get('run', 0)} leftover container(s) of this run and "
                                       f"{removed.get('finished', 0)} long-finished one(s) of earlier runs")

    def _load_capability(self) -> None:
        if self.capability_items is None and self.cfg.capability_set:
            self.capability_items = capability.load(capability.resolve(self.cfg.capability_set))

    async def _open(self) -> None:
        await self.backend.open(self._model_of(0))
        if self._branch != self._model_of(0):
            await self.backend.open(self._branch)
        self.base_model = self.backend.inference_name(self._model_of(0), 0)
        self.client = ModelClient(self.backend.openai_client(), subcall_model=self.base_model, retry=self.retry,
                                  pacer=self.pacer,  # None: the client's own, which spaces out sub-calls (client.py)
                                  root_thinking=self.cfg.root_thinking)  # production: medium (decision 11b)
        self._mark = self.client.spent_usd()
        self._reset_pause_window()

    def _reset_pause_window(self) -> None:
        self.pause = InfraPause.for_run(self.cfg.kind)
        for row in self.state["steps"]:
            if row["step"] > self.state.get("acknowledged_through", 0):
                self.pause.window.append((row["infra"]["episodes"], row["infra"]["infra"]))

    # ------------------------------------------------------------------ spend

    def _commit_spend(self, split: str) -> float:
        """Move the client's spend since the last commit into the state, under "train" or "eval"."""
        spent = self.client.spent_usd()
        delta, self._mark = spent - self._mark, spent
        spend = self.state["spend"]
        spend[f"{split}_usd"] = round(spend[f"{split}_usd"] + delta, 6)
        spend["total_usd"] = round(spend["train_usd"] + spend["eval_usd"], 6)
        self.state["live"] = None
        return delta

    def _live(self, step: int, phase: str) -> None:
        """The heartbeat: what's running, and the spend not yet committed, so an interruption loses no money."""
        self.state["live"] = {"step": step, "phase": phase, "at": time.time(),
                              "uncommitted_usd": round(self.client.spent_usd() - self._mark, 6)}
        self._save()

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            if self.state.get("live"):
                self._live(self.state["live"]["step"], self.state["live"]["phase"])

    def _settle_interrupted(self) -> None:
        live = self.state.get("live")
        if not live:
            return
        lost = live.get("uncommitted_usd", 0.0)
        split = "eval" if live["phase"] in ("eval", "capability") else "train"
        spend = self.state["spend"]
        spend[f"{split}_usd"] = round(spend[f"{split}_usd"] + lost, 6)
        spend["interrupted_usd"] = round(spend["interrupted_usd"] + lost, 6)
        spend["total_usd"] = round(spend["train_usd"] + spend["eval_usd"], 6)
        self.state["live"] = None
        self._notice("interrupted", f"the run stopped during step {live['step']} ({live['phase']}); "
                                    f"${lost:.4f} of inference it had used is counted as spent")

    def _affordable(self, split: str, estimate: float) -> bool:
        ceiling = self.cfg.ceilings.get(split)
        return ceiling is None or self.state["spend"][f"{split}_usd"] + estimate <= ceiling

    def _episode_estimate(self) -> float:
        """Per-episode cost for an eval not yet measured: recent training episodes, with a 1.5x margin."""
        rows = [r for r in self.state["steps"][-3:] if r["infra"]["episodes"]]
        if not rows:
            return self.cfg.eval_episode_estimate_usd
        return 1.5 * max(r["cost"]["step_usd"] / r["infra"]["episodes"] for r in rows)

    def _next_step_estimate(self) -> float:
        costs = [r["cost"]["step_usd"] for r in self.state["steps"][-3:]]
        return max(costs) if costs else self.cfg.first_step_estimate_usd

    # ------------------------------------------------------------------ the loop

    async def _loop(self) -> dict:
        self._heartbeat = asyncio.create_task(self._beat())
        try:
            while True:
                c = self._latest()
                blocked = await self._run_due_evals(c)
                self._snapshot_if_kept(c)
                await self._prune()
                outcome = blocked or await self._check(c)
                if outcome == "rollback":
                    continue
                if outcome is not None:
                    return self.state
                await self._step(c + 1)
        except BaseException as exc:  # recorded for the dashboard; resume() picks up from here
            interrupted = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            if self.state.get("live"):  # the exact spend not yet committed, for resume() to count
                self.state["live"]["uncommitted_usd"] = round(self.client.spent_usd() - self._mark, 6)
            self.state["status"] = "interrupted" if interrupted else "crashed"
            self.state["stop"] = {"rule": "interrupted" if interrupted else "error", "reason": repr(exc)[:500],
                                  "at_step": self._latest(), "at": time.time()}
            self._notice(self.state["status"], f"the script stopped: {exc!r}"[:500])
            raise
        finally:
            self._heartbeat.cancel()
            self._save()

    async def _step(self, n: int) -> None:
        cfg, st = self.cfg, self.state
        root_model = self._model_for(n - 1)
        path = transcript_path(self.run_dir, "train", n)
        if path.exists():  # an earlier attempt at this step was interrupted before it was recorded
            self._move(path, self.run_dir / "abandoned" / "interrupted" / f"step{n:04d}-{int(time.time())}.jsonl")
        started, t0 = time.time(), time.monotonic()
        self.client.reset_stats()  # the step's request stats leave out the evals before it
        self._live(n, "rollouts")
        retired = self.retirement.ids()
        batch = self.draw(cfg.tasks_per_step, cfg.mix, self.ctl.levels(), n, **({"exclude": retired} if retired else {}))
        fallback = self._fallback_slots(n, batch)
        groups = await run_groups(batch, cfg.rollouts, client=self.client, root_model=root_model, run_id=cfg.run,
                                  sandbox_factory=self.sandbox_factory, prompt_version=cfg.prompt_version,
                                  **self.sandbox_kwargs)
        t_rollouts = time.monotonic() - t0
        TranscriptLog(path).extend([episode_record(ep, split="train", run=cfg.run, step=n, sample=sample,
                                                   key=f"train/step{n}/{ep.task_id}/{sample}")
                                    for group in groups for sample, ep in enumerate(group)])
        step_filter = filter_step(groups)
        controller_before = self.ctl.state_dict()
        retirement_before = self.retirement.state_dict()
        moves = self.ctl.update(n, [ep for group in groups for ep in group])
        newly_retired = self.retirement.update(n, groups)
        clean = monitor.scored(groups)
        self._check_template(clean)
        self._live(n, "kl")
        t1 = time.monotonic()
        kl = await monitor.kl_to_base(self.client, clean, base_model=self.base_model, n=cfg.kl_samples, rng=self.rng)
        t_kl = time.monotonic() - t1
        lr = st["lr"]["current"]
        row = {"step": n, "branch": st["branch"], "status": "rolled_out", "lr": lr, "started_at": started,
               "sampled_from": {"checkpoint": n - 1, "model": root_model}, "trained_to": None,
               **monitor.step_metrics(groups, step_filter, controller=self.ctl, moves=moves, kl=kl,
                                      request_stats=self.client.reset_stats()),
               "filter": step_filter.as_dict(),
               "time": {"rollouts_s": round(t_rollouts, 2), "kl_s": round(t_kl, 2), "train_s": None}}
        row["infra_pause"] = self.pause.update(step_filter)
        cost = self._commit_spend("train")
        st["spend"]["kl_usd"] = round(st["spend"]["kl_usd"] + kl["cost_usd"], 6)
        row["cost"] = {"step_usd": round(cost, 6), "kl_usd": kl["cost_usd"], "spent_train_usd": st["spend"]["train_usd"]}
        st["controller"] = self.ctl.state_dict()
        st["retirement"] = self.retirement.state_dict()
        row["retired"] = {"new": len(newly_retired), "total": len(self.retirement.retired), "left": self._pool_left()}
        row["fallback"] = fallback
        if newly_retired:
            left = row["retired"]["left"]
            self._notice("retired", f"step {n}: {len(newly_retired)} edit_real task(s) retired (all-0 twice or all-1 once); "
                                     f"{len(self.retirement.retired)} retired so far"
                                     + (f", {left} left to draw" if left is not None else ""))
        previous = st["steps"][-1].get("fallback") if st["steps"] else None
        for family, moved in fallback.items():
            if not (previous or {}).get(family):  # once when a stretch starts, not every step of it
                self._notice("fallback", f"step {n}: too few {family} tasks left to draw, so {moved['slots']} of its "
                                         f"slot(s) went to {moved['to']} (12b). This repeats each step the pool is short")
        for mv in moves:
            self._notice("level", f"{mv.family} {'would move' if not mv.applied else 'moved'} {mv.direction} "
                                  f"from level {mv.from_level} to {mv.to_level} (pass rate {mv.pass_rate:.0%} over "
                                  f"{mv.episodes} episodes)")
        if step_filter.hacked_episodes:
            hits = ", ".join(f"{rule} {k}" for rule, k in step_filter.hacks.items() if k)
            self._notice("detectors", f"step {n}: detectors fired on {step_filter.hacked_episodes} episode(s) ({hits})")

        if step_filter.skip_train:
            row["status"] = "skipped"
            st["checkpoints"][str(n)] = dict(self._ckpt(n - 1))
            self._notice("skip", f"step {n} trained nothing: {step_filter.skip_train}")
            self._commit_row(row)
        else:
            train_groups = self._trajectory_groups(step_filter.groups)
            latest_before = await self._backend_latest()
            self._write_pending(n, row, train_groups, latest_before, controller_before, retirement_before)
            self._live(n, "training")
            t2 = time.monotonic()
            backend_step, train_metrics = await self._train(n, train_groups, lr, latest_before)
            self._commit_row(self._trained(row, backend_step, train_metrics, time.monotonic() - t2))
        await self._log_step(n, groups)

    def _trained(self, row: dict, backend_step: int, train_metrics: dict | None, train_s: float | None) -> dict:
        row = dict(row, status="trained", trained_to={"checkpoint": row["step"], "backend_step": backend_step})
        row["train_metrics"] = train_metrics
        row["policy"] = dict(row["policy"], **{ours: (train_metrics or {}).get(theirs)
                                                for theirs, ours in monitor.TRAIN_METRIC_KEYS.items()})
        row["time"] = dict(row["time"], train_s=None if train_s is None else round(train_s, 2))
        self.state["checkpoints"][str(row["step"])] = {"branch": row["branch"], "backend_step": backend_step}
        self.state["spend"]["training_gpu_usd_est"] = round(
            self.state["spend"]["training_gpu_usd_est"] + rconfig.TRAINING_GPU_USD_PER_STEP, 4)
        return row

    def _commit_row(self, row: dict) -> None:
        row["finished_at"] = time.time()
        row["time"]["total_s"] = round(row["finished_at"] - row["started_at"], 2)
        self.state["steps"].append(row)
        self.state["pending"] = None
        self.state["live"] = None
        shutil.rmtree(self.run_dir / "pending", ignore_errors=True)
        self._save()
        p, pol = row["pass"], row["policy"]
        self.say(f"[{self.run}] step {row['step']} {row['status']}: pass {_pct(p['rate'])}, trainable "
                 f"{p['trainable_groups']}/{p['groups']}, entropy {_num(pol['entropy'])}, KL {_num(pol['kl_to_base'])}, "
                 f"${row['cost']['step_usd']:.3f} (train ${self.state['spend']['train_usd']:.2f})")

    async def _backend_latest(self) -> int:
        """The branch's newest backend checkpoint, tried 3 times: a W&B hiccup here mustn't stop the run."""
        poll = self.train_retry[2]
        for attempt in range(3):
            try:
                return await self.backend.latest_step(self._branch)
            except Exception as exc:
                if attempt == 2:
                    raise
                self._notice("backend", f"reading the latest checkpoint failed ({exc!r}"[:200] + "); trying again")
                await asyncio.sleep(poll)
        raise AssertionError("unreachable")

    async def _train(self, n: int, groups: list, lr: float, latest_before: int) -> tuple[int, dict | None]:
        """Step n's train call, kept going through a failed call (TRAIN_SETTLE_S above): a failure that trained anyway
        is accepted without its metrics; one that didn't is sent again. The last failure is raised, and resume decides
        (--retrain-pending or --discard-pending)."""
        retries, settle_s, poll_s = self.train_retry
        for attempt in range(retries + 1):
            try:
                return await self.backend.train(self._branch, groups, **self.cfg.train_args(lr))
            except Exception as exc:
                if attempt == retries:
                    raise
                self._notice("training", f"step {n}'s train call failed ({exc!r}"[:200]
                             + "); waiting to see whether it trained anyway")
                landed = await self._wait_for_checkpoint(latest_before, settle_s, poll_s)
                if landed is not None:
                    self._notice("training", f"step {n} trained anyway (backend step {landed}); its training "
                                             "metrics weren't recorded")
                    return landed, None
                self._notice("training", f"no new checkpoint after {settle_s / 60:.0f} minutes, so step {n}'s batch "
                                         f"is sent again (try {attempt + 2} of {retries + 1})")
        raise AssertionError("unreachable")

    async def _wait_for_checkpoint(self, latest_before: int, settle_s: float, poll_s: float) -> int | None:
        """The branch's new backend step once one past `latest_before` shows up, or None after `settle_s`."""
        deadline = time.monotonic() + settle_s
        while True:
            try:
                latest = await self.backend.latest_step(self._branch)
                if latest > latest_before:
                    return latest
            except Exception:  # keep polling through a hiccup; the deadline still holds
                pass
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(poll_s)

    def _fallback_slots(self, n: int, batch: list) -> dict:
        """A fixed pool's slots that went to its fallback family because too few of its tasks were left (12b)."""
        families = [f for f in rtasks.FALLBACK if f in self.cfg.mix]
        if not families:
            return {}
        planned = rtasks.allocate_step(self.cfg.tasks_per_step, self.cfg.mix, n)
        drawn = collections.Counter(t.family for t in batch)
        return {f: {"to": rtasks.FALLBACK[f], "slots": planned[f] - drawn[f]} for f in families if drawn[f] < planned[f]}

    def _pool_left(self) -> int | None:
        """Fixed-pool tasks not retired yet, or None when the run draws from no fixed pool or its index can't be read."""
        families = [f for f in rtasks.RETIRABLE if f in self.cfg.mix]
        sizes = [rtasks.pool_size(f) for f in families]
        if not families or None in sizes:
            return None
        return sum(sizes) - sum(v.get("family") in families for v in self.retirement.retired.values())

    def _trajectory_groups(self, groups: list[list[Episode]]) -> list:
        import art

        return [art.TrajectoryGroup([to_trajectory(ep) for ep in group]) for group in groups]

    async def _log_step(self, n: int, groups: list[list[Episode]]) -> None:
        """The step's trajectories and metrics to W&B through ART. A logging failure never stops the run."""
        import art

        row = self.state["steps"][-1]
        log_groups = [art.TrajectoryGroup([to_trajectory(ep, with_choices=False) for ep in group
                                           if ep.infra_error is None],
                                          exceptions=[InfraError(ep.infra_error) for ep in group if ep.infra_error])
                      for group in groups]
        metrics = {**monitor.flat({k: row[k] for k in ("pass", "reward", "policy", "shape", "infra", "cost", "time", "edit")
                                   if k in row}),
                   **{k: v for k, v in (row.get("train_metrics") or {}).items() if isinstance(v, (int, float))},
                   "rlm/lr": row["lr"], "rlm/spend/total_usd": self.state["spend"]["total_usd"]}
        try:
            await self.backend.log(self._branch, log_groups, split="train", step=n, metrics=metrics)
        except Exception as exc:  # W&B can hiccup; the local JSON is still complete
            self._notice("logging", f"W&B logging failed at step {n}: {exc!r}")

    # ------------------------------------------------------------------ pending training

    def _write_pending(self, n: int, row: dict, train_groups: list, latest_before: int, controller_before: dict,
                       retirement_before: dict | None = None) -> None:
        folder = self.run_dir / "pending"
        folder.mkdir(parents=True, exist_ok=True)
        batch = folder / f"step{n:04d}.pkl"
        try:
            batch.write_bytes(pickle.dumps(train_groups))
        except Exception as exc:  # retraining the batch then needs the operator to redo the step instead
            batch = None
            self._notice("pending", f"the step {n} batch couldn't be saved for a retrain: {exc!r}")
        self.state["pending"] = {"step": n, "branch": self.state["branch"], "row": row, "latest_before": latest_before,
                                 "batch": None if batch is None else str(batch.relative_to(self.run_dir)),
                                 "controller_before": controller_before, "retirement_before": retirement_before,
                                 "sent_at": time.time()}
        self._save()

    async def _settle_pending(self, *, retrain: bool, discard: bool) -> None:
        pending = self.state.get("pending")
        if not pending:
            return
        n, name, row = pending["step"], self._model_of(pending["branch"]), pending["row"]
        latest = await self.backend.latest_step(name)
        if latest != pending["latest_before"]:
            self._commit_row(self._trained(row, latest, None, None))
            self._notice("pending", f"step {n}'s training finished while the script was down (backend step {latest}); "
                                    "its training metrics weren't recorded")
        elif retrain:
            if not pending.get("batch") or not (self.run_dir / pending["batch"]).exists():
                raise ResumeBlocked(f"step {n}'s batch wasn't saved; resume with --discard-pending to redo the step")
            groups = pickle.loads((self.run_dir / pending["batch"]).read_bytes())
            t = time.monotonic()
            backend_step, metrics = await self.backend.train(name, groups, **self.cfg.train_args(row["lr"]))
            self._commit_row(self._trained(row, backend_step, metrics, time.monotonic() - t))
            self._notice("pending", f"step {n}'s saved batch was trained on resume")
        elif discard:
            self.ctl = DifficultyController.from_state(pending["controller_before"])
            self.state["controller"] = self.ctl.state_dict()
            self.retirement = rtasks.Retirement(pending.get("retirement_before"))
            self.state["retirement"] = self.retirement.state_dict()
            self.state["pending"] = None
            shutil.rmtree(self.run_dir / "pending", ignore_errors=True)
            self._notice("pending", f"step {n} was discarded and will be redone (its ${row['cost']['step_usd']:.3f} "
                                    "of rollouts stays counted)")
        else:
            raise ResumeBlocked(
                f"step {n} was sent for training, but the backend has no new checkpoint yet (latest {latest}). "
                "If that training job may still be running, wait a few minutes and resume again. To train the saved "
                "batch now, resume with --retrain-pending; to redo the step from new rollouts, --discard-pending.")

    # ------------------------------------------------------------------ evals and the audit

    def _has_eval(self, c: int, name: str) -> bool:
        return any(e["checkpoint"] == c and e["name"] == name for e in self.state["evals"])

    def _due_evals(self, c: int) -> list[str]:
        cfg, names = self.cfg, []
        if cfg.selection and cfg.eval_every and ((c > 0 and c % cfg.eval_every == 0) or (c == 0 and cfg.eval_at_start)):
            names.append(cfg.selection)
            if self.capability_items:
                names.append("capability")
            elif not self.state.get("capability_noted"):
                self.state["capability_noted"] = True
                self._notice("capability", "no capability prompt set yet (plan step 7), so stop rule 3 only "
                                           "watches for a collapse after the rollback")
        if cfg.final_eval and cfg.max_steps and c == cfg.max_steps:
            names.append(cfg.final_eval)
        return [name for name in names if not self._has_eval(c, name)]

    def _eval_estimate(self, name: str, items: int, per_item: float) -> float:
        past = [e["cost_usd"] for e in self.state["evals"] if e["name"] == name and e.get("ran", 1)]
        return max(past[-2:]) if past else items * per_item

    async def _run_due_evals(self, c: int) -> str | None:
        due = self._due_evals(c)
        model = self._model_for(c)
        for name in due:
            if name == "capability":
                estimate = self._eval_estimate(name, len(self.capability_items), CAPABILITY_ITEM_ESTIMATE_USD)
                if not self._affordable("eval", estimate):
                    return self._finish("stopped", "budget", self._budget_text("eval", estimate, f"the {name} check"))
                self._live(c, "capability")
                result = await capability.run(self.capability_items, client=self.client, model=model)
                TranscriptLog(self.run_dir / "capability" / f"step{c:04d}.jsonl").extend(
                    [{**record, "checkpoint": c, "model": model} for record in result.records])
                entry = {"checkpoint": c, "name": name, "summary": result.summary, "ran": len(result.records)}
                metrics = {f"capability/{k}": v for k, v in result.summary.items() if isinstance(v, (int, float))}
                await self._log_metrics(c, metrics)
            else:
                tasks, samples = self.eval_sets(name)
                estimate = self._eval_estimate(name, len(tasks) * samples, self._episode_estimate())
                if not self._affordable("eval", estimate):
                    return self._finish("stopped", "budget", self._budget_text("eval", estimate, f"the {name} eval"))
                self._live(c, "eval")
                result = await run_eval(tasks, name=name, step=c, run_dir=self.run_dir, run=self.run, client=self.client,
                                        root_model=model, samples=samples, art_model=self.backend.art_model(self._branch),
                                        sandbox_factory=self.sandbox_factory, prompt_version=self.cfg.prompt_version,
                                        **self.sandbox_kwargs)
                entry = {"checkpoint": c, "name": name, "summary": result.summary, "ran": result.ran}
            entry.update(branch=self.state["branch"], model=model, at=time.time(), cost_usd=round(self._commit_spend("eval"), 6))
            self.state["evals"].append(entry)
            self._update_best()
            acc = entry["summary"].get("accuracy")
            self._notice("eval", f"{name} at checkpoint {c}: accuracy {_pct(acc)} over {entry['summary'].get('n', entry['summary'].get('graded'))} "
                                 f"(${entry['cost_usd']:.3f})")
            self._save()
        if c > 0 and (c % AUDIT_EVERY == 0 or c == self.cfg.max_steps) and str(c) not in self.state["audits"]:
            self._audit_sample(c)
        return None

    async def _log_metrics(self, c: int, metrics: dict) -> None:
        try:
            await self.backend.log(self._branch, None, split="val", step=c, metrics=metrics)
        except Exception as exc:
            self._notice("logging", f"W&B logging failed at checkpoint {c}: {exc!r}")

    def _best_eval_name(self) -> str | None:
        return self.cfg.selection or self.cfg.final_eval

    def _update_best(self) -> None:
        best = stoprules.best_checkpoint(stoprules.eval_points(self.state["evals"], self._best_eval_name() or ""))
        self.state["best"] = None if best is None else {"checkpoint": best[0], "accuracy": best[1],
                                                         "eval": self._best_eval_name()}

    def _audit_sample(self, c: int) -> None:
        """Write the Claude audit's sample for checkpoint c (decision 5a, documentation/audit-procedure.md)."""
        eval_name = self.cfg.selection or (self.cfg.final_eval if c == self.cfg.max_steps else None) or "selection"
        audit = _audit_module()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            ok = audit.checkpoint(self.run_dir, c, audit.CHECKPOINT_EVERY, eval_name, audit.DEFAULT_TRAIN,
                                  audit.DEFAULT_VAL, audit.DEFAULT_FLAGGED, force=False) == 0
        detail = (out.getvalue() if ok else err.getvalue()).strip()
        self.state["audits"][str(c)] = {"status": "pending" if ok else "no_sample", "eval": eval_name,
                                        "sample": str((self.run_dir / "audit" / f"step{c:04d}" / "sample.md")
                                                      .relative_to(self.results)) if ok else None,
                                        "detail": detail[-300:]}
        self._notice("audit", f"Claude audit sample for checkpoint {c} is ready" if ok
                     else f"no audit sample for checkpoint {c}: {detail[-200:]}")

    def _audit_hacks(self, *, all_: bool = False) -> list[dict]:
        """Hack verdicts among audited wins, and each audited checkpoint's clean share (for the dashboard)."""
        found = []
        for path in sorted((self.run_dir / "audit").glob("step*/verdicts.json")):
            try:
                verdicts = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            c = verdicts["checkpoint"]
            wins = [e for e in verdicts["episodes"] if e["kind"] == "win"]
            judged = [e for e in wins if e.get("verdict")]
            entry = self.state["audits"].setdefault(str(c), {"status": "pending"})
            entry.update(audited=len(judged), sampled=len(wins),
                         clean_share=sum(e["verdict"] == "clean" for e in judged) / len(judged) if judged else None,
                         status="audited" if judged and len(judged) == len(wins) else entry.get("status", "pending"))
            for e in wins:
                key = f"{c}/{e['episode_id']}"
                if e.get("verdict") == "hack" and (all_ or key not in self.state["acknowledged_audit_hacks"]):
                    found.append({"checkpoint": c, "episode_id": e["episode_id"], "key": key, "note": e.get("note")})
        return found

    # ------------------------------------------------------------------ checkpoints

    def _kept(self) -> set[int]:
        latest = self._latest()
        kept = {0, latest} | set(range(0, latest + 1, self.cfg.keep_every))
        kept |= {e["checkpoint"] for e in self.state["evals"]}
        if self.state.get("best"):
            kept.add(self.state["best"]["checkpoint"])
        return {k for k in kept if str(k) in self.state["checkpoints"]}

    def _snapshot_path(self, c: int) -> Path:
        return self.run_dir / "checkpoints" / f"step{c:04d}.json"

    def _snapshot_if_kept(self, c: int) -> None:
        """Save what resuming from checkpoint c needs, and rewrite config.json (spec §8)."""
        if c not in self._kept() or (c != 0 and c % self.cfg.keep_every and not any(
                e["checkpoint"] == c for e in self.state["evals"])):
            return
        path = self._snapshot_path(c)
        ckpt = self._ckpt(c)
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("branch") == ckpt["branch"] and saved.get("backend_step") == ckpt["backend_step"]:
                return
        if self._git is None:
            self._git = rconfig.git_state()
        pinned = rconfig.pinned(self.cfg, lr=self.state["lr"], controller_state=self.ctl.state_dict(),
                                spend=self.state["spend"], template=self.state["template_fingerprint"], git=self._git)
        snapshot = {"checkpoint": c, "branch": ckpt["branch"], "backend_step": ckpt["backend_step"],
                    "model": self._model_for(c), "lr": dict(self.state["lr"]), "controller": self.ctl.state_dict(),
                    "retirement": self.retirement.state_dict(),
                    "saved_at": time.time(), "config": pinned}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=1, default=str), encoding="utf-8")
        (self.run_dir / "config.json").write_text(json.dumps(pinned, indent=1, default=str), encoding="utf-8")
        self.state["kept"] = sorted(self._kept())
        self._save()

    async def _prune(self) -> None:
        """Delete this branch's checkpoints that aren't kept (spec §8). Unknown backend steps are never touched."""
        if not self.prune_enabled:
            return
        kept = self._kept()
        b = self.state["branch"]
        ours = {}
        for c, ckpt in self.state["checkpoints"].items():
            if ckpt["branch"] == b:
                ours.setdefault(ckpt["backend_step"], set()).add(int(c))
        keep = {s for s, cs in ours.items() if cs & kept}
        try:
            existing = await self.backend.list_steps(self._branch)
            latest = max(existing, default=0)
            delete = sorted(s for s in existing if s in ours and s not in keep and s not in (0, latest))
            if delete:
                await self.backend.delete_steps(self._branch, delete)
                self.state["pruned"].extend({"branch": b, "backend_step": s} for s in delete)
                self.state["kept"] = sorted(kept)
                self._save()
        except Exception as exc:
            self._notice("prune", f"checkpoint pruning failed and will be retried: {exc!r}")

    def _resolve_checkpoint(self, value: str | int | None) -> int | None:
        if value in (None, "latest"):
            return None
        if value == "best":
            if not self.state.get("best"):
                raise ValueError("no eval has picked a best checkpoint yet")
            return self.state["best"]["checkpoint"]
        c = int(value)
        if c not in self._kept() or (c and not self._snapshot_path(c).exists()):
            raise ValueError(f"checkpoint {c} isn't kept; kept checkpoints: {sorted(self._kept())}")
        return c

    async def _rollback(self, target: int, *, reason: str, new_lr: float | None) -> None:
        """Continue from kept checkpoint `target` on a new branch forked from it (nothing is deleted)."""
        st = self.state
        snapshot = None
        if target:
            path = self._snapshot_path(target)
            if not path.exists():
                raise ValueError(f"checkpoint {target} has no snapshot to resume from")
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        src = self._ckpt(target)
        old, new = st["branch"], len(st["branches"])
        name = branch_name(self.run, new)
        if src["backend_step"]:  # checkpoint 0 is the base model: a new model starts there anyway
            await self.backend.branch_from(name, self._model_of(src["branch"]), src["backend_step"])
        else:
            await self.backend.open(name)
        latest = await self.backend.latest_step(name)
        if latest != 0:  # the new branch must start from the seeded step 0, or training continues from elsewhere
            raise RuntimeError(f"{name} already has checkpoint {latest}; a branch from checkpoint {target} needs a new "
                               "model, so nothing was trained")
        folder = self.run_dir / "abandoned" / f"b{old}-after{target}"
        for path in self._files_after(target):
            self._move(path, folder / path.relative_to(self.run_dir))
        st["abandoned"].append({"branch": old, "after": target, "reason": reason, "at": time.time(),
                                "steps": [r for r in st["steps"] if r["step"] > target],
                                "evals": [e for e in st["evals"] if e["checkpoint"] > target],
                                "files": str(folder.relative_to(self.results))})
        st["steps"] = [r for r in st["steps"] if r["step"] <= target]
        st["evals"] = [e for e in st["evals"] if e["checkpoint"] <= target]
        st["checkpoints"] = {k: v for k, v in st["checkpoints"].items() if int(k) <= target}
        st["audits"] = {k: v for k, v in st["audits"].items() if int(k) <= target}
        if snapshot is not None:
            self.ctl = DifficultyController.from_state(snapshot["controller"])
            self.retirement = rtasks.Retirement(snapshot.get("retirement"))
            lr = snapshot["lr"]["current"]
        else:
            self.ctl = DifficultyController(rtasks.max_levels(self.cfg.mix), frozen=self.cfg.frozen_controller)
            self.retirement = rtasks.Retirement()
            lr = self.cfg.learning_rate
        st["controller"] = self.ctl.state_dict()
        st["retirement"] = self.retirement.state_dict()
        st["lr"]["current"] = new_lr if new_lr is not None else lr
        st["branch"] = new
        st["branches"].append({"branch": new, "model": name, "at": time.time(), "reason": reason,
                               "from": {"checkpoint": target, "branch": src["branch"], "backend_step": src["backend_step"]}})
        st["collapse_since"] = target
        st["acknowledged_through"] = min(st["acknowledged_through"], target)
        st["rule5_since"] = min(st["rule5_since"], target)
        st["kept"] = sorted(self._kept())
        self._update_best()
        self._reset_pause_window()
        self._notice("rollback", f"continuing from checkpoint {target} on branch {new} ({name}, started from its LoRA "
                                 f"with a fresh optimizer) at learning rate {st['lr']['current']:g}: {reason}. Steps after "
                                 f"{target} are kept under {folder.name}")
        self._save()

    def _files_after(self, target: int) -> list[Path]:
        def step_of(path: Path) -> int | None:
            stem = path.stem if path.is_file() else path.name
            return int(stem[4:8]) if stem.startswith("step") and stem[4:8].isdigit() else None

        out = []
        for pattern in ("transcripts/**/step*.jsonl", "capability/step*.jsonl", "checkpoints/step*.json", "audit/step*"):
            for path in self.run_dir.glob(pattern):
                s = step_of(path)
                if s is not None and s > target:
                    out.append(path)
        return out

    @staticmethod
    def _move(src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

    def _check_template(self, episodes: list[Episode]) -> None:
        for ep in episodes:
            if ep.choices:
                ids = (ep.choices[0].model_extra or {}).get("prompt_token_ids")
                if ids:
                    fingerprint = rconfig.template_fingerprint(ids)
                    known = self.state["template_fingerprint"]
                    if known is None:
                        self.state["template_fingerprint"] = fingerprint
                    elif fingerprint != known:
                        self._notice("template", "the server's rendering of the system prompt changed "
                                                 f"({known[:12]} -> {fingerprint[:12]}): check the chat template")
                        self.state["template_fingerprint"] = fingerprint
                return

    # ------------------------------------------------------------------ rules

    def _budget_text(self, split: str, estimate: float, what: str) -> str:
        spend = self.state["spend"][f"{split}_usd"]
        return (f"{what} would take the {split} spend from ${spend:.2f} past its ${self.cfg.ceilings[split]:g} ceiling "
                f"(estimate ${estimate:.2f})")

    async def _check(self, c: int) -> str | None:
        """Decide what happens after checkpoint c: None to keep going, "rollback", or the run's final status."""
        st, cfg = self.state, self.cfg
        rows = st["steps"]
        last = rows[-1] if rows else None
        reasons: list[tuple[str, str, str]] = []  # (status, rule, reason)

        if last and last["step"] > st["acknowledged_through"] and last.get("infra_pause"):
            reasons.append(("paused", "infra", last["infra_pause"]))

        frying_reason = None
        signal = stoprules.collapse(rows, since_step=st["collapse_since"])
        if signal and f"{st['branch']}:{signal.step}" not in st["collapse_noted"]:
            st["collapse_noted"].append(f"{st['branch']}:{signal.step}")
            lr = st["lr"]
            if cfg.lr_rule == "fallback" and not lr["fallback_used"] and lr["current"] > lr["floor"]:
                kept = [k for k in self._kept() if k == 0 or self._snapshot_path(k).exists()]
                target = stoprules.last_good_checkpoint(signal.first_bad_step, kept)
                lr["fallback_used"] = True
                lr["fallback"] = {"at_step": signal.step, "signal": signal.kind, "to_checkpoint": target,
                                  "from_lr": lr["current"], "to_lr": lr["floor"]}
                self._notice("lr_fallback", f"{signal.text}. Rolling back to checkpoint {target} and continuing at "
                                            f"{lr['floor']:g} (spec §5.2)")
                await self._rollback(target, reason=f"LR fallback after a collapse ({signal.kind})", new_lr=lr["floor"])
                return "rollback"
            if lr["fallback_used"]:
                frying_reason = f"a collapse came back after the rollback to {lr['floor']:g}: {signal.text}"
            elif cfg.kind == "production":
                reasons.append(("paused", "collapse", f"{signal.text}. The run is already at {lr['current']:g}, the "
                                                      "lowest rate the LR rule allows, so the next move is your call"))
            else:
                self._notice("collapse", f"{signal.text} (logged only: the second seed's curves decide production's LR)")
        warning = stoprules.collapse_warning(rows, since_step=st["collapse_since"])
        if warning and not stoprules.collapse_warning(rows[:-1], since_step=st["collapse_since"]):  # once a stretch
            self._notice("collapse", f"warning, no rollback: {warning}")

        audit_hacks = self._audit_hacks()
        points = stoprules.eval_points(st["evals"], cfg.selection or "")
        level_ups = [m["step"] for m in st["controller"]["history"] if m["applied"] and m["to_level"] > m["from_level"]]
        statuses = [stoprules.plateau(points, level_ups), stoprules.overfitting(points, rows),
                    stoprules.frying(st["evals"], collapse_after_rollback=frying_reason),
                    stoprules.saturation(self.ctl.summary()),
                    stoprules.hacking(rows, audit_hacks, since_step=st["rule5_since"])]
        st["stop_rules"] = {s.rule: s.as_dict() for s in statuses}
        for s in statuses:
            if s.fired:
                reasons.append(("paused" if s.action == "pause" else "stopped", s.rule, s.text))

        if cfg.max_steps and c >= cfg.max_steps:
            reasons.append(("finished", "max_steps", f"reached {cfg.max_steps} steps"))
        stop_file = self.run_dir / STOP_FILE
        if stop_file.exists():
            reason = stop_file.read_text(encoding="utf-8").strip() or "stop requested"
            stop_file.unlink()
            reasons.append(("stopped", "operator", f"stop switch: {reason}"))
        elif self.stop_requested:
            reasons.append(("stopped", "operator", self.stop_requested))
        elif self.stop_after is not None and c >= self.stop_after:
            reasons.append(("stopped", "operator", f"stop after step {self.stop_after}"))
        if not reasons:
            estimate = self._next_step_estimate()
            if not self._affordable("train", estimate):
                text = self._budget_text("train", estimate, f"step {c + 1}")
                recent = stoprules.new_bests(points)[-2:]
                if len(recent) == 2 and all(recent):
                    text += ". The last 2 evals were both new bests, so the research would spend the reserve: your call (12a)"
                reasons.append(("stopped", "budget", text))
        if not reasons:
            self._save()
            return None
        order = {"paused": 0, "stopped": 1, "finished": 2}
        status, rule, reason = sorted(reasons, key=lambda r: order[r[0]])[0]
        for other in reasons:
            if other[1] != rule:
                self._notice("rule", f"also: {other[1]}: {other[2]}")
        return self._finish(status, rule, reason)

    def _finish(self, status: str, rule: str, reason: str) -> str:
        st = self.state
        st["status"] = status
        st["stop"] = {"rule": rule, "reason": reason, "at_step": self._latest(), "at": time.time(),
                      "best": st.get("best")}
        if self.cfg.kind == "second_seed" and status == "finished":
            st["lr_decision"] = stoprules.lr_decision(st["steps"], high=rconfig.LR_PRODUCTION_HIGH,
                                                      floor=rconfig.LR_FLOOR)
            self._notice("lr_decision", st["lr_decision"]["note"])
        best = st.get("best")
        tail = f" Best checkpoint: {best['checkpoint']} ({_pct(best['accuracy'])} on {best['eval']})." if best else ""
        self._notice(status, f"{rule}: {reason}.{tail}")
        self._save()
        return status


_AUDIT = None


def _audit_module():
    """scripts/audit.py, loaded once in this process (a new process costs seconds of imports in WSL2)."""
    global _AUDIT
    if _AUDIT is None:
        spec = importlib.util.spec_from_file_location("rlm_audit_script", REPO / "scripts" / "audit.py")
        _AUDIT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_AUDIT)
    return _AUDIT


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _num(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"
