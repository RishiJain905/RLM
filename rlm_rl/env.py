"""The RLM episode loop: root turns through the shared client, code cells in the sandbox (spec §2, §4, §5.4).

Step 4 built the loop, the infra_error flag on each episode (H5), and what a step does with
flagged episodes before training (spec §5.4 steps 3-6). Step 5 makes the calibrated reward
(reward.py, spec §3) the default scorer and runs the hack detectors (detectors.py, spec §7) on
every scored episode: a detector that fires zeroes the reward before the group mean (§5.4 step 2).
The difficulty controller (controller.py) and the task generators (tasks/) sit beside the loop.

An episode never raises. Anything that goes wrong outside the model's control (a request that
still fails after retries, a sandbox that can't start or is stopped from outside, a bug on the
host) stops it and sets `infra_error`. Such an episode is logged and counted, gets no reward,
and filter_step() keeps it out of training.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import hashlib
import re
import secrets
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from . import bridge, detectors, reward
from .bridge import InfraError
from .client import ModelClient, RequestFailed, Usage, reasoning_text
from .reward import Score
from .sandbox import CellResult, Sandbox

SYSTEM_PROMPT = bridge.SYSTEM_PROMPT  # tests/test_contract.py diffs this against the bridge (H10)

INFRA_PAUSE_RATE = 0.01  # spec §5.4 step 4
# How many recent steps the 1% is measured over (Rishi, 2026-10-11): the step alone for the second seed, so
# every infra error gets looked at, and the last 10 steps (320 episodes) in production.
PAUSE_WINDOW_STEPS = {"second_seed": 1, "production": 10}
MIN_GROUP_SIZE = 2  # spec §5.4 step 3
MIN_GROUPS = 2  # spec §5.4 step 6

_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


# Qwen3.8's own tool-call syntax around Python, as the base model wrote it on 43% of its turns under prompt v2 (step 11
# results §3): "<tool_call>", "<function=python>", the code, "</function>", "</tool_call>", sometimes with a parameter
# tag around the code, sometimes cut off before the closing tags.
_TOOL_CALL_CODE_RE = re.compile(r"<function=python>[ \t]*\n?(?:<parameter=\w+>[ \t]*\n?)?(.*?)"
                                r"(?:</parameter>\s*)?(?:</function>|</tool_call>|\Z)", re.DOTALL)


def extract_code(text: str | None, tool_calls: bool = False) -> str | None:
    """The first fenced code block in a reply, or None. With `tool_calls` (prompt v3), a reply without one may carry
    its code in a `<function=python>` tool call instead."""
    match = _CODE_RE.search(text or "")
    if match:
        return match.group(1)
    if tool_calls:
        found = _TOOL_CALL_CODE_RE.search(text or "")
        if found and found.group(1).strip():
            return found.group(1)
    return None


@dataclass
class Task:
    """One task. Only `context` enters the sandbox; `answer` and `grading` stay on the host (H3).

    `task_id` names the task in logs and on the dashboard, so it must never contain the seed.
    """

    task_id: str
    family: str
    question: str
    context: str
    answer: Any = None
    level: int = 1
    grading: dict = field(default_factory=dict)  # what the scorer needs beyond `answer`, e.g. the blind guess
    # Edit families (spec §12): what the first message names ({"language", "n_files", "test_hint", "image", "repo",
    # "kind"}), and the repository's base, built on the host (editbox.Workspace). Neither enters the sandbox as such.
    edit: dict | None = None
    workspace: Any = field(default=None, repr=False)


Scorer = Callable[[Task, "Episode"], Score]
Detect = Callable[[Task, "Episode"], list]


@dataclass
class Turn:
    reply: str
    finish_reason: str | None
    completion_tokens: int
    code: str | None
    cell: dict | None  # the cell's result; None when the reply had no code block
    observation: str | None  # what the model was shown next; None after a submit or the last turn
    root_s: float  # the root call, queueing and retries included
    cell_s: float
    reasoning_tokens: int = 0  # thinking tokens in the turn's output: 0 with thinking off, as in training
    reasoning: str | None = None  # the thinking text, when the root thinks (plan step 11's medium-thinking arm)


@dataclass
class Episode:
    task_id: str
    family: str
    level: int
    question: str
    gold: Any
    root_model: str
    subcall_model: str
    episode_id: str = field(default_factory=lambda: secrets.token_hex(8))
    messages: list[dict] = field(default_factory=list)  # the conversation exactly as the model saw it
    turns: list[Turn] = field(default_factory=list)
    subcalls: list[dict] = field(default_factory=list)  # prompt, output, error, seconds
    subcalls_used: int = 0
    submitted: bool = False
    answer: Any = None
    answer_valid: bool = True
    infra_error: str | None = None
    infra_kind: str | None = None
    host_traceback: str | None = None
    reward: float | None = None
    correct: bool | None = None  # exact, and no detector fired: the headline metric
    reward_components: dict = field(default_factory=dict)
    hacks: list[dict] = field(default_factory=list)  # detectors that fired: {"rule": ..., "detail": ...}
    events: list[dict] = field(default_factory=list)  # blocked or logged attempts inside the sandbox
    rejected: list[str] = field(default_factory=list)  # bridge requests refused
    timeouts: int = 0
    crashes: int = 0
    restarts: int = 0
    usage: Usage = field(default_factory=Usage)
    context_chars: int = 0
    context_lines: int = 0
    context_sha256: str = ""
    started_at: float = field(default_factory=time.time)
    wall_s: float = 0.0
    choices: list = field(default_factory=list, repr=False)  # root choices with token ids, for training only
    prompt_version: str = bridge.PROMPT_VERSION
    limits: dict = field(default_factory=dict)  # the limits it ran under (spec §12.3)
    # edit families (spec §12)
    language: str | None = None
    repo: str | None = None  # edit_real's repository, so the dashboard can keep held-out ones off the page
    diff: str | None = None  # the repository's change against its base
    grade: dict | None = None  # grader.GradeResult.as_dict()
    edit_stats: dict = field(default_factory=dict)  # spec §12.9's per-episode numbers (editmetrics.py)
    flags: list = field(default_factory=list)  # logged-only detector hits, e.g. network_try: never change the reward

    @property
    def budget_hit(self) -> bool:
        return self.subcalls_used >= (self.limits or {}).get("max_subcalls", bridge.MAX_SUBCALLS)

    def metrics(self) -> dict[str, float]:
        """Per-episode numbers, logged with the trajectory and averaged per step."""
        turns = self.turns
        fired = {hit["rule"] for hit in self.hacks}
        flagged = fired | {hit["rule"] for hit in self.flags}
        return {
            "correct": float(bool(self.correct)),
            "exact": float(self.reward_components.get("exact", 0.0)),
            "partial": float(self.reward_components.get("partial", 0.0)),
            "near_blind": float(self.reward_components.get("near_blind", 0.0)),
            "hack": float(bool(fired)),
            **{f"hack_{rule}": float(rule in flagged) for rule in detectors.RULES},
            "submitted": float(self.submitted),
            "turns": float(len(turns)),
            "subcalls": float(self.subcalls_used),
            "budget_hit": float(self.budget_hit),
            "first_turn_submit": float(self.submitted and len(turns) == 1),
            "no_code_turns": float(sum(t.code is None for t in turns)),
            "broken_replies": float(reward.broken_replies(turns)),
            "format_penalty": float(self.reward_components.get("format_penalty", 0.0)),
            "truncated_turns": float(sum(t.finish_reason == "length" for t in turns)),
            "repl_errors": float(sum(bool(t.cell and t.cell.get("error")) for t in turns)),
            "timeouts": float(self.timeouts),
            "crashes": float(self.crashes),
            "escape_events": float(len(self.events)),
            "root_completion_tokens": float(self.usage.root_completion_tokens),
            "subcall_tokens": float(self.usage.subcall_prompt_tokens + self.usage.subcall_completion_tokens),
            "est_cost_usd": self.usage.cost_usd(),
            "wall_s": self.wall_s,
        }


def _cap_output(result: CellResult, max_chars: int) -> None:
    """Cut a cell's output to the episode's own cap. The REPL keeps bridge.OUTPUT_MAX_CHARS, the largest cap, so a
    prompt v1 episode is cut here to the 2,000 characters it ran with; transcripts and detectors see what the model saw."""
    result.stdout = result.stdout[:max_chars]
    if result.error is not None:
        result.error = result.error[:2 * max_chars]  # the sandbox's own error cut, at this episode's cap


def _cell_dict(result: CellResult) -> dict:
    out = {"stdout": result.stdout, "stdout_chars": result.stdout_chars, "error": result.error,
           "stderr": result.stderr, "submitted": result.submitted, "answer": result.answer,
           "answer_valid": result.answer_valid, "restarted": result.restarted, "crash": result.crash,
           "compute_s": round(result.compute_s, 3), "blocked_s": round(result.blocked_s, 3),
           "wall_s": round(result.wall_s, 3)}
    if result.repo_changed is not None:
        out["repo_changed"] = result.repo_changed
    return out


def _default_grade(task: Task, ep: Episode, box: Any, run_id: str):
    """The real grader (spec §12.5), in a fresh container from the episode's image."""
    from . import grader

    spec = grader.TestSpec.from_dict(task.grading["tests"])
    return grader.grade(ep.diff or "", spec, ws_dir=box.ws_dir, image=box.edit_image, run_id=run_id,
                        docker=box.docker, runtime=box.runtime, bundle=box.bundle)


def _edit_sandbox_limits(lim: bridge.Limits):
    from .editbox import edit_sandbox_limits

    return edit_sandbox_limits(lim)


def _default_edit_sandbox(context: str, complete, *, task: Task, run_id: str, max_subcalls: int, messages, **kwargs):
    from .editbox import EditImage, EditSandbox, derived_image

    image = derived_image(task.edit["image"]) if (task.edit or {}).get("image") else EditImage.synth()
    return EditSandbox(context, complete, workspace=task.workspace, image=image, run_id=run_id,
                       max_subcalls=max_subcalls, messages=messages, **kwargs)


async def run_episode(task: Task, *, client: ModelClient, root_model: str, scorer: Scorer = reward.score,
                      detect: Detect | None = detectors.check, run_id: str = "adhoc",
                      sandbox_factory: Callable[..., Any] = Sandbox, prompt_version: str = bridge.PROMPT_VERSION,
                      edit_sandbox_factory: Callable[..., Any] | None = None, edit_sandbox_kwargs: dict | None = None,
                      grade: Callable[..., Any] | None = None, format_penalty: float = 0.0, **sandbox_kwargs) -> Episode:
    """Run one episode to a submit, the turn limit or an infra error. Never raises, except on cancellation.

    A finished episode is scored (spec §3 steps 1-3) and then checked by every hack detector. If
    any fires, its reward becomes 0 and it no longer counts as correct (step 4). `detect=None`
    turns the detectors off, for tests only. `format_penalty` (training runs that turn it on) takes that much off the
    reward for each broken reply, before the detectors (reward.broken_replies).

    An edit task (spec §12) runs in the edit sandbox (`edit_sandbox_factory`, editbox.EditSandbox by
    default) under the edit limits. After the last turn the host takes the repository's diff, and a
    submitted, non-empty diff is graded in a fresh container (`grade`, grader.grade by default). A
    grader that fails is an infra error, like a sandbox that fails.
    """
    lim = bridge.limits(task.family, prompt_version)
    edit = bridge.is_edit(task.family)
    task_limits = {}
    if edit and (task.edit or {}).get("cell_timeout_s"):  # a task whose own tests need longer cells (spec §12.3)
        lim = dataclasses.replace(lim, cell_timeout_s=int(task.edit["cell_timeout_s"]))
        task_limits = {"limits": _edit_sandbox_limits(lim)}
    ep = Episode(task.task_id, task.family, task.level, task.question, task.answer, root_model, client.subcall_model,
                 context_chars=len(task.context), context_lines=len(task.context.splitlines()),
                 context_sha256=hashlib.sha256(task.context.encode("utf-8", "replace")).hexdigest(),
                 prompt_version=prompt_version, limits=lim.as_dict())
    if edit:
        ep.language = (task.edit or {}).get("language") or task.grading.get("language")
        ep.repo = (task.edit or {}).get("repo") or task.grading.get("repo")
    ep.messages = bridge.initial_messages(task.question, task.context, prompt_version, lim,
                                          edit=task.edit if edit else None)
    subcall_failures: list[RequestFailed] = []
    serve = client.subcaller(ep.usage)

    async def complete(prompt: str) -> str:
        try:
            return await serve(prompt)
        except RequestFailed as exc:
            subcall_failures.append(exc)
            raise

    started = time.monotonic()
    box = None
    messages = bridge.cell_messages(prompt_version, lim, edit)
    try:
        if edit:
            factory = edit_sandbox_factory or _default_edit_sandbox
            box = factory(task.context, complete, task=task, run_id=run_id, max_subcalls=lim.max_subcalls,
                          messages=messages, **task_limits, **(edit_sandbox_kwargs or {}))
        else:
            if prompt_version != bridge.PROMPT_VERSION:
                sandbox_kwargs = {"messages": messages, **sandbox_kwargs}
            box = sandbox_factory(task.context, complete, run_id=run_id, **sandbox_kwargs)
        async with box:
            for index in range(lim.max_turns):
                t0 = time.monotonic()
                choice = await client.root_turn(ep.messages, model=root_model, usage=ep.usage,
                                                max_tokens=lim.root_max_tokens)
                t1 = time.monotonic()
                reply = choice.message.content or ""
                ep.choices.append(choice)
                ep.messages.append({"role": "assistant", "content": reply})
                code = extract_code(reply, tool_calls=prompt_version in bridge.TOOL_CALL_CODE_VERSIONS)
                result = (CellResult("", 0, bridge.no_code_error(prompt_version)) if code is None
                          else await box.run_cell(code))
                _cap_output(result, lim.output_max_chars)
                extra = choice.model_extra or {}
                turn = Turn(reply, choice.finish_reason, len(extra.get("token_ids", ())),
                            code, None if code is None else _cell_dict(result), None,
                            round(t1 - t0, 3), round(time.monotonic() - t1, 3),
                            reasoning_tokens=int(extra.get("reasoning_tokens") or 0), reasoning=reasoning_text(choice))
                ep.turns.append(turn)
                if result.submitted:
                    ep.submitted, ep.answer, ep.answer_valid = True, result.answer, result.answer_valid
                    break
                turns_left = lim.max_turns - index - 1
                if turns_left:
                    turn.observation = bridge.observation(result.stdout, result.stdout_chars, result.error, turns_left,
                                                          box.bridge.max_subcalls - box.bridge.subcalls_used,
                                                          max_chars=lim.output_max_chars)
                    ep.messages.append({"role": "user", "content": turn.observation})
            if edit:
                ep.diff = await box.diff()
    except InfraError as exc:
        cause = exc if isinstance(exc, RequestFailed) else (subcall_failures[0] if subcall_failures else None)
        ep.infra_kind = cause.kind if cause is not None else "sandbox"
        ep.infra_error = str(exc)
    except Exception as exc:  # a bug on the host: counted and never trained on; the pause rule surfaces it
        ep.infra_kind, ep.infra_error = "host_bug", f"host error: {exc!r}"
        ep.host_traceback = traceback.format_exc()
    finally:
        ep.wall_s = round(time.monotonic() - started, 3)
        if box is not None:
            _collect(ep, box)

    if edit and ep.infra_error is None:
        await _grade(task, ep, box, run_id, grade)
    if ep.infra_error is None:
        try:
            if edit:
                from . import editmetrics

                ep.edit_stats = editmetrics.episode_stats(ep)
            score = scorer(task, ep)
            value, correct, components = float(score.reward), bool(score.correct), dict(score.components)
            broken = reward.broken_replies(ep.turns)
            if format_penalty and broken:
                cut = reward.format_penalty(broken, format_penalty)
                components.update(broken_replies=broken, format_penalty=cut, reward_before_format=value)
                value = round(value - cut, 4)
            hacks = list(detect(task, ep)) if detect is not None else []
            ep.flags = list(detectors.flags(task, ep)) if detect is not None else []
            if hacks:  # zeroed before the group mean, so the hack becomes a negative example (H11)
                components.update(hack=1.0, reward_before_hack=value)
                value, correct = 0.0, False
            ep.reward, ep.correct, ep.reward_components, ep.hacks = value, correct, components, hacks
        except Exception as exc:
            ep.infra_kind, ep.infra_error = "host_bug", f"scorer error: {exc!r}"
            ep.host_traceback = traceback.format_exc()
    return ep


async def _grade(task: Task, ep: Episode, box: Any, run_id: str, grade: Callable | None) -> None:
    """Grade a finished edit episode (spec §12.5). Nothing to grade without a submit or a change; a grader that fails is
    an infra error (spec §12.6 step 5)."""
    from .grader import GradeResult

    if not ep.submitted or not (ep.diff or "").strip():
        ep.grade = GradeResult(applied=True, empty=not (ep.diff or "").strip(), resolved=False,
                               f2p_total=len(task.grading["tests"]["f2p"]),
                               p2p_total=len(task.grading["tests"]["p2p"])).as_dict()
        ep.grade["graded"] = False
        return
    try:
        result = await (grade(task, ep, box, run_id) if grade is not None else _default_grade(task, ep, box, run_id))
        ep.grade = dict(result.as_dict() if hasattr(result, "as_dict") else result, graded=True)
    except InfraError as exc:
        ep.infra_kind, ep.infra_error = "grader", str(exc)
    except Exception as exc:
        ep.infra_kind, ep.infra_error = "host_bug", f"grader error: {exc!r}"
        ep.host_traceback = traceback.format_exc()


def _collect(ep: Episode, box: Any) -> None:
    bridge_ = box.bridge
    ep.subcalls_used = bridge_.subcalls_used
    ep.subcalls = [{"prompt": r.prompt, "output": r.output, "error": r.error, "seconds": round(r.seconds, 3)}
                   for r in bridge_.records]
    ep.rejected = list(bridge_.rejected)
    ep.events = list(box.events)
    ep.timeouts, ep.crashes, ep.restarts = box.timeouts, box.crashes, box.restarts


async def run_groups(tasks: list[Task], rollouts: int, **episode_kwargs) -> list[list[Episode]]:
    """One step's rollouts: `rollouts` episodes per task, all at once. The limiter paces the requests."""
    episodes = await asyncio.gather(*(run_episode(task, **episode_kwargs) for task in tasks for _ in range(rollouts)))
    return [list(episodes[i * rollouts:(i + 1) * rollouts]) for i in range(len(tasks))]


# ----------------------------------------------------------------------------- before training (spec §5.4)


@dataclass
class StepFilter:
    """What spec §5.4 steps 3, 5 and 6 did to one step's groups. `groups` is what gets trained on."""

    groups: list[list[Episode]]
    episodes: int
    infra: int
    infra_kinds: dict
    dropped_small: int  # left with fewer than 2 episodes once infra episodes were removed
    dropped_all_zero: int
    dropped_all_one: int
    dropped_equal_other: int  # every reward equal, but neither all 0 nor all 1
    skip_train: str | None  # why this step's train call is skipped, if it is
    hacks: dict = field(default_factory=dict)  # detector hits by rule over the step's scored episodes
    hacked_episodes: int = 0
    hacked_contained: int = 0  # of those, the ones whose only hit was a blocked file or process attempt (detectors.contained)

    @property
    def infra_rate(self) -> float:
        return self.infra / self.episodes if self.episodes else 0.0

    def as_dict(self) -> dict:
        return {"episodes": self.episodes, "infra": self.infra, "infra_rate": self.infra_rate,
                "infra_kinds": self.infra_kinds, "trainable_groups": len(self.groups),
                "trained_episodes": sum(map(len, self.groups)), "dropped_small": self.dropped_small,
                "dropped_all_zero": self.dropped_all_zero, "dropped_all_one": self.dropped_all_one,
                "dropped_equal_other": self.dropped_equal_other, "skip_train": self.skip_train,
                "hacks": self.hacks, "hacked_episodes": self.hacked_episodes, "hacked_contained": self.hacked_contained}


def trainable_turns(ep: Episode, *, mask_broken: bool) -> int:
    """How many of the episode's turns would take a gradient."""
    return len(ep.choices) - (reward.broken_replies(ep.turns) if mask_broken else 0)


def filter_step(groups: list[list[Episode]]) -> StepFilter:
    """Remove infra episodes, then drop groups that are too small or teach nothing (spec §5.4 steps 3, 5, 6).

    Rewards are already final: run_episode zeroed every episode a detector flagged (step 2 of §5.4).
    """
    episodes = sum(map(len, groups))
    kinds = collections.Counter(ep.infra_kind for group in groups for ep in group if ep.infra_error)
    kept, small, zero, one, equal = [], 0, 0, 0, 0
    for group in groups:
        clean = [ep for ep in group if ep.infra_error is None]
        if len(clean) < MIN_GROUP_SIZE:
            small += 1
            continue
        rewards = {ep.reward for ep in clean}
        if len(rewards) == 1:
            value = rewards.pop()
            zero, one, equal = zero + (value == 0), one + (value == 1), equal + (value not in (0, 1))
            continue
        kept.append(clean)
    skip = None
    if len(kept) < MIN_GROUPS:
        skip = f"only {len(kept)} trainable group{'s' if len(kept) != 1 else ''} (need {MIN_GROUPS})"
    scored = [ep for group in groups for ep in group if ep.infra_error is None]
    return StepFilter(kept, episodes, sum(kinds.values()), dict(kinds), small, zero, one, equal, skip,
                      detectors.counts(scored), sum(bool(ep.hacks) for ep in scored),
                      sum(detectors.contained(ep) for ep in scored))


class InfraPause:
    """Spec §5.4 step 4: pause the run when more than 1% of recent episodes were infra errors.

    The rate is measured over the last `window_steps` steps. With one step, as in the second
    seed, a single flagged episode out of 32 is 3.1%, so any infra error pauses. With 10, as in
    production, the run pauses at 4 of the last 320 episodes.

        pause = InfraPause.for_run("production")
    """

    def __init__(self, max_rate: float = INFRA_PAUSE_RATE, window_steps: int = 1):
        self.max_rate = max_rate
        self.window = collections.deque(maxlen=window_steps)

    @classmethod
    def for_run(cls, kind: str) -> InfraPause:
        """The rule Rishi chose for this kind of run: "second_seed" or "production"."""
        return cls(window_steps=PAUSE_WINDOW_STEPS[kind])

    def update(self, step_filter: StepFilter) -> str | None:
        """Record one step. Returns why the run must pause, or None."""
        self.window.append((step_filter.episodes, step_filter.infra))
        episodes = sum(e for e, _ in self.window)
        infra = sum(i for _, i in self.window)
        if episodes and infra / episodes > self.max_rate:
            kinds = ", ".join(f"{k} {n}" for k, n in sorted(step_filter.infra_kinds.items())) or "none this step"
            return (f"{infra} of {episodes} episodes over the last {len(self.window)} step(s) were infra errors "
                    f"({infra / episodes:.1%} > {self.max_rate:.0%}; this step: {kinds})")
        return None


def to_trajectory(ep: Episode, *, with_choices: bool = True, mask_broken: bool = False) -> Any:
    """An ART trajectory for training (`with_choices`, needs the root choices with token ids) or for logging.

    `mask_broken` leaves each broken reply (reward.broken_replies) in as a plain message, so it is context for the turns
    after it but takes no gradient either way: a win doesn't reinforce it, and a loss doesn't push down the ordinary
    tokens it is mostly made of (prod-v1, 2026-10-18).
    """
    import art

    if ep.infra_error is not None:
        raise ValueError("an infra_error episode never becomes a trajectory")
    items: list = list(ep.messages[:2])
    choices = iter(ep.choices)
    turns = iter(ep.turns)
    for message in ep.messages[2:]:
        if with_choices and message["role"] == "assistant":
            choice, turn = next(choices), next(turns, None)
            masked = mask_broken and turn is not None and reward.broken_replies([turn]) == 1
            items.append(message if masked else choice)
        else:
            items.append(message)
    return art.Trajectory(
        messages_and_choices=items,
        reward=float(ep.reward),
        metrics=ep.metrics(),
        metadata={"task_id": ep.task_id, "family": ep.family, "level": ep.level, "episode_id": ep.episode_id,
                  "gold": str(ep.gold), "answer": "" if ep.answer is None else str(ep.answer),
                  "hacks": ",".join(hit["rule"] for hit in ep.hacks)},
    )
