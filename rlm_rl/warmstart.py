"""The coding warm start (step-12 plan step 3; prod-v2, Rishi 2026-10-19): SFT on correct coding episodes that
really delegated, so RL has delegating wins to reinforce.

prod-v2's step 1: base Qwen solved none of its 12 build-from-spec episodes and delegated almost nothing in coding
(1.2 sub-calls an episode), so the delegation bonus had nothing to reward. RL mostly amplifies what a policy already
does (step-12-multistep-plan.md, problem 5), so a short SFT first shows it the habit:

1. Sampling (scripts/warm_start.py sample): coding tasks from the training pools, run on the base model with
   TEACHER_EDIT appended to the system prompt (env.run_episode(system_extra=)): a concrete delegating workflow.
2. Selection (`select`): episodes that are right, unhacked, without infra errors, and delegated by the bonus's own
   gate (training.delegated).
3. Data (`sft_trajectories`): the teacher hint is stripped, so the model learns to delegate unprompted. Each kept turn
   becomes its own sample: the conversation before it exactly as the model saw it when sampling (earlier replies
   without their thinking), then that turn's thinking and reply as the only target (ART assistant_turns="last").
   Qwen's chat template keeps thinking only on the last turn, so one sample per episode would train every earlier
   turn without its thinking. A broken reply, or one with no code, is never a target.
4. Training (TrainingRun.warm_start): one SFT call on the run's model; the latest checkpoint then points at the
   result, so the next RL step samples from the warm-started weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable

from . import bridge, reward

TEACHER_EDIT = (
    "How to work on this task (a worked method; follow it):\n"
    "1. Don't read big files in small slices yourself. In your first turns, read each file the task names into a "
    "variable and give it to llm() with the task (one sub-call reads up to 60,000 characters). Ask for what you need: "
    "the helpers, conventions and existing code the change depends on, and a first draft of each function to write, or "
    "the most likely cause of the failure and its fix. Use llm_batch for parts that don't depend on each other.\n"
    "2. Write the drafts into the files with edit_file or write_file by about a third of your turns.\n"
    "3. Run the tests early. When one fails, give llm() the failing output and the code it exercises, and ask for a "
    "fix; apply it and run the tests again.\n"
    "4. When the tests pass, call submit() with a one-line summary.")

SEPARATOR = "\n\n"


def strip_teacher(system: str, teacher: str = TEACHER_EDIT) -> str:
    """The system prompt as the policy sees it in training: the sampled one without the teacher hint."""
    suffix = SEPARATOR + teacher
    if not system.endswith(suffix):
        raise ValueError("this episode's system prompt doesn't end with the teacher hint")
    return system[:-len(suffix)]


def keep(record: dict) -> bool:
    """A sampled episode worth imitating: right, unhacked, no infra error, and really delegated."""
    from .training import delegated

    if record.get("infra_error") or record.get("hacks") or not record.get("correct"):
        return False
    return delegated(SimpleNamespace(subcalls=record.get("subcalls") or [], family=record["family"],
                                     context_chars=record.get("context_chars") or 0))


def _turn(record_turn: dict) -> SimpleNamespace:
    return SimpleNamespace(**{k: record_turn.get(k) for k in ("reply", "finish_reason", "code", "completion_tokens")})


def sft_samples(record: dict, teacher: str = TEACHER_EDIT) -> list[list[dict]]:
    """One message list per kept turn of a sampled episode: the stripped system prompt, the conversation up to that
    turn as it was sampled, and the turn's reply with its thinking (`reasoning_content`) last."""
    messages = [dict(m) for m in record["messages"]]
    messages[0] = {**messages[0], "content": strip_teacher(messages[0]["content"], teacher)}
    turns = list(record.get("turns") or [])
    out, k = [], 0
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        turn = turns[k] if k < len(turns) else None
        k += 1
        if turn is None or turn.get("code") is None or reward.broken_replies([_turn(turn)]):
            continue
        target = {"role": "assistant", "content": message["content"]}
        if turn.get("reasoning"):
            target["reasoning_content"] = turn["reasoning"]
        out.append(messages[:i] + [target])
    return out


def sft_trajectories(records: Iterable[dict], teacher: str = TEACHER_EDIT) -> list[Any]:
    """ART trajectories for TrainingRun.warm_start, from the kept records."""
    import art

    trajectories = []
    for record in records:
        for messages in sft_samples(record, teacher):
            trajectories.append(art.Trajectory(messages_and_choices=messages, reward=1.0,
                                               metadata={"task_id": record["task_id"], "family": record["family"]}))
    return trajectories


def summary(records: list[dict]) -> dict:
    """Counts by family, for the notice and the thread: sampled, right, kept, and the sampling cost."""
    out: dict = {}
    for r in records:
        f = out.setdefault(r["family"], {"sampled": 0, "right": 0, "kept": 0, "cost_usd": 0.0})
        f["sampled"] += 1
        f["right"] += bool(r.get("correct"))
        f["kept"] += keep(r)
        f["cost_usd"] = round(f["cost_usd"] + float((r.get("usage") or {}).get("est_cost_usd") or 0.0), 4)
    return out


assert bridge.is_edit("edit_spec") and bridge.is_edit("edit_bugs")
