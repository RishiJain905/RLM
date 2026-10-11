"""Evals with every episode logged in full (H9, research §8.4).

run_eval() runs a fixed task list against one checkpoint and appends one transcript line per
episode to results/<run>/transcripts/val/<name>/step<NNNN>.jsonl as each one finishes.
Rerunning the same eval at the same checkpoint resumes it: finished episodes are skipped, infra
episodes run again, and the reader keeps the newest line per episode, so the eval ends with
exactly one transcript per episode. Infra episodes are left out of accuracy and counted.

When an ART model is given, the whole eval (not just this call's episodes) is also logged with
`model.log(groups, split="val", step=step)`, so a resumed eval logs complete groups. That writes
.art/.../trajectories/val/<step>.parquet locally and the summary metrics to W&B. Logging
metrics is free; it spends no inference credit.
"""

from __future__ import annotations

import asyncio
import collections
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import bridge, detectors, reward, tasks as task_sets
from .bridge import InfraError
from .client import ModelClient
from .env import Scorer, Task, run_episode
from .sandbox import Sandbox
from .transcripts import TranscriptLog, episode_record, transcript_path

SYSTEM_PROMPT = bridge.SYSTEM_PROMPT  # tests/test_contract.py diffs this against the bridge (H10)

MAX_EPISODES_AT_ONCE = 32  # sandboxes running at once; the limiter paces the requests themselves

# The selection set until plan step 7 builds the production one: 64 ticket tasks, 32 per family, across all 5
# levels at fixed seeds (research §8.4). It stands in so the loop's eval, best-checkpoint and stop-rule code runs.
SELECTION_TICKETS_V1 = {(family, level): count for family in ("tickets_all", "tickets_customer")
                        for level, count in {1: 7, 2: 7, 3: 6, 4: 6, 5: 6}.items()}


@dataclass
class EvalResult:
    name: str
    step: int
    path: Path
    summary: dict
    ran: int  # episodes this call ran; the rest were already logged
    records: list[dict]


# The production selection set (plan step 7, research §8.1): 64 fixed tasks that pick the checkpoint. Held-out
# seeds of every training family at fixed levels, the selection repos LCA training never sees, and one held-out
# generator family, the ticket tasks, which production never trains on. The cyber share isn't built yet; when it
# is, selection-v2 gives it its ~35% (~22 tasks).
SELECTION_V1 = {
    **{("aggregation", level): 4 for level in range(1, 6)},
    **{("callgraph", level): 2 for level in range(1, 6)},
    **{("needle", level): 2 for level in range(1, 6)},
    **{("repo_loc_select", level): 3 for level in range(1, 5)},
    **{(family, level): 2 for family in ("tickets_all", "tickets_customer") for level in (1, 3, 5)},
}

# The held-out test (research §8.1): touched once, in plan step 13. Held-out sources, behaviours, call styles and
# question templates, LCA's Python test split (each item once) and a fixed sample of OOLONG-synth's test split.
HELDOUT_TEST_V1 = {
    **{("aggregation_heldout", level): 8 for level in range(1, 6)},
    **{("callgraph_heldout", level): 4 for level in range(1, 6)},
    **{("needle_heldout", level): 4 for level in range(1, 6)},
}
HELDOUT_OOLONG_PER_BUCKET = 10


def eval_set(name: str) -> tuple[list[Task], int]:
    """An eval set the training loop runs, by name: (tasks, samples per task)."""
    if name == "heldout-seed1":  # the first seed's 50 held-out tasks, 2 samples each (plan step 10)
        from .tasks.tickets import legacy_tasks

        return legacy_tasks(50, 99_999), 2
    if name == "selection-tickets-v1":
        return task_sets.fixed_set(name, SELECTION_TICKETS_V1), 1
    if name == "selection-v1":
        return task_sets.fixed_set(name, SELECTION_V1), 1
    if name == "heldout-test-v1":
        from .tasks import oolong, repoloc

        generated = task_sets.fixed_set(name, HELDOUT_TEST_V1)
        lca = repoloc.test_tasks(task_sets.private_seed_list(f"{name}-lca", len(repoloc.items("test"))), name)
        return generated + lca + oolong.load(HELDOUT_OOLONG_PER_BUCKET), 1
    raise KeyError(f"no eval set named {name!r}")


def episode_key(name: str, step: int, task_id: str, sample: int) -> str:
    return f"{name}/step{step}/{task_id}/{sample}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[dict]) -> dict:
    """Accuracy and episode shape over finished episodes; infra episodes are counted separately."""
    clean = [r for r in records if not r.get("infra_error")]
    infra = [r for r in records if r.get("infra_error")]
    n = len(clean)
    accuracy = _mean([float(bool(r["correct"])) for r in clean])
    families, levels = collections.defaultdict(list), collections.defaultdict(list)
    hacks = dict.fromkeys(detectors.RULES, 0)
    for r in clean:
        families[r["family"]].append(float(bool(r["correct"])))
        levels[f"{r['family']}/L{r['level']}"].append(float(bool(r["correct"])))
        for hit in r.get("hacks", ()):
            hacks[hit["rule"]] = hacks.get(hit["rule"], 0) + 1

    def metric(name):
        return _mean([r["metrics"][name] for r in clean])

    return {
        "episodes": len(records),
        "n": n,
        "accuracy": accuracy,
        "accuracy_se": math.sqrt(accuracy * (1 - accuracy) / n) if n else None,
        "mean_reward": _mean([r["reward"] for r in clean]),
        "submitted": metric("submitted"),
        "first_turn_submit": metric("first_turn_submit"),
        "turns": metric("turns"),
        "subcalls": metric("subcalls"),
        "budget_hit": metric("budget_hit"),
        "truncated_turns": metric("truncated_turns"),
        "escape_events": metric("escape_events"),
        "hacked": sum(bool(r.get("hacks")) for r in clean),
        "hacks": hacks,
        "infra_episodes": len(infra),
        "infra_kinds": dict(collections.Counter(r.get("infra_kind") for r in infra)),
        "est_cost_usd": round(sum(r["usage"]["est_cost_usd"] for r in records), 4),
        "by_family": {f: {"n": len(v), "accuracy": _mean(v)} for f, v in sorted(families.items())},
        "by_level": {k: {"n": len(v), "accuracy": _mean(v)} for k, v in sorted(levels.items())},
    }


async def run_eval(tasks: list[Task], *, name: str, step: int, run_dir: str | Path, run: str, client: ModelClient,
                   root_model: str, scorer: Scorer = reward.score, samples: int = 1, art_model: Any = None,
                   max_episodes: int = MAX_EPISODES_AT_ONCE, sandbox_factory: Callable[..., Any] = Sandbox,
                   **sandbox_kwargs) -> EvalResult:
    """Run (or resume) one eval of checkpoint `step` and log every episode."""
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique within an eval")
    path = transcript_path(run_dir, "val", step, name)
    log = TranscriptLog(path)
    done = log.done_keys()
    planned = {episode_key(name, step, t.task_id, s): (t, s) for t in tasks for s in range(samples)}
    todo = [(key, task, sample) for key, (task, sample) in planned.items() if key not in done]
    gate = asyncio.Semaphore(max_episodes)

    async def one(key: str, task: Task, sample: int) -> None:
        async with gate:
            ep = await run_episode(task, client=client, root_model=root_model, scorer=scorer, run_id=run,
                                   sandbox_factory=sandbox_factory, **sandbox_kwargs)
        log.append(episode_record(ep, split="val", run=run, step=step, key=key, eval_name=name, sample=sample))

    await asyncio.gather(*(one(*job) for job in todo))
    records = [r for r in log.records() if r["key"] in planned]
    summary = summarize(records)
    if art_model is not None:
        await log_to_art(art_model, records, step=step, name=name, summary=summary)
    return EvalResult(name, step, path, summary, len(todo), records)


def record_trajectory(record: dict) -> Any:
    """An ART trajectory for logging, rebuilt from a transcript record (plain messages, no token ids)."""
    import art

    return art.Trajectory(
        messages_and_choices=record["messages"],
        reward=float(record["reward"]),
        metrics=record["metrics"],
        metadata={"task_id": record["task_id"], "family": record["family"], "level": record["level"],
                  "episode_id": record["episode_id"], "eval": record["eval"] or "", "gold": str(record["gold"]),
                  "answer": "" if record["answer"] is None else str(record["answer"]),
                  "hacks": ",".join(hit["rule"] for hit in record.get("hacks", ()))},
    )


async def log_to_art(art_model: Any, records: list[dict], *, step: int, name: str, summary: dict) -> None:
    """Log a finished eval through ART: one group per task, infra episodes as the group's exceptions."""
    import art

    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)
    groups = [art.TrajectoryGroup([record_trajectory(r) for r in rs if not r.get("infra_error")],
                                  exceptions=[InfraError(r["infra_error"]) for r in rs if r.get("infra_error")])
              for rs in by_task.values()]
    scalars = {f"{name}/{k}": float(v) for k, v in summary.items() if isinstance(v, (int, float)) and v is not None}
    await art_model.log(groups, split="val", step=step, metrics=scalars)
