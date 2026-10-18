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
# generator family, the ticket tasks, which production never trains on.
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

# Spec v2 §12.10. selection-v2 (64 tasks): 48 read tasks, the read families in the new mix's proportions (25/10/10/15
# of 60) plus the ticket tasks as the held-out generator family, and 16 edit tasks: 6 held-out seeds of edit_synth, one
# per kind and level below, and the 10 edit_real tasks of the selection split (repositories training never sees).
SELECTION_V2_READ = {
    **{("aggregation", level): n for level, n in {1: 4, 2: 4, 3: 3, 4: 3, 5: 3}.items()},
    **{("callgraph", level): n for level, n in {1: 2, 2: 2, 3: 1, 4: 1, 5: 1}.items()},
    **{("needle", level): n for level, n in {1: 1, 2: 2, 3: 2, 4: 1, 5: 1}.items()},
    **{("repo_loc_select", level): n for level, n in {1: 3, 2: 2, 3: 2, 4: 2}.items()},
    **{(family, level): 1 for family in ("tickets_all", "tickets_customer") for level in (1, 2, 3, 5)},
}
SELECTION_V2_SYNTH = [("fix", 1), ("callers", 2), ("feature", 3), ("fix", 4), ("callers", 5), ("feature", 2)]
# selection-v3 (Rishi, 2026-10-17, after step 11): selection-v2 without its 8 ticket tasks, which production never trains
# on and which the base model found too easy (tickets_customer 100% in every arm), and with 8 more held-out seeds of
# the trained read families in their place, at levels 3-5, where the set was thinnest. With them the read tasks follow
# the mix (20/8/8/12 of 48). selection-v2's other 56 tasks are kept as they were (same seeds), so step 11's numbers on
# them carry over; the 8 new ones have seeds of their own. repo_loc_select reads only the selection repositories,
# which training never sees.
SELECTION_V3_EXTRA = {("aggregation", 3): 1, ("aggregation", 4): 1, ("aggregation", 5): 1, ("callgraph", 4): 1,
                      ("needle", 5): 1, ("repo_loc_select", 3): 1, ("repo_loc_select", 4): 2}
SELECTION_V3_DROPPED = ("tickets_all", "tickets_customer")
# selection-v4 (step-12 task shift, documentation/step-12-multistep-plan.md step 5, 2026-10-18): the families built to
# need sub-calls, in the new mix's proportions (research 30 / context search 20 / build from a spec 30 / find and fix
# 20 of 48). research_select reads only MuSiQue's and QAMPARI's dev questions, which training never sees; aggregation
# gets held-out seeds of aggregation-v3; the coding tasks come from their families' select pools (repositories or
# tasks training never sees), spread evenly over the levels. Checkpoint 20 is re-scored on it as the new baseline.
SELECTION_V4_READ = {**{("research_select", level): n for level, n in {1: 3, 2: 3, 3: 3, 4: 3, 5: 2}.items()},
                     **{("aggregation", level): 2 for level in range(1, 6)}}
SELECTION_V4_EDIT = {"edit_spec_select": 14, "edit_bugs_select": 10}
# heldout-test-v2: heldout-test-v1 plus 60 edit tasks, 20 edit_synth_heldout (every kind at every level, plus five
# more) and the 40 edit_real tasks of the test split (held-out repositories, at least 15 of them feature tasks).
HELDOUT_V2_SYNTH = [(kind, level) for level in range(1, 6) for kind in ("fix", "callers", "feature")] + \
                   [("fix", 3), ("callers", 3), ("feature", 4), ("callers", 4), ("feature", 5)]


def _synth_set(name: str, family: str, plan: list[tuple[str, int]]) -> list[Task]:
    from .tasks import editsynth

    seeds = task_sets.private_seed_list(name, len(plan))
    return [editsynth.make_task(seed, family, level, f"{name}-{i:02d}", kind=kind)
            for i, (seed, (kind, level)) in enumerate(zip(seeds, plan))]


def _spread(tasks: list[Task], n: int) -> list[Task]:
    """n tasks spread evenly over a list sorted by level, so every level is represented."""
    tasks = sorted(tasks, key=lambda t: (t.level, t.task_id))
    if len(tasks) <= n:
        return tasks
    return [tasks[round(i * (len(tasks) - 1) / (n - 1))] for i in range(n)] if n > 1 else tasks[:1]


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
    if name == "selection-v2":
        from .tasks import editreal

        read = task_sets.fixed_set(name, SELECTION_V2_READ)
        synth = _synth_set(f"{name}-edit_synth", "edit_synth", SELECTION_V2_SYNTH)
        return read + synth + editreal.eval_tasks("select", "edit_real_select"), 1
    if name == "selection-v3":
        v2, _ = eval_set("selection-v2")  # the same 56 kept tasks, same seeds and task ids
        kept = [t for t in v2 if t.family not in SELECTION_V3_DROPPED]
        read = [t for t in kept if not bridge.is_edit(t.family)]
        extra = task_sets.fixed_set(f"{name}-extra", SELECTION_V3_EXTRA)
        return read + extra + [t for t in kept if bridge.is_edit(t.family)], 1
    if name == "selection-v4":
        read = task_sets.fixed_set(name, SELECTION_V4_READ)
        edit = [t for family, n in SELECTION_V4_EDIT.items()
                for t in _spread(task_sets.generator(family).eval_tasks("select", family), n)]
        return read + edit, 1
    if name == "heldout-test-v2":
        from .tasks import editreal

        v1, _ = eval_set("heldout-test-v1")
        synth = _synth_set(f"{name}-edit_synth_heldout", "edit_synth_heldout", HELDOUT_V2_SYNTH)
        return v1 + synth + editreal.eval_tasks("test", "edit_real_test"), 1
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
    families, levels, languages = (collections.defaultdict(list) for _ in range(3))
    hacks = dict.fromkeys(detectors.RULES, 0)
    for r in clean:
        families[r["family"]].append(float(bool(r["correct"])))
        levels[f"{r['family']}/L{r['level']}"].append(float(bool(r["correct"])))
        if r.get("language") and str(r["family"]).startswith(bridge.EDIT_PREFIX):
            languages[str(r["language"]).lower()].append(float(bool(r["correct"])))
        for hit in list(r.get("hacks", ())) + list(r.get("flags") or ()):
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
        "by_language": {k: {"n": len(v), "accuracy": _mean(v)} for k, v in sorted(languages.items())},
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
                  "language": record.get("language") or "",
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
