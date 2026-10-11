"""OOLONG-synth's own test split, as held-out test tasks (plan step 7, H7). Eval only: it fails the licence bar.

OOLONG-synth is the benchmark our aggregation generator imitates. Its test split uses 8 source sets that the
generator never sees (metaphors, negation, formality, IMDB, app reviews, Yahoo Answers, MultiNLI, AG News) and
its own line format and question wording, so it is the held-out test of the aggregation skill.

`load()` picks a fixed, stratified sample by context length bucket and source set. Questions are kept as
OOLONG wrote them, except that the sentence asking for a "Label: answer" style reply is replaced by one
asking for the bare answer, because the answer goes through `submit()` and is scored by exact match.
Date and month answers are left out (137 of 5,200 items) because their expected format isn't stated.
"""

from __future__ import annotations

import ast
import functools
import random
import re

from .. import data
from ..env import Task

FAMILY = "oolong_synth_test"
GENERATOR_VERSION = "oolong-synth-test@f0d59eaf"  # a fixed dataset, pinned by revision
BUCKETS = (16_384, 32_768, 65_536, 131_072, 262_144)  # tokens; ~65K to ~1M characters
_FORM_RE = re.compile(r"\s*Give your final answer in the form .*$", re.S)
_REPLY = {"NUMERIC": "Answer with a single integer.", "LABEL": "Answer with the label only.",
          "USER": "Answer with the user ID only.",
          "COMPARISON": "Answer with exactly one of: more common than, less common than, same frequency as."}


def _files():
    return sorted(data.path("hf/oolongbench__oolong-synth/data").glob("test-*.parquet"))


@functools.lru_cache(maxsize=None)
def _index() -> tuple[dict, ...]:
    import pyarrow.parquet as pq

    out = []
    for f in _files():
        for row in pq.read_table(f, columns=["id", "context_len", "dataset", "task", "answer_type"]).to_pylist():
            out.append({**row, "file": f.name})
    return tuple(out)


def load(per_bucket: int = 10, buckets=BUCKETS, seed: int = 0) -> list[Task]:
    """A fixed sample: `per_bucket` items per length bucket, spread over the 8 source sets."""
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    usable = [r for r in _index() if r["context_len"] in buckets and r["answer_type"].split(".")[-1] in _REPLY]
    chosen = []
    for bucket in buckets:
        rows = sorted((r for r in usable if r["context_len"] == bucket), key=lambda r: r["id"])
        by_set: dict[str, list[dict]] = {}
        for r in rows:
            by_set.setdefault(r["dataset"], []).append(r)
        names = sorted(by_set)
        rng.shuffle(names)
        for i in range(per_bucket):
            chosen.append((bucket, rng.choice(by_set[names[i % len(names)]])))
    tasks = []
    for bucket, pick in chosen:
        table = pq.read_table(data.path("hf/oolongbench__oolong-synth/data", pick["file"]),
                              columns=["id", "context_window_text", "question", "answer", "answer_type", "num_labels"],
                              filters=[("id", "=", pick["id"])])
        row = table.to_pylist()[0]
        kind = row["answer_type"].split(".")[-1]
        answer = ast.literal_eval(row["answer"])[0]
        question = _FORM_RE.sub("", row["question"]).strip() + " " + _REPLY[kind]
        n_lines = row["context_window_text"].count("\n") + 1
        if kind == "NUMERIC":
            grading = {"type": "int", "blind": round(n_lines / max(1, row["num_labels"]))}
            answer = int(answer)
        else:
            grading = {"type": "entity"}
            answer = str(answer)
        grading.update(kind=kind.lower(), dataset=pick["dataset"], oolong_id=str(pick["id"]))
        tasks.append(Task(f"{FAMILY}-{bucket // 1024}k-{pick['id']}", FAMILY, question, row["context_window_text"],
                          answer=answer, level=BUCKETS.index(bucket) + 1, grading=grading))
    return tasks
