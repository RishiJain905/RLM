"""AA-LCR, Artificial Analysis's long-context reasoning benchmark, as a post-training benchmark (plan step 13). Eval only.

100 questions, each over a set of real documents (company and industry reports, government consultations, papers,
legal and marketing texts) of ~100K tokens, whose answers have to be worked out from several of them. Artificial
Analysis publishes scores for most open models (Qwen3.8 27B 82%, Gemma 4 31B 70% on v1.1, read directly in one prompt),
so it is the benchmark that puts our checkpoint next to models of its class.

The documents go into `context` in the order `data_source_filenames` gives, each under the research family's header
(`# ===== doc N: <title>`), the format production trained on. The question is kept as written. Answers are free text
(lists, rankings, values with units), so the episode's own text score is only a first pass: scripts/benchmark.py
`judge` grades every answer with Artificial Analysis's v1.1 equality-checker prompts, and that is the reported number.

`load()` returns the 100 questions in one fixed order that spreads each document category evenly, so a run the
spending ceiling stops early still has about the whole set's category mix, and two arms stopped at different points
share their first questions. The order is in the task id (`aa-lcr-<rank>-q<question>`), because baselines.plan_jobs
starts a read arm's episodes in task-id order.
"""

from __future__ import annotations

import csv
import functools
import random
import unicodedata

from .. import data
from ..env import Task
from .research import HEADER

FAMILY = "aa_lcr"
VERSION = "aa-lcr@9a77ef56"  # v1.1 (September 2026), pinned by revision
DIR = "hf/ArtificialAnalysis__AA-LCR"
TEMPLATE = (
    "{question}\n\n`context` holds the documents this question is about, each headed `# ===== doc N: <title>`. "
    "The answer isn't stated in any one place: work it out from what several documents say. Submit only the final "
    "answer, in the form the question asks for."
)


def _rows() -> list[dict]:
    with open(data.path(DIR, "AA-LCR_Dataset.csv"), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _documents(row: dict) -> list[tuple[str, str]]:
    folder = data.path(DIR, "extracted", "lcr", row["document_category"].replace(" ", "_"), row["document_set_id"])
    # the zip stores some names decomposed ("s" + combining cedilla) where the CSV has the composed "ş"
    files = {unicodedata.normalize("NFC", p.name): p for p in folder.iterdir()}
    docs = []
    for name in row["data_source_filenames"].split(";"):
        text = files[unicodedata.normalize("NFC", name)].read_text(encoding="utf-8")
        docs.append((name.rsplit(".", 1)[0], text))
    return docs


def _order(rows: list[dict], seed: int) -> list[dict]:
    """Shuffle within each category, then spread each category evenly over the order (its i-th of n questions sits at
    (i + 0.5) / n), so every prefix keeps about the whole set's category mix (63 of the 100 are company documents)."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: int(r["question_id"])):
        by_cat.setdefault(row["document_category"], []).append(row)
    keyed = []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        rng.shuffle(items)
        keyed += [((i + 0.5) / len(items), cat, row) for i, row in enumerate(items)]
    return [row for _, _, row in sorted(keyed, key=lambda k: (k[0], k[1]))]


@functools.lru_cache(maxsize=None)
def load(seed: int = 0) -> tuple[Task, ...]:
    tasks = []
    for rank, row in enumerate(_order(_rows(), seed)):
        docs = _documents(row)
        context = "\n\n".join(f"{HEADER.format(n=n, title=title)}\n{text}" for n, (title, text) in enumerate(docs, 1))
        answer = row["answer"].strip()
        grading = {"type": "text", "aliases": [], "blind": None, "category": row["document_category"],
                   "question_id": int(row["question_id"]), "docs": len(docs), "input_tokens": int(row["input_tokens"]),
                   "source": VERSION}
        tasks.append(Task(f"aa-lcr-{rank:03d}-q{int(row['question_id']):03d}", FAMILY,
                          TEMPLATE.format(question=row["question"]), context, answer=answer, grading=grading))
    return tuple(tasks)
