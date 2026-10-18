"""Research tasks: multi-hop and list questions over a large corpus of Wikipedia text (step 12 multistep plan, step 1).

A context is a shuffled corpus of documents, each headed `# ===== doc 137: <title>`. It is far bigger than the root
can print (12-20 turns x 8,000 characters), and keyword hits are mostly look-alike distractors, so the root has to
read: hop by hop, or by handing chunks to sub-calls.

Two kinds of task, from two sources:
- multi-hop (~70%), from MuSiQue-Ans (CC BY 4.0). The question's own 20 paragraphs (its supporting paragraphs and the
  dataset's lexically similar distractors) are packed among paragraphs of other questions until the level's size.
  Part of the padding is drawn from paragraphs that share a word with the question, so a keyword search hits many
  documents. The hop-2 entity only appears after hop 1 is read. Graded as free text ("text"): exact match against
  the answer or an alias, else 0.8 x token-F1.
- list (~30%), from QAMPARI's single-relation questions (CC0; Wikipedia text). One document per Wikipedia page, built
  from the dataset's proof sentences: one per answer, plus pages that name the question's entity in another role
  (hard negatives), pages that share the question's other words, and random pages. The answer is the list of the
  answer documents' titles, graded by set-F1 against a blind guess of [] ("list").

The build (scripts/build_research_data.py) cleans QAMPARI: an answer whose proof sentence doesn't name the question's
entity can't be checked by reading, so it leaves both the answer and the corpus; a page that names the entity but
reads like an answer is never a distractor (Wikidata's lists miss answers); intersection and language questions are
left out. Padding never names the entity, so a search for it finds exactly the answers and the hard negatives: a
search alone earns set-F1 of about 0.3-0.6, and reading earns the rest.

Families:
- `research`: training, on MuSiQue's and QAMPARI's train splits.
- `research_select`: the selection split: half of MuSiQue's dev questions (whole groups of questions that share a
  sub-question) and QAMPARI's dev split. Never trained on.
- `research_test`: held-out test only (H7): the other half of MuSiQue dev and QAMPARI's test split, with separately
  written question templates.
Each split pads only with its own split's text, and no question, MuSiQue sub-question or QAMPARI entity is shared
across splits.

Levels: 2-hop at ~75K characters up to 4-hop at ~700K; list questions grow from 5-6 answers to 10-15 over the same
sizes, with more hard negatives.
"""

from __future__ import annotations

import functools
import math
import random
import re
import secrets

from .. import data
from ..env import Task
from ..reward import normalize_text

GENERATOR_VERSION = "research-v1"
FAMILIES = ("research", "research_select", "research_test")
SPLIT = {"research": "train", "research_select": "select", "research_test": "test"}
TRAIN_SOURCES = ("musique", "qampari")

LEVELS = {1: (60_000, 90_000), 2: (120_000, 180_000), 3: (220_000, 300_000), 4: (380_000, 480_000),
          5: (600_000, 800_000)}
MAX_LEVEL = max(LEVELS)
HOPS = {1: (2,), 2: (2, 3), 3: (3,), 4: (3, 4), 5: (4,)}
LIST_ANSWERS = {1: (5, 6), 2: (5, 8), 3: (6, 10), 4: (8, 12), 5: (10, 15)}
LIST_SHARE = 0.3
# The share of a context's padding drawn from documents that share a word with the question *(mine)*.
NEAR_SHARE = {1: 0.25, 2: 0.3, 3: 0.35, 4: 0.4, 5: 0.45}
# Hard negatives. A gold document always names the question's entity, and padding never does, so a search for the
# entity finds the answers plus the hard negatives: a list task needs at least NEG_RATIO x its answer count of them
# (fewer only when a small eval pool has none such), and holds up to NEG_CAP *(mine)*.
NEG_RATIO = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.0, 5: 1.0}
NEG_CAP = {1: 15, 2: 25, 3: 40, 4: 60, 5: 80}
PAGE_MAX_CHARS = 1_000  # a QAMPARI page document keeps its needed sentences, then others up to this

HEADER = "# ===== doc {n}: {title}"
HEADER_RE = re.compile(r"^# ===== doc (\d+): (.*)$", re.M)
# Sizes count every header as a 4-digit doc number, so a context is never over its level's top; the shorter numbers
# of the first 999 documents save up to 1,107 characters, which this margin over the bottom covers.
HEADER_SLACK = 1_200

TRAIN_HOP = (
    "Answer this question from the documents in `context`:\n\n{question}\n\n`context` is a large collection of "
    "Wikipedia paragraphs, each headed `# ===== doc N: <title>`. The facts you need are spread over several "
    "documents, each step's answer leading to the next, and many other documents look related but aren't. Answer "
    "with the final answer only: a name, place, date or number, written as in the documents.",
    "{question}\n\n`context` holds a large corpus of Wikipedia paragraphs, each starting with a line "
    "`# ===== doc N: <title>`. Work the question out step by step from those paragraphs. Reply with just the final "
    "answer, in a few words, as the documents write it.",
)
TRAIN_LIST = (
    "{question}\n\n`context` is a large collection of short Wikipedia extracts, each headed "
    "`# ===== doc N: <title>`. Several documents are about an answer to the question; many others mention the same "
    "names but aren't. Answer with a list of the titles of every document that is about an answer, copied exactly "
    "from their headers.",
    "Question: {question}\n\nEach document in `context` starts with a line `# ===== doc N: <title>` and describes "
    "the subject in its title. Find every document whose subject answers the question (there is more than one). "
    "Answer with a list of those documents' titles, written exactly as in the headers.",
)
HELDOUT_HOP = (
    "{question}\nThe sections of the reference corpus in `context` (each marked `# ===== doc N: <title>`) hold the "
    "facts this needs, among many unrelated or misleading sections. Give only the final answer, in a few words.",
)
HELDOUT_LIST = (
    "{question}\nThe reference corpus in `context` has sections marked `# ===== doc N: <title>`. Name all the "
    "sections whose subject is a correct answer. Reply with the section titles exactly as they appear, as a list.",
)

STOPWORDS = frozenset("""a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each etc few for from further had has have
having he her here hers him his how i if in into is it its itself just me more most my no nor not now of off on once
one ones only or other our out over own same she should so some such than that the their theirs them then there these
they this those through to too under until up very was we were what when where which while who whom whose why will
with would you your yours name named list many much kind type""".split())


def training_sources() -> tuple[str, ...]:
    """The data sources the training family reads (rlm_rl/data.py SOURCES), for the licence bar (7b)."""
    return TRAIN_SOURCES


def content_words(text: str) -> list[str]:
    """A question's content words: lowercase, 3+ characters, not stopwords, in order of first use."""
    words = [w for w in re.findall(r"[^\W_]+", text.lower()) if len(w) >= 3 and w not in STOPWORDS]
    return list(dict.fromkeys(words))


def _capital_terms(question: str) -> list[str]:
    """Capitalized words after the first, 4+ characters: the question's names, for keeping same-name titles out."""
    words = re.findall(r"[^\W_]+", question)[1:]
    return sorted({w.lower() for w in words if w[0].isupper() and len(w) >= 4 and w.lower() not in STOPWORDS})


def _word_re(words: list[str]) -> re.Pattern | None:
    """Any of `words` as a whole word, ignoring case."""
    if not words:
        return None
    return re.compile(r"(?<![^\W_])(?:" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) +
                      r")(?![^\W_])", re.IGNORECASE)


def entity_re(names: list[str]) -> re.Pattern | None:
    """Any of `names` as a run of whole words, whatever the punctuation between them ("Toodyay, W.A." matches
    "toodyay w a"): how scripts/build_research_data.py finds the pages that name an entity."""
    runs = sorted({tuple(re.findall(r"[^\W_]+", n.lower())) for n in names} - {()}, key=len, reverse=True)
    if not runs:
        return None
    return re.compile(r"(?<![^\W_])(?:" + "|".join(r"[\W_]+".join(map(re.escape, run)) for run in runs) +
                      r")(?![^\W_])", re.IGNORECASE)


# ----------------------------------------------------------------------------- pools


def _split(family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"unknown research family {family!r}")
    return SPLIT[family]


@functools.lru_cache(maxsize=None)
def _musique(split: str) -> dict:
    return data.research_pool("musique")["splits"][split]


@functools.lru_cache(maxsize=None)
def _qampari(split: str) -> dict:
    return data.research_pool("qampari")["splits"][split]


@functools.lru_cache(maxsize=None)
def _by_hops(split: str) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, q in enumerate(_musique(split)["questions"]):
        out.setdefault(int(q["hops"]), []).append(i)
    return out


@functools.lru_cache(maxsize=None)
def _shared_titles(split: str) -> frozenset[int]:
    """QAMPARI pages whose title equals another's but for case: never padding, so a title names one document."""
    seen: dict[str, list[int]] = {}
    for pid, (title, _) in enumerate(_qampari(split)["pages"]):
        seen.setdefault(title.casefold(), []).append(pid)
    return frozenset(pid for pids in seen.values() if len(pids) > 1 for pid in pids)


@functools.lru_cache(maxsize=None)
def list_pool(split: str, level: int) -> tuple[int, ...]:
    """The QAMPARI questions a level draws from: the level's answer count and enough hard negatives. A small eval
    pool without such questions relaxes the negatives (to half, then none), then the answer count (nearest)."""
    questions = _qampari(split)["questions"]
    lo, hi = LIST_ANSWERS[level]
    for ratio in (NEG_RATIO[level], NEG_RATIO[level] / 2, 0.0):
        pool = tuple(i for i, q in enumerate(questions)
                     if lo <= len(q["gold"]) <= hi and len(q["neg"]) >= math.ceil(ratio * len(q["gold"])))
        if pool:
            return pool
    best = min(abs(len(q["gold"]) - lo) + abs(len(q["gold"]) - hi) for q in questions)
    return tuple(i for i, q in enumerate(questions) if abs(len(q["gold"]) - lo) + abs(len(q["gold"]) - hi) == best)


def _doc_len(title: str, text: str) -> int:
    return len(HEADER.format(n=1000, title=title)) + len(text) + 2  # the header, the text, a newline, a blank line


def _render(docs: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"{HEADER.format(n=n, title=title)}\n{text}" for n, (title, text) in enumerate(docs, 1))


def _fill(rng: random.Random, chosen: list, size: int, budget: int, hi: int, candidates, near_re, near_budget: float,
          doc_of, skip) -> tuple[list, int]:
    """Pad `chosen` with candidate ids up to `budget` characters (never past `hi`): first ones whose text matches
    `near_re`, up to `near_budget` characters, then any. `doc_of(i)` gives (title, text); `skip(i, title, text)`
    leaves a candidate out."""
    order = list(candidates)
    rng.shuffle(order)
    taken = set(chosen)
    near = 0
    for pass_near in (True, False):
        if pass_near and near_re is None:
            continue
        for i in order:
            if size >= budget or (pass_near and near >= near_budget):
                break
            if i in taken:
                continue
            title, text = doc_of(i)
            if skip(i, title, text):
                taken.add(i)  # never a candidate again
                continue
            if pass_near and not near_re.search(text):
                continue
            n = _doc_len(title, text)
            if size + n > hi:
                continue
            chosen.append(i)
            taken.add(i)
            size += n
            if pass_near:
                near += n
    return chosen, size


# ----------------------------------------------------------------------------- multi-hop (MuSiQue)


def _hop_task(rng: random.Random, split: str, level: int, budget: int) -> tuple[str, str, dict, list, list[int]]:
    pool = _musique(split)
    paras, pad_ok = pool["paragraphs"], pool["pad_ok"]
    by_hops = _by_hops(split)
    hops = rng.choice([h for h in HOPS[level] if h in by_hops] or sorted(by_hops))
    q = pool["questions"][rng.choice(by_hops[hops])]
    own = list(dict.fromkeys(q["paras"]))
    own_titles = {paras[i][0] for i in own}
    names = _capital_terms(q["q"])
    name_re = _word_re(names)
    size = sum(_doc_len(*paras[i]) for i in own)

    def skip(i, title, text):
        return not pad_ok[i] or title in own_titles or (name_re is not None and name_re.search(title) is not None)

    near_budget = NEAR_SHARE[level] * (budget - size)
    chosen, _ = _fill(rng, list(own), size, budget, LEVELS[level][1], range(len(paras)),
                      _word_re(content_words(q["q"])), near_budget, lambda i: paras[i], skip)
    rng.shuffle(chosen)
    docs = [(paras[i][0], paras[i][1]) for i in chosen]
    gold = set(q["gold"])
    gold_docs = sorted(n for n, i in enumerate(chosen, 1) if i in gold)
    return q["q"], q["a"], {"aliases": list(q["aliases"]), "hops": hops}, docs, gold_docs


# ----------------------------------------------------------------------------- list (QAMPARI)


def _page_text(page: list, must=()) -> str:
    """A page's document: the sentences it must keep, then others in order up to PAGE_MAX_CHARS."""
    sentences = page[1]
    keep = set(must)
    total = sum(len(sentences[i]) + 1 for i in keep)
    for i, s in enumerate(sentences):
        if i not in keep and total + len(s) + 1 <= PAGE_MAX_CHARS:
            keep.add(i)
            total += len(s) + 1
    return " ".join(sentences[i] for i in sorted(keep))


def _list_task(rng: random.Random, split: str, level: int, budget: int) -> tuple[str, list, dict, list, list[int]]:
    pool = _qampari(split)
    pages = pool["pages"]
    q = pool["questions"][rng.choice(list_pool(split, level))]
    gold = {int(pid): sents for pid, sents in q["gold"]}
    negs = [(int(pid), sents) for pid, sents in q["neg"]]
    negs = rng.sample(negs, min(len(negs), NEG_CAP[level]))
    names = entity_re(q["entities"])
    excluded = set(q["exclude"]) | set(gold) | {pid for pid, _ in negs} | _shared_titles(split)
    texts = {pid: _page_text(pages[pid], sents) for pid, sents in list(gold.items()) + negs}
    chosen = list(texts)
    size = sum(_doc_len(pages[pid][0], texts[pid]) for pid in chosen)

    def doc_of(i):
        if i not in texts:
            texts[i] = _page_text(pages[i])
        return pages[i][0], texts[i]

    def skip(i, title, text):  # padding never names the question's entity: such a page could be an answer
        return i in excluded or (names is not None and names.search(f"{title} {text}") is not None)

    entity_words = {w for e in q["entities"] for w in content_words(e)}
    near_words = [w for w in content_words(q["q"]) if w not in entity_words]
    near_budget = NEAR_SHARE[level] * (budget - size)
    chosen, _ = _fill(rng, chosen, size, budget, LEVELS[level][1], range(len(pages)), _word_re(near_words),
                      near_budget, doc_of, skip)
    rng.shuffle(chosen)
    docs = [(pages[i][0], texts[i]) for i in chosen]
    gold_docs = sorted(n for n, i in enumerate(chosen, 1) if i in gold)
    answer = sorted(pages[pid][0] for pid in gold)
    return q["q"], answer, {"answers": len(answer), "negatives": len(negs), "qtype": q["type"]}, docs, gold_docs


# ----------------------------------------------------------------------------- tasks


def _kind(rng: random.Random) -> str:
    return "list" if rng.random() < LIST_SHARE else "multihop"


def build(seed: int, family: str, level: int = 1, task_id: str | None = None) -> tuple[Task, dict]:
    """A task and, for tests and audits only, which documents are gold. The Task carries no source id, gold flag or
    answer outside `answer` and `grading`, which stay on the host (H3)."""
    split = _split(family)
    if level not in LEVELS:
        raise ValueError(f"no research level {level} (levels 1-{MAX_LEVEL})")
    task_id = task_id or f"{family}-L{level}-{secrets.token_hex(4)}"
    rng = random.Random(seed)
    kind = _kind(rng)
    lo, hi = LEVELS[level]
    budget = rng.randint(lo + HEADER_SLACK, hi)
    heldout = family == "research_test"
    if kind == "multihop":
        question, answer, extra, docs, gold_docs = _hop_task(rng, split, level, budget)
        template = rng.choice(HELDOUT_HOP if heldout else TRAIN_HOP)
        # "ignore": the question's own words, which earn no partial credit (reward.score_text)
        grading = {"type": "text", "aliases": extra["aliases"], "blind": None,
                   "ignore": sorted(set((normalize_text(question) or "").split())), "hops": extra["hops"],
                   "source": "musique"}
    else:
        question, answer, extra, docs, gold_docs = _list_task(rng, split, level, budget)
        template = rng.choice(HELDOUT_LIST if heldout else TRAIN_LIST)
        grading = {"type": "list", "item": "entity", "blind": [], "answers": extra["answers"],
                   "negatives": extra["negatives"], "source": "qampari"}
    grading.update(kind=kind, code_only=False, docs=len(docs))
    task = Task(task_id, family, template.format(question=question), _render(docs), answer=answer, level=level,
                grading=grading)
    return task, {"gold_docs": gold_docs, "question": question, "kind": kind, **extra}


def make_task(seed: int, family: str, level: int = 1, task_id: str | None = None) -> Task:
    """One task from a seed. The task id never contains the seed (H3)."""
    return build(seed, family, level, task_id)[0]
