"""Download and build the research family's data: MuSiQue and QAMPARI (step 12 multistep plan, step 1).

    python -I scripts/build_research_data.py download   # fetch the raw files into new, empty folders; check sha256
    python -I scripts/build_research_data.py musique    # derived/research/musique-v1.json.gz
    python -I scripts/build_research_data.py qampari    # derived/research/qampari-v1.json.gz
    python -I scripts/build_research_data.py all       # download (skipping files already there), then both

The downloads are untrusted data. This script reads them only as JSON lines (and the QAMPARI zip only through
`zipfile`, never extracting it), runs under `python -I`, and writes only under the data root (rlm_rl/data.py).
Nothing here calls a model or spends credit.

Splits. `research` trains on each dataset's train split. MuSiQue's dev split is cut in two by whole groups of
questions that share a sub-question: one half for `research_select`, the other for `research_test`. QAMPARI's dev
split is `research_select` and its test split `research_test`. No MuSiQue sub-question is shared across splits
(checked here), no QAMPARI entity or near-identical answer set is (dropped here), and an earlier split never pads
a context with a later split's answer text (`pad_ok`, and the page pools).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import data  # noqa: E402
from rlm_rl.reward import normalize_text  # noqa: E402
from rlm_rl.tasks import research  # noqa: E402

MQ_DIR = "hf/dgslibisey__MuSiQue"
MQ_REV = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
QP_DIR = "github/samsam3232__qampari"
# (path under the data root, URL, sha256). The two text files are records of the licences.
DOWNLOADS = (
    (f"{MQ_DIR}/musique_ans_v1.0_train.jsonl",
     f"https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/{MQ_REV}/musique_ans_v1.0_train.jsonl",
     "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"),
    (f"{MQ_DIR}/musique_ans_v1.0_dev.jsonl",
     f"https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/{MQ_REV}/musique_ans_v1.0_dev.jsonl",
     "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"),
    (f"{MQ_DIR}/UPSTREAM_README.md", "https://raw.githubusercontent.com/StonyBrookNLP/musique/main/README.md",
     "7a3fb6a960f3fa122d2a1a81b73b9cf5a39ac983b11735d158490947960b9359"),
    (f"{QP_DIR}/qampari.zip", "https://aggreg-qa.s3.amazonaws.com/qampari.zip",
     "f061d10879aa2ef645db680bfc699fe6e756e922e48dc6c6a98c4979e091c1c8"),
    (f"{QP_DIR}/LICENSE", "https://raw.githubusercontent.com/samsam3232/qampari/master/LICENSE",
     "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"),
)

MUSIQUE_OUT = data.RESEARCH_POOLS["musique"]
QAMPARI_OUT = data.RESEARCH_POOLS["qampari"]
SPLIT_SALT = "research-split-v1"

# Only single-relation questions, whose answer is the page its proof is on. An intersection question ("films Mani
# Ratnam both wrote and produced") needs both facts, but its one-sentence proofs rarely state both, so the answer
# can't be checked by reading; and when they do, searching for both names finds exactly the answers.
QAMPARI_TYPES = ("wikidata_simple",)
MIN_ANSWERS, MAX_ANSWERS = 5, 15
MIN_PROVEN = 0.6  # at least this share of a question's answers must be checkable by reading
MAX_TITLE = 100
SENTENCE_CHARS = (20, 800)
NEG_STORE = 100  # hard negatives kept per question
TRAIN_FILLER_PAGES = 40_000  # other train pages kept for padding
JACCARD_DUP = 0.5
# A page that names the question's entity is a hard negative only if it looks like a different relation: its naming
# sentence uses fewer than two of the question's other words (crudely stemmed; all of them if there's only one), and
# it isn't written like an answer's proof (under TEMPLATE_SAME of its words, less titles, entity names and
# stopwords, appear in one). QAMPARI's answer lists come from Wikidata and miss answers, so a page that fails either
# test may well be a missing answer: it's left out of that question's contexts altogether.
# Language questions ("who spoke Burmese as a native language") are proven by nationality ("a Burmese politician"),
# so every other Burmese person reads like an answer: they're dropped.
LANGUAGE_RE = re.compile(r"\b(?:native|mother tongue|speaks?|spoke|speakers?|language)\b", re.I)
TEMPLATE_SAME = 0.6
STEMS = {"wrote": "writ", "written": "writ", "writes": "writ", "writer": "writ", "writing": "writ", "died": "die",
         "dies": "die", "death": "die", "born": "born", "birth": "born", "won": "win", "wins": "win", "winner": "win"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def clean(text: str) -> str:
    return " ".join(str(text).split())


def write_json_gz(rel: str, obj) -> Path:
    out = data.derived(rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(out)
    return out


# ----------------------------------------------------------------------------- download


def download(args) -> None:
    """Each raw file into its folder, which must be new or hold only files of this list. Hashes are checked."""
    expected = collections.defaultdict(set)
    for rel, _, _ in DOWNLOADS:
        expected[str(Path(rel).parent)].add(Path(rel).name)
    for folder, names in expected.items():
        path = data.path(folder)
        stray = sorted(p.name for p in path.iterdir() if p.name not in names) if path.exists() else []
        if stray:
            raise SystemExit(f"{path} holds files this script didn't fetch ({stray}); use a new, empty folder")
    for rel, url, digest in DOWNLOADS:
        dest = data.path(rel)
        if dest.exists():
            if sha256(dest) != digest:
                raise SystemExit(f"{dest} exists with a different sha256; move it aside and fetch again")
            log(f"have {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        log(f"fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        got = sha256(part)
        if got != digest:
            part.unlink()
            raise SystemExit(f"{url}: sha256 {got}, expected {digest}")
        part.replace(dest)
        log(f"  {dest.stat().st_size / 1e6:.1f} MB, sha256 ok")


# ----------------------------------------------------------------------------- MuSiQue


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _subq(row: dict) -> list[int]:
    return sorted(int(d["id"]) for d in row["question_decomposition"])


def musique_dev_split(rows: list[dict]) -> dict[str, str]:
    """MuSiQue question id -> "select" or "test", by whole groups of questions joined through shared sub-questions,
    so no sub-question is in both."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        ids = _subq(r)
        for i in ids[1:]:
            parent[find(i)] = find(ids[0])
    out = {}
    for r in rows:
        key = min(find(i) for i in _subq(r))
        byte = hashlib.sha256(f"{SPLIT_SALT}:{key}".encode()).digest()[0]
        out[r["id"]] = "select" if byte % 2 == 0 else "test"
    return out


def _answer_in_question(answer: str, question: str) -> bool:
    a, q = normalize_text(answer) or "", normalize_text(question) or ""
    return bool(a) and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", q) is not None


def musique(args) -> None:
    raw = {s: _read_jsonl(data.path(MQ_DIR, f"musique_ans_v1.0_{s}.jsonl")) for s in ("train", "dev")}
    dev_split = musique_dev_split(raw["dev"])
    groups = {"train": raw["train"], "select": [r for r in raw["dev"] if dev_split[r["id"]] == "select"],
              "test": [r for r in raw["dev"] if dev_split[r["id"]] == "test"]}
    subq = {s: {i for r in rows for i in _subq(r)} for s, rows in groups.items()}
    for a, b in (("train", "select"), ("train", "test"), ("select", "test")):
        if subq[a] & subq[b]:
            raise SystemExit(f"MuSiQue {a} and {b} share {len(subq[a] & subq[b])} sub-questions")
    gold_text = {s: {clean(p["paragraph_text"]) for r in rows for p in r["paragraphs"] if p["is_supporting"]}
                 for s, rows in groups.items()}
    # an earlier split never pads with a later split's supporting paragraphs
    reserved = {"train": gold_text["select"] | gold_text["test"], "select": gold_text["test"], "test": set()}
    out = {"version": "musique-v1", "source": MQ_DIR, "revision": MQ_REV, "splits": {}}
    for split, rows in groups.items():
        table: list[list[str]] = []
        index: dict[tuple[str, str], int] = {}
        questions, skipped = [], collections.Counter()
        for r in rows:
            hops = int(r["id"][0])
            answer = clean(r["answer"])
            if not r.get("answerable", True):
                skipped["unanswerable"] += 1
                continue
            if len(r["question_decomposition"]) != hops:
                skipped["hop count"] += 1
                continue
            if normalize_text(answer) is None or _answer_in_question(answer, r["question"]):
                skipped["answer empty or in the question"] += 1
                continue
            pids = []
            for p in r["paragraphs"]:
                key = (clean(p["title"]), clean(p["paragraph_text"]))
                if key not in index:
                    index[key] = len(table)
                    table.append(list(key))
                pids.append(index[key])
            gold = sorted({pids[i] for i, p in enumerate(r["paragraphs"]) if p["is_supporting"]})
            if len(gold) < 2:
                skipped["fewer than 2 supporting paragraphs"] += 1
                continue
            aliases = sorted({clean(a) for a in r.get("answer_aliases") or [] if normalize_text(a)} - {answer})
            questions.append({"id": r["id"], "q": clean(r["question"]), "a": answer, "aliases": aliases, "hops": hops,
                              "paras": pids, "gold": gold, "subq": _subq(r)})
        pad_ok = [0 if text in reserved[split] else 1 for _, text in table]
        out["splits"][split] = {"paragraphs": table, "pad_ok": pad_ok, "questions": questions}
        hops = collections.Counter(q["hops"] for q in questions)
        log(f"musique {split}: {len(questions)} questions {dict(sorted(hops.items()))}, {len(table)} paragraphs "
            f"({sum(len(t) for _, t in table) / 1e6:.1f}M chars, {len(pad_ok) - sum(pad_ok)} kept out of padding); "
            f"skipped {dict(skipped)}")
    path = write_json_gz(MUSIQUE_OUT, out)
    log(f"musique -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------- QAMPARI


def title_of(url: str) -> str:
    title = urllib.parse.unquote(url.rsplit("/wiki/", 1)[-1]).split("#")[0]
    return clean(title.replace("_", " "))


def tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", text.lower()))


def clean_sentence(text: str, title: str) -> str | None:
    """A proof sentence as document text: spacing fixed, empty brackets dropped, a missing subject restored."""
    s = clean(text)
    s = re.sub(r"\s*\([^\w]*\)", "", s)  # "kumudra () is", "(, ; )"
    s = re.sub(r"\s+([,.;:!?)])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s).strip(" ,;")
    if not SENTENCE_CHARS[0] <= len(s) <= SENTENCE_CHARS[1]:
        return None
    if re.match(r"(?:is|was|are|were|has|had)\b", s, re.I):  # QAMPARI cut the subject off: put the page name back
        s = re.sub(r" \(.*\)$", "", title) + " " + s[0].lower() + s[1:]
    return s[0].upper() + s[1:]


def _entities(row: dict) -> list[str]:
    names = []
    for e in row["entities"]:
        for name in [e.get("entity_text", ""), *(e.get("aliases") or [])]:
            name = clean(name)
            if len(name) >= 3 and re.search(r"[^\W\d_]", name):
                names.append(name)
    return sorted(set(names), key=lambda n: (-len(n), n))


class Mentions:
    """Which sentences mention which questions' entities, by whole-word token sequences."""

    def __init__(self, entity_sets: dict[int, list[str]]):
        self.by_tokens: dict[tuple, set[int]] = collections.defaultdict(set)
        for qi, names in entity_sets.items():
            for name in names:
                if toks := tokens(name):
                    self.by_tokens[toks].add(qi)
        self.first = {t[0] for t in self.by_tokens}
        self.longest = max((len(t) for t in self.by_tokens), default=0)

    def find(self, sentence: str) -> set[int]:
        toks = tokens(sentence)
        hits: set[int] = set()
        for i, tok in enumerate(toks):
            if tok not in self.first:
                continue
            for n in range(1, self.longest + 1):
                if found := self.by_tokens.get(toks[i:i + n]):
                    hits |= found
        return hits


def _qampari_split(rows: list[dict]) -> tuple[dict, list[dict]]:
    """A split's pages (title -> cleaned sentences) and its candidate questions, before cross-split checks."""
    pages: dict[str, list[str]] = {}

    def add(title: str, text: str) -> int | None:
        """A proof sentence onto its page; returns its index. QAMPARI's proofs overlap (one is often another plus a
        sentence), so a sentence inside one already there is that one, and one that holds another replaces it."""
        s = clean_sentence(text, title)
        if s is None:
            return None
        sents = pages.setdefault(title, [])
        key = s.casefold()
        for i, x in enumerate(sents):
            if key in x.casefold():
                return i
            if x.casefold() in key:
                sents[i] = s
                return i
        sents.append(s)
        return len(sents) - 1

    for r in rows:  # every proof of the split, whatever its question type, makes the page pool
        for a in r["answer_list"]:
            for p in a["proof"]:
                add(title_of(p["found_in_url"]), p["proof_text"])
    candidates = []
    for r in rows:
        qtype = r["qid"].split("__")[1]
        if qtype not in QAMPARI_TYPES:
            continue
        names = _entities(r)
        if not names:
            continue
        mention = Mentions({0: names})
        proven: dict[str, set[int]] = {}  # answer page -> its proof sentences that name the question's entity
        unproven: set[str] = set()  # every other page an answer or its proof points to: never a distractor
        answers = set()
        for a in r["answer_list"]:
            page = title_of(a["answer_url"]) if a.get("answer_url") else None
            answers.add(page or clean(a["answer_text"]).casefold())
            sids = set()
            for p in a["proof"]:
                pt = title_of(p["found_in_url"])
                if page and pt == page:
                    sid = add(pt, p["proof_text"])
                    if sid is not None and mention.find(pages[pt][sid]):
                        sids.add(sid)
                else:
                    unproven.add(pt)
            if page and sids:
                proven.setdefault(page, set()).update(sids)
            elif page:
                unproven.add(page)
        candidates.append({"qid": r["qid"], "q": clean(r["question_text"]), "type": qtype, "entities": names,
                           "entity_urls": sorted({e["entity_url"] for e in r["entities"] if e.get("entity_url")}),
                           "proven": proven, "unproven": unproven - set(proven), "n_answers": len(answers)})
    return pages, candidates


def _stem(word: str) -> str:
    return STEMS.get(word, word[:5])


def _sentence_shape(sentence: str, drop: set[str]) -> set[str]:
    return set(tokens(sentence)) - drop - research.STOPWORDS


def _clear_negative(c: dict, pages: dict, title: str, sids: list[int]) -> bool:
    """Whether a page that names the question's entity reads as a different relation (see TEMPLATE_SAME)."""
    entity_words = {t for name in c["entities"] for t in tokens(name)}
    relation = {_stem(w) for w in research.content_words(c["q"]) if w not in entity_words}
    for si in sids:
        if relation and len(relation & {_stem(t) for t in tokens(pages[title][si])}) >= min(2, len(relation)):
            return False
    neg_shapes = [_sentence_shape(pages[title][si], entity_words | set(tokens(title))) for si in sids]
    for gold_title, gold_sids in c["proven"].items():
        for gi in gold_sids:
            g = _sentence_shape(pages[gold_title][gi], entity_words | set(tokens(gold_title)))
            if any(len(n) >= 3 and len(g & n) / len(n) >= TEMPLATE_SAME for n in neg_shapes):
                return False
    return True


def _usable(c: dict) -> str | None:
    """Why a candidate question can't be used, or None."""
    titles = list(c["proven"])
    if LANGUAGE_RE.search(c["q"]):
        return "a language question"
    if not MIN_ANSWERS <= len(titles) <= MAX_ANSWERS:
        return "answers checkable by reading not 5-15"
    if len(titles) < MIN_PROVEN * c["n_answers"]:
        return "too few answers checkable by reading"
    if any(re.search(r"[,;\[\]|\n]", t) or len(t) > MAX_TITLE for t in titles):
        return "an answer title with a list separator"
    if len({t.casefold() for t in titles}) < len(titles):
        return "answer titles equal ignoring case"
    if any(re.search(rf"(?<!\w){re.escape(t.casefold())}(?!\w)", c["q"].casefold()) for t in titles):
        return "an answer named in the question"  # "Which settlement is in Vavuniya District?": Vavuniya
    return None


def qampari(args) -> None:
    with zipfile.ZipFile(data.path(QP_DIR, "qampari.zip")) as z:
        raw = {}
        for split, member in (("train", "train"), ("select", "dev"), ("test", "test")):
            with z.open(f"qampari_data/{member}_data.jsonl") as f:
                raw[split] = [json.loads(line) for line in f if line.strip()]
    built = {split: _qampari_split(rows) for split, rows in raw.items()}
    log("qampari: pages and candidates read")

    # Cross-split checks, against every eval candidate (not only the kept ones), to be conservative.
    def answer_pages(c):
        return set(c["proven"]) | c["unproven"]

    later = {"train": ("select", "test"), "select": ("test",), "test": ()}
    out = {"version": "qampari-v1", "source": QP_DIR, "splits": {}}
    for split in ("test", "select", "train"):
        pages, candidates = built[split]
        reserved_pages: set[str] = set()
        eval_urls: set[str] = set()
        eval_sets: list[set[str]] = []
        for other in later[split]:
            for c in built[other][1]:
                reserved_pages |= answer_pages(c)
                eval_urls |= set(c["entity_urls"])
                eval_sets.append(set(c["proven"]))
        page_index = collections.defaultdict(list)
        for i, s in enumerate(eval_sets):
            for t in s:
                page_index[t].append(i)
        kept, skipped, texts = [], collections.Counter(), set()

        def why_not(c) -> str | None:
            if why := _usable(c):
                return why
            if set(c["entity_urls"]) & eval_urls:
                return "entity in a later split"
            mine = set(c["proven"])
            near = {i for t in mine for i in page_index.get(t, ())}
            if any(len(mine & eval_sets[i]) / len(mine | eval_sets[i]) >= JACCARD_DUP for i in near):
                return "answer set near one in a later split"
            if mine & reserved_pages:
                return "an answer page is a later split's answer"
            if c["q"].casefold() in texts:
                return "duplicate question text"
            return None

        for c in candidates:
            if why := why_not(c):
                skipped[why] += 1
                continue
            kept.append(c)
            texts.add(c["q"].casefold())
        # pages a context may hold: none of a later split's answers
        titles = sorted(t for t in pages if t not in reserved_pages and pages[t])
        # hard negatives: pages that name a question's entity but aren't its answers
        mentions = Mentions({qi: c["entities"] for qi, c in enumerate(kept)})
        negs: dict[int, dict[str, list[int]]] = collections.defaultdict(dict)
        for t in titles:
            for si, s in enumerate(pages[t]):
                for qi in mentions.find(s):
                    c = kept[qi]
                    if t not in c["proven"] and t not in c["unproven"]:
                        negs[qi].setdefault(t, []).append(si)
        rng = random.Random(f"{SPLIT_SALT}:{split}")
        unclear = 0
        for qi, c in enumerate(kept):
            found = sorted(negs.get(qi, {}).items())
            clear = [(t, s) for t, s in found if _clear_negative(c, pages, t, s)]
            unclear += len(found) - len(clear)
            # an unclear page is never a distractor for this question: it may be a missing answer
            c["unproven"] = c["unproven"] | ({t for t, _ in found} - {t for t, _ in clear})
            found = clear
            if len(found) > NEG_STORE:
                found = sorted(random.Random(f"{SPLIT_SALT}:{c['qid']}").sample(found, NEG_STORE))
            c["neg"] = found
        if split == "train":  # keep answers, negatives and a fixed sample of other pages
            needed = {t for c in kept for t in list(c["proven"]) + [n for n, _ in c["neg"]]}
            others = [t for t in titles if t not in needed]
            titles = sorted(needed | set(rng.sample(others, min(TRAIN_FILLER_PAGES, len(others)))))
        pid = {t: i for i, t in enumerate(titles)}
        questions = []
        for c in kept:
            questions.append({
                "qid": c["qid"], "q": c["q"], "type": c["type"], "entities": c["entities"],
                "entity_urls": c["entity_urls"],
                "gold": [[pid[t], sorted(s)] for t, s in sorted(c["proven"].items())],
                "neg": [[pid[t], s] for t, s in c["neg"] if t in pid],
                "exclude": sorted(pid[t] for t in c["unproven"] if t in pid)})
        out["splits"][split] = {"pages": [[t, pages[t]] for t in titles], "questions": questions}
        n_neg = sorted(len(q["neg"]) for q in questions)
        answers = collections.Counter(len(q["gold"]) for q in questions)
        log(f"qampari {split}: {len(questions)} questions, types "
            f"{dict(collections.Counter(q['type'] for q in questions))}, answers {dict(sorted(answers.items()))}, "
            f"hard negatives p25/50/75 {[n_neg[int(p * (len(n_neg) - 1))] for p in (0.25, 0.5, 0.75)] if n_neg else []}, "
            f"{len(titles)} pages ({sum(len(s) for t in titles for s in pages[t]) / 1e6:.1f}M chars); "
            f"{unclear} entity pages left out as possible answers; skipped {dict(skipped)}")
    path = write_json_gz(QAMPARI_OUT, out)
    log(f"qampari -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["download", "musique", "qampari", "all"])
    args = parser.parse_args()
    steps = {"download": [download], "musique": [musique], "qampari": [qampari],
             "all": [download, musique, qampari]}[args.what]
    for step in steps:
        step(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
