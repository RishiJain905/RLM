"""Open-book check for the research family (research-v3; after the pilot, 2026-10-18).

The base model, thinking off, gets a research question with only its gold documents and answers it. A question it
answers right is "valid": its gold documents really lead to its gold answer. The pilot found two of six level-3
MuSiQue tasks broken that way (a supporting paragraph about the wrong person; a gold "Italian Republic" against the
right answer "Italy"). Once this file exists, rlm_rl/tasks/research.py draws only valid questions.

- MuSiQue: the question's supporting paragraphs, shuffled; valid if the answer is right under the text grader
  (reward.score_answer, type "text", with the gold's aliases).
- QAMPARI: the answers' proof sentences and every stored hard-negative page's naming sentences, shuffled, as
  `# ===== doc N: <title>` documents; valid if the listed titles reach set-F1 >= VALID_LIST_F1. A list whose "hard
  negatives" read like answers (Wikidata's lists miss answers) fails here.

Which questions: exactly those scripts/closed_book.py checked (its file's "checked" index sets), so research.usable()
can require checked closed-book AND not known AND valid. Run inside WSL2 from the repo root:

    .venv-linux/bin/python -I scripts/open_book.py plan               # $0: how many calls, projected cost
    .venv-linux/bin/python -I scripts/open_book.py run --live --ceiling 8
    .venv-linux/bin/python -I scripts/open_book.py summary

The results go to the data root (derived/research/open-book-v1.json). A run keeps them in open-book-v1.partial.json
as it goes, so a stopped run resumes, and research.py only reads the final file. The W&B key is read from .env and
never printed.
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import config, data, reward  # noqa: E402
from rlm_rl.client import PRICE_INPUT, PRICE_OUTPUT  # noqa: E402
from rlm_rl.tasks import research  # noqa: E402

MODEL_NAME = "prod-v1"  # step 0 is the base model, as in scripts/closed_book.py
VALID_LIST_F1 = 0.8
SHUFFLE_SEED = 20261018
# Training questions checked per source (and per hop count for MuSiQue): a run draws a few hundred research tasks, so
# a fixed sample is plenty; the select and test splits are checked in full.
TRAIN_SAMPLE = {"musique": {2: 600, 3: 1500, 4: 1200}, "qampari": 700}
MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")  # a list line's bullet or number
AT_ONCE = 32
SAVE_EVERY = 250
MAX_TOKENS = {"musique": 64, "qampari": 400}
CHARS_PER_TOKEN = 3.5  # for the projected cost only *(mine)*
EST_OUTPUT_TOKENS = {"musique": 8, "qampari": 10}  # per answer; a list names ~one title per answer

ASK_ONE = ("Answer the question using only the documents below. The facts are spread over several documents, each "
           "step's answer leading to the next. Reply with only the final answer, in a few words.\n\n{docs}\n\n"
           "Question: {q}")
ASK_LIST = ("Answer the question using only the documents below. Some documents are about an answer to the "
            "question; others mention the same names but aren't answers. List the title of every document that is "
            "about an answer, one per line, copied exactly from its `# ===== doc N: <title>` header, with nothing "
            "else.\n\n{docs}\n\nQuestion: {q}")


def out_path() -> Path:
    return data.derived(research.OPEN_BOOK)


def partial_path() -> Path:
    return out_path().with_suffix(".partial.json")


def plan() -> list[tuple[str, str, int]]:
    """(source, split, question index): every question the closed-book check checked, in a fixed order."""
    checked = research.closed_book()
    if checked is None:
        raise SystemExit(f"no closed-book results at {data.derived(research.CLOSED_BOOK)}: run scripts/closed_book.py")
    items = []
    rng = random.Random(f"{SHUFFLE_SEED}:plan")
    for split in ("train", "select", "test"):
        for source in ("musique", "qampari"):
            row = checked["results"].get(source, {}).get(split)
            if not row:
                continue
            # questions the closed-book check found known are never drawn, so they need no check
            left = sorted(set(row["checked"]) - set(row["known"]))
            if split == "train":  # a training run draws a few hundred: a fixed sample per hop count keeps it ~$2.5
                if source == "musique":
                    by_hops: dict[int, list[int]] = {}
                    for i in left:
                        by_hops.setdefault(int(research._musique(split)["questions"][i]["hops"]), []).append(i)
                    left = sorted(i for hops, ids in by_hops.items()
                                  for i in rng.sample(ids, min(len(ids), TRAIN_SAMPLE["musique"].get(hops, 0))))
                else:
                    left = sorted(rng.sample(left, min(len(left), TRAIN_SAMPLE["qampari"])))
            items += [(source, split, i) for i in left]
    return items


def documents(source: str, split: str, i: int) -> tuple[str, list[str]]:
    """The documents one check sends, rendered as in a task, and the gold answer's titles (lists) or []."""
    rng = random.Random(f"{SHUFFLE_SEED}:{source}:{split}:{i}")
    if source == "musique":
        pool = research._musique(split)
        q = pool["questions"][i]
        docs = [tuple(pool["paragraphs"][pid]) for pid in dict.fromkeys(q["gold"])]
        rng.shuffle(docs)
        return research._render(docs), []
    pool = research._qampari(split)
    q = pool["questions"][i]
    pages = pool["pages"]
    picked = [(int(pid), sids) for pid, sids in q["gold"]] + [(int(pid), sids) for pid, sids in q["neg"]]
    docs = [(pages[pid][0], " ".join(pages[pid][1][s] for s in sorted(sids))) for pid, sids in picked]
    rng.shuffle(docs)
    return research._render(docs), sorted(pages[int(pid)][0] for pid, _ in q["gold"])


def prompt(source: str, split: str, i: int) -> str:
    pool = research._musique(split) if source == "musique" else research._qampari(split)
    docs, _ = documents(source, split, i)
    return (ASK_ONE if source == "musique" else ASK_LIST).format(docs=docs, q=pool["questions"][i]["q"])


def judge(source: str, split: str, i: int, reply: str) -> tuple[bool, float]:
    """(valid, score) for one reply."""
    if source == "musique":
        q = research._musique(split)["questions"][i]
        grading = {"type": "text", "aliases": list(q["aliases"]), "blind": None,
                   "ignore": sorted(set((reward.normalize_text(q["q"]) or "").split()))}
        s = reward.score_answer(reply.strip(), q["a"], grading)
        return bool(s.correct), s.reward
    _, gold = documents(source, split, i)
    items = [MARKER.sub("", line).strip().strip("`") for line in reply.splitlines() if line.strip()]
    items = [re.sub(r"^#+\s*=+\s*doc \d+:\s*", "", item) for item in items]  # a header copied whole
    f1 = reward.set_f1(reward.normalize_set(items, "entity"), reward.normalize_set(gold, "entity"))
    return f1 >= VALID_LIST_F1, f1


def projected_usd(items: list[tuple[str, str, int]]) -> tuple[float, dict]:
    """The projected cost of `items` from their prompts' length, and the prompt characters by (source, split)."""
    usd, chars = 0.0, {}
    for source, split, i in items:
        n = len(prompt(source, split, i))
        chars[(source, split)] = chars.get((source, split), 0) + n
        answers = 1 if source == "musique" else len(research._qampari(split)["questions"][i]["gold"])
        usd += (n / CHARS_PER_TOKEN * PRICE_INPUT + EST_OUTPUT_TOKENS[source] * answers * PRICE_OUTPUT) / 1e6
    return usd, chars


def load() -> dict:
    for path in (partial_path(), out_path()):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"version": "open-book-v1", "model": None, "closed_book": (research.closed_book() or {}).get("version"),
            "rules": {"list_f1": VALID_LIST_F1, "thinking": "off", "seed": SHUFFLE_SEED,
                      "musique": "supporting paragraphs only; right under the text grader",
                      "qampari": "proof sentences and every stored hard negative; set-F1 >= list_f1"},
            "results": {}, "spent_usd": 0.0}


def save(state: dict, final: bool = False) -> None:
    path = out_path() if final else partial_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


async def run(args) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY; never printed
    from rlm_rl.backend import ArtBackend
    from rlm_rl.client import ModelClient

    state = load()
    done = {(src, split, i) for src, by in state["results"].items() for split, row in by.items()
            for i in row["checked"]}
    todo = [item for item in plan() if item not in done]
    usd, _ = projected_usd(todo)
    print(f"{len(done)} checked, {len(todo)} to go, ~${usd:.2f}; spent ${state['spent_usd']:.2f}; "
          f"ceiling ${args.ceiling:g}")
    backend = ArtBackend(project=config.PROJECT, base_model=config.BASE_MODEL)
    try:
        await backend.open(MODEL_NAME)
        base = backend.inference_name(MODEL_NAME, 0)
        state["model"] = base
        client = ModelClient(backend.openai_client(), subcall_model=base)
        spent0 = state["spent_usd"]
        gate = asyncio.Semaphore(AT_ONCE)
        failed = 0
        since_save = 0

        async def one(item):
            nonlocal failed, since_save
            source, split, i = item
            async with gate:
                if spent0 + client.spent_usd() >= args.ceiling:
                    return
                try:
                    reply = await client.chat([{"role": "user", "content": prompt(source, split, i)}], model=base,
                                              max_tokens=MAX_TOKENS[source], temperature=0.0)
                except Exception as exc:  # noqa: BLE001 - counted and retried on the next run
                    failed += 1
                    print(f"failed {source}/{split}/{i}: {type(exc).__name__}", file=sys.stderr)
                    return
            valid, score = judge(source, split, i, reply)
            row = state["results"].setdefault(source, {}).setdefault(split, {"checked": [], "valid": [], "scores": {}})
            row["checked"].append(i)
            row["scores"][str(i)] = round(score, 3)
            if valid:
                row["valid"].append(i)
            since_save += 1
            if since_save >= SAVE_EVERY:
                since_save = 0
                state["spent_usd"] = round(spent0 + client.spent_usd(), 4)
                state["updated_at"] = time.time()
                save(state)
                print(f"  {sum(len(r['checked']) for by in state['results'].values() for r in by.values())} checked, "
                      f"${state['spent_usd']:.2f}")

        await asyncio.gather(*(one(item) for item in todo))
        state["spent_usd"] = round(spent0 + client.spent_usd(), 4)
        state["updated_at"] = time.time()
        complete = not failed and spent0 + client.spent_usd() < args.ceiling
        save(state, final=complete)
        if complete:
            partial_path().unlink(missing_ok=True)
            print(f"complete: {out_path()}")
        summary(state)
        if not complete:
            print(f"not complete ({failed} call(s) failed or the ceiling was reached); run again to finish. "
                  "research.py ignores the open-book check until then")
    finally:
        await backend.close()


def summary(state: dict | None = None) -> None:
    state = state or load()
    known = (research.closed_book() or {}).get("results") or {}
    for source, by in state["results"].items():
        for split, row in by.items():
            n, v = len(row["checked"]), len(row["valid"])
            unknown = set(row["checked"]) - set(known.get(source, {}).get(split, {}).get("known", []))
            usable = len(unknown & set(row["valid"]))
            print(f"{source:8} {split:6} checked {n:6}  valid {v:6} ({v / n:.0%})  usable (valid, not known) {usable}"
                  if n else f"{source} {split}: none")
    print(f"spent ${state['spent_usd']:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("run")
    p.add_argument("--live", action="store_true", required=True, help="spends W&B inference credit")
    p.add_argument("--ceiling", type=float, required=True, help="hard stop, $")
    sub.add_parser("summary")
    args = parser.parse_args()
    if args.command == "plan":
        items = plan()
        usd, chars = projected_usd(items)
        by: dict = {}
        for source, split, _ in items:
            by[(source, split)] = by.get((source, split), 0) + 1
        for key, n in sorted(by.items()):
            print(key, n, f"calls, {chars[key] / n:,.0f} prompt chars each on average")
        print(f"{len(items)} calls, {sum(chars.values()) / 1e6:.1f}M prompt characters, ~${usd:.2f}")
    elif args.command == "run":
        asyncio.run(run(args))
    else:
        summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
