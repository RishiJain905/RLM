"""Closed-book check for the research family (research-v2; Rishi, 2026-10-18: no questions the model already knows).

The base model answers each research question with no documents, thinking off. A question it answers right is
"known": a task built on it can be solved without reading, so it would teach nothing about delegating, and
rlm_rl/tasks/research.py never draws it. Once this file exists, only checked questions are drawn.

Which questions are checked: every 3- and 4-hop MuSiQue train question and a fixed sample of the 2-hop ones (level 1
only), a fixed sample of QAMPARI's train lists, and every question of the select and test splits. At ~$0.0001 a call
that is about $1.3. Run inside WSL2 from the repo root:

    .venv-linux/bin/python -I scripts/closed_book.py plan               # $0: how many calls, projected cost
    .venv-linux/bin/python -I scripts/closed_book.py run --live --ceiling 2

The results go to the data root (derived/research/closed-book-v1.json), saved as it goes, so a stopped run resumes.
The W&B key is read from .env and never printed.
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
from rlm_rl.tasks import research  # noqa: E402

# The base model is prod-v1's step 0. A freshly registered model's step 0 answered only HTTP 500s for minutes
# (2026-10-18), while prod-v1's answered at once.
MODEL_NAME = "prod-v1"
TWO_HOP_SAMPLE = 2500  # of 14,374 2-hop train questions; level 1 draws only these
LIST_SAMPLE = 2000  # of 7,659 QAMPARI train lists
SAMPLE_SEED = 20261018
KNOWN_TEXT_REWARD = 0.6  # a multi-hop answer is known if right, or if its token-F1 credit reaches this (F1 0.75)
KNOWN_LIST_F1 = 0.5  # a list is known if the model names at least this set-F1 of it from memory
EST_USD_PER_CALL = 0.0001
MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")  # a list line's bullet or number
AT_ONCE = 32
SAVE_EVERY = 250

ASK_ONE = ("Answer this question from your own knowledge, without looking anything up. Reply with only the answer, "
           "in a few words. If you are not sure, give your best guess.\n\nQuestion: {q}")
ASK_LIST = ("Answer this question from your own knowledge, without looking anything up. List every answer you know, "
            "one per line, with nothing else.\n\nQuestion: {q}")


def out_path() -> Path:
    return data.derived(research.CLOSED_BOOK)


def plan() -> list[tuple[str, str, int]]:
    """(source, split, question index) to check, in a fixed order."""
    rng = random.Random(SAMPLE_SEED)
    items = []
    for split in ("train", "select", "test"):
        questions = research._musique(split)["questions"]
        if split == "train":
            two = [i for i, q in enumerate(questions) if int(q["hops"]) == 2]
            keep = set(rng.sample(two, min(TWO_HOP_SAMPLE, len(two))))
            idx = [i for i, q in enumerate(questions) if int(q["hops"]) > 2 or i in keep]
        else:
            idx = list(range(len(questions)))
        items += [("musique", split, i) for i in idx]
        lists = research._qampari(split)["questions"]
        idx = sorted(rng.sample(range(len(lists)), min(LIST_SAMPLE, len(lists)))) if split == "train" \
            else list(range(len(lists)))
        items += [("qampari", split, i) for i in idx]
    return items


def partial_path() -> Path:
    """Where a run in progress keeps its results: research.py only reads the final file, so a half-done check never
    shrinks the pools."""
    return out_path().with_suffix(".partial.json")


def load() -> dict:
    for path in (partial_path(), out_path()):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"version": "closed-book-v1", "model": None, "rules": {
        "text_reward": KNOWN_TEXT_REWARD, "list_f1": KNOWN_LIST_F1, "thinking": "off",
        "two_hop_sample": TWO_HOP_SAMPLE, "list_sample": LIST_SAMPLE, "seed": SAMPLE_SEED},
        "results": {}, "spent_usd": 0.0}


def save(state: dict, final: bool = False) -> None:
    path = out_path() if final else partial_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


def judge(source: str, split: str, i: int, reply: str) -> tuple[bool, float]:
    """(known, score) for one reply."""
    if source == "musique":
        q = research._musique(split)["questions"][i]
        grading = {"type": "text", "aliases": list(q["aliases"]), "blind": None,
                   "ignore": sorted(set((reward.normalize_text(q["q"]) or "").split()))}
        s = reward.score_answer(reply.strip(), q["a"], grading)
        return bool(s.correct or s.reward >= KNOWN_TEXT_REWARD), s.reward
    q = research._qampari(split)["questions"][i]
    pages = research._qampari(split)["pages"]
    gold = sorted(pages[int(pid)][0] for pid, _ in q["gold"])
    items = [MARKER.sub("", line).strip() for line in reply.splitlines() if line.strip()]
    f1 = reward.set_f1(reward.normalize_set(items, "entity"), reward.normalize_set(gold, "entity"))
    return f1 >= KNOWN_LIST_F1, f1


async def run(args) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY; never printed
    from rlm_rl.backend import ArtBackend
    from rlm_rl.client import ModelClient

    state = load()
    done = {(src, split, i) for src, by in state["results"].items() for split, row in by.items()
            for i in row["checked"]}
    todo = [item for item in plan() if item not in done]
    print(f"{len(done)} checked, {len(todo)} to go, ~${len(todo) * EST_USD_PER_CALL:.2f}; "
          f"spent ${state['spent_usd']:.2f}; ceiling ${args.ceiling:g}")
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
                pool = research._musique(split) if source == "musique" else research._qampari(split)
                text = (ASK_ONE if source == "musique" else ASK_LIST).format(q=pool["questions"][i]["q"])
                try:
                    reply = await client.chat([{"role": "user", "content": text}], model=base,
                                              max_tokens=48 if source == "musique" else 300, temperature=0.0)
                except Exception as exc:  # noqa: BLE001 - counted and retried on the next run
                    failed += 1
                    print(f"failed {source}/{split}/{i}: {type(exc).__name__}", file=sys.stderr)
                    return
            known, score = judge(source, split, i, reply)
            row = state["results"].setdefault(source, {}).setdefault(split, {"checked": [], "known": [], "scores": {}})
            row["checked"].append(i)
            row["scores"][str(i)] = round(score, 3)
            if known:
                row["known"].append(i)
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
                  "research.py keeps drawing from every question until then")
    finally:
        await backend.close()


def summary(state: dict | None = None) -> None:
    state = state or load()
    for source, by in state["results"].items():
        for split, row in by.items():
            n, k = len(row["checked"]), len(row["known"])
            print(f"{source:8} {split:6} checked {n:6}  known {k:5} ({k / n:.0%})" if n else f"{source} {split}: none")
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
        by = {}
        for source, split, _ in items:
            by[(source, split)] = by.get((source, split), 0) + 1
        for key, n in sorted(by.items()):
            print(key, n)
        print(f"{len(items)} calls, ~${len(items) * EST_USD_PER_CALL:.2f}")
    elif args.command == "run":
        asyncio.run(run(args))
    else:
        summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
