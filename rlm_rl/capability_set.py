"""The capability check's prompts (plan step 7; research §8.5): ~100 short prompts that catch frying.

`build()` samples a fixed set, in the JSONL format `rlm_rl/capability.py` runs, from four sources:
- IFEval (Apache-2.0): 30 prompts whose instructions all translate into capability.py's machine checks
- MMLU-Pro (MIT): 40 multiple-choice questions, stratified over its 14 subjects
- CRUXEval (MIT): 20 "what does this function return?" items, the short coding set
- 10 chat prompts written here, each with a machine check

All of it is eval-only, so the licence bar (7b) doesn't apply, and none of it is ever trained on.
`scripts/build_data.py capability` writes it to derived/capability/<VERSION>.jsonl on the data root.
"""

from __future__ import annotations

import ast
import json
import random
import re

from . import data

VERSION = "capability-v1"
COUNTS = {"ifeval": 30, "mmlu_pro": 40, "cruxeval": 20, "chat": 10}

CHAT = [
    ("Reply with exactly the word: ready", [{"kind": "exact", "value": "ready"}]),
    ("What is 17 times 23? Reply with the number only.", [{"kind": "number", "value": 391}, {"kind": "max_words", "value": 3}]),
    ("Name a prime number between 20 and 30. Reply with the number only.",
     [{"kind": "regex", "value": r"^\s*(23|29)\s*\.?\s*$"}]),
    ("Write a haiku about rain, using only lowercase letters.", [{"kind": "lowercase"}, {"kind": "max_words", "value": 30}]),
    ("How do you say 'thank you' in French? Reply with the translation only.",
     [{"kind": "contains", "value": "merci"}, {"kind": "max_words", "value": 4}]),
    ("Give a JSON object with the keys name and age for a person called Ana who is 31. Output only the JSON.",
     [{"kind": "json"}, {"kind": "contains", "value": "Ana"}, {"kind": "contains", "value": "31"}]),
    ("Spell the word 'necessary' backwards. Reply with the reversed word only.",
     [{"kind": "contains", "value": "yrassecen"}, {"kind": "max_words", "value": 3}]),
    ("In one word, what is the capital of Japan?", [{"kind": "contains", "value": "tokyo"}, {"kind": "max_words", "value": 3}]),
    ("List the three primary colours of light, separated by commas, and nothing else.",
     [{"kind": "contains", "value": "red"}, {"kind": "contains", "value": "green"}, {"kind": "contains", "value": "blue"},
      {"kind": "max_words", "value": 6}]),
    ("Explain in fewer than 25 words why ice floats on water.", [{"kind": "max_words", "value": 24},
                                                                 {"kind": "regex", "value": r"(?i)dens"}]),
]


def _ifeval_checks(ids: list[str], kwargs: list[dict]) -> list[dict] | None:
    """capability.py checks for an IFEval prompt, or None if any of its instructions has no faithful check."""
    checks = []
    for iid, kw in zip(ids, kwargs):
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        if iid == "change_case:english_lowercase":
            checks.append({"kind": "lowercase"})
        elif iid == "change_case:english_capital":
            checks.append({"kind": "regex", "value": r"^[^a-z]*$"})
        elif iid == "punctuation:no_comma":
            checks.append({"kind": "not_contains", "value": ","})
        elif iid == "detectable_format:json_format":
            checks.append({"kind": "json"})
        elif iid == "detectable_format:title":
            checks.append({"kind": "regex", "value": r"<<[^\n]+>>"})
        elif iid == "startend:quotation":
            checks.append({"kind": "regex", "value": r'^\s*".*"\s*$'})
        elif iid == "startend:end_checker" and "end_phrase" in kw:
            checks.append({"kind": "regex", "value": re.escape(kw["end_phrase"]) + r"\s*$"})
        elif iid == "detectable_content:postscript" and "postscript_marker" in kw:
            checks.append({"kind": "contains", "value": kw["postscript_marker"]})
        elif iid == "keywords:existence" and kw.get("keywords"):
            checks += [{"kind": "contains", "value": k} for k in kw["keywords"]]
        elif iid == "keywords:forbidden_words" and kw.get("forbidden_words"):
            checks += [{"kind": "regex", "value": rf"(?i)^(?!.*\b{re.escape(w)}\b)"} for w in kw["forbidden_words"]]
        elif iid == "length_constraints:number_words" and "num_words" in kw:
            n = int(kw["num_words"])
            checks.append({"kind": "max_words", "value": n - 1} if kw.get("relation") == "less than"
                          else {"kind": "min_words", "value": n})
        else:
            return None
    return checks


def _ifeval(rng: random.Random, n: int) -> list[dict]:
    rows = [json.loads(line) for line in data.path("hf/google__IFEval/ifeval_input_data.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    usable = []
    for row in rows:
        checks = _ifeval_checks(row["instruction_id_list"], row["kwargs"])
        if checks:
            usable.append({"id": f"ifeval-{row['key']}", "source": "ifeval", "prompt": row["prompt"], "checks": checks})
    return rng.sample(usable, n)


def _mmlu_pro(rng: random.Random, n: int) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(data.path("hf/TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet"),
                         columns=["question_id", "question", "options", "answer", "category"]).to_pylist()
    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row)
    cats = sorted(by_cat)
    picked = []
    for i in range(n):
        row = rng.choice(by_cat[cats[i % len(cats)]])
        options = row["options"] if isinstance(row["options"], list) else ast.literal_eval(row["options"])
        letters = "ABCDEFGHIJ"
        body = "\n".join(f"({letters[j]}) {opt}" for j, opt in enumerate(options))
        prompt = (f"{row['question']}\n\n{body}\n\nAnswer with the letter of the correct option, in the form "
                  "'The answer is (X)'.")
        picked.append({"id": f"mmlu-{row['question_id']}", "source": "mmlu_pro", "prompt": prompt,
                       "checks": [{"kind": "choice", "value": row["answer"]}]})
    return picked


def _cruxeval(rng: random.Random, n: int) -> list[dict]:
    rows = [json.loads(line) for line in data.path("hf/cruxeval-org__cruxeval/test.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    usable = []
    for row in rows:
        try:
            value = ast.literal_eval(row["output"])
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, bool):
            check = {"kind": "regex", "value": rf"(?i)^\W*{value}\W*$"}
        elif isinstance(value, int):
            check = {"kind": "number", "value": value}
        elif isinstance(value, str) and value and len(value) <= 40 and "\n" not in value:
            check = {"kind": "exact", "value": value}
        else:
            continue
        prompt = (f"Here is a Python function:\n\n```python\n{row['code']}\n```\n\nWhat does `f({row['input']})` "
                  "return? Reply with the value only.")
        usable.append({"id": f"crux-{row['id']}", "source": "cruxeval", "prompt": prompt, "checks": [check]})
    return rng.sample(usable, n)


def build(rng: random.Random) -> list[dict]:
    items = _ifeval(rng, COUNTS["ifeval"]) + _mmlu_pro(rng, COUNTS["mmlu_pro"]) + _cruxeval(rng, COUNTS["cruxeval"])
    items += [{"id": f"chat-{i:02d}", "source": "chat", "prompt": p, "checks": c} for i, (p, c) in enumerate(CHAT)]
    return items


def path():
    return data.derived("capability", f"{VERSION}.jsonl")
