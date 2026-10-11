"""OOLONG-style aggregation tasks over labelled utterances (plan step 7, the 30% aggregation share of decision 7a).

OOLONG-synth itself fails the licence bar (decision 7b), so this generator rebuilds its method on permissive
data. A context is a log of short messages, each with a date and a user id but no label. The question asks
for an aggregate over the labels (how many messages are about travel, which users sent at least 5 banking
messages), so the model has to classify every message by reading it, then count with code. 30% of
questions need no labels at all (dates and users only), so code alone solves them (research §3.1).

Sources: CLINC150 utterances grouped by domain (CC-BY-3.0) and MASSIVE utterances grouped by scenario
(CC-BY-4.0). The label is the domain or scenario, never the fine intent, so each label has hundreds of
distinct messages and a top-level context repeats almost none.

Per seed: the label set, the label mix, the users and their ids, the date range, the line format and the
question all change (research §3.1, renamed entities). The answer is never in the context.

Families:
- `aggregation`: the training family, on the CLINC150 and MASSIVE train and validation splits.
- `aggregation_heldout`: held-out test only (H7). A source never trained on (HuffPost headlines), the
  sources' test splits are not used anywhere, and separately written line formats and question templates.

Levels go from the 20-step tasks' size (~35K characters) to ~400K, which ~20 sub-calls of ~20K characters
cover (spec §6). Each level up also adds labels and users.
"""

from __future__ import annotations

import datetime as dt
import random
import secrets

from .. import data
from ..env import Task

GENERATOR_VERSION = "aggregation-v1"
FAMILIES = ("aggregation", "aggregation_heldout")
TRAIN_SOURCES = ("clinc150", "clinc150-domains", "massive")
HELDOUT_SOURCES = ("huffpost",)

# Characters per context at each level: (fewest, most), with the label count and the user range.
LEVELS = {1: (32_000, 40_000), 2: (70_000, 90_000), 3: (130_000, 170_000), 4: (230_000, 290_000),
          5: (360_000, 420_000)}
LABELS_AT = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}
USERS_AT = {1: (6, 10), 2: (10, 16), 3: (14, 24), 4: (20, 32), 5: (26, 40)}
MAX_LEVEL = max(LEVELS)

# What each label means, shown in the question. CLINC150 domains, then MASSIVE scenarios.
DESCRIPTIONS = {
    "banking": "bank accounts: balances, transactions, transfers, bills and payments, routing numbers, PINs, frozen accounts, fraud",
    "credit_cards": "credit cards: limits, APR, rewards, credit scores, card fees, new, lost, damaged or declined cards",
    "kitchen_and_dining": "food: recipes, ingredients, calories, cooking times, restaurants and reservations",
    "home": "the household: shopping and to-do lists, reminders, calendar, orders, smart home, playing music",
    "auto_and_commute": "cars and driving: fuel, oil, tyres, maintenance, traffic, directions, distances, rides",
    "travel": "trips: flights, hotels, car rental, luggage, visas, vaccines, plugs, exchange rates, translation, time zones",
    "utility": "quick tools: weather, time, date, alarms, timers, calls, texts, calculator, conversions, spelling, definitions, dice, coin flips, finding the phone",
    "work": "the job: meetings, PTO, holidays, payday, direct deposit, income, taxes, W-2, 401k, insurance",
    "small_talk": "chit-chat with the assistant: greetings, thanks, goodbyes, jokes, fun facts, questions about the assistant itself",
    "meta": "controlling the assistant: renaming it or the user, its accent, speed, volume, language, whisper mode, settings, syncing, yes/no/maybe/cancel/repeat",
    "alarm": "setting, changing or removing alarms",
    "audio": "device volume and audio settings",
    "calendar": "calendar events, meetings and reminders",
    "cooking": "recipes and cooking instructions",
    "datetime": "asking the current time or date, time zones",
    "email": "reading, writing or managing email and contacts",
    "iot": "smart home devices: lights, plugs, vacuum, coffee machine",
    "lists": "creating and editing lists",
    "music": "music preferences, song information, likes and dislikes",
    "news": "news headlines and topics",
    "qa": "factual questions, definitions, maths, currency, stocks",
    "recommendation": "recommendations for places, events, movies",
    "social": "social media posts and complaints to companies",
    "takeaway": "ordering takeaway food and delivery",
    "transport": "trains, taxis, tickets and traffic",
    "weather": "the weather and forecasts",
    # Held-out source: HuffPost categories (eval only)
    "SPORTS": "sport", "FOOD & DRINK": "food, drink and recipes", "TRAVEL": "travel and destinations",
    "TECH": "technology", "RELIGION": "religion and faith", "WEDDINGS": "weddings", "SCIENCE": "science",
    "BUSINESS": "business and the economy", "CRIME": "crime", "ENTERTAINMENT": "film, TV, music and celebrities",
    "ENVIRONMENT": "the environment and climate", "EDUCATION": "schools and education",
}
CLINC_LABELS = ("banking", "credit_cards", "kitchen_and_dining", "home", "auto_and_commute", "travel", "utility",
                "work", "small_talk", "meta")
# MASSIVE's `general` and `play` scenarios overlap the others too much to grade fairly.
MASSIVE_LABELS = ("alarm", "audio", "calendar", "cooking", "datetime", "email", "iot", "lists", "music", "news",
                  "qa", "recommendation", "social", "takeaway", "transport", "weather")
HELDOUT_LABELS = ("SPORTS", "FOOD & DRINK", "TRAVEL", "TECH", "RELIGION", "WEDDINGS", "SCIENCE", "BUSINESS",
                  "CRIME", "ENTERTAINMENT", "ENVIRONMENT", "EDUCATION")
# Pairs too close to tell apart reliably, never asked about in the same task.
INCOMPATIBLE = {frozenset(p) for p in [("banking", "credit_cards"), ("small_talk", "meta"),
                                       ("alarm", "datetime"), ("calendar", "alarm"), ("music", "audio"),
                                       ("transport", "travel"), ("news", "qa"), ("BUSINESS", "TECH"),
                                       ("ENVIRONMENT", "SCIENCE"), ("FOOD & DRINK", "TRAVEL")]}

# Question kinds and how often each is drawn. The last two need no labels, so code alone answers them.
# The code-only kinds get 30%, so the whole 7a mix stays near research §3.1's ~20% code-solvable tasks.
KINDS = {"label_count": 0.22, "user_label_count": 0.12, "labels_over": 0.13, "users_with_label": 0.08,
         "window_label_count": 0.15, "user_window_count": 0.18, "most_active_user": 0.12}
CODE_ONLY_KINDS = ("user_window_count", "most_active_user")

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")

# Line formats. Training uses the first three; the held-out test uses its own (H7).
TRAIN_FORMATS = ("{iso} | user {user} | {text}", "[{mon} {day}, {year}] user#{user}: {text}",
                 "{day:02d}/{month:02d}/{year} - id {user} - {text}")
HELDOUT_FORMATS = ("On {weekday} {day} {monname} {year}, account {user} posted: {text}",
                   "{year}.{month:02d}.{day:02d} <{user}> {text}")
TRAIN_HEADERS = ("Messages sent to a voice assistant, one per line: date, user id, message.",
                 "Assistant request log. Each line has a date, the id of the user who sent it, and what they said.")
HELDOUT_HEADERS = ("Headlines shared by readers of a news app, one per line, with the date and the reader's account.",)

# Question templates. {labels} is the label list with descriptions. Held-out templates are written separately,
# and scripts/data_check.py checks their n-gram overlap with these.
TRAIN_TEMPLATES = {
    "label_count": ["Every message belongs to exactly one of these topics:\n{labels}\nHow many messages are about "
                    "the topic '{label}'? Answer with a single integer.",
                    "Each line is about one of these topics:\n{labels}\nCount the messages whose topic is '{label}'. "
                    "Answer with a single integer."],
    "user_label_count": ["Every message belongs to exactly one of these topics:\n{labels}\nHow many messages from "
                         "user {user} are about '{label}'? Answer with a single integer.",
                         "Each line is about one of these topics:\n{labels}\nCount the '{label}' messages sent by "
                         "user {user}. Answer with a single integer."],
    "labels_over": ["Every message belongs to exactly one of these topics:\n{labels}\nWhich topics have more than "
                    "{threshold} messages? Answer with a list of topic names.",
                    "Each line is about one of these topics:\n{labels}\nList every topic that more than {threshold} "
                    "messages are about. Answer with a list of topic names."],
    "users_with_label": ["Every message belongs to exactly one of these topics:\n{labels}\nWhich users sent at "
                         "least {threshold} messages about '{label}'? Answer with a list of user ids.",
                         "Each line is about one of these topics:\n{labels}\nList the users with {threshold} or "
                         "more '{label}' messages. Answer with a list of user ids."],
    "window_label_count": ["Every message belongs to exactly one of these topics:\n{labels}\nHow many messages "
                           "about '{label}' were sent from {start} to {end}, both days included? Answer with a "
                           "single integer.",
                           "Each line is about one of these topics:\n{labels}\nBetween {start} and {end} "
                           "(inclusive), how many '{label}' messages were sent? Answer with a single integer."],
    "user_window_count": ["How many messages did user {user} send from {start} to {end}, both days included? "
                          "Answer with a single integer.",
                          "Count the messages user {user} sent between {start} and {end} (inclusive). Answer with "
                          "a single integer."],
    "most_active_user": ["Which user sent the most messages? Answer with the user id.",
                         "Give the id of the user with the largest number of messages. Answer with the user id."],
}
HELDOUT_TEMPLATES = {
    "label_count": ["Classify each headline into one of the following categories:\n{labels}\nWhat is the total "
                    "number of headlines in category '{label}'? Reply with one integer."],
    "user_label_count": ["Classify each headline into one of the following categories:\n{labels}\nFor account "
                         "{user} only, what is the total number of '{label}' headlines? Reply with one integer."],
    "labels_over": ["Classify each headline into one of the following categories:\n{labels}\nName the categories "
                    "whose headline total exceeds {threshold}. Reply with the category names, comma separated."],
    "users_with_label": ["Classify each headline into one of the following categories:\n{labels}\nName the "
                         "accounts that posted {threshold} or more '{label}' headlines. Reply with the account "
                         "numbers, comma separated."],
    "window_label_count": ["Classify each headline into one of the following categories:\n{labels}\nConsidering "
                           "only posts dated {start} through {end}, what is the total number of '{label}' "
                           "headlines? Reply with one integer."],
    "user_window_count": ["What is the total number of posts by account {user} dated {start} through {end}? Reply "
                          "with one integer."],
    "most_active_user": ["Name the account that posted more often than any other. Reply with the account number."],
}


def _variant(family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"unknown aggregation family {family!r}")
    return "heldout" if family.endswith("_heldout") else "train"


def _pools(variant: str, rng: random.Random, k: int, need: int) -> dict[str, tuple[str, ...]]:
    """k compatible labels and their message pools. MASSIVE labels are used only where the pool is big enough."""
    if variant == "heldout":
        candidates = [(data.huffpost_pools(), HELDOUT_LABELS)]
    else:
        candidates = [(data.clinc_domain_pools(), CLINC_LABELS), (data.massive_scenario_pools(), MASSIVE_LABELS)]
        if rng.random() < 0.4:  # MASSIVE first 40% of the time *(mine)*; CLINC150's pools are the biggest
            candidates.reverse()
    for pools, names in candidates:
        usable = [n for n in names if len(pools.get(n, ())) >= need]
        rng.shuffle(usable)
        chosen: list[str] = []
        for name in usable:
            if all(frozenset((name, other)) not in INCOMPATIBLE for other in chosen):
                chosen.append(name)
            if len(chosen) == k:
                return {name: pools[name] for name in chosen}
    raise ValueError(f"no {k} compatible labels with {need}+ messages each")


def _user_ids(rng: random.Random, n: int) -> list[str]:
    ids: set[str] = set()
    while len(ids) < n:
        ids.add(str(rng.randint(10_000, 99_999)))
    return sorted(ids, key=lambda _: rng.random())


def _fmt_date(d: dt.date) -> str:
    return f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"


def _render_line(fmt: str, d: dt.date, user: str, text: str) -> str:
    return fmt.format(iso=d.isoformat(), mon=_MONTHS[d.month - 1][:3], monname=_MONTHS[d.month - 1],
                      weekday=d.strftime("%A"), day=d.day, month=d.month, year=d.year, user=user, text=text)


def _build(rng: random.Random, family: str, level: int, task_id: str) -> Task:
    variant = _variant(family)
    if level not in LEVELS:
        raise ValueError(f"no aggregation level {level} (levels 1-{MAX_LEVEL})")
    fmt = rng.choice(TRAIN_FORMATS if variant == "train" else HELDOUT_FORMATS)
    header = rng.choice(TRAIN_HEADERS if variant == "train" else HELDOUT_HEADERS)
    budget = rng.randint(*LEVELS[level])
    k = LABELS_AT[level]
    est_line = len(_render_line(fmt, dt.date(2024, 9, 28), "12345", "x" * 38)) + 1
    n_est = budget // est_line
    pools = _pools(variant, rng, k, need=int(1.3 * n_est / k))
    labels = list(pools)
    label_w = [rng.uniform(0.4, 1.6) for _ in labels]
    users = _user_ids(rng, rng.randint(*USERS_AT[level]))
    user_w = [1 / (i + 1) ** 0.7 for i in range(len(users))]  # a few heavy users, a long tail
    start = dt.date(2019, 1, 1) + dt.timedelta(days=rng.randint(0, 6 * 365))
    span = rng.randint(120, 540)

    remaining = {name: list(rng.sample(pools[name], len(pools[name]))) for name in labels}
    rows, size = [], len(header) + 1
    while size < budget:
        label = rng.choices(labels, label_w)[0]
        if not remaining[label]:
            remaining[label] = list(rng.sample(pools[label], len(pools[label])))  # a repeat only after the pool runs out
        text = remaining[label].pop()
        row = (start + dt.timedelta(days=rng.randint(0, span)), rng.choices(users, user_w)[0], label, text)
        rows.append(row)
        size += len(_render_line(fmt, row[0], row[1], text)) + 1
    rows.sort(key=lambda r: r[0])
    context = "\n".join([header] + [_render_line(fmt, d, u, t) for d, u, _, t in rows])

    question, answer, grading = _question(rng, variant, rows, labels, users, level)
    return Task(task_id, family, question, context, answer=answer, level=level, grading=grading)


def _label_list(labels: list[str]) -> str:
    return "\n".join(f"- {name}: {DESCRIPTIONS[name]}" for name in labels)


def _window(rng: random.Random, rows) -> tuple[dt.date, dt.date]:
    first, last = rows[0][0], rows[-1][0]
    days = (last - first).days
    length = max(7, int(days * rng.uniform(0.15, 0.5)))
    begin = first + dt.timedelta(days=rng.randint(0, max(0, days - length)))
    return begin, begin + dt.timedelta(days=length)


def _question(rng: random.Random, variant: str, rows, labels, users, level):
    """The question, its gold answer and its grading, with the blind guess (spec §3)."""
    templates = TRAIN_TEMPLATES if variant == "train" else HELDOUT_TEMPLATES
    n, k = len(rows), len(labels)
    n_users_mean = sum(USERS_AT[level]) / 2
    counts = {name: sum(r[2] == name for r in rows) for name in labels}
    by_user = {u: [r for r in rows if r[1] == u] for u in users}
    users = [u for u in users if by_user[u]]
    kinds, weights = list(KINDS), list(KINDS.values())
    for _ in range(50):
        kind = rng.choices(kinds, weights)[0]
        fill = {"labels": _label_list(labels)}
        if kind == "label_count":
            label = rng.choice(labels)
            fill["label"] = label
            answer, grading = counts[label], {"type": "int", "blind": round(n / k)}
        elif kind == "user_label_count":
            user, label = rng.choice(users), rng.choice(labels)
            answer = sum(r[2] == label for r in by_user[user])
            if answer == 0:
                continue
            fill.update(user=user, label=label)
            grading = {"type": "int", "blind": round(n / (k * n_users_mean))}
        elif kind == "labels_over":
            ordered = sorted(counts.values())
            gaps = [(ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1) if ordered[i + 1] - ordered[i] >= 2]
            if not gaps:
                continue
            _, i = rng.choice(sorted(gaps)[-3:])
            threshold = (ordered[i] + ordered[i + 1]) // 2
            answer = sorted(name for name in labels if counts[name] > threshold)
            fill["threshold"] = threshold
            grading = {"type": "list", "item": "entity", "blind": list(labels)}
        elif kind == "users_with_label":
            label = rng.choice(labels)
            per_user = sorted((sum(r[2] == label for r in by_user[u]) for u in users), reverse=True)
            cut = max(2, per_user[max(1, int(len(per_user) * rng.uniform(0.2, 0.5))) - 1])
            answer = sorted(u for u in users if sum(r[2] == label for r in by_user[u]) >= cut)
            if not answer or len(answer) == len(users):
                continue
            fill.update(label=label, threshold=cut)
            grading = {"type": "list", "item": "entity", "blind": list(users)}  # every user in the log *(mine)*
        elif kind == "window_label_count":
            label = rng.choice(labels)
            begin, end = _window(rng, rows)
            answer = sum(r[2] == label and begin <= r[0] <= end for r in rows)
            frac = ((end - begin).days + 1) / ((rows[-1][0] - rows[0][0]).days + 1)
            fill.update(label=label, start=_fmt_date(begin), end=_fmt_date(end))
            grading = {"type": "int", "blind": round(n * frac / k)}
        elif kind == "user_window_count":
            user = rng.choice(users)
            begin, end = _window(rng, rows)
            answer = sum(begin <= r[0] <= end for r in by_user[user])
            if answer == 0:
                continue
            frac = ((end - begin).days + 1) / ((rows[-1][0] - rows[0][0]).days + 1)
            fill.update(user=user, start=_fmt_date(begin), end=_fmt_date(end))
            grading = {"type": "int", "blind": round(n * frac / n_users_mean)}
        else:  # most_active_user
            sizes = sorted((len(by_user[u]) for u in users), reverse=True)
            if len(sizes) > 1 and sizes[0] == sizes[1]:
                continue
            answer = max(users, key=lambda u: len(by_user[u]))
            grading = {"type": "entity"}
        grading["kind"] = kind
        grading["code_only"] = kind in CODE_ONLY_KINDS
        return rng.choice(templates[kind]).format(**fill), answer, grading
    raise RuntimeError("could not draw a well-posed question")  # never seen in testing


def make_task(seed: int, family: str, level: int = 1, task_id: str | None = None) -> Task:
    """One task from a seed. The task id never contains the seed (H3)."""
    task_id = task_id or f"{family}-L{level}-{secrets.token_hex(4)}"
    return _build(random.Random(seed), family, level, task_id)
