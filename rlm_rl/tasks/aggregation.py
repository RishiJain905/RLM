"""OOLONG-style aggregation tasks over labelled texts (plan step 7; 45% of the production mix after decision 7c).

OOLONG-synth itself fails the licence bar (decision 7b), so this generator rebuilds its method on permissive
data. A context is a log of short texts, each with a date and a user id but no label. The question asks for
an aggregate over the labels (how many messages are about travel, which users filed at least 5 provisions
about taxes), so the model has to classify every line by reading it, then count with code. In v2, 30% of
questions needed no labels at all (dates and users only), so code alone solved them (research §3.1); v3 has none.

Training sources, one per task, each in its own register:
- CLINC150 voice-assistant utterances grouped by domain (CC-BY-3.0)
- MASSIVE voice-assistant utterances grouped by scenario (CC-BY-4.0)
- LEDGAR contract provisions grouped by clause type (CC-BY-4.0), long lines of legal text
- arXiv paper titles grouped by archive (CC0), short lines of scientific text
LEDGAR and arXiv were added when decision 7c raised this family from 30% to 45%, so the bigger share isn't
spent on one kind of text. Labels are coarse (a domain, a clause type, an archive), so each has hundreds to
thousands of distinct texts, and a top-level context repeats almost none.

Per seed: the source, the label set, the label mix, the users and their ids, the date range, the line format
and the question all change (research §3.1, renamed entities). The answer is never in the context.

Families:
- `aggregation`: the training family, on the sources' train and validation splits.
- `aggregation_heldout`: held-out test only (H7). A source never trained on (HuffPost headlines), with
  separately written line formats and question templates.

Levels go from the 20-step tasks' size (~35K characters) to ~400K, which ~20 sub-calls of ~20K characters
cover (spec §6). Each level up also adds labels and users.

Generator versions. `aggregation-v3` (2026-10-18, the step 12 plan's "aggregation v2"; see the v3 section below)
is what training draws: only kinds that need a semantic label for every line, the denser OOLONG / OOLONG-Pairs
kinds (pairs of users, first and last dates, a user's top topic), OOLONG's count credit, and no line that names
a label. `aggregation-v2` is kept unchanged for the eval sets frozen before v3, which rebuild their tasks from
private seeds (`FROZEN_EVAL_SETS`), so their numbers stay comparable. One fix reaches v2 too: user ids no longer
follow the process's string hash seed (`_user_ids`), so a v2 task is now the same in every process; before, each
process gave the users other ids (the same task otherwise).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import math
import random
import re
import secrets
import statistics
from dataclasses import dataclass
from typing import Callable

from .. import data, reward
from ..env import Task

GENERATOR_VERSION = "aggregation-v3"  # v3: dense kinds, count credit, re-draws (step 12); v2 added LEDGAR and arXiv (7c)
LEGACY_VERSION = "aggregation-v2"
# Eval sets built before v3. tasks.fixed_set names their tasks "<set>-<family>-L<level>-<i>", and a task id with one of
# these prefixes is built by the v2 generator, so selection-v1/v2/v3 and heldout-test-v1/v2 rebuild the same tasks from
# their private seeds. Every other task (training draws, and any set built later, such as selection-v4) is v3.
FROZEN_EVAL_SETS = ("selection-v1", "selection-v2", "selection-v3-extra", "heldout-test-v1")
FAMILIES = ("aggregation", "aggregation_heldout")
TRAIN_SOURCES = ("clinc150", "clinc150-domains", "massive", "ledgar", "arxiv-titles")
HELDOUT_SOURCES = ("huffpost",)

# Characters per context at each level: (fewest, most), with the label count and the user range.
LEVELS = {1: (32_000, 40_000), 2: (70_000, 90_000), 3: (130_000, 170_000), 4: (230_000, 290_000),
          5: (360_000, 420_000)}
LABELS_AT = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}
USERS_AT = {1: (6, 10), 2: (10, 16), 3: (14, 24), 4: (20, 32), 5: (26, 40)}
MAX_LEVEL = max(LEVELS)


@dataclass(frozen=True)
class Source:
    """One labelled text collection: its labels with descriptions, the pairs too close to ask about together,
    how a context introduces it, and what its lines are called in a question."""

    name: str
    pools: Callable[[], dict]
    labels: dict  # label -> description shown in the question
    incompatible: tuple = ()  # label pairs never asked about in the same task
    headers: tuple = ()
    item: str = "message"
    items: str = "messages"
    verb: str = "sent"  # past tense, for "sent by user X"
    verb_base: str = "send"
    weight: float = 1.0  # how often a training task draws this source *(mine)*


CLINC = Source(
    "clinc150", data.clinc_domain_pools,
    {"banking": "bank accounts: balances, transactions, transfers, bills and payments, routing numbers, PINs, "
                "frozen accounts, fraud",
     "credit_cards": "credit cards: limits, APR, rewards, credit scores, card fees, new, lost, damaged or declined cards",
     "kitchen_and_dining": "food: recipes, ingredients, calories, cooking times, restaurants and reservations",
     "home": "the household: shopping and to-do lists, reminders, calendar, orders, smart home, playing music",
     "auto_and_commute": "cars and driving: fuel, oil, tyres, maintenance, traffic, directions, distances, rides",
     "travel": "trips: flights, hotels, car rental, luggage, visas, vaccines, plugs, exchange rates, translation, "
               "time zones",
     "utility": "quick tools: weather, time, date, alarms, timers, calls, texts, calculator, conversions, spelling, "
                "definitions, dice, coin flips, finding the phone",
     "work": "the job: meetings, PTO, holidays, payday, direct deposit, income, taxes, W-2, 401k, insurance",
     "small_talk": "chit-chat with the assistant: greetings, thanks, goodbyes, jokes, fun facts, questions about the "
                   "assistant itself",
     "meta": "controlling the assistant: renaming it or the user, its accent, speed, volume, language, whisper mode, "
             "settings, syncing, yes/no/maybe/cancel/repeat"},
    (("banking", "credit_cards"), ("small_talk", "meta")),
    ("Messages sent to a voice assistant, one per line: date, user id, message.",
     "Assistant request log. Each line has a date, the id of the user who sent it, and what they said."),
    weight=0.30)

MASSIVE = Source(
    # MASSIVE's `general` and `play` scenarios overlap the others too much to grade fairly, so they're left out.
    "massive", data.massive_scenario_pools,
    {"alarm": "setting, changing or removing alarms", "audio": "device volume and audio settings",
     "calendar": "calendar events, meetings and reminders", "cooking": "recipes and cooking instructions",
     "datetime": "asking the current time or date, time zones", "email": "reading, writing or managing email and contacts",
     "iot": "smart home devices: lights, plugs, vacuum, coffee machine", "lists": "creating and editing lists",
     "music": "music preferences, song information, likes and dislikes", "news": "news headlines and topics",
     "qa": "factual questions, definitions, maths, currency, stocks",
     "recommendation": "recommendations for places, events, movies",
     "social": "social media posts and complaints to companies", "takeaway": "ordering takeaway food and delivery",
     "transport": "trains, taxis, tickets and traffic", "weather": "the weather and forecasts"},
    (("alarm", "datetime"), ("calendar", "alarm"), ("music", "audio"), ("transport", "travel"), ("news", "qa")),
    CLINC.headers, weight=0.20)

LEDGAR = Source(
    "ledgar", data.ledgar_pools,
    {"Governing Laws": "which jurisdiction's law governs the agreement",
     "Counterparts": "signing in separate copies, including electronic or faxed signatures",
     "Notices": "how and where formal notices must be delivered",
     "Entire Agreements": "the agreement replaces all earlier understandings between the parties",
     "Severability": "an invalid provision doesn't void the rest of the agreement",
     "Survival": "which obligations continue after the agreement ends",
     "Amendments": "how the agreement may be changed",
     "Assignments": "transferring rights or obligations to another party",
     "Expenses": "who pays the costs and fees of the transaction",
     "Taxes": "tax payments, withholding and tax treatment",
     "Insurances": "insurance coverage a party must keep",
     "Terminations": "when and how the agreement can be ended",
     "Confidentiality": "keeping information secret",
     "Indemnifications": "compensating the other party for losses or claims",
     "Waiver Of Jury Trials": "giving up the right to a trial by jury",
     "Base Salary": "an employee's annual base pay"},
    (("Waiver Of Jury Trials", "Governing Laws"), ("Amendments", "Entire Agreements"), ("Terminations", "Survival"),
     ("Expenses", "Indemnifications"), ("Expenses", "Taxes"), ("Base Salary", "Taxes")),
    ("Contract provisions filed by companies, one per line: date filed, the filer's user id, the provision.",
     "A clause library. Each line has the date a clause was filed, the id of the user who filed it, and its text."),
    item="provision", items="provisions", verb="filed", verb_base="file", weight=0.25)

ARXIV = Source(
    "arxiv-titles", data.arxiv_title_pools,
    {"mathematics": "pure and applied mathematics", "astrophysics": "stars, galaxies, cosmology and the universe",
     "condensed matter physics": "solids, materials, superconductors, soft matter",
     "computer science": "algorithms, software, machine learning, networks",
     "high-energy physics": "particles, quantum field theory, colliders, string theory",
     "quantum physics": "quantum information, quantum optics, quantum foundations",
     "general relativity": "gravitation, black holes, gravitational waves",
     "nuclear physics": "atomic nuclei and nuclear matter",
     "quantitative biology": "genes, cells, neurons, populations and evolution",
     "statistics": "statistical methods, inference, data analysis",
     "quantitative finance": "markets, pricing, portfolios, risk",
     "nonlinear sciences": "chaos, solitons, self-organising and complex systems"},
    (("high-energy physics", "general relativity"), ("high-energy physics", "nuclear physics"),
     ("high-energy physics", "astrophysics"), ("astrophysics", "general relativity"),
     ("condensed matter physics", "quantum physics"), ("mathematics", "statistics"),
     ("computer science", "statistics"), ("computer science", "mathematics"), ("quantitative finance", "statistics"),
     ("nonlinear sciences", "condensed matter physics"), ("nonlinear sciences", "mathematics")),
    ("Papers posted to a preprint server, one per line: date posted, the submitter's user id, the title.",
     "Preprint submissions. Each line has the date, the id of the user who posted the paper, and its title."),
    item="paper", items="papers", verb="posted", verb_base="post", weight=0.25)

HELDOUT = Source(
    "huffpost", data.huffpost_pools,
    {"SPORTS": "sport", "FOOD & DRINK": "food, drink and recipes", "TRAVEL": "travel and destinations",
     "TECH": "technology", "RELIGION": "religion and faith", "WEDDINGS": "weddings", "SCIENCE": "science",
     "BUSINESS": "business and the economy", "CRIME": "crime", "ENTERTAINMENT": "film, TV, music and celebrities",
     "ENVIRONMENT": "the environment and climate", "EDUCATION": "schools and education"},
    (("BUSINESS", "TECH"), ("ENVIRONMENT", "SCIENCE"), ("FOOD & DRINK", "TRAVEL")),
    ("Headlines shared by readers of a news app, one per line, with the date and the reader's account.",),
    item="headline", items="headlines", verb="posted", verb_base="post")

TRAIN_SET = (CLINC, MASSIVE, LEDGAR, ARXIV)
DESCRIPTIONS = {label: desc for src in TRAIN_SET + (HELDOUT,) for label, desc in src.labels.items()}

# v2's question kinds and how often each is drawn. The last two need no labels, so code alone answers them.
# The code-only kinds get 30%, so the whole production mix stays near research §3.1's ~20% code-solvable tasks.
# v3's kinds are KINDS, in the v3 section below.
KINDS_V2 = {"label_count": 0.22, "user_label_count": 0.12, "labels_over": 0.13, "users_with_label": 0.08,
            "window_label_count": 0.15, "user_window_count": 0.18, "most_active_user": 0.12}
CODE_ONLY_KINDS = ("user_window_count", "most_active_user")

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")

# Line formats. Training uses the first three; the held-out test uses its own (H7).
TRAIN_FORMATS = ("{iso} | user {user} | {text}", "[{mon} {day}, {year}] user#{user}: {text}",
                 "{day:02d}/{month:02d}/{year} - id {user} - {text}")
HELDOUT_FORMATS = ("On {weekday} {day} {monname} {year}, account {user} posted: {text}",
                   "{year}.{month:02d}.{day:02d} <{user}> {text}")

# Question templates. {labels} is the label list with descriptions; {item}, {items}, {verb} and {verb_base} name
# the source's lines. Held-out templates are written separately, and scripts/data_check.py checks their n-gram
# overlap with these.
TRAIN_TEMPLATES = {
    "label_count": ["Every {item} belongs to exactly one of these topics:\n{labels}\nHow many {items} are about "
                    "the topic '{label}'? Answer with a single integer.",
                    "Each line is about one of these topics:\n{labels}\nCount the {items} whose topic is '{label}'. "
                    "Answer with a single integer."],
    "user_label_count": ["Every {item} belongs to exactly one of these topics:\n{labels}\nHow many {items} from "
                         "user {user} are about '{label}'? Answer with a single integer.",
                         "Each line is about one of these topics:\n{labels}\nCount the '{label}' {items} {verb} by "
                         "user {user}. Answer with a single integer."],
    "labels_over": ["Every {item} belongs to exactly one of these topics:\n{labels}\nWhich topics have more than "
                    "{threshold} {items}? Answer with a list of topic names.",
                    "Each line is about one of these topics:\n{labels}\nList every topic that more than {threshold} "
                    "{items} are about. Answer with a list of topic names."],
    "users_with_label": ["Every {item} belongs to exactly one of these topics:\n{labels}\nWhich users {verb} at "
                         "least {threshold} {items} about '{label}'? Answer with a list of user ids.",
                         "Each line is about one of these topics:\n{labels}\nList the users with {threshold} or "
                         "more '{label}' {items}. Answer with a list of user ids."],
    "window_label_count": ["Every {item} belongs to exactly one of these topics:\n{labels}\nHow many {items} "
                           "about '{label}' were {verb} from {start} to {end}, both days included? Answer with a "
                           "single integer.",
                           "Each line is about one of these topics:\n{labels}\nBetween {start} and {end} "
                           "(inclusive), how many '{label}' {items} were {verb}? Answer with a single integer."],
    "user_window_count": ["How many {items} did user {user} {verb_base} from {start} to {end}, both days included? "
                          "Answer with a single integer.",
                          "Count the {items} user {user} {verb} between {start} and {end} (inclusive). Answer with "
                          "a single integer."],
    "most_active_user": ["Which user {verb} the most {items}? Answer with the user id.",
                         "Give the id of the user with the largest number of {items}. Answer with the user id."],
    # v3 only (the v2 generator never draws these keys)
    "user_window_label_count": ["Every {item} belongs to exactly one of these topics:\n{labels}\nHow many {items} "
                                "about '{label}' did user {user} {verb_base} from {start} to {end}, both days "
                                "included? Answer with a single integer.",
                                "Each line is about one of these topics:\n{labels}\nBetween {start} and {end} "
                                "(inclusive), how many '{label}' {items} were {verb} by user {user}? Answer with a "
                                "single integer."],
    "pairs_both": ["Every {item} belongs to exactly one of these topics:\n{labels}\nWhich pairs of users each {verb} "
                   "at least {threshold} {items} about '{label}' and at least {threshold} {items} about '{label2}'? "
                   "Answer with a list of pairs, each written as the two user ids joined by \"|\", smaller id first.",
                   "Each line is about one of these topics:\n{labels}\nList every pair of users in which both users "
                   "have {threshold} or more '{label}' {items} and {threshold} or more '{label2}' {items}. Write each "
                   "pair as the two user ids joined by \"|\", smaller id first."],
    "pairs_same_top": ["Every {item} belongs to exactly one of these topics:\n{labels}\nA user's main topic is the "
                       "topic most of their {items} are about. Among users {user_list}, which pairs have the same "
                       "main topic? Answer with a list of pairs, each written as the two user ids joined by \"|\", "
                       "smaller id first.",
                       "Each line is about one of these topics:\n{labels}\nConsider only users {user_list}. List "
                       "every pair of them whose most common topic is the same. Write each pair as the two user ids "
                       "joined by \"|\", smaller id first."],
    "timeline_first": ["Every {item} belongs to exactly one of these topics:\n{labels}\nOn what date did user {user} "
                       "first {verb_base} a {item} about '{label}'? Answer with the date as YYYY-MM-DD.",
                       "Each line is about one of these topics:\n{labels}\nGive the date of the first '{label}' "
                       "{item} {verb} by user {user}. Answer with the date as YYYY-MM-DD."],
    "timeline_last": ["Every {item} belongs to exactly one of these topics:\n{labels}\nOn what date did user {user} "
                      "last {verb_base} a {item} about '{label}'? Answer with the date as YYYY-MM-DD.",
                      "Each line is about one of these topics:\n{labels}\nGive the date of the most recent "
                      "'{label}' {item} {verb} by user {user}. Answer with the date as YYYY-MM-DD."],
    "top_topic": ["Every {item} belongs to exactly one of these topics:\n{labels}\nWhich topic are most of the "
                  "{items} {verb} by user {user} about? Answer with the topic name.",
                  "Each line is about one of these topics:\n{labels}\nWhat is the most common topic among user "
                  "{user}'s {items}? Answer with the topic name."],
    "second_topic": ["Every {item} belongs to exactly one of these topics:\n{labels}\nWhich topic is the second most "
                     "common among the {items} {verb} by user {user}? Answer with the topic name.",
                     "Each line is about one of these topics:\n{labels}\nRank the topics of user {user}'s {items} "
                     "by how many {items} each has. Which topic comes second? Answer with the topic name."],
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
    # v3 only
    "user_window_label_count": ["Classify each headline into one of the following categories:\n{labels}\nFor "
                                "account {user}, counting posts dated {start} through {end}, what is the total "
                                "number of '{label}' headlines? Reply with one integer."],
    "pairs_both": ["Classify each headline into one of the following categories:\n{labels}\nFind every "
                   "two-account combination where each account posted {threshold} or more '{label}' headlines and "
                   "{threshold} or more '{label2}' headlines. Reply with the combinations, comma separated, each as "
                   "the lower account number, a vertical bar, then the higher account number."],
    "pairs_same_top": ["Classify each headline into one of the following categories:\n{labels}\nAn account's leading "
                       "category is the one most of its headlines fall into. Of accounts {user_list}, which "
                       "two-account combinations share a leading category? Reply with the combinations, comma "
                       "separated, each as the lower account number, a vertical bar, then the higher account "
                       "number."],
    "timeline_first": ["Classify each headline into one of the following categories:\n{labels}\nWhat was the earliest "
                       "day account {user} posted a '{label}' headline? Reply in the form YYYY-MM-DD."],
    "timeline_last": ["Classify each headline into one of the following categories:\n{labels}\nWhat was the latest "
                      "day account {user} posted a '{label}' headline? Reply in the form YYYY-MM-DD."],
    "top_topic": ["Classify each headline into one of the following categories:\n{labels}\nFor account {user}, which "
                  "category holds the largest number of its headlines? Reply with the category name."],
    "second_topic": ["Classify each headline into one of the following categories:\n{labels}\nFor account {user}, "
                     "which category holds the second largest number of its headlines? Reply with the category "
                     "name."],
}


def _variant(family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"unknown aggregation family {family!r}")
    return "heldout" if family.endswith("_heldout") else "train"


_MEDIAN_CHARS: dict[str, float] = {}


def median_chars(source: Source) -> float:
    """The median length of a source's texts, which sets how many lines fill a level."""
    if source.name not in _MEDIAN_CHARS:
        pools = source.pools()
        _MEDIAN_CHARS[source.name] = statistics.median(len(t) for label in source.labels for t in pools.get(label, ()))
    return _MEDIAN_CHARS[source.name]


def _choose(source: Source, rng: random.Random, k: int, need: int) -> dict[str, tuple[str, ...]] | None:
    """k compatible labels from one source, each with at least `need` texts, or None if it hasn't enough."""
    pools = source.pools()
    blocked = {frozenset(pair) for pair in source.incompatible}
    usable = [label for label in source.labels if len(pools.get(label, ())) >= need]
    rng.shuffle(usable)
    chosen: list[str] = []
    for label in usable:
        if all(frozenset((label, other)) not in blocked for other in chosen):
            chosen.append(label)
        if len(chosen) == k:
            return {label: pools[label] for label in chosen}
    return None


def _render_line(fmt: str, d: dt.date, user: str, text: str) -> str:
    return fmt.format(iso=d.isoformat(), mon=_MONTHS[d.month - 1][:3], monname=_MONTHS[d.month - 1],
                      weekday=d.strftime("%A"), day=d.day, month=d.month, year=d.year, user=user, text=text)


def _source_and_labels(rng: random.Random, variant: str, fmt: str, budget: int, k: int):
    """A source drawn by weight (falling back to the others if it can't fill k labels), and its labels' pools."""
    order = [HELDOUT] if variant == "heldout" else []
    if not order:
        left = list(TRAIN_SET)
        while left:  # a weighted draw without replacement
            pick = rng.choices(left, [s.weight for s in left])[0]
            order.append(pick)
            left.remove(pick)
    for source in order:
        line = len(_render_line(fmt, dt.date(2024, 9, 28), "12345", "")) + median_chars(source) + 1
        chosen = _choose(source, rng, k, need=int(1.3 * budget / line / k))
        if chosen:
            return source, chosen
    raise ValueError(f"no source has {k} compatible labels with enough texts")


def _user_ids(rng: random.Random, n: int) -> list[str]:
    ids: set[str] = set()
    while len(ids) < n:
        ids.add(str(rng.randint(10_000, 99_999)))
    # Shuffled from sorted order. Until 2026-10-18 this shuffled the set's own order, which follows the process's
    # string hash seed, so each process gave the users other ids (the same tasks up to renaming the users).
    return sorted(sorted(ids), key=lambda _: rng.random())


def _fmt_date(d: dt.date) -> str:
    return f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"


def _build(rng: random.Random, family: str, level: int, task_id: str) -> Task:
    variant = _variant(family)
    if level not in LEVELS:
        raise ValueError(f"no aggregation level {level} (levels 1-{MAX_LEVEL})")
    fmt = rng.choice(TRAIN_FORMATS if variant == "train" else HELDOUT_FORMATS)
    budget = rng.randint(*LEVELS[level])
    source, pools = _source_and_labels(rng, variant, fmt, budget, LABELS_AT[level])
    header = rng.choice(source.headers)
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

    question, answer, grading = _question(rng, variant, source, rows, labels, users, level)
    grading["source"] = source.name
    return Task(task_id, family, question, context, answer=answer, level=level, grading=grading)


def _label_list(labels: list[str]) -> str:
    return "\n".join(f"- {name}: {DESCRIPTIONS[name]}" for name in labels)


def _window(rng: random.Random, rows) -> tuple[dt.date, dt.date]:
    first, last = rows[0][0], rows[-1][0]
    days = (last - first).days
    length = max(7, int(days * rng.uniform(0.15, 0.5)))
    begin = first + dt.timedelta(days=rng.randint(0, max(0, days - length)))
    return begin, begin + dt.timedelta(days=length)


def _question(rng: random.Random, variant: str, source: Source, rows, labels, users, level):
    """The question, its gold answer and its grading, with the blind guess (spec §3)."""
    templates = TRAIN_TEMPLATES if variant == "train" else HELDOUT_TEMPLATES
    words = {"item": source.item, "items": source.items, "verb": source.verb, "verb_base": source.verb_base}
    n, k = len(rows), len(labels)
    n_users_mean = sum(USERS_AT[level]) / 2
    counts = {name: sum(r[2] == name for r in rows) for name in labels}
    by_user = {u: [r for r in rows if r[1] == u] for u in users}
    users = [u for u in users if by_user[u]]
    kinds, weights = list(KINDS_V2), list(KINDS_V2.values())
    for _ in range(50):
        kind = rng.choices(kinds, weights)[0]
        fill = {"labels": _label_list(labels), **words}
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


# ----------------------------------------------------------------------------- v3: dense kinds (step 12)
#
# Why: in prod-v1 the code-only kinds were solved by regex, and the semantic kinds rarely earned reward. Exact counts
# over hundreds of lines are brittle, near-miss credit collapsed when the metadata-only blind guess (lines / labels)
# was close, and many lines named their own label (89% of MASSIVE `alarm` texts say "alarm", 99% of LEDGAR
# `Counterparts` provisions say "counterparts"), so a regex could count them. The RLM paper (arXiv 2512.24601) finds
# sub-calls pay off on information-dense tasks: OOLONG, where every line needs a semantic label, and OOLONG-Pairs,
# a quadratic aggregation over users, where RLMs gain the most.
#
# What changes:
# - Kinds: no code-only kind. The five semantic v2 kinds stay; three dense ones are added: `pairs` (pairs of users
#   who both wrote enough about two topics, or whose main topics match), `timeline` (the first or last date a user
#   wrote about a topic) and `top_topic` (a user's most, or second most, common topic, only with a clear lead).
# - Lines: no line names any label the question lists (`label_pattern`); texts that do are left out of the task's
#   pools (`clean_pools`). Topic words other than the label names stay: telling the topic is the semantic work.
# - Users: each writes about 2-4 of the task's topics, at their own rates, and minor topics may start late or stop
#   early. So per-user questions aren't answered by the global mix, how active a user is (which code can count) says
#   little about what they write, and a topic's first or last date is spread over the user's dates.
# - Counts: graded by reward.py's "count" type (OOLONG: 0.8 x 0.75^|err|). A task whose metadata-only guesses (0,
#   and the scope's lines / labels) would earn more than BLIND_MAX is re-drawn from a deterministic sub-seed.
# - Lists: the blind guess is the best "most active users" guess (any prefix of the users by line count, which code
#   alone gives), so the calibrated list credit pays nothing for it; a task where it reaches LIST_BLIND_F1_MAX is
#   re-drawn.
# - Sizes: a count depends on SCOPE_LINES[level] lines, few enough to label carefully with sub-calls and too many to
#   eyeball; set questions avoid thresholds that a single mislabelled line would flip (FRAGILE_MAX).

KINDS = {"label_count": 0.12, "user_label_count": 0.12, "window_label_count": 0.08, "labels_over": 0.06,
         "users_with_label": 0.08, "pairs": 0.26, "timeline": 0.14, "top_topic": 0.14}
COUNT_KINDS = ("label_count", "user_label_count", "window_label_count")

# Labels per level. Level 5 keeps level 4's 7: with the texts that name a label left out, LEDGAR has no 8 compatible
# labels with enough texts and arXiv's conflict graph has no 8-set, so 8 would leave level 5 to CLINC alone.
LABELS_V3 = {1: 4, 2: 5, 3: 6, 4: 7, 5: 7}
LABEL_SHUFFLES = 30
LABEL_SPREAD = 0.6  # sigma of the log-normal topic weights
TOPICS_PER_USER = {4: (2, 3), 5: (2, 3), 6: (2, 4), 7: (2, 4), 8: (2, 5)}  # by label count
# Lines a count depends on (whole log, a window, a user, or a user in a window), by level. Wide enough that no fixed
# "share of the scope" guess earns much (the best one ~0.07-0.10 per level, tests/test_aggregation_v3.py), few enough
# to label carefully with sub-calls.
SCOPE_LINES = {1: (120, 300), 2: (150, 450), 3: (180, 600), 4: (220, 750), 5: (260, 900)}
USER_WINDOW_SHARE = 0.3  # how often a user count is asked over a date window when the user's lines allow one
MIN_USER_LINES = {1: 10, 2: 15, 3: 20, 4: 25, 5: 30}  # a user a top-topic question can be about
PAIR_MEMBERS = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9}  # most users in a pairs answer (so at most 10-36 pairs), at least 3
PAIR_THRESHOLDS = {1: (1, 2), 2: (1, 2), 3: (2, 3), 4: (2, 3), 5: (2, 3)}
BLIND_MAX = 0.05  # what a metadata-only guess may earn on a count
LIST_BLIND_F1_MAX = 0.4
FRAGILE_MAX = 1  # users whose membership one mislabelled line would flip
TOP_MARGIN = 0.05  # a top topic leads the next by max(2, this share of the user's lines)
LABELS_OVER_GAP = 0.02  # labels_over's threshold sits in a gap of max(2, this share of the lines)
LATE_START, EARLY_STOP = 0.5, 0.5  # how often a user's minor topic starts late, or stops early
# Timeline: the user wrote on at least TIMELINE_DATES[level] dates, and on at least TIMELINE_OTHER_DATES before the
# first (after the last) one. The answer's place among the user's dates is drawn at random (TIMELINE_SLACK), so no
# fixed "the user's n-th date" guess is often right.
TIMELINE_DATES = {1: 15, 2: 20, 3: 25, 4: 30, 5: 30}
TIMELINE_OTHER_DATES = 3
TIMELINE_SLACK = 0.1
MAX_DRAWS = 60
QUESTION_TRIES = 3


def version_for(task_id: str | None) -> str:
    """The generator a task id gets: v2 for the eval sets frozen before v3, v3 for everything else."""
    if task_id and any(task_id.startswith(f"{name}-") for name in FROZEN_EVAL_SETS):
        return LEGACY_VERSION
    return GENERATOR_VERSION


def _sub_seed(seed: int, tag) -> int:
    digest = hashlib.sha256(f"{GENERATOR_VERSION}:{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def label_pattern(names) -> re.Pattern:
    """Any of these label names as words in a text: case and the punctuation between words ignored, singular or
    plural, '&' read as 'and'."""
    alternatives = []
    for name in sorted(names):
        words = re.findall(r"[a-z0-9]+|&", name.lower())
        *head, last = ["&" if w == "and" else w for w in words]
        forms = {last, last + "s", last + "es"}
        if last.endswith("y"):
            forms.add(last[:-1] + "ies")
        if last.endswith("ies"):
            forms.add(last[:-3] + "y")
        if last.endswith("s") and not last.endswith(("ss", "us", "is", "ws")):  # "news" is not the plural of "new"
            forms.update({last[:-1], last[:-2]} if last.endswith("es") else {last[:-1]})
        parts = [r"(?:and|&)" if w == "&" else re.escape(w) for w in head]
        tail = "(?:" + "|".join(re.escape(f) for f in sorted(forms, key=lambda f: (-len(f), f))) + ")"
        alternatives.append(r"\W+".join(parts + [tail]))
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alternatives) + r")(?![a-z0-9])", re.I)


_MENTIONS: dict[str, dict[str, list]] = {}
_OWN_SIZE: dict[str, dict[str, int]] = {}
_OWN_MEDIAN: dict[str, float] = {}


def _mentions(source: Source) -> dict[str, list[tuple[str, frozenset]]]:
    """A source's texts by label, each with the source's label names it mentions (`label_pattern`). Built once."""
    if source.name not in _MENTIONS:
        pools, patterns = source.pools(), {name: label_pattern([name]) for name in source.labels}
        _MENTIONS[source.name] = {
            label: [(t, frozenset(name for name, p in patterns.items() if p.search(t))) for t in pools.get(label, ())]
            for label in source.labels}
        own = {label: [t for t, named in entries if label not in named]
               for label, entries in _MENTIONS[source.name].items()}
        _OWN_SIZE[source.name] = {label: len(texts) for label, texts in own.items()}
        _OWN_MEDIAN[source.name] = statistics.median(len(t) for texts in own.values() for t in texts)
    return _MENTIONS[source.name]


def clean_pools(source: Source, labels) -> dict[str, tuple[str, ...]]:
    """v3's pools for a task about `labels`: each label's texts, minus any that names one of `labels`."""
    chosen, entries = frozenset(labels), _mentions(source)
    return {label: tuple(t for t, named in entries[label] if not named & chosen) for label in labels}


@dataclass
class _Log:
    source: Source
    labels: list  # in the order the question lists them
    rows: list  # (date, user, label, text), by date
    by_user: dict  # user -> rows, by date
    active: list  # users with a line, most lines first: what code alone can count
    context: str


def _weighted_sample(rng: random.Random, items: list, weights: list, k: int) -> list:
    items, weights, out = list(items), list(weights), []
    for _ in range(min(k, len(items))):
        i = rng.choices(range(len(items)), weights)[0]
        out.append(items.pop(i))
        weights.pop(i)
    return out


def _log_v3(rng: random.Random, variant: str, level: int) -> _Log:
    fmt = rng.choice(TRAIN_FORMATS if variant == "train" else HELDOUT_FORMATS)
    budget = rng.randint(*LEVELS[level])
    k = LABELS_V3[level]
    order = [HELDOUT] if variant == "heldout" else []
    if not order:
        left = list(TRAIN_SET)
        while left:  # a weighted draw without replacement, as in v2
            pick = rng.choices(left, [s.weight for s in left])[0]
            order.append(pick)
            left.remove(pick)
    for source in order:
        _mentions(source)
        line = len(_render_line(fmt, dt.date(2024, 9, 28), "12345", "")) + _OWN_MEDIAN[source.name] + 1
        need = int(1.3 * budget / line / k)
        blocked = {frozenset(pair) for pair in source.incompatible}
        usable = [label for label in source.labels if _OWN_SIZE[source.name][label] >= need]
        found = None
        for _ in range(LABEL_SHUFFLES):  # a few random greedy passes: some conflict graphs have few k-sets
            rng.shuffle(usable)
            labels: list[str] = []
            for label in usable:
                if len(labels) < k and all(frozenset((label, other)) not in blocked for other in labels):
                    labels.append(label)
            if len(labels) == k:
                pools = clean_pools(source, labels)  # also without texts that name another chosen label
                if all(len(pools[name]) >= need for name in labels):
                    found = labels
                    break
        if found:
            break
    else:
        raise ValueError(f"no source has {k} compatible labels with enough texts")
    labels = found

    weight = {name: math.exp(rng.gauss(0, LABEL_SPREAD)) for name in labels}
    users = _user_ids(rng, rng.randint(*USERS_AT[level]))
    user_w = [1 / (i + 1) ** 0.7 for i in range(len(users))]  # a few heavy users, a long tail
    start = dt.date(2019, 1, 1) + dt.timedelta(days=rng.randint(0, 6 * 365))
    span = rng.randint(120, 540)
    lo, hi = TOPICS_PER_USER[k]
    prefs = {}
    for user in users:
        topics = _weighted_sample(rng, labels, [weight[name] for name in labels], rng.randint(lo, hi))
        rates = [weight[name] * rng.uniform(0.5, 2.0) for name in topics]
        main = max(range(len(topics)), key=lambda i: rates[i])
        periods = []  # the days each topic is active: the main one all along, others may start late or stop early
        for i in range(len(topics)):
            begin = rng.randint(0, int(0.7 * span)) if i != main and rng.random() < LATE_START else 0
            end = rng.randint(begin + int(0.2 * span), span) if i != main and rng.random() < EARLY_STOP else span
            periods.append((begin, end))
        prefs[user] = (topics, rates, periods)
    header = rng.choice(source.headers)

    remaining = {name: list(rng.sample(pools[name], len(pools[name]))) for name in labels}
    rows, size = [], len(header) + 1
    while size < budget:
        user = rng.choices(users, user_w)[0]
        day = rng.randint(0, span)
        topics, rates, periods = prefs[user]
        live = [i for i, (begin, end) in enumerate(periods) if begin <= day <= end]
        label = topics[rng.choices(live, [rates[i] for i in live])[0]]
        if not remaining[label]:
            remaining[label] = list(rng.sample(pools[label], len(pools[label])))  # a repeat only after the pool runs out
        text = remaining[label].pop()
        row = (start + dt.timedelta(days=day), user, label, text)
        rows.append(row)
        size += len(_render_line(fmt, row[0], row[1], text)) + 1
    rows.sort(key=lambda r: r[0])
    by_user: dict[str, list] = {}
    for row in rows:
        by_user.setdefault(row[1], []).append(row)
    active = sorted(by_user, key=lambda u: (-len(by_user[u]), u))
    context = "\n".join([header] + [_render_line(fmt, d, u, t) for d, u, _, t in rows])
    return _Log(source, labels, rows, by_user, active, context)


def _shuffled(rng: random.Random, items) -> list:
    items = list(items)
    rng.shuffle(items)
    return items


def _count_grading(gold: int, scope: int, k: int) -> dict | None:
    """The "count" grading, or None if a metadata-only guess (0, or the scope's lines / labels) would earn more than
    BLIND_MAX, or the count is 0."""
    grading = {"type": "count", "blind": round(scope / k), "scope_lines": scope}
    if gold == 0 or any(reward.score_answer(g, gold, grading).reward > BLIND_MAX for g in (0, grading["blind"])):
        return None
    return grading


def _scope_window(rows: list, rng: random.Random, lo: int, hi: int):
    """A date window over `rows` (by date) that holds lo to hi of them: (first day, last day, rows inside), or None."""
    if len(rows) < lo:
        return None
    target = rng.randint(lo, min(hi, len(rows)))
    i = rng.randint(0, len(rows) - target)
    begin, end = rows[i][0], rows[i + target - 1][0]
    inside = [r for r in rows if begin <= r[0] <= end]
    return (begin, end, inside) if lo <= len(inside) <= hi else None


def _count_question(rng: random.Random, log: _Log, rows: list):
    """The first label (in random order) whose count over `rows` passes `_count_grading`: (label, gold, grading)."""
    for label in _shuffled(rng, log.labels):
        gold = sum(r[2] == label for r in rows)
        grading = _count_grading(gold, len(rows), len(log.labels))
        if grading:
            return label, gold, grading
    return None


def _ask_window_label_count(rng, log, level):
    window = _scope_window(log.rows, rng, *SCOPE_LINES[level])
    found = window and _count_question(rng, log, window[2])
    if not found:
        return None
    label, gold, grading = found
    return "window_label_count", {"label": label, "start": _fmt_date(window[0]), "end": _fmt_date(window[1])}, \
        gold, grading


def _ask_label_count(rng, log, level):
    if len(log.rows) > SCOPE_LINES[level][1]:  # too many lines to count well: the same count over a window
        return _ask_window_label_count(rng, log, level)
    found = _count_question(rng, log, log.rows)
    if not found:
        return None
    label, gold, grading = found
    return "label_count", {"label": label}, gold, grading


def _ask_user_label_count(rng, log, level):
    lo, hi = SCOPE_LINES[level]
    users = [u for u in log.active if len(log.by_user[u]) >= lo]
    if not users:
        return None
    user = rng.choice(users)
    rows, key, fill = log.by_user[user], "user_label_count", {"user": user}
    if len(rows) > hi or (len(rows) >= 2 * lo and rng.random() < USER_WINDOW_SHARE):  # the user's lines in a window
        window = _scope_window(rows, rng, lo, hi)
        if not window:
            return None
        rows, key = window[2], "user_window_label_count"
        fill.update(start=_fmt_date(window[0]), end=_fmt_date(window[1]))
    found = _count_question(rng, log, rows)
    if not found:
        return None
    label, gold, grading = found
    return key, {**fill, "label": label}, gold, grading


def _ask_labels_over(rng, log, level):
    counts = {name: sum(r[2] == name for r in log.rows) for name in log.labels}
    ordered, k = sorted(counts.values()), len(log.labels)
    gap = max(2, math.ceil(LABELS_OVER_GAP * len(log.rows)))
    options = [i for i in range(k - 1) if ordered[i + 1] - ordered[i] >= gap and 1 <= k - 1 - i <= k // 2]
    if not options:
        return None
    i = rng.choice(options)
    threshold = (ordered[i] + ordered[i + 1]) // 2
    answer = sorted(name for name in log.labels if counts[name] > threshold)
    return "labels_over", {"threshold": threshold}, answer, {"type": "list", "item": "entity", "blind": list(log.labels)}


def _pair(a: str, b: str) -> str:
    return "|".join(sorted((a, b), key=int))


def _pairs_of(users) -> list[str]:
    return sorted(_pair(a, b) for a, b in itertools.combinations(users, 2))


def _best_prefix(order: list, gold: list, expand: Callable[[list], list]) -> tuple[list, float]:
    """The metadata guess a list task is calibrated against: the most active users first (`order`), cut wherever
    the guess scores best against the gold (an oracle cut, so the strongest such guess), expanded to answer items."""
    target, best = set(gold), ([], -1.0)
    for m in range(1, len(order) + 1):
        guess = expand(order[:m])
        f1 = reward.set_f1(set(guess), target)
        if f1 > best[1]:
            best = (guess, f1)
    return best


def _ask_users_with_label(rng, log, level):
    users = log.active
    for label in _shuffled(rng, log.labels):
        per = {u: sum(r[2] == label for r in log.by_user[u]) for u in users}
        for cut in _shuffled(rng, range(2, max(per.values()) + 1)):
            answer = sorted(u for u in users if per[u] >= cut)
            if not 2 <= len(answer) <= max(2, len(users) // 2):
                continue
            if sum(per[u] in (cut - 1, cut) for u in users) > FRAGILE_MAX:
                continue
            blind, f1 = _best_prefix(users, answer, lambda us: sorted(us))
            if f1 > LIST_BLIND_F1_MAX:
                continue
            return "users_with_label", {"label": label, "threshold": cut}, answer, \
                {"type": "list", "item": "entity", "blind": blind, "blind_f1": round(f1, 4)}
    return None


def _ask_pairs_both(rng, log, level):
    users = log.active
    per = {name: {u: sum(r[2] == name for r in log.by_user[u]) for u in users} for name in log.labels}
    for x, y in _shuffled(rng, itertools.combinations(log.labels, 2)):
        for t in _shuffled(rng, PAIR_THRESHOLDS[level]):
            members = [u for u in users if per[x][u] >= t and per[y][u] >= t]
            if not 3 <= len(members) <= PAIR_MEMBERS[level]:
                continue
            fragile = sum(min(per[x][u], per[y][u]) == t if u in members else
                          (per[x][u] == t - 1 and per[y][u] >= t) or (per[y][u] == t - 1 and per[x][u] >= t)
                          for u in users)
            if fragile > FRAGILE_MAX:
                continue
            answer = _pairs_of(members)
            blind, f1 = _best_prefix(users, answer, _pairs_of)
            if f1 > LIST_BLIND_F1_MAX:
                continue
            first, second = (x, y) if rng.random() < 0.5 else (y, x)
            return "pairs_both", {"label": first, "label2": second, "threshold": t}, answer, \
                {"type": "list", "item": "entity", "blind": blind, "blind_f1": round(f1, 4)}
    return None


def _top_order(log: _Log, user: str) -> tuple[list, list]:
    """A user's topics by line count (ties by name), with their counts; topics they never wrote about come last."""
    counts = {name: sum(r[2] == name for r in log.by_user[user]) for name in log.labels}
    order = sorted(log.labels, key=lambda name: (-counts[name], name))
    return order, [counts[name] for name in order]


def _margin(log: _Log, user: str) -> int:
    return max(2, math.ceil(TOP_MARGIN * len(log.by_user[user])))


def _clear_top(log: _Log, user: str, level: int) -> bool:
    """The user has enough lines, and their top topic leads the next by the margin."""
    counts = _top_order(log, user)[1]
    return len(log.by_user[user]) >= MIN_USER_LINES[level] and counts[0] - counts[1] >= _margin(log, user)


def _ask_pairs_same_top(rng, log, level):
    m = PAIR_MEMBERS[level]
    clear = [u for u in log.active if _clear_top(log, u, level)]
    if len(clear) < m:
        return None
    for _ in range(20):
        listed = sorted(rng.sample(clear, m), key=int)
        top = {u: _top_order(log, u)[0][0] for u in listed}
        answer = sorted(_pair(a, b) for a, b in itertools.combinations(listed, 2) if top[a] == top[b])
        if len(answer) < 2:
            continue
        by_activity = [u for u in log.active if u in top]
        blind, f1 = _best_prefix(by_activity, answer, _pairs_of)
        if f1 > LIST_BLIND_F1_MAX:
            continue
        return "pairs_same_top", {"user_list": ", ".join(listed)}, answer, \
            {"type": "list", "item": "entity", "blind": blind, "blind_f1": round(f1, 4)}
    return None


def _ask_pairs(rng, log, level):
    askers = [_ask_pairs_both, _ask_pairs_same_top]
    for ask in _shuffled(rng, askers):
        posed = ask(rng, log, level)
        if posed:
            return posed
    return None


def _ask_timeline(rng, log, level):
    """The first (or last) date a user wrote about a topic. The answer's place among the user's dates, counted from
    the start (or the end), is drawn first, as a share of their dates, and the question is a (user, topic) whose
    answer sits within TIMELINE_SLACK of it."""
    first = rng.random() < 0.5
    want = rng.uniform(0.0, 1.0)
    candidates = []
    for user in log.active:
        rows = log.by_user[user]
        dates = sorted({r[0] for r in rows})
        if len(dates) < TIMELINE_DATES[level]:
            continue
        seq = dates if first else dates[::-1]
        for label in log.labels:
            days = [r[0] for r in rows if r[2] == label]
            if not days:
                continue
            place = seq.index(min(days) if first else max(days))
            if place >= TIMELINE_OTHER_DATES and abs(place / (len(seq) - 1) - want) <= TIMELINE_SLACK:
                candidates.append((user, label, seq[place], seq[0]))
    if not candidates:
        return None
    user, label, target, blind = rng.choice(candidates)
    # The blind guess is the user's own first (last) date, which code alone finds.
    return f"timeline_{'first' if first else 'last'}", {"user": user, "label": label}, target.isoformat(), \
        {"type": "entity", "blind": blind.isoformat()}


def _ask_top_topic(rng, log, level):
    users = [u for u in log.active if len(log.by_user[u]) >= MIN_USER_LINES[level]]
    rank = 2 if len(log.labels) >= 5 and rng.random() < 0.3 else 1
    for user in _shuffled(rng, users):
        order, counts = _top_order(log, user)
        margin = _margin(log, user)
        if counts[0] - counts[1] < margin or (rank == 2 and (counts[1] - counts[2] < margin or not counts[1])):
            continue
        # The blind guess is every topic the question lists: a name has no partial credit, and copied_answer
        # (detectors.py) leaves the listed topics alone.
        return ("top_topic" if rank == 1 else "second_topic"), {"user": user}, order[rank - 1], \
            {"type": "entity", "blind": list(log.labels)}
    return None


ASKERS = {"label_count": _ask_label_count, "user_label_count": _ask_user_label_count,
          "window_label_count": _ask_window_label_count, "labels_over": _ask_labels_over,
          "users_with_label": _ask_users_with_label, "pairs": _ask_pairs, "timeline": _ask_timeline,
          "top_topic": _ask_top_topic}


def _build_v3(seed: int, family: str, level: int, task_id: str, kind: str | None = None) -> Task:
    """A v3 task. The kind is drawn once from the seed (so the mix is exactly KINDS); the log and question are
    re-drawn from deterministic sub-seeds until the question is well posed and no metadata guess scores."""
    variant = _variant(family)
    if level not in LEVELS:
        raise ValueError(f"no aggregation level {level} (levels 1-{MAX_LEVEL})")
    if kind is None:
        kind = random.Random(_sub_seed(seed, "kind")).choices(list(KINDS), list(KINDS.values()))[0]
    if kind not in KINDS:
        raise ValueError(f"no v3 aggregation kind {kind!r}")
    templates = TRAIN_TEMPLATES if variant == "train" else HELDOUT_TEMPLATES
    for attempt in range(MAX_DRAWS):
        rng = random.Random(_sub_seed(seed, attempt))
        log = _log_v3(rng, variant, level)
        for _ in range(QUESTION_TRIES):
            posed = ASKERS[kind](rng, log, level)
            if not posed:
                continue
            key, fill, answer, grading = posed
            src = log.source
            question = rng.choice(templates[key]).format(
                labels=_label_list(log.labels), item=src.item, items=src.items, verb=src.verb,
                verb_base=src.verb_base, **fill)
            grading.update(kind=kind, template=key, code_only=False, source=src.name, generator=GENERATOR_VERSION,
                           redraws=attempt)
            return Task(task_id, family, question, log.context, answer=answer, level=level, grading=grading)
    raise RuntimeError(f"could not pose a {kind} question in {MAX_DRAWS} draws")


def make_task(seed: int, family: str, level: int = 1, task_id: str | None = None, *, version: str | None = None,
              kind: str | None = None) -> Task:
    """One task from a seed. The task id never contains the seed (H3).

    The generator is v3 unless the task id belongs to an eval set frozen before v3 (`version_for`), or `version`
    names one. `kind` (v3 only) asks for one kind of question instead of drawing it from KINDS.
    """
    task_id = task_id or f"{family}-L{level}-{secrets.token_hex(4)}"
    version = version or version_for(task_id)
    if version == LEGACY_VERSION:
        if kind is not None:
            raise ValueError("the v2 generator draws its own kind")
        return _build(random.Random(seed), family, level, task_id)
    if version != GENERATOR_VERSION:
        raise ValueError(f"unknown aggregation generator {version!r}")
    return _build_v3(seed, family, level, task_id, kind)
