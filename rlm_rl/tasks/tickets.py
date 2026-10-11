"""Support-ticket counting tasks, ported from the 20-step run with difficulty levels (plan step 5).

Each ticket is assembled from interchangeable parts (opener, body, detail, closer), so a context
of a few hundred tickets has almost no repeated text. Problem reports are paraphrased without
shared keywords, and the distractors reuse the same vocabulary (praise about a fast courier,
questions about shipping fees, explicit "no problem with X" statements). Keyword counting gets
it wrong; reading each ticket gets it right.

Two families, as in the 20-step run:
- tickets_all: how many tickets report one kind of problem
- tickets_customer: how many tickets from one customer report any problem

Level 1 is the 20-step run's size (250-380 tickets, ~32-48K characters). Each level up adds
tickets, to ~400K characters at level 5, which ~20 sub-calls of ~20K characters cover (spec §6).
The texts, names and items are unchanged, so level 1 reproduces the 20-step run's task
distribution, and `legacy_tasks()` rebuilds its exact task lists from their seeds.

Everything here runs on the host (H3). The answer is an integer, and the blind guess (spec §3)
is the line count times the target's expected share: what a policy could learn to answer from
the prompt's metadata without reading.
"""

from __future__ import annotations

import random
import secrets

from ..env import Task

GENERATOR_VERSION = "tickets-v2"  # v1 was rlm_rl/archive/ticket_tasks.py (level 1 only, float answers)
FAMILIES = ("tickets_all", "tickets_customer")

# Tickets per context at each level: (fewest, most). ~128 characters a ticket.
LEVELS = {1: (250, 380), 2: (500, 760), 3: (950, 1400), 4: (1600, 2300), 5: (2600, 3400)}
MAX_LEVEL = max(LEVELS)

_OPENERS = ["Hi,", "Hello team,", "Hey there.", "Good morning,", "Hi support,", "Quick note:", "Hello,", "Hiya,",
            "To whom it may concern,", "Hey,", ""]
_CLOSERS = ["Thanks.", "Please advise.", "Appreciate it!", "Cheers.", "Let me know.", "Thank you!", "Regards.",
            "Hope to hear back soon.", "", ""]

_PROBLEMS = {
    "delivery": [
        "the parcel never turned up even though the app claims it was handed over",
        "my {item} arrived with the box caved in and the {item} cracked",
        "the courier left it with a neighbour I've never met and now it's gone",
        "tracking has been frozen at the regional hub for {n} days",
        "only the charger came, the {item} itself wasn't in the box",
        "I took the day off to receive it and nobody ever knocked",
        "the order went to my old flat despite me updating the address",
        "it's now {n} days past the promised date and still nothing",
        "the driver tossed it over the fence and it landed in a puddle",
        "the box I received contained someone else's order",
    ],
    "billing": [
        "my card shows two identical charges for the {item}",
        "there's a mystery {n} dollar line on my statement I never agreed to",
        "you promised money back for the {item} {n} days ago and my bank shows nothing",
        "the plan renewed at a higher price without any notice",
        "I cancelled last month yet another payment went out",
        "the coupon was accepted at checkout but I was charged full price",
        "I was invoiced for express shipping that I didn't pick",
        "the receipt total doesn't match what left my account",
    ],
    "technical": [
        "the app freezes on a white screen whenever I open my orders",
        "the reset link in your email leads to a page that won't load",
        "clicking pay does absolutely nothing on your site",
        "the tracking page throws an error for every order number I enter",
        "I get signed out every few minutes on the website",
        "photo uploads for my {item} review keep failing halfway",
        "the search bar returns nothing, even for products on your homepage",
        "two-factor codes never arrive so I can't log in",
    ],
}

_NON_PROBLEMS = [
    "the courier was lovely and the {item} came a day early",
    "no complaints about shipping this time, it arrived in perfect condition",
    "your billing team sorted out my question in five minutes, great service",
    "the new app version is so much faster, nice work",
    "how much do you charge for express delivery to Canada?",
    "do you offer gift wrapping on the {item}?",
    "can I change the colour of my {item} before it ships?",
    "is there a student discount on subscriptions?",
    "just wanted to say the {item} is fantastic and I'd buy it again",
    "what are your support hours over the holidays?",
    "my payment went through fine, I only need a copy of the invoice for work",
    "I had no trouble logging in, I'd just like to update my email address",
    "the parcel arrived safely, could you tell me how to register the warranty?",
    "will the {item} come back in stock soon?",
]

_DETAILS = ["Order #{order}.", "This is order {order}.", "Ref {order}.", "", "", "My account email is on file.",
            "I've been a customer for {n} years.", "This is the second time this month.", "", "Bought it as a gift."]

_NAMES = ["Maria Lopez", "Wei Chen", "Aisha Khan", "Tom Becker", "Priya Nair", "Lucas Silva", "Emma Brown",
          "Omar Haddad", "Sofia Rossi", "Kenji Sato", "Nina Ivanova", "Jack Wilson", "Grace Kim", "Diego Ramos"]
_ITEMS = ["lamp", "kettle", "backpack", "headphones", "desk chair", "blender", "jacket", "monitor", "toaster", "speaker"]

_LABELS = {"delivery": "the delivery or shipping of an order (late, lost, misdelivered, damaged in transit, or incomplete)",
           "billing": "billing or payment (wrong or duplicate charges, unexpected fees, missing refunds, price changes)",
           "technical": "a technical fault in the app or website (crashes, errors, login or page failures)"}

_KINDS = ["delivery", "billing", "technical", "none"]
_WEIGHTS = [1.0, 0.8, 0.8, 1.6]
# The expected share of tickets that count, which is all the prompt's metadata can tell a policy.
SHARE = {kind: w / sum(_WEIGHTS) for kind, w in zip(_KINDS, _WEIGHTS)}
CUSTOMER_SHARE = (1 - SHARE["none"]) / len(_NAMES)


def _ticket(rng: random.Random, kind: str) -> str:
    fill = {"item": rng.choice(_ITEMS), "n": rng.randint(2, 15), "order": rng.randint(10000, 99999)}
    body = rng.choice(_PROBLEMS[kind] if kind in _PROBLEMS else _NON_PROBLEMS).format(**fill)
    parts = [rng.choice(_OPENERS), body[0].upper() + body[1:] + ".", rng.choice(_DETAILS).format(**fill), rng.choice(_CLOSERS)]
    return " ".join(p for p in parts if p)


def _tickets(rng: random.Random, n: int) -> list[tuple[int, str, str, str]]:
    out = []
    for i in range(n):
        kind = rng.choices(_KINDS, _WEIGHTS)[0]
        out.append((20000 + i, rng.choice(_NAMES), kind, _ticket(rng, kind)))
    return out


def _render(tickets) -> str:
    return "\n".join(f"[ticket {tid}] from {name}: {text}" for tid, name, _, text in tickets)


def _build(rng: random.Random, family: str, level: int, task_id: str) -> Task:
    """One task, drawing from `rng` in the same order as the 20-step generator."""
    if family not in FAMILIES:
        raise ValueError(f"unknown ticket family {family!r}")
    if level not in LEVELS:
        raise ValueError(f"no ticket level {level} (levels 1-{MAX_LEVEL})")
    tickets = _tickets(rng, rng.randint(*LEVELS[level]))
    target = rng.choice(list(_LABELS))
    if family == "tickets_all":
        answer = sum(kind == target for _, _, kind, _ in tickets)
        question = f"How many tickets report a problem with {_LABELS[target]}? Answer with a single integer."
        share = SHARE[target]
    else:
        who = rng.choice(_NAMES)
        answer = sum(name == who and kind != "none" for _, name, kind, _ in tickets)
        question = (f"How many tickets from {who} report an actual problem of any kind (delivery, billing, or "
                    "technical)? Questions, praise, and requests are not problems. Answer with a single integer.")
        share = CUSTOMER_SHARE
    blind = round(len(tickets) * share)  # the line count is in the prompt, and there is one ticket per line
    return Task(task_id, family, question, _render(tickets), answer=answer, level=level,
                grading={"type": "int", "blind": blind})


def make_task(seed: int, family: str, level: int = 1, task_id: str | None = None) -> Task:
    """One task from a seed. The task id never contains the seed (H3); pass a stable one for eval tasks."""
    task_id = task_id or f"{family}-L{level}-{secrets.token_hex(4)}"
    return _build(random.Random(seed), family, level, task_id)


def legacy_tasks(n: int, seed: int, prefix: str = "legacy") -> list[Task]:
    """The 20-step run's task lists, exactly: `make_ticket_tasks(n, seed)` from rlm_rl/archive/ticket_tasks.py,
    alternating tickets_all and tickets_customer from one generator. With seed 99_999 and n=50 this is
    the first seed's held-out eval set."""
    rng = random.Random(seed)
    return [_build(rng, FAMILIES[i % 2], 1, f"{prefix}-{i:03d}") for i in range(n)]
