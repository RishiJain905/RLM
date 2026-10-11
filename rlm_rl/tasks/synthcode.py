"""Synthetic Python code bases for the two code generator families (plan step 7, the 20% code share of decision 7a).

Each task is a generated Python package rendered as one text, file after file. Function and module names are
drawn fresh per seed from neutral word lists, so a name says nothing about what the function does, and nothing
can be answered from pretraining memory (research §3.1).

Families:
- `callgraph`: "which functions can end up calling X?" The gold answer comes from the generator's own call
  graph, so it is exact even where the calls are hard to see. At level 1, and in a quarter of tasks above it,
  every call is a plain call, which Python's `ast` resolves. Otherwise, from level 2 up, some calls go through
  an alias, a dispatch table or a `getattr` built from string pieces, and comments mention functions that are
  never called, so a regex over names over- and under-counts and the model has to read. Set-F1 against the
  gold set.
- `needle`: RepoQA's needle-function task. The question describes what a function does in plain English, and
  the answer is its qualified name. Near-neighbour decoys (the same loop over odd items instead of even, the
  last match instead of the first) sit elsewhere in the package. Exact match.
- `callgraph_heldout` and `needle_heldout`: held-out test only (H7). Separately written questions; call styles
  never used in training (`functools.partial` and lambda wrappers); behaviours never described in training.

Levels share the other families' sizes, ~35K to ~400K characters. Each level up also adds call depth and hidden
call styles (call graph) or decoys (needle).
"""

from __future__ import annotations

import random
import secrets
from dataclasses import dataclass, field

from ..env import Task
from . import behaviours as bh

GENERATOR_VERSION = "synthcode-v1"
FAMILIES = ("callgraph", "needle", "callgraph_heldout", "needle_heldout")
LEVELS = {1: (32_000, 40_000), 2: (70_000, 90_000), 3: (130_000, 170_000), 4: (230_000, 290_000),
          5: (360_000, 420_000)}
MAX_LEVEL = max(LEVELS)

# How each call is written, by level. Training never uses `partial` or `lambda`; the held-out test does.
CALL_STYLES = {1: {"direct": 1.0},
               2: {"direct": 0.85, "alias": 0.15},
               3: {"direct": 0.70, "alias": 0.15, "table": 0.15},
               4: {"direct": 0.60, "alias": 0.15, "table": 0.15, "getattr": 0.10},
               5: {"direct": 0.50, "alias": 0.15, "table": 0.20, "getattr": 0.15}}
HELDOUT_STYLES = {"partial": 0.15, "lambda": 0.10}
PLAIN_SHARE = 0.25  # call-graph tasks above level 1 drawn with plain calls only, so `ast` alone solves them (§3.1)
DECOY_COMMENT_RATE = {1: 0.0, 2: 0.10, 3: 0.20, 4: 0.25, 5: 0.30}
CALLERS_RANGE = {1: (2, 6), 2: (3, 9), 3: (4, 12), 4: (5, 15), 5: (6, 18)}  # how many functions reach the target
MIN_DEPTH = {1: 1, 2: 2, 3: 3, 4: 3, 5: 4}  # the longest chain from a caller down to the target
NEEDLE_DECOYS = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3}  # same-group decoys; other groups' behaviours fill in too
NEEDLE_BEHAVIOURS = {1: 6, 2: 10, 3: 16, 4: 24, 5: 32}  # behaviour functions in the package, target included
PER_MODULE = {1: (6, 12), 2: (8, 16), 3: (12, 24), 4: (16, 32), 5: (20, 40)}  # filler functions per module
# Mean rendered sizes, measured on generated packages: a filler function with its calls, a file's header and
# imports, and a behaviour function. `_grow` uses them to hit each level's size.
FILLER_CHARS, FILE_CHARS, BEHAVIOUR_CHARS = 215, 200, 250
SIZE_FACTOR = {1: 1.15, 2: 1.05, 3: 1.0, 4: 0.97, 5: 0.94}  # hidden call styles add text as the level rises

VERBS = ("load", "parse", "merge", "apply", "build", "check", "emit", "fold", "scan", "render", "resolve", "sync",
         "trim", "wrap", "pack", "split", "tally", "probe", "route", "stage", "flush", "seed", "patch", "shape",
         "sort", "pick", "sweep", "mark", "drain", "align", "bind", "carry", "clip", "coerce", "cache", "compose",
         "extract", "gather", "index", "join", "lift", "mint", "nudge", "order", "prune", "queue", "rank", "relay",
         "score", "settle", "shift", "slice", "spawn", "stash", "stitch", "tag", "tune", "unpack", "weigh", "yield")
NOUNS = ("ledger", "batch", "token", "frame", "record", "cursor", "quota", "shard", "bucket", "packet", "parcel",
         "lane", "slot", "chunk", "entry", "ticket", "voucher", "bundle", "cell", "grid", "ring", "node", "edge",
         "pane", "tile", "draft", "span", "window", "track", "stream", "signal", "metric", "badge", "crate", "crumb",
         "flag", "glyph", "hinge", "juncture", "keel", "lattice", "manifest", "nib", "orbit", "pivot", "quill",
         "rivet", "sprocket", "tether", "umbra", "vane", "wick", "yarn", "zone", "atlas", "beacon", "cipher", "dial")
QUALIFIERS = ("", "", "", "_all", "_safe", "_fast", "_raw", "_v2", "_once", "_many", "_inplace", "_strict")
PACKAGES = ("tidepool", "ledgerkit", "brightwell", "cobaltline", "fernhouse", "quarrystone", "marrowgate", "silverbirch",
            "copperleaf", "harrowfield", "lanternworks", "oakhollow")
MODULE_NOUNS = ("core", "io", "models", "utils", "views", "store", "queue", "jobs", "sync", "report", "schema",
                "config", "cache", "events", "hooks", "rules", "tasks", "stats", "export", "audit", "index",
                "pipeline", "routing", "session", "billing", "inventory", "metrics", "workers", "storage", "transform")
ARG_NAMES = ("items", "values", "data", "rows", "payload", "records", "batch", "entries", "seq", "buf")
LOCAL_NAMES = ("acc", "out", "res", "tmp", "cur", "seen", "parts", "found", "best", "head", "tail", "box", "pool")
LOOP_NAMES = ("x", "item", "elem", "val", "entry", "piece", "unit", "row")
ALIAS_NAMES = ("step", "handler", "op", "runner", "fn", "action", "hook")

# Filler bodies. Each ends with a top-level `return`; calls are inserted before it.
FILLER = (
    "total = {c1}\nfor {it} in {a}:\n    total += {it} * {c2}\nreturn total",
    "{v} = [{it} for {it} in {a} if {it} % {c1} == {c3}]\nreturn {v}",
    "{v} = str({a}).strip()\nif len({v}) > {c1}:\n    {v} = {v}[:{c1}]\nreturn {v}",
    "if not {a}:\n    return {c1}\nreturn len({a}) * {c2} - {c3}",
    "{v} = sorted({a})\nreturn {v}[len({v}) // 2] if {v} else None",
    "{v} = {{}}\nfor {it} in {a}:\n    {v}[{it}] = {v}.get({it}, 0) + 1\nreturn {v}",
    "{v} = max({a}, default={c1})\nreturn {v} - min({a}, default={c1})",
    "{v} = []\nfor {i}, {it} in enumerate({a}):\n    if {i} % {c1} == 0:\n        {v}.append({it})\nreturn {v}",
    "{v} = 0\nwhile {v} < {c1} and {v} < len({a}):\n    {v} += {c2}\nreturn {v}",
    "{v} = dict({a}) if isinstance({a}, dict) else {{}}\nreturn {{key: {it} for key, {it} in {v}.items() if {it}}}",
    "{v} = list({a})[{c3}:{c1}]\n{v}.reverse()\nreturn {v}",
    "{v} = sum(1 for {it} in {a} if {it})\nreturn {v} > {c2}",
)
DOCSTRINGS = ("Helper used by the pipeline.", "Internal step; keep the signature stable.",
              "Called during the nightly run.", "Kept for backwards compatibility.", "See the module notes.")
DECOY_COMMENTS = ("# {name} is the older path; it is not used here.", "# do not route through {name} (it re-sorts).",
                  "# mirrors {name} but keeps the original order", "# {name} used to be called here before 2.0")

TRAIN_QUESTIONS = {
    "callgraph": ["Which functions can end up calling `{target}`, directly or through other functions? List every one "
                  "as module.function (for example `module_name.function_name`).",
                  "List all functions from which a chain of calls reaches `{target}`. Give each as "
                  "module.function (for example `module_name.function_name`)."],
    "needle": ["Somewhere in this package is a function that {description}. What is it? Answer with its qualified "
               "name as module.function (for example `module_name.function_name`).",
               "Find the function that {description}. Answer with module.function (for example "
               "`module_name.function_name`)."],
}
HELDOUT_QUESTIONS = {
    "callgraph": ["Name every function whose execution may invoke {target} somewhere down the call stack. Reply with "
                  "their names written as module.function, comma separated."],
    "needle": ["One routine in the repository {description}. Reply with that routine's module and name joined by "
               "a dot."],
}


@dataclass
class Fn:
    module: str
    name: str
    args: tuple[str, ...]
    body: list[str]  # body lines, unindented
    calls: list[tuple[str, str]] = field(default_factory=list)  # (style, qualified callee)
    comments: list[str] = field(default_factory=list)
    docstring: str | None = None
    behaviour: str | None = None

    @property
    def qname(self) -> str:
        return f"{self.module}.{self.name}"


class Repo:
    """A generated package: modules, functions and how each call is written."""

    def __init__(self, rng: random.Random, heldout: bool):
        self.rng = rng
        self.heldout = heldout
        self.package = rng.choice(PACKAGES)
        self.modules: list[str] = []
        self.fns: list[Fn] = []
        self._names: set[str] = set()
        self.tables: dict[str, dict[str, str]] = {}  # module -> {key: qualified callee}
        self.partials: dict[str, dict[str, str]] = {}  # module -> {wrapper name: qualified callee}

    # -- names
    def new_module(self) -> str:
        while True:
            first, second = self.rng.sample(MODULE_NOUNS, 2)
            name = first if self.rng.random() < 0.4 else f"{first}_{second}"
            if name not in self.modules:
                self.modules.append(name)
                return name

    def new_name(self) -> str:
        while True:
            name = f"{self.rng.choice(VERBS)}_{self.rng.choice(NOUNS)}{self.rng.choice(QUALIFIERS)}"
            if name not in self._names:
                self._names.add(name)
                return name

    def locals_(self) -> dict[str, str]:
        v, w = self.rng.sample(LOCAL_NAMES, 2)
        return {"v": v, "w": w, "it": self.rng.choice(LOOP_NAMES), "i": "idx"}

    # -- functions
    def filler(self, module: str) -> Fn:
        rng = self.rng
        a = rng.choice(ARG_NAMES)
        names = self.locals_()
        body = rng.choice(FILLER).format(a=a, c1=rng.randint(2, 9), c2=rng.randint(2, 9), c3=rng.randint(0, 1), **names)
        doc = rng.choice(DOCSTRINGS) if rng.random() < 0.3 else None
        fn = Fn(module, self.new_name(), (a,), body.splitlines(), docstring=doc)
        self.fns.append(fn)
        return fn

    def behaviour_fn(self, module: str, behaviour: bh.Behaviour) -> Fn:
        a, b = self.rng.sample(ARG_NAMES, 2) if behaviour.args == 2 else (self.rng.choice(ARG_NAMES), "")
        names = {**self.locals_(), "a": a, "b": b or "n"}
        args = (a, names["b"]) if behaviour.args == 2 else (a,)
        fn = Fn(module, self.new_name(), args, behaviour.template.format(**names).splitlines(), behaviour=behaviour.key)
        self.fns.append(fn)
        return fn

    # -- calls
    def add_call(self, caller: Fn, callee: Fn, styles: dict[str, float]) -> None:
        style = self.rng.choices(list(styles), list(styles.values()))[0]
        caller.calls.append((style, callee.qname))

    def _ref(self, caller: Fn, qname: str) -> str:
        module, name = qname.split(".")
        return name if module == caller.module else f"{module}.{name}"

    def call_lines(self, caller: Fn) -> list[str]:
        rng, out = self.rng, []
        arg = caller.args[0]
        for style, qname in caller.calls:
            module, name = qname.split(".")
            var = rng.choice(LOCAL_NAMES) + "_" + str(rng.randint(1, 9))
            if style == "direct":
                out.append(f"{var} = {self._ref(caller, qname)}({arg})")
            elif style == "alias":
                alias = rng.choice(ALIAS_NAMES)
                out += [f"{alias} = {self._ref(caller, qname)}", f"{var} = {alias}({arg})"]
            elif style == "table":
                table = self.tables.setdefault(caller.module, {})
                key = f"{rng.choice(NOUNS)}-{len(table)}"
                table[key] = qname
                out.append(f"{var} = _ROUTES[{key!r}]({arg})")
            elif style == "getattr":
                cut = rng.randint(2, len(name) - 2)
                owner = "sys.modules[__name__]" if module == caller.module else module
                out.append(f"{var} = getattr({owner}, {name[:cut]!r} + {name[cut:]!r})({arg})")
            elif style == "partial":
                wrappers = self.partials.setdefault(caller.module, {})
                wrapper = f"_{rng.choice(VERBS)}_{rng.choice(NOUNS)}_{len(wrappers)}"
                wrappers[wrapper] = qname
                out.append(f"{var} = {wrapper}()")
            elif style == "lambda":
                out += [f"later = lambda v: {self._ref(caller, qname)}(v)", f"{var} = later({arg})"]
            else:
                raise ValueError(style)
        return out

    # -- rendering
    def render_fn(self, fn: Fn) -> str:
        body = list(fn.body)
        calls = self.call_lines(fn)
        if calls:
            body = body[:-1] + calls + body[-1:]
        lines = [f"def {fn.name}({', '.join(fn.args)}):"]
        if fn.docstring:
            lines.append(f'    """{fn.docstring}"""')
        for comment in fn.comments:
            lines.append(f"    {comment}")
        lines += ["    " + line if line else "" for line in body]
        return "\n".join(lines)

    def render(self) -> str:
        files = []
        for module in self.modules:
            fns = [f for f in self.fns if f.module == module]
            self.rng.shuffle(fns)
            rendered = [self.render_fn(f) for f in fns]  # renders calls first, so tables and partials are known
            imports = sorted({q.split(".")[0] for f in fns for _, q in f.calls if q.split(".")[0] != module})
            styles = {s for f in fns for s, _ in f.calls}
            head = [f'"""{self.package}.{module}"""', ""]
            if "getattr" in styles:
                head.append("import sys")
            if self.partials.get(module):
                head.append("import functools")
            head += [f"from . import {m}" for m in imports]
            tail = []
            if self.tables.get(module):
                entries = ",\n".join(f"    {k!r}: {q.split('.')[1] if q.split('.')[0] == module else q}"
                                     for k, q in self.tables[module].items())
                tail = ["", "", "_ROUTES = {\n" + entries + ",\n}"]
            for wrapper, qname in self.partials.get(module, {}).items():
                ref = qname.split(".")[1] if qname.split(".")[0] == module else qname
                tail += ["", f"{wrapper} = functools.partial({ref}, [])"]
            text = "\n".join(head) + "\n\n\n" + "\n\n\n".join(rendered) + "\n" + "\n".join(tail)
            files.append(f"# ===== file: {self.package}/{module}.py =====\n{text.rstrip()}\n")
        return "\n".join(files)


def _variant(family: str) -> tuple[str, bool]:
    if family not in FAMILIES:
        raise ValueError(f"unknown code family {family!r}")
    return family.removesuffix("_heldout"), family.endswith("_heldout")


def _grow(repo: Repo, budget: int, level: int) -> None:
    """Add modules of filler functions until the rendered package is about `budget` characters."""
    est, budget = 0, budget * SIZE_FACTOR[level]
    while est < budget:
        module = repo.new_module()
        est += FILE_CHARS
        for _ in range(repo.rng.randint(*PER_MODULE[level])):
            repo.filler(module)
            est += FILLER_CHARS


def _transitive_callers(fns: list[Fn], target: str) -> tuple[set[str], int]:
    """Every function that can reach `target`, and the longest chain length down to it."""
    callers: dict[str, set[str]] = {}
    for f in fns:
        for _, q in f.calls:
            callers.setdefault(q, set()).add(f.qname)
    depth = {target: 0}
    frontier, seen = [target], set()
    while frontier:
        node = frontier.pop()
        for c in callers.get(node, ()):
            if depth.get(c, -1) < depth[node] + 1:
                depth[c] = depth[node] + 1
                frontier.append(c)
            seen.add(c)
    return seen, max(depth.values())


def _wire_calls(repo: Repo, level: int, styles: dict[str, float]) -> None:
    """A random call DAG: each function calls up to 3 functions ranked below it, mostly nearby ones."""
    rng = repo.rng
    order = list(repo.fns)
    rng.shuffle(order)
    for rank, fn in enumerate(order):
        if rank == 0 or fn.behaviour:  # behaviour functions stay pure leaves, so running one is a fair check
            continue
        n_calls = rng.choices((0, 1, 2, 3), (0.3, 0.4, 0.2, 0.1))[0]
        for _ in range(n_calls):
            lo = max(0, rank - 25)
            callee = order[rng.randint(lo, rank - 1)]
            if all(q != callee.qname for _, q in fn.calls):
                repo.add_call(fn, callee, styles)


def _decoy_comments(repo: Repo, level: int, exclude: set[str]) -> None:
    rng = repo.rng
    for fn in repo.fns:
        if rng.random() < DECOY_COMMENT_RATE[level]:
            called = {q for _, q in fn.calls}
            options = [f for f in repo.fns if f.qname not in called and f is not fn and f.qname not in exclude]
            if options:
                other = rng.choice(options)
                ref = other.name if other.module == fn.module else other.qname
                fn.comments.append(rng.choice(DECOY_COMMENTS).format(name=ref))


def _callgraph(rng: random.Random, family: str, level: int, heldout: bool, task_id: str) -> Task:
    plain = level == 1 or (not heldout and rng.random() < PLAIN_SHARE)
    styles = {"direct": 1.0} if plain else dict(CALL_STYLES[level])
    if heldout and not plain:
        styles["direct"] -= sum(HELDOUT_STYLES.values())
        styles.update(HELDOUT_STYLES)
    for _ in range(40):
        repo = Repo(rng, heldout)
        _grow(repo, rng.randint(*LEVELS[level]), level)
        _wire_calls(repo, level, styles)
        lo, hi = CALLERS_RANGE[level]
        candidates = []
        for fn in repo.fns:
            callers, depth = _transitive_callers(repo.fns, fn.qname)
            if lo <= len(callers) <= hi and depth >= MIN_DEPTH[level]:
                candidates.append((fn, callers))
        if candidates:
            target, callers = rng.choice(candidates)
            break
    else:
        raise RuntimeError("no call-graph target in range")  # never seen in testing
    if not plain:
        _decoy_comments(repo, level, exclude=set())
    context = repo.render()
    templates = HELDOUT_QUESTIONS if heldout else TRAIN_QUESTIONS
    question = rng.choice(templates["callgraph"]).format(target=target.qname)
    everything = sorted(f.qname for f in repo.fns if f.qname != target.qname)
    hidden = sum(style != "direct" for f in repo.fns for style, q in f.calls if f.qname in callers or q == target.qname)
    grading = {"type": "list", "item": "entity", "blind": everything,  # every function in the package *(mine)*
               "kind": "callgraph", "code_only": plain, "hidden_calls": hidden}
    return Task(task_id, family, question, context, answer=sorted(callers), level=level, grading=grading)


def _needle(rng: random.Random, family: str, level: int, heldout: bool, task_id: str) -> Task:
    pool = bh.HELDOUT_GROUPS if heldout else bh.TRAIN_GROUPS
    group = rng.choice(sorted(pool))
    target_b = rng.choice(pool[group])
    decoys = [b for b in pool[group] if b is not target_b]
    rng.shuffle(decoys)
    chosen = [target_b] + decoys[:NEEDLE_DECOYS[level]]
    others = [b for g, bs in sorted(pool.items()) if g != group for b in bs]
    rng.shuffle(others)
    if heldout:  # held-out packages also carry training behaviours, so there are enough look-alikes
        others += [b for b in bh.TRAIN if b not in others]
    chosen += others[:max(0, NEEDLE_BEHAVIOURS[level] - len(chosen))]

    repo = Repo(rng, heldout)
    _grow(repo, rng.randint(*LEVELS[level]) - BEHAVIOUR_CHARS * len(chosen), level)
    placed = {b.key: repo.behaviour_fn(rng.choice(repo.modules), b) for b in chosen}
    _wire_calls(repo, level, dict(CALL_STYLES[min(level, 2)]))
    _decoy_comments(repo, level, exclude={placed[target_b.key].qname})
    target = placed[target_b.key]
    templates = HELDOUT_QUESTIONS if heldout else TRAIN_QUESTIONS
    description = rng.choice(target_b.descriptions)
    question = rng.choice(templates["needle"]).format(description=description)
    grading = {"type": "entity", "kind": "needle", "code_only": False, "behaviour": target_b.key,
               "decoys": [placed[b.key].qname for b in chosen if b is not target_b]}
    return Task(task_id, family, question, repo.render(), answer=target.qname, level=level, grading=grading)


def make_task(seed: int, family: str, level: int = 1, task_id: str | None = None) -> Task:
    """One task from a seed. The task id never contains the seed (H3)."""
    kind, heldout = _variant(family)
    if level not in LEVELS:
        raise ValueError(f"no code level {level} (levels 1-{MAX_LEVEL})")
    task_id = task_id or f"{family}-L{level}-{secrets.token_hex(4)}"
    rng = random.Random(seed)
    build = _callgraph if kind == "callgraph" else _needle
    return build(rng, family, level, heldout, task_id)
