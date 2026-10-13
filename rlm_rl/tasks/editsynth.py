"""edit_synth: code-editing tasks in generated Python packages (spec §12.7, code-editing-plan §4.2).

Each task is a generated, standard-library-only package written as real files (the edit sandbox
mounts it at /work/repo), with a small visible `tests/` folder the model can run with
`python3 -m unittest discover -s tests`. Names come from neutral word lists, fresh per seed, so
nothing can be answered from memory. Three kinds, each the edit twin of a read family:

- `fix` (needle's twin): a public function in `<pkg>.api` returns wrong results. The bug is a
  mutation in the behaviour function a few calls below it; look-alike behaviours from the same
  group sit elsewhere as decoys. Hidden tests compare the function with its reference on fresh
  inputs, and every other public function with its own.
- `callers` (callgraph's twin): a shared helper sits on the call paths of several public
  functions. Rename it, or give it a required keyword-only parameter, and update every caller:
  direct calls, aliases, dispatch tables and getattr calls built from string pieces. Hidden tests
  check the new name or parameter, and that every public function still works, so a missed
  caller fails.
- `feature` (new): add an option to a public function and thread it through the 2-4 modules down
  to the function that does the work. Hidden tests call every listed function with the option on
  fresh inputs, and check that the default changes nothing.

Hidden test inputs are drawn on the host from the seed and differ from any example in the task
text, so hard-coding the examples earns nothing (spec §12.6 H1). Hidden tests never enter the
episode's repository: the grader restores them (grader.py).

Levels 1-5 grow the package from ~20K to ~200K characters, the call chains, the callers, the
modules an option crosses (1 to 4 files to change), the decoys and the hidden call styles.
`edit_synth_heldout` (eval only, H7) uses mutation kinds, feature options, helper-parameter
semantics and task wording that training never sees, and behaviours training never describes.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import secrets
from dataclasses import dataclass, field
from typing import Callable

from ..env import Task
from . import behaviours as bh
from . import synthcode as sc

GENERATOR_VERSION = "editsynth-v1"
FAMILIES = ("edit_synth", "edit_synth_heldout")
KINDS = ("fix", "callers", "feature")
LEVELS = {1: (18_000, 24_000), 2: (40_000, 55_000), 3: (75_000, 95_000), 4: (120_000, 145_000),
          5: (175_000, 215_000)}
MAX_LEVEL = max(LEVELS)
LANGUAGE = "python"
TEST_HINT = "python3 -m unittest discover -s tests"
TEST_RUNNER = "unittest"

CHAIN_DEPTH = {1: 2, 2: 3, 3: 3, 4: 4, 5: 5}  # functions from a public entry point down to its behaviour
ENTRIES = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}  # public functions in api.py
DECOYS = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3}  # look-alike behaviours from the target's group
CALLERS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 5}  # callers kind: functions that use the shared helper
FEATURE_SPAN = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}  # feature kind: functions below the entry point that take the option
CALL_STYLES = {1: ("direct",), 2: ("direct", "alias"), 3: ("direct", "alias", "table"),
               4: ("direct", "alias", "table", "getattr"), 5: ("direct", "alias", "table", "getattr")}
FILLER_CHARS = 165  # a filler function's mean rendered size, for hitting a level's size (measured)
ENTRY_CHARS = 900  # an entry point's chain and behaviour, rendered
HIDDEN_INPUTS = 12
VISIBLE_EXAMPLES = 3

# Mutations for the fix kind, as (pattern, replacement) on a behaviour's template. "+1" bumps a digit. Held-out kinds
# never appear in training (H7).
TRAIN_MUTATIONS = [
    (r" > ", " >= "), (r" >= ", " > "), (r" < ", " <= "), (r" <= ", " < "), (r" == ", " != "), (r" != ", " == "),
    (r" \+ ", " - "), (r" - ", " + "), (r" \* ", " + "), (r"(?<![\w.{])([0-9])(?![\w.}])", "+1"),
]
HELDOUT_MUTATIONS = [
    (r"\bmin\(", "max("), (r"\bmax\(", "min("), (r" and ", " or "), (r" or ", " and "), (r" \+= ", " -= "),
    (r"\bif (?!not )", "if not "), (r"\bsorted\(", "list("), (r"\breversed\(", "list("),
]

# Feature options: name -> (default, the value the hidden tests use, the transform on the input list, the sentence).
TRAIN_OPTIONS = {
    "skip_negative": (False, True, lambda xs, v: [x for x in xs if x >= 0] if v else xs,
                      "When `skip_negative` is true, negative numbers in the input list are ignored.",
                      "{a} = [x for x in {a} if x >= 0] if {o} else {a}"),
    "unique": (False, True, lambda xs, v: list(dict.fromkeys(xs)) if v else xs,
               "When `unique` is true, repeated numbers in the input list are dropped first, keeping the first of each.",
               "{a} = list(dict.fromkeys({a})) if {o} else {a}"),
    "absolute": (False, True, lambda xs, v: [abs(x) for x in xs] if v else xs,
                 "When `absolute` is true, every number in the input list is replaced by its absolute value first.",
                 "{a} = [abs(x) for x in {a}] if {o} else {a}"),
    "first": (None, 5, lambda xs, v: xs[:v] if v is not None else xs,
              "When `first` is a number n, only the first n items of the input list are used.",
              "{a} = {a}[:{o}] if {o} is not None else {a}"),
}
HELDOUT_OPTIONS = {
    "stride": (1, 2, lambda xs, v: xs[::v], "With `stride` k, only every k-th item of the input list is used, starting "
                                            "with the first.", "{a} = {a}[::{o}]"),
    "clamp": (None, 4, lambda xs, v: [max(-v, min(v, x)) for x in xs] if v is not None else xs,
              "With `clamp` c, each number in the input list is first limited to the range from -c to c.",
              "{a} = [max(-{o}, min({o}, x)) for x in {a}] if {o} is not None else {a}"),
    "drop_last": (False, True, lambda xs, v: xs[:-1] if v else xs,
                  "With `drop_last` set, the last item of the input list is left out.",
                  "{a} = {a}[:-1] if {o} else {a}"),
    "shift": (0, 3, lambda xs, v: [x + v for x in xs], "With `shift` d, d is added to every number in the input list "
                                                       "before anything else happens.", "{a} = [x + {o} for x in {a}]"),
}
# Callers kind, param variant: (name, the sentence, the value that keeps today's behaviour, the value the hidden tests
# use, the gold return line). Training uses `limit`; the held-out test its twin `skip`.
TRAIN_PARAM = ("limit", "When `limit` is an integer, it returns only the first `limit` items of its result; when "
                        "`limit` is None, it behaves exactly as it does now.", None, 2,
               "return {x}[:{p}] if {p} is not None else {x}")
HELDOUT_PARAM = ("skip", "It drops the first `skip` items of its result, so `skip=0` keeps today's behaviour.", 0, 1,
                 "return {x}[{p}:] if {p} else {x}")

TRAIN_TEXTS = {
    "fix": ["`{pkg}.api.{entry}` gives wrong results. For example, `{entry}({ex})` returns `{got}`, but it should "
            "return `{want}`. The function is supposed to {description}. Find the bug and fix it without changing any "
            "function's signature.",
            "There is a bug behind `{pkg}.api.{entry}`: `{entry}({ex})` comes back as `{got}` instead of `{want}`. It "
            "should {description}. Fix the code so it does, keeping every signature as it is."],
    "rename": ["Rename the function `{module}.{old}` to `{new}`. Update every place in the package that uses it, so "
               "that every public function in `{pkg}.api` keeps working exactly as before. Don't keep the old name "
               "around as an alias."],
    "param": ["`{module}.{fn}` needs a new required keyword-only parameter `{param}`. {semantics} Update every caller "
              "in the package to pass `{param}={keep}`, so that every public function in `{pkg}.api` keeps working "
              "exactly as before."],
    "feature": ["Add a keyword argument `{opt}` (default `{default}`) to `{pkg}.api.{entry}`. {semantics} The same "
                "keyword, with the same meaning, must also be accepted by {chain}, which `{entry}` goes through, so "
                "thread it all the way down. With the default value, everything must behave exactly as it does now."],
}
HELDOUT_TEXTS = {
    "fix": ["Bug report: calling {entry} on {ex} produced {got}; the expected value is {want}. The routine's job is "
            "to {description}. Please correct the implementation (public signatures must stay the same)."],
    "rename": ["Our style guide wants `{module}.{old}` to be called `{new}` from now on. Carry the new name through "
               "the codebase, removing the old one, and make sure no public routine in `{pkg}.api` changes behaviour."],
    "param": ["Please extend `{module}.{fn}` with a mandatory keyword-only argument named `{param}`. {semantics} "
              "Every existing call site has to supply `{param}={keep}` so the routines exposed in `{pkg}.api` "
              "behave as they did."],
    "feature": ["Feature request: `{pkg}.api.{entry}` should take an optional `{opt}` argument, defaulting to "
                "`{default}`. {semantics} Plumb it through {chain} too, each honouring it identically, while calls "
                "that omit it see no change at all."],
}


# ----------------------------------------------------------------------------- inputs for behaviours


def _int_list(rng, lo=-9, hi=20, n=(0, 12)):
    return [rng.randint(lo, hi) for _ in range(rng.randint(*n))]


_VOCAB = ("alpha", "beta", "gamma", "delta", "pi", "zeta", "omega", "Kappa", "mu", "Sigma", "tau", "rho", "nu", "x",
          "yy", "Echo", "lima", "Oscar", "papa", "quebec", "a", "Bb")


def _words(rng):
    return " ".join(rng.choice(_VOCAB) for _ in range(rng.randint(0, 7)))


def _kv(rng, sep, eq):
    keys = ("a", "b", "key", "x", "name", "id", "v")
    parts = [f"{rng.choice(keys)}{eq}{rng.randint(0, 99)}" for _ in range(rng.randint(0, 4))]
    if rng.random() < 0.3:
        parts.append("bad")
    return f"{sep} ".join(parts)


def _nested(rng, depth=0):
    return [_nested(rng, depth + 1) if depth < 2 and rng.random() < 0.3 else rng.randint(-5, 9)
            for _ in range(rng.randint(0, 4))]


INPUTS: dict[str, Callable] = {
    "filtered_sum": lambda r: (_int_list(r),), "running": lambda r: (_int_list(r),),
    "centre": lambda r: (_int_list(r),), "search": lambda r: (_int_list(r, -6, 6),),
    "dedupe": lambda r: (_int_list(r, 0, 6),), "rotate": lambda r: (_int_list(r), r.randint(0, 7)),
    "words": lambda r: (_words(r),), "chars": lambda r: (_words(r),), "frequency": lambda r: (_words(r),),
    "palindrome": lambda r: (r.choice(["Abba", "abc", "Never odd or even", "ab ba", "noon", "Taco cat"])
                             + r.choice(["", "x", "!", " "]),),
    "chunk": lambda r: (_int_list(r), r.randint(1, 4)), "flatten": lambda r: (_nested(r),),
    "digits": lambda r: (r.randint(-99999, 99999),), "casing": lambda r: (_words(r),),
    "pairs": lambda r: (_int_list(r, -3, 6, (0, 8)), r.randint(-2, 6)), "bits": lambda r: (r.randint(-300, 300),),
    "ranking": lambda r: (_int_list(r, -4, 8),),
    "brackets": lambda r: ("".join(r.choice("(()a)b(") for _ in range(r.randint(0, 10))),),
}
INT_LIST_GROUPS = ("filtered_sum", "running", "centre", "search", "dedupe", "rotate", "chunk", "pairs", "ranking")


def inputs_for(b: bh.Behaviour, rng: random.Random) -> tuple:
    if b.group == "cipher":
        return (_words(rng), rng.randint(0, 30)) if b.args == 2 else (_words(rng),)
    if b.group == "parse":
        return (_kv(rng, ";", "=") if "semicolon" in b.key else _kv(rng, ",", ":"),)
    return INPUTS[b.group](rng)


def takes_int_list(b: bh.Behaviour) -> bool:
    return b.group in INT_LIST_GROUPS


def _copy(args) -> list:
    return json.loads(json.dumps(list(args)))


def _ref(b: bh.Behaviour, args, transform=None, value=None):
    args = _copy(args)
    if transform is not None:
        args[0] = transform(args[0], value)
    return b.reference(*args)


def _lit(value) -> str:
    return repr(value)


# ----------------------------------------------------------------------------- the package model


@dataclass
class Call:
    callee: str  # function id
    style: str  # direct, alias, table, getattr
    var: str
    alias: str = ""
    key: str = ""
    cut: float = 0.5


@dataclass
class Fn:
    fid: str
    module: str
    name: str
    params: list
    body: list  # lines; "@CALLS" renders every call, "@CALL:<i>" one call
    calls: list = field(default_factory=list)
    doc: str | None = None
    role: str = "filler"  # filler, entry, chain, helper, behaviour
    behaviour: bh.Behaviour | None = None
    names: dict = field(default_factory=dict)  # a behaviour's template names


class Package:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.name = rng.choice(sc.PACKAGES)
        self.modules: list[str] = []
        self.fns: dict[str, Fn] = {}
        self._names: set[str] = set()
        self._n = 0

    def module(self) -> str:
        while True:
            first, second = self.rng.sample(sc.MODULE_NOUNS, 2)
            name = first if self.rng.random() < 0.4 else f"{first}_{second}"
            if name not in self.modules and name not in ("api", "tests"):
                self.modules.append(name)
                return name

    def fname(self) -> str:
        while True:
            name = f"{self.rng.choice(sc.VERBS)}_{self.rng.choice(sc.NOUNS)}{self.rng.choice(sc.QUALIFIERS)}"
            if name not in self._names:
                self._names.add(name)
                return name

    def add(self, module: str, params: list, body: list, role: str = "filler", doc: str | None = None,
            behaviour: bh.Behaviour | None = None, names: dict | None = None) -> Fn:
        self._n += 1
        fn = Fn(f"f{self._n}", module, self.fname(), list(params), list(body), doc=doc, role=role,
                behaviour=behaviour, names=names or {})
        self.fns[fn.fid] = fn
        return fn

    def filler(self, module: str) -> Fn:
        rng = self.rng
        a = rng.choice(sc.ARG_NAMES)
        v, w = rng.sample(sc.LOCAL_NAMES, 2)
        body = rng.choice(sc.FILLER).format(a=a, c1=rng.randint(2, 9), c2=rng.randint(2, 9), c3=rng.randint(0, 1),
                                            v=v, w=w, it=rng.choice(sc.LOOP_NAMES), i="idx").splitlines()
        return self.add(module, [a], body[:-1] + ["@CALLS", body[-1]],
                        doc=rng.choice(sc.DOCSTRINGS) if rng.random() < 0.3 else None)

    def call(self, caller: Fn, callee: Fn, style: str) -> Call:
        rng = self.rng
        c = Call(callee.fid, style, f"{rng.choice(sc.LOCAL_NAMES)}_{rng.randint(1, 9)}", alias=rng.choice(sc.ALIAS_NAMES),
                 key=f"{rng.choice(sc.NOUNS)}-{rng.randint(10, 99)}", cut=rng.uniform(0.3, 0.7))
        caller.calls.append(c)
        return c

    # -- rendering. `names` renames functions by id; `variant` overrides parts of functions:
    #    {"params": extra parameter text, "body": lines, "pre": lines before the body, "kwargs": {callee id: text}}

    def _ref(self, caller: Fn, fid: str, names: dict) -> str:
        fn = self.fns[fid]
        name = names.get(fid, fn.name)
        return name if fn.module == caller.module else f"{fn.module}.{name}"

    def _call_lines(self, fn: Fn, c: Call, names: dict, tables: dict, kwargs: dict) -> list[str]:
        target = self.fns[c.callee]
        name = names.get(c.callee, target.name)
        ref = self._ref(fn, c.callee, names)
        args = ", ".join(fn.params[:len(target.params)]) + kwargs.get(c.callee, "")  # the helper takes one argument
        if c.style == "direct":
            return [f"{c.var} = {ref}({args})"]
        if c.style == "alias":
            return [f"{c.alias} = {ref}", f"{c.var} = {c.alias}({args})"]
        if c.style == "table":
            if target.module == fn.module:  # module level, after the functions it names
                tables.setdefault(fn.module, {})[c.key] = name
                return [f"{c.var} = _ROUTES[{c.key!r}]({args})"]
            # a table of another module's functions lives in the function, so imports can't cycle at load time
            return [f"_routes = {{{c.key!r}: {ref}}}", f"{c.var} = _routes[{c.key!r}]({args})"]
        if c.style == "getattr":
            cut = max(2, min(len(name) - 2, round(len(name) * c.cut)))
            owner = "sys.modules[__name__]" if target.module == fn.module else target.module
            return [f"{c.var} = getattr({owner}, {name[:cut]!r} + {name[cut:]!r})({args})"]
        raise ValueError(c.style)

    def _render_fn(self, fn: Fn, names: dict, tables: dict, variant: dict) -> str:
        over = variant.get(fn.fid, {})
        kwargs = over.get("kwargs", {})
        lines = [f"def {names.get(fn.fid, fn.name)}({', '.join(fn.params)}{over.get('params', '')}):"]
        if fn.doc:
            lines.append(f'    """{fn.doc}"""')
        body = list(over.get("pre", []))
        for line in over.get("body", fn.body):
            if line == "@CALLS":
                for c in fn.calls:
                    body += self._call_lines(fn, c, names, tables, kwargs)
            elif line.startswith("@CALL:"):
                body += self._call_lines(fn, fn.calls[int(line[6:])], names, tables, kwargs)
            else:
                body.append(line)
        lines += ["    " + line if line else "" for line in body]
        return "\n".join(lines)

    def render(self, names: dict | None = None, variant: dict | None = None) -> dict[str, str]:
        names, variant = names or {}, variant or {}
        files = {}
        for module in list(self.modules) + ["api"]:
            fns = [f for f in self.fns.values() if f.module == module]
            tables: dict = {}
            rendered = [self._render_fn(f, names, tables, variant) for f in fns]
            used = sorted({self.fns[c.callee].module for f in fns for c in f.calls} - {module})
            head = [f'"""{self.name}.{module}"""', ""]
            if any(c.style == "getattr" and self.fns[c.callee].module == module for f in fns for c in f.calls):
                head.append("import sys")
            head += [f"from . import {m}" for m in used]
            tail = []
            if tables.get(module):
                entries = ",\n".join(f"    {k!r}: {v}" for k, v in tables[module].items())
                tail = ["", "", "_ROUTES = {\n" + entries + ",\n}"]
            text = "\n".join(head) + "\n\n\n" + "\n\n\n".join(rendered) + "\n" + "\n".join(tail)
            files[f"{self.name}/{module}.py"] = text.rstrip() + "\n"
        files[f"{self.name}/__init__.py"] = f'"""{self.name}: a small utility package."""\n\nfrom . import api\n'
        return files


@dataclass
class Entry:
    fn: Fn  # the public function in api.py
    behaviour: bh.Behaviour
    chain: list  # Fn, from the entry point down to (not including) the behaviour function
    core: Fn  # the behaviour function


def _grow(pkg: Package, budget: int, rng: random.Random) -> None:
    """Filler functions (synthcode's bodies) until the package is about `budget` characters, with a call DAG."""
    est = 0
    while est < budget:
        module = rng.choice(pkg.modules) if pkg.modules and rng.random() < 0.6 else pkg.module()
        for _ in range(rng.randint(4, 10)):
            pkg.filler(module)
            est += FILLER_CHARS
    fillers = [f for f in pkg.fns.values() if f.role == "filler"]
    rng.shuffle(fillers)
    for rank, fn in enumerate(fillers):
        for _ in range(rng.choices((0, 1, 2), (0.5, 0.35, 0.15))[0]):
            if rank:
                callee = fillers[rng.randint(max(0, rank - 20), rank - 1)]
                if all(c.callee != callee.fid for c in fn.calls):
                    pkg.call(fn, callee, "direct")


def _make_entry(pkg: Package, b: bh.Behaviour, depth: int, rng: random.Random, modules: list[str]) -> Entry:
    """api.<entry> -> chain functions (depth - 1 of them, in `modules`) -> the behaviour function."""
    params = rng.sample(sc.ARG_NAMES, 2) if b.args == 2 else [rng.choice(sc.ARG_NAMES)]
    v, w = rng.sample(sc.LOCAL_NAMES, 2)
    names = {"a": params[0], "b": params[1] if b.args == 2 else "n", "v": v, "w": w, "it": rng.choice(sc.LOOP_NAMES)}
    core = pkg.add(modules[-1], params, b.template.format(**names).splitlines(), role="behaviour", behaviour=b,
                   names=names)
    entry = pkg.add("api", params, [], role="entry", doc=rng.choice(("Public entry point.", "Part of the public API.",
                                                                     None)))
    chain = [entry]
    for i in range(depth - 1):
        pre = rng.choice(([f"{params[0]} = list({params[0]})"] if b.group in INT_LIST_GROUPS else [], [],
                          [f"if {params[0]} is None:", f"    {params[0]} = []"]))
        chain.append(pkg.add(modules[min(i, len(modules) - 1)], params, pre, role="chain",
                             doc=rng.choice(sc.DOCSTRINGS) if rng.random() < 0.3 else None))
    for upper, lower in zip(chain, chain[1:] + [core]):
        c = pkg.call(upper, lower, "direct")
        upper.body = upper.body + ["@CALLS", f"return {c.var}"]
    return Entry(entry, b, chain, core)


def _pick(rng, pool_groups: dict, n: int, n_decoys: int, need: Callable | None):
    groups = sorted(g for g, bs in pool_groups.items() if len(bs) >= 2 and any(need is None or need(b) for b in bs))
    group = rng.choice(groups)
    target = rng.choice([b for b in pool_groups[group] if need is None or need(b)])
    decoys = [b for b in pool_groups[group] if b is not target]
    rng.shuffle(decoys)
    others = [b for g, bs in sorted(pool_groups.items()) if g != group for b in bs]
    rng.shuffle(others)
    chosen = [target] + decoys[:n_decoys]
    return chosen + others[:max(0, n - len(chosen))]


def _modules(pkg: Package, n: int, rng: random.Random) -> list[str]:
    mods = list(pkg.modules)
    rng.shuffle(mods)
    while len(mods) < n:
        mods.append(pkg.module())
    return mods[:n]


def _function_from(text: str, name: str) -> Callable:
    namespace: dict = {}
    exec(compile(text, "<editsynth>", "exec"), namespace)  # our own generated code, run on the host
    return namespace[name]


def _fresh(b: bh.Behaviour, rng: random.Random, n: int, avoid: list) -> list:
    seen = {json.dumps(_copy(a)) for a in avoid}
    out = []
    for _ in range(n * 60):
        if len(out) == n:
            break
        args = inputs_for(b, rng)
        key = json.dumps(_copy(args))
        if key not in seen:
            seen.add(key)
            out.append(args)
    return out


# ----------------------------------------------------------------------------- tests


def _asserts(ref: str, cases: list, kw: str = "") -> list[str]:
    return [f"        self.assertEqual({ref}({', '.join(_lit(a) for a in args)}{kw}), {_lit(want)})"
            for args, want in cases]


def _test_module(pkg: str, cls: str, methods: dict, imports: list) -> str:
    lines = ["import unittest", "", *[f"from {pkg} import {m}" for m in sorted(set(imports))], "", "",
             f"class {cls}(unittest.TestCase):"]
    for name, body in methods.items():
        lines += [f"    def {name}(self):", *(body or ["        pass"]), ""]
    return "\n".join(lines + ["", "if __name__ == '__main__':", "    unittest.main()", ""])


def _entry_cases(e: Entry, rng, n: int, avoid=()) -> list:
    return [(a, _ref(e.behaviour, a)) for a in _fresh(e.behaviour, rng, n, list(avoid))]


# ----------------------------------------------------------------------------- the three kinds


@dataclass
class Built:
    kind: str
    pkg: Package
    base: dict  # path -> text: the episode's repository
    gold: dict  # path -> text: what the gold change writes
    hidden: dict  # path -> text: restored by the grader
    f2p: list
    p2p: list
    scope: list
    issue: str
    target: dict  # where the change goes (for the blind checks, H1)


def _finish(rng, pkg: Package, kind: str, entries: list, base: dict, gold: dict, methods: dict, f2p_names: list,
            issue: str, target: dict, skip_visible: set) -> Built:
    cls_v, mod_v = f"Test{pkg.name.capitalize()}", f"test_{pkg.name}_api"
    visible = {}
    for e in entries:
        if e.fn.fid not in skip_visible:
            visible[f"test_{e.fn.name}"] = _asserts(f"api.{e.fn.name}", _entry_cases(e, rng, VISIBLE_EXAMPLES))
    hidden_mod = f"test_{rng.choice(sc.NOUNS)}_{secrets.token_hex(3)}"
    cls_h = f"Test{rng.choice(sc.NOUNS).capitalize()}{rng.choice(sc.VERBS).capitalize()}"
    hidden = {f"tests/{hidden_mod}.py": _test_module(pkg.name, cls_h, methods, ["api", *pkg.modules])}
    v_keys = [f"tests.{mod_v}.{cls_v}.{n}" for n in visible]
    h_keys = {n: f"tests.{hidden_mod}.{cls_h}.{n}" for n in methods}
    readme = f"# {pkg.name}\n\nA small utility package.\n\nRun the tests with:\n\n    {TEST_HINT}\n"
    files = {**base, "tests/__init__.py": "", f"tests/{mod_v}.py": _test_module(pkg.name, cls_v, visible, ["api"]),
             "README.md": readme, ".gitignore": "__pycache__/\n*.pyc\n"}
    return Built(kind, pkg, files, gold, hidden, [h_keys[n] for n in f2p_names],
                 v_keys + [k for n, k in h_keys.items() if n not in f2p_names],
                 [f"tests.{hidden_mod}", f"tests.{mod_v}"], issue, target)


def _mutate(e: Entry, rng: random.Random, heldout: bool):
    """A mutation of the behaviour that changes its result on at least a quarter of random inputs and never raises."""
    b, core = e.behaviour, e.core
    candidates = []
    for pattern, repl in (HELDOUT_MUTATIONS if heldout else TRAIN_MUTATIONS):
        for m in re.finditer(pattern, b.template):
            new = str(int(m.group(1)) + 1) if repl == "+1" else repl
            candidates.append(b.template[:m.start()] + new + b.template[m.end():])
    rng.shuffle(candidates)
    probe = [inputs_for(b, rng) for _ in range(40)]
    header = f"def {core.name}({', '.join(core.params)}):\n"
    for mutated in candidates:
        try:
            body = mutated.format(**core.names).splitlines()
            wrong = _function_from(header + "\n".join("    " + x if x else "" for x in body), core.name)
            differs = sum(wrong(*_copy(a)) != _ref(b, a) for a in probe)
        except Exception:
            continue
        if differs >= 10:
            return body, wrong, probe
    return None


def _build_fix(rng, pkg, entries, texts, heldout) -> Built | None:
    target = entries[0]
    found = _mutate(target, rng, heldout)
    if found is None:
        return None
    body, wrong, probe = found
    core, b = target.core, target.behaviour
    base = pkg.render(variant={core.fid: {"body": body}})
    path = f"{pkg.name}/{core.module}.py"
    gold = {path: pkg.render()[path]}
    ex = next(a for a in probe if wrong(*_copy(a)) != _ref(b, a))
    for _ in range(8):  # fresh inputs, at least 4 of which the bug gets wrong, so the hidden test fails on the base
        fresh = _fresh(b, rng, HIDDEN_INPUTS, [ex])
        if sum(wrong(*_copy(a)) != _ref(b, a) for a in fresh) >= 4:
            break
    else:
        return None
    issue = rng.choice(texts["fix"]).format(pkg=pkg.name, entry=target.fn.name, ex=", ".join(_lit(a) for a in ex),
                                           got=_lit(wrong(*_copy(ex))), want=_lit(_ref(b, ex)),
                                           description=rng.choice(b.descriptions))
    name = f"test_{target.fn.name}_matches_reference"
    methods = {name: _asserts(f"api.{target.fn.name}", [(a, _ref(b, a)) for a in fresh])}
    for e in entries[1:]:
        methods[f"test_{e.fn.name}_unchanged"] = _asserts(f"api.{e.fn.name}", _entry_cases(e, rng, HIDDEN_INPUTS))
    tgt = {"path": path, "function": core.name, "kind": "fix"}
    return _finish(rng, pkg, "fix", entries, base, gold, methods, [name], issue, tgt, {target.fn.fid})


def _build_callers(rng, pkg, entries, level, texts, heldout) -> Built | None:
    """A shared helper on the call paths of several public functions, called in every style the level allows."""
    users = entries[:min(CALLERS[level], len(entries))]
    x = rng.choice(sc.ARG_NAMES)
    first = (users[0].chain[1:] or users[0].chain)[0]  # the helper sits beside its first caller: fewer files at level 1
    helper = pkg.add(first.module if first.module != "api" else rng.choice(pkg.modules), [x],
                     [f"if {x} is None:", f"    return []", f"return {x}"],
                     role="helper", doc=rng.choice(("Treats a missing input as empty.", "Normalizes its input.", None)))
    styles = CALL_STYLES[level]
    callers = []
    for i, e in enumerate(users):
        caller = rng.choice(e.chain[1:] or e.chain)
        if caller in callers:
            continue
        c = pkg.call(caller, helper, styles[i % len(styles)])
        caller.calls.insert(0, caller.calls.pop())  # the helper's call comes first, and its result feeds the rest
        at = caller.body.index("@CALLS")
        caller.body[at:at + 1] = ["@CALL:0", f"{caller.params[0]} = {c.var}", "@CALL:1"]
        callers.append(caller)
    if len(callers) < 2:
        return None
    mod = helper.module
    if rng.random() < 0.5:  # rename
        new = pkg.fname()
        base, gold_all = pkg.render(), pkg.render(names={helper.fid: new})
        issue = rng.choice(texts["rename"]).format(module=f"{pkg.name}.{mod}", old=helper.name, new=new, pkg=pkg.name)
        methods = {f"test_{new}_exists": [f"        self.assertEqual({mod}.{new}([3, 1]), [3, 1])",
                                          f"        self.assertEqual({mod}.{new}(None), [])"],
                   f"test_{helper.name}_is_gone": [f"        self.assertFalse(hasattr({mod}, {helper.name!r}))"]}
        f2p_names = list(methods)
        tgt = {"path": f"{pkg.name}/{mod}.py", "function": helper.name, "kind": "rename", "new": new}
    else:  # a required keyword-only parameter
        param, semantics, keep, value, ret = HELDOUT_PARAM if heldout else TRAIN_PARAM
        variant = {helper.fid: {"params": f", *, {param}",
                                "body": [f"if {x} is None:", f"    {x} = []", ret.format(x=x, p=param)]}}
        for caller in callers:
            variant[caller.fid] = {"kwargs": {helper.fid: f", {param}={keep!r}"}}
        base, gold_all = pkg.render(), pkg.render(variant=variant)
        issue = rng.choice(texts["param"]).format(module=f"{pkg.name}.{mod}", fn=helper.name, param=param,
                                                 semantics=semantics, keep=repr(keep), pkg=pkg.name)
        sample = [4, 7, 1, 9, 2]
        cut = sample[:value] if param == "limit" else sample[value:]
        methods = {f"test_{helper.name}_{param}": [
                       f"        self.assertEqual({mod}.{helper.name}({sample!r}, {param}={value!r}), {cut!r})",
                       f"        self.assertEqual({mod}.{helper.name}({sample!r}, {param}={keep!r}), {sample!r})"],
                   f"test_{helper.name}_{param}_required": [
                       "        with self.assertRaises(TypeError):", f"            {mod}.{helper.name}({sample!r})"]}
        f2p_names = list(methods)
        tgt = {"path": f"{pkg.name}/{mod}.py", "function": helper.name, "kind": "param", "param": param}
    gold = {p: t for p, t in gold_all.items() if base.get(p) != t}
    for e in entries:  # a missed caller breaks the public function above it
        methods[f"test_{e.fn.name}_unchanged"] = _asserts(f"api.{e.fn.name}", _entry_cases(e, rng, HIDDEN_INPUTS))
    return _finish(rng, pkg, "callers", entries, base, gold, methods, f2p_names, issue, tgt, set())


def _build_feature(rng, pkg, entries, texts, heldout) -> Built | None:
    target = entries[0]
    options = HELDOUT_OPTIONS if heldout else TRAIN_OPTIONS
    opt = rng.choice(sorted(options))
    default, value, transform, semantics, apply_line = options[opt]
    chain, core, b = target.chain, target.core, target.behaviour
    variant = {fn.fid: {"params": f", {opt}={default!r}", "kwargs": {fn.calls[0].callee: f", {opt}={opt}"}}
               for fn in chain}
    a = core.params[0]
    variant[core.fid] = {"params": f", {opt}={default!r}", "pre": [f"{a} = list({a})", apply_line.format(a=a, o=opt)]}
    base, gold_all = pkg.render(), pkg.render(variant=variant)
    gold = {p: t for p, t in gold_all.items() if base.get(p) != t}
    listed = chain[1:] + [core]
    issue = rng.choice(texts["feature"]).format(opt=opt, default=repr(default), pkg=pkg.name, entry=target.fn.name,
                                               semantics=semantics,
                                               chain=", ".join(f"`{pkg.name}.{f.module}.{f.name}`" for f in listed))
    for _ in range(10):  # inputs on which the option changes the result, so the hidden tests fail on the base
        fresh = _fresh(b, rng, HIDDEN_INPUTS, [])
        if sum(_ref(b, a, transform, value) != _ref(b, a) for a in fresh) >= 4:
            break
    else:
        return None
    methods = {}
    for fn in chain + [core]:
        ref = f"{fn.module}.{fn.name}"
        methods[f"test_{fn.name}_{opt}"] = _asserts(ref, [(a, _ref(b, a, transform, value)) for a in fresh],
                                                    f", {opt}={value!r}")
    f2p_names = list(methods)
    methods[f"test_{target.fn.name}_default_unchanged"] = _asserts(f"api.{target.fn.name}",
                                                                   _entry_cases(target, rng, 6, fresh))
    for e in entries[1:]:
        methods[f"test_{e.fn.name}_unchanged"] = _asserts(f"api.{e.fn.name}", _entry_cases(e, rng, HIDDEN_INPUTS))
    tgt = {"path": f"{pkg.name}/{core.module}.py", "function": core.name, "kind": "feature", "option": opt}
    return _finish(rng, pkg, "feature", entries, base, gold, methods, f2p_names, issue, tgt, set())


def _build(rng: random.Random, kind: str, level: int, heldout: bool) -> Built | None:
    pool = bh.HELDOUT_GROUPS if heldout else bh.TRAIN_GROUPS
    pkg = Package(rng)
    for _ in range(2 + level):
        pkg.module()
    lo, hi = LEVELS[level]
    _grow(pkg, rng.randint(lo, hi) - ENTRY_CHARS * ENTRIES[level] - 1500, rng)
    chosen = _pick(rng, pool, ENTRIES[level], DECOYS[level], takes_int_list if kind == "feature" else None)
    if heldout:  # held-out packages also carry training behaviours, so there are enough of them
        extra = [b for b in bh.TRAIN if b not in chosen]
        rng.shuffle(extra)
        chosen += extra[:max(0, ENTRIES[level] - len(chosen))]
    entries = []
    for i, b in enumerate(chosen[:ENTRIES[level]]):
        depth = (FEATURE_SPAN[level] + 1 if kind == "feature" else CHAIN_DEPTH[level]) if i == 0 \
            else rng.randint(2, CHAIN_DEPTH[level])
        entries.append(_make_entry(pkg, b, depth, rng, _modules(pkg, max(1, depth - 1), rng)))
    texts = HELDOUT_TEXTS if heldout else TRAIN_TEXTS
    if kind == "fix":
        return _build_fix(rng, pkg, entries, texts, heldout)
    if kind == "callers":
        return _build_callers(rng, pkg, entries, level, texts, heldout)
    return _build_feature(rng, pkg, entries, texts, heldout)


# ----------------------------------------------------------------------------- the Task


def file_list(files: dict) -> str:
    return "\n".join(f"{path}  {len(text):,}" for path, text in sorted(files.items()))


def make_task(seed: int, family: str, level: int = 1, task_id: str | None = None, kind: str | None = None) -> Task:
    """One edit_synth task from a seed. The task id never contains the seed (H3)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown edit_synth family {family!r}")
    if level not in LEVELS:
        raise ValueError(f"no edit_synth level {level} (levels 1-{MAX_LEVEL})")
    rng = random.Random(seed)
    kind = kind or rng.choice(KINDS)
    if kind not in KINDS:
        raise ValueError(f"unknown edit_synth kind {kind!r}")
    for _ in range(30):
        built = _build(rng, kind, level, family.endswith("_heldout"))
        if built is not None:
            return to_task(built, family, level, task_id or f"{family}-L{level}-{secrets.token_hex(4)}")
    raise RuntimeError("no edit_synth task could be built")  # never seen in testing


def to_task(built: Built, family: str, level: int, task_id: str) -> Task:
    from ..editbox import Workspace
    from ..grader import TestSpec

    base = dict(built.base)
    key = "synth-" + hashlib.sha256((GENERATOR_VERSION + json.dumps(base, sort_keys=True)).encode()).hexdigest()[:24]

    def build(folder, files=base):
        for path, text in files.items():
            target = folder / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")

    context = f"{built.issue}\n\nRepository files (path, size in characters):\n{file_list(base)}\n"
    spec = TestSpec(runner=TEST_RUNNER, scope=built.scope, f2p=built.f2p, p2p=built.p2p, language=LANGUAGE,
                    hidden=built.hidden)
    grading = {"type": "edit", "kind": built.kind, "language": LANGUAGE, "tests": spec.as_dict(),
               "gold_files": built.gold, "target": built.target,
               "size_chars": sum(len(t) for p, t in base.items() if p.startswith(built.pkg.name + "/")),
               "files_to_change": len(built.gold), "blind": None}
    edit = {"language": LANGUAGE, "n_files": len(base), "test_hint": TEST_HINT, "image": None, "repo": None,
            "kind": built.kind}
    return Task(task_id, family, built.issue, context, answer=None, level=level, grading=grading, edit=edit,
                workspace=Workspace(key, build))
