"""edit_bugs: find and fix several linked injected bugs from their symptoms only (edit tasks v2).

Research basis: Breakpoint's discovery mode (arXiv 2506.00172: corrupted functions the agent must
locate from failing tests, harder as more functions and files are involved) and SWE-smith's
procedural bugs (operator swaps, off-by-one, flipped branches and the like). Splitting is the
efficient path: the root reads the failures, has sub-calls investigate suspects in parallel (each
failing test, or each candidate module), then fixes and checks.

scripts/build_edit_tasks_v2.py builds the pool in the repositories edit_real validated. Each
candidate bug is one small, syntax-preserving change in a function the tests cover; it is kept only
if, on its own, it makes between 1 and 40 tests fail (subtle, not fatal). A task injects k of them
into k linked functions (call-graph neighbours, or functions the same tests run), every one of which
still shows in the combined run: some failing test of each bug's own failures still fails. The task
text gives only symptoms: the names of failing tests and their assertion messages (no traceback, no
file or line of library code), and how many faults there are. Hidden tests: every test that fails
with the bugs and passes on the original (fail-to-pass) and the other tests of those files
(pass-to-pass); the failing tests' files are taken out of the repository.

Levels (bugs, files they span): 1: 1-2 in one file; 2: 2-3 in 1-2 files; 3: 3-4 across two files;
4: 4-5 across 2-3 files; 5: 5-7 across 3-4 files.

Splits: `edit_bugs` trains; `edit_bugs_select` and `edit_bugs_test` come from repositories training
never sees and are eval-only (H7). Drawing, retirement and exclusion work as edit_real's.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass

from ..env import Task
from . import editspec
from .editspec import Fn, Repo, Source

GENERATOR_VERSION = "editbugs-v1"
FAMILIES = ("edit_bugs", "edit_bugs_select", "edit_bugs_test")
SPLIT = {"edit_bugs": "train", "edit_bugs_select": "select", "edit_bugs_test": "test"}
MAX_LEVEL = 5
INDEX = "edit/v2/bugs_index.json"  # relative to the data root
KIND = "bugs"
# level -> ((fewest, most) bugs, (fewest, most) files)
LEVELS = {1: ((1, 2), (1, 1)), 2: ((2, 3), (1, 2)), 3: ((3, 4), (2, 2)), 4: ((4, 5), (2, 3)), 5: ((5, 7), (3, 4))}
MAX_SINGLE_FAILS = 40  # a candidate bug that fails more tests on its own is too blunt to be a subtle fault
MAX_LISTED = 10  # failures quoted in the task text

POOL = editspec.Pool("edit_bugs", FAMILIES, SPLIT, INDEX, GENERATOR_VERSION, "bugs")


def index_path():
    return POOL.index_path()


def records(split: str) -> list[dict]:
    return POOL.records(split)


def training_sources() -> tuple[str, ...]:
    return POOL.training_sources()


def check_licences() -> None:
    POOL.check_licences()


def make_task(seed: int, family: str, level: int = 1, task_id: str | None = None, exclude=()) -> Task:
    """A task from the family's split at (or nearest) `level`, chosen by the seed, skipping retired ids."""
    return POOL.make_task(seed, family, level, task_id, exclude)


def pool_size(family: str = "edit_bugs") -> int:
    return POOL.pool_size(family)


def eval_tasks(split: str, family: str) -> list[Task]:
    return POOL.eval_tasks(split, family)


def to_task(record: dict, family: str, task_id: str | None = None) -> Task:
    return POOL.to_task(record, family, task_id)


# ----------------------------------------------------------------------------- candidate bugs (host only)


@dataclass
class Mutation:
    id: str  # "<function key>@<line>:<kind>"
    fn: str
    path: str
    kind: str
    line: int  # 1-based line of the change
    start: int  # character offsets in the file's text (\n line ends)
    end: int
    old: str
    new: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# How likely each kind is picked when a function offers several (subtle kinds first).
KIND_WEIGHT = {"boundary": 4, "off_by_one": 4, "swapped_arguments": 3, "dropped_condition": 3, "arithmetic": 2,
               "boolean": 2, "wrong_branch": 2, "equality": 1, "membership": 1, "flipped_constant": 1}
COMPARE = {ast.Lt: (r"<", "<=", "boundary"), ast.LtE: (r"<=", "<", "boundary"), ast.Gt: (r">", ">=", "boundary"),
           ast.GtE: (r">=", ">", "boundary"), ast.Eq: (r"==", "!=", "equality"), ast.NotEq: (r"!=", "==", "equality"),
           ast.In: (r"\bin\b", "not in", "membership"), ast.NotIn: (r"\bnot\s+in\b", "in", "membership"),
           ast.Is: (r"\bis\b(?!\s+not\b)", "is not", "equality"), ast.IsNot: (r"\bis\s+not\b", "is", "equality")}
SKIP_SWAP = {"isinstance", "issubclass", "getattr", "setattr", "hasattr", "max", "min", "print", "super", "zip", "dict",
             "set", "list", "tuple", "sorted", "sum", "any", "all", "format", "join", "append", "extend", "update",
             "add", "map", "filter", "enumerate", "range", "open", "warn", "warning", "debug", "info", "error"}


def _parents(root) -> dict:
    out = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def _own_nodes(stmts: list):
    """Every node of these statements, not of functions or classes nested in them."""
    stack = list(stmts)
    while stack:
        n = stack.pop()
        yield n
        for child in ast.iter_child_nodes(n):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stack.append(child)


def function_node(repo: Repo, fn: Fn):
    tree = repo.modules[fn.path].tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == fn.def_line \
                and node.name == fn.name:
            return node
    return None


def _gap_op(src: Source, a, b, pattern: str, repl: str) -> tuple[int, int, str] | None:
    """The operator between two nodes (text between them, parentheses included), replaced."""
    lo, hi = src.span(a)[1], src.span(b)[0]
    gap = src.text[lo:hi]
    hits = list(re.finditer(pattern, gap))
    if len(hits) != 1 or "#" in gap:
        return None
    m = hits[0]
    return lo + m.start(), lo + m.end(), repl


def candidates(repo: Repo, fn: Fn) -> list[tuple[str, int, int, int, str]]:
    """Every single-site bug this function's body offers: (kind, line, start, end, new text)."""
    node = function_node(repo, fn)
    if node is None or not fn.stubbable:
        return []
    src = repo.modules[fn.path].source
    body = node.body[1:] if fn.doc else node.body
    parents = _parents(node)
    out = []

    def add(kind, line, span):
        if span is not None:
            out.append((kind, line, span[0], span[1], span[2]))

    for n in _own_nodes(body):
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in COMPARE:
            pattern, repl, kind = COMPARE[type(n.ops[0])]
            add(kind, n.lineno, _gap_op(src, n.left, n.comparators[0], pattern, repl))
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
            if any(isinstance(x, (ast.JoinedStr, ast.List, ast.Tuple)) or
                   (isinstance(x, ast.Constant) and isinstance(x.value, (str, bytes))) for x in (n.left, n.right)):
                continue  # concatenation: the swap would only crash
            old, new = (r"\+", "-") if isinstance(n.op, ast.Add) else (r"-", "+")
            add("arithmetic", n.lineno, _gap_op(src, n.left, n.right, old, new))
        elif isinstance(n, ast.AugAssign) and isinstance(n.op, (ast.Add, ast.Sub)):
            if isinstance(n.value, ast.Constant) and isinstance(n.value.value, (str, bytes)):
                continue
            old, new = (r"\+=", "-=") if isinstance(n.op, ast.Add) else (r"-=", "+=")
            add("arithmetic", n.lineno, _gap_op(src, n.target, n.value, old, new))
        elif isinstance(n, ast.BoolOp) and len(n.values) == 2:
            a, b = n.values
            word = "and" if isinstance(n.op, ast.And) else "or"
            span = _gap_op(src, a, b, rf"\b{word}\b", "or" if word == "and" else "and")
            add("boolean", n.lineno, span)
            if span is not None:
                lo, hi = src.span(n)
                kw_start, kw_end = span[0], span[1]
                add("dropped_condition", n.lineno, (lo, hi, src.text[lo:kw_start].rstrip()))
                add("dropped_condition", b.lineno, (lo, hi, src.text[kw_end:hi].lstrip()))
        elif isinstance(n, (ast.If, ast.IfExp)) and not isinstance(n.test, ast.Compare):  # not `while`: it hangs
            test = n.test
            lo, hi = src.span(test)
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                inner = src.segment(test.operand)
                if isinstance(test.operand, (ast.BoolOp, ast.IfExp, ast.Lambda, ast.NamedExpr)):
                    inner = f"({inner})"
                add("wrong_branch", test.lineno, (lo, hi, inner))
            elif isinstance(test, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
                add("wrong_branch", test.lineno, (lo, hi, "not " + src.segment(test)))
            elif isinstance(test, ast.BoolOp) and "\n" not in src.segment(test):
                add("wrong_branch", test.lineno, (lo, hi, f"not ({src.segment(test)})"))
        elif isinstance(n, ast.Constant) and type(n.value) is int and 0 <= n.value <= 16:
            parent = parents.get(n)
            seg = src.segment(n)
            if seg != str(n.value) or not isinstance(parent, (ast.Compare, ast.Subscript, ast.Slice, ast.BinOp,
                                                               ast.UnaryOp, ast.Call)):
                continue
            if isinstance(parent, ast.Call) and not (isinstance(parent.func, ast.Name)
                                                     and parent.func.id in ("range", "islice")):
                continue
            lo, hi = src.span(n)
            delta = 1 if n.value == 0 or isinstance(parent, ast.UnaryOp) else random.Random(lo).choice((1, -1))
            add("off_by_one", n.lineno, (lo, hi, str(n.value + delta)))
        elif isinstance(n, ast.Constant) and type(n.value) is bool:
            parent = parents.get(n)
            if isinstance(parent, (ast.Return, ast.keyword, ast.Assign)):
                lo, hi = src.span(n)
                add("flipped_constant", n.lineno, (lo, hi, str(not n.value)))
        elif isinstance(n, ast.Call) and len(n.args) >= 2 and not any(isinstance(a, ast.Starred) for a in n.args):
            name = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ""
            if not name or name in SKIP_SWAP:
                continue
            for i in range(len(n.args) - 1):
                a, b = n.args[i], n.args[i + 1]
                sa, sb = src.segment(a), src.segment(b)
                if sa == sb or "\n" in sa or "\n" in sb:
                    continue
                lo, hi = src.span(a)[0], src.span(b)[1]
                mid = src.text[src.span(a)[1]:src.span(b)[0]]
                add("swapped_arguments", n.lineno, (lo, hi, sb + mid + sa))
                break
    return out


def apply(text: str, muts: list[Mutation]) -> str:
    """`text` with each mutation applied (they never overlap: one per function)."""
    for m in sorted(muts, key=lambda m: m.start, reverse=True):
        if text[m.start:m.end] != m.old:
            raise ValueError(f"{m.id}: the text at the mutation's span changed")
        text = text[:m.start] + m.new + text[m.end:]
    return text


def mutations(repo: Repo, fn: Fn, rng: random.Random, limit: int = 3) -> list[Mutation]:
    """Up to `limit` candidate bugs in one function, on different lines, each one that still parses and changes the
    program; kinds drawn with KIND_WEIGHT."""
    text = repo.modules[fn.path].source.text
    original = ast.dump(ast.parse(text))
    options = candidates(repo, fn)
    rng.shuffle(options)
    options.sort(key=lambda o: -KIND_WEIGHT.get(o[0], 1) * rng.random())
    out, lines = [], set()
    for kind, line, start, end, new in options:
        if line in lines or len(out) >= limit:
            continue
        m = Mutation(f"{fn.key}@{line}:{kind}", fn.key, fn.path, kind, line, start, end, text[start:end], new)
        if m.old == m.new:
            continue
        try:
            changed = apply(text, [m])
            if ast.dump(ast.parse(changed)) == original:
                continue
        except (SyntaxError, ValueError):
            continue
        lines.add(line)
        out.append(m)
    return out


# ----------------------------------------------------------------------------- linked bug sets


def bug_sets(repo: Repo, tests_of: dict[str, set], good: dict[str, list[dict]], level: int, rng: random.Random,
             want: int, used: dict | None = None, max_use: int = 2, max_fails: int = 150, total: dict | None = None,
             max_total: int = 5, max_files: int | None = None) -> list[list[dict]]:
    """Up to `want` sets of screened bugs for a level: k linked functions (static call-graph neighbours first, then
    functions run by the same tests), one bug each, each function in at most `max_use` of the level's sets and
    `max_total` of the family's. `good` maps a function key to its screened bugs (dicts with "mutation" and "fails")."""
    (kmin, kmax), files = LEVELS[level]
    used = used if used is not None else {}
    total = total if total is not None else {}
    cand = {k for k, ms in good.items() if ms}
    static, co = editspec.graph(repo, cand, tests_of)
    seeds = sorted(cand)
    rng.shuffle(seeds)
    seeds.sort(key=lambda s: used.get(s, 0) + total.get(s, 0))
    out, seen = [], set()
    for seed in seeds:
        if len(out) >= want:
            break
        allowed = {c for c in cand if used.get(c, 0) < max_use and total.get(c, 0) < max_total}
        if seed not in allowed:
            continue
        k = rng.randint(kmin, kmax)
        fns = [seed] if k == 1 and files[0] == 1 else editspec.grow(repo, seed, k, files, static, co, rng, allowed)
        if not fns or frozenset(fns) in seen:
            continue
        picked = [rng.choice(good[f]) for f in fns]
        fails = set().union(*(set(p["fails"]) for p in picked))
        if len(fails) > max_fails:
            continue
        if max_files and len(editspec.test_files_of(fns, tests_of)) > max_files:
            continue  # its tests span too many files to hide
        seen.add(frozenset(fns))
        for f in fns:
            used[f] = used.get(f, 0) + 1
            total[f] = total.get(f, 0) + 1
        out.append(sorted(picked, key=lambda p: p["mutation"]["id"]))
    return out


# ----------------------------------------------------------------------------- the task text


PATH_LINE = re.compile(r"(/work/repo/|/testbed/)?[\w./-]+\.py:\d+(:\d+)?")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def clean_message(message: str, max_lines: int = 8, max_chars: int = 500) -> str:
    """A failure's message as a symptom: no file paths or line numbers, at most a few lines."""
    text = ANSI.sub("", message or "").replace("/work/repo/", "").replace("/testbed/", "")
    lines = []
    for line in text.splitlines():
        line = PATH_LINE.sub("<library code>", line.rstrip())
        if line.strip():
            lines.append(line)
        if len(lines) >= max_lines:
            break
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[:max_chars].rstrip() + " ..."


def issue_text(repo_name: str, k: int, failures: list[tuple[str, str]]) -> str:
    """The task text: symptoms only. `failures` are (test id, message) pairs."""
    short = repo_name.split("/")[-1]
    head = (f"After a recent change, some of {short}'s tests fail. These are failures the test run reported (the "
            "failing test files are not in your checkout, so you can't run them; you can write your own checks):")
    items = []
    for i, (test, message) in enumerate(failures, 1):
        msg = clean_message(message) or "(the test failed without a message)"
        items.append(f"{i}. `{test}`\n" + "\n".join("   " + line for line in msg.splitlines()))
    faults = "one fault" if k == 1 else f"{k} separate faults"
    tail = (f"The failures come from {faults} in the library's own code; the tests are correct. Find and fix "
            f"{'it' if k == 1 else 'every one of them'} without changing the library's intended behaviour anywhere "
            "else. Don't change any signature, and don't change the tests.")
    return "\n\n".join([head, "\n".join(items), tail]) + "\n"


def listed_failures(f2p: list[str], bugs: list[dict], calls: dict[str, set], rng: random.Random) -> list[str] | None:
    """The failing tests the text quotes: for each bug, one of its own failures that still fails with every bug in
    (the one that runs the fewest other bugged functions), then others up to MAX_LISTED. None if some bug shows no
    failure of its own in the combined run."""
    f2p_set = set(f2p)
    fns = {b["mutation"]["fn"] for b in bugs}
    listed = []
    for b in bugs:
        own = sorted(set(b["fails"]) & f2p_set)
        if not own:
            return None
        others = lambda t: len((calls.get(t) or set()) & (fns - {b["mutation"]["fn"]}))  # noqa: E731
        own.sort(key=lambda t: (others(t), len(t), t))
        if own[0] not in listed:
            listed.append(own[0])
    extra = sorted(f2p_set - set(listed))
    rng.shuffle(extra)
    want = min(MAX_LISTED, max(len(listed) + 2, 4))
    listed += extra[:max(0, want - len(listed))]
    return sorted(listed)
