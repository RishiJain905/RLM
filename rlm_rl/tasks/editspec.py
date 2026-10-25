"""edit_spec: implement a cluster of interdependent functions from a requirements document (edit tasks v2).

Research basis: SWE-Dev (arXiv 2505.16975, feature development from a requirements document),
SWE-smith's LM-rewrite (a function's body removed, its signature and docstring kept) and Breakpoint
(arXiv 2506.00172, several linked functions at once). The task is built so that splitting it is the
efficient path: the root reads the document, plans the interfaces and their order, has sub-calls
draft the bodies (each fits in one ~4,000-token draft), integrates and tests.

scripts/build_edit_tasks_v2.py builds the pool from the permissive Python repositories edit_real
already validated (their derived images and snapshots, D:\\RLM-data\\edit): a cluster of k
call-graph neighbours that the repository's own tests cover has its bodies replaced by
`raise NotImplementedError` (signatures and docstrings stay), and a requirements document is written
from the signatures, docstrings, the module's role and how the rest of the code uses them. Hidden
tests: every existing test that fails on the stubbed repository and passes on the original
(fail-to-pass), with the other tests of those files (pass-to-pass); their files are taken out of the
repository, as for edit_real. Nothing from a hidden test is in the document.

Levels (functions stubbed, files they span): 1: 2-3 in one file; 2: 3-4 in one file; 3: 4-6 across
two files; 4: 6-8 across 2-3 files; 5: 8-12 across 2-4 files, or a whole module.

Splits: `edit_spec` trains; `edit_spec_select` and `edit_spec_test` come from repositories training
never sees (edit_real's selection and test repositories) and are eval-only (H7). The pool is fixed,
so drawing works as edit_real's: a seed picks a task at (or nearest) a level, and `exclude` keeps
retired tasks out.

This module also holds the host-side Python analysis both v2 families use (editbugs.py imports it):
the function inventory, the static call graph, stubs and cluster growth. It runs on the host only (H3).
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..env import Task
from . import PoolEmpty

GENERATOR_VERSION = "editspec-v1"
FAMILIES = ("edit_spec", "edit_spec_select", "edit_spec_test")
SPLIT = {"edit_spec": "train", "edit_spec_select": "select", "edit_spec_test": "test"}
MAX_LEVEL = 5
INDEX = "edit/v2/spec_index.json"  # relative to the data root
KIND = "spec"
# level -> ((fewest, most) functions, (fewest, most) files)
LEVELS = {1: ((2, 3), (1, 1)), 2: ((3, 4), (1, 1)), 3: ((4, 6), (2, 2)), 4: ((6, 8), (2, 3)), 5: ((8, 12), (2, 4))}
STUB = "raise NotImplementedError"


# ----------------------------------------------------------------------------- the fixed pool (shared with edit_bugs)


@functools.lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relevel_by_rank(records: list[dict], key: str, levels: int = MAX_LEVEL) -> list[dict]:
    """Copies of `records` whose levels are their rank by `key` (smallest first, then id) in `levels` equal bins.

    prod-v2 (2026-10-25): level 1 of build from a spec (2-3 functions in one file) held tasks base Qwen solved 11 times in
    12 and tasks it never solved. What told them apart was the size of the code to write: every task with gold_lines up
    to 21 but one was solved at least 4 times in 6, every task from 26 lines up but one at most twice. So the training
    split is levelled by gold_lines instead of the function count."""
    ranked = sorted(records, key=lambda r: (r.get(key) or 0, r["id"]))
    n = len(ranked)
    return [{**r, "level": min(levels, 1 + i * levels // n), "stored_level": r.get("level")} for i, r in enumerate(ranked)]


class Pool:
    """A fixed, validated task pool read from an index on the data drive, drawn like edit_real's (seed, nearest level,
    retired ids excluded). Records have edit_real's index format, so its workspace builder and the grader take them."""

    def __init__(self, name: str, families: tuple, split: dict, index: str, version: str, prefix: str,
                 relevel_train: str | None = None):
        self.name, self.families, self.split, self.index_rel = name, families, split, index
        self.version, self.prefix = version, prefix
        # A record field whose rank sets the training split's levels, in equal bins (relevel_by_rank); the eval
        # splits keep their stored levels, so the selection set is unchanged.
        self.relevel_train = relevel_train

    def index_path(self) -> Path:
        from .. import data

        return data.path(self.index_rel)

    def index(self) -> dict:
        path = self.index_path()
        if not path.exists():
            raise FileNotFoundError(f"no {self.name} task index at {path}: build it with scripts/build_edit_tasks_v2.py")
        return _load(str(path), path.stat().st_mtime)

    def records(self, split: str) -> list[dict]:
        records = [r for r in self.index()["tasks"] if r["split"] == split]
        if self.relevel_train and split == self.split[self.families[0]]:
            records = relevel_by_rank(records, self.relevel_train)
        return records

    def training_sources(self) -> tuple[str, ...]:
        """The data sources the training split reads (rlm_rl/data.py SOURCES), for the licence bar (7b)."""
        try:
            return tuple(sorted({r["source"] for r in self.records("train")}))
        except FileNotFoundError:
            return ()

    def check_licences(self) -> None:
        from .editreal import PERMISSIVE

        for r in self.records("train"):
            if r.get("licence") not in PERMISSIVE:
                raise ValueError(f"{self.name}: {r['repo']} is licensed {r.get('licence')!r}, which the licence bar "
                                 "keeps out of training (7b)")

    def pick(self, pool: list[dict], level: int, rng: random.Random) -> dict:
        by_level: dict[int, list] = {}
        for r in pool:
            by_level.setdefault(int(r["level"]), []).append(r)
        if not by_level:
            raise PoolEmpty(f"no {self.name} task left to draw")
        best = min(by_level, key=lambda lv: (abs(lv - level), lv > level))  # the nearest level, a lower one on a tie
        return rng.choice(sorted(by_level[best], key=lambda r: r["id"]))

    def make_task(self, seed: int, family: str, level: int = 1, task_id: str | None = None, exclude=()) -> Task:
        if family not in self.families:
            raise ValueError(f"unknown {self.name} family {family!r}")
        excluded = set(exclude or ())
        pool = [r for r in self.records(self.split[family]) if r["id"] not in excluded]
        return self.to_task(self.pick(pool, level, random.Random(seed)), family, task_id)

    def pool_size(self, family: str) -> int:
        return len(self.records(self.split[family]))

    def eval_tasks(self, split: str, family: str) -> list[Task]:
        return [self.to_task(r, family, f"{family}-{i:03d}")
                for i, r in enumerate(sorted(self.records(split), key=lambda r: r["id"]))]

    def to_task(self, record: dict, family: str, task_id: str | None = None) -> Task:
        from ..editbox import Workspace
        from .editreal import _build_workspace, _context

        key_src = json.dumps({k: record.get(k) for k in ("id", "image", "snapshot", "snapshot_sha256", "break_patch",
                                                         "remove", "replace")}, sort_keys=True)
        key = f"{self.prefix}-" + hashlib.sha256((self.version + key_src).encode()).hexdigest()[:24]
        targets = record.get("targets") or []
        grading = {"type": "edit", "kind": record["kind"], "language": record["language"], "repo": record["repo"],
                   "tests": record["tests"], "gold_patch": record.get("gold_patch"), "source": record.get("source"),
                   "instance_id": None, "record_id": record["id"], "files_to_change": record.get("gold_files"),
                   "gold_lines": record.get("gold_lines"), "targets": targets, "k": len(targets), "blind": None}
        edit = {"language": record["language"], "n_files": len(record["files"]), "test_hint": record["test_hint"],
                "image": record["image"], "repo": record["repo"], "kind": record["kind"]}
        if record.get("cell_timeout_s"):
            edit["cell_timeout_s"] = int(record["cell_timeout_s"])
        return Task(task_id or record["id"], family, record["issue"], _context(record), answer=None,
                    level=int(record["level"]), grading=grading, edit=edit,
                    workspace=Workspace(key, functools.partial(_build_workspace, record)))


POOL = Pool("edit_spec", FAMILIES, SPLIT, INDEX, GENERATOR_VERSION, "spec", relevel_train="gold_lines")


def index_path() -> Path:
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


def pool_size(family: str = "edit_spec") -> int:
    return POOL.pool_size(family)


def eval_tasks(split: str, family: str) -> list[Task]:
    return POOL.eval_tasks(split, family)


def to_task(record: dict, family: str, task_id: str | None = None) -> Task:
    return POOL.to_task(record, family, task_id)


# ----------------------------------------------------------------------------- Python analysis (host only)

SKIP_DIRS = {"vendor", "node_modules", "third_party", "thirdparty", "examples", "example", "_examples", "docs", "doc",
             "build", "dist", "benches", "bench", "benchmarks", "benchmark", "testdata", "fixtures", ".github",
             "scripts", "performance", "tests", "test"}
SKIP_FILES = {"setup.py", "conftest.py", "noxfile.py", "__main__.py", "_version.py", "version.py"}
MIN_BODY, MAX_BODY = 2, 60  # lines a stubbed or mutated body may have


class Source:
    """A file's text with line offsets; ast columns are UTF-8 byte offsets, turned into character offsets here."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.split("\n")
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line) + 1)

    def offset(self, lineno: int, col: int) -> int:
        line = self.lines[lineno - 1]
        return self.starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8", "ignore"))

    def span(self, node) -> tuple[int, int]:
        return self.offset(node.lineno, node.col_offset), self.offset(node.end_lineno, node.end_col_offset)

    def segment(self, node) -> str:
        a, b = self.span(node)
        return self.text[a:b]


@dataclass
class Fn:
    key: str  # "path::Qual.name"
    path: str
    module: str
    qualname: str
    name: str
    cls: str | None
    first: int  # 1-based line of the first decorator, or of `def`
    def_line: int  # 1-based
    end: int  # 1-based last line
    body_start: int  # 0-based first line a stub replaces (after the docstring)
    body_end: int  # 0-based, exclusive
    indent: str
    signature: str  # decorators and the `def` line(s)
    doc: str
    kind: str  # function, method, classmethod, staticmethod, property
    generator: bool = False
    is_async: bool = False
    stubbable: bool = False
    refs: set = field(default_factory=set)  # function keys this one references (calls or passes)

    @property
    def body_lines(self) -> int:
        return self.body_end - self.body_start

    @property
    def public(self) -> bool:
        return not self.name.startswith("_")


@dataclass
class Usage:
    target: str  # function key used
    where: str  # function key using it, or "path::<module>"
    path: str
    line: int  # 1-based
    text: str  # the stripped source line


@dataclass
class Module:
    path: str
    name: str
    source: Source
    tree: ast.Module
    functions: dict = field(default_factory=dict)  # top-level name -> fn key
    classes: dict = field(default_factory=dict)  # name -> {"methods": {name: key}, "bases": [ast exprs], "doc": str}
    imports: dict = field(default_factory=dict)  # alias -> ("module", dotted) | ("from", dotted, attr)
    doc: str = ""


@dataclass
class Repo:
    modules: dict  # path -> Module
    by_name: dict  # dotted -> path
    fns: dict  # key -> Fn
    usages: list  # Usage, resolved references from anywhere in the source
    import_time: set = field(default_factory=set)  # functions run when their module is imported (static_import_time)


def is_source(path: str) -> bool:
    from ..grader import is_test_path

    parts = path.split("/")
    return (path.endswith(".py") and not is_test_path(path, "python") and not any(p in SKIP_DIRS for p in parts[:-1])
            and parts[-1] not in SKIP_FILES and not parts[-1].startswith("test"))


def module_names(paths) -> dict[str, str]:
    """Each source file's dotted module name, from the package folders (with __init__.py) above it."""
    pathset = set(paths)
    out = {}
    for p in paths:
        parts = p[:-3].split("/")
        dirs = parts[:-1]
        i = len(dirs)
        while i > 0 and "/".join(dirs[:i]) + "/__init__.py" in pathset:
            i -= 1
        mod = parts[i:]
        if mod and mod[-1] == "__init__":
            mod = mod[:-1]
        if mod:
            out[p] = ".".join(mod)
    return out


def _is_stub_body(body: list) -> bool:
    if len(body) != 1:
        return False
    b = body[0]
    if isinstance(b, ast.Pass) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)):
        return True
    if isinstance(b, ast.Raise) and b.exc is not None:
        exc = b.exc.func if isinstance(b.exc, ast.Call) else b.exc
        return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"
    return isinstance(b, ast.Return) and isinstance(b.value, ast.Name) and b.value.id == "NotImplemented"


def _walk_own(node):
    """The nodes of a function's own body, not of functions, lambdas or classes nested in it."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(n))


def _make_fn(mod: Module, node, cls: str | None) -> Fn:
    lines = mod.source.lines
    deco = [d.lineno for d in node.decorator_list]
    first = min(deco + [node.lineno])
    doc = ast.get_docstring(node) or ""
    body = node.body
    stmts = body[1:] if doc and isinstance(body[0], ast.Expr) else body
    names = {d.id if isinstance(d, ast.Name) else d.attr if isinstance(d, ast.Attribute) else
             (d.func.id if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) else "")
             for d in node.decorator_list}
    kind = ("property" if names & {"property", "cached_property", "setter", "getter"} else "classmethod"
            if "classmethod" in names else "staticmethod" if "staticmethod" in names else
            "method" if cls else "function")
    qual = f"{cls}.{node.name}" if cls else node.name
    sig_end = body[0].lineno - 1  # lines before the docstring or first statement
    signature = "\n".join(lines[first - 1:sig_end]).rstrip()
    fn = Fn(key=f"{mod.path}::{qual}", path=mod.path, module=mod.name, qualname=qual, name=node.name, cls=cls,
            first=first, def_line=node.lineno, end=node.end_lineno, body_start=0, body_end=0, indent="",
            signature=signature, doc=doc, kind=kind, is_async=isinstance(node, ast.AsyncFunctionDef))
    fn.generator = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in _walk_own(node))
    if not stmts or _is_stub_body(stmts) or names & {"overload", "abstractmethod", "abstractproperty"}:
        return fn
    if node.name.startswith("__") and node.name.endswith("__"):
        return fn
    start = stmts[0]
    if start.lineno == node.lineno or (doc and start.lineno <= body[0].end_lineno):
        return fn  # a one-line function, or a statement on the docstring's line
    line = lines[start.lineno - 1]
    indent = re.match(r"\s*", line).group(0)
    if len(indent.encode()) != start.col_offset:
        return fn  # the first statement shares its line with the signature
    fn.body_start, fn.body_end, fn.indent = start.lineno - 1, node.end_lineno, indent
    fn.stubbable = MIN_BODY <= fn.body_lines <= MAX_BODY
    return fn


def _parse_module(path: str, name: str, text: str) -> Module | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    mod = Module(path, name, Source(text), tree, doc=ast.get_docstring(tree) or "")
    package = name if path.endswith("__init__.py") else name.rpartition(".")[0]
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    mod.imports[a.asname] = ("module", a.name)
                else:
                    mod.imports[a.name.split(".")[0]] = ("module", a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join(parts + ([node.module] if node.module else []))
            for a in node.names:
                if a.name != "*":
                    mod.imports[a.asname or a.name] = ("from", base, a.name)
    return mod


def analyze(files: dict[str, str]) -> Repo:
    """The function inventory, imports and resolved references of a repository's Python source (tests excluded)."""
    srcs = {p: t for p, t in files.items() if is_source(p)}
    names = module_names(srcs)
    modules, fns = {}, {}
    for path in sorted(names):
        mod = _parse_module(path, names[path], srcs[path].replace("\r\n", "\n"))
        if mod is None:
            continue
        modules[path] = mod
        for node in mod.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = _make_fn(mod, node, None)
                fns[fn.key] = fn
                mod.functions[node.name] = fn.key
            elif isinstance(node, ast.ClassDef):
                info = {"methods": {}, "bases": list(node.bases), "doc": ast.get_docstring(node) or "", "node": node}
                mod.classes[node.name] = info
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fn = _make_fn(mod, child, node.name)
                        # a property's setter has the getter's name: keep the first definition only
                        if child.name not in info["methods"]:
                            fns[fn.key] = fn
                            info["methods"][child.name] = fn.key
    repo = Repo(modules, {m.name: p for p, m in modules.items()}, fns, [])
    _resolve_refs(repo)
    repo.import_time = static_import_time(repo)
    return repo


def static_import_time(repo: Repo) -> set[str]:
    """Functions that run when a module is imported: decorators, and calls in module or class bodies (outside any
    function). Stubbing one breaks every import of the package; the profile's collection list may miss them when a
    conftest imported the package before the tracer started."""
    out = set()
    for mod in repo.modules.values():
        def target(expr, mod=mod):
            t = _expr_target(repo, mod, expr.func if isinstance(expr, ast.Call) else expr, None)
            if t and t[0] == "fn" and t[1] in repo.fns:
                out.add(t[1])

        def visit(node, target=target):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for d in child.decorator_list:
                        target(d)
                        visit(d)
                    for d in child.args.defaults + [d for d in child.args.kw_defaults if d is not None]:
                        visit(d)
                    continue  # the body runs when called, not at import
                if isinstance(child, ast.Lambda):
                    continue
                if isinstance(child, ast.ClassDef):
                    for d in child.decorator_list:
                        target(d)
                if isinstance(child, ast.Call):
                    target(child)
                visit(child)

        visit(mod.tree)
    return out


def _lookup(repo: Repo, mod: Module, name: str, depth: int = 0):
    """What a module-level name is: ("fn", key), ("class", path, name), ("module", path) or None."""
    if depth > 4:
        return None
    if name in mod.functions:
        return ("fn", mod.functions[name])
    if name in mod.classes:
        return ("class", mod.path, name)
    imp = mod.imports.get(name)
    if imp is None:
        return None
    if imp[0] == "module":
        return ("module", repo.by_name[imp[1]]) if imp[1] in repo.by_name else None
    _, base, attr = imp
    if f"{base}.{attr}" in repo.by_name:
        return ("module", repo.by_name[f"{base}.{attr}"])
    if base in repo.by_name:
        return _lookup(repo, repo.modules[repo.by_name[base]], attr, depth + 1)
    return None


def _method(repo: Repo, path: str, cls: str, name: str, depth: int = 0) -> str | None:
    if depth > 4 or path not in repo.modules:
        return None
    mod = repo.modules[path]
    info = mod.classes.get(cls)
    if info is None:
        return None
    if name in info["methods"]:
        return info["methods"][name]
    for base in info["bases"]:
        target = _expr_target(repo, mod, base, None)
        if target and target[0] == "class":
            found = _method(repo, target[1], target[2], name, depth + 1)
            if found:
                return found
    return None


def _chain(node) -> list[str] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id] + parts[::-1]
    return None


def _expr_target(repo: Repo, mod: Module, node, fn: Fn | None):
    """What a Name or attribute chain refers to, where it can be told statically."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) \
            and node.value.func.id == "super" and fn is not None and fn.cls:
        info = mod.classes.get(fn.cls) or {}
        for base in info.get("bases", []):
            t = _expr_target(repo, mod, base, None)
            if t and t[0] == "class":
                found = _method(repo, t[1], t[2], node.attr)
                if found:
                    return ("fn", found)
        return None
    parts = _chain(node)
    if not parts:
        return None
    head, rest = parts[0], parts[1:]
    if head in ("self", "cls") and fn is not None and fn.cls:
        if len(rest) == 1:
            found = _method(repo, mod.path, fn.cls, rest[0])
            return ("fn", found) if found else None
        return None
    target = _lookup(repo, mod, head)
    for attr in rest:
        if target is None:
            return None
        if target[0] == "module":
            sub = repo.modules[target[1]]
            dotted = f"{sub.name}.{attr}"
            target = ("module", repo.by_name[dotted]) if dotted in repo.by_name else _lookup(repo, sub, attr)
        elif target[0] == "class":
            found = _method(repo, target[1], target[2], attr)
            target = ("fn", found) if found else None
        else:
            return None
    return target


def _local_names(node) -> set[str]:
    """Names a function binds itself (parameters, assignments, loop, `with` and `except` targets, local imports), so
    a use of one is never taken for the module-level function of the same name. Nested functions count too."""
    out, glob = set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not node:
            out.add(n.name)
        elif isinstance(n, ast.Global):
            glob |= set(n.names)
    return out - glob


def _resolve_refs(repo: Repo) -> None:
    for path, mod in repo.modules.items():
        owners = []  # (node, fn or None) whose own body we scan
        for node in mod.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owners.append((node, repo.fns.get(mod.functions.get(node.name, ""))))
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        key = mod.classes[node.name]["methods"].get(child.name)
                        fn = repo.fns.get(key) if key else None
                        if fn is not None and fn.def_line == child.lineno:
                            owners.append((child, fn))
                    else:
                        owners.append((child, None))
            else:
                owners.append((node, None))
        for node, fn in owners:
            seen_nodes = set()
            walk = [node] if fn is None else list(ast.iter_child_nodes(node))
            local = _local_names(node) if fn is not None else set()
            stack = walk
            while stack:
                n = stack.pop()
                if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(getattr(n, "ctx", None), ast.Load) \
                        and id(n) not in seen_nodes:
                    head = (_chain(n) or [""])[0]
                    if head in local and head not in ("self", "cls"):
                        stack.extend(ast.iter_child_nodes(n))
                        continue  # a parameter or local variable that shadows a module-level name
                    target = _expr_target(repo, mod, n, fn)
                    if target and target[0] == "fn" and target[1] in repo.fns:
                        # the whole chain is resolved: its inner parts aren't separate references
                        inner = n.value if isinstance(n, ast.Attribute) else None
                        while isinstance(inner, ast.Attribute):
                            seen_nodes.add(id(inner))
                            inner = inner.value
                        if fn is None or target[1] != fn.key:
                            if fn is not None:
                                fn.refs.add(target[1])
                            where = fn.key if fn is not None else f"{path}::<module>"
                            line = mod.source.lines[n.lineno - 1].strip()
                            repo.usages.append(Usage(target[1], where, path, n.lineno, line))
                stack.extend(ast.iter_child_nodes(n))


# ----------------------------------------------------------------------------- tests and the call map


def profile_index(repo: Repo) -> dict[tuple[str, str], list[tuple[int, int, str]]]:
    """(path, function name) -> [(first line, def line, key)], to match a profiled code object to a function."""
    out: dict = {}
    for fn in repo.fns.values():
        out.setdefault((fn.path, fn.name), []).append((fn.first, fn.def_line, fn.key))
    return out


def map_profile(repo: Repo, profile: dict[str, list[str]]) -> tuple[dict[str, set], dict[str, set], set]:
    """From the lab's per-test call lists ("path:firstlineno:name"): (test -> functions it ran, function -> tests that
    ran it, functions run while the tests were collected, i.e. at import)."""
    idx = profile_index(repo)
    cache: dict[str, str | None] = {}

    def key_of(entry: str) -> str | None:
        if entry not in cache:
            path, line, name = entry.rsplit(":", 2)
            hit = None
            for first, def_line, key in idx.get((path, name), ()):
                if first <= int(line) <= def_line:
                    hit = key
            cache[entry] = hit
        return cache[entry]

    calls, tests_of, at_import = {}, {}, set()
    for test, entries in profile.items():
        keys = {k for k in map(key_of, entries) if k}
        if test == "<collect>":
            at_import = keys
            continue
        calls[test] = keys
        for k in keys:
            tests_of.setdefault(k, set()).add(test)
    return calls, tests_of, at_import


def test_file(test_key: str) -> str:
    return test_key.split("::", 1)[0]


# ----------------------------------------------------------------------------- stubs


def split_newlines(raw: str) -> tuple[str, str] | None:
    """A file's text with \\n line ends and the line end to write it back with; None for mixed line ends."""
    crlf = raw.count("\r\n")
    if 0 < crlf < raw.count("\n"):
        return None
    return (raw.replace("\r\n", "\n"), "\r\n") if crlf else (raw, "\n")


def stub_text(text: str, fns: list[Fn]) -> str:
    """`text` (\\n line ends) with each function's body, after its docstring, replaced by `raise NotImplementedError`."""
    lines = text.split("\n")
    for fn in sorted(fns, key=lambda f: f.body_start, reverse=True):
        lines[fn.body_start:fn.body_end] = [fn.indent + STUB]
    out = "\n".join(lines)
    ast.parse(out)  # still valid Python
    return out


# ----------------------------------------------------------------------------- clusters


def graph(repo: Repo, cand: set[str], tests_of: dict[str, set]) -> tuple[dict, dict]:
    """(static neighbours, co-tested neighbours: functions some test runs together) among candidate functions, both
    undirected."""
    static = {k: set() for k in cand}
    for k in cand:
        for r in repo.fns[k].refs:
            if r in cand and r != k:
                static[k].add(r)
                static[r].add(k)
    co = {k: set() for k in cand}
    keys = sorted(cand)
    for i, a in enumerate(keys):
        ta = tests_of.get(a, set())
        for b in keys[i + 1:]:
            if not ta.isdisjoint(tests_of.get(b, ())):
                co[a].add(b)
                co[b].add(a)
    return static, co


def grow(repo: Repo, seed: str, k: int, files: tuple[int, int], static: dict, co: dict, rng: random.Random,
         allowed: set) -> list[str] | None:
    """A connected cluster of k functions from `seed`: static call-graph neighbours first, then functions tested
    together with a member in the same file, then in another file; it spans between files[0] and files[1] files."""
    cluster, fileset = [seed], {repo.fns[seed].path}
    while len(cluster) < k:
        need = files[0] - len(fileset)  # new files the cluster still has to reach
        left = k - len(cluster)
        options = []
        for member in cluster:
            here = repo.fns[member].path
            for nbrs, static_edge in ((static[member], True), (co[member], False)):
                for n in nbrs:
                    if n in cluster or n not in allowed:
                        continue
                    tier = 0 if static_edge else 1 if repo.fns[n].path == here else 2
                    new_file = repo.fns[n].path not in fileset
                    if (new_file and len(fileset) >= files[1]) or (need > 0 and left <= need and not new_file):
                        continue
                    # while it spans too few files, a function in a new one first; after that, stay in its files
                    place = 0 if new_file == (need > 0) else 1
                    options.append(((place, tier, 0 if repo.fns[n].doc else 1, rng.random()), n))
        if not options:
            return None
        options.sort()
        n = options[0][1]
        cluster.append(n)
        fileset.add(repo.fns[n].path)
    if not files[0] <= len(fileset) <= files[1]:
        return None
    return cluster


def eligible(repo: Repo, tests_of: dict[str, set], at_import: set) -> set[str]:
    """Functions a task may stub or mutate: a body of the right size, run by at least one test, never at import (by
    the profile or by static_import_time)."""
    return {k for k, fn in repo.fns.items() if fn.stubbable and tests_of.get(k) and k not in at_import
            and k not in repo.import_time}


def test_files_of(keys, tests_of: dict[str, set]) -> set[str]:
    """The test files whose tests run any of these functions: what the task would hide."""
    return {test_file(t) for k in keys for t in tests_of.get(k, ())}


def spec_clusters(repo: Repo, tests_of: dict[str, set], at_import: set, level: int, rng: random.Random, want: int,
                  used: dict | None = None, max_use: int = 2, max_tests: int | None = None, total: dict | None = None,
                  max_total: int = 5, max_files: int | None = None) -> list[list[str]]:
    """Up to `want` distinct clusters for a level, each function in at most `max_use` of the level's clusters (`used`)
    and `max_total` of the family's (`total`, across levels)."""
    (kmin, kmax), files = LEVELS[level]
    max_tests = max_tests or {1: 150, 2: 200, 3: 300, 4: 400, 5: 500}[level]
    max_lines = {1: 70, 2: 110, 3: 170, 4: 240, 5: 380}[level]
    used = used if used is not None else {}
    total = total if total is not None else {}
    cand = eligible(repo, tests_of, at_import)
    static, co = graph(repo, cand, tests_of)
    seeds = sorted(cand)
    rng.shuffle(seeds)
    seeds.sort(key=lambda s: (used.get(s, 0) + total.get(s, 0), 0 if repo.fns[s].doc else 1))
    out, seen = [], set()
    for seed in seeds:
        if len(out) >= want:
            break
        allowed = {c for c in cand if used.get(c, 0) < max_use and total.get(c, 0) < max_total}
        if seed not in allowed:
            continue
        k = rng.randint(kmin, kmax)
        cluster = grow(repo, seed, k, files, static, co, rng, allowed)
        if level == 5 and cluster is None:
            cluster = whole_module(repo, repo.fns[seed].path, allowed, kmin, kmax + 4)
        if not cluster or frozenset(cluster) in seen:
            continue
        lines = sum(repo.fns[c].body_lines for c in cluster)
        tests = set().union(*(tests_of[c] for c in cluster))
        if lines > max_lines or lines < 2 * len(cluster) + 2 or len(tests) > max_tests:
            continue
        if max_files and len(test_files_of(cluster, tests_of)) > max_files:
            continue  # its tests span too many files to hide
        if not any(repo.fns[c].doc for c in cluster):
            continue  # nothing in it is documented: the requirements would be guesswork
        seen.add(frozenset(cluster))
        for c in cluster:
            used[c] = used.get(c, 0) + 1
            total[c] = total.get(c, 0) + 1
        out.append(sorted(cluster, key=lambda c: (repo.fns[c].path, repo.fns[c].def_line)))
    return out


def whole_module(repo: Repo, path: str, allowed: set, kmin: int, kmax: int) -> list[str] | None:
    keys = [k for k, fn in repo.fns.items() if fn.path == path and k in allowed and fn.stubbable]
    return keys if kmin <= len(keys) <= kmax else None


# ----------------------------------------------------------------------------- the requirements document


def _first_paragraph(doc: str, limit: int = 400) -> str:
    para = (doc or "").strip().split("\n\n")[0]
    para = " ".join(para.split())
    return para if len(para) <= limit else para[:limit].rsplit(" ", 1)[0] + " ..."


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line if line.strip() else "" for line in text.split("\n"))


def _describe_kind(fn: Fn) -> str:
    bits = []
    if fn.kind == "property":
        bits.append(f"a property of `{fn.cls}`")
    elif fn.kind in ("classmethod", "staticmethod"):
        bits.append(f"a {fn.kind.replace('method', ' method')} of `{fn.cls}`")
    elif fn.cls:
        bits.append(f"a method of `{fn.cls}`")
    else:
        bits.append("a module-level function")
    if fn.generator:
        bits.append("a generator (it yields its results)")
    if fn.is_async:
        bits.append("a coroutine")
    return ", ".join(bits)


DOC_MAX = 1500  # characters of a docstring quoted in the document

GENERIC_METHODS = {"get", "set", "items", "keys", "values", "append", "extend", "update", "pop", "add", "copy", "run",
                   "process", "format", "parse", "load", "dump", "dumps", "loads", "read", "write", "close", "open",
                   "render", "match", "split", "join", "strip", "replace", "find", "index", "count", "sort", "clear",
                   "remove", "insert", "send", "next", "reset", "call", "apply", "visit", "compile", "build", "check"}


def stubbed_lines(repo: Repo, cluster) -> set[str]:
    """The stripped lines a cluster's stubs take out: the document never quotes one (a sibling method may share it)."""
    out = set()
    for k in cluster:
        fn = repo.fns[k]
        out |= {line.strip() for line in repo.modules[fn.path].source.lines[fn.body_start:fn.body_end] if line.strip()}
    return out


def usage_lines(repo: Repo, target: str, cluster: set[str], limit: int = 4) -> list[str]:
    """How code outside the cluster uses a function: resolved references first, then, for a distinctive method name,
    attribute calls on objects the analysis can't type. A line that also appears in a stubbed body is left out."""
    fn = repo.fns[target]
    hidden = stubbed_lines(repo, cluster)
    long_hidden = [h for h in hidden if len(h) >= 25]  # editdata.leaks' threshold

    def quotes(text: str) -> bool:
        return text in hidden or any(h in text for h in long_hidden)

    out, seen = [], set()
    for u in sorted(repo.usages, key=lambda u: (u.path != fn.path, u.path, u.line)):
        if u.target != target or u.where in cluster or quotes(u.text):
            continue
        if (u.path, u.line) in seen:
            continue
        seen.add((u.path, u.line))
        where = u.where.split("::", 1)[1]
        where = "module level" if where == "<module>" else f"`{where}`"
        out.append(f"`{u.path}`, {where}: `{u.text[:160]}`")
        if len(out) >= limit:
            return out
    unique = sum(1 for f in repo.fns.values() if f.name == fn.name) == 1  # else `.name(` may be another class's
    if fn.cls and unique and fn.name not in GENERIC_METHODS and len(fn.name) >= 6:
        pattern = re.compile(rf"\.{re.escape(fn.name)}\(")
        stubbed = {(repo.fns[c].path, line) for c in cluster
                   for line in range(repo.fns[c].first, repo.fns[c].end + 1)}
        for path, mod in sorted(repo.modules.items()):
            for i, line in enumerate(mod.source.lines, 1):
                if (path, i) in seen or (path, i) in stubbed or not pattern.search(line) or quotes(line.strip()):
                    continue
                seen.add((path, i))
                out.append(f"`{path}`, line {i}: `{line.strip()[:160]}`")
                if len(out) >= limit:
                    return out
    return out


def requirements(repo_name: str, repo: Repo, cluster: list[str]) -> str:
    """The task's requirements document: what each missing function is for, its interface, its documentation, how the
    rest of the code uses it, and how the missing pieces depend on each other. Nothing from the hidden tests."""
    members = set(cluster)
    fns = [repo.fns[k] for k in cluster]
    files = sorted({f.path for f in fns})
    short = repo_name.split("/")[-1]
    parts = [f"Implement the missing code in {short} ({len(fns)} functions in {len(files)} "
             f"file{'s' if len(files) > 1 else ''}).",
             "Each function or method listed below has had its body replaced by `raise NotImplementedError`; its "
             "signature and docstring are unchanged. Write the bodies so that the library behaves as its documentation "
             "and its callers expect. Hidden tests exercise this code, and the repository's other tests must keep "
             "passing. Don't change any signature, and don't change the tests."]
    n = 0
    for path in files:
        mod = repo.modules[path]
        section = [f"## `{path}`"]
        if mod.doc:
            section.append(f"Module `{mod.name}`: {_first_paragraph(mod.doc)}")
        for fn in [f for f in fns if f.path == path]:
            n += 1
            lines = [f"### {n}. `{fn.qualname}`", "```python", fn.signature, "```", f"It is {_describe_kind(fn)}."]
            if fn.cls:
                cdoc = _first_paragraph(mod.classes.get(fn.cls, {}).get("doc", ""), 300)
                if cdoc:
                    lines.append(f"`{fn.cls}`: {cdoc}")
            doc = fn.doc.strip()
            doc = doc if len(doc) <= DOC_MAX else doc[:DOC_MAX].rsplit("\n", 1)[0] + "\n[... the rest is in the file]"
            lines.append("Docstring:\n" + _indent(doc) if doc else "It has no docstring.")
            uses = sorted(r for r in fn.refs if r in members)
            if uses:
                lines.append("It relies on these other missing functions: "
                             + ", ".join(f"`{repo.fns[u].qualname}`" for u in uses) + ".")
            users = sorted(k for k in members if fn.key in repo.fns[k].refs)
            if users:
                lines.append("It is used by these other missing functions: "
                             + ", ".join(f"`{repo.fns[u].qualname}`" for u in users) + ".")
            seen = usage_lines(repo, fn.key, members)
            if seen:
                lines.append("How the rest of the code uses it:\n" + "\n".join(f"- {s}" for s in seen))
            section.append("\n".join(lines))
        parts.append("\n\n".join(section))
    order = build_order(repo, cluster)
    if len(cluster) > 1 and any(repo.fns[k].refs & members for k in cluster):
        parts.append("## How the pieces fit\nA dependency order (each function after the missing ones it relies on): "
                     + ", ".join(f"`{repo.fns[k].qualname}`" for k in order) + ".")
    return "\n\n".join(parts).strip() + "\n"


def build_order(repo: Repo, cluster: list[str]) -> list[str]:
    """The cluster with each function after the members it references (cycles broken by definition order)."""
    members = set(cluster)
    done, out = set(), []

    def visit(k, stack):
        if k in done or k in stack:
            return
        stack.add(k)
        for r in sorted(repo.fns[k].refs & members):
            visit(r, stack)
        stack.discard(k)
        done.add(k)
        out.append(k)

    for k in cluster:
        visit(k, set())
    return out
