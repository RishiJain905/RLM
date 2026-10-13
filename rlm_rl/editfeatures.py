"""Feature-removal tasks for edit_real (code-editing-plan §4.1, decision E-f).

Pick a public function or method that existing tests cover, replace its body with a stub that still
builds, and ask for it back. The tests that fail with the stub are the hidden fail-to-pass set
(found by validation, editdata.validate). The task text is written separately, from the function's
signature and doc comment and what the tests check, and must not quote the removed code.

Function boundaries come from Python's `ast`, and for Go, Rust, Java, TypeScript and JavaScript from
a brace matcher that skips strings, characters and comments; candidates whose boundaries look
doubtful (nested functions, one-line bodies, generated or vendored code) are left out.

| Language   | Stub                                                       |
|------------|------------------------------------------------------------|
| Python     | `raise NotImplementedError` (the docstring stays)          |
| Go         | `panic("not implemented")`, and imports left unused go too |
| Rust       | `unimplemented!()`                                         |
| Java       | `throw new UnsupportedOperationException("not implemented");` |
| TypeScript, JavaScript | `throw new Error("not implemented");`          |
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

MIN_BODY_LINES, MAX_BODY_LINES = 3, 45
SKIP_DIRS = {"vendor", "node_modules", "third_party", "thirdparty", "examples", "example", "_examples", "docs", "doc",
             "build", "dist", "target", "benches", "bench", "benchmarks", "testdata", "fixtures", ".github", "scripts"}
SOURCE_SUFFIX = {"python": (".py",), "go": (".go",), "rust": (".rs",), "java": (".java",),
                 "typescript": (".ts", ".tsx"), "javascript": (".js", ".mjs", ".cjs")}


@dataclass
class Function:
    path: str
    name: str
    qualname: str
    start: int  # 0-based line of the signature
    body_start: int  # 0-based first line of the body to replace
    body_end: int  # 0-based line after the body (exclusive)
    indent: str
    signature: str
    doc: str = ""
    tests: list = field(default_factory=list)  # test files that mention the name


def _skip(path: str, language: str, is_test) -> bool:
    parts = path.split("/")
    if any(p in SKIP_DIRS for p in parts[:-1]) or is_test(path, language):
        return True
    name = parts[-1]
    return name.endswith((".d.ts", ".min.js", ".pb.go", "_generated.go", ".gen.go")) or name.startswith("zz_")


# ----------------------------------------------------------------------------- Python


def python_functions(path: str, text: str) -> list[Function]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    lines = text.split("\n")
    out = []

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
                if name.startswith("_") or any(isinstance(d, ast.Name) and d.id in ("overload", "abstractmethod")
                                               or isinstance(d, ast.Attribute) and d.attr in ("overload", "abstractmethod")
                                               for d in child.decorator_list):
                    continue
                body = child.body
                doc = ast.get_docstring(child) or ""
                first = body[1] if doc and len(body) > 1 else body[0]
                if doc and len(body) == 1:
                    continue
                if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in ast.walk(child)
                       if n is not child):
                    continue  # nested definitions: boundaries are fine, but the stub would hide more than one thing
                body_start, body_end = first.lineno - 1, child.end_lineno
                n = body_end - body_start
                if not MIN_BODY_LINES <= n <= MAX_BODY_LINES:
                    continue
                if any(isinstance(n2, (ast.Yield, ast.YieldFrom)) for n2 in ast.walk(child)):
                    continue  # a generator's stub would change its type
                indent = re.match(r"\s*", lines[body_start]).group(0)
                sig = "\n".join(lines[child.lineno - 1:body[0].lineno - 1]).strip() if body else ""
                out.append(Function(path, name, f"{owner}.{name}" if owner else name, child.lineno - 1, body_start,
                                    body_end, indent, sig.split(":\n")[0], doc))
    visit(tree, None)
    return out


def python_stub(text: str, fn: Function) -> str:
    lines = text.split("\n")
    return "\n".join(lines[:fn.body_start] + [f"{fn.indent}raise NotImplementedError"] + lines[fn.body_end:])


# ----------------------------------------------------------------------------- brace languages


def _scan_braces(text: str, open_at: int, language: str) -> int | None:
    """Index just past the brace that closes the one at `open_at`, skipping strings and comments."""
    i, depth, n = open_at, 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if two == "/*":
            j = text.find("*/", i + 2)
            if j < 0:
                return None
            i = j + 2
            continue
        if c in "\"'`":
            if language == "rust" and c == "'":  # a char literal or a lifetime
                m = re.match(r"'(\\.|[^\\'])'", text[i:i + 6])
                i += len(m.group(0)) if m else 1
                continue
            if language == "go" and c == "'":
                m = re.match(r"'(\\.[^']*|[^\\'])'", text[i:i + 12])
                i += len(m.group(0)) if m else 1
                continue
            if language == "rust" and c == '"' and i > 0 and text[i - 1] == "r":
                hashes = 0
                j = i - 1
                end = '"'
                k = text.find(end, i + 1)
                i = n if k < 0 else k + 1
                continue
            quote = c
            if language == "go" and c == "`":
                k = text.find("`", i + 1)
                i = n if k < 0 else k + 1
                continue
            if language == "java" and text[i:i + 3] == '"""':
                k = text.find('"""', i + 3)
                i = n if k < 0 else k + 3
                continue
            j = i + 1
            while j < n and text[j] != quote:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "\n" and quote != "`":
                    break
                if quote == "`" and text[j:j + 2] == "${":  # template literal expression
                    end = _scan_braces(text, j + 1, language)
                    if end is None:
                        return None
                    j = end
                    continue
                j += 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


SIGNATURES = {
    "go": re.compile(r"^func (?:\([^)]*\) )?([A-Z]\w*)(?:\[[^\]]*\])?\([^)]*\)[^{\n]*\{\s*$", re.M),
    "rust": re.compile(r"^(\s*)pub(?:\([^)]*\))? (?:const |async |unsafe )*fn ([a-z]\w*)\b[^;{]*\{\s*$", re.M),
    "java": re.compile(r"^(\s*)(?:public|protected) (?!class|interface|enum|record|abstract)(?:static |final |synchronized )*"
                       r"(?:<[^>]+> )?[\w<>\[\],.? ]+ ([a-z]\w*)\([^)]*\)(?: throws [\w., ]+)?\s*\{\s*$", re.M),
    "typescript": re.compile(r"^(\s*)(?:export (?:default )?)?(?:async )?function\*? ?([A-Za-z]\w*)\s*(?:<[^>]*>)?\([^)]*\)"
                             r"(?:\s*:\s*[^{\n]+)?\s*\{\s*$|^(\s*)(?:public |static |async |override )*([a-z]\w*)\s*\([^)]*\)"
                             r"(?:\s*:\s*[^{\n]+)?\s*\{\s*$", re.M),
}
SIGNATURES["javascript"] = SIGNATURES["typescript"]
STUBS = {"go": 'panic("not implemented")', "rust": "unimplemented!()",
         "java": 'throw new UnsupportedOperationException("not implemented");',
         "typescript": 'throw new Error("not implemented");', "javascript": 'throw new Error("not implemented");'}
KEYWORDS = {"if", "for", "while", "switch", "catch", "function", "return", "else", "constructor", "super", "new", "do",
            "try", "with", "match", "loop", "get", "set", "static", "describe", "it", "test"}


def _doc_above(lines: list[str], start: int) -> str:
    doc = []
    i = start - 1
    while i >= 0 and (lines[i].strip().startswith(("//", "*", "/**", "///", "#[", "@")) or lines[i].strip() == "*/"):
        doc.append(lines[i].strip())
        i -= 1
    return "\n".join(reversed(doc))


def brace_functions(path: str, text: str, language: str) -> list[Function]:
    out = []
    lines = text.split("\n")
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line) + 1)
    in_test_block = _rust_test_ranges(text) if language == "rust" else []
    for m in SIGNATURES[language].finditer(text):
        groups = [g for g in m.groups() if g is not None]
        name = next((g for g in reversed(groups) if re.fullmatch(r"[A-Za-z]\w*", g)), None)
        if not name or name in KEYWORDS or name.startswith("_"):
            continue
        open_at = text.find("{", m.start(), m.end())
        close_at = _scan_braces(text, open_at, language)
        if close_at is None:
            continue
        line_of = lambda pos: max(i for i, s in enumerate(starts) if s <= pos)  # noqa: E731
        sig_line, open_line, close_line = line_of(m.start()), line_of(open_at), line_of(close_at - 1)
        if any(a <= sig_line <= b for a, b in in_test_block):
            continue
        body_start, body_end = open_line + 1, close_line  # lines strictly between the braces
        n = body_end - body_start
        if not MIN_BODY_LINES <= n <= MAX_BODY_LINES or lines[close_line].strip() != "}":
            continue
        body = "\n".join(lines[body_start:body_end])
        if language in ("typescript", "javascript") and re.search(r"\byield\b", body):
            continue
        indent = re.match(r"\s*", lines[sig_line]).group(0)
        inner = re.match(r"\s*", lines[body_start]).group(0) or indent + ("\t" if language == "go" else "    ")
        out.append(Function(path, name, name, sig_line, body_start, body_end, inner, lines[sig_line].strip(),
                            _doc_above(lines, sig_line)))
    return out


def _rust_test_ranges(text: str) -> list[tuple[int, int]]:
    from .grader import _rust_test_blocks

    return [(a - 1, b - 1) for a, b in _rust_test_blocks(text.split("\n"))]


def brace_stub(text: str, fn: Function, language: str) -> str:
    lines = text.split("\n")
    out = "\n".join(lines[:fn.body_start] + [f"{fn.indent}{STUBS[language]}"] + lines[fn.body_end:])
    return go_drop_unused_imports(out) if language == "go" else out


def go_drop_unused_imports(text: str) -> str:
    """Remove imports a stubbed Go file no longer uses (gofmt'd sources: one import per line)."""
    m = re.search(r"^import \(\n(.*?)^\)", text, re.M | re.S)
    single = re.findall(r'^import (?:(\w+) )?"([^"]+)"\s*$', text, re.M)
    block_lines = m.group(1).split("\n") if m else []
    keep, body = [], text[m.end():] if m else text
    for line in block_lines:
        mm = re.match(r'\s*(?:(\w+|\.) )?"([^"]+)"', line)
        if not mm:
            keep.append(line)
            continue
        alias, pathname = mm.group(1), mm.group(2)
        if alias in ("_", "."):
            keep.append(line)
            continue
        name = alias or pathname.rstrip("/").split("/")[-1]
        if re.fullmatch(r"v\d+", name):
            name = pathname.rstrip("/").split("/")[-2]
        name = name.replace("-", "_").split(".")[0]
        if re.search(rf"\b{re.escape(name)}\.", body):
            keep.append(line)
    if m:
        new_block = "import (\n" + "\n".join(keep) + ")"
        text = text[:m.start()] + new_block + text[m.end():]
    for alias, pathname in single:
        name = alias or pathname.split("/")[-1]
        rest = re.sub(r'^import .*$', "", text, flags=re.M)
        if alias not in ("_", ".") and not re.search(rf"\b{re.escape(name)}\.", rest):
            text = re.sub(rf'^import (?:{re.escape(alias)} )?"{re.escape(pathname)}"\s*\n', "", text, flags=re.M)
    return text


# ----------------------------------------------------------------------------- candidates


def functions(tree: Path, language: str, is_test) -> list[Function]:
    out = []
    for p in sorted(tree.rglob("*")):
        rel = str(p.relative_to(tree)).replace("\\", "/")
        if not p.is_file() or p.is_symlink() or not rel.endswith(SOURCE_SUFFIX[language]) or _skip(rel, language, is_test):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if language == "python":
            out.extend(python_functions(rel, text))
        else:
            out.extend(brace_functions(rel, text, language))
    return out


def test_mentions(tree: Path, language: str, is_test, names: set[str]) -> dict[str, list[str]]:
    """Test files that mention each name as a whole word."""
    mentions: dict[str, list[str]] = {n: [] for n in names}
    for p in sorted(tree.rglob("*")):
        rel = str(p.relative_to(tree)).replace("\\", "/")
        if not p.is_file() or not is_test(rel, language) or not rel.endswith(SOURCE_SUFFIX[language]):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        words = set(re.findall(r"[A-Za-z_]\w*", text))
        for n in names & words:
            mentions[n].append(rel)
    return mentions


def stub(text: str, fn: Function, language: str) -> str:
    return python_stub(text, fn) if language == "python" else brace_stub(text, fn, language)


# ----------------------------------------------------------------------------- our own injected bugs

# Operator and boundary mutations, the kinds SWE-smith's procedural bugs use (func_pm_flip_operators and the like),
# applied to one line of a function body that tests mention. Validation keeps one only if some test fails on it
# and passes on the fix.
MUTATIONS = [
    (r"(?<![=!<>])==(?!=)", "!=", "flipped equality"), (r"!=(?!=)", "==", "flipped inequality"),
    (r"===", "!==", "flipped strict equality"), (r"!==", "===", "flipped strict inequality"),
    (r"(?<![<>=!-])<=(?!=)", "<", "boundary"), (r"(?<![<>=!-])>=(?!=)", ">", "boundary"),
    (r"(?<![<>=!\-])<(?![<=>])", "<=", "boundary"), (r"(?<![<>=!\-])>(?![>=])", ">=", "boundary"),
    (r"&&", "||", "flipped logic"), (r"\|\|", "&&", "flipped logic"),
    (r"\+ 1\b", "- 1", "off by one"), (r"- 1\b", "+ 1", "off by one"), (r"\btrue\b", "false", "flipped flag"),
    (r"\bfalse\b", "true", "flipped flag"),
]
SKIP_LINE = re.compile(r"^\s*(//|/\*|\*|#|import |use |package |@)|<\w+(,\s*\w+)*>|->|=>\s*$")


def mutations(text: str, fn: Function, language: str, rng, limit: int = 6) -> list[tuple[str, str, int]]:
    """Up to `limit` single-line mutants of a function: (mutated text, kind, 0-based line)."""
    lines = text.split("\n")
    options = []
    for i in range(fn.body_start, fn.body_end):
        line = lines[i]
        if SKIP_LINE.search(line) or line.count('"') % 2 or "`" in line:
            continue
        # string contents masked at the same length, so offsets found in `code` apply to `line`
        code = re.sub(r"\"(\\.|[^\"\\])*\"|'(\\.|[^'\\])*'" if language != "rust" else r"\"(\\.|[^\"\\])*\"",
                      lambda m: m.group(0)[0] + "x" * (len(m.group(0)) - 2) + m.group(0)[-1], line)
        for pattern, repl, kind in MUTATIONS:
            for m in re.finditer(pattern, code):
                if language == "rust" and kind == "boundary" and re.search(r"[A-Za-z_]<|>::|&'", code):
                    continue  # generics and lifetimes, not comparisons
                if language in ("typescript", "java") and kind == "boundary" and re.search(r"\w<\w", code):
                    continue
                options.append((i, m.start(), m.end(), repl, kind))
    rng.shuffle(options)
    out, seen = [], set()
    for i, a, b, repl, kind in options:
        if i in seen:
            continue
        seen.add(i)
        new = lines[i][:a] + repl + lines[i][b:]
        out.append(("\n".join(lines[:i] + [new] + lines[i + 1:]), kind, i))
        if len(out) >= limit:
            break
    return out
