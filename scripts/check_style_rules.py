"""Project style-rule checker invoked by ``make lint``.

Catches drift that ruff cannot express. The first four checks run across the
whole tree (a pattern is forbidden everywhere); the fifth runs only on lines
added relative to the base branch (it stops NEW smells without forcing a
cleanup of pre-existing ones).

1. Em dashes (``—``) in ``src/`` or ``tests/``. Project rule from AGENTS.md.
2. Divider comments (``# ----`` / ``# ====``) used to group code in ``src/``.
   Project rule: prefer modules/classes for grouping.
3. Historical-narrative docstrings in ``src/`` (``previously``, ``used to``,
   ``migrated from``, ``preserves the historical``, ``for backward``). Project
   rule: docstrings describe what the code IS, not what it was.
4. Stale single-file path references in ``src/`` for modules that have since
   become packages (``catalog.py``, ``store.py``, ``gen.py``, ``ingest.py``,
   ``commands.py``, ``handlers.py``, ``api.py``, ``clustering_embedding.py``,
   ``worker_process.py``), and the phrase ``original X.py``. Project rule:
   docstrings name the current path.
5. New occurrences of the AGENTS.md "Code-Smell Triggers" (getattr-by-name on
   owned attributes, getattr-with-default on typed fields, owned-attribute
   type-ignores, production host-narrowing, string-typed closed sets,
   module-level mutable globals) on lines added in ``src/`` vs the base
   branch. Resolving the base is best-effort: when git history is unavailable
   (shallow CI checkout, no ``origin/main``) the check is skipped, not failed.

Inline opt-out comments: ``# style-check: allow-history`` skips the
historical-narrative check on that line; ``# style-check: allow-smell`` skips
the code-smell check on that added line (use it with a written justification
for genuinely dynamic reflection).

Exits 0 when clean, 1 with one ``path:line:reason`` per finding when violations
are found.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"

EM_DASH_EXCLUDED_FILENAMES: set[str] = set()

EM_DASH = "—"
DIVIDER_RE = re.compile(r"^\s*#\s*[-=]{4,}\s*$")

# Patterns that flag back-compat scaffolding language (the AGENTS.md
# "No Back-Compat Scaffolding" rule's surface symptoms). Each is matched
# case-insensitively against a single line. The patterns are deliberately
# narrow: they catch phrasings that almost only appear in scaffolding
# docstrings and comments, not generic descriptions of current behaviour.
HISTORICAL_PATTERNS = (
    re.compile(r"\bpreviously\s*[,;]", re.IGNORECASE),
    re.compile(r"\bpreviously\s+(this|the|we|it|they|all|each)\b", re.IGNORECASE),
    re.compile(r"\bpreserves the historical\b", re.IGNORECASE),
    re.compile(r"\bfor backward(s)?(\s+compat)\b", re.IGNORECASE),
    re.compile(r"\blegacy mock\b", re.IGNORECASE),
    re.compile(r"\bso existing imports keep working\b", re.IGNORECASE),
)
ALLOW_HISTORY_TAG = "# style-check: allow-history"

# Module names that became packages after the tidy-module-organization
# restructure. Any reference to ``<name>.py`` in ``src/`` is a stale
# single-file path: docstrings and comments should name the package
# (``lilbee.catalog``, ``catalog/download.py``, etc.) instead.
STALE_SINGLE_FILE_RE = re.compile(
    r"\b(catalog|store|gen|ingest|commands|handlers|api|clustering_embedding"
    r"|worker_process)\.py\b"
)

# "the original X.py" / "original foo.py" phrasing is historical narrative
# pointing at a file that no longer exists in its single-file form.
ORIGINAL_FILE_RE = re.compile(r"\boriginal\s+[a-z_][a-z0-9_]*\.py\b", re.IGNORECASE)


# Names that always denote text file I/O, whatever they are called on:
# ``Path.read_text`` / ``Path.write_text`` take no mode, and the builtin
# ``open`` and ``NamedTemporaryFile`` are files by definition.
_ALWAYS_TEXT_CALLS = frozenset({"read_text", "write_text"})
_TEXT_IO_CALLS = _ALWAYS_TEXT_CALLS | {"open", "NamedTemporaryFile"}

# subprocess decodes its pipes with the locale's encoding too, whenever text
# mode is asked for. Same defect, different call: the output of a GPU tool or a
# probe script is decoded before any of this code sees it.
_SUBPROCESS_CALLS = frozenset({"run", "Popen", "check_output", "check_call", "call"})
_TEXT_MODE_KEYWORDS = frozenset({"text", "universal_newlines"})

# A ``.open`` attribute call is only a file open when it says so. ``os.open``
# returns a descriptor and ``webbrowser.open`` takes a URL; neither accepts an
# encoding, so reporting them would be a finding nobody can resolve. Require
# either a literal ``Path(...)`` receiver or a first argument that is a real
# mode string, which is what tells a file open from its homonyms.
_FILE_MODE_RE = re.compile(r"^[rwxa][bt+]*$")
# ``open``'s mode is its second positional argument; the rest are keyword-only.
_OPEN_MODE_POSITION = 1


def _call_name(node: ast.Call) -> str | None:
    """The bare function name of *node*, ignoring whatever it is called on."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _mode_argument(node: ast.Call, name: str) -> str | None:
    """The literal mode *node* opens with, or None when it names none."""
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    position = _OPEN_MODE_POSITION if name == "open" and isinstance(node.func, ast.Name) else 0
    if name == "open" and len(node.args) > position:
        first = node.args[position]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def _is_file_io(node: ast.Call, name: str) -> bool:
    """Whether *node* is text file I/O this rule governs.

    Everything but a bare ``.open`` attribute is unambiguous by name. For that
    one, a ``Path(...)`` receiver or a mode-shaped first argument separates a
    file open from ``os.open`` and ``webbrowser.open``.
    """
    if name != "open" or isinstance(node.func, ast.Name):
        return True
    receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
    if isinstance(receiver, ast.Call) and _call_name(receiver) == "Path":
        return True
    mode = _mode_argument(node, name)
    return mode is not None and bool(_FILE_MODE_RE.match(mode))


def _asks_for_text_pipes(node: ast.Call) -> bool:
    """Whether *node* puts subprocess into text mode, which decodes its pipes.

    Without a text-mode keyword the pipes stay bytes and there is nothing to
    decode; ``capture_output`` alone does not change that.
    """
    return any(
        keyword.arg in _TEXT_MODE_KEYWORDS
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in node.keywords
    )


def _opens_in_binary_mode(node: ast.Call, name: str) -> bool:
    """Whether *node* reads bytes, which have no encoding to declare.

    ``NamedTemporaryFile`` defaults to ``w+b``, so it is binary until a text
    mode says otherwise; ``open`` defaults to ``r``, so it is the reverse.
    """
    if name in _ALWAYS_TEXT_CALLS:
        return False
    mode = _mode_argument(node, name)
    if mode is None:
        return name == "NamedTemporaryFile"
    return "b" in mode


def _unspecified_encoding_hits(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line, call name)`` for text file I/O that names no encoding.

    Without ``encoding=`` Python decodes with the locale's, so a file holding
    anything outside cp1252 raises ``UnicodeDecodeError`` on Windows while
    passing everywhere else. Ruff expresses this as PLW1514, but only under
    preview mode, which turns on 900 unrelated findings across the tree.

    An unparsable file yields nothing rather than raising: this runs over many
    files, and a syntax error is already every other tool's finding.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in _TEXT_IO_CALLS and name not in _SUBPROCESS_CALLS:
            continue
        if any(keyword.arg == "encoding" for keyword in node.keywords):
            continue
        if name in _SUBPROCESS_CALLS:
            if _asks_for_text_pipes(node):
                yield node.lineno, name
            continue
        if not _is_file_io(node, name) or _opens_in_binary_mode(node, name):
            continue
        yield node.lineno, name


def _encoding_finding(path: Path | str, lineno: int, name: str) -> str:
    return (
        f"{path}:{lineno}: {name}() without encoding= decodes as the locale's "
        "(pass encoding='utf-8', or open in binary mode)"
    )


def _check_unspecified_encoding(paths: Iterable[Path]) -> Iterator[str]:
    """Yield findings for every text file I/O call in *paths* naming no encoding."""
    for path in paths:
        for lineno, name in _unspecified_encoding_hits(path):
            yield _encoding_finding(path, lineno, name)


def _check_new_unspecified_encoding(added: Iterable[tuple[str, int, str]]) -> Iterator[str]:
    """Yield findings only where this branch added the offending line.

    Whole-tree enforcement would mean touching 700-odd call sites, nearly all of
    them ``tmp_path`` writes in tests that cannot fail. Diff scoping stops the
    bug arriving without demanding that cleanup first, matching how the
    code-smell check above is scoped.
    """
    added_lines: dict[str, set[int]] = {}
    for rel_path, lineno, _text in added:
        added_lines.setdefault(rel_path, set()).add(lineno)
    for rel_path, linenos in sorted(added_lines.items()):
        for lineno, name in _unspecified_encoding_hits(REPO_ROOT / rel_path):
            if lineno in linenos:
                yield _encoding_finding(rel_path, lineno, name)


def _iter_python_files(*roots: Path) -> Iterator[Path]:
    """Yield every ``*.py`` file under each existing root."""
    for root in roots:
        if not root.exists():
            continue
        yield from sorted(root.rglob("*.py"))


def _check_em_dashes(paths: Iterable[Path]) -> Iterator[str]:
    """Yield ``path:line:reason`` for every em-dash hit outside excluded files."""
    for path in paths:
        if path.name in EM_DASH_EXCLUDED_FILENAMES:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if EM_DASH in line:
                yield f"{path}:{lineno}: em-dash forbidden (use a period or comma)"


def _check_divider_comments(paths: Iterable[Path]) -> Iterator[str]:
    """Yield findings for ``# ----`` / ``# ====`` divider comments in src/."""
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if DIVIDER_RE.match(line):
                yield (
                    f"{path}:{lineno}: divider comment forbidden "
                    "(group with modules or classes instead)"
                )


def _check_historical_narrative(paths: Iterable[Path]) -> Iterator[str]:
    """Yield findings for historical-narrative phrases in src/.

    Lines carrying ``# style-check: allow-history`` are skipped so docs that
    genuinely need the history can opt in explicitly.
    """
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if ALLOW_HISTORY_TAG in line:
                continue
            for pattern in HISTORICAL_PATTERNS:
                match = pattern.search(line)
                if match is not None:
                    yield (
                        f"{path}:{lineno}: historical-narrative phrase "
                        f"{match.group(0)!r} (rewrite to describe current "
                        f"behaviour, or annotate with `{ALLOW_HISTORY_TAG}`)"
                    )
                    break


def _check_stale_single_file_paths(paths: Iterable[Path]) -> Iterator[str]:
    """Yield findings for ``catalog.py`` / ``store.py`` / etc. references in src/.

    These names are now packages; docstrings and comments must name the
    current path (``lilbee.catalog``, ``catalog/download.py``).
    """
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stale = STALE_SINGLE_FILE_RE.search(line)
            if stale is not None:
                yield (
                    f"{path}:{lineno}: stale single-file path {stale.group(0)!r} "
                    f"(name the current package or sub-module instead)"
                )
                continue
            original = ORIGINAL_FILE_RE.search(line)
            if original is not None:
                yield (
                    f"{path}:{lineno}: historical-file phrase {original.group(0)!r} "
                    f"(name the current module without the 'original' qualifier)"
                )


ALLOW_SMELL_TAG = "# style-check: allow-smell"

# getattr-with-default on an object field. Named so the dunder exclusion below
# (a legitimate getattr on `__dunder__` attributes) can reference it directly.
_GETATTR_DEFAULT_RE = re.compile(r'getattr\([^,]+, "[^"]+",')

# Each entry mirrors one AGENTS.md "Code-Smell Triggers" grep. Patterns match
# the added line's content (the leading ``+`` already stripped).
SMELL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r'getattr\(self, "'),
        "getattr-by-name on an owned attribute (declare it in __init__ with a type)",
    ),
    (
        _GETATTR_DEFAULT_RE,
        "getattr-with-default on an object field (tighten the type, don't paper over it)",
    ),
    (
        re.compile(r"# type: ignore\[attr-defined\]"),
        "type-ignore on an owned attribute (declare the attribute on the class)",
    ),
    (
        re.compile(r"isinstance\(self\.app, LilbeeApp\)"),
        "production host-narrowing for tests (declare `app: LilbeeApp`, use LilbeeAppHost)",
    ),
    (
        re.compile(r"\b(?:task|kind|role|event_type|status|mode): str\b"),
        "string-typed closed set (convert to a StrEnum at the boundary)",
    ),
    (
        re.compile(r"^\s*global \w+"),
        "module-level mutable global (encapsulate on a class)",
    ),
)

_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$")
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _smell_base_ref() -> str | None:
    """Return the merge-base sha with the upstream default branch, or None.

    Tries ``origin/main`` then ``main``; returns None when neither resolves so
    the caller can skip the diff-scoped check instead of failing.
    """
    for branch in ("origin/main", "main"):
        try:
            out = subprocess.run(
                ["git", "merge-base", "HEAD", branch],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        ref = out.stdout.strip()
        if ref:
            return ref
    return None


def _git_diff_src(base: str) -> str:
    """Return ``git diff --unified=0`` of ``src/`` against the base sha."""
    return _git_diff(base, "src")


def _git_diff(base: str, *paths: str) -> str:
    """Return ``git diff --unified=0`` of *paths* against the base sha."""
    out = subprocess.run(
        ["git", "diff", "--unified=0", base, "--", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def _parse_added_lines(diff_text: str) -> Iterator[tuple[str, int, str]]:
    """Yield ``(path, new_line_number, content)`` for each added line.

    Parses ``git diff --unified=0`` output: ``+`` lines (excluding the ``+++``
    header) are additions, ``-`` lines never advance the new-file counter, and
    ``+++ /dev/null`` deletions are skipped.
    """
    path: str | None = None
    lineno = 0
    for line in diff_text.splitlines():
        file_match = _DIFF_FILE_RE.match(line)
        if file_match is not None:
            target = file_match.group(1)
            path = None if target == "/dev/null" else target
            continue
        hunk_match = _DIFF_HUNK_RE.match(line)
        if hunk_match is not None:
            lineno = int(hunk_match.group(1))
            continue
        if path is None or not line.startswith("+"):
            continue
        yield path, lineno, line[1:]
        lineno += 1


def _check_code_smells(added: Iterable[tuple[str, int, str]]) -> Iterator[str]:
    """Yield findings for AGENTS.md code-smell triggers on added Python lines."""
    for path, lineno, content in added:
        if not path.endswith(".py") or ALLOW_SMELL_TAG in content:
            continue
        for pattern, reason in SMELL_PATTERNS:
            match = pattern.search(content)
            if match is None:
                continue
            # getattr on a dunder attribute is legitimate dynamic reflection.
            if pattern is _GETATTR_DEFAULT_RE and '"__' in match.group(0):
                continue
            yield f"{path}:{lineno}: code smell -- {reason}"
            break


def main() -> int:
    src_files = list(_iter_python_files(SRC_DIR))
    test_files = list(_iter_python_files(TESTS_DIR))

    findings: list[str] = []
    findings.extend(_check_em_dashes(src_files + test_files))
    findings.extend(_check_divider_comments(src_files))
    findings.extend(_check_historical_narrative(src_files))
    findings.extend(_check_stale_single_file_paths(src_files))

    base = _smell_base_ref()
    if base is not None:
        findings.extend(_check_code_smells(_parse_added_lines(_git_diff_src(base))))
        findings.extend(
            _check_new_unspecified_encoding(_parse_added_lines(_git_diff(base, "src", "tests")))
        )

    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
