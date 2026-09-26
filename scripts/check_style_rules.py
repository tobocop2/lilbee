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
6. Code in ``src/`` outside the TUI that lets Rich parse text as markup: a
   rich ``Console`` built anywhere but ``lilbee.runtime.console`` (whose
   ``PlainConsole`` has markup off), ``markup=`` with any value but ``False``,
   ``Text.from_markup``, rich's ``print``, ``get_console``, ``inspect``
   and ``progress.track``, a subclass of rich's ``Console``, star imports
   from rich, and the rich modules whose constructors parse a string as
   markup (markup, panel, prompt, spinner, status); a ``SpinnerColumn``
   built with ``finished_text=``; a ``TextColumn`` built from a non-constant
   format without ``markup=False``; and a ``Progress``/``Live`` built
   without ``console=`` whose bound result calls ``.print`` or ``.log``
   (both use rich's global console, which has markup on). Imports and
   attribute chains resolve to dotted names; a name rebound by plain
   assignment (``K = Console``) is not followed.
7. New occurrences in ``tests/`` of a patch or monkeypatch on ``get_services``,
   whether the target names the literal string
   ``lilbee.app.services.get_services`` or resolves through an import alias
   (``patch.object(svc_mod, "get_services", ...)`` /
   ``monkeypatch.setattr(svc_mod, "get_services", ...)`` where ``svc_mod`` is
   bound to ``lilbee.app.services``). A module first imported while that
   attribute is patched binds its own ``from ... import get_services`` to the
   stand-in and keeps it for the rest of the xdist worker (bb-25eyz). Use
   ``set_services()`` from ``lilbee.app.services`` instead, which the
   ``_reset_services_after_test`` conftest fixture clears after every test. A
   test that asserts ``get_services`` itself was (not) called needs the real
   patch; opt out with ``# style-check: allow-smell``.

Inline opt-out comments: ``# style-check: allow-history`` skips the
historical-narrative check on that line; ``# style-check: allow-smell`` skips
the code-smell check on that added line (also used by the ``get_services``
root-patch check).

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

# subprocess decodes its pipes with the locale's encoding in text mode.
_SUBPROCESS_CALLS = frozenset({"run", "Popen", "check_output", "check_call", "call"})
# Names that mean subprocess wherever they appear, so they need no receiver.
_DISTINCTIVE_SUBPROCESS_CALLS = frozenset({"Popen", "check_output", "check_call"})
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

    A bare ``.open`` needs a ``Path(...)`` receiver or a mode-shaped first
    argument; ``os.open`` and ``webbrowser.open`` accept no encoding.
    """
    if name != "open" or isinstance(node.func, ast.Name):
        return True
    receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
    if isinstance(receiver, ast.Call) and _call_name(receiver) == "Path":
        return True
    mode = _mode_argument(node, name)
    return mode is not None and bool(_FILE_MODE_RE.match(mode))


def _is_subprocess_call(node: ast.Call) -> bool:
    """Whether *node* is subprocess's own call, not a same-named method elsewhere.

    ``run`` and ``call`` are generic, so they need the receiver; ``Popen`` and
    ``check_output`` are not.
    """
    if isinstance(node.func, ast.Attribute):
        return isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
    return isinstance(node.func, ast.Name) and node.func.id in _DISTINCTIVE_SUBPROCESS_CALLS


def _asks_for_text_pipes(node: ast.Call) -> bool:
    """Whether *node* asks subprocess for text pipes; without it they stay bytes.

    Direct keywords only: a kwargs dict splatted with ``**`` hides the mode.
    """
    return any(
        keyword.arg in _TEXT_MODE_KEYWORDS
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in node.keywords
    )


def _opens_in_binary_mode(node: ast.Call, name: str) -> bool:
    """Whether *node* reads bytes, which have no encoding to declare.

    ``NamedTemporaryFile`` defaults to ``w+b`` and ``open`` to ``r``, so the
    absent-mode default is opposite between them.
    """
    if name in _ALWAYS_TEXT_CALLS:
        return False
    mode = _mode_argument(node, name)
    if mode is None:
        return name == "NamedTemporaryFile"
    return "b" in mode


def _unspecified_encoding_hits(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line, call name)`` for text I/O in *path* that names no encoding.

    An unparsable file yields nothing; a syntax error is already every other
    tool's finding.
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
            if _is_subprocess_call(node) and _asks_for_text_pipes(node):
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
    """Yield findings only where this branch added the offending line."""
    added_lines: dict[str, set[int]] = {}
    for rel_path, lineno, _text in added:
        added_lines.setdefault(rel_path, set()).add(lineno)
    for rel_path, linenos in sorted(added_lines.items()):
        for lineno, name in _unspecified_encoding_hits(REPO_ROOT / rel_path):
            if lineno in linenos:
                yield _encoding_finding(rel_path, lineno, name)


_GET_SERVICES_ROOT_PATCH_RE = re.compile(r"""["']lilbee\.app\.services\.get_services["']""")

SERVICES_MODULE_DOTTED = "lilbee.app.services"
GET_SERVICES_ATTR = "get_services"
# ``patch.object`` however ``patch`` got imported: bare (``from unittest.mock
# import patch``), off an imported ``mock``/``unittest.mock`` module, or a
# dotted ``unittest.mock.patch``.
_PATCH_OBJECT_RECEIVERS = frozenset({"unittest.mock.patch", "mock.patch"})
# The receiver and the attribute-name string are always the first two
# positional arguments, in ``patch.object``, ``monkeypatch.setattr`` and the
# builtin three-argument ``setattr`` alike.
_PATCH_TARGET_MIN_ARGS = 2


def _get_services_root_patch_finding(path: str, lineno: int) -> str:
    return (
        f"{path}:{lineno}: patches lilbee.app.services.get_services directly "
        "(use set_services() from lilbee.app.services instead, or "
        f"{ALLOW_SMELL_TAG} on a test that asserts the getter itself)"
    )


def _is_patch_object_call(node: ast.Call, bindings: dict[str, str]) -> bool:
    """Whether *node* calls ``patch.object(...)``, however ``patch`` was imported."""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "object"
        and _dotted_name(func.value, bindings) in _PATCH_OBJECT_RECEIVERS
    )


def _targets_get_services_alias(node: ast.Call, bindings: dict[str, str]) -> bool:
    """Whether *node* names ``get_services`` on a value bound to the services module.

    Covers ``patch.object(svc_mod, "get_services", ...)`` and
    ``monkeypatch.setattr(svc_mod, "get_services", ...)`` (and the builtin
    three-argument ``setattr``) for any import alias of
    ``lilbee.app.services``: the target's first two positional arguments are
    the receiver and the attribute name in every one of those call shapes.
    """
    if len(node.args) < _PATCH_TARGET_MIN_ARGS:
        return False
    attr_name = node.args[1]
    if not (isinstance(attr_name, ast.Constant) and attr_name.value == GET_SERVICES_ATTR):
        return False
    return _dotted_name(node.args[0], bindings) == SERVICES_MODULE_DOTTED


def _get_services_alias_hits(path: Path) -> Iterator[int]:
    """Yield line numbers of a patch/setattr on the services module's alias.

    ``patch.object`` and ``monkeypatch.setattr`` name their target by
    reference, not by the string the ``get_services`` root-patch regex
    matches, so this resolves the reference through the file's own imports
    with rule 6's dotted-name helpers (``_import_bindings`` / `_dotted_name`)
    instead of duplicating that resolution.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return
    bindings = _import_bindings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in ("object", "setattr"):
            continue
        if name == "object" and not _is_patch_object_call(node, bindings):
            continue
        if _targets_get_services_alias(node, bindings):
            yield node.lineno


def _literal_string_patch_findings(added_by_path: dict[str, dict[int, str]]) -> Iterator[str]:
    """Yield findings for the literal quoted ``get_services`` root-attribute string."""
    for path, lines in added_by_path.items():
        if not path.endswith(".py"):
            continue
        for lineno, content in lines.items():
            if ALLOW_SMELL_TAG in content:
                continue
            if _GET_SERVICES_ROOT_PATCH_RE.search(content) is not None:
                yield _get_services_root_patch_finding(path, lineno)


def _alias_patch_findings(added_by_path: dict[str, dict[int, str]]) -> Iterator[str]:
    """Yield findings for a patch/setattr on an import alias of the services module."""
    for path, lines in sorted(added_by_path.items()):
        if not path.endswith(".py"):
            continue
        for lineno in _get_services_alias_hits(REPO_ROOT / path):
            content = lines.get(lineno)
            if content is not None and ALLOW_SMELL_TAG not in content:
                yield _get_services_root_patch_finding(path, lineno)


def _check_get_services_root_patch(added: Iterable[tuple[str, int, str]]) -> Iterator[str]:
    """Yield findings for a new patch of get_services, by string or by alias.

    A module first imported while ``lilbee.app.services.get_services`` is
    patched binds its own ``from ... import get_services`` to the stand-in
    and keeps it for the rest of the xdist worker (bb-25eyz). ``set_services``
    replaces the singleton directly, so every caller's binding resolves
    through it instead.
    """
    added_by_path: dict[str, dict[int, str]] = {}
    for path, lineno, content in added:
        added_by_path.setdefault(path, {})[lineno] = content

    yield from _literal_string_patch_findings(added_by_path)
    yield from _alias_patch_findings(added_by_path)


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


MARKUP_EXEMPT_DIRS = (SRC_DIR / "lilbee" / "cli" / "tui",)
PLAIN_CONSOLE_MODULE = SRC_DIR / "lilbee" / "runtime" / "console.py"
MARKUP_PARSING_MODULES = frozenset(
    {"rich.markup", "rich.panel", "rich.prompt", "rich.spinner", "rich.status"}
)
MARKUP_PARSING_CALLABLES = frozenset(
    {
        "rich.console.Console",
        "rich.get_console",
        "rich.inspect",
        "rich.print",
        "rich.progress.track",
    }
)
SPINNER_COLUMN_DOTTED = "rich.progress.SpinnerColumn"
SPINNER_FINISHED_TEXT_KW = "finished_text"
TEXT_COLUMN_DOTTED = "rich.progress.TextColumn"
PROGRESS_LIVE_DOTTED = frozenset({"rich.progress.Progress", "rich.live.Live"})
CONSOLE_KW = "console"
MARKUP_KW = "markup"
_PRINT_OR_LOG_ATTRS = frozenset({"print", "log"})


def _is_markup_parser(dotted: str) -> bool:
    """True for a rich module or callable that parses strings as markup."""
    return dotted in MARKUP_PARSING_CALLABLES or any(
        dotted == module or dotted.startswith(f"{module}.") for module in MARKUP_PARSING_MODULES
    )


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    """Map each name an import binds to the dotted rich path it stands for."""
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bindings[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    bindings[top] = top
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bindings


def _dotted_name(expr: ast.expr, bindings: dict[str, str]) -> str | None:
    """The dotted import path *expr* refers to, or None when it is not an imported name."""
    if isinstance(expr, ast.Name):
        return bindings.get(expr.id)
    if isinstance(expr, ast.Attribute):
        base = _dotted_name(expr.value, bindings)
        return f"{base}.{expr.attr}" if base else None
    return None


def _markup_import_hits(node: ast.stmt) -> Iterator[tuple[int, str]]:
    """Yield a finding for an import of a rich markup parser or a rich star import."""
    if isinstance(node, ast.Import):
        targets = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
        if node.module.split(".")[0] == "rich" and any(a.name == "*" for a in node.names):
            yield node.lineno, f"star-imports from {node.module}"
            return
        targets = [f"{node.module}.{alias.name}" for alias in node.names]
    else:
        return
    for target in targets:
        if _is_markup_parser(target) and target != "rich.console.Console":
            yield node.lineno, f"imports {target}, which parses strings as markup"


def _spinner_column_hit(node: ast.Call, bindings: dict[str, str]) -> Iterator[tuple[int, str]]:
    """Yield a finding for a SpinnerColumn built with finished_text=, which renders as markup."""
    if _dotted_name(node.func, bindings) != SPINNER_COLUMN_DOTTED:
        return
    if any(kw.arg == SPINNER_FINISHED_TEXT_KW for kw in node.keywords):
        yield node.lineno, "builds SpinnerColumn with finished_text=, which parses as markup"


def _text_column_hit(node: ast.Call, bindings: dict[str, str]) -> Iterator[tuple[int, str]]:
    """Yield a finding for a TextColumn built from a non-constant format without markup=False.

    A constant format string is the codebase's own template text, already
    reviewed; a computed one may embed unreviewed content directly into what
    TextColumn renders as markup by default.
    """
    if _dotted_name(node.func, bindings) != TEXT_COLUMN_DOTTED or not node.args:
        return
    fmt = node.args[0]
    if isinstance(fmt, ast.Constant) and isinstance(fmt.value, str):
        return
    markup_off = any(
        kw.arg == MARKUP_KW and isinstance(kw.value, ast.Constant) and kw.value.value is False
        for kw in node.keywords
    )
    if not markup_off:
        yield node.lineno, "builds TextColumn from a non-constant format without markup=False"


def _print_or_log_receiver(value: ast.expr) -> str | None:
    """The bound name behind a ``.print``/``.log`` receiver: itself, or its ``.console``.

    ``Progress``/``Live`` without ``console=`` fall back to ``rich.get_console()``,
    so ``p.console`` names that same global console and ``p.console.print(...)``
    parses markup exactly like ``p.print(...)`` does.
    """
    if isinstance(value, ast.Name):
        return value.id
    if (
        isinstance(value, ast.Attribute)
        and value.attr == CONSOLE_KW
        and isinstance(value.value, ast.Name)
    ):
        return value.value.id
    return None


def _print_or_log_receiver_names(tree: ast.Module) -> frozenset[str]:
    """Names that ``.print(``/``.log(`` (directly, or via ``.console``) is called on in *tree*.

    File-scoped, and coarse the same way the module's own name-rebinding
    blind spot is: it does not confirm a given Progress/Live binding is the
    one that gets printed through, only that some name spelled the same way
    does somewhere in the file.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _PRINT_OR_LOG_ATTRS
        ):
            name = _print_or_log_receiver(node.func.value)
            if name is not None:
                names.add(name)
    return frozenset(names)


def _assign_target_names(target: ast.expr) -> Iterator[str]:
    """Every plain name *target* binds, recursing into tuple/list unpacking."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _assign_target_names(elt)


def _named_calls(value: ast.expr, target: ast.expr) -> Iterator[tuple[ast.Call, str | None]]:
    """Pair each call *value* holds with the name *target* binds it to.

    Walks a tuple/list target against a tuple/list value in lockstep, so
    ``p, x = Progress(), 1`` pairs the call with ``p`` and not with the whole
    right-hand side.
    """
    if isinstance(value, ast.Call):
        yield value, next(_assign_target_names(target), None)
    elif isinstance(value, (ast.Tuple, ast.List)) and isinstance(target, (ast.Tuple, ast.List)):
        for sub_value, sub_target in zip(value.elts, target.elts, strict=False):
            yield from _named_calls(sub_value, sub_target)


def _progress_or_live_bindings(stmt: ast.stmt) -> Iterator[tuple[ast.Call, str | None]]:
    """Every (Progress/Live call, bound name) pair one statement produces.

    Covers a plain, annotated, or multi-target assign, tuple/list unpacking
    where the call is one element, and ``with ... as name:``. A target this
    cannot name (an attribute, a starred element) still yields the call with
    ``name=None``, so a console-less construction is seen even when nothing
    downstream can prove it gets printed through.
    """
    if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        yield from _named_calls(stmt.value, stmt.target)
    elif isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            yield from _named_calls(stmt.value, target)
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            if isinstance(item.context_expr, ast.Call):
                as_name = item.optional_vars
                name = next(_assign_target_names(as_name), None) if as_name is not None else None
                yield item.context_expr, name


def _progress_live_hit(
    call: ast.Call,
    name: str | None,
    bindings: dict[str, str],
    print_or_log_names: frozenset[str],
) -> Iterator[tuple[int, str]]:
    """Yield a finding for a console-less Progress/Live whose binding is printed through."""
    target = _dotted_name(call.func, bindings)
    if target not in PROGRESS_LIVE_DOTTED:
        return
    if any(kw.arg == CONSOLE_KW for kw in call.keywords):
        return
    if name is not None and name in print_or_log_names:
        yield (
            call.lineno,
            f"builds {target} without console= and calls .print/.log on it",
        )


def _class_def_hit(node: ast.ClassDef, bindings: dict[str, str]) -> Iterator[tuple[int, str]]:
    """Yield a finding for a class that subclasses rich's Console."""
    if any(_dotted_name(base, bindings) == "rich.console.Console" for base in node.bases):
        yield node.lineno, "subclasses rich.console.Console, which keeps markup on"


def _call_hits(node: ast.Call, bindings: dict[str, str]) -> Iterator[tuple[int, str]]:
    """Yield every finding a single call expression produces on its own terms."""
    target = _dotted_name(node.func, bindings)
    if target is not None and _is_markup_parser(target):
        yield node.lineno, f"calls {target}, which parses strings as markup"
    if isinstance(node.func, ast.Attribute) and node.func.attr == "from_markup":
        yield node.lineno, "parses a string as markup with Text.from_markup"
    for kw in node.keywords:
        markup_off = isinstance(kw.value, ast.Constant) and kw.value.value is False
        if kw.arg == MARKUP_KW and not markup_off:
            yield node.lineno, "passes markup= a value other than False"
    yield from _spinner_column_hit(node, bindings)
    yield from _text_column_hit(node, bindings)


def _node_hits(
    node: ast.AST, bindings: dict[str, str], print_or_log_names: frozenset[str]
) -> Iterator[tuple[int, str]]:
    """Yield every finding one AST node produces, dispatched by node type."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield from _markup_import_hits(node)
    if isinstance(node, ast.ClassDef):
        yield from _class_def_hit(node, bindings)
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.With, ast.AsyncWith)):
        for call, name in _progress_or_live_bindings(node):
            yield from _progress_live_hit(call, name, bindings, print_or_log_names)
    if isinstance(node, ast.Call):
        yield from _call_hits(node, bindings)


def _markup_parser_hits(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line, reason)`` for each way *path* lets Rich parse text as markup."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return
    bindings = _import_bindings(tree)
    print_or_log_names = _print_or_log_receiver_names(tree)
    for node in ast.walk(tree):
        yield from _node_hits(node, bindings, print_or_log_names)


def _check_markup_parsers(paths: Iterable[Path]) -> Iterator[str]:
    """Yield findings for code outside the TUI that lets Rich parse text as markup."""
    for path in paths:
        if path == PLAIN_CONSOLE_MODULE or any(path.is_relative_to(d) for d in MARKUP_EXEMPT_DIRS):
            continue
        for lineno, reason in _markup_parser_hits(path):
            yield (
                f"{path}:{lineno}: {reason} (print through lilbee.runtime.console.PlainConsole "
                "and style with Text, styled() or style=)"
            )


def main() -> int:
    src_files = list(_iter_python_files(SRC_DIR))
    test_files = list(_iter_python_files(TESTS_DIR))

    findings: list[str] = []
    findings.extend(_check_em_dashes(src_files + test_files))
    findings.extend(_check_divider_comments(src_files))
    findings.extend(_check_historical_narrative(src_files))
    findings.extend(_check_stale_single_file_paths(src_files))
    findings.extend(_check_markup_parsers(src_files))

    base = _smell_base_ref()
    if base is not None:
        findings.extend(_check_code_smells(_parse_added_lines(_git_diff_src(base))))
        findings.extend(
            _check_new_unspecified_encoding(_parse_added_lines(_git_diff(base, "src", "tests")))
        )
        findings.extend(
            _check_get_services_root_patch(_parse_added_lines(_git_diff(base, "tests")))
        )

    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
