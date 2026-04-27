"""Project style-rule checker invoked by ``make lint``.

Catches four classes of drift that ruff cannot express:

1. Em dashes (``—``) in ``src/`` or ``tests/``. Project rule from AGENTS.md.
2. Divider comments (``# ----`` / ``# ====``) used to group code in ``src/``.
   Project rule: prefer modules/classes for grouping.
3. Historical-narrative docstrings in ``src/`` (``previously``, ``used to``,
   ``migrated from``, ``preserves the historical``, ``for backward``). Project
   rule: docstrings describe what the code IS, not what it was.
4. Stale single-file path references in ``src/`` for modules that have since
   become packages (``catalog.py``, ``store.py``, ``gen.py``, ``ingest.py``,
   ``commands.py``, ``handlers.py``, ``api.py``, ``clustering_embedding.py``,
   ``worker_process.py``, ``llama_cpp_provider.py``), and the phrase
   ``original X.py``. Project rule: docstrings name the current path.

Lines tagged with the inline opt-out comment ``# style-check: allow-history``
are skipped for the historical-narrative check (only).

Exits 0 when clean, 1 with one ``path:line:reason`` per finding when violations
are found.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"

# featured_models.toml carries em dashes in legitimate model display names.
EM_DASH_EXCLUDED_FILENAMES = {"featured_models.toml"}

EM_DASH = "—"
DIVIDER_RE = re.compile(r"^\s*#\s*[-=]{4,}\s*$")

# Patterns that flag a comment or docstring as historical narrative (not a
# description of current behaviour). Each is a regex matched case-insensitively
# against the line. The patterns target sentence-shaped historical phrasings
# rather than the bare word, so descriptive uses like "model used to embed
# chunks" or "previously-written file" do not trip the check.
HISTORICAL_PATTERNS = (
    re.compile(r"\bused to\s+(be|live|return|raise|run|exist|wrap|hold)\b", re.IGNORECASE),
    re.compile(r"\bpreviously\s*[,;]", re.IGNORECASE),
    re.compile(r"\bpreviously\s+(this|the|we|it|they|all|each)\b", re.IGNORECASE),
    re.compile(r"\bmigrated from\b", re.IGNORECASE),
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
    r"|worker_process|llama_cpp_provider)\.py\b"
)

# "the original X.py" / "original foo.py" phrasing is historical narrative
# pointing at a file that no longer exists in its single-file form.
ORIGINAL_FILE_RE = re.compile(r"\boriginal\s+[a-z_][a-z0-9_]*\.py\b", re.IGNORECASE)


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


def main() -> int:
    src_files = list(_iter_python_files(SRC_DIR))
    test_files = list(_iter_python_files(TESTS_DIR))

    findings: list[str] = []
    findings.extend(_check_em_dashes(src_files + test_files))
    findings.extend(_check_divider_comments(src_files))
    findings.extend(_check_historical_narrative(src_files))
    findings.extend(_check_stale_single_file_paths(src_files))

    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
