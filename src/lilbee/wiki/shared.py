"""Shared wiki utilities: frontmatter parsing, constants, page targets."""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

MIN_CLUSTER_SOURCES = 3  # minimum unique sources for a synthesis page

# Held by every mutating wiki entry point (build, synthesize, prune, draft
# accept, lint's log append, post-ingest update) while it writes. Pages,
# index.md and log.md are shared files that CLI, TUI, MCP and HTTP all reach
# inside one process, so the serialization lives with the writers rather than
# on any one surface. Re-entrant: a writer holding it calls helpers that take
# it again (a prune that lints, a lint that records a log entry).
WIKI_BUILD_LOCK = threading.RLock()


class WikiSubdir(StrEnum):
    """Filesystem subdirectory under ``$data_root/$wiki_dir/``."""

    SUMMARIES = "summaries"
    SYNTHESIS = "synthesis"
    CONCEPTS = "concepts"
    ENTITIES = "entities"
    DRAFTS = "drafts"
    ARCHIVE = "archive"


class WikiPageType(StrEnum):
    """Kind of wiki page. Values are used as frontmatter/API labels."""

    SUMMARY = "summary"
    SYNTHESIS = "synthesis"
    CONCEPT = "concept"
    ENTITY = "entity"
    DRAFT = "draft"
    ARCHIVE = "archive"


WIKI_CONTENT_SUBDIRS: tuple[WikiSubdir, ...] = (
    WikiSubdir.SUMMARIES,
    WikiSubdir.SYNTHESIS,
    WikiSubdir.CONCEPTS,
    WikiSubdir.ENTITIES,
)


def count_pages_in(wiki_root: Path, subdirs: Sequence[WikiSubdir]) -> int:
    """Count ``.md`` pages under *subdirs* of *wiki_root*, at any depth."""
    total = 0
    for subdir in subdirs:
        directory = wiki_root / subdir
        if directory.is_dir():
            total += sum(1 for _ in directory.rglob("*.md"))
    return total


def total_wiki_pages(wiki_root: Path) -> int:
    """Count published ``.md`` pages across every wiki content subdir.

    ``wiki build`` writes concepts/entities/synthesis pages, while summaries come
    from draft-accept, so counting only summaries (+ drafts) reports zero pages
    after a normal build even though searchable pages exist.
    """
    return count_pages_in(wiki_root, WIKI_CONTENT_SUBDIRS)


WIKI_DISABLED_ERROR = "wiki not enabled"

# Generic, path-free error for a draft slug that fails traversal validation.
# Shared across every transport (REST/CLI/MCP/TUI) so the absolute candidate
# path from validate_path_within is never echoed to a caller.
INVALID_DRAFT_SLUG_ERROR = "invalid draft slug"

# PENDING-marker keyword phrases written into ``drafts/<slug>.md`` by the
# batched generator and matched by the drafts-review surface. Centralized
# here so the gen-side writer and the drafts-side reader agree on the
# exact wording. Changing a keyword here requires updating any cached
# markers on disk (one-shot find -delete or a regen).
PENDING_MARKER_KEYWORD_PARSE = "PENDING: batch parse failed"
PENDING_MARKER_KEYWORD_COLLISION = "PENDING: concept slug collision"


class PendingKind(StrEnum):
    """Reason a wiki draft is in ``drafts/`` instead of a published page.

    Derived from a draft's leading marker line and surfaced through
    ``DraftInfo.pending_kind`` to CLI / HTTP / MCP callers. StrEnum members
    serialise as their string value, so the JSON payload stays a plain
    string. ``DRIFT`` is display-only, never written to disk, but exposed so
    consumers don't hard-code ``"drift"``.
    """

    PARSE = "parse"
    COLLISION = "collision"
    DRIFT = "drift"


class WikiLogAction(StrEnum):
    """Verbs written into ``wiki/log.md`` audit-trail entries.

    Distinct from WIKI_STATUS_* (which are result statuses returned to
    CLI/MCP/HTTP callers); these label internal audit-trail rows.
    """

    GENERATED = "generated"
    BUILD = "build"
    SYNTHESIZE = "synthesize"
    INGEST = "ingest"
    LINT = "lint"
    PRUNE = "prune"


SUBDIR_TO_TYPE: dict[str, WikiPageType] = {
    WikiSubdir.SUMMARIES.value: WikiPageType.SUMMARY,
    WikiSubdir.SYNTHESIS.value: WikiPageType.SYNTHESIS,
    WikiSubdir.CONCEPTS.value: WikiPageType.CONCEPT,
    WikiSubdir.ENTITIES.value: WikiPageType.ENTITY,
    WikiSubdir.DRAFTS.value: WikiPageType.DRAFT,
    WikiSubdir.ARCHIVE.value: WikiPageType.ARCHIVE,
}

# One source of truth for sidebar-style headings keyed by page type.
# Consumed by ``wiki/index.py`` and the TUI sidebar via
# ``cli/tui/messages.WIKI_TYPE_HEADINGS``.
WIKI_TYPE_HEADINGS: dict[WikiPageType, str] = {
    WikiPageType.CONCEPT: "Concepts",
    WikiPageType.ENTITY: "Entities",
    WikiPageType.SUMMARY: "Source Summaries",
    WikiPageType.SYNTHESIS: "Synthesis",
}


@dataclass(frozen=True)
class PageTarget:
    """Grouping of page location fields for wiki generation."""

    wiki_root: Path
    subdir: str
    slug: str
    wiki_source: str
    page_type: str
    label: str


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* via a temp file and ``os.replace``, creating parents.

    A crash mid-write leaves the previous page intact rather than a truncated
    one. ``mkstemp`` creates the temp file owner-only and ``os.replace`` keeps
    that mode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Extract YAML frontmatter fields from a wiki page string.

    A draft carries its marker comments above the frontmatter (drift,
    collision, origin), so the leading comment run is skipped before the
    opening delimiter is looked for. Without that every marked draft parses
    as having no frontmatter at all. Uses line-by-line scanning so ``---``
    inside YAML content is not mistaken for the closing delimiter.
    """
    lines = text.splitlines()
    start = 0
    while start < len(lines) and lines[start].lstrip().startswith("<!--"):
        start += 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines) or lines[start].strip() != "---":
        return {}
    end_idx: int | None = None
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}
    block = "\n".join(lines[start + 1 : end_idx])
    try:
        return yaml.safe_load(block) or {}
    except yaml.YAMLError:
        return {}
