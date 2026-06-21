"""Shared wiki utilities: frontmatter parsing, constants, page targets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

MIN_CLUSTER_SOURCES = 3  # minimum unique sources for a synthesis page


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


def total_wiki_pages(wiki_root: Path) -> int:
    """Count published ``.md`` pages across every wiki content subdir.

    ``wiki build`` writes concepts/entities/synthesis pages, while summaries come
    from draft-accept, so counting only summaries (+ drafts) reports zero pages
    after a normal build even though searchable pages exist.
    """
    total = 0
    for subdir in WIKI_CONTENT_SUBDIRS:
        directory = wiki_root / subdir
        if directory.exists():
            total += sum(1 for _ in directory.rglob("*.md"))
    return total


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

    The string value is what lands in the ``pending_kind`` YAML
    frontmatter field and is surfaced verbatim through
    ``DraftInfo.pending_kind`` to CLI / HTTP / MCP callers.
    StrEnum members serialise as their string value, so the YAML/JSON
    round-trip stays a plain string. ``DRIFT`` is display-only, never
    written to disk, but exposed so consumers don't hard-code ``"drift"``.
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
    INGEST = "ingest"
    LINT = "lint"


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


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Extract YAML frontmatter fields from a wiki page string.
    Uses line-by-line scanning so ``---`` inside YAML content is not
    mistaken for the closing delimiter.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}
    block = "\n".join(lines[1:end_idx])
    try:
        return yaml.safe_load(block) or {}
    except yaml.YAMLError:
        return {}
