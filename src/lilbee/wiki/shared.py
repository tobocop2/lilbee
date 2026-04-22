"""Shared wiki utilities — frontmatter parsing, constants, slug generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

MIN_CLUSTER_SOURCES = 3  # minimum unique sources for a synthesis page

SUMMARIES_SUBDIR = "summaries"
SYNTHESIS_SUBDIR = "synthesis"
CONCEPTS_SUBDIR = "concepts"
ENTITIES_SUBDIR = "entities"
DRAFTS_SUBDIR = "drafts"
ARCHIVE_SUBDIR = "archive"


class WikiPageType(StrEnum):
    """Kind of wiki page. Values are used as frontmatter/API labels."""

    SUMMARY = "summary"
    SYNTHESIS = "synthesis"
    CONCEPT = "concept"
    ENTITY = "entity"
    DRAFT = "draft"
    ARCHIVE = "archive"


WIKI_CONTENT_SUBDIRS: tuple[str, ...] = (
    SUMMARIES_SUBDIR,
    SYNTHESIS_SUBDIR,
    CONCEPTS_SUBDIR,
    ENTITIES_SUBDIR,
)

WIKI_DISABLED_ERROR = "wiki not enabled"

# wiki/log.md action labels. Distinct from WIKI_STATUS_* (which are result
# statuses returned to CLI/MCP/HTTP callers); these are internal audit trail
# verbs written into the log file.
WIKI_LOG_ACTION_GENERATED = "generated"
WIKI_LOG_ACTION_BUILD = "build"
WIKI_LOG_ACTION_INGEST = "ingest"
WIKI_LOG_ACTION_LINT = "lint"

# Auto-generated citation block markers, shared across the wiki layer.
# The writers in ``wiki.citation`` emit these exact strings; the
# ``wiki.links`` rewriter treats anything after the comment as citation
# content and leaves it untouched.
CITATION_BLOCK_SEP = "---"
CITATION_BLOCK_COMMENT = "<!-- citations (auto-generated from _citations table -- do not edit) -->"

SUBDIR_TO_TYPE: dict[str, WikiPageType] = {
    SUMMARIES_SUBDIR: WikiPageType.SUMMARY,
    SYNTHESIS_SUBDIR: WikiPageType.SYNTHESIS,
    CONCEPTS_SUBDIR: WikiPageType.CONCEPT,
    ENTITIES_SUBDIR: WikiPageType.ENTITY,
    DRAFTS_SUBDIR: WikiPageType.DRAFT,
    ARCHIVE_SUBDIR: WikiPageType.ARCHIVE,
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

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9-]")


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


def make_slug(label: str) -> str:
    """Turn a concept label into a filesystem-safe slug.
    Lowercases, replaces spaces with hyphens, slashes with double-hyphens,
    and strips non-alphanumeric characters (except hyphens).
    """
    slug = label.lower().replace(" ", "-").replace("/", "--")
    return _SLUG_CLEAN_RE.sub("", slug)
