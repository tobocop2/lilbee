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
_DISPLAY_STRUCTURAL_RE = re.compile(r"[|#>]+")
_DISPLAY_WHITESPACE_RE = re.compile(r"\s+")

LABEL_SANITY_MIN_LEN = 3
LABEL_SANITY_MIN_ALNUM_RATIO = 0.5
_STRUCTURAL_CHARS = frozenset("|#>")


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

    Lowercases, maps whitespace to single hyphens and slashes to double
    hyphens (path encoding), strips anything outside ``[a-z0-9-]``, and
    trims leading and trailing hyphens. Returns ``""`` when no sluggable
    characters remain; callers must treat an empty slug as "skip this
    entity" so the generator never writes a file called ``.md``.

    Internal hyphen runs from the ``/`` path encoding are preserved;
    only leading and trailing hyphens (e.g. ``--body`` from a stripped
    ``| | Body``) are removed.
    """
    slug = label.lower().replace(" ", "-").replace("/", "--")
    slug = _SLUG_CLEAN_RE.sub("", slug)
    return slug.strip("-")


def is_valid_label(label: str) -> bool:
    """Reject structural-noise labels before aggregation.

    Catches the noise patterns observed in QA: empty/very-short
    fragments (``cro``-class length gates), markdown table delimiters
    (``| | designer``), page-number ordinals (``158 vehicle``), and
    punctuation-heavy tokens that slipped past NER. Left intentionally
    coarse: the alnum-ratio gate plus the structural-character gate
    catch ~90% of the QA noise without rejecting legitimate labels
    like ``E-mail`` or ``C++``.
    """
    stripped = label.strip()
    if len(stripped) < LABEL_SANITY_MIN_LEN:
        return False
    if stripped[0].isdigit():
        return False
    if any(ch in _STRUCTURAL_CHARS for ch in stripped):
        return False
    alnum = sum(1 for ch in stripped if ch.isalnum())
    return alnum / len(stripped) >= LABEL_SANITY_MIN_ALNUM_RATIO


def clean_label_for_display(label: str) -> str:
    """Return a prompt-safe version of *label* for the ``{topic}`` slot.

    Removes markdown-structural characters and collapses internal
    whitespace so the LLM never sees ``| | designer`` and echoes it
    into the generated H1. Preserves the original capitalization so
    proper nouns (``Chevrolet Caprice``, ``iPhone``) survive intact;
    the model title-cases lowercase common nouns on its own.
    """
    clean = _DISPLAY_STRUCTURAL_RE.sub("", label)
    return _DISPLAY_WHITESPACE_RE.sub(" ", clean).strip()
