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

# PENDING-marker keyword phrases written into ``drafts/<slug>.md`` by the
# batched generator and matched by the drafts-review surface. Centralized
# here so the gen-side writer and the drafts-side reader agree on the
# exact wording. Changing a keyword here requires updating any cached
# markers on disk (one-shot find -delete or a regen).
PENDING_MARKER_KEYWORD_PARSE = "PENDING: batch parse failed"
PENDING_MARKER_KEYWORD_COLLISION = "PENDING: concept slug collision"

# Values written into the ``pending_kind`` frontmatter field and
# surfaced verbatim through ``DraftInfo.pending_kind`` to CLI / HTTP /
# MCP callers. Kept as plain string constants (not an enum) because the
# value round-trips through YAML and JSON without translation.
PENDING_KIND_PARSE = "parse"
PENDING_KIND_COLLISION = "collision"
# Display-only default shown to users when a draft has no PENDING marker
# (i.e. a regular drift draft). Never written into
# ``DraftInfo.pending_kind`` on disk; consumers fall back to this
# constant instead of hard-coding ``"drift"``.
PENDING_KIND_DRIFT = "drift"

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

# Characters that signal markdown-structural noise in a concept label.
# Single source of truth for both ``is_valid_label`` (membership check)
# and ``clean_label_for_display`` (regex strip).
_STRUCTURAL_CHARS = frozenset("|#>")
_DISPLAY_STRUCTURAL_RE = re.compile(f"[{re.escape(''.join(_STRUCTURAL_CHARS))}]+")
_DISPLAY_WHITESPACE_RE = re.compile(r"\s+")

LABEL_SANITY_MIN_LEN = 3
LABEL_SANITY_MIN_ALNUM_RATIO = 0.5


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

    Catches the noise patterns observed in QA (bb-8b7s):

    - empty or sub-three-char fragments,
    - markdown table delimiters (``| | designer``),
    - page-number-prefixed tokens (``158 vehicle``),
    - paren-prefixed numerics (``(7.0 l)`` — would otherwise slug to
      ``70-l`` after punctuation cleanup),
    - hyphen-prefixed fragments (``-answers`` — trailing text from
      markdown bracket-link extraction).

    Requires the first non-whitespace character to be a Unicode letter
    so any non-alpha prefix (digit, bracket, hyphen, punctuation) is
    rejected up front. Legitimate labels like ``E-mail`` or ``iPhone``
    pass. Still permissive on three-char fragments like ``cro`` /
    ``fus``; A3's entity-type filter and ``wiki_entity_min_mentions``
    catch those downstream.
    """
    stripped = label.strip()
    if len(stripped) < LABEL_SANITY_MIN_LEN:
        return False
    if not stripped[0].isalpha():
        return False
    if any(ch in _STRUCTURAL_CHARS for ch in stripped):
        return False
    alnum = sum(1 for ch in stripped if ch.isalnum())
    return alnum / len(stripped) >= LABEL_SANITY_MIN_ALNUM_RATIO


def clean_label_for_display(label: str) -> str:
    """Return a prompt-safe version of *label* for the ``{topic}`` slot.

    Defense-in-depth behind :func:`is_valid_label`: a concept or entity
    label that reached this function already passed the sanity gate
    and should not contain ``|#>`` in practice. The structural-char
    strip here guards against a future code path that bypasses the
    gate (synthesis cluster labels sourced from ``concept_nodes``,
    user-supplied topics, tests). The always-useful work is whitespace
    normalization: spaCy surface forms can carry internal runs of
    whitespace that would reach the H1 verbatim.

    Preserves the original capitalization so proper nouns
    (``Chevrolet Caprice``, ``iPhone``) survive intact; the model
    title-cases lowercase common nouns on its own.
    """
    clean = _DISPLAY_STRUCTURAL_RE.sub("", label)
    return _DISPLAY_WHITESPACE_RE.sub(" ", clean).strip()
