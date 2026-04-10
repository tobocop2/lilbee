"""Shared wiki utilities — frontmatter parsing, constants, slug generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MIN_CLUSTER_SOURCES = 3  # minimum unique sources for a synthesis page

SUBDIR_TO_TYPE: dict[str, str] = {
    "summaries": "summary",
    "concepts": "concept",
    "drafts": "draft",
    "archive": "archive",
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
