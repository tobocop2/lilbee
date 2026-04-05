"""Shared wiki utilities — frontmatter parsing, constants, slug generation."""

from __future__ import annotations

import re
from typing import Any

import yaml

SUBDIR_TO_TYPE: dict[str, str] = {
    "summaries": "summary",
    "concepts": "concept",
    "drafts": "draft",
    "archive": "archive",
}

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9-]")


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Extract YAML frontmatter fields from a wiki page string."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    block = text[3:end]
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
