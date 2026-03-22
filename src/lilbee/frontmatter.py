"""Extract YAML frontmatter and inline hashtags from markdown files.

Parsing approach inspired by gno (gmickel/gno) frontmatter.ts which
supports YAML frontmatter, Logseq properties, and inline hashtags
with code-block-aware extraction and NFC tag normalization.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import yaml

_FRONTMATTER_RE = re.compile(r"^---\r?\n([\s\S]*?)(?:\r?\n)?---(?:\r?\n|$)")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
_HASHTAG_RE = re.compile(r"(?<!\S)#([a-zA-Z][\w/-]*)")


@dataclass(frozen=True)
class FrontmatterResult:
    """Parsed frontmatter metadata."""

    title: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    author: str = ""
    date: str = ""
    body: str = ""


def _normalize_tag(tag: str) -> str:
    """Normalize a tag: NFC unicode, lowercase, strip whitespace."""
    return unicodedata.normalize("NFC", tag.strip().lower())


def _extract_tags(raw: object) -> list[str]:
    """Extract tags from various YAML formats (list, string, comma-separated)."""
    if isinstance(raw, list):
        return [_normalize_tag(str(t)) for t in raw if str(t).strip()]
    if isinstance(raw, str):
        return [_normalize_tag(t) for t in raw.split(",") if t.strip()]
    return []


def _extract_hashtags(text: str) -> list[str]:
    """Extract inline #hashtags from body text, skipping code blocks."""
    cleaned = _CODE_BLOCK_RE.sub("", text)
    return [_normalize_tag(m.group(1)) for m in _HASHTAG_RE.finditer(cleaned)]


def parse_frontmatter(text: str) -> FrontmatterResult:
    """Parse YAML frontmatter and inline hashtags from markdown text.

    Returns metadata and the body text with frontmatter stripped.
    """

    body = text
    title = ""
    tags: list[str] = []
    author = ""
    date = ""

    match = _FRONTMATTER_RE.match(text)
    if match:
        raw_yaml = match.group(1)
        body = text[match.end() :]
        try:
            data = yaml.safe_load(raw_yaml)
            if isinstance(data, dict):
                title = str(data.get("title", ""))
                tags = _extract_tags(data.get("tags"))
                author = str(data.get("author", ""))
                date_val = data.get("date", "")
                date = str(date_val) if date_val else ""
        except yaml.YAMLError:
            pass

    inline_tags = _extract_hashtags(body)
    all_tags = list(dict.fromkeys(tags + inline_tags))

    return FrontmatterResult(
        title=title,
        tags=tuple(all_tags),
        author=author,
        date=date,
        body=body,
    )
