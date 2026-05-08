"""Wiki browse: shared page listing, reading, and resolution logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from lilbee.core.security import validate_path_within
from lilbee.wiki.grammar import CODE_FENCE_RE, H1_RE
from lilbee.wiki.index import parse_source_count
from lilbee.wiki.shared import (
    SUBDIR_TO_TYPE,
    WIKI_CONTENT_SUBDIRS,
    WikiSubdir,
    parse_frontmatter,
)

# Wiki paths under the root take the form ``<subdir>/<slug>.md``; only paths
# with at least this many components carry a meaningful page type.
_WIKI_PATH_MIN_PARTS = 2


@dataclass
class WikiPageInfo:
    """Summary metadata for a wiki page."""

    slug: str
    title: str
    page_type: str
    source_count: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON responses."""
        return {
            "slug": self.slug,
            "title": self.title,
            "page_type": self.page_type,
            "source_count": self.source_count,
            "created_at": self.created_at,
        }


@dataclass
class WikiPageContent:
    """Full content of a wiki page with parsed frontmatter."""

    slug: str
    title: str
    content: str
    frontmatter: dict[str, Any] = field(default_factory=dict)


def list_md_files(directory: Path) -> list[Path]:
    """Return sorted markdown files in a directory (non-recursive)."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.md"))


def _page_type_from_path(path: Path, wiki_root: Path) -> str:
    """Determine page type from its location relative to wiki root."""
    try:
        relative = path.relative_to(wiki_root)
    except ValueError:
        return "unknown"
    parts = relative.parts
    if len(parts) >= _WIKI_PATH_MIN_PARTS:
        return SUBDIR_TO_TYPE.get(parts[0], "unknown")
    return "unknown"


def _slug_from_path(path: Path, wiki_root: Path) -> str:
    """Build a URL slug from a wiki page path."""
    relative = path.relative_to(wiki_root)
    return str(relative.with_suffix("")).replace("\\", "/")


def _extract_h1_title(text: str) -> str | None:
    """Return the first top-level heading from markdown body, ignoring fenced code blocks."""
    in_fence = False
    for line in text.splitlines():
        if CODE_FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if m := H1_RE.match(line):
            return m.group(1).strip()
    return None


def _resolve_page_title(fm: dict[str, Any], text: str, path: Path) -> str:
    """Pick a page title. Frontmatter wins; body H1 beats slug-title-case fallback.

    Wiki generation does not emit a frontmatter title today, so without the H1
    step every page would render as the slug (e.g. 'Cv Manual' for cv-manual.md).
    """
    if (fm_title := fm.get("title")) is not None:
        return str(fm_title)
    if (h1 := _extract_h1_title(text)) is not None:
        return h1
    return path.stem.replace("-", " ").title()


def build_page_info(path: Path, wiki_root: Path) -> WikiPageInfo:
    """Build a WikiPageInfo from a markdown file on disk."""
    text = path.read_text(encoding="utf-8")
    fm = parse_frontmatter(text)
    slug = _slug_from_path(path, wiki_root)
    title = _resolve_page_title(fm, text, path)
    page_type = _page_type_from_path(path, wiki_root)
    source_count = parse_source_count(text)
    raw_at = fm.get("generated_at", "")
    # yaml.safe_load returns datetime/date objects for date-like strings
    created_at = raw_at.isoformat() if isinstance(raw_at, (datetime, date)) else str(raw_at)
    return WikiPageInfo(
        slug=slug,
        title=title,
        page_type=page_type,
        source_count=source_count,
        created_at=created_at,
    )


def find_page(wiki_root: Path, slug: str) -> Path | None:
    """Resolve a slug to a wiki page path, or None if not found.
    Validates the resolved path stays within wiki_root to prevent
    path traversal attacks.
    """
    candidate = wiki_root / f"{slug}.md"
    try:
        validate_path_within(candidate, wiki_root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _list_md_files_recursive(directory: Path) -> list[Path]:
    """Sorted markdown files under *directory* at any depth."""
    if not directory.is_dir():
        return []
    return sorted(directory.rglob("*.md"))


def list_pages(wiki_root: Path) -> list[WikiPageInfo]:
    """List all wiki pages under summaries/ and synthesis/ at any nesting depth."""
    pages: list[WikiPageInfo] = []
    for subdir in WIKI_CONTENT_SUBDIRS:
        for path in _list_md_files_recursive(wiki_root / subdir):
            pages.append(build_page_info(path, wiki_root))
    return pages


def list_draft_pages(wiki_root: Path) -> list[WikiPageInfo]:
    """List draft pages that failed the quality gate (recurses into per-source dirs)."""
    return [
        build_page_info(path, wiki_root)
        for path in _list_md_files_recursive(wiki_root / WikiSubdir.DRAFTS)
    ]


def read_page(wiki_root: Path, slug: str) -> WikiPageContent | None:
    """Read a wiki page's content and parsed frontmatter.
    Returns None if the page does not exist or the slug escapes wiki_root.
    """
    path = find_page(wiki_root, slug)
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    fm = parse_frontmatter(text)
    title = _resolve_page_title(fm, text, path)
    return WikiPageContent(slug=slug, title=title, content=text, frontmatter=fm)
