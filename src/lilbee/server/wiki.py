"""Wiki layer route handlers — page listing, reading, citations, lint, generation, pruning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from litestar import get, post
from litestar.exceptions import NotFoundException
from litestar.params import Parameter

from lilbee.config import cfg
from lilbee.server.models import WikiPageSummary


def _wiki_root() -> Path:
    """Resolve the wiki directory under data_root."""
    return cfg.data_root / cfg.wiki_dir


def _require_wiki() -> None:
    """Raise 404 if the wiki feature is disabled."""
    if not cfg.wiki:
        raise NotFoundException(detail="wiki not enabled")


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Extract YAML frontmatter fields from a wiki page string."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip()
    result: dict[str, Any] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip()] = value.strip()
    return result


def _list_md_files(directory: Path) -> list[Path]:
    """Return sorted markdown files in a directory (non-recursive)."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.md"))


_SUBDIR_TO_TYPE: dict[str, str] = {
    "summaries": "summary",
    "concepts": "concept",
    "drafts": "draft",
    "archive": "archive",
}


def _page_type_from_path(path: Path, wiki_root: Path) -> str:
    """Determine page type from its location relative to wiki root."""
    try:
        relative = path.relative_to(wiki_root)
    except ValueError:
        return "unknown"
    parts = relative.parts
    if len(parts) >= 2:
        return _SUBDIR_TO_TYPE.get(parts[0], "unknown")
    return "unknown"


def _slug_from_path(path: Path, wiki_root: Path) -> str:
    """Build a URL slug from a wiki page path."""
    relative = path.relative_to(wiki_root)
    return str(relative.with_suffix("")).replace("\\", "/")


def _find_page(slug: str) -> Path | None:
    """Resolve a slug to a wiki page path, or None if not found."""
    candidate = _wiki_root() / f"{slug}.md"
    return candidate if candidate.is_file() else None


def _build_summary(path: Path, wiki_root: Path) -> WikiPageSummary:
    """Build a WikiPageSummary from a markdown file on disk."""
    text = path.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    slug = _slug_from_path(path, wiki_root)
    title = fm.get("title", path.stem.replace("-", " ").title())
    page_type = _page_type_from_path(path, wiki_root)
    sources = fm.get("sources", "")
    source_count = len(sources.strip("[]").split(",")) if sources else 0
    created_at = fm.get("generated_at", "")
    return WikiPageSummary(
        slug=slug,
        title=title,
        page_type=page_type,
        source_count=source_count,
        created_at=created_at,
    )


@get("/api/wiki")
async def wiki_list_route() -> list[dict[str, Any]]:
    """List all wiki pages across subdirectories.

    If wiki/index.md exists, regenerate it first to ensure freshness,
    then build the page list from disk.
    """
    _require_wiki()
    root = _wiki_root()

    index_path = root / "index.md"
    if index_path.is_file():
        from lilbee.wiki.index import update_wiki_index

        update_wiki_index()

    pages: list[WikiPageSummary] = []
    for subdir in ("summaries", "concepts"):
        for path in _list_md_files(root / subdir):
            pages.append(_build_summary(path, root))
    return [p.model_dump() for p in pages]


@get("/api/wiki/drafts")
async def wiki_drafts_route() -> list[dict[str, Any]]:
    """List draft pages that failed the quality gate."""
    _require_wiki()
    root = _wiki_root()
    drafts: list[WikiPageSummary] = []
    for path in _list_md_files(root / "drafts"):
        drafts.append(_build_summary(path, root))
    return [d.model_dump() for d in drafts]


@get("/api/wiki/citations")
async def wiki_citations_reverse_route(
    source: str = Parameter(query="source", default=""),
) -> list[dict[str, Any]]:
    """Reverse citation lookup: which wiki pages cite a given source."""
    _require_wiki()
    if not source:
        return []
    # Stub: return empty until citation table is implemented
    return []


@get("/api/wiki/lint/{task_id:str}")
async def wiki_lint_status_route(task_id: str) -> dict[str, Any]:
    """Poll lint task status by task ID."""
    _require_wiki()
    return {"task_id": task_id, "status": "not_implemented", "issues": []}


@get("/api/wiki/{slug:path}")
async def wiki_read_route(slug: str) -> dict[str, Any]:
    """Read a specific wiki page as markdown, or its citations."""
    _require_wiki()
    slug = slug.lstrip("/")
    if slug.endswith("/citations"):
        real_slug = slug.removesuffix("/citations")
        return _citations_for_slug(real_slug)
    path = _find_page(slug)
    if path is None:
        raise NotFoundException(detail=f"wiki page not found: {slug}")
    text = path.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    return {
        "slug": slug,
        "title": fm.get("title", path.stem.replace("-", " ").title()),
        "content": text,
    }


def _citations_for_slug(slug: str) -> dict[str, Any]:
    """Return citation chain for a wiki page."""
    path = _find_page(slug)
    if path is None:
        raise NotFoundException(detail=f"wiki page not found: {slug}")
    # Stub: return empty until citation table is implemented
    return {"slug": slug, "citations": []}


@post("/api/wiki/lint")
async def wiki_lint_route() -> dict[str, Any]:
    """Trigger a full wiki lint (background task)."""
    _require_wiki()
    return {"status": "not_implemented"}


@post("/api/wiki/generate/{source:path}")
async def wiki_generate_route(source: str) -> dict[str, Any]:
    """Trigger wiki generation for a source document (background)."""
    _require_wiki()
    return {"status": "not_implemented", "source": source.lstrip("/")}


@post("/api/wiki/prune")
async def wiki_prune_route() -> dict[str, Any]:
    """Trigger pruning of stale/orphaned wiki pages."""
    _require_wiki()
    from lilbee.services import get_services
    from lilbee.wiki.prune import prune_wiki

    report = prune_wiki(get_services().store)
    return {
        "records": [
            {
                "wiki_source": r.wiki_source,
                "action": r.action.value,
                "reason": r.reason,
            }
            for r in report.records
        ],
        "archived": report.archived_count,
        "flagged": report.flagged_count,
    }
