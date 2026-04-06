"""Wiki layer route handlers — page listing, reading, citations, lint, generation, pruning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from litestar import get, post
from litestar.exceptions import NotFoundException
from litestar.params import Parameter

from lilbee import services as svc_mod
from lilbee.config import cfg
from lilbee.security import validate_path_within
from lilbee.server.models import (
    WikiCitationsResult,
    WikiGenerateResult,
    WikiLintResult,
    WikiLintStatusResult,
    WikiPageDetail,
    WikiPageSummary,
    WikiPruneRecordResponse,
    WikiPruneResult,
)
from lilbee.wiki import gen as gen_mod
from lilbee.wiki import lint as lint_mod
from lilbee.wiki import prune as prune_mod
from lilbee.wiki.browse import (
    WikiPageInfo,
    _build_page_info,
    _list_md_files,
    find_page,
    list_pages,
    read_page,
)
from lilbee.wiki.index import update_wiki_index


def _wiki_root() -> Path:
    """Resolve the wiki directory under data_root."""
    return cfg.data_root / cfg.wiki_dir


def _require_wiki() -> None:
    """Raise 404 if the wiki feature is disabled."""
    if not cfg.wiki:
        raise NotFoundException(detail="wiki not enabled")


def _find_page(slug: str) -> Path | None:
    """Resolve a slug to a wiki page path via the browse module."""
    return find_page(_wiki_root(), slug)


def _build_summary(path: Path, wiki_root: Path) -> WikiPageSummary:
    """Build a WikiPageSummary from a markdown file on disk.

    Delegates to browse._build_page_info and converts to pydantic.
    """
    return _summary_from_info(_build_page_info(path, wiki_root))


def _summary_from_info(info: WikiPageInfo) -> WikiPageSummary:
    """Convert a browse WikiPageInfo to a server WikiPageSummary."""
    return WikiPageSummary(
        slug=info.slug,
        title=info.title,
        page_type=info.page_type,
        source_count=info.source_count,
        created_at=info.created_at,
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
        update_wiki_index()

    pages = list_pages(root)
    return [_summary_from_info(p).model_dump() for p in pages]


@get("/api/wiki/drafts")
async def wiki_drafts_route() -> list[dict[str, Any]]:
    """List draft pages that failed the quality gate."""
    _require_wiki()
    root = _wiki_root()
    drafts: list[WikiPageSummary] = []
    for path in _list_md_files(root / "drafts"):
        drafts.append(_summary_from_info(_build_page_info(path, root)))
    return [d.model_dump() for d in drafts]


@get("/api/wiki/citations")
async def wiki_citations_reverse_route(
    source: str = Parameter(query="source", default=""),
) -> list[dict[str, Any]]:
    """Reverse citation lookup: which wiki pages cite a given source."""
    _require_wiki()
    if not source:
        return []
    records = svc_mod.get_services().store.get_citations_for_source(source)
    return [dict(r) for r in records]


@get("/api/wiki/lint/{task_id:str}")
async def wiki_lint_status_route(task_id: str) -> WikiLintStatusResult:
    """Poll lint task status by task ID."""
    _require_wiki()
    return WikiLintStatusResult(task_id=task_id)


@get("/api/wiki/{slug:path}")
async def wiki_read_route(slug: str) -> WikiPageDetail | WikiCitationsResult:
    """Read a specific wiki page as markdown, or its citations."""
    _require_wiki()
    slug = slug.lstrip("/")
    if slug.endswith("/citations"):
        real_slug = slug.removesuffix("/citations")
        return _citations_for_slug(real_slug)
    result = read_page(_wiki_root(), slug)
    if result is None:
        raise NotFoundException(detail=f"wiki page not found: {slug}")
    return WikiPageDetail(
        slug=result.slug,
        title=result.title,
        content=result.content,
    )


def _citations_for_slug(slug: str) -> WikiCitationsResult:
    """Return citation chain for a wiki page."""
    path = _find_page(slug)
    if path is None:
        raise NotFoundException(detail=f"wiki page not found: {slug}")
    wiki_source = f"{cfg.wiki_dir}/{slug}.md"
    records = svc_mod.get_services().store.get_citations_for_wiki(wiki_source)
    return WikiCitationsResult(slug=slug, citations=[dict(r) for r in records])


@post("/api/wiki/lint")
async def wiki_lint_route() -> WikiLintResult:
    """Trigger a full wiki lint."""
    _require_wiki()
    report = lint_mod.lint_all(svc_mod.get_services().store)
    return WikiLintResult(
        issues=[i.to_dict() for i in report.issues],
        errors=report.error_count,
        warnings=report.warning_count,
    )


@post("/api/wiki/generate/{source:path}")
async def wiki_generate_route(source: str) -> WikiGenerateResult:
    """Trigger wiki generation for a source document."""
    _require_wiki()
    source = source.lstrip("/")

    try:
        validate_path_within(cfg.documents_dir / source, cfg.documents_dir)
    except ValueError:
        raise NotFoundException(detail=f"invalid source path: {source}") from None

    svc = svc_mod.get_services()
    chunks = svc.store.get_chunks_by_source(source)
    result = gen_mod.generate_summary_page(source, chunks, svc.provider, svc.store)
    if result is None:
        return WikiGenerateResult(status="failed", source=source)
    return WikiGenerateResult(status="generated", source=source, path=str(result))


@post("/api/wiki/prune")
async def wiki_prune_route() -> WikiPruneResult:
    """Trigger pruning of stale/orphaned wiki pages."""
    _require_wiki()
    report = prune_mod.prune_wiki(svc_mod.get_services().store)
    return WikiPruneResult(
        records=[
            WikiPruneRecordResponse(**r.to_dict()) for r in report.records
        ],
        archived=report.archived_count,
        flagged=report.flagged_count,
    )
