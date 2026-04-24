"""Wiki layer route handlers — page listing, reading, citations, lint, generation, pruning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from litestar import delete, get, post
from litestar.exceptions import NotFoundException
from litestar.params import Parameter

from lilbee import services as svc_mod
from lilbee.config import cfg
from lilbee.server.auth import read_only
from lilbee.server.models import (
    DraftInfoResponse,
    WikiCitationRecord,
    WikiCitationsResult,
    WikiDraftAcceptResponse,
    WikiDraftDiffResponse,
    WikiDraftRejectResponse,
    WikiLintIssueItem,
    WikiLintResult,
    WikiPageDetail,
    WikiPruneRecordResponse,
    WikiPruneResult,
)
from lilbee.wiki import lint as lint_mod
from lilbee.wiki import prune as prune_mod
from lilbee.wiki.browse import (
    find_page,
    list_pages,
    read_page,
)
from lilbee.wiki.drafts import (
    accept_draft,
    diff_draft,
    list_drafts,
    reject_draft,
)
from lilbee.wiki.index import update_wiki_index
from lilbee.wiki.shared import WIKI_DISABLED_ERROR


def _wiki_root() -> Path:
    """Resolve the wiki directory under data_root."""
    return cfg.data_root / cfg.wiki_dir


def _require_wiki() -> None:
    """Raise 404 if the wiki feature is disabled."""
    if not cfg.wiki:
        raise NotFoundException(detail=WIKI_DISABLED_ERROR)


def _find_page(slug: str) -> Path | None:
    """Resolve a slug to a wiki page path via the browse module."""
    return find_page(_wiki_root(), slug)


@get("/api/wiki")
@read_only
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
    return [p.to_dict() for p in pages]


@get("/api/wiki/drafts")
@read_only
async def wiki_drafts_route() -> list[DraftInfoResponse]:
    """List pending wiki drafts with drift, faithfulness, and pending-marker info."""
    _require_wiki()
    return [DraftInfoResponse(**d.to_dict()) for d in list_drafts(_wiki_root())]


@get("/api/wiki/drafts/diff/{slug:path}")
@read_only
async def wiki_draft_diff_route(slug: str) -> WikiDraftDiffResponse:
    """Return the unified diff of a draft against its published counterpart.

    The ``diff`` action prefix precedes the slug because Litestar's
    ``{slug:path}`` parameter is greedy and does not support a fixed
    trailing segment. Keeping the action as a literal prefix lets
    nested slugs (``cars/caprice``) flow through unchanged.
    """
    _require_wiki()
    slug = slug.lstrip("/")
    try:
        diff = diff_draft(slug, _wiki_root())
    except FileNotFoundError as exc:
        raise NotFoundException(detail=f"draft not found: {slug}") from exc
    return WikiDraftDiffResponse(slug=slug, diff=diff)


@post("/api/wiki/drafts/accept/{slug:path}")
async def wiki_draft_accept_route(slug: str) -> WikiDraftAcceptResponse:
    """Accept a draft: overwrite the published page and re-index its chunks.

    See :func:`wiki_draft_diff_route` for the action-prefix rationale.
    """
    _require_wiki()
    slug = slug.lstrip("/")
    store = svc_mod.get_services().store
    try:
        result = accept_draft(slug, _wiki_root(), store)
    except FileNotFoundError as exc:
        raise NotFoundException(detail=f"draft not found: {slug}") from exc
    return WikiDraftAcceptResponse(**result.to_dict())


@delete("/api/wiki/drafts/{slug:path}", status_code=200)
async def wiki_draft_reject_route(slug: str) -> WikiDraftRejectResponse:
    """Reject a draft: delete the draft file without touching the published page."""
    _require_wiki()
    slug = slug.lstrip("/")
    try:
        reject_draft(slug, _wiki_root())
    except FileNotFoundError as exc:
        raise NotFoundException(detail=f"draft not found: {slug}") from exc
    return WikiDraftRejectResponse(slug=slug)


@get("/api/wiki/citations")
@read_only
async def wiki_citations_reverse_route(
    source: str = Parameter(query="source", default=""),
) -> list[WikiCitationRecord]:
    """Reverse citation lookup: which wiki pages cite a given source."""
    _require_wiki()
    if not source:
        return []
    records = svc_mod.get_services().store.get_citations_for_source(source)
    return [WikiCitationRecord(**r) for r in records]


@get("/api/wiki/{slug:path}")
@read_only
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
    return WikiCitationsResult(slug=slug, citations=[WikiCitationRecord(**r) for r in records])


@post("/api/wiki/lint")
async def wiki_lint_route() -> WikiLintResult:
    """Trigger a full wiki lint."""
    _require_wiki()
    report = lint_mod.lint_all(svc_mod.get_services().store)
    return WikiLintResult(
        issues=[WikiLintIssueItem(**i.to_dict()) for i in report.issues],
        errors=report.error_count,
        warnings=report.warning_count,
    )


@post("/api/wiki/prune")
async def wiki_prune_route() -> WikiPruneResult:
    """Trigger pruning of stale/orphaned wiki pages."""
    _require_wiki()
    report = prune_mod.prune_wiki(svc_mod.get_services().store)
    return WikiPruneResult(
        records=[WikiPruneRecordResponse(**r.to_dict()) for r in report.records],
        archived=report.archived_count,
        flagged=report.flagged_count,
    )
