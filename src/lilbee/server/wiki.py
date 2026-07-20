"""Wiki layer route handlers: page listing, reading, citations, lint, generation, pruning.

Every route here needs the session token. Wiki pages are generated from the
user's own corpus, so a page body, a citation list, and even the page titles
in a listing are their content, not public reference material.
"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any

from litestar import delete, get, patch, post
from litestar.exceptions import ClientException, NotFoundException
from litestar.params import Parameter

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.core.security import PathTraversalError
from lilbee.server.models import (
    DraftInfoResponse,
    WikiBuildResult,
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
    WikiStatusResult,
    WikiSynthesizeResult,
)
from lilbee.wiki import lint as lint_mod
from lilbee.wiki import prune as prune_mod
from lilbee.wiki import run_full_build, run_full_synthesize
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
from lilbee.wiki.shared import (
    INVALID_DRAFT_SLUG_ERROR,
    WIKI_DISABLED_ERROR,
    WikiSubdir,
    total_wiki_pages,
)


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
async def wiki_list_route() -> list[dict[str, Any]]:
    """List all wiki pages across subdirectories.

    Reads the tree without touching it. This route is unauthenticated, so it
    used to hand any caller a way to rewrite index.md on repeat and to race the
    build path over the same file. The listing never depended on that write
    anyway: it is built by walking pages, and every path that changes the tree
    (build, update, synthesize, prune, draft-accept) refreshes the index itself.
    """
    _require_wiki()
    # list_pages walks the whole tree; offload so the listing doesn't block
    # the event loop.
    pages = await asyncio.to_thread(list_pages, _wiki_root())
    return [p.to_dict() for p in pages]


@get("/api/wiki/drafts")
async def wiki_drafts_route() -> list[DraftInfoResponse]:
    """List pending wiki drafts with drift, faithfulness, and pending-marker info."""
    _require_wiki()
    return [DraftInfoResponse(**d.to_dict()) for d in list_drafts(_wiki_root())]


@get("/api/wiki/drafts/diff/{slug:path}")
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
    except PathTraversalError as exc:
        raise ClientException(detail=INVALID_DRAFT_SLUG_ERROR) from exc
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
        # accept_draft overwrites a published page and refreshes the index, the
        # same artifacts a build writes, so it shares the build mutex. It also
        # re-chunks and embeds, so it runs off the event loop.
        async with _wiki_build_lock():
            result = await asyncio.to_thread(accept_draft, slug, _wiki_root(), store)
    except FileNotFoundError as exc:
        raise NotFoundException(detail=f"draft not found: {slug}") from exc
    except PathTraversalError as exc:
        raise ClientException(detail=INVALID_DRAFT_SLUG_ERROR) from exc
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
    except PathTraversalError as exc:
        raise ClientException(detail=INVALID_DRAFT_SLUG_ERROR) from exc
    return WikiDraftRejectResponse(slug=slug)


@get("/api/wiki/citations")
async def wiki_citations_reverse_route(
    source: str = Parameter(query="source", default=""),
) -> list[WikiCitationRecord]:
    """Reverse citation lookup: which wiki pages cite a given source."""
    _require_wiki()
    if not source:
        return []
    # get_citations_for_source queries LanceDB; offload like wiki_lint_route.
    records = await asyncio.to_thread(svc_mod.get_services().store.get_citations_for_source, source)
    return [WikiCitationRecord(**r) for r in records]


@get("/api/wiki/{slug:path}")
async def wiki_read_route(slug: str) -> WikiPageDetail | WikiCitationsResult:
    """Read a specific wiki page as markdown, or its citations."""
    _require_wiki()
    slug = slug.lstrip("/")
    if slug.endswith("/citations"):
        real_slug = slug.removesuffix("/citations")
        return await _citations_for_slug(real_slug)
    result = read_page(_wiki_root(), slug)
    if result is None:
        raise NotFoundException(detail=f"wiki page not found: {slug}")
    return WikiPageDetail(
        slug=result.slug,
        title=result.title,
        content=result.content,
    )


async def _citations_for_slug(slug: str) -> WikiCitationsResult:
    """Return citation chain for a wiki page."""
    path = _find_page(slug)
    if path is None:
        raise NotFoundException(detail=f"wiki page not found: {slug}")
    wiki_source = f"{cfg.wiki_dir}/{slug}.md"
    # get_citations_for_wiki queries LanceDB; offload like wiki_lint_route.
    records = await asyncio.to_thread(
        svc_mod.get_services().store.get_citations_for_wiki, wiki_source
    )
    return WikiCitationsResult(slug=slug, citations=[WikiCitationRecord(**r) for r in records])


@post("/api/wiki/lint")
async def wiki_lint_route() -> WikiLintResult:
    """Trigger a full wiki lint."""
    _require_wiki()
    # lint_all scans every page and embeds; offload so it doesn't block the loop.
    report = await asyncio.to_thread(lint_mod.lint_all, svc_mod.get_services().store)
    return WikiLintResult(
        issues=[WikiLintIssueItem(**i.to_dict()) for i in report.issues],
        errors=report.error_count,
        warnings=report.warning_count,
    )


@post("/api/wiki/prune")
async def wiki_prune_route() -> WikiPruneResult:
    """Trigger pruning of stale/orphaned wiki pages."""
    _require_wiki()
    # prune archives pages and rewrites the index, so it takes the build mutex
    # like the other writers. It also walks the whole tree and store, so it runs
    # off the event loop.
    async with _wiki_build_lock():
        report = await asyncio.to_thread(prune_mod.prune_wiki, svc_mod.get_services().store)
    return WikiPruneResult(
        records=[WikiPruneRecordResponse(**r.to_dict()) for r in report.records],
        archived=report.archived_count,
        flagged=report.flagged_count,
    )


# Serialize wiki builds: ``run_full_build`` writes pages, the wiki index, and
# the wiki log; two concurrent calls would corrupt those. The lock is created
# lazily because ``Lock()`` requires a running event loop.
_WIKI_BUILD_LOCK: asyncio.Lock | None = None


def _wiki_build_lock() -> asyncio.Lock:
    """Return the per-process wiki-build mutex, creating it on first call."""
    global _WIKI_BUILD_LOCK
    if _WIKI_BUILD_LOCK is None:
        _WIKI_BUILD_LOCK = asyncio.Lock()
    return _WIKI_BUILD_LOCK


def _reset_wiki_build_lock() -> None:
    """Test hook: clear the per-process wiki-build mutex.

    Mirrors ``handlers._reset_ingest_locks`` so a test that creates the
    lock under one event loop doesn't leak it into the next test.
    """
    global _WIKI_BUILD_LOCK
    _WIKI_BUILD_LOCK = None


@post("/api/wiki/build")
async def wiki_build_route() -> WikiBuildResult:
    """Build the concept and entity wiki across all ingested sources.

    The build is CPU- and IO-bound (LLM calls, embeddings, file writes) so
    it runs in a worker thread; concurrent build/update requests serialize
    on a per-process lock so they don't corrupt the wiki index.
    """
    _require_wiki()
    async with _wiki_build_lock():
        result = await asyncio.to_thread(run_full_build, cfg)
    return WikiBuildResult(**result)


@patch("/api/wiki/update")
async def wiki_update_route() -> WikiBuildResult:
    """Refresh the concept and entity wiki after an ingest. Currently a full rebuild."""
    _require_wiki()
    async with _wiki_build_lock():
        result = await asyncio.to_thread(run_full_build, cfg)
    return WikiBuildResult(**result)


@post("/api/wiki/synthesize")
async def wiki_synthesize_route() -> WikiSynthesizeResult:
    """Generate synthesis pages for concept clusters spanning 3+ sources.

    Shares the wiki-build mutex so synthesis can't race a build/update
    over the same on-disk wiki tree.
    """
    _require_wiki()
    async with _wiki_build_lock():
        result = await asyncio.to_thread(run_full_synthesize, cfg)
    return WikiSynthesizeResult(**result)


@get("/api/wiki/status")
async def wiki_status_route() -> WikiStatusResult:
    """Wiki layer status: page counts and recent lint counts.

    Token-gated, unlike the other wiki reads: the lint behind it walks every
    page and issues a store query each, so leaving it open made it an
    unauthenticated way to spend the machine's IO budget. It still answers
    while the wiki is disabled so a client can render the disabled state
    without a second round trip to /api/config.
    """
    root = _wiki_root()
    if not cfg.wiki or not root.exists():
        # A disabled wiki can still have a tree left over from an earlier
        # build; report the disabled state rather than linting it.
        return WikiStatusResult(wiki_enabled=cfg.wiki)

    summaries_dir = root / WikiSubdir.SUMMARIES
    drafts_dir = root / WikiSubdir.DRAFTS
    summaries = list(summaries_dir.rglob("*.md")) if summaries_dir.exists() else []
    drafts = list(drafts_dir.rglob("*.md")) if drafts_dir.exists() else []

    # record_log=False: a status poll is a read, and appending a LINT entry to
    # log.md on every poll is what that flag exists to avoid. The MCP and CLI
    # status surfaces already pass it.
    report = await asyncio.to_thread(
        partial(lint_mod.lint_all, svc_mod.get_services().store, record_log=False)
    )
    return WikiStatusResult(
        wiki_enabled=cfg.wiki,
        summaries=len(summaries),
        drafts=len(drafts),
        pages=total_wiki_pages(root),
        lint_errors=report.error_count,
        lint_warnings=report.warning_count,
    )
