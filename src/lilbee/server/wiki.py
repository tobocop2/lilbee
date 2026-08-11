"""Wiki layer route handlers: page listing, reading, citations, lint, generation, pruning.

Every route needs the token: pages are generated from the user's own corpus,
so even the titles in a listing are their content.
"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any

from litestar import MediaType, Response, delete, get, patch, post
from litestar.exceptions import ClientException, NotFoundException
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import FromPath, FromQuery
from litestar.response import Stream
from litestar.status_codes import HTTP_200_OK, HTTP_409_CONFLICT

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.core.security import PathTraversalError
from lilbee.data.store import Store
from lilbee.server import handlers
from lilbee.server.handlers.sse import SSE_MEDIA_TYPE
from lilbee.server.models import (
    DraftInfoResponse,
    WikiBuildDryRunResult,
    WikiCitationRecord,
    WikiCitationsResult,
    WikiDraftAcceptResponse,
    WikiDraftDiffResponse,
    WikiDraftRejectResponse,
    WikiEntityCandidateResponse,
    WikiIndexResult,
    WikiLintIssueItem,
    WikiLintResult,
    WikiPageDetail,
    WikiPruneRecordResponse,
    WikiPruneResult,
    WikiStatusResult,
    WikiWipeResult,
)
from lilbee.wiki import lint as lint_mod
from lilbee.wiki import prune as prune_mod
from lilbee.wiki import wipe as wipe_mod
from lilbee.wiki.browse import (
    find_page,
    list_pages,
    read_page,
)
from lilbee.wiki.drafts import (
    DraftAcceptError,
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
from lilbee.wiki.stubs import WikiStub, load_stub_index, ungenerated_stubs


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

    Reads the tree without touching it. The listing is built by walking pages,
    and every path that changes the tree (build, update, synthesize, prune,
    draft-accept) refreshes index.md itself, so a read never rewrites it.
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
    # list_drafts reads every draft and stats its published counterpart;
    # offload like the page listing.
    drafts = await asyncio.to_thread(list_drafts, _wiki_root())
    return [DraftInfoResponse(**d.to_dict()) for d in drafts]


@get("/api/wiki/drafts/diff/{slug:path}")
async def wiki_draft_diff_route(slug: FromPath[str]) -> WikiDraftDiffResponse:
    """Return the unified diff of a draft against its published counterpart.

    The ``diff`` action prefix precedes the slug because Litestar's
    ``{slug:path}`` parameter is greedy and does not support a fixed
    trailing segment. Keeping the action as a literal prefix lets
    nested slugs (``cars/caprice``) flow through unchanged.
    """
    _require_wiki()
    slug = slug.lstrip("/")
    try:
        # diff_draft reads both files and diffs them; offload like the listing.
        diff = await asyncio.to_thread(diff_draft, slug, _wiki_root())
    except FileNotFoundError as exc:
        raise NotFoundException(detail=f"draft not found: {slug}") from exc
    except PathTraversalError as exc:
        raise ClientException(detail=INVALID_DRAFT_SLUG_ERROR) from exc
    return WikiDraftDiffResponse(slug=slug, diff=diff)


@post("/api/wiki/drafts/accept/{slug:path}")
async def wiki_draft_accept_route(slug: FromPath[str]) -> WikiDraftAcceptResponse:
    """Accept a draft: overwrite the published page and re-index its chunks.

    See :func:`wiki_draft_diff_route` for the action-prefix rationale.
    """
    _require_wiki()
    slug = slug.lstrip("/")
    store = svc_mod.get_services().store
    try:
        # accept_draft re-chunks and embeds, so it runs off the event loop; it
        # takes the wiki build mutex itself.
        result = await asyncio.to_thread(accept_draft, slug, _wiki_root(), store)
    except FileNotFoundError as exc:
        raise NotFoundException(detail=f"draft not found: {slug}") from exc
    except DraftAcceptError as exc:
        raise ClientException(detail=str(exc), status_code=HTTP_409_CONFLICT) from exc
    except PathTraversalError as exc:
        raise ClientException(detail=INVALID_DRAFT_SLUG_ERROR) from exc
    return WikiDraftAcceptResponse(**result.to_dict())


@delete("/api/wiki/drafts/{slug:path}", status_code=200)
async def wiki_draft_reject_route(slug: FromPath[str]) -> WikiDraftRejectResponse:
    """Reject a draft: delete the draft file without touching the published page."""
    _require_wiki()
    slug = slug.lstrip("/")
    try:
        # reject takes the wiki build mutex, so it runs off the event loop.
        await asyncio.to_thread(reject_draft, slug, _wiki_root())
    except FileNotFoundError as exc:
        raise NotFoundException(detail=f"draft not found: {slug}") from exc
    except PathTraversalError as exc:
        raise ClientException(detail=INVALID_DRAFT_SLUG_ERROR) from exc
    return WikiDraftRejectResponse(slug=slug)


@get("/api/wiki/citations")
async def wiki_citations_reverse_route(
    source: FromQuery[str] = "",
) -> list[WikiCitationRecord]:
    """Reverse citation lookup: which wiki pages cite a given source."""
    _require_wiki()
    if not source:
        raise ClientException(detail="pass ?source=<document path> to look up citing wiki pages")
    # get_citations_for_source queries LanceDB; offload like wiki_lint_route.
    records = await asyncio.to_thread(svc_mod.get_services().store.get_citations_for_source, source)
    return [WikiCitationRecord(**r) for r in records]


@get("/api/wiki/{slug:path}")
async def wiki_read_route(slug: FromPath[str]) -> WikiPageDetail | WikiCitationsResult:
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
        frontmatter=result.frontmatter,
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
async def wiki_lint_route(
    wiki_source: FromQuery[str] = "",
) -> WikiLintResult:
    """Lint the wiki; an empty ``wiki_source`` lints every page.

    Same single-page argument as ``lilbee wiki lint <page>`` and the
    ``wiki_lint`` MCP tool, and the same issue counts.
    """
    _require_wiki()
    store = svc_mod.get_services().store
    # Either arm reads every cited source and embeds; offload so a lint of a
    # large wiki does not block the loop.
    report = await asyncio.to_thread(_lint_report, wiki_source, store)
    return WikiLintResult(
        issues=[WikiLintIssueItem(**i.to_dict()) for i in report.issues],
        total=len(report.issues),
        errors=report.error_count,
        warnings=report.warning_count,
    )


def _lint_report(wiki_source: str, store: Store) -> lint_mod.LintReport:
    """Lint one page or the whole wiki, as the CLI and MCP surfaces do."""
    if wiki_source:
        return lint_mod.LintReport(issues=lint_mod.lint_wiki_page(wiki_source, store))
    return lint_mod.lint_all(store)


@post("/api/wiki/prune")
async def wiki_prune_route() -> WikiPruneResult:
    """Trigger pruning of stale/orphaned wiki pages."""
    _require_wiki()
    # prune walks the whole tree and store, so it runs off the event loop; it
    # takes the wiki build mutex itself.
    report = await asyncio.to_thread(prune_mod.prune_wiki, svc_mod.get_services().store)
    return WikiPruneResult(
        records=[WikiPruneRecordResponse(**r.to_dict()) for r in report.records],
        archived=report.archived_count,
        flagged=report.flagged_count,
        reconciled=report.reconciled_count,
    )


@post("/api/wiki/index")
async def wiki_index_route() -> WikiIndexResult:
    """Rebuild the browse index of pages the corpus could have.

    Spends no LLM call. Extraction walks every chunk, so it runs off the event
    loop and takes the wiki build mutex itself.
    """
    _require_wiki()
    from lilbee.wiki.stubs import refresh_stub_index

    stubs = await asyncio.to_thread(refresh_stub_index, svc_mod.get_services().store)
    return WikiIndexResult(entries=len(stubs))


@get("/api/wiki/stubs")
async def wiki_stubs_route() -> list[WikiEntityCandidateResponse]:
    """List the pages the corpus names that nothing has written yet.

    The other half of the browse tree: ``GET /api/wiki`` walks written pages,
    this lists the ones a client can ask for. Entries whose page now exists are
    left out, so a client never offers to regenerate a live page. Costs no LLM
    call, and ``slug`` is what ``POST /api/wiki/generate/{slug}`` takes.
    """
    _require_wiki()
    # Reading the index stats every candidate's page and draft; offload like
    # the page listing.
    stubs = await asyncio.to_thread(_ungenerated_stubs)
    return [
        WikiEntityCandidateResponse(
            slug=stub.slug,
            label=stub.label,
            kind=stub.kind,
            type_hint=stub.type_hint,
            mentions=stub.mentions,
            sources=list(stub.sources),
        )
        for stub in stubs
    ]


def _ungenerated_stubs() -> list[WikiStub]:
    """The indexed subjects with no page yet, in slug order."""
    return ungenerated_stubs(load_stub_index(), _wiki_root())


@post("/api/wiki/generate/{slug:path}", media_type=SSE_MEDIA_TYPE)
async def wiki_generate_route(slug: FromPath[str]) -> Stream:
    """Generate one indexed page, streaming progress. Costs a single LLM call.

    One model call takes long enough to look hung, so the response is an SSE
    stream like /api/wiki/build: wiki_phase and wiki_page events, then a done
    event carrying the slug the read route accepts. 404 when the slug names
    nothing in the index; an entry whose sources are gone surfaces as an
    error event, since the run discovers it after the stream has started.
    """
    _require_wiki()
    slug = slug.lstrip("/")
    from lilbee.wiki.lazy import resolve_stub

    # Resolving reads the index file; offload like the stub listing.
    if await asyncio.to_thread(resolve_stub, slug) is None:
        raise NotFoundException(detail=f"no indexed page named {slug!r}")
    return Stream(handlers.wiki_generate_stream(slug), media_type=SSE_MEDIA_TYPE)


@delete("/api/wiki", status_code=200)
async def wiki_wipe_route() -> WikiWipeResult:
    """Delete every generated wiki page and its indexed rows.

    Answers while the wiki is disabled, unlike the other write routes: turning
    the setting off is exactly when a client needs to clear what was already
    generated. The wipe touches the whole tree and the store, so it runs off
    the event loop and takes the wiki build mutex itself.
    """
    report = await asyncio.to_thread(wipe_mod.wipe_wiki, svc_mod.get_services().store)
    return WikiWipeResult(
        pages_removed=report.pages_removed,
        sources_cleared=report.sources_cleared,
        rows_deleted=report.rows_deleted,
    )


@post(
    "/api/wiki/build",
    media_type=SSE_MEDIA_TYPE,
    responses={
        HTTP_200_OK: ResponseSpec(
            data_container=WikiBuildDryRunResult,
            media_type=MediaType.JSON,
            generate_examples=False,
            description="Entity candidates a build would cover, when dry_run=true.",
        )
    },
)
async def wiki_build_route(
    dry_run: FromQuery[bool] = False,
) -> Stream | Response[WikiBuildDryRunResult]:
    """Build the concept and entity wiki across all ingested sources.

    A build issues per-source LLM calls and embeddings and can run for a long
    time, so the response is 201 with an SSE stream: wiki_phase and wiki_page
    events while it runs, then a done event carrying the summary. The work runs
    in a worker thread and holds the wiki build mutex, so a second request
    streams its own progress only once the first run finishes.

    ``dry_run=true`` creates nothing, so it answers 200 with plain JSON: the
    NER entity candidates a build would cover, with no LLM call made. The two
    arms carry different content types, so each is declared under its own
    status code.
    """
    _require_wiki()
    if dry_run:
        return await _build_dry_run()
    return Stream(handlers.wiki_build_stream(), media_type=SSE_MEDIA_TYPE)


async def _build_dry_run() -> Response[WikiBuildDryRunResult]:
    """Extract entity candidates off the event loop and shape them for the wire."""
    from lilbee.wiki.generation import DRY_RUN_CONCEPT_NOTE, preview_build_entities

    rows = await asyncio.to_thread(preview_build_entities, cfg)
    result = WikiBuildDryRunResult(
        entities=[WikiEntityCandidateResponse(**row) for row in rows],
        count=len(rows),
        note=DRY_RUN_CONCEPT_NOTE,
    )
    return Response(result, media_type=MediaType.JSON, status_code=HTTP_200_OK)


@patch("/api/wiki/update", media_type=SSE_MEDIA_TYPE)
async def wiki_update_route() -> Stream:
    """Refresh the concept and entity wiki after an ingest.

    A full rebuild, streamed like /api/wiki/build.
    """
    _require_wiki()
    return Stream(handlers.wiki_build_stream(), media_type=SSE_MEDIA_TYPE)


@post("/api/wiki/synthesize", media_type=SSE_MEDIA_TYPE)
async def wiki_synthesize_route() -> Stream:
    """Generate synthesis pages for concept clusters spanning 3+ sources.

    Streams per-cluster progress and shares the wiki build mutex, so synthesis
    can't race a build over the same on-disk wiki tree.
    """
    _require_wiki()
    return Stream(handlers.wiki_synthesize_stream(), media_type=SSE_MEDIA_TYPE)


@get("/api/wiki/status")
async def wiki_status_route() -> WikiStatusResult:
    """Wiki layer status: page counts and recent lint counts.

    Answers while the wiki is disabled so a client can render the disabled
    state without a second round trip to /api/config.
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
