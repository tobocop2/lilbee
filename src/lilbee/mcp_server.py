"""MCP server exposing lilbee as tools for AI agents."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import json
import logging
import os
import re
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast
from weakref import WeakKeyDictionary

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import Tool as MCPTool

from lilbee.app.memory import (
    MEMORY_DISABLED_HINT,
    forget,
    list_memories,
    memory_enabled,
    recall,
    remember,
)
from lilbee.app.placement import (
    PlacementView,
    get_placement,
    placement_refused_message,
    preview_placement,
    set_placement,
)
from lilbee.app.search import clean_result
from lilbee.app.services import get_services, reset_services, reset_store
from lilbee.app.settings import (
    SettingInfo,
    apply_settings_update,
    get_setting,
    list_settings,
    provider_reset_refused_message,
    requires_services_reset,
    reset_settings,
)
from lilbee.catalog.types import ModelSource
from lilbee.core.config import cfg
from lilbee.core.config.enums import CrawlRenderMode
from lilbee.core.settings import overlay_persisted_settings
from lilbee.core.system import LOCAL_ROOT_DIRNAME, canonical_data_root
from lilbee.crawler import crawler_available, is_url, require_valid_crawl_url
from lilbee.crawler.task import get_task, start_crawl
from lilbee.data.store import (
    EmbeddingModelMismatchError,
    MemoryKind,
    MemorySource,
    SearchScope,
    agent_owner,
    scope_to_chunk_type,
)
from lilbee.runtime.cancellation import TaskCancelledError
from lilbee.sessions import (
    AGENT_SESSIONS_DISABLED_HINT,
    MessageRole,
    Session,
    SessionMessage,
    SessionNotFoundError,
    SessionOrigin,
    SessionOwnershipError,
    TitleSource,
    agent_sessions_enabled,
)
from lilbee.wiki.shared import (
    INVALID_DRAFT_SLUG_ERROR,
    WIKI_DISABLED_ERROR,
    WikiSubdir,
    total_wiki_pages,
)

if TYPE_CHECKING:
    from lilbee.providers.fleet.placement_spec import PlacementSpec

log = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "Local search engine over the user's files, code, and crawled pages. "
    "For any question about the user's own documents or codebase -- a lookup, a "
    "find-in-docs, 'where is X', 'how does Y work here' -- call lilbee_search first "
    "and answer from its cited chunks. Prefer it over web-fetch or file-read tools: "
    "those cannot see the indexed corpus."
)


class _TransportState:
    """Process-level MCP transport facts.

    ``http_mounted`` is True only when MCP is exposed over the shared
    streamable-http daemon (set by build_mcp_mount). On that transport multiple
    agents share one process and one global cfg/Services singleton, so
    vault-switching (init) and factory reset are refused: switching or tearing
    down the store under concurrent in-flight handlers is a use-after-close /
    identity race. stdio (one agent per process) keeps both.
    """

    http_mounted: bool = False


_transport = _TransportState()


def set_http_mounted(value: bool) -> None:
    """Mark whether this process serves MCP over the shared HTTP daemon."""
    _transport.http_mounted = value


_F = TypeVar("_F", bound=Callable[..., Any])

# Set by the offload for the duration of one sync handler and read by the long
# ones. A parameter would reach the generated tool schema and invite an agent to
# pass it; the handlers keep the plain signature that in-process callers use.
_CANCEL: ContextVar[threading.Event | None] = ContextVar("lilbee_mcp_cancel", default=None)


def _caller_cancelled() -> threading.Event | None:
    """The stop token for the handler running on this thread, if the offload set one."""
    return _CANCEL.get()


def _offload_sync(fn: _F) -> _F:
    """Run a sync tool handler off the event loop; async handlers pass through.

    The bundled mcp SDK calls sync tool handlers directly on the event loop, so
    under the shared streamable-http daemon one slow handler would stall every
    connected agent. ``functools.wraps`` preserves the wrapped signature so the
    generated tool schema is unchanged.

    ``abandon_on_cancel`` releases the caller the moment it cancels. A worker
    thread cannot be interrupted, so the handler still runs to completion; the
    default would additionally hold the cancelling task until it did, which on
    the shared mount keeps a connection open for a request nobody is waiting
    for. Work that must actually stop takes a cancel token instead, as the
    ingest tools do.
    """
    if inspect.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def _runner(*args: Any, **kwargs: Any) -> Any:
        with _cancel_token() as token:
            _CANCEL.set(token)
            return await anyio.to_thread.run_sync(
                functools.partial(fn, *args, **kwargs), abandon_on_cancel=True
            )

    return cast("_F", _runner)


# (handler, wire name override, gate) applied by build_mcp_server per instance.
_REGISTRATIONS: list[tuple[Callable[..., Any], str | None, Callable[[], bool] | None]] = []


def _tool(fn: _F) -> _F:
    """Register *fn* as an MCP tool with sync handlers offloaded off the loop.

    Returns the original callable so in-process callers (tests, the stdio
    fallback) keep the synchronous API while the schema sees the offloaded form.
    """
    _REGISTRATIONS.append((_offload_sync(fn), None, None))
    return fn


def _tool_named(name: str) -> Callable[[_F], _F]:
    """Register an MCP tool under an explicit wire *name* (sync handlers offloaded)."""

    def deco(fn: _F) -> _F:
        _REGISTRATIONS.append((_offload_sync(fn), name, None))
        return fn

    return deco


def _tool_if(when: Callable[[], bool]) -> Callable[[_F], _F]:
    """Register an MCP tool gated on *when*, evaluated at server-build time.

    The function stays importable so direct callers (tests, in-process
    fallback) can still reach it. A server built after a config change
    carries the current tool surface; live servers keep theirs.
    """
    if not callable(when):
        raise TypeError("_tool_if takes a zero-arg callable, evaluated per build")

    def deco(fn: _F) -> _F:
        _REGISTRATIONS.append((_offload_sync(fn), None, when))
        return fn

    return deco


def _wiki_enabled() -> bool:
    return cfg.wiki


def _error(msg: str) -> dict[str, Any]:
    """Uniform error envelope MCP tool handlers return on a failure path.

    Typed as ``dict[str, Any]`` rather than a TypedDict so it composes
    with the success-side returns under the existing handler signatures
    without forcing every caller to widen its return type.
    """
    return {"error": msg}


def _wiki_tool(fn: _F) -> _F:
    """Register a wiki MCP tool: gated at build time, re-checked per call.

    A live server keeps the tool surface it was built with, so every handler
    re-reads ``cfg.wiki`` and refuses once the setting is turned off at runtime.
    """

    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if not cfg.wiki:
            return _error(WIKI_DISABLED_ERROR)
        return fn(*args, **kwargs)

    checked = cast("_F", guarded)
    _tool_if(_wiki_enabled)(checked)
    return checked


@_tool
def search(
    query: str, top_k: int | None = None, scope: str = SearchScope.BOTH.value
) -> list[dict[str, Any]] | dict[str, Any]:
    """Search the user's indexed documents, code, and crawled pages; prefer it over web-fetch or
    file-read tools. Returns chunks with citations. ``scope``: "both" (default) / "raw" / "wiki"."""
    if not query or not query.strip():
        return _error("query must not be empty")
    try:
        chunk_type = scope_to_chunk_type(scope)
    except ValueError:
        # Smaller models routinely echo prose like "indexed docs" or "all"
        # back as the scope value. Treat unrecognised scopes as the default
        # "both" rather than a hard failure so the request still does work.
        log.warning("lilbee_search: unknown scope %r, falling back to %r", scope, SearchScope.BOTH)
        chunk_type = scope_to_chunk_type(SearchScope.BOTH.value)
    if top_k is not None and top_k < 1:
        # Same lenient stance as the scope fallback above: do the search with
        # the configured default rather than hard-failing the agent's call.
        log.warning("lilbee_search: top_k %d is not positive, using the default", top_k)
        top_k = None
    effective_top_k = top_k if top_k is not None else cfg.top_k
    try:
        results = get_services().searcher.search(
            query, top_k=effective_top_k, chunk_type=chunk_type
        )
        return [clean_result(r) for r in results]
    except EmbeddingModelMismatchError as exc:
        # Structured so an agent can offer to adopt the index's embedder
        # rather than parse prose out of a generic error. Names the embedder,
        # which the HTTP search route keeps out of its generic 503.
        return {
            "error": str(exc),
            "code": "INDEX_EMBEDDER_MISMATCH",
            "persisted_model": exc.persisted_model,
            "persisted_dim": exc.persisted_dim,
            "adoptable": exc.dims_match,
        }
    except Exception as exc:
        return _error(str(exc))


@_tool
def status() -> dict[str, Any]:
    """Show indexed documents, configuration, and chunk counts."""
    sources = get_services().store.get_sources()
    return {
        "config": {
            "documents_dir": str(cfg.documents_dir),
            "data_dir": str(cfg.data_dir),
            "chat_model": cfg.chat_model,
            "embedding_model": cfg.embedding_model,
            "vision_model": cfg.vision_model,
            "reranker_model": cfg.reranker_model,
            "enable_ocr": cfg.enable_ocr,
            "num_ctx": cfg.num_ctx,
            "num_ctx_max": cfg.num_ctx_max,
            "chat_n_ctx_target": cfg.chat_n_ctx_target,
            "flash_attention": cfg.flash_attention,
            "kv_cache_type": cfg.kv_cache_type.value,
            "n_gpu_layers": cfg.n_gpu_layers,
            "cpu_moe": cfg.cpu_moe,
            "n_cpu_moe": cfg.n_cpu_moe,
            "main_gpu": cfg.main_gpu,
            "gpu_devices": cfg.gpu_devices,
        },
        "sources": [
            {"filename": s["filename"], "chunk_count": s["chunk_count"]}
            for s in sorted(sources, key=lambda x: x["filename"])
        ],
        "total_chunks": sum(s["chunk_count"] for s in sources),
        "entities": _entity_status_dict(),
    }


def _entity_status_dict() -> dict[str, Any] | None:
    """Entity types + extracted rows, mirroring the HTTP status section."""
    from lilbee.app.status import entity_status

    section = entity_status()
    return section.model_dump() if section is not None else None


@contextmanager
def _cancel_token() -> Iterator[threading.Event]:
    """A stop token for a long operation, set on any abnormal exit.

    An agent cancelling a tool call gets what the HTTP surface gets on a client
    disconnect: the work stops at its next boundary and keeps what it finished,
    rather than running the rest of a corpus, download or wiki build for a
    request that has gone away. The work runs on a thread asyncio cannot
    interrupt, which is why cancellation alone does not reach it and the token
    has to be polled.

    Set on every exception, not only cancellation: a pass that raised is over
    either way, and a caller vanishing does not always surface as a cancel.
    """
    token = threading.Event()
    try:
        yield token
    except BaseException:
        token.set()
        raise


@_tool
async def sync(force_rebuild: bool = False, retry_skipped: bool = False) -> dict[str, Any]:
    """Sync the documents directory into the vector store.

    ``force_rebuild`` drops every table and re-ingests. ``retry_skipped``
    clears failed-file skip markers without dropping the store.
    """
    from lilbee.data.ingest import sync as run_sync

    with _cancel_token() as cancel:
        return (
            await run_sync(
                quiet=True,
                force_rebuild=force_rebuild,
                retry_skipped=retry_skipped,
                cancel=cancel,
            )
        ).model_dump()


@_tool
async def add(
    paths: list[str],
    force: bool = False,
    enable_ocr: bool | None = None,
    ocr_timeout: float | None = None,
    render_mode: CrawlRenderMode | None = None,
) -> dict[str, Any]:
    """Add files, directories, or URLs to the knowledge base, then sync.
    Paths must be absolute. URLs are fetched as single pages; use ``crawl`` for sites."""
    from lilbee.app.ingest import register_sources
    from lilbee.data.ingest import sync as run_sync

    errors: list[str] = []
    valid: list[Path] = []
    urls: list[str] = []
    for p_str in paths:
        if is_url(p_str):
            urls.append(p_str)
        else:
            p = Path(p_str)
            if not p.exists():
                errors.append(p_str)
            else:
                valid.append(p)

    # Crawl URLs
    crawled_count = 0
    if urls:
        from lilbee.crawler import crawler_available

        if not crawler_available():
            return _error("Web crawling requires: pip install 'lilbee[crawler]'")
        from lilbee.crawler import crawl_and_save

        for url in urls:
            try:
                # URL validation resolves the host (blocking DNS); run it off the
                # event loop like the sibling crawl tool does.
                await anyio.to_thread.run_sync(require_valid_crawl_url, url)
            except ValueError as exc:
                errors.append(f"{url}: {exc}")
                continue
            # add fetches single pages (depth=0); site crawls go through the crawl tool
            crawled_paths = await crawl_and_save(url, depth=0, render_mode=render_mode)
            crawled_count += len(crawled_paths)

    # Registration touches config.toml (a locked read-modify-write); keep the
    # blocking disk I/O off the event loop.
    reg_result = await anyio.to_thread.run_sync(
        functools.partial(register_sources, valid, force=force)
    )

    from lilbee.app.ingest import temporary_ocr_config

    with temporary_ocr_config(enable_ocr, ocr_timeout), _cancel_token() as cancel:
        sync_result = (await run_sync(quiet=True, cancel=cancel)).model_dump()

    result: dict[str, Any] = {
        "command": "add",
        "copied": reg_result.registered,
        "skipped": reg_result.skipped,
        "tracked": reg_result.tracked,
        "crawled": crawled_count,
        "errors": errors,
        "sync": sync_result,
    }
    if errors or sync_result.get("failed"):
        result["warning"] = "some files could not be processed"
    return result


@_tool_if(crawler_available)
async def crawl(
    url: str,
    depth: int | None = 0,
    max_pages: int | None = None,
    render_mode: CrawlRenderMode | None = None,
    include_subdomains: bool = False,
) -> dict[str, Any]:
    """Start a non-blocking crawl; poll via ``crawl_status(task_id)``.
    ``depth=0`` (default) = single URL, ``N`` = follow links N levels,
    ``null`` = whole site. ``render_mode``: "http"/"browser"."""
    from lilbee.crawler import crawler_available

    if not crawler_available():
        return _error("Web crawling requires: pip install 'lilbee[crawler]'")
    # Mirror the REST CrawlRequest bounds so a negative value is a clean error,
    # not an unbounded crawl.
    if depth is not None and depth < 0:
        return _error("depth must be 0 or greater (pass depth=null to crawl the whole site)")
    if max_pages is not None and max_pages < 0:
        return _error("max_pages must be 0 or greater (0 = unlimited, omit for the safety cap)")
    try:
        # URL validation resolves the host (blocking DNS), so it runs off the loop.
        # The crawl itself must be scheduled ON the loop: start_crawl uses
        # asyncio.create_task, which requires a running event loop.
        await anyio.to_thread.run_sync(require_valid_crawl_url, url)
    except ValueError as exc:
        return _error(str(exc))

    task_id = start_crawl(
        url,
        depth=depth,
        max_pages=max_pages,
        render_mode=render_mode,
        include_subdomains=include_subdomains,
    )
    return {"status": "started", "task_id": task_id, "url": url}


@_tool_if(crawler_available)
def crawl_status(task_id: str) -> dict[str, Any]:
    """Poll a crawl task by id; returns ``{status, pages, error}``."""
    task = get_task(task_id)
    if task is None:
        return _error(f"No task found with id: {task_id}")
    return {
        "task_id": task.task_id,
        "url": task.url,
        "status": task.status.value,
        "pages_crawled": task.pages_crawled,
        "pages_total": task.pages_total,
        "pages_failed": task.pages_failed,
        "failure_reasons": task.failure_reasons,
        "error": task.error,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
    }


@_tool
def crawl_cancel(task_id: str) -> dict[str, Any]:
    """Stop a running crawl started by ``crawl``. Pages already saved are kept."""
    from lilbee.crawler.task import cancel_crawl

    if get_task(task_id) is None:
        return _error(f"No task found with id: {task_id}")
    return {"command": "crawl_cancel", "task_id": task_id, "cancelling": cancel_crawl(task_id)}


@_tool
def init(path: str = "") -> dict[str, Any]:
    """Initialize a local ``.lilbee/`` knowledge base; empty path = cwd.

    Switches the MCP session to use it for subsequent calls.
    """
    if _transport.http_mounted:
        return _error(
            "init is unavailable on the HTTP server: it is bound to one vault and "
            "shared by every connected client. Start a separate server for another vault."
        )
    # Canonical so this vault keys the same lock paths a CLI or server process
    # would derive for the same directory.
    base = canonical_data_root(path) if path else canonical_data_root(Path.cwd())
    root = base / LOCAL_ROOT_DIRNAME

    created = False
    if not root.is_dir():
        (root / "documents").mkdir(parents=True)
        (root / "data").mkdir(parents=True)
        (root / ".gitignore").write_text("data/\n", encoding="utf-8")
        created = True

    # Switch MCP session to this project's KB. Overlay any persisted
    # config.toml so per-vault model / generation settings take effect,
    # matching the CLI's --data-dir behaviour. Env export mirrors
    # cli/app.py::_apply_data_root for worker-log parity.
    cfg.data_root = base
    cfg.documents_dir = root / "documents"
    cfg.data_dir = root / "data"
    cfg.lancedb_dir = root / "data" / "lancedb"
    os.environ["LILBEE_DATA"] = str(base)
    overlay_persisted_settings(base)
    reset_services()

    return {"command": "init", "path": str(root), "created": created}


@_tool
def remove(names: list[str]) -> dict[str, Any]:
    """Remove documents from the index by source name, folder, or glob pattern.

    Source files are never deleted. A folder name removes every document indexed
    beneath it; a glob (``*``/``?``/``[]``) removes every matching source."""
    from lilbee.app.ingest import remove_documents_durably

    result = remove_documents_durably(names)
    return {"command": "remove", "removed": result.removed, "not_found": result.not_found}


@_tool
def list_documents() -> dict[str, Any]:
    """List all indexed documents with their chunk counts."""
    sources = get_services().store.get_sources()
    return {
        "documents": [
            {"filename": s["filename"], "chunk_count": s.get("chunk_count", 0)} for s in sources
        ],
        "total": len(sources),
    }


_AGENT_ORIGINS = frozenset({SessionOrigin.MCP})


def _require_agent_session(session_id: str) -> Session:
    """Human conversations are private: foreign ids answer not-found, so an
    agent cannot even probe which ones exist."""
    session = get_services().session_store.get(session_id)
    if session.meta.origin is not SessionOrigin.MCP:
        raise SessionNotFoundError(session_id)
    return session


@_tool_if(agent_sessions_enabled)
def sessions_list() -> dict[str, Any]:
    """List the agent's sessions, newest first."""
    if not agent_sessions_enabled():
        return _error(AGENT_SESSIONS_DISABLED_HINT)
    metas = get_services().session_store.list(origins=_AGENT_ORIGINS)
    return {"sessions": [asdict(meta) for meta in metas], "total": len(metas)}


@_tool_if(agent_sessions_enabled)
def session_get(session_id: str) -> dict[str, Any]:
    """Return one agent session: metadata, transcript, summary."""
    if not agent_sessions_enabled():
        return _error(AGENT_SESSIONS_DISABLED_HINT)
    try:
        session = _require_agent_session(session_id)
    except SessionNotFoundError as exc:
        return _error(str(exc))
    return {
        "meta": asdict(session.meta),
        "messages": [
            {
                "role": message.role.value,
                "content": message.content,
                "sources": list(message.sources),
                "ts": message.ts,
            }
            for message in session.messages
        ],
        # What compaction folded the older turns into (empty if never compacted).
        # An agent that resumes and continues the conversation needs it, or it
        # rebuilds history without what was already condensed.
        "summary": session.summary,
    }


@_tool_if(agent_sessions_enabled)
def session_create(model_ref: str, scope: str = "both") -> dict[str, Any]:
    """Start a saved chat session; returns its id."""
    if not agent_sessions_enabled():
        return _error(AGENT_SESSIONS_DISABLED_HINT)
    session_id = get_services().session_store.create(
        model_ref=model_ref, scope=scope, origin=SessionOrigin.MCP
    )
    return {"id": session_id, "model_ref": model_ref, "scope": scope}


@_tool_if(agent_sessions_enabled)
def session_add_message(
    session_id: str,
    role: MessageRole,
    content: str,
    sources: list[str] | None = None,
    claim: bool = False,
) -> dict[str, Any]:
    """Append one turn; a foreign session errors unless claim=True (ask first)."""
    if not agent_sessions_enabled():
        return _error(AGENT_SESSIONS_DISABLED_HINT)
    store = get_services().session_store
    try:
        # Re-coerce: the MCP layer passes the enum, but a raw string still
        # arrives via direct library calls, and a bad one must error cleanly.
        message = SessionMessage(
            role=MessageRole(role), content=content, sources=tuple(sources or ())
        )
        if claim:
            store.transfer(session_id, SessionOrigin.MCP)
        store.add_message(session_id, message, surface=SessionOrigin.MCP)
    except (SessionNotFoundError, SessionOwnershipError) as exc:
        return _error(str(exc))
    except ValueError as exc:
        return _error(f"invalid role {role!r}: {exc}")
    return {"id": session_id, "added": True}


@_tool_if(agent_sessions_enabled)
def session_set_summary(session_id: str, summary: str) -> dict[str, Any]:
    """Replace an agent session's compaction summary."""
    if not agent_sessions_enabled():
        return _error(AGENT_SESSIONS_DISABLED_HINT)
    try:
        _require_agent_session(session_id)
        get_services().session_store.set_summary(session_id, summary)
    except SessionNotFoundError as exc:
        return _error(str(exc))
    return {"id": session_id, "summary": summary}


@_tool_if(agent_sessions_enabled)
def session_rename(session_id: str, title: str) -> dict[str, Any]:
    """Rename an agent session."""
    if not agent_sessions_enabled():
        return _error(AGENT_SESSIONS_DISABLED_HINT)
    try:
        _require_agent_session(session_id)
        get_services().session_store.set_title(session_id, title, TitleSource.CUSTOM)
    except SessionNotFoundError as exc:
        return _error(str(exc))
    return {"id": session_id, "title": title}


@_tool_if(agent_sessions_enabled)
def session_delete(session_id: str) -> dict[str, Any]:
    """Delete an agent session."""
    if not agent_sessions_enabled():
        return _error(AGENT_SESSIONS_DISABLED_HINT)
    try:
        _require_agent_session(session_id)
        get_services().session_store.delete(session_id)
    except SessionNotFoundError as exc:
        return _error(str(exc))
    return {"id": session_id, "deleted": True}


@_tool
def export_dataset(output: str, fmt: str = "", source: str = "") -> dict[str, Any]:
    """Write the per-page {source, page, text} dataset to a file (no vectors).

    ``fmt`` is parquet/jsonl (empty infers from the suffix); ``source`` limits to one file.
    """
    from lilbee.app.dataset import DatasetError, export_to_path

    try:
        summary = export_to_path(Path(output), fmt, source or None, cancel=_caller_cancelled())
    except DatasetError as exc:
        return _error(str(exc))
    return summary.model_dump()


@_tool
async def import_dataset(dataset: str, fmt: str = "", ctx: Context | None = None) -> dict[str, Any]:
    """Import a per-page text dataset, re-embedding under the current model.

    Replaces existing copies; imported sources are detached so sync won't delete them.
    """
    from lilbee.app.dataset import DatasetError, import_from_path
    from lilbee.runtime.progress import EmbedEvent, EventType, ProgressEvent

    loop = asyncio.get_running_loop()

    with _cancel_token() as cancel:

        def on_progress(event_type: EventType, data: ProgressEvent) -> None:
            # Raise rather than return: the embed work runs off the loop, so a
            # quiet return would re-embed the whole dataset for a caller that
            # has gone. Checked before the isinstance filter so the stop lands
            # on every event, not only the ones that map to a percent.
            if cancel.is_set():
                raise TaskCancelledError
            # EMBED events carry chunk/total_chunks; other event types don't map to a percent.
            if ctx is None or not isinstance(data, EmbedEvent):
                return
            future = asyncio.run_coroutine_threadsafe(
                ctx.report_progress(
                    progress=float(data.chunk), total=float(data.total_chunks), message=data.file
                ),
                loop,
            )
            future.add_done_callback(_log_progress_failure)

        try:
            summary = await import_from_path(Path(dataset), fmt, on_progress=on_progress)
        except DatasetError as exc:
            return _error(str(exc))
    return summary.model_dump()


@_tool
def reset(confirm: bool = False) -> dict[str, Any]:
    """Factory reset: delete all documents and indexed data. Requires ``confirm=true``."""
    if _transport.http_mounted:
        return _error(
            "reset is unavailable on the HTTP server: it would wipe the shared index for "
            "every connected client. Run it from the CLI or the stdio MCP server."
        )
    if not confirm:
        return _error("pass confirm=true to confirm deletion")
    from lilbee.app.reset import perform_reset

    result = perform_reset().model_dump()
    # Reopen LanceDB against the empty data dir; keep providers loaded.
    reset_store()
    return result


@_wiki_tool
def wiki_lint(wiki_source: str = "") -> dict[str, Any]:
    """Lint wiki pages; empty ``wiki_source`` lints all."""
    from lilbee.wiki.lint import LintReport, lint_all, lint_wiki_page

    store = get_services().store
    report = (
        LintReport(issues=lint_wiki_page(wiki_source, store))
        if wiki_source
        else lint_all(store, cancel=_caller_cancelled())
    )
    return {
        "command": "wiki_lint",
        "issues": [i.to_dict() for i in report.issues],
        "total": len(report.issues),
        "errors": report.error_count,
        "warnings": report.warning_count,
    }


@_wiki_tool
def wiki_citations(wiki_source: str = "", source: str = "") -> dict[str, Any]:
    """List a wiki page's citations, or with ``source``, the pages citing that document.

    Pass exactly one of ``wiki_source`` (forward) or ``source`` (reverse).
    """
    if bool(wiki_source) == bool(source):
        return _error("pass either wiki_source or source, not both")
    store = get_services().store
    if source:
        records = store.get_citations_for_source(source)
        return {
            "command": "wiki_citations",
            "source": source,
            "citations": [dict(r) for r in records],
            "total": len(records),
        }
    records = store.get_citations_for_wiki(wiki_source)
    return {
        "command": "wiki_citations",
        "wiki_source": wiki_source,
        "citations": [dict(r) for r in records],
        "total": len(records),
    }


@_tool
def wiki_status() -> dict[str, Any]:
    """Show wiki layer status: page counts, recent lint issues.

    Registered even when the wiki is disabled, like the HTTP status route, so a
    caller can read the disabled state instead of finding no tool at all.
    """
    from lilbee.wiki.lint import lint_all

    wiki_root = cfg.data_root / cfg.wiki_dir
    if not cfg.wiki or not wiki_root.exists():
        # Same keys as the enabled arm so a client can key on lint_errors in
        # either state; a disabled wiki reports zeros rather than being linted.
        return {
            "wiki_enabled": cfg.wiki,
            WikiSubdir.SUMMARIES: 0,
            WikiSubdir.DRAFTS: 0,
            "pages": 0,
            "lint_errors": 0,
            "lint_warnings": 0,
        }

    summaries_dir = wiki_root / WikiSubdir.SUMMARIES
    drafts_dir = wiki_root / WikiSubdir.DRAFTS
    summaries = list(summaries_dir.rglob("*.md")) if summaries_dir.exists() else []
    drafts = list(drafts_dir.rglob("*.md")) if drafts_dir.exists() else []

    # Read-only status: lint for counts without appending to the audit log.
    report = lint_all(get_services().store, record_log=False, cancel=_caller_cancelled())
    return {
        "wiki_enabled": cfg.wiki,
        WikiSubdir.SUMMARIES: len(summaries),
        WikiSubdir.DRAFTS: len(drafts),
        "pages": total_wiki_pages(wiki_root),
        "lint_errors": report.error_count,
        "lint_warnings": report.warning_count,
    }


@_wiki_tool
def wiki_list() -> dict[str, Any]:
    """List wiki pages with metadata."""
    from dataclasses import asdict

    from lilbee.wiki.browse import list_pages

    wiki_root = cfg.data_root / cfg.wiki_dir
    pages = list_pages(wiki_root)
    return {
        "command": "wiki_list",
        "pages": [asdict(p) for p in pages],
        "total": len(pages),
    }


@_wiki_tool
def wiki_read(slug: str) -> dict[str, Any]:
    """Read a wiki page's content + frontmatter by slug."""
    from dataclasses import asdict

    from lilbee.wiki.browse import read_page

    wiki_root = cfg.data_root / cfg.wiki_dir
    result = read_page(wiki_root, slug)
    if result is None:
        return _error(f"wiki page not found: {slug}")
    return {"command": "wiki_read", **asdict(result)}


@_wiki_tool
def wiki_build(dry_run: bool = False) -> dict[str, Any]:
    """Build the concept and entity wiki across all ingested sources. Blocks until done.

    ``dry_run=True`` returns the NER entity candidates a build would cover and
    makes no LLM call.
    """
    from lilbee.wiki import run_full_build
    from lilbee.wiki.generation import DRY_RUN_CONCEPT_NOTE, preview_build_entities

    if dry_run:
        rows = preview_build_entities(cfg)
        return {
            "command": "wiki_build",
            "dry_run": True,
            "entities": rows,
            "count": len(rows),
            "note": DRY_RUN_CONCEPT_NOTE,
        }
    return {"command": "wiki_build", **run_full_build(cfg, cancel=_caller_cancelled())}


@_wiki_tool
def wiki_update() -> dict[str, Any]:
    """Refresh the concept and entity wiki after an ingest. A full rebuild; blocks until done."""
    from lilbee.wiki import run_full_build

    return {"command": "wiki_update", **run_full_build(cfg, cancel=_caller_cancelled())}


@_wiki_tool
def wiki_synthesize() -> dict[str, Any]:
    """Generate synthesis pages for concept clusters with three or more sources."""
    from lilbee.wiki import run_full_synthesize

    return {"command": "wiki_synthesize", **run_full_synthesize(cfg, cancel=_caller_cancelled())}


@_wiki_tool
def wiki_prune() -> dict[str, Any]:
    """Prune stale and orphaned wiki pages."""
    from lilbee.wiki.prune import prune_wiki

    report = prune_wiki(get_services().store, cancel=_caller_cancelled())
    return {
        "command": "wiki_prune",
        "records": [r.to_dict() for r in report.records],
        "archived": report.archived_count,
        "flagged": report.flagged_count,
        "reconciled": report.reconciled_count,
    }


@_wiki_tool
def wiki_index() -> dict[str, Any]:
    """Rebuild the browse index of pages the corpus could have. No LLM call."""
    from lilbee.wiki.stubs import refresh_stub_index

    stubs = refresh_stub_index(get_services().store)
    return {"command": "wiki_index", "entries": len(stubs)}


@_wiki_tool
def wiki_generate(slug: str) -> dict[str, Any]:
    """Generate one indexed wiki page. Costs a single LLM call and is GPU-heavy."""
    from lilbee.wiki.browse import page_slug
    from lilbee.wiki.lazy import UnknownStubError, generate_stub_page

    try:
        path = generate_stub_page(slug, get_services().store)
    except UnknownStubError as exc:
        return _error(str(exc))
    if path is None:
        return _error(f"index entry for {slug} is stale; its sources are gone")
    # The read surfaces address pages by section, so answer with that slug.
    read_slug = page_slug(path, cfg.data_root / cfg.wiki_dir)
    return {"command": "wiki_generate", "slug": read_slug, "path": path.as_posix()}


@_tool
def wiki_wipe(confirm: bool = False) -> dict[str, Any]:
    """Delete every generated wiki page and its indexed rows. Pass ``confirm=true``.

    Registered even with the wiki disabled, because turning the setting off
    leaves the pages generated earlier in place.
    """
    if not confirm:
        return _error("pass confirm=true to delete the wiki; this cannot be undone")
    from lilbee.wiki.wipe import wipe_wiki

    report = wipe_wiki(get_services().store)
    if not report.rows_deleted:
        return _error(report.summary())
    return {
        "command": "wiki_wipe",
        "pages_removed": report.pages_removed,
        "sources_cleared": report.sources_cleared,
    }


def _setting_info_to_dict(info: SettingInfo) -> dict[str, Any]:
    """Render a SettingInfo as a JSON-safe dict for the MCP wire format."""
    return {
        "key": info.key,
        "value": _json_safe(info.value),
        "default": _json_safe(info.default),
        "type": info.type,
        "nullable": info.nullable,
        "group": info.group.value,
        "help": info.help_text,
        "choices": list(info.choices) if info.choices else None,
        "reindex_required": info.reindex_required,
    }


def _json_safe(value: Any) -> Any:
    """Coerce Path / frozenset / tuple to JSON-friendly primitives."""
    if isinstance(value, str | int | float | bool | list | type(None)):
        return value
    return str(value)


@_tool
def settings_list(group: str = "") -> dict[str, Any]:
    """List writable lilbee settings (each with value, default, type, help, choices).

    ``group`` filters by group name (case-insensitive); empty returns all.
    """

    try:
        infos = list_settings(group or None)
    except ValueError as exc:
        return _error(str(exc))
    return {
        "command": "settings_list",
        "settings": [_setting_info_to_dict(info) for info in infos],
        "total": len(infos),
    }


@_tool
def settings_get(key: str) -> dict[str, Any]:
    """Get a single setting's current value + metadata."""

    try:
        info = get_setting(key)
    except KeyError as exc:
        return _error(str(exc))
    return {"command": "settings_get", "setting": _setting_info_to_dict(info)}


@_tool
def settings_set(updates: dict[str, Any]) -> dict[str, Any]:
    """Atomically update writable settings; rolls back on validation error.
    Persists to config.toml; returns ``{updated, reindex_required}``."""
    if _transport.http_mounted and requires_services_reset(updates):
        return _error(provider_reset_refused_message("Switching"))
    try:
        result = apply_settings_update(updates)
    except (ValueError, TypeError) as exc:
        return _error(str(exc))
    return {
        "command": "settings_set",
        "updated": result.updated,
        "reindex_required": result.reindex_required,
    }


@_tool
def settings_reset(keys: list[str]) -> dict[str, Any]:
    """Reset writable settings to their built-in defaults."""
    if _transport.http_mounted and requires_services_reset(dict.fromkeys(keys)):
        return _error(provider_reset_refused_message("Resetting"))
    try:
        result = reset_settings(keys)
    except (ValueError, TypeError) as exc:
        return _error(str(exc))
    return {
        "command": "settings_reset",
        "updated": result.updated,
        "reindex_required": result.reindex_required,
    }


@_tool
def model_list(source: str = "", task: str = "") -> dict[str, Any]:
    """List installed models. ``source`` is ``native`` / ``remote``; ``task`` filters by role."""
    from lilbee.app.models import list_models_data
    from lilbee.catalog.types import ModelTask

    try:
        src = ModelSource.parse(source)
    except ValueError as exc:
        return _error(str(exc))
    try:
        parsed_task = ModelTask(task) if task else None
    except ValueError as exc:
        return _error(str(exc))
    return list_models_data(source=src, task=parsed_task).model_dump()


@_tool
def catalog_browse(
    task: str = "",
    search: str = "",
    size: str = "",
    installed: bool | None = None,
    featured: bool | None = None,
    sort: str = "featured",
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Browse the lilbee model catalog. ``task``: chat/embedding/vision/rerank.
    ``size``: small/medium/large/huge, by parameter count.
    ``sort``: featured/downloads/name/size_asc/size_desc."""
    from lilbee.catalog.query import get_catalog
    from lilbee.catalog.types import CatalogSize, CatalogSort, ModelTask

    try:
        parsed_task = ModelTask(task) if task else None
        parsed_size = CatalogSize(size) if size else None
        parsed_sort = CatalogSort(sort)
    except ValueError as exc:
        return _error(str(exc))
    try:
        result = get_catalog(
            task=parsed_task,
            search=search,
            size=parsed_size,
            installed=installed,
            featured=featured,
            sort=parsed_sort,
            limit=limit,
            offset=offset,
            model_manager=get_services().model_manager,
        )
    except ValueError as exc:
        return _error(str(exc))
    return {
        "command": "catalog_browse",
        "total": result.total,
        "limit": result.limit,
        "offset": result.offset,
        "has_more": result.has_more,
        "models": [
            {
                "ref": m.hf_repo,
                "display_name": m.display_name,
                "task": m.task.value,
                "size_gb": m.size_gb,
                "min_ram_gb": m.min_ram_gb,
                "downloads": m.downloads,
                "featured": m.featured,
                "description": m.description,
                "architecture": m.architecture,
                "compat": m.compat.value,
            }
            for m in result.models
        ],
    }


@_tool
def model_show(model: str) -> dict[str, Any]:
    """Show catalog and installed metadata for a model ref."""
    from lilbee.app.models import show_model_data
    from lilbee.modelhub.model_manager import ModelNotFoundError

    try:
        return show_model_data(model).model_dump()
    except ModelNotFoundError as exc:
        return _error(str(exc))


def _log_progress_failure(future: concurrent.futures.Future[None]) -> None:
    """Log report_progress failures without raising.

    Progress notifications are best-effort: a failure should not abort
    an in-flight pull.
    """
    try:
        future.result()
    except Exception:
        log.warning("MCP report_progress failed", exc_info=True)


@_tool
async def model_pull(
    model: str,
    source: str = ModelSource.NATIVE.value,
    allow_unsupported: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Download a model and stream progress.

    ``source`` is ``native`` (GGUF) or ``remote`` (SDK).
    ``allow_unsupported`` overrides the supported-architecture refusal.
    """
    from lilbee.app.models import pull_model_data
    from lilbee.catalog import DownloadProgress
    from lilbee.catalog.compat import SUPPORTED_ARCHS, UnsupportedArchError

    try:
        src = ModelSource.parse(source) or ModelSource.NATIVE
    except ValueError as exc:
        return _error(str(exc))

    loop = asyncio.get_running_loop()

    with _cancel_token() as cancel:

        def on_update(p: DownloadProgress) -> None:
            # Raise rather than return: the download runs on a thread asyncio
            # cannot interrupt, so returning would leave a multi-GB pull going
            # for a caller that has gone. Same idiom the HTTP pull uses.
            if cancel.is_set():
                raise TaskCancelledError
            if ctx is None:
                return
            future = asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=float(p.percent), total=100.0, message=p.detail),
                loop,
            )
            future.add_done_callback(_log_progress_failure)

        try:
            result = await asyncio.to_thread(
                pull_model_data,
                model,
                src,
                on_update=on_update,
                allow_unsupported=allow_unsupported,
                cancel=cancel,
            )
        except UnsupportedArchError as exc:
            return {
                "ok": False,
                "command": "model_pull",
                "error": {
                    "code": "unsupported_arch",
                    "arch": exc.architecture,
                    "ref": exc.ref,
                    "supported_examples": sorted(SUPPORTED_ARCHS)[:5],
                    "total_supported": len(SUPPORTED_ARCHS),
                },
            }
        except (RuntimeError, PermissionError) as exc:
            return _error(str(exc))
    return result.model_dump()


@_tool
def model_rm(model: str, source: str = "") -> dict[str, Any]:
    """Remove an installed model. Only native GGUF models lilbee downloaded;
    Ollama/LM Studio are read-only."""
    from lilbee.app.models import remove_model_data

    try:
        src = ModelSource.parse(source)
        return remove_model_data(model, source=src).model_dump()
    except ValueError as exc:
        return _error(str(exc))


@_wiki_tool
def wiki_drafts_list() -> dict[str, Any]:
    """List pending wiki drafts. Read-only: promotion is reserved for the human
    surfaces (CLI, TUI, and the authenticated HTTP API)."""
    from lilbee.wiki.drafts import list_drafts

    wiki_root = cfg.data_root / cfg.wiki_dir
    drafts = list_drafts(wiki_root)
    return {
        "command": "wiki_drafts_list",
        "drafts": [d.to_dict() for d in drafts],
        "total": len(drafts),
    }


@_wiki_tool
def wiki_drafts_diff(slug: str) -> dict[str, Any]:
    """Unified diff of a draft against its published counterpart."""
    from lilbee.core.security import PathTraversalError
    from lilbee.wiki.drafts import diff_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        diff = diff_draft(slug, wiki_root)
    except FileNotFoundError as exc:
        return _error(str(exc))
    except PathTraversalError:
        return _error(INVALID_DRAFT_SLUG_ERROR)
    return {"command": "wiki_drafts_diff", "slug": slug, "diff": diff}


def _collapse_nullable_anyof(prop: dict[str, Any]) -> None:
    """Collapse ``anyOf: [{type: X}, {type: null}]`` to ``{type: X}`` in place.

    Pydantic emits ``T | None`` parameters as a two-arm anyOf with a null
    branch. The null branch carries no information the model needs to pick
    or shape its call, but it costs tokens at every dispatch. Drop it.
    """
    arms = prop.get("anyOf")
    if not isinstance(arms, list):
        return
    non_null = [a for a in arms if isinstance(a, dict) and a.get("type") != "null"]
    if len(non_null) == 1 and len(non_null) < len(arms):
        prop.pop("anyOf", None)
        for key, value in non_null[0].items():
            prop.setdefault(key, value)


def _strip_property_noise(prop: dict[str, Any]) -> None:
    """Drop tokens that don't change the model's behavior."""
    prop.pop("title", None)
    prop.pop("default", None)
    _collapse_nullable_anyof(prop)
    if prop.get("additionalProperties") is True:
        prop.pop("additionalProperties", None)


def _flatten_tool_description(text: str) -> str:
    """Flatten a triple-quoted tool docstring for the tools wire.

    The summary line carries no indent while continuation lines are indented to
    the source, so ``textwrap.dedent`` alone is a no-op (the common prefix is the
    empty string) and leaves source indentation on every body line -- including
    deeper-indented Args lines. Strip each line so the model sees flat text;
    blank lines are kept so paragraph breaks survive.
    """
    return "\n".join(line.strip() for line in text.strip().splitlines())


def _strip_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Trim auto-generated noise from a tool's input schema, on a copy.

    Drops:
    - SDK/Pydantic ``title`` keys (per-schema + per-property). Tools the
      model picks by name don't need a separate display title.
    - ``default`` values on properties: clients send what they want and
      omitted fields fall back server-side.
    - ``additionalProperties: true`` on dict params: Pydantic emits it for
      every ``dict[str, Any]`` but it's the JSON Schema default behavior.
    - The ``null`` arm of ``anyOf: [{type: X}, {type: null}]`` unions for
      ``T | None`` defaults; the null branch is implicit.

    A roughly 25-35% reduction in the serialized tools payload, which matters
    most for small-context (16K) chat models where the tools surface was
    previously eating ~60% of the budget.
    """
    schema = deepcopy(schema)
    schema.pop("title", None)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for prop in properties.values():
            if isinstance(prop, dict):
                _strip_property_noise(prop)
    return schema


_NO_WIKI_SCOPE_HINT = ' No wiki layer here: use scope "raw" or "both".'


class LilbeeMCP(MCPServer):
    """MCP server that trims its tools wire and keeps it current with config."""

    async def list_tools(self) -> list[MCPTool]:
        """The registered tools with schema noise stripped and flat descriptions.

        The transforms run on the wire representation per request, never on the
        stored registrations, so they cannot drift out of sync with config. The
        ``search`` description advertises only the scopes this corpus has: when
        wiki generation is off, a model that guesses ``scope="wiki"`` gets a
        silent fallback to the full pool, so raw/both only.
        """
        tools = await super().list_tools()
        for tool in tools:
            tool.input_schema = _strip_schema(tool.input_schema)
            if isinstance(tool.description, str):
                description = _flatten_tool_description(tool.description)
                if tool.name == "search" and not cfg.wiki:
                    description += _NO_WIKI_SCOPE_HINT
                tool.description = description
        return tools


def _client_name(ctx: Context | None) -> str:
    """The MCP client's self-reported name from the initialize handshake, or empty."""
    if ctx is None:
        return ""
    params = ctx.session.client_params
    return params.clientInfo.name if params is not None else ""


def _slug(value: str) -> str:
    """Lowercase, hyphenated id fragment; falls back to ``generic`` when empty."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "generic"


# Per-connection fallback ids for agents that report no identity. Keyed by the
# live MCP session so each connection gets a distinct, stable namespace instead
# of every unidentified agent colliding on a shared one. WeakKeyDictionary drops
# entries once the session is collected, so this does not grow unbounded. The
# lock guards the get-or-create because sync tool handlers run on the offload
# threadpool, so concurrent connections (and weakref-removal callbacks) can
# touch the mapping from different threads.
_ANON_OWNER_IDS: WeakKeyDictionary[object, str] = WeakKeyDictionary()
_ANON_OWNER_LOCK = threading.Lock()


def _anon_owner_id(ctx: Context | None) -> str:
    """A stable per-connection id for an agent that reported no identity.

    Without this, two unidentified agents would both slug to ``generic`` and
    share a memory namespace; keying on the session keeps them isolated.
    """
    if ctx is None:
        return "anonymous"
    session = ctx.session
    with _ANON_OWNER_LOCK:
        existing = _ANON_OWNER_IDS.get(session)
        if existing is None:
            existing = f"anon-{uuid.uuid4().hex[:12]}"
            _ANON_OWNER_IDS[session] = existing
        return existing


def _derive_owner(agent_id: str, ctx: Context | None) -> str:
    """Resolve the calling agent's stable owner namespace.

    Precedence: explicit ``agent_id`` argument, then the ``LILBEE_AGENT_ID`` env var
    (pinned in the client's MCP config), then the MCP client name, then a stable
    per-connection fallback so unidentified agents never share a namespace.
    """
    explicit = agent_id or os.environ.get("LILBEE_AGENT_ID", "")
    resolved = explicit or _client_name(ctx)
    if resolved:
        return agent_owner(_slug(resolved))
    return agent_owner(_slug(_anon_owner_id(ctx)))


@_tool_if(memory_enabled)
def memory_remember(
    text: str,
    kind: MemoryKind = MemoryKind.FACT,
    shared: bool = False,
    agent_id: str = "",
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Store a durable memory. ``kind``: "fact" (similarity-recalled) or "preference" (always on).
    ``shared`` exposes it to the human's TUI/CLI."""
    if not memory_enabled():
        return _error(MEMORY_DISABLED_HINT)
    owner = _derive_owner(agent_id, ctx)
    memory_id = remember(text, owner=owner, kind=kind, source=MemorySource.AGENT, shared=shared)
    return {"ok": True, "id": memory_id, "owner": owner}


@_tool_if(memory_enabled)
def memory_recall(
    query: str, limit: int = 0, agent_id: str = "", ctx: Context | None = None
) -> dict[str, Any]:
    """Recall this agent's memories (plus any the human shared) relevant to *query*."""
    if not memory_enabled():
        return _error(MEMORY_DISABLED_HINT)
    owner = _derive_owner(agent_id, ctx)
    memories = recall(query, owner, top_k=limit if limit > 0 else None)
    return {
        "memories": [
            {"id": m.id, "text": m.text, "kind": m.kind.value, "owner": m.owner} for m in memories
        ]
    }


@_tool_if(memory_enabled)
def memory_list(agent_id: str = "", ctx: Context | None = None) -> dict[str, Any]:
    """List every memory in this agent's namespace (any kind, newest first)."""
    if not memory_enabled():
        return _error(MEMORY_DISABLED_HINT)
    owner = _derive_owner(agent_id, ctx)
    memories = list_memories(owner)
    return {
        "memories": [
            {"id": m.id, "text": m.text, "kind": m.kind.value, "shared": m.shared} for m in memories
        ]
    }


@_tool_if(memory_enabled)
def memory_forget(memory_id: str, agent_id: str = "", ctx: Context | None = None) -> dict[str, Any]:
    """Delete one of this agent's own memories by id (agent_id scopes the namespace)."""
    if not memory_enabled():
        return _error(MEMORY_DISABLED_HINT)
    owner = _derive_owner(agent_id, ctx)
    if not forget(memory_id, owner=owner):
        return _error(f"No memory '{memory_id}' in this agent's namespace.")
    return {"ok": True, "id": memory_id}


def _placement_dict(view: PlacementView) -> dict[str, Any]:
    from lilbee.server.models import PlacementResponse

    return PlacementResponse.from_view(view).model_dump(mode="json")


def _placement_guard(serialize: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run a placement query and serialize it, returning a structured error on failure."""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.fleet.placement_spec import PlacementError

    try:
        return serialize()
    except (PlacementError, ProviderError) as exc:
        return _error(str(exc))


def _placement_result(action: Callable[[], PlacementView]) -> dict[str, Any]:
    """Run a placement action and serialize its view, returning a structured error on failure."""
    return _placement_guard(lambda: _placement_dict(action()))


def _parse_spec(spec: dict[str, Any] | None) -> PlacementSpec | None:
    from lilbee.providers.fleet.placement_spec import PlacementSpec

    return PlacementSpec.from_json(json.dumps(spec)) if spec else None


@_tool_named("get_gpus")
def get_gpus_tool() -> dict[str, Any]:
    """List detected GPUs with free/total VRAM (the placement HTTP /api/gpus equivalent)."""

    def _body() -> dict[str, Any]:
        from lilbee.cli.tui import messages as msg
        from lilbee.providers.fleet.gpu_stats import probe_intel_util_hint

        view = get_placement()
        hint = probe_intel_util_hint(view.gpus)
        return {
            "gpus": _placement_dict(view)["gpus"],
            "notice": msg.intel_util_hint_text(hint) if hint else None,
        }

    return _placement_guard(_body)


@_tool_named("get_placement")
def get_placement_tool() -> dict[str, Any]:
    """Show the current effective multi-GPU model placement."""
    return _placement_result(get_placement)


@_tool_named("preview_placement")
def preview_placement_tool(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Preview what a placement spec (or auto, when omitted) would place. No changes made."""
    return _placement_result(lambda: preview_placement(_parse_spec(spec)))


@_tool_named("set_placement")
def set_placement_tool(spec: dict[str, Any]) -> dict[str, Any]:
    """Set and apply a manual multi-GPU placement spec (persists to config).

    The spec maps a role ("chat"/"embed"/"rerank"/"vision") to a placement, e.g.
    ``{"chat": {"devices": [0, 1], "tensor_split": [1, 1]}}``. ``devices`` is the
    GPU indices (get_gpus lists them); ``tensor_split`` is optional per-device
    weights (omit for an even split). Omit a role to leave it auto-placed.
    """
    from lilbee.providers.fleet.placement_spec import PlacementSpec

    # set_placement restarts the shared fleet's moved roles: gate it on the
    # shared HTTP transport exactly like the REST PUT/DELETE placement routes.
    if _transport.http_mounted and not cfg.allow_http_placement:
        return _error(placement_refused_message())
    # Always build a spec (even {}) so an empty/invalid one is rejected, not cleared.
    return _placement_result(lambda: set_placement(PlacementSpec.from_json(json.dumps(spec))))


@_tool_named("clear_placement")
def clear_placement_tool() -> dict[str, Any]:
    """Clear the manual placement and return to automatic placement."""
    if _transport.http_mounted and not cfg.allow_http_placement:
        return _error(placement_refused_message())
    return _placement_result(lambda: set_placement(None))


def build_mcp_server() -> LilbeeMCP:
    """Build an MCP server carrying every tool registered in this module.

    Each transport builds its own instance: the SDK caches one
    ``StreamableHTTPSessionManager`` per server and its ``run()`` is single-use,
    so a shared server cannot back two apps in one process. Gates registered
    via ``_tool_if`` are evaluated here, against current config.
    """
    server = LilbeeMCP("lilbee", instructions=_INSTRUCTIONS)
    for fn, name, gate in _REGISTRATIONS:
        if gate is None or gate():
            server.add_tool(fn, name=name)
    return server


_PARENT_DEATH_CLEANUP_S = 5.0


def _exit_on_parent_death() -> None:
    """Release engine membership best-effort, then hard-exit promptly.

    ``os._exit`` skips atexit, so an explicit release stops this process's engine
    when it was the last user and keeps the machine clean. But this watchdog's one
    contract is to exit promptly on parent death, so the release runs on a daemon
    thread joined with a short deadline: a peer holding an engine build lock can
    never keep the orphaned process alive with its models resident. The kernel
    releases this process's user lock on exit regardless, so a skipped release only
    defers the engine stop to the peers' reap and the idle TTL.
    """
    cleanup = threading.Thread(target=reset_services, name="parent-death-cleanup", daemon=True)
    cleanup.start()
    cleanup.join(timeout=_PARENT_DEATH_CLEANUP_S)
    os._exit(0)


def main() -> None:
    """Entry point for the stdio MCP server."""
    # Preload so the first tool call doesn't pay the cold-start cost
    # of provider/embedder/store init. Failures (missing model, bad
    # config) still surface on the first tool call rather than crashing
    # the server before it attaches to stdio.
    try:
        get_services()
    except Exception:
        log.debug("MCP pre-warm failed; services will init on first call", exc_info=True)

    from lilbee.parent_monitor import parse_parent_pid, watch_parent_thread

    parent_pid = parse_parent_pid()
    if parent_pid is not None:
        watch_parent_thread(parent_pid, _exit_on_parent_death)

    build_mcp_server().run()
