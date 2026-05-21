"""MCP server exposing lilbee as tools for AI agents."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from mcp.server.fastmcp import Context, FastMCP

from lilbee.app.search import clean_result
from lilbee.app.services import get_services, reset_services, reset_store
from lilbee.app.settings import (
    SettingInfo,
    apply_settings_update,
    get_setting,
    list_settings,
    reset_settings,
)
from lilbee.catalog.types import ModelSource
from lilbee.core.config import cfg
from lilbee.core.settings import overlay_persisted_settings
from lilbee.core.system import LOCAL_ROOT_DIRNAME
from lilbee.crawler import crawler_available, is_url, require_valid_crawl_url
from lilbee.crawler.task import get_task, start_crawl
from lilbee.data.store import SearchScope, scope_to_chunk_type
from lilbee.wiki.shared import (
    WIKI_DISABLED_ERROR,
    WikiSubdir,
)

log = logging.getLogger(__name__)

mcp = FastMCP("lilbee", instructions="Local RAG knowledge base. Search indexed documents.")

_F = TypeVar("_F", bound=Callable[..., Any])


def _tool_if(condition: bool) -> Callable[[_F], _F]:
    """Register a function as an MCP tool only when *condition* is true.

    The function stays importable so direct callers (tests, in-process
    fallback) can still reach it. Whether the tool appears in the MCP
    schema is fixed at import time; changing the gating config requires
    a server restart.
    """
    if condition:
        # mcp.tool() returns Callable[[Callable[..., Any]], Callable[..., Any]];
        # cast to the typed-callable shape so generic call sites narrow correctly.
        return cast("Callable[[_F], _F]", mcp.tool())

    def _passthrough(fn: _F) -> _F:
        return fn

    return _passthrough


def _error(msg: str) -> dict[str, Any]:
    """Uniform error envelope MCP tool handlers return on a failure path.

    Typed as ``dict[str, Any]`` rather than a TypedDict so it composes
    with the success-side returns under the existing handler signatures
    without forcing every caller to widen its return type.
    """
    return {"error": msg}


@mcp.tool()
def search(
    query: str, top_k: int | None = None, scope: str = SearchScope.BOTH.value
) -> list[dict[str, Any]] | dict[str, Any]:
    """Search the knowledge base for relevant document chunks.

    ``top_k`` defaults to ``cfg.top_k``. ``scope`` is ``"raw"``, ``"wiki"``,
    or ``"both"``. Returns chunks sorted by relevance.
    """
    if not query or not query.strip():
        return _error("query must not be empty")
    try:
        chunk_type = scope_to_chunk_type(scope)
    except ValueError as exc:
        return _error(str(exc))
    effective_top_k = top_k if top_k is not None else cfg.top_k
    try:
        results = get_services().searcher.search(
            query, top_k=effective_top_k, chunk_type=chunk_type
        )
        results = [r for r in results if r.distance is None or r.distance <= cfg.max_distance]
        return [clean_result(r) for r in results]
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
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
            "main_gpu": cfg.main_gpu,
            "gpu_devices": cfg.gpu_devices,
        },
        "sources": [
            {"filename": s["filename"], "chunk_count": s["chunk_count"]}
            for s in sorted(sources, key=lambda x: x["filename"])
        ],
        "total_chunks": sum(s["chunk_count"] for s in sources),
    }


@mcp.tool()
async def sync(force_rebuild: bool = False, retry_skipped: bool = False) -> dict[str, Any]:
    """Sync the documents directory into the vector store.

    ``force_rebuild`` drops every table and re-ingests. ``retry_skipped``
    clears failed-file skip markers without dropping the store.
    """
    from lilbee.data.ingest import sync as run_sync

    return (
        await run_sync(quiet=True, force_rebuild=force_rebuild, retry_skipped=retry_skipped)
    ).model_dump()


@mcp.tool()
async def add(
    paths: list[str],
    force: bool = False,
    enable_ocr: bool | None = None,
    ocr_timeout: float | None = None,
) -> dict[str, Any]:
    """Add files, directories, or URLs to the knowledge base, then sync.

    Paths must be absolute. URLs (http(s)://) are crawled as markdown.
    ``enable_ocr`` forces OCR on/off; ``ocr_timeout`` overrides the
    per-page OCR cap for this call.
    """
    from lilbee.app.ingest import copy_files
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
                require_valid_crawl_url(url)
            except ValueError as exc:
                errors.append(f"{url}: {exc}")
                continue
            crawled_paths = await crawl_and_save(url)
            crawled_count += len(crawled_paths)

    copy_result = copy_files(valid, force=force)

    from lilbee.app.ingest import temporary_ocr_config

    with temporary_ocr_config(enable_ocr, ocr_timeout):
        sync_result = (await run_sync(quiet=True)).model_dump()

    result: dict[str, Any] = {
        "command": "add",
        "copied": copy_result.copied,
        "skipped": copy_result.skipped,
        "crawled": crawled_count,
        "errors": errors,
        "sync": sync_result,
    }
    if errors or sync_result.get("failed"):
        result["warning"] = "some files could not be processed"
    return result


@_tool_if(crawler_available())
def crawl(
    url: str,
    depth: int | None = None,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """Start a non-blocking crawl; poll via ``crawl_status(task_id)``.

    ``depth=None`` crawls the whole site, ``0`` is single-URL, positive ints
    cap follow depth. ``max_pages=None`` is unlimited.
    """
    from lilbee.crawler import crawler_available

    if not crawler_available():
        return _error("Web crawling requires: pip install 'lilbee[crawler]'")
    try:
        require_valid_crawl_url(url)
    except ValueError as exc:
        return _error(str(exc))

    task_id = start_crawl(url, depth=depth, max_pages=max_pages)
    return {"status": "started", "task_id": task_id, "url": url}


@_tool_if(crawler_available())
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
        "error": task.error,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
    }


@mcp.tool()
def init(path: str = "") -> dict[str, Any]:
    """Initialize a local ``.lilbee/`` knowledge base; empty path = cwd.

    Switches the MCP session to use it for subsequent calls.
    """
    base = Path(path) if path else Path.cwd()
    root = base / LOCAL_ROOT_DIRNAME

    created = False
    if not root.is_dir():
        (root / "documents").mkdir(parents=True)
        (root / "data").mkdir(parents=True)
        (root / ".gitignore").write_text("data/\n")
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


@mcp.tool()
def remove(names: list[str], delete_files: bool = False) -> dict[str, Any]:
    """Remove documents by source name; ``delete_files=true`` also deletes the file on disk."""
    result = get_services().store.remove_documents(
        names, delete_files=delete_files, documents_dir=cfg.documents_dir
    )
    return {"command": "remove", "removed": result.removed, "not_found": result.not_found}


@mcp.tool()
def list_documents() -> dict[str, Any]:
    """List all indexed documents with their chunk counts."""
    sources = get_services().store.get_sources()
    return {
        "documents": [
            {"filename": s["filename"], "chunk_count": s.get("chunk_count", 0)} for s in sources
        ],
        "total": len(sources),
    }


@mcp.tool()
def reset(confirm: bool = False) -> dict[str, Any]:
    """Factory reset: delete all documents and indexed data. Requires ``confirm=true``."""
    if not confirm:
        return _error("pass confirm=true to confirm deletion")
    from lilbee.app.reset import perform_reset

    result = perform_reset().model_dump()
    # Reopen LanceDB against the empty data dir; keep providers loaded.
    reset_store()
    return result


@_tool_if(cfg.wiki)
def wiki_lint(wiki_source: str = "") -> dict[str, Any]:
    """Lint wiki pages; empty ``wiki_source`` lints all."""
    from lilbee.wiki.lint import lint_all, lint_wiki_page

    store = get_services().store
    if wiki_source:
        issues = lint_wiki_page(wiki_source, store)
    else:
        report = lint_all(store)
        issues = report.issues
    return {
        "command": "wiki_lint",
        "issues": [i.to_dict() for i in issues],
        "total": len(issues),
    }


@_tool_if(cfg.wiki)
def wiki_citations(wiki_source: str) -> dict[str, Any]:
    """List citations for a wiki page."""
    records = get_services().store.get_citations_for_wiki(wiki_source)
    return {
        "command": "wiki_citations",
        "wiki_source": wiki_source,
        "citations": [dict(r) for r in records],
        "total": len(records),
    }


@_tool_if(cfg.wiki)
def wiki_status() -> dict[str, Any]:
    """Show wiki layer status: page counts, recent lint issues."""
    from lilbee.wiki.lint import lint_all

    wiki_root = cfg.data_root / cfg.wiki_dir
    if not wiki_root.exists():
        return {"wiki_enabled": cfg.wiki, "pages": 0, "issues": 0}

    summaries_dir = wiki_root / WikiSubdir.SUMMARIES
    drafts_dir = wiki_root / WikiSubdir.DRAFTS
    summaries = list(summaries_dir.rglob("*.md")) if summaries_dir.exists() else []
    drafts = list(drafts_dir.rglob("*.md")) if drafts_dir.exists() else []

    report = lint_all(get_services().store)
    return {
        "wiki_enabled": cfg.wiki,
        WikiSubdir.SUMMARIES: len(summaries),
        WikiSubdir.DRAFTS: len(drafts),
        "pages": len(summaries) + len(drafts),
        "lint_errors": report.error_count,
        "lint_warnings": report.warning_count,
    }


@_tool_if(cfg.wiki)
def wiki_list() -> dict[str, Any]:
    """List wiki pages with metadata."""
    if not cfg.wiki:
        return _error(WIKI_DISABLED_ERROR)
    from dataclasses import asdict

    from lilbee.wiki.browse import list_pages

    wiki_root = cfg.data_root / cfg.wiki_dir
    pages = list_pages(wiki_root)
    return {
        "command": "wiki_list",
        "pages": [asdict(p) for p in pages],
        "total": len(pages),
    }


@_tool_if(cfg.wiki)
def wiki_read(slug: str) -> dict[str, Any]:
    """Read a wiki page's content + frontmatter by slug."""
    if not cfg.wiki:
        return _error(WIKI_DISABLED_ERROR)
    from dataclasses import asdict

    from lilbee.wiki.browse import read_page

    wiki_root = cfg.data_root / cfg.wiki_dir
    result = read_page(wiki_root, slug)
    if result is None:
        return _error(f"wiki page not found: {slug}")
    return {"command": "wiki_read", **asdict(result)}


@_tool_if(cfg.wiki)
def wiki_build() -> dict[str, Any]:
    """Build the concept and entity wiki across all ingested sources."""
    if not cfg.wiki:
        return _error(WIKI_DISABLED_ERROR)
    from lilbee.wiki import run_full_build

    return {"command": "wiki_build", **run_full_build(cfg)}


@_tool_if(cfg.wiki)
def wiki_update() -> dict[str, Any]:
    """Refresh the concept and entity wiki after an ingest. Currently a full rebuild."""
    if not cfg.wiki:
        return _error(WIKI_DISABLED_ERROR)
    from lilbee.wiki import run_full_build

    return {"command": "wiki_update", **run_full_build(cfg)}


@_tool_if(cfg.wiki)
def wiki_synthesize() -> dict[str, Any]:
    """Generate synthesis pages for concept clusters with three or more sources."""
    if not cfg.wiki:
        return _error(WIKI_DISABLED_ERROR)
    from lilbee.wiki import run_full_synthesize

    return {"command": "wiki_synthesize", **run_full_synthesize(cfg)}


@_tool_if(cfg.wiki)
def wiki_prune() -> dict[str, Any]:
    """Prune stale and orphaned wiki pages."""
    from lilbee.wiki.prune import prune_wiki

    report = prune_wiki(get_services().store)
    return {
        "command": "wiki_prune",
        "records": [r.to_dict() for r in report.records],
        "archived": report.archived_count,
        "flagged": report.flagged_count,
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


@mcp.tool()
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


@mcp.tool()
def settings_get(key: str) -> dict[str, Any]:
    """Get a single setting's current value + metadata."""

    try:
        info = get_setting(key)
    except KeyError as exc:
        return _error(str(exc))
    return {"command": "settings_get", "setting": _setting_info_to_dict(info)}


@mcp.tool()
def settings_set(updates: dict[str, Any]) -> dict[str, Any]:
    """Atomically update writable settings; rolls back on any validation error.

    Persists to ``config.toml`` and invalidates in-process caches. Returns
    ``{updated, reindex_required}``; ``reindex_required=true`` means run
    ``sync(force_rebuild=true)`` to refresh the index.
    """

    try:
        result = apply_settings_update(updates)
    except (ValueError, TypeError) as exc:
        return _error(str(exc))
    return {
        "command": "settings_set",
        "updated": result.updated,
        "reindex_required": result.reindex_required,
    }


@mcp.tool()
def settings_reset(keys: list[str]) -> dict[str, Any]:
    """Reset writable settings to their built-in defaults."""

    try:
        result = reset_settings(keys)
    except (ValueError, TypeError) as exc:
        return _error(str(exc))
    return {
        "command": "settings_reset",
        "updated": result.updated,
        "reindex_required": result.reindex_required,
    }


@mcp.tool()
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


@mcp.tool()
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
    """Browse the lilbee model catalog (featured + Hugging Face).

    ``task`` is ``chat`` / ``embedding`` / ``vision`` / ``rerank``.
    ``size`` is ``small`` / ``medium`` / ``large``. ``installed`` filters
    by install state, ``featured`` toggles the curated list, ``sort`` is
    one of ``featured`` / ``downloads`` / ``name`` / ``size_asc`` /
    ``size_desc``. Returns paginated model records.
    """
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
            }
            for m in result.models
        ],
    }


@mcp.tool()
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


@mcp.tool()
async def model_pull(
    model: str,
    source: str = ModelSource.NATIVE.value,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Download a model and stream progress. ``source`` is ``native`` (GGUF) or ``remote`` (SDK)."""
    from lilbee.app.models import pull_model_data
    from lilbee.catalog import DownloadProgress

    try:
        src = ModelSource.parse(source) or ModelSource.NATIVE
    except ValueError as exc:
        return _error(str(exc))

    loop = asyncio.get_running_loop()

    def on_update(p: DownloadProgress) -> None:
        if ctx is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            ctx.report_progress(progress=float(p.percent), total=100.0, message=p.detail),
            loop,
        )
        future.add_done_callback(_log_progress_failure)

    try:
        result = await asyncio.to_thread(pull_model_data, model, src, on_update=on_update)
    except (RuntimeError, PermissionError) as exc:
        return _error(str(exc))
    return result.model_dump()


@mcp.tool()
def model_rm(model: str, source: str = "") -> dict[str, Any]:
    """Remove an installed model.

    Args:
        model: Model ref to remove.
        source: Restrict to "native" or "remote"; empty = both.
    """
    from lilbee.app.models import remove_model_data

    try:
        src = ModelSource.parse(source)
    except ValueError as exc:
        return _error(str(exc))
    return remove_model_data(model, source=src).model_dump()


@_tool_if(cfg.wiki)
def wiki_drafts_list() -> dict[str, Any]:
    """List pending wiki drafts (read-only; accept/reject are CLI-only)."""
    from lilbee.wiki.drafts import list_drafts

    wiki_root = cfg.data_root / cfg.wiki_dir
    drafts = list_drafts(wiki_root)
    return {
        "command": "wiki_drafts_list",
        "drafts": [d.to_dict() for d in drafts],
        "total": len(drafts),
    }


@_tool_if(cfg.wiki)
def wiki_drafts_diff(slug: str) -> dict[str, Any]:
    """Unified diff of a draft against its published counterpart."""
    from lilbee.wiki.drafts import diff_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        diff = diff_draft(slug, wiki_root)
    except FileNotFoundError as exc:
        return _error(str(exc))
    return {"command": "wiki_drafts_diff", "slug": slug, "diff": diff}


def _strip_schema_noise() -> None:
    """Trim auto-generated noise from every registered tool's schema before
    it ships on the OpenAI tools wire for each chat request.

    Drops:
    - FastMCP/Pydantic ``title`` keys (per-schema + per-property). Tools the
      model picks by name don't need a separate display title.
    - Triple-quoted docstring indentation on the tool description. The model
      sees a flat sentence instead of multi-line text with 4-space prefixes.

    Runs once after every ``@mcp.tool()`` decoration in this module has fired.
    """
    for info in mcp._tool_manager._tools.values():
        params = info.parameters
        if isinstance(params, dict):
            params.pop("title", None)
            properties = params.get("properties")
            if isinstance(properties, dict):
                for prop in properties.values():
                    if isinstance(prop, dict):
                        prop.pop("title", None)
        if isinstance(info.description, str):
            info.description = textwrap.dedent(info.description).strip()


_strip_schema_noise()


def main() -> None:
    """Entry point for the MCP server."""
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
        watch_parent_thread(parent_pid, lambda: os._exit(0))

    mcp.run()
