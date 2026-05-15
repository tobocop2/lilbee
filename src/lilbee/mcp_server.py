"""MCP server exposing lilbee as tools for AI agents.

Tool handler bodies use function-local ``from lilbee.X import ...`` to keep
``lilbee mcp`` boot fast (the same startup discipline AGENTS.md mandates for
Typer command bodies). Heavy chains pulled in lazily here:
``data.ingest`` / ``wiki.*`` / ``wiki.drafts`` (spaCy via the wiki ingest
pipeline), ``crawler`` (crawl4ai + Playwright), ``app.models`` /
``modelhub.model_manager`` / ``catalog`` (HF discovery + `huggingface_hub`).
``app.ingest`` / ``app.reset`` are individually light but transitively reach
``data.ingest`` via the runtime sync handlers, so they share the policy.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from lilbee.app.search import clean_result
from lilbee.app.services import get_services, reset_services, reset_store
from lilbee.core.config import cfg
from lilbee.core.settings import overlay_persisted_settings
from lilbee.core.system import LOCAL_ROOT_DIRNAME
from lilbee.crawler import is_url, require_valid_crawl_url
from lilbee.crawler.task import get_task, start_crawl
from lilbee.data.store import SearchScope, scope_to_chunk_type
from lilbee.wiki.shared import (
    WIKI_DISABLED_ERROR,
    WikiSubdir,
)

log = logging.getLogger(__name__)

mcp = FastMCP("lilbee", instructions="Local RAG knowledge base. Search indexed documents.")


def _error(msg: str) -> dict[str, Any]:
    """Uniform error envelope MCP tool handlers return on a failure path.

    Typed as ``dict[str, Any]`` rather than a TypedDict so it composes
    with the success-side returns under the existing handler signatures
    without forcing every caller to widen its return type.
    """
    return {"error": msg}


@mcp.tool()
def search(
    query: str, top_k: int = 5, scope: str = SearchScope.BOTH.value
) -> list[dict[str, Any]] | dict[str, Any]:
    """Search the knowledge base for relevant document chunks.

    ``scope`` picks the pool: ``"raw"`` (source chunks), ``"wiki"`` (wiki
    page bodies), or ``"both"`` (default, unfiltered). Returns chunks
    sorted by relevance. No LLM call -- uses pre-computed embeddings.
    """
    if not query or not query.strip():
        return _error("query must not be empty")
    try:
        chunk_type = scope_to_chunk_type(scope)
    except ValueError as exc:
        return _error(str(exc))
    try:
        results = get_services().searcher.search(query, top_k=top_k, chunk_type=chunk_type)
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
    """Sync documents directory with the vector store.

    Args:
        force_rebuild: Drop every table and re-ingest from scratch (equivalent
            to ``lilbee rebuild``). Also clears the failed-file skip markers.
        retry_skipped: Clear the failed-file skip markers so files that were
            skipped on a previous sync get another attempt, without dropping
            the store.
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
    """Add files, directories, or URLs to the knowledge base and sync.
    Copies the given paths into the documents directory, then ingests them.
    URLs (http:// or https://) are fetched as markdown and saved to _web/.
    Paths must be absolute and accessible from this machine.

    Args:
        paths: Absolute file/directory paths or URLs to add.
        force: Overwrite files that already exist in the knowledge base.
        enable_ocr: Force vision OCR on (True), off (False), or auto-detect
            from chat model capabilities (None/omit).
        ocr_timeout: Per-page timeout in seconds for vision OCR. Overrides
            the configured default for this invocation only.
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


@mcp.tool()
def crawl(
    url: str,
    depth: int | None = None,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """Crawl a web page and add it to the knowledge base (non-blocking).
    Launches the crawl as a background task and returns immediately with a
    task_id. Use crawl_status(task_id) to poll progress.

    Args:
        url: The URL to crawl (must start with http:// or https://).
        depth: None (default) crawls the whole site; 0 fetches only this URL;
            positive int caps link-follow depth.
        max_pages: None (default) means no page limit. Positive int caps total
            pages fetched.
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


@mcp.tool()
def crawl_status(task_id: str) -> dict[str, Any]:
    """Check the status of a running crawl task.
    Returns the current state including status, pages crawled, and any error.
    Use this to poll after crawl returns a task_id.

    Args:
        task_id: The task ID returned by crawl.
    """
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
    """Initialize a local .lilbee/ knowledge base in a directory.
    Creates .lilbee/ with documents/, data/, and .gitignore.
    If path is empty, uses the current working directory.
    Also switches the MCP session to use this knowledge base for
    subsequent tool calls.
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
    """Remove documents from the knowledge base by source name.
    Args:
        names: Source filenames to remove (as shown by status).
        delete_files: Also delete the physical files from the documents directory.
    """
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
    """Delete all documents and data (full factory reset).
    WARNING: This permanently removes all indexed documents and vector data.
    Pass confirm=true to proceed.
    """
    if not confirm:
        return _error("pass confirm=true to confirm deletion")
    from lilbee.app.reset import perform_reset

    result = perform_reset().model_dump()
    # Reopen LanceDB against the empty data dir; keep providers loaded.
    reset_store()
    return result


@mcp.tool()
def wiki_lint(wiki_source: str = "") -> dict[str, Any]:
    """Lint wiki pages for citation staleness, missing sources, and unmarked claims.
    If wiki_source is provided, lint only that page. Otherwise, lint all wiki pages.

    Args:
        wiki_source: Path like "wiki/summaries/doc.md". Empty = lint all.
    """
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


@mcp.tool()
def wiki_citations(wiki_source: str) -> dict[str, Any]:
    """Get all citations for a wiki page.
    Args:
        wiki_source: Wiki page path, e.g. "wiki/summaries/doc.md".
    """
    records = get_services().store.get_citations_for_wiki(wiki_source)
    return {
        "command": "wiki_citations",
        "wiki_source": wiki_source,
        "citations": [dict(r) for r in records],
        "total": len(records),
    }


@mcp.tool()
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


@mcp.tool()
def wiki_list() -> dict[str, Any]:
    """List all wiki pages (summaries and concepts) with metadata.
    Returns page slugs, titles, types, source counts, and creation dates.
    """
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


@mcp.tool()
def wiki_read(slug: str) -> dict[str, Any]:
    """Read a wiki page's content and frontmatter by slug.
    Args:
        slug: Page slug like "summaries/my-doc" or "concepts/typing".
    """
    if not cfg.wiki:
        return _error(WIKI_DISABLED_ERROR)
    from dataclasses import asdict

    from lilbee.wiki.browse import read_page

    wiki_root = cfg.data_root / cfg.wiki_dir
    result = read_page(wiki_root, slug)
    if result is None:
        return _error(f"wiki page not found: {slug}")
    return {"command": "wiki_read", **asdict(result)}


@mcp.tool()
def wiki_build() -> dict[str, Any]:
    """Build the concept and entity wiki across all ingested sources.

    Returns ``{paths, entities, count}``.
    """
    if not cfg.wiki:
        return _error(WIKI_DISABLED_ERROR)
    from lilbee.wiki import run_full_build

    return {"command": "wiki_build", **run_full_build(cfg)}


@mcp.tool()
def wiki_update() -> dict[str, Any]:
    """Refresh the concept and entity wiki after an ingest. Currently a full rebuild."""
    if not cfg.wiki:
        return _error(WIKI_DISABLED_ERROR)
    from lilbee.wiki import run_full_build

    return {"command": "wiki_update", **run_full_build(cfg)}


@mcp.tool()
def wiki_synthesize() -> dict[str, Any]:
    """Generate synthesis pages for concept clusters spanning three or more sources.

    Returns the list of synthesis page paths written to disk. When no
    cluster meets the 3+ source threshold, returns an empty list and
    ``count: 0``.
    """
    if not cfg.wiki:
        return _error(WIKI_DISABLED_ERROR)
    from lilbee.wiki import run_full_synthesize

    return {"command": "wiki_synthesize", **run_full_synthesize(cfg)}


@mcp.tool()
def wiki_prune() -> dict[str, Any]:
    """Prune stale and orphaned wiki pages.
    Archives pages whose sources are all deleted or whose concept cluster
    dropped below 3 live sources. Flags pages with >50% stale citations
    for regeneration.
    """
    from lilbee.wiki.prune import prune_wiki

    report = prune_wiki(get_services().store)
    return {
        "command": "wiki_prune",
        "records": [r.to_dict() for r in report.records],
        "archived": report.archived_count,
        "flagged": report.flagged_count,
    }


@mcp.tool()
def model_list(source: str = "", task: str = "") -> dict[str, Any]:
    """List installed models across native and SDK-backend sources.

    Args:
        source: Filter by source: "native", "remote", or "" for all.
        task: Filter by task: "chat", "embedding", "vision", "rerank", or "" for all.
    """
    from lilbee.app.models import list_models_data
    from lilbee.catalog.types import ModelSource, ModelTask

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
    source: str = "native",
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Download a model, streaming progress via MCP notifications.

    Args:
        model: Model ref to pull (e.g. "Qwen/Qwen3-0.6B-GGUF" or
            "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf").
        source: "native" (HuggingFace GGUF) or "remote" (SDK-managed).
    """
    from lilbee.app.models import pull_model_data
    from lilbee.catalog import DownloadProgress
    from lilbee.catalog.types import ModelSource

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
    from lilbee.catalog.types import ModelSource

    try:
        src = ModelSource.parse(source)
    except ValueError as exc:
        return _error(str(exc))
    return remove_model_data(model, source=src).model_dump()


@mcp.tool()
def wiki_drafts_list() -> dict[str, Any]:
    """List pending wiki drafts with drift, faithfulness, and pairing info.

    Read-only. Accept and reject are CLI-only (destructive, explicit).
    """
    from lilbee.wiki.drafts import list_drafts

    wiki_root = cfg.data_root / cfg.wiki_dir
    drafts = list_drafts(wiki_root)
    return {
        "command": "wiki_drafts_list",
        "drafts": [d.to_dict() for d in drafts],
        "total": len(drafts),
    }


@mcp.tool()
def wiki_drafts_diff(slug: str) -> dict[str, Any]:
    """Return a unified diff of the draft against its published counterpart.

    Args:
        slug: Draft slug (e.g. ``"chevrolet"``).
    """
    from lilbee.wiki.drafts import diff_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        diff = diff_draft(slug, wiki_root)
    except FileNotFoundError as exc:
        return _error(str(exc))
    return {"command": "wiki_drafts_diff", "slug": slug, "diff": diff}


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
