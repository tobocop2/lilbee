"""MCP server exposing lilbee as tools for AI agents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from lilbee.config import cfg
from lilbee.crawl_task import get_task, start_crawl
from lilbee.crawler import is_url, require_valid_crawl_url

if TYPE_CHECKING:
    from lilbee.store import SearchChunk

mcp = FastMCP("lilbee", instructions="Local RAG knowledge base. Search indexed documents.")


@mcp.tool()
def lilbee_search(query: str, top_k: int = 5) -> list[dict]:
    """Search the knowledge base for relevant document chunks.

    Returns chunks sorted by relevance. No LLM call — uses pre-computed embeddings.
    """
    from lilbee.services import get_services

    results = get_services().searcher.search(query, top_k=top_k)
    return [clean(r) for r in results]


@mcp.tool()
def lilbee_status() -> dict:
    """Show indexed documents, configuration, and chunk counts."""
    from lilbee.services import get_services

    sources = get_services().store.get_sources()
    return {
        "config": {
            "documents_dir": str(cfg.documents_dir),
            "data_dir": str(cfg.data_dir),
            "chat_model": cfg.chat_model,
            "embedding_model": cfg.embedding_model,
            **({"vision_model": cfg.vision_model} if cfg.vision_model else {}),
        },
        "sources": [
            {"filename": s["filename"], "chunk_count": s["chunk_count"]}
            for s in sorted(sources, key=lambda x: x["filename"])
        ],
        "total_chunks": sum(s["chunk_count"] for s in sources),
    }


@mcp.tool()
async def lilbee_sync() -> dict:
    """Sync documents directory with the vector store."""
    from lilbee.ingest import sync

    return (await sync(quiet=True)).model_dump()


@mcp.tool()
async def lilbee_add(
    paths: list[str],
    force: bool = False,
    vision_model: str = "",
) -> dict:
    """Add files, directories, or URLs to the knowledge base and sync.

    Copies the given paths into the documents directory, then ingests them.
    URLs (http:// or https://) are fetched as markdown and saved to _web/.
    Paths must be absolute and accessible from this machine.

    Args:
        paths: Absolute file/directory paths or URLs to add.
        force: Overwrite files that already exist in the knowledge base.
        vision_model: Vision model for scanned PDF OCR
            (e.g. "maternion/LightOnOCR-2:latest"). If empty, uses
            the configured default. If no model is configured,
            scanned PDFs are skipped.
    """
    from lilbee.cli.helpers import copy_files
    from lilbee.ingest import sync

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
            return {"error": "Web crawling requires: pip install 'lilbee[crawler]'"}
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

    old_vision = cfg.vision_model
    if vision_model:
        cfg.vision_model = vision_model
    try:
        sync_result = (await sync(quiet=True, force_vision=bool(vision_model))).model_dump()
    finally:
        if vision_model:
            cfg.vision_model = old_vision

    return {
        "command": "add",
        "copied": copy_result.copied,
        "skipped": copy_result.skipped,
        "crawled": crawled_count,
        "errors": errors,
        "sync": sync_result,
    }


@mcp.tool()
def lilbee_crawl(
    url: str,
    depth: int = 0,
    max_pages: int = 50,
) -> dict:
    """Crawl a web page and add it to the knowledge base (non-blocking).

    Launches the crawl as a background task and returns immediately with a
    task_id. Use lilbee_crawl_status(task_id) to poll progress.

    Args:
        url: The URL to crawl (must start with http:// or https://).
        depth: Maximum link-following depth (0 = single page only).
        max_pages: Maximum number of pages to fetch (default: 50).
    """
    from lilbee.crawler import crawler_available

    if not crawler_available():
        return {"error": "Web crawling requires: pip install 'lilbee[crawler]'"}
    try:
        require_valid_crawl_url(url)
    except ValueError as exc:
        return {"error": str(exc)}

    task_id = start_crawl(url, depth=depth, max_pages=max_pages)
    return {"status": "started", "task_id": task_id, "url": url}


@mcp.tool()
def lilbee_crawl_status(task_id: str) -> dict:
    """Check the status of a running crawl task.

    Returns the current state including status, pages crawled, and any error.
    Use this to poll after lilbee_crawl returns a task_id.

    Args:
        task_id: The task ID returned by lilbee_crawl.
    """
    task = get_task(task_id)
    if task is None:
        return {"error": f"No task found with id: {task_id}"}
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
def lilbee_init(path: str = "") -> dict:
    """Initialize a local .lilbee/ knowledge base in a directory.

    Creates .lilbee/ with documents/, data/, and .gitignore.
    If path is empty, uses the current working directory.
    """
    base = Path(path) if path else Path.cwd()
    if not base.resolve().is_relative_to(Path.home().resolve()):
        return {"error": "Path must be within your home directory"}
    root = base / ".lilbee"
    if root.is_dir():
        return {"command": "init", "path": str(root), "created": False}

    (root / "documents").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    (root / ".gitignore").write_text("data/\n")
    return {"command": "init", "path": str(root), "created": True}


@mcp.tool()
def lilbee_remove(names: list[str], delete_files: bool = False) -> dict:
    """Remove documents from the knowledge base by source name.

    Args:
        names: Source filenames to remove (as shown by lilbee_status).
        delete_files: Also delete the physical files from the documents directory.
    """
    from lilbee.services import get_services

    store = get_services().store
    known = {s["filename"] for s in store.get_sources()}
    removed: list[str] = []
    not_found: list[str] = []
    docs_resolved = cfg.documents_dir.resolve()
    for name in names:
        if name not in known:
            not_found.append(name)
            continue
        store.delete_by_source(name)
        store.delete_source(name)
        removed.append(name)
        if delete_files:
            path = cfg.documents_dir / name
            if not path.resolve().is_relative_to(docs_resolved):
                continue
            if path.exists():
                path.unlink()
    return {"command": "remove", "removed": removed, "not_found": not_found}


@mcp.tool()
def lilbee_list_documents() -> dict:
    """List all indexed documents with their chunk counts."""
    from lilbee.services import get_services

    sources = get_services().store.get_sources()
    return {
        "documents": [
            {"filename": s["filename"], "chunk_count": s.get("chunk_count", 0)} for s in sources
        ],
        "total": len(sources),
    }


@mcp.tool()
def lilbee_reset() -> dict:
    """Delete all documents and data (full factory reset).

    WARNING: This permanently removes all indexed documents and vector data.
    """
    from lilbee.cli import perform_reset

    return perform_reset().model_dump()


def clean(result: SearchChunk) -> dict[str, object]:
    """Convert SearchChunk to a JSON-friendly dict."""
    return result.model_dump(exclude={"vector"}, exclude_none=True)


def main() -> None:
    """Entry point for the MCP server."""
    mcp.run()
