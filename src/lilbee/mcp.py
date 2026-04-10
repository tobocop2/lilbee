"""MCP server exposing lilbee as tools for AI agents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from lilbee.config import cfg
from lilbee.crawl_task import get_task, start_crawl
from lilbee.crawler import is_url, require_valid_crawl_url
from lilbee.security import validate_path_within
from lilbee.services import get_services

if TYPE_CHECKING:
    from lilbee.store import SearchChunk

mcp = FastMCP("lilbee", instructions="Local RAG knowledge base. Search indexed documents.")


@mcp.tool()
def lilbee_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search the knowledge base for relevant document chunks.
    Returns chunks sorted by relevance. No LLM call — uses pre-computed embeddings.
    """
    results = get_services().searcher.search(query, top_k=top_k)
    return [clean(r) for r in results]


@mcp.tool()
def lilbee_status() -> dict[str, Any]:
    """Show indexed documents, configuration, and chunk counts."""
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
async def lilbee_sync() -> dict[str, Any]:
    """Sync documents directory with the vector store."""
    from lilbee.ingest import sync

    return (await sync(quiet=True)).model_dump()


@mcp.tool()
async def lilbee_add(
    paths: list[str],
    force: bool = False,
    vision_model: str = "",
) -> dict[str, Any]:
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
    from lilbee.cli.helpers import copy_files, temporary_vision_model
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

    with temporary_vision_model(vision_model):
        sync_result = (await sync(quiet=True, force_vision=bool(vision_model))).model_dump()

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
) -> dict[str, Any]:
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
def lilbee_crawl_status(task_id: str) -> dict[str, Any]:
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
def lilbee_init(path: str = "") -> dict[str, Any]:
    """Initialize a local .lilbee/ knowledge base in a directory.
    Creates .lilbee/ with documents/, data/, and .gitignore.
    If path is empty, uses the current working directory.
    """
    base = Path(path) if path else Path.cwd()
    try:
        validate_path_within(base, Path.home())
    except ValueError:
        return {"error": "Path must be within your home directory"}
    root = base / ".lilbee"
    if root.is_dir():
        return {"command": "init", "path": str(root), "created": False}

    (root / "documents").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    (root / ".gitignore").write_text("data/\n")
    return {"command": "init", "path": str(root), "created": True}


@mcp.tool()
def lilbee_remove(names: list[str], delete_files: bool = False) -> dict[str, Any]:
    """Remove documents from the knowledge base by source name.
    Args:
        names: Source filenames to remove (as shown by lilbee_status).
        delete_files: Also delete the physical files from the documents directory.
    """
    result = get_services().store.remove_documents(
        names, delete_files=delete_files, documents_dir=cfg.documents_dir
    )
    return {"command": "remove", "removed": result.removed, "not_found": result.not_found}


@mcp.tool()
def lilbee_list_documents() -> dict[str, Any]:
    """List all indexed documents with their chunk counts."""
    sources = get_services().store.get_sources()
    return {
        "documents": [
            {"filename": s["filename"], "chunk_count": s.get("chunk_count", 0)} for s in sources
        ],
        "total": len(sources),
    }


@mcp.tool()
def lilbee_reset() -> dict[str, Any]:
    """Delete all documents and data (full factory reset).
    WARNING: This permanently removes all indexed documents and vector data.
    """
    from lilbee.cli import perform_reset

    return perform_reset().model_dump()


@mcp.tool()
def lilbee_wiki_lint(wiki_source: str = "") -> dict[str, Any]:
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
def lilbee_wiki_citations(wiki_source: str) -> dict[str, Any]:
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
def lilbee_wiki_status() -> dict[str, Any]:
    """Show wiki layer status: page counts, recent lint issues."""
    from lilbee.wiki.lint import lint_all

    wiki_root = cfg.data_root / cfg.wiki_dir
    if not wiki_root.exists():
        return {"wiki_enabled": cfg.wiki, "pages": 0, "issues": 0}

    summaries = (
        list((wiki_root / "summaries").rglob("*.md")) if (wiki_root / "summaries").exists() else []
    )
    drafts = list((wiki_root / "drafts").rglob("*.md")) if (wiki_root / "drafts").exists() else []

    report = lint_all(get_services().store)
    return {
        "wiki_enabled": cfg.wiki,
        "summaries": len(summaries),
        "drafts": len(drafts),
        "pages": len(summaries) + len(drafts),
        "lint_errors": report.error_count,
        "lint_warnings": report.warning_count,
    }


@mcp.tool()
def lilbee_wiki_list() -> dict[str, Any]:
    """List all wiki pages (summaries and concepts) with metadata.
    Returns page slugs, titles, types, source counts, and creation dates.
    """
    if not cfg.wiki:
        return {"error": "wiki not enabled"}
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
def lilbee_wiki_read(slug: str) -> dict[str, Any]:
    """Read a wiki page's content and frontmatter by slug.
    Args:
        slug: Page slug like "summaries/my-doc" or "concepts/typing".
    """
    if not cfg.wiki:
        return {"error": "wiki not enabled"}
    from dataclasses import asdict

    from lilbee.wiki.browse import read_page

    wiki_root = cfg.data_root / cfg.wiki_dir
    result = read_page(wiki_root, slug)
    if result is None:
        return {"error": f"wiki page not found: {slug}"}
    return {"command": "wiki_read", **asdict(result)}


@mcp.tool()
def lilbee_wiki_prune() -> dict[str, Any]:
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


def clean(result: SearchChunk) -> dict[str, object]:
    """Convert SearchChunk to a JSON-friendly dict."""
    return result.model_dump(exclude={"vector"}, exclude_none=True)


def main() -> None:
    """Entry point for the MCP server."""
    mcp.run()
