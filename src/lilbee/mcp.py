"""MCP server exposing lilbee as tools for AI agents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from lilbee.store import SearchChunk

mcp = FastMCP("lilbee", instructions="Local RAG knowledge base. Search indexed documents.")


@mcp.tool()
def lilbee_search(query: str, top_k: int = 5) -> list[dict]:
    """Search the knowledge base for relevant document chunks.

    Returns chunks sorted by relevance. No LLM call — uses pre-computed embeddings.
    """
    from lilbee.query import search_context

    results = search_context(query, top_k=top_k)
    return [clean(r) for r in results]


@mcp.tool()
def lilbee_status() -> dict:
    """Show indexed documents, configuration, and chunk counts."""
    from lilbee.config import cfg
    from lilbee.store import get_sources

    sources = get_sources()
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
    """Add files or directories to the knowledge base and sync.

    Copies the given paths into the documents directory, then ingests them.
    Paths must be absolute and accessible from this machine.

    Args:
        paths: Absolute file or directory paths to add.
        force: Overwrite files that already exist in the knowledge base.
        vision_model: Ollama vision model for scanned PDF OCR
            (e.g. "maternion/LightOnOCR-2:latest"). If empty, uses
            the configured default. If no model is configured,
            scanned PDFs are skipped.
    """
    from lilbee.cli.helpers import copy_files
    from lilbee.config import cfg
    from lilbee.ingest import sync

    errors: list[str] = []
    valid: list[Path] = []
    for p_str in paths:
        p = Path(p_str)
        if not p.exists():
            errors.append(p_str)
        else:
            valid.append(p)

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
        "errors": errors,
        "sync": sync_result,
    }


@mcp.tool()
def lilbee_init(path: str = "") -> dict:
    """Initialize a local .lilbee/ knowledge base in a directory.

    Creates .lilbee/ with documents/, data/, and .gitignore.
    If path is empty, uses the current working directory.
    """
    root = Path(path) / ".lilbee" if path else Path.cwd() / ".lilbee"
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
    from lilbee.config import cfg
    from lilbee.store import delete_by_source, delete_source, get_sources

    known = {s["filename"] for s in get_sources()}
    removed: list[str] = []
    not_found: list[str] = []
    for name in names:
        if name not in known:
            not_found.append(name)
            continue
        delete_by_source(name)
        delete_source(name)
        removed.append(name)
        if delete_files:
            path = cfg.documents_dir / name
            if path.exists():
                path.unlink()
    return {"command": "remove", "removed": removed, "not_found": not_found}


@mcp.tool()
def lilbee_list_documents() -> dict:
    """List all indexed documents with their chunk counts."""
    from lilbee.store import get_sources

    sources = get_sources()
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
