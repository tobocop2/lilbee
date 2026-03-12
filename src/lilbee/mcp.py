"""MCP server exposing lilbee as tools for AI agents."""

from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

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

    return asdict(await sync(quiet=True))


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

    old_vision = getattr(cfg, "vision_model", "")
    if vision_model:
        cfg.vision_model = vision_model  # type: ignore[attr-defined]
    try:
        sync_result = asdict(await sync(quiet=True))
    finally:
        if vision_model:
            cfg.vision_model = old_vision  # type: ignore[attr-defined]

    return {
        "command": "add",
        "copied": copy_result.copied,
        "skipped": copy_result.skipped,
        "errors": errors,
        "sync": sync_result,
    }


@mcp.tool()
def lilbee_reset() -> dict:
    """Delete all documents and data (full factory reset).

    WARNING: This permanently removes all indexed documents and vector data.
    """
    from lilbee.cli import perform_reset

    return perform_reset()


def clean(result: dict) -> dict:
    """Strip vector field and rename _distance for output."""
    cleaned = {k: v for k, v in result.items() if k != "vector"}
    if "_distance" in cleaned:
        cleaned["distance"] = cleaned.pop("_distance")
    return cleaned


def main() -> None:
    """Entry point for the MCP server."""
    mcp.run()
