"""MCP server exposing lilbee as tools for AI agents."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lilbee", instructions="Local RAG knowledge base. Search indexed documents.")


@mcp.tool()
def lilbee_search(query: str, top_k: int = 5) -> list[dict]:
    """Search the knowledge base for relevant document chunks.

    Returns chunks sorted by relevance. No LLM call — uses pre-computed embeddings.
    """
    from lilbee.query import search_context

    results = search_context(query, top_k=top_k)
    return [_clean(r) for r in results]


@mcp.tool()
def lilbee_status() -> dict:
    """Show indexed documents, configuration, and chunk counts."""
    from lilbee.config import CHAT_MODEL, DATA_DIR, DOCUMENTS_DIR, EMBEDDING_MODEL
    from lilbee.store import get_sources

    sources = get_sources()
    return {
        "config": {
            "documents_dir": str(DOCUMENTS_DIR),
            "data_dir": str(DATA_DIR),
            "chat_model": CHAT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
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
    from dataclasses import asdict

    from lilbee.ingest import sync

    return asdict(await sync(quiet=True))


@mcp.tool()
def lilbee_reset() -> dict:
    """Delete all documents and data (full factory reset).

    WARNING: This permanently removes all indexed documents and vector data.
    """
    from lilbee.cli import _perform_reset

    return _perform_reset()


def _clean(result: dict) -> dict:
    """Strip vector field and rename _distance for output."""
    cleaned = {k: v for k, v in result.items() if k != "vector"}
    if "_distance" in cleaned:
        cleaned["distance"] = cleaned.pop("_distance")
    return cleaned


def main() -> None:
    """Entry point for the MCP server."""
    mcp.run()
