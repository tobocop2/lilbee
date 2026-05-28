"""LanceDB vector store package."""

from __future__ import annotations

from .core import Store
from .lance_helpers import (
    ensure_table,
    escape_sql_string,
    install_lancedb_thread_error_suppressor,
    safe_delete,
)
from .ranking import cosine_sim, mmr_rerank
from .types import (
    ChunkType,
    ChunkWrite,
    CitationRecord,
    EmbeddingModelMismatchError,
    RemoveResult,
    SearchChunk,
    SearchScope,
    SourceRecord,
    scope_to_chunk_type,
)

__all__ = [
    "ChunkType",
    "ChunkWrite",
    "CitationRecord",
    "EmbeddingModelMismatchError",
    "RemoveResult",
    "SearchChunk",
    "SearchScope",
    "SourceRecord",
    "Store",
    "cosine_sim",
    "ensure_table",
    "escape_sql_string",
    "install_lancedb_thread_error_suppressor",
    "mmr_rerank",
    "safe_delete",
    "scope_to_chunk_type",
]
