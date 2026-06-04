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
    LOCAL_OWNER,
    ChunkType,
    CitationRecord,
    EmbeddingModelMismatchError,
    MemoryKind,
    MemoryRow,
    MemorySource,
    RemoveResult,
    SearchChunk,
    SearchScope,
    SourceRecord,
    agent_owner,
    is_agent_owner,
    scope_to_chunk_type,
)

__all__ = [
    "LOCAL_OWNER",
    "ChunkType",
    "CitationRecord",
    "EmbeddingModelMismatchError",
    "MemoryKind",
    "MemoryRow",
    "MemorySource",
    "RemoveResult",
    "SearchChunk",
    "SearchScope",
    "SourceRecord",
    "Store",
    "agent_owner",
    "cosine_sim",
    "ensure_table",
    "escape_sql_string",
    "install_lancedb_thread_error_suppressor",
    "is_agent_owner",
    "mmr_rerank",
    "safe_delete",
    "scope_to_chunk_type",
]
