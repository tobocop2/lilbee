"""Public dataclasses, TypedDicts, enums, and constants for the store package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

# How often readers re-check the manifest for new versions from other processes.
# Zero means strong consistency (every read checks); higher values reduce disk I/O
# on slow media (HDD) at the cost of serving slightly stale data.
READ_CONSISTENCY_INTERVAL = timedelta(seconds=5)


class ChunkType(StrEnum):
    """Values for the ``chunk_type`` column.

    Everything ingests as ``RAW`` except wiki pages written by the wiki
    producer; callers filter with ``Store.search(chunk_type=...)``.
    """

    RAW = "raw"
    WIKI = "wiki"


# ``schema_version`` is an integer for forward-compat. Bump only if we ever need to
# add or rename a meta column without forcing every store to drop_all.
META_SCHEMA_VERSION = 1

# Always-true predicate used to clear the single-row ``_meta`` table before re-insert.
# Lance's ``Table.delete`` requires a SQL where clause; this matches every row without
# coupling the deletion to any specific column's value domain.
META_DELETE_ALL_PREDICATE = "schema_version IS NOT NULL"


class SearchScope(StrEnum):
    """What the user wants to search over.

    Values are used as-is on CLI flags, MCP params, and HTTP query strings.
    ``BOTH`` resolves to a ``None`` ``chunk_type`` (no filter); the two
    others map 1:1 to the chunks-table values.
    """

    RAW = ChunkType.RAW
    WIKI = ChunkType.WIKI
    BOTH = "both"


def scope_to_chunk_type(scope: SearchScope | str | None) -> ChunkType | None:
    """Translate a user-facing scope into a ``Store.search`` ``chunk_type`` arg.

    ``None``/``"both"`` → no filter. ``"raw"`` / ``"wiki"`` → the matching
    ``ChunkType``. Raises ``ValueError`` on any other string.
    """
    if scope is None:
        return None
    normalized = SearchScope(scope)
    if normalized is SearchScope.BOTH:
        return None
    return ChunkType(normalized.value)


class SearchChunk(BaseModel):
    """A search result from LanceDB.
    Hybrid results have ``relevance_score`` set (higher = better).
    Vector-only results have ``distance`` set (lower = better).
    """

    model_config = ConfigDict(populate_by_name=True)

    source: str
    content_type: str
    chunk_type: ChunkType = ChunkType.RAW

    @field_validator("chunk_type", mode="before")
    @classmethod
    def _coerce_none_chunk_type(cls, v: str | None) -> str:
        """LanceDB rows from before the chunk_type column was added return None."""
        return v if v is not None else ChunkType.RAW

    page_start: int
    page_end: int
    line_start: int
    line_end: int
    chunk: str
    chunk_index: int
    vector: list[float] = Field(repr=False)
    distance: float | None = Field(None, alias="_distance")
    relevance_score: float | None = Field(None, alias="_relevance_score")


class SourceRecord(TypedDict):
    """A tracked source document record."""

    filename: str
    file_hash: str
    ingested_at: str
    chunk_count: int
    source_type: str


class CitationRecord(TypedDict):
    """A citation linking a wiki chunk to a specific source location."""

    wiki_source: str
    wiki_chunk_index: int
    citation_key: str
    claim_type: str
    source_filename: str
    source_hash: str
    page_start: int
    page_end: int
    line_start: int
    line_end: int
    excerpt: str
    created_at: str


class StoreMeta(TypedDict):
    """Single-row store metadata recording the embedding model used to build the store.

    Compatibility is checked before every read and write. When ``cfg.embedding_model``
    or ``cfg.embedding_dim`` drifts from the persisted row, the store refuses to serve
    until ``lilbee rebuild`` (CLI) or ``POST /api/sync {"force_rebuild": true}`` (HTTP)
    rewrites the chunks under the new model.

    ``updated_at`` is an ISO 8601 UTC timestamp produced by ``datetime.isoformat()``;
    kept as ``str`` to match the LanceDB ``utf8`` schema column.
    """

    embedding_model: str
    embedding_dim: int
    schema_version: int
    updated_at: str


class EmbeddingModelMismatchError(RuntimeError):
    """Raised when stored vectors were built with a different embedding model than ``cfg``.

    Carries a user-facing message naming both the persisted and the configured model and
    pointing at the two recovery paths (``lilbee rebuild`` and ``POST /api/sync`` with
    ``force_rebuild=true``).
    """


@dataclass
class RemoveResult:
    """Result of a remove_documents operation."""

    removed: list[str]
    not_found: list[str]
