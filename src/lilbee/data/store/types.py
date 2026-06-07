"""Public dataclasses, TypedDicts, enums, and constants for the store package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import NamedTuple, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

# How often readers re-check the manifest for new versions from other processes.
# Zero means strong consistency (every read checks); higher values reduce disk I/O
# on slow media (HDD) at the cost of serving slightly stale data.
READ_CONSISTENCY_INTERVAL = timedelta(seconds=5)


class ChunkWrite(NamedTuple):
    """One document's chunks plus its source-table update, for a batched write.

    ``Store.write_chunks_batch`` folds many of these into a single locked
    transaction so bulk ingest doesn't pay a write-lock acquisition per document.
    """

    source: str
    file_hash: str
    records: list[dict]
    needs_cleanup: bool


class ChunkType(StrEnum):
    """Values for the ``chunk_type`` column.

    Everything ingests as ``RAW`` except wiki pages written by the wiki
    producer; callers filter with ``Store.search(chunk_type=...)``.
    """

    RAW = "raw"
    WIKI = "wiki"


class SourceType(StrEnum):
    """Values for the ``_sources.source_type`` column.

    ``DOCUMENT`` mirrors a file under ``documents/`` and is managed by the
    file-driven sync. ``IMPORTED`` is detached: it came from ``lilbee import``
    and has no backing file, so sync must not treat it as a missing document.
    """

    DOCUMENT = "document"
    IMPORTED = "imported"


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
    Reranked results have ``rerank_score`` set (higher = better).
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
    rerank_score: float | None = None


class SourceRecord(TypedDict):
    """A tracked source document record."""

    filename: str
    file_hash: str
    ingested_at: str
    chunk_count: int
    source_type: str


class PageTextRecord(TypedDict):
    """One row of the per-page text dataset, matching ``_page_texts``."""

    source: str
    page: int
    text: str
    content_type: str


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


class MemoryKind(StrEnum):
    """Whether a memory is an always-injected preference or a similarity-recalled fact."""

    PREFERENCE = "preference"
    FACT = "fact"


class MemorySource(StrEnum):
    """Provenance of a memory: user-typed, LLM-extracted, or agent-written."""

    MANUAL = "manual"
    EXTRACTED = "extracted"
    AGENT = "agent"


# Memory owner namespaces. ``"local"`` is the single human (TUI/CLI/REST); agents own
# ``"agent:<id>"`` namespaces. The prefix lives only here so it is never hand-spliced.
LOCAL_OWNER = "local"
AGENT_OWNER_PREFIX = "agent:"


def agent_owner(agent_id: str) -> str:
    """Owner string for an agent identity (``"opencode"`` -> ``"agent:opencode"``)."""
    return f"{AGENT_OWNER_PREFIX}{agent_id}"


def is_agent_owner(owner: str) -> bool:
    """True when *owner* is an agent namespace rather than the local human."""
    return owner.startswith(AGENT_OWNER_PREFIX)


class MemoryRow(BaseModel):
    """A long-term memory entry in the per-library ``_memories`` table.

    Built from a LanceDB row via ``MemoryRow(**row)`` (which coerces the ``kind``
    and ``source`` strings to enums) and written back via ``model_dump(mode="json")``.
    Extra keys like a search ``_distance`` are ignored on construction.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    owner: str
    shared: bool
    kind: MemoryKind
    source: MemorySource
    text: str
    vector: list[float] = Field(repr=False)
    created_at: str
    updated_at: str


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
    """Raised when stored vectors were built with a different embedder than ``cfg``.

    Carries the persisted and configured refs and dims so each surface renders its
    own recovery affordance (TUI prompt, CLI command, REST body) from the facts.
    """

    def __init__(
        self,
        *,
        persisted_model: str,
        persisted_dim: int,
        current_model: str,
        current_dim: int,
    ) -> None:
        self.persisted_model = persisted_model
        self.persisted_dim = persisted_dim
        self.current_model = current_model
        self.current_dim = current_dim
        super().__init__(self._build_message())

    @property
    def dims_match(self) -> bool:
        """True when the index is adoptable by switching embedder alone (same dim)."""
        return self.persisted_dim == self.current_dim

    def _build_message(self) -> str:
        if self.dims_match:
            return (
                f"This index was built with embedding model '{self.persisted_model}', "
                f"but lilbee is configured to use '{self.current_model}'. Configure lilbee "
                f"to use '{self.persisted_model}' to search this index, or rebuild it under "
                f"'{self.current_model}'."
            )
        return (
            f"This index was built with embedding model '{self.persisted_model}' "
            f"(dim {self.persisted_dim}), which differs from the current "
            f"'{self.current_model}' (dim {self.current_dim}). The dimensions differ, "
            f"so rebuild the index under '{self.current_model}' to use it."
        )


@dataclass
class RemoveResult:
    """Result of a remove_documents operation."""

    removed: list[str]
    not_found: list[str]
