"""Degradations a running server can report to a client.

Each subsystem owns the state behind its own warnings and exposes them; the
health handler aggregates. Nothing here holds state, so there is no registry to
keep in sync with the components that produce it.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class WarningCode(StrEnum):
    """Stable identifier for a degradation, so a client can branch on it."""

    FTS_UNAVAILABLE = "fts_unavailable"
    """Keyword search failed and queries fall back to vector-only recall."""
    EMBEDDING_PREFIX_MISMATCH = "embedding_prefix_mismatch"
    """Queries are prefixed but stored documents are not; retrieval is degraded."""
    INDEX_EMBEDDING_MISMATCH = "index_embedding_mismatch"
    """The index was built with another embedding model; search refuses until they agree."""
    SCALAR_INDEX_UNAVAILABLE = "scalar_index_unavailable"
    """A scalar index is registered but its files are gone; filtered search is degraded."""
    EMBED_WINDOW_BELOW_CHUNK = "embed_window_below_chunk"
    """The embedding model accepts fewer tokens than chunk_size, so chunks are cut to its cap."""
    PLACEMENT_DIVERGED = "placement_diverged"
    """The engine allocated materially more or less GPU memory than planned."""
    HOST_SPILL = "host_spill"
    """The engine keeps part of a model's weights in system memory, where it runs slower."""
    LIBRARY_UPGRADING ="library_upgrading"
    """A first open is upgrading the library's format; requests that need it wait."""
    ENGINE_FALLBACK = "engine_fallback"
    """A role runs on another engine than the one chosen, or the stored choices are unreadable."""
    HYDE_BLEND_FAILED = "hyde_blend_failed"
    """A search ran without its hypothetical-answer step because that step failed."""
    QUERY_EXPANSION_FAILED = "query_expansion_failed"
    """A search ran without model-written query variants because that step failed."""
    CONCEPT_BOOST_FAILED = "concept_boost_failed"
    """A search ran without the concept re-rank because that step failed."""
    CONCEPT_EXPANSION_FAILED = "concept_expansion_failed"
    """A search ran without concept query variants because that step failed."""
    RERANK_FAILED = "rerank_failed"
    """A search ran without the cross-encoder re-rank because the reranker failed."""
    GPU_NOT_SERVED = "gpu_not_served"
    """The engine pack that installed on first use cannot use this computer's GPU."""
    ROUTER_OUTDATED = "router_outdated"
    """The llama-swap router is too old to copy the output of an engine that writes no log."""


def warnings_payload(warnings: Iterable[HealthWarning]) -> list[dict[str, Any]]:
    """The warnings as JSON-ready dicts, the shape ``/api/health`` serves."""
    return [w.model_dump(mode="json") for w in warnings]


class HealthWarning(BaseModel):
    """One active degradation, with the remedy the user can act on."""

    code: WarningCode
    message: str
    """User-facing description of what is degraded."""
    remedy: str | None = None
    """The action that clears it, when there is one."""

    @property
    def line(self) -> str:
        """The message followed by the remedy, for a one-line display."""
        return f"{self.message} {self.remedy}" if self.remedy else self.message
