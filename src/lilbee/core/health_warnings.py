"""Degradations a running server can report to a client.

Each subsystem owns the state behind its own warnings and exposes them; the
health handler aggregates. Nothing here holds state, so there is no registry to
keep in sync with the components that produce it.
"""

from __future__ import annotations

from enum import StrEnum

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
    """The embedder's window is below the chunk budget; chunks are sized in its tokens."""
    PLACEMENT_DIVERGED = "placement_diverged"
    """The engine allocated materially more or less GPU memory than planned."""


class HealthWarning(BaseModel):
    """One active degradation, with the remedy the user can act on."""

    code: WarningCode
    message: str
    """User-facing description of what is degraded."""
    remedy: str | None = None
    """The action that clears it, when there is one."""
