"""Source clustering abstraction for wiki synthesis pages.

The wiki synthesis layer groups related documents into clusters and generates
one cross-source page per cluster. Multiple clustering strategies are possible
(embedding-space, concept graph, LLM topic extraction, etc). This module
defines the ``SourceClusterer`` protocol that decouples the wiki from any
specific implementation. See ``clustering_embedding`` for the default
backend and ``concepts.ConceptGraphClusterer`` for the graph-based adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SourceCluster:
    """A group of related documents identified by a clustering strategy."""

    cluster_id: str
    """Opaque stable identifier, used for filesystem slugs."""

    label: str
    """Human-readable topic label for the cluster."""

    sources: frozenset[str]
    """Set of source document filenames in the cluster."""


@runtime_checkable
class SourceClusterer(Protocol):
    """Finds clusters of related source documents for cross-source synthesis."""

    def available(self) -> bool:
        """Return True if this clusterer can produce clusters in the current env."""
        ...

    def get_clusters(self, min_sources: int = 3) -> list[SourceCluster]:
        """Return clusters spanning at least ``min_sources`` distinct documents."""
        ...
