"""Source clustering abstraction for wiki synthesis pages.

Defines the :class:`SourceClusterer` protocol, the :class:`ClustererBackend`
enum of known backend identifiers, and the :class:`Clusterer` facade. The
facade is the single class the services container constructs and it picks
the right backend from ``config.wiki_clusterer`` so callers never need to
know which implementation they got.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lilbee.config import ClustererBackend

if TYPE_CHECKING:
    from lilbee.config import Config
    from lilbee.store import Store

log = logging.getLogger(__name__)


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


def _select_backend(config: Config, store: Store) -> SourceClusterer:
    """Pick a backend based on ``config.wiki_clusterer`` with safe fallback.

    Concrete backends are imported inside the function to break a hard
    circular dependency: ``clustering_embedding`` re-exports
    :class:`SourceCluster` from this module, so importing it at module
    level here would fail during package initialization.
    """
    from lilbee.clustering_embedding import EmbeddingClusterer
    from lilbee.concepts import ConceptGraphClusterer

    if config.wiki_clusterer == ClustererBackend.CONCEPTS:
        graph_clusterer = ConceptGraphClusterer(config, store)
        if graph_clusterer.available():
            return graph_clusterer
        log.warning(
            "wiki_clusterer=concepts but the [graph] extra is not installed or "
            "the concept graph has not been built. Falling back to the "
            "embedding clusterer."
        )
    return EmbeddingClusterer(config, store)


class Clusterer:
    """Wiki synthesis clusterer facade with backend selection."""

    def __init__(self, config: Config, store: Store) -> None:
        self._backend: SourceClusterer = _select_backend(config, store)

    @property
    def backend(self) -> SourceClusterer:
        """Return the underlying backend (useful for tests and introspection)."""
        return self._backend

    def available(self) -> bool:
        return self._backend.available()

    def get_clusters(self, min_sources: int = 3) -> list[SourceCluster]:
        return self._backend.get_clusters(min_sources=min_sources)
