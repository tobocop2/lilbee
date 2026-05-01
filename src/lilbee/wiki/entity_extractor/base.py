"""Protocol and record types for entity/concept extractors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lilbee.data.store import SearchChunk


class EntityKind(StrEnum):
    """Whether an ``ExtractedEntity`` is a concept or a proper-noun entity."""

    CONCEPT = "concept"
    ENTITY = "entity"


@dataclass(frozen=True)
class ChunkRef:
    """Stable identifier for a chunk inside the store."""

    source: str
    chunk_index: int


@dataclass(frozen=True)
class ExtractedEntity:
    """One concept or entity discovered in the corpus.

    All fields are populated by the extractor regardless of strategy, so
    downstream page generation, [[link]] rewriting, and index building
    never branch on which extractor ran.
    """

    slug: str
    kind: EntityKind
    label: str
    type_hint: str
    chunk_refs: tuple[ChunkRef, ...]


@runtime_checkable
class EntityExtractor(Protocol):
    """Strategy that turns a chunk corpus into ``ExtractedEntity`` records."""

    def extract(self, chunks: list[SearchChunk]) -> list[ExtractedEntity]:
        """Return the deduplicated set of concepts and entities in *chunks*."""
        ...
