"""Entity and concept extraction strategies for the concept/entity wiki.

Each strategy reads a corpus of chunks and produces a deduplicated list of
``ExtractedEntity`` records. The factory selects the strategy at runtime
based on ``cfg.wiki_entity_mode``.
"""

from __future__ import annotations

from lilbee.wiki.entity_extractor.base import (
    ChunkRef,
    EntityExtractor,
    EntityKind,
    ExtractedEntity,
)
from lilbee.wiki.entity_extractor.factory import get_entity_extractor

__all__ = [
    "ChunkRef",
    "EntityExtractor",
    "EntityKind",
    "ExtractedEntity",
    "get_entity_extractor",
]
