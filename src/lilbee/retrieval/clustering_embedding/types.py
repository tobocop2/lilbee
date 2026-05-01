"""Type definitions for the chunk-level mutual-kNN clusterer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkRecord:
    """Lightweight view of one chunk row used by the clusterer."""

    source: str
    chunk_index: int
    text: str
    tokens: list[str] = field(default_factory=list)
