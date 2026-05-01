"""Fully LLM-driven entity extractor.

Stub: the real implementation asks the LLM to propose a schema AND tag
every chunk. Most accurate, most expensive (O(N) LLM calls per ingest).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.wiki.entity_extractor.base import ExtractedEntity

if TYPE_CHECKING:
    from lilbee.core.config import Config
    from lilbee.data.store import SearchChunk
    from lilbee.providers.base import LLMProvider


class LlmTaggedExtractor:
    """LLM-proposed schema plus per-chunk LLM tagging."""

    def __init__(self, provider: LLMProvider, config: Config) -> None:
        self._provider = provider
        self._config = config

    def extract(self, chunks: list[SearchChunk]) -> list[ExtractedEntity]:
        raise NotImplementedError("LlmTaggedExtractor.extract is not yet implemented")
