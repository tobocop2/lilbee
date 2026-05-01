"""NerConceptsExtractor plus an LLM-proposed domain type schema.

Stub: the real implementation wraps ``NerConceptsExtractor`` and issues
one LLM call against a representative corpus sample to propose custom
entity types. Cached in ``wiki/_schema.yaml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.wiki.entity_extractor.base import ExtractedEntity

if TYPE_CHECKING:
    from lilbee.core.config import Config
    from lilbee.data.store import SearchChunk
    from lilbee.providers.base import LLMProvider


class NerConceptsPlusLlmTypesExtractor:
    """NER + concepts with an LLM-proposed domain schema layered on top."""

    def __init__(self, provider: LLMProvider, config: Config) -> None:
        self._provider = provider
        self._config = config

    def extract(self, chunks: list[SearchChunk]) -> list[ExtractedEntity]:
        raise NotImplementedError("NerConceptsPlusLlmTypesExtractor.extract is not yet implemented")
