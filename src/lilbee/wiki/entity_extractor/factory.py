"""Runtime selector for the entity-extraction strategy."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from lilbee.core.config import WikiEntityMode
from lilbee.wiki.entity_extractor.base import EntityExtractor
from lilbee.wiki.entity_extractor.llm_tagged import LlmTaggedExtractor
from lilbee.wiki.entity_extractor.ner_concepts import NerConceptsExtractor
from lilbee.wiki.entity_extractor.ner_concepts_plus_llm_types import (
    NerConceptsPlusLlmTypesExtractor,
)

if TYPE_CHECKING:
    from lilbee.core.config import Config
    from lilbee.providers.base import LLMProvider

log = logging.getLogger(__name__)

_EXTRACTOR_BY_MODE: dict[
    WikiEntityMode,
    Callable[[LLMProvider, Config], EntityExtractor],
] = {
    WikiEntityMode.NER_ENTITIES: NerConceptsExtractor,
    WikiEntityMode.NER_CONCEPTS_PLUS_LLM_TYPES: NerConceptsPlusLlmTypesExtractor,
    WikiEntityMode.LLM_TAGGED: LlmTaggedExtractor,
}

# Implementations whose ``extract`` actually runs. Modes outside this set
# are accepted for forward compatibility (so config files and env vars
# keep parsing) but fall back to ``NER_ENTITIES`` with a warning.
_IMPLEMENTED_MODES: frozenset[WikiEntityMode] = frozenset({WikiEntityMode.NER_ENTITIES})


def effective_entity_mode(mode: WikiEntityMode) -> WikiEntityMode:
    """The mode that will actually run for *mode*, applying the fallback.

    Unimplemented strategies resolve to ``NER_ENTITIES``; provenance records this
    so the audit reflects the extractor that ran, not the configured request.
    """
    return mode if mode in _IMPLEMENTED_MODES else WikiEntityMode.NER_ENTITIES


def get_entity_extractor(
    mode: WikiEntityMode, provider: LLMProvider, config: Config
) -> EntityExtractor:
    """Return an ``EntityExtractor`` implementation for *mode*.

    Unimplemented strategies fall back to ``NER_ENTITIES`` with a
    warning so a user who flips the config to a stub never crashes a
    build or sync mid-flight.
    """
    if mode not in _IMPLEMENTED_MODES:
        log.warning(
            "Entity-extraction mode %r is not yet implemented; falling back to %r",
            mode.value,
            WikiEntityMode.NER_ENTITIES.value,
        )
    effective = effective_entity_mode(mode)
    factory = _EXTRACTOR_BY_MODE[effective]
    return factory(provider, config)
