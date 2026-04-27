"""StrEnum types used by :mod:`lilbee.config`."""

from enum import StrEnum


class ClustererBackend(StrEnum):
    """Known wiki clusterer backends."""

    EMBEDDING = "embedding"
    CONCEPTS = "concepts"


class WikiEntityMode(StrEnum):
    """Strategy used to extract entities for the wiki.

    The extractor emits typed NER entities only. Concept pages are
    proposed by the LLM inside the per-source batched call in
    :mod:`lilbee.wiki.generation`. The enum values reflect that
    extractor responsibility.
    """

    NER_ENTITIES = "ner_entities"
    NER_CONCEPTS_PLUS_LLM_TYPES = "ner_concepts_plus_llm_types"
    LLM_TAGGED = "llm_tagged"
