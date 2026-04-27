"""StrEnum types used by :mod:`lilbee.config`."""

from enum import StrEnum


class ClustererBackend(StrEnum):
    """Known wiki clusterer backends."""

    EMBEDDING = "embedding"
    CONCEPTS = "concepts"


class WikiEntityMode(StrEnum):
    """Strategy used to extract entities for the wiki.

    Phase D: the extractor no longer emits concepts: concept pages
    are proposed by the LLM inside the per-source batched call in
    ``wiki.gen``. The enum values reflect the extractor's current
    responsibility (typed NER entities only).
    """

    NER_ENTITIES = "ner_entities"
    NER_CONCEPTS_PLUS_LLM_TYPES = "ner_concepts_plus_llm_types"
    LLM_TAGGED = "llm_tagged"
