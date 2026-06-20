"""StrEnum types used by :mod:`lilbee.config`."""

from enum import StrEnum


class ChatMode(StrEnum):
    """How chat turns route through retrieval. ``search`` uses retrieval; ``chat`` skips it."""

    SEARCH = "search"
    CHAT = "chat"


class LlmProvider(StrEnum):
    """Inference backend that ``create_provider`` builds.

    ``auto`` prefix-routes: native GGUF refs to the local llama-server engine,
    remote-prefixed refs (``ollama/``, ``openai/``, ...) to the SDK backend.
    ``remote`` forces the SDK backend.
    """

    AUTO = "auto"
    REMOTE = "remote"


class RerankerType(StrEnum):
    """How the reranker GGUF is served. ``auto`` detects by architecture."""

    AUTO = "auto"
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"


class CrawlRenderMode(StrEnum):
    """How a crawl fetches pages. ``http`` uses no browser; ``browser`` runs Chromium with JS."""

    HTTP = "http"
    BROWSER = "browser"


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


class KvCacheType(StrEnum):
    """KV cache element type. ``q8_0`` / ``q4_0`` require flash attention."""

    F16 = "f16"
    F32 = "f32"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"


# Bytes per KV element for memory budgeting. The quantized variants are
# ~1 byte of data plus shared scales, close enough for context-fit math.
KV_CACHE_TYPE_BYTES: dict[KvCacheType, int] = {
    KvCacheType.F16: 2,
    KvCacheType.F32: 4,
    KvCacheType.Q8_0: 1,
    KvCacheType.Q4_0: 1,
}
