"""StrEnum types used by :mod:`lilbee.config`."""

from enum import StrEnum


class ReasoningMode(StrEnum):
    """How ``/v1/chat/completions`` presents a reasoning model's thinking.

    ``separate`` reports thinking in ``reasoning_content`` (OpenAI-compatible).
    ``inline`` streams thinking as ordinary ``content`` text with the
    ``<think>`` markers stripped, for clients
    that never render ``reasoning_content``. ``off`` asks the model not to
    think; thinking templates honor the request and other templates ignore it.
    """

    SEPARATE = "separate"
    INLINE = "inline"
    OFF = "off"


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


class TableModel(StrEnum):
    """xberg's table structure recognition model, used when layout detection is on.

    The ``slanet_*`` variants are the docling-parity lineage; ``tatr`` is xberg's
    older default. ``disabled`` skips structure recognition.
    """

    DISABLED = "disabled"
    TATR = "tatr"
    SLANET_AUTO = "slanet_auto"
    SLANET_PLUS = "slanet_plus"
    SLANET_WIRED = "slanet_wired"
    SLANET_WIRELESS = "slanet_wireless"


class KvCacheType(StrEnum):
    """KV cache element type. ``q8_0`` / ``q4_0`` require flash attention."""

    F16 = "f16"
    F32 = "f32"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"


# Bytes per KV element for memory budgeting, from llama.cpp's block layouts:
# q8_0 stores 32 elements in 34 bytes (2-byte scale + 32 data bytes) and q4_0
# stores 32 elements in 18 bytes (2-byte scale + 16 bytes of packed nibbles).
# Rounding q4_0 up to a whole byte charged its cache at almost twice the real
# size and nearly halved the context window the dynamic picker granted.
KV_CACHE_TYPE_BYTES: dict[KvCacheType, float] = {
    KvCacheType.F16: 2.0,
    KvCacheType.F32: 4.0,
    KvCacheType.Q8_0: 34 / 32,
    KvCacheType.Q4_0: 18 / 32,
}
