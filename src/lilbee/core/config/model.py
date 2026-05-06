"""The :class:`Config` dataclass and the ``cfg`` singleton.

The settings sources, TOML parser, and the resilient builder that falls
back to defaults on stale-config validation failures live here too. Every
``from lilbee.core.config import cfg`` resolves through ``lilbee.core.config.__init__``
to the same instance defined at module bottom.
"""

import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .defaults import (
    DEFAULT_ALLOWED_NER_LABELS,
    DEFAULT_CORS_ORIGIN_REGEX,
    DEFAULT_CRAWL_EXCLUDE_PATTERNS,
    DEFAULT_GENERAL_SYSTEM_PROMPT,
    DEFAULT_IGNORE_DIRS,
    DEFAULT_RAG_SYSTEM_PROMPT,
)
from .enums import ChatMode, ClustererBackend, KvCacheType, WikiEntityMode
from .parsing import parse_bool
from .validators import ConfigField

log = logging.getLogger(__name__)

# Sentinel for unset Path-typed fields. ``Field(default=Path())`` produces an
# instance equal to this, so the model_validator can distinguish "user passed
# the default" from "user explicitly set a value".
_UNSET_PATH = Path()


class Config(BaseSettings):
    """Runtime configuration: one singleton instance, mutated by CLI overrides."""

    model_config = SettingsConfigDict(
        env_prefix="LILBEE_",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    # Paths: resolved from env/defaults in model_validator(mode='before')
    data_root: Path = Field(default=Path())
    # Writable so plugin-managed servers can pivot storage to a vault path on
    # first boot; rebuild the index after migrating.
    documents_dir: Path = ConfigField(default=Path(), writable=True)
    data_dir: Path = Field(default=Path())
    lancedb_dir: Path = Field(default=Path())
    models_dir: Path = Field(default=Path())
    # Obsidian vault root; when set, search results carry a vault-relative
    # ``vault_path`` for native-UI deep-links.
    vault_base: Path | None = ConfigField(default=None, writable=True)

    chat_model: str = Field(default="Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", min_length=1)
    embedding_model: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf",
        min_length=1,
    )
    # Vision OCR model for scanned PDFs and image-only pages. Empty = disabled;
    # there is no cross-role fallback onto the chat model even if multimodal.
    vision_model: str = ConfigField(default="", public=True)
    embedding_dim: int = Field(default=768, ge=1)
    chunk_size: int = ConfigField(default=512, ge=64, writable=True, reindex=True)
    chunk_overlap: int = ConfigField(default=100, ge=0, writable=True, reindex=True)
    max_embed_chars: int = Field(default=2000, ge=1)
    top_k: int = ConfigField(default=10, ge=1, writable=True)
    max_distance: float = ConfigField(default=0.9, ge=0.0, writable=True)
    # Minimum RRF relevance score for hybrid search results (0.0 = no filtering).
    min_relevance_score: float = ConfigField(default=0.0, ge=0.0, writable=True)
    adaptive_threshold: bool = Field(default=False)
    rag_system_prompt: str = ConfigField(
        default=DEFAULT_RAG_SYSTEM_PROMPT, min_length=1, writable=True
    )
    general_system_prompt: str = ConfigField(
        default=DEFAULT_GENERAL_SYSTEM_PROMPT, min_length=1, writable=True
    )
    chat_mode: str = ConfigField(default=ChatMode.SEARCH.value, writable=True)
    ignore_dirs: frozenset[str] = Field(default=DEFAULT_IGNORE_DIRS)
    # OCR for scanned PDFs via vision-capable chat model.
    # None = auto-detect (use OCR if chat model is vision-capable).
    # True = force OCR regardless of detection.
    # False = disable OCR entirely.
    enable_ocr: bool | None = ConfigField(default=None, writable=True)
    # Per-page timeout in seconds for vision OCR (0 = no limit).
    ocr_timeout: float = ConfigField(default=120.0, ge=0.0, writable=True)
    # Max concurrent vision-OCR requests per PDF. Default 1 (serial): raise
    # only when the vision model is network-hosted with meaningful latency
    # (remote API, separate Ollama host). Local GPU models contend on a
    # single device and get slower with concurrency > 1.
    vision_concurrency: int = ConfigField(default=1, ge=1, writable=True)
    # Outer wall-clock budget for the PDF-extract subprocess: load buffer
    # plus per_page_budget * pages. Tune up for slow hardware (M1 Pro vision
    # is ~5min/page) or down for fast hardware. ocr_timeout still governs
    # the per-page timeout inside the subprocess.
    vision_load_budget_s: float = ConfigField(default=300.0, ge=0.0, writable=True)
    vision_per_page_budget_s: float = ConfigField(default=600.0, ge=0.0, writable=True)

    # Tesseract fallback wall-clock timeout per file, seconds. 0 = no cap.
    tesseract_timeout: float = ConfigField(default=60.0, ge=0.0, writable=True)
    semantic_chunking: bool = ConfigField(default=False, writable=True)
    topic_threshold: float = ConfigField(default=0.75, ge=0.0, le=1.0, writable=True)
    server_host: str = "127.0.0.1"
    server_port: int = Field(default=0, ge=0, le=65535)
    cors_origins: list[str] = Field(default_factory=list)
    cors_origin_regex: str = Field(default=DEFAULT_CORS_ORIGIN_REGEX)
    # Seconds between SSE heartbeat events when the producer queue is idle.
    # Must stay well below the plugin's STREAM_IDLE_TIMEOUT_MS (120s) so a
    # single long-running vision OCR page can't starve the client into aborting.
    sse_heartbeat_interval: float = ConfigField(default=30.0, ge=0.0, writable=True)
    json_mode: bool = False
    temperature: float | None = ConfigField(default=None, ge=0.0, writable=True)
    top_p: float | None = ConfigField(default=None, ge=0.0, le=1.0, writable=True)
    top_k_sampling: int | None = ConfigField(default=None, ge=1, writable=True)
    # 1.1 is llama.cpp's default. Leaving this at None caused n-gram loops
    # ("tire tire tire...") on some open-weights models.
    repeat_penalty: float | None = ConfigField(default=1.1, ge=0.0, writable=True)
    num_ctx: int | None = ConfigField(default=None, ge=1, writable=True)
    max_tokens: int | None = ConfigField(default=4096, ge=1, writable=True)
    seed: int | None = ConfigField(default=None, writable=True)
    llm_provider: str = ConfigField(default="auto", writable=True)
    remote_base_url: str = ConfigField(default="http://localhost:11434", writable=True)
    llm_api_key: str = ConfigField(default="", writable=True, write_only=True)
    openai_api_key: str = ConfigField(default="", writable=True, write_only=True)
    anthropic_api_key: str = ConfigField(default="", writable=True, write_only=True)
    gemini_api_key: str = ConfigField(default="", writable=True, write_only=True)

    # Retrieval quality knobs.

    # Max chunks per source in top-k; prevents one large file monopolizing results.
    diversity_max_per_source: int = ConfigField(default=3, ge=1, writable=True)

    # MMR relevance/diversity tradeoff; 0 = max diversity, 1 = pure relevance
    # (Carbonell & Goldstein 1998).
    mmr_lambda: float = ConfigField(default=0.5, ge=0.0, le=1.0, writable=True)

    # Extra candidates retrieved for MMR reranking (multiplies top_k).
    candidate_multiplier: int = ConfigField(default=3, ge=1, writable=True)

    # LLM-generated alternative queries for expansion. 0 disables.
    query_expansion_count: int = ConfigField(default=3, ge=0, writable=True)

    # Skip LLM expansion when tokenized query length ≤ this. The LLM round-trip
    # dominates latency on small local models; short queries already have strong
    # BM25/vector signal. Concept-graph expansion still runs. 0 disables the skip.
    expansion_short_query_tokens: int = ConfigField(default=2, ge=0, writable=True)

    # Cosine-distance step when adaptive-widening retry kicks in.
    adaptive_threshold_step: float = ConfigField(default=0.2, gt=0.0, writable=True)

    # Reject expansion variants below expansion_similarity_threshold.
    expansion_guardrails: bool = ConfigField(default=True, writable=True)

    # Min cosine similarity between question and variant embeddings.
    expansion_similarity_threshold: float = ConfigField(default=0.5, ge=0.0, le=1.0, writable=True)

    # Sigmoid-normalized BM25 score above which query expansion is skipped.
    expansion_skip_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # Min BM25 top-1 vs top-2 gap to skip expansion.
    expansion_skip_gap: float = Field(default=0.15, ge=0.0, le=1.0)

    # Chunks included in LLM context after adaptive selection.
    max_context_sources: int = ConfigField(default=5, ge=1, writable=True)

    # HyDE (Gao et al. 2022): hypothetical-answer embedding search. +~500ms.
    hyde: bool = ConfigField(default=False, writable=True)

    # HyDE result weight relative to real-doc search (0.0-1.0).
    hyde_weight: float = ConfigField(default=0.7, ge=0.0, le=1.0, writable=True)

    # HyDE prompt template. Must contain {question} placeholder.
    hyde_prompt: str = (
        "Write a 50-100 word passage that directly answers this question as if "
        "it were an excerpt from a real document. Do not include any preamble, "
        "just write the passage.\n\nQuestion: {question}"
    )

    # Reranker model ref. Empty disables reranking. Native GGUFs use
    # llama-cpp rank pooling; hosted refs (cohere/voyage/jina/together/hf-tei)
    # need the backend extra.
    reranker_model: str = ConfigField(default="", public=True)

    # Candidate count sent to the reranker.
    rerank_candidates: int = ConfigField(default=20, ge=1, writable=True, public=True)

    # Date-range filter; only fires when a temporal keyword is detected.
    temporal_filtering: bool = ConfigField(default=True, writable=True)

    # If True, emit <think>…</think> content as separate SSE reasoning events;
    # if False, strip it silently.
    show_reasoning: bool = ConfigField(default=False, writable=True)

    # Web crawling.

    # Optional global ceilings. None = no ceiling.
    crawl_max_depth: int | None = ConfigField(default=None, ge=0, writable=True)
    crawl_max_pages: int | None = ConfigField(default=None, ge=1, writable=True)

    # Per-URL fetch timeout, seconds.
    crawl_timeout: int = ConfigField(default=30, ge=1, writable=True)

    # 0 = unlimited, default = CPU count.
    crawl_max_concurrent: int = Field(default=0, ge=0)

    # Seconds between periodic syncs during crawl. 0 = sync only at end.
    crawl_sync_interval: int = ConfigField(default=30, ge=0, writable=True)

    # Per-request delay + jitter (defaults chosen to be gentler than crawl4ai's).
    crawl_mean_delay: float = ConfigField(default=0.5, ge=0.0, writable=True)
    crawl_max_delay_range: float = ConfigField(default=0.5, ge=0.0, writable=True)

    # In-flight requests per crawl.
    crawl_concurrent_requests: int = ConfigField(default=3, ge=1, writable=True)

    # Per-domain rate-limiter that backs off on HTTP 429/503 and retries.
    crawl_retry_on_rate_limit: bool = ConfigField(default=True, writable=True)
    crawl_retry_base_delay_min: float = ConfigField(default=1.0, ge=0.0, writable=True)
    crawl_retry_base_delay_max: float = ConfigField(default=3.0, ge=0.0, writable=True)
    crawl_retry_max_backoff: float = ConfigField(default=30.0, ge=0.0, writable=True)
    crawl_retry_max_attempts: int = ConfigField(default=3, ge=0, writable=True)

    # Regex patterns dropped at link-discovery time. Defaults block CMS
    # scaffolding (WordPress admin, archives, tracking params, etc.).
    crawl_exclude_patterns: list[str] = ConfigField(
        default_factory=lambda: list(DEFAULT_CRAWL_EXCLUDE_PATTERNS),
        writable=True,
    )

    # Fraction of GPU/unified memory reserved for loaded models.
    gpu_memory_fraction: float = ConfigField(default=0.75, ge=0.1, le=1.0, writable=True)

    # Seconds a model stays loaded after last use. 0 = unload immediately.
    model_keep_alive: int = ConfigField(default=300, ge=0, writable=True)

    # Per-call deadline for one pool round-trip (send + recv). Embed batches
    # larger than this on slow machines surface as TimeoutError; raise for
    # heavy ingest jobs.
    worker_pool_call_timeout_s: float = ConfigField(default=300.0, gt=0.0, writable=True)

    # Spawn every configured role at startup instead of on first use. Trades
    # a slower TUI mount (~1-3s per worker, cold-started in parallel) for a
    # responsive first interaction. Roles whose model is unset are skipped,
    # so a setup with only chat + embed never spawns rerank or vision.
    # Set to false for headless / scripted use where the first call doesn't
    # need to be fast.
    worker_pool_eager_start: bool = ConfigField(default=True, writable=True)

    # Idle worker reap. A worker that has been quiet for this many seconds
    # is shut down to free RAM/VRAM; the next request respawns it.
    # ``0`` disables reaping (workers stay up until TUI exit).
    worker_pool_max_idle_s: float = ConfigField(default=300.0, ge=0.0, writable=True)

    # Upper bound for the dynamic n_ctx picker. The picker chooses the
    # largest 256-multiple ctx that fits in available memory and the
    # model's training window; this caps it at a sane ceiling.
    num_ctx_max: int = ConfigField(default=16384, ge=512, writable=True)

    # Flash attention. None (default) = on with TypeError fallback for
    # older llama-cpp-python builds, True = force on, False = off.
    # Resolves the 'padding V cache to 1024' warning on models with
    # uneven per-layer V dims (e.g. Gemma3) and saves ~25% KV memory.
    flash_attention: bool | None = ConfigField(default=None, writable=True)

    # KV cache element type. q8_0 / q4_0 halve or quarter cache memory
    # but require flash attention to be enabled.
    kv_cache_type: KvCacheType = ConfigField(default=KvCacheType.F16, writable=True)

    # Number of model layers to offload to GPU. None (default) = all
    # layers, 0 = CPU only, positive int = partial offload. Useful when a
    # discrete GPU has less VRAM than the model needs.
    n_gpu_layers: int | None = ConfigField(default=None, writable=True)

    # True = Markdown widget for chat; False = plain Static (faster).
    markdown_rendering: bool = True

    # TUI theme name; persists the last Ctrl+T pick across sessions.
    theme: str = ConfigField(default="gruvbox", writable=True)

    # Per-model generation defaults set via apply_model_defaults().
    _model_defaults: Any = None

    # Wiki layer. LLM-maintained synthesis pages with citation provenance.
    # Off by default; flip to True (or set LILBEE_WIKI=1) to enable. When off,
    # the Wiki view tab and the chat ModelBar's scope picker are both hidden.
    wiki: bool = ConfigField(default=False, writable=True)
    wiki_dir: str = "wiki"
    wiki_prune_raw: bool = ConfigField(default=False, writable=True)

    # Minimum cosine similarity between a page body and the mean of its
    # source chunk vectors before a page is published (below → drafts).
    # Replaces the old LLM-based faithfulness score: mean-of-chunks is a
    # deterministic, zero-LLM-call signal that routes topic-drifted
    # pages to drafts without the 0.0 to 1.0 ambiguity of a model-emitted
    # number. Tuning knob: swap to per-chunk max or top-K-mean if the
    # default 0.5 produces false drafts.
    wiki_embedding_faithfulness_threshold: float = ConfigField(
        default=0.5, ge=0.0, le=1.0, writable=True
    )

    # Per-call output token cap for wiki generation. Without this a
    # reasoning model (Qwen3, DeepSeek-R1) can burn the full context
    # window emitting <think> tokens before the actual answer, taking
    # minutes per page. Default leaves headroom for a typical reasoning
    # budget plus a real response (~1000 output + ~1000 slack).
    wiki_summary_max_tokens: int = ConfigField(default=2048, ge=256, writable=True)

    # Wiki generation is a structured-output task: the model must emit the
    # block separators, the citation footnotes, and verbatim quotes. The
    # usual chat default (~0.8) is too creative for that. Lowering the
    # sampling temperature makes the model stick to the template and quote
    # more faithfully. 0.1 leaves just enough slack to avoid hard loops.
    wiki_temperature: float = ConfigField(default=0.1, ge=0.0, le=2.0, writable=True)

    # Fraction of citations that must be stale before a wiki page is flagged.
    wiki_stale_citation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Fraction of content changed that triggers human-review drift guard.
    wiki_drift_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # LLM prompt templates for wiki page generation. Writable so advanced
    # users can override them from /settings, config.toml, or
    # ``LILBEE_WIKI_*_PROMPT`` env vars. Templates must keep the expected
    # ``{placeholders}``. If you remove one the generator will crash on
    # first use. The defaults below are the only reason the pipeline
    # works out of the box.
    wiki_summary_prompt: str = ConfigField(
        writable=True,
        default=(
            "You are a knowledge compiler. Given the source chunks below from a single "
            "document, write a concise wiki summary page in markdown.\n\n"
            "Rules:\n"
            "1. Every factual claim MUST have an inline citation [^src1], [^src2], etc.\n"
            "2. Cite the EXACT text from the source that supports each claim by quoting it.\n"
            "3. For interpretations or connections not directly stated in the source, "
            "mark with [*inference*].\n"
            "4. Use blockquotes (>) for directly cited facts.\n"
            "5. End with a citation block in this format:\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: {source_name}, excerpt: "exact quoted text"\n'
            '[^src2]: {source_name}, excerpt: "exact quoted text"\n\n'
            "Source document: {source_name}\n\n"
            "Chunks:\n{chunks_text}\n\n"
            "Write the wiki summary page now. Start with a heading."
        ),
    )
    wiki_synthesis_prompt: str = ConfigField(
        writable=True,
        default=(
            "You are a knowledge compiler. Given source chunks from MULTIPLE documents "
            "about related concepts, write a synthesis wiki page in markdown that connects "
            "ideas across sources.\n\n"
            "Rules:\n"
            "1. Every factual claim MUST have an inline citation [^src1], [^src2], etc.\n"
            "2. Cite the EXACT text from the source that supports each claim by quoting it.\n"
            "3. For connections, interpretations, or patterns you identify across sources, "
            "mark with [*inference*].\n"
            "4. Use blockquotes (>) for directly cited facts.\n"
            "5. Reference each source by its filename when drawing connections.\n"
            "6. End with a citation block in this format:\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: {{source_name}}, excerpt: "exact quoted text"\n'
            '[^src2]: {{source_name}}, excerpt: "exact quoted text"\n\n'
            "Topic: {topic}\n\n"
            "Sources:\n{source_list}\n\n"
            "Chunks:\n{chunks_text}\n\n"
            "Write the synthesis page now. Start with a heading."
        ),
    )

    # Wiki synthesis clusterer backend. CONCEPTS requires the [graph] extra
    # and falls back to EMBEDDING when unavailable.
    wiki_clusterer: ClustererBackend = ConfigField(
        default=ClustererBackend.EMBEDDING, writable=True
    )

    # Neighborhood size for the mutual-kNN graph. 0 = auto-scale from corpus size.
    wiki_clusterer_k: int = ConfigField(default=0, ge=0, writable=True)

    # LazyGraphRAG-style concept graph. Requires the [graph] extra.
    concept_graph: bool = ConfigField(default=True, writable=True)

    # Weight of concept overlap boost relative to vector similarity.
    concept_boost_weight: float = ConfigField(default=0.3, ge=0.0, le=1.0, writable=True)

    # Floor on post-boost distance to stop weak boosts from promoting marginal hits.
    concept_boost_floor: float = ConfigField(default=0.05, ge=0.0, writable=True)

    # Max noun-phrase concepts extracted per chunk.
    concept_max_per_chunk: int = ConfigField(default=10, ge=1, writable=True)

    # spaCy NER labels kept by the wiki entity extractor. Anything not
    # in this set (QUANTITY, CARDINAL, DATE, TIME, MONEY, PERCENT,
    # ORDINAL, ...) is dropped before aggregation. Override via
    # LILBEE_CONCEPT_ALLOWED_ENT_TYPES as a comma-separated list.
    concept_allowed_ent_types: frozenset[str] = Field(default=DEFAULT_ALLOWED_NER_LABELS)

    # Strategy used to extract entities for the concept/entity wiki.
    # NER_ENTITIES (default) pulls typed NER entities with spaCy; concept
    # pages are proposed by the LLM inside the per-source batched call,
    # not by the extractor. NER_CONCEPTS_PLUS_LLM_TYPES layers an
    # LLM-proposed domain schema on top. LLM_TAGGED asks the LLM to tag
    # every chunk (most expensive). Unimplemented modes fall back to
    # NER_ENTITIES.
    wiki_entity_mode: WikiEntityMode = ConfigField(
        default=WikiEntityMode.NER_ENTITIES, writable=True
    )

    # Minimum distinct chunk mentions before an entity or concept earns
    # its own wiki page. Filters one-off noise.
    wiki_entity_min_mentions: int = ConfigField(default=3, ge=1, writable=True)

    # Maximum chunks passed into each concept or entity page generation
    # call. Caps context size so one page does not blow the context
    # window on a prolific topic.
    wiki_concept_max_chunks_per_page: int = ConfigField(default=25, ge=1, writable=True)

    # Maximum number of related concepts the model is asked to list in
    # the `## Related` section of each page.
    wiki_related_max: int = ConfigField(default=8, ge=0, writable=True)

    # Auto-update cap: if a single sync touches more than this many
    # concept or entity pages, skip the per-slug regeneration and tell
    # the user to run `lilbee wiki update` explicitly. Keeps a surprise
    # bulk import from firing hundreds of LLM calls.
    wiki_ingest_update_cap: int = ConfigField(default=20, ge=1, writable=True)

    # Whether the per-source batched call asks the LLM to curate
    # concept pages alongside the pre-extracted entity list. False →
    # entity sections only, no concept curation (incremental ingest
    # path uses this to avoid churning concept slugs per source-touch).
    wiki_extract_concepts: bool = ConfigField(default=True, writable=True)

    # Minimum chunk count a source must contribute before it is eligible
    # for concept curation. Sources below the floor still get a batched
    # call when they have entities (the prompt writes entity-only
    # sections); sources below the floor with zero entities are skipped
    # entirely. Prevents boilerplate / TOC / appendix documents from
    # burning an LLM call to invent "concepts".
    wiki_batch_min_chunks: int = ConfigField(default=3, ge=1, writable=True)

    # Prompt template for the per-source batched call. Placeholders:
    # {source}, {entity_list}, {chunks_text}, {concept_instruction}.
    # {concept_instruction} is filled with a concept-curation paragraph
    # when concepts are requested, or the empty string otherwise.
    wiki_entity_batch_prompt: str = ConfigField(
        writable=True,
        default=(
            "You are writing wiki sections based on these chunks from {source}.\n\n"
            "{concept_instruction}"
            "Write a wiki section for each of these NER ENTITIES: {entity_list}\n\n"
            "Format each section exactly as:\n"
            "## Name\n"
            "{{content with [^src1]-style citations}}\n\n"
            "Rules:\n"
            "1. Every factual claim MUST have an inline citation [^src1], [^src2], etc.\n"
            "2. Cite the EXACT text from the source that supports each claim by quoting it.\n"
            "3. For interpretations or connections not directly stated, mark with [*inference*].\n"
            "4. Use blockquotes (>) for directly cited facts.\n"
            "5. End the response with a citation block in this format:\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: {{source_name}}, excerpt: "exact quoted text"\n'
            '[^src2]: {{source_name}}, excerpt: "exact quoted text"\n\n'
            "Source chunks:\n{chunks_text}\n"
        ),
    )

    # Class variable: not a settings field
    _toml_cache: ClassVar[dict[str, Any]] = {}

    @field_validator(
        "temperature",
        "top_p",
        "repeat_penalty",
        "top_k_sampling",
        "num_ctx",
        "seed",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("chat_mode", mode="before")
    @classmethod
    def _normalize_chat_mode(cls, v: Any) -> str:
        """Coerce chat_mode to a ChatMode value; default ChatMode.SEARCH."""
        if v is None or v == "":
            return ChatMode.SEARCH.value
        candidate = str(v).strip().lower()
        try:
            return ChatMode(candidate).value
        except ValueError as exc:
            valid = ", ".join(repr(m.value) for m in ChatMode)
            raise ValueError(f"chat_mode must be one of {{{valid}}}, got {v!r}") from exc

    @field_validator("enable_ocr", mode="before")
    @classmethod
    def _parse_enable_ocr(cls, v: Any) -> bool | None:
        """Parse enable_ocr from env var string or direct value.

        Accepts: true/false/1/0/yes/no (case-insensitive), empty string
        or None for auto-detect.
        """
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.strip().lower() in ("", "auto", "none"):
                return None
            try:
                return parse_bool(v)
            except ValueError:
                pass
        return bool(v)

    @field_validator("flash_attention", mode="before")
    @classmethod
    def _parse_flash_attention(cls, v: Any) -> bool | None:
        """Auto/on/off tri-state: empty/auto/none -> None, else parse bool."""
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.strip().lower() in ("", "auto", "none"):
                return None
            try:
                return parse_bool(v)
            except ValueError:
                return None
        return bool(v)

    @field_validator("n_gpu_layers", mode="before")
    @classmethod
    def _parse_n_gpu_layers(cls, v: Any) -> int | None:
        """Auto -> None, ``cpu`` alias -> 0, integers parsed verbatim."""
        if v is None:
            return None
        if isinstance(v, str):
            label = v.strip().lower()
            if label in ("", "auto", "none"):
                return None
            if label == "cpu":
                return 0
            try:
                return int(label)
            except ValueError:
                log.warning("Invalid LILBEE_N_GPU_LAYERS=%r, using auto", v)
                return None
        return int(v)

    @field_validator("semantic_chunking", mode="before")
    @classmethod
    def _parse_semantic_chunking(cls, v: Any) -> bool:
        """Parse from env string; invalid values warn and fall back to False."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            try:
                return parse_bool(v)
            except ValueError:
                log.warning("Invalid LILBEE_SEMANTIC_CHUNKING=%r, using default False", v)
                return False
        return bool(v)

    @field_validator(
        "chat_model", "embedding_model", "vision_model", "reranker_model", mode="after"
    )
    @classmethod
    def _normalize_model_tag(cls, v: str, info: ValidationInfo) -> str:
        """Validate and canonicalize a model ref; blank clears optional roles."""
        if not v or not v.strip():
            if info.field_name in {"chat_model", "embedding_model"}:
                raise ValueError(f"{info.field_name} must not be blank")
            return ""
        from lilbee.providers.model_ref import parse_model_ref

        return parse_model_ref(v).for_openai_prefix()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("crawl_exclude_patterns", mode="before")
    @classmethod
    def _split_crawl_exclude_patterns(cls, v: Any) -> Any:
        """Accept newline-separated strings from env vars / plain-text config.

        Regex commonly uses commas (e.g. `{2,4}`) and pipes (alternation), so
        newline is the only separator safe to use for this field. TOML lists
        and JSON arrays pass through unchanged.
        """
        if isinstance(v, str):
            return [p.strip() for p in v.splitlines() if p.strip()]
        return v

    @field_validator("crawl_exclude_patterns", mode="after")
    @classmethod
    def _validate_crawl_exclude_patterns(cls, v: list[str]) -> list[str]:
        """Reject any entry that isn't a valid Python regex.

        These patterns are compiled at crawl time. An invalid pattern there
        surfaces as an opaque mid-crawl error; catching it at PATCH time gives
        the user a 400 with a pointer to the bad entry.
        """
        import re

        bad: list[str] = []
        for i, pattern in enumerate(v):
            try:
                re.compile(pattern)
            except re.error as exc:
                bad.append(f"[{i}] {pattern!r}: {exc}")
        if bad:
            raise ValueError("invalid regex in crawl_exclude_patterns:\n  " + "\n  ".join(bad))
        return v

    @field_validator("ignore_dirs", mode="before")
    @classmethod
    def _merge_ignore_dirs(cls, v: Any) -> frozenset[str]:
        if isinstance(v, str):
            extra = frozenset(name.strip() for name in v.split(",") if name.strip())
            return DEFAULT_IGNORE_DIRS | extra
        if isinstance(v, (set, frozenset, list)):
            return DEFAULT_IGNORE_DIRS | frozenset(v)
        return DEFAULT_IGNORE_DIRS

    @field_validator("concept_allowed_ent_types", mode="before")
    @classmethod
    def _parse_ent_types(cls, v: Any) -> frozenset[str]:
        """Replace-semantics override: a narrowed set is used as-is,
        not unioned with defaults. A user asking for ``PERSON,ORG``
        wants exactly those kinds. Accepts comma-separated strings
        from env and list / set / frozenset from code. Empty input
        falls back to :data:`DEFAULT_ALLOWED_NER_LABELS` so an empty
        env var does not silently disable the gate.
        """
        if isinstance(v, str):
            parts = frozenset(name.strip().upper() for name in v.split(",") if name.strip())
            return parts or DEFAULT_ALLOWED_NER_LABELS
        if isinstance(v, (set, frozenset, list)):
            parts = frozenset(str(x).upper() for x in v)
            return parts or DEFAULT_ALLOWED_NER_LABELS
        return DEFAULT_ALLOWED_NER_LABELS

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: Any) -> Any:
        from lilbee.core.system import canonical_models_dir, default_data_dir, find_local_root

        if not isinstance(data, dict):  # pragma: no cover
            return data

        if data.get("data_root") in (None, _UNSET_PATH):
            data_env = os.environ.get("LILBEE_DATA", "").strip()
            if data_env:
                data["data_root"] = Path(data_env)
            else:
                local = find_local_root()
                data["data_root"] = local if local is not None else default_data_dir()
        root = data["data_root"]
        if data.get("documents_dir") in (None, _UNSET_PATH):
            data["documents_dir"] = root / "documents"
        if data.get("data_dir") in (None, _UNSET_PATH):
            data["data_dir"] = root / "data"
        if data.get("lancedb_dir") in (None, _UNSET_PATH):
            data["lancedb_dir"] = root / "data" / "lancedb"
        if data.get("models_dir") in (None, _UNSET_PATH):
            data["models_dir"] = canonical_models_dir()

        return data

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        from lilbee.core.system import default_data_dir, find_local_root

        data_env = os.environ.get("LILBEE_DATA", "")
        if data_env:
            toml_dir = Path(data_env)
        else:
            local = find_local_root()
            toml_dir = local if local else default_data_dir()
        toml_path = toml_dir / "config.toml"

        plain_env = _PlainEnvSource(settings_cls, env_prefix="LILBEE_", env_ignore_empty=True)
        sources: list[Any] = [init_settings, plain_env]
        if toml_path.exists() and os.environ.get("LILBEE_SKIP_TOML_CONFIG") != "1":
            sources.append(_TomlSource(settings_cls, toml_path))
        return tuple(sources)

    @property
    def model_defaults(self) -> Any:
        """Per-model generation defaults (read-only). Set via apply_model_defaults()."""
        return self._model_defaults

    def apply_model_defaults(self, defaults: Any) -> None:
        """Store per-model generation defaults for 3-layer merge."""
        object.__setattr__(self, "_model_defaults", defaults)

    def clear_model_defaults(self) -> None:
        """Reset per-model defaults to None."""
        object.__setattr__(self, "_model_defaults", None)

    def generation_options(self, **overrides: Any) -> dict[str, Any]:
        """Merge model defaults, user config, and per-call overrides, dropping None."""
        result = _model_defaults_dict(self._model_defaults)
        user_fields: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k_sampling,
            "repeat_penalty": self.repeat_penalty,
            "num_ctx": self.num_ctx,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }
        for k, v in user_fields.items():
            if v is not None:
                result[k] = v
        for k, v in overrides.items():
            if v is not None:
                result[k] = v
        return result


def _model_defaults_dict(defaults: Any) -> dict[str, Any]:
    """Non-None fields of a ModelDefaults instance as a dict."""
    if defaults is None:
        return {}
    from dataclasses import fields as dc_fields

    return {
        f.name: getattr(defaults, f.name)
        for f in dc_fields(defaults)
        if getattr(defaults, f.name) is not None
    }


class _PlainEnvSource:
    """Reads LILBEE_* env vars as plain strings so field validators handle parsing."""

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        env_prefix: str,
        env_ignore_empty: bool = True,
    ) -> None:
        self._prefix = env_prefix
        self._ignore_empty = env_ignore_empty
        self._fields = set(settings_cls.model_fields)

    def __call__(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field_name in self._fields:
            env_key = f"{self._prefix}{field_name.upper()}"
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            if self._ignore_empty and raw == "":
                continue
            result[field_name] = raw
        return result


class _TomlSource:
    """Custom pydantic-settings source that reads config.toml."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        self._path = path

    def __call__(self) -> dict[str, Any]:
        import tomllib

        try:
            with self._path.open("rb") as f:
                data = tomllib.load(f)
        except (ValueError, OSError):
            log.warning("Failed to read %s, ignoring", self._path)
            return {}
        return {k: str(v) for k, v in data.items()}


def _build_cfg() -> tuple[Config, Exception | None]:
    """Build cfg; on stale-config validation failure, fall back to defaults.

    A persisted ``config.toml`` from before a breaking schema change can
    contain values the new validators reject. Crashing at module import
    means every command (``lilbee --help`` included) emits a Python
    traceback. Falling back to env+defaults lets the package load; the
    CLI / TUI surfaces the original error before doing real work.
    """
    try:
        return Config(), None
    except Exception as exc:
        os.environ["LILBEE_SKIP_TOML_CONFIG"] = "1"
        try:
            return Config(), exc
        finally:
            os.environ.pop("LILBEE_SKIP_TOML_CONFIG", None)


cfg, config_load_error = _build_cfg()
