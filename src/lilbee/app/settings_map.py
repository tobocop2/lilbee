"""Shared settings map for interactive configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic_core import PydanticUndefined

from lilbee.app.themes import DARK_THEMES
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode, ClustererBackend, KvCacheType, WikiEntityMode


class RenderStyle(StrEnum):
    """How a setting is displayed in /settings."""

    COMPACT = "compact"
    FULL = "full"
    LIST_COLLAPSED = "list_collapsed"
    MULTILINE = "multiline"


class SettingGroup(StrEnum):
    """Logical bucket names rendered by ``/settings`` and ``settings_list``."""

    MODELS = "Models"
    GENERATION = "Generation"
    RETRIEVAL = "Retrieval"
    INGEST = "Ingest"
    WIKI = "Wiki"
    MEMORY = "Memory"
    CRAWLING = "Crawling"
    LOCAL_SERVERS = "Local-Servers"
    API_KEYS = "API-Keys"
    SYSTEM = "System"
    DISPLAY = "Display"
    GENERAL = "General"


@dataclass(frozen=True)
class SettingDef:
    """Metadata for an interactive setting.

    ``writable`` is a TUI rendering hint: fields marked ``writable=False``
    (the model role slots) get a dedicated picker rather than an inline
    editor, and the ``/set`` slash command refuses them. The actual
    write contract for HTTP / MCP / programmatic surfaces lives in
    ``config_meta.WRITABLE_CONFIG_FIELDS`` + ``MODEL_ROLE_FIELDS`` and
    is enforced by ``app.settings.apply_settings_update``.

    ``hidden`` keeps the setting out of the TUI settings screen while
    leaving it reachable via ``lilbee set`` and the ``LILBEE_*`` env
    var: use it for transport/server knobs that aren't relevant to a
    typical TUI session.
    """

    type: type
    nullable: bool
    writable: bool = True
    render: RenderStyle = field(default=RenderStyle.COMPACT)
    group: SettingGroup = SettingGroup.GENERAL
    help_text: str = ""
    choices: tuple[str, ...] | None = None
    hidden: bool = False


def get_default(key: str) -> object:
    """Return the cfg default for a setting key."""
    field_info = type(cfg).model_fields[key]
    if field_info.default_factory is not None:
        return field_info.default_factory()  # type: ignore[call-arg]
    if field_info.default is PydanticUndefined:
        return None
    return field_info.default


SETTINGS_MAP: dict[str, SettingDef] = {
    "chat_model": SettingDef(
        str,
        nullable=False,
        writable=False,
        group=SettingGroup.MODELS,
        help_text="LLM used for chat generation (vision and reranking are separate slots)",
    ),
    "vision_model": SettingDef(
        str,
        nullable=True,
        writable=False,
        group=SettingGroup.MODELS,
        help_text="Vision model for scanned PDF OCR (empty = disabled; Tesseract only)",
    ),
    "enable_ocr": SettingDef(
        bool,
        nullable=True,
        group=SettingGroup.INGEST,
        help_text="Vision OCR for scanned PDFs (empty = auto-detect from vision_model)",
    ),
    "ocr_timeout": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Per-page timeout in seconds for vision OCR (0 = no limit)",
    ),
    "vision_load_budget_s": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text=(
            "Wall-clock seconds reserved for the vision worker to load the"
            " model. Total PDF-OCR budget = load_budget + ocr_timeout * pages."
        ),
    ),
    "semantic_chunking": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Opt-in topic-aware chunker (default off; may fragment numbered procedures)",
    ),
    "topic_threshold": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Topic-boundary similarity threshold, 0.0-1.0, used when semantic chunking is on",
    ),
    "embedding_model": SettingDef(
        str,
        nullable=False,
        writable=False,
        group=SettingGroup.MODELS,
        help_text="Model used to embed document chunks",
    ),
    "reranker_model": SettingDef(
        str,
        nullable=True,
        writable=False,
        group=SettingGroup.MODELS,
        help_text="Cross-encoder model for result reranking",
    ),
    "temperature": SettingDef(
        float,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Sampling temperature (higher = more creative)",
    ),
    "top_p": SettingDef(
        float,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Nucleus sampling cutoff probability",
    ),
    "top_k_sampling": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Top-K sampling: number of tokens to consider",
    ),
    "repeat_penalty": SettingDef(
        float,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Penalty for repeating tokens",
    ),
    "num_ctx": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Context window size in tokens. Leave empty to size automatically "
            "(aims for chat_n_ctx_target, ceiling at num_ctx_max or training_ctx)."
        ),
    ),
    "num_ctx_max": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Explicit ceiling for the dynamic context picker. Leave empty to "
            "use the model's training_ctx from GGUF metadata as the only "
            "ceiling. Set to cap below training_ctx (saves KV memory)."
        ),
    ),
    "chat_n_ctx_target": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "Working context the dynamic picker aims for. Fits a RAG turn "
            "with reasoning headroom; raise for long-document chat."
        ),
    ),
    "flash_attention": SettingDef(
        bool,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Flash attention. Empty (auto) tries it on with a fallback for older "
            "llama-cpp-python builds; resolves the V-cache padding warning on "
            "models with uneven per-layer V dims."
        ),
    ),
    "kv_cache_type": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "KV cache element type. q8_0 / q4_0 halve or quarter cache memory "
            "but require flash attention to be enabled."
        ),
        choices=tuple(t.value for t in KvCacheType),
    ),
    "n_gpu_layers": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Layers to offload to GPU. Empty = all (recommended), 0 = CPU only, "
            "positive int = partial offload for tight VRAM."
        ),
    ),
    "gpu_devices": SettingDef(
        str,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Restrict llama.cpp to specific GPU indexes on dual-GPU machines "
            "(e.g. NVIDIA dGPU + integrated). Comma-separated, like '0' or '0,1'. "
            "Applies to Vulkan, CUDA, and ROCm. Requires a restart to take effect."
        ),
    ),
    "main_gpu": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Primary GPU index for llama.cpp when multiple devices are visible. "
            "Empty = let llama.cpp pick (index 0). Set this together with "
            "gpu_devices to pin inference to a specific card."
        ),
    ),
    "seed": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Random seed for reproducible output",
    ),
    "rag_system_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.MULTILINE,
        group=SettingGroup.GENERATION,
        help_text="System prompt sent when answering with retrieved context",
    ),
    "general_system_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.MULTILINE,
        group=SettingGroup.GENERATION,
        help_text="System prompt sent when there are no documents to ground the answer",
    ),
    "chat_mode": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.GENERATION,
        choices=tuple(m.value for m in ChatMode),
        help_text="search runs every chat turn through document retrieval; chat skips it",
    ),
    "top_k": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Number of chunks returned by search",
    ),
    "rerank_candidates": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Candidate pool size for reranking",
    ),
    "show_reasoning": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.DISPLAY,
        help_text="Show model reasoning/thinking tokens in output",
    ),
    "lilbee_name": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.DISPLAY,
        help_text=(
            "Human-readable label for this lilbee, shown in the status bar. "
            "Empty falls back to 'global' for the platform default dir or "
            "to the project path (~-substituted and left-truncated)."
        ),
    ),
    "show_lilbee_path": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.DISPLAY,
        help_text=(
            "Show the full absolute path in the status bar: expands 'global' "
            "to its on-disk path and skips ~ substitution / truncation."
        ),
    ),
    "theme": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.DISPLAY,
        help_text="TUI color theme. Cycle with Ctrl+T; the active theme persists across sessions.",
        choices=tuple(DARK_THEMES),
    ),
    "wiki": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Enable the wiki layer (synthesis pages with citations)",
    ),
    "wiki_dir": SettingDef(
        str,
        nullable=False,
        writable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Directory under data_root where wiki pages live (set via env / config.toml only)"
        ),
    ),
    "wiki_prune_raw": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Delete raw chunks after summarizing into the wiki",
    ),
    "wiki_embedding_faithfulness_threshold": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Minimum cosine similarity (0-1) between a generated page and "
            "the mean of its source chunk vectors before publishing. "
            "Pages below the threshold route to drafts/."
        ),
    ),
    "wiki_stale_citation_threshold": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Fraction of stale citations that triggers page regeneration",
    ),
    "wiki_drift_threshold": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Max fraction of changed lines before regeneration requires review",
    ),
    "wiki_clusterer": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Synthesis clusterer backend (embedding or concepts)",
        choices=tuple(b.value for b in ClustererBackend),
    ),
    "wiki_entity_mode": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Entity extraction strategy "
            "(ner_entities = default, typed NER entities; "
            "plus_llm_types = NER + LLM-proposed schema; "
            "llm_tagged = LLM tags every chunk)"
        ),
        choices=tuple(m.value for m in WikiEntityMode),
    ),
    "wiki_entity_min_mentions": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Minimum chunk mentions before an entity or concept gets its own page",
    ),
    "wiki_concept_max_chunks_per_page": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Maximum chunks passed into each concept or entity page generation call",
    ),
    "wiki_related_max": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Maximum related concepts listed in the `## Related` section of each page",
    ),
    "wiki_ingest_update_cap": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Touched-page cap for auto-update after sync. "
            "Beyond this count, run `lilbee wiki update` manually."
        ),
    ),
    "wiki_summary_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.FULL,
        group=SettingGroup.WIKI,
        help_text=(
            "Prompt for per-source summary pages. "
            "Must keep the {source_name} and {chunks_text} placeholders."
        ),
    ),
    "wiki_synthesis_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.FULL,
        group=SettingGroup.WIKI,
        help_text=(
            "Prompt for cross-source synthesis pages. "
            "Must keep {topic}, {source_list}, and {chunks_text}."
        ),
    ),
    "wiki_entity_batch_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.FULL,
        group=SettingGroup.WIKI,
        help_text=(
            "Prompt for the per-source batched call. "
            "Must keep {source}, {entity_list}, {chunks_text}, and {concept_instruction}."
        ),
    ),
    "wiki_extract_concepts": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Whether the per-source batched call asks the LLM to curate concept pages "
            "alongside the pre-extracted entity list."
        ),
    ),
    "wiki_batch_min_chunks": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Minimum chunks a source must contribute before its batched call includes "
            "concept curation. Sources below the floor skip the concept-curation "
            "instruction; sources with zero entities AND below the floor are skipped entirely."
        ),
    ),
    "wiki_clusterer_k": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Mutual-kNN neighborhood size for the clusterer (0 = auto)",
    ),
    "memory_enabled": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Master switch for long-term chat memory (off by default)",
    ),
    "memory_auto_extract": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Auto-save durable facts and preferences from each TUI turn (needs memory on)",
    ),
    "memory_top_k": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Maximum facts recalled into context per turn",
    ),
    "memory_max_distance": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Recall cutoff distance, 0.0-1.0 (lower is stricter)",
    ),
    "memory_token_budget": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Token cap on the recalled-memory block added to the prompt",
    ),
    "memory_max_per_owner": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Soft cap before the oldest memories are evicted",
        hidden=True,
    ),
    "memory_dedup_distance": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Near-duplicate distance below which a new memory updates the old",
        hidden=True,
    ),
    "crawl_max_depth": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.CRAWLING,
        help_text="Optional recursion-depth cap (blank = no cap; per-crawl values win)",
    ),
    "crawl_max_pages": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.CRAWLING,
        help_text="Optional global cap on total pages per crawl (blank = no cap).",
    ),
    "crawl_safety_max_pages": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Default page bound for an unbounded crawl, so a hostile site cannot "
        "exhaust the disk. An explicit max-pages overrides it; raise this to crawl "
        "larger sites unbounded.",
    ),
    "crawl_timeout": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Per-page fetch timeout in seconds",
    ),
    "crawl_sync_interval": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Seconds between periodic re-syncs during a crawl (0 = sync only at end)",
    ),
    "crawl_mean_delay": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Seconds between in-flight requests within a single crawl",
    ),
    "crawl_max_delay_range": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Random jitter (seconds) added on top of mean delay",
    ),
    "crawl_concurrent_requests": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Concurrent in-flight URLs within one crawl",
    ),
    "crawl_retry_on_rate_limit": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Enable per-domain backoff and retries on HTTP 429/503",
    ),
    "crawl_retry_base_delay_min": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Minimum base-delay (seconds) on rate-limit responses",
    ),
    "crawl_retry_base_delay_max": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Maximum base-delay (seconds) on rate-limit responses",
    ),
    "crawl_retry_max_backoff": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Upper bound on any single backoff wait (seconds)",
    ),
    "crawl_retry_max_attempts": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Retry count per URL when a rate-limit code comes back",
    ),
    "crawl_exclude_patterns": SettingDef(
        list,
        nullable=False,
        group=SettingGroup.CRAWLING,
        render=RenderStyle.LIST_COLLAPSED,
        help_text=(
            "Regex patterns that skip URLs at link-discovery time during "
            "recursive crawls. One per line."
        ),
    ),
    "openrouter_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        help_text="OpenRouter API key (enables frontier models in chat picker)",
    ),
    "gemini_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        help_text="Google Gemini API key (enables frontier models in chat picker)",
    ),
    "anthropic_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        help_text="Anthropic API key (enables frontier models in chat picker)",
    ),
    "openai_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        help_text="OpenAI API key (enables frontier models in chat picker)",
    ),
    "mistral_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        help_text="Mistral API key (enables frontier models in chat picker)",
    ),
    "deepseek_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        help_text="DeepSeek API key (enables frontier models in chat picker)",
    ),
    "hf_token": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.SYSTEM,
        help_text=(
            "HuggingFace access token. Avoids the unauthenticated download "
            "rate limit and unlocks gated repos. Stored in plain text in "
            "config.toml. Env vars (LILBEE_HF_TOKEN, HF_TOKEN) override."
        ),
    ),
    "chunk_size": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Document chunk size in tokens (changes invalidate the index)",
    ),
    "chunk_overlap": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Tokens of overlap between adjacent chunks (preserves context across boundaries)",
    ),
    "tesseract_timeout": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Per-page Tesseract timeout in seconds (used when no vision model is set)",
    ),
    "worker_pool_call_timeout_s": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text=(
            "Per-call deadline for one worker-pool round-trip in seconds. "
            "Raise this for very large embed batches on slow machines"
        ),
    ),
    "worker_pool_eager_start": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text=(
            "Spawn every registered worker at TUI startup instead of on first use. "
            "Trades 1-3 seconds of cold-start per role for first-call latency"
        ),
    ),
    "worker_pool_max_idle_s": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text=(
            "Shut a worker down after this many seconds idle to free RAM/VRAM. "
            "0 disables idle reaping"
        ),
    ),
    "max_tokens": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Hard cap on generated tokens per response (blank = no cap)",
    ),
    "max_reasoning_chars": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "Maximum reasoning characters before lilbee forces the model to answer "
            "(0 = unlimited; per-model overrides apply on top)"
        ),
    ),
    "model_keep_alive": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text="Seconds the loaded model stays warm between calls (0 = unload immediately)",
    ),
    "gpu_memory_fraction": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text="Fraction of GPU memory the model is allowed to claim (0.1-1.0)",
    ),
    "candidate_multiplier": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Candidate-pool multiplier over top_k before reranking",
    ),
    "ann_index_threshold": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Chunk count to start building an ANN vector index (0 = always flat search)",
    ),
    "max_distance": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Maximum vector distance for retrieval matches (lower = stricter)",
    ),
    "min_relevance_score": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Minimum RRF relevance score for hybrid search results (0.0 = no filter)",
    ),
    "max_context_sources": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Maximum unique sources contributing chunks to a single answer",
    ),
    "diversity_max_per_source": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Maximum chunks accepted from any one source (caps source dominance)",
    ),
    "mmr_lambda": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text=(
            "MMR lambda balancing relevance vs diversity (0 = max diversity, 1 = max relevance)"
        ),
    ),
    "temporal_filtering": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Detect temporal queries and bias retrieval toward recent chunks",
    ),
    "hyde": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Use HyDE (hypothetical answer expansion) to broaden retrieval",
    ),
    "hyde_weight": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Weight on the HyDE-generated query vector when blending with the original",
    ),
    "query_expansion_count": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Number of paraphrase expansions per query (0 disables expansion)",
    ),
    "expansion_similarity_threshold": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Minimum cosine similarity an expansion must keep with the original query",
    ),
    "expansion_short_query_tokens": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Queries at or below this token count skip expansion (saves a model call)",
    ),
    "expansion_guardrails": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Drop expansions that diverge from the original intent",
    ),
    "adaptive_threshold_step": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Step size for adaptive relevance-score relaxation when initial recall is empty",
    ),
    "concept_graph": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Boost retrieval scores for chunks that share concepts with the query",
    ),
    "concept_boost_weight": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Maximum boost (0-1) the concept graph can add to a chunk's relevance",
    ),
    "concept_boost_floor": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Minimum cosine similarity needed before the concept graph boosts a chunk",
    ),
    "concept_max_per_chunk": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Maximum concept tags stored per chunk (caps graph density)",
    ),
    "documents_dir": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.SYSTEM,
        help_text="Local documents root that lilbee sync ingests (blank = data_root/documents)",
    ),
    "vault_base": SettingDef(
        str,
        nullable=True,
        group=SettingGroup.SYSTEM,
        help_text="Markdown vault root; results carry a vault-relative path (blank = none)",
    ),
    "sse_heartbeat_interval": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.SYSTEM,
        help_text="Seconds between SSE keep-alive frames sent to idle HTTP stream clients",
        hidden=True,
    ),
    "llm_provider": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        choices=("auto", "llama-cpp", "remote"),
        help_text=(
            "Provider routing: auto picks the first key present; force a specific one when set"
        ),
    ),
    "ollama_base_url": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.LOCAL_SERVERS,
        help_text="Ollama server URL (blank uses http://localhost:11434)",
    ),
    "lm_studio_base_url": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.LOCAL_SERVERS,
        help_text="LM Studio server URL (blank uses http://localhost:1234/v1)",
    ),
    "wiki_summary_max_tokens": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Maximum tokens generated per wiki page",
    ),
    "wiki_temperature": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Temperature used for wiki page synthesis (low = stay close to sources)",
    ),
}
