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

from lilbee.core.system import scaled_chat_ctx_target_default

from .defaults import (
    DEFAULT_ALLOWED_NER_LABELS,
    DEFAULT_CORS_ORIGIN_REGEX,
    DEFAULT_CRAWL_EXCLUDE_PATTERNS,
    DEFAULT_GENERAL_SYSTEM_PROMPT,
    DEFAULT_IGNORE_DIRS,
    DEFAULT_RAG_SYSTEM_PROMPT,
)
from .enums import (
    ChatMode,
    ClustererBackend,
    CrawlRenderMode,
    KvCacheType,
    LlmProvider,
    RerankerType,
    WikiEntityMode,
)
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
    # Markdown vault root; when set, search results carry a vault-relative
    # ``vault_path`` so a host UI can deep-link into the vault.
    vault_base: Path | None = ConfigField(default=None, writable=True)

    # Human-readable label for the active lilbee. Empty falls back to
    # "global" for the platform default dir, otherwise the project path
    # (~-substituted and left-truncated to a hard cap).
    lilbee_name: str = ConfigField(default="", writable=True)
    # If True, the status bar pill shows the full absolute path: expands
    # "global" to the on-disk platform-default path and skips the
    # ~-substitution / left-truncation for project paths. Toggled by F4.
    show_lilbee_path: bool = ConfigField(default=False, writable=True)

    # Whether an agent launcher (opencode, hermes) registers lilbee's MCP search
    # tool into the agent's config. Per-launch --mcp/--no-mcp overrides it.
    agent_mcp_enabled: bool = ConfigField(default=True, writable=True)

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
    # Gate for the pre-ask sync; --no-sync overrides per invocation.
    auto_sync: bool = ConfigField(default=True, writable=True)
    max_embed_chars: int = Field(default=2000, ge=1)
    top_k: int = ConfigField(default=12, ge=1, writable=True)
    max_distance: float = ConfigField(default=0.75, ge=0.0, writable=True)
    # Floor for hybrid-search relevance scores (0.0 = no filtering). lilbee
    # surfaces LanceDB's raw RRF sum, not a normalized score: with K=60 a
    # chunk ranked first in both the vector and FTS lists tops out near
    # 1/61 + 1/61 ~= 0.033, so any positive floor above that silently drops
    # every result. Keep this at 0.0 unless the RRF scores are normalized first.
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
    # Per-page timeout in seconds for vision OCR (0 = no limit). Sized so a dense
    # full-page scan finishes on modest hardware; a raised vision_ocr_max_tokens
    # needs matching headroom here.
    ocr_timeout: float = ConfigField(default=300.0, ge=0.0, writable=True)
    # Outer wall-clock budget for the streamed pool drain: load grace plus
    # per_page * pages. Tune up for slow hardware (M1 Pro vision is
    # ~5min/page) or down for fast hardware. ocr_timeout still governs the
    # per-page expectation that drives the total budget.
    vision_load_budget_s: float = ConfigField(default=300.0, ge=0.0, writable=True)
    # Hard cap on tokens generated per OCR page. A real page is well under this;
    # the cap bounds the occasional runaway repetition loop (a page that loops to
    # tens of thousands of chars) which otherwise dominates a scan's OCR time.
    # Raising it lengthens per-page generation on dense scans, so give ocr_timeout
    # matching headroom.
    vision_ocr_max_tokens: int = ConfigField(default=4096, ge=256, writable=True)
    # Pages OCR'd concurrently, and the vision server's continuous-batching slots.
    # A single-page decode underutilizes a modern GPU (~half SM); batching several
    # pages raises throughput. Each slot adds KV cache, so lower it on small GPUs.
    vision_ocr_concurrency: int = ConfigField(default=4, ge=1, writable=True)

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
    temperature: float | None = ConfigField(default=0.1, ge=0.0, writable=True)
    top_p: float | None = ConfigField(default=0.9, ge=0.0, le=1.0, writable=True)
    top_k_sampling: int | None = ConfigField(default=40, ge=1, writable=True)
    # 1.1 is llama.cpp's default. Leaving this at None caused n-gram loops
    # ("tire tire tire...") on some open-weights models.
    repeat_penalty: float | None = ConfigField(default=1.1, ge=0.0, writable=True)
    num_ctx: int | None = ConfigField(default=None, ge=1, writable=True)
    max_tokens: int | None = ConfigField(default=4096, ge=1, writable=True)
    seed: int | None = ConfigField(default=None, writable=True)
    llm_provider: LlmProvider = ConfigField(default=LlmProvider.AUTO, writable=True)
    # Path to a llama-server binary. Empty = use the bundled lilbee-engine
    # wheel binary, else a llama-server on PATH.
    llama_server_path: str = ConfigField(default="", writable=True)
    # Per-server local model-manager URLs. Blank means "use the server's spec
    # default" (resolved in providers.local_servers.config_urls); the default
    # URL literal lives only in the spec, which core must not import.
    ollama_base_url: str = ConfigField(default="", writable=True)
    lm_studio_base_url: str = ConfigField(default="", writable=True)
    llm_api_key: str = ConfigField(default="", writable=True, write_only=True)
    openrouter_api_key: str = ConfigField(default="", writable=True, write_only=True)
    gemini_api_key: str = ConfigField(default="", writable=True, write_only=True)
    anthropic_api_key: str = ConfigField(default="", writable=True, write_only=True)
    openai_api_key: str = ConfigField(default="", writable=True, write_only=True)
    mistral_api_key: str = ConfigField(default="", writable=True, write_only=True)
    deepseek_api_key: str = ConfigField(default="", writable=True, write_only=True)
    hf_token: str = ConfigField(default="", writable=True, write_only=True)

    # Retrieval quality knobs.

    # Max chunks per source in top-k; prevents one large file monopolizing results.
    diversity_max_per_source: int = ConfigField(default=5, ge=1, writable=True)

    # MMR relevance/diversity tradeoff; 0 = max diversity, 1 = pure relevance
    # (Carbonell & Goldstein 1998).
    mmr_lambda: float = ConfigField(default=0.5, ge=0.0, le=1.0, writable=True)

    # Extra candidates retrieved for MMR reranking (multiplies top_k).
    candidate_multiplier: int = ConfigField(default=3, ge=1, writable=True)

    # Chunk count at/above which sync builds an approximate (ANN) vector index
    # so search stays fast at millions of vectors. Below this, search uses exact
    # flat scan (faster and exact for small vaults). 0 disables the ANN index.
    ann_index_threshold: int = ConfigField(default=50_000, ge=0, writable=True)

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
    max_context_sources: int = ConfigField(default=8, ge=1, writable=True)

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

    # Reranker model ref. Empty disables reranking. Native GGUFs run on
    # llama-server (rank pooling or LLM logprob scoring); hosted refs
    # (cohere/voyage/jina/together/hf-tei) need the backend extra.
    reranker_model: str = ConfigField(default="", public=True)

    # auto detects cross-encoder vs LLM reranker by GGUF arch; override forces one.
    reranker_type: RerankerType = ConfigField(default=RerankerType.AUTO, writable=True, public=True)
    # Relevance prompt for LLM rerankers; empty uses the built-in generic template.
    # A format string with {query} and {document} placeholders.
    reranker_prompt: str = ConfigField(default="", writable=True, public=True)

    # Long-term chat memory. Off by default (opt-in): when disabled the whole
    # subsystem is dormant and the write surfaces respond with an enable hint.
    memory_enabled: bool = ConfigField(default=False, writable=True)

    # Facts recalled by similarity per turn (preferences are always injected).
    memory_top_k: int = ConfigField(default=5, ge=0, writable=True)

    # Cosine-distance ceiling for fact recall; stricter than the document default
    # because a tiny memory corpus floods at the wider document threshold.
    memory_max_distance: float = ConfigField(default=0.6, ge=0.0, le=1.0, writable=True)

    # Char/4 token budget for the injected memory block.
    memory_token_budget: int = ConfigField(default=512, ge=0, writable=True)

    # Per-owner soft cap; oldest memories evicted past it (runaway-write guard).
    memory_max_per_owner: int = ConfigField(default=200, ge=1, writable=True)

    # Cosine distance below which a new memory is treated as a duplicate of an
    # existing same-owner memory and updates it in place instead of inserting.
    memory_dedup_distance: float = ConfigField(default=0.05, ge=0.0, le=1.0, writable=True)

    # LLM pass that extracts memories from the chat loop. Off by default; extracted
    # memories are saved directly and recalled like any other memory.
    memory_auto_extract: bool = ConfigField(default=False, writable=True)

    # Candidate count sent to the reranker.
    rerank_candidates: int = ConfigField(default=60, ge=1, writable=True, public=True)

    # Date-range filter; only fires when a temporal keyword is detected.
    temporal_filtering: bool = ConfigField(default=True, writable=True)

    # If True, emit <think>…</think> content as separate SSE reasoning events;
    # if False, strip it silently.
    show_reasoning: bool = ConfigField(default=False, writable=True)

    # Maximum reasoning characters before lilbee forces the model to answer.
    # Per-model overrides apply on top of this default. Approx N/4 tokens.
    # 0 disables the cap (unlimited reasoning; accept the runaway-loop risk).
    max_reasoning_chars: int = ConfigField(default=64_000, ge=0, writable=True)

    # Web crawling.

    # How crawls fetch pages. ``http`` (default) uses a plain HTTP client with
    # no browser, the lightweight path for static / server-rendered sites.
    # ``browser`` launches a tuned Chromium with JavaScript enabled for sites
    # that render content client-side, at a much higher memory cost.
    crawl_render_mode: CrawlRenderMode = ConfigField(default=CrawlRenderMode.HTTP, writable=True)

    # Browser-mode memory levers (only used when crawl_render_mode is browser).
    # Recycle the Chromium process every N fetched pages to cap RSS growth on a
    # long recursive crawl; 0 disables recycling. Raise on a roomy machine for
    # fewer restarts, lower it if memory is tight.
    crawl_browser_recycle_pages: int = ConfigField(default=50, ge=0, writable=True)

    # Extra Chromium launch flags for browser-mode crawls. Defaults trim shared
    # memory and GPU use; override to pass site- or environment-specific flags.
    crawl_browser_extra_args: list[str] = ConfigField(
        default_factory=lambda: ["--disable-dev-shm-usage", "--disable-gpu"],
        writable=True,
    )

    # Optional global ceilings. None = no ceiling.
    crawl_max_depth: int | None = ConfigField(default=None, ge=0, writable=True)
    crawl_max_pages: int | None = ConfigField(default=None, ge=1, writable=True)

    # Default page bound for an unbounded crawl (no explicit max_pages /
    # crawl_max_pages), so a hostile site can't exhaust the disk by default.
    # An explicit limit overrides it; raise this to crawl larger sites unbounded.
    crawl_safety_max_pages: int = ConfigField(default=5_000, ge=1, writable=True)

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

    # Data-parallel replicas of the embed / vision role across GPUs: N independent
    # servers, round-robined, so large-scale ingest fans the embedding / OCR work
    # across the whole box. 0 means "auto": one replica per detected GPU, capped by
    # the VRAM left after the persistent query fleet (chat, one embed, rerank, one
    # vision) is reserved. A positive value pins the count. The extra replicas are
    # ingest-only and reclaimed when ingest ends; the persistent query embedder /
    # vision (replica 0) always exists if its model fits.
    embed_replicas: int = ConfigField(default=0, ge=0, writable=True)
    vision_replicas: int = ConfigField(default=0, ge=0, writable=True)

    # Seconds a model stays loaded after last use. 0 = unload immediately.
    model_keep_alive: int = ConfigField(default=300, ge=0, writable=True)

    # Spawn every configured role server at startup instead of on first use.
    # Trades a slower TUI mount (the role servers cold-start in parallel) for a
    # responsive first interaction. Roles whose model is unset are skipped, so a
    # setup with only chat + embed never spawns rerank or vision. Set to false
    # for headless / scripted use where the first call doesn't need to be fast.
    worker_pool_eager_start: bool = ConfigField(default=True, writable=True)

    # Working n_ctx the dynamic picker aims for. Default scales with
    # total host RAM (see core.system.chat_ctx_target_for_total_bytes):
    # <16 GiB -> 8192, 16-32 -> 12288, 32-64 -> 16384, >=64 -> 24576.
    # 8192 is the floor; the picker still clamps to training_ctx and
    # host headroom.
    chat_n_ctx_target: int = ConfigField(
        default_factory=scaled_chat_ctx_target_default,
        ge=512,
        writable=True,
    )

    # Explicit ceiling for the dynamic n_ctx picker. ``None`` (default)
    # lets the model's training_ctx from GGUF metadata be the ceiling,
    # so a 128K-context model can reach for it on a host with the RAM
    # to back it. Set explicitly to cap below the model's training_ctx.
    num_ctx_max: int | None = ConfigField(default=None, ge=512, writable=True)

    # Flash attention. None (default) = on, True = force on, False = off
    # for backends or models where it misbehaves.
    # Resolves the 'padding V cache to 1024' warning on models with
    # uneven per-layer V dims (e.g. Gemma3) and saves ~25% KV memory.
    flash_attention: bool | None = ConfigField(default=None, writable=True)

    # KV cache element type. q8_0 (default) halves cache memory vs f16
    # with no measurable quality loss for chat; q4_0 quarters it with a
    # small quality cost. Both require flash attention to be enabled.
    kv_cache_type: KvCacheType = ConfigField(default=KvCacheType.Q8_0, writable=True)

    # Number of model layers to offload to GPU. None (default) = all
    # layers, 0 = CPU only, positive int = partial offload. Useful when a
    # discrete GPU has less VRAM than the model needs.
    n_gpu_layers: int | None = ConfigField(default=None, writable=True)

    # GPU device picker for dual-GPU machines (typical laptop case:
    # discrete NVIDIA + integrated Intel/AMD). The Vulkan backend
    # enumerates every adapter the system exposes and may pick the
    # integrated one first, producing stalls or OOMs that look like
    # llama.cpp bugs. Setting ``gpu_devices`` constrains visibility
    # before the servers spawn, pinning inference to the chosen device(s).
    #
    # Accepts a comma-separated list of device indexes ("0", "1",
    # "0,1") and applies it to every backend simultaneously:
    # ``GGML_VK_VISIBLE_DEVICES`` for Vulkan, ``CUDA_VISIBLE_DEVICES``
    # for CUDA, ``HIP_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES`` for
    # ROCm. Setting one variable that the active backend ignores is
    # harmless, so we set all four rather than detecting the build.
    #
    # Must be set before the first llama.cpp call; in practice that
    # means via ``LILBEE_GPU_DEVICES`` or ``config.toml`` (TUI edits
    # only take effect after a restart). ``None`` (default) hands off
    # to the autodetect in ``providers/fleet/gpu_select.py``,
    # which parses ``vulkaninfo --summary`` and pins the discrete
    # adapter when one is present. The autodetect is silent on failure
    # (no vulkaninfo, single device, parse error), leaving the
    # Vulkan-loader's default ordering in place.
    gpu_devices: str | None = ConfigField(default=None, writable=True)

    # Primary GPU index passed to ``Llama(main_gpu=...)``. Only matters
    # when multiple devices remain visible after ``gpu_devices``; with
    # a single visible device, llama.cpp ignores this. ``None``
    # (default) lets llama.cpp pick (index 0).
    main_gpu: int | None = ConfigField(default=None, writable=True)

    # Manual GPU placement override stored as a JSON scalar (the config.toml store
    # is flat, and core must not depend on the provider PlacementSpec type). When
    # set, it fully replaces the automatic placement planner: each active role pins
    # to the listed device indices, with an optional tensor_split and replica count.
    # Edited via the placement CLI/MCP/HTTP/TUI surfaces rather than the generic
    # settings list, so public=False. None hands off to the VRAM-aware auto planner.
    placement: str | None = ConfigField(default=None, writable=True, public=False)

    # Allow PUT/DELETE /api/placement to apply or clear placement over HTTP.
    # Off by default because applying placement restarts the shared fleet's moved roles, which
    # is unsafe across concurrent HTTP clients. Turn it on (LILBEE_ALLOW_HTTP_PLACEMENT=1)
    # only for a single-client / owned deployment: the plugin's managed local
    # server, or a personally-owned pod where one operator runs `lilbee serve`.
    allow_http_placement: bool = Field(default=False)

    # True = Markdown widget for chat; False = plain Static (faster).
    markdown_rendering: bool = True

    # TUI theme name; persists the last Ctrl+T pick across sessions.
    theme: str = ConfigField(default="rose-pine", writable=True)

    # Per-model generation defaults set via apply_model_defaults().
    _model_defaults: Any = None

    # Wiki layer. LLM-maintained synthesis pages with citation provenance.
    # Off by default; flip to True (or set LILBEE_WIKI=1) to enable. When off,
    # the Wiki view tab and the chat ModelBar's scope picker are both hidden.
    wiki: bool = ConfigField(default=False, writable=True)
    # Read-only: changing the directory at runtime strands prior wiki pages
    # under the old path. Users who want a different location set it via
    # LILBEE_WIKI_DIR / config.toml before the first wiki_build.
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
    wiki_stale_citation_threshold: float = ConfigField(default=0.5, ge=0.0, le=1.0, writable=True)

    # Fraction of content changed that triggers human-review drift guard.
    wiki_drift_threshold: float = ConfigField(default=0.3, ge=0.0, le=1.0, writable=True)

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
    concept_max_per_chunk: int = ConfigField(default=5, ge=1, writable=True)

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

    @field_validator("lilbee_name", mode="after")
    @classmethod
    def _strip_lilbee_name(cls, value: str) -> str:
        """Strip whitespace; an empty string signals 'use the path-derived label'."""
        return value.strip()

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
                pass  # fall through to bool() coercion below for unrecognised strings
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
                log.warning("Invalid flash_attention=%r, using auto", v)
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

    @field_validator("main_gpu", mode="before")
    @classmethod
    def _parse_main_gpu(cls, v: Any) -> int | None:
        """Empty/auto strings -> None, integers parsed verbatim."""
        if v is None:
            return None
        if isinstance(v, str):
            label = v.strip().lower()
            if label in ("", "auto", "none"):
                return None
            try:
                return int(label)
            except ValueError:
                log.warning("Invalid LILBEE_MAIN_GPU=%r, using auto", v)
                return None
        return int(v)

    @field_validator("gpu_devices", mode="before")
    @classmethod
    def _parse_gpu_devices(cls, v: Any) -> str | None:
        """Normalize device list: strip whitespace, drop empties, keep order."""
        if v is None:
            return None
        if isinstance(v, str):
            label = v.strip().lower()
            if label in ("", "auto", "all", "none"):
                return None
            parts = [p.strip() for p in v.split(",") if p.strip()]
            if not parts:
                return None
            for part in parts:
                if not part.lstrip("-").isdigit():
                    log.warning("Invalid LILBEE_GPU_DEVICES=%r, ignoring", v)
                    return None
            return ",".join(parts)
        return str(v)

    @field_validator("placement", mode="before")
    @classmethod
    def _parse_placement(cls, v: Any) -> str | None:
        """Blank/None -> None; validate a JSON string or PlacementSpec; store JSON."""
        from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec

        if v is None:
            return None
        if isinstance(v, PlacementSpec):
            json_str = v.to_json()
            PlacementSpec.from_json(json_str)  # re-validate a directly-built spec
            return json_str
        if isinstance(v, str):
            if v.strip() == "":
                return None
            PlacementSpec.from_json(v)
            return v
        raise PlacementError("placement must be a JSON string or PlacementSpec")

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

    @field_validator("ollama_base_url", "lm_studio_base_url", mode="after")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        """Canonicalize a local-server URL once at the write boundary."""
        return v.rstrip("/")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("crawl_browser_extra_args", mode="before")
    @classmethod
    def _split_crawl_browser_extra_args(cls, v: Any) -> Any:
        """Accept a newline-separated string, matching how the field is persisted.

        ``app.settings`` joins list values with newlines before writing them to
        ``config.toml`` as a scalar string. Without this inverse, reload cannot
        coerce that string to ``list[str]`` and the whole config.toml is dropped.
        TOML lists and JSON arrays pass through unchanged.
        """
        if isinstance(v, str):
            return [a.strip() for a in v.splitlines() if a.strip()]
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

        if not isinstance(data, dict):
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
        # Empty strings represent "no persisted value" for nullable scalar
        # fields (legacy from set_setting writing "" for None). Pydantic
        # can't coerce "" to int|None, so dropping them here lets the field
        # default apply rather than crashing the whole Config load. TOML's
        # native types (lists, ints, bools) pass through untouched: stringifying
        # turned a list field's ["a", "b"] into the literal "['a', 'b']".
        return {k: v for k, v in data.items() if v != ""}


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

# Canonicalize LILBEE_DATA at the cfg.data_root resolution boundary so
# spawn-context worker subprocesses inherit the same data root.
# ``setdefault`` preserves a user-set value.
os.environ.setdefault("LILBEE_DATA", str(cfg.data_root))
