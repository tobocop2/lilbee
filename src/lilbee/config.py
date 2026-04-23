"""Application configuration for lilbee.

All settings can be overridden via environment variables prefixed with LILBEE_.
Uses pydantic-settings for automatic env var loading with TOML config file support.
"""

import logging
import os
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClustererBackend(StrEnum):
    """Known wiki clusterer backends."""

    EMBEDDING = "embedding"
    CONCEPTS = "concepts"


def ConfigField(
    *args: Any,
    writable: bool = False,
    reindex: bool = False,
    write_only: bool = False,
    public: bool = True,
    **kwargs: Any,
) -> Any:
    """Wrap pydantic ``Field`` and attach metadata via ``json_schema_extra``."""
    extra: dict[str, bool] = {}
    if writable:
        extra["writable"] = True
    if reindex:
        extra["reindex"] = True
    if write_only:
        extra["write_only"] = True
    if not public:
        extra["public"] = False
    if extra:
        kwargs["json_schema_extra"] = extra
    return Field(*args, **kwargs)


log = logging.getLogger(__name__)

# Skips the per-role catalog-task check when set AND pytest is imported.
# The two-signal gate prevents a leaked env var from disabling role
# validation in production.
_SKIP_MODEL_TASK_VALIDATION_ENV = "LILBEE_SKIP_MODEL_TASK_VALIDATION"


def _model_task_validation_bypassed() -> bool:
    """True iff the task-validator should be skipped (test fixtures only)."""
    if not os.environ.get(_SKIP_MODEL_TASK_VALIDATION_ENV):
        return False
    return sys.modules.get("pytest") is not None


_MODEL_FIELD_TO_TASK: dict[str, str] = {
    "chat_model": "chat",
    "embedding_model": "embedding",
    "vision_model": "vision",
    "reranker_model": "rerank",
}


def _find_model_catalog_entry(ref: str) -> Any:
    """Look up *ref* in the featured catalog.

    ``find_catalog_entry`` handles the ref variants the validator sees
    (``name``, ``name:tag``, ``display_name``, ``hf_repo``, ``hf_repo:tag``,
    provider-prefixed), so the caller doesn't need a fallback chain.
    """
    # circular: catalog -> config -> catalog (catalog imports ``cfg``).
    from lilbee.catalog import find_catalog_entry

    return find_catalog_entry(ref)


def _enforce_role_match(ref: str, entry: Any, field_name: str) -> None:
    """Raise ValueError if *entry*'s task doesn't match *field_name*'s role."""
    from lilbee.models import ModelTask

    want = ModelTask(_MODEL_FIELD_TO_TASK[field_name])
    if entry.task == want:
        return
    from lilbee.server.handlers import format_task_mismatch

    raise ValueError(format_task_mismatch(ref, ModelTask(entry.task), want))


def validate_model_task_assignment(field_name: str, ref: str) -> str:
    """Validate a model role assignment and return its canonical catalog ref.

    This was previously a Pydantic ``field_validator`` on ``Config`` that
    fired on every ``setattr``, which meant a single PATCH /api/config
    touching several fields rescanned the featured catalog index once
    per field. Callers now invoke this explicitly from the handler layer
    so the catalog lookup runs at most once per field per request, and
    zero times for unrelated fields.

    Returns the catalog's canonical ``name:tag`` (for registry-key match)
    or the input ``ref`` unchanged when validation is bypassed in tests.
    Raises ``ValueError`` on unknown refs or role-task mismatch.
    """
    if not ref or not ref.strip() or _model_task_validation_bypassed():
        return ref
    entry = _find_model_catalog_entry(ref)
    if entry is None:
        raise ValueError(
            f"Model '{ref}' is not in the featured catalog. "
            "Pick a featured model for this role, or install one via "
            "POST /api/models/pull with a known catalog ref."
        )
    _enforce_role_match(ref, entry, field_name)
    canonical: str = entry.ref
    return canonical


_BOOL_TRUE = frozenset({"true", "1", "yes"})
_BOOL_FALSE = frozenset({"false", "0", "no"})


def _parse_bool(raw: str) -> bool:
    """Parse true/1/yes or false/0/no; raises ValueError on anything else."""
    normalized = raw.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    raise ValueError(f"Invalid boolean: {raw!r}")


DEFAULT_IGNORE_DIRS = frozenset(
    {
        "node_modules",
        "__pycache__",
        "venv",
        "build",
        "dist",
        "target",
        "vendor",
        "_build",
        "coverage",
        "htmlcov",
    }
)

# Shared HTTP timeout (seconds) for backend catalog / management calls
# (Ollama /api/tags, /api/show, /v1/models, OpenAI-compatible endpoints).
# Not exposed as a user config field because changing it is a debugging
# maneuver, not a deployment knob.
DEFAULT_HTTP_TIMEOUT = 30.0

CHUNKS_TABLE = "chunks"
SOURCES_TABLE = "_sources"
CITATIONS_TABLE = "_citations"
CONCEPT_NODES_TABLE = "concept_nodes"
CONCEPT_EDGES_TABLE = "concept_edges"
CHUNK_CONCEPTS_TABLE = "chunk_concepts"

# Default URL-exclusion patterns for recursive crawls. Categorized so each
# group is inspectable and easy to trim. Users extend or replace via
# LILBEE_CRAWL_EXCLUDE_PATTERNS (newline-separated) or a
# `crawl_exclude_patterns = [...]` list in config.toml.
# All patterns are Python regex (use_glob=False at the call site).

# WordPress scaffolding: admin UIs, API endpoints, RPC endpoints, numeric
# permalink stubs, and the Elementor page-builder staging area.
_WP_EXCLUDE: tuple[str, ...] = (
    r"/wp-admin/",
    r"/wp-login(\.php)?",
    r"/wp-json/",
    r"/xmlrpc\.php",
    r"/wp-cron\.php",
    r"/wp-includes/",
    r"/wp-content/uploads/",
    r"\?p=\d+",
    r"\?page_id=\d+",
    r"\?cat=\d+",
    r"/elementor-\d+",
    r"\?elementor_library",
)

# Pagination and archive permalinks (WP + other CMSes share this shape).
_ARCHIVE_EXCLUDE: tuple[str, ...] = (
    r"/page/\d+/?$",
    r"\?paged?=\d+",
    r"/20\d{2}(/\d{2}(/\d{2})?)?/?$",
    r"/tag/",
    r"/category/",
    r"/author/",
    r"/archives?/?$",
    r"/comment-page-\d+",
)

# Syndication feeds (content-duplicated in HTML pages).
_FEED_EXCLUDE: tuple[str, ...] = (
    r"/feed/?$",
    r"/feed/atom/?$",
    r"/feed/rdf/?$",
    r"/comments/feed/?$",
    r"/rss/?$",
)

# Duplicate views of the same canonical page (AMP, print, preview).
_DUPLICATE_VIEW_EXCLUDE: tuple[str, ...] = (
    r"/amp/?$",
    r"\?amp=",
    r"\?print=",
    r"/print/?$",
    r"\?preview=",
)

# WP attachment URLs (point at media, not content pages).
_ATTACHMENT_EXCLUDE: tuple[str, ...] = (
    r"/attachment/",
    r"\?attachment_id=",
)

# Auth and account flows (generic across CMSes and e-commerce platforms).
_AUTH_EXCLUDE: tuple[str, ...] = (
    r"/login",
    r"/logout",
    r"/register",
    r"/signup",
    r"/signin",
    r"/account",
    r"/my-account/",
    r"/profile",
    r"/password-reset",
    r"/forgot-password",
)

# E-commerce transactional flows (cart / checkout / compare / etc.).
_ECOMMERCE_EXCLUDE: tuple[str, ...] = (
    r"/cart",
    r"/checkout",
    r"/wishlist",
    r"/orders?",
    r"/compare",
    r"/products\.json",
    r"/collections/.+/products/.+\?page=",
)

# Marketing tracking query parameters. One alternation so the regex engine
# scans the URL once. Each listed key is a widely-seen tracking field; see
# https://en.wikipedia.org/wiki/UTM_parameters and vendor docs for origins.
_TRACKING_EXCLUDE: tuple[str, ...] = (
    (
        r"[?&]("
        r"utm_[a-z_]+"
        r"|fbclid|gclid|msclkid|yclid"
        r"|mc_cid|mc_eid"
        r"|_hsenc|_hsmi|hsCtaTracking"
        r"|mkt_tok|mkt_[a-z_]+"
        r"|trk|trkInfo"
        r"|dm_i"
        r"|vero_id|vero_conv"
        r"|oly_anon_id|oly_enc_id"
        r"|igshid"
        r"|pk_campaign|pk_source|pk_medium|pk_[a-z_]+"
        r"|_ga"
        r"|ref|referrer"
        r"|affiliate|aff_id|aff_ref|aff|partner"
        r"|srsltid"
        r"|share|replytocom"
        r")="
    ),
)

# Site-meta URLs and non-HTML resources. crawl4ai also filters by
# Content-Type at fetch time; this filter saves the fetch entirely.
_META_EXCLUDE: tuple[str, ...] = (
    r"/sitemap[^/]*\.xml",
    r"/robots\.txt",
    r"/humans\.txt",
    r"/favicon\.ico",
    r"/\.well-known/",
    r"\.(jpe?g|png|gif|webp|avif|svg|ico|pdf|docx?|xlsx?|pptx?|zip|tar|gz|mp3|mp4|webm|ogg|ttf|woff2?|css|js|map|json|xml)(\?.*)?$",
)

DEFAULT_CRAWL_EXCLUDE_PATTERNS: tuple[str, ...] = (
    *_WP_EXCLUDE,
    *_ARCHIVE_EXCLUDE,
    *_FEED_EXCLUDE,
    *_DUPLICATE_VIEW_EXCLUDE,
    *_ATTACHMENT_EXCLUDE,
    *_AUTH_EXCLUDE,
    *_ECOMMERCE_EXCLUDE,
    *_TRACKING_EXCLUDE,
    *_META_EXCLUDE,
)


_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, direct assistant grounded in the provided context. "
    "Answer using only the context — if it doesn't contain enough information, "
    "say so rather than guessing. Be specific: quote relevant passages and "
    "reference context by number (e.g. [1], [2]) inline. Prefer exact values "
    "over approximations. For code, prefer working examples over abstract "
    "explanations. Keep responses concise unless asked to elaborate."
)

# Default regex for the CORS allow-origin filter. Covers:
#   - Obsidian desktop (Electron renderer uses app://obsidian.md)
#   - Obsidian iOS (Capacitor webview uses capacitor://localhost)
#   - Any http(s) localhost origin, including ports (Android Obsidian, local dev tools)
#   - IPv4 and IPv6 loopback literals
# Auth is still enforced on mutating endpoints regardless of CORS — see server/auth.py.
_DEFAULT_CORS_ORIGIN_REGEX = (
    r"^(app://obsidian\.md"
    r"|capacitor://localhost"
    r"|https?://localhost(:\d+)?"
    r"|https?://127\.0\.0\.1(:\d+)?"
    r"|https?://\[::1\](:\d+)?)$"
)


class Config(BaseSettings):
    """Runtime configuration — one singleton instance, mutated by CLI overrides."""

    model_config = SettingsConfigDict(
        env_prefix="LILBEE_",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    # Paths — resolved from env/defaults in model_validator(mode='before')
    data_root: Path = Field(default=Path())
    documents_dir: Path = Field(default=Path())
    data_dir: Path = Field(default=Path())
    lancedb_dir: Path = Field(default=Path())
    models_dir: Path = Field(default=Path())

    chat_model: str = Field(default="qwen3:0.6b", min_length=1)
    embedding_model: str = Field(default="nomic-embed-text:v1.5", min_length=1)
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
    system_prompt: str = ConfigField(default=_DEFAULT_SYSTEM_PROMPT, min_length=1, writable=True)
    ignore_dirs: frozenset[str] = Field(default=DEFAULT_IGNORE_DIRS)
    # OCR for scanned PDFs via vision-capable chat model.
    # None = auto-detect (use OCR if chat model is vision-capable).
    # True = force OCR regardless of detection.
    # False = disable OCR entirely.
    enable_ocr: bool | None = ConfigField(default=None, writable=True)
    # Per-page timeout in seconds for vision OCR (0 = no limit).
    ocr_timeout: float = ConfigField(default=120.0, ge=0.0, writable=True)

    # Wall-clock timeout in seconds for the Tesseract OCR fallback per
    # file. Large scanned PDFs can otherwise block an ingest worker for
    # many minutes and make the TUI feel frozen. 0 disables the cap.
    tesseract_timeout: float = ConfigField(default=60.0, ge=0.0, writable=True)
    semantic_chunking: bool = ConfigField(default=False, writable=True)
    topic_threshold: float = ConfigField(default=0.75, ge=0.0, le=1.0, writable=True)
    server_host: str = "127.0.0.1"
    server_port: int = Field(default=0, ge=0, le=65535)
    cors_origins: list[str] = Field(default_factory=list)
    cors_origin_regex: str = Field(default=_DEFAULT_CORS_ORIGIN_REGEX)
    json_mode: bool = False
    temperature: float | None = ConfigField(default=None, ge=0.0, writable=True)
    top_p: float | None = ConfigField(default=None, ge=0.0, le=1.0, writable=True)
    top_k_sampling: int | None = ConfigField(default=None, ge=1, writable=True)
    # 1.1 is the llama.cpp default and the value most open-weights chat
    # models are tuned with. A None here made chat prone to n-gram loops
    # ("tire tire tire ...") on some models. Users can still override or
    # disable via settings / config.toml.
    repeat_penalty: float | None = ConfigField(default=1.1, ge=0.0, writable=True)
    num_ctx: int | None = ConfigField(default=None, ge=1, writable=True)
    max_tokens: int | None = ConfigField(default=4096, ge=1, writable=True)
    seed: int | None = ConfigField(default=None, writable=True)
    llm_provider: str = ConfigField(default="auto", writable=True)
    backend_base_url: str = ConfigField(default="http://localhost:11434", writable=True)
    llm_api_key: str = ConfigField(default="", writable=True, write_only=True)
    openai_api_key: str = ConfigField(default="", writable=True, write_only=True)
    anthropic_api_key: str = ConfigField(default="", writable=True, write_only=True)
    gemini_api_key: str = ConfigField(default="", writable=True, write_only=True)

    # Retrieval quality knobs — defaults chosen from academic research and grantflow
    # and academic literature (see docs/superpowers/specs/2026-03-22-feature-parity-design.md)

    # Max chunks per source document in results. Prevents one large file from
    # dominating all top-k slots. 3 balances coverage vs diversity.
    diversity_max_per_source: int = ConfigField(default=3, ge=1, writable=True)

    # MMR relevance/diversity tradeoff. 0.0 = max diversity, 1.0 = pure relevance.
    # 0.5 is the standard default from Carbonell & Goldstein 1998.
    mmr_lambda: float = ConfigField(default=0.5, ge=0.0, le=1.0, writable=True)

    # How many extra candidates to retrieve for MMR reranking.
    # 3x gives enough candidates to find diverse results without excessive latency.
    candidate_multiplier: int = ConfigField(default=3, ge=1, writable=True)

    # Number of LLM-generated alternative queries for expansion.
    # 3 variants covers lexical + semantic angles. Set to 0 to disable expansion.
    query_expansion_count: int = ConfigField(default=3, ge=0, writable=True)

    # Cosine distance threshold step for adaptive widening.
    # When too few results are found, threshold widens by this amount per retry.
    # 0.2 gives 4 steps from typical 0.3 start to 1.0 cap.
    adaptive_threshold_step: float = ConfigField(default=0.2, gt=0.0, writable=True)

    # Reject expansion variants below expansion_similarity_threshold.
    expansion_guardrails: bool = ConfigField(default=True, writable=True)

    # Minimum cosine similarity (question vs variant embedding).
    # Calibrate per embedding model.
    expansion_similarity_threshold: float = ConfigField(default=0.5, ge=0.0, le=1.0, writable=True)

    # BM25 confidence score above which query expansion is skipped entirely.
    # Based on 90th percentile of sigmoid-normalized BM25 score distribution.
    # Higher = expansion runs more often. Calibrate per-corpus.
    expansion_skip_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # Minimum gap between top-1 and top-2 BM25 scores to skip expansion.
    # Approximately 1 standard deviation of typical score spread.
    expansion_skip_gap: float = Field(default=0.15, ge=0.0, le=1.0)

    # Maximum chunks included in LLM context after adaptive selection.
    # More = more complete answers but higher latency and token cost.
    max_context_sources: int = ConfigField(default=5, ge=1, writable=True)

    # Enable HyDE (Hypothetical Document Embeddings) for search.
    # Gao et al. 2022. Adds ~500ms per query. Best for vague queries.
    hyde: bool = ConfigField(default=False, writable=True)

    # Weight for HyDE results relative to original search (0.0-1.0).
    # Lower = less trust in hypothetical documents.
    hyde_weight: float = ConfigField(default=0.7, ge=0.0, le=1.0, writable=True)

    # HyDE prompt template. Must contain {question} placeholder.
    hyde_prompt: str = (
        "Write a 50-100 word passage that directly answers this question as if "
        "it were an excerpt from a real document. Do not include any preamble, "
        "just write the passage.\n\nQuestion: {question}"
    )

    # Reranker model for search results. Empty = disabled. Native GGUF refs
    # run via llama-cpp-python's pooling_type=LLAMA_POOLING_TYPE_RANK; hosted
    # refs (cohere/voyage/jina/together/hf-tei) require the litellm extra.
    reranker_model: str = ConfigField(default="", public=True)

    # Number of candidates to rerank with cross-encoder.
    rerank_candidates: int = ConfigField(default=20, ge=1, writable=True, public=True)

    # Enable temporal filtering (date-based result filtering).
    # Only activates when temporal keywords detected in query.
    temporal_filtering: bool = ConfigField(default=True, writable=True)

    # Show reasoning model thinking process (<think>...</think> tags).
    # When False, thinking is stripped silently. When True, emitted as
    # separate SSE events (event: reasoning) for UI rendering.
    show_reasoning: bool = ConfigField(default=False, writable=True)

    # Web crawling settings
    # Optional global ceiling on recursion depth. None (default) = no ceiling;
    # callers decide. Set a positive int in config.toml as a safety cap.
    crawl_max_depth: int | None = ConfigField(default=None, ge=0, writable=True)

    # Optional global ceiling on total pages per crawl. None (default) = no ceiling.
    crawl_max_pages: int | None = ConfigField(default=None, ge=1, writable=True)

    # Per-page timeout in seconds for fetching a URL.
    crawl_timeout: int = ConfigField(default=30, ge=1, writable=True)

    # Maximum concurrent crawl operations (0 = unlimited, default = CPU count).
    crawl_max_concurrent: int = Field(default=0, ge=0)

    # Seconds between periodic syncs during crawl (0 = sync only at end).
    crawl_sync_interval: int = ConfigField(default=30, ge=0, writable=True)

    # Uniform delay between in-flight requests within a single crawl.
    # crawl4ai's own defaults are 0.1 / 0.3; ours are friendlier to hosts.
    crawl_mean_delay: float = ConfigField(default=0.5, ge=0.0, writable=True)

    # Random jitter added to crawl_mean_delay (seconds).
    crawl_max_delay_range: float = ConfigField(default=0.5, ge=0.0, writable=True)

    # Concurrent in-flight requests within a single crawl. crawl4ai default
    # is 5; we use 3 by default to be gentler.
    crawl_concurrent_requests: int = ConfigField(default=3, ge=1, writable=True)

    # Enable the per-domain RateLimiter that backs off on HTTP 429/503 and
    # retries. Set False to disable the dispatcher entirely.
    crawl_retry_on_rate_limit: bool = ConfigField(default=True, writable=True)

    # Randomized base-delay range the RateLimiter picks from on rate-limit
    # responses (seconds). Pair: (min, max).
    crawl_retry_base_delay_min: float = ConfigField(default=1.0, ge=0.0, writable=True)
    crawl_retry_base_delay_max: float = ConfigField(default=3.0, ge=0.0, writable=True)

    # Upper bound on any single backoff wait (seconds).
    crawl_retry_max_backoff: float = ConfigField(default=30.0, ge=0.0, writable=True)

    # Retry count per URL when rate-limit codes come back.
    crawl_retry_max_attempts: int = ConfigField(default=3, ge=0, writable=True)

    # Regex patterns that skip URLs at link-discovery time during recursive
    # crawls. Defaults block WordPress scaffolding (pagination, archives,
    # tracking query params) that inflates useful-page count by 5-7x without
    # adding content. Empty list disables the filter.
    crawl_exclude_patterns: list[str] = ConfigField(
        default_factory=lambda: list(DEFAULT_CRAWL_EXCLUDE_PATTERNS),
        writable=True,
    )

    # Fraction of GPU/unified memory available for loaded models.
    # 0.75 leaves headroom for the OS and other processes.
    gpu_memory_fraction: float = ConfigField(default=0.75, ge=0.1, le=1.0, writable=True)

    # Seconds a model stays loaded after last use. 0 = unload immediately.
    model_keep_alive: int = ConfigField(default=300, ge=0, writable=True)

    # Run embedding and vision inference in a subprocess to avoid GIL blocking.
    # Applies only to the llama-cpp provider.
    subprocess_embed: bool = ConfigField(default=False, writable=True)

    # Use Markdown widget for chat responses in the TUI. When False, uses
    # plain Static text (faster rendering, no formatting).
    markdown_rendering: bool = True

    # Per-model generation defaults (not serialized, not a config field).
    # Set via apply_model_defaults() when switching models.
    _model_defaults: Any = None

    # Wiki layer — LLM-maintained synthesis pages with citation provenance.
    # On by default; no extras required. Set to False to hide the Wiki view
    # and disable wiki generation/sync.
    wiki: bool = True
    wiki_dir: str = "wiki"
    wiki_prune_raw: bool = ConfigField(default=False, writable=True)
    wiki_faithfulness_threshold: float = ConfigField(default=0.7, ge=0.0, le=1.0, writable=True)

    # Fraction of citations that must be stale before a wiki page is flagged
    # for regeneration during pruning. 0.5 = flag when >50% are stale.
    wiki_stale_citation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Maximum fraction of content that may change before a regeneration is
    # flagged for human review instead of overwriting the existing page.
    # 0.3 = 30% of lines changed triggers the drift guard.
    wiki_drift_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # LLM prompt templates for wiki page generation. Override via env vars
    # LILBEE_WIKI_SUMMARY_PROMPT, LILBEE_WIKI_FAITHFULNESS_PROMPT,
    # LILBEE_WIKI_SYNTHESIS_PROMPT. Must contain the expected {placeholders}.
    wiki_summary_prompt: str = (
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
    )
    wiki_faithfulness_prompt: str = (
        "You are a fact-checker. Given source chunks and a wiki summary page generated "
        "from them, score the summary's faithfulness to the sources on a scale of 0.0 "
        "to 1.0.\n\n"
        "Criteria:\n"
        "- 1.0 = every claim is directly supported by the source chunks\n"
        "- 0.5 = some claims are supported, some are unsupported extrapolations\n"
        "- 0.0 = the summary contains fabricated information\n\n"
        "Source chunks:\n{chunks_text}\n\n"
        "Wiki summary:\n{wiki_text}\n\n"
        "Respond with ONLY a number between 0.0 and 1.0. Nothing else."
    )
    wiki_synthesis_prompt: str = (
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
    )

    # Wiki synthesis clusterer backend. EMBEDDING (default, no extra deps)
    # runs chunk-level mutual kNN + label propagation over chunk embeddings.
    # CONCEPTS uses the concept graph adapter and requires the [graph] extra;
    # when [graph] is missing the services factory logs a warning and falls
    # back to EMBEDDING.
    wiki_clusterer: ClustererBackend = ConfigField(
        default=ClustererBackend.EMBEDDING, writable=True
    )

    # Neighborhood size for the embedding clusterer's mutual-kNN graph.
    # 0 means "auto-scale from corpus size" via clamp(log2(N)+2, 5, 20).
    # Raise to get larger, looser clusters; lower for tighter, smaller ones.
    wiki_clusterer_k: int = ConfigField(default=0, ge=0, writable=True)

    # Enable concept graph (LazyGraphRAG-style index). Extracts noun phrases
    # from chunks, builds a co-occurrence graph, and uses it to boost search
    # results and expand queries. Requires spacy + networkx + graspologic-native.
    concept_graph: bool = ConfigField(default=True, writable=True)

    # Weight for concept overlap boosting in search results (0.0-1.0).
    # Higher = concept overlap matters more relative to vector similarity.
    concept_boost_weight: float = ConfigField(default=0.3, ge=0.0, le=1.0, writable=True)

    # Minimum distance after concept boost. Prevents boost from making
    # marginally relevant results appear artificially close.
    concept_boost_floor: float = ConfigField(default=0.05, ge=0.0, writable=True)

    # Maximum noun-phrase concepts extracted per chunk.
    # Caps extraction to avoid noise from very long chunks.
    concept_max_per_chunk: int = ConfigField(default=10, ge=1, writable=True)

    # Class variable — not a settings field
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
                return _parse_bool(v)
            except ValueError:
                pass
        return bool(v)

    @field_validator("semantic_chunking", mode="before")
    @classmethod
    def _parse_semantic_chunking(cls, v: Any) -> bool:
        """Parse from env string; invalid values warn and fall back to False."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            try:
                return _parse_bool(v)
            except ValueError:
                log.warning("Invalid LILBEE_SEMANTIC_CHUNKING=%r, using default False", v)
                return False
        return bool(v)

    @field_validator(
        "chat_model", "embedding_model", "vision_model", "reranker_model", mode="after"
    )
    @classmethod
    def _normalize_model_tag(cls, v: str, info: ValidationInfo) -> str:
        """Ensure model names always have an explicit tag and canonical prefix.

        Whitespace-only values are coerced to "" for roles that allow empty
        (vision, reranker), and rejected for required roles (chat, embedding)
        to prevent bypassing ``min_length=1``.
        """
        if not v or not v.strip():
            if info.field_name in {"chat_model", "embedding_model"}:
                raise ValueError(f"{info.field_name} must not be blank")
            return ""
        from lilbee.providers.model_ref import parse_model_ref

        return parse_model_ref(v).for_openai_prefix()

    # Model-task validation lives in the handler layer now (see
    # lilbee.server.handlers._validate_config_updates and the
    # validate_model_task_assignment helper below). Running it inside a
    # field_validator fired on every setattr, and a single PATCH /api/config
    # with six fields would rescan the featured catalog six times.

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

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: Any) -> Any:
        from lilbee.platform import canonical_models_dir, default_data_dir, find_local_root

        if not isinstance(data, dict):  # pragma: no cover
            return data

        _UNSET = Path()

        if data.get("data_root") in (None, _UNSET):
            data_env = os.environ.get("LILBEE_DATA", "").strip()
            if data_env:
                data["data_root"] = Path(data_env)
            else:
                local = find_local_root()
                data["data_root"] = local if local is not None else default_data_dir()
        root = data["data_root"]
        if data.get("documents_dir") in (None, _UNSET):
            data["documents_dir"] = root / "documents"
        if data.get("data_dir") in (None, _UNSET):
            data["data_dir"] = root / "data"
        if data.get("lancedb_dir") in (None, _UNSET):
            data["lancedb_dir"] = root / "data" / "lancedb"
        if data.get("models_dir") in (None, _UNSET):
            data["models_dir"] = canonical_models_dir()

        if "LILBEE_BACKEND_BASE_URL" not in os.environ:
            ollama_host = os.environ.get("OLLAMA_HOST")
            if ollama_host:
                data["backend_base_url"] = ollama_host

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
        from lilbee.platform import default_data_dir, find_local_root

        data_env = os.environ.get("LILBEE_DATA", "")
        if data_env:
            toml_dir = Path(data_env)
        else:
            local = find_local_root()
            toml_dir = local if local else default_data_dir()
        toml_path = toml_dir / "config.toml"

        plain_env = _PlainEnvSource(settings_cls, env_prefix="LILBEE_", env_ignore_empty=True)
        sources: list[Any] = [init_settings, plain_env]
        if toml_path.exists():
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
        """Build LLM generation options with 3-layer merge.
        Layer 1 (base): model defaults from ``_model_defaults``
        Layer 2 (override): user config fields — only if explicitly set (not None)
        Layer 3 (override): per-call ``overrides`` parameter

        Remaps ``top_k_sampling`` to the provider's ``top_k`` key.
        Filters out ``None`` values so the provider uses its model defaults.
        """
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
    """Convert a ModelDefaults instance to a dict with provider key names.
    Remaps ``top_k`` to the provider's ``top_k`` key (same name for model defaults).
    Filters out None values.
    """
    if defaults is None:
        return {}
    from dataclasses import fields as dc_fields

    return {
        f.name: getattr(defaults, f.name)
        for f in dc_fields(defaults)
        if getattr(defaults, f.name) is not None
    }


class _PlainEnvSource:
    """Env source that reads LILBEE_* env vars as plain strings.
    Avoids pydantic-settings' default JSON parsing of complex types (list, frozenset)
    so that comma-separated values like ``LILBEE_CORS_ORIGINS=a,b`` pass through to
    field validators instead of failing JSON decode.
    """

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
            return {k: str(v) for k, v in data.items()}
        except (ValueError, OSError):
            log.warning("Failed to read %s, ignoring", self._path)
            return {}


cfg = Config()
