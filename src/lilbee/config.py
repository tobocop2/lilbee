"""Application configuration for lilbee.

All settings can be overridden via environment variables prefixed with LILBEE_.
"""

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lilbee import settings
from lilbee.platform import canonical_models_dir, default_data_dir, env, env_int

log = logging.getLogger(__name__)

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

CHUNKS_TABLE = "chunks"
SOURCES_TABLE = "_sources"
CONCEPT_NODES_TABLE = "concept_nodes"
CONCEPT_EDGES_TABLE = "concept_edges"
CHUNK_CONCEPTS_TABLE = "chunk_concepts"


class Config(BaseModel):
    """Runtime configuration — one singleton instance, mutated by CLI overrides."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data_root: Path
    documents_dir: Path
    data_dir: Path
    lancedb_dir: Path
    models_dir: Path
    chat_model: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dim: int = Field(ge=1)
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)
    max_embed_chars: int = Field(ge=1)
    top_k: int = Field(ge=1)
    max_distance: float = Field(ge=0.0)
    system_prompt: str = Field(min_length=1)
    ignore_dirs: frozenset[str]
    vision_model: str = ""
    vision_timeout: float = Field(default=120.0, ge=0.0)
    server_host: str = "127.0.0.1"
    server_port: int = Field(default=0, ge=0, le=65535)
    cors_origins: list[str] = Field(default_factory=list)
    json_mode: bool = False
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k_sampling: int | None = Field(default=None, ge=1)
    repeat_penalty: float | None = Field(default=None, ge=0.0)
    num_ctx: int | None = Field(default=None, ge=1)
    seed: int | None = None
    llm_provider: str = "auto"
    ollama_url: str = "http://localhost:11434"
    llm_api_key: str = ""

    # Retrieval quality knobs — defaults chosen from academic research and grantflow
    # and academic literature (see docs/superpowers/specs/2026-03-22-feature-parity-design.md)

    # Max chunks per source document in results. Prevents one large file from
    # dominating all top-k slots. 3 balances coverage vs diversity.
    diversity_max_per_source: int = Field(default=3, ge=1)

    # MMR relevance/diversity tradeoff. 0.0 = max diversity, 1.0 = pure relevance.
    # 0.5 is the standard default from Carbonell & Goldstein 1998.
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)

    # How many extra candidates to retrieve for MMR reranking.
    # 3x gives enough candidates to find diverse results without excessive latency.
    candidate_multiplier: int = Field(default=3, ge=1)

    # Number of LLM-generated alternative queries for expansion.
    # 3 variants covers lexical + semantic angles. Set to 0 to disable expansion.
    query_expansion_count: int = Field(default=3, ge=0)

    # Cosine distance threshold step for adaptive widening.
    # When too few results are found, threshold widens by this amount per retry.
    # 0.2 gives 4 steps from typical 0.3 start to 1.0 cap.
    adaptive_threshold_step: float = Field(default=0.2, gt=0.0)

    # Validate LLM-generated expansion variants to prevent query drift.
    # Checks token overlap with original query (≥ 0.3) and deduplicates
    # near-identical variants (cosine similarity > 0.85).
    expansion_guardrails: bool = True

    # BM25 confidence score above which query expansion is skipped entirely.
    # Based on 90th percentile of sigmoid-normalized BM25 score distribution.
    # Higher = expansion runs more often. Calibrate per-corpus.
    expansion_skip_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # Minimum gap between top-1 and top-2 BM25 scores to skip expansion.
    # Approximately 1 standard deviation of typical score spread.
    expansion_skip_gap: float = Field(default=0.15, ge=0.0, le=1.0)

    # Maximum chunks included in LLM context after adaptive selection.
    # More = more complete answers but higher latency and token cost.
    max_context_sources: int = Field(default=5, ge=1)

    # Enable HyDE (Hypothetical Document Embeddings) for search.
    # Gao et al. 2022. Adds ~500ms per query. Best for vague queries.
    hyde: bool = False

    # Weight for HyDE results relative to original search (0.0-1.0).
    # Lower = less trust in hypothetical documents.
    hyde_weight: float = Field(default=0.7, ge=0.0, le=1.0)

    # Cross-encoder model for reranking. Empty = disabled.
    # Requires sentence-transformers installed.
    reranker_model: str = ""

    # Number of candidates to rerank with cross-encoder.
    rerank_candidates: int = Field(default=20, ge=1)

    # Enable temporal filtering (date-based result filtering).
    # Only activates when temporal keywords detected in query.
    temporal_filtering: bool = True

    # Show reasoning model thinking process (<think>...</think> tags).
    # When False, thinking is stripped silently. When True, emitted as
    # separate SSE events (event: reasoning) for UI rendering.
    show_reasoning: bool = False

    # Web crawling settings
    # Maximum link-following depth for recursive crawls.
    crawl_max_depth: int = Field(default=2, ge=0)

    # Maximum pages to fetch in a single crawl operation.
    crawl_max_pages: int = Field(default=50, ge=1)

    # Per-page timeout in seconds for fetching a URL.
    crawl_timeout: int = Field(default=30, ge=1)

    # Maximum concurrent crawl operations (0 = unlimited, default = CPU count).
    crawl_max_concurrent: int = Field(default=0, ge=0)

    # Seconds between periodic syncs during crawl (0 = sync only at end).
    crawl_sync_interval: int = Field(default=30, ge=0)

    # Enable concept graph (LazyGraphRAG-style index). Extracts noun phrases
    # from chunks, builds a co-occurrence graph, and uses it to boost search
    # results and expand queries. Requires spacy + networkx + graspologic-native.
    concept_graph: bool = True

    # Weight for concept overlap boosting in search results (0.0-1.0).
    # Higher = concept overlap matters more relative to vector similarity.
    concept_boost_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Maximum noun-phrase concepts extracted per chunk.
    # Caps extraction to avoid noise from very long chunks.
    concept_max_per_chunk: int = Field(default=10, ge=1)

    def generation_options(self, **overrides: Any) -> dict[str, Any]:
        """Build Ollama generation options from config fields and overrides.

        Remaps ``top_k_sampling`` to Ollama's ``top_k`` key.
        Filters out ``None`` values so Ollama uses its model defaults.
        """
        mapping: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k_sampling,
            "repeat_penalty": self.repeat_penalty,
            "num_ctx": self.num_ctx,
            "seed": self.seed,
        }
        mapping.update(overrides)
        return {k: v for k, v in mapping.items() if v is not None}

    @classmethod
    def from_env(cls) -> "Config":
        """Build config from environment variables and settings file."""
        data_root = _resolve_data_root()
        chat_model = _load_chat_model(data_root)
        vision_model = _load_vision_model(data_root)
        vision_timeout = _parse_vision_timeout()

        extra = env("IGNORE", "")
        ignore_dirs = DEFAULT_IGNORE_DIRS | frozenset(
            name.strip() for name in extra.split(",") if name.strip()
        )

        _DEFAULT_SYSTEM_PROMPT = (
            "You are a precise, direct assistant grounded in the provided context. "
            "Answer using only the context — if it doesn't contain enough information, "
            "say so rather than guessing. Be specific: quote relevant passages, cite file "
            "paths, and prefer exact values over approximations. For code, prefer working "
            "examples over abstract explanations. Keep responses concise unless asked to "
            "elaborate."
        )

        return cls(
            data_root=data_root,
            documents_dir=data_root / "documents",
            data_dir=data_root / "data",
            lancedb_dir=data_root / "data" / "lancedb",
            models_dir=canonical_models_dir(),
            chat_model=chat_model,
            embedding_model=_load_setting(
                data_root, "embedding_model", "EMBEDDING_MODEL", "nomic-embed-text", str
            ),
            embedding_dim=_load_setting(data_root, "embedding_dim", "EMBEDDING_DIM", 768, int),
            chunk_size=_load_setting(data_root, "chunk_size", "CHUNK_SIZE", 512, int),
            chunk_overlap=_load_setting(data_root, "chunk_overlap", "CHUNK_OVERLAP", 100, int),
            max_embed_chars=_load_setting(
                data_root, "max_embed_chars", "MAX_EMBED_CHARS", 2000, int
            ),
            top_k=_load_setting(data_root, "top_k", "TOP_K", 10, int),
            max_distance=_load_setting(data_root, "max_distance", "MAX_DISTANCE", 0.7, float),
            system_prompt=_load_setting(
                data_root,
                "system_prompt",
                "SYSTEM_PROMPT",
                _DEFAULT_SYSTEM_PROMPT,
                str,
            ),
            ignore_dirs=ignore_dirs,
            vision_model=vision_model,
            vision_timeout=vision_timeout,
            server_host=env("SERVER_HOST", "127.0.0.1"),
            server_port=env_int("SERVER_PORT", 0),
            cors_origins=_parse_cors_origins(),
            temperature=_load_setting(data_root, "temperature", "TEMPERATURE", None, float),
            top_p=_load_setting(data_root, "top_p", "TOP_P", None, float),
            top_k_sampling=_load_setting(data_root, "top_k_sampling", "TOP_K_SAMPLING", None, int),
            repeat_penalty=_load_setting(
                data_root, "repeat_penalty", "REPEAT_PENALTY", None, float
            ),
            num_ctx=_load_setting(data_root, "num_ctx", "NUM_CTX", None, int),
            seed=_load_setting(data_root, "seed", "SEED", None, int),
            llm_provider=env("LLM_PROVIDER", "auto"),
            ollama_url=env(
                "OLLAMA_URL",
                # Fall back to OLLAMA_HOST for backwards compat
                os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            ),
            llm_api_key=env("LLM_API_KEY", ""),
            diversity_max_per_source=_load_setting(
                data_root, "diversity_max_per_source", "DIVERSITY_MAX_PER_SOURCE", 3, int
            ),
            mmr_lambda=_load_setting(data_root, "mmr_lambda", "MMR_LAMBDA", 0.5, float),
            candidate_multiplier=_load_setting(
                data_root, "candidate_multiplier", "CANDIDATE_MULTIPLIER", 3, int
            ),
            query_expansion_count=_load_setting(
                data_root, "query_expansion_count", "QUERY_EXPANSION_COUNT", 3, int
            ),
            adaptive_threshold_step=_load_setting(
                data_root, "adaptive_threshold_step", "ADAPTIVE_THRESHOLD_STEP", 0.2, float
            ),
            expansion_guardrails=_load_setting(
                data_root, "expansion_guardrails", "EXPANSION_GUARDRAILS", True, bool
            ),
            expansion_skip_threshold=_load_setting(
                data_root, "expansion_skip_threshold", "EXPANSION_SKIP_THRESHOLD", 0.8, float
            ),
            expansion_skip_gap=_load_setting(
                data_root, "expansion_skip_gap", "EXPANSION_SKIP_GAP", 0.15, float
            ),
            max_context_sources=_load_setting(
                data_root, "max_context_sources", "MAX_CONTEXT_SOURCES", 5, int
            ),
            hyde=_load_setting(data_root, "hyde", "HYDE", False, bool),
            hyde_weight=_load_setting(data_root, "hyde_weight", "HYDE_WEIGHT", 0.7, float),
            reranker_model=_load_setting(data_root, "reranker_model", "RERANKER_MODEL", "", str),
            rerank_candidates=_load_setting(
                data_root, "rerank_candidates", "RERANK_CANDIDATES", 20, int
            ),
            temporal_filtering=_load_setting(
                data_root, "temporal_filtering", "TEMPORAL_FILTERING", True, bool
            ),
            show_reasoning=_load_setting(
                data_root, "show_reasoning", "SHOW_REASONING", False, bool
            ),
            crawl_max_depth=_load_setting(data_root, "crawl_max_depth", "CRAWL_MAX_DEPTH", 2, int),
            crawl_max_pages=_load_setting(data_root, "crawl_max_pages", "CRAWL_MAX_PAGES", 50, int),
            crawl_timeout=_load_setting(data_root, "crawl_timeout", "CRAWL_TIMEOUT", 30, int),
            crawl_max_concurrent=_load_setting(
                data_root,
                "crawl_max_concurrent",
                "CRAWL_MAX_CONCURRENT",
                os.cpu_count() or 4,
                int,
            ),
            crawl_sync_interval=_load_setting(
                data_root, "crawl_sync_interval", "CRAWL_SYNC_INTERVAL", 30, int
            ),
            concept_graph=_load_setting(data_root, "concept_graph", "CONCEPT_GRAPH", True, bool),
            concept_boost_weight=_load_setting(
                data_root, "concept_boost_weight", "CONCEPT_BOOST_WEIGHT", 0.3, float
            ),
            concept_max_per_chunk=_load_setting(
                data_root, "concept_max_per_chunk", "CONCEPT_MAX_PER_CHUNK", 10, int
            ),
        )


def _load_setting(data_root: Path, key: str, env_var: str, default: Any, typ: type) -> Any:
    """Load setting with precedence: LILBEE_<ENV> env > config.toml > default."""
    raw = os.environ.get(f"LILBEE_{env_var}")
    if raw is not None:
        return typ(raw)
    try:
        saved = settings.get(data_root, key)
    except (ValueError, OSError):
        saved = None
    if saved:
        return typ(saved)
    return default


def _resolve_data_root() -> Path:
    """Determine the data root: LILBEE_DATA env > local .lilbee/ > platform default."""
    data_env = env("DATA", "")
    if data_env:
        return Path(data_env)

    from lilbee.platform import find_local_root

    local = find_local_root()
    if local is not None:
        return local

    return default_data_dir()


def _load_chat_model(data_root: Path) -> str:
    """Resolve chat model: LILBEE_CHAT_MODEL env > persisted setting > default."""
    chat_model = env("CHAT_MODEL", "qwen3:8b")
    if "LILBEE_CHAT_MODEL" not in os.environ:
        try:
            saved = settings.get(data_root, "chat_model")
        except (ValueError, OSError):
            saved = None
        if saved:
            chat_model = saved
    return chat_model


def _load_vision_model(data_root: Path) -> str:
    """Resolve vision model: LILBEE_VISION_MODEL env > persisted setting > empty."""
    vision_model_env = os.environ.get("LILBEE_VISION_MODEL", "").strip()
    if vision_model_env:
        return vision_model_env
    try:
        return settings.get(data_root, "vision_model") or ""
    except (ValueError, OSError):
        return ""


def _parse_vision_timeout() -> float:
    """Parse LILBEE_VISION_TIMEOUT env var, returning default on invalid input."""
    raw = os.environ.get("LILBEE_VISION_TIMEOUT", "").strip()
    if not raw:
        return 120.0
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid LILBEE_VISION_TIMEOUT=%r, ignoring", raw)
        return 120.0


def _parse_cors_origins() -> list[str]:
    """Parse LILBEE_CORS_ORIGINS env var (comma-separated list of origins)."""
    raw = os.environ.get("LILBEE_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]


cfg = Config.from_env()
