"""Application configuration for lilbee.

All settings can be overridden via environment variables prefixed with LILBEE_.
Uses pydantic-settings for automatic env var loading with TOML config file support.
"""

import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
CITATIONS_TABLE = "_citations"
CONCEPT_NODES_TABLE = "concept_nodes"
CONCEPT_EDGES_TABLE = "concept_edges"
CHUNK_CONCEPTS_TABLE = "chunk_concepts"

_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, direct assistant grounded in the provided context. "
    "Answer using only the context — if it doesn't contain enough information, "
    "say so rather than guessing. Be specific: quote relevant passages, cite file "
    "paths, and prefer exact values over approximations. For code, prefer working "
    "examples over abstract explanations. Keep responses concise unless asked to "
    "elaborate."
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

    chat_model: str = Field(default="qwen3:8b", min_length=1)
    embedding_model: str = Field(default="nomic-embed-text", min_length=1)
    embedding_dim: int = Field(default=768, ge=1)
    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=100, ge=0)
    max_embed_chars: int = Field(default=2000, ge=1)
    top_k: int = Field(default=10, ge=1)
    max_distance: float = Field(default=0.7, ge=0.0)
    system_prompt: str = Field(default=_DEFAULT_SYSTEM_PROMPT, min_length=1)
    ignore_dirs: frozenset[str] = Field(default=DEFAULT_IGNORE_DIRS)
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
    litellm_base_url: str = "http://localhost:11434"
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
    # Checks token overlap with original query (>= 0.3) and deduplicates
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

    # HyDE prompt template. Must contain {question} placeholder.
    hyde_prompt: str = (
        "Write a 50-100 word passage that directly answers this question as if "
        "it were an excerpt from a real document. Do not include any preamble, "
        "just write the passage.\n\nQuestion: {question}"
    )

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

    # Wiki layer — LLM-maintained synthesis pages with citation provenance.
    # Requires optional ``wiki`` extra: ``pip install lilbee[wiki]``.
    wiki: bool = False
    wiki_dir: str = "wiki"
    wiki_prune_raw: bool = False
    wiki_faithfulness_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Maximum fraction of content that may change before a regeneration is
    # flagged for human review instead of overwriting the existing page.
    # 0.3 = 30% of lines changed triggers the drift guard.
    wiki_drift_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

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

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
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

        if "LILBEE_LITELLM_BASE_URL" not in os.environ:
            ollama_host = os.environ.get("OLLAMA_HOST")
            if ollama_host:
                data["litellm_base_url"] = ollama_host

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

    def generation_options(self, **overrides: Any) -> dict[str, Any]:
        """Build LLM generation options from config fields and overrides.

        Remaps ``top_k_sampling`` to the provider's ``top_k`` key.
        Filters out ``None`` values so the provider uses its model defaults.
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
