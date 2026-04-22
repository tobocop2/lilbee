"""Shared settings map for interactive configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic_core import PydanticUndefined

from lilbee.config import ClustererBackend, cfg


class RenderStyle(StrEnum):
    """How a setting is displayed in /settings."""

    COMPACT = "compact"
    FULL = "full"
    LIST_COLLAPSED = "list_collapsed"


@dataclass(frozen=True)
class SettingDef:
    """Metadata for an interactive setting."""

    type: type
    nullable: bool
    writable: bool = True
    render: RenderStyle = field(default=RenderStyle.COMPACT)
    group: str = "General"
    help_text: str = ""
    choices: tuple[str, ...] | None = None


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
        group="Models",
        help_text="LLM used for chat and generation",
    ),
    "enable_ocr": SettingDef(
        bool,
        nullable=True,
        group="Ingest",
        help_text="Vision OCR for scanned PDFs (empty = auto-detect from chat model)",
    ),
    "ocr_timeout": SettingDef(
        float,
        nullable=False,
        group="Ingest",
        help_text="Per-page timeout in seconds for vision OCR (0 = no limit)",
    ),
    "semantic_chunking": SettingDef(
        bool,
        nullable=False,
        group="Ingest",
        help_text="Opt-in topic-aware chunker (default off; may fragment numbered procedures)",
    ),
    "topic_threshold": SettingDef(
        float,
        nullable=False,
        group="Ingest",
        help_text="Topic-boundary similarity threshold, 0.0-1.0, used when semantic chunking is on",
    ),
    "embedding_model": SettingDef(
        str,
        nullable=False,
        writable=False,
        group="Models",
        help_text="Model used to embed document chunks",
    ),
    "reranker_model": SettingDef(
        str,
        nullable=True,
        group="Models",
        help_text="Cross-encoder model for result reranking",
    ),
    "temperature": SettingDef(
        float,
        nullable=True,
        group="Generation",
        help_text="Sampling temperature (higher = more creative)",
    ),
    "top_p": SettingDef(
        float,
        nullable=True,
        group="Generation",
        help_text="Nucleus sampling cutoff probability",
    ),
    "top_k_sampling": SettingDef(
        int,
        nullable=True,
        group="Generation",
        help_text="Top-K sampling: number of tokens to consider",
    ),
    "repeat_penalty": SettingDef(
        float,
        nullable=True,
        group="Generation",
        help_text="Penalty for repeating tokens",
    ),
    "num_ctx": SettingDef(
        int,
        nullable=True,
        group="Generation",
        help_text="Context window size in tokens",
    ),
    "seed": SettingDef(
        int,
        nullable=True,
        group="Generation",
        help_text="Random seed for reproducible output",
    ),
    "system_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.FULL,
        group="Generation",
        help_text="System prompt sent before every conversation",
    ),
    "top_k": SettingDef(
        int,
        nullable=False,
        group="Retrieval",
        help_text="Number of chunks returned by search",
    ),
    "rerank_candidates": SettingDef(
        int,
        nullable=False,
        group="Retrieval",
        help_text="Candidate pool size for reranking",
    ),
    "show_reasoning": SettingDef(
        bool,
        nullable=False,
        group="Display",
        help_text="Show model reasoning/thinking tokens in output",
    ),
    "wiki": SettingDef(
        bool,
        nullable=False,
        group="Wiki",
        help_text="Enable the wiki layer (synthesis pages with citations)",
    ),
    "wiki_dir": SettingDef(
        str,
        nullable=False,
        group="Wiki",
        help_text="Directory under data_root where wiki pages are stored",
    ),
    "wiki_prune_raw": SettingDef(
        bool,
        nullable=False,
        group="Wiki",
        help_text="Delete raw chunks after summarizing into the wiki",
    ),
    "wiki_faithfulness_threshold": SettingDef(
        float,
        nullable=False,
        group="Wiki",
        help_text="Minimum faithfulness score (0-1) to accept a generated page",
    ),
    "wiki_stale_citation_threshold": SettingDef(
        float,
        nullable=False,
        group="Wiki",
        help_text="Fraction of stale citations that triggers page regeneration",
    ),
    "wiki_drift_threshold": SettingDef(
        float,
        nullable=False,
        group="Wiki",
        help_text="Max fraction of changed lines before regeneration requires review",
    ),
    "wiki_clusterer": SettingDef(
        str,
        nullable=False,
        group="Wiki",
        help_text="Synthesis clusterer backend (embedding or concepts)",
        choices=tuple(b.value for b in ClustererBackend),
    ),
    "wiki_clusterer_k": SettingDef(
        int,
        nullable=False,
        group="Wiki",
        help_text="Mutual-kNN neighborhood size for the clusterer (0 = auto)",
    ),
    "wiki_summary_max_tokens": SettingDef(
        int,
        nullable=False,
        group="Wiki",
        help_text="Max output tokens per generated wiki page (caps reasoning runaway)",
    ),
    "wiki_faithfulness_max_tokens": SettingDef(
        int,
        nullable=False,
        group="Wiki",
        help_text="Max output tokens for the faithfulness score call",
    ),
    "wiki_temperature": SettingDef(
        float,
        nullable=False,
        group="Wiki",
        help_text="Sampling temperature for wiki generation (low = follow template exactly)",
    ),
    "crawl_max_depth": SettingDef(
        int,
        nullable=True,
        group="Crawling",
        help_text="Optional recursion-depth cap (blank = no cap; per-crawl values win)",
    ),
    "crawl_max_pages": SettingDef(
        int,
        nullable=True,
        group="Crawling",
        help_text="Optional global cap on total pages per crawl (blank = no cap).",
    ),
    "crawl_timeout": SettingDef(
        int,
        nullable=False,
        group="Crawling",
        help_text="Per-page fetch timeout in seconds",
    ),
    "crawl_sync_interval": SettingDef(
        int,
        nullable=False,
        group="Crawling",
        help_text="Seconds between periodic re-syncs during a crawl (0 = sync only at end)",
    ),
    "crawl_mean_delay": SettingDef(
        float,
        nullable=False,
        group="Crawling",
        help_text="Seconds between in-flight requests within a single crawl",
    ),
    "crawl_max_delay_range": SettingDef(
        float,
        nullable=False,
        group="Crawling",
        help_text="Random jitter (seconds) added on top of mean delay",
    ),
    "crawl_concurrent_requests": SettingDef(
        int,
        nullable=False,
        group="Crawling",
        help_text="Concurrent in-flight URLs within one crawl",
    ),
    "crawl_retry_on_rate_limit": SettingDef(
        bool,
        nullable=False,
        group="Crawling",
        help_text="Enable per-domain backoff and retries on HTTP 429/503",
    ),
    "crawl_retry_base_delay_min": SettingDef(
        float,
        nullable=False,
        group="Crawling",
        help_text="Minimum base-delay (seconds) on rate-limit responses",
    ),
    "crawl_retry_base_delay_max": SettingDef(
        float,
        nullable=False,
        group="Crawling",
        help_text="Maximum base-delay (seconds) on rate-limit responses",
    ),
    "crawl_retry_max_backoff": SettingDef(
        float,
        nullable=False,
        group="Crawling",
        help_text="Upper bound on any single backoff wait (seconds)",
    ),
    "crawl_retry_max_attempts": SettingDef(
        int,
        nullable=False,
        group="Crawling",
        help_text="Retry count per URL when a rate-limit code comes back",
    ),
    "crawl_exclude_patterns": SettingDef(
        list,
        nullable=False,
        group="Crawling",
        render=RenderStyle.LIST_COLLAPSED,
        help_text=(
            "Regex patterns that skip URLs at link-discovery time during "
            "recursive crawls. One per line."
        ),
    ),
    "openai_api_key": SettingDef(
        str,
        nullable=False,
        group="API-Keys",
        help_text="OpenAI API key (enables frontier models in chat picker)",
    ),
    "anthropic_api_key": SettingDef(
        str,
        nullable=False,
        group="API-Keys",
        help_text="Anthropic API key (enables frontier models in chat picker)",
    ),
    "gemini_api_key": SettingDef(
        str,
        nullable=False,
        group="API-Keys",
        help_text="Google Gemini API key (enables frontier models in chat picker)",
    ),
}
