"""Shared settings map for interactive configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic_core import PydanticUndefined

from lilbee.core.config import ClustererBackend, WikiEntityMode, cfg


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
        help_text="LLM used for chat generation (vision and reranking are separate slots)",
    ),
    "vision_model": SettingDef(
        str,
        nullable=True,
        writable=False,
        group="Models",
        help_text="Vision model for scanned PDF OCR (empty = disabled; Tesseract only)",
    ),
    "enable_ocr": SettingDef(
        bool,
        nullable=True,
        group="Ingest",
        help_text="Vision OCR for scanned PDFs (empty = auto-detect from vision_model)",
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
        writable=False,
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
        help_text="Context window size in tokens. Leave empty for the safe chat default (8192).",
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
    "theme": SettingDef(
        str,
        nullable=False,
        group="Display",
        help_text="TUI color theme. Cycle with Ctrl+T; the active theme persists across sessions.",
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
    "wiki_embedding_faithfulness_threshold": SettingDef(
        float,
        nullable=False,
        group="Wiki",
        help_text=(
            "Minimum cosine similarity (0-1) between a generated page and "
            "the mean of its source chunk vectors before publishing. "
            "Pages below the threshold route to drafts/."
        ),
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
    "wiki_entity_mode": SettingDef(
        str,
        nullable=False,
        group="Wiki",
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
        group="Wiki",
        help_text="Minimum chunk mentions before an entity or concept gets its own page",
    ),
    "wiki_concept_max_chunks_per_page": SettingDef(
        int,
        nullable=False,
        group="Wiki",
        help_text="Maximum chunks passed into each concept or entity page generation call",
    ),
    "wiki_related_max": SettingDef(
        int,
        nullable=False,
        group="Wiki",
        help_text="Maximum related concepts listed in the `## Related` section of each page",
    ),
    "wiki_ingest_update_cap": SettingDef(
        int,
        nullable=False,
        group="Wiki",
        help_text=(
            "Touched-page cap for auto-update after sync. "
            "Beyond this count, run `lilbee wiki update` manually."
        ),
    ),
    "wiki_summary_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.FULL,
        group="Wiki",
        help_text=(
            "Prompt for per-source summary pages. "
            "Must keep the {source_name} and {chunks_text} placeholders."
        ),
    ),
    "wiki_synthesis_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.FULL,
        group="Wiki",
        help_text=(
            "Prompt for cross-source synthesis pages. "
            "Must keep {topic}, {source_list}, and {chunks_text}."
        ),
    ),
    "wiki_entity_batch_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.FULL,
        group="Wiki",
        help_text=(
            "Prompt for the per-source batched call. "
            "Must keep {source}, {entity_list}, {chunks_text}, and {concept_instruction}."
        ),
    ),
    "wiki_extract_concepts": SettingDef(
        bool,
        nullable=False,
        group="Wiki",
        help_text=(
            "Whether the per-source batched call asks the LLM to curate concept pages "
            "alongside the pre-extracted entity list."
        ),
    ),
    "wiki_batch_min_chunks": SettingDef(
        int,
        nullable=False,
        group="Wiki",
        help_text=(
            "Minimum chunks a source must contribute before its batched call includes "
            "concept curation. Sources below the floor skip the concept-curation "
            "instruction; sources with zero entities AND below the floor are skipped entirely."
        ),
    ),
    "wiki_clusterer_k": SettingDef(
        int,
        nullable=False,
        group="Wiki",
        help_text="Mutual-kNN neighborhood size for the clusterer (0 = auto)",
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
