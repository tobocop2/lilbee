"""Shared settings map for interactive configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from lilbee.config import ClustererBackend


class RenderStyle(StrEnum):
    """How a setting is displayed in /settings."""

    COMPACT = "compact"
    FULL = "full"


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
