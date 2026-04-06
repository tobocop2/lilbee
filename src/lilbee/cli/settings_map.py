"""Shared settings map for interactive configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


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
    "vision_model": SettingDef(
        str,
        nullable=True,
        writable=False,
        group="Models",
        help_text="Vision model for OCR on images and PDFs",
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
}
