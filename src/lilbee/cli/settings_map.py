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

    cfg_attr: str
    type: type
    nullable: bool
    writable: bool = True
    render: RenderStyle = field(default=RenderStyle.COMPACT)


SETTINGS_MAP: dict[str, SettingDef] = {
    "chat_model": SettingDef("chat_model", str, nullable=False, writable=False),
    "vision_model": SettingDef("vision_model", str, nullable=True, writable=False),
    "embedding_model": SettingDef("embedding_model", str, nullable=False, writable=False),
    "top_k": SettingDef("top_k", int, nullable=False),
    "temperature": SettingDef("temperature", float, nullable=True),
    "top_p": SettingDef("top_p", float, nullable=True),
    "top_k_sampling": SettingDef("top_k_sampling", int, nullable=True),
    "repeat_penalty": SettingDef("repeat_penalty", float, nullable=True),
    "num_ctx": SettingDef("num_ctx", int, nullable=True),
    "seed": SettingDef("seed", int, nullable=True),
    "system_prompt": SettingDef("system_prompt", str, nullable=False, render=RenderStyle.FULL),
    "show_reasoning": SettingDef("show_reasoning", bool, nullable=False),
    "reranker_model": SettingDef("reranker_model", str, nullable=True),
    "rerank_candidates": SettingDef("rerank_candidates", int, nullable=False),
}
