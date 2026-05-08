"""Domain enums shared by catalog, modelhub, and surfaces.

These types live in catalog/ so the layering stays one-way: catalog is the
foundation, modelhub queries catalog, and surfaces (CLI/server/MCP/TUI)
import from either layer.
"""

from __future__ import annotations

from enum import Enum, StrEnum


class ModelTask(StrEnum):
    """Task classification for models."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    VISION = "vision"
    RERANK = "rerank"


class ModelSource(Enum):
    """Where a model is stored."""

    NATIVE = "native"  # lilbee's GGUF files in cfg.models_dir
    REMOTE = "remote"  # Models managed by a remote SDK-backed service

    @classmethod
    def parse(cls, value: str | None) -> ModelSource | None:
        """Parse a user-supplied source string. Empty or None means 'all'.

        Raises ValueError on any other non-empty input so callers get a
        consistent error type regardless of entry point (CLI, MCP, server).
        """
        if value is None or value == "":
            return None
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"invalid source {value!r}; expected one of: {valid}") from exc
