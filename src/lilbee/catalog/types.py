"""Domain enums for model task and source classification."""

from __future__ import annotations

from enum import StrEnum


class ModelTask(StrEnum):
    """Task classification for models."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    VISION = "vision"
    RERANK = "rerank"


class ModelSource(StrEnum):
    """Where a model is stored.

    StrEnum (not Enum) so ``ModelSource.NATIVE == "native"`` is True at
    runtime. Pre-existing tests, JSON-decoded payloads from older clients,
    and server handler comparisons against bare strings keep working
    without an explicit ``.value`` lookup or enum import.
    """

    NATIVE = "native"  # lilbee's GGUF files in cfg.models_dir
    REMOTE = "remote"  # Models managed by a remote SDK-backed service
    FRONTIER = "frontier"  # Cloud API-key models (Gemini/OpenAI/Anthropic/…)
    OLLAMA = "ollama"  # Models served by a reachable Ollama endpoint

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


class KeyStatus(StrEnum):
    """Whether a hosted provider's API key is configured."""

    READY = "ready"
    MISSING_KEY = "missing_key"


class CatalogSize(StrEnum):
    """Size bucket for catalog filtering."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class CatalogSort(StrEnum):
    """Sort key for catalog browse results."""

    FEATURED = "featured"
    DOWNLOADS = "downloads"
    NAME = "name"
    SIZE_ASC = "size_asc"
    SIZE_DESC = "size_desc"


class ModelCompat(StrEnum):
    """Whether the bundled llama.cpp runtime supports a model's architecture."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
