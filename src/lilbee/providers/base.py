"""Base protocol and exceptions for LLM providers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol


class ProviderError(Exception):
    """Raised when an LLM provider operation fails."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        self.provider = provider
        super().__init__(message)


class LLMProvider(Protocol):
    """Protocol for pluggable LLM backends."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, return list of vectors."""
        ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | Iterator[str]:
        """Chat completion. Returns str for non-stream, Iterator[str] for stream."""
        ...

    def list_models(self) -> list[str]:
        """List available model identifiers."""
        ...

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Download a model. Raises NotImplementedError if not supported."""
        ...

    def show_model(self, model: str) -> dict[str, str] | None:
        """Return model metadata, or None if backend doesn't expose it."""
        ...
