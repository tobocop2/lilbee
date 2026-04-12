"""Base protocol and exceptions for LLM providers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol

from pydantic import BaseModel


class LLMOptions(BaseModel):
    """Validated options passed to LLM providers.
    Only these fields are forwarded — everything else is rejected
    to prevent injection of sensitive parameters like api_base or api_key.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    num_predict: int | None = None
    repeat_penalty: float | None = None
    num_ctx: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return only non-None values as a dict."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


def filter_options(options: dict[str, Any]) -> dict[str, Any]:
    """Validate and filter generation options through LLMOptions model."""
    return LLMOptions(**options).to_dict()


class ProviderError(Exception):
    """Raised when an LLM provider operation fails."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        self.provider = provider
        super().__init__(message)


ChatMessage = dict[str, str]


class LLMProvider(Protocol):
    """Protocol for pluggable LLM backends."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, return list of vectors."""
        ...

    def chat(
        self,
        messages: list[ChatMessage],
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

    def show_model(self, model: str) -> dict[str, Any] | None:
        """Return model metadata, or None if backend doesn't expose it."""
        ...

    def get_capabilities(self, model: str) -> list[str]:
        """Return capability tags (e.g. ``["completion", "vision"]``) for *model*.

        Returns an empty list when the backend does not support capability
        reporting or the model is not found.
        """
        ...

    def shutdown(self) -> None:
        """Release resources (e.g. background threads). No-op if nothing to clean up."""
        ...
