"""Base protocol and exceptions for LLM providers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, overload, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from lilbee.providers.worker.transport import (
        ChatResult,
        ChatStreamItem,
        OcrBackend,
    )
    from lilbee.vision import PageText

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ClosableIterator(Iterator[T_co], Protocol[T_co]):
    """An iterator that releases resources when ``close()`` is called.

    Streaming chat responses use this to guarantee the upstream model lock
    is released even when callers truncate the stream before exhaustion.
    Concrete implementations may additionally satisfy ``AsyncIterator``
    (the llama-cpp pool wrapper does, for non-blocking dispatch in
    ``server/chat_dispatch``); the Protocol underspecifies because some
    backends only expose a sync generator.
    """

    def close(self) -> None: ...


class LLMOptions(BaseModel):
    """Validated options passed to LLM providers.
    Only these fields are forwarded: everything else is rejected
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


class ContextWindowExceededError(ProviderError):
    """Raised when a chat prompt does not fit the loaded model's context window."""

    def __init__(self, message: str) -> None:
        super().__init__(message, provider="llama-cpp")

    @classmethod
    def from_counts(
        cls, *, requested: int, available: int, model: str
    ) -> ContextWindowExceededError:
        """Build with the canonical user-facing message from structured token counts."""
        return cls(
            f"Prompt of {requested} tokens exceeds the {available}-token "
            f"context window of {model!r}."
        )


ChatMessage = dict[str, Any]
"""One chat message; ``content`` may be a string or a list of content blocks."""


class LLMProvider(Protocol):
    """Protocol for pluggable LLM backends."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, return list of vectors."""
        ...

    @overload
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        stream: Literal[False] = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult: ...

    @overload
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        stream: Literal[True],
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ClosableIterator[ChatStreamItem]: ...

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult | ClosableIterator[ChatStreamItem]:
        """Chat completion.

        Returns :class:`ChatResult` for non-streaming requests and a
        :class:`ClosableIterator` of :data:`ChatStreamItem` for streaming
        requests (text deltas plus tool-call deltas). Pass ``tools`` to
        enable tool-calling on backends whose ``supports_tools(model)``
        returns True.
        """
        ...

    def supports_tools(self, model_ref: str) -> bool:
        """Return True iff the backend can route tool calls for *model_ref*."""
        ...

    def vision_ocr(
        self,
        png_bytes: bytes,
        model: str,
        prompt: str = "",
        *,
        timeout: float | None = None,
    ) -> str:
        """OCR one page image; ``timeout`` seconds, ``None``/``0`` = no cap."""
        ...

    def pdf_ocr(
        self,
        path: Path,
        *,
        backend: OcrBackend,
        model: str = "",
        per_page_timeout_s: float | None = None,
        quiet: bool = True,
        on_progress: Callable[..., None] | None = None,
    ) -> list[PageText]:
        """OCR every page of a PDF, returning per-page text in input order.

        Backends that cannot OCR scanned PDFs locally raise
        :class:`NotImplementedError`; ingest callers catch and log this.
        """
        ...

    def list_models(self) -> list[str]:
        """List available model identifiers."""
        ...

    def list_chat_models(self, provider: str) -> list[str]:
        """List frontier chat models the provider is aware of for *provider*.

        Returns the unfiltered upstream catalog (whatever litellm
        exposes for API providers; an empty list for backends like
        native llama-cpp that have no notion of external catalogs).
        """
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

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Score *candidates* for their relevance to *query*, one float per candidate.

        The backend resolves the reranker model from ``cfg.reranker_model``.
        Callers MUST check ``cfg.reranker_model`` is non-empty before
        calling; use :meth:`supports_rerank` for UI-render decisions.

        Returns: list of floats in input order, higher = more relevant.
        Empty ``candidates`` returns ``[]``.
        Raises :class:`ProviderError` when the backend does not support
        reranking or ``cfg.reranker_model`` is empty.
        """
        ...

    def supports_rerank(self) -> bool:
        """Capability probe: can this backend rerank *if* a model is configured?

        Pure capability check, NOT "a reranker is currently active". An
        empty ``cfg.reranker_model`` returns ``True`` so the settings UI
        keeps the picker visible; callers that need to know whether
        reranking is actually configured must check ``bool(cfg.reranker_model)``
        separately. ``rerank()`` is the gated path that requires a
        non-empty value.
        """
        return False

    def shutdown(self) -> None:
        """Release resources (e.g. background threads). No-op if nothing to clean up."""
        ...

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        """Drop loaded-model state; ``None`` evicts all, else only that path. No-op default."""
        return

    def warm_up_pool(self) -> None:
        """Eagerly register configured roles so :meth:`WorkerPool.start_eager` has work to do.

        Default no-op so providers without a worker pool (SDK / routing
        wrappers) can be passed to ``Services`` unchanged. Implemented by
        :class:`LlamaCppProvider` to register chat / embed / rerank / vision
        roles whose model is configured.
        """
        return
