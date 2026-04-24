"""Protocol and value types for SDK-backed LLM backends.

A backend hides one third-party SDK. The ``SdkLLMProvider`` speaks to
backends exclusively through the ``LlmSdkBackend`` Protocol and the
value types defined here, so SDK response objects never leak outside
the adapter.

This module is intentionally dependency-free (no SDK imports, no
lilbee provider imports beyond the shared base types).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from lilbee.providers.model_ref import ProviderModelRef

# Single source of truth for per-provider API key configuration.
# Maps (provider_name, config_field, env_var, display_label). Backend-agnostic:
# OpenAI-compatible SDKs all read these env vars at call time.
PROVIDER_KEYS: tuple[tuple[str, str, str, str], ...] = (
    ("openai", "openai_api_key", "OPENAI_API_KEY", "OpenAI"),
    ("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY", "Anthropic"),
    ("gemini", "gemini_api_key", "GEMINI_API_KEY", "Gemini"),
)

# Derived set of config field names (for checking which updates touch API keys).
API_KEY_FIELDS: frozenset[str] = frozenset(t[1] for t in PROVIDER_KEYS)

# Display names for the active backend the SDK is talking to. The
# adapter's own identity is exposed separately via provider_name.
OLLAMA_BACKEND_NAME = "Ollama"
OPENAI_BACKEND_NAME = "OpenAI"
ANTHROPIC_BACKEND_NAME = "Anthropic"
GEMINI_BACKEND_NAME = "Gemini"
REMOTE_BACKEND_NAME = "Remote"

_BACKEND_URL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("localhost:11434", OLLAMA_BACKEND_NAME),
    ("ollama", OLLAMA_BACKEND_NAME),
    ("openai", OPENAI_BACKEND_NAME),
    ("anthropic", ANTHROPIC_BACKEND_NAME),
    ("googleapis", GEMINI_BACKEND_NAME),
    ("gemini", GEMINI_BACKEND_NAME),
)


def detect_backend_name(base_url: str) -> str:
    """Return the display name of the backend behind ``base_url``.

    Adapter-agnostic; any SDK implementation can delegate to this helper.
    Falls back to ``REMOTE_BACKEND_NAME`` when the URL matches none of
    the known patterns.
    """
    url_lower = base_url.lower()
    for pattern, name in _BACKEND_URL_PATTERNS:
        if pattern in url_lower:
            return name
    return REMOTE_BACKEND_NAME


@dataclass(frozen=True)
class CompletionResult:
    """Single-shot chat completion result returned by a backend."""

    content: str
    finish_reason: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class StreamChunk:
    """One delta yielded during a streaming chat completion."""

    content: str
    finish_reason: str | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding vectors returned by a backend for a batch of inputs."""

    vectors: list[list[float]]
    model: str | None = None


@dataclass(frozen=True)
class CompletionRequest:
    """Backend-agnostic request for a single completion call.

    ``ref`` carries the parsed model reference; the adapter converts it
    to the wire format its SDK expects. ``messages`` is the raw lilbee
    message list (may contain ``images`` bytes); the adapter formats it
    for its SDK. ``api_base`` is populated for local/Ollama deployments
    and omitted for API-hosted models.
    """

    ref: ProviderModelRef
    messages: list[dict[str, Any]]
    options: dict[str, Any] = field(default_factory=dict)
    api_base: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class EmbeddingRequest:
    """Backend-agnostic request for an embedding call."""

    ref: ProviderModelRef
    inputs: list[str]
    api_base: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class RerankRequest:
    """Backend-agnostic rerank request."""

    ref: ProviderModelRef
    query: str
    candidates: list[str]
    api_base: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class RerankResult:
    """Rerank scores returned by a backend, one per candidate in input order."""

    scores: list[float]
    model: str | None = None


class LlmSdkBackend(Protocol):
    """Protocol every LLM SDK adapter must satisfy.

    The provider calls these methods through the Protocol only; SDK
    response objects never cross the seam. Methods with a natural
    "not supported" signal are documented below.

    Lifecycle: ``available()`` is the cheap install check called before
    any other method; ``configure_logging`` runs once at first use.
    ``complete`` / ``complete_stream`` / ``embed`` are the hot-path
    operations. ``list_models`` / ``list_chat_models`` / ``pull_model``
    / ``show_model`` are catalog helpers and may raise
    ``NotImplementedError`` or return empty values when unsupported.

    Error contract: implementations must raise only ``ProviderError`` or
    ``NotImplementedError`` from any method. ``SdkLLMProvider`` wraps any
    other exception at the seam; adapters should translate SDK-specific
    errors (httpx errors, third-party SDK exceptions) into
    ``ProviderError`` so the provider can pass them through.
    """

    @property
    def provider_name(self) -> str:
        """Stable identifier used when wrapping errors in ``ProviderError``."""
        ...

    def active_backend_name(self, base_url: str) -> str:
        """Return the display name of the backend the adapter is talking to.

        ``"Ollama"`` for an Ollama URL, ``"OpenAI"`` for an OpenAI URL,
        etc.; unknown URLs fall back to ``"Remote"``. The adapter's own
        identity is exposed separately through ``provider_name``.
        """
        ...

    def available(self) -> bool:
        """Return True when the underlying SDK is importable."""
        ...

    def configure_logging(self, *, suppress_debug: bool) -> None:
        """Apply backend-level logging toggles (best-effort no-op if unsupported)."""
        ...

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run a single-shot chat completion."""
        ...

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Run a streaming chat completion, yielding content chunks."""
        ...

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Embed a batch of inputs, returning one vector per input."""
        ...

    def rerank(self, request: RerankRequest) -> RerankResult:
        """Score *candidates* against *query*, returning one float per candidate.

        Raise ``NotImplementedError`` if the backend has no rerank API.
        An empty ``request.candidates`` returns ``RerankResult([])``
        without an SDK call.
        """
        ...

    def list_models(self, *, base_url: str, api_key: str) -> list[str]:
        """List model identifiers visible to the backend. Return [] if unsupported."""
        ...

    def list_chat_models(self, provider: str) -> list[str]:
        """List chat-mode models from the SDK's catalog for *provider*.

        Return ``[]`` if the backend has no catalog of frontier models.
        Unlike ``list_models``, this is a static pricing/capability table,
        not a runtime HTTP probe.
        """
        ...

    def pull_model(
        self,
        model: str,
        *,
        base_url: str,
        on_progress: Callable[..., Any] | None = None,
    ) -> None:
        """Pull a model. Raise NotImplementedError if unsupported."""
        ...

    def show_model(self, model: str, *, base_url: str) -> dict[str, Any] | None:
        """Return model metadata dict or None if unsupported / not found."""
        ...
