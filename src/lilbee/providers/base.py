"""Base protocol and exceptions for LLM providers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, overload, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from lilbee.providers.roles import WorkerRole
    from lilbee.providers.warm_progress import WarmProgress

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ClosableIterator(Iterator[T_co], Protocol[T_co]):
    """An iterator that releases resources when ``close()`` is called.

    Streaming chat responses use this to guarantee upstream resources (the
    fleet's in-flight request slot) are released even when callers truncate
    the stream before exhaustion. Generators satisfy this implicitly.
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
    stop: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return only non-None values as a dict."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


def filter_options(options: dict[str, Any]) -> dict[str, Any]:
    """Validate and filter generation options through LLMOptions model."""
    return LLMOptions(**options).to_dict()


def normalize_generation_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Validate options and map them to the per-call set an OpenAI/llama-server body takes.

    ``filter_options`` validates against :class:`LLMOptions`; ``num_predict`` then
    becomes ``max_tokens`` and ``num_ctx`` is dropped (a model-load param, not a
    per-call one). Shared by the fleet and SDK option translators so the mapping
    lives in one place.
    """
    if not options:
        return {}
    filtered = filter_options(options)
    if "num_predict" in filtered:
        filtered["max_tokens"] = filtered.pop("num_predict")
    filtered.pop("num_ctx", None)
    return filtered


class ProviderErrorKind(StrEnum):
    """Provider-agnostic category of a failed provider call.

    Classified by exception type at each backend boundary so callers can
    branch on the kind instead of matching message strings (which are
    provider-specific and drift between SDK versions).
    """

    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    NOT_FOUND = "not_found"
    BAD_REQUEST = "bad_request"
    CONNECTION = "connection"
    SERVER = "server"
    UNKNOWN = "unknown"


class ProviderError(Exception):
    """Raised when an LLM provider operation fails.

    ``kind`` is the provider-agnostic category; backends that can't classify a
    failure leave it ``UNKNOWN``.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        kind: ProviderErrorKind = ProviderErrorKind.UNKNOWN,
    ) -> None:
        self.provider = provider
        self.kind = kind
        super().__init__(message)


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class ToolCall:
    """One tool/function call the model requested.

    ``arguments`` is the raw JSON-encoded argument object (OpenAI's shape), left
    as a string so the caller decides how to parse and validate it. ``id`` is the
    server-assigned call id, echoed back in the tool result message.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatToolResult:
    """A chat turn that may carry tool calls alongside (or instead of) text.

    ``tool_calls`` is empty for an ordinary text answer; ``content`` is empty when
    the model returned only tool calls. Both can be populated when a model emits
    commentary plus a call.
    """

    content: str
    tool_calls: list[ToolCall]


class FinishReason(StrEnum):
    """Why a chat completion stopped, mirroring OpenAI's vocabulary."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"

    @classmethod
    def coerce(cls, raw: object) -> FinishReason:
        """Map a backend-supplied finish_reason to a member, defaulting to STOP.

        Both the streaming and non-streaming paths read finish_reason from the
        backend; an unknown or non-string value (a model that omits it) falls
        back to STOP so the dispatch reports an ordinary end-of-turn.
        """
        if isinstance(raw, str):
            try:
                return cls(raw)
            except ValueError:
                return cls.STOP
        return cls.STOP


@dataclass(frozen=True)
class TokenUsage:
    """Prompt / completion token counts for one chat call.

    Defaults to zero so a backend that reports no usage block still yields a
    well-formed result; the fleet populates these from llama-server's ``usage``.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class ChatResult:
    """Structured result from a non-streaming chat call.

    ``tool_calls`` is empty for an ordinary text answer; ``text`` is empty when
    the model returned only tool calls. ``usage`` carries the backend's token
    counts (zero when unreported). The canonical chat dispatch reads these to
    build its OpenAI/Anthropic-shaped response.
    """

    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: TokenUsage = TokenUsage()


@dataclass(frozen=True)
class ToolCallDelta:
    """Partial tool-call delta in a streaming response, accumulated by ``index``.

    ``id`` and ``name`` arrive on the opener frame for a call; ``arguments_delta``
    accumulates across subsequent frames at the same ``index``.
    """

    index: int
    id: str | None
    name: str | None
    arguments_delta: str | None


@dataclass(frozen=True)
class StreamFinish:
    """Terminal frame carrying why a streaming chat call stopped.

    Emitted once, near the end of the stream, so the dispatch can report the
    same finish_reason the non-streaming path already surfaces, notably
    ``length`` on a max_tokens truncation. Tool-call streams already infer
    TOOL_USE from their deltas, so a finish frame never downgrades that.
    """

    reason: FinishReason


ChatStreamItem = str | ToolCallDelta | TokenUsage | StreamFinish
"""One frame yielded by a streaming chat call: text token, tool-call delta, the
final token-usage summary, or the finish-reason terminator (each emitted once,
last, when the backend reports them)."""


@dataclass(frozen=True)
class EmbeddingEndpoint:
    """OpenAI-compatible embeddings endpoint, for routing third-party embedders
    (e.g. xberg's semantic chunker) at lilbee's own fleet instead of a download."""

    base_url: str
    model: str
    api_key: str


class LLMProvider(Protocol):
    """Protocol for pluggable LLM backends."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, return list of vectors."""
        ...

    def embedding_endpoint(self) -> EmbeddingEndpoint | None:
        """The OpenAI-compatible embeddings endpoint to hand to a third-party
        embedder, or None when there isn't a routable one (caller falls back)."""
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

        Non-streaming returns a :class:`ChatResult` (assistant text, any
        tool-call frames, and a finish reason). Streaming returns a
        :class:`ClosableIterator` of :data:`ChatStreamItem` (text tokens
        interleaved with :class:`ToolCallDelta` frames). ``tools`` is the
        OpenAI function-tool list; ``tool_choice`` is ``"auto"`` / ``"none"`` /
        ``"required"`` or a ``{"type": "function", ...}`` selector. A model
        that lacks tool support returns an empty ``tool_calls`` / yields no
        tool deltas rather than erroring.
        """
        ...

    def supports_tools(self, model_ref: str) -> bool:
        """Return True iff the backend can route tool calls for *model_ref*.

        Default False so backends without a tool path are never offered tools;
        tool-capable backends override this with a real probe.
        """
        return False

    def chat_with_tools(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> ChatToolResult:
        """Non-streaming chat that may return tool calls.

        ``tools`` is the OpenAI function-tool list; ``tool_choice`` is ``"auto"``
        / ``"none"`` / ``"required"`` or a specific ``{"type": "function", ...}``
        selector. Backends without tool support raise :class:`ProviderError`.
        """
        raise ProviderError("This backend does not support tool calling.")

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

    def list_models(self) -> list[str]:
        """List available model identifiers."""
        ...

    def list_chat_models(self, provider: str) -> list[str]:
        """List frontier chat models the provider is aware of for *provider*.

        Returns the unfiltered upstream catalog (whatever litellm
        exposes for API providers; an empty list for local backends
        like the llama-server fleet that have no notion of external
        catalogs).
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

    def drop_loaded_models_async(self) -> None:
        """Drop all loaded-model state off the caller's thread. No-op default.

        Like :meth:`invalidate_load_cache` with no path, but the teardown (which
        stops every server and waits on each process) runs on a background thread
        so a settings change that touches a role-agnostic load key never blocks
        the UI / request thread. The next call rebuilds with current cfg.
        """
        self.invalidate_load_cache()

    def warm_up_pool(self) -> None:
        """Eagerly start the configured role servers so the first call lands warm.

        Default no-op so providers without managed servers (SDK / routing
        wrappers) can be passed to ``Services`` unchanged. Implemented by
        :class:`FleetProvider` to spawn the chat / embed / rerank / vision
        servers whose model is configured.
        """
        return

    def cancel_inference(self) -> None:
        """Interrupt any in-flight generation. No-op default.

        The fleet engine stops a llama-server mid-generation by client
        disconnect (the caller closes the active stream), so there is no abort
        flag to flip; SDK and routing wrappers have nothing to interrupt here.
        """
        return

    def reload_role(self, role: WorkerRole, *, wait: bool = False) -> None:
        """Drop and respawn just *role*'s model so it picks up changed cfg.

        Default no-op for providers without per-role model servers. The fleet
        respawns only that role's server; other roles and their in-flight work
        are left untouched. ``wait=True`` blocks until the respawn finishes (for a
        caller already off the event loop); the default returns immediately.
        """
        return

    def role_ready(self, role: WorkerRole) -> bool:
        """Whether *role* has a healthy server now, without starting one.

        Default ``True``: providers without managed servers (SDK / routing
        wrappers) are always reachable. The fleet returns ``False`` while a role
        is still cold-starting so surfaces can show a warming state.
        """
        del role
        return True

    def max_concurrent_chats(self) -> int:
        """Upper bound on simultaneous chat generations this provider can serve.

        Default ``1``: a single in-process model cannot take concurrent generate
        calls, so chat is serialized. A server-backed provider that batches (the
        fleet) overrides this with its slot capacity, so the chat admission gate
        lets that many run at once instead of one at a time.
        """
        return 1

    def served_chat_ctx(self) -> int | None:
        """Per-slot context the active chat server runs with, or None if unknown.

        A client trims its conversation to this so a long agentic session fits
        the model's actual window instead of overflowing. Default ``None``:
        providers without a managed context (SDK wrappers) advertise nothing.
        """
        return None

    def warm_progress(self) -> WarmProgress | None:
        """Snapshot of the chat model's cold-load progress, or None when idle.

        A launcher streams this to render a real progress bar while a large chat
        model loads. Default ``None``: providers without a managed load (SDK /
        routing wrappers) expose nothing, so a launcher falls back to a plain
        spinner. The fleet returns live read / engine-load state.
        """
        return None

    def add_spawn_listener(
        self,
        *,
        on_spawning: Callable[[WorkerRole], None] | None = None,
        on_spawned: Callable[[WorkerRole], None] | None = None,
    ) -> None:
        """Subscribe to server (re)spawn lifecycle events. No-op default.

        The fleet calls ``on_spawning`` before a role's server starts and
        ``on_spawned`` once it is healthy, so the TUI can surface cold-start and
        reload progress. Providers without managed servers ignore it.
        """
        return
