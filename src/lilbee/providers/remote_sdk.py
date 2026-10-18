"""liter-llm implementation of the ``LlmSdkBackend`` Protocol.

The only module that imports ``liter_llm``; the SDK's wire conventions live here.
"""

from __future__ import annotations

import base64
import functools
import json
import logging
from collections.abc import Callable, Iterator
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import Any

import httpx

from lilbee.core.config import DEFAULT_HTTP_TIMEOUT
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.local_servers import (
    OLLAMA,
    detect_local_server,
    local_server_for_key,
    openai_models_url,
)
from lilbee.providers.model_ref import ProviderModelRef
from lilbee.providers.sdk_backend import (
    CompletionRequest,
    CompletionResult,
    EmbeddingRequest,
    EmbeddingResult,
    RerankRequest,
    RerankResult,
    SdkToolCall,
    SdkToolCallDelta,
    StreamChunk,
    detect_backend_name,
)
from lilbee.runtime import asyncio_loop

log = logging.getLogger(__name__)

_PROVIDER_NAME = "remote"
# Distribution and import names of the client library behind this adapter.
SDK_DIST = "liter-llm"
SDK_MODULE = "liter_llm"

# Seconds a single provider request may run. liter-llm has no unbounded default,
# so a long non-streaming generation needs an explicit ceiling.
_REQUEST_TIMEOUT_S = 600

REMOTE_SDK_MISSING_MSG = (
    "Remote and API models need the lilbee[providers] extra. If you installed lilbee "
    "with uv tool install, run your install command again with liter-llm added to the "
    "extras. If you installed lilbee with pip, run: pip install 'lilbee[providers]'."
)


@functools.cache
def remote_sdk_available() -> bool:
    """Return True if the ``liter-llm`` distribution is installed.

    Reads package metadata instead of importing the native module, so the
    Settings screen's compose does not pay for the import.
    """
    try:
        _dist_version(SDK_DIST)
    except PackageNotFoundError:
        return False
    return True


def _require_sdk() -> Any:
    """Import ``liter_llm`` or raise a user-facing ProviderError with install steps."""
    try:
        import liter_llm
    except ImportError as exc:
        raise ProviderError(REMOTE_SDK_MISSING_MSG, provider=_PROVIDER_NAME) from exc
    return liter_llm


# User-facing message per recognised error kind. Each names the problem against
# {model} and makes clear the cause sits with the user's provider account or
# network, not with lilbee. UNKNOWN has no entry and falls back to the raw error.
_KIND_MESSAGES: dict[ProviderErrorKind, str] = {
    ProviderErrorKind.RATE_LIMIT: (
        "{model} is rate-limited or out of quota. That's a limit on your provider "
        "API key, not a lilbee problem. Check your plan and billing with the "
        "provider, or pick a different model."
    ),
    ProviderErrorKind.AUTH: (
        "{model} rejected your API key. Check that the key is set correctly and has "
        "access to this model. That's between your key and the provider, not a lilbee problem."
    ),
    ProviderErrorKind.NOT_FOUND: (
        "The provider doesn't offer {model} on your account. "
        "Pick a different model or check the name."
    ),
    ProviderErrorKind.CONTEXT_OVERFLOW: (
        "This conversation is too long for {model}'s context window. "
        "Start a new chat or pick a model with a larger context."
    ),
    ProviderErrorKind.BAD_REQUEST: (
        "The provider rejected the request for {model}. Check the model name and your settings."
    ),
    ProviderErrorKind.CONNECTION: (
        "Couldn't reach the provider for {model}, or it timed out. Check your "
        "connection and base URL, then try again or pick a different model."
    ),
    ProviderErrorKind.SERVER: (
        "The provider for {model} is unavailable right now. That's on the provider's "
        "side, not a lilbee problem. Try again shortly or pick a different model."
    ),
}

# liter-llm exception class name -> error kind. Every HTTP status maps to one
# typed class in liter-llm, and the classes are flat (each subclasses
# ``LiterLlmError`` directly), so this one table covers all providers. Names, not
# classes, so the table stays importable without the native module.
_ERROR_KINDS: dict[str, ProviderErrorKind] = {
    "AuthenticationError": ProviderErrorKind.AUTH,
    "NotFoundError": ProviderErrorKind.NOT_FOUND,
    "RateLimitedError": ProviderErrorKind.RATE_LIMIT,
    "ProviderQuotaExceededError": ProviderErrorKind.RATE_LIMIT,
    "ContextWindowExceededError": ProviderErrorKind.CONTEXT_OVERFLOW,
    "BadRequestError": ProviderErrorKind.BAD_REQUEST,
    "ContentPolicyError": ProviderErrorKind.BAD_REQUEST,
    "EndpointNotSupportedError": ProviderErrorKind.BAD_REQUEST,
    "LiterLlmTimeoutError": ProviderErrorKind.CONNECTION,
    "NetworkError": ProviderErrorKind.CONNECTION,
    "StreamingError": ProviderErrorKind.CONNECTION,
    "ServiceUnavailableError": ProviderErrorKind.SERVER,
    "ServerError": ProviderErrorKind.SERVER,
}

# Operation labels prefixed onto the fallback message for an unrecognised error.
_CHAT_FAILED = "Chat failed"
_EMBED_FAILED = "Embedding failed"
_RERANK_FAILED = "Rerank failed"


def _classify_error(exc: BaseException) -> ProviderErrorKind:
    """Map a liter-llm exception to a ``ProviderErrorKind`` by type, never by message."""
    return _ERROR_KINDS.get(type(exc).__name__, ProviderErrorKind.UNKNOWN)


def _provider_error(fallback: str, exc: Exception, model: str) -> ProviderError:
    """Wrap a liter-llm failure as a ``ProviderError`` classified by type.

    Recognised kinds get a blob-free, user-facing message; unrecognised ones
    keep the raw ``{fallback}: {exc}`` shape so nothing is lost when debugging.
    """
    kind = _classify_error(exc)
    template = _KIND_MESSAGES.get(kind)
    message = template.format(model=model) if template is not None else f"{fallback}: {exc}"
    return ProviderError(message, provider=_PROVIDER_NAME, kind=kind)


def _route_model(ref: ProviderModelRef, api_base: str | None) -> str:
    """Format *ref* for liter-llm using the ``provider/model`` convention.

    API and local-server refs already carry their canonical prefix. A bare
    ``local`` ref forced through the SDK (``llm_provider=remote``) gets the
    prefix of whichever local server its ``api_base`` points at.
    """
    if ref.is_api or local_server_for_key(ref.provider) is not None:
        return ref.for_openai_prefix()
    if api_base and (spec := detect_local_server(api_base)) is not None:
        return spec.qualify(ref.name)
    return ref.name


def _wire_base_url(api_base: str | None) -> str | None:
    """Return the OpenAI-compatible endpoint root for a local server, or ``None``.

    liter-llm posts to ``{base_url}/chat/completions`` for every provider, and
    Ollama and LM Studio both serve that route under ``/v1``. A base that
    already ends in ``/v1`` is kept as it is.
    """
    if not api_base:
        return None
    trimmed = api_base.rstrip("/").removesuffix("/v1")
    return f"{trimmed}/v1"


@functools.lru_cache(maxsize=32)
def _client(model: str, base_url: str | None, api_key: str | None) -> Any:
    """Return the liter-llm client for one model, endpoint and key (cached)."""
    liter_llm = _require_sdk()
    options = liter_llm.ClientOptions(
        api_key=api_key or "",
        base_url=base_url,
        timeout_secs=_REQUEST_TIMEOUT_S,
        model_hint=model,
    )
    return liter_llm.create_client_with_options(options)


def _format_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages with inline image bytes into OpenAI content parts.

    Messages without ``images`` pass through untouched.
    """
    formatted: list[dict[str, Any]] = []
    for msg in messages:
        if "images" in msg:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": msg.get("content", "")}]
            for img in msg["images"]:
                if isinstance(img, bytes):
                    b64 = base64.b64encode(img).decode()
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
            formatted.append({"role": msg["role"], "content": content_parts})
        else:
            formatted.append(msg)
    return formatted


# lilbee option -> chat request field, for the options liter-llm takes as typed fields.
_REQUEST_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "response_format",
        "tool_choice",
    }
)
# Options sent in the request body outside the typed fields. liter-llm's
# ``reasoning_effort`` field only takes low/medium/high; lilbee sends ``none``
# to turn thinking off.
_BODY_FIELDS = frozenset({"reasoning_effort"})
# Options the OpenAI-compatible route cannot carry. They were Ollama native
# options (``options.num_ctx`` and friends) and are dropped here.
_UNSUPPORTED_OPTIONS = frozenset({"top_k", "num_ctx", "repeat_penalty"})


def _chat_fields(request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
    """Translate a ``CompletionRequest`` into liter-llm chat request fields."""
    liter_llm = _require_sdk()
    fields: dict[str, Any] = {
        "model": _route_model(request.ref, request.api_base),
        "messages": _format_messages(request.messages),
        "stream": stream,
    }
    extra_body: dict[str, Any] = {}
    for key, value in request.options.items():
        if key == "num_predict":
            fields["max_tokens"] = value
        elif key == "tools":
            fields["tools"] = [liter_llm.ChatCompletionTool.from_json(json.dumps(t)) for t in value]
        elif key in _REQUEST_FIELDS:
            fields[key] = value
        elif key in _BODY_FIELDS:
            extra_body[key] = value
        elif key in _UNSUPPORTED_OPTIONS:
            log.debug("liter-llm cannot send option %r to %s; dropped", key, fields["model"])
        else:
            extra_body[key] = value
    if extra_body:
        fields["extra_body"] = extra_body
    return fields


def _tool_call(call: Any) -> SdkToolCall:
    """Build one ``SdkToolCall`` from a liter-llm tool call."""
    return SdkToolCall(
        id=str(call.id or ""),
        name=str(call.function.name or ""),
        arguments=str(call.function.arguments or ""),
    )


def _tool_call_delta(call: Any) -> SdkToolCallDelta:
    """Build one ``SdkToolCallDelta`` from a streaming tool-call slot.

    Empty-string ``name`` / ``arguments`` are normalised to ``None`` so the
    SDK stream shape matches the native worker's deltas (the dispatch's
    ``_StreamState`` gates on ``is not None``; emitting ``""`` produces a
    spurious empty ContentBlockDelta on every opener).
    """
    function = call.function
    return SdkToolCallDelta(
        index=call.index,
        id=str(call.id) if call.id else None,
        name=str(function.name) if function is not None and function.name else None,
        arguments_delta=(
            str(function.arguments) if function is not None and function.arguments else None
        ),
    )


def _finish_reason(choice: Any) -> str | None:
    """Return a choice's ``finish_reason`` as its wire string, or ``None``."""
    reason = choice.finish_reason
    return str(reason) if reason is not None else None


class RemoteSdkBackend:
    """``LlmSdkBackend`` adapter backed by the ``liter-llm`` SDK."""

    @property
    def provider_name(self) -> str:
        """Stable identifier used when wrapping errors in ``ProviderError``."""
        return _PROVIDER_NAME

    def active_backend_name(self, base_url: str) -> str:
        """Return the display name of the backend ``base_url`` points at."""
        return detect_backend_name(base_url)

    def available(self) -> bool:
        """Return True if the underlying SDK is installed."""
        return remote_sdk_available()

    def supports_tools(self, _model_ref: str) -> bool:
        """Optimistic: all SDK-routed refs report tool support.

        A model that lacks a tool template just returns an empty
        ``tool_calls`` array, which the dispatch handles as a normal
        end-of-turn.
        """
        return True

    def configure_logging(self, *, suppress_debug: bool) -> None:
        """Do nothing: liter-llm prints no debug banner, so there is nothing to suppress."""

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run a single-shot completion through ``liter_llm.chat``."""
        liter_llm = _require_sdk()
        model = request.ref.for_display()
        try:
            fields = _chat_fields(request, stream=False)
            client = _client(fields["model"], _wire_base_url(request.api_base), request.api_key)
            response = asyncio_loop.run(
                liter_llm.chat(client, liter_llm.ChatCompletionRequest(**fields))
            )
        except Exception as exc:
            raise _provider_error(_CHAT_FAILED, exc, model) from exc
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        return CompletionResult(
            content=(message.content if message is not None else None) or "",
            finish_reason=_finish_reason(choice) if choice is not None else None,
            model=response.model or None,
            tool_calls=tuple(_tool_call(c) for c in (message.tool_calls or []))
            if message is not None
            else (),
        )

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream a completion through ``liter_llm.chat_stream``.

        The first chunk is read before returning, so a connection or auth
        failure raises here, as it does for a single-shot call.
        """
        liter_llm = _require_sdk()
        model = request.ref.for_display()
        try:
            fields = _chat_fields(request, stream=True)
            client = _client(fields["model"], _wire_base_url(request.api_base), request.api_key)
            stream = liter_llm.chat_stream(client, liter_llm.ChatCompletionRequest(**fields))
            first = asyncio_loop.run(_next_chunk(stream))
        except Exception as exc:
            raise _provider_error(_CHAT_FAILED, exc, model) from exc
        return self._stream_chunks(stream, first, model)

    @staticmethod
    def _stream_chunks(stream: Any, first: Any, model: str) -> Iterator[StreamChunk]:
        """Yield ``StreamChunk`` values from a liter-llm chunk stream.

        Exceptions raised mid-iteration are classified into ``ProviderError``
        so the semantic layer sees a consistent error type regardless of
        where the SDK failed.
        """
        chunk = first
        try:
            while chunk is not None:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is not None:
                    delta = choice.delta
                    content = (delta.content if delta is not None else None) or ""
                    finish_reason = _finish_reason(choice)
                    tool_call_deltas = tuple(
                        _tool_call_delta(c)
                        for c in ((delta.tool_calls if delta is not None else None) or [])
                    )
                    if content or finish_reason or tool_call_deltas:
                        yield StreamChunk(
                            content=content,
                            finish_reason=finish_reason,
                            tool_call_deltas=tool_call_deltas,
                        )
                chunk = asyncio_loop.run(_next_chunk(stream))
        except Exception as exc:
            raise _provider_error(_CHAT_FAILED, exc, model) from exc

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Embed inputs through ``liter_llm.embed``."""
        liter_llm = _require_sdk()
        model = _route_model(request.ref, request.api_base)
        try:
            client = _client(model, _wire_base_url(request.api_base), request.api_key)
            response = asyncio_loop.run(
                liter_llm.embed(
                    client, liter_llm.EmbeddingRequest(model=model, input=request.inputs)
                )
            )
        except Exception as exc:
            raise _provider_error(_EMBED_FAILED, exc, request.ref.for_display()) from exc
        # Order by the response's ``index`` rather than arrival order: a proxy or
        # gateway may return the batch out of order, and the consumer zips vectors
        # to inputs positionally, so a reorder would silently mis-pair every chunk
        # with the wrong vector.
        ordered = sorted(response.data, key=lambda item: item.index)
        # Reordering is not the only way the positional zip breaks. A gateway that
        # drops an item, or repeats an index, yields a batch that still sorts
        # cleanly but no longer corresponds one-to-one with the inputs. A
        # spec-conforming response carries exactly one item per input, indexed
        # 0..n-1, so anything else is refused rather than silently mis-paired.
        expected = len(request.inputs)
        if [item.index for item in ordered] != list(range(expected)):
            raise ProviderError(
                f"Embedding response does not match the request: expected {expected} "
                f"vectors indexed 0-{expected - 1}, got {len(ordered)}. The endpoint "
                "returned an incomplete or misindexed batch.",
                provider=_PROVIDER_NAME,
                kind=ProviderErrorKind.SERVER,
            )
        return EmbeddingResult(
            vectors=[list(item.embedding) for item in ordered], model=response.model or None
        )

    def rerank(self, request: RerankRequest) -> RerankResult:
        """Rerank documents via ``liter_llm.rerank`` (Cohere, Voyage, Jina, Together, HF TEI).

        The SDK returns results sorted by relevance; we restore input
        order via each result's ``index`` so scores line up with the
        caller's ``candidates`` list.
        """
        if not request.candidates:
            return RerankResult(scores=[])
        liter_llm = _require_sdk()
        model = _route_model(request.ref, request.api_base)
        try:
            client = _client(model, _wire_base_url(request.api_base), request.api_key)
            response = asyncio_loop.run(
                liter_llm.rerank(
                    client,
                    liter_llm.RerankRequest(
                        model=model, query=request.query, documents=request.candidates
                    ),
                )
            )
        except Exception as exc:
            raise _provider_error(_RERANK_FAILED, exc, request.ref.for_display()) from exc
        scores = [0.0] * len(request.candidates)
        for item in response.results:
            scores[item.index] = float(item.relevance_score)
        return RerankResult(scores=scores)

    def list_models(self, *, base_url: str, api_key: str) -> list[str]:
        """List models from Ollama (``/api/tags``) or an OpenAI-compatible ``/v1/models``."""
        clean_base = base_url.rstrip("/")
        spec = detect_local_server(clean_base)
        if spec is OLLAMA:
            return self._list_ollama_models(clean_base)
        return self._list_openai_models(clean_base, api_key)

    def list_chat_models(self, provider: str) -> list[str]:
        """Return no models: liter-llm cannot enumerate its catalog by provider.

        GAP: liter-llm answers ``model_info(name)`` for one known name but has
        no call that lists the catalog, so the frontier model picker has no
        source. The picker shows no hosted models until liter-llm adds one.
        """
        log.debug("liter-llm has no catalog listing; no chat models for %s", provider)
        return []

    @staticmethod
    def _list_ollama_models(base_url: str) -> list[str]:
        """List models via the Ollama ``/api/tags`` endpoint."""
        try:
            resp = httpx.get(f"{base_url}/api/tags", timeout=DEFAULT_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cannot list models: {exc}", provider=_PROVIDER_NAME) from exc

    @staticmethod
    def _list_openai_models(base_url: str, api_key: str) -> list[str]:
        """List models via an OpenAI-compatible ``/v1/models`` endpoint."""
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = httpx.get(
                openai_models_url(base_url), headers=headers, timeout=DEFAULT_HTTP_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.HTTPError:
            log.debug("Failed to list models via /v1/models", exc_info=True)
            return []

    def pull_model(
        self,
        model: str,
        *,
        base_url: str,
        on_progress: Callable[..., Any] | None = None,
    ) -> None:
        """Refuse to pull: local servers (Ollama, LM Studio) are read-only.

        Their models are managed in their own app and surface here once
        present, so lilbee never downloads them over the network.
        """
        spec = detect_local_server(base_url.rstrip("/"))
        server = spec.display_name if spec is not None else "This server"
        raise ProviderError(
            f"{server} doesn't download models over the network. "
            f"Add the model in its own app, then pick it here.",
            provider=_PROVIDER_NAME,
        )

    def show_model(self, model: str, *, base_url: str) -> dict[str, Any] | None:
        """Get model info via the Ollama ``/api/show`` endpoint.

        Returns the raw ``parameters`` text and the ``capabilities`` list
        (newer Ollama versions) so callers can check for vision support.
        Returns ``None`` for servers without a metadata endpoint (LM Studio).
        """
        clean_base = base_url.rstrip("/")
        spec = detect_local_server(clean_base)
        if spec is None or not spec.supports_show:
            return None
        # Ollama's API uses bare model names; the routing-layer prefix has
        # to come off before the request goes out.
        ollama_name = model.removeprefix(OLLAMA.wire_prefix)
        try:
            resp = httpx.post(
                f"{clean_base}/api/show",
                json={"name": ollama_name},
                timeout=DEFAULT_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError:
            return None

        result: dict[str, Any] = {}

        params = data.get("parameters", "")
        if isinstance(params, str) and params:
            result["parameters"] = params
        elif params:
            result["parameters"] = str(params)

        capabilities = data.get("capabilities")
        if isinstance(capabilities, list):
            result["capabilities"] = capabilities

        return result or None


async def _next_chunk(stream: Any) -> Any:
    """Return the next chunk of an async chunk stream, or ``None`` when it ends."""
    try:
        return await stream.__anext__()
    except StopAsyncIteration:
        return None
