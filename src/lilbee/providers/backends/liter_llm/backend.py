"""``LlmSdkBackend`` implementation backed by liter-llm."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from lilbee.core.config import DEFAULT_HTTP_TIMEOUT
from lilbee.providers import availability
from lilbee.providers.backends.liter_llm.client import _client, _require_sdk
from lilbee.providers.backends.liter_llm.config import (
    _chat_fields,
    _route_model,
    _wire_base_url,
    rerank_route,
)
from lilbee.providers.backends.liter_llm.errors import (
    _CHAT_FAILED,
    _EMBED_FAILED,
    _PROVIDER_NAME,
    _RERANK_FAILED,
    _provider_error,
)
from lilbee.providers.backends.liter_llm.events import (
    _finish_reason,
    _next_chunk,
    _reasoning,
    _text,
    _tool_call,
    _tool_call_delta,
    _usage,
)
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.local_servers import (
    OLLAMA,
    LocalServerSpec,
    detect_local_server,
    openai_models_url,
)
from lilbee.providers.reasoning import ThinkInliner, inline_reasoning
from lilbee.providers.sdk_backend import (
    CompletionRequest,
    CompletionResult,
    EmbeddingRequest,
    EmbeddingResult,
    RerankRequest,
    RerankResult,
    SdkToolCallDelta,
    StreamChunk,
    detect_backend_name,
)
from lilbee.runtime import asyncio_loop

log = logging.getLogger(__name__)

# The catalog's ``mode`` value for a chat model; embedding, image and audio models carry others.
_CHAT_MODE = "chat"
# lilbee provider key -> catalog provider id, where the two differ. The routing key
# ``gemini`` is the catalog's ``google``. Every other lilbee provider uses the same id.
_CATALOG_PROVIDER = {"gemini": "google"}


class LiterLlmBackend:
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
        return availability.provider_sdk_available()

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
            client = _client(
                fields["model"], _wire_base_url(fields["model"], request.api_base), request.api_key
            )
            response = asyncio_loop.run(
                liter_llm.chat(client, liter_llm.ChatCompletionRequest(**fields))
            )
        except Exception as exc:
            raise _provider_error(_CHAT_FAILED, exc, model) from exc
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        content = _text(message.content) if message is not None else ""
        if message is not None and request.inline_reasoning:
            content = inline_reasoning(_reasoning(message), content)
        return CompletionResult(
            content=content,
            finish_reason=_finish_reason(choice.finish_reason) if choice is not None else None,
            usage=_usage(response.usage),
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
            client = _client(
                fields["model"], _wire_base_url(fields["model"], request.api_base), request.api_key
            )
            stream = liter_llm.chat_stream(client, liter_llm.ChatCompletionRequest(**fields))
            first = asyncio_loop.run(_next_chunk(stream))
        except Exception as exc:
            raise _provider_error(_CHAT_FAILED, exc, model) from exc
        return self._stream_chunks(stream, first, model, request.inline_reasoning)

    @staticmethod
    def _stream_chunks(stream: Any, first: Any, model: str, inline: bool) -> Iterator[StreamChunk]:
        """Yield ``StreamChunk`` values from a liter-llm chunk stream.

        Exceptions raised mid-iteration are classified into ``ProviderError``
        so the semantic layer sees a consistent error type regardless of
        where the SDK failed.
        """
        chunk = first
        inliner = ThinkInliner(enabled=inline)
        try:
            while chunk is not None:
                choice = chunk.choices[0] if chunk.choices else None
                text = ""
                finish_reason = None
                tool_call_deltas: tuple[SdkToolCallDelta, ...] = ()
                if choice is not None:
                    delta = choice.delta
                    text = inliner.feed(_reasoning(delta), delta.content or "")
                    finish_reason = _finish_reason(choice.finish_reason)
                    tool_call_deltas = tuple(_tool_call_delta(c) for c in delta.tool_calls or [])
                usage = _usage(chunk.usage) if chunk.usage is not None else None
                if finish_reason:
                    text += inliner.finish()
                if text or finish_reason or tool_call_deltas or usage is not None:
                    yield StreamChunk(
                        content=text,
                        finish_reason=finish_reason,
                        tool_call_deltas=tool_call_deltas,
                        usage=usage,
                    )
                chunk = asyncio_loop.run(_next_chunk(stream))
            if tail := inliner.finish():
                yield StreamChunk(content=tail)
        except Exception as exc:
            raise _provider_error(_CHAT_FAILED, exc, model) from exc

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Embed inputs through ``liter_llm.embed``."""
        liter_llm = _require_sdk()
        model = _route_model(request.ref, request.api_base)
        try:
            client = _client(model, _wire_base_url(model, request.api_base), request.api_key)
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
        model, base_url = rerank_route(request.ref, request.api_base, request.rerank_wire)
        try:
            client = _client(model, base_url, request.api_key)
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

    def list_models(self, *, spec: LocalServerSpec, base_url: str, api_key: str) -> list[str]:
        """List models from Ollama (``/api/tags``) or an OpenAI-compatible ``/v1/models``.

        *spec* is the configured server being listed, so the kind never depends on the
        address. Ollama is listed without a key, as on the request path.
        """
        clean_base = base_url.rstrip("/")
        if spec is OLLAMA:
            return self._list_ollama_models(clean_base)
        return self._list_openai_models(clean_base, api_key)

    def list_chat_models(self, provider: str) -> list[str]:
        """Return the bare chat-model names liter-llm's built-in catalog holds for *provider*.

        The catalog is embedded, so the call needs no network and no key. Names
        come back without the ``provider/`` prefix, sorted; the routing layer adds
        the prefix back. A provider the catalog does not hold, or an empty one,
        gives an empty list: the library reads an empty id as "every provider".
        A library that is installed but cannot import also gives an empty list,
        so the model pickers show no hosted models instead of failing.
        """
        catalog_id = _CATALOG_PROVIDER.get(provider, provider)
        if not catalog_id:
            return []
        try:
            sdk = _require_sdk()
        except ProviderError:
            log.debug("Provider library unavailable; no hosted chat models", exc_info=True)
            return []
        models = sdk.list_catalog_models(catalog_id)
        return sorted({m.model for m in models if m.mode == _CHAT_MODE})

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
