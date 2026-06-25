"""SDK-agnostic LLM provider implementing the public ``LLMProvider`` Protocol.

``SdkLLMProvider`` owns the semantic layer: auth key injection, option
translation, model-ref parsing, error wrapping, and lazy one-shot
backend initialization (``configure_logging`` + ``inject_provider_keys``
on first use). It speaks to the underlying SDK exclusively through an
``LlmSdkBackend``, so swapping SDKs is a one-file adapter change.

Zero direct SDK imports live here. The adapter owns SDK-specific
concerns like wire-format prefixes (``ollama/``) and OpenAI content-parts
schema for image inputs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, overload

from lilbee.core.config import cfg
from lilbee.providers.base import (
    ChatMessage,
    ChatResult,
    ChatStreamItem,
    ChatToolResult,
    ClosableIterator,
    EmbeddingEndpoint,
    FinishReason,
    LLMProvider,
    ProviderError,
    StreamFinish,
    ToolCall,
    ToolCallDelta,
)
from lilbee.providers.local_servers import LOCAL_SERVER_KEYS
from lilbee.providers.local_servers.config_urls import base_url_for, configured_local_servers
from lilbee.providers.model_ref import ProviderModelRef, parse_model_ref, translate_options
from lilbee.providers.sdk_backend import (
    PROVIDER_KEYS,
    CompletionRequest,
    EmbeddingRequest,
    LlmSdkBackend,
    RerankRequest,
)

log = logging.getLogger(__name__)


def _api_base_for(ref: ProviderModelRef) -> str | None:
    """Endpoint for a local-server ref; ``None`` for hosted APIs (no base needed)."""
    if ref.provider in LOCAL_SERVER_KEYS:
        return base_url_for(ref.provider)
    return None


def inject_provider_keys() -> None:
    """Copy per-provider API keys from config into ``os.environ``.

    OpenAI-compatible SDKs read provider-specific env vars
    (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...) at call time. This
    bridges lilbee's config system to that convention. Explicit env
    vars are never overwritten so users can still override via their
    shell.
    """
    for _, cfg_field, env_var, _ in PROVIDER_KEYS:
        # No default: every PROVIDER_KEYS field is a declared config attribute, so
        # a typo in the table should surface as AttributeError, not silently read "".
        value = getattr(cfg, cfg_field)
        if value and not os.environ.get(env_var):
            os.environ[env_var] = value


class SdkLLMProvider(LLMProvider):
    """Provider that delegates SDK calls to an ``LlmSdkBackend``."""

    def __init__(
        self,
        backend: LlmSdkBackend,
        *,
        api_key: str = "",
    ) -> None:
        self._backend = backend
        self._api_key = api_key
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Apply one-shot backend setup before the first call.

        Runs ``configure_logging(suppress_debug=cfg.json_mode)`` and
        ``inject_provider_keys()`` exactly once, regardless of whether
        the first operation is ``chat``, ``embed``, or a catalog query.
        Both steps happen together because the backend's first SDK
        import must see (a) the debug flag applied, and (b) per-provider
        API keys in ``os.environ``.
        """
        if self._initialized:
            return
        try:
            self._backend.configure_logging(suppress_debug=cfg.json_mode)
        except (ImportError, AttributeError):
            log.debug("backend.configure_logging failed", exc_info=True)
        inject_provider_keys()
        self._initialized = True

    def embedding_endpoint(self) -> EmbeddingEndpoint | None:
        # Remote SDK backends: keep a third-party embedder (xberg's semantic
        # chunker) on its own preset rather than routing it through the user's
        # API credentials/quota.
        return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via the configured backend."""
        self._ensure_initialized()
        ref = parse_model_ref(cfg.embedding_model)
        request = EmbeddingRequest(
            ref=ref,
            inputs=texts,
            api_base=_api_base_for(ref),
            api_key=self._api_key or None,
        )
        try:
            result = self._backend.embed(request)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Embedding failed: {exc}", provider=self._backend.provider_name
            ) from exc
        return result.vectors

    @overload
    def chat(
        self,
        messages: list[dict[str, str]],
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
        messages: list[dict[str, str]],
        *,
        stream: Literal[True],
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ClosableIterator[ChatStreamItem]: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult | ClosableIterator[ChatStreamItem]:
        """Chat completion via the configured backend.

        Non-streaming returns a :class:`ChatResult` carrying the assistant
        text, any tool-call frames the model emitted, and a finish reason.
        Streaming yields :data:`ChatStreamItem` frames (text tokens and
        tool-call deltas).
        """
        self._ensure_initialized()
        ref = parse_model_ref(model or cfg.chat_model)
        if tools and not self.supports_tools(model or cfg.chat_model):
            chosen = model or cfg.chat_model
            raise ProviderError(
                f"Model {chosen!r} does not support tool calls. Pick a different "
                f"chat model that advertises tool support, or remove tools from "
                f"the request.",
                provider=self._backend.provider_name,
            )
        translated = translate_options(options, ref) if options else {}
        if tools is not None:
            translated["tools"] = tools
        if tool_choice is not None:
            translated["tool_choice"] = tool_choice
        request = CompletionRequest(
            ref=ref,
            messages=list(messages),
            options=translated,
            api_base=_api_base_for(ref),
            api_key=self._api_key or None,
        )
        if stream:
            return self._chat_stream(request)
        try:
            result = self._backend.complete(request)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Chat failed: {exc}", provider=self._backend.provider_name
            ) from exc
        return ChatResult(
            text=result.content,
            tool_calls=tuple(
                ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments) for tc in result.tool_calls
            ),
            finish_reason=FinishReason.coerce(result.finish_reason),
        )

    def supports_tools(self, model_ref: str) -> bool:
        """Delegate to the backend's ``supports_tools`` probe."""
        return self._backend.supports_tools(model_ref)

    def chat_with_tools(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> ChatToolResult:
        """Tool-calling chat for remote/SDK models.

        The base stub raises, but this backend advertises tool support and ``chat``
        already forwards tools/tool_choice, so route through it instead of refusing.
        """
        result = self.chat(
            # Pass each message through whole: a tool conversation carries
            # ``tool_calls`` / ``tool_call_id`` / ``name`` that link an assistant
            # call to its result, and stripping to role+content breaks that chain.
            [dict(m) for m in messages],
            stream=False,
            options=options,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        return ChatToolResult(content=result.text, tool_calls=list(result.tool_calls))

    def _chat_stream(self, request: CompletionRequest) -> ClosableIterator[ChatStreamItem]:
        """Yield content tokens and tool-call deltas from a streaming completion.

        Exceptions surfaced by the backend at either call time or during
        iteration are re-raised as ``ProviderError`` so callers always
        see a consistent error type.
        """
        try:
            stream = self._backend.complete_stream(request)
            for chunk in stream:
                if chunk.content:
                    yield chunk.content
                for delta in chunk.tool_call_deltas:
                    yield ToolCallDelta(
                        index=delta.index,
                        id=delta.id,
                        name=delta.name,
                        arguments_delta=delta.arguments_delta,
                    )
                if chunk.finish_reason is not None:
                    # The closing chunk's finish_reason lets the dispatch report
                    # length/stop, matching the non-streaming path.
                    yield StreamFinish(reason=FinishReason.coerce(chunk.finish_reason))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Chat failed: {exc}", provider=self._backend.provider_name
            ) from exc

    def vision_ocr(
        self,
        png_bytes: bytes,
        model: str,
        prompt: str = "",
        *,
        timeout: float | None = None,
    ) -> str:
        """OCR via a multipart chat completion; ``timeout`` enforced via thread pool."""
        from lilbee.vision import build_vision_messages, resolve_ocr_prompt

        messages = build_vision_messages(prompt or resolve_ocr_prompt(model), png_bytes)
        if timeout and timeout > 0:
            from concurrent.futures import ThreadPoolExecutor

            # Don't use the context manager: its __exit__ shutdown(wait=True) would
            # block until a hung call returns, so the caller would not be freed at
            # the deadline. Shut down without waiting (matching the fleet OCR path);
            # a wedged call's worker thread lives until the backend httpx timeout.
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(self.chat, messages, stream=False, model=model)
                result = future.result(timeout=timeout)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        else:
            result = self.chat(messages, stream=False, model=model)
        if not isinstance(result, ChatResult):
            raise ProviderError(
                f"Vision OCR returned non-text response ({type(result).__name__}).",
                provider=self._backend.provider_name,
            )
        return result.text

    def list_models(self) -> list[str]:
        """List models across every configured local server.

        A single unreachable server is logged and skipped so its outage does not
        drop the models served by the other reachable servers.
        """
        names: list[str] = []
        for spec, base_url in configured_local_servers():
            try:
                names.extend(self._backend.list_models(base_url=base_url, api_key=self._api_key))
            except NotImplementedError:
                continue
            except Exception as exc:
                log.debug("Skipping unreachable local server %s: %s", spec.key, exc)
        return names

    def list_chat_models(self, provider: str) -> list[str]:
        """List frontier chat models known to the backend for *provider*.

        Initializes the backend first so ``cfg.json_mode`` suppression is
        applied before the SDK import inside the backend runs.
        """
        self._ensure_initialized()
        try:
            return self._backend.list_chat_models(provider)
        except NotImplementedError:
            return []
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Listing chat models failed: {exc}", provider=self._backend.provider_name
            ) from exc

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Pull a model via the backend."""
        try:
            base_url = _api_base_for(parse_model_ref(model)) or ""
            self._backend.pull_model(model, base_url=base_url, on_progress=on_progress)
        except NotImplementedError as exc:
            raise ProviderError(
                f"Cannot pull model {model!r}: backend does not support pulling",
                provider=self._backend.provider_name,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Cannot pull model {model!r}: {exc}", provider=self._backend.provider_name
            ) from exc

    def show_model(self, model: str) -> dict[str, Any] | None:
        """Return model metadata, or None when unsupported or not found."""
        try:
            base_url = _api_base_for(parse_model_ref(model)) or ""
            return self._backend.show_model(model, base_url=base_url)
        except NotImplementedError:
            return None
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Showing model {model!r} failed: {exc}", provider=self._backend.provider_name
            ) from exc

    def get_capabilities(self, model: str) -> list[str]:
        """Return capability tags from ``show_model`` output, or ``[]``."""
        info = self.show_model(model)
        if info is None:
            return []
        caps = info.get("capabilities", [])
        return caps if isinstance(caps, list) else []

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Rerank candidates via the SDK backend using ``cfg.reranker_model``."""
        if not candidates:
            return []
        self._ensure_initialized()
        ref = parse_model_ref(cfg.reranker_model)
        request = RerankRequest(
            ref=ref,
            query=query,
            candidates=candidates,
            api_base=_api_base_for(ref),
            api_key=self._api_key or None,
        )
        try:
            result = self._backend.rerank(request)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Rerank failed: {exc}", provider=self._backend.provider_name
            ) from exc
        return result.scores

    def supports_rerank(self) -> bool:
        """SDK-backed rerank is available when the underlying SDK is importable."""
        return self._backend.available()

    def available(self) -> bool:
        """Return True when the configured SDK backend can service catalog calls."""
        return self._backend.available()

    def shutdown(self) -> None:
        """SDK-backed providers hold no lilbee-side resources."""

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        """No-op: cloud backends have no local model cache to evict."""
