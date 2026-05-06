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
from typing import Any

from lilbee.core.config import cfg
from lilbee.providers.base import ClosableIterator, LLMProvider, ProviderError
from lilbee.providers.model_ref import parse_model_ref, translate_options
from lilbee.providers.sdk_backend import (
    PROVIDER_KEYS,
    CompletionRequest,
    EmbeddingRequest,
    LlmSdkBackend,
    RerankRequest,
)

log = logging.getLogger(__name__)


def inject_provider_keys() -> None:
    """Copy per-provider API keys from config into ``os.environ``.

    OpenAI-compatible SDKs read provider-specific env vars
    (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...) at call time. This
    bridges lilbee's config system to that convention. Explicit env
    vars are never overwritten so users can still override via their
    shell.
    """
    for _, cfg_field, env_var, _ in PROVIDER_KEYS:
        value = getattr(cfg, cfg_field, "")
        if value and not os.environ.get(env_var):
            os.environ[env_var] = value


class SdkLLMProvider(LLMProvider):
    """Provider that delegates SDK calls to an ``LlmSdkBackend``."""

    def __init__(
        self,
        backend: LlmSdkBackend,
        *,
        base_url: str = "http://localhost:11434",
        api_key: str = "",
    ) -> None:
        self._backend = backend
        self._base_url = base_url.rstrip("/")
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via the configured backend."""
        self._ensure_initialized()
        ref = parse_model_ref(cfg.embedding_model)
        request = EmbeddingRequest(
            ref=ref,
            inputs=texts,
            api_base=self._base_url if ref.needs_api_base else None,
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

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | ClosableIterator[str]:
        """Chat completion via the configured backend."""
        self._ensure_initialized()
        ref = parse_model_ref(model or cfg.chat_model)
        translated = translate_options(options, ref) if options else {}
        request = CompletionRequest(
            ref=ref,
            messages=list(messages),
            options=translated,
            api_base=self._base_url if ref.needs_api_base else None,
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
        return result.content

    def _chat_stream(self, request: CompletionRequest) -> ClosableIterator[str]:
        """Yield content tokens from a streaming completion.

        Exceptions surfaced by the backend at either call time or during
        iteration are re-raised as ``ProviderError`` so callers always
        see a consistent error type.
        """
        try:
            stream = self._backend.complete_stream(request)
            for chunk in stream:
                if chunk.content:
                    yield chunk.content
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
        from lilbee.vision import OCR_PROMPT, build_vision_messages

        messages = build_vision_messages(prompt or OCR_PROMPT, png_bytes)
        if timeout and timeout > 0:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.chat, messages, stream=False, model=model)
                result = future.result(timeout=timeout)
        else:
            result = self.chat(messages, stream=False, model=model)
        if not isinstance(result, str):
            raise ProviderError(
                f"Vision OCR returned non-text response ({type(result).__name__}).",
                provider=self._backend.provider_name,
            )
        return result

    def list_models(self) -> list[str]:
        """List models from the backend (empty list on SDK errors)."""
        try:
            return self._backend.list_models(base_url=self._base_url, api_key=self._api_key)
        except NotImplementedError:
            return []
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Listing models failed: {exc}", provider=self._backend.provider_name
            ) from exc

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
            self._backend.pull_model(model, base_url=self._base_url, on_progress=on_progress)
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
            return self._backend.show_model(model, base_url=self._base_url)
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
            api_base=self._base_url if ref.needs_api_base else None,
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
