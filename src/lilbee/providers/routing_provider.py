"""Routing provider: prefix-based dispatch between the SDK backend and llama-cpp."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, overload

from lilbee.catalog import is_rerank_ref
from lilbee.core.config import cfg
from lilbee.providers.base import ClosableIterator, LLMProvider, ProviderError
from lilbee.providers.litellm_sdk import LitellmSdkBackend
from lilbee.providers.model_ref import ProviderModelRef, parse_model_ref
from lilbee.providers.sdk_llm_provider import SdkLLMProvider
from lilbee.providers.worker.transport import OcrBackend
from lilbee.vision import PageText

log = logging.getLogger(__name__)

_NATIVE_GGUF_REF_MIN_SLASHES = 2
"""``<org>/<repo>/<filename>.gguf`` has at least two slashes."""


class RoutingProvider(LLMProvider):
    """Dispatches calls based on the model ref prefix.

    ``ollama/``, ``openai/``, ``anthropic/``, ``gemini/`` go to the SDK
    provider. Other refs (the HuggingFace ``<org>/<repo>/<file>.gguf``
    shape) go to llama-cpp, which resolves them against the native
    registry. A registry miss surfaces the native ProviderError
    unchanged, rather than silently falling through to a remote backend.
    """

    def __init__(self) -> None:
        self._llama_cpp: LLMProvider | None = None
        self._sdk_provider: SdkLLMProvider | None = None

    def _get_llama_cpp(self) -> LLMProvider:
        if self._llama_cpp is None:
            from lilbee.providers.llama_cpp import LlamaCppProvider

            self._llama_cpp = LlamaCppProvider()
        return self._llama_cpp

    def _get_sdk_provider(self) -> SdkLLMProvider:
        if self._sdk_provider is None:
            self._sdk_provider = SdkLLMProvider(
                LitellmSdkBackend(),
                base_url=cfg.remote_base_url,
                api_key=cfg.llm_api_key,
            )
        return self._sdk_provider

    def _pick_backend(self, ref: ProviderModelRef) -> LLMProvider:
        """Pick the backend for *ref* purely by prefix."""
        if ref.is_remote:
            return self._get_sdk_provider()
        return self._get_llama_cpp()

    def embed(self, texts: list[str]) -> list[list[float]]:
        ref = parse_model_ref(cfg.embedding_model)
        return self._pick_backend(ref).embed(texts)

    @overload
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: Literal[False] = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: Literal[True],
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> ClosableIterator[str]: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | ClosableIterator[str]:
        ref = parse_model_ref(model or cfg.chat_model)
        backend = self._pick_backend(ref)
        # Split on stream so each call resolves to a specific overload; the
        # base impl signature accepts bool but the @overloads on the LLMProvider
        # Protocol require Literal narrowing at the boundary.
        if stream:
            return backend.chat(messages, stream=True, options=options, model=model)
        return backend.chat(messages, stream=False, options=options, model=model)

    def vision_ocr(
        self,
        png_bytes: bytes,
        model: str,
        prompt: str = "",
        *,
        timeout: float | None = None,
    ) -> str:
        """Dispatch by ``model``'s ref prefix, same rules as :meth:`chat`."""
        ref = parse_model_ref(model)
        return self._pick_backend(ref).vision_ocr(png_bytes, model, prompt, timeout=timeout)

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
        """Dispatch by ``model``'s ref prefix, same rules as :meth:`vision_ocr`.

        Hosted refs reach :class:`SdkLLMProvider`, which raises
        ``NotImplementedError`` for PDF OCR; native refs reach the
        llama-cpp pool worker. ``model`` is empty when the caller wants
        the configured ``cfg.vision_model`` to drive the dispatch.
        """
        ref = parse_model_ref(model or cfg.vision_model)
        return self._pick_backend(ref).pdf_ocr(
            path,
            backend=backend,
            model=model,
            per_page_timeout_s=per_page_timeout_s,
            quiet=quiet,
            on_progress=on_progress,
        )

    def list_models(self) -> list[str]:
        """Return the union of native and SDK-visible models.

        Both halves are wrapped so an unreachable remote backend or a
        missing native registry does not mask the other.
        """
        native: set[str] = set()
        with contextlib.suppress(Exception):
            native = set(self._get_llama_cpp().list_models())
        sdk = self._get_sdk_provider()
        if not sdk.available():
            return sorted(native)
        try:
            remote = set(sdk.list_models())
        except Exception:
            return sorted(native)
        return sorted(native | remote)

    def list_chat_models(self, provider: str) -> list[str]:
        """Delegate to the SDK backend; native llama-cpp has no catalog."""
        sdk = self._get_sdk_provider()
        if not sdk.available():
            return []
        return sdk.list_chat_models(provider)

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Pull via the SDK backend if installed, otherwise raise."""
        sdk = self._get_sdk_provider()
        if not sdk.available():
            raise ProviderError(f"Cannot pull model {model!r}: no pull-capable backend available")
        sdk.pull_model(model, on_progress=on_progress)

    def show_model(self, model: str) -> dict[str, Any] | None:
        """Show model info from the backend selected by the ref prefix."""
        ref = parse_model_ref(model)
        return self._pick_backend(ref).show_model(model)

    def get_capabilities(self, model: str) -> list[str]:
        """Return capability tags from the backend selected by the ref prefix."""
        ref = parse_model_ref(model)
        return self._pick_backend(ref).get_capabilities(model)

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Dispatch rerank to the backend that owns ``cfg.reranker_model``.

        Native GGUF refs go to llama-cpp; hosted refs go through the SDK
        provider. Raises ``ProviderError`` when ``cfg.reranker_model`` is
        empty or the selected backend does not support reranking.
        """
        if not cfg.reranker_model:
            raise ProviderError("No reranker configured. Set cfg.reranker_model first.")
        if _is_native_rerank_ref(cfg.reranker_model):
            return self._get_llama_cpp().rerank(query, candidates)
        sdk = self._get_sdk_provider()
        if not sdk.supports_rerank():
            raise ProviderError(
                f"Cannot rerank with {cfg.reranker_model!r}: "
                "hosted rerank backend not available. "
                "Install the 'litellm' extra to enable hosted reranking."
            )
        return sdk.rerank(query, candidates)

    def supports_rerank(self) -> bool:
        """Capability probe: can the routed backend rerank if configured?

        Pure capability check, NOT "a reranker is currently active". An
        empty ``cfg.reranker_model`` returns ``True`` so the settings UI
        keeps the picker visible; callers that need to know whether
        reranking is actually configured must check ``bool(cfg.reranker_model)``
        separately. Delegates to the backend that would handle the
        configured model when one is set.
        """
        model = cfg.reranker_model
        if not model:
            return True
        if _is_native_rerank_ref(model):
            return self._get_llama_cpp().supports_rerank()
        return self._get_sdk_provider().supports_rerank()

    def shutdown(self) -> None:
        """Shut down sub-providers to release resources."""
        if self._llama_cpp is not None:
            self._llama_cpp.shutdown()
        if self._sdk_provider is not None:
            self._sdk_provider.shutdown()

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        """Forward to the native side only; the SDK side has no local cache."""
        if self._llama_cpp is not None:
            self._llama_cpp.invalidate_load_cache(model_path)

    def warm_up_pool(self) -> None:
        """Forward to the native side; the SDK side has no worker pool.

        Lazily constructs the llama-cpp provider if it isn't already up so
        eager-start during ``Services`` boot still warms the configured
        native roles, even when the user hasn't issued a chat call yet.
        """
        self._get_llama_cpp().warm_up_pool()


def _is_native_rerank_ref(model: str) -> bool:
    """Return True iff *model* should route to the native llama-cpp rerank worker.

    Two acceptance paths:

    1. The ref resolves to a featured rerank catalog entry (the historical
       fast path).
    2. The ref has the native HuggingFace GGUF shape
       ``<org>/<repo>/<filename>.gguf`` (two slashes, ``.gguf`` suffix). This
       lets users point ``cfg.reranker_model`` at any installed native GGUF
       reranker instead of only the ones that ship in ``FEATURED_ALL``.
       Non-GGUF refs without a known SDK prefix still raise downstream
       through ``parse_model_ref``.
    """
    if not model:
        return False
    if is_rerank_ref(model):
        return True
    return model.lower().endswith(".gguf") and model.count("/") >= _NATIVE_GGUF_REF_MIN_SLASHES
