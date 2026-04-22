"""Routing provider: prefix-based dispatch between the SDK backend and llama-cpp."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from typing import Any

from lilbee.catalog import is_rerank_ref
from lilbee.config import cfg
from lilbee.providers.base import LLMProvider, ProviderError
from lilbee.providers.litellm_sdk import LitellmSdkBackend, litellm_available
from lilbee.providers.model_ref import ProviderModelRef, parse_model_ref
from lilbee.providers.sdk_llm_provider import SdkLLMProvider

log = logging.getLogger(__name__)


class RoutingProvider(LLMProvider):
    """Dispatches calls based on the model ref prefix.

    ``ollama/``, ``openai/``, ``anthropic/``, ``gemini/`` go to the SDK
    provider. Unprefixed refs (``qwen3:8b``) go to llama-cpp, which
    resolves them against the native registry. A registry miss surfaces
    the native ProviderError unchanged, rather than silently falling
    through to a remote backend.
    """

    def __init__(self) -> None:
        self._llama_cpp: LLMProvider | None = None
        self._sdk_provider: LLMProvider | None = None

    def _get_llama_cpp(self) -> LLMProvider:  # pragma: no cover
        if self._llama_cpp is None:
            from lilbee.providers.llama_cpp_provider import LlamaCppProvider

            self._llama_cpp = LlamaCppProvider()
        return self._llama_cpp

    def _get_sdk_provider(self) -> LLMProvider:  # pragma: no cover
        if self._sdk_provider is None:
            self._sdk_provider = SdkLLMProvider(
                LitellmSdkBackend(),
                base_url=cfg.litellm_base_url,
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

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | Iterator[str]:
        ref = parse_model_ref(model or cfg.chat_model)
        return self._pick_backend(ref).chat(messages, stream=stream, options=options, model=model)

    def list_models(self) -> list[str]:
        """Return the union of native and SDK-visible models.

        Both halves are wrapped so an unreachable remote backend or a
        missing native registry does not mask the other.
        """
        native: set[str] = set()
        with contextlib.suppress(Exception):
            native = set(self._get_llama_cpp().list_models())
        if not litellm_available():
            return sorted(native)
        try:  # pragma: no cover
            remote = set(self._get_sdk_provider().list_models())  # pragma: no cover
        except Exception:  # pragma: no cover
            return sorted(native)  # pragma: no cover
        return sorted(native | remote)  # pragma: no cover

    def list_chat_models(self, provider: str) -> list[str]:
        """Delegate to the SDK backend; native llama-cpp has no catalog."""
        if not litellm_available():
            return []
        return self._get_sdk_provider().list_chat_models(provider)

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Pull via the SDK backend if installed, otherwise raise."""
        if not litellm_available():
            raise ProviderError(f"Cannot pull model {model!r}: no pull-capable backend available")
        self._get_sdk_provider().pull_model(model, on_progress=on_progress)  # pragma: no cover

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

        GGUF refs present in the native rerank catalog go to llama-cpp;
        everything else (Cohere, Voyage, Jina, Together AI, HF TEI) is
        treated as a hosted reranker and goes through the SDK provider.
        Hosted dispatch requires the ``litellm`` extra; otherwise we
        raise with a user-facing hint, mirroring ``pull_model`` /
        ``list_models``.
        """
        if _is_native_rerank_ref(cfg.reranker_model):
            return self._get_llama_cpp().rerank(query, candidates)
        if not litellm_available():
            raise ProviderError(
                f"Cannot rerank with {cfg.reranker_model!r}: litellm extra not installed"
            )
        return self._get_sdk_provider().rerank(query, candidates)

    def supports_rerank(self) -> bool:
        """Delegate to the backend that would handle ``cfg.reranker_model``.

        An empty reranker model (= reranking disabled) always counts as
        "supported" since disabling is a valid user choice — the UI
        should still surface the picker so the user can re-enable it.
        """
        model = cfg.reranker_model
        if not model:
            return True
        if _is_native_rerank_ref(model):
            return self._get_llama_cpp().supports_rerank()
        return litellm_available()

    def shutdown(self) -> None:
        """Shut down sub-providers to release resources."""
        if self._llama_cpp is not None:
            self._llama_cpp.shutdown()
        if self._sdk_provider is not None:
            self._sdk_provider.shutdown()


def _is_native_rerank_ref(model: str) -> bool:
    """Return True if *model* resolves to a featured rerank catalog entry.

    Thin alias over :func:`lilbee.catalog.is_rerank_ref` so the rerank
    dispatch logic and the llama-cpp provider both go through the same
    canonical check.
    """
    return is_rerank_ref(model)
