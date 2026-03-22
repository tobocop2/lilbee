"""Routing provider — auto-selects backend per model based on where it lives."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from lilbee.providers.base import LLMProvider, ProviderError

log = logging.getLogger(__name__)


class RoutingProvider(LLMProvider):
    """Routes each call to the correct backend based on model source.

    If a model exists in Ollama, uses litellm (Ollama backend).
    Otherwise, uses llama-cpp for local GGUF files.
    """

    def __init__(self) -> None:
        self._llama_cpp: LLMProvider | None = None
        self._litellm: LLMProvider | None = None
        self._ollama_models: set[str] | None = None

    def _get_llama_cpp(self) -> LLMProvider:
        if self._llama_cpp is None:
            from lilbee.providers.llama_cpp_provider import LlamaCppProvider

            self._llama_cpp = LlamaCppProvider()
        return self._llama_cpp

    def _get_litellm(self) -> LLMProvider:
        if self._litellm is None:
            from lilbee.config import cfg
            from lilbee.providers.litellm_provider import LiteLLMProvider

            self._litellm = LiteLLMProvider(base_url=cfg.ollama_url)
        return self._litellm

    def _ollama_available(self) -> set[str]:
        """Return set of models available in Ollama, cached per provider lifetime."""
        if self._ollama_models is not None:
            return self._ollama_models
        try:
            self._ollama_models = set(self._get_litellm().list_models())
        except (ProviderError, Exception):
            log.debug("Ollama not reachable, using local models only")
            self._ollama_models = set()
        return self._ollama_models

    def _is_in_ollama(self, model: str) -> bool:
        return model in self._ollama_available()

    def _provider_for(self, model: str) -> LLMProvider:
        """Pick the right provider for a given model name."""
        if self._is_in_ollama(model):
            return self._get_litellm()
        return self._get_llama_cpp()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed via whichever backend has the embedding model."""
        from lilbee.config import cfg

        return self._provider_for(cfg.embedding_model).embed(texts)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | Iterator[str]:
        """Chat via whichever backend has the model."""
        from lilbee.config import cfg

        resolved = model or cfg.chat_model
        return self._provider_for(resolved).chat(
            messages, stream=stream, options=options, model=model
        )

    def list_models(self) -> list[str]:
        """Return the union of Ollama and native GGUF models."""
        import contextlib

        native: set[str] = set()
        with contextlib.suppress(Exception):
            native = set(self._get_llama_cpp().list_models())
        ollama = self._ollama_available()
        return sorted(native | ollama)

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Pull via Ollama if available, otherwise raise."""
        if self._ollama_available() is not None:
            try:
                self._get_litellm().pull_model(model, on_progress=on_progress)
                self.invalidate_cache()
                return
            except (ProviderError, Exception):
                pass
        raise ProviderError(f"Cannot pull model {model!r}: no pull-capable backend available")

    def show_model(self, model: str) -> dict[str, str] | None:
        """Try Ollama first (has metadata), fall back to llama-cpp (returns None)."""
        if self._is_in_ollama(model):
            return self._get_litellm().show_model(model)
        return self._get_llama_cpp().show_model(model)

    def invalidate_cache(self) -> None:
        """Clear cached Ollama model list (after pull/delete)."""
        self._ollama_models = None
