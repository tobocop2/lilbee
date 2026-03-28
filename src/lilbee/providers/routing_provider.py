"""Routing provider — auto-selects backend per model based on where it lives."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from lilbee.providers.base import LLMProvider, ProviderError

log = logging.getLogger(__name__)


class RoutingProvider(LLMProvider):
    """Routes each call to the correct backend based on model source.

    If litellm is installed and models are reachable via its API, uses litellm.
    Otherwise, uses llama-cpp for local GGUF files.
    """

    def __init__(self) -> None:
        self._llama_cpp: LLMProvider | None = None
        self._litellm: LLMProvider | None = None
        self._remote_models: set[str] | None = None

    def _get_llama_cpp(self) -> LLMProvider:
        if self._llama_cpp is None:
            from lilbee.providers.llama_cpp_provider import LlamaCppProvider

            self._llama_cpp = LlamaCppProvider()
        return self._llama_cpp

    def _get_litellm(self) -> LLMProvider:
        if self._litellm is None:
            from lilbee.config import cfg
            from lilbee.providers.litellm_provider import LiteLLMProvider

            self._litellm = LiteLLMProvider(base_url=cfg.litellm_base_url)
        return self._litellm

    def _litellm_models(self) -> set[str]:
        """Return set of models available via litellm, cached per provider lifetime."""
        if self._remote_models is not None:
            return self._remote_models

        from lilbee.providers.litellm_provider import litellm_available

        if not litellm_available():
            self._remote_models = set()
            return self._remote_models
        try:
            self._remote_models = set(self._get_litellm().list_models())
        except (ProviderError, Exception):
            log.debug("litellm backend not reachable, using local models only")
            self._remote_models = set()
        return self._remote_models

    def _is_in_litellm(self, model: str) -> bool:
        return model in self._litellm_models()

    def _provider_for(self, model: str) -> LLMProvider:
        """Pick the right provider for a given model name."""
        if self._is_in_litellm(model):
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
        """Return the union of litellm and native GGUF models."""
        import contextlib

        native: set[str] = set()
        with contextlib.suppress(Exception):
            native = set(self._get_llama_cpp().list_models())
        remote = self._litellm_models()
        return sorted(native | remote)

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Pull via litellm if available, otherwise raise."""
        if len(self._litellm_models()) > 0:
            try:
                self._get_litellm().pull_model(model, on_progress=on_progress)
                self.invalidate_cache()
                return
            except (ProviderError, Exception):
                pass
        raise ProviderError(f"Cannot pull model {model!r}: no pull-capable backend available")

    def show_model(self, model: str) -> dict[str, str] | None:
        """Try litellm first (has metadata), fall back to llama-cpp (returns None)."""
        if self._is_in_litellm(model):
            return self._get_litellm().show_model(model)
        return self._get_llama_cpp().show_model(model)

    def invalidate_cache(self) -> None:
        """Clear cached litellm model list (after pull/delete)."""
        self._remote_models = None

    def shutdown(self) -> None:
        """Shut down sub-providers to release resources."""
        if self._llama_cpp is not None:
            self._llama_cpp.shutdown()
        if self._litellm is not None:
            self._litellm.shutdown()
