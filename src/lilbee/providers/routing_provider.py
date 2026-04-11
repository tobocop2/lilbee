"""Routing provider — auto-selects backend based on litellm availability."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from lilbee.config import cfg
from lilbee.providers.base import LLMProvider, ProviderError

log = logging.getLogger(__name__)


class RoutingProvider(LLMProvider):
    """Routes all calls to litellm if available, otherwise llama-cpp.
    When litellm is installed and the backend is reachable, all operations
    go through litellm. Otherwise, falls back to llama-cpp for local GGUF files.
    """

    def __init__(self) -> None:
        self._llama_cpp: LLMProvider | None = None
        self._litellm: LLMProvider | None = None
        self._use_litellm: bool | None = None

    def _get_llama_cpp(self) -> LLMProvider:  # pragma: no cover
        if self._llama_cpp is None:
            from lilbee.providers.llama_cpp_provider import LlamaCppProvider

            self._llama_cpp = LlamaCppProvider()
        return self._llama_cpp

    def _get_litellm(self) -> LLMProvider:  # pragma: no cover
        if self._litellm is None:
            from lilbee.providers.litellm_provider import LiteLLMProvider

            self._litellm = LiteLLMProvider(base_url=cfg.litellm_base_url)
        return self._litellm

    def _should_use_litellm(self) -> bool:
        """Check once whether litellm is installed and reachable, then cache."""
        if self._use_litellm is not None:
            return self._use_litellm

        from lilbee.providers.litellm_provider import litellm_available

        if not litellm_available():
            self._use_litellm = False
            return False
        try:  # pragma: no cover
            self._get_litellm().list_models()  # pragma: no cover
            self._use_litellm = True  # pragma: no cover
        except Exception:  # pragma: no cover
            log.debug("litellm backend not reachable, using llama-cpp")  # pragma: no cover
            self._use_litellm = False  # pragma: no cover
        return self._use_litellm  # pragma: no cover

    def _provider(self) -> LLMProvider:
        """Return the active provider."""
        if self._should_use_litellm():
            return self._get_litellm()
        return self._get_llama_cpp()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider().embed(texts)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | Iterator[str]:
        return self._provider().chat(messages, stream=stream, options=options, model=model)

    def list_models(self) -> list[str]:
        """Return models from the active provider."""
        import contextlib

        native: set[str] = set()
        with contextlib.suppress(Exception):
            native = set(self._get_llama_cpp().list_models())
        if self._should_use_litellm():
            try:  # pragma: no cover
                remote = set(self._get_litellm().list_models())  # pragma: no cover
                return sorted(native | remote)  # pragma: no cover
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        return sorted(native)

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Pull via litellm if available, otherwise raise."""
        if self._should_use_litellm():
            try:  # pragma: no cover
                self._get_litellm().pull_model(model, on_progress=on_progress)  # pragma: no cover
                self.invalidate_cache()  # pragma: no cover
                return  # pragma: no cover
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        raise ProviderError(f"Cannot pull model {model!r}: no pull-capable backend available")

    def show_model(self, model: str) -> dict[str, str] | None:
        """Show model info from the active provider."""
        return self._provider().show_model(model)

    def invalidate_cache(self) -> None:
        """Clear cached litellm detection (after pull/delete)."""
        self._use_litellm = None

    def shutdown(self) -> None:
        """Shut down sub-providers to release resources."""
        if self._llama_cpp is not None:
            self._llama_cpp.shutdown()
        if self._litellm is not None:
            self._litellm.shutdown()
