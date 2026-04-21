"""Routing provider — auto-selects backend based on litellm availability."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from lilbee.config import cfg
from lilbee.providers.base import LLMProvider, ProviderError
from lilbee.providers.model_ref import parse_model_ref

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

            self._litellm = LiteLLMProvider(base_url=cfg.litellm_base_url, api_key=cfg.llm_api_key)
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

    def _native_has(self, name: str) -> bool:
        """Return True if the model resolves natively via the local registry.

        Treats registry / path lookup failures as 'not native' so routing
        falls through to litellm when the ref isn't a local GGUF. This
        keeps Ollama-backed refs like ``nomic-embed-text:v1.5`` from
        crashing in llama-cpp when no native copy is installed (bb-0ud4).
        Any other exception (services-container bootstrap errors, etc.)
        gets logged at debug level so it doesn't silently mask real
        routing regressions.
        """
        from lilbee.providers.llama_cpp_provider import resolve_model_path

        try:
            resolve_model_path(name)
        except ProviderError:
            return False
        except Exception:
            log.debug("Native registry probe for %r raised unexpectedly", name, exc_info=True)
            return False
        return True

    def _pick_backend(self, ref: Any) -> LLMProvider:
        """Choose litellm vs llama-cpp for a parsed ref.

        Remote (API or Ollama-prefixed) refs always go to litellm. Bare
        refs prefer llama-cpp when a native copy exists, and fall through
        to litellm otherwise so litellm can try to reach them via Ollama
        (it auto-prefixes bare refs with ``ollama/`` when configured
        against an Ollama base URL).
        """
        if ref.is_api or ref.provider == "ollama":
            return self._get_litellm()
        if self._native_has(ref.name):
            return self._get_llama_cpp()
        if self._should_use_litellm():
            return self._get_litellm()
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

    def show_model(self, model: str) -> dict[str, Any] | None:
        """Show model info from the active provider."""
        return self._provider().show_model(model)

    def get_capabilities(self, model: str) -> list[str]:
        """Return capability tags from the active provider."""
        return self._provider().get_capabilities(model)

    def invalidate_cache(self) -> None:
        """Clear cached litellm detection (after pull/delete)."""
        self._use_litellm = None

    def shutdown(self) -> None:
        """Shut down sub-providers to release resources."""
        if self._llama_cpp is not None:
            self._llama_cpp.shutdown()
        if self._litellm is not None:
            self._litellm.shutdown()
