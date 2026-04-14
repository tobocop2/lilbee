"""LiteLLM provider for external LLM backends.

For users who manage models with external tools like Ollama, or want to connect
to frontier AI APIs (OpenAI, Anthropic, Azure). Install with ``pip install
lilbee[litellm]``. By default, lilbee manages its own models via llama-cpp.
"""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from lilbee.config import cfg
from lilbee.providers.base import LLMProvider, ProviderError
from lilbee.providers.model_ref import parse_model_ref, translate_options

log = logging.getLogger(__name__)

# HTTP timeout for litellm API calls (seconds)
_HTTP_TIMEOUT = 30

_OLLAMA_URL_PATTERNS = ("localhost:11434", "127.0.0.1:11434", "ollama")


# Single source of truth for per-provider API key configuration.
# Maps (litellm_provider_name, config_field, env_var, display_label).
# Used by inject_provider_keys(), discover_api_models(), and config update handler.
PROVIDER_KEYS: tuple[tuple[str, str, str, str], ...] = (
    ("openai", "openai_api_key", "OPENAI_API_KEY", "OpenAI"),
    ("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY", "Anthropic"),
    ("gemini", "gemini_api_key", "GEMINI_API_KEY", "Gemini"),
)

# Derived set of config field names (for checking which updates touch API keys).
API_KEY_FIELDS: frozenset[str] = frozenset(t[1] for t in PROVIDER_KEYS)


def _is_ollama(base_url: str) -> bool:
    """Return True if *base_url* points to an Ollama instance."""
    url_lower = base_url.lower()
    return any(p in url_lower for p in _OLLAMA_URL_PATTERNS)


def inject_provider_keys() -> None:
    """Copy per-provider API keys from config into ``os.environ``.

    litellm reads provider-specific env vars (``OPENAI_API_KEY``, etc.)
    at call time. This bridges the gap between lilbee's config system
    and litellm's env-var-based auth. Explicit env vars are never
    overwritten so users can still override via their shell.
    """
    for _, cfg_field, env_var, _ in PROVIDER_KEYS:
        value = getattr(cfg, cfg_field, "")
        if value and not os.environ.get(env_var):
            os.environ[env_var] = value


def litellm_available() -> bool:
    """Return True if litellm is installed."""
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


class LiteLLMProvider(LLMProvider):
    """Provider backed by litellm for external APIs (Ollama, OpenAI, Azure, etc.)."""

    @staticmethod
    def available() -> bool:
        """Return True if litellm is installed."""
        return litellm_available()

    def __init__(self, *, base_url: str = "http://localhost:11434", api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def _model_name(self, model: str | None = None) -> str:
        """Route model name for litellm via parse_model_ref."""
        ref = parse_model_ref(model or cfg.chat_model)
        if ref.is_api:
            return ref.for_litellm()
        if _is_ollama(self._base_url):
            return f"ollama/{ref.name}"
        return ref.name

    def _embed_model_name(self) -> str:
        """Route embedding model name for litellm."""
        ref = parse_model_ref(cfg.embedding_model)
        if ref.is_api:
            return ref.for_litellm()
        if _is_ollama(self._base_url):
            return f"ollama/{ref.name}"
        return ref.name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via litellm."""
        import litellm

        try:
            ref = parse_model_ref(cfg.embedding_model)
            routed = self._embed_model_name()
            embed_kwargs: dict[str, Any] = {
                "model": routed,
                "input": texts,
            }
            if ref.needs_api_base:
                embed_kwargs["api_base"] = self._base_url
            if self._api_key:
                embed_kwargs["api_key"] = self._api_key
            response = litellm.embedding(**embed_kwargs)
            return [item["embedding"] for item in response["data"]]
        except Exception as exc:
            raise ProviderError(f"Embedding failed: {exc}", provider="litellm") from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | Iterator[str]:
        """Chat completion via litellm."""
        import litellm

        inject_provider_keys()
        ref = parse_model_ref(model or cfg.chat_model)
        formatted = _format_messages(messages)
        routed = self._model_name(model)
        kwargs: dict[str, Any] = {
            "model": routed,
            "messages": formatted,
            "stream": stream,
        }
        if ref.needs_api_base:
            kwargs["api_base"] = self._base_url
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if options:
            kwargs.update(translate_options(options, ref))

        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            raise ProviderError(f"Chat failed: {exc}", provider="litellm") from exc

        if stream:
            return _stream_tokens(response)
        return response.choices[0].message.content or ""

    def list_models(self) -> list[str]:
        """List models from the backend.

        Uses the Ollama ``/api/tags`` endpoint when the URL looks like
        Ollama, otherwise falls back to the OpenAI-compatible
        ``/v1/models`` endpoint.
        """
        if _is_ollama(self._base_url):
            return self._list_ollama_models()
        return self._list_openai_models()

    def _list_ollama_models(self) -> list[str]:
        """List models via the Ollama /api/tags endpoint."""
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cannot list models: {exc}", provider="litellm") from exc

    def _list_openai_models(self) -> list[str]:
        """List models via the OpenAI-compatible /v1/models endpoint."""
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = httpx.get(f"{self._base_url}/v1/models", headers=headers, timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.HTTPError:
            log.debug("Failed to list models via /v1/models", exc_info=True)
            return []

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Pull a model via the /api/pull endpoint."""
        try:
            with (
                httpx.Client(timeout=None) as client,
                client.stream(
                    "POST",
                    f"{self._base_url}/api/pull",
                    json={"name": model, "stream": True},
                ) as resp,
            ):
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    import json

                    event = json.loads(line)
                    if on_progress:
                        on_progress(event)
                    if event.get("status") == "success":
                        break
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cannot pull model {model!r}: {exc}", provider="litellm") from exc

    def show_model(self, model: str) -> dict[str, Any] | None:
        """Get model info via the /api/show endpoint.

        Parses and caches per-model generation defaults from the
        ``parameters`` field. Also extracts the ``capabilities`` list
        (newer Ollama versions) so callers can check for vision support.

        Returns a dict with ``"parameters"`` and/or ``"capabilities"``
        keys, or None on error.
        """
        try:
            resp = httpx.post(
                f"{self._base_url}/api/show",
                json={"name": model},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            result: dict[str, Any] = {}

            params = data.get("parameters", "")
            if isinstance(params, str) and params:
                _cache_ollama_defaults(model, params)
                result["parameters"] = params
            elif params:
                _cache_ollama_defaults(model, str(params))
                result["parameters"] = str(params)

            capabilities = data.get("capabilities")
            if isinstance(capabilities, list):
                result["capabilities"] = capabilities

            return result or None
        except httpx.HTTPError:
            return None

    def get_capabilities(self, model: str) -> list[str]:
        """Return capability tags for *model* from the backend.

        Uses the Ollama ``/api/show`` ``capabilities`` array when
        available; returns an empty list on error or if the backend
        does not support capabilities.
        """
        info = self.show_model(model)
        if info is None:
            return []
        caps = info.get("capabilities", [])
        return caps if isinstance(caps, list) else []

    def shutdown(self) -> None:
        """No resources to release for litellm provider."""


def _format_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert messages with images to litellm content format."""
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


def _cache_ollama_defaults(model: str, params_text: str) -> None:
    """Parse Ollama parameters and store in the model defaults cache."""
    from lilbee.model_defaults import parse_kv_parameters, set_defaults

    defaults = parse_kv_parameters(params_text)
    set_defaults(model, defaults)


def _stream_tokens(response: Any) -> Iterator[str]:
    """Extract content tokens from a streaming completion response."""
    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content
