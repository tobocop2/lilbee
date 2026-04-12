"""LiteLLM provider for external LLM backends.

For users who manage models with external tools like Ollama, or want to connect
to frontier AI APIs (OpenAI, Anthropic, Azure). Install with ``pip install
lilbee[litellm]``. By default, lilbee manages its own models via llama-cpp.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from lilbee.config import cfg
from lilbee.providers.base import LLMProvider, ProviderError, filter_options

log = logging.getLogger(__name__)

# HTTP timeout for litellm API calls (seconds)
_HTTP_TIMEOUT = 30

# litellm routes local models via the ollama/ prefix. Any model without this
# prefix is assumed to be an ollama-hosted model and gets the prefix added.
_OLLAMA_PREFIX = "ollama/"


def _prefix_ollama(name: str) -> str:
    """Prefix ``name`` with ``ollama/`` for litellm routing if not already prefixed."""
    return name if name.startswith(_OLLAMA_PREFIX) else f"{_OLLAMA_PREFIX}{name}"


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
        """Prefix model name with ollama/ for litellm routing when needed."""
        return _prefix_ollama(model or cfg.chat_model)

    def _embed_model_name(self) -> str:
        """Prefix embedding model with ollama/ for litellm routing when needed."""
        return _prefix_ollama(cfg.embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via litellm."""
        import litellm

        try:
            response = litellm.embedding(
                model=self._embed_model_name(),
                input=texts,
                api_base=self._base_url,
                api_key=self._api_key or None,
            )
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

        formatted = _format_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model_name(model),
            "messages": formatted,
            "stream": stream,
            "api_base": self._base_url,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if options:
            kwargs.update(filter_options(options))

        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            raise ProviderError(f"Chat failed: {exc}", provider="litellm") from exc

        if stream:
            return _stream_tokens(response)
        return response.choices[0].message.content or ""

    def list_models(self) -> list[str]:
        """List models via the /api/tags endpoint."""
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cannot list models: {exc}", provider="litellm") from exc

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
