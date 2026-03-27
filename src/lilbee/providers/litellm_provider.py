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

from lilbee.providers.base import LLMProvider, ProviderError

log = logging.getLogger(__name__)

# HTTP timeout for litellm API calls (seconds)
_HTTP_TIMEOUT = 30


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
        from lilbee.config import cfg

        resolved = model or cfg.chat_model
        if not resolved.startswith("ollama/"):
            return f"ollama/{resolved}"
        return resolved

    def _embed_model_name(self) -> str:
        """Prefix embedding model with ollama/ for litellm routing when needed."""
        from lilbee.config import cfg

        name = cfg.embedding_model
        if not name.startswith("ollama/"):
            return f"ollama/{name}"
        return name

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
        messages: list[dict[str, Any]],
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
            kwargs.update(options)

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

    def show_model(self, model: str) -> dict[str, str] | None:
        """Get model info via the /api/show endpoint."""
        try:
            resp = httpx.post(
                f"{self._base_url}/api/show",
                json={"name": model},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            params = data.get("parameters", "")
            if isinstance(params, str):
                if not params:
                    return None
                return {"parameters": params}
            if params:
                return {"parameters": str(params)}
            return None
        except httpx.HTTPError:
            return None


def _format_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _stream_tokens(response: Any) -> Iterator[str]:
    """Extract content tokens from a streaming completion response."""
    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content
