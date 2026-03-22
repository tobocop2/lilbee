"""Llama.cpp provider for local GGUF inference."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from lilbee.providers.base import LLMProvider, ProviderError

log = logging.getLogger(__name__)


class LlamaCppProvider(LLMProvider):
    """Provider backed by llama-cpp-python for local GGUF model inference."""

    def __init__(self) -> None:
        self._chat_llm: Any | None = None
        self._embed_llm: Any | None = None

    def _get_chat_llm(self, model: str | None = None) -> Any:
        """Lazy-load a Llama instance for chat."""
        from lilbee.config import cfg

        resolved_model = model or cfg.chat_model
        model_path = _resolve_model_path(resolved_model)

        cached = getattr(self._chat_llm, "_model_path", None)
        if self._chat_llm is None or cached != str(model_path):
            self._chat_llm = _load_llama(model_path, embedding=False)
            self._chat_llm._model_path = str(model_path)
        return self._chat_llm

    def _get_embed_llm(self) -> Any:
        """Lazy-load a Llama instance for embeddings."""
        from lilbee.config import cfg

        model_path = _resolve_model_path(cfg.embedding_model)

        if self._embed_llm is None or getattr(self._embed_llm, "_model_path", None) != str(
            model_path
        ):
            self._embed_llm = _load_llama(model_path, embedding=True)
            self._embed_llm._model_path = str(model_path)
        return self._embed_llm

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using llama-cpp-python."""
        llm = self._get_embed_llm()
        response = llm.create_embedding(input=texts)
        return [item["embedding"] for item in response["data"]]

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | Iterator[str]:
        """Chat completion using llama-cpp-python."""
        llm = self._get_chat_llm(model)
        kwargs: dict[str, Any] = {}
        if options:
            kwargs.update(options)
        response = llm.create_chat_completion(messages=messages, stream=stream, **kwargs)
        if stream:
            return _stream_tokens(response)
        return response["choices"][0]["message"]["content"] or ""

    def list_models(self) -> list[str]:
        """List .gguf files in the models directory."""
        from lilbee.config import cfg

        models_dir = cfg.models_dir
        if not models_dir.exists():
            return []
        return sorted(p.name for p in models_dir.glob("*.gguf"))

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Not supported directly — catalog.py handles downloads."""
        raise NotImplementedError(
            f"llama-cpp provider cannot pull model {model!r}. "
            "Download GGUF files manually or use the catalog."
        )

    def show_model(self, model: str) -> dict[str, str] | None:
        """llama-cpp doesn't expose model metadata."""
        return None


def _resolve_model_path(model: str) -> Path:
    """Resolve a model name to a .gguf file path."""
    from lilbee.config import cfg

    models_dir = cfg.models_dir

    # Direct path
    if model.endswith(".gguf"):
        path = Path(model)
        if path.exists():
            return path
        # Try in models_dir
        candidate = models_dir / model
        if candidate.exists():
            return candidate
        raise ProviderError(f"Model file not found: {model}", provider="llama-cpp")

    # Try common extensions
    for ext in (".gguf",):
        candidate = models_dir / f"{model}{ext}"
        if candidate.exists():
            return candidate

    raise ProviderError(
        f"Model {model!r} not found in {models_dir}. "
        f"Available: {[p.name for p in models_dir.glob('*.gguf')] if models_dir.exists() else []}",
        provider="llama-cpp",
    )


def _load_llama(model_path: Path, *, embedding: bool) -> Any:
    """Load a llama_cpp.Llama instance."""
    from llama_cpp import Llama

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "embedding": embedding,
        "verbose": False,
    }
    return Llama(**kwargs)


def _stream_tokens(response: Any) -> Iterator[str]:
    """Extract content tokens from a streaming chat completion response."""
    for chunk in response:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if content:
            yield content
