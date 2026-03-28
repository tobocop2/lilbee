"""Llama.cpp provider for local GGUF inference.

Includes a thread-safe batching queue for embeddings so that concurrent
ingest threads don't hit the non-thread-safe Llama object simultaneously.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.providers.base import LLMProvider, ProviderError, filter_options

if TYPE_CHECKING:
    from lilbee.registry import ModelRegistry

log = logging.getLogger(__name__)

_BATCH_WINDOW_S = 0.01  # 10ms — collect concurrent requests before dispatching

_registry: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    """Lazy-init and cache a ModelRegistry for the current models_dir."""
    global _registry
    if _registry is None:
        from lilbee.config import cfg
        from lilbee.registry import ModelRegistry as _Cls

        _registry = _Cls(cfg.models_dir)
    return _registry


@dataclass
class _EmbedRequest:
    """A single embedding request submitted to the batch queue."""

    texts: list[str]
    future: Future[list[list[float]]]


class LlamaCppProvider(LLMProvider):
    """Provider backed by llama-cpp-python for local GGUF model inference.

    Embedding calls are funnelled through a single background worker thread
    that batches concurrent requests into one ``create_embedding`` call.
    Chat calls are serialized via a lock (no batching possible).
    """

    def __init__(self) -> None:
        self._chat_llm: Any | None = None
        self._embed_llm: Any | None = None
        self._embed_queue: queue.Queue[_EmbedRequest | None] = queue.Queue()
        self._chat_lock = threading.Lock()
        self._worker = threading.Thread(target=self._embed_worker, daemon=True)
        self._worker.start()

    def _embed_worker(self) -> None:
        """Background thread: drain queue, batch, inference, dispatch results."""
        while True:
            first = self._embed_queue.get()
            if first is None:
                break

            batch: list[_EmbedRequest] = [first]
            shutting_down = False
            deadline = time.monotonic() + _BATCH_WINDOW_S
            while time.monotonic() < deadline:
                try:
                    req = self._embed_queue.get_nowait()
                    if req is None:
                        shutting_down = True
                        break
                    batch.append(req)
                except queue.Empty:
                    time.sleep(0.001)
                    continue

            self._dispatch_batch(batch)
            if shutting_down:
                break

    def _dispatch_batch(self, batch: list[_EmbedRequest]) -> None:
        """Serialize embedding requests one-by-one and resolve all futures."""
        llm = self._get_embed_llm()
        for req in batch:
            try:
                response = llm.create_embedding(input=req.texts)
                vectors = [item["embedding"] for item in response["data"]]
                req.future.set_result(vectors)
            except Exception as exc:
                if not req.future.done():
                    req.future.set_exception(exc)

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
        """Submit embedding request to the batch queue. Thread-safe."""
        fut: Future[list[list[float]]] = Future()
        self._embed_queue.put(_EmbedRequest(texts=texts, future=fut))
        return fut.result()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | Iterator[str]:
        """Chat completion — serialized via lock (Llama is not thread-safe)."""
        self._chat_lock.acquire()
        try:
            llm = self._get_chat_llm(model)
            kwargs: dict[str, Any] = {}
            if options:
                kwargs.update(filter_options(options))
            response = llm.create_chat_completion(messages=messages, stream=stream, **kwargs)
            if stream:
                return _LockedStreamIterator(response, self._chat_lock)
            result: str = response["choices"][0]["message"]["content"] or ""
            return result
        finally:
            if not stream:
                self._chat_lock.release()

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

    def shutdown(self) -> None:
        """Stop the embed worker thread."""
        self._embed_queue.put(None)
        self._worker.join(timeout=2)


class _LockedStreamIterator:
    """Wraps a streaming response so the chat lock is held until iteration ends.

    The lock must already be acquired by the caller; this iterator releases it
    when the underlying stream is exhausted (or on explicit close).
    """

    def __init__(self, response: Any, lock: threading.Lock) -> None:
        self._response = response
        self._lock = lock
        self._released = False

    def __iter__(self) -> _LockedStreamIterator:
        return self

    def __next__(self) -> str:
        try:
            while True:
                try:
                    chunk = next(self._response)
                except StopIteration:
                    self._release()
                    raise
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content: str | None = delta.get("content")
                if content:
                    return content
        except StopIteration:
            raise
        except Exception:
            self._release()
            raise

    def _release(self) -> None:
        if not self._released:
            self._released = True
            self._lock.release()

    def close(self) -> None:
        """Explicitly release the lock if the stream is abandoned early."""
        self._release()

    def __del__(self) -> None:  # pragma: no cover
        self._release()


def _resolve_model_path(model: str) -> Path:
    """Resolve a model name to a .gguf file path.

    Resolution order:
    1. Registry manifest -> blob
    2. Direct .gguf filename in models_dir
    3. Append .gguf extension to model name
    4. Prefix match (e.g. "nomic-embed-text" -> "nomic-embed-text-v1.5.Q4_K_M.gguf")
    """
    from lilbee.config import cfg

    registry = _get_registry()
    try:
        return registry.resolve(model)
    except (KeyError, ValueError):
        pass

    models_dir = cfg.models_dir

    # Direct path
    if model.endswith(".gguf"):
        candidate = models_dir / Path(model).name
        if candidate.exists():
            from lilbee.security import validate_path_within

            return validate_path_within(candidate, models_dir)
        raise ProviderError(f"Model file not found: {model}", provider="llama-cpp")

    # Try common extensions
    for ext in (".gguf",):
        candidate = models_dir / f"{model}{ext}"
        if candidate.exists():
            return candidate

    # Prefix match: "nomic-embed-text" matches "nomic-embed-text-v1.5.Q4_K_M.gguf"
    # Uses *.gguf glob + startswith filter to avoid glob injection
    if models_dir.exists():
        candidates = sorted(p for p in models_dir.glob("*.gguf") if p.name.startswith(model))
        if candidates:
            return candidates[0]

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
