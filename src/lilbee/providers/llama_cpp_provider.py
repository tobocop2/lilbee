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
from typing import Any

from lilbee.providers.base import LLMProvider, ProviderError, filter_options

log = logging.getLogger(__name__)

_BATCH_WINDOW_S = 0.01  # 10ms — collect concurrent requests before dispatching


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
    Vision models are loaded with a CLIP chat handler for image understanding.
    """

    def __init__(self) -> None:
        from lilbee.config import cfg
        from lilbee.providers.model_cache import MemoryAwareModelCache

        self._cache = MemoryAwareModelCache(
            max_memory_fraction=cfg.gpu_memory_fraction,
            keep_alive_seconds=cfg.model_keep_alive,
            loader=_load_llama,
        )
        self._vision_llm: Any | None = None
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
        """Serialize embedding requests and resolve all futures.

        Embeds one text at a time because some model architectures (e.g.
        nomic-bert) fail with llama_decode -1 on multi-text batches.
        """
        llm = self._get_embed_llm()
        for req in batch:
            try:
                vectors: list[list[float]] = []
                for text in req.texts:
                    response = _embed_one(llm, text)
                    vectors.append(response)
                req.future.set_result(vectors)
            except Exception as exc:
                if not req.future.done():
                    req.future.set_exception(exc)

    def _get_chat_llm(self, model: str | None = None) -> Any:
        """Load or return a cached Llama instance for chat."""
        from lilbee.config import cfg

        resolved_model = model or cfg.chat_model

        if _is_vision_model(resolved_model):
            return self._get_vision_llm(resolved_model)

        model_path = _resolve_model_path(resolved_model)
        return self._cache.load_model(model_path, embedding=False)

    def _get_vision_llm(self, model: str) -> Any:
        """Lazy-load a Llama instance with a vision chat handler."""
        model_path = _resolve_model_path(model)

        cached = getattr(self._vision_llm, "_model_path", None)
        if self._vision_llm is None or cached != str(model_path):
            self._vision_llm = _load_vision_llama(model_path)
            self._vision_llm._model_path = str(model_path)
        return self._vision_llm

    def _get_embed_llm(self) -> Any:
        """Load or return a cached Llama instance for embeddings."""
        from lilbee.config import cfg

        model_path = _resolve_model_path(cfg.embedding_model)
        return self._cache.load_model(model_path, embedding=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Submit embedding request to the batch queue. Thread-safe."""
        fut: Future[list[list[float]]] = Future()
        self._embed_queue.put(_EmbedRequest(texts=texts, future=fut))
        return fut.result()

    def chat(
        self,
        messages: list[dict[str, str]],
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
                filtered = filter_options(options)
                if "num_predict" in filtered:
                    filtered["max_tokens"] = filtered.pop("num_predict")
                kwargs.update(filtered)
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
        """Return model metadata from GGUF headers."""
        try:
            path = _resolve_model_path(model)
        except ProviderError:
            return None
        return _read_gguf_metadata(path)

    def shutdown(self) -> None:
        """Stop the embed worker thread and unload all cached models."""
        self._embed_queue.put(None)
        self._worker.join(timeout=2)
        self._cache.unload_all()


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


_STDERR_LOCK = threading.Lock()


def _suppress_stderr(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call *fn* with C-level stderr suppressed.

    llama.cpp prints noisy messages (e.g. 'init: embeddings required...')
    that bypass Python logging. This redirects fd 2 to /dev/null for the
    duration of the call. A lock serializes access to fd 2 so concurrent
    threads don't corrupt each other's file descriptors.
    """
    import os

    with _STDERR_LOCK:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            return fn(*args, **kwargs)
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)


def _embed_one(llm: Any, text: str) -> list[float]:
    """Embed a single text with llama.cpp stderr noise suppressed."""
    response = _suppress_stderr(llm.create_embedding, input=[text])
    result: list[float] = response["data"][0]["embedding"]
    return result


def _read_gguf_metadata(model_path: Path) -> dict[str, str] | None:
    """Read metadata from a GGUF file's headers via llama-cpp-python.

    Returns a dict with keys like 'architecture', 'context_length',
    'embedding_length', 'chat_template', 'file_type'.
    """
    from llama_cpp import Llama

    llm = _suppress_stderr(
        Llama, model_path=str(model_path), vocab_only=True, verbose=False, n_gpu_layers=0
    )
    try:
        raw = llm.metadata or {}
        result: dict[str, str] = {}
        if "general.architecture" in raw:
            result["architecture"] = str(raw["general.architecture"])
        arch = raw.get("general.architecture", "llama")
        ctx_key = f"{arch}.context_length"
        if ctx_key in raw:
            result["context_length"] = str(raw[ctx_key])
        emb_key = f"{arch}.embedding_length"
        if emb_key in raw:
            result["embedding_length"] = str(raw[emb_key])
        if "tokenizer.chat_template" in raw:
            result["chat_template"] = str(raw["tokenizer.chat_template"])
        if "general.file_type" in raw:
            result["file_type"] = str(raw["general.file_type"])
        if "general.name" in raw:
            result["name"] = str(raw["general.name"])
        return result or None
    finally:
        llm.close()


def _resolve_model_path(model: str) -> Path:
    """Resolve a model name to a .gguf file path.

    Resolution order:
    1. Registry manifest -> blob
    2. Direct .gguf filename in models_dir
    3. Append .gguf extension to model name
    4. Prefix match (e.g. "nomic-embed-text" -> "nomic-embed-text-v1.5.Q4_K_M.gguf")
    """
    from lilbee.config import cfg
    from lilbee.services import get_services

    registry = get_services().registry
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

    from lilbee.config import cfg

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "embedding": embedding,
        "verbose": False,
        "n_gpu_layers": -1,  # Offload all layers to GPU (Metal/CUDA)
    }
    if cfg.num_ctx is not None:
        kwargs["n_ctx"] = cfg.num_ctx
    else:
        # n_ctx=0 tells llama.cpp to use the model's training context.
        # Without this, llama.cpp defaults to 512 tokens which is too small
        # for most embedding models (e.g. nomic-embed-text trains at 2048).
        kwargs["n_ctx"] = 0

    if embedding:
        # llama-cpp-python defaults n_batch = min(n_ctx, 512), silently
        # truncating embeddings to 512 tokens. Set n_batch = n_ctx so each
        # text can use the model's full context window.
        if kwargs["n_ctx"] == 0:
            meta = _read_gguf_metadata(model_path)
            ctx_len = int(meta.get("context_length", 2048)) if meta else 2048
        else:
            ctx_len = kwargs["n_ctx"]
        kwargs["n_batch"] = ctx_len
        kwargs["n_ubatch"] = ctx_len

    return _suppress_stderr(Llama, **kwargs)


def _is_vision_model(model: str) -> bool:
    """Check if a model name corresponds to a vision model in the catalog."""
    from lilbee.config import cfg

    if model == cfg.vision_model and cfg.vision_model:
        return True

    from lilbee.catalog import FEATURED_VISION

    model_lower = model.lower()
    return any(
        model_lower in entry.name.lower() or model_lower in entry.hf_repo.lower()
        for entry in FEATURED_VISION
    )


def _find_mmproj_for_model(model_path: Path) -> Path:
    """Find the mmproj (CLIP projection) file for a vision model.

    Searches the same directory as the model for mmproj .gguf files.
    Raises ProviderError if no mmproj file is found.
    """
    from lilbee.catalog import find_mmproj_file

    # Try catalog-aware lookup first
    mmproj = find_mmproj_file(model_path.stem)
    if mmproj is not None:
        return mmproj

    # Fallback: look in same directory as the model
    model_dir = model_path.parent
    mmproj_files = sorted(p for p in model_dir.glob("*mmproj*.gguf"))
    if mmproj_files:
        return mmproj_files[0]

    raise ProviderError(
        f"No mmproj (CLIP projection) file found for vision model {model_path.name}. "
        f"Download the mmproj file to {model_dir} or re-download the vision model "
        "through the catalog to get both files.",
        provider="llama-cpp",
    )


def _load_vision_llama(model_path: Path, mmproj_path: Path | None = None) -> Any:
    """Load a Llama instance with a vision chat handler.

    The chat handler (Llava15ChatHandler) processes image content in messages
    through the mmproj CLIP projection model.
    """
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler

    from lilbee.config import cfg

    if mmproj_path is None:
        mmproj_path = _find_mmproj_for_model(model_path)

    log.info("Loading vision model %s with mmproj %s", model_path.name, mmproj_path.name)
    chat_handler = _suppress_stderr(Llava15ChatHandler, clip_model_path=str(mmproj_path))

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "chat_handler": chat_handler,
        "verbose": False,
        "n_gpu_layers": -1,
    }
    if cfg.num_ctx is not None:
        kwargs["n_ctx"] = cfg.num_ctx
    else:
        kwargs["n_ctx"] = 0

    return _suppress_stderr(Llama, **kwargs)
