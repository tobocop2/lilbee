"""Llama.cpp provider: class, model loader, and path resolution.

Includes a thread-safe batching queue for embeddings so that concurrent
ingest threads don't hit the non-thread-safe Llama object simultaneously.
When subprocess_embed is enabled, embedding and vision calls are delegated
to a persistent child process to avoid GIL contention.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from lilbee.catalog import is_rerank_ref
from lilbee.core.config import DEFAULT_NUM_CTX, cfg
from lilbee.core.services import get_services
from lilbee.providers.base import LLMProvider, ProviderError, filter_options
from lilbee.providers.llama_cpp.batching import (
    BATCH_WINDOW_S,
    EMBED_FUTURE_TIMEOUT_S,
    RERANK_FUTURE_TIMEOUT_S,
    EmbedRequest,
    RerankRequest,
    compute_rerank_scores,
    embed_one,
)
from lilbee.providers.llama_cpp.gguf_meta import (
    find_mmproj_for_model,
    read_gguf_metadata,
)
from lilbee.providers.llama_cpp.log_dispatch import (
    install_llama_log_handler,
    suppress_native_stderr,
)
from lilbee.providers.model_cache import (
    MODE_CHAT,
    MODE_EMBED,
    MODE_RERANK,
    LoaderMode,
    MemoryAwareModelCache,
)
from lilbee.providers.worker import WorkerManager

log = logging.getLogger(__name__)

# Settings baked into Llama() at load time, or whose change picks a
# different model file. Sampling params are read per-call and excluded.
LOAD_AFFECTING_KEYS = frozenset(
    {
        "num_ctx",
        "chat_model",
        "embedding_model",
        "vision_model",
        "reranker_model",
    }
)


class LlamaCppProvider(LLMProvider):
    """Provider backed by llama-cpp-python for local GGUF model inference.
    Embedding calls are funnelled through a single background worker thread
    that batches concurrent requests into one ``create_embedding`` call.
    Chat calls are serialized via a lock (no batching possible).
    Vision models are loaded with a CLIP chat handler for image understanding.
    """

    def __init__(self) -> None:
        self._cache = MemoryAwareModelCache(
            max_memory_fraction=cfg.gpu_memory_fraction,
            keep_alive_seconds=cfg.model_keep_alive,
            loader=load_llama,
        )
        self._embed_queue: queue.Queue[EmbedRequest | None] = queue.Queue()
        self._rerank_queue: queue.Queue[RerankRequest | None] = queue.Queue()
        self._chat_lock = threading.Lock()
        self._embed_thread = threading.Thread(target=self._embed_worker, daemon=True)
        self._embed_thread.start()
        self._rerank_thread = threading.Thread(target=self._rerank_worker, daemon=True)
        self._rerank_thread.start()
        self._subprocess_worker: WorkerManager | None = None
        self._subprocess_enabled = cfg.subprocess_embed

    def _embed_worker(self) -> None:
        """Background thread: drain queue, batch, inference, dispatch results."""
        while True:
            first = self._embed_queue.get()
            if first is None:
                break

            batch: list[EmbedRequest] = [first]
            shutting_down = False
            deadline = time.monotonic() + BATCH_WINDOW_S
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

    def _dispatch_batch(self, batch: list[EmbedRequest]) -> None:
        """Serialize embedding requests and resolve all futures.
        Embeds one text at a time because some model architectures (e.g.
        nomic-bert) fail with llama_decode -1 on multi-text batches.
        """
        try:
            llm = self._get_embed_llm()
        except Exception as exc:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
            return
        for req in batch:
            try:
                vectors: list[list[float]] = []
                for text in req.texts:
                    response = embed_one(llm, text)
                    vectors.append(response)
                req.future.set_result(vectors)
            except Exception as exc:
                if not req.future.done():
                    req.future.set_exception(exc)

    def _rerank_worker(self) -> None:
        """Background thread: drain rerank queue, serialize through the model.

        The queue is unbounded; back-pressure comes from callers awaiting
        their futures synchronously.
        """
        while True:
            req = self._rerank_queue.get()
            if req is None:
                break
            self._dispatch_rerank(req)

    def _dispatch_rerank(self, req: RerankRequest) -> None:
        """Run a single rerank request and resolve its future."""
        try:
            llm = self._get_rerank_llm()
        except Exception as exc:
            if not req.future.done():
                req.future.set_exception(exc)
            return
        try:
            scores = compute_rerank_scores(llm, req.query, req.candidates)
            req.future.set_result(scores)
        except Exception as exc:
            if not req.future.done():
                req.future.set_exception(exc)

    def _get_chat_llm(self, model: str | None = None) -> Any:
        """Load or return a cached Llama instance for chat.

        Vision OCR has its own entry point (``vision_ocr``); the chat path
        never substitutes a vision model, even if the chat pick is multimodal.
        """
        resolved_model = model or cfg.chat_model
        model_path = resolve_model_path(resolved_model)
        return self._cache.load_model(model_path, mode=MODE_CHAT)

    def _get_embed_llm(self) -> Any:
        """Load or return a cached Llama instance for embeddings."""
        model_path = resolve_model_path(cfg.embedding_model)
        return self._cache.load_model(model_path, mode=MODE_EMBED)

    def _get_rerank_llm(self) -> Any:
        """Load or return a cached Llama instance for reranking."""
        model_name = cfg.reranker_model
        if not model_name:
            raise ProviderError(
                "No reranker model configured. Set cfg.reranker_model first.",
                provider="llama-cpp",
            )
        model_path = resolve_model_path(model_name)
        return self._cache.load_model(model_path, mode=MODE_RERANK)

    def _get_subprocess_worker(self) -> WorkerManager:
        """Lazy-create and return the subprocess worker."""
        if self._subprocess_worker is None:
            self._subprocess_worker = WorkerManager()
        return self._subprocess_worker

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts. Delegates to subprocess worker if enabled, with fallback."""
        if self._subprocess_enabled:
            try:
                return self._get_subprocess_worker().embed(texts)
            except (OSError, RuntimeError) as exc:
                log.warning("Subprocess embed failed, falling back to in-process: %s", exc)
                self._subprocess_enabled = False
        fut: Future[list[list[float]]] = Future()
        self._embed_queue.put(EmbedRequest(texts=texts, future=fut))
        return fut.result(timeout=EMBED_FUTURE_TIMEOUT_S)

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Score *candidates* by relevance to *query*, queued through a single worker."""
        if not candidates:
            return []
        fut: Future[list[float]] = Future()
        self._rerank_queue.put(RerankRequest(query=query, candidates=candidates, future=fut))
        return fut.result(timeout=RERANK_FUTURE_TIMEOUT_S)

    def supports_rerank(self) -> bool:
        """llama-cpp can rerank iff llama-cpp-python exposes the rank pooling type."""
        return _llama_cpp_has_rank_pooling()

    def vision_ocr(self, png_bytes: bytes, model: str, prompt: str = "") -> str:
        """Run vision OCR via the subprocess worker."""
        return self._get_subprocess_worker().vision_ocr(png_bytes, model, prompt)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | Iterator[str]:
        """Chat completion: serialized via lock (Llama is not thread-safe)."""
        self._chat_lock.acquire()
        try:
            llm = self._get_chat_llm(model)
            kwargs: dict[str, Any] = {}
            if options:
                filtered = filter_options(options)
                if "num_predict" in filtered:
                    filtered["max_tokens"] = filtered.pop("num_predict")
                filtered.pop("num_ctx", None)  # model-load param, not per-call
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
        """List installed models from registry."""
        registry = get_services().registry
        return sorted(m.ref for m in registry.list_installed())

    def list_chat_models(self, provider: str) -> list[str]:
        """llama-cpp has no frontier-provider catalog; always ``[]``."""
        return []

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Not supported directly: ``lilbee.catalog`` handles downloads."""
        raise NotImplementedError(
            f"llama-cpp provider cannot pull model {model!r}. "
            "Download GGUF files manually or use the catalog."
        )

    def show_model(self, model: str) -> dict[str, Any] | None:
        """Return model metadata from GGUF headers."""
        try:
            path = resolve_model_path(model)
        except ProviderError:
            return None
        return read_gguf_metadata(path)

    def get_capabilities(self, model: str) -> list[str]:
        """Detect capabilities from local GGUF files.

        Rerank models return ``["rerank"]``; cross-encoder GGUFs cannot
        generate text. Other models report ``"completion"``, plus
        ``"vision"`` when an mmproj sidecar is present.
        """
        if _is_rerank_model(model):
            return ["rerank"]
        caps: list[str] = ["completion"]
        try:
            path = resolve_model_path(model)
        except ProviderError:
            log.debug("resolve_model_path failed for %s", model, exc_info=True)
            return caps
        try:
            find_mmproj_for_model(path)
            caps.append("vision")
        except ProviderError:
            log.debug("no mmproj for %s", model, exc_info=True)
        return caps

    def shutdown(self) -> None:
        """Stop workers and unload all cached models."""
        self._embed_queue.put(None)
        self._embed_thread.join(timeout=2)
        self._rerank_queue.put(None)
        self._rerank_thread.join(timeout=2)
        if self._subprocess_worker is not None:
            self._subprocess_worker.stop()
            self._subprocess_worker = None
        self._cache.unload_all()

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        """Evict cached models so the next call reloads with current settings."""
        if model_path is None:
            self._cache.unload_all()
        else:
            self._cache.unload_path(model_path)


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
        """Exhaust the underlying C iterator, then release the lock.

        Simply releasing the lock without finishing inference leaves the
        llama-cpp model in an inconsistent state. The next streaming call
        would hang because the C runtime is still processing the previous
        request. Draining the iterator ensures inference completes cleanly.
        """
        if not self._released:
            try:
                for _ in self._response:
                    pass
            except Exception:  # noqa: S110 -- best-effort drain during release; ignore partial-read errors
                pass
            self._release()

    def __del__(self) -> None:  # pragma: no cover
        self._release()


def resolve_model_path(model: str) -> Path:
    """Resolve a model name to a .gguf file path.
    Resolution order:
    1. Registry (canonical source for installed models)
    2. Absolute path (if it points to an existing file)
    """
    registry = get_services().registry
    try:
        return registry.resolve(model)
    except (KeyError, ValueError):
        pass

    # Absolute path to a .gguf file
    candidate = Path(model)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise ProviderError(f"Model file not found: {model}", provider="llama-cpp")

    raise ProviderError(
        f"Model {model!r} not found in registry. "
        f"Install it via the catalog or 'lilbee model pull'.",
        provider="llama-cpp",
    )


def _llama_cpp_has_rank_pooling() -> bool:
    """Return True iff the installed llama-cpp-python exposes ``LLAMA_POOLING_TYPE_RANK``."""
    try:
        from llama_cpp import LLAMA_POOLING_TYPE_RANK  # noqa: F401
    except ImportError:
        return False
    return True


def load_llama(model_path: Path, *, mode: LoaderMode) -> Any:
    """Load a llama_cpp.Llama instance in chat, embed, or rerank mode.

    Rerank mode sets ``embedding=True`` and ``pooling_type=LLAMA_POOLING_TYPE_RANK``
    so llama.cpp emits cross-encoder scores instead of token embeddings.
    """
    from llama_cpp import Llama

    install_llama_log_handler()
    embedding = mode in (MODE_EMBED, MODE_RERANK)
    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "embedding": embedding,
        "verbose": False,
        "n_gpu_layers": -1,  # Offload all layers to GPU (Metal/CUDA)
    }
    if cfg.num_ctx is not None:
        kwargs["n_ctx"] = cfg.num_ctx
    elif embedding:
        # n_ctx=0 -> llama.cpp uses the model's training context (else 512).
        kwargs["n_ctx"] = 0
    else:
        # Cap chat at DEFAULT_NUM_CTX so 128K+ training contexts don't OOM.
        training_ctx = DEFAULT_NUM_CTX
        try:
            meta = read_gguf_metadata(model_path)
        except Exception:
            log.debug("read_gguf_metadata failed for %s", model_path, exc_info=True)
            meta = None
        if meta:
            training_ctx = int(meta.get("context_length", DEFAULT_NUM_CTX))
        kwargs["n_ctx"] = min(training_ctx, DEFAULT_NUM_CTX)

    if embedding:
        # llama-cpp-python defaults n_batch = min(n_ctx, 512), silently
        # truncating embeddings to 512 tokens. Set n_batch = n_ctx so each
        # text can use the model's full context window.
        if kwargs["n_ctx"] == 0:
            meta = read_gguf_metadata(model_path)
            ctx_len = int(meta.get("context_length", 2048)) if meta else 2048
        else:
            ctx_len = kwargs["n_ctx"]
        kwargs["n_batch"] = ctx_len
        kwargs["n_ubatch"] = ctx_len

    if mode == MODE_RERANK:
        from llama_cpp import LLAMA_POOLING_TYPE_RANK

        kwargs["pooling_type"] = LLAMA_POOLING_TYPE_RANK

    try:
        return suppress_native_stderr(Llama, **kwargs)
    except ValueError as exc:
        wrapped = _wrap_llama_load_error(model_path, kwargs, exc)
        if wrapped is None:
            raise
        raise wrapped from exc


def _wrap_llama_load_error(
    model_path: Path, kwargs: dict[str, Any], exc: ValueError
) -> ValueError | None:
    """Diagnostic ValueError for opaque llama.cpp load failures, or None to pass through."""
    err = str(exc)
    if "llama_context" not in err and "load model from file" not in err:
        return None
    try:
        size_gb = model_path.stat().st_size / (1024**3) if model_path.exists() else 0.0
    except OSError:  # pragma: no cover
        size_gb = 0.0
    n_ctx = kwargs.get("n_ctx", 0)
    n_ctx_label = n_ctx or "model default"
    parts = [
        f"Failed to load {model_path.name} ({size_gb:.1f} GB) with n_ctx={n_ctx_label}.",
    ]
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / (1024**3)
        parts.append(f"Host has {free_gb:.1f} GB free RAM.")
    except Exception as psu_exc:  # pragma: no cover
        log.debug("psutil unavailable: %s", psu_exc)
    parts.append(
        "Try a smaller model, lower LILBEE_NUM_CTX, or close other processes to free RAM. "
        f"(llama.cpp: {err})"
    )
    return ValueError(" ".join(parts))


def _is_rerank_model(model: str) -> bool:
    """Check if *model* is an exact rerank catalog entry by ref or hf_repo."""
    if not model:
        return False
    return is_rerank_ref(model)
