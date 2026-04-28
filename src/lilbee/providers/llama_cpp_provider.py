"""Llama.cpp provider for local GGUF inference.

Includes a thread-safe batching queue for embeddings so that concurrent
ingest threads don't hit the non-thread-safe Llama object simultaneously.
When subprocess_embed is enabled, embedding and vision calls are delegated
to a persistent child process to avoid GIL contention.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gguf import GGUFReader, GGUFValueType

from lilbee.catalog import is_rerank_ref
from lilbee.config import DEFAULT_NUM_CTX, KvCacheType, cfg
from lilbee.providers.base import LLMProvider, ProviderError, filter_options
from lilbee.providers.model_cache import (
    KV_CACHE_TYPE_BYTES,
    MODE_CHAT,
    MODE_EMBED,
    MODE_RERANK,
    LoaderMode,
    compute_dynamic_ctx,
    get_available_memory,
    kv_bytes_per_token,
)
from lilbee.services import get_services

if TYPE_CHECKING:
    from lilbee.providers.worker_process import WorkerProcess

log = logging.getLogger(__name__)

_llama_log = logging.getLogger("lilbee.llama_cpp")

# ggml.h log levels (not exposed by llama-cpp-python).
_GGML_LOG_LEVEL_INFO = 1
_GGML_LOG_LEVEL_WARN = 2
_GGML_LOG_LEVEL_ERROR = 3
_GGML_LOG_LEVEL_DEBUG = 4
_GGML_LOG_LEVEL_CONT = 5

# WARN demotes to INFO so noisy auto-corrections stay silent at the default WARNING level.
_GGML_TO_PY_LEVEL = {
    _GGML_LOG_LEVEL_INFO: logging.DEBUG,
    _GGML_LOG_LEVEL_WARN: logging.INFO,
    _GGML_LOG_LEVEL_ERROR: logging.ERROR,
    _GGML_LOG_LEVEL_DEBUG: logging.DEBUG,
}

# Substrings llama.cpp emits at GGML_LOG_LEVEL_ERROR but which are
# advisory: the model still loads correctly. Demoted to WARNING so users
# don't think their setup is broken.
_GGML_ERROR_SOFT_DEMOTE = (
    "special_eos_id is not in special_eog_ids",
    "embeddings required but some input tokens were not marked as outputs",
    "n_ctx_seq",  # 'n_ctx_seq (X) > n_ctx_train (Y)' -- our embed clamp prevents this
    "tokenizer config may be incorrect",
)

_BATCH_WINDOW_S = 0.01  # 10ms — collect concurrent requests before dispatching

# Cap on tokens consumed during _LockedStreamIterator.close()'s drain.
# A runaway model (e.g. Qwen3-0.6B in a never-closing <think> loop)
# would otherwise block close() indefinitely.
_LOCKED_STREAM_DRAIN_CAP = 1024
_EMBED_FUTURE_TIMEOUT_S = 300.0  # Safety net: max wait for embed result
_RERANK_FUTURE_TIMEOUT_S = 300.0  # Safety net: max wait for rerank result

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


@dataclass
class _EmbedRequest:
    """A single embedding request submitted to the batch queue."""

    texts: list[str]
    future: Future[list[list[float]]]


@dataclass
class _RerankRequest:
    """A single rerank request submitted to the batch queue."""

    query: str
    candidates: list[str]
    future: Future[list[float]]


class LlamaCppProvider(LLMProvider):
    """Provider backed by llama-cpp-python for local GGUF model inference.
    Embedding calls are funnelled through a single background worker thread
    that batches concurrent requests into one ``create_embedding`` call.
    Chat calls are serialized via a lock (no batching possible).
    Vision models are loaded with a CLIP chat handler for image understanding.
    """

    def __init__(self) -> None:
        from lilbee.providers.model_cache import MemoryAwareModelCache

        self._cache = MemoryAwareModelCache(
            max_memory_fraction=cfg.gpu_memory_fraction,
            keep_alive_seconds=cfg.model_keep_alive,
            loader=load_llama,
        )
        self._embed_queue: queue.Queue[_EmbedRequest | None] = queue.Queue()
        self._rerank_queue: queue.Queue[_RerankRequest | None] = queue.Queue()
        self._chat_lock = threading.Lock()
        self._embed_thread = threading.Thread(target=self._embed_worker, daemon=True)
        self._embed_thread.start()
        self._rerank_thread = threading.Thread(target=self._rerank_worker, daemon=True)
        self._rerank_thread.start()
        self._subprocess_worker: WorkerProcess | None = None
        self._subprocess_enabled = cfg.subprocess_embed

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

    def _dispatch_rerank(self, req: _RerankRequest) -> None:
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

    def _get_subprocess_worker(self) -> WorkerProcess:
        """Lazy-create and return the subprocess worker."""
        if self._subprocess_worker is None:
            from lilbee.providers.worker_process import WorkerProcess as WP  # noqa: N817

            self._subprocess_worker = WP()
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
        self._embed_queue.put(_EmbedRequest(texts=texts, future=fut))
        return fut.result(timeout=_EMBED_FUTURE_TIMEOUT_S)

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Score *candidates* by relevance to *query*, queued through a single worker."""
        if not candidates:
            return []
        fut: Future[list[float]] = Future()
        self._rerank_queue.put(_RerankRequest(query=query, candidates=candidates, future=fut))
        return fut.result(timeout=_RERANK_FUTURE_TIMEOUT_S)

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
        """Chat completion — serialized via lock (Llama is not thread-safe)."""
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
        """Not supported directly — catalog.py handles downloads."""
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
        """Drain (capped) the underlying C iterator, then release the lock.

        Simply releasing the lock without finishing inference leaves the
        llama-cpp model in an inconsistent state. Draining lets inference
        complete cleanly. The cap (``_LOCKED_STREAM_DRAIN_CAP``) keeps a
        runaway think loop from blocking close() indefinitely; once the
        cap fires we accept the inconsistent state in exchange for not
        hanging the UI.
        """
        if not self._released:
            try:
                for i, _ in enumerate(self._response):
                    if i >= _LOCKED_STREAM_DRAIN_CAP:
                        break
            except Exception:  # noqa: S110 -- best-effort drain during release; ignore partial-read errors
                pass
            self._release()

    def __del__(self) -> None:  # pragma: no cover
        self._release()


_STDERR_LOCK = threading.Lock()

# ctypes does not retain a Python reference to the wrapped callback;
# this module-level handle keeps it alive for the process lifetime.
_llama_log_callback: Any = None
_llama_log_installed = False
_llama_log_pending: dict[int, str] = {}
_llama_log_pending_level: int = _GGML_LOG_LEVEL_INFO


def _llama_log_dispatch(level: int, text_bytes: bytes, _user_data: Any) -> None:
    """Dispatch one llama.cpp log message; CONT chunks are coalesced on newline."""
    global _llama_log_pending_level
    try:
        text = text_bytes.decode("utf-8", errors="replace") if text_bytes else ""
    except Exception:  # pragma: no cover
        return

    if level == _GGML_LOG_LEVEL_CONT:
        _llama_log_pending[0] = _llama_log_pending.get(0, "") + text
    else:
        if 0 in _llama_log_pending:
            buffered = _llama_log_pending.pop(0).rstrip()
            if buffered:
                _llama_log.log(_resolve_ggml_level(_llama_log_pending_level, buffered), buffered)
        _llama_log_pending_level = level
        _llama_log_pending[0] = text

    if "\n" in _llama_log_pending.get(0, ""):
        full = _llama_log_pending.pop(0).rstrip()
        if full:
            _llama_log.log(_resolve_ggml_level(_llama_log_pending_level, full), full)


def _resolve_ggml_level(ggml_level: int, text: str) -> int:
    """Translate ggml log level to Python, demoting known-advisory ERRORs to WARNING."""
    py_level = _GGML_TO_PY_LEVEL.get(ggml_level, logging.DEBUG)
    if py_level == logging.ERROR and any(s in text for s in _GGML_ERROR_SOFT_DEMOTE):
        return logging.WARNING
    return py_level


def install_llama_log_handler() -> None:
    """Route llama.cpp logs through Python logging. Idempotent."""
    global _llama_log_callback, _llama_log_installed
    if _llama_log_installed:
        return
    import llama_cpp

    _llama_log_callback = llama_cpp.llama_log_callback(_llama_log_dispatch)
    llama_cpp.llama_log_set(_llama_log_callback, None)
    _llama_log_installed = True


def suppress_native_stderr(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call *fn* with C-level stderr suppressed.
    llama.cpp prints noisy messages (e.g. 'init: embeddings required...')
    that bypass Python logging. This redirects fd 2 to /dev/null for the
    duration of the call. A lock serializes access to fd 2 so concurrent
    threads don't corrupt each other's file descriptors.
    """
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


def embed_one(llm: Any, text: str) -> list[float]:
    """Embed a single text with llama.cpp stderr noise suppressed."""
    response = suppress_native_stderr(llm.create_embedding, input=[text])
    result: list[float] = response["data"][0]["embedding"]
    return result


def read_gguf_metadata(model_path: Path) -> dict[str, str] | None:
    """Read metadata from a GGUF file's headers via llama-cpp-python.
    Returns a dict with keys like 'architecture', 'context_length',
    'embedding_length', 'chat_template', 'file_type', plus the
    KV-cache-shape fields ('block_count', 'head_count_kv', 'key_length',
    'value_length') used to size n_ctx against host memory.
    """
    from llama_cpp import Llama

    install_llama_log_handler()
    llm = suppress_native_stderr(
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
        for arch_key, out_key in (
            (f"{arch}.block_count", "block_count"),
            (f"{arch}.attention.head_count_kv", "head_count_kv"),
            (f"{arch}.attention.head_count", "head_count"),
            (f"{arch}.attention.key_length", "key_length"),
            (f"{arch}.attention.value_length", "value_length"),
        ):
            if arch_key in raw:
                result[out_key] = str(raw[arch_key])
        if "tokenizer.chat_template" in raw:
            result["chat_template"] = str(raw["tokenizer.chat_template"])
        if "general.file_type" in raw:
            result["file_type"] = str(raw["general.file_type"])
        if "general.name" in raw:
            result["name"] = str(raw["general.name"])
        return result or None
    finally:
        llm.close()


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
        f"Install it via the catalog or 'lilbee models install'.",
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

    Chat picks ``n_ctx`` dynamically against host memory (unless
    ``cfg.num_ctx`` is set) and enables flash attention with a fallback for
    older llama-cpp-python builds. Rerank uses ``pooling_type=RANK`` so
    llama.cpp emits cross-encoder scores instead of token embeddings.
    """
    from llama_cpp import Llama

    install_llama_log_handler()
    embedding = mode in (MODE_EMBED, MODE_RERANK)
    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "embedding": embedding,
        "verbose": False,
        "n_gpu_layers": _resolve_n_gpu_layers(embedding=embedding),
    }

    if embedding:
        # Embedding/rerank: clamp n_ctx to the model's training context.
        # Passing a chat-sized cfg.num_ctx through here triggers
        # ``n_ctx_seq > n_ctx_train`` warnings and wastes KV memory.
        embed_meta = _safe_read_gguf_metadata(model_path)
        embed_train_ctx = int((embed_meta or {}).get("context_length", "2048"))
        if cfg.num_ctx is not None:
            kwargs["n_ctx"] = min(cfg.num_ctx, embed_train_ctx)
        else:
            kwargs["n_ctx"] = 0  # 0 -> llama.cpp uses the model's training context
    elif cfg.num_ctx is not None:
        kwargs["n_ctx"] = cfg.num_ctx
    else:
        meta = _safe_read_gguf_metadata(model_path)
        kwargs["n_ctx"] = _resolve_chat_ctx(model_path, meta)
        log.info(
            "Chat n_ctx=%d for %s (dynamic, training_ctx=%s)",
            kwargs["n_ctx"],
            model_path.name,
            (meta or {}).get("context_length", "unknown"),
        )

    if embedding:
        # llama-cpp-python defaults n_batch = min(n_ctx, 512), silently
        # truncating embeddings to 512 tokens. Set n_batch = n_ctx so each
        # text can use the model's full context window.
        ctx_len = embed_train_ctx if kwargs["n_ctx"] == 0 else kwargs["n_ctx"]
        kwargs["n_batch"] = ctx_len
        kwargs["n_ubatch"] = ctx_len

    if mode == MODE_RERANK:
        from llama_cpp import LLAMA_POOLING_TYPE_RANK

        kwargs["pooling_type"] = LLAMA_POOLING_TYPE_RANK

    if not embedding:
        _apply_flash_attention(kwargs)
        _apply_kv_cache_type(kwargs)

    return _construct_llama(Llama, model_path, kwargs)


_MAX_OOM_RETRIES = 2
_CTX_QUANTUM = 256
_CTX_FLOOR = 512


def _safe_read_gguf_metadata(model_path: Path) -> dict[str, str] | None:
    """Best-effort GGUF metadata read, returning None on any failure."""
    try:
        return read_gguf_metadata(model_path)
    except Exception:
        log.debug("read_gguf_metadata failed for %s", model_path, exc_info=True)
        return None


def _resolve_chat_ctx(model_path: Path, meta: dict[str, str] | None) -> int:
    """Pick a chat n_ctx that fits available memory.

    Honors ``LILBEE_NUM_CTX_MAX`` as the upper bound. Falls back to the
    static ``min(training_ctx, DEFAULT_NUM_CTX)`` cap if memory accounting
    goes wrong (e.g., psutil missing).
    """
    training_ctx = DEFAULT_NUM_CTX
    if meta:
        try:
            training_ctx = int(meta.get("context_length", DEFAULT_NUM_CTX))
        except (TypeError, ValueError):
            training_ctx = DEFAULT_NUM_CTX
    ceiling = cfg.num_ctx_max or 16384

    try:
        model_bytes = model_path.stat().st_size
        available = get_available_memory(cfg.gpu_memory_fraction)
        kv_per_tok = kv_bytes_per_token(meta, _kv_elem_bytes_for_cfg())
        return compute_dynamic_ctx(
            model_bytes=model_bytes,
            available_bytes=available,
            training_ctx=training_ctx,
            kv_bytes_per_tok=kv_per_tok,
            ceiling=ceiling,
        )
    except (OSError, ValueError):
        log.debug("dynamic ctx sizing failed for %s, using static cap", model_path, exc_info=True)
        return min(training_ctx, DEFAULT_NUM_CTX)


def _kv_elem_bytes_for_cfg() -> int:
    """Bytes per KV element implied by the configured cache type."""
    return KV_CACHE_TYPE_BYTES.get(cfg.kv_cache_type.value, 2)


_N_GPU_LAYERS_AUTO = -1


def _resolve_n_gpu_layers(*, embedding: bool) -> int:
    """Resolve ``cfg.n_gpu_layers`` to llama-cpp-python's offload integer.

    Embedding loads always use all layers; chat honors ``cfg.n_gpu_layers``
    (None -> all, 0 -> CPU only, positive int -> partial offload).
    """
    if embedding or cfg.n_gpu_layers is None:
        return _N_GPU_LAYERS_AUTO
    return cfg.n_gpu_layers


def _apply_flash_attention(kwargs: dict[str, Any]) -> None:
    """Set ``flash_attn`` per ``cfg.flash_attention`` (None=auto, True/False=force)."""
    if cfg.flash_attention is False:
        return
    # None (auto) and True both pass flash_attn=True; the construct loop
    # drops it on TypeError if llama-cpp-python doesn't support it.
    kwargs["flash_attn"] = True


def _apply_kv_cache_type(kwargs: dict[str, Any]) -> None:
    """Map ``cfg.kv_cache_type`` to llama-cpp-python ``type_k`` / ``type_v``."""
    if cfg.kv_cache_type is KvCacheType.F16:
        return
    type_map = _ggml_type_map()
    if type_map is None:
        log.debug("llama_cpp internal types unavailable; skipping KV quant")
        return
    ggml_type = type_map.get(cfg.kv_cache_type)
    if ggml_type is None:  # pragma: no cover -- defensive against new enum values
        return
    kwargs["type_k"] = ggml_type
    kwargs["type_v"] = ggml_type


def _ggml_type_map() -> dict[KvCacheType, Any] | None:
    """Resolve llama-cpp-python's GGML_TYPE_* constants, or None on older builds."""
    try:
        from llama_cpp import llama_cpp as _llc
    except Exception:  # pragma: no cover -- only fires on llama-cpp-python without _llc
        return None
    return {
        KvCacheType.F32: getattr(_llc, "GGML_TYPE_F32", None),
        KvCacheType.F16: getattr(_llc, "GGML_TYPE_F16", None),
        KvCacheType.Q8_0: getattr(_llc, "GGML_TYPE_Q8_0", None),
        KvCacheType.Q4_0: getattr(_llc, "GGML_TYPE_Q4_0", None),
    }


def _construct_llama(llama_cls: Any, model_path: Path, kwargs: dict[str, Any]) -> Any:
    """Call ``llama_cls(**kwargs)`` with FA fallback and OOM-retry-with-halved-ctx.

    Each loop iteration either returns the loaded model, raises (failure
    or unrelated TypeError), or continues with halved n_ctx; the loop is
    therefore structurally exhaustive and never falls through.
    """
    fa_dropped = False
    for attempt in range(_MAX_OOM_RETRIES + 1):
        try:
            return suppress_native_stderr(llama_cls, **kwargs)
        except TypeError as exc:
            if not _drop_flash_attn_if_unsupported(exc, kwargs, fa_dropped):
                raise
            fa_dropped = True
            continue
        except ValueError as exc:
            if attempt == _MAX_OOM_RETRIES or not _is_load_oom(exc):
                _raise_load_error(model_path, kwargs, exc)
            if not _halve_ctx_for_retry(kwargs, exc):
                _raise_load_error(model_path, kwargs, exc)
    raise RuntimeError("unreachable: _construct_llama loop fell through")  # pragma: no cover


def _drop_flash_attn_if_unsupported(
    exc: TypeError, kwargs: dict[str, Any], already_dropped: bool
) -> bool:
    """If the TypeError is about an unsupported ``flash_attn`` kwarg, drop it."""
    if already_dropped or "flash_attn" not in kwargs or "flash_attn" not in str(exc):
        return False
    log.info("llama-cpp-python rejected flash_attn=True; retrying without it")
    kwargs.pop("flash_attn", None)
    return True


def _halve_ctx_for_retry(kwargs: dict[str, Any], exc: ValueError) -> bool:
    """Halve n_ctx (and matching batch sizes) for an OOM retry. Returns False if no progress."""
    current_ctx = int(kwargs.get("n_ctx", 0) or 0)
    if current_ctx <= 0:
        return False
    new_ctx = max(_CTX_FLOOR, (current_ctx // 2 // _CTX_QUANTUM) * _CTX_QUANTUM)
    if new_ctx >= current_ctx:
        return False
    log.warning(
        "llama.cpp load failed at n_ctx=%d (%s); retrying at n_ctx=%d",
        current_ctx,
        str(exc).splitlines()[0],
        new_ctx,
    )
    kwargs["n_ctx"] = new_ctx
    for key in ("n_batch", "n_ubatch"):
        if key in kwargs:
            kwargs[key] = new_ctx
    return True


def _raise_load_error(model_path: Path, kwargs: dict[str, Any], exc: ValueError) -> None:
    """Raise the wrapped diagnostic for a llama.cpp load failure, or re-raise as-is."""
    wrapped = _wrap_llama_load_error(model_path, kwargs, exc)
    if wrapped is None:
        raise exc
    raise wrapped from exc


def _is_load_oom(exc: ValueError) -> bool:
    """Does this ValueError look like a llama.cpp memory failure?"""
    err = str(exc)
    return "llama_context" in err or "load model from file" in err


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
        "Try a smaller model, lower LILBEE_NUM_CTX, set LILBEE_KV_CACHE_TYPE=q8_0, "
        "or close other processes to free RAM. "
        f"(llama.cpp: {err})"
    )
    return ValueError(" ".join(parts))


def _is_rerank_model(model: str) -> bool:
    """Check if *model* is an exact rerank catalog entry by ref or hf_repo."""
    if not model:
        return False
    return is_rerank_ref(model)


_RERANK_PAIR_SEPARATOR = "</s></s>"


def compute_rerank_scores(llm: Any, query: str, candidates: list[str]) -> list[float]:
    """Score *candidates* against *query* via llama.cpp reranker embeddings.

    ``pooling_type=LLAMA_POOLING_TYPE_RANK`` requires the pair pre-joined
    as ``query</s></s>candidate``; passing them as two inputs makes
    ``llama_decode`` fail with ``-1``.
    """
    scores: list[float] = []
    for candidate in candidates:
        pair = f"{query}{_RERANK_PAIR_SEPARATOR}{candidate}"
        response = suppress_native_stderr(llm.create_embedding, input=pair)
        score = _extract_rerank_score(response)
        scores.append(score)
    return scores


def _extract_rerank_score(response: dict[str, Any]) -> float:
    """Extract a single relevance score from a pooling_type=RANK response.

    Raises ``ProviderError`` with the observed shape for anything other
    than a non-empty ``list[float]`` so upstream format changes surface.
    """
    data = response.get("data") or []
    if not data:
        raise ProviderError("Reranker returned no data", provider="llama-cpp")
    embedding = data[-1].get("embedding")
    if isinstance(embedding, list) and embedding and isinstance(embedding[0], (int, float)):
        return float(embedding[0])
    raise ProviderError(
        "Reranker returned unexpected score shape "
        f"(got {type(embedding).__name__}: {embedding!r}); "
        "llama-cpp-python may have changed its response format",
        provider="llama-cpp",
    )


_HF_BLOBS_DIR_NAME = "blobs"
_HF_SNAPSHOTS_DIR_NAME = "snapshots"


def _find_mmproj_in_hf_snapshots(model_dir: Path) -> Path | None:
    """Walk an HF-cache ``blobs/`` dir up to its sibling ``snapshots/`` tree."""
    if model_dir.name != _HF_BLOBS_DIR_NAME:
        return None
    snapshots_dir = model_dir.parent / _HF_SNAPSHOTS_DIR_NAME
    if not snapshots_dir.is_dir():
        return None
    for snapshot in snapshots_dir.iterdir():
        candidates = sorted(snapshot.glob("*mmproj*.gguf"))
        if candidates:
            return candidates[0]
    return None


def _find_mmproj_in_flat_dir(model_dir: Path) -> Path | None:
    """Glob ``*mmproj*.gguf`` siblings of a model GGUF (sideloaded layout)."""
    candidates = sorted(model_dir.glob("*mmproj*.gguf"))
    return candidates[0] if candidates else None


def find_mmproj_for_model(model_path: Path) -> Path:
    """Find the mmproj (CLIP projection) file for a vision model.

    Resolution order: (1) catalog lookup scoped to ``FEATURED_VISION``,
    (2) HuggingFace-cache ``snapshots/`` sibling of ``blobs/``,
    (3) same-directory glob for flat sideloaded layouts.
    Raises ``ProviderError`` if none find a file.
    """
    from lilbee.catalog import find_mmproj_file

    found = (
        find_mmproj_file(model_path.stem)
        or _find_mmproj_in_hf_snapshots(model_path.parent)
        or _find_mmproj_in_flat_dir(model_path.parent)
    )
    if found is not None:
        return found

    raise ProviderError(
        f"No mmproj (CLIP projection) file found for vision model {model_path.name}. "
        f"Download the mmproj file to {model_path.parent} or re-download the vision "
        "model through the catalog to get both files.",
        provider="llama-cpp",
    )


_CLIP_PROJECTOR_TYPE_KEY = "clip.projector_type"


def read_mmproj_projector_type(mmproj_path: Path) -> str | None:
    """Read ``clip.projector_type`` from a GGUF mmproj without loading the model."""
    try:
        reader = GGUFReader(str(mmproj_path))
        field = reader.get_field(_CLIP_PROJECTOR_TYPE_KEY)
    except Exception:
        log.debug("Failed to read mmproj metadata from %s", mmproj_path, exc_info=True)
        return None
    if field is None or field.types[-1] != GGUFValueType.STRING:
        return None
    return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
