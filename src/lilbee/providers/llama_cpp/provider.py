"""Llama.cpp provider: class, model loader, and path resolution.

Includes a thread-safe batching queue for embeddings so that concurrent
ingest threads don't hit the non-thread-safe Llama object simultaneously.
With ``cfg.worker_pool_enabled = True`` (the default) embed routes through
a persistent worker subprocess so the asyncio loop stays responsive under
load. The legacy ``subprocess_embed`` per-call worker is the fallback when
the pool is disabled. When both flags are off, embeddings run in-process
through the batching thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from lilbee.catalog import is_rerank_ref
from lilbee.core.config import DEFAULT_NUM_CTX, cfg
from lilbee.core.config.enums import KV_CACHE_TYPE_BYTES, KvCacheType
from lilbee.core.services import get_services
from lilbee.providers.base import ClosableIterator, LLMProvider, ProviderError, filter_options
from lilbee.providers.llama_cpp.abort_signal import abort_callback, clear_abort
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
    import_llama_cpp,
    install_llama_log_handler,
    suppress_native_stderr,
)
from lilbee.providers.model_cache import (
    MODE_CHAT,
    MODE_EMBED,
    MODE_RERANK,
    LoaderMode,
    MemoryAwareModelCache,
    compute_dynamic_ctx,
    get_available_memory,
    kv_bytes_per_token,
)
from lilbee.providers.worker import WorkerManager
from lilbee.providers.worker.embed_worker import embed_worker_main
from lilbee.providers.worker.pool import PoolRuntime, RoleAccessor, WorkerPool
from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import WorkerError

log = logging.getLogger(__name__)

# Cap on tokens consumed during ``_LockedStreamIterator.close()``'s drain. A
# runaway model (e.g. Qwen3-0.6B stuck in a never-closing ``<think>`` loop)
# would otherwise block ``close()`` indefinitely.
_LOCKED_STREAM_DRAIN_CAP = 1024

# Chat-load OOM retry knobs. The OOM wrapper halves ``n_ctx`` (rounded down to
# the next ``_CTX_QUANTUM`` multiple) up to ``_MAX_OOM_RETRIES`` times before
# raising. ``_CTX_FLOOR`` is the smallest ``n_ctx`` we'll attempt.
_MAX_OOM_RETRIES = 2
_CTX_QUANTUM = 256
_CTX_FLOOR = 512

# Sentinel passed to ``llama-cpp-python`` for "offload all layers".
_N_GPU_LAYERS_AUTO = -1


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
        self._pool: WorkerPool | None = None
        self._pool_runtime: PoolRuntime | None = None
        self._pool_embed_accessor: RoleAccessor | None = None
        self._pool_lock = threading.Lock()

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
        # Each new dispatch starts with a fresh abort flag: a previous
        # request_abort() unblocks the prior in-flight call but must not
        # latch and break this one.
        clear_abort()
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
        # See ``_dispatch_batch`` for why we clear at the start of each dispatch.
        clear_abort()
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

    def _get_pool_embed_accessor(self) -> RoleAccessor:
        """Lazy-create the pool, register the embed role, return its accessor."""
        with self._pool_lock:
            if self._pool_embed_accessor is not None:
                return self._pool_embed_accessor
            self._pool = WorkerPool()
            self._pool_runtime = PoolRuntime()
            self._pool_runtime.start()
            self._pool_embed_accessor = self._pool.register(
                "embed",
                embed_worker_main,
                _make_embed_role_config_factory(),
            )
            return self._pool_embed_accessor

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts. Routes through the persistent worker pool by default.

        Order of preference:

        1. ``cfg.worker_pool_enabled`` (default True): per-role pool worker.
        2. ``cfg.subprocess_embed``: legacy per-call WorkerManager.
        3. In-process via the batching thread.

        Pool failures fall back to the in-process path so a worker crash
        does not break ingest in-flight; the next call retries the pool.
        """
        if cfg.worker_pool_enabled:
            try:
                return self._embed_via_pool(texts)
            except (OSError, RuntimeError, WorkerError) as exc:
                log.warning("Pool embed failed, falling back to in-process: %s", exc)
        if self._subprocess_enabled:
            try:
                return self._get_subprocess_worker().embed(texts)
            except (OSError, RuntimeError) as exc:
                log.warning("Subprocess embed failed, falling back to in-process: %s", exc)
                self._subprocess_enabled = False
        fut: Future[list[list[float]]] = Future()
        self._embed_queue.put(EmbedRequest(texts=texts, future=fut))
        return fut.result(timeout=EMBED_FUTURE_TIMEOUT_S)

    def _embed_via_pool(self, texts: list[str]) -> list[list[float]]:
        """Run one embed batch through the persistent pool worker."""
        accessor = self._get_pool_embed_accessor()
        runtime = self._pool_runtime
        assert runtime is not None  # noqa: S101  set by _get_pool_embed_accessor
        result = runtime.run_sync(
            accessor.call("embed", texts, timeout=cfg.worker_pool_call_timeout_s),
            timeout=cfg.worker_pool_call_timeout_s,
        )
        if not isinstance(result, list):
            raise WorkerError(
                "ProtocolError",
                f"Pool embed returned {type(result).__name__}, expected list[list[float]].",
                "",
            )
        return result

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

    def vision_ocr(
        self, png_bytes: bytes, model: str, prompt: str = "", *, timeout: float | None = None
    ) -> str:
        """Run vision OCR via the subprocess worker, honouring an optional per-call timeout."""
        return self._get_subprocess_worker().vision_ocr(png_bytes, model, prompt, timeout=timeout)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | ClosableIterator[str]:
        """Chat completion: serialized via lock (Llama is not thread-safe)."""
        self._chat_lock.acquire()
        # Clear AFTER the lock acquires so a concurrent chat can't clobber a
        # mid-stream cancel still being honored by the prior holder.
        clear_abort()
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
        self._shutdown_pool()
        self._cache.unload_all()

    def _shutdown_pool(self) -> None:
        """Drain pool workers and stop the runtime loop. Idempotent."""
        with self._pool_lock:
            pool = self._pool
            runtime = self._pool_runtime
            self._pool = None
            self._pool_runtime = None
            self._pool_embed_accessor = None
        if pool is not None and runtime is not None:
            try:
                runtime.run_sync(pool.shutdown(), timeout=10.0)
            except (TimeoutError, RuntimeError, OSError) as exc:
                log.warning("Pool shutdown raised %s; forcing runtime stop", exc)
            runtime.shutdown(timeout=5.0)

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        """Evict cached models so the next call reloads with current settings.

        Also tears down the pool's per-role workers; the next call will
        respawn them with the new ``cfg.embedding_model``. In-place model
        swap inside a worker is a follow-up; for now the lazy respawn
        on the next call is the simpler correctness story.
        """
        if model_path is None:
            self._cache.unload_all()
        else:
            self._cache.unload_path(model_path)
        self._shutdown_pool()


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


def _make_embed_role_config_factory() -> Callable[[], RoleConfig]:
    """Return a factory that resolves cfg.embedding_model at spawn time.

    The pool calls the factory on every spawn (lazy or restart) so model
    swaps in cfg propagate without an explicit invalidation call.
    """
    from lilbee.providers.model_cache import MODE_EMBED

    def _make() -> RoleConfig:
        return RoleConfig(
            role="embed",
            model_path=resolve_model_path(cfg.embedding_model),
            mode=MODE_EMBED,
        )

    return _make


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
    # supports_rerank() can be called before any model load (feature detection
    # for status / catalog UIs), so route through import_llama_cpp() first to
    # surface the libvulkan hint rather than a raw OSError on bare Linux. A
    # genuinely-missing llama_cpp package surfaces as ImportError and means
    # "no rerank support"; a libvulkan-flavored OSError is a real install
    # error and must propagate as ProviderError to the caller.
    try:
        import_llama_cpp()
        from llama_cpp import LLAMA_POOLING_TYPE_RANK  # noqa: F401
    except ImportError:
        return False
    return True


def load_llama(model_path: Path, *, mode: LoaderMode) -> Any:
    """Load a llama_cpp.Llama in chat, embed, or rerank mode."""
    Llama = import_llama_cpp().Llama  # noqa: N806

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


def _safe_read_gguf_metadata(model_path: Path) -> dict[str, str] | None:
    """Best-effort GGUF metadata read, returning None on any failure."""
    try:
        return read_gguf_metadata(model_path)
    except Exception:
        log.debug("read_gguf_metadata failed for %s", model_path, exc_info=True)
        return None


def _resolve_chat_ctx(model_path: Path, meta: dict[str, str] | None) -> int:
    """Pick the largest 256-multiple n_ctx that fits in available memory."""
    training_ctx = DEFAULT_NUM_CTX
    if meta:
        try:
            training_ctx = int(meta.get("context_length", DEFAULT_NUM_CTX))
        except (TypeError, ValueError):
            training_ctx = DEFAULT_NUM_CTX
    ceiling = cfg.num_ctx_max

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
    return KV_CACHE_TYPE_BYTES[cfg.kv_cache_type]


def _resolve_n_gpu_layers(*, embedding: bool) -> int:
    """Resolve ``cfg.n_gpu_layers`` (None=all) to llama-cpp's offload integer."""
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
    # Fresh abort flag per load: a prior request_abort() that interrupted
    # an inference must not latch and abort the next model swap.
    clear_abort()
    kwargs.setdefault("abort_callback", abort_callback)
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
