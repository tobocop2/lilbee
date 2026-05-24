"""Llama.cpp provider: class, model loader, and path resolution."""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import re
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, cast, overload

from lilbee.app.services import get_services
from lilbee.catalog import is_rerank_ref
from lilbee.core.config import DEFAULT_NUM_CTX, cfg
from lilbee.core.config.enums import KV_CACHE_TYPE_BYTES, KvCacheType
from lilbee.providers.base import (
    ClosableIterator,
    ContextWindowExceededError,
    LLMProvider,
    ProviderError,
    filter_options,
)
from lilbee.providers.families import detect as detect_profile
from lilbee.providers.llama_cpp.abort_signal import abort_callback, clear_abort
from lilbee.providers.llama_cpp.gguf_meta import (
    find_mmproj_for_model,
    read_gguf_metadata,
    train_ctx_from_meta,
)
from lilbee.providers.llama_cpp.log_dispatch import (
    import_llama_cpp,
    install_llama_log_handler,
    suppress_native_stderr,
)
from lilbee.providers.model_cache import (
    LoaderMode,
    compute_dynamic_ctx,
    get_available_memory,
    kv_bytes_per_token,
)
from lilbee.providers.worker.chat_worker import chat_worker_main
from lilbee.providers.worker.embed_worker import embed_worker_main
from lilbee.providers.worker.pool import PoolRuntime, RoleAccessor
from lilbee.providers.worker.rerank_worker import rerank_worker_main
from lilbee.providers.worker.transport import (
    ChatRequest,
    ChatResult,
    ChatStreamItem,
    OcrBackend,
    PdfOcrRequest,
    RerankPayload,
    RoleConfig,
    VisionRequest,
    WorkerRole,
)
from lilbee.providers.worker.transport_pipe import WorkerCrashError, WorkerError
from lilbee.providers.worker.vision_worker import vision_worker_main
from lilbee.providers.worker.wire_kinds import WireKind
from lilbee.runtime.progress import EventType, ExtractEvent
from lilbee.vision import PageText, PdfOcrChunk, pdf_page_count

log = logging.getLogger(__name__)

# Vision OCR sentinel used when no per-call timeout and no ``cfg.ocr_timeout``
# is set. 24h is effectively "no cap" for the round-trip wait loop.
_VISION_NO_CAP_TIMEOUT_S = 86_400.0

_LLAMA_CONTEXT_PATCH_LOCK = threading.Lock()
"""Serialises overlapping ``_llama_n_seq_max`` callers inside one process.

The shim mutates ``llama_cpp.internals.LlamaContext.__init__`` globally
while the with-block is open. Worker subprocesses each load one model
serially today, but the lock keeps the contract safe if a future caller
loads two models concurrently.
"""


@contextlib.contextmanager
def _llama_n_seq_max(n_seq_max: int) -> Any:
    """Set ``context_params.n_seq_max`` on the next ``LlamaContext`` constructed.

    Workaround for llama-cpp-python upstream issue #2051 (``n_seq_max``
    not exposed as a Llama kwarg). See ``docs/architecture.md`` for the
    full rationale and the upstream-fix removal hint.
    """
    from llama_cpp import internals

    with _LLAMA_CONTEXT_PATCH_LOCK:
        original = internals.LlamaContext.__init__

        def patched(self: Any, *, model: Any, params: Any, verbose: bool) -> None:
            params.n_seq_max = n_seq_max
            original(self, model=model, params=params, verbose=verbose)

        internals.LlamaContext.__init__ = patched  # type: ignore[method-assign,assignment]
        try:
            yield
        finally:
            internals.LlamaContext.__init__ = original  # type: ignore[method-assign]


# Cap on tokens drained during ``_PoolChatStreamIterator.close()`` after a
# mid-stream cancel. A runaway model (Qwen3-0.6B stuck in a never-closing
# ``<think>`` loop) would otherwise block close() indefinitely.
_CHAT_STREAM_DRAIN_CAP = 1024

# Chat-load OOM retry knobs. The OOM wrapper halves ``n_ctx`` (rounded down to
# the next ``_CTX_QUANTUM`` multiple) up to ``_MAX_OOM_RETRIES`` times before
# raising. ``_CTX_FLOOR`` is the smallest ``n_ctx`` we'll attempt.
_MAX_OOM_RETRIES = 2
_CTX_QUANTUM = 256
_CTX_FLOOR = 512

# Minimum usable post-load n_ctx for a chat model. Some GGUF quants ship with
# ``<arch>.context_length`` missing or zero; llama-cpp silently falls back to
# its own 512 default, which is too small for any real chat request. We refuse
# to register such a model rather than failing opaquely on the first turn.
_MIN_CHAT_CTX = 2048

# Sentinel passed to ``llama-cpp-python`` for "offload all layers".
_N_GPU_LAYERS_AUTO = -1

# Jinja chat templates flag tool support by referencing one of these
# names inside an expression / statement block. Anchoring the match to
# the Jinja delimiters rules out incidental hits like "this template has
# no tools" prose; the names must appear as identifiers inside ``{% ... %}``
# or ``{{ ... }}``.
_TOOL_TEMPLATE_PATTERN = re.compile(r"\{[%{][^}]*\b(?:tools|tool_calls|functions|function_calls)\b")


class LlamaCppProvider(LLMProvider):
    """Provider backed by llama-cpp-python for local GGUF model inference."""

    def __init__(self) -> None:
        self._pool_lock = threading.Lock()
        self._registered_roles: set[WorkerRole] = set()

    @staticmethod
    def _raise_chat_worker_error(exc: WorkerError) -> NoReturn:
        """Translate a worker-side chat error into the right parent-side exception."""
        if exc.original_type == ContextWindowExceededError.__name__:
            requested, usable, n_ctx = _parse_context_overflow_breakdown(exc.message)
            raise ContextWindowExceededError(
                exc.message,
                requested=requested,
                usable_budget=usable,
                n_ctx=n_ctx,
            ) from exc
        raise ProviderError(
            LlamaCppProvider._worker_error_message("Chat", exc),
            provider="llama-cpp",
        ) from exc

    @staticmethod
    def _worker_error_message(role_label: str, exc: WorkerError) -> str:
        """Render a user-facing message that names the role and points at the log.

        ``WorkerCrashError`` already embeds the log path in its message; for
        plain ``WorkerError`` (the worker reported an exception or returned
        a malformed reply) the surfaced text is the worker's exception
        repr so the user sees enough to file a bug report.
        """
        detail = str(exc)
        if detail.endswith("."):  # the wrapper supplies its own sentence-final period
            detail = detail[:-1]
        if isinstance(exc, WorkerCrashError):
            return f"{role_label} worker exited unexpectedly. {detail}. Please try again."
        return f"{role_label} worker reported an error: {detail}. Please try again."

    def _pool_runtime(self) -> PoolRuntime:
        """Return the Services-owned :class:`PoolRuntime`, starting it lazily."""
        runtime = get_services().pool_runtime
        runtime.start()
        return runtime

    def _get_pool_accessor(
        self,
        role: WorkerRole,
        worker_main: Any,
        config_factory: Callable[[], RoleConfig],
    ) -> RoleAccessor:
        """Register *role* on the Services pool the first time it is used.

        Subsequent calls return the same accessor without touching the
        pool state. Registration is gated by ``self._pool_lock`` so two
        concurrent first-callers do not race to register the role twice.
        """
        pool = get_services().worker_pool
        with self._pool_lock:
            if role not in self._registered_roles:
                pool.register(role, worker_main, config_factory)
                self._registered_roles.add(role)
        return pool.accessor(role)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via the persistent pool worker.

        Worker crashes and timeouts surface as :class:`ProviderError`;
        the pool respawns the embed role lazily on the next call.
        """
        accessor = self._get_pool_accessor(
            WorkerRole.EMBED, embed_worker_main, _make_role_config_factory(WorkerRole.EMBED)
        )
        runtime = self._pool_runtime()
        try:
            result = runtime.run_sync(
                accessor.call(WireKind.EMBED, texts, timeout=cfg.worker_pool_call_timeout_s),
                timeout=cfg.worker_pool_call_timeout_s,
            )
            if not isinstance(result, list):
                raise WorkerError(
                    "ProtocolError",
                    f"Pool embed returned {type(result).__name__}, expected list[list[float]].",
                    "",
                )
        except WorkerError as exc:
            raise ProviderError(
                self._worker_error_message("Embedding", exc),
                provider="llama-cpp",
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(
                "Embedding worker timed out. Please try again.",
                provider="llama-cpp",
            ) from exc
        return result

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Score *candidates* by relevance to *query* via the pool worker."""
        if not candidates:
            return []
        accessor = self._get_pool_accessor(
            WorkerRole.RERANK, rerank_worker_main, _make_role_config_factory(WorkerRole.RERANK)
        )
        runtime = self._pool_runtime()
        try:
            result = runtime.run_sync(
                accessor.call(
                    WireKind.RERANK,
                    RerankPayload(query=query, candidates=candidates),
                    timeout=cfg.worker_pool_call_timeout_s,
                ),
                timeout=cfg.worker_pool_call_timeout_s,
            )
            if not isinstance(result, list):
                raise WorkerError(
                    "ProtocolError",
                    f"Pool rerank returned {type(result).__name__}, expected list[float].",
                    "",
                )
        except WorkerError as exc:
            raise ProviderError(
                self._worker_error_message("Rerank", exc),
                provider="llama-cpp",
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(
                "Rerank worker timed out. Please try again.",
                provider="llama-cpp",
            ) from exc
        return result

    def supports_rerank(self) -> bool:
        """llama-cpp can rerank iff llama-cpp-python exposes the rank pooling type."""
        return _llama_cpp_has_rank_pooling()

    def vision_ocr(
        self, png_bytes: bytes, model: str, prompt: str = "", *, timeout: float | None = None
    ) -> str:
        """Run vision OCR via the persistent pool worker."""
        accessor = self._get_pool_accessor(
            WorkerRole.VISION, vision_worker_main, _make_role_config_factory(WorkerRole.VISION)
        )
        runtime = self._pool_runtime()
        budget = self._vision_call_budget(timeout)
        request = VisionRequest(png_bytes=png_bytes, prompt=prompt, model=model or None)
        try:
            result = runtime.run_sync(
                accessor.call(WireKind.VISION, request, timeout=budget),
                timeout=budget,
            )
            if not isinstance(result, str):
                raise WorkerError(
                    "ProtocolError",
                    f"Pool vision_ocr returned {type(result).__name__}, expected str.",
                    "",
                )
        except WorkerError as exc:
            raise ProviderError(
                self._worker_error_message("Vision", exc),
                provider="llama-cpp",
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(
                "Vision worker timed out. Please try again.",
                provider="llama-cpp",
            ) from exc
        return result

    @staticmethod
    def _vision_call_budget(timeout: float | None) -> float:
        """Wall-clock budget for one vision_ocr call (per-call > cfg.ocr_timeout > no cap)."""
        effective = timeout if timeout is not None else cfg.ocr_timeout
        return float(effective) if effective and effective > 0 else _VISION_NO_CAP_TIMEOUT_S

    def pdf_ocr(
        self,
        path: Path,
        *,
        backend: OcrBackend,
        model: str = "",
        per_page_timeout_s: float | None = None,
        quiet: bool = True,
        on_progress: Callable[..., None] | None = None,
    ) -> list[PageText]:
        """Run multi-page vision PDF OCR via the persistent vision worker.

        ``per_page_timeout_s`` is *per page*. The total wall-clock cap on
        the streamed drain is ``pages * per_page + cfg.vision_load_budget_s``
        (load grace), so a 100-page scan with a 60 s per-page budget gets
        ~6000 s + load, not 60 s for the whole document.
        """
        accessor = self._get_pool_accessor(
            WorkerRole.VISION, vision_worker_main, _make_role_config_factory(WorkerRole.VISION)
        )
        runtime = self._pool_runtime()
        budget = self._pdf_drain_budget(path, per_page_timeout_s)
        del quiet  # accepted for Protocol parity; worker has no Rich progress to suppress.
        request = PdfOcrRequest(
            path=str(path),
            backend=backend,
            model=model,
        )
        progress = on_progress

        async def _drain() -> list[PageText]:
            pages: list[PageText] = []
            stream = cast(AsyncIterator[Any], accessor.stream(WireKind.PDF_OCR, request))
            async for frame in stream:
                if not isinstance(frame, PdfOcrChunk):
                    raise ProviderError(
                        f"PDF OCR worker streamed unexpected frame type {type(frame).__name__}.",
                        provider="llama-cpp",
                    )
                pages.append(PageText(frame.page, frame.text))
                if progress is not None:
                    progress(
                        EventType.EXTRACT,
                        ExtractEvent(file=path.name, page=frame.page, total_pages=frame.total),
                    )
            return pages

        try:
            return runtime.run_sync(_drain(), timeout=budget)
        except WorkerError as exc:
            raise ProviderError(
                self._worker_error_message("PDF OCR", exc),
                provider="llama-cpp",
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(
                "PDF OCR worker timed out. Please try again.",
                provider="llama-cpp",
            ) from exc

    def _pdf_drain_budget(self, path: Path, per_page_timeout_s: float | None) -> float:
        """Total drain timeout = page_count * per_page + vision_load_budget_s."""
        if not per_page_timeout_s or per_page_timeout_s <= 0:
            return _VISION_NO_CAP_TIMEOUT_S
        try:
            pages = pdf_page_count(path)
        except Exception:
            # If we can't probe pages upfront the worker will still try,
            # but we lose the precise budget; fall back to no-cap so the
            # parent doesn't kill a valid run on a probe failure.
            return _VISION_NO_CAP_TIMEOUT_S
        return float(pages) * per_page_timeout_s + cfg.vision_load_budget_s

    @overload
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: Literal[False] = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult: ...

    @overload
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: Literal[True],
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ClosableIterator[ChatStreamItem]: ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult | ClosableIterator[ChatStreamItem]:
        """Chat completion via the persistent pool worker.

        Streaming returns a :class:`ClosableIterator` whose ``close()``
        flips the worker's abort flag so in-flight generation drains
        cleanly. Non-streaming returns a :class:`ChatResult` carrying the
        assistant text and any tool-call frames the model emitted.
        """
        accessor = self._get_pool_accessor(
            WorkerRole.CHAT, chat_worker_main, _make_role_config_factory(WorkerRole.CHAT)
        )
        runtime = self._pool_runtime()
        accessor.clear_abort()  # honor mid-stream cancels from the previous turn
        request = ChatRequest(
            messages=messages,
            stream=stream,
            options=self._chat_kwargs_from_options(options) or None,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        if stream:
            return _PoolChatStreamIterator(
                runtime=runtime,
                accessor=accessor,
                async_iter=accessor.stream(WireKind.CHAT, request),
            )
        try:
            result = runtime.run_sync(
                accessor.call(WireKind.CHAT, request, timeout=cfg.worker_pool_call_timeout_s),
                timeout=cfg.worker_pool_call_timeout_s,
            )
            if not isinstance(result, ChatResult):
                raise WorkerError(
                    "ProtocolError",
                    f"Pool chat returned {type(result).__name__}, expected ChatResult.",
                    "",
                )
        except WorkerError as exc:
            self._raise_chat_worker_error(exc)
        except TimeoutError as exc:
            raise ProviderError(
                "Chat worker timed out. Please try again.",
                provider="llama-cpp",
            ) from exc
        return result

    def supports_tools(self, model_ref: str) -> bool:
        """True iff *model_ref*'s GGUF chat template references tool tokens.

        Cached on ``(path, mtime)`` so a tool-bearing chat doesn't read the
        GGUF header from disk on every request. A re-quantised file at the
        same path invalidates automatically because its mtime changes.
        """
        try:
            path = resolve_model_path(model_ref)
        except (ProviderError, OSError):
            log.debug("supports_tools: resolve_model_path failed for %s", model_ref, exc_info=True)
            return False
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        return _supports_tools_cached(str(path), mtime_ns)

    @staticmethod
    def _chat_kwargs_from_options(options: dict[str, Any] | None) -> dict[str, Any]:
        """Translate user-facing options into llama-cpp create_chat_completion kwargs."""
        if not options:
            return {}
        filtered = filter_options(options)
        if "num_predict" in filtered:
            filtered["max_tokens"] = filtered.pop("num_predict")
        filtered.pop("num_ctx", None)  # model-load param, not per-call
        return filtered

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

    def warm_up_pool(self) -> None:
        """Register roles for every configured model. Idempotent.

        Called by ``Services`` when ``cfg.worker_pool_eager_start`` is on so
        ``WorkerPool.start_eager()`` has roles to spawn. Roles whose model is
        unset are skipped; this lets a setup with only ``chat_model`` +
        ``embedding_model`` configured eager-start exactly those two and not
        pay rerank or vision spawn cost.
        """
        for role, _spec in _ROLE_SPECS.items():
            if not _is_role_configured(role):
                continue
            entrypoint = _ROLE_ENTRYPOINTS[role]
            self._get_pool_accessor(role, entrypoint, _make_role_config_factory(role))

    def shutdown(self) -> None:
        """Drop pool registrations so a follow-up provider can re-register cleanly."""
        self._release_pool_roles()

    def _release_pool_roles(self) -> None:
        """Drop our registrations on the Services pool so the next call respawns.

        Safe even when Services has not yet been built (early shutdown
        on import-time failure). Holds ``self._pool_lock`` so a concurrent
        ``_get_pool_accessor`` does not race the role removal.
        """
        with self._pool_lock:
            roles = tuple(self._registered_roles)
            self._registered_roles.clear()
        if not roles:
            return
        from lilbee.providers.worker.pool import PoolShutdownError

        services = get_services()
        runtime = services.pool_runtime
        for role in roles:
            try:
                runtime.run_sync(services.worker_pool.release(role), timeout=10.0)
            except PoolShutdownError:
                # Pool already shut down (atexit ordering during a CLI exit
                # tears down the pool runtime before this provider). Nothing
                # to release; silent no-op.
                pass
            except (TimeoutError, RuntimeError, OSError) as exc:
                log.warning("Pool release of role=%s raised %s", role, exc)

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        """Drop the pool's per-role workers so the next call respawns with current settings.

        The ``model_path`` argument is accepted for protocol parity with
        other providers but does not narrow the scope: workers reload all
        their roles on respawn anyway.
        """
        del model_path
        self._release_pool_roles()


class _PoolChatStreamIterator:
    """Sync facade over an async chat-stream iterator from the worker pool.

    Each ``__next__`` submits one ``__anext__`` to the pool's runtime
    loop and blocks for the result. ``close()`` flips the worker's abort
    flag so any in-flight generation stops at the next token-tick;
    in-flight chunks already in the pipe still drain.
    """

    def __init__(
        self,
        *,
        runtime: PoolRuntime,
        accessor: RoleAccessor,
        async_iter: Any,
    ) -> None:
        self._runtime = runtime
        self._accessor = accessor
        self._async_iter = async_iter
        self._exhausted = False

    def __iter__(self) -> _PoolChatStreamIterator:
        return self

    def __aiter__(self) -> _PoolChatStreamIterator:
        return self

    async def __anext__(self) -> ChatStreamItem:
        if self._exhausted:
            raise StopAsyncIteration
        try:
            chunk: ChatStreamItem = await self._async_iter.__anext__()
            return chunk
        except StopAsyncIteration:
            self._exhausted = True
            raise
        except WorkerError as exc:
            self._exhausted = True
            LlamaCppProvider._raise_chat_worker_error(exc)

    def __next__(self) -> ChatStreamItem:
        if self._exhausted:
            raise StopIteration
        try:
            chunk: ChatStreamItem = self._runtime.run_sync(
                self._async_iter.__anext__(),
                timeout=cfg.worker_pool_call_timeout_s,
            )
            return chunk
        except StopAsyncIteration:
            self._exhausted = True
            raise StopIteration from None
        except WorkerError as exc:
            # Mid-stream worker errors translate the same way the non-stream
            # path does: context overflow surfaces as ContextWindowExceededError,
            # everything else as a generic ProviderError.
            self._exhausted = True
            LlamaCppProvider._raise_chat_worker_error(exc)
        except TimeoutError as exc:
            self._exhausted = True
            raise ProviderError(
                "Chat worker timed out mid-stream. Please try again.",
                provider="llama-cpp",
            ) from exc

    def close(self) -> None:
        """Cancel mid-stream and drain remaining tokens from the pipe.

        Drain is bounded by ``_CHAT_STREAM_DRAIN_CAP`` so a stuck
        worker cannot block close() indefinitely; once the cap fires we
        accept the partial-state for not hanging the UI.
        """
        if self._exhausted:
            return
        self._accessor.cancel()
        drained = 0
        while drained < _CHAT_STREAM_DRAIN_CAP:
            try:
                next(self)
            except StopIteration:
                break
            except Exception:
                break
            drained += 1
        self._accessor.clear_abort()
        self._exhausted = True

    def __del__(self) -> None:  # pragma: no cover
        with contextlib.suppress(Exception):
            self.close()


@dataclass(frozen=True)
class _RoleSpec:
    """Per-role recipe for building a :class:`RoleConfig` from cfg."""

    cfg_attr: str
    mode: str


_ROLE_SPECS: dict[WorkerRole, _RoleSpec] = {
    WorkerRole.EMBED: _RoleSpec(cfg_attr="embedding_model", mode=LoaderMode.EMBED),
    WorkerRole.RERANK: _RoleSpec(cfg_attr="reranker_model", mode=LoaderMode.RERANK),
    WorkerRole.CHAT: _RoleSpec(cfg_attr="chat_model", mode=LoaderMode.CHAT),
    # Vision uses a custom mtmd loader (not load_llama); the mode hint is
    # documentation only, the vision worker calls load_vision_llama directly.
    WorkerRole.VISION: _RoleSpec(cfg_attr="vision_model", mode="vision"),
}


_ROLE_ENTRYPOINTS: dict[WorkerRole, Callable[..., None]] = {
    WorkerRole.EMBED: embed_worker_main,
    WorkerRole.RERANK: rerank_worker_main,
    WorkerRole.CHAT: chat_worker_main,
    WorkerRole.VISION: vision_worker_main,
}


def _is_role_configured(role: WorkerRole) -> bool:
    """True iff the cfg attribute for *role* holds a non-empty model name."""
    return bool(getattr(cfg, _ROLE_SPECS[role].cfg_attr))


def _make_role_config_factory(role: WorkerRole) -> Callable[[], RoleConfig]:
    """Return a factory that resolves the role's configured model at spawn time.

    The pool calls the factory on every spawn (lazy or restart) so model
    swaps in cfg propagate without an explicit invalidation call.
    """
    spec = _ROLE_SPECS[role]

    def _make() -> RoleConfig:
        model_name = getattr(cfg, spec.cfg_attr)
        if not model_name:
            raise ProviderError(
                f"No {role} model configured. Set cfg.{spec.cfg_attr} first.",
                provider="llama-cpp",
            )
        return RoleConfig(
            role=role,
            model_path=resolve_model_path(model_name),
            mode=spec.mode,
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


def load_llama(
    model_path: Path,
    *,
    mode: LoaderMode,
    abort_callback_override: Any = None,
) -> Any:
    """Load a llama_cpp.Llama in chat, embed, or rerank mode.

    ``abort_callback_override`` lets pool workers bind a callback that
    reads the worker's shared ``mp.Value`` abort flag.
    """
    Llama = import_llama_cpp().Llama  # noqa: N806

    install_llama_log_handler()
    embedding = mode in (LoaderMode.EMBED, LoaderMode.RERANK)
    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "embedding": embedding,
        "verbose": False,
        "n_gpu_layers": _resolve_n_gpu_layers(embedding=embedding),
    }
    if cfg.main_gpu is not None:
        kwargs["main_gpu"] = cfg.main_gpu

    # Read GGUF metadata at most once and share it across all the kwargs
    # builders that need it (n_ctx resolution + chat_format override).
    # Each call constructs a Llama(vocab_only=True), so caching here keeps
    # the load path to one metadata pass instead of two.
    meta: dict[str, str] | None = None

    if embedding:
        # Embedding/rerank uses the model's training context unconditionally.
        # cfg.num_ctx is a chat-tuned setting; propagating it here used to
        # clamp the rerank model below what a query+candidate pair needs and
        # produced "llama_decode returned 1" on every other query when the
        # user picked a small chat ctx for a low-RAM box. The explicit
        # ``embed_train_ctx`` value (instead of ``0`` for "use model
        # default") keeps the OOM-retry path working: ``_halve_ctx_for_retry``
        # cannot bisect from 0.
        meta = read_gguf_metadata(model_path)
        embed_train_ctx = train_ctx_from_meta(
            meta, fallback=_EMBED_FALLBACK_CTX, model_path=model_path
        )
        kwargs["n_ctx"] = embed_train_ctx
    elif cfg.num_ctx is not None:
        kwargs["n_ctx"] = cfg.num_ctx
    else:
        meta = read_gguf_metadata(model_path)
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
        kwargs["n_batch"] = kwargs["n_ctx"]
        kwargs["n_ubatch"] = kwargs["n_ctx"]

    if mode == LoaderMode.RERANK:
        from llama_cpp import LLAMA_POOLING_TYPE_RANK

        kwargs["pooling_type"] = LLAMA_POOLING_TYPE_RANK

    if not embedding:
        _apply_flash_attention(kwargs)
        _apply_kv_cache_type(kwargs)
        chat_meta = meta if meta is not None else read_gguf_metadata(model_path)
        _apply_chat_format_override(kwargs, model_path, chat_meta)

    if abort_callback_override is not None:
        kwargs["abort_callback"] = abort_callback_override

    if embedding:
        from lilbee.providers.llama_cpp.batching import EMBED_N_SEQ_MAX

        with _llama_n_seq_max(EMBED_N_SEQ_MAX):
            llm = _construct_llama(Llama, model_path, kwargs)
    else:
        llm = _construct_llama(Llama, model_path, kwargs)
    if mode == LoaderMode.CHAT:
        _validate_chat_context_window(llm, model_path)
    return llm


def _validate_chat_context_window(llm: Any, model_path: Path) -> None:
    """Refuse a chat model whose post-load ``n_ctx`` is below the chat minimum.

    Triggered by GGUFs that report ``context_length=0`` (broken quant metadata):
    llama-cpp silently falls back to its 512-token default, which can't fit any
    realistic prompt and surfaces as an opaque 500 on the first request.
    """
    actual = int(llm.n_ctx())
    if actual >= _MIN_CHAT_CTX:
        return
    with contextlib.suppress(Exception):
        llm.close()
    raise ProviderError(
        f"Chat model {model_path.name!r} loaded with n_ctx={actual}, which is below "
        f"the {_MIN_CHAT_CTX}-token minimum for chat. This usually means the GGUF's "
        "metadata is broken (missing or zero context_length). Try a different quant "
        "of the model, or set 'num_ctx' (LILBEE_NUM_CTX) in lilbee config to "
        "override.",
        provider="llama-cpp",
    )


@functools.lru_cache(maxsize=32)
def _supports_tools_cached(path_str: str, _mtime_ns: int) -> bool:
    """Memoised tool-template probe keyed on the GGUF's path + mtime.

    The mtime arg participates in the cache key only; a re-quantised file at
    the same path invalidates automatically because its mtime changes.
    """
    from lilbee.providers.worker.response_parser.schemas import get_schemas

    meta = read_gguf_metadata(Path(path_str))
    if not isinstance(meta, dict):
        return False
    # A matching family profile is itself evidence the model is tool-capable:
    # the family ships a response-parsing schema lilbee uses to extract calls,
    # so even families whose embedded chat template doesn't trip the generic
    # Jinja probe (e.g. ERNIE's <tool_call> markers, LFM2's <|tool_list_start|>)
    # still register as tool-capable when the profile is registered.
    profile = detect_profile(meta, ref=path_str)
    if profile is not None and (
        profile.chat_format_override is not None or profile.family in get_schemas()
    ):
        return True
    template = meta.get("chat_template")
    if not isinstance(template, str):
        return False
    return _TOOL_TEMPLATE_PATTERN.search(template) is not None


# Capture requested / usable / n_ctx out of an overflow-error message body.
_CTX_BREAKDOWN_RE = re.compile(
    r"Prompt of (\d+) tokens exceeds the usable budget of (\d+) tokens "
    r"\(n_ctx=(\d+)"
)
_CTX_RUNTIME_RE = re.compile(r"Prompt of (\d+) tokens exceeded the (\d+)-token context window")


def _parse_context_overflow_breakdown(message: str) -> tuple[int, int, int]:
    """Recover ``(requested, usable_budget, n_ctx)`` from an overflow message."""
    match = _CTX_BREAKDOWN_RE.search(message)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    match = _CTX_RUNTIME_RE.search(message)
    if match:
        n_ctx = int(match.group(2))
        return int(match.group(1)), n_ctx, n_ctx
    return 0, 0, 0


# Fallback used when an embedding GGUF reports zero, negative, or
# unparseable ``context_length`` in its metadata header. Some published
# nomic-embed and Qwen3 GGUFs in the wild report ``0`` (the b473 QA dump
# logged ``n_ctx_seq (512) > n_ctx_train (0)``). 2048 is the documented
# training-context for the smallest featured embedder that uses it
# (Google's EmbeddingGemma-300m, see
# https://huggingface.co/google/embeddinggemma-300m), and llama.cpp
# tolerates n_ctx > n_ctx_train with a warning, so the larger nomic
# embedder still loads cleanly under the same fallback.
_EMBED_FALLBACK_CTX = 2048


def _resolve_chat_ctx(model_path: Path, meta: dict[str, str] | None) -> int:
    """Pick n_ctx aiming for ``cfg.chat_n_ctx_target``, clamped to model + host.

    When ``cfg.num_ctx_max`` is ``None`` the model's training_ctx is the only
    ceiling, so a long-context model can grow past the target if the host
    has the RAM to back it. Setting ``num_ctx_max`` explicitly caps below
    training_ctx for per-host policy reasons.
    """
    training_ctx = train_ctx_from_meta(meta, fallback=DEFAULT_NUM_CTX, model_path=model_path)
    # Some GGUFs under-declare context_length vs the model's real trained window
    # (Command-R7B is a 128K model whose GGUF says 8192). A family profile can
    # correct that; the dynamic picker still bounds the result by the chat target
    # and available VRAM, so this lifts the ceiling without forcing a giant KV.
    profile = detect_profile(meta, ref=str(model_path))
    if profile is not None and profile.context_length_override is not None:
        training_ctx = profile.context_length_override
    ceiling = cfg.num_ctx_max if cfg.num_ctx_max is not None else training_ctx

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
            target=cfg.chat_n_ctx_target,
        )
    except (OSError, ValueError):
        log.debug("dynamic ctx sizing failed for %s, using static cap", model_path, exc_info=True)
        return min(training_ctx, cfg.chat_n_ctx_target)


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


def _tool_args_as_json_strings(messages: Any) -> Any:
    """Return *messages* with assistant tool-call arguments as JSON strings.

    lilbee normalises ``tool_calls[].function.arguments`` to a dict for GGUF
    templates that iterate them, but HF chat templates follow the OpenAI wire
    shape and concatenate ``arguments`` as a string (functionary's template does
    ``'>>>' + name + '\\n' + arguments``). Re-serialise at render time only;
    the windowed messages the GGUF path uses are left untouched.
    """
    rendered = []
    for msg in messages:
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if not tool_calls:
            rendered.append(msg)
            continue
        fixed_calls = []
        for call in tool_calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict):
                fixed_calls.append(call)
                continue
            args = fn.get("arguments")
            if isinstance(args, dict):
                fixed_calls.append({**call, "function": {**fn, "arguments": json.dumps(args)}})
            else:
                fixed_calls.append(call)
        rendered.append({**msg, "tool_calls": fixed_calls})
    return rendered


def _apply_hf_template_chat_handler(kwargs: dict[str, Any], hf_tokenizer_repo: str) -> bool:
    """Render with the HF tokenizer's own jinja chat template, not the GGUF's.

    Some GGUFs ship a stripped template that drops tool definitions (functionary)
    or no template at all (Command-R), so the model never sees the tools. The
    upstream HF repo carries a tool-aware template; load it and hand llama-cpp a
    ``Jinja2ChatFormatter`` built from it. The formatter only renders the prompt;
    lilbee's own response parser extracts the tool calls from the generated text,
    so this stays independent of llama-cpp's per-preset tool parsing. Returns
    True when the handler was installed.
    """
    try:
        from llama_cpp.llama_chat_format import ChatFormatterResponse, Jinja2ChatFormatter
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_repo)
        if not tokenizer.chat_template:
            return False

        class _HFTemplateChatFormatter(Jinja2ChatFormatter):
            """Render with the tokenizer's own ``apply_chat_template``.

            ``Jinja2ChatFormatter`` compiles a single template string in a sandbox
            that lacks the ``loopcontrols`` extension and can't pick between a
            tokenizer's named templates (Command-R ships a ``{default, tool_use, rag}``
            dict and uses ``{% break %}``). ``apply_chat_template`` handles both, so
            delegate rendering to it and inherit the base class's generation handler.
            """

            def __init__(self) -> None:
                self._tokenizer = tokenizer
                self.eos_token = tokenizer.eos_token or ""
                self.bos_token = tokenizer.bos_token or ""
                self.add_generation_prompt = True
                self.stop_token_ids = None

            def __call__(
                self, *, messages: Any, tools: Any = None, **_: Any
            ) -> ChatFormatterResponse:
                # tokenize=False makes apply_chat_template return the rendered str.
                prompt = cast(
                    str,
                    self._tokenizer.apply_chat_template(
                        _tool_args_as_json_strings(messages),
                        tools=tools,
                        add_generation_prompt=True,
                        tokenize=False,
                    ),
                )
                return ChatFormatterResponse(prompt=prompt, stop=[self.eos_token])

        kwargs["chat_handler"] = _HFTemplateChatFormatter().to_chat_handler()
    except Exception:
        log.warning(
            "Failed to build HF-template chat handler from %s; tool calls may fail",
            hf_tokenizer_repo,
            exc_info=True,
        )
        return False
    log.info("Rendering with HF chat template from %s", hf_tokenizer_repo)
    return True


def _apply_chat_format_override(
    kwargs: dict[str, Any], model_path: Path, meta: dict[str, str] | None
) -> None:
    """Swap the GGUF's embedded chat template for the family profile's preset."""
    profile = detect_profile(meta, ref=str(model_path))
    if profile is None:
        return
    if (
        profile.render_with_hf_template
        and profile.hf_tokenizer_repo is not None
        and _apply_hf_template_chat_handler(kwargs, profile.hf_tokenizer_repo)
    ):
        return
    if profile.chat_format_override is None:
        return
    # llama-cpp-python accepts the StrEnum value (str subclass) directly.
    kwargs["chat_format"] = profile.chat_format_override.value
    log.info(
        "Chat format override for %s: %s (family=%s)",
        model_path.name,
        profile.chat_format_override.value,
        profile.family.value,
    )

    if profile.hf_tokenizer_repo is not None:
        try:
            from llama_cpp.llama_tokenizer import LlamaHFTokenizer

            kwargs["tokenizer"] = LlamaHFTokenizer.from_pretrained(profile.hf_tokenizer_repo)
            log.info(
                "Loaded HF tokenizer %s for %s",
                profile.hf_tokenizer_repo,
                model_path.name,
            )
        except Exception:
            log.warning(
                "Failed to load HF tokenizer %s for chat_format=%s; tool calls may fail",
                profile.hf_tokenizer_repo,
                profile.chat_format_override.value,
                exc_info=True,
            )


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


_UNSUPPORTED_ARCH_PATTERNS = (
    "unknown model architecture",
    "unknown architecture",
)

# llama-cpp's "unknown architecture" messages embed the offending name in
# one of two shapes: ``architecture: 'name'`` or ``architecture 'name'``.
# The named group is optional; if extraction fails we still wrap the error,
# just without the architecture label.
_ARCH_NAME_RE = re.compile(
    r"(?:unknown\s+(?:model\s+)?architecture)\s*[:\s]+'?([A-Za-z0-9_.\-]+)'?",
    re.IGNORECASE,
)


def _wrap_unsupported_architecture(model_path: Path, exc: ValueError) -> ValueError | None:
    """Wrap a llama-cpp ``unknown architecture`` ValueError with a usable hint."""
    err = str(exc)
    lower = err.lower()
    if not any(p in lower for p in _UNSUPPORTED_ARCH_PATTERNS):
        return None
    match = _ARCH_NAME_RE.search(err)
    arch_clause = f" architecture {match.group(1)!r}" if match else ""
    # When the architecture name can't be extracted we still don't leak the
    # raw upstream string; it lives in the chained exception's __cause__ for
    # debugging, and falls into the unfiltered log lines for ops.
    log.warning("llama-cpp rejected load of %s: %s", model_path.name, err)
    return ValueError(
        f"Model {model_path.name!r} uses{arch_clause} which lilbee's native "
        f"runtime doesn't support yet. Pick a different model from the catalog, "
        f"or set LILBEE_REMOTE_BASE_URL (e.g. to a running Ollama) and select "
        f"the model there."
    )


def _wrap_llama_load_error(
    model_path: Path, kwargs: dict[str, Any], exc: ValueError
) -> ValueError | None:
    """Diagnostic ValueError for opaque llama.cpp load failures, or None to pass through."""
    arch_wrap = _wrap_unsupported_architecture(model_path, exc)
    if arch_wrap is not None:
        return arch_wrap
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
