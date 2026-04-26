"""Framework-agnostic route handlers for the lilbee HTTP server.

Every public function is a plain async callable — no framework imports.
Return types are dicts (JSON responses), lists, or async generators of SSE strings.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import json
import logging
import mimetypes
import threading
import time
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from lilbee import settings
from lilbee.cli.helpers import clean_result, copy_files, gather_status, get_version
from lilbee.config import Config, cfg
from lilbee.config_meta import (
    MODEL_ROLE_FIELDS as _MODEL_ROLE_FIELDS,
)
from lilbee.config_meta import (
    PUBLIC_CONFIG_FIELDS as _PUBLIC_CONFIG_FIELDS,
)
from lilbee.config_meta import (
    REINDEX_FIELDS,
    WRITABLE_CONFIG_FIELDS,
)
from lilbee.model_manager import ModelSource, get_model_manager
from lilbee.models import ModelTask
from lilbee.progress import DetailedProgressCallback, EventType, ProgressEvent, SseEvent
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.sdk_backend import API_KEY_FIELDS
from lilbee.providers.sdk_llm_provider import inject_provider_keys
from lilbee.results import DocumentResult, group
from lilbee.security import validate_path_within
from lilbee.server.models import (
    AddSummary,
    AskResponse,
    CatalogEntryResponse,
    CleanedChunk,
    ConfigResponse,
    ConfigUpdateResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentRemoveResponse,
    ExternalModelsResponse,
    HealthResponse,
    InstalledModelEntry,
    ModelsCatalogResponse,
    ModelsDeleteResponse,
    ModelsInstalledResponse,
    ModelsShowResponse,
    SetModelResponse,
    SourceContentResponse,
    StatusResponse,
    SyncSummary,
)
from lilbee.services import get_services

if TYPE_CHECKING:
    from lilbee.catalog import CatalogModel
    from lilbee.ingest import SyncResult
    from lilbee.query import ChatMessage

log = logging.getLogger(__name__)

# Windows mimetypes reads from the registry, which may not define ``.md``
# as ``text/markdown``. Pin the mapping at import time; ``add_type`` is
# idempotent so repeated imports are safe.
mimetypes.add_type("text/markdown", ".md")

MAX_ADD_FILES = 100


class ModelCatalogEntry(BaseModel):
    """A single model in the catalog."""

    name: str
    size_gb: float
    min_ram_gb: float
    description: str
    installed: bool


class ModelCatalogSection(BaseModel):
    """A single-role catalog section with active model and installed list."""

    active: str
    catalog: list[ModelCatalogEntry]
    installed: list[str]


class ModelsResponse(BaseModel):
    """Response for GET /api/models: one catalog section per role."""

    chat: ModelCatalogSection
    embedding: ModelCatalogSection
    vision: ModelCatalogSection
    reranker: ModelCatalogSection


# ``ModelTask.RERANK.value`` is ``"rerank"`` but the route is ``/api/models/reranker``,
# so this mapping is needed to build correct redirect URLs in 422 responses.
TASK_ENDPOINT_PATH: dict[ModelTask, str] = {
    ModelTask.CHAT: "chat",
    ModelTask.EMBEDDING: "embedding",
    ModelTask.VISION: "vision",
    ModelTask.RERANK: "reranker",
}


def format_task_mismatch(ref: str, entry_task: ModelTask, expected_task: ModelTask) -> str:
    """Build the 422 body when a role slot is assigned a model of the wrong task."""
    endpoint = TASK_ENDPOINT_PATH[entry_task]
    return (
        f"Model '{ref}' is a {entry_task} model, not {expected_task}. "
        f"Set it via PUT /api/models/{endpoint} instead."
    )


def sse_event(event: str, data: Any) -> str:
    """Format a single Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_error(message: str) -> str:
    """Format an SSE error event."""
    return sse_event(SseEvent.ERROR, {"message": message})


def sse_done(data: dict[str, Any]) -> str:
    """Format an SSE done event."""
    return sse_event(SseEvent.DONE, data)


def _resolve_generation_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert raw options dict to GenerationOptions, or None."""
    return cfg.generation_options(**options) if options else None


class SseStream:
    """Context object for SSE streaming with cancellation support.
    Bundles the queue, cancel event, and progress callback that every SSE
    endpoint needs.  Call :meth:`drain` to yield events until the task
    completes or the client disconnects.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.cancel = threading.Event()
        self.loop = asyncio.get_running_loop()
        self.callback: DetailedProgressCallback = self._build_callback()

    def _build_callback(self) -> DetailedProgressCallback:
        """Create a progress callback that serializes events into the queue.
        Safe to call from both the event-loop thread and worker threads.
        """
        loop = self.loop
        queue = self.queue

        def _callback(event_type: EventType, data: ProgressEvent) -> None:
            serialized = data.model_dump() if isinstance(data, BaseModel) else data
            payload = f"event: {event_type}\ndata: {json.dumps(serialized)}\n\n"
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                queue.put_nowait(payload)
            else:
                loop.call_soon_threadsafe(queue.put_nowait, payload)

        return _callback

    async def drain(
        self, task: asyncio.Task[Any] | asyncio.Future[Any], label: str
    ) -> AsyncGenerator[str, None]:
        """Yield SSE strings until a sentinel arrives; cancel *task* on client disconnect.

        Emits a ``heartbeat`` event whenever the producer queue stays
        idle longer than ``cfg.sse_heartbeat_interval`` seconds so
        clients that enforce a stream-idle timeout don't abort.
        """
        last_yielded = time.monotonic()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                except TimeoutError:
                    now = time.monotonic()
                    heartbeat_interval = cfg.sse_heartbeat_interval
                    if heartbeat_interval > 0 and now - last_yielded >= heartbeat_interval:
                        last_yielded = now
                        yield sse_event(SseEvent.HEARTBEAT, {"ts": time.time()})
                    # Fallback for producers that die without a sentinel.
                    if task.done() and self.queue.empty():
                        break
                    continue
                if item is None:
                    break
                last_yielded = time.monotonic()
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            log.info("%s cancelled by client", label)
            self.cancel.set()
            task.cancel()


async def health() -> HealthResponse:
    """Return service health and version."""
    return HealthResponse(status="ok", version=get_version())


async def status() -> StatusResponse:
    """Return config, sources, and chunk counts."""
    raw = gather_status()
    return StatusResponse(**raw.model_dump(exclude_none=True))


async def search(q: str, top_k: int = 5, chunk_type: str | None = None) -> list[DocumentResult]:
    """Search and return grouped DocumentResults."""
    if not q or not q.strip():
        raise ValueError("query must not be empty")
    results = get_services().searcher.search(q, top_k=top_k, chunk_type=chunk_type)
    results = [r for r in results if r.distance is None or r.distance <= cfg.max_distance]
    return group(results)


async def ask(
    question: str,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AskResponse:
    """One-shot RAG answer. Returns answer and sources."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    opts = _resolve_generation_options(options)
    result = get_services().searcher.ask_raw(
        question, top_k=top_k, options=opts, chunk_type=chunk_type
    )
    return AskResponse(
        answer=result.answer,
        sources=[CleanedChunk(**clean_result(s)) for s in result.sources],
    )


def _run_llm_stream(
    messages: list[ChatMessage],
    opts: dict[str, Any] | None,
    queue: asyncio.Queue[str | None],
    cancel: threading.Event,
    error_holder: list[str],
) -> None:
    """Stream LLM tokens into a queue from a worker thread."""
    from lilbee.reasoning import filter_reasoning

    try:
        provider = get_services().provider
        stream = provider.chat(
            cast("list[dict[str, Any]]", messages),
            stream=True,
            options=opts or None,
            model=cfg.chat_model,
        )
        for st in filter_reasoning(cast(Iterator[str], stream), show=cfg.show_reasoning):
            if cancel.is_set():
                break
            if st.content:
                event_type = SseEvent.REASONING if st.is_reasoning else SseEvent.TOKEN
                queue.put_nowait(sse_event(event_type, {"token": st.content}))
    except Exception as exc:
        error_holder.append(str(exc))
    finally:
        queue.put_nowait(None)


async def _stream_rag_response(
    question: str,
    history: list[ChatMessage] | None = None,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AsyncGenerator[str, None]:
    """Shared SSE streaming for ask_stream and chat_stream."""
    yield ""  # force generator

    rag = get_services().searcher.build_rag_context(
        question, top_k=top_k, history=history, chunk_type=chunk_type
    )
    if rag is None:
        yield sse_error("No relevant documents found.")
        return

    results, messages = rag
    opts = _resolve_generation_options(options) or cfg.generation_options()

    sse = SseStream()
    error_holder: list[str] = []

    executor_fut = sse.loop.run_in_executor(
        None, _run_llm_stream, messages, opts, sse.queue, sse.cancel, error_holder
    )
    task = asyncio.ensure_future(executor_fut)
    async for event in sse.drain(task, "RAG stream"):
        yield event

    if error_holder:
        log.warning("Stream error: %s", error_holder[0])
        yield sse_error("Internal error")
        sse.cancel.set()
        return

    # Ensure executor thread has finished before yielding final events
    await executor_fut

    yield sse_event(SseEvent.SOURCES, [clean_result(s) for s in results])
    yield sse_done({})


def ask_stream(
    question: str,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events: token, sources, done."""
    return _stream_rag_response(question, top_k=top_k, options=options, chunk_type=chunk_type)


async def chat(
    question: str,
    history: list[ChatMessage],
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AskResponse:
    """Chat with history. Returns answer and sources."""
    opts = _resolve_generation_options(options)
    result = get_services().searcher.ask_raw(
        question, top_k=top_k, history=history, options=opts, chunk_type=chunk_type
    )
    return AskResponse(
        answer=result.answer,
        sources=[CleanedChunk(**clean_result(s)) for s in result.sources],
    )


def chat_stream(
    question: str,
    history: list[ChatMessage],
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events with chat history support."""
    return _stream_rag_response(
        question, history=history, top_k=top_k, options=options, chunk_type=chunk_type
    )


async def _run_sync_with_sentinel(sse: SseStream, enable_ocr: bool | None) -> SyncResult:
    """Run ingest.sync() and guarantee the drain sentinel is enqueued."""
    from lilbee.cli.helpers import temporary_ocr_config
    from lilbee.ingest import sync

    try:
        with temporary_ocr_config(enable_ocr):
            return await sync(quiet=True, on_progress=sse.callback, cancel=sse.cancel)
    finally:
        sse.queue.put_nowait(None)


# Registry lock serializes lock creation and the check-and-acquire step to
# avoid a TOCTOU between locked() and acquire() under concurrent /api/add.
_INGEST_LOCKS: dict[str, asyncio.Lock] = {}
_INGEST_LOCK_REGISTRY: asyncio.Lock | None = None


# Types that can carry script even within an "inline-rendered" category.
# Keep the deny narrow and explicit. Broadening this set is a security-relevant
# change — file an issue with the ``security`` label before adding entries.
_RAW_INLINE_RENDER_DENY: frozenset[str] = frozenset(
    {
        "text/html",
        "text/javascript",
        "application/javascript",
        "application/xhtml+xml",
        "text/css",
        "image/svg+xml",
    }
)


def _is_safe_for_inline_render(content_type: str) -> bool:
    """Whether ``raw=1`` may serve this Content-Type as-is.

    Trusted categories (``text/*``, ``image/*``, ``application/pdf``) pass
    through, with named exceptions for types that embed executable script.
    Everything else degrades to ``application/octet-stream`` so an attacker-
    renamed file (e.g. ``evil.html``) cannot trick a browser into rendering
    it inline within the plugin origin.
    """
    if content_type in _RAW_INLINE_RENDER_DENY:
        return False
    if content_type == "application/pdf":
        return True
    return content_type.startswith("text/") or content_type.startswith("image/")


def _get_registry_lock() -> asyncio.Lock:
    """Return the registry lock, creating it on the running loop if needed."""
    global _INGEST_LOCK_REGISTRY
    if _INGEST_LOCK_REGISTRY is None:
        _INGEST_LOCK_REGISTRY = asyncio.Lock()
    return _INGEST_LOCK_REGISTRY


def _reset_ingest_locks() -> None:
    """Test hook: clear per-source locks and the registry lock."""
    global _INGEST_LOCK_REGISTRY
    _INGEST_LOCKS.clear()
    _INGEST_LOCK_REGISTRY = None


async def _try_acquire_source(name: str) -> asyncio.Lock | None:
    """Acquire the lock for ``name`` or return ``None`` if already held."""
    async with _get_registry_lock():
        lock = _INGEST_LOCKS.get(name)
        if lock is None:
            lock = asyncio.Lock()
            _INGEST_LOCKS[name] = lock
        if lock.locked():
            return None
        await lock.acquire()
        return lock


def _canonical_source_name(p_str: str) -> str:
    """Match the basename ``copy_files`` writes under ``cfg.documents_dir``."""
    return Path(p_str).name


async def sync_stream(*, enable_ocr: bool | None = None) -> AsyncGenerator[str, None]:
    """Trigger sync, yield SSE progress events, then done event."""
    sse = SseStream()
    task = asyncio.create_task(_run_sync_with_sentinel(sse, enable_ocr))
    async for event in sse.drain(task, "Sync stream"):
        yield event
    if not sse.cancel.is_set() and task.done() and not task.cancelled():
        exc = task.exception()
        if exc is not None:
            yield sse_error(str(exc))
            return
        yield sse_done(task.result().model_dump())


async def _run_add(
    paths: list[str],
    force: bool,
    enable_ocr: bool | None,
    ocr_timeout: float | None,
    sse: SseStream,
) -> AddSummary:
    """Copy files and sync, returning the summary for the final done event."""
    from lilbee.cli.helpers import temporary_ocr_config
    from lilbee.ingest import sync

    try:
        errors: list[str] = []
        valid: list[Path] = []
        for p_str in paths:
            p = Path(p_str)
            if not p.exists():
                errors.append(p_str)
            else:
                valid.append(p)

        copy_result = copy_files(valid, force=force)

        if sse.cancel.is_set():
            return AddSummary(copied=copy_result.copied, skipped=copy_result.skipped, errors=errors)

        with temporary_ocr_config(enable_ocr, ocr_timeout):
            sync_result = await sync(quiet=True, on_progress=sse.callback, cancel=sse.cancel)

        return AddSummary(
            copied=copy_result.copied,
            skipped=copy_result.skipped,
            errors=errors,
            sync=SyncSummary(**sync_result.model_dump()),
        )
    finally:
        sse.queue.put_nowait(None)


async def _acquire_add_locks(
    paths: list[str],
) -> tuple[list[tuple[str, asyncio.Lock]], list[str]]:
    """Return ``(acquired, busy)`` partitioning of ``paths`` by lock state."""
    acquired: list[tuple[str, asyncio.Lock]] = []
    busy: list[str] = []
    seen: set[str] = set()
    for p_str in paths:
        name = _canonical_source_name(p_str)
        if name in seen:
            continue
        seen.add(name)
        lock = await _try_acquire_source(name)
        if lock is None:
            busy.append(name)
        else:
            acquired.append((name, lock))
    return acquired, busy


def _release_add_locks(acquired: list[tuple[str, asyncio.Lock]]) -> None:
    """Release every lock in ``acquired``. Safe to call multiple times."""
    while acquired:
        _, lock = acquired.pop()
        if lock.locked():
            lock.release()


def validate_add_paths(
    data: dict[str, Any],
) -> tuple[list[str], bool, bool | None, float | None]:
    """Validate add-files input. Raises ValueError on bad input."""
    paths = data.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("'paths' must be a non-empty list of strings")
    if len(paths) > MAX_ADD_FILES:
        raise ValueError(f"Too many files: {len(paths)} exceeds limit of {MAX_ADD_FILES}")

    for p_str in paths:
        validate_path_within(cfg.documents_dir / Path(p_str).name, cfg.documents_dir)

    force = bool(data.get("force", False))
    enable_ocr, ocr_timeout = _parse_ocr_params(data)
    return paths, force, enable_ocr, ocr_timeout


def _parse_ocr_params(data: dict[str, Any]) -> tuple[bool | None, float | None]:
    """Extract and coerce OCR parameters from a request dict."""
    enable_ocr = data.get("enable_ocr")
    ocr_timeout = data.get("ocr_timeout")
    if enable_ocr is not None:
        enable_ocr = bool(enable_ocr)
    if ocr_timeout is not None:
        ocr_timeout = float(ocr_timeout)
    return enable_ocr, ocr_timeout


async def add_files_stream(data: dict[str, Any]) -> AsyncGenerator[str, None]:
    """Copy files, sync, and yield SSE progress events.

    Contended sources emit ``already_ingesting`` and the stream closes
    without a ``done`` event, signalling the client to wait rather than retry.
    """
    paths = data.get("paths", [])
    force = bool(data.get("force", False))
    enable_ocr, ocr_timeout = _parse_ocr_params(data)

    acquired, busy = await _acquire_add_locks(paths)
    try:
        for name in busy:
            log.info("Rejecting /api/add for %s: already ingesting", name)
            yield sse_event(SseEvent.ALREADY_INGESTING, {"source": name})

        if not acquired:
            return

        sse = SseStream()
        task = asyncio.create_task(_run_add(paths, force, enable_ocr, ocr_timeout, sse))
        try:
            async for event in sse.drain(task, "Add files stream"):
                yield event
            if not sse.cancel.is_set() and task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    yield sse_error(str(exc))
                    return
                summary = task.result()
                yield sse_done(summary.model_dump())
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
    finally:
        _release_add_locks(acquired)


def _catalog_section(
    featured: tuple[CatalogModel, ...],
    active: str,
    installed: set[str],
) -> ModelCatalogSection:
    """Build a ModelCatalogSection from a featured-catalog tuple."""
    return ModelCatalogSection(
        active=active,
        catalog=[
            ModelCatalogEntry(
                name=m.display_name,
                size_gb=m.size_gb,
                min_ram_gb=m.min_ram_gb,
                description=m.description,
                installed=m.ref in installed,
            )
            for m in featured
        ],
        installed=sorted(installed),
    )


async def list_models() -> ModelsResponse:
    """Return per-role catalogs (chat, embedding, vision, reranker) with active selections.

    Uses the unfiltered installed set so a single ref lights up in every
    catalog section it legitimately matches.
    """
    from lilbee.catalog import (
        FEATURED_CHAT,
        FEATURED_EMBEDDING,
        FEATURED_RERANK,
        FEATURED_VISION,
    )

    installed = set(get_model_manager().list_installed())

    return ModelsResponse(
        chat=_catalog_section(FEATURED_CHAT, cfg.chat_model, installed),
        embedding=_catalog_section(FEATURED_EMBEDDING, cfg.embedding_model, installed),
        vision=_catalog_section(FEATURED_VISION, cfg.vision_model, installed),
        reranker=_catalog_section(FEATURED_RERANK, cfg.reranker_model, installed),
    )


async def _set_model(
    field: Literal["chat_model", "embedding_model", "vision_model", "reranker_model"],
    model: str,
) -> SetModelResponse:
    """Shared helper for switching a model field."""
    setattr(cfg, field, model)
    settings.set_value(cfg.data_root, field, model)
    return SetModelResponse(model=model)


def _require_model_available(model: str) -> str:
    """Return the normalized installed model ref; raises ValueError when unavailable.

    Accepts catalog ``name:tag``, HuggingFace repo id, display name, or
    provider-prefixed ref.
    """
    from lilbee.catalog import find_catalog_entry
    from lilbee.models import ensure_tag

    entry = find_catalog_entry(model)
    normalized = entry.ref if entry is not None else ensure_tag(model)
    available = get_services().provider.list_models()
    # ``available`` lists bare tags from /api/tags; stored refs may carry an
    # ``ollama/`` prefix. Match on either form so both client styles work.
    bare = parse_model_ref(normalized).name
    if normalized in available or bare in available:
        return normalized
    # Providers may report the HuggingFace repo form instead of the catalog
    # ``name:tag``. When the input resolved to a catalog entry, accept that
    # entry's ``hf_repo`` (with or without ``:latest``) as an equivalent
    # installed form so the canonical ref is still returned.
    if entry is not None:
        hf_candidates = {entry.hf_repo, ensure_tag(entry.hf_repo)}
        if hf_candidates.intersection(available):
            return normalized
    raise ValueError(f"Model '{normalized}' is not available. Pull it first or check the name.")


def _build_task_to_field() -> dict[ModelTask, str]:
    """Invert config's ``_MODEL_FIELD_TO_TASK`` so the two maps stay in sync."""
    from lilbee.config import _MODEL_FIELD_TO_TASK

    return {ModelTask(task): field for field, task in _MODEL_FIELD_TO_TASK.items()}


_TASK_TO_FIELD: dict[ModelTask, str] = _build_task_to_field()


def _require_model_for_task(model: str, expected: ModelTask, *, allow_empty: bool = False) -> str:
    """Validate *model* is installed locally AND passes the catalog task check.

    Empty string unsets the role when *allow_empty* is True. Catalog +
    task validation delegates to ``validate_model_task_assignment`` so
    the handler and config paths share a single implementation.
    """
    from lilbee.config import validate_model_task_assignment

    if allow_empty and not model.strip():
        return ""
    normalized = _require_model_available(model)
    return validate_model_task_assignment(_TASK_TO_FIELD[expected], normalized, allow_bypass=False)


async def set_chat_model(model: str) -> SetModelResponse:
    """Switch active chat model. Validates installation and catalog task."""
    normalized = _require_model_for_task(model, ModelTask.CHAT)
    return await _set_model("chat_model", normalized)


async def set_embedding_model(model: str) -> SetModelResponse:
    """Switch embedding model. Validates installation and catalog task."""
    normalized = _require_model_for_task(model, ModelTask.EMBEDDING)
    return await _set_model("embedding_model", normalized)


async def set_vision_model(model: str) -> SetModelResponse:
    """Switch vision OCR model. Empty string unsets it (vision OCR disabled)."""
    normalized = _require_model_for_task(model, ModelTask.VISION, allow_empty=True)
    return await _set_model("vision_model", normalized)


async def set_reranker_model(model: str) -> SetModelResponse:
    """Switch reranker model. Empty string unsets it (reranking disabled)."""
    normalized = _require_model_for_task(model, ModelTask.RERANK, allow_empty=True)
    return await _set_model("reranker_model", normalized)


_MIN_CHUNK_SIZE = 64


def _validate_config_updates(updates: dict[str, Any]) -> None:
    """Reject unknown fields, null values on non-nullable fields, and invalid ranges."""
    for key, value in updates.items():
        if key not in WRITABLE_CONFIG_FIELDS:
            raise ValueError(f"Unknown or read-only config field: {key}")
        if value is None and not WRITABLE_CONFIG_FIELDS[key]:
            raise ValueError(f"Field '{key}' does not accept null")
    chunk_val = updates.get("chunk_size")
    if isinstance(chunk_val, int) and chunk_val < _MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be >= {_MIN_CHUNK_SIZE}")


def _apply_config_updates(updates: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Apply updates to the in-memory config, rolling back on error.
    Returns (fields_to_persist, fields_to_delete) for disk write.
    """
    snapshot = {k: getattr(cfg, k) for k in updates}
    to_persist: dict[str, str] = {}
    to_delete: list[str] = []
    try:
        for key, value in updates.items():
            if value is None:
                setattr(cfg, key, None)
                to_delete.append(key)
            else:
                setattr(cfg, key, value)
                to_persist[key] = str(getattr(cfg, key))
    except Exception:
        for k, v in snapshot.items():
            setattr(cfg, k, v)
        raise
    return to_persist, to_delete


async def update_config(updates: dict[str, Any]) -> ConfigUpdateResponse:
    """Partial update of writable config fields.
    Algorithm: validate-then-apply with rollback.

    1. Validate all keys and null-acceptability upfront (no mutations yet).
       This catches typos and bad input before anything changes.
    2. Snapshot current values, then apply each update via setattr (pydantic's
       validate_assignment catches type errors). If any field fails type
       validation, roll back ALL fields from the snapshot so the config
       stays consistent — no half-applied updates.
    3. Persist to disk in batch (one file write for sets, one for deletes)
       rather than per-field, avoiding partial writes on crash.

    Why not just setattr-and-save per field? A multi-field PATCH like
    {"chunk_size": 1024, "chunk_overlap": "bad"} would leave chunk_size
    changed but chunk_overlap unchanged — the caller gets an error but
    the config is silently modified. The snapshot/rollback prevents that.
    """
    _validate_config_updates(updates)
    to_persist, to_delete = _apply_config_updates(updates)
    if to_persist:
        settings.update_values(cfg.data_root, to_persist)
    if to_delete:
        settings.delete_values(cfg.data_root, to_delete)
    if API_KEY_FIELDS & set(updates):
        inject_provider_keys()
    reindex_required = bool(REINDEX_FIELDS & set(updates))
    return ConfigUpdateResponse(updated=list(updates), reindex_required=reindex_required)


async def delete_documents(
    names: list[str], *, delete_files: bool = False
) -> DocumentRemoveResponse:
    """Remove documents from the knowledge base by source name."""
    result = get_services().store.remove_documents(names, delete_files=delete_files)
    return DocumentRemoveResponse(removed=result.removed, not_found=result.not_found)


async def list_documents(
    search: str = "",
    limit: int = 50,
    offset: int = 0,
) -> DocumentListResponse:
    """Return indexed documents with metadata, paginated and filterable.

    Pagination and the filename filter are pushed into LanceDB via
    ``Store.get_sources(search=..., limit=..., offset=...)`` and the
    total comes from ``Store.count_sources(search=...)`` so neither
    call materializes the full SOURCES table per request.
    """
    store = get_services().store
    search_term = search or None
    page = store.get_sources(search=search_term, limit=limit, offset=offset)
    total = store.count_sources(search=search_term)
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                filename=s["filename"],
                chunk_count=s.get("chunk_count", 0),
                ingested_at=s.get("ingested_at", ""),
            )
            for s in page
        ],
        total=total,
        limit=limit,
        offset=offset,
        has_more=len(page) > 0 and (offset + len(page)) < total,
    )


async def get_config() -> ConfigResponse:
    """Return all user-facing configuration values."""
    dumped = cfg.model_dump()
    result = {k: v for k, v in dumped.items() if k in _PUBLIC_CONFIG_FIELDS}
    return ConfigResponse(**result)


async def get_source_content(
    source: str, raw: bool = False
) -> SourceContentResponse | tuple[bytes, str]:
    """Return a stored source file: JSON with markdown text for text types, or
    ``(bytes, content_type)`` when *raw* is True. Binary types return empty
    markdown so clients know to re-request with ``raw=1``.
    """
    from lilbee.wiki.index import parse_title

    if not source or not source.strip():
        raise ValueError("source must not be empty")
    documents_dir = cfg.documents_dir
    resolved = validate_path_within(documents_dir / source, documents_dir)
    if not resolved.is_file():
        raise FileNotFoundError(source)

    content_type, _ = mimetypes.guess_type(resolved.name)
    if content_type is None:
        content_type = "application/octet-stream"

    if raw:
        # Cap raw responses to inline-render-safe categories; anything else
        # degrades to a binary download so attacker-renamed files (e.g.
        # evil.html) can't trick the embedding browser into running script
        # under our origin.
        served_type = (
            content_type if _is_safe_for_inline_render(content_type) else "application/octet-stream"
        )
        return resolved.read_bytes(), served_type

    if not content_type.startswith("text/"):
        return SourceContentResponse(markdown="", content_type=content_type, title=None)

    text = resolved.read_text(encoding="utf-8", errors="replace")
    title = parse_title(text) or None
    return SourceContentResponse(markdown=text, content_type=content_type, title=title)


@functools.cache
def _compute_config_defaults() -> dict[str, Any]:
    """Materialize Config defaults once per process."""
    defaults: dict[str, Any] = {}
    for name, info in Config.model_fields.items():
        is_writable_public = name in WRITABLE_CONFIG_FIELDS and name in _PUBLIC_CONFIG_FIELDS
        if not is_writable_public and name not in _MODEL_ROLE_FIELDS:
            continue
        value = info.get_default(call_default_factory=True)
        if value is PydanticUndefined:  # pragma: no cover
            continue
        defaults[name] = value
    return defaults


async def get_config_defaults() -> ConfigResponse:
    """Return canonical defaults for every public config field.

    Covers writable fields (resettable via PATCH /api/config) and the
    model-role fields (resettable via PUT /api/models/<role>).

    Deepcopies the cached dict so callers that mutate the response
    (list-valued fields like ``crawl_exclude_patterns``) cannot poison
    subsequent calls.
    """
    return ConfigResponse(**copy.deepcopy(_compute_config_defaults()))


async def models_show(model: str) -> ModelsShowResponse:
    """Return model metadata/parameters. Returns empty model if unavailable."""
    provider = get_services().provider
    result = provider.show_model(model)
    return ModelsShowResponse(**(result or {}))


def _parse_source(source: str) -> ModelSource:
    """Convert a source string to ModelSource enum."""
    return ModelSource(source)


async def models_catalog(
    task: str | None = None,
    search: str = "",
    size: str | None = None,
    installed: bool | None = None,
    featured: bool | None = None,
    sort: str = "featured",
    limit: int = 20,
    offset: int = 0,
) -> ModelsCatalogResponse:
    """Return paginated model catalog with installed status."""
    from lilbee.catalog import enrich_catalog, get_catalog

    result = get_catalog(
        task=task,
        search=search,
        size=size,
        installed=installed,
        featured=featured,
        sort=sort,
        limit=limit,
        offset=offset,
    )

    registry = get_services().registry
    installed_names = {f"{m.name}:{m.tag}" for m in registry.list_installed()}
    enriched = enrich_catalog(result, installed_names)

    return ModelsCatalogResponse(
        total=result.total,
        limit=result.limit,
        offset=result.offset,
        has_more=result.has_more,
        models=[
            CatalogEntryResponse(
                name=e.name,
                tag=e.tag,
                hf_repo=e.hf_repo,
                task=e.task,
                display_name=e.display_name,
                param_count=e.param_count,
                size_gb=e.size_gb,
                min_ram_gb=e.min_ram_gb,
                description=e.description,
                quality_tier=e.quality_tier,
                featured=e.featured,
                downloads=e.downloads,
                installed=e.installed,
                source=e.source,
            )
            for e in enriched
        ],
    )


async def models_installed() -> ModelsInstalledResponse:
    """Return list of installed models with their source."""
    manager = get_model_manager()
    names = manager.list_installed()
    models = []
    for name in names:
        src = manager.get_source(name)
        source_str = src.value if src is not None else ModelSource.REMOTE.value
        models.append(InstalledModelEntry(name=name, source=source_str))
    return ModelsInstalledResponse(models=models)


async def models_pull(model: str, *, source: str = "native") -> AsyncGenerator[str, None]:
    """Yield SSE progress events while pulling a model in real time.
    Sets a cancel event on client disconnect so the pull stops.
    """
    manager = get_model_manager()
    src = _parse_source(source)
    sse = SseStream()

    def _pull_blocking() -> None:
        def _on_progress(data: dict[str, Any]) -> None:
            if sse.cancel.is_set():
                return
            payload = sse_event(SseEvent.PROGRESS, data)
            sse.loop.call_soon_threadsafe(sse.queue.put_nowait, payload)

        def _on_bytes(downloaded: int, total: int) -> None:
            if sse.cancel.is_set():
                return
            payload = sse_event(SseEvent.PROGRESS, {"current": downloaded, "total": total})
            sse.loop.call_soon_threadsafe(sse.queue.put_nowait, payload)

        try:
            manager.pull(model, src, on_progress=_on_progress, on_bytes=_on_bytes)
        except Exception as exc:
            sse.loop.call_soon_threadsafe(sse.queue.put_nowait, sse_error(str(exc)))
        finally:
            sse.loop.call_soon_threadsafe(sse.queue.put_nowait, None)

    task = asyncio.ensure_future(asyncio.to_thread(_pull_blocking))
    async for event in sse.drain(task, "Model pull stream"):
        yield event


async def models_delete(model: str, *, source: str = "native") -> ModelsDeleteResponse:
    """Delete a model. Returns deletion status, model name, and freed space."""
    manager = get_model_manager()
    src = _parse_source(source)
    deleted = manager.remove(model, src)
    return ModelsDeleteResponse(deleted=deleted, model=model, freed_gb=0.0)


async def crawl_stream(
    url: str, depth: int | None = None, max_pages: int | None = None
) -> AsyncGenerator[str, None]:
    """Stream crawl progress as SSE events.
    Emits crawl_start, crawl_page, crawl_done events, then a final done event
    with the list of files written. On error emits crawl_error.
    Sets a cancel event on client disconnect so the crawl stops between pages.

    On first use, Chromium isn't installed yet. The stream inlines
    setup_start/progress/done events before the crawl begins so the
    Obsidian plugin's Task Center renders a matching 'setup' pill (bb-wq8g).
    """
    sse = SseStream()

    async def _run_crawl() -> list[Path]:
        from lilbee.crawler import crawl_and_save

        # crawl_and_save runs the Chromium bootstrap itself on first use,
        # relaying setup_* events through the same on_progress callback
        # so the SSE stream carries them before any crawl_* events.
        try:
            return await crawl_and_save(
                url, depth=depth, max_pages=max_pages, on_progress=sse.callback, cancel=sse.cancel
            )
        finally:
            sse.queue.put_nowait(None)

    task = asyncio.create_task(_run_crawl())
    async for event in sse.drain(task, "Crawl stream"):
        yield event
    if not sse.cancel.is_set() and task.done() and not task.cancelled():
        exc = task.exception()
        if exc is not None:
            yield sse_error(str(exc))
            return
        paths = task.result()
        yield sse_done({"files_written": [str(p) for p in paths]})


_EXTERNAL_MODELS_TTL = 60


class _ExternalModelsCache:
    """TTL cache for external model listings (no module-level mutable global)."""

    def __init__(self) -> None:
        self._time: float = 0.0
        self._key: str = ""
        self._result: ExternalModelsResponse | None = None

    def get(self, key: str) -> ExternalModelsResponse | None:
        now = time.monotonic()
        if self._result and key == self._key and (now - self._time) < _EXTERNAL_MODELS_TTL:
            return self._result
        return None

    def set(self, key: str, result: ExternalModelsResponse) -> None:
        self._time = time.monotonic()
        self._key = key
        self._result = result


_external_cache = _ExternalModelsCache()


async def list_external_models() -> ExternalModelsResponse:
    """Query the provider for available models via its list_models() API."""
    key = f"{cfg.remote_base_url}:{cfg.llm_api_key or ''}"
    cached = _external_cache.get(key)
    if cached:
        return cached

    try:
        models = await asyncio.to_thread(get_services().provider.list_models)
        result = ExternalModelsResponse(models=models)
        _external_cache.set(key, result)
        return result
    except Exception as exc:
        log.warning("Failed to list external models: %s", exc)
        return ExternalModelsResponse(models=[], error=str(exc))
