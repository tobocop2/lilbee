"""Framework-agnostic route handlers for the lilbee HTTP server.

Every public function is a plain async callable — no framework imports.
Return types are dicts (JSON responses), lists, or async generators of SSE strings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import types
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from lilbee import settings
from lilbee.cli.helpers import clean_result, copy_files, gather_status, get_version
from lilbee.config import Config, cfg
from lilbee.progress import DetailedProgressCallback, EventType, SseEvent
from lilbee.results import group, to_dicts
from lilbee.security import validate_path_within
from lilbee.server.models import (
    CatalogEntryResponse,
    ConfigResponse,
    ConfigUpdateResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentRemoveResponse,
    ExternalModelsResponse,
    InstalledModelEntry,
    ModelsCatalogResponse,
    ModelsDeleteResponse,
    ModelsInstalledResponse,
    ModelsShowResponse,
)

if TYPE_CHECKING:
    from lilbee.model_manager import ModelSource
    from lilbee.query import ChatMessage

log = logging.getLogger(__name__)

MAX_ADD_FILES = 100


def _get_extra(info: FieldInfo, key: str, default: bool = False) -> bool:
    """Read a boolean flag from a field's ``json_schema_extra``."""
    extra = info.json_schema_extra
    if isinstance(extra, dict):
        return bool(extra.get(key, default))
    return default


def _is_nullable(info: FieldInfo) -> bool:
    """Return True if ``None`` is part of the field's type union."""
    origin = get_origin(info.annotation)
    if origin is Union or origin is types.UnionType:
        return type(None) in get_args(info.annotation)
    return False


def _derive_field_sets() -> tuple[
    types.MappingProxyType[str, bool], frozenset[str], frozenset[str]
]:
    """Derive writable, reindex, and public field sets from Config metadata."""
    writable: dict[str, bool] = {}
    reindex: set[str] = set()
    public: set[str] = set()
    for name, info in Config.model_fields.items():
        if _get_extra(info, "writable"):
            writable[name] = _is_nullable(info)
            if not _get_extra(info, "write_only") and _get_extra(info, "public", default=True):
                public.add(name)
            if _get_extra(info, "reindex"):
                reindex.add(name)
        elif name in {"chat_model", "embedding_model", "vision_model"}:
            public.add(name)
    return types.MappingProxyType(writable), frozenset(reindex), frozenset(public)


WRITABLE_CONFIG_FIELDS, REINDEX_FIELDS, _PUBLIC_CONFIG_FIELDS = _derive_field_sets()


class ModelCatalogEntry(BaseModel):
    """A single model in the catalog."""

    name: str
    size_gb: float
    min_ram_gb: float
    description: str
    installed: bool


class ModelCatalogSection(BaseModel):
    """Chat or vision model catalog with active model and installed list."""

    active: str
    catalog: list[ModelCatalogEntry]
    installed: list[str]


class ModelsResponse(BaseModel):
    """Response for the list-models endpoint."""

    chat: ModelCatalogSection
    vision: ModelCatalogSection


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


def _make_sse_callback(queue: asyncio.Queue[str | None]) -> DetailedProgressCallback:
    """Return a progress callback that serializes events into an asyncio queue.

    Safe to call from both the event loop thread (async code) and worker
    threads (``asyncio.to_thread`` / ``run_in_executor``).
    """
    loop = asyncio.get_running_loop()

    def _callback(event_type: EventType, data: dict[str, Any]) -> None:
        payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            queue.put_nowait(payload)
        else:
            loop.call_soon_threadsafe(queue.put_nowait, payload)

    return _callback


async def sse_generator(queue: asyncio.Queue[str | None]) -> AsyncGenerator[bytes, None]:
    """Yield SSE-formatted bytes from a queue until sentinel (None) is received."""
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item.encode()


async def health() -> dict[str, str]:
    """Return service health and version."""
    return {"status": "ok", "version": get_version()}


async def status() -> dict[str, Any]:
    """Return config, sources, and chunk counts."""
    return gather_status().model_dump(exclude_none=True)


async def search(q: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search and return grouped DocumentResults as dicts."""
    from lilbee.services import get_services

    results = get_services().searcher.search(q, top_k=top_k)
    grouped = group(results)
    return to_dicts(grouped)


async def ask(
    question: str, top_k: int = 0, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One-shot RAG answer. Returns {answer, sources[]}."""
    from lilbee.services import get_services

    opts = _resolve_generation_options(options)
    result = get_services().searcher.ask_raw(question, top_k=top_k, options=opts)
    return {
        "answer": result.answer,
        "sources": [clean_result(s) for s in result.sources],
    }


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
        from lilbee.services import get_services

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


async def _drain_token_queue(queue: asyncio.Queue[str | None]) -> AsyncGenerator[str, None]:
    """Yield SSE strings from the token queue until sentinel."""
    while True:
        event = await queue.get()
        if event is None:
            break
        yield event


async def _stream_rag_response(
    question: str,
    history: list[ChatMessage] | None = None,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    """Shared SSE streaming for ask_stream and chat_stream."""
    yield ""  # force generator

    from lilbee.services import get_services

    rag = get_services().searcher.build_rag_context(question, top_k=top_k, history=history)
    if rag is None:
        yield sse_error("No relevant documents found.")
        return

    results, messages = rag
    opts = _resolve_generation_options(options) or cfg.generation_options()

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    cancel = threading.Event()
    error_holder: list[str] = []

    loop = asyncio.get_running_loop()
    executor_fut = loop.run_in_executor(
        None, _run_llm_stream, messages, opts, queue, cancel, error_holder
    )
    try:
        async for event in _drain_token_queue(queue):
            yield event
    except (asyncio.CancelledError, GeneratorExit):
        log.info("Stream cancelled by client")
        cancel.set()
        return

    if error_holder:
        log.warning("Stream error: %s", error_holder[0])
        yield sse_error("Internal error")
        cancel.set()
        return

    # Ensure executor thread has finished before yielding final events
    await executor_fut

    yield sse_event(SseEvent.SOURCES, [clean_result(s) for s in results])
    yield sse_done({})


def ask_stream(
    question: str, top_k: int = 0, options: dict[str, Any] | None = None
) -> AsyncGenerator[str, None]:
    """Yield SSE events: token, sources, done."""
    return _stream_rag_response(question, top_k=top_k, options=options)


async def chat(
    question: str,
    history: list[ChatMessage],
    top_k: int = 0,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chat with history. Returns {answer, sources[]}."""
    from lilbee.services import get_services

    opts = _resolve_generation_options(options)
    result = get_services().searcher.ask_raw(question, top_k=top_k, history=history, options=opts)
    return {
        "answer": result.answer,
        "sources": [clean_result(s) for s in result.sources],
    }


def chat_stream(
    question: str,
    history: list[ChatMessage],
    top_k: int = 0,
    options: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events with chat history support."""
    return _stream_rag_response(question, history=history, top_k=top_k, options=options)


async def sync_stream(*, force_vision: bool = False) -> AsyncGenerator[str, None]:
    """Trigger sync, yield SSE progress events, then done event."""
    from lilbee.ingest import SyncResult, sync

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    callback = _make_sse_callback(queue)

    async def run_sync() -> SyncResult:
        return await sync(quiet=True, on_progress=callback, force_vision=force_vision)

    task = asyncio.create_task(run_sync())
    while not task.done() or not queue.empty():
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.1)
        except TimeoutError:
            continue
        if item is not None:
            yield item
    yield sse_done(task.result().model_dump())


async def _run_add(
    paths: list[str],
    force: bool,
    vision_model: str,
    queue: asyncio.Queue[str | None],
) -> None:
    """Copy files and sync, pushing SSE events to the queue."""
    from lilbee.ingest import sync

    callback = _make_sse_callback(queue)

    errors: list[str] = []
    valid: list[Path] = []
    for p_str in paths:
        p = Path(p_str)
        if not p.exists():
            errors.append(p_str)
        else:
            valid.append(p)

    copy_result = copy_files(valid, force=force)

    from lilbee.cli.helpers import temporary_vision_model

    with temporary_vision_model(vision_model):
        sync_result = await sync(quiet=True, force_vision=bool(vision_model), on_progress=callback)

    summary = {
        "copied": copy_result.copied,
        "skipped": copy_result.skipped,
        "errors": errors,
        "sync": sync_result.model_dump(),
    }
    payload = f"event: summary\ndata: {json.dumps(summary)}\n\n"
    queue.put_nowait(payload)
    queue.put_nowait(None)  # sentinel


AddResult = tuple[list[str], asyncio.Queue[str | None], asyncio.Task[None]]


async def add_files(data: dict[str, Any]) -> AddResult:
    """Validate and start the add-files operation.

    Returns (paths, queue, task) for the Litestar adapter to stream.
    Raises ValueError on validation failure.
    """
    paths = data.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("'paths' must be a non-empty list of strings")
    if len(paths) > MAX_ADD_FILES:
        raise ValueError(f"Too many files: {len(paths)} exceeds limit of {MAX_ADD_FILES}")

    for p_str in paths:
        # Validate that the resolved target inside documents_dir won't escape
        validate_path_within(cfg.documents_dir / Path(p_str).name, cfg.documents_dir)

    force = bool(data.get("force", False))
    vision_model = str(data.get("vision_model", "") or "")

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    task = asyncio.create_task(_run_add(paths, force, vision_model, queue))
    return paths, queue, task


async def list_models() -> ModelsResponse:
    """Return chat and vision model catalogs with installed status."""
    from lilbee.models import MODEL_CATALOG, VISION_CATALOG, list_installed_models

    installed = set(list_installed_models())
    chat_installed = set(list_installed_models(exclude_vision=True))
    vision_names = {v.name for v in VISION_CATALOG}

    response = ModelsResponse(
        chat=ModelCatalogSection(
            active=cfg.chat_model,
            catalog=[
                ModelCatalogEntry(
                    name=m.name,
                    size_gb=m.size_gb,
                    min_ram_gb=m.min_ram_gb,
                    description=m.description,
                    installed=m.name in installed,
                )
                for m in MODEL_CATALOG
            ],
            installed=sorted(chat_installed),
        ),
        vision=ModelCatalogSection(
            active=cfg.vision_model,
            catalog=[
                ModelCatalogEntry(
                    name=m.name,
                    size_gb=m.size_gb,
                    min_ram_gb=m.min_ram_gb,
                    description=m.description,
                    installed=m.name in installed,
                )
                for m in VISION_CATALOG
            ],
            installed=sorted(m for m in installed if m in vision_names),
        ),
    )
    return response


async def _set_model(
    field: Literal["chat_model", "vision_model", "embedding_model"],
    model: str,
    *,
    normalize: bool = False,
) -> dict[str, str]:
    """Shared helper for switching a model field. Returns {model}."""
    if normalize:
        from lilbee.models import ensure_tag

        model = ensure_tag(model)
    setattr(cfg, field, model)
    settings.set_value(cfg.data_root, field, model)
    return {"model": model}


async def set_chat_model(model: str) -> dict[str, str]:
    """Switch active chat model. Returns {model}."""
    return await _set_model("chat_model", model, normalize=True)


async def set_vision_model(model: str) -> dict[str, str]:
    """Switch active vision model. Pass empty string to disable. Returns {model}."""
    return await _set_model("vision_model", model)


async def set_embedding_model(model: str) -> dict[str, str]:
    """Switch embedding model. Returns {model}."""
    return await _set_model("embedding_model", model)


def _validate_config_updates(updates: dict[str, Any]) -> None:
    """Reject unknown fields and null values on non-nullable fields."""
    for key, value in updates.items():
        if key not in WRITABLE_CONFIG_FIELDS:
            raise ValueError(f"Unknown or read-only config field: {key}")
        if value is None and not WRITABLE_CONFIG_FIELDS[key]:
            raise ValueError(f"Field '{key}' does not accept null")


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
    reindex_required = bool(REINDEX_FIELDS & set(updates))
    return ConfigUpdateResponse(updated=list(updates), reindex_required=reindex_required)


async def delete_documents(
    names: list[str], *, delete_files: bool = False
) -> DocumentRemoveResponse:
    """Remove documents from the knowledge base by source name."""
    from lilbee.services import get_services

    result = get_services().store.remove_documents(names, delete_files=delete_files)
    return DocumentRemoveResponse(removed=result.removed, not_found=result.not_found)


async def list_documents(
    search: str = "",
    limit: int = 50,
    offset: int = 0,
) -> DocumentListResponse:
    """Return indexed documents with metadata, paginated and filterable."""
    from lilbee.services import get_services

    sources = get_services().store.get_sources()
    if search:
        search_lower = search.lower()
        sources = [s for s in sources if search_lower in s["filename"].lower()]
    total = len(sources)
    page = sources[offset : offset + limit]
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
    )


async def get_config() -> ConfigResponse:
    """Return all user-facing configuration values."""
    from lilbee.reranker import reranker_available

    dumped = cfg.model_dump()
    result = {k: v for k, v in dumped.items() if k in _PUBLIC_CONFIG_FIELDS}
    if reranker_available():
        result["reranker_model"] = dumped["reranker_model"]
        result["rerank_candidates"] = dumped["rerank_candidates"]
    return ConfigResponse(**result)


async def models_show(model: str) -> ModelsShowResponse:
    """Return model metadata/parameters. Returns empty model if unavailable."""
    from lilbee.services import get_services

    provider = get_services().provider
    result = provider.show_model(model)
    return ModelsShowResponse(**(result or {}))


def _parse_source(source: str) -> ModelSource:
    """Convert a source string to ModelSource enum."""
    from lilbee.model_manager import ModelSource

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
    from lilbee.services import get_services

    provider = get_services().provider
    installed_names = set(provider.list_models())
    enriched = enrich_catalog(result, installed_names)

    return ModelsCatalogResponse(
        total=result.total,
        limit=result.limit,
        offset=result.offset,
        models=[
            CatalogEntryResponse(
                name=e.name,
                display_name=e.display_name,
                size_gb=e.size_gb,
                min_ram_gb=e.min_ram_gb,
                description=e.description,
                quality_tier=e.quality_tier,
                installed=e.installed,
                source=e.source,
            )
            for e in enriched
        ],
    )


async def models_installed() -> ModelsInstalledResponse:
    """Return list of installed models with their source."""
    from lilbee.model_manager import ModelSource, get_model_manager

    manager = get_model_manager()
    names = manager.list_installed()
    models = []
    for name in names:
        src = manager.get_source(name)
        source_str = src.value if src is not None else ModelSource.LITELLM.value
        models.append(InstalledModelEntry(name=name, source=source_str))
    return ModelsInstalledResponse(models=models)


async def models_pull(model: str, *, source: str = "native") -> AsyncGenerator[str, None]:
    """Yield SSE progress events while pulling a model in real time."""
    from lilbee.model_manager import get_model_manager

    manager = get_model_manager()
    src = _parse_source(source)
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _pull_blocking() -> None:
        def _on_progress(data: dict[str, Any]) -> None:
            payload = sse_event(SseEvent.PROGRESS, data)
            loop.call_soon_threadsafe(queue.put_nowait, payload)

        try:
            manager.pull(model, src, on_progress=_on_progress)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, sse_error(str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.ensure_future(asyncio.to_thread(_pull_blocking))
    while not task.done() or not queue.empty():
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.1)
        except TimeoutError:  # pragma: no cover — async polling race
            continue
        if item is None:
            break
        yield item


async def models_delete(model: str, *, source: str = "litellm") -> ModelsDeleteResponse:
    """Delete a model. Returns deletion status, model name, and freed space."""
    from lilbee.model_manager import get_model_manager

    manager = get_model_manager()
    src = _parse_source(source)
    deleted = manager.remove(model, src)
    return ModelsDeleteResponse(deleted=deleted, model=model, freed_gb=0.0)


async def crawl_stream(url: str, depth: int = 0, max_pages: int = 50) -> AsyncGenerator[str, None]:
    """Stream crawl progress as SSE events.

    Emits crawl_start, crawl_page, crawl_done events, then a final done event
    with the list of files written. On error emits crawl_error.
    """
    from lilbee.crawler import require_valid_crawl_url

    require_valid_crawl_url(url)

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    callback = _make_sse_callback(queue)

    async def _run_crawl() -> list[Path]:
        from lilbee.crawler import crawl_and_save

        return await crawl_and_save(url, depth=depth, max_pages=max_pages, on_progress=callback)

    task = asyncio.create_task(_run_crawl())
    while not task.done() or not queue.empty():
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.1)
        except TimeoutError:
            continue
        if item is not None:
            yield item

    exc = task.exception()
    if exc is not None:
        yield sse_error(str(exc))
        return

    paths = task.result()
    yield sse_done({"files_written": [str(p) for p in paths]})


_EXTERNAL_MODELS_TTL = 60
_external_cache: tuple[float, str, ExternalModelsResponse | None] = (0.0, "", None)


async def list_external_models() -> ExternalModelsResponse:
    """Query the provider for available models via its list_models() API."""
    global _external_cache
    import asyncio

    cache_time, cache_key, cache_result = _external_cache
    key = f"{cfg.litellm_base_url}:{cfg.llm_api_key or ''}"
    now = time.monotonic()
    if cache_result and key == cache_key and (now - cache_time) < _EXTERNAL_MODELS_TTL:
        return cache_result

    try:
        from lilbee.services import get_services

        models = await asyncio.to_thread(get_services().provider.list_models)
        result = ExternalModelsResponse(models=models)
        _external_cache = (now, key, result)
        return result
    except Exception as exc:
        log.warning("Failed to list external models: %s", exc)
        return ExternalModelsResponse(models=[], error=str(exc))
