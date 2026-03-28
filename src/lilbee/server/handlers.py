"""Framework-agnostic route handlers for the lilbee HTTP server.

Every public function is a plain async callable — no framework imports.
Return types are dicts (JSON responses), lists, or async generators of SSE strings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from lilbee import settings
from lilbee.cli.helpers import clean_result, copy_files, gather_status, get_version
from lilbee.config import cfg
from lilbee.progress import DetailedProgressCallback, EventType, SseEvent
from lilbee.results import group, to_dicts
from lilbee.security import validate_path_within

if TYPE_CHECKING:
    from lilbee.model_manager import ModelSource
    from lilbee.query import ChatMessage

log = logging.getLogger(__name__)

MAX_ADD_FILES = 100

_PUBLIC_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "chat_model",
        "embedding_model",
        "vision_model",
        "litellm_base_url",
        "system_prompt",
        "top_k",
        "max_distance",
        "chunk_size",
        "chunk_overlap",
        "temperature",
        "top_p",
        "top_k_sampling",
        "repeat_penalty",
        "num_ctx",
        "seed",
        "llm_provider",
        "diversity_max_per_source",
        "mmr_lambda",
        "candidate_multiplier",
        "query_expansion_count",
        "adaptive_threshold_step",
        "show_reasoning",
    }
)


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


async def list_models() -> dict[str, Any]:
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
    return response.model_dump()


async def set_chat_model(model: str) -> dict[str, str]:
    """Switch active chat model. Returns {model}."""
    from lilbee.models import ensure_tag

    tagged = ensure_tag(model)
    cfg.chat_model = tagged
    settings.set_value(cfg.data_root, "chat_model", tagged)
    return {"model": tagged}


async def set_vision_model(model: str) -> dict[str, str]:
    """Switch active vision model. Pass empty string to disable. Returns {model}."""
    cfg.vision_model = model
    settings.set_value(cfg.data_root, "vision_model", model)
    return {"model": model}


async def delete_documents(names: list[str], *, delete_files: bool = False) -> dict[str, Any]:
    """Remove documents from the knowledge base by source name."""
    from lilbee.services import get_services

    result = get_services().store.remove_documents(names, delete_files=delete_files)
    return {"removed": result.removed, "not_found": result.not_found}


async def list_documents(
    search: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Return indexed documents with metadata, paginated and filterable."""
    from lilbee.services import get_services

    sources = get_services().store.get_sources()
    if search:
        search_lower = search.lower()
        sources = [s for s in sources if search_lower in s["filename"].lower()]
    total = len(sources)
    page = sources[offset : offset + limit]
    return {
        "documents": [
            {
                "filename": s["filename"],
                "chunk_count": s.get("chunk_count", 0),
                "ingested_at": s.get("ingested_at", ""),
            }
            for s in page
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


async def get_config() -> dict[str, Any]:
    """Return all user-facing configuration values."""
    from lilbee.reranker import reranker_available

    dumped = cfg.model_dump()
    result = {k: v for k, v in dumped.items() if k in _PUBLIC_CONFIG_FIELDS}
    if reranker_available():
        result["reranker_model"] = dumped["reranker_model"]
        result["rerank_candidates"] = dumped["rerank_candidates"]
    return result


async def models_show(model: str) -> dict[str, Any]:
    """Return model metadata/parameters. Returns empty dict if unavailable."""
    from lilbee.services import get_services

    provider = get_services().provider
    result = provider.show_model(model)
    return result if result is not None else {}


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
) -> dict[str, Any]:
    """Return paginated model catalog with installed status."""
    from lilbee.catalog import get_catalog

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

    models = []
    for m in result.models:
        source = "litellm" if m.name in installed_names else "native"
        models.append(
            {
                "name": m.name,
                "size_gb": m.size_gb,
                "min_ram_gb": m.min_ram_gb,
                "description": m.description,
                "installed": m.name in installed_names,
                "source": source,
            }
        )

    return {
        "total": result.total,
        "limit": result.limit,
        "offset": result.offset,
        "models": models,
    }


async def models_installed() -> dict[str, Any]:
    """Return list of installed models with their source."""
    from lilbee.model_manager import ModelSource, get_model_manager

    manager = get_model_manager()
    names = manager.list_installed()
    models = []
    for name in names:
        src = manager.get_source(name)
        source_str = src.value if src is not None else ModelSource.LITELLM.value
        models.append({"name": name, "source": source_str})
    return {"models": models}


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
        except TimeoutError:
            continue
        if item is None:
            break
        yield item


async def models_delete(model: str, *, source: str = "litellm") -> dict[str, Any]:
    """Delete a model. Returns {deleted, model, freed_gb}."""
    from lilbee.model_manager import get_model_manager

    manager = get_model_manager()
    src = _parse_source(source)
    deleted = manager.remove(model, src)
    return {"deleted": deleted, "model": model, "freed_gb": 0.0}


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
