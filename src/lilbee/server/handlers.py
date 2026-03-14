"""Framework-agnostic route handlers for the lilbee HTTP server.

Every public function is a plain async callable — no framework imports.
Return types are dicts (JSON responses), lists, or async generators of SSE strings.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lilbee.progress import DetailedProgressCallback

log = logging.getLogger(__name__)

MAX_ADD_FILES = 100


def sse_event(event: str, data: Any) -> str:
    """Format a single Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _make_sse_callback(queue: asyncio.Queue[str | None]) -> DetailedProgressCallback:
    """Return a progress callback that serializes events into an asyncio queue.

    Safe to call from both the event loop thread (async code) and worker
    threads (``asyncio.to_thread`` / ``run_in_executor``).
    """
    loop = asyncio.get_event_loop()

    def _callback(event_type: str, data: dict[str, Any]) -> None:
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


async def _sse_generator(queue: asyncio.Queue[str | None]) -> AsyncGenerator[bytes, None]:
    """Yield SSE-formatted bytes from a queue until sentinel (None) is received."""
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item.encode()


async def health() -> dict[str, str]:
    """Return service health and version."""
    from lilbee.cli.helpers import get_version

    return {"status": "ok", "version": get_version()}


async def status() -> dict[str, Any]:
    """Return config, sources, and chunk counts."""
    from lilbee.cli.helpers import gather_status

    return gather_status()


async def search(q: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search and return grouped DocumentResults as dicts."""
    from lilbee.query import search_context
    from lilbee.results import group, to_dicts

    results = search_context(q, top_k=top_k)
    grouped = group(results)
    return to_dicts(grouped)


async def ask(question: str, top_k: int = 0) -> dict[str, Any]:
    """One-shot RAG answer. Returns {answer, sources[]}."""
    from lilbee.cli.helpers import clean_result
    from lilbee.query import ask_raw

    result = ask_raw(question, top_k=top_k)
    return {
        "answer": result.answer,
        "sources": [clean_result(s) for s in result.sources],
    }


async def ask_stream(question: str, top_k: int = 0) -> AsyncGenerator[str, None]:
    """Yield SSE events: token, sources, done."""
    yield ""  # force generator
    from lilbee.cli.helpers import clean_result
    from lilbee.config import cfg
    from lilbee.query import (
        _CONTEXT_TEMPLATE,
        build_context,
        search_context,
        sort_by_relevance,
    )

    results = search_context(question, top_k=top_k)
    if not results:
        yield sse_event("error", {"message": "No relevant documents found."})
        return

    results = sort_by_relevance(results)
    context = build_context(results)
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)
    messages: list[dict[str, str]] = [{"role": "system", "content": cfg.system_prompt}]
    messages.append({"role": "user", "content": prompt})

    try:
        import ollama as ollama_client

        stream = await asyncio.to_thread(
            lambda: ollama_client.chat(model=cfg.chat_model, messages=messages, stream=True)
        )
        for chunk in stream:
            token = chunk.message.content
            if token:
                yield sse_event("token", {"token": token})
    except Exception as exc:
        yield sse_event("error", {"message": str(exc)})
        return

    yield sse_event("sources", [clean_result(s) for s in results])
    yield sse_event("done", {})


async def chat(question: str, history: list[dict[str, str]], top_k: int = 0) -> dict[str, Any]:
    """Chat with history. Returns {answer, sources[]}."""
    from lilbee.cli.helpers import clean_result
    from lilbee.query import ask_raw

    result = ask_raw(question, top_k=top_k, history=history)
    return {
        "answer": result.answer,
        "sources": [clean_result(s) for s in result.sources],
    }


async def chat_stream(
    question: str, history: list[dict[str, str]], top_k: int = 0
) -> AsyncGenerator[str, None]:
    """Yield SSE events with chat history support."""
    yield ""  # force generator
    from lilbee.cli.helpers import clean_result
    from lilbee.config import cfg
    from lilbee.query import (
        _CONTEXT_TEMPLATE,
        build_context,
        search_context,
        sort_by_relevance,
    )

    results = search_context(question, top_k=top_k)
    if not results:
        yield sse_event("error", {"message": "No relevant documents found."})
        return

    results = sort_by_relevance(results)
    context = build_context(results)
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)
    messages: list[dict[str, str]] = [{"role": "system", "content": cfg.system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    try:
        import ollama as ollama_client

        stream = await asyncio.to_thread(
            lambda: ollama_client.chat(model=cfg.chat_model, messages=messages, stream=True)
        )
        for chunk in stream:
            token = chunk.message.content
            if token:
                yield sse_event("token", {"token": token})
    except Exception as exc:
        yield sse_event("error", {"message": str(exc)})
        return

    yield sse_event("sources", [clean_result(s) for s in results])
    yield sse_event("done", {})


async def sync_stream(*, force_vision: bool = False) -> AsyncGenerator[str, None]:
    """Trigger sync, yield SSE progress events, then done event."""
    from lilbee.ingest import sync

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    callback = _make_sse_callback(queue)

    async def run_sync() -> object:
        return await sync(quiet=True, on_progress=callback, force_vision=force_vision)

    task = asyncio.create_task(run_sync())
    while not task.done() or not queue.empty():
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.1)
        except TimeoutError:
            continue
        if item is not None:
            yield item
    result = task.result()
    yield sse_event("done", asdict(result))  # type: ignore[call-overload]


async def _run_add(
    paths: list[str],
    force: bool,
    vision_model: str,
    queue: asyncio.Queue[str | None],
) -> None:
    """Copy files and sync, pushing SSE events to the queue."""
    from lilbee.cli.helpers import copy_files
    from lilbee.config import cfg
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

    old_vision = cfg.vision_model
    if vision_model:
        cfg.vision_model = vision_model
    try:
        sync_result = await sync(quiet=True, force_vision=bool(vision_model), on_progress=callback)
    finally:
        if vision_model:
            cfg.vision_model = old_vision

    summary = {
        "copied": copy_result.copied,
        "skipped": copy_result.skipped,
        "errors": errors,
        "sync": asdict(sync_result),
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

    force = bool(data.get("force", False))
    vision_model = str(data.get("vision_model", "") or "")

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    task = asyncio.create_task(_run_add(paths, force, vision_model, queue))
    return paths, queue, task


async def list_models() -> dict[str, Any]:
    """Return chat and vision model catalogs with installed status."""
    from lilbee.cli.chat import list_ollama_models
    from lilbee.config import cfg
    from lilbee.models import MODEL_CATALOG, VISION_CATALOG

    installed = set(list_ollama_models())
    chat_installed = set(list_ollama_models(exclude_vision=True))
    vision_names = {v.name for v in VISION_CATALOG}
    return {
        "chat": {
            "active": cfg.chat_model,
            "catalog": [
                {
                    "name": m.name,
                    "size_gb": m.size_gb,
                    "min_ram_gb": m.min_ram_gb,
                    "description": m.description,
                    "installed": m.name in installed,
                }
                for m in MODEL_CATALOG
            ],
            "installed": sorted(chat_installed),
        },
        "vision": {
            "active": cfg.vision_model,
            "catalog": [
                {
                    "name": m.name,
                    "size_gb": m.size_gb,
                    "min_ram_gb": m.min_ram_gb,
                    "description": m.description,
                    "installed": m.name in installed,
                }
                for m in VISION_CATALOG
            ],
            "installed": sorted(m for m in installed if m in vision_names),
        },
    }


async def pull_model(model: str) -> AsyncGenerator[str, None]:
    """Pull an Ollama model, yielding SSE progress events."""
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _pull() -> None:
        import ollama as ollama_client

        for event in ollama_client.pull(model, stream=True):
            total = event.total or 0
            completed = event.completed or 0
            queue.put_nowait(
                sse_event(
                    "progress",
                    {
                        "model": model,
                        "status": event.status or "",
                        "total": total,
                        "completed": completed,
                    },
                )
            )
        queue.put_nowait(None)  # sentinel

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _pull)
    while True:
        event = await queue.get()
        if event is None:
            break
        yield event
    yield sse_event("done", {"model": model})


async def set_chat_model(model: str) -> dict[str, str]:
    """Switch active chat model. Returns {model}."""
    from lilbee import settings
    from lilbee.config import cfg
    from lilbee.models import ensure_tag

    tagged = ensure_tag(model)
    cfg.chat_model = tagged
    settings.set_value(cfg.data_root, "chat_model", tagged)
    return {"model": tagged}


async def set_vision_model(model: str) -> dict[str, str]:
    """Switch active vision model. Pass empty string to disable. Returns {model}."""
    from lilbee import settings
    from lilbee.config import cfg

    cfg.vision_model = model
    settings.set_value(cfg.data_root, "vision_model", model)
    return {"model": model}
