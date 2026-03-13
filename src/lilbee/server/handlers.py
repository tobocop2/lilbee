"""Framework-agnostic route handlers for the lilbee HTTP server.

Every public function is a plain async callable — no framework imports.
Return types are dicts (JSON responses), lists, or async generators of SSE strings.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any


def sse_event(event: str, data: Any) -> str:
    """Format a single Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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

    queue: asyncio.Queue[str] = asyncio.Queue()

    def on_progress(name: str, status_str: str, current: int, total: int) -> None:
        queue.put_nowait(
            sse_event(
                "progress",
                {"file": name, "status": status_str, "current": current, "total": total},
            )
        )

    async def run_sync() -> object:
        return await sync(quiet=True, on_progress=on_progress, force_vision=force_vision)

    task = asyncio.create_task(run_sync())
    while not task.done() or not queue.empty():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield event
        except TimeoutError:
            continue
    result = task.result()
    yield sse_event("done", asdict(result))  # type: ignore[call-overload]


async def list_models() -> dict[str, Any]:
    """Return chat and vision model catalogs with installed status."""
    from lilbee.cli.chat import list_ollama_models
    from lilbee.config import cfg
    from lilbee.models import MODEL_CATALOG, VISION_CATALOG

    installed = set(list_ollama_models())
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
            "installed": sorted(installed),
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
