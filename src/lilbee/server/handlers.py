"""Request handlers for the lilbee HTTP API."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from litestar import post
from litestar.exceptions import ValidationException
from litestar.response import Stream

from lilbee.progress import DetailedProgressCallback

log = logging.getLogger(__name__)

# Hard cap on files per single /api/add request
MAX_ADD_FILES = 100


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


@post("/api/add")
async def add_files(data: dict[str, Any]) -> Stream:
    """Add files to the knowledge base with streaming SSE progress.

    Accepts JSON: ``{paths: string[], force?: bool, vision_model?: string}``.
    Streams SSE events: file_start, extract, embed, file_done, done, summary.
    """
    paths = data.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValidationException("'paths' must be a non-empty list of strings")
    if len(paths) > MAX_ADD_FILES:
        raise ValidationException(f"Too many files: {len(paths)} exceeds limit of {MAX_ADD_FILES}")

    force = bool(data.get("force", False))
    vision_model = str(data.get("vision_model", "") or "")

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    task = asyncio.create_task(_run_add(paths, force, vision_model, queue))

    async def _stream() -> AsyncGenerator[bytes, None]:
        async for chunk in _sse_generator(queue):
            yield chunk
        await task

    return Stream(_stream(), media_type="text/event-stream")
