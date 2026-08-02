"""Wiki build and synthesis handlers (SSE-streamed)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Callable
from typing import Any

from lilbee.core.config import cfg
from lilbee.runtime.progress import DetailedProgressCallback
from lilbee.server.handlers.sse import SseStream
from lilbee.wiki import run_full_build, run_full_synthesize

_Summary = dict[str, Any]


async def _wiki_run_stream(
    run: Callable[[DetailedProgressCallback, threading.Event], _Summary], label: str
) -> AsyncGenerator[str, None]:
    """Run a wiki job off the event loop, yielding its progress as SSE.

    Emits wiki_phase and wiki_page events while the job runs, then a done event
    carrying the run summary. The run gets the stream's cancel event, so a
    client disconnect stops it at the next source boundary; without that the
    worker keeps building the whole corpus and holds the wiki build mutex,
    blocking every other surface.
    """
    sse = SseStream()

    async def _run() -> _Summary:
        try:
            return await asyncio.to_thread(run, sse.callback, sse.cancel)
        finally:
            sse.queue.put_nowait(None)

    task = asyncio.create_task(_run())
    async for event in sse.drain(task, label):
        yield event
    frame = sse.terminal_frame(task, dict)
    if frame is not None:
        yield frame


async def wiki_build_stream() -> AsyncGenerator[str, None]:
    """Build the concept and entity wiki, streaming progress."""
    async for event in _wiki_run_stream(
        lambda on_progress, cancel: dict(run_full_build(cfg, on_progress, cancel)),
        "Wiki build stream",
    ):
        yield event


async def wiki_synthesize_stream() -> AsyncGenerator[str, None]:
    """Generate synthesis pages for cross-source clusters, streaming progress."""
    async for event in _wiki_run_stream(
        lambda on_progress, cancel: dict(run_full_synthesize(cfg, on_progress, cancel)),
        "Wiki synthesize stream",
    ):
        yield event
