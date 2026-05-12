"""Crawl streaming handler."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

from lilbee.server.handlers.sse import SseStream, sse_done, sse_error


async def crawl_stream(
    url: str, depth: int | None = None, max_pages: int | None = None
) -> AsyncGenerator[str, None]:
    """Stream crawl progress as SSE events.
    Emits crawl_start, crawl_page, crawl_done events, then a final done event
    with the list of files written. On error emits crawl_error.
    Sets a cancel event on client disconnect so the crawl stops between pages.

    On first use, Chromium isn't installed yet. The stream inlines
    setup_start/progress/done events before the crawl begins so a stream
    consumer can render a matching 'setup' progress indicator.
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
