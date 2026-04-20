"""Setup routes — status and bootstrap for optional runtime components.

Currently exposes Playwright Chromium bootstrap (needed for /crawl). The
bb-wq8g contract mirrors what the TUI does in ``TaskBarController.ensure_chromium``
so the Obsidian plugin's Task Center can render a matching ``setup`` pill.

Endpoints:
    GET  /setup/crawler/status → { installed, component, browsers_path }
    POST /setup/crawler         → text/event-stream of setup_start →
                                   setup_progress → setup_done → done
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from litestar import get, post
from litestar.response import Stream

from lilbee.crawler import (
    _playwright_browsers_path,
    bootstrap_chromium,
    playwright_chromium_installed,
)
from lilbee.server.handlers import SseStream, sse_done, sse_error


@get("/setup/crawler/status")
async def setup_crawler_status_route() -> dict[str, Any]:
    """Return whether the Chromium browser is installed."""
    return {
        "installed": playwright_chromium_installed(),
        "component": "chromium",
        "browsers_path": str(_playwright_browsers_path()),
    }


async def _bootstrap_crawler_stream() -> AsyncGenerator[str, None]:
    sse = SseStream()

    async def _run() -> None:
        try:
            await bootstrap_chromium(on_progress=sse.callback)
        finally:
            sse.queue.put_nowait(None)

    task = asyncio.create_task(_run())
    async for event in sse.drain(task, "Crawler setup stream"):
        yield event
    if task.done() and not task.cancelled():
        exc = task.exception()
        if exc is not None:
            yield sse_error(str(exc))
            return
    yield sse_done({})


@post("/setup/crawler")
async def setup_crawler_route() -> Stream:
    """Stream the Chromium bootstrap subprocess as SSE events."""
    return Stream(_bootstrap_crawler_stream(), media_type="text/event-stream")


__all__ = [
    "setup_crawler_route",
    "setup_crawler_status_route",
]
