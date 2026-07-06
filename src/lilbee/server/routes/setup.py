"""Setup routes: status and bootstrap for optional runtime components.

Currently exposes Playwright Chromium bootstrap (needed for /crawl). The
SSE event sequence mirrors what the TUI does in
``TaskBarController.ensure_chromium`` so a stream consumer can render a
matching ``setup`` progress indicator.

Endpoints:
    GET  /setup/crawler/status → { installed, package_installed,
                                   chromium_installed, component, browsers_path }
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
    bootstrap_chromium,
    chromium_installed,
    crawler_available,
    crawler_browsers_path,
)
from lilbee.server.auth import read_only
from lilbee.server.handlers import SseStream, sse_done, sse_error


@get("/setup/crawler/status")
@read_only
async def setup_crawler_status_route() -> dict[str, Any]:
    """Return whether the crawler is fully ready (Python package + Chromium)."""
    package_installed = crawler_available()
    chromium_ok = chromium_installed()
    return {
        "installed": package_installed and chromium_ok,
        "package_installed": package_installed,
        "chromium_installed": chromium_ok,
        "component": "chromium",
        "browsers_path": str(crawler_browsers_path()),
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
