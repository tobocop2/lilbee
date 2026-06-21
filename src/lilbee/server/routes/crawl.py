"""Crawl route handler."""

from __future__ import annotations

import anyio
from litestar import post
from litestar.exceptions import ValidationException
from litestar.response import Stream

from lilbee.server import handlers
from lilbee.server.models import CrawlRequest


@post("/api/crawl")
async def crawl_route(data: CrawlRequest) -> Stream:
    """Crawl a URL with streaming SSE progress events."""
    from lilbee.crawler import require_valid_crawl_url

    try:
        # URL validation resolves the host (blocking DNS), so it runs off the loop,
        # mirroring the MCP crawl tool so neither async transport stalls the loop.
        await anyio.to_thread.run_sync(require_valid_crawl_url, data.url)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    gen = handlers.crawl_stream(
        url=data.url, depth=data.depth, max_pages=data.max_pages, render_mode=data.render_mode
    )
    return Stream(gen, media_type="text/event-stream")
