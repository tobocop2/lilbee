"""A stream whose producer was cancelled ends without a done or error frame.

``terminal_frame`` returns None for a cancelled producer, and every streaming
handler then closes the stream silently instead of inventing a terminal event.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from unittest.mock import patch

import pytest

from lilbee.server import handlers
from lilbee.server.handlers import analyze as analyze_handlers
from lilbee.server.handlers import wiki as wiki_handlers
from lilbee.server.handlers.sse import SseStream, sse_event
from lilbee.server.routes import setup as setup_routes


async def _cancelled(*_args: object, **_kwargs: object) -> None:
    raise asyncio.CancelledError


def _cancelled_in_thread(*_args: object) -> dict[str, object]:
    raise asyncio.CancelledError


async def _collect(stream: AsyncGenerator[str, None]) -> list[str]:
    return [frame async for frame in stream]


@pytest.mark.parametrize(
    ("target", "make_stream"),
    [
        ("lilbee.crawler.crawl_and_save", lambda: handlers.crawl_stream("https://example.com")),
        ("lilbee.data.ingest.sync", lambda: handlers.sync_stream()),
        ("lilbee.app.dataset.import_from_bytes", lambda: handlers.import_stream(b"{}", "json")),
        ("lilbee.server.routes.setup.bootstrap_chromium", setup_routes._bootstrap_crawler_stream),
    ],
    ids=["crawl", "sync", "import", "crawler-setup"],
)
async def test_a_cancelled_producer_ends_the_stream_without_a_terminal_frame(
    target: str, make_stream: Callable[[], AsyncGenerator[str, None]]
) -> None:
    with (
        patch("lilbee.crawler.bootstrap.chromium_installed", return_value=True),
        patch(target, _cancelled),
    ):
        frames = await _collect(make_stream())
    assert frames == []


async def test_a_cancelled_wiki_job_ends_the_stream_without_a_terminal_frame() -> None:
    frames = await _collect(wiki_handlers._wiki_run_stream(_cancelled_in_thread, "Wiki stream"))
    assert frames == []


async def test_a_cancelled_analyze_run_ends_the_stream_without_a_terminal_frame() -> None:
    request = analyze_handlers.AnalyzeRequest(directory=None, apply=False)
    with patch.object(analyze_handlers, "_run", _cancelled):
        frames = await _collect(analyze_handlers.analyze_stream(request))
    assert frames == []


async def test_drain_skips_a_second_sentinel_left_in_the_queue() -> None:
    sse = SseStream()
    task = asyncio.ensure_future(asyncio.sleep(0))
    first = sse_event("progress", {"n": 1})
    second = sse_event("progress", {"n": 2})
    sse.queue.put_nowait(first)
    sse.queue.put_nowait(None)
    sse.queue.put_nowait(None)
    sse.queue.put_nowait(second)
    frames = [frame async for frame in sse.drain(task, "Test stream")]
    assert frames == [first, second]
