"""Wiki build and synthesis handlers (SSE-streamed)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

from lilbee.core.config import cfg
from lilbee.runtime.progress import DetailedProgressCallback
from lilbee.server.handlers.sse import SseStream
from lilbee.wiki import run_full_build, run_full_synthesize

_Summary = dict[str, Any]


async def _wiki_run_stream(
    run: Callable[[DetailedProgressCallback], _Summary], label: str
) -> AsyncGenerator[str, None]:
    """Run a wiki job off the event loop, yielding its progress as SSE.

    Emits wiki_phase and wiki_page events while the job runs, then a done event
    carrying the run summary.
    """
    sse = SseStream()

    async def _run() -> _Summary:
        try:
            return await asyncio.to_thread(run, sse.callback)
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
        lambda on_progress: dict(run_full_build(cfg, on_progress)), "Wiki build stream"
    ):
        yield event


async def wiki_synthesize_stream() -> AsyncGenerator[str, None]:
    """Generate synthesis pages for cross-source clusters, streaming progress."""
    async for event in _wiki_run_stream(
        lambda on_progress: dict(run_full_synthesize(cfg, on_progress)), "Wiki synthesize stream"
    ):
        yield event
