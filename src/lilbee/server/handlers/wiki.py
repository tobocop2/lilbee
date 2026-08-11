"""Wiki build, synthesis, and single-page generation handlers (SSE-streamed)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Callable
from functools import partial
from typing import Any

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.runtime.progress import DetailedProgressCallback
from lilbee.server.handlers.sse import SseStream
from lilbee.server.models import WikiGenerateResult
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


async def wiki_generate_stream(slug: str) -> AsyncGenerator[str, None]:
    """Generate one indexed page, streaming progress.

    The done event carries the written page's read slug and path; a stale
    index entry surfaces as an error event.
    """
    async for event in _wiki_run_stream(partial(_generate_one_page, slug), "Wiki generate stream"):
        yield event


def _generate_one_page(
    slug: str, on_progress: DetailedProgressCallback, cancel: threading.Event
) -> _Summary:
    """Write one indexed page and shape the done payload for the wire."""
    from lilbee.wiki.browse import page_slug
    from lilbee.wiki.lazy import generate_stub_page

    path = generate_stub_page(
        slug, svc_mod.get_services().store, on_progress=on_progress, cancel=cancel
    )
    if path is None:
        raise RuntimeError(f"index entry for {slug} is stale; its sources are gone")
    result = WikiGenerateResult(
        slug=page_slug(path, cfg.data_root / cfg.wiki_dir), path=path.as_posix()
    )
    return result.model_dump()
