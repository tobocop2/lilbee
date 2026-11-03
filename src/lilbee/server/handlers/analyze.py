"""Analyze handlers: the SSE-streamed run, and the analyze tip state."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from litestar.exceptions import HTTPException
from litestar.status_codes import HTTP_503_SERVICE_UNAVAILABLE

from lilbee.app.analyze import AnalyzeReport, AnalyzeRequest, hide_tip, run_analysis, tip_state
from lilbee.app.profiles import file_failure_message
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.server.handlers.sse import SseStream
from lilbee.server.models import AnalyzeResponse, AnalyzeStateResponse


async def _run(sse: SseStream, request: AnalyzeRequest) -> AnalyzeReport:
    """Run analyze with the stream's progress and cancel; the drain ends when this task does."""
    try:
        return await run_analysis(
            get_services().profile_store, request, on_progress=sse.callback, cancel=sse.cancel
        )
    except OSError as exc:
        raise RuntimeError(file_failure_message(exc)) from exc


def _done_payload(report: AnalyzeReport) -> dict[str, object]:
    return AnalyzeResponse.from_report(report).model_dump(mode="json")


async def analyze_stream(request: AnalyzeRequest) -> AsyncGenerator[str, None]:
    """Yield ``analyze`` progress events, then ``done`` with the report or ``error``."""
    sse = SseStream()
    task = asyncio.create_task(_run(sse, request))
    async for event in sse.drain(task, "Analyze stream"):
        yield event
    frame = sse.terminal_frame(task, _done_payload)
    if frame is not None:
        yield frame


async def analyze_state() -> AnalyzeStateResponse:
    """The project's analyze tip state."""
    return AnalyzeStateResponse.from_state(await asyncio.to_thread(tip_state, cfg.data_root))


async def dismiss_tip() -> AnalyzeStateResponse:
    """Hide the analyze tip for the project and return the new state; a failed write is 503."""
    try:
        await asyncio.to_thread(hide_tip, cfg.data_root)
    except OSError as exc:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=file_failure_message(exc)
        ) from exc
    return await analyze_state()
