"""Analyze routes: the streamed run, the tip state, and hiding the tip."""

from __future__ import annotations

import asyncio

from litestar import Router, get, post
from litestar.exceptions import ValidationException
from litestar.response import Stream
from litestar.status_codes import HTTP_200_OK

from lilbee.app.analyze import AnalyzeRequest, server_directory, validate_request
from lilbee.server.handlers import analyze as handlers
from lilbee.server.handlers.sse import SSE_MEDIA_TYPE
from lilbee.server.models import AnalyzeRequestBody, AnalyzeStateResponse


def _request(data: AnalyzeRequestBody) -> AnalyzeRequest:
    return AnalyzeRequest(
        directory=server_directory(data.directory),
        apply=data.apply,
        save=data.save,
        target=data.target,
    )


@post("/api/analyze", media_type=SSE_MEDIA_TYPE)
async def analyze_route(data: AnalyzeRequestBody | None = None) -> Stream:
    """Read the corpus or a folder and recommend a profile, streaming ``analyze`` progress.

    The ``done`` event carries the report; a refused request answers 400 before any file
    is read. A client that disconnects before the save starts cancels the run; nothing is saved.
    """
    try:
        request = _request(data if data is not None else AnalyzeRequestBody())
        await asyncio.to_thread(validate_request, request)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    return Stream(handlers.analyze_stream(request), media_type=SSE_MEDIA_TYPE)


@get("/api/analyze/state")
async def analyze_state_route() -> AnalyzeStateResponse:
    """Whether the project was analyzed, hid the tip, and would see the tip now."""
    return await handlers.analyze_state()


@post("/api/analyze/dismiss", status_code=HTTP_200_OK)
async def analyze_dismiss_route() -> AnalyzeStateResponse:
    """Hide the analyze tip for this project."""
    return await handlers.dismiss_tip()


analyze_router = Router(
    path="/",
    route_handlers=[analyze_route, analyze_state_route, analyze_dismiss_route],
)
