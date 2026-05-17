"""Anthropic-compatible ``POST /v1/messages`` route and router."""

from __future__ import annotations

import hmac
import logging
from collections.abc import AsyncGenerator
from typing import Any

from litestar import Request, Response, Router, post
from litestar.enums import MediaType
from litestar.response import Stream

from lilbee.server.auth import read_only, session_manager
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalStreamEvent,
)
from lilbee.server.chat_dispatch.concurrency import (
    ChatBusyError,
    acquire_or_raise_busy,
    chat_lock,
)
from lilbee.server.chat_dispatch.dispatch import (
    ModelDoesNotSupportToolsError,
    ModelNotFoundError,
    dispatch_chat,
    dispatch_chat_stream,
)
from lilbee.server.messages_api.errors import (
    MessagesErrorType,
    messages_error_body,
    status_for_error_type,
)
from lilbee.server.messages_api.streaming import encode_messages_sse
from lilbee.server.messages_api.translate import (
    canonical_stream_to_messages_events,
    canonical_to_messages_response,
    messages_to_canonical_request,
)

log = logging.getLogger(__name__)


def _error_response(error_type: MessagesErrorType, message: str) -> Response:
    return Response(
        content=messages_error_body(error_type, message),
        status_code=status_for_error_type(error_type),
        media_type=MediaType.JSON,
    )


def _authenticate(request: Request) -> bool:
    """Accept Anthropic ``x-api-key`` or OpenAI ``Authorization: Bearer`` header."""
    if session_manager.token is None:
        return True  # auth disabled (tests / dev)
    api_key = request.headers.get("x-api-key", "")
    if api_key and hmac.compare_digest(api_key, session_manager.token):
        return True
    bearer = request.headers.get("authorization", "")
    return bool(bearer) and hmac.compare_digest(bearer, f"Bearer {session_manager.token}")


async def _gated_byte_stream(
    byte_stream: AsyncGenerator[bytes, None],
) -> AsyncGenerator[bytes, None]:
    """Wrap *byte_stream* so the chat lock releases on completion or disconnect."""
    try:
        async for chunk in byte_stream:
            yield chunk
    finally:
        chat_lock().release()


@post("/v1/messages", status_code=200)
@read_only
async def messages_route(request: Request, data: dict[str, Any]) -> Response | Stream:
    """Anthropic-compatible chat completion (streaming + non-streaming + tools)."""
    if not _authenticate(request):
        return _error_response(MessagesErrorType.AUTHENTICATION, "Missing or invalid API key")

    try:
        canonical_req = messages_to_canonical_request(data)
    except ValueError as exc:
        return _error_response(MessagesErrorType.INVALID_REQUEST, str(exc))

    if canonical_req.stream:
        return await _start_stream(canonical_req)
    return await _run_non_stream(canonical_req)


def _busy_response() -> Response:
    return Response(
        content=messages_error_body(
            MessagesErrorType.OVERLOADED, "Chat backend is busy. Retry shortly."
        ),
        status_code=429,
        media_type=MediaType.JSON,
        headers={"Retry-After": "1"},
    )


async def _run_non_stream(canonical_req: CanonicalChatRequest) -> Response:
    try:
        acquire_or_raise_busy()
    except ChatBusyError:
        return _busy_response()
    await chat_lock().acquire()
    try:
        try:
            resp = dispatch_chat(canonical_req)
        except ModelNotFoundError as exc:
            return _error_response(MessagesErrorType.NOT_FOUND, f"Model {exc.model!r} not found")
        except ModelDoesNotSupportToolsError as exc:
            return _error_response(
                MessagesErrorType.INVALID_REQUEST,
                f"Model {exc.model!r} does not support tool calls",
            )
        return Response(
            content=canonical_to_messages_response(resp),
            status_code=200,
            media_type=MediaType.JSON,
        )
    finally:
        chat_lock().release()


async def _start_stream(canonical_req: CanonicalChatRequest) -> Response | Stream:
    """Validate, acquire lock, and open the SSE stream.

    Validation runs by stepping the dispatch generator once so a bad
    model or unsupported tools yield a JSON 4xx instead of a half-open
    event stream. The probed first event is then replayed into the
    encoder so no frames are lost.
    """
    try:
        acquire_or_raise_busy()
    except ChatBusyError:
        return _busy_response()
    await chat_lock().acquire()

    canonical_stream = dispatch_chat_stream(canonical_req)
    try:
        first_event = await canonical_stream.__anext__()
    except ModelNotFoundError as exc:
        chat_lock().release()
        return _error_response(MessagesErrorType.NOT_FOUND, f"Model {exc.model!r} not found")
    except ModelDoesNotSupportToolsError as exc:
        chat_lock().release()
        return _error_response(
            MessagesErrorType.INVALID_REQUEST,
            f"Model {exc.model!r} does not support tool calls",
        )

    async def replay() -> AsyncGenerator[CanonicalStreamEvent, None]:
        yield first_event
        async for ev in canonical_stream:
            yield ev

    async def sse_bytes() -> AsyncGenerator[bytes, None]:
        async for chunk in encode_messages_sse(canonical_stream_to_messages_events(replay())):
            yield chunk

    return Stream(_gated_byte_stream(sse_bytes()), media_type="text/event-stream")


messages_router = Router(path="", route_handlers=[messages_route])
