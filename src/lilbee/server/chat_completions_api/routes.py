"""HTTP routes for ``/v1/models`` and ``/v1/chat/completions``."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime

from litestar import Request, Router, get, post
from litestar.exceptions import ValidationException
from litestar.response import Response, Stream

from lilbee.app.services import get_services
from lilbee.catalog.types import ModelTask
from lilbee.server.auth import read_only, session_manager
from lilbee.server.chat_completions_api.errors import (
    CompletionsErrorCode,
    completions_error_body,
)
from lilbee.server.chat_completions_api.models import (
    CompletionsRequest,
    CompletionsResponse,
    ModelEntry,
    ModelsListResponse,
)
from lilbee.server.chat_completions_api.streaming import encode_completions_sse
from lilbee.server.chat_completions_api.translate import (
    canonical_stream_to_completions_chunks,
    canonical_to_completions_response,
    completions_to_canonical_request,
)
from lilbee.server.chat_dispatch.canonical import CanonicalChatRequest
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

log = logging.getLogger(__name__)


@get("/v1/models")
@read_only
async def list_models_endpoint(request: Request) -> Response:
    """Return all installed chat models in the ``/v1/models`` shape."""
    auth_error = _auth_failure(request)
    if auth_error is not None:
        return auth_error

    registry = get_services().registry
    chat_models = [m for m in registry.list_installed() if m.task == ModelTask.CHAT]
    payload = ModelsListResponse(
        data=[
            ModelEntry(id=m.ref, created=_parse_created(m.downloaded_at))
            for m in sorted(chat_models, key=lambda m: m.ref)
        ]
    )
    return Response(payload.model_dump(), media_type="application/json")


@post("/v1/chat/completions", status_code=200)
@read_only
async def chat_completions_endpoint(
    request: Request, data: CompletionsRequest
) -> Response | Stream:
    """``/v1/chat/completions`` (stream + non-stream + tools)."""
    auth_error = _auth_failure(request)
    if auth_error is not None:
        return auth_error

    try:
        req = completions_to_canonical_request(data)
    except ValueError as exc:
        # Request shape is wire-valid but carries something we can't translate
        # (e.g. image content). Surface as 400 instead of a generic 500.
        return _error_response(400, CompletionsErrorCode.INVALID_REQUEST, str(exc))

    try:
        acquire_or_raise_busy()
    except ChatBusyError:
        return _error_response(
            429,
            CompletionsErrorCode.RATE_LIMIT_EXCEEDED,
            "Backend is busy. Retry in a moment.",
            headers={"Retry-After": "1"},
        )

    lock = chat_lock()
    await lock.acquire()

    if req.stream:
        return Stream(
            _gated_completions_stream(req, lock),
            media_type="text/event-stream",
        )
    return await _run_non_stream(req, lock)


async def _run_non_stream(req: CanonicalChatRequest, lock: asyncio.Lock) -> Response:
    """Dispatch a non-streaming chat call, translating errors to the wire envelope."""
    try:
        resp = dispatch_chat(req)
    except ModelNotFoundError as exc:
        return _error_response(404, CompletionsErrorCode.MODEL_NOT_FOUND, str(exc))
    except ModelDoesNotSupportToolsError as exc:
        return _error_response(400, CompletionsErrorCode.MODEL_DOES_NOT_SUPPORT_TOOLS, str(exc))
    except Exception:
        log.exception("chat_completions_endpoint failed")
        return _error_response(
            500,
            CompletionsErrorCode.INTERNAL_ERROR,
            "Internal server error. Check the server logs for details.",
        )
    finally:
        lock.release()
    body: CompletionsResponse = canonical_to_completions_response(resp, response_id=_response_id())
    return Response(body.model_dump(exclude_none=True), media_type="application/json")


async def _gated_completions_stream(
    req: CanonicalChatRequest, lock: asyncio.Lock
) -> AsyncGenerator[bytes, None]:
    """Drive ``dispatch_chat_stream`` -> translate -> SSE-encode, releasing *lock* on exit.

    Pre-flight errors (unknown model, tools-against-non-tool-model) are
    surfaced as a single SSE ``data:`` frame carrying the error
    envelope, then ``[DONE]``. The lock is released in ``finally`` so
    natural completion, exception, and client disconnect (GeneratorExit)
    all unwind cleanly.
    """
    try:
        try:
            events = dispatch_chat_stream(req)
            chunks = canonical_stream_to_completions_chunks(
                events, model=req.model, response_id=_response_id()
            )
            async for frame in encode_completions_sse(chunks):
                yield frame
        except ModelNotFoundError as exc:
            yield _sse_error_frame(CompletionsErrorCode.MODEL_NOT_FOUND, str(exc))
        except ModelDoesNotSupportToolsError as exc:
            yield _sse_error_frame(CompletionsErrorCode.MODEL_DOES_NOT_SUPPORT_TOOLS, str(exc))
        except Exception:
            log.exception("chat_completions stream failed")
            yield _sse_error_frame(
                CompletionsErrorCode.INTERNAL_ERROR,
                "Internal server error. Check the server logs for details.",
            )
    finally:
        lock.release()


def _sse_error_frame(code: CompletionsErrorCode, message: str) -> bytes:
    """SSE frame carrying an error body, followed by ``[DONE]``."""
    body = completions_error_body(code, message)
    payload = json.dumps(body, separators=(",", ":"))
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode()


def _error_response(
    status: int,
    code: CompletionsErrorCode,
    message: str,
    *,
    headers: dict[str, str] | None = None,
) -> Response:
    return Response(
        completions_error_body(code, message),
        status_code=status,
        headers=headers or {},
        media_type="application/json",
    )


def _auth_failure(request: Request) -> Response | None:
    """Return a 401 error response if the bearer token is missing/wrong, else None."""
    auth_header = request.headers.get("authorization", "")
    if session_manager.validate(auth_header):
        return None
    return _error_response(401, CompletionsErrorCode.INVALID_API_KEY, "Missing or invalid API key.")


def _validation_exception_handler(_: Request, exc: ValidationException) -> Response:
    """Wrap Litestar's body-parse failures in the OpenAI error envelope."""
    return _error_response(400, CompletionsErrorCode.INVALID_REQUEST, _format_validation(exc))


def _format_validation(exc: ValidationException) -> str:
    """Render a litestar/pydantic ValidationException as a single user-facing string.

    Litestar wraps pydantic errors as ``{"key": "field_name", "message": "..."}``
    entries on ``exc.extra``; we flatten them into a semicolon-joined string so
    the OpenAI envelope carries the same field names a client expects to see.
    """
    items: list[dict[str, str]] = exc.extra if isinstance(exc.extra, list) else []
    parts = [f"{err.get('key') or ''}: {err.get('message', '')}".lstrip(": ") for err in items]
    return "; ".join(parts) if parts else str(exc.detail)


def _parse_created(downloaded_at: str | None) -> int:
    """Best-effort ISO-8601 to Unix-timestamp conversion; zero on failure."""
    if not downloaded_at:
        return 0
    try:
        return int(datetime.fromisoformat(downloaded_at).timestamp())
    except ValueError:
        return 0


def _response_id() -> str:
    """OpenAI-style ``chatcmpl-*`` id."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


completions_router = Router(
    path="/",
    route_handlers=[list_models_endpoint, chat_completions_endpoint],
    exception_handlers={ValidationException: _validation_exception_handler},
)
