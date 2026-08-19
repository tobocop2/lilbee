"""HTTP route for the Anthropic-compatible ``/v1/messages``."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator

from litestar import Request, Router, post
from litestar.background_tasks import BackgroundTask
from litestar.exceptions import NotAuthorizedException, ValidationException
from litestar.response import Response, Stream

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.config.enums import ReasoningMode
from lilbee.retrieval.reasoning import effective_reasoning_cap
from lilbee.server.anthropic_api.errors import anthropic_error_body, anthropic_error_type
from lilbee.server.anthropic_api.models import (
    _THINKING_DISABLED,
    AnthropicEventType,
    MessagesRequest,
    MessagesResponse,
)
from lilbee.server.anthropic_api.streaming import encode_anthropic_event, encode_anthropic_sse
from lilbee.server.anthropic_api.translate import (
    canonical_stream_to_anthropic_events,
    canonical_to_messages_response,
    messages_to_canonical_request,
    resolve_reasoning_mode,
)
from lilbee.server.auth import auth_checked_in_handler, session_manager
from lilbee.server.chat_completions_api.errors import (
    CompletionsErrorCode,
    classify_provider_error,
)
from lilbee.server.chat_dispatch.canonical import CanonicalChatRequest
from lilbee.server.chat_dispatch.concurrency import (
    ChatBusyError,
    ChatSlotGuard,
    acquire_chat_slot_or_busy,
)
from lilbee.server.chat_dispatch.dispatch import (
    dispatch_chat_stream,
    preflight_chat_request,
)
from lilbee.server.chat_dispatch.reasoning_cap import (
    budget_capped_chars,
    cap_aware_chat,
    cap_aware_chat_stream,
)
from lilbee.server.handlers.sse import SSE_MEDIA_TYPE
from lilbee.server.validation_format import format_validation

log = logging.getLogger(__name__)

_INTERNAL_ERROR_MESSAGE = "Internal server error. Check the server logs for details."


async def _auth_before_request(request: Request) -> Response | None:
    """Reject an unauthenticated caller before Litestar parses the body.

    Same rationale as the completions surface: the auth answer must ride this
    surface's own error envelope, and it must win over body validation.
    """
    return _auth_failure(request)


@post("/v1/messages", status_code=200, before_request=_auth_before_request)
@auth_checked_in_handler
async def messages_endpoint(request: Request, data: MessagesRequest) -> Response | Stream:
    """``/v1/messages`` (stream + non-stream + tools), Anthropic wire format.

    ``stream: true`` switches the 200 response from JSON to the Anthropic SSE
    event stream; the request body picks the arm, matching Anthropic's own
    contract.
    """
    # Thinking is opt-in per request; the setting only presents it.
    mode = resolve_reasoning_mode(data.thinking, default=cfg.messages_reasoning)
    cap_chars = budget_capped_chars(effective_reasoning_cap(), _budget_tokens(data))
    try:
        req = messages_to_canonical_request(data, mode=mode)
    except ValueError as exc:
        # Wire-valid but untranslatable (image content, bare tool choice).
        return _error_response(400, CompletionsErrorCode.INVALID_REQUEST, str(exc))

    preflight = await _preflight_resolved_model(req)
    if isinstance(preflight, Response):
        return preflight
    resolved_model = preflight

    try:
        await acquire_chat_slot_or_busy(get_services().provider.max_concurrent_chats())
    except ChatBusyError:
        return _error_response(
            429,
            CompletionsErrorCode.RATE_LIMIT_EXCEEDED,
            "Backend is busy. Retry in a moment.",
            headers={"Retry-After": "1"},
        )

    guard = ChatSlotGuard()
    if req.stream:
        # The after-send hook frees the slot when a disconnect lands before the
        # generator's first iteration (its finally never runs in that case).
        return Stream(
            _gated_messages_stream(
                req, guard, model=resolved_model, mode=mode, cap_chars=cap_chars
            ),
            media_type=SSE_MEDIA_TYPE,
            background=BackgroundTask(guard.release),
        )
    return await _run_non_stream(
        req, guard, canonical_model=resolved_model, mode=mode, cap_chars=cap_chars
    )


def _budget_tokens(data: MessagesRequest) -> int | None:
    """The thinking budget this request asks for; ``disabled`` carries none."""
    if data.thinking is None or data.thinking.type == _THINKING_DISABLED:
        return None
    return data.thinking.budget_tokens


async def _preflight_resolved_model(req: CanonicalChatRequest) -> str | Response:
    """Validate *req* before any streaming response starts (see completions)."""
    try:
        return await asyncio.to_thread(preflight_chat_request, req)
    except Exception as exc:
        classified = classify_provider_error(exc)
        if classified is None:
            return _internal_error_response()
        return _error_response(classified.http_status, classified.code, classified.message)


async def _run_non_stream(
    req: CanonicalChatRequest,
    guard: ChatSlotGuard,
    *,
    canonical_model: str,
    mode: ReasoningMode = ReasoningMode.SEPARATE,
    cap_chars: int = 0,
) -> Response:
    """Dispatch a non-streaming chat call, translating errors to the envelope."""
    try:
        resp = await asyncio.to_thread(
            cap_aware_chat, req, canonical_model=canonical_model, cap_chars=cap_chars
        )
    except Exception as exc:
        classified = classify_provider_error(exc)
        if classified is None:
            return _internal_error_response()
        return _error_response(classified.http_status, classified.code, classified.message)
    finally:
        await guard.release()
    body: MessagesResponse = canonical_to_messages_response(
        resp, response_id=_response_id(), mode=mode
    )
    return Response(body.model_dump(), media_type="application/json")


async def _gated_messages_stream(
    req: CanonicalChatRequest,
    guard: ChatSlotGuard,
    *,
    model: str,
    mode: ReasoningMode = ReasoningMode.SEPARATE,
    cap_chars: int = 0,
) -> AsyncGenerator[bytes, None]:
    """Drive dispatch -> translate -> SSE-encode, freeing the slot on exit.

    A mid-stream failure surfaces as Anthropic's ``event: error`` frame; the
    headers are already flushed at 200 by then, so the frame is the only
    channel left.
    """
    response_id = _response_id()
    try:
        try:
            events = cap_aware_chat_stream(
                dispatch_chat_stream(req, canonical_model=model),
                req,
                canonical_model=model,
                cap_chars=cap_chars,
            )
            pairs = canonical_stream_to_anthropic_events(
                events, model=model, response_id=response_id, mode=mode
            )
            async for frame in encode_anthropic_sse(pairs):
                yield frame
        except Exception as exc:
            classified = classify_provider_error(exc)
            if classified is None:
                log.exception("anthropic messages stream failed")
                body = anthropic_error_body("api_error", _INTERNAL_ERROR_MESSAGE)
            else:
                body = anthropic_error_body(
                    anthropic_error_type(classified.code), classified.message
                )
            yield encode_anthropic_event(AnthropicEventType.ERROR, body)
    finally:
        await guard.release()


def _internal_error_response() -> Response:
    """Log and return the generic api_error 500 envelope."""
    log.exception("messages_endpoint failed")
    return _error_response(500, CompletionsErrorCode.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE)


def _error_response(
    status: int,
    code: CompletionsErrorCode,
    message: str,
    *,
    headers: dict[str, str] | None = None,
) -> Response:
    return Response(
        anthropic_error_body(anthropic_error_type(code), message),
        status_code=status,
        headers=headers or {},
        media_type="application/json",
    )


def _auth_failure(request: Request) -> Response | None:
    """Return a 401 envelope if the bearer token is missing/wrong, else None.

    Claude Code sends ``ANTHROPIC_AUTH_TOKEN`` as a bearer Authorization
    header, which is exactly the session token check every /v1 route runs.
    """
    auth_header = request.headers.get("authorization", "")
    try:
        authorized = session_manager.validate(auth_header)
    except NotAuthorizedException:
        authorized = False
    if authorized:
        return None
    return _error_response(401, CompletionsErrorCode.INVALID_API_KEY, "Missing or invalid API key.")


def _validation_exception_handler(_: Request, exc: ValidationException) -> Response:
    """Wrap Litestar's body-parse failures in the Anthropic error envelope."""
    return _error_response(400, CompletionsErrorCode.INVALID_REQUEST, format_validation(exc))


def _response_id() -> str:
    """Anthropic-style ``msg_*`` id."""
    return f"msg_{uuid.uuid4().hex[:24]}"


anthropic_router = Router(
    path="/",
    route_handlers=[messages_endpoint],
    exception_handlers={ValidationException: _validation_exception_handler},
)
