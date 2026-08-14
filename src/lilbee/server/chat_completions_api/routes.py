"""HTTP routes for ``/v1/models`` and ``/v1/chat/completions``."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime

from litestar import Request, Router, get, post
from litestar.background_tasks import BackgroundTask
from litestar.exceptions import NotAuthorizedException, ValidationException
from litestar.response import Response, Stream

from lilbee.app.services import get_services
from lilbee.catalog.types import ModelTask
from lilbee.core.config import cfg
from lilbee.core.config.enums import ReasoningMode
from lilbee.providers.model_ref import default_first, with_configured_remote_chat
from lilbee.server.auth import auth_checked_in_handler, session_manager
from lilbee.server.chat_completions_api.errors import (
    CompletionsErrorCode,
    classify_provider_error,
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
    ChatSlotGuard,
    acquire_chat_slot_or_busy,
)
from lilbee.server.chat_dispatch.dispatch import (
    dispatch_chat,
    dispatch_chat_stream,
    preflight_chat_request,
)
from lilbee.server.handlers.sse import SSE_MEDIA_TYPE
from lilbee.server.validation_format import format_validation

log = logging.getLogger(__name__)

_MULTI_CHOICE_MESSAGE = "lilbee serves one choice per request; set n to 1 or omit it."


@get("/v1/models")
@auth_checked_in_handler
async def list_models_endpoint(request: Request) -> Response:
    """Return installed chat models in the ``/v1/models`` shape, the configured one leading."""
    auth_error = _auth_failure(request)
    if auth_error is not None:
        return auth_error

    # list_installed walks the model filesystem and served_chat_ctx may probe the
    # engine; run both off the event loop like the sibling chat-completions route.
    payload = await asyncio.to_thread(_build_models_list_payload)
    # exclude_none like the sibling completions responses: context_window is
    # deliberately None for every model but the active one, and emitting it as
    # null adds a field OpenAI clients do not expect on a model entry.
    return Response(payload.model_dump(exclude_none=True), media_type="application/json")


def _build_models_list_payload() -> ModelsListResponse:
    """Synchronous /v1/models body: blocking registry walk + engine ctx probe."""
    services = get_services()
    # The served window applies to the active chat model; advertise it so a
    # client trims history to fit instead of overflowing on a long session.
    served_ctx = services.provider.served_chat_ctx()
    installed = {m.ref: m for m in services.registry.list_installed() if m.task == ModelTask.CHAT}
    # A remote-configured chat model has no registry entry but is still listed,
    # configured model first (the launcher's picker order).
    listed = with_configured_remote_chat(sorted(installed), cfg.chat_model)
    refs = default_first(listed, cfg.chat_model)
    # A ref without a registry entry carries the newest native timestamp so a
    # client sorting by created desc does not bury the model lilbee serves.
    fallback_created = max((_parse_created(m.downloaded_at) for m in installed.values()), default=0)
    return ModelsListResponse(
        data=[
            ModelEntry(
                id=ref,
                created=_parse_created(installed[ref].downloaded_at)
                if ref in installed
                else fallback_created,
                context_window=served_ctx if ref == cfg.chat_model else None,
            )
            for ref in refs
        ]
    )


async def _auth_before_request(request: Request) -> Response | None:
    """Reject an unauthenticated caller before Litestar parses the body.

    The endpoint is marked @auth_checked_in_handler so AuthMiddleware defers to
    the handler, which answers a bad token with the OpenAI error envelope
    rather than Litestar's 401 shape. But binding the body as a handler
    parameter made Litestar parse and pydantic-validate the whole payload
    first, so an unauthenticated caller got request_max_body_size worth of
    JSON processed on every request, and a malformed body was answered with a
    400 naming the failing fields instead of ever reaching the 401. Litestar
    runs before_request ahead of kwargs resolution and short-circuits on a
    returned value, so this is the one place the check can sit.
    """
    return _auth_failure(request)


@post("/v1/chat/completions", status_code=200, before_request=_auth_before_request)
@auth_checked_in_handler
async def chat_completions_endpoint(
    request: Request, data: CompletionsRequest
) -> Response | Stream:
    """``/v1/chat/completions`` (stream + non-stream + tools).

    ``stream: true`` switches the 200 response from JSON to an SSE stream of
    chat.completion.chunk frames. The request body picks the arm, which OpenAPI
    cannot express, so the schema declares the JSON default and this contract
    matches OpenAI's own.
    """
    rejection = _reject_before_dispatch(request, data)
    if rejection is not None:
        return rejection

    # The request field wins over the completions_reasoning setting, so a
    # client that can send extra body fields picks its mode per call.
    mode = data.reasoning if data.reasoning is not None else cfg.completions_reasoning
    try:
        req = completions_to_canonical_request(data, mode=mode)
    except ValueError as exc:
        # Request shape is wire-valid but carries something we can't translate
        # (e.g. image content). Surface as 400 instead of a generic 500.
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
        include_usage = bool(data.stream_options and data.stream_options.include_usage)
        # The after-send hook frees the slot when a disconnect lands before the
        # generator's first iteration (its finally never runs in that case).
        return Stream(
            _gated_completions_stream(
                req, guard, model=resolved_model, include_usage=include_usage, mode=mode
            ),
            media_type=SSE_MEDIA_TYPE,
            background=BackgroundTask(guard.release),
        )
    return await _run_non_stream(req, guard, canonical_model=resolved_model, mode=mode)


def _reject_before_dispatch(request: Request, data: CompletionsRequest) -> Response | None:
    """Multi-choice and unsupported-param checks before any dispatch work.

    Returns a 4xx Response to short-circuit, or None to proceed. Unmapped OpenAI
    params (``response_format``, ``logprobs``, and any other unknown field) are
    accepted but logged at debug so a client learns they had no effect. Auth is
    not checked here: it runs in the before_request hook, ahead of body parsing.
    """
    if data.n is not None and data.n > 1:
        return _error_response(400, CompletionsErrorCode.INVALID_REQUEST, _MULTI_CHOICE_MESSAGE)
    if data.model_extra:
        log.debug("chat/completions ignoring unsupported params: %s", sorted(data.model_extra))
    return None


async def _preflight_resolved_model(req: CanonicalChatRequest) -> str | Response:
    """Validate *req* before any streaming response starts, returning the resolved model.

    A 4xx body here is reachable by any OpenAI-compatible client; once a
    Stream is returned the headers are flushed at 200 and downstream errors
    can only travel via SSE frames which not every client surfaces cleanly.
    Runs the preflight in a thread: a lapsed model-discovery TTL makes it do
    blocking HTTP probes that must not stall the event loop. Returns the
    canonical resolved model string when *req* is fit to dispatch, so the
    streaming response echoes the same model the non-streaming path does.
    """
    try:
        return await asyncio.to_thread(preflight_chat_request, req)
    except Exception as exc:
        classified = classify_provider_error(exc)
        if classified is None:
            # Mirror _run_non_stream: an unclassified failure still rides the
            # OpenAI error envelope, not a bare framework 500.
            return _internal_error_response()
        return _error_response(classified.http_status, classified.code, classified.message)


_INTERNAL_ERROR_MESSAGE = "Internal server error. Check the server logs for details."


def _internal_error_response() -> Response:
    """Log and return the generic internal_error 500 envelope."""
    log.exception("chat_completions_endpoint failed")
    return _error_response(500, CompletionsErrorCode.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE)


async def _run_non_stream(
    req: CanonicalChatRequest,
    guard: ChatSlotGuard,
    *,
    canonical_model: str,
    mode: ReasoningMode = ReasoningMode.SEPARATE,
) -> Response:
    """Dispatch a non-streaming chat call, translating errors to the wire envelope."""
    try:
        # dispatch_chat blocks for the whole generation; run it off the event loop
        # so a slow chat does not stall other admitted requests. The preflight
        # already resolved the model, so hand it in to avoid re-running it.
        resp = await asyncio.to_thread(dispatch_chat, req, canonical_model=canonical_model)
    except Exception as exc:
        classified = classify_provider_error(exc)
        if classified is None:
            return _internal_error_response()
        return _error_response(classified.http_status, classified.code, classified.message)
    finally:
        await guard.release()
    body: CompletionsResponse = canonical_to_completions_response(
        resp, response_id=_response_id(), mode=mode
    )
    return Response(body.model_dump(exclude_none=True), media_type="application/json")


async def _gated_completions_stream(
    req: CanonicalChatRequest,
    guard: ChatSlotGuard,
    *,
    model: str,
    include_usage: bool = False,
    mode: ReasoningMode = ReasoningMode.SEPARATE,
) -> AsyncGenerator[bytes, None]:
    """Drive ``dispatch_chat_stream`` -> translate -> SSE-encode, freeing the slot on exit.

    Pre-flight errors (unknown model, tools-against-non-tool-model) are
    surfaced as a single SSE ``data:`` frame carrying the error
    envelope, then ``[DONE]``. The chat slot is released in ``finally`` so
    natural completion, exception, and client disconnect (GeneratorExit)
    all unwind cleanly; a disconnect before the first iteration is covered
    by the route's after-send release of the same guard.
    """
    # One id for the whole stream, error frame included: the frame is shaped as a
    # real chunk so SDK clients parse it, and those accumulate by chunk id, so a
    # fresh id made the error look like part of a different completion.
    response_id = _response_id()
    try:
        try:
            events = dispatch_chat_stream(req, canonical_model=model)
            chunks = canonical_stream_to_completions_chunks(
                events,
                model=model,
                response_id=response_id,
                include_usage=include_usage,
                mode=mode,
            )
            async for frame in encode_completions_sse(chunks):
                yield frame
        except Exception as exc:
            classified = classify_provider_error(exc)
            if classified is None:
                log.exception("chat_completions stream failed")
                yield _sse_error_frame(
                    CompletionsErrorCode.INTERNAL_ERROR,
                    _INTERNAL_ERROR_MESSAGE,
                    model=model,
                    response_id=response_id,
                )
            else:
                yield _sse_error_frame(
                    classified.code, classified.message, model=model, response_id=response_id
                )
    finally:
        await guard.release()


def _sse_error_frame(
    code: CompletionsErrorCode,
    message: str,
    *,
    model: str = "",
    response_id: str | None = None,
) -> bytes:
    """SSE frame carrying a mid-stream error in OpenAI's chunk-shaped wire format.

    OpenAI-SDK clients only parse ``chat.completion.chunk``-shaped frames, so the
    error rides a real chunk (empty delta, inline ``error`` field) followed by
    ``[DONE]`` rather than a bare error frame. ``finish_reason`` stays null, as
    in OpenAI's non-final chunks: a concrete reason like ``"length"`` would tell
    clients the answer was merely truncated, and some auto-continue on it.
    """
    body = completions_error_body(code, message)
    chunk: dict[str, object] = {
        "id": response_id or _response_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "error": body["error"],
    }
    payload = json.dumps(chunk, separators=(",", ":"))
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
    """Return a 401 error response if the bearer token is missing/wrong, else None.

    validate() fails closed by raising when auth is uninitialized; surface that
    as the same OpenAI 401 envelope rather than letting it escape as a 500.
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
    """Wrap Litestar's body-parse failures in the OpenAI error envelope."""
    return _error_response(400, CompletionsErrorCode.INVALID_REQUEST, format_validation(exc))


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
