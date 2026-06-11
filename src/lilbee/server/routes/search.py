"""Search, ask, ask_stream, chat, and chat_stream route handlers."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import NoReturn

from litestar import get, post
from litestar.background_tasks import BackgroundTask
from litestar.exceptions import HTTPException, ValidationException
from litestar.params import Parameter
from litestar.response import Stream

from lilbee.core.results import DocumentResult
from lilbee.data.store import EmbeddingModelMismatchError, scope_to_chunk_type
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.retrieval.query import ChatMessage as ChatMessageDict
from lilbee.server import handlers
from lilbee.server.auth import read_only
from lilbee.server.chat_dispatch.concurrency import (
    ChatBusyError,
    ChatSlotGuard,
    acquire_chat_slot_or_busy,
    release_chat_slot,
)
from lilbee.server.chat_dispatch.dispatch import (
    ModelDoesNotSupportToolsError,
    ModelNotFoundError,
)
from lilbee.server.handlers.sse import sse_error
from lilbee.server.models import (
    AskRequest,
    AskResponse,
    ChatRequest,
)

_BAD_REQUEST_STATUS = 400
_NOT_FOUND_STATUS = 404
_SERVICE_UNAVAILABLE_STATUS = 503

# Shipped clients read /api 401/429 as lilbee-session signals, so upstream kinds stay 503.
_API_PROVIDER_KIND_STATUSES: dict[ProviderErrorKind, int] = {
    ProviderErrorKind.CONTEXT_OVERFLOW: _BAD_REQUEST_STATUS,
    ProviderErrorKind.NOT_FOUND: _NOT_FOUND_STATUS,
}

log = logging.getLogger(__name__)


def _embedding_mismatch_http(exc: EmbeddingModelMismatchError) -> HTTPException:
    """Translate an embedder mismatch into a 409 carrying the facts to adopt.

    The client renders its own confirm-to-adopt prompt from ``extra`` and, on
    confirm, sets the embedder via ``PUT /api/models/embedding`` then retries.
    The server never switches embedder unprompted.
    """
    return HTTPException(
        status_code=409,
        detail=str(exc),
        extra={
            "persisted_model": exc.persisted_model,
            "persisted_dim": exc.persisted_dim,
            "current_model": exc.current_model,
            "adoptable": exc.dims_match,
        },
    )


def _raise_chat_http_error(exc: Exception) -> NoReturn:
    """Translate a chat/RAG failure into the Litestar HTTP envelope.

    ValueError is a 422 validation error; a typed dispatch error or a
    kind-mapped ProviderError carries its own status; anything else is a
    503 carrying the failure message.
    """
    if isinstance(exc, ValueError):
        raise ValidationException(str(exc)) from exc
    raise HTTPException(status_code=_api_chat_error_status(exc), detail=str(exc)) from exc


def _api_chat_error_status(exc: Exception) -> int:
    """HTTP status for a non-stream /api chat failure; unmapped kinds stay 503."""
    if isinstance(exc, ModelNotFoundError):
        return _NOT_FOUND_STATUS
    if isinstance(exc, ModelDoesNotSupportToolsError):
        return _BAD_REQUEST_STATUS
    if isinstance(exc, ProviderError):
        return _API_PROVIDER_KIND_STATUSES.get(exc.kind, _SERVICE_UNAVAILABLE_STATUS)
    return _SERVICE_UNAVAILABLE_STATUS


async def _acquire_chat_lock_or_raise() -> None:
    """Translate the canonical busy signal into Litestar's HTTP 429 envelope."""
    from lilbee.app.services import get_services

    try:
        await acquire_chat_slot_or_busy(get_services().provider.max_concurrent_chats())
    except ChatBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "1"}) from exc


async def _gated_stream(
    generator: AsyncGenerator[str, None],
    guard: ChatSlotGuard,
) -> AsyncGenerator[str, None]:
    """Wrap *generator* so the chat lock is released when the stream ends.

    The lock must already be held when this is called. Release happens on
    natural completion, exception, and client-disconnect (GeneratorExit
    fires the ``finally`` block); a disconnect before the first iteration
    never enters this body, so the route also releases *guard* from the
    response's after-send hook. A failure inside the generator becomes an
    SSE error event; raising after the 201 headers would drop the connection
    with no body for the client to read.
    """
    try:
        async for chunk in generator:
            yield chunk
    except Exception as exc:
        log.exception("streaming chat handler failed")
        yield sse_error(str(exc))
    finally:
        await guard.release()


def _slot_gated_sse(generator: AsyncGenerator[str, None], guard: ChatSlotGuard) -> Stream:
    """SSE Stream whose chat slot is freed by the generator or the after-send hook."""
    return Stream(
        _gated_stream(generator, guard),
        media_type="text/event-stream",
        background=BackgroundTask(guard.release),
    )


@get("/api/search")
@read_only
async def search_route(
    q: str = Parameter(query="q"),
    top_k: int = Parameter(query="top_k", default=5, le=100),
    chunk_type: str | None = Parameter(query="chunk_type", default=None),
) -> list[DocumentResult]:
    """Search indexed documents by semantic similarity. No LLM call required."""
    try:
        chunk_type = scope_to_chunk_type(chunk_type)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    try:
        return await handlers.search(q, top_k=top_k, chunk_type=chunk_type)
    except EmbeddingModelMismatchError as exc:
        raise _embedding_mismatch_http(exc) from exc
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@post("/api/ask")
async def ask_route(data: AskRequest) -> AskResponse:
    """One-shot RAG question returning an answer with source chunks."""
    await _acquire_chat_lock_or_raise()
    try:
        return await handlers.ask(
            question=data.question,
            top_k=data.top_k,
            options=data.options,
            chunk_type=data.chunk_type,
        )
    except EmbeddingModelMismatchError as exc:
        raise _embedding_mismatch_http(exc) from exc
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    except Exception as exc:
        _raise_chat_http_error(exc)
    finally:
        await release_chat_slot()


@post("/api/ask/stream")
async def ask_stream_route(data: AskRequest) -> Stream:
    """Streaming SSE version of ask, emitting token-by-token answer chunks."""
    await _acquire_chat_lock_or_raise()
    return _slot_gated_sse(
        handlers.ask_stream(
            question=data.question,
            top_k=data.top_k,
            options=data.options,
            chunk_type=data.chunk_type,
        ),
        ChatSlotGuard(),
    )


@post("/api/chat")
async def chat_route(data: ChatRequest) -> AskResponse:
    """RAG chat with conversation history, returning an answer with sources."""
    await _acquire_chat_lock_or_raise()
    history: list[ChatMessageDict] = [
        ChatMessageDict(role=m.role, content=m.content) for m in data.history
    ]
    try:
        return await handlers.chat(
            question=data.question,
            history=history,
            top_k=data.top_k,
            options=data.options,
            chunk_type=data.chunk_type,
        )
    except EmbeddingModelMismatchError as exc:
        raise _embedding_mismatch_http(exc) from exc
    except Exception as exc:
        _raise_chat_http_error(exc)
    finally:
        await release_chat_slot()


@post("/api/chat/stream")
async def chat_stream_route(data: ChatRequest) -> Stream:
    """Streaming SSE version of chat with conversation history."""
    await _acquire_chat_lock_or_raise()
    history: list[ChatMessageDict] = [
        ChatMessageDict(role=m.role, content=m.content) for m in data.history
    ]
    return _slot_gated_sse(
        handlers.chat_stream(
            question=data.question,
            history=history,
            top_k=data.top_k,
            options=data.options,
            chunk_type=data.chunk_type,
        ),
        ChatSlotGuard(),
    )
