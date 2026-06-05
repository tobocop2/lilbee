"""Search, ask, ask_stream, chat, and chat_stream route handlers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from litestar import get, post
from litestar.exceptions import HTTPException, ValidationException
from litestar.params import Parameter
from litestar.response import Stream

from lilbee.core.results import DocumentResult
from lilbee.data.store import EmbeddingModelMismatchError, scope_to_chunk_type
from lilbee.retrieval.query import ChatMessage as ChatMessageDict
from lilbee.server import handlers
from lilbee.server.auth import read_only
from lilbee.server.models import (
    AskRequest,
    AskResponse,
    ChatRequest,
)

# Process-wide lock that gates the two streaming chat endpoints to one
# in-flight request at a time. The llama-cpp provider already serializes
# concurrent chat() calls under a thread lock, so a second concurrent
# stream blocks the client for many seconds with no feedback. Returning
# 429 + Retry-After fast lets clients surface a real error and decide.
# The lock binds to the worker's running event loop on first acquire.
_chat_inflight_lock = asyncio.Lock()


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


def _acquire_chat_lock_or_raise() -> None:
    """Non-blocking acquire on the running loop thread; raise 429 on contention.

    Race-free because route handlers run on a single event loop thread and
    ``Lock.acquire()`` on a free lock returns synchronously without yielding.
    The check + acquire is atomic from the loop's perspective, no ``await``
    can intervene between the two calls.
    """
    if _chat_inflight_lock.locked():
        raise HTTPException(status_code=429, headers={"Retry-After": "1"})


async def _gated_stream(
    generator: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Wrap *generator* so the chat lock is released when the stream ends.

    The lock must already be held when this is called. Release happens on
    natural completion, exception, and client-disconnect (GeneratorExit
    fires the ``finally`` block).
    """
    try:
        async for chunk in generator:
            yield chunk
    finally:
        _chat_inflight_lock.release()


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
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@post("/api/ask/stream")
async def ask_stream_route(data: AskRequest) -> Stream:
    """Streaming SSE version of ask, emitting token-by-token answer chunks."""
    _acquire_chat_lock_or_raise()
    await _chat_inflight_lock.acquire()
    return Stream(
        _gated_stream(
            handlers.ask_stream(
                question=data.question,
                top_k=data.top_k,
                options=data.options,
                chunk_type=data.chunk_type,
            ),
        ),
        media_type="text/event-stream",
    )


@post("/api/chat")
async def chat_route(data: ChatRequest) -> AskResponse:
    """RAG chat with conversation history, returning an answer with sources."""
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


@post("/api/chat/stream")
async def chat_stream_route(data: ChatRequest) -> Stream:
    """Streaming SSE version of chat with conversation history."""
    _acquire_chat_lock_or_raise()
    await _chat_inflight_lock.acquire()
    history: list[ChatMessageDict] = [
        ChatMessageDict(role=m.role, content=m.content) for m in data.history
    ]
    return Stream(
        _gated_stream(
            handlers.chat_stream(
                question=data.question,
                history=history,
                top_k=data.top_k,
                options=data.options,
                chunk_type=data.chunk_type,
            ),
        ),
        media_type="text/event-stream",
    )
