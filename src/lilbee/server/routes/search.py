"""Search, ask, ask_stream, chat, and chat_stream route handlers."""

from __future__ import annotations

from litestar import get, post
from litestar.params import Parameter
from litestar.response import Stream

from lilbee.query import ChatMessage as ChatMessageDict
from lilbee.results import DocumentResult
from lilbee.server import handlers
from lilbee.server.auth import read_only
from lilbee.server.models import (
    AskRequest,
    AskResponse,
    ChatRequest,
)


@get("/api/search")
@read_only
async def search_route(
    q: str = Parameter(query="q"),
    top_k: int = Parameter(query="top_k", default=5, le=100),
) -> list[DocumentResult]:
    """Search indexed documents by semantic similarity. No LLM call required."""
    return await handlers.search(q, top_k=top_k)


@post("/api/ask")
async def ask_route(data: AskRequest) -> AskResponse:
    """One-shot RAG question returning an answer with source chunks."""
    return await handlers.ask(
        question=data.question,
        top_k=data.top_k,
        options=data.options,
    )


@post("/api/ask/stream")
async def ask_stream_route(data: AskRequest) -> Stream:
    """Streaming SSE version of ask, emitting token-by-token answer chunks."""
    return Stream(
        handlers.ask_stream(
            question=data.question,
            top_k=data.top_k,
            options=data.options,
        ),
        media_type="text/event-stream",
    )


@post("/api/chat")
async def chat_route(data: ChatRequest) -> AskResponse:
    """RAG chat with conversation history, returning an answer with sources."""
    history: list[ChatMessageDict] = [
        ChatMessageDict(role=m.role, content=m.content) for m in data.history
    ]
    return await handlers.chat(
        question=data.question,
        history=history,
        top_k=data.top_k,
        options=data.options,
    )


@post("/api/chat/stream")
async def chat_stream_route(data: ChatRequest) -> Stream:
    """Streaming SSE version of chat with conversation history."""
    history: list[ChatMessageDict] = [
        ChatMessageDict(role=m.role, content=m.content) for m in data.history
    ]
    return Stream(
        handlers.chat_stream(
            question=data.question,
            history=history,
            top_k=data.top_k,
            options=data.options,
        ),
        media_type="text/event-stream",
    )
