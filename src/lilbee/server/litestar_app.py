"""Litestar adapter — wires framework-agnostic handlers to HTTP routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from litestar import Litestar, get, post, put
from litestar.config.cors import CORSConfig
from litestar.exceptions import ValidationException
from litestar.openapi import OpenAPIConfig
from litestar.params import Parameter
from litestar.response import Stream

from lilbee.cli.helpers import get_version
from lilbee.config import cfg
from lilbee.query import ChatMessage as ChatMessageDict
from lilbee.server import handlers
from lilbee.server.handlers import _sse_generator
from lilbee.server.models import (
    AddRequest,
    AskRequest,
    AskResponse,
    ChatRequest,
    CleanedChunk,
    HealthResponse,
    SetModelRequest,
    SetModelResponse,
    SyncRequest,
)


def _clean_to_model(raw: dict) -> CleanedChunk:
    """Convert a raw cleaned dict to a CleanedChunk model."""
    return CleanedChunk(**raw)


@get("/api/health")
async def health_route() -> HealthResponse:
    """Service health check returning server version and uptime status."""
    raw = await handlers.health()
    return HealthResponse(**raw)


@get("/api/status")
async def status_route() -> dict[str, Any]:
    """Current configuration, indexed document sources, and chunk counts."""
    return await handlers.status()


@get("/api/search")
async def search_route(
    q: str = Parameter(query="q"),
    top_k: int = Parameter(query="top_k", default=5),
) -> list[dict[str, Any]]:
    """Search indexed documents by semantic similarity. No LLM call required."""
    return await handlers.search(q, top_k=top_k)


@post("/api/ask")
async def ask_route(data: AskRequest) -> AskResponse:
    """One-shot RAG question returning an answer with source chunks."""
    raw = await handlers.ask(
        question=data.question,
        top_k=data.top_k,
        options=data.options,
    )
    return AskResponse(
        answer=raw["answer"],
        sources=[_clean_to_model(s) for s in raw["sources"]],
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
    raw = await handlers.chat(
        question=data.question,
        history=history,
        top_k=data.top_k,
        options=data.options,
    )
    return AskResponse(
        answer=raw["answer"],
        sources=[_clean_to_model(s) for s in raw["sources"]],
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


@post("/api/sync")
async def sync_route(data: SyncRequest | None = None) -> Stream:
    """Re-index changed documents with streaming SSE progress events."""
    force_vision = data.force_vision if data else False
    return Stream(
        handlers.sync_stream(force_vision=force_vision),
        media_type="text/event-stream",
    )


@post("/api/add")
async def add_route(data: AddRequest) -> Stream:
    """Add files to the knowledge base with streaming SSE progress."""
    try:
        _paths, queue, task = await handlers.add_files(data.model_dump())
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc

    async def _stream() -> AsyncGenerator[bytes, None]:
        async for chunk in _sse_generator(queue):
            yield chunk
        await task

    return Stream(_stream(), media_type="text/event-stream", status_code=201)


@get("/api/models")
async def models_list_route() -> dict[str, Any]:
    """Available chat and vision models from Ollama."""
    return await handlers.list_models()


@put("/api/models/chat")
async def models_set_chat_route(data: SetModelRequest) -> SetModelResponse:
    """Switch the active chat model used for RAG answers."""
    raw = await handlers.set_chat_model(model=data.model)
    return SetModelResponse(**raw)


@put("/api/models/vision")
async def models_set_vision_route(data: SetModelRequest) -> SetModelResponse:
    """Switch the active vision model used for image and PDF OCR."""
    raw = await handlers.set_vision_model(model=data.model)
    return SetModelResponse(**raw)


def create_app() -> Litestar:
    """Create the Litestar application instance."""
    cors = CORSConfig(
        allow_origins=cfg.cors_origins,
        allow_origin_regex=r"^http://localhost(:\d+)?$",
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type"],
    )
    return Litestar(
        route_handlers=[
            health_route,
            status_route,
            search_route,
            ask_route,
            ask_stream_route,
            chat_route,
            chat_stream_route,
            sync_route,
            add_route,
            models_list_route,
            models_set_chat_route,
            models_set_vision_route,
        ],
        cors_config=cors,
        openapi_config=OpenAPIConfig(
            title="lilbee",
            description="Local knowledge base REST API",
            version=get_version(),
            path="/schema",
        ),
    )
