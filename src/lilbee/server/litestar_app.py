"""Litestar adapter — wires framework-agnostic handlers to HTTP routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from litestar import Litestar, get, post, put
from litestar.config.cors import CORSConfig
from litestar.exceptions import ValidationException
from litestar.params import Parameter
from litestar.response import Stream

from lilbee.server import handlers
from lilbee.server.handlers import _sse_generator


@get("/api/health")
async def health_route() -> dict[str, str]:
    return await handlers.health()


@get("/api/status")
async def status_route() -> dict[str, Any]:
    return await handlers.status()


@get("/api/search")
async def search_route(
    q: str = Parameter(query="q"),
    top_k: int = Parameter(query="top_k", default=5),
) -> list[dict[str, Any]]:
    return await handlers.search(q, top_k=top_k)


@post("/api/ask")
async def ask_route(data: dict[str, Any]) -> dict[str, Any]:
    return await handlers.ask(
        question=data["question"],
        top_k=data.get("top_k", 0),
        options=data.get("options"),
    )


@post("/api/ask/stream")
async def ask_stream_route(data: dict[str, Any]) -> Stream:
    return Stream(
        handlers.ask_stream(
            question=data["question"],
            top_k=data.get("top_k", 0),
            options=data.get("options"),
        ),
        media_type="text/event-stream",
    )


@post("/api/chat")
async def chat_route(data: dict[str, Any]) -> dict[str, Any]:
    return await handlers.chat(
        question=data["question"],
        history=data.get("history", []),
        top_k=data.get("top_k", 0),
        options=data.get("options"),
    )


@post("/api/chat/stream")
async def chat_stream_route(data: dict[str, Any]) -> Stream:
    return Stream(
        handlers.chat_stream(
            question=data["question"],
            history=data.get("history", []),
            top_k=data.get("top_k", 0),
            options=data.get("options"),
        ),
        media_type="text/event-stream",
    )


@post("/api/sync")
async def sync_route(data: dict[str, Any] | None = None) -> Stream:
    force_vision = (data or {}).get("force_vision", False)
    return Stream(
        handlers.sync_stream(force_vision=force_vision),
        media_type="text/event-stream",
    )


@post("/api/add")
async def add_route(data: dict[str, Any]) -> Stream:
    """Add files to the knowledge base with streaming SSE progress."""
    try:
        _paths, queue, task = await handlers.add_files(data)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc

    async def _stream() -> AsyncGenerator[bytes, None]:
        async for chunk in _sse_generator(queue):
            yield chunk
        await task

    return Stream(_stream(), media_type="text/event-stream", status_code=201)


@get("/api/models")
async def models_list_route() -> dict[str, Any]:
    return await handlers.list_models()


@put("/api/models/chat")
async def models_set_chat_route(data: dict[str, Any]) -> dict[str, str]:
    return await handlers.set_chat_model(model=data["model"])


@put("/api/models/vision")
async def models_set_vision_route(data: dict[str, Any]) -> dict[str, str]:
    return await handlers.set_vision_model(model=data["model"])


def create_app() -> Litestar:
    """Create the Litestar application instance."""
    cors = CORSConfig(
        allow_origins=["app://obsidian.md"],
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
    )
