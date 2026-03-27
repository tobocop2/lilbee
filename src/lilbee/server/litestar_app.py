"""Litestar adapter — wires framework-agnostic handlers to HTTP routes."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from litestar import Litestar, delete, get, post, put
from litestar.config.cors import CORSConfig
from litestar.exceptions import ValidationException
from litestar.openapi import OpenAPIConfig
from litestar.params import Parameter
from litestar.response import Stream

from lilbee.cli.helpers import get_version
from lilbee.config import cfg
from lilbee.query import ChatMessage as ChatMessageDict
from lilbee.server import handlers
from lilbee.server.handlers import sse_generator
from lilbee.server.models import (
    AddRequest,
    AskRequest,
    AskResponse,
    ChatRequest,
    CleanedChunk,
    CrawlRequest,
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
        async for chunk in sse_generator(queue):
            yield chunk
        await task

    return Stream(_stream(), media_type="text/event-stream", status_code=201)


@get("/api/models")
async def models_list_route() -> dict[str, Any]:
    """Available chat and vision models."""
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


@get("/api/models/catalog")
async def models_catalog_route(
    task: str | None = Parameter(query="task", default=None),
    search: str = Parameter(query="search", default=""),
    size: str | None = Parameter(query="size", default=None),
    featured: bool | None = Parameter(query="featured", default=None),
    sort: str = Parameter(query="sort", default="featured"),
    limit: int = Parameter(query="limit", default=20),
    offset: int = Parameter(query="offset", default=0),
) -> dict[str, Any]:
    """Browse the model catalog with optional filters."""
    return await handlers.models_catalog(
        task=task,
        search=search,
        size=size,
        featured=featured,
        sort=sort,
        limit=limit,
        offset=offset,
    )


@get("/api/models/installed")
async def models_installed_route() -> dict[str, Any]:
    """List installed models with their source (native or litellm)."""
    return await handlers.models_installed()


@post("/api/models/pull")
async def models_pull_route(data: dict[str, Any]) -> Stream:
    """Pull a model with streaming SSE progress events."""
    model = data.get("model", "")
    source = data.get("source", "native")
    return Stream(
        handlers.models_pull(model, source=source),
        media_type="text/event-stream",
    )


@post("/api/models/show")
async def models_show_route(data: SetModelRequest) -> dict[str, Any]:
    """Get model metadata and parameter defaults."""
    return await handlers.models_show(model=data.model)


@delete("/api/models/{model:str}", status_code=200)
async def models_delete_route(model: str, source: str = "native") -> dict[str, Any]:
    """Delete a model from the specified source."""
    return await handlers.models_delete(model, source=source)


@get("/api/config")
async def config_route() -> dict[str, Any]:
    """Return all user-facing configuration values."""
    return await handlers.get_config()


@get("/api/documents")
async def documents_list_route(
    search: str = Parameter(query="search", default=""),
    limit: int = Parameter(query="limit", default=50),
    offset: int = Parameter(query="offset", default=0),
) -> dict[str, Any]:
    """List indexed documents with metadata, paginated and searchable."""
    return await handlers.list_documents(search=search, limit=limit, offset=offset)


@post("/api/documents/remove")
async def documents_remove_route(data: dict[str, Any]) -> dict[str, Any]:
    """Remove documents from the knowledge base by source name."""
    names = data.get("names", [])
    delete_files = data.get("delete_files", False)
    return await handlers.delete_documents(names, delete_files=delete_files)


@post("/api/crawl")
async def crawl_route(data: CrawlRequest) -> Stream:
    """Crawl a URL with streaming SSE progress events (crawl_start, crawl_page, crawl_done)."""
    try:
        gen = handlers.crawl_stream(url=data.url, depth=data.depth, max_pages=data.max_pages)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    return Stream(gen, media_type="text/event-stream")


log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: Litestar) -> AsyncIterator[None]:
    """Pre-load LLM provider and embedding model on server startup."""
    try:
        from lilbee.providers.factory import get_provider

        get_provider()
        log.info("LLM provider pre-loaded")
    except Exception:
        log.warning("Failed to pre-load LLM provider", exc_info=True)
    try:
        from lilbee import embedder

        embedder.validate_model()
        log.info("Embedding model validated")
    except Exception:
        log.warning("Failed to validate embedding model", exc_info=True)
    yield


def create_app() -> Litestar:
    """Create the Litestar application instance."""
    cors = CORSConfig(
        allow_origins=cfg.cors_origins,
        allow_origin_regex=r"^http://localhost(:\d+)?$",
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )
    return Litestar(
        lifespan=[_lifespan],
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
            config_route,
            models_catalog_route,
            models_installed_route,
            models_pull_route,
            models_show_route,
            models_delete_route,
            documents_list_route,
            documents_remove_route,
            crawl_route,
        ],
        cors_config=cors,
        openapi_config=OpenAPIConfig(
            title="lilbee",
            description="Local knowledge base REST API",
            version=get_version(),
            path="/schema",
        ),
    )
