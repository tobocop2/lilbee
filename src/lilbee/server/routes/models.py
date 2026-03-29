"""Model management route handlers: catalog, installed, pull, show, delete, set."""

from __future__ import annotations

from typing import Any

from litestar import delete, get, post, put
from litestar.params import Parameter
from litestar.response import Stream
from pydantic import BaseModel

from lilbee.server import handlers
from lilbee.server.auth import read_only
from lilbee.server.models import SetModelRequest, SetModelResponse


class PullRequest(BaseModel):
    """Request body for /api/models/pull."""

    model: str
    source: str = "native"


@get("/api/models")
@read_only
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
@read_only
async def models_catalog_route(
    task: str | None = Parameter(query="task", default=None),
    search: str = Parameter(query="search", default=""),
    size: str | None = Parameter(query="size", default=None),
    featured: bool | None = Parameter(query="featured", default=None),
    sort: str = Parameter(query="sort", default="featured"),
    limit: int = Parameter(query="limit", default=20, le=1000),
    offset: int = Parameter(query="offset", default=0, ge=0),
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
@read_only
async def models_installed_route() -> dict[str, Any]:
    """List installed models with their source (native or litellm)."""
    return await handlers.models_installed()


@post("/api/models/pull")
async def models_pull_route(data: PullRequest) -> Stream:
    """Pull a model with streaming SSE progress events."""
    return Stream(
        handlers.models_pull(data.model, source=data.source),
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
