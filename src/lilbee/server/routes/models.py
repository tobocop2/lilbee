"""Model management route handlers: catalog, installed, pull, show, delete, set."""

from __future__ import annotations

from litestar import delete, get, post, put
from litestar.exceptions import HTTPException
from litestar.params import Parameter
from litestar.response import Stream
from pydantic import BaseModel

from lilbee.modelhub.role_validator import TaskMismatchError
from lilbee.server import handlers
from lilbee.server.auth import read_only
from lilbee.server.handlers import ModelsResponse, format_task_mismatch
from lilbee.server.models import (
    ExternalModelsResponse,
    ModelsCatalogResponse,
    ModelsDeleteResponse,
    ModelsInstalledResponse,
    ModelsShowResponse,
    SetModelRequest,
    SetModelResponse,
)


def _task_mismatch_detail(exc: ValueError) -> str:
    """Format a 422 detail string, expanding TaskMismatchError into HTTP guidance."""
    if isinstance(exc, TaskMismatchError):
        return format_task_mismatch(exc.ref, exc.entry_task, exc.expected_task)
    return str(exc)


class PullRequest(BaseModel):
    """Request body for /api/models/pull."""

    model: str
    source: str = "native"
    allow_unsupported: bool = False


@get("/api/models")
@read_only
async def models_list_route() -> ModelsResponse:
    """Available chat, embedding, vision, and reranker models."""
    return await handlers.list_models()


@get("/api/models/external")
@read_only
async def models_external_route() -> ExternalModelsResponse:
    """Discover models available from the configured external provider."""
    return await handlers.list_external_models()


@put("/api/models/chat")
async def models_set_chat_route(data: SetModelRequest) -> SetModelResponse:
    """Switch the active chat model used for RAG answers."""
    try:
        return await handlers.set_chat_model(model=data.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=_task_mismatch_detail(exc)) from exc


@put("/api/models/embedding")
async def models_set_embedding_route(data: SetModelRequest) -> SetModelResponse:
    """Switch the active embedding model."""
    try:
        return await handlers.set_embedding_model(model=data.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=_task_mismatch_detail(exc)) from exc


@put("/api/models/vision")
async def models_set_vision_route(data: SetModelRequest) -> SetModelResponse:
    """Switch the active vision model for scanned PDF OCR. Empty disables OCR."""
    try:
        return await handlers.set_vision_model(model=data.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=_task_mismatch_detail(exc)) from exc


@put("/api/models/reranker")
async def models_set_reranker_route(data: SetModelRequest) -> SetModelResponse:
    """Switch the active reranker model. Empty disables reranking."""
    try:
        return await handlers.set_reranker_model(model=data.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=_task_mismatch_detail(exc)) from exc


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
) -> ModelsCatalogResponse:
    """Browse the model catalog with optional filters."""
    try:
        return await handlers.models_catalog(
            task=task,
            search=search,
            size=size,
            featured=featured,
            sort=sort,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@get("/api/models/installed")
@read_only
async def models_installed_route() -> ModelsInstalledResponse:
    """List installed models with their source (native or remote)."""
    return await handlers.models_installed()


@post("/api/models/pull")
async def models_pull_route(data: PullRequest) -> Stream:
    """Pull a model with streaming SSE progress events."""
    # Validate before opening the stream so an unsupported arch is a real 409, not
    # an in-stream abort after the 200 SSE headers have flushed.
    await handlers.enforce_pull_arch_compat(
        data.model, source=data.source, allow_unsupported=data.allow_unsupported
    )
    return Stream(
        handlers.models_pull(
            data.model, source=data.source, allow_unsupported=data.allow_unsupported
        ),
        media_type="text/event-stream",
    )


@post("/api/models/show")
async def models_show_route(data: SetModelRequest) -> ModelsShowResponse:
    """Get model metadata and parameter defaults."""
    return await handlers.models_show(model=data.model)


@delete("/api/models/{model:str}", status_code=200)
async def models_delete_route(model: str, source: str = "native") -> ModelsDeleteResponse:
    """Delete a model from the specified source."""
    return await handlers.models_delete(model, source=source)
