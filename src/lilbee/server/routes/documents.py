"""Document management route handlers: add, list, remove, sync."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from litestar import get, post
from litestar.exceptions import ValidationException
from litestar.params import Parameter
from litestar.response import Stream
from pydantic import BaseModel, Field

from lilbee.server import handlers
from lilbee.server.auth import read_only
from lilbee.server.handlers import sse_generator
from lilbee.server.models import (
    AddRequest,
    DocumentListResponse,
    DocumentRemoveResponse,
    SyncRequest,
)


class RemoveRequest(BaseModel):
    """Request body for /api/documents/remove."""

    names: list[str] = Field(max_length=100)
    delete_files: bool = False


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
        result = await handlers.add_files(data.model_dump())
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc

    async def _stream() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in sse_generator(result.queue):
                yield chunk
            await result.task
        except (GeneratorExit, Exception):
            result.cancel.set()

    return Stream(_stream(), media_type="text/event-stream", status_code=201)


@get("/api/documents")
@read_only
async def documents_list_route(
    search: str = Parameter(query="search", default=""),
    limit: int = Parameter(query="limit", default=50, le=1000),
    offset: int = Parameter(query="offset", default=0, ge=0),
) -> DocumentListResponse:
    """List indexed documents with metadata, paginated and searchable."""
    return await handlers.list_documents(search=search, limit=limit, offset=offset)


@post("/api/documents/remove")
async def documents_remove_route(data: RemoveRequest) -> DocumentRemoveResponse:
    """Remove documents from the knowledge base by source name."""
    return await handlers.delete_documents(data.names, delete_files=data.delete_files)
