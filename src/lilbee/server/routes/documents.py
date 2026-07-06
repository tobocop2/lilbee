"""Document management route handlers: add, list, remove, sync."""

from __future__ import annotations

from litestar import Request, Response, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.exceptions import ValidationException
from litestar.params import Body, Parameter
from litestar.response import Stream
from pydantic import BaseModel, Field

from lilbee.server import handlers
from lilbee.server.auth import read_only
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
    """Re-index changed documents with streaming SSE progress events.

    Pass ``{"force_rebuild": true}`` to wipe the store and re-ingest every file
    under the current ``cfg.embedding_model``. This is the recovery path after
    a ``PUT /api/models/embedding`` that returned ``reindex_required=true``.
    Pass ``{"retry_skipped": true}`` for the lighter path: retry the files that
    failed a previous sync without dropping the store.
    """
    enable_ocr = data.enable_ocr if data else None
    force_rebuild = data.force_rebuild if data else False
    retry_skipped = data.retry_skipped if data else False
    return Stream(
        handlers.sync_stream(
            enable_ocr=enable_ocr, force_rebuild=force_rebuild, retry_skipped=retry_skipped
        ),
        media_type="text/event-stream",
    )


@post("/api/add")
async def add_route(data: AddRequest) -> Stream:
    """Add files to the knowledge base with streaming SSE progress."""
    try:
        paths, force, enable_ocr, ocr_timeout = handlers.validate_add_paths(data.model_dump())
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    return Stream(
        handlers.add_files_stream(
            paths, force=force, enable_ocr=enable_ocr, ocr_timeout=ocr_timeout
        ),
        media_type="text/event-stream",
        status_code=201,
    )


@post("/api/add/upload")
async def add_upload_route(
    data: list[UploadFile] = Body(media_type=RequestEncodingType.MULTI_PART),
) -> Stream:
    """Ingest uploaded file content with streaming SSE progress.

    Unlike /api/add, which reads server-side paths, this accepts the client's raw
    file bytes. That lets a client whose files the server cannot read by path --
    e.g. the plugin or CLI in external mode against a remote lilbee / GPU box --
    ingest its own local files by uploading them straight to the server.
    """
    files: list[tuple[str, bytes]] = []
    for upload in data:
        files.append((upload.filename, await upload.read()))
    try:
        cleaned = handlers.validate_uploads(files)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    return Stream(
        handlers.add_uploads_stream(cleaned),
        media_type="text/event-stream",
        status_code=201,
    )


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


@get("/api/export")
@read_only
async def export_route(
    fmt: str = Parameter(query="format", default=""),
    source: str = Parameter(query="source", default=""),
) -> Response[bytes]:
    """Download the per-page text dataset as a file (parquet by default)."""
    from lilbee.app.dataset import DatasetError, export_to_bytes

    try:
        payload = export_to_bytes(fmt, source or None)
    except DatasetError as exc:
        raise ValidationException(str(exc)) from exc
    return Response(
        content=payload.data,
        media_type="application/octet-stream",
        headers={"content-disposition": f'attachment; filename="pages.{payload.fmt}"'},
    )


@post("/api/import")
async def import_route(
    request: Request,
    fmt: str = Parameter(query="format", default=""),
) -> Stream:
    """Import an uploaded per-page dataset with streaming SSE progress events.

    The request body is the raw dataset bytes; ``?format=parquet|jsonl`` is
    required since there is no filename to infer from. Bounded by the server's
    body-size limit; larger datasets use the path-based CLI/MCP import.
    """
    from lilbee.app.dataset import DatasetError, require_format

    try:
        require_format(fmt)
    except DatasetError as exc:
        raise ValidationException(str(exc)) from exc
    return Stream(
        handlers.import_stream(await request.body(), fmt),
        media_type="text/event-stream",
        status_code=201,
    )
