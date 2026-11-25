"""Document management route handlers: add, list, remove, sync, export.

Every route needs the token, reads included: the listing names the user's
files and ``/api/export`` serializes the whole corpus.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Annotated

from litestar import Request, Response, get, post
from litestar.datastructures import UploadFile
from litestar.exceptions import ClientException, NotFoundException, ValidationException
from litestar.params import FromQuery, MultipartBody, QueryParameter
from litestar.response import Stream
from litestar.status_codes import HTTP_200_OK, HTTP_409_CONFLICT
from pydantic import BaseModel, Field

from lilbee.core.config import validate_ocr_timeout
from lilbee.data.store import DOCUMENTS_PAGE_DEFAULT, DOCUMENTS_PAGE_MAX
from lilbee.runtime.add_registry import AddIdRunningError
from lilbee.server import handlers
from lilbee.server.content_disposition import CONTENT_DISPOSITION
from lilbee.server.handlers.sse import SSE_MEDIA_TYPE
from lilbee.server.models import (
    ADD_ID_MAX_LENGTH,
    ADD_ID_PATTERN,
    AddCancelRequest,
    AddCancelResponse,
    AddRequest,
    DocumentListResponse,
    DocumentRemoveResponse,
    SyncRequest,
)


class AddIdConflictException(ClientException):
    """An add with the same ``add_id`` already runs."""

    status_code = HTTP_409_CONFLICT


def _add_response(open_stream: Callable[[], AsyncGenerator[str, None]]) -> Stream:
    """The SSE answer of an add; opening the stream reserves its id, or answers 409."""
    try:
        stream = open_stream()
    except AddIdRunningError as exc:
        raise AddIdConflictException(detail=str(exc)) from exc
    return Stream(stream, media_type=SSE_MEDIA_TYPE, status_code=201)


class RemoveRequest(BaseModel):
    """Request body for /api/documents/remove."""

    names: list[str] = Field(max_length=100)


@post("/api/sync", media_type=SSE_MEDIA_TYPE)
async def sync_route(data: SyncRequest | None = None) -> Stream:
    """Re-index changed documents with streaming SSE progress events.

    Pass ``{"force_rebuild": true}`` to re-ingest every file over the index
    under the current ``cfg.embedding_model``. This is the recovery path after
    a ``PUT /api/models/embedding`` that returned ``reindex_required=true``.
    Pass ``{"retry_skipped": true}`` for the lighter path: retry the files that
    failed a previous sync and read partly read files again, without dropping
    the store.
    Pass ``{"prune_ignored": true}`` to also drop sources a ``.lilbeeignore``
    now excludes; without it, sync leaves already-indexed sources alone.
    """
    ocr = data.ocr if data else None
    ocr_timeout = data.ocr_timeout if data else None
    force_rebuild = data.force_rebuild if data else False
    retry_skipped = data.retry_skipped if data else False
    prune_ignored = data.prune_ignored if data else False
    try:
        validate_ocr_timeout(ocr_timeout)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    return Stream(
        handlers.sync_stream(
            ocr=ocr,
            ocr_timeout=ocr_timeout,
            force_rebuild=force_rebuild,
            retry_skipped=retry_skipped,
            prune_ignored=prune_ignored,
        ),
        media_type=SSE_MEDIA_TYPE,
    )


@post("/api/add", media_type=SSE_MEDIA_TYPE, raises=[AddIdConflictException])
async def add_route(data: AddRequest) -> Stream:
    """Add files to the knowledge base with streaming SSE progress.

    An ``add_id`` names the add, so ``POST /api/add/cancel`` can stop it.
    """
    try:
        paths, force, ocr, ocr_timeout = handlers.validate_add_paths(data.model_dump())
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    return _add_response(
        lambda: handlers.add_files_stream(
            paths, force=force, ocr=ocr, ocr_timeout=ocr_timeout, add_id=data.add_id
        )
    )


@post("/api/add/upload", media_type=SSE_MEDIA_TYPE, raises=[AddIdConflictException])
async def add_upload_route(
    data: MultipartBody[list[UploadFile]],
    ocr: FromQuery[str | None] = None,
    ocr_timeout: FromQuery[float | None] = None,
    add_id: Annotated[
        str | None,
        QueryParameter(min_length=1, max_length=ADD_ID_MAX_LENGTH, pattern=ADD_ID_PATTERN),
    ] = None,
) -> Stream:
    """Ingest uploaded file content with streaming SSE progress.

    Unlike /api/add, which reads server-side paths, this accepts the client's raw
    file bytes. That lets a client whose files the server cannot read by path --
    e.g. the plugin or CLI in external mode against a remote lilbee / GPU box --
    ingest its own local files by uploading them straight to the server.
    ``ocr``, ``ocr_timeout`` and ``add_id`` are query parameters because the
    request body is the upload's raw multipart file list.
    """
    # Names first, bytes second: reading every part before validating cost a
    # full in-memory copy of a payload that was going to be rejected anyway.
    try:
        names = handlers.validate_upload_names([upload.filename for upload in data])
        ocr_mode = handlers.parse_ocr_mode(ocr)
        validate_ocr_timeout(ocr_timeout)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc
    cleaned = [(name, await upload.read()) for name, upload in zip(names, data, strict=True)]
    return _add_response(
        lambda: handlers.add_uploads_stream(
            cleaned, ocr=ocr_mode, ocr_timeout=ocr_timeout, add_id=add_id
        )
    )


@post("/api/add/cancel", status_code=HTTP_200_OK, raises=[NotFoundException])
async def add_cancel_route(data: AddCancelRequest) -> AddCancelResponse:
    """Stop the add or upload that runs under ``add_id``, as the user's cancel.

    The add removes what it did not finish: a source with no file indexed is no
    longer added, and an uploaded file the server did not index is deleted. The
    stream of the add still ends with its ``done`` event, which carries
    ``cancelled`` and ``not_added``. A stream that only closes keeps its sources.
    Answers 404 when no add runs under ``add_id``.
    """
    if not handlers.cancel_add(data.add_id):
        raise NotFoundException(detail=f"No add is running with the add_id '{data.add_id}'")
    return AddCancelResponse(add_id=data.add_id, cancelling=True)


@get("/api/documents")
async def documents_list_route(
    search: FromQuery[str] = "",
    limit: Annotated[int, QueryParameter(ge=1, le=DOCUMENTS_PAGE_MAX)] = DOCUMENTS_PAGE_DEFAULT,
    offset: Annotated[int, QueryParameter(ge=0)] = 0,
) -> DocumentListResponse:
    """List indexed documents with metadata, paginated and searchable."""
    return await handlers.list_documents(search=search, limit=limit, offset=offset)


@post("/api/documents/remove")
async def documents_remove_route(data: RemoveRequest) -> DocumentRemoveResponse:
    """Remove documents from the knowledge base by source name."""
    return await handlers.delete_documents(data.names)


@get("/api/export", media_type="application/octet-stream")
async def export_route(
    fmt: Annotated[str, QueryParameter(name="format")] = "",
    source: FromQuery[str] = "",
) -> Response[bytes]:
    """Download the per-page text dataset as a file (parquet by default).

    The media type is declared on the decorator as well as on the returned
    Response for the same reason the streaming routes declare theirs: litestar
    documents the content type from the decorator, so without it the schema
    promises JSON and a generated client parses a parquet file as text.
    """
    from lilbee.app.dataset import DatasetError, export_to_bytes

    try:
        # export_to_bytes serializes the whole per-page dataset into memory;
        # offload so a large export doesn't stall every other request, matching
        # get_source_content's own off-loop read.
        payload = await asyncio.to_thread(export_to_bytes, fmt, source or None)
    except DatasetError as exc:
        raise ValidationException(str(exc)) from exc
    return Response(
        content=payload.data,
        media_type="application/octet-stream",
        headers={CONTENT_DISPOSITION: f'attachment; filename="pages.{payload.fmt}"'},
    )


@post("/api/import", media_type=SSE_MEDIA_TYPE)
async def import_route(
    request: Request,
    fmt: Annotated[str, QueryParameter(name="format")] = "",
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
        media_type=SSE_MEDIA_TYPE,
        status_code=201,
    )
