"""Session routes: list, get, markdown, create, append, fork, summary, rename, delete.

Every route requires the token, reads included: a transcript is at least as
personal as the memory store next door.
"""

from __future__ import annotations

import re
from urllib.parse import quote

from litestar import Response, delete, get, patch, post, put
from litestar.datastructures import ResponseHeader
from litestar.exceptions import NotFoundException
from litestar.params import FromPath

from lilbee.data.types import MARKDOWN_MIME
from lilbee.server.handlers.sessions import (
    add_session_message,
    claim_session,
    create_session,
    delete_session,
    fork_session,
    get_session,
    get_session_markdown,
    list_sessions,
    rename_session,
    set_session_summary,
)
from lilbee.server.models import (
    SessionCreateRequest,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionForkRequest,
    SessionListResponse,
    SessionMessageCreateRequest,
    SessionRenameRequest,
    SessionRenameResponse,
    SessionSummaryRequest,
)

CONTENT_DISPOSITION = "Content-Disposition"
# Printable ASCII except the quote and backslash a quoted filename cannot hold bare.
_FILENAME_UNSAFE_RE = re.compile(r"[^ !#-\[\]-~]")
_FILENAME_PLACEHOLDER = "_"


def _attachment_disposition(filename: str) -> str:
    """An RFC 6266 attachment header: an ASCII *filename* fallback plus the exact UTF-8 name."""
    fallback = _FILENAME_UNSAFE_RE.sub(_FILENAME_PLACEHOLDER, filename)
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"


@get("/api/sessions")
async def sessions_list_route() -> SessionListResponse:
    """List saved conversations, newest first."""
    return await list_sessions()


@get("/api/sessions/{session_id:str}")
async def session_get_route(session_id: FromPath[str]) -> SessionDetailResponse:
    """Return a conversation's metadata and full transcript."""
    return await get_session(session_id)


@get(
    "/api/sessions/{session_id:str}/markdown",
    media_type=MARKDOWN_MIME,
    raises=[NotFoundException],
    response_headers=[
        ResponseHeader(
            name=CONTENT_DISPOSITION,
            description="The export's file name: the title slug and the session id prefix.",
            documentation_only=True,
        )
    ],
)
async def session_markdown_route(session_id: FromPath[str]) -> Response[str]:
    """Return a conversation as a markdown document, named for download."""
    export = await get_session_markdown(session_id)
    return Response(
        export.markdown,
        media_type=MARKDOWN_MIME,
        headers={CONTENT_DISPOSITION: _attachment_disposition(export.filename)},
    )


@post("/api/sessions")
async def session_create_route(data: SessionCreateRequest) -> SessionDetailResponse:
    """Start a new conversation."""
    return await create_session(data)


@post("/api/sessions/{session_id:str}/messages")
async def session_add_message_route(
    session_id: FromPath[str], data: SessionMessageCreateRequest
) -> SessionDetailResponse:
    """Append a turn to a conversation."""
    return await add_session_message(session_id, data)


@post("/api/sessions/{session_id:str}/fork")
async def session_fork_route(
    session_id: FromPath[str], data: SessionForkRequest | None = None
) -> SessionDetailResponse:
    """Start a new conversation from a copy of this one's leading messages."""
    return await fork_session(session_id, data)


@post("/api/sessions/{session_id:str}/claim")
async def session_claim_route(session_id: FromPath[str]) -> SessionDetailResponse:
    """Claim a conversation for this surface so it can append."""
    return await claim_session(session_id)


@put("/api/sessions/{session_id:str}/summary")
async def session_set_summary_route(
    session_id: FromPath[str], data: SessionSummaryRequest
) -> SessionDetailResponse:
    """Replace a conversation's compaction summary."""
    return await set_session_summary(session_id, data)


@patch("/api/sessions/{session_id:str}")
async def session_rename_route(
    session_id: FromPath[str], data: SessionRenameRequest
) -> SessionRenameResponse:
    """Rename a conversation."""
    return await rename_session(session_id, data.title)


@delete("/api/sessions/{session_id:str}", status_code=200)
async def session_delete_route(session_id: FromPath[str]) -> SessionDeleteResponse:
    """Delete a conversation."""
    return await delete_session(session_id)
