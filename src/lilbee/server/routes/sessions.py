"""Session routes: list, get, create, append, summary, rename, delete.

Every route requires the bearer token, reads included. A conversation
transcript is at least as personal as the memory store, which is gated for the
same reason. The two ``GET`` routes used to be marked read-only, with a
docstring claiming that let "a read-only session token" browse history; there
is no such token, and the marker means no auth check at all, so both reads
answered callers that sent no Authorization header.
"""

from __future__ import annotations

from litestar import delete, get, patch, post, put

from lilbee.server.handlers.sessions import (
    add_session_message,
    claim_session,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    rename_session,
    set_session_summary,
)
from lilbee.server.models import (
    SessionCreateRequest,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionListResponse,
    SessionMessageCreateRequest,
    SessionRenameRequest,
    SessionRenameResponse,
    SessionSummaryRequest,
)


@get("/api/sessions")
async def sessions_list_route() -> SessionListResponse:
    """List saved conversations, newest first."""
    return await list_sessions()


@get("/api/sessions/{session_id:str}")
async def session_get_route(session_id: str) -> SessionDetailResponse:
    """Return a conversation's metadata and full transcript."""
    return await get_session(session_id)


@post("/api/sessions")
async def session_create_route(data: SessionCreateRequest) -> SessionDetailResponse:
    """Start a new conversation."""
    return await create_session(data)


@post("/api/sessions/{session_id:str}/messages")
async def session_add_message_route(
    session_id: str, data: SessionMessageCreateRequest
) -> SessionDetailResponse:
    """Append a turn to a conversation."""
    return await add_session_message(session_id, data)


@post("/api/sessions/{session_id:str}/claim")
async def session_claim_route(session_id: str) -> SessionDetailResponse:
    """Claim a conversation for this surface so it can append."""
    return await claim_session(session_id)


@put("/api/sessions/{session_id:str}/summary")
async def session_set_summary_route(
    session_id: str, data: SessionSummaryRequest
) -> SessionDetailResponse:
    """Replace a conversation's compaction summary."""
    return await set_session_summary(session_id, data)


@patch("/api/sessions/{session_id:str}")
async def session_rename_route(
    session_id: str, data: SessionRenameRequest
) -> SessionRenameResponse:
    """Rename a conversation."""
    return await rename_session(session_id, data.title)


@delete("/api/sessions/{session_id:str}", status_code=200)
async def session_delete_route(session_id: str) -> SessionDeleteResponse:
    """Delete a conversation."""
    return await delete_session(session_id)
