"""Session routes: list, get, rename, delete.

The two ``GET`` routes are ``@read_only`` so a read-only session token can browse
and resume history; rename and delete are unmarked so read-only tokens cannot
mutate.
"""

from __future__ import annotations

from litestar import delete, get, patch

from lilbee.server.auth import read_only
from lilbee.server.handlers.sessions import (
    delete_session,
    get_session,
    list_sessions,
    rename_session,
)
from lilbee.server.models import (
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionListResponse,
    SessionRenameRequest,
    SessionRenameResponse,
)


@get("/api/sessions")
@read_only
async def sessions_list_route() -> SessionListResponse:
    """List saved conversations, newest first."""
    return await list_sessions()


@get("/api/sessions/{session_id:str}")
@read_only
async def session_get_route(session_id: str) -> SessionDetailResponse:
    """Return a conversation's metadata and full transcript."""
    return await get_session(session_id)


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
