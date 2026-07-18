"""Session route handlers: list, get, rename, forget.

Reads and mutations go through the process ``SessionStore`` on the services
container. A missing session id surfaces as a 404.
"""

from __future__ import annotations

from litestar.exceptions import ClientException, NotFoundException
from litestar.status_codes import HTTP_409_CONFLICT

from lilbee.app.services import get_services
from lilbee.server.models import (
    SessionCreateRequest,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionListResponse,
    SessionMessageCreateRequest,
    SessionMessageItem,
    SessionMetaItem,
    SessionRenameResponse,
    SessionSummaryRequest,
)
from lilbee.sessions import (
    HUMAN_ORIGINS,
    SESSIONS_DISABLED_HINT,
    Session,
    SessionMessage,
    SessionMeta,
    SessionNotFoundError,
    SessionOrigin,
    SessionOwnershipError,
    SessionStore,
    TitleSource,
    sessions_enabled,
)


def _require_sessions() -> None:
    """Raise 404 if session persistence is disabled (on by default)."""
    if not sessions_enabled():
        raise NotFoundException(detail=SESSIONS_DISABLED_HINT)


def _store() -> SessionStore:
    _require_sessions()
    return get_services().session_store


def _meta_item(meta: SessionMeta) -> SessionMetaItem:
    return SessionMetaItem(
        id=meta.id,
        title=meta.title,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        model_ref=meta.model_ref,
        scope=meta.scope,
        message_count=meta.message_count,
        origin=meta.origin.value,
    )


def _detail(session: Session) -> SessionDetailResponse:
    return SessionDetailResponse(
        meta=_meta_item(session.meta),
        messages=[
            SessionMessageItem(
                role=message.role,
                content=message.content,
                sources=list(message.sources),
                ts=message.ts,
            )
            for message in session.messages
        ],
        summary=session.summary,
    )


async def list_sessions() -> SessionListResponse:
    """Return every session's metadata, newest first."""
    return SessionListResponse(
        sessions=[_meta_item(meta) for meta in _store().list(origins=HUMAN_ORIGINS)]
    )


async def get_session(session_id: str) -> SessionDetailResponse:
    """Return a session's metadata and transcript, or 404 if unknown."""
    try:
        session = _store().get(session_id)
    except SessionNotFoundError as exc:
        raise NotFoundException(detail=str(exc)) from exc
    return _detail(session)


async def create_session(data: SessionCreateRequest) -> SessionDetailResponse:
    """Start a new conversation and return it (empty transcript, no summary)."""
    session_id = _store().create(
        model_ref=data.model_ref, scope=data.scope, origin=SessionOrigin.HTTP
    )
    return _detail(_store().get(session_id))


async def add_session_message(
    session_id: str, data: SessionMessageCreateRequest
) -> SessionDetailResponse:
    """Append one turn to a conversation and return it, or 404 if unknown."""
    message = SessionMessage(role=data.role, content=data.content, sources=tuple(data.sources))
    try:
        _store().add_message(session_id, message, surface=SessionOrigin.HTTP)
    except SessionNotFoundError as exc:
        raise NotFoundException(detail=str(exc)) from exc
    except SessionOwnershipError as exc:
        # 409, not 403: the resource exists and the token is fine; the session
        # is owned elsewhere, and claiming it is the documented resolution.
        raise ClientException(detail=str(exc), status_code=HTTP_409_CONFLICT) from exc
    return _detail(_store().get(session_id))


async def claim_session(session_id: str) -> SessionDetailResponse:
    """Claim a conversation for the HTTP surface, or 404 if unknown."""
    try:
        _store().transfer(session_id, SessionOrigin.HTTP)
    except SessionNotFoundError as exc:
        raise NotFoundException(detail=str(exc)) from exc
    return _detail(_store().get(session_id))


async def set_session_summary(
    session_id: str, data: SessionSummaryRequest
) -> SessionDetailResponse:
    """Replace a conversation's compaction summary, or 404 if unknown."""
    try:
        _store().set_summary(session_id, data.summary)
    except SessionNotFoundError as exc:
        raise NotFoundException(detail=str(exc)) from exc
    return _detail(_store().get(session_id))


async def rename_session(session_id: str, title: str) -> SessionRenameResponse:
    """Rename a session, or 404 if unknown."""
    try:
        _store().set_title(session_id, title, TitleSource.CUSTOM)
    except SessionNotFoundError as exc:
        raise NotFoundException(detail=str(exc)) from exc
    return SessionRenameResponse(id=session_id, title=title)


async def delete_session(session_id: str) -> SessionDeleteResponse:
    """Delete a session, or 404 if unknown."""
    try:
        _store().delete(session_id)
    except SessionNotFoundError as exc:
        raise NotFoundException(detail=str(exc)) from exc
    return SessionDeleteResponse(id=session_id, deleted=True)
