"""Session route handlers: list, get, rename, forget.

Reads and mutations go through the process ``SessionStore`` on the services
container. A missing session id surfaces as a 404.
"""

from __future__ import annotations

from litestar.exceptions import NotFoundException

from lilbee.app.services import get_services
from lilbee.server.models import (
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionListResponse,
    SessionMessageItem,
    SessionMetaItem,
    SessionRenameResponse,
)
from lilbee.sessions import Session, SessionMeta, SessionNotFoundError, SessionStore, TitleSource


def _store() -> SessionStore:
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
    )


async def list_sessions() -> SessionListResponse:
    """Return every session's metadata, newest first."""
    return SessionListResponse(sessions=[_meta_item(meta) for meta in _store().list()])


async def get_session(session_id: str) -> SessionDetailResponse:
    """Return a session's metadata and transcript, or 404 if unknown."""
    try:
        session = _store().get(session_id)
    except SessionNotFoundError as exc:
        raise NotFoundException(detail=str(exc)) from exc
    return _detail(session)


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
