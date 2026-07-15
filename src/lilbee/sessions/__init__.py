"""Chat session persistence: append-only JSONL logs managed by ``SessionStore``."""

from __future__ import annotations

from lilbee.sessions.store import (
    SESSIONS_DIRNAME,
    UNTITLED_SESSION_TITLE,
    MessageRole,
    Session,
    SessionEventType,
    SessionMessage,
    SessionMeta,
    SessionNotFoundError,
    SessionStore,
    TitleSource,
    derive_title,
)

__all__ = [
    "SESSIONS_DIRNAME",
    "UNTITLED_SESSION_TITLE",
    "MessageRole",
    "Session",
    "SessionEventType",
    "SessionMessage",
    "SessionMeta",
    "SessionNotFoundError",
    "SessionStore",
    "TitleSource",
    "derive_title",
]
