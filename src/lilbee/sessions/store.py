"""Append-only JSONL store for chat sessions.

Each session is one ``<id>.jsonl`` file under ``cfg.data_dir/sessions``. The file
is a strictly append-only event log: one JSON object per line, appended and
fsynced, never rewritten. Event types are ``meta`` (first line), ``title``
(newest wins, so rename appends rather than rewrites), and ``message``. The only
corruption an append log can suffer is a torn final line, which the reader skips.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from lilbee.core.config import cfg

SESSIONS_DIRNAME = "sessions"
UNTITLED_SESSION_TITLE = "Untitled chat"
TITLE_MAX_LEN = 60
TITLE_ELLIPSIS = "…"


class SessionEventType(StrEnum):
    """The tag on every line of a session log."""

    META = "meta"
    TITLE = "title"
    MESSAGE = "message"


class MessageRole(StrEnum):
    """Author of a chat message."""

    USER = "user"
    ASSISTANT = "assistant"


class TitleSource(StrEnum):
    """Where a session title came from."""

    AUTO = "auto"
    CUSTOM = "custom"


@dataclass(frozen=True)
class SessionMessage:
    """One turn in a session. ``ts`` is stamped by the store on write."""

    role: MessageRole
    content: str
    sources: tuple[str, ...] = ()
    ts: str = ""


@dataclass(frozen=True)
class SessionMeta:
    """Session metadata, reconstructed from the log without its message bodies."""

    id: str
    title: str
    created_at: str
    updated_at: str
    model_ref: str
    scope: str
    message_count: int


@dataclass(frozen=True)
class Session:
    """A session's metadata plus its full transcript."""

    meta: SessionMeta
    messages: tuple[SessionMessage, ...]


class SessionNotFoundError(Exception):
    """Raised when a session id has no backing file."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"No session with id {session_id!r}")
        self.session_id = session_id


def derive_title(text: str) -> str:
    """Title a session from its first user message: first line, truncated."""
    stripped = text.strip()
    if not stripped:
        return UNTITLED_SESSION_TITLE
    first = stripped.splitlines()[0]
    if len(first) > TITLE_MAX_LEN:
        return first[:TITLE_MAX_LEN] + TITLE_ELLIPSIS
    return first


def _message_from_event(event: dict[str, Any], ts: str) -> SessionMessage:
    """Reconstruct one message from its ``message`` event line."""
    return SessionMessage(
        role=MessageRole(event["role"]),
        content=event["content"],
        sources=tuple(event.get("sources", [])),
        ts=ts,
    )


class SessionStore:
    """Reads and appends session logs under ``cfg.data_dir/sessions``.

    The directory is resolved late-bound from ``cfg`` on every call, so the store
    follows a reconfigured data dir (and test isolation) without reconstruction.
    ``clock`` is injectable for deterministic tests.
    """

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def _dir(self) -> Path:
        return cfg.data_dir / SESSIONS_DIRNAME

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.jsonl"

    def _now(self) -> str:
        return self._clock().isoformat()

    def _require(self, session_id: str) -> Path:
        path = self._path(session_id)
        if not path.exists():
            raise SessionNotFoundError(session_id)
        return path

    @staticmethod
    def _write_event(path: Path, event: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    @staticmethod
    def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line; skip it

    def create(self, model_ref: str, scope: str) -> str:
        """Start a new session with a ``meta`` line and return its id."""
        session_id = uuid4().hex
        self._dir.mkdir(parents=True, exist_ok=True)
        now = self._now()
        self._write_event(
            self._path(session_id),
            {
                "type": SessionEventType.META,
                "id": session_id,
                "created_at": now,
                "model_ref": model_ref,
                "scope": scope,
                "ts": now,
            },
        )
        return session_id

    def add_message(self, session_id: str, message: SessionMessage) -> None:
        """Append one message event to an existing session."""
        self._write_event(
            self._require(session_id),
            {
                "type": SessionEventType.MESSAGE,
                "role": message.role,
                "content": message.content,
                "sources": list(message.sources),
                "ts": self._now(),
            },
        )

    def set_title(self, session_id: str, title: str, source: TitleSource) -> None:
        """Append a title event; the newest title wins on read."""
        self._write_event(
            self._require(session_id),
            {"type": SessionEventType.TITLE, "title": title, "source": source, "ts": self._now()},
        )

    def delete(self, session_id: str) -> None:
        """Remove a session's file."""
        self._require(session_id).unlink()

    def get(self, session_id: str) -> Session:
        """Replay a session's log into its reconstructed view."""
        return self._fold(session_id, self._require(session_id))

    def list(self) -> list[SessionMeta]:
        """All sessions' metadata, newest first."""
        if not self._dir.exists():
            return []
        metas = [self._fold(path.stem, path).meta for path in self._dir.glob("*.jsonl")]
        return sorted(metas, key=lambda meta: (meta.updated_at, meta.id), reverse=True)

    def _fold(self, session_id: str, path: Path) -> Session:
        created_at = ""
        model_ref = ""
        scope = ""
        title = UNTITLED_SESSION_TITLE
        updated_at = ""
        messages: list[SessionMessage] = []
        for event in self._iter_events(path):
            ts = event.get("ts", "")
            updated_at = ts
            event_type = event.get("type")
            if event_type == SessionEventType.META:
                created_at = event["created_at"]
                model_ref = event["model_ref"]
                scope = event["scope"]
            elif event_type == SessionEventType.TITLE:
                title = event["title"]
            elif event_type == SessionEventType.MESSAGE:
                messages.append(_message_from_event(event, ts))
        meta = SessionMeta(
            id=session_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            model_ref=model_ref,
            scope=scope,
            message_count=len(messages),
        )
        return Session(meta=meta, messages=tuple(messages))
