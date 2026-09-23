"""Append-only JSONL store for chat sessions.

Each session is one ``<id>.jsonl`` file under ``cfg.data_dir/sessions``. The file
is a strictly append-only event log: one JSON object per line, appended and
fsynced, never rewritten. Event types are ``meta`` (first line), ``title``
(newest wins, so rename appends rather than rewrites), ``message``, and
``summary`` (newest wins; compaction's condensed view of the turns that no
longer fit the prompt). The only corruption an append log can suffer is a torn
final line, which the reader skips. A fork's file is written whole once, then
only appended to.
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

from filelock import FileLock

from lilbee.core.config import cfg
from lilbee.core.security import write_private_text

SESSIONS_DIRNAME = "sessions"
SESSIONS_DISABLED_HINT = (
    "Sessions are off. Turn them on with /set sessions_enabled true in the TUI, "
    "settings_set via MCP, or sessions_enabled = true in config.toml."
)
AGENT_SESSIONS_DISABLED_HINT = (
    "Agent sessions are off. Turn them on with settings_set mcp_sessions_enabled "
    "true, /set mcp_sessions_enabled true in the TUI, or mcp_sessions_enabled = "
    "true in config.toml."
)
# Bounds a wedged lock holder; a healthy append holds the lock for milliseconds.
_APPEND_LOCK_TIMEOUT_S = 10
UNTITLED_SESSION_TITLE = "Untitled chat"
TITLE_MAX_LEN = 60
TITLE_ELLIPSIS = "…"
FORK_TITLE_SUFFIX = " (fork {number})"


def sessions_enabled() -> bool:
    """True when session persistence is on for the human surfaces (default on)."""
    return cfg.sessions_enabled


def agent_sessions_enabled() -> bool:
    """True when agent (MCP) sessions are on (default off).

    Independent of ``sessions_enabled``: the two govern separate domains, so an
    agent's working state can be off while a human's conversations are saved.
    """
    return cfg.mcp_sessions_enabled


class SessionEventType(StrEnum):
    """The tag on every line of a session log."""

    META = "meta"
    TITLE = "title"
    MESSAGE = "message"
    SUMMARY = "summary"
    ORIGIN = "origin"


class SessionOrigin(StrEnum):
    """The surface a session belongs to: whoever created it, or was last
    explicitly transferred to. Appends from any other surface are refused, so
    an agent cannot splice its turns into a conversation a human owns."""

    TUI = "tui"
    MCP = "mcp"
    HTTP = "http"
    CLI = "cli"


# The surfaces a human drives directly. Their sessions are one conversation
# space (start in Obsidian, continue in the TUI); agent sessions are working
# state and stay out of it unless asked for.
HUMAN_ORIGINS: frozenset[SessionOrigin] = frozenset(
    {SessionOrigin.TUI, SessionOrigin.HTTP, SessionOrigin.CLI}
)


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
    origin: SessionOrigin = SessionOrigin.TUI
    """Owning surface. Files written before ownership existed carry no origin;
    the only writer then was the TUI, so that is the fallback."""
    forked_from: str = ""
    """Id of the session this one was forked from; empty when it is not a fork."""


@dataclass(frozen=True)
class Session:
    """A session's metadata plus its full transcript."""

    meta: SessionMeta
    messages: tuple[SessionMessage, ...]
    # Rolling summary of the turns compaction has folded away, empty until the
    # conversation first outgrows the prompt budget. It lives here rather than on
    # the meta because only replaying a session needs it: listing does not, and
    # carrying a paragraph per session would bloat the drawer's hot path and
    # every HTTP/MCP list payload.
    summary: str = ""


class SessionNotFoundError(Exception):
    """Raised when a session id has no backing file."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"No session with id {session_id!r}")
        self.session_id = session_id


def _may_append(surface: SessionOrigin, owner: SessionOrigin) -> bool:
    """Whether *surface* may append to a session owned by *owner*.

    The human surfaces are one conversation space (the same person in the TUI,
    Obsidian, or the shell), so they append to each other's sessions freely.
    Agent sessions are working state: only the agent surface appends to them,
    and it appends to nothing else without an explicit claim.
    """
    if surface is owner:
        return True
    return surface in HUMAN_ORIGINS and owner in HUMAN_ORIGINS


class SessionOwnershipError(Exception):
    """Raised when a surface appends to a session another surface owns."""

    def __init__(self, session_id: str, owner: SessionOrigin, surface: SessionOrigin) -> None:
        super().__init__(
            f"Session {session_id!r} belongs to the {owner.value} surface; "
            f"claim it before appending from {surface.value}."
        )
        self.session_id = session_id
        self.owner = owner
        self.surface = surface


class SessionForkRangeError(ValueError):
    """Raised when a fork point lies outside the source session's transcript."""

    def __init__(self, session_id: str, message_count: int, available: int) -> None:
        super().__init__(
            f"Cannot fork session {session_id!r} after {message_count} messages: "
            f"choose a count from 0 to {available}."
        )
        self.session_id = session_id
        self.message_count = message_count
        self.available = available


def derive_title(text: str) -> str:
    """Title a session from its first user message: first line, truncated."""
    stripped = text.strip()
    if not stripped:
        return UNTITLED_SESSION_TITLE
    first = stripped.splitlines()[0]
    if len(first) > TITLE_MAX_LEN:
        return first[:TITLE_MAX_LEN] + TITLE_ELLIPSIS
    return first


def _fork_title(source_title: str, number: int) -> str:
    """``<source title> (fork N)``, the source part clipped so the whole fits TITLE_MAX_LEN."""
    suffix = FORK_TITLE_SUFFIX.format(number=number)
    room = TITLE_MAX_LEN - len(suffix)
    if len(source_title) <= room:
        return source_title + suffix
    return source_title[: room - len(TITLE_ELLIPSIS)] + TITLE_ELLIPSIS + suffix


def _fork_count(source: Session, message_count: int | None) -> int:
    """The number of leading messages to copy, checked against *source*'s transcript."""
    available = len(source.messages)
    if message_count is None:
        return available
    if not 0 <= message_count <= available:
        raise SessionForkRangeError(source.meta.id, message_count, available)
    return message_count


def _message_event(message: SessionMessage, ts: str) -> dict[str, Any]:
    """The ``message`` event line for *message*, stamped *ts*."""
    return {
        "type": SessionEventType.MESSAGE,
        "role": message.role,
        "content": message.content,
        "sources": list(message.sources),
        "ts": ts,
    }


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
        # path -> (size, mtime, meta) from the last fold of that file; see _meta_for.
        self._meta_cache: dict[Path, tuple[int, float, SessionMeta]] = {}

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
        # Per-session lock: two writers on one id (a second process, another
        # surface) serialize instead of interleaving lines. Appends take
        # milliseconds, so a blocked writer waits, never fails, under any
        # realistic contention; the timeout only bounds a wedged holder.
        with (
            FileLock(str(path) + ".lock", timeout=_APPEND_LOCK_TIMEOUT_S),
            path.open("a", encoding="utf-8") as fh,
        ):
            fh.write(json.dumps(event) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    @staticmethod
    def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
        # A torn multi-byte character decodes here, outside the try that skips
        # torn lines; replacing makes it a JSON failure that try can catch.
        with path.open(encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line; skip it

    @staticmethod
    def _meta_event(
        session_id: str,
        model_ref: str,
        scope: str,
        origin: SessionOrigin,
        now: str,
        forked_from: str,
    ) -> dict[str, Any]:
        return {
            "type": SessionEventType.META,
            "id": session_id,
            "created_at": now,
            "model_ref": model_ref,
            "scope": scope,
            "origin": origin,
            "forked_from": forked_from,
            "ts": now,
        }

    def create(self, model_ref: str, scope: str, origin: SessionOrigin = SessionOrigin.TUI) -> str:
        """Start a new session owned by *origin* and return its id."""
        session_id = uuid4().hex
        self._dir.mkdir(parents=True, exist_ok=True)
        meta = self._meta_event(session_id, model_ref, scope, origin, self._now(), forked_from="")
        self._write_event(self._path(session_id), meta)
        return session_id

    def fork(
        self,
        session_id: str,
        *,
        message_count: int | None = None,
        origin: SessionOrigin = SessionOrigin.TUI,
    ) -> str:
        """Copy the first *message_count* messages (all when None) into a new session.

        The fork is owned by *origin*, which must be allowed to append to the
        source. The source is read once and never written.
        """
        source = self.get(session_id)
        if not _may_append(origin, source.meta.origin):
            raise SessionOwnershipError(session_id, source.meta.origin, origin)
        count = _fork_count(source, message_count)
        fork_id = uuid4().hex
        events = self._fork_events(fork_id, source, count, origin)
        write_private_text(self._path(fork_id), "".join(json.dumps(e) + "\n" for e in events))
        return fork_id

    def _fork_events(
        self, fork_id: str, source: Session, count: int, origin: SessionOrigin
    ) -> list[dict[str, Any]]:
        """A fork's whole log; the title event is last so ``updated_at`` is the fork time."""
        now = self._now()
        meta = source.meta
        events = [self._meta_event(fork_id, meta.model_ref, meta.scope, origin, now, meta.id)]
        events += [_message_event(message, message.ts) for message in source.messages[:count]]
        if count == len(source.messages) and source.summary:
            events.append({"type": SessionEventType.SUMMARY, "summary": source.summary, "ts": now})
        title = _fork_title(meta.title, self._fork_number(meta.id))
        events.append(
            {"type": SessionEventType.TITLE, "title": title, "source": TitleSource.AUTO, "ts": now}
        )
        return events

    def _fork_number(self, source_id: str) -> int:
        """1 plus the number of existing forks of *source_id*."""
        return 1 + sum(1 for meta in self.list() if meta.forked_from == source_id)

    def add_message(
        self, session_id: str, message: SessionMessage, *, surface: SessionOrigin | None = None
    ) -> None:
        """Append one message event to an existing session.

        With *surface* given, the append is refused unless that surface owns the
        session (see ``transfer``); without it the caller is a library embedder
        that manages its own store and ownership does not apply.
        """
        path = self._require(session_id)
        if surface is not None:
            meta = self._meta_for(path)
            if meta is not None and not _may_append(surface, meta.origin):
                raise SessionOwnershipError(session_id, meta.origin, surface)
        self._write_event(path, _message_event(message, self._now()))

    def transfer(self, session_id: str, origin: SessionOrigin) -> None:
        """Append an origin event handing the session to *origin*; newest wins.

        This is the explicit bridge between the human and agent domains: an
        agent claims a session whose id the user handed it, and POST /claim
        brings one back. Never implicit in an append.
        """
        self._write_event(
            self._require(session_id),
            {"type": SessionEventType.ORIGIN, "origin": origin, "ts": self._now()},
        )

    def set_title(self, session_id: str, title: str, source: TitleSource) -> None:
        """Append a title event; the newest title wins on read."""
        self._write_event(
            self._require(session_id),
            {"type": SessionEventType.TITLE, "title": title, "source": source, "ts": self._now()},
        )

    def set_summary(self, session_id: str, summary: str) -> None:
        """Append a summary event; the newest summary wins on read.

        Compaction folds the oldest turns into a summary once they no longer fit
        the prompt. The messages themselves stay in the log untouched: the
        transcript the user scrolls is always complete, and only what is fed to
        the model is condensed.
        """
        self._write_event(
            self._require(session_id),
            {"type": SessionEventType.SUMMARY, "summary": summary, "ts": self._now()},
        )

    def delete(self, session_id: str) -> None:
        """Remove a session's file."""
        self._require(session_id).unlink()

    def get(self, session_id: str) -> Session:
        """Replay a session's log into its reconstructed view."""
        meta, messages, summary = self._replay(
            session_id, self._require(session_id), collect_messages=True
        )
        return Session(meta=meta, messages=messages, summary=summary)

    def list(self, origins: frozenset[SessionOrigin] | None = None) -> list[SessionMeta]:
        """Sessions' metadata, newest first; *origins* narrows to those surfaces.

        Listing replays every event of every session, so it is the one hot path
        here: the drawer runs it on open. Messages are not materialised (only
        counted), and each file's meta is memoised against its size and mtime so
        reopening a vault that has not changed costs one stat() per session.
        """
        if not self._dir.exists():
            return []
        paths = list(self._dir.glob("*.jsonl"))
        metas = [meta for meta in (self._meta_for(path) for path in paths) if meta is not None]
        if origins is not None:
            metas = [meta for meta in metas if meta.origin in origins]
        # Drop cache entries for sessions that no longer exist, so a long-lived
        # store does not pin the meta of every session ever deleted.
        live = {path for path in paths}
        self._meta_cache = {p: v for p, v in self._meta_cache.items() if p in live}
        return sorted(metas, key=lambda meta: (meta.updated_at, meta.id), reverse=True)

    def _meta_for(self, path: Path) -> SessionMeta | None:
        """Meta for one session file, reusing the last fold when it is unchanged.

        The log is append-only, so any new event grows the file: size plus mtime
        is enough to notice a change. A file that grows between the stat and the
        read is simply re-folded on the next list(), never served stale.

        Returns None when the file goes away underneath us, which is routine: the
        CLI or another surface can delete a session while the drawer is listing.
        Reading it instead would raise straight out of list().
        """
        try:
            stat = path.stat()
        except OSError:
            return None
        cached = self._meta_cache.get(path)
        if cached is not None and cached[0] == stat.st_size and cached[1] == stat.st_mtime:
            return cached[2]
        try:
            meta = self._replay(path.stem, path, collect_messages=False)[0]
        except OSError:
            return None
        self._meta_cache[path] = (stat.st_size, stat.st_mtime, meta)
        return meta

    def _replay(
        self, session_id: str, path: Path, *, collect_messages: bool
    ) -> tuple[SessionMeta, tuple[SessionMessage, ...], str]:
        """Fold a session's event log into its meta, messages and summary.

        ``collect_messages=False`` is for listing, which needs only the count:
        building a SessionMessage per message across a whole vault is pure waste.
        """
        created_at = ""
        model_ref = ""
        scope = ""
        title = UNTITLED_SESSION_TITLE
        updated_at = ""
        summary = ""
        origin = SessionOrigin.TUI
        forked_from = ""
        message_count = 0
        messages: list[SessionMessage] = []
        for event in self._iter_events(path):
            ts = event.get("ts", "")
            updated_at = ts
            event_type = event.get("type")
            if event_type == SessionEventType.META:
                created_at = event["created_at"]
                model_ref = event["model_ref"]
                scope = event["scope"]
                origin = SessionOrigin(event.get("origin", SessionOrigin.TUI))
                forked_from = event.get("forked_from", "")
            elif event_type == SessionEventType.ORIGIN:
                origin = SessionOrigin(event["origin"])
            elif event_type == SessionEventType.TITLE:
                title = event["title"]
            elif event_type == SessionEventType.SUMMARY:
                summary = event["summary"]
            elif event_type == SessionEventType.MESSAGE:
                message_count += 1
                if collect_messages:
                    messages.append(_message_from_event(event, ts))
        meta = SessionMeta(
            id=session_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            model_ref=model_ref,
            scope=scope,
            message_count=message_count,
            origin=origin,
            forked_from=forked_from,
        )
        return meta, tuple(messages), summary
