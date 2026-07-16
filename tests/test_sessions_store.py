"""Tests for the append-only JSONL session store."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from lilbee.core.config import cfg
from lilbee.sessions.store import (
    SESSIONS_DIRNAME,
    UNTITLED_SESSION_TITLE,
    MessageRole,
    Session,
    SessionMessage,
    SessionNotFoundError,
    SessionStore,
    TitleSource,
    derive_title,
)  # SESSIONS_DIRNAME / UNTITLED_SESSION_TITLE are internal, imported from the submodule


class _FakeClock:
    """Deterministic, monotonically increasing UTC clock."""

    def __init__(self) -> None:
        self._t = datetime(2026, 7, 14, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self._t += timedelta(seconds=1)
        return self._t


@pytest.fixture
def store(tmp_path) -> Iterator[SessionStore]:
    cfg.data_dir = tmp_path / "data"
    yield SessionStore(clock=_FakeClock())


def _msg(content: str, role: MessageRole = MessageRole.USER, sources=()) -> SessionMessage:
    return SessionMessage(role=role, content=content, sources=tuple(sources), ts="")


def test_create_returns_id_and_writes_meta_line(store: SessionStore, tmp_path) -> None:
    session_id = store.create(model_ref="gpt-oss-20b", scope="both")
    path = tmp_path / "data" / SESSIONS_DIRNAME / f"{session_id}.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    meta = json.loads(lines[0])
    assert meta["type"] == "meta"
    assert meta["model_ref"] == "gpt-oss-20b"
    assert meta["scope"] == "both"


def test_new_session_has_untitled_title_and_no_messages(store: SessionStore) -> None:
    session_id = store.create(model_ref="m", scope="both")
    session = store.get(session_id)
    assert isinstance(session, Session)
    assert session.meta.title == UNTITLED_SESSION_TITLE
    assert session.meta.message_count == 0
    assert session.messages == ()


def test_add_message_round_trips_role_content_sources(store: SessionStore) -> None:
    session_id = store.create(model_ref="m", scope="both")
    store.add_message(session_id, _msg("what are the specs?"))
    store.add_message(
        session_id,
        _msg("85 Nm.", role=MessageRole.ASSISTANT, sources=["manual.pdf", "specs.pdf"]),
    )
    session = store.get(session_id)
    assert session.meta.message_count == 2
    assert session.messages[0].role == MessageRole.USER
    assert session.messages[0].content == "what are the specs?"
    assert session.messages[1].role == MessageRole.ASSISTANT
    assert session.messages[1].sources == ("manual.pdf", "specs.pdf")


def test_set_title_auto_then_custom_latest_wins(store: SessionStore) -> None:
    session_id = store.create(model_ref="m", scope="both")
    store.set_title(session_id, "auto title", TitleSource.AUTO)
    assert store.get(session_id).meta.title == "auto title"
    store.set_title(session_id, "my name", TitleSource.CUSTOM)
    assert store.get(session_id).meta.title == "my name"


def test_every_mutator_appends_and_never_rewrites(store: SessionStore, tmp_path) -> None:
    session_id = store.create(model_ref="m", scope="both")
    path = tmp_path / "data" / SESSIONS_DIRNAME / f"{session_id}.jsonl"
    store.set_title(session_id, "t", TitleSource.AUTO)
    store.add_message(session_id, _msg("hi"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line_)["type"] for line_ in lines] == ["meta", "title", "message"]


def test_updated_at_is_last_event_timestamp(store: SessionStore) -> None:
    session_id = store.create(model_ref="m", scope="both")
    created = store.get(session_id).meta.created_at
    store.add_message(session_id, _msg("hi"))
    updated = store.get(session_id).meta.updated_at
    assert updated > created


def test_list_newest_first(store: SessionStore) -> None:
    first = store.create(model_ref="m", scope="both")
    second = store.create(model_ref="m", scope="both")
    store.add_message(first, _msg("bump the older one"))  # first now has the latest event
    ordered = [meta.id for meta in store.list()]
    assert ordered == [first, second]


def test_new_session_has_no_summary(store: SessionStore) -> None:
    session_id = store.create(model_ref="m", scope="both")
    assert store.get(session_id).summary == ""


def test_set_summary_is_readable_and_newest_wins(store: SessionStore) -> None:
    """Compaction appends; the latest summary is the one replayed."""
    session_id = store.create(model_ref="m", scope="both")
    store.set_summary(session_id, "they discussed torque specs")
    store.set_summary(session_id, "they discussed torque specs, then oil")
    assert store.get(session_id).summary == "they discussed torque specs, then oil"


def test_set_summary_never_rewrites_the_transcript(store: SessionStore) -> None:
    """The log the user scrolls stays whole; only the prompt is condensed."""
    session_id = store.create(model_ref="m", scope="both")
    store.add_message(session_id, _msg("first"))
    store.add_message(session_id, _msg("second", role=MessageRole.ASSISTANT))
    store.set_summary(session_id, "a summary")
    session = store.get(session_id)
    assert [m.content for m in session.messages] == ["first", "second"]
    assert session.meta.message_count == 2


def test_set_summary_on_missing_session_raises(store: SessionStore) -> None:
    with pytest.raises(SessionNotFoundError):
        store.set_summary("nope", "x")


def test_summary_is_not_carried_on_list_metadata(store: SessionStore) -> None:
    """Listing must not pay for summary text it never shows."""
    session_id = store.create(model_ref="m", scope="both")
    store.set_summary(session_id, "a long summary paragraph")
    assert not hasattr(store.list()[0], "summary")


def test_list_reflects_appends_after_an_earlier_list(store: SessionStore) -> None:
    """A second list() must see new messages, not a stale cached meta."""
    session_id = store.create(model_ref="m", scope="both")
    assert store.list()[0].message_count == 0
    store.add_message(session_id, _msg("hello"))
    assert store.list()[0].message_count == 1


def test_list_reflects_rename_after_an_earlier_list(store: SessionStore) -> None:
    """A newer title event must win over a cached meta from a previous list()."""
    session_id = store.create(model_ref="m", scope="both")
    store.set_title(session_id, "before", TitleSource.AUTO)
    assert store.list()[0].title == "before"
    store.set_title(session_id, "after", TitleSource.CUSTOM)
    assert store.list()[0].title == "after"


def test_list_drops_deleted_sessions_after_an_earlier_list(store: SessionStore) -> None:
    """A cached meta must not resurrect a session whose file is gone."""
    session_id = store.create(model_ref="m", scope="both")
    assert len(store.list()) == 1
    store.delete(session_id)
    assert store.list() == []


def test_list_skips_a_session_deleted_while_listing(store: SessionStore, tmp_path) -> None:
    """Another surface deleting a session mid-list must not raise out of list().

    The drawer lists while the CLI (or the chat's own recovery path) can delete;
    a file that vanishes between glob and read is skipped, not fatal.
    """
    keep = store.create(model_ref="m", scope="both")
    doomed = store.create(model_ref="m", scope="both")
    real_stat = Path.stat

    def stat_as_if_doomed_vanished(self: Path, *args, **kwargs):
        if self.stem == doomed:
            raise FileNotFoundError(self)
        return real_stat(self, *args, **kwargs)

    with patch.object(Path, "stat", stat_as_if_doomed_vanished):
        metas = store.list()
    assert [m.id for m in metas] == [keep]


def test_list_skips_a_session_that_vanishes_before_it_is_read(store: SessionStore) -> None:
    """The race can also land between the stat and the read; same contract."""
    keep = store.create(model_ref="m", scope="both")
    doomed = store.create(model_ref="m", scope="both")
    real_open = Path.open

    def open_as_if_doomed_vanished(self: Path, *args, **kwargs):
        if self.stem == doomed:
            raise FileNotFoundError(self)
        return real_open(self, *args, **kwargs)

    with patch.object(Path, "open", open_as_if_doomed_vanished):
        metas = store.list()
    assert [m.id for m in metas] == [keep]


def test_list_empty_when_no_sessions(store: SessionStore) -> None:
    assert store.list() == []


def test_list_empty_when_dir_absent(store: SessionStore, tmp_path) -> None:
    assert not (tmp_path / "data" / SESSIONS_DIRNAME).exists()
    assert store.list() == []


def test_delete_removes_the_file(store: SessionStore, tmp_path) -> None:
    session_id = store.create(model_ref="m", scope="both")
    store.delete(session_id)
    assert not (tmp_path / "data" / SESSIONS_DIRNAME / f"{session_id}.jsonl").exists()
    assert store.list() == []


@pytest.mark.parametrize("op", ["get", "add", "title", "delete"])
def test_unknown_id_raises(store: SessionStore, op: str) -> None:
    with pytest.raises(SessionNotFoundError):
        if op == "get":
            store.get("nope")
        elif op == "add":
            store.add_message("nope", _msg("hi"))
        elif op == "title":
            store.set_title("nope", "t", TitleSource.CUSTOM)
        else:
            store.delete("nope")


def test_torn_final_line_is_skipped(store: SessionStore, tmp_path) -> None:
    session_id = store.create(model_ref="m", scope="both")
    store.add_message(session_id, _msg("intact"))
    path = tmp_path / "data" / SESSIONS_DIRNAME / f"{session_id}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"type": "message", "role": "user", "conte')  # torn write
    session = store.get(session_id)
    assert session.meta.message_count == 1
    assert session.messages[0].content == "intact"


def test_blank_lines_are_ignored(store: SessionStore, tmp_path) -> None:
    session_id = store.create(model_ref="m", scope="both")
    path = tmp_path / "data" / SESSIONS_DIRNAME / f"{session_id}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n")
    assert store.get(session_id).meta.message_count == 0


def test_derive_title_truncates_long_first_message() -> None:
    short = derive_title("hi there")
    assert short == "hi there"
    long = derive_title("word " * 40)
    assert len(long) <= 61  # cap plus the ellipsis
    assert long.endswith("…")


def test_derive_title_uses_first_line_only() -> None:
    assert derive_title("first line\nsecond line") == "first line"


def test_derive_title_blank_falls_back_to_untitled() -> None:
    assert derive_title("   ") == UNTITLED_SESSION_TITLE


def test_default_clock_produces_iso_timestamp(tmp_path) -> None:
    cfg.data_dir = tmp_path / "data"
    real = SessionStore()  # no injected clock: exercises the real UTC clock
    session_id = real.create(model_ref="m", scope="both")
    created = real.get(session_id).meta.created_at
    datetime.fromisoformat(created)  # parses without raising
