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
    HUMAN_ORIGINS,
    SESSIONS_DIRNAME,
    TITLE_ELLIPSIS,
    TITLE_MAX_LEN,
    UNTITLED_SESSION_TITLE,
    MessageRole,
    Session,
    SessionForkRangeError,
    SessionMessage,
    SessionNotFoundError,
    SessionOrigin,
    SessionOwnershipError,
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


# --- surface ownership (sessions belong to the surface that created them) ---


def test_origin_is_stamped_at_creation_and_defaults_to_tui(store: SessionStore) -> None:
    assert store.get(store.create(model_ref="m", scope="both")).meta.origin == SessionOrigin.TUI
    sid = store.create(model_ref="m", scope="both", origin=SessionOrigin.MCP)
    assert store.get(sid).meta.origin == SessionOrigin.MCP


def test_a_file_without_an_origin_reads_as_tui(store: SessionStore, tmp_path) -> None:
    """Session files written before ownership existed have no origin field; the
    only writer back then was the TUI, so that is what they are."""
    sid = store.create(model_ref="m", scope="both")
    path = tmp_path / "data" / "sessions" / f"{sid}.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    for line in lines:
        line.pop("origin", None)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    assert store.get(sid).meta.origin == SessionOrigin.TUI


def test_appending_from_a_foreign_surface_is_refused(store: SessionStore) -> None:
    """An MCP agent must not splice its turns into a human's TUI conversation."""
    sid = store.create(model_ref="m", scope="both")  # origin: tui
    message = SessionMessage(role=MessageRole.USER, content="agent turn")
    with pytest.raises(SessionOwnershipError) as exc:
        store.add_message(sid, message, surface=SessionOrigin.MCP)
    assert exc.value.owner == SessionOrigin.TUI
    assert store.get(sid).meta.message_count == 0, "the refused turn must not land"


def test_appending_from_the_owning_surface_is_allowed(store: SessionStore) -> None:
    sid = store.create(model_ref="m", scope="both", origin=SessionOrigin.HTTP)
    store.add_message(
        sid, SessionMessage(role=MessageRole.USER, content="q"), surface=SessionOrigin.HTTP
    )
    assert store.get(sid).meta.message_count == 1


def test_append_without_a_surface_keeps_working(store: SessionStore) -> None:
    """Library callers that predate ownership pass no surface; policy is opt-in
    at the surface boundaries, not sprung on every embedder."""
    sid = store.create(model_ref="m", scope="both", origin=SessionOrigin.MCP)
    store.add_message(sid, SessionMessage(role=MessageRole.USER, content="q"))
    assert store.get(sid).meta.message_count == 1


def test_human_surfaces_append_to_each_others_sessions(store: SessionStore) -> None:
    """TUI, HTTP, and CLI are one conversation space: the same person in the
    terminal, Obsidian, or the shell needs no claim dance between them."""
    sid = store.create(model_ref="m", scope="both")  # origin: tui
    store.add_message(
        sid, SessionMessage(role=MessageRole.USER, content="q"), surface=SessionOrigin.HTTP
    )
    store.add_message(
        sid, SessionMessage(role=MessageRole.USER, content="q"), surface=SessionOrigin.CLI
    )
    assert store.get(sid).meta.message_count == 2
    assert store.get(sid).meta.origin == SessionOrigin.TUI, "no transfer involved"


def test_human_surfaces_do_not_append_to_agent_sessions(store: SessionStore) -> None:
    """The domain boundary cuts both ways: agent working state is not a
    human surface's to write into without a claim."""
    sid = store.create(model_ref="m", scope="both", origin=SessionOrigin.MCP)
    with pytest.raises(SessionOwnershipError):
        store.add_message(
            sid, SessionMessage(role=MessageRole.USER, content="q"), surface=SessionOrigin.TUI
        )


def test_list_filters_by_origin(store: SessionStore) -> None:
    """Surfaces scope their listings: human surfaces never see agent working
    state, agents never see human conversations."""
    human = store.create(model_ref="m", scope="both")
    agent = store.create(model_ref="m", scope="both", origin=SessionOrigin.MCP)
    assert [m.id for m in store.list(origins=HUMAN_ORIGINS)] == [human]
    assert [m.id for m in store.list(origins=frozenset({SessionOrigin.MCP}))] == [agent]
    assert {m.id for m in store.list()} == {human, agent}, "None means everything"


def test_transfer_claims_the_session_for_the_new_surface(store: SessionStore) -> None:
    """claim over MCP (or POST /claim back) is the explicit bridge between
    the human and agent domains."""
    sid = store.create(model_ref="m", scope="both", origin=SessionOrigin.MCP)
    store.transfer(sid, SessionOrigin.TUI)
    assert store.get(sid).meta.origin == SessionOrigin.TUI
    store.add_message(
        sid, SessionMessage(role=MessageRole.USER, content="q"), surface=SessionOrigin.TUI
    )
    with pytest.raises(SessionOwnershipError):
        store.add_message(
            sid, SessionMessage(role=MessageRole.USER, content="q"), surface=SessionOrigin.MCP
        )


def test_transfer_appends_rather_than_rewrites(store: SessionStore, tmp_path) -> None:
    sid = store.create(model_ref="m", scope="both")
    path = tmp_path / "data" / "sessions" / f"{sid}.jsonl"
    before = path.read_bytes()
    store.transfer(sid, SessionOrigin.HTTP)
    assert path.read_bytes().startswith(before), "append-only holds for origin events"


def test_concurrent_appends_do_not_interleave(store: SessionStore) -> None:
    """Two writers on one session id must serialize through the per-session
    lock: every line lands whole, every message survives."""
    import threading

    sid = store.create(model_ref="m", scope="both")
    errors: list[Exception] = []

    def writer(tag: str) -> None:
        try:
            for i in range(25):
                store.add_message(
                    sid, SessionMessage(role=MessageRole.USER, content=f"{tag}-{i} " + "x" * 400)
                )
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b", "c")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    session = store.get(sid)
    assert session.meta.message_count == 75, "every append must land exactly once"
    contents = {m.content.split()[0] for m in session.messages}
    assert contents == {f"{t}-{i}" for t in ("a", "b", "c") for i in range(25)}


def _seed_conversation(store: SessionStore, origin: SessionOrigin = SessionOrigin.TUI) -> str:
    """Q1, A1, Q2, A2 under the title "Torque", with a compaction summary."""
    sid = store.create(model_ref="m", scope="both", origin=origin)
    store.set_title(sid, "Torque", TitleSource.AUTO)
    store.add_message(sid, _msg("Q1"))
    store.add_message(sid, _msg("A1", role=MessageRole.ASSISTANT, sources=["manual.pdf"]))
    store.add_message(sid, _msg("Q2"))
    store.add_message(sid, _msg("A2", role=MessageRole.ASSISTANT))
    store.set_summary(sid, "notes on Q1")
    return sid


@pytest.mark.parametrize("message_count", [None, 4])
def test_whole_fork_copies_every_message_and_the_summary(
    store: SessionStore, message_count: int | None
) -> None:
    sid = _seed_conversation(store)
    fork = store.get(store.fork(sid, message_count=message_count))
    assert fork.messages == store.get(sid).messages
    assert fork.summary == "notes on Q1"
    assert fork.meta.message_count == 4


@pytest.mark.parametrize("message_count", [0, 2])
def test_partial_fork_copies_the_prefix_and_drops_the_summary(
    store: SessionStore, message_count: int
) -> None:
    sid = _seed_conversation(store)
    fork = store.get(store.fork(sid, message_count=message_count))
    assert fork.messages == store.get(sid).messages[:message_count]
    assert fork.summary == ""


@pytest.mark.parametrize("message_count", [-1, 5])
def test_fork_outside_the_transcript_raises_range_error(
    store: SessionStore, message_count: int
) -> None:
    sid = _seed_conversation(store)
    with pytest.raises(SessionForkRangeError) as err:
        store.fork(sid, message_count=message_count)
    assert "0 to 4" in str(err.value)
    assert [meta.id for meta in store.list()] == [sid]


def test_fork_preserves_message_timestamps(store: SessionStore) -> None:
    sid = _seed_conversation(store)
    fork = store.get(store.fork(sid))
    assert [m.ts for m in fork.messages] == [m.ts for m in store.get(sid).messages]
    assert all(m.ts for m in fork.messages)


def test_fork_records_its_source_and_copies_model_and_scope(store: SessionStore) -> None:
    sid = store.create(model_ref="qwen3:8b", scope="wiki")
    meta = store.get(store.fork(sid)).meta
    assert meta.forked_from == sid
    assert (meta.model_ref, meta.scope) == ("qwen3:8b", "wiki")


def test_a_session_that_is_not_a_fork_has_no_source(store: SessionStore) -> None:
    assert store.get(store.create(model_ref="m", scope="both")).meta.forked_from == ""


def test_a_file_without_forked_from_reads_as_not_a_fork(store: SessionStore, tmp_path) -> None:
    sid = store.create(model_ref="m", scope="both")
    path = tmp_path / "data" / SESSIONS_DIRNAME / f"{sid}.jsonl"
    meta_line = json.loads(path.read_text(encoding="utf-8"))
    del meta_line["forked_from"]
    path.write_text(json.dumps(meta_line) + "\n", encoding="utf-8")
    assert store.get(sid).meta.forked_from == ""


def test_a_long_title_is_clipped_so_the_fork_suffix_fits(store: SessionStore) -> None:
    sid = store.create(model_ref="m", scope="both")
    store.set_title(sid, "t" * TITLE_MAX_LEN, TitleSource.AUTO)
    title = store.get(store.fork(sid)).meta.title
    kept = TITLE_MAX_LEN - len(" (fork 1)") - len(TITLE_ELLIPSIS)
    assert title == "t" * kept + TITLE_ELLIPSIS + " (fork 1)"
    assert len(title) == TITLE_MAX_LEN


def test_a_short_title_is_kept_whole_in_the_fork_title(store: SessionStore) -> None:
    sid = store.create(model_ref="m", scope="both")
    fitting = "t" * (TITLE_MAX_LEN - len(" (fork 1)"))
    store.set_title(sid, fitting, TitleSource.AUTO)
    assert store.get(store.fork(sid)).meta.title == fitting + " (fork 1)"


def test_fork_titles_number_the_forks_of_one_source(store: SessionStore) -> None:
    sid = _seed_conversation(store)
    first = store.fork(sid)
    second = store.fork(sid, message_count=2)
    assert store.get(first).meta.title == "Torque (fork 1)"
    assert store.get(second).meta.title == "Torque (fork 2)"


def test_fork_is_stamped_now_and_sorts_first(store: SessionStore) -> None:
    """No summary, so only the title event, written last at fork time, can make it newest."""
    sid = store.create(model_ref="m", scope="both")
    store.add_message(sid, _msg("Q1"))
    store.add_message(sid, _msg("A1", role=MessageRole.ASSISTANT))
    store.create(model_ref="m", scope="both")
    fork_id = store.fork(sid)
    newest = store.list()[0]
    assert newest.id == fork_id
    assert newest.created_at == newest.updated_at
    assert newest.created_at > store.get(sid).meta.updated_at


def test_fork_and_source_are_independent(store: SessionStore) -> None:
    sid = _seed_conversation(store)
    fork_id = store.fork(sid, message_count=2)
    store.add_message(fork_id, _msg("Q3 in the fork"))
    store.add_message(sid, _msg("Q3 in the source"))
    assert [m.content for m in store.get(sid).messages][-1] == "Q3 in the source"
    assert [m.content for m in store.get(fork_id).messages] == ["Q1", "A1", "Q3 in the fork"]
    assert store.get(sid).meta.message_count == 5


def test_fork_never_writes_to_the_source(store: SessionStore, tmp_path) -> None:
    sid = _seed_conversation(store)
    path = tmp_path / "data" / SESSIONS_DIRNAME / f"{sid}.jsonl"
    before = path.read_bytes()
    store.fork(sid)
    assert path.read_bytes() == before


def test_fork_of_a_fork_keeps_lineage_and_prefix(store: SessionStore) -> None:
    sid = _seed_conversation(store)
    child = store.fork(sid, message_count=2)
    grandchild = store.get(store.fork(child, message_count=1))
    assert grandchild.meta.forked_from == child
    assert grandchild.meta.title == "Torque (fork 1) (fork 1)"
    assert [m.content for m in grandchild.messages] == ["Q1"]


def test_fork_is_owned_by_the_forking_surface(store: SessionStore) -> None:
    sid = _seed_conversation(store)
    fork_id = store.fork(sid, origin=SessionOrigin.HTTP)
    assert store.get(fork_id).meta.origin == SessionOrigin.HTTP


def test_fork_across_the_human_agent_boundary_is_refused(store: SessionStore) -> None:
    sid = _seed_conversation(store, origin=SessionOrigin.MCP)
    with pytest.raises(SessionOwnershipError):
        store.fork(sid, origin=SessionOrigin.HTTP)
    assert [meta.id for meta in store.list()] == [sid]


def test_fork_of_an_unknown_session_raises(store: SessionStore) -> None:
    with pytest.raises(SessionNotFoundError):
        store.fork("nope")
