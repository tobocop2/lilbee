"""The /fork slash command: picker, switch, and every refusal."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import OptionList

from lilbee.app.services import get_services, set_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.widgets.fork_picker import ForkPicker
from lilbee.cli.tui.widgets.message import AssistantMessage
from lilbee.cli.tui.widgets.notice_dialog import NoticeDialog
from lilbee.core.config import cfg
from lilbee.retrieval.query.compaction import CompactionResult
from lilbee.retrieval.reasoning import StreamToken
from lilbee.sessions import MessageRole, SessionMessage, SessionOrigin, TitleSource
from tests._lilbee_app_test_host import await_chat, pump_until
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def _services():
    store = MagicMock()
    store.get_sources.return_value = []
    set_services(make_mock_services(store=store))
    yield
    set_services(None)


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    with (
        patch("lilbee.cli.tui.app.chat_ready", return_value=True),
        patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False),
        patch("lilbee.cli.tui.widgets.model_bar.ModelBar.on_mount"),
    ):
        yield


@pytest.fixture
def sessions():
    return get_services().session_store


def _seed(store) -> str:
    """Q1, A1, Q2, A2 under the title "Torque"."""
    session_id = store.create(model_ref=cfg.chat_model, scope="both")
    store.set_title(session_id, "Torque", TitleSource.AUTO)
    for role, content in (
        (MessageRole.USER, "Q1"),
        (MessageRole.ASSISTANT, "A1"),
        (MessageRole.USER, "Q2\nsecond line"),
        (MessageRole.ASSISTANT, "A2"),
    ):
        store.add_message(session_id, SessionMessage(role=role, content=content))
    return session_id


async def _submit(pilot, text: str) -> None:
    """Type *text* into the chat input and press enter."""
    keys = ["slash" if ch == "/" else ch for ch in text]
    await pilot.press(*keys, "enter")
    await pilot.pause()


async def _open_picker(app, pilot) -> ForkPicker:
    await _submit(pilot, "/fork")
    assert await pump_until(pilot, lambda: isinstance(app.screen, ForkPicker))
    picker = app.screen
    assert isinstance(picker, ForkPicker)
    return picker


def _notified(mock_notify: MagicMock) -> list[str]:
    return [str(call.args[0]) for call in mock_notify.call_args_list]


def _row(answer: str, question: str) -> str:
    """The text of the picker row for *answer*, with the question it answers under it."""
    return f"{answer}\n{msg.FORK_PICKER_ANSWER_TO.format(question=question)}"


def _labels(picker: ForkPicker) -> list[str]:
    rows = picker.query_one("#fork-list", OptionList)
    return [str(rows.get_option_at_index(i).prompt) for i in range(rows.option_count)]


async def test_picker_lists_each_answer_newest_first(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        picker = await _open_picker(app, pilot)
        assert _labels(picker) == [_row("A2", "Q2"), _row("A1", "Q1")]


async def test_picking_an_answer_forks_through_it_and_leaves_the_input_empty(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await _open_picker(app, pilot)
        with patch.object(screen, "notify") as notify:
            await pilot.press("down", "enter")
            assert await pump_until(pilot, lambda: screen.session_id not in (None, source))
        fork = sessions.get(screen.session_id)
        assert fork.meta.forked_from == source
        assert [m.content for m in fork.messages] == ["Q1", "A1"]
        assert screen._chat_input.value == ""
        assert _notified(notify) == [msg.FORK_DONE.format(title="Torque (fork 1)")]
        assert len(screen.query(AssistantMessage)) == 1
        assert sessions.get(source).meta.message_count == 4


async def test_a_fork_from_normal_mode_lands_in_an_empty_input_in_insert_mode(sessions):
    """The palette runs /fork from NORMAL mode with a draft in the input; the fork clears it."""
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.press(*"draft")
        assert await pump_until(pilot, lambda: screen._chat_input.value == "draft")
        await pilot.press("escape")
        assert await pump_until(pilot, lambda: not screen._insert_mode)
        await pilot.press("ctrl+p")
        await pilot.pause()
        await pilot.press(*"/fork", "enter")
        assert await pump_until(pilot, lambda: isinstance(app.screen, ForkPicker))
        await pilot.press("enter")
        assert await pump_until(pilot, lambda: screen.session_id not in (None, source))
        assert screen._chat_input.value == ""
        assert screen._insert_mode
        assert screen._chat_input.has_focus


async def test_the_latest_answer_forks_everything(sessions):
    source = _seed(sessions)
    sessions.set_summary(source, "notes")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await _open_picker(app, pilot)
        await pilot.press("enter")
        assert await pump_until(pilot, lambda: screen.session_id not in (None, source))
        fork = sessions.get(screen.session_id)
        assert fork.meta.message_count == 4
        assert fork.summary == "notes"
        assert screen._summary == "notes"
        assert screen._chat_input.value == ""


async def test_a_question_with_no_answer_yet_is_not_a_fork_point(sessions):
    source = _seed(sessions)
    sessions.add_message(source, SessionMessage(role=MessageRole.USER, content="Q3"))
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        picker = await _open_picker(app, pilot)
        assert _labels(picker) == [_row("A2", "Q2"), _row("A1", "Q1")]
        await pilot.press("enter")
        assert await pump_until(pilot, lambda: screen.session_id not in (None, source))
        assert sessions.get(screen.session_id).meta.message_count == 4


async def test_an_answer_with_no_question_before_it_shows_only_the_answer(sessions):
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    for role, content in (
        (MessageRole.ASSISTANT, "Welcome"),
        (MessageRole.USER, "Q1"),
        (MessageRole.ASSISTANT, "A1"),
    ):
        sessions.add_message(session_id, SessionMessage(role=role, content=content))
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(session_id)
        picker = await _open_picker(app, pilot)
        assert _labels(picker) == [_row("A1", "Q1"), "Welcome"]
        await pilot.press("down", "enter")
        assert await pump_until(pilot, lambda: screen.session_id not in (None, session_id))
        assert sessions.get(screen.session_id).meta.message_count == 1


async def test_fork_with_no_answer_yet_says_there_is_nothing_to_fork(sessions):
    source = sessions.create(model_ref=cfg.chat_model, scope="both")
    sessions.add_message(source, SessionMessage(role=MessageRole.USER, content="Q1"))
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "/fork")
        assert _notified(notify) == [msg.FORK_NO_ANSWER]
        assert not isinstance(app.screen, ForkPicker)
        assert [meta.id for meta in sessions.list()] == [source]


async def test_escape_closes_the_picker_without_forking(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await _open_picker(app, pilot)
        await pilot.press("escape")
        assert await pump_until(pilot, lambda: not isinstance(app.screen, ForkPicker))
        assert screen.session_id == source
        assert len(sessions.list()) == 1


async def test_picker_positions_come_from_the_log_after_compaction(sessions):
    """Compaction trims the in-memory history; the fork point must still be a log index."""
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        cfg.chat_n_ctx_target = 512
        cfg.chat_compaction = True
        # The live history starts on an answer, so after the fold its questions
        # sit at other indices, with other text, than the log's.
        with screen._history_lock:
            screen._history = [
                {
                    "role": "assistant" if i % 2 == 0 else "user",
                    "content": f"live {i} " + "x" * 1200,
                }
                for i in range(6)
            ]
        with patch.object(
            get_services().searcher,
            "summarize_history",
            return_value=CompactionResult(summary="NOTES", condensed=4, stranded=0),
        ):
            threading.Thread(
                target=screen._compact_history,
                args=(source, screen._conversation_generation),
            ).start()
            assert await pump_until(pilot, lambda: screen._summary == "NOTES")
        assert [m["role"] for m in screen._history] != ["user", "assistant", "user", "assistant"]
        picker = await _open_picker(app, pilot)
        assert _labels(picker) == [_row("A2", "Q2"), _row("A1", "Q1")]
        await pilot.press("down", "enter")
        assert await pump_until(pilot, lambda: screen.session_id not in (None, source))
        assert sessions.get(screen.session_id).meta.message_count == 2


async def test_fork_while_streaming_is_refused_as_busy(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        screen.streaming = True
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "/fork")
        assert msg.CHAT_BUSY in _notified(notify)
        assert not isinstance(app.screen, ForkPicker)
        screen.streaming = False


async def test_fork_with_sessions_off_shows_the_sessions_notice(sessions, monkeypatch):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        monkeypatch.setattr(cfg, "sessions_enabled", False)
        await _submit(pilot, "/fork")
        assert await pump_until(pilot, lambda: isinstance(app.screen, NoticeDialog))


async def test_fork_before_the_first_message_says_there_is_nothing_to_fork(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "/fork")
        assert _notified(notify) == [msg.FORK_NO_SESSION]
        assert not isinstance(app.screen, ForkPicker)


async def test_fork_of_a_deleted_session_says_it_is_gone(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        sessions.delete(source)
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "/fork")
        assert _notified(notify) == [msg.FORK_SESSION_GONE]
        assert not isinstance(app.screen, ForkPicker)


async def test_a_session_deleted_while_the_picker_is_open_says_it_is_gone(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await _open_picker(app, pilot)
        sessions.delete(source)
        with patch.object(screen, "notify") as notify:
            await pilot.press("enter")
            assert await pump_until(pilot, lambda: not isinstance(app.screen, ForkPicker))
        assert _notified(notify) == [msg.FORK_SESSION_GONE]
        assert sessions.list() == []
        assert screen.session_id == source


async def _join(pilot, thread: threading.Thread) -> None:
    """Wait for *thread* while pumping the app, which its call_from_thread hops need."""
    assert await pump_until(pilot, lambda: not thread.is_alive())


class _Reply:
    """An answer that streams *first*, holds until released, then streams *rest*."""

    def __init__(self, first: str, rest: str = "", *, held: bool = False) -> None:
        self.first = first
        self.rest = rest
        self.started = threading.Event()
        self.release = threading.Event()
        if not held:
            self.release.set()

    def __call__(self, *_args: object, **_kwargs: object):
        return self._tokens()

    def _tokens(self):
        yield StreamToken(content=self.first, is_reasoning=False)
        self.started.set()
        self.release.wait()
        if self.rest:
            yield StreamToken(content=self.rest, is_reasoning=False)


@contextmanager
def _answering(reply: _Reply):
    """Answer every question with *reply* from a ready engine."""
    with (
        patch.object(ChatScreen, "_await_chat_engine", return_value=True),
        patch.object(get_services().searcher, "ask_stream", side_effect=reply),
    ):
        try:
            yield
        finally:
            reply.release.set()


def _failing_assistant_save(store, error: Exception):
    """An add_message that saves user turns and raises *error* for the reply."""
    real_add = store.add_message

    def add(session_id, message, **kwargs):
        if message.role == MessageRole.ASSISTANT:
            raise error
        real_add(session_id, message, **kwargs)

    return add


async def _turn_ends(pilot, screen) -> None:
    assert await pump_until(pilot, lambda: not screen.streaming)


async def test_fork_in_the_gap_before_the_reply_is_saved_is_refused(sessions):
    """The busy gate stays up until the assistant reply is on disk.

    Otherwise /fork snapshots the source without the reply, and the switch
    redirects the reply into the fork.
    """
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.pause()
        persist_started, release = threading.Event(), threading.Event()
        real_add = sessions.add_message

        def slow_add(session_id, message, **kwargs):
            if message.role == MessageRole.ASSISTANT:
                persist_started.set()
                release.wait()
            real_add(session_id, message, **kwargs)

        with (
            _answering(_Reply("A3")),
            patch.object(sessions, "add_message", side_effect=slow_add),
            patch.object(screen, "notify") as notify,
        ):
            try:
                await _submit(pilot, "Q3")
                assert await pump_until(pilot, persist_started.is_set)
                await _submit(pilot, "/fork")
            finally:
                release.set()
            await _turn_ends(pilot, screen)
        assert msg.CHAT_BUSY in _notified(notify)
        assert not isinstance(app.screen, ForkPicker)
        assert sessions.get(source).messages[-1].content == "A3"
        assert len(sessions.list()) == 1


async def test_the_reply_lands_in_the_session_its_turn_started_in(sessions):
    source = _seed(sessions)
    other = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.pause()
        reply = _Reply("A3", held=True)
        with _answering(reply):
            await _submit(pilot, "Q3")
            assert await pump_until(pilot, reply.started.is_set)
            screen._session_id = other
            reply.release.set()
            await _turn_ends(pilot, screen)
        assert sessions.get(source).messages[-1].content == "A3"
        assert sessions.get(other).meta.message_count == 4


async def test_a_failed_save_still_finishes_the_turn_and_says_so(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.pause()
        add = _failing_assistant_save(sessions, OSError("disk full"))
        with (
            _answering(_Reply("A3")),
            patch.object(sessions, "add_message", side_effect=add),
            patch.object(screen, "notify") as notify,
            patch.object(AssistantMessage, "finish") as finish,
        ):
            await _submit(pilot, "Q3")
            await _turn_ends(pilot, screen)
            await pilot.pause()
        finish.assert_called_once()
        assert _notified(notify) == [msg.SESSIONS_SAVE_FAILED.format(error="disk full")]


async def test_an_unexpected_save_error_still_clears_the_busy_gate(sessions):
    """The turn's end is posted from the worker's outermost finally, so a crash still ends it."""
    source = _seed(sessions)
    threads: list[threading.Thread] = []
    body = ChatScreen._stream_response.__wrapped__

    def start_on_a_thread(self, turn, chunk_type):
        thread = threading.Thread(target=body, args=(self, turn, chunk_type))
        thread.start()
        threads.append(thread)

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.pause()
        errors: list[BaseException] = []
        with (
            _answering(_Reply("A3")),
            patch.object(ChatScreen, "_stream_response", start_on_a_thread),
            patch.object(
                sessions,
                "add_message",
                side_effect=_failing_assistant_save(sessions, RuntimeError("bug")),
            ),
            patch("threading.excepthook", lambda args: errors.append(args.exc_value)),
        ):
            await _submit(pilot, "Q3")
            assert threads, "the turn must have started"
            await _join(pilot, threads[0])
            await _turn_ends(pilot, screen)
        assert [str(e) for e in errors] == ["bug"]


async def test_fork_after_a_cancel_waits_for_the_reply_the_source_keeps(sessions):
    """A cancelled turn keeps the chat busy until its partial reply is saved.

    A fork in that window would snapshot the source without the reply the
    source then keeps.
    """
    source = _seed(sessions)
    reply = _Reply("A3-partial", " more", held=True)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with _answering(reply):
            await _submit(pilot, "Q3")
            assert await pump_until(pilot, reply.started.is_set)
            await pilot.press("ctrl+c")
            with patch.object(screen, "notify") as notify:
                await _submit(pilot, "/fork")
            assert not isinstance(app.screen, ForkPicker)
            refused = _notified(notify)
            reply.release.set()
            await _turn_ends(pilot, screen)
        assert sessions.get(source).messages[-1].content == "A3-partial"
        # The refused /fork stays in the input, so a second Enter sends it.
        await pilot.press("enter")
        assert await pump_until(pilot, lambda: isinstance(app.screen, ForkPicker))
        await pilot.press("enter")
        assert await pump_until(pilot, lambda: screen.session_id not in (None, source))
        assert refused == [msg.CHAT_STOPPING]
        assert sessions.get(screen.session_id).messages[-1].content == "A3-partial"


async def test_fork_during_a_fold_after_cancel_is_refused(sessions):
    """The fold runs inside the turn, so a cancel mid-fold still refuses /fork.

    A fork in that window would take the source's summary into a partial fork.
    """
    source = _seed(sessions)
    folding, release = threading.Event(), threading.Event()

    def slow_summarize(*_args, **_kwargs):
        folding.set()
        release.wait()
        return CompactionResult(summary="NOTES", condensed=4, stranded=0)

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.pause()
        cfg.chat_n_ctx_target = 512
        cfg.chat_compaction = True
        with screen._history_lock:
            screen._history = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 1200}
                for i in range(6)
            ]
        with (
            _answering(_Reply("")),
            patch.object(get_services().searcher, "summarize_history", side_effect=slow_summarize),
            patch.object(screen, "notify") as notify,
        ):
            try:
                await _submit(pilot, "Q3")
                assert await pump_until(pilot, folding.is_set)
                await pilot.press("ctrl+c")
                await _submit(pilot, "/fork")
            finally:
                release.set()
            await _turn_ends(pilot, screen)
        assert msg.CHAT_STOPPING in _notified(notify)
        assert not isinstance(app.screen, ForkPicker)
        assert [meta.id for meta in sessions.list()] == [source]


async def test_a_turn_stopped_before_it_ran_ends_and_lets_fork_proceed(sessions):
    """Stopped before its body started, the turn still ends, asks nothing and saves nothing."""
    source = _seed(sessions)
    reply = _Reply("A3")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with _answering(reply):
            # One synchronous step: the stop lands before the worker first runs.
            screen._send_message("Q3")
            screen.action_cancel_stream()
            await _turn_ends(pilot, screen)
            await _open_picker(app, pilot)
        assert not reply.started.is_set()
        assert [m.content for m in sessions.get(source).messages][-1] == "Q3"


async def test_forking_a_session_an_agent_claimed_is_refused(sessions):
    """A session an agent claimed mid-chat belongs to the agent now."""
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await _open_picker(app, pilot)
        sessions.transfer(source, SessionOrigin.MCP)
        with patch.object(screen, "notify") as notify:
            await pilot.press("enter")
            assert await pump_until(pilot, lambda: not isinstance(app.screen, ForkPicker))
        assert len(_notified(notify)) == 1
        assert "belongs to the mcp surface" in _notified(notify)[0]
        assert len(sessions.list()) == 1
