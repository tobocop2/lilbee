"""The /fork slash command: picker, switch, prefill, and every refusal."""

from __future__ import annotations

import threading
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
from lilbee.sessions import MessageRole, SessionMessage, SessionOrigin, TitleSource
from tests._lilbee_app_test_host import await_chat, pump_until
from tests.conftest import make_mock_services

_WAIT_S = 5.0


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


async def test_picker_lists_the_whole_conversation_and_each_question(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        picker = await _open_picker(app, pilot)
        rows = picker.query_one("#fork-list", OptionList)
        labels = [str(rows.get_option_at_index(i).prompt) for i in range(rows.option_count)]
        assert labels == [
            msg.FORK_PICKER_WHOLE,
            msg.FORK_PICKER_BEFORE.format(line="Q1"),
            msg.FORK_PICKER_BEFORE.format(line="Q2"),
        ]


async def test_picking_a_question_forks_before_it_and_prefills_it(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await _open_picker(app, pilot)
        with patch.object(screen, "notify") as notify:
            await pilot.press("down", "down", "enter")
            assert await pump_until(pilot, lambda: screen.session_id not in (None, source))
        fork = sessions.get(screen.session_id)
        assert fork.meta.forked_from == source
        assert [m.content for m in fork.messages] == ["Q1", "A1"]
        assert screen._chat_input.value == "Q2\nsecond line"
        assert _notified(notify) == [msg.FORK_DONE.format(title="Torque (fork 1)")]
        assert len(screen.query(AssistantMessage)) == 1
        assert sessions.get(source).meta.message_count == 4


async def test_a_fork_from_normal_mode_lands_in_the_prefilled_input(sessions):
    """The palette runs /fork from NORMAL mode; the prefilled question must be editable."""
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.press("escape")
        assert await pump_until(pilot, lambda: not screen._insert_mode)
        await pilot.press("ctrl+p")
        await pilot.pause()
        await pilot.press(*"/fork", "enter")
        assert await pump_until(pilot, lambda: isinstance(app.screen, ForkPicker))
        await pilot.press("down", "down", "enter")
        assert await pump_until(pilot, lambda: screen.session_id not in (None, source))
        assert screen._chat_input.value == "Q2\nsecond line"
        assert screen._insert_mode
        assert screen._chat_input.has_focus


async def test_whole_conversation_forks_everything_and_leaves_the_input_empty(sessions):
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
        rows = picker.query_one("#fork-list", OptionList)
        labels = [str(rows.get_option_at_index(i).prompt) for i in range(rows.option_count)]
        assert labels == [
            msg.FORK_PICKER_WHOLE,
            msg.FORK_PICKER_BEFORE.format(line="Q1"),
            msg.FORK_PICKER_BEFORE.format(line="Q2"),
        ]
        await pilot.press("down", "down", "enter")
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


def _start_finalize(screen, session_id: str | None, reply: str) -> threading.Thread:
    """Run the worker-side end of a turn off the main thread, as production does."""
    widget = screen.query(AssistantMessage).last()
    thread = threading.Thread(
        target=screen._finalize_stream, args=(widget, [], [reply], session_id)
    )
    thread.start()
    return thread


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
                release.wait(_WAIT_S)
            real_add(session_id, message, **kwargs)

        screen.streaming = True
        with (
            patch.object(sessions, "add_message", side_effect=slow_add),
            patch.object(screen, "notify") as notify,
        ):
            worker = _start_finalize(screen, source, "A3")
            assert await pump_until(pilot, persist_started.is_set)
            await _submit(pilot, "/fork")
            release.set()
            await _join(pilot, worker)
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
        screen._session_id = other
        await _join(pilot, _start_finalize(screen, source, "A3"))
        assert sessions.get(source).messages[-1].content == "A3"
        assert sessions.get(other).meta.message_count == 4


async def test_a_failed_save_still_finishes_the_turn_and_says_so(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.pause()
        widget = screen.query(AssistantMessage).last()
        screen.streaming = True
        with (
            patch.object(sessions, "add_message", side_effect=OSError("disk full")),
            patch.object(screen, "notify") as notify,
            patch.object(widget, "finish") as finish,
        ):
            await _join(pilot, _start_finalize(screen, source, "A3"))
            await pilot.pause()
        assert not screen.streaming
        finish.assert_called_once()
        assert _notified(notify) == [msg.SESSIONS_SAVE_FAILED.format(error="disk full")]


async def test_an_unexpected_save_error_still_clears_the_busy_gate(sessions):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.pause()
        screen.streaming = True
        errors: list[BaseException] = []
        with (
            patch.object(sessions, "add_message", side_effect=RuntimeError("bug")),
            patch("threading.excepthook", lambda args: errors.append(args.exc_value)),
        ):
            await _join(pilot, _start_finalize(screen, source, "A3"))
            await pilot.pause()
        assert not screen.streaming
        assert [str(e) for e in errors] == ["bug"]


def _hold_then_finish(started: threading.Event, release: threading.Event, reply: str):
    """A stream body that is still running after a cancel, then saves *reply*."""

    def body(self, question, widget, chunk_type, *, session_id, generation):
        started.set()
        release.wait(_WAIT_S)
        self._finalize_stream(widget, [], [reply], session_id)

    return body


async def test_fork_after_a_cancel_waits_for_the_reply_the_source_keeps(sessions):
    """A cancel drops the busy gate while the worker still saves the partial reply.

    A fork in that window would snapshot the source without the reply the
    source then keeps.
    """
    source = _seed(sessions)
    started, release = threading.Event(), threading.Event()
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        body = _hold_then_finish(started, release, "A3-partial")
        with patch.object(ChatScreen, "_do_stream_response", body):
            await _submit(pilot, "Q3")
            assert await pump_until(pilot, started.is_set)
            await pilot.press("ctrl+c")
            assert await pump_until(pilot, lambda: not screen.streaming)
            with patch.object(screen, "notify") as notify:
                await _submit(pilot, "/fork")
            assert not isinstance(app.screen, ForkPicker)
            refused = _notified(notify)
            release.set()
            assert await pump_until(pilot, lambda: not screen._live_streams)
        assert sessions.get(source).messages[-1].content == "A3-partial"
        await _open_picker(app, pilot)
        await pilot.press("enter")
        assert await pump_until(pilot, lambda: screen.session_id not in (None, source))
        assert refused == [msg.FORK_WHILE_FINISHING]
        assert sessions.get(screen.session_id).messages[-1].content == "A3-partial"


async def test_fork_waits_for_every_cancelled_body_not_just_the_first(sessions):
    """After a cancel the user can start a new turn while the old body still runs."""
    source = _seed(sessions)
    held = {q: (threading.Event(), threading.Event()) for q in ("Q3", "Q4")}

    def body(self, question, widget, chunk_type, *, session_id, generation):
        started, release = held[question]
        started.set()
        release.wait(_WAIT_S)
        self._finalize_stream(widget, [], [f"{question}-partial"], session_id)

    async def fork_is_refused() -> bool:
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "/fork")
        return _notified(notify) == [msg.FORK_WHILE_FINISHING]

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with patch.object(ChatScreen, "_do_stream_response", body):
            for question in ("Q3", "Q4"):
                await _submit(pilot, question)
                assert await pump_until(pilot, held[question][0].is_set)
                await pilot.press("ctrl+c")
                assert await pump_until(pilot, lambda: not screen.streaming)
            assert await fork_is_refused()
            held["Q3"][1].set()
            assert await pump_until(pilot, lambda: len(sessions.get(source).messages) == 7)
            assert await fork_is_refused()
            held["Q4"][1].set()
            assert await pump_until(pilot, lambda: not screen._live_streams)
        await _open_picker(app, pilot)


async def test_fork_during_a_fold_after_cancel_is_refused(sessions):
    """The fold runs inside the stream worker, so a cancel mid-fold still refuses /fork.

    A fork in that window would take the source's summary into a partial fork.
    """
    source = _seed(sessions)
    folding, release = threading.Event(), threading.Event()

    def slow_summarize(*_args, **_kwargs):
        folding.set()
        release.wait(_WAIT_S)
        return CompactionResult(summary="NOTES", condensed=4, stranded=0)

    def folding_body(self, question, widget, chunk_type, *, session_id, generation):
        self._compact_history(session_id, generation)
        self._finalize_stream(widget, [], [], session_id)

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
            patch.object(ChatScreen, "_do_stream_response", folding_body),
            patch.object(get_services().searcher, "summarize_history", side_effect=slow_summarize),
            patch.object(screen, "notify") as notify,
        ):
            await _submit(pilot, "Q3")
            assert await pump_until(pilot, folding.is_set)
            await pilot.press("ctrl+c")
            assert await pump_until(pilot, lambda: not screen.streaming)
            await _submit(pilot, "/fork")
            release.set()
            assert await pump_until(pilot, lambda: not screen._live_streams)
        assert msg.FORK_WHILE_FINISHING in _notified(notify)
        assert not isinstance(app.screen, ForkPicker)
        assert [meta.id for meta in sessions.list()] == [source]


async def test_a_worker_cancelled_before_it_ran_does_not_block_fork(sessions):
    """Cancelled before its body started, the worker saves nothing, so /fork may proceed."""
    source = _seed(sessions)
    ran: list[bool] = []

    def body(self, question, widget, chunk_type, *, session_id, generation):
        ran.append(True)

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with patch.object(ChatScreen, "_do_stream_response", body):
            # One synchronous step: the worker's task is cancelled before it first runs.
            screen._send_message("Q3")
            screen._cancel_inflight_stream(msg.STREAM_CANCELLED)
            await _open_picker(app, pilot)
        assert ran == []
        assert not screen._live_streams


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
