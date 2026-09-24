"""A chat turn owns the busy flag: only its own finish step clears it."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage
from lilbee.core.config import cfg
from lilbee.retrieval.query.compaction import CompactionResult
from lilbee.retrieval.reasoning import StreamToken
from lilbee.sessions import MessageRole, SessionMessage
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
        patch("lilbee.app.placement.chat_engine_ready", return_value=True),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False),
        patch("lilbee.cli.tui.widgets.model_bar.ModelBar.on_mount"),
    ):
        yield


class _HeldStream:
    """An answer that streams one token, then holds until released."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.questions: list[str] = []

    def __call__(self, question: str, **_kwargs: object):
        self.questions.append(question)
        return self._tokens()

    def _tokens(self):
        yield StreamToken(content="partial", is_reasoning=False)
        self.started.set()
        self.release.wait(_WAIT_S)
        yield StreamToken(content=" tail", is_reasoning=False)


@pytest.fixture
def held():
    stream = _HeldStream()
    with patch.object(get_services().searcher, "ask_stream", side_effect=stream):
        yield stream
    stream.release.set()


async def _submit(pilot, text: str) -> None:
    keys = ["slash" if ch == "/" else ch for ch in text]
    await pilot.press(*keys, "enter")
    await pilot.pause()


def _notified(mock_notify: MagicMock) -> list[str]:
    return [str(call.args[0]) for call in mock_notify.call_args_list]


async def _start_held_turn(app, pilot, held: _HeldStream, question: str = "Q1") -> ChatScreen:
    screen = await await_chat(app, pilot)
    await _submit(pilot, question)
    assert await pump_until(pilot, held.started.is_set)
    return screen


async def _finish(pilot, screen: ChatScreen, held: _HeldStream) -> None:
    held.release.set()
    assert await pump_until(pilot, lambda: not screen.streaming)


async def test_a_cancelled_turn_keeps_the_chat_busy_until_its_body_ends(held):
    """A submit while the cancelled body still runs is refused, not started."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _start_held_turn(app, pilot, held)
        await pilot.press("ctrl+c")
        await pilot.pause()
        busy_after_cancel = screen.streaming
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "Q2")
        await _finish(pilot, screen, held)
        assert busy_after_cancel
        assert msg.CHAT_STOPPING in _notified(notify)
        assert held.questions == ["Q1"]
        assert screen._history == [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "partial"},
        ]


async def test_ctrl_c_before_the_body_starts_still_runs_its_finish_step(held):
    """The body always runs, sees the stop, and exits without asking the model."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        real_body = ChatScreen._do_stream_response
        with patch.object(
            ChatScreen, "_do_stream_response", autospec=True, side_effect=real_body
        ) as body:
            # One synchronous step: the stop lands before the worker first runs.
            screen._send_message("Q1")
            screen.action_cancel_stream()
            assert await pump_until(pilot, lambda: body.called and not screen.streaming)
        assert held.questions == []
        assert body.call_count == 1


async def test_slash_cancel_before_the_body_starts_still_runs_its_finish_step(held):
    """/cancel stops the turn cooperatively, so the body still runs and ends it."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        real_body = ChatScreen._do_stream_response
        with patch.object(
            ChatScreen, "_do_stream_response", autospec=True, side_effect=real_body
        ) as body:
            screen._send_message("Q1")
            screen.run_command("/cancel")
            assert await pump_until(pilot, lambda: body.called and not screen.streaming)
        assert held.questions == []


async def test_a_second_ctrl_c_while_stopping_quits(held):
    """The first Ctrl+C stops the answer and hides itself; the second one quits."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _start_held_turn(app, pilot, held)
        await pilot.press("ctrl+c")
        await pilot.pause()
        stopping = screen.streaming and screen.stopping
        cancel_offered = screen.check_action("cancel_stream", ())
        with patch.object(app, "exit") as exit_app:
            await pilot.press("ctrl+c")
            await pilot.pause()
        await _finish(pilot, screen, held)
        assert stopping
        assert not cancel_offered
        exit_app.assert_called_once()


async def test_a_stale_turn_leaves_the_resumed_chat_alone(held):
    """After a resume, the old turn's end shows no toast and touches no chip or bubble."""
    store = get_services().session_store
    other = store.create(model_ref=cfg.chat_model, scope="both")
    store.add_message(other, SessionMessage(role=MessageRole.USER, content="older"))
    cfg.chat_mode = "search"
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with patch.object(ChatScreen, "_embedding_ready", return_value=True):
            screen = await _start_held_turn(app, pilot, held)
            first_session = screen.session_id
            screen.resume_session(other)
            await pilot.pause()
            with (
                patch.object(screen, "notify") as notify,
                patch.object(screen, "_refresh_context_usage") as refresh_chip,
                patch.object(AssistantMessage, "finish") as finish,
            ):
                await _finish(pilot, screen, held)
                await pilot.pause()
        assert msg.CHAT_MODE_SEARCH_NO_RESULTS not in _notified(notify)
        refresh_chip.assert_not_called()
        finish.assert_not_called()
        assert store.get(first_session).messages[-1].content == "partial"
        assert screen._history == [{"role": "user", "content": "older"}]


async def test_ctrl_c_stops_only_the_answer(held):
    """An unrelated worker survives a Ctrl+C on the answer."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _start_held_turn(app, pilot, held)
        other = app.run_worker(asyncio.sleep(_WAIT_S), group="unrelated")
        await pilot.press("ctrl+c")
        await pilot.pause()
        other_cancelled = other.is_cancelled
        other.cancel()
        await _finish(pilot, screen, held)
        assert not other_cancelled


async def test_the_cancel_note_is_the_last_text_in_the_bubble(held):
    """Tokens buffered at the cancel land before the note, never after it."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _start_held_turn(app, pilot, held)
        bubble = screen.query(AssistantMessage).last()
        await pilot.press("ctrl+c")
        await pilot.pause()
        await _finish(pilot, screen, held)
        assert "".join(bubble._content_parts).endswith(msg.STREAM_CANCELLED)


async def test_a_queued_model_switch_waits_for_the_body_to_end(held):
    """The fleet restart never runs under a body that is still reading."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _start_held_turn(app, pilot, held)
        with patch.object(screen, "_reload_chat_model_worker") as reload_worker:
            screen.apply_model_change()
            await pilot.press("ctrl+c")
            await pilot.pause()
            reloaded_while_stopping = reload_worker.called
            await _finish(pilot, screen, held)
            await pilot.pause()
        assert not reloaded_while_stopping
        reload_worker.assert_called_once()


async def test_adopt_and_retry_during_a_live_turn_is_refused(held):
    """The retried question is refused with a toast and never mounted or saved."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _start_held_turn(app, pilot, held)
        with (
            patch("lilbee.app.models.adopt_embedder"),
            patch.object(screen, "notify") as notify,
        ):
            await asyncio.to_thread(screen._do_adopt_and_retry, "embedder", "Q2")
            await pilot.pause()
        questions = [m._text for m in screen.query(UserMessage)]
        await _finish(pilot, screen, held)
        assert msg.CHAT_BUSY in _notified(notify)
        assert questions == ["Q1"]
        assert held.questions == ["Q1"]


async def test_clear_leaves_unrelated_workers_running():
    """/clear starts a new conversation; it does not stop other jobs."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        other = app.run_worker(asyncio.sleep(_WAIT_S), group="unrelated")
        await _submit(pilot, "/clear")
        other_cancelled = other.is_cancelled
        other.cancel()
        assert not other_cancelled


async def test_a_turn_ends_while_the_chat_screen_is_not_shown(held):
    """The end of a turn reaches the chat screen while another view is on top."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _start_held_turn(app, pilot, held)
        app.switch_view(msg.SESSIONS_VIEW)
        assert await pump_until(pilot, lambda: app.screen is not screen)
        await _finish(pilot, screen, held)
        assert not screen.streaming


async def test_a_turn_stopped_before_its_body_starts_does_not_fold(held):
    """A stop that lands before the body runs skips the fold and its saved summary."""
    calls: list[int] = []

    def summarize(*_args: object, **_kwargs: object) -> CompactionResult:
        calls.append(1)
        return CompactionResult(summary="NOTES", condensed=4, stranded=0)

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        cfg.chat_n_ctx_target = 512
        cfg.chat_compaction = True
        with screen._history_lock:
            screen._history = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 1200}
                for i in range(6)
            ]
        with patch.object(get_services().searcher, "summarize_history", side_effect=summarize):
            # One synchronous step: the stop lands before the worker first runs.
            screen._send_message("Q1")
            screen.action_cancel_stream()
            assert await pump_until(pilot, lambda: not screen.streaming)
        assert calls == []
        assert screen._summary == ""
        assert held.questions == []


async def test_a_send_during_a_fold_after_cancel_loses_no_turn():
    """A turn sent while a cancelled fold still runs cannot fold the same prefix again.

    Every turn is afterwards either in the history or in the summary, exactly once.
    """
    folding, release = threading.Event(), threading.Event()
    asked: list[str] = []
    compacted: list[int] = []

    def summarize(dropped, summary, **_kwargs):
        # The first fold holds; any later one returns at once, so both can finish.
        if not folding.is_set():
            folding.set()
            release.wait(_WAIT_S)
        labels = "".join(f"|{m['content'][:4]}" for m in dropped)
        return CompactionResult(summary=summary + labels, condensed=len(dropped), stranded=0)

    def answer(question: str, **_kwargs: object):
        asked.append(question)
        return iter([StreamToken(content=f"A-{question}", is_reasoning=False)])

    turns = [f"m{i:02d}:" + "x" * 1200 for i in range(6)]
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        cfg.chat_n_ctx_target = 512
        cfg.chat_compaction = True
        with screen._history_lock:
            screen._history = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": content}
                for i, content in enumerate(turns)
            ]
        searcher = get_services().searcher
        real_compacted = screen._on_history_compacted

        def record_compacted(condensed: int, stranded: int) -> None:
            compacted.append(condensed)
            real_compacted(condensed, stranded)

        with (
            patch.object(searcher, "summarize_history", side_effect=summarize) as fold,
            patch.object(searcher, "ask_stream", side_effect=answer),
            patch.object(screen, "_on_history_compacted", side_effect=record_compacted),
        ):
            await _submit(pilot, "Q1")
            assert await pump_until(pilot, folding.is_set)
            await pilot.press("ctrl+c")
            await pilot.pause()
            await _submit(pilot, "Q2")
            release.set()
            assert await pump_until(pilot, lambda: not screen.streaming)
            # A refused Q2 is still the draft; Enter sends it once the chat is idle.
            await pilot.press("enter")
            assert await pump_until(pilot, lambda: "Q2" in asked and not screen.streaming)
            assert await pump_until(pilot, lambda: len(compacted) == fold.call_count)
        with screen._history_lock:
            held_by_summary = [label for label in screen._summary.split("|") if label]
            in_history = [m["content"][:4] for m in screen._history]
        everything = held_by_summary + in_history
        expected = [t[:4] for t in turns] + ["Q1", "Q2", "A-Q2"]
        assert sorted(everything) == sorted(expected)
