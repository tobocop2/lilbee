"""A prompt joins the conversation and its saved session only once the engine can serve it."""

from __future__ import annotations

import threading
from collections.abc import Callable
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.chat import ChatScreen, _Turn
from lilbee.cli.tui.widgets.message import AssistantMessage
from lilbee.retrieval.reasoning import StreamToken
from tests._async_wait import wait_for_turn_end
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


def _history(screen: ChatScreen) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in screen._history]


def _bubble_text(screen: ChatScreen) -> str:
    return "".join(list(screen.query(AssistantMessage))[-1]._content_parts)


def _engine(*, ready: bool, wait: Callable[..., bool] | None = None, error: str | None = None):
    """Stand in for the engine the wait loop watches; the real wait path in the screen runs."""
    return (
        patch("lilbee.app.placement.chat_engine_ready", return_value=ready),
        patch("lilbee.app.placement.request_engine_warm"),
        patch("lilbee.app.placement.chat_warm_error", return_value=error),
        patch("lilbee.app.placement.wait_chat_ready", side_effect=wait or (lambda **_kw: False)),
    )


def _answering(asked: list[tuple[str, list[dict]]]):
    def ask_stream(question, history=None, chunk_type=None):
        asked.append((question, [dict(m) for m in (history or [])]))
        yield StreamToken(content="answer", is_reasoning=False)

    return patch.object(get_services().searcher, "ask_stream", side_effect=ask_stream)


async def _send_and_settle(pilot, screen: ChatScreen, text: str) -> None:
    screen._send_message(text)
    await wait_for_turn_end(pilot, screen)


async def test_failed_load_leaves_no_turn_in_the_history():
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        p1, p2, p3, p4 = _engine(ready=False, error="out of memory")
        with p1, p2, p3, p4:
            await _send_and_settle(pilot, screen, "will not run")
        assert msg.ENGINE_LOAD_FAILED.format(error="out of memory") in _bubble_text(screen)
        assert screen._history == []


async def test_failed_load_saves_no_session_message(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        p1, p2, p3, p4 = _engine(ready=False, error="out of memory")
        with p1, p2, p3, p4:
            await _send_and_settle(pilot, screen, "will not run")
        assert sessions.list() == []
        assert screen.session_id is None


async def test_prompt_after_a_failed_load_carries_no_orphan(sessions):
    app = LilbeeApp()
    asked: list[tuple[str, list[dict]]] = []
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        p1, p2, p3, p4 = _engine(ready=False, error="out of memory")
        with p1, p2, p3, p4:
            await _send_and_settle(pilot, screen, "first try")
        with _answering(asked), patch("lilbee.app.placement.chat_engine_ready", return_value=True):
            await _send_and_settle(pilot, screen, "second try")
        assert asked == [("second try", [])]
        assert _history(screen) == [("user", "second try"), ("assistant", "answer")]
        assert [m.content for m in sessions.get(screen.session_id).messages] == [
            "second try",
            "answer",
        ]


async def test_wait_with_no_engine_leaves_no_turn(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        p1, p2, p3, p4 = _engine(ready=False, error=None)
        with p1, p2, p3, p4:
            await _send_and_settle(pilot, screen, "into the void")
        assert _bubble_text(screen) == msg.ENGINE_NOT_READY
        assert screen._history == []
        assert sessions.list() == []


async def test_stop_during_the_wait_leaves_no_turn(sessions):
    app = LilbeeApp()
    waiting = threading.Event()
    asked: list[tuple[str, list[dict]]] = []

    def hold(*, should_abort, **_kw) -> bool:
        waiting.set()
        while not should_abort():
            threading.Event().wait(0.01)
        return False

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        p1, p2, p3, p4 = _engine(ready=False, wait=hold)
        with p1, p2, p3, p4, _answering(asked):
            screen._send_message("never mind")
            assert await pump_until(pilot, waiting.is_set), "the turn must reach the wait"
            screen.action_cancel_stream()
            await wait_for_turn_end(pilot, screen)
        assert screen._history == []
        assert sessions.list() == []
        assert asked == []


async def test_stop_after_ready_keeps_the_turn(sessions):
    app = LilbeeApp()
    started = threading.Event()
    release = threading.Event()

    def ask_stream(question, history=None, chunk_type=None):
        yield StreamToken(content="partial", is_reasoning=False)
        started.set()
        release.wait()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with (
            patch("lilbee.app.placement.chat_engine_ready", return_value=True),
            patch.object(get_services().searcher, "ask_stream", side_effect=ask_stream),
        ):
            try:
                screen._send_message("stop me")
                assert await pump_until(pilot, started.is_set)
                screen.action_cancel_stream()
            finally:
                release.set()
            await wait_for_turn_end(pilot, screen)
        assert _history(screen) == [("user", "stop me"), ("assistant", "partial")]
        assert [m.content for m in sessions.get(screen.session_id).messages] == [
            "stop me",
            "partial",
        ]


async def test_reset_during_the_wait_admits_nothing(sessions):
    app = LilbeeApp()
    waiting = threading.Event()
    asked: list[tuple[str, list[dict]]] = []

    def hold(*, should_abort, **_kw) -> bool:
        waiting.set()
        while not should_abort():
            threading.Event().wait(0.01)
        return False

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        p1, p2, p3, p4 = _engine(ready=False, wait=hold)
        with p1, p2, p3, p4, _answering(asked):
            screen._send_message("old prompt")
            assert await pump_until(pilot, waiting.is_set), "the turn must reach the wait"
            screen._reset_conversation()
            await wait_for_turn_end(pilot, screen)
        assert screen._history == []
        assert sessions.list() == []
        assert asked == []


async def test_turn_admitted_when_the_engine_serves(sessions):
    app = LilbeeApp()
    asked: list[tuple[str, list[dict]]] = []

    def serve(*, should_abort, **_kw) -> bool:
        return True

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        p1, p2, p3, p4 = _engine(ready=False, wait=serve)
        with p1, p2, p3, p4, _answering(asked):
            await _send_and_settle(pilot, screen, "after the load")
        assert asked == [("after the load", [])]
        assert _history(screen) == [("user", "after the load"), ("assistant", "answer")]
        assert [m.content for m in sessions.get(screen.session_id).messages] == [
            "after the load",
            "answer",
        ]


async def test_a_refused_admission_ends_the_turn_without_asking(sessions):
    app = LilbeeApp()
    asked: list[tuple[str, list[dict]]] = []
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with (
            patch("lilbee.app.placement.chat_engine_ready", return_value=True),
            patch.object(ChatScreen, "_admit_turn_from_thread", return_value=False),
            _answering(asked),
        ):
            await _send_and_settle(pilot, screen, "refused")
        assert asked == []
        assert screen._history == []
        assert sessions.list() == []


async def test_a_turn_of_a_replaced_conversation_is_not_admitted():
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        stale = _Turn("old", MagicMock(), screen._conversation_generation - 1, None)
        assert screen._admit_turn(stale) is False
        assert screen._history == []
        assert stale.session_id is None


async def test_a_stopped_turn_is_not_admitted(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        stopped = _Turn("stopped", MagicMock(), screen._conversation_generation, None)
        stopped.stop.set()
        assert screen._admit_turn(stopped) is False
        assert screen._history == []
        assert sessions.list() == []


@pytest.mark.parametrize("error", [RuntimeError("app is not running"), OSError("queue closed")])
def test_admission_refused_when_the_call_to_the_app_fails(error):
    screen = ChatScreen.__new__(ChatScreen)
    app = MagicMock()
    app.call_from_thread.side_effect = error
    with patch.object(ChatScreen, "app", new_callable=PropertyMock, return_value=app):
        assert screen._admit_turn_from_thread(_Turn("q", MagicMock(), 0, None)) is False
    app.call_from_thread.assert_called_once()


def test_admission_is_not_asked_of_a_stopped_app():
    screen = ChatScreen.__new__(ChatScreen)
    app = MagicMock(is_running=False)
    with patch.object(ChatScreen, "app", new_callable=PropertyMock, return_value=app):
        assert screen._admit_turn_from_thread(_Turn("q", MagicMock(), 0, None)) is False
    app.call_from_thread.assert_not_called()


@pytest.mark.parametrize("error", [AttributeError("no app"), RuntimeError("no active app")])
def test_admission_refused_when_the_screen_has_no_app(error):
    screen = ChatScreen.__new__(ChatScreen)
    with patch.object(ChatScreen, "app", new_callable=PropertyMock, side_effect=error):
        assert screen._admit_turn_from_thread(_Turn("q", MagicMock(), 0, None)) is False
