"""A prompt sent before the engine is ready waits inside its bubble, input stays live."""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest
from textual.app import ComposeResult

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF, make_mock_services
from lilbee.app.services import set_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.chat import _engine_status_text
from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.core.config import cfg
from lilbee.providers.warm_progress import WarmPhase, WarmProgress
from tests._lilbee_app_test_host import LilbeeAppHost


class _ChatHost(LilbeeAppHost):
    def __init__(self) -> None:
        super().__init__()
        from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen())


@pytest.fixture
def _warming_services():
    """Mock services whose chat role is mid-warm (not ready)."""
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    services = make_mock_services()
    services.provider.role_ready.return_value = False
    services.provider.warm_progress.return_value = WarmProgress(phase=WarmPhase.READING_WEIGHTS)
    set_services(services)
    with (
        mock.patch("lilbee.cli.tui.screens.chat.needs_setup", return_value=False),
        mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=True),
    ):
        yield services


async def test_input_stays_live_and_submits_while_warming(_warming_services):
    """A cold engine no longer locks the input; the prompt goes through."""
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen._chat_input.disabled is False
        with mock.patch.object(screen, "notify") as notify:
            rejected = screen._reject_submit_when_busy()
        assert rejected is False
        notify.assert_not_called()


async def test_not_installed_model_does_not_trap_input(_warming_services):
    # No warm in flight (model not installed): input must stay usable, not locked.
    _warming_services.provider.role_ready.return_value = False
    _warming_services.provider.warm_progress.return_value = None
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen._chat_input.disabled is False
        assert app.screen.query_one("#chat-input", ChatInput) is screen._chat_input


async def test_await_engine_returns_at_once_when_ready(_warming_services):
    """A ready engine costs the stream worker nothing."""
    _warming_services.provider.role_ready.return_value = True
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        widget = mock.MagicMock()
        with mock.patch("lilbee.app.placement.wait_chat_ready") as waited:
            assert await asyncio.to_thread(screen._await_chat_engine, widget) is True
        waited.assert_not_called()
        widget.set_thinking_status.assert_not_called()


async def test_await_engine_paints_progress_then_proceeds(_warming_services):
    """The load's snapshots land in the bubble's thinking row, then clear."""
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        widget = mock.MagicMock()
        snapshot = WarmProgress(
            phase=WarmPhase.READING_WEIGHTS, bytes_done=1, bytes_total=4, model_ref="a/b/c.gguf"
        )

        def _wait(*, on_progress, should_abort):
            assert should_abort() is False
            on_progress(snapshot)
            return True

        with (
            mock.patch("lilbee.app.placement.chat_engine_ready", return_value=False),
            mock.patch("lilbee.app.placement.wait_chat_ready", side_effect=_wait),
            mock.patch(
                "lilbee.cli.tui.screens.chat._get_worker",
                return_value=mock.MagicMock(is_cancelled=False),
            ),
        ):
            assert await asyncio.to_thread(screen._await_chat_engine, widget) is True
        painted = [c.args[0] for c in widget.set_thinking_status.call_args_list]
        assert painted[0] == msg.ENGINE_LOADING  # labelled before the first snapshot
        assert _engine_status_text(snapshot) in painted
        assert painted[-1] == ""  # the status clears once the engine is ready


async def test_await_engine_cancelled_wait_stays_silent(_warming_services):
    """A cancelled prompt ends the wait without painting an error."""
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        widget = mock.MagicMock()
        with (
            mock.patch("lilbee.app.placement.chat_engine_ready", return_value=False),
            mock.patch("lilbee.app.placement.wait_chat_ready", return_value=False),
            mock.patch(
                "lilbee.cli.tui.screens.chat._get_worker",
                return_value=mock.MagicMock(is_cancelled=True),
            ),
        ):
            assert await asyncio.to_thread(screen._await_chat_engine, widget) is False
        widget.append_content.assert_not_called()


async def test_await_engine_failure_lands_in_the_bubble(_warming_services):
    """A failed load renders its reason and the model hint where the answer would be."""
    _warming_services.provider.warm_progress.return_value = WarmProgress(
        phase=WarmPhase.ERROR, error="out of memory"
    )
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        widget = mock.MagicMock()
        with (
            mock.patch("lilbee.app.placement.chat_engine_ready", return_value=False),
            mock.patch("lilbee.app.placement.wait_chat_ready", return_value=False),
            mock.patch(
                "lilbee.cli.tui.screens.chat._get_worker",
                return_value=mock.MagicMock(is_cancelled=False),
            ),
        ):
            assert await asyncio.to_thread(screen._await_chat_engine, widget) is False
        rendered = widget.append_content.call_args[0][0]
        assert msg.ENGINE_LOAD_FAILED.format(error="out of memory") in rendered
        assert msg.ENGINE_FAILED_HINT in rendered


async def test_await_engine_stall_reports_not_ready(_warming_services):
    """A wait that ends with no failure on record still explains itself."""
    _warming_services.provider.warm_progress.return_value = None
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        widget = mock.MagicMock()
        with (
            mock.patch("lilbee.app.placement.chat_engine_ready", return_value=False),
            mock.patch("lilbee.app.placement.wait_chat_ready", return_value=False),
            mock.patch(
                "lilbee.cli.tui.screens.chat._get_worker",
                return_value=mock.MagicMock(is_cancelled=False),
            ),
        ):
            assert await asyncio.to_thread(screen._await_chat_engine, widget) is False
        assert widget.append_content.call_args[0][0] == msg.ENGINE_NOT_READY


async def test_warm_tip_shows_once_and_only_when_warm_is_off(_warming_services, monkeypatch):
    """The keep-warm tip toasts on the first cold wait, never again, never when warm."""
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        monkeypatch.setattr(cfg, "keep_engine_warm", False)
        with mock.patch.object(screen, "notify") as notify:
            await asyncio.to_thread(screen._show_warm_tip_once)
            await asyncio.to_thread(screen._show_warm_tip_once)
        assert notify.call_count == 1
        assert notify.call_args[0][0] == msg.ENGINE_WARM_TIP

        screen._warm_tip_shown = False
        monkeypatch.setattr(cfg, "keep_engine_warm", True)
        with mock.patch.object(screen, "notify") as notify:
            await asyncio.to_thread(screen._show_warm_tip_once)
        notify.assert_not_called()


def test_engine_status_text_reports_bytes_or_phase():
    """Byte progress renders as a percentage; other phases fall back to the label."""
    reading = WarmProgress(
        phase=WarmPhase.READING_WEIGHTS, bytes_done=1, bytes_total=2, model_ref="a/b/c.gguf"
    )
    assert "50%" in _engine_status_text(reading)
    assert _engine_status_text(WarmProgress(phase=WarmPhase.LOADING_ENGINE)) == msg.ENGINE_LOADING
    no_bytes = WarmProgress(phase=WarmPhase.READING_WEIGHTS)
    assert _engine_status_text(no_bytes) == msg.ENGINE_LOADING
