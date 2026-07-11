"""The chat input is held until the model warms, not errored into a bubble."""

from __future__ import annotations

from unittest import mock

import pytest
from textual.app import ComposeResult

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF, make_mock_services
from lilbee.app.services import set_services
from lilbee.cli.tui import messages as msg
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
        mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False),
        mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=True),
    ):
        yield services


async def test_input_locked_and_submit_held_while_warming(_warming_services):
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen.chat_warming is True
        assert screen._chat_input.disabled is True
        with (
            mock.patch.object(screen, "notify") as notify,
            mock.patch.object(screen, "_send_message") as send,
        ):
            rejected = screen._reject_submit_when_busy()
        assert rejected is True
        send.assert_not_called()
        assert notify.call_args[0][0] == msg.CHAT_WARMING


async def test_ready_transition_unlocks_input(_warming_services):
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen._chat_input.disabled is True
        # The model finishes warming and can serve.
        _warming_services.provider.role_ready.return_value = True
        screen._poll_chat_warming()
        await pilot.pause()
        assert screen.chat_warming is False
        assert screen._chat_input.disabled is False


async def test_not_installed_model_does_not_trap_input(_warming_services):
    # No warm in flight (model not installed): input must stay usable, not locked.
    _warming_services.provider.role_ready.return_value = False
    _warming_services.provider.warm_progress.return_value = None
    app = _ChatHost()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen.chat_warming is False
        assert screen._chat_input.disabled is False
        assert app.screen.query_one("#chat-input", ChatInput) is screen._chat_input
