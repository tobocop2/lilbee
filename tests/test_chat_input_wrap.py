"""ChatInput grows to show a wrapped prompt, not just on a literal newline."""

from __future__ import annotations

from unittest import mock

import pytest
from textual.app import ComposeResult

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import LilbeeAppHost

_LONG_PROMPT = (
    "What is the full pre-trip checklist for towing a 3,500 pound boat trailer "
    "on the highway for several hours, covering tires, brakes, bearings, and the coupler?"
)


class _ChatHost(LilbeeAppHost):
    """Mounts a real ChatScreen so chat.tcss height rules apply."""

    def __init__(self) -> None:
        super().__init__()
        from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen())


@pytest.fixture(autouse=True)
def _chat_ready_env():
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    with (
        mock.patch("lilbee.cli.tui.screens.chat.needs_setup", return_value=False),
        mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=True),
    ):
        yield


async def test_short_prompt_stays_single_row():
    app = _ChatHost()
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        chat_input = app.screen.query_one("#chat-input", ChatInput)
        chat_input.value = "hi"
        await pilot.pause()
        assert not chat_input.has_class("-multiline")


async def test_long_prompt_wraps_and_grows_the_box():
    app = _ChatHost()
    async with app.run_test(size=(56, 40)) as pilot:
        await pilot.pause()
        chat_input = app.screen.query_one("#chat-input", ChatInput)
        chat_input.value = _LONG_PROMPT
        await pilot.pause()
        assert chat_input.has_class("-multiline")
        assert chat_input.size.height > 3


async def test_grows_when_terminal_narrows_after_typing():
    app = _ChatHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        chat_input = app.screen.query_one("#chat-input", ChatInput)
        # Fits one row at 120 cols, so it starts single-row.
        chat_input.value = "a prompt that comfortably fits on a single row at a wide terminal width"
        await pilot.pause()
        assert not chat_input.has_class("-multiline")
        # Narrowing the terminal wraps it; the box must grow, not clip.
        await pilot.resize_terminal(48, 40)
        await pilot.pause()
        await pilot.pause()
        assert chat_input.has_class("-multiline")


async def test_literal_newline_still_grows():
    app = _ChatHost()
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        chat_input = app.screen.query_one("#chat-input", ChatInput)
        chat_input.value = "line one\nline two"
        await pilot.pause()
        assert chat_input.has_class("-multiline")
