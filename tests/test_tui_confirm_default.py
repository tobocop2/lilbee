"""A confirm dialog opens on No, so a bare Enter keeps the current state."""

from __future__ import annotations

from unittest import mock

from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost, await_chat


class _DialogApp(LilbeeAppHost):
    """Opens one confirm dialog over the host and records its answer."""

    def __init__(self) -> None:
        super().__init__()
        self.answers: list[bool | None] = []

    def compose(self) -> ComposeResult:
        yield Static("")

    def on_mount(self) -> None:
        self.push_screen(ConfirmDialog("Delete it", "Delete the thing?"), self.answers.append)


async def test_a_bare_enter_answers_no() -> None:
    app = _DialogApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.focused is not None and app.focused.id == "confirm-no"
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: app.answers == [False])


async def test_enter_on_the_yes_pill_answers_yes() -> None:
    app = _DialogApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("left")
        assert app.focused is not None and app.focused.id == "confirm-yes"
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: app.answers == [True])


async def test_the_y_key_answers_yes_and_escape_answers_no() -> None:
    app = _DialogApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("y")
        assert await wait_until(pilot, lambda: app.answers == [True])

    other = _DialogApp()
    async with other.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        assert await wait_until(pilot, lambda: other.answers == [False])


class _ChatApp(LilbeeAppHost):
    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


async def test_a_bare_enter_on_the_reset_prompt_wipes_nothing() -> None:
    app = _ChatApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        with mock.patch("lilbee.app.reset.perform_reset") as wipe:
            chat.request_reset()
            assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
            await pilot.press("enter")
            assert await wait_until(pilot, lambda: not isinstance(app.screen, ConfirmDialog))
            await pilot.pause()
        wipe.assert_not_called()
