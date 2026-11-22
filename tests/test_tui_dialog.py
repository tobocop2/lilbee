"""A dialog closes once: a second answer leaves the screen below it alone."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

import pytest
from textual import events
from textual.screen import ModalScreen, Screen

import lilbee.cli.tui
from lilbee.cli.tui.screens.command_palette import LilbeeCommandPalette
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.dialog import Dialog
from lilbee.cli.tui.widgets.fork_picker import ForkPicker, ForkPoint
from lilbee.cli.tui.widgets.slash_command_catalog import SlashCommandCatalog
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost, app_reads_queue, send_key_burst


class _Below(Screen[None]):
    """Takes every key itself, so a key that comes after the dialog goes no further."""

    def on_key(self, event: events.Key) -> None:
        event.stop()


class _Host(LilbeeAppHost):
    """Holds a dialog over a screen of its own, and keeps the answers."""

    def __init__(self, dialog: Dialog[Any]) -> None:
        super().__init__()
        self.below = _Below()
        self.dialog = dialog
        self.answers: list[object] = []

    def on_mount(self) -> None:
        self.push_screen(self.below)
        self.push_screen(self.dialog, self.answers.append)


@pytest.mark.parametrize(
    ("build", "focused", "answer"),
    [
        pytest.param(lambda: ConfirmDialog("Title", "Text"), "#confirm-no", False, id="pill"),
        pytest.param(SlashCommandCatalog, "#catalog-filter", "/sessions", id="input"),
        pytest.param(
            lambda: ForkPicker([ForkPoint(2, "Answer", "Question")]), "#fork-list", 2, id="list"
        ),
    ],
)
@pytest.mark.parametrize("presses", [2, 3])
async def test_enters_pressed_before_a_dialog_closes_leave_the_screen_below(
    build: Callable[[], Dialog[Any]], focused: str, answer: object, presses: int
) -> None:
    app = _Host(build())
    async with app.run_test(size=(100, 30)) as pilot:
        # app.screen raises before the host pushes its first screen.
        assert await wait_until(pilot, lambda: app.dialog.is_mounted and app.screen is app.dialog)
        target = app.dialog.query_one(focused)
        target.focus()
        assert await wait_until(pilot, lambda: app.focused is target)
        stack = app.screen_stack
        send_key_burst(app, *["enter"] * presses)
        assert await app_reads_queue(app, pilot)
        # The dialog stops after it has read every message posted before the close.
        assert await wait_until(pilot, lambda: not app.dialog.is_running)
        assert await app_reads_queue(app, pilot)
        assert app.screen_stack == stack[:-1]
        assert app.screen is app.below
    assert app.answers == [answer]


@pytest.mark.parametrize("answers", [1, 2])
@pytest.mark.parametrize("caller", ["app", "dialog", "above", "top"])
async def test_a_covered_dialog_closes_itself_and_the_screens_above_it(
    caller: str, answers: int
) -> None:
    """A dialog can get its answer while other screens are open over it, from any of them."""
    app = _Host(ConfirmDialog("Title", "Text"))
    above = ConfirmDialog("Above", "Text")
    above_answers: list[bool | None] = []
    top: Screen[None] = Screen()
    callers = {"app": app, "dialog": app.dialog, "above": above, "top": top}
    async with app.run_test(size=(100, 30)) as pilot:
        assert await wait_until(pilot, lambda: app.dialog.is_mounted and app.screen is app.dialog)
        stack = app.screen_stack
        await app.push_screen(above, above_answers.append)
        await app.push_screen(top)

        def _answer() -> None:
            for _ in range(answers):
                app.dialog.dismiss(True)

        callers[caller].call_later(_answer)
        assert await wait_until(pilot, lambda: not app.dialog.is_running)
        assert app.screen_stack == stack[:-1]
        assert app.screen is app.below
        assert await wait_until(pilot, lambda: bool(above_answers and app.answers))
        assert not above.is_running
        assert not top.is_running
    assert app.answers == [True]
    assert above_answers == [None]


async def test_a_result_callback_answers_its_dialog_under_another_screen() -> None:
    """Two pickers open over a dialog: the answer of the top one answers the dialog."""
    app = _Host(ConfirmDialog("Title", "Text"))
    pickers = [ConfirmDialog("First", "Text"), ConfirmDialog("Second", "Text")]

    def _answered(answer: bool | None) -> None:
        if answer:
            app.dialog.dismiss(True)

    def _push() -> None:
        for picker in pickers:
            app.push_screen(picker, _answered)

    async with app.run_test(size=(100, 30)) as pilot:
        assert await wait_until(pilot, lambda: app.dialog.is_mounted and app.screen is app.dialog)
        stack = app.screen_stack
        app.dialog.call_later(_push)
        assert await wait_until(pilot, lambda: app.screen is pickers[1])
        pickers[1].dismiss(True)
        assert await wait_until(pilot, lambda: not app.dialog.is_running)
        assert app.screen_stack == stack[:-1]
        assert app.screen is app.below
        assert await wait_until(pilot, lambda: bool(app.answers))
        assert not pickers[0].is_running
    assert app.answers == [True]


def test_every_modal_of_the_tui_is_a_dialog() -> None:
    """A modal on Textual's own base closes the screen below it on a second answer."""
    for module in pkgutil.walk_packages(lilbee.cli.tui.__path__, "lilbee.cli.tui."):
        importlib.import_module(module.name)
    found: set[type[ModalScreen[Any]]] = set()
    queue: list[type[ModalScreen[Any]]] = [ModalScreen]
    while queue:
        for kind in queue.pop().__subclasses__():
            if kind not in found:
                found.add(kind)
                queue.append(kind)
    ours = {kind for kind in found if kind.__module__.startswith("lilbee.cli.tui.")}
    assert {ConfirmDialog, ForkPicker, LilbeeCommandPalette} <= ours
    assert {kind for kind in ours if kind.dismiss is not Dialog.dismiss} == set()
