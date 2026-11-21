"""The model bar and Settings switch the role late, once the picked engine is saved."""

from __future__ import annotations

import threading
from typing import Any
from unittest import mock

import pytest
from textual.app import ComposeResult
from textual.widgets import OptionList

from lilbee.app import engine_service
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_commands import EngineCommandResult
from lilbee.cli.tui.screens import model_picker
from lilbee.cli.tui.screens.engine_picker import EnginePickerModal
from lilbee.cli.tui.screens.model_picker import ModelPickerModal
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.model_bar import ModelOption, ModelPickerButton
from lilbee.cli.tui.widgets.model_list import ModelList
from lilbee.core.config import cfg
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.test_tui_model_picker_engine import (  # noqa: F401 - _engines is the autouse fixture
    GGUF_REF,
    OPENVINO_ROW,
    OTHER_REF,
    _engine_step,
    _engine_tasks,
    _engines,
    _offer_openvino_for_download,
    _open,
    _PickerApp,
    _SettingsApp,
    _stored,
    _toasts,
)

THIRD_REF = "org/third/third.gguf"
OPTIONS = [
    ModelOption("Gemma", GGUF_REF),
    ModelOption("Other", OTHER_REF),
    ModelOption("Third", THIRD_REF),
]
SKIPPED = msg.MODEL_PICKER_LATE_PICK_SKIPPED.format(model=OTHER_REF)


class _BarApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield ModelPickerButton(scope="chat", button_id="late-bar")


def _no_picker(app: LilbeeAppHost) -> bool:
    return not any(isinstance(s, ModelPickerModal | EnginePickerModal) for s in app.screen_stack)


async def _search_and_pick(app: LilbeeAppHost, pilot, word: str) -> None:
    """Type *word* into the open picker and press Enter on its match."""
    assert await wait_until(pilot, lambda: isinstance(app.screen, ModelPickerModal))
    modal = app.screen
    assert await wait_until(pilot, lambda: bool(modal.query("#picker-list")))
    ml = modal.query_one("#picker-list", ModelList)
    assert await wait_until(pilot, lambda: ml.option_count > 0)
    await pilot.press(*word)

    def matched() -> bool:
        row = ml.highlighted_row()
        return row is not None and word in row.ref

    assert await wait_until(pilot, matched)
    await pilot.press("enter")


async def _pick_with_openvino(app: LilbeeAppHost, pilot, word: str) -> None:
    await _search_and_pick(app, pilot, word)
    step = await _engine_step(app, pilot)
    step.query_one(OptionList).highlighted = OPENVINO_ROW
    await pilot.press("enter")


async def _pick_with_the_saved_engine(app: LilbeeAppHost, pilot, word: str) -> None:
    await _search_and_pick(app, pilot, word)
    await _engine_step(app, pilot)
    await pilot.press("enter")


def _hold_the_install(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Offer openvino as a download whose install waits on the returned event."""
    _offer_openvino_for_download(monkeypatch)
    release = threading.Event()

    def install(_name: str, **_kw: Any) -> None:
        assert release.wait(30)

    monkeypatch.setattr(engine_service, "install", install)
    return release


def _recorder(seen: list[str | None]) -> Any:
    """A stand-in for the model write that records each ref and puts it in the role."""

    def record(_host: Any, *, key: str, ref: str | None, **_kw: Any) -> None:
        seen.append(ref)
        if ref is not None:
            setattr(cfg, key, ref)

    return record


async def _open_bar_picker(app: _BarApp, pilot) -> None:
    await pilot.pause()
    bar = app.query_one("#late-bar", ModelPickerButton)
    bar.set_options(list(OPTIONS))
    bar.open_picker()


async def _open_settings_picker(app: _SettingsApp, pilot) -> None:
    assert await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
    screen = app.screen
    assert isinstance(screen, SettingsScreen)
    screen._push_model_picker("chat_model", "chat", list(OPTIONS))


async def test_the_model_bar_keeps_the_old_model_until_the_engine_is_saved() -> None:
    seen: list[tuple[str | None, dict[str, str]]] = []

    def record(_host: Any, *, key: str, ref: str | None, **_kw: Any) -> None:
        seen.append((ref, _stored()))

    app = _BarApp()
    with (
        mock.patch("lilbee.cli.tui.widgets.model_bar.apply_model_pick", record),
        mock.patch.object(ModelPickerButton, "_focus_chat_prompt") as focus_prompt,
    ):
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_bar_picker(app, pilot)
            await _pick_with_openvino(app, pilot, "oth")
            assert await wait_until(pilot, lambda: _no_picker(app))
            focused = app.focused
            assert await wait_until(pilot, lambda: len(seen) == 2)
            assert seen == [(None, {}), (OTHER_REF, {OTHER_REF: "openvino"})]
            assert cfg.chat_model == GGUF_REF
            # Only the close of the picker hands focus to the chat prompt, not the late pick.
            assert focus_prompt.call_count == 1
            assert app.focused is focused


async def test_the_model_bar_late_pick_does_not_undo_a_newer_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _hold_the_install(monkeypatch)
    seen: list[str | None] = []
    app = _BarApp()
    with mock.patch("lilbee.cli.tui.widgets.model_bar.apply_model_pick", _recorder(seen)):
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_bar_picker(app, pilot)
            await _pick_with_openvino(app, pilot, "oth")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
            await pilot.press("y")
            assert await wait_until(pilot, lambda: len(_engine_tasks(app)) == 1)
            app.query_one("#late-bar", ModelPickerButton).open_picker()
            await _pick_with_the_saved_engine(app, pilot, "thi")
            assert await wait_until(pilot, lambda: cfg.chat_model == THIRD_REF)
            release.set()
            assert await wait_until(pilot, lambda: SKIPPED in _toasts(app))
            assert _stored() == {OTHER_REF: "openvino"}
            assert seen == [None, THIRD_REF]
            assert cfg.chat_model == THIRD_REF


async def test_settings_puts_the_model_in_the_role_once_the_engine_is_saved() -> None:
    seen: list[tuple[str | None, dict[str, str]]] = []

    def record(_host: Any, *, key: str, ref: str | None, **_kw: Any) -> None:
        seen.append((ref, _stored()))

    app = _SettingsApp()
    with mock.patch("lilbee.cli.tui.screens.settings.apply_model_pick", record):
        async with app.run_test(size=(140, 50)) as pilot:
            await _open_settings_picker(app, pilot)
            await _pick_with_openvino(app, pilot, "oth")
            assert await wait_until(pilot, lambda: _no_picker(app))
            focused = app.focused
            assert await wait_until(pilot, lambda: len(seen) == 2)
            assert seen == [(None, {}), (OTHER_REF, {OTHER_REF: "openvino"})]
            assert app.focused is focused


async def test_the_settings_late_pick_does_not_undo_a_newer_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _hold_the_install(monkeypatch)
    seen: list[str | None] = []
    app = _SettingsApp()
    with mock.patch("lilbee.cli.tui.screens.settings.apply_model_pick", _recorder(seen)):
        async with app.run_test(size=(140, 50)) as pilot:
            await _open_settings_picker(app, pilot)
            await _pick_with_openvino(app, pilot, "oth")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
            await pilot.press("y")
            assert await wait_until(pilot, lambda: len(_engine_tasks(app)) == 1)
            await _open_settings_picker(app, pilot)
            await _pick_with_the_saved_engine(app, pilot, "thi")
            assert await wait_until(pilot, lambda: cfg.chat_model == THIRD_REF)
            release.set()
            assert await wait_until(pilot, lambda: SKIPPED in _toasts(app))
            assert seen == [None, THIRD_REF]
            assert cfg.chat_model == THIRD_REF


async def test_a_refused_engine_save_keeps_the_old_model() -> None:
    app = _PickerApp()
    refused = EngineCommandResult(False, "The engine choice was not saved: disk full")
    with mock.patch(
        "lilbee.cli.tui.engine_install_flow.choose_engine_for_model", return_value=refused
    ):
        async with app.run_test(size=(120, 40)) as pilot:
            await _open(app, pilot)
            await pilot.press("enter")
            step = await _engine_step(app, pilot)
            step.query_one(OptionList).highlighted = OPENVINO_ROW
            await pilot.press("enter")
            assert await wait_until(pilot, lambda: refused.text in _toasts(app))
            await pilot.pause()
            assert app.results == [None]
            assert app.later == []
            assert cfg.engine_choices is None


def test_the_late_pick_runs_while_the_role_holds_the_model_of_the_pick() -> None:
    app, later = mock.Mock(), mock.Mock()
    model_picker._pick_if_role_unchanged(app, "chat_model", GGUF_REF, OTHER_REF, later)
    later.assert_called_once_with(OTHER_REF)
    app.notify.assert_not_called()


def test_the_late_pick_says_nothing_when_the_role_already_holds_the_pick() -> None:
    cfg.chat_model = OTHER_REF
    app, later = mock.Mock(), mock.Mock()
    model_picker._pick_if_role_unchanged(app, "chat_model", GGUF_REF, OTHER_REF, later)
    later.assert_not_called()
    app.notify.assert_not_called()
