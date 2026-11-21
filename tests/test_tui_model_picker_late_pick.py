"""The model bar and Settings switch the role late, once the picked engine is saved."""

from __future__ import annotations

import threading
from typing import Any
from unittest import mock

import pytest
from textual.app import ComposeResult
from textual.widgets import OptionList

from lilbee.app import engine_service
from lilbee.app.settings import SettingsUpdateResult
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_commands import EngineCommandResult
from lilbee.cli.tui.pick_order import PickOrder
from lilbee.cli.tui.screens import model_picker
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.engine_picker import EnginePickerModal
from lilbee.cli.tui.screens.model_picker import ModelPickerModal
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.model_bar import ModelOption, ModelPickerButton, RoleRow
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
    """A stand-in for the model write that records each ref and puts it in the role.

    It tells the app of the change as the real write does.
    """

    def record(host: Any, *, key: str, ref: str | None, **_kw: Any) -> None:
        seen.append(ref)
        if ref is not None:
            setattr(cfg, key, ref)
            host.app.publish_settings([key])

    return record


def _store(values: dict[str, Any]) -> SettingsUpdateResult:
    """A stand-in for the config write under the real model write: cfg takes the values."""
    for key, value in values.items():
        setattr(cfg, key, value)
    return SettingsUpdateResult(updated=list(values), reindex_required=False)


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
            assert await wait_until(pilot, lambda: SKIPPED in _toasts())
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
            assert await wait_until(pilot, lambda: SKIPPED in _toasts())
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
            assert await wait_until(pilot, lambda: refused.text in _toasts())
            await pilot.pause()
            assert app.results == [None]
            assert app.later == []
            assert cfg.engine_choices is None


async def _confirm_the_download(app: LilbeeAppHost, pilot) -> None:
    assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
    await pilot.press("y")


CALLERS = [
    pytest.param(
        _BarApp,
        "lilbee.cli.tui.widgets.model_bar.apply_model_pick",
        _open_bar_picker,
        id="model-bar",
    ),
    pytest.param(
        _SettingsApp,
        "lilbee.cli.tui.screens.settings.apply_model_pick",
        _open_settings_picker,
        id="settings",
    ),
]


@pytest.mark.parametrize(("app_type", "model_write", "open_picker"), CALLERS)
async def test_a_late_pick_loses_when_the_role_is_set_back_to_its_old_model(
    monkeypatch: pytest.MonkeyPatch, app_type: Any, model_write: str, open_picker: Any
) -> None:
    release = _hold_the_install(monkeypatch)
    seen: list[str | None] = []
    app = app_type()
    with mock.patch(model_write, _recorder(seen)):
        async with app.run_test(size=(140, 50)) as pilot:
            await open_picker(app, pilot)
            await _pick_with_openvino(app, pilot, "oth")
            await _confirm_the_download(app, pilot)
            assert await wait_until(pilot, lambda: len(_engine_tasks(app)) == 1)
            await open_picker(app, pilot)
            await _pick_with_the_saved_engine(app, pilot, "thi")
            assert await wait_until(pilot, lambda: cfg.chat_model == THIRD_REF)
            await open_picker(app, pilot)
            await _pick_with_the_saved_engine(app, pilot, "fil")
            assert await wait_until(pilot, lambda: cfg.chat_model == GGUF_REF)
            release.set()
            assert await wait_until(pilot, lambda: SKIPPED in _toasts())
            assert _stored() == {OTHER_REF: "openvino"}
            assert seen == [None, THIRD_REF, GGUF_REF]
            assert cfg.chat_model == GGUF_REF


@pytest.mark.parametrize(("app_type", "model_write", "open_picker"), CALLERS)
async def test_a_late_pick_loses_to_a_pick_of_the_model_the_role_holds(
    monkeypatch: pytest.MonkeyPatch, app_type: Any, model_write: str, open_picker: Any
) -> None:
    release = _hold_the_install(monkeypatch)
    seen: list[str | None] = []
    app = app_type()
    with mock.patch(model_write, _recorder(seen)):
        async with app.run_test(size=(140, 50)) as pilot:
            await open_picker(app, pilot)
            await _pick_with_openvino(app, pilot, "oth")
            await _confirm_the_download(app, pilot)
            assert await wait_until(pilot, lambda: len(_engine_tasks(app)) == 1)
            await open_picker(app, pilot)
            await _pick_with_the_saved_engine(app, pilot, "fil")
            assert await wait_until(pilot, lambda: _no_picker(app))
            release.set()
            assert await wait_until(pilot, lambda: SKIPPED in _toasts())
            assert _stored() == {OTHER_REF: "openvino"}
            assert OTHER_REF not in seen
            assert cfg.chat_model == GGUF_REF


@pytest.mark.parametrize(("app_type", "model_write", "open_picker"), CALLERS)
async def test_the_newer_of_two_late_picks_wins(
    monkeypatch: pytest.MonkeyPatch, app_type: Any, model_write: str, open_picker: Any
) -> None:
    _offer_openvino_for_download(monkeypatch)
    first, second = threading.Event(), threading.Event()
    installs: list[threading.Event] = []

    def install(_name: str, **_kw: Any) -> None:
        gate = second if installs else first
        installs.append(gate)
        assert gate.wait(30)

    monkeypatch.setattr(engine_service, "install", install)
    seen: list[str | None] = []
    app = app_type()
    with mock.patch(model_write, _recorder(seen)):
        async with app.run_test(size=(140, 50)) as pilot:
            await open_picker(app, pilot)
            await _pick_with_openvino(app, pilot, "oth")
            await _confirm_the_download(app, pilot)
            assert await wait_until(pilot, lambda: len(_engine_tasks(app)) == 1)
            await open_picker(app, pilot)
            await _pick_with_openvino(app, pilot, "thi")
            await _confirm_the_download(app, pilot)
            assert await wait_until(pilot, lambda: _no_picker(app))
            first.set()
            assert await wait_until(pilot, lambda: SKIPPED in _toasts())
            assert cfg.chat_model == GGUF_REF
            second.set()
            assert await wait_until(pilot, lambda: cfg.chat_model == THIRD_REF)
            assert _stored() == {OTHER_REF: "openvino", THIRD_REF: "openvino"}
            assert seen == [None, None, THIRD_REF]
            assert msg.MODEL_PICKER_LATE_PICK_SKIPPED.format(model=THIRD_REF) not in _toasts()


class _VisionBarApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield ModelPickerButton(scope="vision", button_id="late-bar")


async def _a_model_write_beats_the_late_pick(
    monkeypatch: pytest.MonkeyPatch, app: LilbeeAppHost, key: str, write: Any, written: str
) -> None:
    """Pick a model whose engine installs, run *write*, then let the install end."""
    setattr(cfg, key, GGUF_REF)
    release = _hold_the_install(monkeypatch)
    with (
        mock.patch("lilbee.cli.tui.app.apply_settings_update", _store),
        mock.patch("lilbee.cli.tui.widgets.model_pick.get_services"),
    ):
        async with app.run_test(size=(140, 50)) as pilot:
            await _open_bar_picker(app, pilot)
            await _pick_with_openvino(app, pilot, "oth")
            await _confirm_the_download(app, pilot)
            assert await wait_until(pilot, lambda: len(_engine_tasks(app)) == 1)
            write(app)
            assert await wait_until(pilot, lambda: getattr(cfg, key) == written)
            release.set()
            assert await wait_until(pilot, lambda: SKIPPED in _toasts())
            assert _stored() == {OTHER_REF: "openvino"}
            assert getattr(cfg, key) == written


async def test_a_late_pick_loses_to_a_model_command_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write(app: LilbeeAppHost) -> None:
        ChatScreen._cmd_model(mock.Mock(app=app), THIRD_REF)

    await _a_model_write_beats_the_late_pick(monkeypatch, _BarApp(), "chat_model", write, THIRD_REF)


async def test_a_late_pick_loses_to_a_catalog_use_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write(app: LilbeeAppHost) -> None:
        CatalogScreen._activate(mock.Mock(app=app), "chat_model", THIRD_REF, "Using Third")

    await _a_model_write_beats_the_late_pick(monkeypatch, _BarApp(), "chat_model", write, THIRD_REF)


async def test_a_late_pick_loses_to_the_turn_off_pill_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write(app: LilbeeAppHost) -> None:
        RoleRow.on_click(mock.Mock(app=app, _key="vision_model"), mock.Mock())

    await _a_model_write_beats_the_late_pick(
        monkeypatch, _VisionBarApp(), "vision_model", write, ""
    )


def _app_with_picks() -> tuple[mock.Mock, PickOrder]:
    app = mock.Mock()
    app.model_picks = PickOrder()
    return app, app.model_picks


def test_the_late_pick_runs_while_it_is_the_newest_pick_of_the_role() -> None:
    (app, picks), later = _app_with_picks(), mock.Mock()
    number = picks.take("chat_model", OTHER_REF)
    picks.take("embedding_model", THIRD_REF)
    model_picker._pick_if_newest(app, "chat_model", number, OTHER_REF, later)
    later.assert_called_once_with(OTHER_REF)
    app.notify.assert_not_called()


def test_the_late_pick_says_nothing_when_the_newer_pick_is_the_same_model() -> None:
    (app, picks), later = _app_with_picks(), mock.Mock()
    number = picks.take("chat_model", OTHER_REF)
    picks.take("chat_model", OTHER_REF)
    model_picker._pick_if_newest(app, "chat_model", number, OTHER_REF, later)
    later.assert_not_called()
    app.notify.assert_not_called()
