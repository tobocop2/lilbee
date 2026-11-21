"""The model picker shows the engine of each model and asks for it when a model is picked."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import pytest
from textual.widgets import Checkbox, DataTable, Input, OptionList, TabbedContent

from lilbee.app import engine_service
from lilbee.app.engine_service import EngineRow, EngineState
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.engine_picker import EnginePickerModal
from lilbee.cli.tui.screens.model_picker import (
    BROWSE_CATALOG_REF,
    ModelPickerModal,
)
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.settings_engines import (
    CHOOSE_ID,
    ENGINES_PANE_ID,
    TABLE_ID,
)
from lilbee.cli.tui.widgets.model_bar import ModelOption
from lilbee.cli.tui.widgets.model_list import ModelList
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.roles import WorkerRole
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conformance.fake_adapter import BundledFakeAdapter

GGUF_REF = "org/repo/file.gguf"
OTHER_REF = "org/other/other.gguf"
MLX_REF = "org/repo#mlx"
REMOTE_REF = "openai/gpt-4o"
OPENVINO_ROW = 2


class _Gguf(BundledFakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})
    served_roles = frozenset(WorkerRole)


class _Openvino(BundledFakeAdapter):
    name = "openvino"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})


class _Mlx(BundledFakeAdapter):
    name = "mlx"
    formats = frozenset({"mlx"})
    architectures = frozenset({"mlx_arch"})


class _PickerApp(LilbeeAppHost):
    """Opens the model picker over the host and records what it returns."""

    options: ClassVar[list[ModelOption]] = []

    def __init__(self) -> None:
        super().__init__()
        self.results: list[str | None] = []

    def on_mount(self) -> None:
        self.push_screen(
            ModelPickerModal(scope="chat", options=list(self.options)), self.results.append
        )


class _SettingsApp(LilbeeAppHost):
    def on_mount(self) -> None:
        self.push_screen(SettingsScreen())


@pytest.fixture(autouse=True)
def _engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.chat_model = GGUF_REF
    cfg.embedding_model = ""
    cfg.vision_model = ""
    cfg.reranker_model = ""
    cfg.preferred_engine = EngineSetting.AUTO
    cfg.engine_choices = None
    cfg.choose_engine_on_pick = True
    fakes = (
        (EngineName.LLAMA, _Gguf()),
        (EngineName.OPENVINO, _Openvino()),
        (EngineName.MLX, _Mlx()),
    )
    for name in set(choice.ADAPTERS) - {name for name, _ in fakes}:
        monkeypatch.delitem(choice.ADAPTERS, name)  # type: ignore[attr-defined]
    for name, adapter in fakes:
        monkeypatch.setitem(choice.ADAPTERS, name, adapter)  # type: ignore[arg-type]
    monkeypatch.setattr(engine_service, "pack_backend", lambda _host: "metal")
    _PickerApp.options = [ModelOption("Gemma", GGUF_REF), ModelOption("Other", OTHER_REF)]
    yield
    for field in type(snapshot).model_fields:
        setattr(cfg, field, getattr(snapshot, field))


def _toasts(app: LilbeeAppHost) -> list[str]:
    return [n.message for n in app._notifications]


def _rows(modal: ModelPickerModal) -> list[str]:
    """The text of each row of the model list."""
    ml = modal.query_one("#picker-list", ModelList)
    return [str(ml.get_option_at_index(i).prompt) for i in range(ml.option_count)]


async def _open(app: _PickerApp, pilot) -> ModelPickerModal:
    await pilot.pause()
    modal = app.screen
    assert isinstance(modal, ModelPickerModal)
    assert await wait_until(pilot, lambda: modal.query_one("#picker-list", ModelList).row_count > 0)
    return modal


async def _engine_step(app: LilbeeAppHost, pilot) -> EnginePickerModal:
    assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
    modal = app.screen
    assert isinstance(modal, EnginePickerModal)
    return modal


async def _back_on_list(app: LilbeeAppHost, pilot) -> None:
    assert await wait_until(pilot, lambda: isinstance(app.screen, ModelPickerModal))


def _stored() -> dict[str, str]:
    return json.loads(cfg.engine_choices) if cfg.engine_choices else {}


async def test_a_row_names_its_engine_and_says_when_lilbee_chose_it() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open(app, pilot)
        first, second, browse = _rows(modal)
        assert "Gemma" in first and "engine: ■ llama.cpp (auto)" in first
        assert "Other" in second and "engine: ■ llama.cpp (auto)" in second
        assert "engine:" not in browse


async def test_a_saved_engine_shows_without_the_auto_tag() -> None:
    cfg.engine_choices = json.dumps({GGUF_REF: "openvino"})
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        first = _rows(await _open(app, pilot))[0]
        assert "engine: ■ openvino" in first
        assert "(auto)" not in first


async def test_a_preferred_engine_is_a_choice_so_it_shows_without_the_auto_tag() -> None:
    cfg.preferred_engine = EngineSetting.OPENVINO
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        first = _rows(await _open(app, pilot))[0]
        assert "engine: ■ openvino" in first
        assert "(auto)" not in first


async def test_a_model_that_no_engine_loads_says_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX)
    _PickerApp.options = [ModelOption("Gemma", MLX_REF)]
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        assert "engine: none" in _rows(await _open(app, pilot))[0]


async def test_a_remote_model_shows_no_engine_and_skips_the_engine_step() -> None:
    _PickerApp.options = [ModelOption("GPT", REMOTE_REF)]
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open(app, pilot)
        assert "engine:" not in _rows(modal)[0]
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: app.results == [REMOTE_REF])
        assert not isinstance(app.screen, EnginePickerModal)


async def test_one_model_in_two_formats_gives_two_rows_that_differ_in_text() -> None:
    _PickerApp.options = [
        ModelOption("Gemma", GGUF_REF),
        ModelOption("Gemma", MLX_REF),
        ModelOption("Other", OTHER_REF),
    ]
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        rows = _rows(await _open(app, pilot))
        assert "Gemma (gguf)" in rows[0]
        assert "Gemma (mlx)" in rows[1]
        assert "Other (" not in rows[2]


async def test_the_key_line_names_enter_and_e() -> None:
    app = _PickerApp()
    async with app.run_test(size=(80, 20)) as pilot:
        modal = await _open(app, pilot)
        hint = str(modal.query_one("#picker-hint").render())
        assert "Enter use model" in hint and "e change engine" in hint
        assert modal.query_one("#picker-hint").region.height == 1


async def test_enter_opens_the_engine_step_with_auto_highlighted_when_nothing_is_saved() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("enter")
        step = await _engine_step(app, pilot)
        assert step.query_one(OptionList).highlighted == 0
        assert app.results == []


async def test_enter_enter_keeps_the_choice_the_user_saved() -> None:
    cfg.engine_choices = json.dumps({GGUF_REF: "openvino"})
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("enter")
        step = await _engine_step(app, pilot)
        assert step.query_one(OptionList).highlighted == OPENVINO_ROW
        with mock.patch("lilbee.cli.tui.screens.model_picker.choose_for_model") as save:
            await pilot.press("enter")
            assert await wait_until(pilot, lambda: app.results == [GGUF_REF])
        save.assert_not_called()
        assert _stored() == {GGUF_REF: "openvino"}


async def test_enter_enter_with_no_saved_engine_saves_nothing() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("enter")
        await _engine_step(app, pilot)
        with mock.patch("lilbee.cli.tui.screens.model_picker.choose_for_model") as save:
            await pilot.press("enter")
            assert await wait_until(pilot, lambda: app.results == [GGUF_REF])
        save.assert_not_called()
        assert cfg.engine_choices is None


async def test_picking_another_engine_saves_it_and_returns_the_model() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("enter")
        step = await _engine_step(app, pilot)
        step.query_one(OptionList).highlighted = OPENVINO_ROW
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: app.results == [GGUF_REF])
        assert _stored() == {GGUF_REF: "openvino"}


async def test_escape_in_the_engine_step_returns_to_the_model_list_and_changes_nothing() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("enter")
        await _engine_step(app, pilot)
        await pilot.press("escape")
        await _back_on_list(app, pilot)
        assert app.results == []
        assert cfg.engine_choices is None


async def test_the_step_is_skipped_when_only_one_engine_can_load_the_file() -> None:
    _PickerApp.options = [ModelOption("Gemma", MLX_REF)]
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: app.results == [MLX_REF])
        assert cfg.engine_choices is None


async def test_with_the_setting_off_enter_returns_the_model_at_once() -> None:
    cfg.choose_engine_on_pick = False
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: app.results == [GGUF_REF])
        assert cfg.engine_choices is None


async def test_the_browse_row_returns_its_ref_without_the_engine_step() -> None:
    _PickerApp.options = []
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open(app, pilot)
        modal.query_one("#picker-list", ModelList).highlighted = 0
        modal.query_one("#picker-list", ModelList).action_select()
        assert await wait_until(pilot, lambda: app.results == [BROWSE_CATALOG_REF])


async def test_e_opens_the_engine_step_and_saves_without_picking_the_model() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open(app, pilot)
        await pilot.press("down")
        await pilot.press("up")
        assert app.focused is modal.query_one("#picker-list", ModelList)
        await pilot.press("e")
        step = await _engine_step(app, pilot)
        step.query_one(OptionList).highlighted = OPENVINO_ROW
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: _stored() == {GGUF_REF: "openvino"})
        await _back_on_list(app, pilot)
        assert await wait_until(pilot, lambda: "engine: ■ openvino" in _rows(modal)[0])
        assert app.results == []
        assert modal.query_one("#picker-list", ModelList).highlighted == 0


async def test_e_keeps_the_highlight_on_the_row_it_changed() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open(app, pilot)
        await pilot.press("down")
        await pilot.press("e")
        step = await _engine_step(app, pilot)
        step.query_one(OptionList).highlighted = OPENVINO_ROW
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: _stored() == {OTHER_REF: "openvino"})
        await _back_on_list(app, pilot)
        assert await wait_until(pilot, lambda: "engine: ■ openvino" in _rows(modal)[1])
        assert modal.query_one("#picker-list", ModelList).highlighted == 1


async def test_ctrl_e_opens_the_engine_step_from_the_search_field() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open(app, pilot)
        assert app.focused is modal.query_one("#picker-search", Input)
        await pilot.press("ctrl+e")
        await _engine_step(app, pilot)


async def test_e_with_the_saved_engine_picked_again_writes_nothing() -> None:
    cfg.engine_choices = json.dumps({GGUF_REF: "openvino"})
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("ctrl+e")
        await _engine_step(app, pilot)
        with mock.patch("lilbee.cli.tui.screens.model_picker.choose_for_model") as save:
            await pilot.press("enter")
            await _back_on_list(app, pilot)
        save.assert_not_called()
        assert _stored() == {GGUF_REF: "openvino"}
        assert not _toasts(app)


async def test_e_says_so_when_one_engine_loads_the_model_or_no_model_is_highlighted() -> None:
    _PickerApp.options = [ModelOption("Gemma", MLX_REF)]
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open(app, pilot)
        await pilot.press("ctrl+e")
        assert await wait_until(pilot, lambda: msg.MODEL_INFO_NO_ENGINE_CHOICE in _toasts(app))
        assert isinstance(app.screen, ModelPickerModal)
        modal.query_one("#picker-list", ModelList).highlighted = None
        await pilot.press("ctrl+e")
        assert not isinstance(app.screen, EnginePickerModal)


async def test_a_download_that_ends_after_the_picker_closed_does_nothing() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open(app, pilot)
        await pilot.press("escape")
        assert await wait_until(pilot, lambda: app.results == [None])
        modal._finish(GGUF_REF)
        modal._repaint()
        await pilot.pause()
        assert app.results == [None]


async def test_an_engine_that_needs_a_download_returns_the_model_only_after_the_install() -> None:
    app = _PickerApp()
    after: list[Callable[[], object]] = []

    def fake_choose(_app: Any, _ref: str, _option: Any, then: Callable[[], object]) -> None:
        after.append(then)

    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        with mock.patch("lilbee.cli.tui.screens.model_picker.choose_for_model", fake_choose):
            await pilot.press("enter")
            step = await _engine_step(app, pilot)
            step.query_one(OptionList).highlighted = OPENVINO_ROW
            await pilot.press("enter")
            assert await wait_until(pilot, lambda: len(after) == 1)
        assert app.results == []
        after[0]()
        assert await wait_until(pilot, lambda: app.results == [GGUF_REF])


async def _open_engines(app: _SettingsApp, pilot) -> Checkbox:
    screen = app.screen
    assert isinstance(screen, SettingsScreen)
    screen.query_one("#settings-tabs", TabbedContent).active = ENGINES_PANE_ID
    table = screen.query_one(f"#{TABLE_ID}", DataTable)
    assert await wait_until(pilot, lambda: table.row_count == 1)
    return screen.query_one(f"#{CHOOSE_ID}", Checkbox)


@pytest.fixture
def one_engine_row(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [EngineRow("llama.cpp", EngineState.BUNDLED, backend="metal")]
    monkeypatch.setattr(engine_service, "engine_rows", lambda *_a, **_k: list(rows))


async def test_the_engines_tab_holds_the_setting_and_it_defaults_to_on(
    one_engine_row: None,
) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(140, 50)) as pilot:
        box = await _open_engines(app, pilot)
        assert box.value is True
        assert str(box.label) == msg.ENGINES_CHOOSE_ON_PICK_LABEL


async def test_the_engines_tab_saves_the_setting(one_engine_row: None) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(140, 50)) as pilot:
        box = await _open_engines(app, pilot)
        box.value = False
        assert await wait_until(pilot, lambda: cfg.choose_engine_on_pick is False)
        stored = (cfg.data_root / "config.toml").read_text(encoding="utf-8")
        assert "choose_engine_on_pick = false" in stored


async def test_a_refused_write_puts_the_box_back(one_engine_row: None) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(140, 50)) as pilot:
        box = await _open_engines(app, pilot)
        with mock.patch.object(app, "set_setting", return_value=False):
            box.value = False
            assert await wait_until(pilot, lambda: box.value is True)
        assert cfg.choose_engine_on_pick is True


def test_the_setting_is_on_by_default() -> None:
    assert type(cfg).model_fields["choose_engine_on_pick"].default is True


async def test_the_box_follows_the_stored_value_on_reload(one_engine_row: None) -> None:
    cfg.choose_engine_on_pick = False
    app = _SettingsApp()
    async with app.run_test(size=(140, 50)) as pilot:
        box = await _open_engines(app, pilot)
        assert box.value is False


async def test_the_setting_is_hidden_from_the_models_tab(one_engine_row: None) -> None:
    from lilbee.cli.tui.screens.settings_widgets import group_settings

    keys = [key for items in group_settings().values() for key, _ in items]
    assert "choose_engine_on_pick" not in keys
