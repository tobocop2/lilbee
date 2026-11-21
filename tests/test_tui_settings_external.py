"""A config.toml change another process writes reaches cfg and an open Settings screen."""

from __future__ import annotations

import threading
from contextlib import contextmanager

from textual import events
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Input, TabbedContent

from lilbee.app.settings import adopt_external_changes, read_config_state
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import LilbeeAppHost, pump_until
from tests.test_tui_settings_perf import _make_eager_settings_screen

SIZE = (80, 24)
RETRIEVAL = "settings-tab-retrieval"
OCR = "settings-tab-ocr_tuning"


class _App(LilbeeAppHost):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(_make_eager_settings_screen()())


def _write_from_another_process(key: str, value: object) -> None:
    """Write config.toml the way the CLI does, leaving this process's cfg alone."""
    persistent_settings.set_value(cfg.data_root, key, value)


async def _open_retrieval(app: _App, pilot) -> tuple[SettingsScreen, Input]:
    assert await pump_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
    screen = app.screen
    assert isinstance(screen, SettingsScreen)
    assert await pump_until(pilot, lambda: bool(screen.query(f"#{RETRIEVAL} Input")))
    tabs = screen.query_one("#settings-tabs", TabbedContent)
    tabs.active = RETRIEVAL
    assert await pump_until(pilot, lambda: tabs.active == RETRIEVAL)
    field = screen.query_one("#ed-top_k", Input)
    assert await pump_until(pilot, lambda: field.value == str(cfg.top_k)), field.value
    return screen, field


async def test_a_tab_change_shows_a_value_another_process_wrote():
    cfg.top_k = 12
    app = _App()
    async with app.run_test(size=SIZE) as pilot:
        screen, field = await _open_retrieval(app, pilot)
        _write_from_another_process("top_k", 31)
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        tabs.active = OCR
        assert await pump_until(pilot, lambda: field.value == "31"), field.value
        assert cfg.top_k == 31


async def test_a_screen_resume_shows_a_value_another_process_wrote():
    cfg.top_k = 12
    app = _App()
    async with app.run_test(size=SIZE) as pilot:
        screen, field = await _open_retrieval(app, pilot)
        await app.push_screen(Screen())
        assert await pump_until(pilot, lambda: app.screen is not screen)
        _write_from_another_process("top_k", 31)
        app.pop_screen()
        assert await pump_until(pilot, lambda: field.value == "31"), field.value
        assert cfg.top_k == 31


async def test_app_focus_takes_the_new_value_but_keeps_the_field_being_typed_in():
    cfg.top_k = 12
    app = _App()
    async with app.run_test(size=SIZE) as pilot:
        _screen, field = await _open_retrieval(app, pilot)
        field.focus()
        assert await pump_until(pilot, lambda: field.has_focus)
        field.value = "25"
        _write_from_another_process("top_k", 31)
        app.post_message(events.AppFocus())
        assert await pump_until(pilot, lambda: cfg.top_k == 31), cfg.top_k
        assert field.value == "25"


def test_an_unchanged_file_changes_nothing():
    reading = read_config_state()
    after, result = adopt_external_changes(reading)
    assert result is None
    assert after is reading


def test_a_process_only_value_survives_a_change_to_another_key():
    persistent_settings.set_value(cfg.data_root, "temperature", 0.7)
    persistent_settings.overlay_persisted_settings(cfg.data_root)
    reading = read_config_state()
    cfg.temperature = 0.2  # a one-off --temperature for this process
    _write_from_another_process("top_k", 31)
    after, result = adopt_external_changes(reading)
    assert result is not None
    assert result.updated == ["top_k"]
    assert cfg.top_k == 31
    assert cfg.temperature == 0.2
    again, second = adopt_external_changes(after)
    assert second is None
    assert again is after


def test_a_file_that_does_not_parse_changes_nothing():
    _write_from_another_process("top_k", 31)
    persistent_settings.overlay_persisted_settings(cfg.data_root)
    reading = read_config_state()
    (cfg.data_root / "config.toml").write_text("top_k = = 9\n", encoding="utf-8")
    after, result = adopt_external_changes(reading)
    assert result is None
    assert after is reading
    assert cfg.top_k == 31


def test_a_folder_and_a_model_the_file_stops_setting_go_back_to_their_startup_values(tmp_path):
    cfg.chat_model = "ollama/start-model"
    start_model = cfg.chat_model  # the model this process started with, say from --model
    reading = read_config_state()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _write_from_another_process("documents_dir", str(elsewhere))
    _write_from_another_process("chat_model", "ollama/file-model")
    reading, _ = adopt_external_changes(reading)
    assert cfg.documents_dir.name == "elsewhere"
    assert cfg.chat_model != start_model
    persistent_settings.delete_values(cfg.data_root, ["documents_dir", "chat_model"])
    _after, result = adopt_external_changes(reading)
    assert result is not None
    assert cfg.documents_dir == cfg.data_root / "documents"  # what a fresh start derives
    assert cfg.chat_model == start_model


def test_a_running_sync_keeps_its_source_folders_until_it_ends(tmp_path):
    folder = tmp_path / "linked"
    folder.mkdir()
    reading = read_config_state()
    with persistent_settings.pin_registry(cfg):
        _write_from_another_process("linked_roots", {"A": str(folder)})
        reading, _ = adopt_external_changes(reading)
        assert cfg.linked_roots == {}
    _after, result = adopt_external_changes(reading)
    assert result is not None
    assert "linked_roots" in result.updated
    assert set(cfg.linked_roots) == {"A"}


def test_a_write_in_progress_reverts_nothing(monkeypatch):
    _write_from_another_process("top_k", 31)
    persistent_settings.overlay_persisted_settings(cfg.data_root)
    reading = read_config_state()
    path = cfg.data_root / "config.toml"
    whole = path.read_text(encoding="utf-8").replace("31", "32")
    write_lock = persistent_settings._config_write_lock
    read_lock = persistent_settings._config_read_lock
    half_written = threading.Event()
    reader_waits = threading.Event()

    def writer() -> None:
        # save() truncates and then writes; an empty file parses as one with no keys.
        with write_lock(cfg.data_root):
            path.write_text("", encoding="utf-8")
            half_written.set()
            reader_waits.wait()
            path.write_text(whole, encoding="utf-8")

    @contextmanager
    def signalling_lock(root):
        reader_waits.set()
        with read_lock(root):
            yield

    monkeypatch.setattr(persistent_settings, "_config_read_lock", signalling_lock)
    thread = threading.Thread(target=writer)
    thread.start()
    half_written.wait()
    _after, _result = adopt_external_changes(reading)
    seen_top_k = cfg.top_k
    reader_waits.set()  # a reader that never took the lock leaves the writer waiting
    thread.join()
    assert seen_top_k == 32


def test_a_value_this_process_already_holds_is_not_applied_again():
    reading = read_config_state()
    cfg.top_k = 31
    _write_from_another_process("top_k", 31)
    _after, result = adopt_external_changes(reading)
    assert result is None
