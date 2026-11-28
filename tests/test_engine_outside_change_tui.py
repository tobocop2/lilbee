"""An outside install, removal or preferred-engine change reaches the Engines tab."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
import zipfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from textual import events
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Select, Static, TabbedContent

from lilbee.app import engine_service
from lilbee.app.engine_choice import CHOICES_KEY
from lilbee.app.engine_service import EngineState
from lilbee.cli.tui.engine_commands import publish_engines_changed
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.settings_engines import ENGINES_PANE_ID, PREFERRED_ID, TABLE_ID
from lilbee.cli.tui.widgets.model_bar import ModelBar
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet.engine_manifest import PinLock
from lilbee.providers.fleet.engine_store import (
    PACK_DIR,
    STAGE_PREFIX,
    EngineFolderError,
    EngineStore,
    engines_root,
)
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost

Entry = Callable[[LilbeeAppHost, Any], Awaitable[None]]


@pytest.fixture(autouse=True)
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One data root and one models folder, the two that a second process also reads."""
    models = tmp_path / "models"
    models.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("LILBEE_DATA", str(data))
    monkeypatch.setattr("lilbee.providers.fleet.engine_store.prune_engines", lambda _models: [])
    monkeypatch.setattr(cfg, "data_root", data)
    monkeypatch.setattr(cfg, "models_dir", models)
    monkeypatch.setattr(cfg, "preferred_engine", EngineSetting.AUTO)
    monkeypatch.setattr(cfg, "engine_choices", None)
    return models


def install_elsewhere(models: Path, engine: str = "tei", version: str = "0.9") -> None:
    """Install a build through a second store, as another process does; the app learns nothing."""
    tool = "llama-server" if "llama" in engine else "text-embeddings-router"
    wheel = models.parent / f"{engine}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        info = zipfile.ZipInfo(f"lilbee_engine_{engine.replace('.', '_')}/bin/{tool}")
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, "#!/bin/sh\n")
        archive.writestr("x-1.dist-info/METADATA", "License-Expression: MIT\n")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock = PinLock(engine, version, "cpu", {}, ("MIT",), sha, tool)
    EngineStore(engines_root(models)).install_from_wheel(lock, wheel)


def remove_elsewhere(engine: str = "tei", version: str = "0.9") -> None:
    engine_service.remove_build(engine, version, "cpu", confirmed=True)


def prefer_elsewhere(engine: str) -> None:
    """Write the preferred engine into config.toml only; this process's settings stay."""
    persistent_settings.set_value(cfg.data_root, "preferred_engine", engine)


class _App(LilbeeAppHost):
    """The model bar on the base screen, the Settings screen on top of it."""

    def compose(self) -> ComposeResult:
        yield ModelBar(id="bar-host")

    def on_mount(self) -> None:
        self.push_screen(SettingsScreen())


class Watch:
    """The keys the app publishes on its settings signal."""

    def __init__(self, app: LilbeeAppHost) -> None:
        self.keys: list[str] = []
        app.settings_changed_signal.subscribe(app, lambda payload: self.keys.append(payload[0]))


def table_keys(app: LilbeeAppHost) -> list[str]:
    table = app.screen.query_one(f"#{TABLE_ID}", DataTable)
    return [str(key.value) for key in table.rows]


def lists_build(app: LilbeeAppHost, engine: str, version: str) -> bool:
    on_disk = {EngineState.INSTALLED, EngineState.DIFFERS_FROM_PIN}
    return any(
        key.startswith(f"{engine}|{version}|") and key.rsplit("|", 1)[1] in on_disk
        for key in table_keys(app)
    )


def hint_of(app: LilbeeAppHost) -> str:
    return str(app.screen.query_one("#engines-hint", Static).render())


def select_of(app: LilbeeAppHost) -> str:
    return str(app.screen.query_one(f"#{PREFERRED_ID}", Select).value)


async def open_engines(app: LilbeeAppHost, pilot: Any) -> None:
    screen = app.screen
    assert isinstance(screen, SettingsScreen)
    screen.query_one("#settings-tabs", TabbedContent).active = ENGINES_PANE_ID
    table = screen.query_one(f"#{TABLE_ID}", DataTable)
    assert await wait_until(pilot, lambda: table.row_count > 0)


async def focus_back(app: LilbeeAppHost, pilot: Any) -> None:
    """The terminal reports that the user came back."""
    app.post_message(events.AppFocus())
    await pilot.pause()


async def resume(app: LilbeeAppHost, pilot: Any) -> None:
    """Another screen opens over Settings and closes, so Settings resumes."""
    await app.push_screen(Screen())
    await pilot.pause()
    await app.pop_screen()
    await pilot.pause()


async def reactivate(app: LilbeeAppHost, pilot: Any) -> None:
    """The user leaves the Engines tab and comes back to it."""
    tabs = app.screen.query_one("#settings-tabs", TabbedContent)
    tabs.active = "settings-tab-models"
    await pilot.pause()
    tabs.active = ENGINES_PANE_ID
    await pilot.pause()


async def question(app: LilbeeAppHost, pilot: Any) -> None:
    """A chat question asks for outside changes first."""
    app.adopt_external_settings()
    await pilot.pause()


ENTRIES: dict[str, Entry] = {
    "focus": focus_back,
    "resume": resume,
    "reactivate": reactivate,
    "question": question,
}


@pytest.mark.parametrize("entry", list(ENTRIES))
async def test_an_install_by_another_process_is_listed_and_published_once(
    world: Path, entry: str
) -> None:
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        assert not lists_build(app, "tei", "0.9")
        watch = Watch(app)
        install_elsewhere(world)
        await ENTRIES[entry](app, pilot)
        assert await wait_until(pilot, lambda: lists_build(app, "tei", "0.9"), max_pauses=40)
        assert watch.keys.count(CHOICES_KEY) == 1


async def test_a_removal_by_another_process_drops_the_row_and_is_published_once(
    world: Path,
) -> None:
    install_elsewhere(world)
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        assert lists_build(app, "tei", "0.9")
        watch = Watch(app)
        remove_elsewhere()
        await focus_back(app, pilot)
        assert await wait_until(pilot, lambda: not lists_build(app, "tei", "0.9"), max_pauses=40)
        assert watch.keys.count(CHOICES_KEY) == 1


async def test_an_event_with_nothing_changed_on_disk_publishes_nothing(world: Path) -> None:
    install_elsewhere(world)
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        watch = Watch(app)
        for entry in ENTRIES.values():
            await entry(app, pilot)
        assert lists_build(app, "tei", "0.9")
        assert watch.keys == []


async def test_the_apps_own_removal_is_published_once_and_the_next_events_add_nothing(
    world: Path,
) -> None:
    install_elsewhere(world)
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        watch = Watch(app)
        remove_elsewhere()
        publish_engines_changed(app)
        assert await wait_until(pilot, lambda: watch.keys.count(CHOICES_KEY) == 1)
        assert await wait_until(pilot, lambda: app.engines_stamp == ())
        await focus_back(app, pilot)
        await question(app, pilot)
        assert watch.keys.count(CHOICES_KEY) == 1


async def test_the_apps_own_publish_goes_out_when_the_stamp_does_not_move(world: Path) -> None:
    install_elsewhere(world)
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        assert await wait_until(pilot, lambda: bool(app.engines_stamp))
        before = app.engines_stamp
        watch = Watch(app)
        publish_engines_changed(app)
        assert await wait_until(pilot, lambda: watch.keys.count(CHOICES_KEY) == 1)
        assert app.engines_stamp == before
        await focus_back(app, pilot)
        assert watch.keys.count(CHOICES_KEY) == 1


async def test_an_outside_install_after_the_apps_own_change_is_published(world: Path) -> None:
    install_elsewhere(world)
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        assert await wait_until(pilot, lambda: bool(app.engines_stamp))
        watch = Watch(app)
        remove_elsewhere()
        publish_engines_changed(app)
        assert await wait_until(pilot, lambda: watch.keys.count(CHOICES_KEY) == 1)
        assert await wait_until(pilot, lambda: app.engines_stamp == ())
        install_elsewhere(world)
        await focus_back(app, pilot)
        assert await wait_until(pilot, lambda: watch.keys.count(CHOICES_KEY) == 2, max_pauses=40)


async def test_a_stage_folder_of_another_process_is_not_a_change(world: Path) -> None:
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        watch = Watch(app)
        stage = engines_root(world) / "tei" / f"{STAGE_PREFIX}abc123"
        stage.mkdir(parents=True)
        (stage / "partial").write_text("x", encoding="utf-8")
        await focus_back(app, pilot)
        assert watch.keys == []
        assert not lists_build(app, "tei", "0.9")


async def test_a_lock_file_of_another_process_is_not_a_change(world: Path) -> None:
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        watch = Watch(app)
        lock = engines_root(world) / ".locks" / "tei.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("", encoding="utf-8")
        await focus_back(app, pilot)
        assert watch.keys == []


async def test_a_build_folder_without_a_record_is_named_in_the_hint(world: Path) -> None:
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        assert "0.9-cpu" not in hint_of(app)
        (engines_root(world) / "tei" / "0.9-cpu").mkdir(parents=True)
        await focus_back(app, pilot)
        assert await wait_until(pilot, lambda: "0.9-cpu" in hint_of(app), max_pauses=40)


async def test_an_install_that_ends_after_a_staged_copy_was_seen_is_published_once(
    world: Path,
) -> None:
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        watch = Watch(app)
        stage = engines_root(world) / "ik_llama.cpp" / f"{STAGE_PREFIX}abc123"
        stage.mkdir(parents=True)
        await focus_back(app, pilot)
        assert watch.keys == []
        stage.rmdir()
        install_elsewhere(world, "ik_llama.cpp")
        await focus_back(app, pilot)
        await focus_back(app, pilot)
        assert await wait_until(
            pilot, lambda: lists_build(app, "ik_llama.cpp", "0.9"), max_pauses=40
        )
        assert watch.keys.count(CHOICES_KEY) == 1


@pytest.mark.parametrize("entry", list(ENTRIES))
async def test_a_preferred_engine_written_by_another_process_shows_on_the_tab(
    world: Path, entry: str
) -> None:
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        assert select_of(app) == "auto"
        prefer_elsewhere("llama.cpp")
        await ENTRIES[entry](app, pilot)
        assert await wait_until(pilot, lambda: select_of(app) == "llama.cpp", max_pauses=40)


async def test_the_tab_shows_an_outside_preferred_engine_and_keeps_an_in_app_setting(
    world: Path,
) -> None:
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        app.set_setting("choose_engine_on_pick", False, ask_rebuild=False)
        await pilot.pause()
        prefer_elsewhere("llama.cpp")
        await focus_back(app, pilot)
        assert await wait_until(pilot, lambda: select_of(app) == "llama.cpp", max_pauses=40)
        assert cfg.choose_engine_on_pick is False


def _touch_later(folder: Path) -> None:
    """Move the change time of *folder* one second on; name, inode and size stay."""
    info = folder.stat()
    os.utime(folder, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))


def test_the_stamp_changes_with_the_change_time_of_a_build_folder(world: Path) -> None:
    (engines_root(world) / "tei" / "0.9-cpu").mkdir(parents=True)
    before = engine_service.build_folders_stamp()
    _touch_later(engines_root(world) / "tei" / "0.9-cpu")
    after = engine_service.build_folders_stamp()
    assert [entry[:2] for entry in after] == [entry[:2] for entry in before]
    assert after != before


async def test_a_build_replaced_in_place_with_the_same_name_is_published(world: Path) -> None:
    install_elsewhere(world)
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        watch = Watch(app)
        _touch_later(engines_root(world) / "tei" / "0.9-cpu")
        await focus_back(app, pilot)
        assert await wait_until(pilot, lambda: watch.keys.count(CHOICES_KEY) == 1, max_pauses=40)


def test_the_stamp_of_a_missing_engines_folder_is_empty(world: Path) -> None:
    assert not engines_root(world).exists()
    assert engine_service.build_folders_stamp() == ()


def test_the_stamp_counts_build_folders_only(world: Path) -> None:
    root = engines_root(world)
    (root / PACK_DIR).mkdir(parents=True)
    (root / ".locks").mkdir()
    (root / "stray.txt").write_text("x", encoding="utf-8")
    (root / "tei" / f"{STAGE_PREFIX}abc").mkdir(parents=True)
    assert engine_service.build_folders_stamp() == ()
    (root / "tei" / "0.9-cpu").mkdir()
    stamp = engine_service.build_folders_stamp()
    assert [name for name, _inode, _time in stamp] == ["tei/0.9-cpu"]


def test_the_stamp_of_an_unreadable_engines_folder_is_empty(
    world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(folder: Path, engine: str) -> list[Any]:
        raise EngineFolderError(folder, engine, PermissionError("denied"))

    (engines_root(world) / "tei" / "0.9-cpu").mkdir(parents=True)
    monkeypatch.setattr("lilbee.providers.fleet.engine_store._entries", refuse)
    assert engine_service.build_folders_stamp() == ()


def test_one_unreadable_engine_folder_leaves_the_other_engines_in_the_stamp(
    world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.providers.fleet import engine_store

    real = engine_store._entries

    def refuse_tei(folder: Path, engine: str) -> list[Any]:
        if engine == "tei":
            raise EngineFolderError(folder, engine, PermissionError("denied"))
        return real(folder, engine)

    root = engines_root(world)
    (root / "tei" / "0.9-cpu").mkdir(parents=True)
    (root / "llama.cpp" / "b1-cpu").mkdir(parents=True)
    monkeypatch.setattr("lilbee.providers.fleet.engine_store._entries", refuse_tei)
    stamp = engine_service.build_folders_stamp()
    assert [name for name, _inode, _time in stamp] == ["llama.cpp/b1-cpu"]


def test_a_build_that_goes_away_during_the_scan_is_left_out(
    world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.providers.fleet import engine_store

    real = engine_store._entries
    root = engines_root(world)
    (root / "tei" / "0.9-cpu").mkdir(parents=True)
    (root / "tei" / "1.0-cpu").mkdir()

    def list_then_move(folder: Path, engine: str) -> list[Any]:
        found = real(folder, engine)
        if engine == "tei":
            (folder / "0.9-cpu").rename(world / "moved-away")
        return found

    monkeypatch.setattr("lilbee.providers.fleet.engine_store._entries", list_then_move)
    stamp = engine_service.build_folders_stamp()
    assert [name for name, _inode, _time in stamp] == ["tei/1.0-cpu"]


async def test_the_app_stamps_the_engines_off_the_ui_thread(
    world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_thread = threading.get_ident()
    threads: list[int] = []
    real = engine_service.build_folders_stamp

    def spy() -> engine_service.BuildStamp:
        threads.append(threading.get_ident())
        return real()

    monkeypatch.setattr(engine_service, "build_folders_stamp", spy)
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        await open_engines(app, pilot)
        assert await wait_until(pilot, lambda: app.engines_stamp is not None)
        watch = Watch(app)
        install_elsewhere(world)
        await focus_back(app, pilot)
        assert await wait_until(pilot, lambda: watch.keys.count(CHOICES_KEY) == 1, max_pauses=40)
        publish_engines_changed(app)
        assert await wait_until(pilot, lambda: len(threads) >= 3, max_pauses=40)
    assert threads
    assert loop_thread not in threads


async def test_a_stamp_from_a_cancelled_check_is_dropped(world: Path) -> None:
    app = _App()
    async with app.run_test(size=(140, 40)) as pilot:
        assert await wait_until(pilot, lambda: app.engines_stamp is not None)
        before = app.engines_stamp
        watch = Watch(app)

        class Cancelled:
            is_cancelled = True

        app._take_engines_stamp(Cancelled(), (("tei/9-cpu", 1, 1),))  # type: ignore[arg-type]
        await pilot.pause()
        assert app.engines_stamp == before
        assert watch.keys == []
