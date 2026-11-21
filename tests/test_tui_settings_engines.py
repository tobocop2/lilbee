"""The Engines tab in Settings: the preferred engine, the engine table, Install and Remove."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from rich.text import Text
from textual.widgets import DataTable, Select, Static, TabbedContent, TabPane

from lilbee.app import engine_service
from lilbee.app.engine_service import EngineOption, EngineRow, EngineState
from lilbee.catalog import runnable
from lilbee.catalog.models import DownloadProgress
from lilbee.cli.tui.engine_install_flow import confirm_text
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.settings_engines import (
    ENGINES_PANE_ID,
    PREFERRED_ID,
    TABLE_ID,
    EnginesPane,
)
from lilbee.cli.tui.task_queue import TaskStatus
from lilbee.cli.tui.widgets.confirm_dialog import CancelFirstConfirmDialog, ConfirmDialog
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.engine_store import EngineStore
from lilbee.providers.fleet.engines import LLAMA, choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.host_gate import PASS
from lilbee.providers.fleet.engines.llama import pack_backend
from lilbee.providers.fleet.host_caps import collect
from lilbee.providers.roles import WorkerRole
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests._published import render_table, wheel_name
from tests.conformance.fake_adapter import BundledFakeAdapter, FakeAdapter

_DOWNLOAD_BYTES = 180_000_000

_LLAMA = EngineRow("llama.cpp", EngineState.BUNDLED, backend="metal", roles=("embed",))
_BUILD_BYTES = 412_000_000
_IK = EngineRow(
    "ik_llama.cpp",
    EngineState.INSTALLED,
    "1.2.3",
    "cuda",
    roles=("chat", "rerank"),
    size_bytes=_BUILD_BYTES,
)
# A build whose size on disk cannot be read.
_TEI = EngineRow("tei", EngineState.INSTALLED, "0.9", "cpu")
_OPENVINO = EngineRow(
    "openvino", EngineState.NOT_INSTALLED, backend="cpu", download_bytes=_DOWNLOAD_BYTES
)


class _SettingsApp(LilbeeAppHost):
    def on_mount(self) -> None:
        self.push_screen(SettingsScreen())


class _Rows:
    """The rows the service returns; a test changes them to model an install or a removal."""

    def __init__(self, rows: list[EngineRow]) -> None:
        self.rows = rows

    def __call__(self, *_a: object, **_k: object) -> list[EngineRow]:
        return list(self.rows)


@pytest.fixture(autouse=True)
def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.preferred_engine = EngineSetting.AUTO
    yield
    for field in type(snapshot).model_fields:
        setattr(cfg, field, getattr(snapshot, field))


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> _Rows:
    rows = _Rows([_LLAMA, _IK, _TEI, _OPENVINO])
    monkeypatch.setattr(engine_service, "engine_rows", rows)
    offered = {name: frozenset({"gguf"}) for name in ("llama.cpp", "ik_llama.cpp", "tei")}
    monkeypatch.setattr(engine_service, "offered_engines", lambda: dict(offered))
    return rows


def _toasts(app: LilbeeAppHost) -> list[str]:
    return [n.message for n in app._notifications]


def _plain(cell: object) -> str:
    assert isinstance(cell, Text)
    return cell.plain


async def _open_engines(app: _SettingsApp, pilot, rows: int = 4) -> DataTable[Text]:
    screen = app.screen
    assert isinstance(screen, SettingsScreen)
    screen.query_one("#settings-tabs", TabbedContent).active = ENGINES_PANE_ID
    table = screen.query_one(f"#{TABLE_ID}", DataTable)
    assert await wait_until(pilot, lambda: table.row_count == rows)
    return table


async def _select_row(pilot, table: DataTable[Text], engine: str) -> None:
    index = next(i for i, key in enumerate(table.rows) if str(key.value).startswith(f"{engine}|"))
    table.focus()
    table.move_cursor(row=index)
    await pilot.pause()
    await pilot.press("enter")


async def _dialog_message(app: LilbeeAppHost, pilot, kind: type[ConfirmDialog]) -> str:
    """The text of the dialog once it is the screen and its message widget is built."""
    assert await wait_until(
        pilot,
        lambda: isinstance(app.screen, kind) and bool(app.screen.query("#confirm-message")),
        timeout=10,
    )
    return str(app.screen.query_one("#confirm-message", Static).render())


# --- the tab


async def test_the_engines_tab_sits_right_after_the_models_tab(service: _Rows) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        ids = [pane.id for pane in screen.query(TabPane)]
        assert ids[ids.index("settings-tab-models") + 1] == ENGINES_PANE_ID
        assert (
            screen._pane_ids[screen._pane_ids.index("settings-tab-models") + 1] == ENGINES_PANE_ID
        )
        assert screen._pane_ids[-1] == "settings-tab-presets"
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        assert str(tabs.get_tab(ENGINES_PANE_ID).label) == "Engines"


async def test_the_table_is_named_engines_and_shows_one_row_per_engine(service: _Rows) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        assert str(app.screen.query_one("#engines-title", Static).render()) == "Engines"
        assert [str(c.label) for c in table.columns.values()] == [
            "Engine",
            "Version",
            "Backend",
            "Size",
            "State",
            "Used by",
            "Action",
        ]
        shown = [[_plain(c) for c in table.get_row_at(i)] for i in range(table.row_count)]
        assert shown == [
            ["■ llama.cpp", "-", "metal", "bundled", "bundled", "used by embed", ""],
            [
                "■ ik_llama.cpp",
                "1.2.3",
                "cuda",
                "412 MB",
                "installed",
                "used by chat, rerank",
                "Remove",
            ],
            ["■ tei", "0.9", "cpu", "-", "installed", "", "Remove"],
            ["■ openvino", "-", "cpu", "180 MB", "not installed", "", "Install"],
        ]


async def test_the_engine_cell_colors_the_mark_and_leaves_the_name_plain(service: _Rows) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        cell = table.get_row_at(0)[0]
        assert isinstance(cell, Text)
        colored = [s for s in cell.spans if s.style and getattr(s.style, "color", None)]
        assert [(s.start, s.end) for s in colored] == [(0, 1)]


async def test_a_missing_engine_with_no_known_size_shows_a_dash(
    service: _Rows,
) -> None:
    service.rows = [EngineRow("openvino", EngineState.NOT_INSTALLED)]
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot, rows=1)
        assert [_plain(c) for c in table.get_row_at(0)] == [
            "■ openvino",
            "-",
            "-",
            "-",
            "not installed",
            "",
            "Install",
        ]


async def test_an_engine_the_user_installed_shows_installed_with_its_roles_and_no_action(
    service: _Rows,
) -> None:
    service.rows = [EngineRow("lemonade", EngineState.USER_INSTALLED, roles=("chat",))]
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot, rows=1)
        assert [_plain(c) for c in table.get_row_at(0)] == [
            "■ lemonade",
            "-",
            "-",
            "-",
            "installed",
            "used by chat",
            "",
        ]
        await _select_row(pilot, table, "lemonade")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)


async def test_an_unreadable_engine_list_shows_a_toast(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> list[EngineRow]:
        raise OSError("disk gone")

    monkeypatch.setattr(engine_service, "engine_rows", _boom)
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert await wait_until(
            pilot, lambda: "The engine list could not be read: disk gone" in _toasts(app)
        )


# --- the preferred engine


def _preferred(app: _SettingsApp) -> Select[str]:
    return app.screen.query_one(f"#{PREFERRED_ID}", Select)


def _options(select: Select[str]) -> list[str]:
    return [str(value) for _prompt, value in select._options]


async def test_the_preferred_engine_offers_auto_and_each_engine_the_service_offers(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The service offers lemonade (a user install) and not tei, whatever the rows hold.
    offered = {name: frozenset({"gguf"}) for name in ("llama.cpp", "lemonade")}
    monkeypatch.setattr(engine_service, "offered_engines", lambda: dict(offered))
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_engines(app, pilot)
        select = _preferred(app)
        assert await wait_until(
            pilot, lambda: _options(select) == ["auto", "llama.cpp", "lemonade"]
        )
        assert select.value == "auto"


async def test_the_preferred_engine_keeps_a_stored_value_that_has_no_build(
    service: _Rows,
) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_engines(app, pilot)
        cfg.preferred_engine = EngineSetting.MLX
        app.screen.query_one(EnginesPane).reload()
        select = _preferred(app)
        assert await wait_until(pilot, lambda: _options(select)[-1] == "mlx")
        assert select.value == "mlx"


async def test_picking_a_preferred_engine_stores_it_through_the_service(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefer = mock.Mock(return_value="ik_llama.cpp")
    monkeypatch.setattr(engine_service, "prefer", prefer)
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_engines(app, pilot)
        _preferred(app).value = "ik_llama.cpp"
        assert await wait_until(pilot, lambda: prefer.called)
        prefer.assert_called_once_with("ik_llama.cpp", False)
        assert await wait_until(
            pilot, lambda: "The preferred engine is ik_llama.cpp." in _toasts(app)
        )


async def test_a_refused_preference_shows_the_refusal_and_resets_the_choice(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    refuse = mock.Mock(
        side_effect=engine_service.EngineRefusedError("Engine mlx is not installed.")
    )
    monkeypatch.setattr(engine_service, "prefer", refuse)
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_engines(app, pilot)
        _preferred(app).value = "ik_llama.cpp"
        assert await wait_until(pilot, lambda: "Engine mlx is not installed." in _toasts(app))
        assert await wait_until(pilot, lambda: _preferred(app).value == "auto")


async def test_a_preference_that_can_orphan_the_vectors_asks_and_stores_on_yes(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = engine_service.EmbedConfirmRequiredError("lib", 7)

    def _prefer(name: str, confirmed: bool = False) -> str:
        if not confirmed:
            raise asked
        return name

    prefer = mock.Mock(side_effect=_prefer)
    monkeypatch.setattr(engine_service, "prefer", prefer)
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_engines(app, pilot)
        _preferred(app).value = "ik_llama.cpp"
        text = await _dialog_message(app, pilot, CancelFirstConfirmDialog)
        assert "Library lib holds 7 documents." in text
        await pilot.press("y")
        assert await wait_until(pilot, lambda: prefer.call_count == 2)
        assert prefer.call_args_list[1] == mock.call("ik_llama.cpp", True)


async def test_declining_the_embed_confirm_keeps_the_preference(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefer = mock.Mock(side_effect=engine_service.EmbedConfirmRequiredError("lib", 7))
    monkeypatch.setattr(engine_service, "prefer", prefer)
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_engines(app, pilot)
        _preferred(app).value = "ik_llama.cpp"
        assert await wait_until(pilot, lambda: isinstance(app.screen, CancelFirstConfirmDialog))
        await pilot.press("n")
        assert await wait_until(pilot, lambda: "The engine is unchanged." in _toasts(app))
        assert prefer.call_count == 1
        assert await wait_until(pilot, lambda: _preferred(app).value == "auto")


# --- Remove


async def test_removing_an_engine_in_use_names_the_roles_and_removes_on_yes(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    remove = mock.Mock(return_value="ik_llama.cpp 1.2.3 (cuda)")
    monkeypatch.setattr(engine_service, "remove_build", remove)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        service.rows = [_LLAMA, _TEI, _OPENVINO]
        await _select_row(pilot, table, "ik_llama.cpp")
        text = await _dialog_message(app, pilot, CancelFirstConfirmDialog)
        assert text == "ik_llama.cpp 1.2.3 (cuda) is used by chat, rerank. Remove it anyway?"
        remove.assert_not_called()
        await pilot.press("y")
        assert await wait_until(pilot, lambda: remove.called)
        remove.assert_called_once_with("ik_llama.cpp", "1.2.3", "cuda")
        assert await wait_until(
            pilot, lambda: "Removed engine ik_llama.cpp 1.2.3 (cuda)." in _toasts(app)
        )
        assert await wait_until(pilot, lambda: table.row_count == 3)


async def test_declining_the_remove_question_removes_nothing(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    remove = mock.Mock()
    monkeypatch.setattr(engine_service, "remove_build", remove)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "ik_llama.cpp")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("n")
        assert await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.pause()
        remove.assert_not_called()
        assert table.row_count == 4


async def test_removing_an_unused_engine_asks_without_naming_roles(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    remove = mock.Mock(return_value="tei 0.9 (cpu)")
    monkeypatch.setattr(engine_service, "remove_build", remove)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "tei")
        text = await _dialog_message(app, pilot, ConfirmDialog)
        assert not isinstance(app.screen, CancelFirstConfirmDialog)
        assert text == "Remove tei 0.9 (cpu) from this computer?"
        await pilot.press("y")
        assert await wait_until(pilot, lambda: remove.called)
        remove.assert_called_once_with("tei", "0.9", "cpu")


async def test_a_refused_remove_shows_the_refusal_as_a_toast(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    refuse = mock.Mock(side_effect=engine_service.EngineRefusedError("The engine is running."))
    monkeypatch.setattr(engine_service, "remove_build", refuse)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "tei")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("y")
        assert await wait_until(pilot, lambda: "The engine is running." in _toasts(app))
        assert table.row_count == 4


_IK_MISSING = EngineRow(
    "ik_llama.cpp", EngineState.NOT_INSTALLED, backend="cpu", download_bytes=_DOWNLOAD_BYTES
)
_IK_STALE = EngineRow(
    "ik_llama.cpp", EngineState.DIFFERS_FROM_PIN, "1.2.3", "cuda", size_bytes=_BUILD_BYTES
)


async def test_a_build_that_differs_from_the_pin_shows_its_size_and_removes_on_yes(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    service.rows = [_IK_MISSING, _IK_STALE]
    remove = mock.Mock(return_value="ik_llama.cpp 1.2.3 (cuda)")
    monkeypatch.setattr(engine_service, "remove_build", remove)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot, rows=2)
        shown = [[_plain(c) for c in table.get_row_at(i)] for i in range(table.row_count)]
        assert shown == [
            ["■ ik_llama.cpp", "-", "cpu", "180 MB", "not installed", "", "Install"],
            ["■ ik_llama.cpp", "1.2.3", "cuda", "412 MB", "differs from the pin", "", "Remove"],
        ]
        service.rows = [_IK_MISSING]
        await _select_row(pilot, table, "ik_llama.cpp|1.2.3")
        text = await _dialog_message(app, pilot, ConfirmDialog)
        assert text == "Remove ik_llama.cpp 1.2.3 (cuda) from this computer?"
        await pilot.press("y")
        assert await wait_until(pilot, lambda: remove.called)
        remove.assert_called_once_with("ik_llama.cpp", "1.2.3", "cuda")
        assert await wait_until(pilot, lambda: table.row_count == 1)


async def test_a_build_that_links_out_of_the_store_offers_no_remove_and_says_why(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    linked = EngineRow("ik_llama.cpp", EngineState.INSTALLED, "1.2.3", "cuda", outside_store=True)
    service.rows = [linked]
    remove = mock.Mock()
    monkeypatch.setattr(engine_service, "remove_build", remove)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot, rows=1)
        assert [_plain(c) for c in table.get_row_at(0)] == [
            "■ ik_llama.cpp",
            "1.2.3",
            "cuda",
            "-",
            "installed",
            "",
            "linked, not removable",
        ]
        await _select_row(pilot, table, "ik_llama.cpp")
        folder = engine_service.engines_folder()
        why = (
            "The folder of ik_llama.cpp 1.2.3 (cuda) is a link to a place outside the engines "
            f"folder {folder}. lilbee does not remove it. Delete the link yourself."
        )
        assert await wait_until(pilot, lambda: why in _toasts(app))
        assert isinstance(app.screen, SettingsScreen)
        remove.assert_not_called()


# --- Install


def _engine_tasks(app: LilbeeAppHost) -> list[Any]:
    queue = app.task_bar.queue
    return [t for t in [*queue.active_tasks, *queue.history] if t.task_type == "engine"]


async def test_install_asks_to_download_then_runs_a_task_and_refreshes_the_table(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    installed: list[str] = []

    def _install(name: str, **kwargs: Any) -> engine_service.InstalledBuild:
        kwargs["on_update"](DownloadProgress(50.0, "90 MB of 180 MB", False))
        assert release.wait(timeout=10)
        installed.append(name)
        service.rows = [_LLAMA, _IK, _TEI, EngineRow("openvino", EngineState.INSTALLED, "1", "cpu")]
        return engine_service.InstalledBuild(name, "1", "cpu")

    monkeypatch.setattr(engine_service, "install", _install)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "openvino")
        text = await _dialog_message(app, pilot, ConfirmDialog)
        assert text == "Download openvino (180 MB)?"
        assert installed == []
        await pilot.press("y")
        assert await wait_until(pilot, lambda: bool(_engine_tasks(app)))
        task = _engine_tasks(app)[0]
        assert task.name == "Install engine openvino"
        assert await wait_until(pilot, lambda: task.progress > 0)
        release.set()
        assert await wait_until(pilot, lambda: task.status is TaskStatus.DONE)
        assert installed == ["openvino"]
        assert await wait_until(
            pilot,
            lambda: _plain(table.get_row_at(table.row_count - 1)[4]) == "installed",
        )


async def test_declining_the_download_installs_nothing(
    service: _Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    install = mock.Mock()
    monkeypatch.setattr(engine_service, "install", install)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "openvino")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("n")
        assert await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.pause()
        install.assert_not_called()
        assert _engine_tasks(app) == []


async def test_the_bundled_row_has_no_action(service: _Rows) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "llama.cpp")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)


def test_the_download_question_says_nothing_of_a_model_outside_the_model_flow() -> None:
    sized = EngineOption("openvino", installed=False, download_bytes=_DOWNLOAD_BYTES)
    sizeless = EngineOption("openvino", installed=False)
    assert confirm_text(sized) == "Download openvino (180 MB) and use it for this model?"
    assert confirm_text(sized, use_for_model=False) == "Download openvino (180 MB)?"
    assert confirm_text(sizeless) == "Download openvino and use it for this model?"
    assert confirm_text(sizeless, use_for_model=False) == "Download openvino?"


# --- the service rows


class _Plain(FakeAdapter):
    name = "ik_llama.cpp"


class _Bundled(BundledFakeAdapter):
    name = "llama.cpp"


class _Missing(FakeAdapter):
    name = "openvino"


_BUILD_FILES = {"bin/llama-server": 300, "lib/libggml.so": 100, "ENGINE-RECORD.json": 12}


def _write_build(engine: str, version: str, backend: str) -> Path:
    """Write a build folder of known size in the engines folder of the config."""
    folder = engine_service.engines_folder() / engine / f"{version}-{backend}"
    for name, size in _BUILD_FILES.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"x" * size)
    return folder


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg.models_dir = tmp_path / "models"
    _write_build("ik_llama.cpp", "1.2.3", "cuda")
    # A new dict, so the undo leaves the real adapters in their own order.
    adapters = {
        EngineName.LLAMA: _Bundled(),
        EngineName.IK_LLAMA: _Plain(),
        EngineName.OPENVINO: _Missing(),
    }
    monkeypatch.setattr(choice, "ADAPTERS", adapters)
    views = [
        engine_service.RoleView(WorkerRole.CHAT, "m", "gguf", "ik_llama.cpp", None),
        engine_service.RoleView(WorkerRole.EMBED, "e", "gguf", "llama.cpp", None),
        engine_service.RoleView(WorkerRole.RERANK, "r", "gguf", "ik_llama.cpp", None),
        engine_service.RoleView(WorkerRole.VISION, "", "", "llama.cpp", None),
    ]
    monkeypatch.setattr(engine_service, "roles", lambda *_a, **_k: views)
    monkeypatch.setattr(
        engine_service,
        "builds",
        lambda *_a, **_k: [engine_service.BuildView("ik_llama.cpp", "1.2.3", "cuda")],
    )
    # The rule of the engine list: the bundled engine and the build that matches the pin.
    monkeypatch.setattr(
        engine_service,
        "installed_engines",
        lambda: {EngineName.LLAMA: frozenset({"gguf"}), EngineName.IK_LLAMA: frozenset({"gguf"})},
    )
    # The fake engines need an NVIDIA GPU; this host runs every engine whatever the test machine.
    monkeypatch.setattr(choice, "host_answer", lambda *_a: PASS)
    monkeypatch.setattr(engine_service, "llama_server_ready", lambda: True)
    monkeypatch.setattr(engine_service, "_bundled_backend", lambda _name: "metal")
    monkeypatch.setattr(engine_service, "_install_gate", lambda _name: SimpleNamespace(ok=True))
    monkeypatch.setattr(
        engine_service,
        "_missing_option",
        lambda name, _adapter: EngineOption(
            str(name), installed=False, backend="cpu", download_bytes=_DOWNLOAD_BYTES
        ),
    )


def test_engine_rows_list_each_build_the_bundled_engine_and_each_missing_engine(
    fleet: None,
) -> None:
    assert engine_service.engine_rows() == [
        EngineRow("llama.cpp", EngineState.BUNDLED, backend="metal", roles=("embed",)),
        EngineRow(
            "ik_llama.cpp",
            EngineState.INSTALLED,
            "1.2.3",
            "cuda",
            roles=("chat", "rerank"),
            size_bytes=sum(_BUILD_FILES.values()),
        ),
        EngineRow(
            "openvino",
            EngineState.NOT_INSTALLED,
            backend="cpu",
            download_bytes=_DOWNLOAD_BYTES,
        ),
    ]


def test_engine_rows_offer_a_bundled_engine_that_is_not_ready_as_a_download(
    fleet: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_service, "llama_server_ready", lambda: False)
    rows = engine_service.engine_rows()
    assert rows[0] == EngineRow(
        "llama.cpp", EngineState.NOT_INSTALLED, backend="cpu", download_bytes=_DOWNLOAD_BYTES
    )


def test_engine_rows_leave_out_an_engine_the_host_cannot_install(
    fleet: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        engine_service,
        "_install_gate",
        lambda name: SimpleNamespace(ok=name != EngineName.OPENVINO),
    )
    assert [row.engine for row in engine_service.engine_rows()] == ["llama.cpp", "ik_llama.cpp"]


_REAL_BUNDLED_BACKEND = engine_service._bundled_backend
_REAL_MISSING_OPTION = engine_service._missing_option
_LLAMA_PACK_BYTES = 9_876_543


@pytest.mark.parametrize("ready", [True, False], ids=["ready", "not-ready"])
def test_the_llama_rows_name_the_pack_backend_of_the_host_with_the_real_reads(
    fleet: None, monkeypatch: pytest.MonkeyPatch, ready: bool
) -> None:
    host = collect(
        system="Darwin",
        machine="arm64",
        cpu_info={"flags": ["avx2"], "brand_raw": "Test CPU"},
        nvidia=None,
        mac_ver="14.5",
    )
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    backend = pack_backend(host)
    assert backend
    monkeypatch.setattr(choice, "ADAPTERS", {**choice.ADAPTERS, EngineName.LLAMA: LLAMA})
    file = wheel_name(EngineName.LLAMA, LLAMA.pin_record()["version"], f"macos-arm64-{backend}")
    monkeypatch.setattr(
        engine_manifest, "MANIFESTS", render_table([(file, "1" * 64, _LLAMA_PACK_BYTES)])
    )
    monkeypatch.setattr(engine_service, "_bundled_backend", _REAL_BUNDLED_BACKEND)
    monkeypatch.setattr(engine_service, "_missing_option", _REAL_MISSING_OPTION)
    monkeypatch.setattr(engine_service, "llama_server_ready", lambda: ready)
    llama = engine_service.engine_rows()[0]
    if ready:
        assert llama == EngineRow(
            "llama.cpp", EngineState.BUNDLED, backend=backend, roles=("embed",)
        )
    else:
        assert llama == EngineRow(
            "llama.cpp",
            EngineState.NOT_INSTALLED,
            backend=backend,
            download_bytes=_LLAMA_PACK_BYTES,
        )


def test_engine_rows_show_the_builds_of_an_engine_the_install_gate_refuses(
    fleet: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The host gate passes, so the build is listed. The pin build is absent, and the install
    # gate refuses the engine, so it is neither installed nor offered as missing.
    monkeypatch.setattr(
        engine_service, "installed_engines", lambda: {EngineName.LLAMA: frozenset({"gguf"})}
    )
    monkeypatch.setattr(
        engine_service,
        "_install_gate",
        lambda name: SimpleNamespace(ok=name != EngineName.IK_LLAMA),
    )
    assert "ik_llama.cpp" not in engine_service.engine_list().not_installed
    assert "ik_llama.cpp" not in engine_service.offered_engines()
    rows = [r for r in engine_service.engine_rows() if r.engine == "ik_llama.cpp"]
    assert rows == [
        EngineRow(
            "ik_llama.cpp",
            EngineState.DIFFERS_FROM_PIN,
            "1.2.3",
            "cuda",
            size_bytes=sum(_BUILD_FILES.values()),
        )
    ]


async def test_the_tab_shows_a_build_of_an_engine_the_install_gate_refuses_with_remove(
    fleet: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        engine_service, "installed_engines", lambda: {EngineName.LLAMA: frozenset({"gguf"})}
    )
    monkeypatch.setattr(
        engine_service,
        "_install_gate",
        lambda name: SimpleNamespace(ok=name != EngineName.IK_LLAMA),
    )
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot, rows=3)
        shown = [[_plain(c) for c in table.get_row_at(i)] for i in range(table.row_count)]
        ik = [row for row in shown if row[0] == "■ ik_llama.cpp"]
        assert len(ik) == 1
        assert (ik[0][1], ik[0][2], ik[0][4], ik[0][6]) == (
            "1.2.3",
            "cuda",
            "differs from the pin",
            "Remove",
        )


def _states(rows: list[EngineRow]) -> dict[str, list[EngineState]]:
    found: dict[str, list[EngineState]] = {}
    for row in rows:
        found.setdefault(row.engine, []).append(row.state)
    return found


def test_engine_rows_count_a_build_that_differs_from_the_pin_as_missing_as_the_list_does(
    fleet: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The store holds ik_llama.cpp 1.2.3, but the pin names another build.
    monkeypatch.setattr(
        engine_service, "installed_engines", lambda: {EngineName.LLAMA: frozenset({"gguf"})}
    )
    rows = engine_service.engine_rows()
    missing = {r.engine for r in rows if r.state is EngineState.NOT_INSTALLED}
    assert missing == set(engine_service.engine_list().not_installed)
    assert _states(rows)["ik_llama.cpp"] == [
        EngineState.NOT_INSTALLED,
        EngineState.DIFFERS_FROM_PIN,
    ]
    assert "ik_llama.cpp" not in engine_service.offered_engines()


def test_engine_rows_show_a_build_that_differs_from_the_pin_after_the_install_row(
    fleet: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The store holds ik_llama.cpp 1.2.3 (cuda), and the pin names another build.
    monkeypatch.setattr(
        engine_service, "installed_engines", lambda: {EngineName.LLAMA: frozenset({"gguf"})}
    )
    listed = engine_service.engine_list()
    stale = [
        (b.engine, b.version, b.backend) for b in listed.builds if b.engine in listed.not_installed
    ]
    rows = [r for r in engine_service.engine_rows() if r.engine == "ik_llama.cpp"]
    assert rows == [
        EngineRow(
            "ik_llama.cpp",
            EngineState.NOT_INSTALLED,
            backend="cpu",
            download_bytes=_DOWNLOAD_BYTES,
        ),
        EngineRow(
            "ik_llama.cpp",
            EngineState.DIFFERS_FROM_PIN,
            "1.2.3",
            "cuda",
            size_bytes=sum(_BUILD_FILES.values()),
        ),
    ]
    assert [(r.engine, r.version, r.backend) for r in rows[1:]] == stale


def test_engine_rows_mark_a_linked_build_folder_as_outside_the_store(
    fleet: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    _files(outside, {"big": 50_000})
    (engine_service.engines_folder() / "ik_llama.cpp" / "1.2.4-cuda").symlink_to(outside)
    monkeypatch.setattr(
        engine_service,
        "builds",
        lambda *_a, **_k: [
            engine_service.BuildView("ik_llama.cpp", "1.2.3", "cuda"),
            engine_service.BuildView("ik_llama.cpp", "1.2.4", "cuda"),
        ],
    )
    rows = [r for r in engine_service.engine_rows() if r.engine == "ik_llama.cpp"]
    assert [(r.version, r.size_bytes, r.outside_store) for r in rows] == [
        ("1.2.3", sum(_BUILD_FILES.values()), False),
        ("1.2.4", None, True),
    ]


def test_engine_rows_count_an_engine_the_user_installed_as_installed_as_the_list_does(
    fleet: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A user binary on PATH: the engine is installed, and the store holds no build of it.
    monkeypatch.setattr(engine_service, "builds", lambda *_a, **_k: [])
    rows = engine_service.engine_rows()
    missing = {r.engine for r in rows if r.state is EngineState.NOT_INSTALLED}
    assert missing == set(engine_service.engine_list().not_installed) == {"openvino"}
    assert EngineRow("ik_llama.cpp", EngineState.USER_INSTALLED, roles=("chat", "rerank")) in rows
    assert "ik_llama.cpp" in engine_service.offered_engines()


# --- the size on disk of a build


def test_the_store_reads_the_size_on_disk_of_a_build(fleet: None) -> None:
    store = EngineStore(engine_service.engines_folder())
    assert store.build_size("ik_llama.cpp", "1.2.3", "cuda") == sum(_BUILD_FILES.values())


def test_a_build_whose_folder_cannot_be_read_has_no_size(fleet: None) -> None:
    store = EngineStore(engine_service.engines_folder())
    assert store.build_size("ik_llama.cpp", "9.9.9", "cuda") is None


def _files(folder: Path, sizes: dict[str, int]) -> None:
    for name, size in sizes.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"x" * size)


def test_the_size_counts_only_its_own_build_folder_of_two_builds_of_one_engine(
    tmp_path: Path,
) -> None:
    root = tmp_path / "engines"
    _files(root / "ik_llama.cpp" / "1.2.3-cuda", {"bin/s": 300, "lib/a.so": 100})
    _files(root / "ik_llama.cpp" / "1.2.2-cuda", {"bin/s": 7000})
    _files(root / "tei" / "0.9-cpu", {"bin/t": 9000})
    store = EngineStore(root)
    assert store.build_size("ik_llama.cpp", "1.2.3", "cuda") == 400
    assert store.build_size("ik_llama.cpp", "1.2.2", "cuda") == 7000


def test_the_size_counts_a_link_inside_the_build_as_the_link_not_its_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    _files(outside, {"big": 50_000})
    root = tmp_path / "engines"
    build = root / "ik_llama.cpp" / "1.2.3-cuda"
    _files(build, {"bin/s": 300})
    (build / "dirlink").symlink_to(outside)
    (build / "filelink").symlink_to(outside / "big")
    link_bytes = (build / "filelink").lstat().st_size
    assert EngineStore(root).build_size("ik_llama.cpp", "1.2.3", "cuda") == 300 + link_bytes


def test_a_build_folder_that_is_a_link_out_of_the_store_has_no_size(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    outside = tmp_path / "outside"
    _files(outside, {"big": 50_000})
    root = tmp_path / "engines"
    (root / "ik_llama.cpp").mkdir(parents=True)
    (root / "ik_llama.cpp" / "1.2.3-cuda").symlink_to(outside)
    store = EngineStore(root)
    with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.engine_store"):
        assert store.build_size("ik_llama.cpp", "1.2.3", "cuda") is None
    assert not store.holds("ik_llama.cpp", "1.2.3", "cuda")
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert messages == [
        "The size of engine ik_llama.cpp 1.2.3 (cuda) is unknown: "
        f"'1.2.3-cuda' is not inside the engines folder {root}."
    ]


def test_a_build_in_an_engine_folder_that_is_a_link_has_no_size(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _files(outside / "1.2.3-cuda", {"big": 50_000})
    root = tmp_path / "engines"
    root.mkdir()
    (root / "ik_llama.cpp").symlink_to(outside)
    store = EngineStore(root)
    assert store.build_size("ik_llama.cpp", "1.2.3", "cuda") is None
    assert not store.holds("ik_llama.cpp", "1.2.3", "cuda")
