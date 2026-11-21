"""The Engines tab in Settings: the preferred engine, the engine table, Install and Remove."""

from __future__ import annotations

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
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.roles import WorkerRole
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conformance.fake_adapter import BundledFakeAdapter, FakeAdapter

_DOWNLOAD_BYTES = 180_000_000

_LLAMA = EngineRow("llama.cpp", EngineState.BUNDLED, backend="metal", roles=("embed",))
_IK = EngineRow("ik_llama.cpp", EngineState.INSTALLED, "1.2.3", "cuda", roles=("chat", "rerank"))
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
                "-",
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


async def test_the_preferred_engine_offers_auto_and_each_engine_that_has_a_build(
    service: _Rows,
) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_engines(app, pilot)
        select = _preferred(app)
        # openvino has no build, so it is not offered
        assert _options(select) == ["auto", "llama.cpp", "ik_llama.cpp", "tei"]
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
        assert await wait_until(pilot, lambda: isinstance(app.screen, CancelFirstConfirmDialog))
        assert "Library lib holds 7 documents." in str(
            app.screen.query_one("#confirm-message", Static).render()
        )
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
        assert await wait_until(pilot, lambda: isinstance(app.screen, CancelFirstConfirmDialog))
        text = str(app.screen.query_one("#confirm-message", Static).render())
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
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        assert not isinstance(app.screen, CancelFirstConfirmDialog)
        assert str(app.screen.query_one("#confirm-message", Static).render()) == (
            "Remove tei 0.9 (cpu) from this computer?"
        )
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
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        assert str(app.screen.query_one("#confirm-message", Static).render()) == (
            "Download openvino (180 MB)?"
        )
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


@pytest.fixture
def fleet(monkeypatch: pytest.MonkeyPatch) -> None:
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
        EngineRow("ik_llama.cpp", EngineState.INSTALLED, "1.2.3", "cuda", roles=("chat", "rerank")),
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
