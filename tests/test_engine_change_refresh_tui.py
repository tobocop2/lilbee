"""An engine install or remove repaints the model bar and the fleet editor."""

from __future__ import annotations

import threading
from typing import Any
from unittest import mock

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.app import engine_service
from lilbee.app.engine_choice import CHOICES_KEY
from lilbee.app.engine_service import RoleView
from lilbee.cli.tui.engine_commands import announce, run_engine_command
from lilbee.cli.tui.task_queue import TaskStatus
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.model_bar import ModelBar
from lilbee.providers.roles import WorkerRole
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.test_tui_settings_engines import (  # noqa: F401
    _config,
    _open_engines,
    _select_row,
    _SettingsApp,
    service,
)

_ON_IK = RoleView(WorkerRole.CHAT, "org/chat-GGUF/chat.gguf", "gguf", "ik_llama.cpp", None)
_FELL_BACK = RoleView(
    WorkerRole.CHAT, "org/chat-GGUF/chat.gguf", "gguf", "llama.cpp", "ik_llama.cpp is gone"
)


class _Roles:
    """The role views the service returns; a test changes them to model a removal."""

    def __init__(self, views: list[RoleView]) -> None:
        self.views = views

    def __call__(self, *_a: object) -> list[RoleView]:
        return list(self.views)


class _BarAndSettingsApp(_SettingsApp):
    """The model bar on the base screen, the Settings screen on top of it."""

    def compose(self) -> ComposeResult:
        yield ModelBar(id="bar-host")


def _notice(app: LilbeeAppHost) -> str:
    bar = app.screen_stack[0].query_one(ModelBar)
    return str(bar.query_one("#model-bar-engine-notice", Static).render())


pytestmark = pytest.mark.usefixtures("service")


@pytest.fixture
def roles(monkeypatch: pytest.MonkeyPatch) -> _Roles:
    views = _Roles([_ON_IK])
    monkeypatch.setattr(engine_service, "roles", views)
    return views


async def test_a_remove_from_the_tab_shows_the_fallback_notice_on_the_model_bar(
    roles: _Roles, monkeypatch: pytest.MonkeyPatch
) -> None:
    def remove(*_a: object, **_k: object) -> str:
        roles.views = [_FELL_BACK]
        return "ik_llama.cpp 1.2.3 (cuda)"

    monkeypatch.setattr(engine_service, "remove_build", remove)
    app = _BarAndSettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        assert _notice(app) == ""
        await _select_row(pilot, table, "ik_llama.cpp")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("y")
        assert await wait_until(pilot, lambda: _notice(app) == "ik_llama.cpp is gone")


async def test_a_remove_from_the_tab_publishes_the_engine_change(
    roles: _Roles, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_service, "remove_build", mock.Mock(return_value="tei 0.9 (cpu)"))
    app = _BarAndSettingsApp()
    heard: list[str] = []
    async with app.run_test(size=(140, 40)) as pilot:
        app.settings_changed_signal.subscribe(app, lambda payload: heard.append(payload[0]))
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "tei")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("y")
        assert await wait_until(pilot, lambda: CHOICES_KEY in heard)


async def test_an_install_from_the_tab_publishes_the_engine_change(
    roles: _Roles, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def install(name: str, **_k: Any) -> engine_service.InstalledBuild:
        assert release.wait(timeout=10)
        return engine_service.InstalledBuild(name, "1", "cpu")

    monkeypatch.setattr(engine_service, "install", install)
    app = _BarAndSettingsApp()
    heard: list[str] = []
    async with app.run_test(size=(140, 40)) as pilot:
        app.settings_changed_signal.subscribe(app, lambda payload: heard.append(payload[0]))
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "openvino")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("y")
        assert await wait_until(pilot, lambda: bool(app.task_bar.queue.active_tasks))
        assert heard == []
        release.set()
        task = [*app.task_bar.queue.active_tasks, *app.task_bar.queue.history][0]
        assert await wait_until(pilot, lambda: task.status is TaskStatus.DONE)
        assert await wait_until(pilot, lambda: CHOICES_KEY in heard)


@pytest.mark.parametrize("command", ["remove tei 0.9 cpu", "install tei"])
async def test_the_slash_command_publishes_the_engine_change(
    command: str, roles: _Roles, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_service, "remove_build", mock.Mock(return_value="tei 0.9 (cpu)"))
    monkeypatch.setattr(
        engine_service,
        "install",
        mock.Mock(return_value=engine_service.InstalledBuild("tei", "0.9", "cpu")),
    )
    app = _BarAndSettingsApp()
    heard: list[str] = []
    async with app.run_test(size=(140, 40)) as pilot:
        app.settings_changed_signal.subscribe(app, lambda payload: heard.append(payload[0]))
        announce(app, run_engine_command(command))
        assert await wait_until(pilot, lambda: CHOICES_KEY in heard)


async def test_a_refused_remove_publishes_nothing(
    roles: _Roles, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = engine_service.EngineRefusedError("not here")
    monkeypatch.setattr(engine_service, "remove_build", mock.Mock(side_effect=refused))
    app = _BarAndSettingsApp()
    heard: list[str] = []
    async with app.run_test(size=(140, 40)) as pilot:
        app.settings_changed_signal.subscribe(app, lambda payload: heard.append(payload[0]))
        result = run_engine_command("remove tei 0.9 cpu")
        assert not result.ok
        announce(app, result)
        await pilot.pause()
        assert heard == []
        assert _notice(app) == ""


class _FleetAndSettingsApp(_SettingsApp):
    """The fleet editor on the base screen, the Settings screen on top of it."""

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.fleet_body import FleetBody

        yield FleetBody()


async def test_a_remove_from_the_tab_repaints_the_fleet_editor(
    roles: _Roles, monkeypatch: pytest.MonkeyPatch
) -> None:
    def remove(*_a: object, **_k: object) -> str:
        roles.views = [_FELL_BACK]
        return "ik_llama.cpp 1.2.3 (cuda)"

    def shown(app: LilbeeAppHost) -> str:
        return str(app.screen_stack[0].query_one("#placement-engines", Static).render())

    monkeypatch.setattr(engine_service, "remove_build", remove)
    app = _FleetAndSettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        assert "ik_llama.cpp is gone" not in shown(app)
        await _select_row(pilot, table, "ik_llama.cpp")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("y")
        assert await wait_until(pilot, lambda: "ik_llama.cpp is gone" in shown(app))
