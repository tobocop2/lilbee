"""Quitting while one of our screens or widgets mounts exits cleanly.

Each real-exit case quits from inside the node's compose, so the exit lands while it
mounts. Textual then mounts none of its children but still sends Mount.

The settings screen also resumes work after a wait: the Profile tab switch after a refresh,
and the profile reload after its thread read. A quit during that wait leaves the screen
running but detached, and the resumed work must not touch it.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Select, TabbedContent
from textual.worker import Worker, WorkerCancelled, WorkerFailed

from lilbee.app.settings_map import SettingDef
from lilbee.cli.tui import app as app_module
from lilbee.cli.tui.mount_guard import MountWhenAttached
from lilbee.cli.tui.screens import settings as settings_module
from lilbee.cli.tui.screens.memories import MemoriesScreen
from lilbee.cli.tui.screens.model_picker import ModelPickerModal
from lilbee.cli.tui.screens.profile_tab import PROFILE_SELECT_ID, ProfileTab, load_snapshot
from lilbee.cli.tui.screens.settings import PROFILE_PANE_ID, SettingsScreen
from lilbee.cli.tui.screens.settings_engines import EnginesPane
from lilbee.cli.tui.screens.settings_presets import PresetDialog, PresetsPane
from lilbee.cli.tui.screens.settings_widgets import make_select
from lilbee.cli.tui.screens.status import StatusScreen
from lilbee.cli.tui.screens.task_center import TaskCenter
from lilbee.cli.tui.screens.wiki_drafts import WikiDraftsScreen
from lilbee.cli.tui.widgets.fleet_body import FleetBody
from lilbee.cli.tui.widgets.model_bar import ModelOption
from lilbee.cli.tui.widgets.profile_line import ProfileLine
from lilbee.cli.tui.widgets.scope_chip import ScopeChip
from lilbee.cli.tui.widgets.session_list import SessionListPanel
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from tests._lilbee_app_test_host import LilbeeAppHost

_W = TypeVar("_W", bound=Widget)


def _exiting(cls: type[_W]) -> type[_W]:
    """Return a subclass of ``cls`` that quits the app from inside its compose."""

    def compose(self: Widget) -> ComposeResult:
        self.app.exit()
        yield from cls.compose(self)

    namespace = {"compose": compose, "__module__": cls.__module__}
    return type(f"Exiting{cls.__name__}", (cls,), namespace)


class _Host(LilbeeAppHost):
    def __init__(self, start: Callable[[App], None]) -> None:
        super().__init__()
        self._start = start

    def on_ready(self) -> None:
        self._start(self)


def _push(make: Callable[[], Screen]) -> Callable[[App], None]:
    return lambda app: app.push_screen(make())


def _hold(make: Callable[[], Widget]) -> Callable[[App], None]:
    class _Holder(Screen[None]):
        def compose(self) -> ComposeResult:
            yield make()

    return _push(_Holder)


def _options() -> list[ModelOption]:
    return [ModelOption("Qwen3 0.6B", "qwen3-0.6b"), ModelOption("Llama 8B", "llama-8b")]


def _picker() -> Screen:
    return _exiting(ModelPickerModal)(scope="chat", options=_options(), pick_later=lambda _r: None)


def _profile(app: App) -> None:
    # switch_view replaces the screen on top, so one must be pushed first, as in the real app
    app.push_screen(Screen())
    app.open_profile_tab()


SITES: dict[str, Callable[[App], None]] = {
    "model-picker": _push(_picker),
    "settings-profile": _profile,
    "status": _push(_exiting(StatusScreen)),
    "task-center": _push(_exiting(TaskCenter)),
    "memories": _push(_exiting(MemoriesScreen)),
    "wiki-drafts": _push(_exiting(WikiDraftsScreen)),
    "view-tabs": _hold(_exiting(ViewTabs)),
    "scope-chip": _hold(_exiting(ScopeChip)),
    "fleet-body": _hold(_exiting(FleetBody)),
    "session-list": _hold(_exiting(SessionListPanel)),
    "preset-dialog": _push(_exiting(PresetDialog)),
}


@pytest.mark.parametrize("site", sorted(SITES))
async def test_quit_while_mounting_exits_cleanly(site, monkeypatch):
    monkeypatch.setitem(app_module._VIEW_FACTORIES, "Settings", _exiting(SettingsScreen))
    app = _Host(SITES[site])
    await app.run_async(headless=True, size=(120, 40))
    assert app._exception is None, f"{site} rc={app.return_code} {app._exception!r}"
    assert app.return_code == 0


@pytest.mark.parametrize("value", ["auto", "unknown"], ids=["preset", "unset"])
async def test_quit_while_a_setting_select_mounts_exits_cleanly(value, monkeypatch):
    """A quit that lands as a settings Select composes must not break its mount handler.

    A tab fills lazily, so a quit can come while its rows still mount. Textual's Select
    then mounts none of its overlay but still looks for it.
    """
    real_compose = Select.compose

    def compose(self: Select) -> ComposeResult:
        self.app.exit()
        yield from real_compose(self)

    monkeypatch.setattr(Select, "compose", compose)
    defn = SettingDef(type=str, nullable=False, group="Test", choices=("auto", "remote"))
    app = _Host(_hold(lambda: make_select("test_key", defn, value)))
    await app.run_async(headless=True, size=(120, 40))
    assert app._exception is None, f"rc={app.return_code} {app._exception!r}"
    assert app.return_code == 0


_TUI_DIR = Path(app_module.__file__).parent


def _builds_plain_select(tree: ast.AST) -> list[int]:
    """Lines that call or subclass textual's Select by its bare name."""

    def is_select(node: ast.expr) -> bool:
        if isinstance(node, ast.Subscript):
            node = node.value
        return isinstance(node, ast.Name) and node.id == "Select"

    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and is_select(node.func):
            lines.append(node.lineno)
        elif isinstance(node, ast.ClassDef):
            lines.extend(node.lineno for base in node.bases if is_select(base))
    return lines


def test_no_tui_module_builds_a_plain_select():
    """Every Select the TUI builds is a GuardedSelect, so a quit mid-mount cannot break it."""
    found = [
        f"{path.relative_to(_TUI_DIR)}:{line}"
        for path in sorted(_TUI_DIR.rglob("*.py"))
        if path.name != "mount_guard.py"
        for line in _builds_plain_select(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert found == [], f"build a GuardedSelect, not Select: {found}"


@pytest.mark.parametrize(
    "source",
    ["Select(x)", "Select[str](x)", "class Mine(Select): ...", "class Mine(Select[str]): ..."],
)
def test_the_plain_select_scan_sees_each_way_to_build_one(source):
    assert _builds_plain_select(ast.parse(source)) == [1]


@pytest.mark.parametrize("source", ["GuardedSelect(x)", "query_one(Select)", "Select.Changed"])
def test_the_plain_select_scan_leaves_other_uses_alone(source):
    assert _builds_plain_select(ast.parse(source)) == []


GUARDED: dict[str, Callable[[], Any]] = {
    "model-picker": lambda: ModelPickerModal(
        scope="chat", options=_options(), pick_later=lambda _r: None
    ),
    "preset-dialog": PresetDialog,
    "settings": SettingsScreen,
    "status": StatusScreen,
    "task-center": TaskCenter,
    "memories": MemoriesScreen,
    "wiki-drafts": WikiDraftsScreen,
    "view-tabs": ViewTabs,
    "scope-chip": ScopeChip,
    "fleet-body": FleetBody,
    "session-list": SessionListPanel,
    "engines-pane": EnginesPane,
    "presets-pane": PresetsPane,
}


@pytest.mark.parametrize("site", sorted(GUARDED))
async def test_unattached_node_drops_mount(site, monkeypatch):
    """A node outside a running app gets no call of its mount handler."""
    node = GUARDED[site]()
    calls: list[object] = []
    monkeypatch.setattr(type(node), "on_mount", lambda _self: calls.append(1))
    assert isinstance(node, MountWhenAttached)
    await node.on_event(events.Mount())
    assert calls == []


async def test_attached_node_gets_its_mount(monkeypatch):
    """A screen pushed onto a running app gets one call of its mount handler."""
    calls: list[object] = []
    monkeypatch.setattr(SettingsScreen, "on_mount", lambda _self: calls.append(1))
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(SettingsScreen())
        await pilot.pause()
    assert calls == [1]


def _other_pane(screen: SettingsScreen) -> str:
    tabs = screen.query_one("#settings-tabs", TabbedContent)
    return next(pane.id for pane in tabs.query("TabPane") if pane.id != PROFILE_PANE_ID)


async def _live_settings(app: App, pilot: Any) -> SettingsScreen:
    """Push a settings screen and wait until its Profile pane has filled."""
    screen = SettingsScreen()
    await app.push_screen(screen)
    for _ in range(4):
        await pilot.pause()
    assert screen.query(ProfileTab), "the Profile pane did not fill"
    return screen


async def test_a_quit_after_the_profile_request_leaves_the_tab_as_it_was():
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _live_settings(app, pilot)
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        other = _other_pane(screen)
        tabs.active = other
        await pilot.pause()
        screen.request_profile_tab()
        app.exit()
        assert not screen.is_attached
    assert app._exception is None
    assert tabs.active == other, "the tab switch ran on a detached screen"


class _RequestsThenQuits(SettingsScreen):
    """Asks for the Profile tab from its own compose, then quits the app."""

    __module__ = SettingsScreen.__module__

    def compose(self) -> ComposeResult:
        assert self.is_attached and not self.is_mounted
        self.request_profile_tab()
        self.app.exit()
        yield from SettingsScreen.compose(self)


async def test_a_profile_request_before_mount_then_a_quit_in_compose_exits_cleanly():
    """The request lands on an attached screen that has not mounted, on any task factory."""
    app = _Host(_push(_RequestsThenQuits))
    await app.run_async(headless=True, size=(120, 40))
    assert app._exception is None, f"rc={app.return_code} {app._exception!r}"
    assert app.return_code == 0


async def test_a_profile_request_on_a_mounted_screen_switches_the_tab():
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _live_settings(app, pilot)
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        tabs.active = _other_pane(screen)
        await pilot.pause()
        screen.request_profile_tab()
        await pilot.pause()
        await pilot.pause()
        assert tabs.active == PROFILE_PANE_ID


async def test_open_profile_tab_from_another_screen_focuses_the_profile_dropdown():
    """The Profile tab is the first pane, so the request shows itself by the focus it moves."""
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(Screen())
        await pilot.pause()
        app.open_profile_tab()
        await pilot.pause()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        assert screen.query_one("#settings-tabs", TabbedContent).active == PROFILE_PANE_ID
        assert app.focused is not None and app.focused.id == PROFILE_SELECT_ID


def _record_fills(monkeypatch) -> list[tuple[str, bool]]:
    """Record each fill of the profile line and the Profile tab, and whether it was attached."""
    fills: list[tuple[str, bool]] = []
    for cls, name in ((ProfileLine, "show"), (ProfileTab, "show"), (ProfileTab, "show_failure")):
        real = getattr(cls, name)

        def fill(self, value, _real=real, _label=f"{cls.__name__}.{name}"):
            fills.append((_label, self.is_attached))
            _real(self, value)

        monkeypatch.setattr(cls, name, fill)
    return fills


def _quit_as_the_read_returns(monkeypatch, app: App, *, fails: bool) -> None:
    """Run the read inline, quit, then return or raise: no await separates the quit and the fill.

    A quit while the read still waits on its thread cancels the worker at its await, so only
    a quit after the read reaches the fill. Running the read inline fixes that order.
    """

    async def to_thread(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        app.exit()
        if fails:
            raise OSError("profiles folder gone")
        return result

    monkeypatch.setattr(settings_module, "asyncio", SimpleNamespace(to_thread=to_thread))


async def _worker_error(worker: Worker[None]) -> BaseException | None:
    try:
        await worker.wait()
    except (WorkerFailed, WorkerCancelled) as exc:
        return exc
    return None


@pytest.mark.parametrize("fails", [False, True], ids=["read", "read-fails"])
async def test_a_quit_after_the_profile_read_fills_nothing(monkeypatch, fails):
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _live_settings(app, pilot)
        fills = _record_fills(monkeypatch)
        _quit_as_the_read_returns(monkeypatch, app, fails=fails)
        error = await _worker_error(screen.reload_profile())
    assert app._exception is None
    assert error is None, repr(error)
    assert fills == [], f"filled on a detached screen: {fills}"


async def test_a_profile_reload_on_a_live_screen_fills_the_line_and_the_tab(monkeypatch):
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _live_settings(app, pilot)
        fills = _record_fills(monkeypatch)
        error = await _worker_error(screen.reload_profile())
        assert error is None
        assert fills == [("ProfileLine.show", True), ("ProfileTab.show", True)]


_TAB_USES: dict[str, Callable[[SettingsScreen], None]] = {
    "failure": lambda screen: screen._show_profile_failure("disk gone"),
    "snapshot": lambda screen: screen._show_profile(load_snapshot()),
    "focus": lambda screen: screen._focus_profile_select(),
}


@pytest.mark.parametrize("use", sorted(_TAB_USES))
async def test_profile_tab_exited_mid_mount_is_left_alone(use, monkeypatch):
    """A Profile tab the exit left in the DOM without children takes no snapshot or focus.

    The screen is attached and fills the pane lazily; the exit lands inside the tab's compose.
    The tab is then in the DOM and flagged mounted, but detached and empty.
    """
    monkeypatch.setattr(settings_module, "ProfileTab", _exiting(ProfileTab))
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = SettingsScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen._populate_pane(PROFILE_PANE_ID)
        for _ in range(5):
            await pilot.pause()
        (tab,) = screen.query(ProfileTab)
        assert tab.is_mounted
        assert not tab.is_attached
        assert not tab.children
        _TAB_USES[use](screen)


class _Probe(MountWhenAttached):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def on_unmount(self) -> None:
        self.seen.append("unmount")


async def test_unattached_node_passes_other_events_to_the_parent():
    """Only Mount is dropped: Unmount still reaches its handler."""
    node = _Probe()
    await node.on_event(events.Unmount())
    assert node.seen == ["unmount"]
