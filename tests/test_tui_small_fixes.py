"""The small TUI fixes: wordmark, fleet apply key, dialog sizing, picker, splash, dialog focus."""

from __future__ import annotations

import hashlib
import inspect
import io
import os
import re
import signal
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.errors import NoWidget
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Input, Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.engine_picker import EnginePickerModal
from lilbee.cli.tui.screens.model_picker import ModelPickerModal
from lilbee.cli.tui.widgets.model_bar import ModelOption
from lilbee.cli.tui.widgets.model_list import ModelList
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests._tui_census import build, discover_surfaces

# The picker tests' fake engines: the picker asks for an engine only when two can load the model.
from tests.test_tui_model_picker_engine import (  # noqa: F401
    GGUF_REF,
    _engines,
    _PickerApp,
)

GIB = 1024**3
SIZES = [(80, 24), (120, 40)]
SIZE_IDS = [f"{w}x{h}" for w, h in SIZES]
# Keys that a terminal eats or that end the program: XOFF, XON, INT, QUIT, SUSP, EOF.
TERMINAL_RESERVED = {"ctrl+s", "ctrl+q", "ctrl+c", "ctrl+\\", "ctrl+z", "ctrl+d"}


# --- the wordmark -------------------------------------------------------------------------------

# sha256 of the eleven rows other than index 4.
_OTHER_ROWS_SHA = "b0d843408a23e9618a397c87b290de5ec66ab0234d2f3bc838ad29e9432f80ef"


def test_every_wordmark_row_has_the_logo_width() -> None:
    from lilbee.runtime import bee_logo

    lengths = [len(line) for line in bee_logo.BEE_LINES]
    assert len(lengths) == 12
    assert set(lengths) == {bee_logo.LOGO_WIDTH}


def test_wordmark_row_five_has_no_ink_in_the_gaps_of_its_neighbours() -> None:
    from lilbee.runtime import bee_logo

    lines = bee_logo.BEE_LINES
    width = max(len(line) for line in lines)
    others = [line.ljust(width) for i, line in enumerate(lines) if i != 4]
    gaps = [c for c in range(width) if all(o[c] == " " for o in others)]
    assert gaps
    padded = lines[4].ljust(width)
    assert [c for c in gaps if padded[c] != " "] == []


def test_the_other_wordmark_rows_do_not_change() -> None:
    from lilbee.runtime import bee_logo

    rest = "\n".join(line for i, line in enumerate(bee_logo.BEE_LINES) if i != 4)
    assert hashlib.sha256(rest.encode()).hexdigest() == _OTHER_ROWS_SHA
    assert bee_logo.LOGO_WIDTH == 55


def test_the_splash_runner_and_the_startup_gate_draw_the_one_wordmark() -> None:
    from lilbee.cli.tui.screens import startup_gate
    from lilbee.runtime import _splash_runner, bee_logo

    assert startup_gate._LOGO.split("\n") == bee_logo.BEE_LINES
    assert _splash_runner.BEE_LINES is bee_logo.BEE_LINES


# --- the second fleet Apply key -----------------------------------------------------------------


def _make_view(*, manual: bool = False):
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.providers.roles import WorkerRole

    return PlacementView(
        gpus=tuple(
            GpuInfo(i, "CUDA", f"CUDA{i}", "NVIDIA A40", 44 * GIB, 44 * GIB) for i in range(2)
        ),
        roles=(
            RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0,), None, 1, "llama.cpp"),
            RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (1,), None, 1, "llama.cpp"),
        ),
        unplaceable=(),
        manual=manual,
        spec_json=None,
    )


class _FleetApp(LilbeeAppHost):
    CSS = ""

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.fleet import FleetScreen

        self.push_screen(FleetScreen())


class _DrawerApp(LilbeeAppHost):
    """A live Input stands in for the chat prompt under the drawer."""

    CSS = ""

    def compose(self) -> ComposeResult:
        yield Input(id="probe-input")


@pytest.fixture
def fleet_applies(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Stub the placement calls; the list collects each spec that an apply sends."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp

    calls: list[object] = []
    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    monkeypatch.setattr(
        fbm, "set_placement", lambda spec: calls.append(spec) or _make_view(manual=True)
    )
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})
    return calls


def _keys_for(owner: type, action: str) -> list[str]:
    return [b.key for b in owner.BINDINGS if getattr(b, "action", None) == action]  # type: ignore[union-attr]


def test_the_fleet_screen_and_drawer_bind_the_same_two_apply_keys() -> None:
    from lilbee.cli.tui.screens.fleet import FleetScreen
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    screen, drawer = _keys_for(FleetScreen, "apply"), _keys_for(FleetDrawer, "apply")
    assert sorted(screen) == ["ctrl+s", "f5"]
    assert sorted(drawer) == sorted(screen)
    assert {"f5"}.isdisjoint(TERMINAL_RESERVED)


def test_preview_and_clear_keys_stay_on_the_screen_and_the_drawer() -> None:
    from lilbee.cli.tui.screens.fleet import FleetScreen
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    for owner in (FleetScreen, FleetDrawer):
        assert _keys_for(owner, "preview") == ["ctrl+r"]
        assert _keys_for(owner, "clear") == ["ctrl+x"]


def test_the_footer_lists_the_secondary_keys_only_in_the_drawer() -> None:
    from lilbee.cli.tui.screens.fleet import FleetScreen
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    def shown(owner: type) -> list[str]:
        return [b.key for b in owner.BINDINGS if getattr(b, "show", False) and b.priority]  # type: ignore[union-attr]

    assert shown(FleetScreen) == ["ctrl+s"]
    assert shown(FleetDrawer) == ["ctrl+r", "ctrl+s", "ctrl+x"]


@pytest.mark.asyncio
async def test_f5_applies_once_on_the_fleet_screen(fleet_applies: list[object]) -> None:
    app = _FleetApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        await pilot.press("ctrl+s")
        assert await wait_until(pilot, lambda: len(fleet_applies) == 1)
        assert await wait_until(pilot, lambda: not body.applying)  # type: ignore[attr-defined]
        await pilot.press("f5")
        assert await wait_until(pilot, lambda: len(fleet_applies) == 2)
        assert await wait_until(pilot, lambda: not body.applying)  # type: ignore[attr-defined]
        assert len(fleet_applies) == 2


@pytest.mark.asyncio
async def test_f5_applies_in_the_drawer_and_not_from_the_prompt(
    fleet_applies: list[object],
) -> None:
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    app = _DrawerApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        assert await wait_until(pilot, lambda: bool(app.screen.query(FleetDrawer)))
        assert app.focused is app.query_one("#probe-input", Input)
        await pilot.press("ctrl+s", "f5")
        await pilot.pause()
        assert fleet_applies == []
        app.screen.query_one(FleetDrawer).query(".dev-toggle").first().focus()
        await pilot.pause()
        await pilot.press("f5")
        assert await wait_until(pilot, lambda: len(fleet_applies) == 1)
        assert await wait_until(pilot, lambda: not app.screen.query_one("FleetBody").applying)  # type: ignore[attr-defined]


def test_the_fleet_hints_name_both_apply_keys() -> None:
    from lilbee.cli.tui.screens.fleet import FleetScreen
    from lilbee.cli.tui.widgets import fleet_body

    for text in (FleetScreen.HELP, fleet_body._HINT, msg.FLEET_STATE_EDITED):
        assert "ctrl+s" in text
        assert "f5" in text


# --- dialog sizing and focus --------------------------------------------------------------------


class _DialogApp(LilbeeAppHost):
    def __init__(self, dialog: Screen[Any]) -> None:
        super().__init__()
        self._dialog = dialog
        self.results: list[object] = []

    def compose(self) -> ComposeResult:
        yield Static("")

    def on_mount(self) -> None:
        self.push_screen(self._dialog, self.results.append)


def _shown_at(screen: Screen[Any], widget: Widget, x: int, y: int) -> bool:
    try:
        return screen.get_widget_at(x, y)[0] is widget
    except NoWidget:
        return False


def _shown(screen: Screen[Any], widget: Widget) -> bool:
    """True when the screen draws the widget at both ends of its region (no parent clips it)."""
    region = widget.region
    return _shown_at(screen, widget, region.x, region.y) and _shown_at(
        screen, widget, region.right - 1, region.bottom - 1
    )


LONG = " ".join(f"word{i}" for i in range(120))
MANY_LINES = "\n".join(f"line {i} of the notice" for i in range(30))
MESSAGES = {"short": "Remove this model?", "long": LONG, "many-lines": MANY_LINES}


def _confirm(message: str) -> Screen[Any]:
    from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

    return ConfirmDialog("Remove model", message)


def _notice(message: str) -> Screen[Any]:
    from lilbee.cli.tui.widgets.notice_dialog import NoticeDialog

    return NoticeDialog("Heads up", message)


BUTTONS = {"confirm": ("confirm-yes", "confirm-no"), "notice": ("notice-dismiss",)}
MAKERS = {"confirm": _confirm, "notice": _notice}
MESSAGE_ID = {"confirm": "#confirm-message", "notice": "#notice-message"}
SHORT_BOX_HEIGHT = 9


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("kind", ["confirm", "notice"])
@pytest.mark.parametrize("name", list(MESSAGES))
async def test_the_dialog_buttons_stay_on_screen_whatever_the_message(
    kind: str, name: str, size: tuple[int, int]
) -> None:
    app = _DialogApp(MAKERS[kind](MESSAGES[name]))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()

        def buttons_shown() -> bool:
            found = [app.screen.query(f"#{i}") for i in BUTTONS[kind]]
            return all(f and _shown(app.screen, f.first()) for f in found)

        assert await wait_until(pilot, buttons_shown, max_pauses=30)
        box = app.screen.query_one("Vertical")
        assert box.region.y >= 0
        assert box.region.bottom <= size[1]
        if name == "short":
            assert box.region.height == SHORT_BOX_HEIGHT


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["confirm", "notice"])
async def test_a_long_message_scrolls_to_its_last_row_and_a_click_answers(kind: str) -> None:
    app = _DialogApp(MAKERS[kind](MANY_LINES))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        label = app.screen.query_one(MESSAGE_ID[kind])
        await wait_until(pilot, lambda: label.region.height > 0)
        scrollers = [a for a in label.ancestors if isinstance(a, ScrollableContainer)]
        assert scrollers
        assert scrollers[0].max_scroll_y > 0
        scrollers[0].scroll_end(animate=False)
        assert await wait_until(
            pilot,
            lambda: _shown_at(app.screen, label, label.region.right - 1, label.region.bottom - 1),
        )
        first_button = f"#{BUTTONS[kind][0]}"
        assert _shown(app.screen, app.screen.query_one(first_button))
        await pilot.click(first_button)
        assert await wait_until(pilot, lambda: bool(app.results))


def _picker_rows(n: int) -> list[ModelOption]:
    return [ModelOption(f"Model {i}", f"org/m{i}/f{i}.gguf") for i in range(n)]


async def _open_picker(app: _PickerApp, pilot: Any) -> ModelPickerModal:
    await pilot.pause()
    modal = app.screen
    assert isinstance(modal, ModelPickerModal)
    assert await wait_until(pilot, lambda: modal.query_one("#picker-list", ModelList).row_count > 0)
    await pilot.pause()
    return modal


PICKER_CAP = {(80, 24): 19, (120, 40): 30}  # min(80% of the screen, 30)
# Rows of the box that are not list content: title, search, hint, padding and borders.
PICKER_CHROME = 13


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("n", [1, 3, 40])
async def test_the_model_picker_is_as_tall_as_its_list_up_to_the_cap(
    n: int, size: tuple[int, int]
) -> None:
    _PickerApp.options = _picker_rows(n)
    app = _PickerApp()
    async with app.run_test(size=size) as pilot:
        modal = await _open_picker(app, pilot)
        root = modal.query_one("#picker-root")
        lst = modal.query_one("#picker-list", ModelList)
        content = lst.virtual_size.height
        assert content > 0
        assert lst.size.height <= content
        assert root.region.height == min(PICKER_CAP[size], content + PICKER_CHROME)


@pytest.mark.asyncio
async def test_the_model_picker_keeps_its_width() -> None:
    _PickerApp.options = _picker_rows(2)
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open_picker(app, pilot)
        assert modal.query_one("#picker-root").region.width == 90


def _apply_dialog() -> Screen[Any]:
    from lilbee.app import profiles
    from lilbee.cli.tui.screens.profile_dialogs import ApplyProfileDialog, SwitchPlan
    from lilbee.core.config.enums import SettingSource
    from lilbee.core.profile_files import ProfileFile

    profile = ProfileFile(
        name="demo",
        description="A profile used to test the dialog.",
        authors=(),
        tested_on=None,
        format=1,
        min_lilbee=None,
        evidence=None,
        values={},
    )
    changes = tuple(
        profiles.DiffRow(
            f"setting_{i}", i, SettingSource.BUILT_IN, i + 1, profiles.ProfileEffect.NOW
        )
        for i in range(24)
    )
    diff = profiles.ProfileDiff(name="demo", changes=changes, kept=(), untouched_count=3)
    return ApplyProfileDialog(SwitchPlan(profile, diff, (), 0, None))


def _info_dialog() -> Screen[Any]:
    from lilbee.cli.tui.screens.model_info import ModelInfoModal
    from lilbee.cli.tui.screens.model_picker import _option_to_row

    return ModelInfoModal(_option_to_row(ModelOption("Gemma", GGUF_REF), "Gemma", None, False))


FOCUS_FACTORIES = {
    "ConfirmDialog": lambda: _confirm(MANY_LINES),
    "NoticeDialog": lambda: _notice(MANY_LINES),
    "ApplyProfileDialog": _apply_dialog,
    "ModelInfoModal": _info_dialog,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("name", list(FOCUS_FACTORIES))
async def test_a_dialog_with_a_scroll_area_focuses_a_control(
    name: str, size: tuple[int, int]
) -> None:
    app = _DialogApp(FOCUS_FACTORIES[name]())
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await wait_until(pilot, lambda: app.focused is not None, max_pauses=30)
        await pilot.pause()
        focused = app.focused
        assert focused is not None
        # A read-only viewer has no control: its body takes the keys.
        if name != "ModelInfoModal":
            assert not isinstance(focused, ScrollableContainer)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["ConfirmDialog", "NoticeDialog", "ApplyProfileDialog"])
async def test_no_scroll_area_holds_focus_while_the_dialog_opens(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    real = Screen.set_focus

    def spy(self: Screen[Any], widget: Widget | None, *args: Any, **kwargs: Any) -> None:
        if widget is not None:
            seen.append(type(widget).__name__)
        return real(self, widget, *args, **kwargs)

    monkeypatch.setattr(Screen, "set_focus", spy)
    app = _DialogApp(FOCUS_FACTORIES[name]())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await wait_until(pilot, lambda: app.focused is not None, max_pauses=30)
        await pilot.pause()
    assert seen
    assert [n for n in seen if n in ("VerticalScroll", "ScrollableContainer")] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
async def test_the_apply_dialog_draws_its_change_table_when_the_pill_has_focus(
    size: tuple[int, int],
) -> None:
    app = _DialogApp(_apply_dialog())
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        table = app.screen.query_one("#apply-changes")
        assert await wait_until(pilot, lambda: table.region.height > 0, max_pauses=30)


def test_every_dialog_with_a_scroll_area_has_a_census_builder() -> None:
    from lilbee.cli.tui.widgets.dialog import Dialog

    surfaces = discover_surfaces()
    scrolling = [
        c
        for c in surfaces
        if issubclass(c, Dialog)
        and re.search(r"VerticalScroll|ScrollableContainer", inspect.getsource(c))
    ]
    assert scrolling
    # The click census opens every surface; a dialog with no builder raises and names itself.
    for dialog in scrolling:
        build(dialog)


# --- the model picker double Enter --------------------------------------------------------------


def _engine_pickers(app: LilbeeAppHost) -> int:
    return sum(isinstance(s, EnginePickerModal) for s in app.screen_stack)


async def _settle(pilot: Any, app: LilbeeAppHost) -> int:
    await wait_until(pilot, lambda: _engine_pickers(app) >= 1)
    for _ in range(10):
        await pilot.pause()
    return _engine_pickers(app)


@pytest.mark.asyncio
async def test_two_selections_before_the_first_is_handled_open_one_engine_picker() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open_picker(app, pilot)
        ml = modal.query_one("#picker-list", ModelList)
        ml.highlighted = 0
        ml.action_select()
        ml.action_select()
        assert await _settle(pilot, app) == 1


@pytest.mark.asyncio
async def test_two_sequential_enter_presses_leave_at_most_one_engine_picker() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_picker(app, pilot)
        await pilot.press("enter", "enter")
        assert await _settle(pilot, app) <= 1


@pytest.mark.asyncio
async def test_the_engine_key_twice_opens_one_engine_picker() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open_picker(app, pilot)
        modal.query_one("#picker-list", ModelList).focus()
        await pilot.pause()
        modal.action_change_engine()
        modal.action_change_engine()
        assert await _settle(pilot, app) == 1


@pytest.mark.asyncio
async def test_the_engine_picker_guard_does_not_stick() -> None:
    app = _PickerApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_picker(app, pilot)
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: _engine_pickers(app) == 1)
        await pilot.press("escape")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ModelPickerModal))
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: _engine_pickers(app) == 1)


@pytest.mark.asyncio
async def test_the_model_info_engine_key_twice_opens_one_engine_picker() -> None:
    from lilbee.cli.tui.screens.model_info import ModelInfoModal

    app = _DialogApp(_info_dialog())
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        info = app.screen
        assert isinstance(info, ModelInfoModal)
        info.action_pick_engine()
        info.action_pick_engine()
        assert await _settle(pilot, app) == 1


@pytest.mark.asyncio
async def test_the_fleet_engine_key_twice_opens_one_engine_picker(
    fleet_applies: list[object],
) -> None:
    from lilbee.cli.tui.widgets.fleet_body import FleetBody
    from lilbee.providers.roles import WorkerRole

    app = _FleetApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one(FleetBody)
        body.pick_engine(WorkerRole.CHAT)
        body.pick_engine(WorkerRole.CHAT)
        assert await _settle(pilot, app) == 1


# --- the splash wait after a kill ---------------------------------------------------------------


class _StuckProcess:
    """A child that survives SIGKILL for the wait (a process in uninterruptible sleep)."""

    pid = 4_000_000

    def __init__(self) -> None:
        self.killed = 0

    def wait(self, timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired("splash", timeout or 0)

    def kill(self) -> None:
        self.killed += 1


def _handle(process: Any) -> Any:
    from lilbee.runtime import splash

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    return splash.SplashHandle(process=process, pipe=io.FileIO(write_fd, "wb"))


@pytest.fixture
def splash_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    from lilbee.runtime import splash

    restored: list[int] = []
    monkeypatch.setattr(splash, "_restore_cursor", lambda: restored.append(1))
    monkeypatch.setattr(splash, "_STOP_TIMEOUT", 0.01)
    yield splash, restored
    splash._active_handle = None


def test_stop_returns_when_the_child_ignores_the_kill(
    splash_env: Any, caplog: pytest.LogCaptureFixture
) -> None:
    splash, restored = splash_env
    proc = _StuckProcess()
    splash.stop(_handle(proc))
    assert proc.killed == 1
    assert restored == [1]
    assert splash._active_handle is None
    assert "outlived SIGKILL" in caplog.text


def test_dismiss_returns_when_the_child_ignores_the_kill(splash_env: Any) -> None:
    splash, _ = splash_env
    proc = _StuckProcess()
    splash._active_handle = _handle(proc)
    splash.dismiss()
    assert proc.killed == 1
    assert splash._active_handle is None


def test_the_exit_hook_returns_when_the_child_ignores_the_kill(splash_env: Any) -> None:
    splash, _ = splash_env
    proc = _StuckProcess()
    splash._active_handle = _handle(proc)
    splash._atexit_cleanup()
    assert proc.killed == 1


def test_a_real_child_that_ignores_the_stop_is_killed_and_reaped(splash_env: Any) -> None:
    splash, _ = splash_env
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL
    )
    splash.stop(_handle(proc))
    assert proc.returncode == -signal.SIGKILL


def test_a_child_that_exits_in_time_is_not_killed(splash_env: Any) -> None:
    splash, restored = splash_env
    proc = SimpleNamespace(killed=0, pid=1)
    proc.wait = lambda timeout=None: 0  # type: ignore[attr-defined]
    proc.kill = lambda: setattr(proc, "killed", proc.killed + 1)  # type: ignore[attr-defined]
    splash.stop(_handle(proc))
    assert proc.killed == 0
    assert restored == [1]
