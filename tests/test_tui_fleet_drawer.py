"""Tests for the FleetDrawer: ctrl+g docks placement beside a live screen."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Input

from tests._lilbee_app_test_host import LilbeeAppHost

GIB = 1024**3


def _make_view(*, manual: bool = False):  # type: ignore[no-untyped-def]
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.providers.roles import WorkerRole

    return PlacementView(
        gpus=tuple(
            GpuInfo(i, "CUDA", f"CUDA{i}", "NVIDIA A40", 44 * GIB, 44 * GIB) for i in range(2)
        ),
        roles=(
            RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0,), None, 1),
            RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (1,), None, 1),
        ),
        unplaceable=(),
        manual=manual,
        spec_json=None,
    )


class _DrawerApp(LilbeeAppHost):
    """Host with a live Input, standing in for the chat prompt underneath."""

    CSS = ""

    def compose(self) -> ComposeResult:
        yield Input(id="probe-input")


@pytest.fixture
def _patched(monkeypatch):  # type: ignore[no-untyped-def]
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})


@pytest.mark.asyncio
async def test_ctrl_g_opens_drawer_and_esc_closes(_patched):
    """ctrl+g docks the FleetDrawer onto the screen; esc removes it."""
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    app = _DrawerApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        await pilot.pause()
        assert app.screen.query(FleetDrawer)
        # opening does not steal focus, so focus a toggle before esc routes here
        app.screen.query_one(FleetDrawer).query(".dev-toggle").first().focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not app.screen.query(FleetDrawer)


@pytest.mark.asyncio
async def test_ctrl_g_toggles_drawer_closed(_patched):
    """A second ctrl+g closes the open drawer rather than stacking another."""
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    app = _DrawerApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert app.screen.query(FleetDrawer)
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert not app.screen.query(FleetDrawer)


@pytest.mark.asyncio
async def test_chat_stays_interactive_while_drawer_open(_patched):
    """The prompt on the left keeps accepting input while the drawer is docked."""
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    app = _DrawerApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert app.screen.query(FleetDrawer)
        # opening the drawer must NOT steal focus from the prompt
        probe = app.query_one("#probe-input", Input)
        assert app.focused is probe
        await pilot.press("h", "i")
        await pilot.pause()
        assert probe.value == "hi"


@pytest.mark.asyncio
async def test_drawer_reflows_screen_to_the_left(_patched):
    """The drawer docks on the right; the prompt sits to its left."""
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    app = _DrawerApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        drawer = app.screen.query_one(FleetDrawer)
        probe = app.query_one("#probe-input", Input)
        assert probe.region.x == 0
        assert drawer.region.x > probe.region.x


class _BarsApp(LilbeeAppHost):
    """Host whose prompt lives in a docked BottomBars, like the chat screen."""

    CSS = ""

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.bottom_bars import BottomBars

        with BottomBars():
            yield Input(id="probe-input")


@pytest.mark.asyncio
async def test_open_drawer_insets_docked_bars(_patched):
    """The docked bottom bar shrinks left of the drawer instead of hiding under it."""
    from lilbee.cli.tui.widgets.bottom_bars import BottomBars
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    app = _BarsApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        bars = app.screen.query_one(BottomBars)
        assert bars.region.right == 120  # full width before opening
        await pilot.press("ctrl+g")
        await pilot.pause()
        drawer = app.screen.query_one(FleetDrawer)
        assert app.screen.has_class("fleet-open")
        # the bar no longer extends under the drawer
        assert bars.region.right <= drawer.region.x
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert not app.screen.has_class("fleet-open")
        assert bars.region.right == 120  # restored to full width


@pytest.mark.asyncio
async def test_drawer_shows_live_gpu_rows(monkeypatch):
    """The drawer hosts FleetBody, whose GPU table renders one row per card."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel, GpuStat

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    stats = {i: GpuStat(i, 50, 10 * GIB, 47 * GIB) for i in range(2)}
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: stats)

    app = _DrawerApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        panel = app.screen.query_one(FleetDrawer).query_one(GpuFleetPanel)
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert str(panel.render()).count("CUDA") == 2


class _FleetTabApp(LilbeeAppHost):
    """Host that already shows a FleetBody, standing in for the Fleet tab."""

    CSS = ""

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.fleet_body import FleetBody

        yield FleetBody()


@pytest.mark.asyncio
async def test_ctrl_g_noop_when_placement_already_shown(_patched):
    """On the Fleet tab (FleetBody already visible) ctrl+g does not add a drawer."""
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    app = _FleetTabApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert not app.screen.query(FleetDrawer)


@pytest.mark.asyncio
async def test_drawer_delegates_editor_actions(_patched, monkeypatch):
    """preview/apply/clear on the drawer delegate to the hosted FleetBody."""
    from lilbee.cli.tui.widgets.fleet_body import FleetBody
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    calls: list[str] = []
    monkeypatch.setattr(FleetBody, "action_preview", lambda self: calls.append("preview"))
    monkeypatch.setattr(FleetBody, "action_apply", lambda self: calls.append("apply"))
    monkeypatch.setattr(FleetBody, "action_clear", lambda self: calls.append("clear"))

    app = _DrawerApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        drawer = app.screen.query_one(FleetDrawer)
        drawer.action_preview()
        drawer.action_apply()
        drawer.action_clear()
    assert calls == ["preview", "apply", "clear"]


def _make_view3():  # type: ignore[no-untyped-def]
    """Three GPUs with chat on two of them, so a chat toggle can be turned off."""
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.providers.roles import WorkerRole

    return PlacementView(
        gpus=tuple(
            GpuInfo(i, "CUDA", f"CUDA{i}", "NVIDIA A40", 44 * GIB, 44 * GIB) for i in range(3)
        ),
        roles=(
            RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0,), None, 1),
            RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (1, 2), None, 1),
        ),
        unplaceable=(),
        manual=False,
        spec_json=None,
    )


@pytest.mark.asyncio
async def test_normal_tab_focuses_drawer_and_enter_toggles(monkeypatch) -> None:
    """Normal-mode Tab jumps into the open drawer's first toggle; Enter on a
    focused chat chip toggles it off without falling into insert mode."""
    from unittest import mock

    from lilbee.app.services import set_services
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp
    from lilbee.cli.tui.widgets.fleet_body import FleetBody
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view3())
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})
    svc = mock.MagicMock()
    svc.provider.list_models.return_value = []
    svc.searcher._embedder.embedding_available.return_value = True
    set_services(svc)
    try:
        cs = "lilbee.cli.tui.screens.chat.ChatScreen"
        with (
            mock.patch(f"{cs}._needs_setup", return_value=False),
            mock.patch(f"{cs}._embedding_ready", return_value=True),
        ):
            app = LilbeeApp()
            async with app.run_test(size=(140, 40)) as pilot:
                await pilot.pause()
                await pilot.press("ctrl+g")
                await pilot.pause()
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
                await pilot.press("tab")
                await pilot.pause()
                assert app.focused is not None
                assert "dev-toggle" in app.focused.classes
                # focus the "on" chat GPU-2 chip and toggle it off with Enter
                app.screen.query_one("#dev-chat-2").focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                body = app.screen.query_one(FleetBody)
                assert 2 not in body._edits[WorkerRole.CHAT].devices
                assert app.screen._insert_mode is False
                # Tab while already inside the drawer advances focus (no crash,
                # no drop into insert mode).
                app.screen.query_one("#dev-chat-1").focus()
                await pilot.pause()
                await pilot.press("tab")
                await pilot.pause()
                assert app.screen._insert_mode is False
    finally:
        set_services(None)


@pytest.mark.asyncio
async def test_command_buttons_are_clickable(_patched, monkeypatch):  # type: ignore[no-untyped-def]
    """Clicking Preview/Apply/Auto fires the editor actions, so the drawer is
    fully mouse-operable without the ctrl+r/s/x keys."""
    from lilbee.cli.tui.widgets.fleet_body import FleetBody

    calls: list[str] = []
    monkeypatch.setattr(FleetBody, "action_preview", lambda self: calls.append("preview"))
    monkeypatch.setattr(FleetBody, "action_apply", lambda self: calls.append("apply"))
    monkeypatch.setattr(FleetBody, "action_clear", lambda self: calls.append("clear"))

    app = _DrawerApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        await pilot.click("#cmd-preview")
        await pilot.click("#cmd-apply")
        await pilot.click("#cmd-auto")
        await pilot.pause()
    assert calls == ["preview", "apply", "clear"]


@pytest.mark.asyncio
async def test_chip_click_toggles_device_off(monkeypatch):  # type: ignore[no-untyped-def]
    """Clicking a GPU chip toggles that device for the role, no keyboard needed."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp
    from lilbee.cli.tui.widgets.fleet_body import FleetBody
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view3())
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})

    app = _DrawerApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        await pilot.click("#dev-chat-2")
        await pilot.pause()
        body = app.screen.query_one(FleetBody)
        assert 2 not in body._edits[WorkerRole.CHAT].devices
        assert "on" not in app.screen.query_one("#dev-chat-2").classes


@pytest.mark.asyncio
async def test_chat_input_enter_yields_when_unfocused() -> None:
    """The chat input's Enter yields (SkipAction) when it lacks focus, so Enter
    reaches a focused drawer toggle instead of submitting the prompt."""
    from textual.actions import SkipAction

    from lilbee.cli.tui.widgets.chat_input import ChatInput

    inp = ChatInput()
    with pytest.raises(SkipAction):
        inp.action_submit()
