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
