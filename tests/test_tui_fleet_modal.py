"""Tests for the FleetModal overlay: open with ctrl+g, dismiss with escape."""

from __future__ import annotations

import pytest

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


class LilbeeTestApp(LilbeeAppHost):
    """Minimal app host for fleet-modal tests."""

    CSS = ""


@pytest.mark.asyncio
async def test_ctrl_g_opens_fleet_modal_and_esc_closes(monkeypatch):
    """ctrl+g opens FleetModal; escape dismisses it and returns to the prior screen."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp
    from lilbee.cli.tui.widgets.fleet_modal import FleetModal

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})

    app = LilbeeTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert isinstance(app.screen, FleetModal)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, FleetModal)


@pytest.mark.asyncio
async def test_ctrl_g_does_not_stack_second_fleet_modal(monkeypatch):
    """Pressing ctrl+g while FleetModal is open is a no-op (re-entry guard)."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp
    from lilbee.cli.tui.widgets.fleet_modal import FleetModal

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})

    app = LilbeeTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert isinstance(app.screen, FleetModal)
        depth_before = len(app.screen_stack)
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert isinstance(app.screen, FleetModal)
        assert len(app.screen_stack) == depth_before


@pytest.mark.asyncio
async def test_fleet_modal_shows_fleet_panel(monkeypatch):
    """FleetModal mounts FleetBody which renders the live GPU table (one row per GPU)."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp
    from lilbee.cli.tui.widgets.fleet_modal import FleetModal
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel, GpuStat

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    stats = {i: GpuStat(i, 50, 10 * 1024**3, 47 * 1024**3) for i in range(2)}
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: stats)

    app = LilbeeTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert isinstance(app.screen, FleetModal)
        panel = app.screen.query_one(GpuFleetPanel)
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert str(panel.render()).count("CUDA") == 2


@pytest.mark.asyncio
async def test_fleet_modal_delegates_preview_apply_clear(monkeypatch):
    """ctrl+r/ctrl+s/ctrl+x inside FleetModal delegate to FleetBody actions."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp
    from lilbee.cli.tui.widgets.fleet_modal import FleetModal

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})

    preview_calls: list[object] = []
    apply_calls: list[object] = []
    clear_calls: list[object] = []

    def _preview(spec):  # type: ignore[no-untyped-def]
        preview_calls.append(spec)
        return _make_view()

    monkeypatch.setattr(fbm, "preview_placement", _preview)
    monkeypatch.setattr(fbm, "set_placement", lambda spec: apply_calls.append(spec))

    app = LilbeeTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert isinstance(app.screen, FleetModal)

        # ctrl+r exercises action_preview on the modal -> FleetBody.action_preview
        await pilot.press("ctrl+r")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert preview_calls

        # ctrl+s exercises action_apply on the modal -> FleetBody.action_apply
        monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
        await pilot.press("ctrl+s")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert apply_calls

        # ctrl+x exercises action_clear on the modal -> FleetBody.action_clear
        monkeypatch.setattr(fbm, "set_placement", lambda spec: clear_calls.append(spec))
        monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
        await pilot.press("ctrl+x")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert clear_calls


@pytest.mark.asyncio
async def test_ctrl_g_noop_on_fleet_screen(monkeypatch):
    """ctrl+g while on the Fleet view (which hosts a FleetBody) does not push a modal."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp
    from lilbee.cli.tui.widgets.fleet_modal import FleetModal

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})

    class _FleetViewApp(LilbeeTestApp):
        def on_mount(self) -> None:
            from lilbee.cli.tui.screens.fleet import FleetScreen

            self.push_screen(FleetScreen())

    app = _FleetViewApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        depth_before = len(app.screen_stack)
        await pilot.press("ctrl+g")
        await pilot.pause()
        # FleetBody already present: guard must prevent a second push
        assert not isinstance(app.screen, FleetModal)
        assert len(app.screen_stack) == depth_before
