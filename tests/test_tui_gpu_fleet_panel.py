"""Tests for the GpuFleetPanel widget.

Drives the real widget via a minimal Textual app host.  GPU probes are
monkeypatched at the module boundary (probe_gpu_stats) so no nvidia-smi
subprocess is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from textual.app import ComposeResult

from tests._lilbee_app_test_host import LilbeeAppHost

GIB = 1024**3


@dataclass(frozen=True)
class _FakeDevice:
    """Minimal _DeviceLike-compatible stub for tests."""

    index: int
    backend: str
    total_bytes: int
    free_bytes: int


def _make_stat(
    index: int,
    utilization_pct: int | None = 42,
    used_bytes: int = 10 * GIB,
    total_bytes: int = 24 * GIB,
) -> object:
    from lilbee.providers.fleet.gpu_stats import GpuStat

    free = total_bytes - used_bytes
    return GpuStat(
        index=index,
        utilization_pct=utilization_pct,
        free_bytes=free,
        total_bytes=total_bytes,
    )


def _make_device(index: int, backend: str = "CUDA", total_bytes: int = 24 * GIB) -> _FakeDevice:
    """Return a _DeviceLike-compatible stub."""
    return _FakeDevice(
        index=index, backend=backend, total_bytes=total_bytes, free_bytes=total_bytes
    )


class _PanelHost(LilbeeAppHost):
    """Minimal app host that mounts GpuFleetPanel directly."""

    CSS = ""

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

        yield GpuFleetPanel()


@pytest.mark.asyncio
async def test_panel_renders_empty_state_with_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without any devices the panel shows the empty-state text."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    monkeypatch.setattr(panel_mod, "probe_gpu_stats", lambda devices: {})

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        rendered = str(panel.render())
        assert panel_mod._EMPTY_TEXT in rendered


@pytest.mark.asyncio
async def test_panel_renders_card_label_and_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """With one CUDA GPU the panel renders the label and VRAM figures."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    stat = _make_stat(0, utilization_pct=55, used_bytes=10 * GIB, total_bytes=24 * GIB)
    monkeypatch.setattr(panel_mod, "probe_gpu_stats", lambda devices: {0: stat})

    device = _make_device(0)
    labels = {0: "CUDA0"}
    names = {0: "NVIDIA A40"}

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        panel.set_devices([device], labels=labels, names=names)
        # Trigger a manual tick so we don't wait for the 1 s timer.
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = str(panel.render())
        assert "CUDA0" in rendered
        # VRAM numerics: 10.0/24G
        assert "10.0" in rendered
        assert "24" in rendered


@pytest.mark.asyncio
async def test_panel_renders_utilization_percentage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Utilization percentage appears in the rendered output for a CUDA GPU."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    stat = _make_stat(0, utilization_pct=73)
    monkeypatch.setattr(panel_mod, "probe_gpu_stats", lambda devices: {0: stat})

    device = _make_device(0)

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        panel.set_devices([device], labels={0: "CUDA0"}, names={0: "RTX 3090"})
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = str(panel.render())
        assert "73%" in rendered


@pytest.mark.asyncio
async def test_panel_shows_dash_when_util_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-CUDA backends (utilization_pct=None) render a dash, not a number."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import _UTIL_DASH, GpuFleetPanel

    stat = _make_stat(0, utilization_pct=None)
    monkeypatch.setattr(panel_mod, "probe_gpu_stats", lambda devices: {0: stat})

    device = _make_device(0, backend="Vulkan")

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        panel.set_devices([device], labels={0: "Vulkan0"}, names={0: "AMD Radeon"})
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = str(panel.render())
        assert _UTIL_DASH.strip() in rendered


@pytest.mark.asyncio
async def test_panel_updates_on_second_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """The panel re-renders when stats change between ticks.

    The initial on_mount probe fires with empty devices (empty result); the
    two explicit ticks return 10% then 90% and we assert each change lands.
    """
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    results = [
        {},  # initial on_mount probe (no devices yet)
        {0: _make_stat(0, utilization_pct=10)},
        {0: _make_stat(0, utilization_pct=90)},
    ]
    probe_calls: list[int] = []

    def _probe(devices: object) -> object:  # type: ignore[no-untyped-def]
        idx = len(probe_calls)
        probe_calls.append(idx)
        return results[min(idx, len(results) - 1)]

    monkeypatch.setattr(panel_mod, "probe_gpu_stats", _probe)

    device = _make_device(0)

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        # Set devices after mount so explicit ticks have the right labels.
        panel.set_devices([device], labels={0: "CUDA0"}, names={0: "A100"})
        await app.workers.wait_for_complete()

        # First explicit tick
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        first_render = str(panel.render())
        assert "10%" in first_render

        # Second explicit tick
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        second_render = str(panel.render())
        assert "90%" in second_render


@pytest.mark.asyncio
async def test_panel_graceful_on_probe_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing probe leaves the previous content intact and does not crash.

    The initial on_mount probe returns an empty dict (no devices set yet).
    The first explicit tick returns good stats with the right labels set.
    The second explicit tick raises; the panel must keep the previous content.
    """
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    good_stat = _make_stat(0, utilization_pct=50)
    probe_calls: list[int] = []

    def _probe(devices: object) -> object:  # type: ignore[no-untyped-def]
        idx = len(probe_calls)
        probe_calls.append(idx)
        if idx == 0:
            # on_mount probe: no devices registered yet
            return {}
        if idx == 1:
            return {0: good_stat}
        raise OSError("nvidia-smi not found")

    monkeypatch.setattr(panel_mod, "probe_gpu_stats", _probe)

    device = _make_device(0)

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        panel.set_devices([device], labels={0: "CUDA0"}, names={0: "A100"})
        await app.workers.wait_for_complete()

        # First explicit tick returns good stats
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        content_after_good = str(panel.render())
        assert "CUDA0" in content_after_good

        # Second explicit tick raises; content must remain unchanged
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        content_after_fail = str(panel.render())
        assert "CUDA0" in content_after_fail


@pytest.mark.asyncio
async def test_panel_timer_stops_on_unmount(monkeypatch: pytest.MonkeyPatch) -> None:
    """The internal timer is cleared when the panel is unmounted."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    monkeypatch.setattr(panel_mod, "probe_gpu_stats", lambda devices: {})

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        # Timer is set after mount
        assert panel._timer is not None
    # After the context exits the app stops; the timer must be None or stopped.
    # The on_unmount callback is the mechanism; we verify it ran without error.
    assert panel._timer is None


@pytest.mark.asyncio
async def test_placement_screen_mounts_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PlacementScreen composes a GpuFleetPanel alongside the GPU table."""
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel
    from tests.test_tui_placement import PlacementTestApp, _make_view

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        # Panel must be present in the placement screen's DOM.
        panel = app.screen.query_one(GpuFleetPanel)
        assert panel is not None


@pytest.mark.asyncio
async def test_placement_screen_passes_devices_to_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """After loading placement the fleet panel receives the GPU device list."""
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel
    from tests.test_tui_placement import PlacementTestApp, _make_view

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        panel = app.screen.query_one(GpuFleetPanel)
        # The placement view has 4 GPUs (indices 0-3); all must be in the panel.
        assert len(panel._devices) == 4
        assert panel._labels[0] == "CUDA0"
        assert panel._names[0] == "NVIDIA A40"
