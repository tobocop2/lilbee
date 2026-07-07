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
    """Minimal DeviceLike-compatible stub for tests."""

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
    """Return a DeviceLike-compatible stub."""
    return _FakeDevice(
        index=index, backend=backend, total_bytes=total_bytes, free_bytes=total_bytes
    )


class _PanelHost(LilbeeAppHost):
    """Minimal app host that mounts GpuFleetPanel directly."""

    CSS = ""

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

        yield GpuFleetPanel()


def test_panel_initial_content_is_loading_not_empty() -> None:
    """Before the first probe returns the panel shows a loading state, not the
    empty-GPUs text, so a multi-GPU box doesn't flash '(no GPUs detected)'."""
    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    panel = GpuFleetPanel()
    content = str(panel.render())
    assert msg.FLEET_GPU_PROBING in content
    assert msg.FLEET_NO_GPUS not in content


@pytest.mark.asyncio
async def test_panel_renders_empty_state_with_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed probe that finds no devices shows the empty-state text."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    monkeypatch.setattr(panel_mod, "probe_gpu_stats", lambda devices: {})

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        panel.set_devices([], labels={})
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = str(panel.render())
        assert msg.FLEET_NO_GPUS in rendered


@pytest.mark.asyncio
async def test_panel_holds_probing_until_devices_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The empty-GPUs text must not appear until the parent device probe completes.

    Regression for the cold-probe flash: the panel's own stat probe returns {}
    on a cold start (before set_devices runs), and that must keep the 'probing'
    placeholder, not flash '(no GPUs detected)'.
    """
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    monkeypatch.setattr(panel_mod, "probe_gpu_stats", lambda devices: {})

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        before_probe = str(panel.render())
        assert msg.FLEET_GPU_PROBING in before_probe
        assert msg.FLEET_NO_GPUS not in before_probe

        # A completed probe that genuinely finds no devices does show the empty state.
        panel.set_devices([], labels={})
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        after_probe = str(panel.render())
        assert msg.FLEET_NO_GPUS in after_probe


@pytest.mark.asyncio
async def test_panel_renders_card_label_and_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """With one CUDA GPU the panel renders the label and VRAM figures."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    stat = _make_stat(0, utilization_pct=55, used_bytes=10 * GIB, total_bytes=24 * GIB)
    monkeypatch.setattr(panel_mod, "probe_gpu_stats", lambda devices: {0: stat})

    device = _make_device(0)
    labels = {0: "CUDA0"}

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        panel.set_devices([device], labels=labels)
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
        panel.set_devices([device], labels={0: "CUDA0"})
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
        panel.set_devices([device], labels={0: "Vulkan0"})
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
        panel.set_devices([device], labels={0: "CUDA0"})
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
        panel.set_devices([device], labels={0: "CUDA0"})
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
async def test_tick_skips_while_probe_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tick during a slow probe skips instead of stacking another worker."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as panel_mod
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    monkeypatch.setattr(panel_mod, "probe_gpu_stats", lambda devices: {})

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(GpuFleetPanel)
        await app.workers.wait_for_complete()
        launches: list[int] = []
        monkeypatch.setattr(panel, "_probe_worker", lambda *a: launches.append(1))
        panel._probing = True  # a probe is still running
        panel._request_stats()
        assert launches == []  # skipped, no pile-up
        panel._probing = False  # the running probe finished
        panel._request_stats()
        assert launches == [1]


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
async def test_fleet_screen_mounts_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FleetScreen composes a GpuFleetPanel alongside the GPU table."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel
    from tests.test_tui_fleet import FleetTestApp, _make_view

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        # Panel must be present in the fleet screen's DOM.
        panel = app.screen.query_one(GpuFleetPanel)
        assert panel is not None


@pytest.mark.asyncio
async def test_fleet_screen_passes_devices_to_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """After loading placement the fleet panel receives the GPU device list."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel
    from tests.test_tui_fleet import FleetTestApp, _make_view

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        panel = app.screen.query_one(GpuFleetPanel)
        # The placement view has 4 GPUs (indices 0-3); all must be in the panel.
        assert len(panel._devices) == 4
        assert panel._labels[0] == "CUDA0"


@pytest.mark.asyncio
async def test_panel_renders_role_badge_one_row_per_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each GPU renders as a single table row carrying its role badge."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as pm
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    stat = lambda i, u: pm.GpuStat(i, u, 15 * 1024**3, 47 * 1024**3)  # noqa: E731
    monkeypatch.setattr(pm, "probe_gpu_stats", lambda d: {1: stat(1, 71), 2: stat(2, 68)})
    app = _PanelHost()
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        p = app.query_one(GpuFleetPanel)
        p.set_devices(
            [_make_device(1), _make_device(2)],
            labels={1: "CUDA1", 2: "CUDA2"},
            roles={1: "chat - Qwen3-235B", 2: "chat - Qwen3-235B"},
        )
        p._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        r = str(p.render())
        assert "chat - Qwen3-235B" in r  # badge present
        assert r.count("CUDA") == 2  # both cards
        # one row per GPU: exactly two non-blank lines, no blank separators
        lines = [ln for ln in r.split("\n") if ln.strip()]
        assert len(lines) == 2
        assert "\n\n" not in r


@pytest.mark.asyncio
async def test_panel_uses_resolved_theme_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Theme tokens from app.theme_variables appear in the rendered markup."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as pm
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    # A hot util (>= _UTIL_HOT) should render the error token color.
    stat = _make_stat(0, utilization_pct=85)
    monkeypatch.setattr(pm, "probe_gpu_stats", lambda d: {0: stat})

    sentinel = "#deadbe"
    fake_theme = {"$error": sentinel, "$success": "#aabbcc", "$warning": "#ccbbaa"}

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        p = app.query_one(GpuFleetPanel)
        # Patch _resolve_theme so it returns our known token dict.
        monkeypatch.setattr(
            p,
            "_resolve_theme",
            lambda: {k.lstrip("$"): v for k, v in fake_theme.items()},
        )
        p.set_devices([_make_device(0)], labels={0: "CUDA0"})
        p._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # _Static__content holds the raw markup string passed to update(); render() strips tags.
        raw_markup = str(p._Static__content)  # isinstance guard: Static name-mangles __content
        # The error token color must appear (util 85% >= _UTIL_HOT triggers error heat).
        assert sentinel in raw_markup


@pytest.mark.asyncio
async def test_badge_role_markup_no_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    """A role string without ' - ' renders without splitting (else branch in _badge_role_markup)."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as pm
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    stat = _make_stat(0, utilization_pct=30)
    monkeypatch.setattr(pm, "probe_gpu_stats", lambda d: {0: stat})

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        p = app.query_one(GpuFleetPanel)
        # "chat" has no " - " separator: exercises the else branch in _badge_role_markup
        p.set_devices([_make_device(0)], labels={0: "CUDA0"}, roles={0: "chat"})
        p._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = str(p.render())
        assert "chat" in rendered


@pytest.mark.asyncio
async def test_resolve_theme_falls_back_on_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_theme returns {} when app.theme_variables raises AttributeError."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as pm
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    monkeypatch.setattr(pm, "probe_gpu_stats", lambda d: {})

    app = _PanelHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        p = app.query_one(GpuFleetPanel)
        # Make theme_variables raise AttributeError so the except branch runs.
        type(p.app).theme_variables = property(  # type: ignore[attr-defined]
            lambda self: (_ for _ in ()).throw(AttributeError("no theme_variables"))
        )
        result = p._resolve_theme()
        assert result == {}


@pytest.mark.asyncio
async def test_update_fleet_panel_noop_when_panel_not_mounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_update_fleet_panel is a no-op when GpuFleetPanel is not in the DOM."""
    import lilbee.cli.tui.widgets.gpu_fleet_panel as gfp
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.fleet_body import FleetBody
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel
    from lilbee.providers.roles import WorkerRole

    view = PlacementView(
        gpus=(GpuInfo(0, "CUDA", "CUDA0", "A40", 44 * GIB, 44 * GIB),),
        roles=(RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (0,), None, 1),),
        unplaceable=(),
        manual=False,
        spec_json=None,
    )
    monkeypatch.setattr(fbm, "get_placement", lambda: view)
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda d: {})

    class _NoPanelHost(LilbeeAppHost):
        CSS = ""

        def compose(self) -> ComposeResult:
            yield FleetBody()

    app = _NoPanelHost()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.query_one(FleetBody)
        # Remove the GpuFleetPanel child so NoMatches fires in _update_fleet_panel.
        panel = body.query_one(GpuFleetPanel)
        await panel.remove()
        await pilot.pause()
        # Must not raise; NoMatches is caught and the method returns early.
        body._update_fleet_panel(view)
