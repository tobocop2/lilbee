"""Tests for the TUI PlacementScreen."""

from __future__ import annotations

import pytest
from textual.widgets import DataTable

from tests._lilbee_app_test_host import LilbeeAppHost

GIB = 1024**3


def _make_view(*, manual: bool = False, unplaceable: tuple = ()):  # type: ignore[type-arg]
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.providers.roles import WorkerRole

    return PlacementView(
        gpus=(
            GpuInfo(0, "CUDA", "CUDA0", "NVIDIA A100", 80 * GIB, 72 * GIB),
            GpuInfo(1, "CUDA", "CUDA1", "NVIDIA A100", 80 * GIB, 80 * GIB),
        ),
        roles=(RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (0, 1), (1, 1), 1),),
        unplaceable=unplaceable,
        manual=manual,
        spec_json=None,
    )


class PlacementTestApp(LilbeeAppHost):
    """Minimal app host for PlacementScreen tests."""

    CSS = ""

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.placement import PlacementScreen

        self.push_screen(PlacementScreen())


@pytest.mark.asyncio
async def test_screen_lists_gpus(monkeypatch):
    """GPU table renders one row per detected GPU."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        table = app.screen.query_one(screen_mod._GPU_TABLE_ID, DataTable)
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_screen_lists_roles(monkeypatch):
    """Role summary reflects placed roles."""
    from textual.widgets import Static

    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        summary = app.screen.query_one(screen_mod._ROLE_SUMMARY_ID, Static)
        rendered = str(summary.render())
        assert "chat" in rendered.lower()


@pytest.mark.asyncio
async def test_screen_unplaceable_shown(monkeypatch):
    """Unplaceable roles appear in the summary."""
    from textual.widgets import Static

    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.providers.roles import WorkerRole

    view = _make_view(unplaceable=(WorkerRole.VISION,))
    monkeypatch.setattr(screen_mod, "get_placement", lambda: view)

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        summary = app.screen.query_one(screen_mod._ROLE_SUMMARY_ID, Static)
        rendered = str(summary.render())
        assert "vision" in rendered.lower()


@pytest.mark.asyncio
async def test_preview_rerenders(monkeypatch):
    """ctrl+r with a valid spec re-renders the GPU table."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())
    preview_calls: list[object] = []

    def _preview(spec):  # type: ignore[no-untyped-def]
        preview_calls.append(spec)
        return _make_view()

    monkeypatch.setattr(screen_mod, "preview_placement", _preview)

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen._spec_text = '{"chat": {"devices": [0]}}'
        await pilot.press("ctrl+r")
        await pilot.pause()
        await pilot.pause()

    assert len(preview_calls) == 1


@pytest.mark.asyncio
async def test_preview_shows_fit_error(monkeypatch):
    """ctrl+r surfaces a PlacementError as a notification, not a crash."""
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.providers.fleet.placement_spec import PlacementError

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    def _boom(spec):  # type: ignore[no-untyped-def]
        raise PlacementError("chat needs 70 GiB but device 0 has 40 GiB free")

    monkeypatch.setattr(screen_mod, "preview_placement", _boom)

    notes: list[str] = []
    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen._spec_text = '{"chat": {"devices": [0]}}'
        monkeypatch.setattr(screen, "notify", lambda msg, **k: notes.append(msg))
        await pilot.press("ctrl+r")
        await pilot.pause()
        await pilot.pause()

    assert any("40 GiB free" in n for n in notes)


@pytest.mark.asyncio
async def test_apply_calls_set_placement(monkeypatch):
    """ctrl+s calls set_placement with parsed spec."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())
    set_calls: list[object] = []
    monkeypatch.setattr(
        screen_mod, "set_placement", lambda spec: set_calls.append(spec) or _make_view()
    )
    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen._spec_text = '{"chat": {"devices": [0]}}'
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()

    assert len(set_calls) == 1


@pytest.mark.asyncio
async def test_clear_calls_set_placement_none(monkeypatch):
    """ctrl+x calls set_placement(None) to restore auto placement."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())
    set_calls: list[object] = []

    def _set(spec):  # type: ignore[no-untyped-def]
        set_calls.append(spec)
        return _make_view()

    monkeypatch.setattr(screen_mod, "set_placement", _set)

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+x")
        await pilot.pause()
        await pilot.pause()

    assert set_calls == [None]


@pytest.mark.asyncio
async def test_apply_bad_json_notifies(monkeypatch):
    """ctrl+s with invalid JSON shows an error notification."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())
    notes: list[str] = []

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen._spec_text = "not-valid-json"
        monkeypatch.setattr(screen, "notify", lambda msg, **k: notes.append(msg))
        await pilot.press("ctrl+s")
        await pilot.pause()

    assert any(notes)


@pytest.mark.asyncio
async def test_go_back_binding(monkeypatch):
    """q pops the screen."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, screen_mod.PlacementScreen)
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.asyncio
async def test_load_placement_error_notifies(monkeypatch):
    """An exception from get_placement on mount shows a notification."""
    from lilbee.cli.tui.screens import placement as screen_mod

    def _boom():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(screen_mod, "get_placement", _boom)
    notes: list[str] = []

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        monkeypatch.setattr(screen, "notify", lambda msg, **k: notes.append(msg))
        screen._load_placement()
        await pilot.pause()

    assert any("probe failed" in n for n in notes)


@pytest.mark.asyncio
async def test_render_view_with_spec_json(monkeypatch):
    """When spec_json is set on the view, the TextArea is seeded."""
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.providers.roles import WorkerRole

    GIB = 1024**3
    view_with_spec = PlacementView(
        gpus=(GpuInfo(0, "CUDA", "CUDA0", "NVIDIA A100", 80 * GIB, 72 * GIB),),
        roles=(RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (0,), None, 1),),
        unplaceable=(),
        manual=True,
        spec_json='{"chat": {"devices": [0]}}',
    )
    monkeypatch.setattr(screen_mod, "get_placement", lambda: view_with_spec)

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen._spec_text == '{"chat": {"devices": [0]}}'


@pytest.mark.asyncio
async def test_apply_worker_error_notifies(monkeypatch):
    """An exception from set_placement shows a notification."""
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.providers.fleet.placement_spec import PlacementError

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    def _oom(spec):  # type: ignore[no-untyped-def]
        raise PlacementError("oom")

    monkeypatch.setattr(screen_mod, "set_placement", _oom)

    notes: list[str] = []
    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen._spec_text = '{"chat": {"devices": [0]}}'
        monkeypatch.setattr(screen, "notify", lambda msg, **k: notes.append(msg))
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()

    assert any("oom" in n for n in notes)


@pytest.mark.asyncio
async def test_clear_worker_error_notifies(monkeypatch):
    """An exception from set_placement(None) during clear shows a notification."""
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.providers.fleet.placement_spec import PlacementError

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    def _fail(spec):  # type: ignore[no-untyped-def]
        raise PlacementError("clear fail")

    monkeypatch.setattr(screen_mod, "set_placement", _fail)

    notes: list[str] = []
    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        monkeypatch.setattr(screen, "notify", lambda msg, **k: notes.append(msg))
        await pilot.press("ctrl+x")
        await pilot.pause()
        await pilot.pause()

    assert any("clear fail" in n for n in notes)


@pytest.mark.asyncio
async def test_go_back_single_screen(monkeypatch):
    """action_go_back with a single-item screen_stack calls switch_view."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        called: list[str] = []
        # Simulate single-screen stack so the else branch is taken.
        monkeypatch.setattr(type(app), "screen_stack", property(lambda self: [screen]))
        monkeypatch.setattr(app, "switch_view", lambda v: called.append(v))
        screen.action_go_back()
        await pilot.pause()

    assert called == ["Chat"]


@pytest.mark.asyncio
async def test_placement_screen_app_harness(monkeypatch):
    """PlacementScreenApp mounts PlacementScreen successfully."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = screen_mod.PlacementScreenApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, screen_mod.PlacementScreen)
