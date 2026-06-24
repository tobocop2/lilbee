"""Tests for the interactive TUI PlacementScreen.

These drive the real widgets (GPU toggle Buttons, replica steppers, key
bindings) rather than poking private state, so the input path is actually
exercised.
"""

from __future__ import annotations

import threading

import pytest
from textual.widgets import Button, DataTable, Static

from tests._lilbee_app_test_host import LilbeeAppHost

GIB = 1024**3


def _make_view(*, manual: bool = False, unplaceable: tuple = (), spec_json=None):  # type: ignore[type-arg]
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.providers.roles import WorkerRole

    return PlacementView(
        gpus=tuple(
            GpuInfo(i, "CUDA", f"CUDA{i}", "NVIDIA A40", 44 * GIB, 44 * GIB) for i in range(4)
        ),
        roles=(
            RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0,), None, 1),
            RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (1,), None, 1),
        ),
        unplaceable=unplaceable,
        manual=manual,
        spec_json=spec_json,
    )


class PlacementTestApp(LilbeeAppHost):
    """Minimal app host for PlacementScreen tests."""

    CSS = ""

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.placement import PlacementScreen

        self.push_screen(PlacementScreen())


def _generated(app) -> str:  # type: ignore[no-untyped-def]
    from lilbee.cli.tui.screens import placement as screen_mod

    return str(app.screen.query_one(screen_mod._GENERATED_ID, Static).render())


@pytest.mark.asyncio
async def test_screen_lists_gpus_with_roles(monkeypatch):
    """GPU table renders one row per GPU and shows which role sits on each card."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        table = app.screen.query_one(screen_mod._GPU_TABLE_ID, DataTable)
        assert table.row_count == 4
        # row 0 (CUDA0) Roles cell == "embed", row 1 (CUDA1) == "chat"
        assert table.get_row_at(0)[4] == "embed"
        assert table.get_row_at(1)[4] == "chat"


@pytest.mark.asyncio
async def test_toggle_device_updates_spec_and_table(monkeypatch):
    """Clicking a GPU toggle for a role updates the spec and the table live."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#dev-embed-2")  # add CUDA2 to embed
        await pilot.pause()
        assert '"embed": {"devices": [0, 2]}' in _generated(app)
        table = app.screen.query_one(screen_mod._GPU_TABLE_ID, DataTable)
        assert table.get_row_at(2)[4] == "embed"


@pytest.mark.asyncio
async def test_cannot_remove_last_device(monkeypatch):
    """A role must keep at least one GPU; toggling its only device is a no-op."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#dev-embed-0")  # embed's only device -> must stay
        await pilot.pause()
        assert '"embed": {"devices": [0]}' in _generated(app)


@pytest.mark.asyncio
async def test_replica_stepper(monkeypatch):
    """The +/- stepper changes replicas (floored at 1) for a replicated role."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#rep-embed-inc")  # 1 -> 2
        await pilot.pause()
        assert '"replicas": 2' in _generated(app)
        await pilot.click("#rep-embed-dec")  # 2 -> 1 (omitted)
        await pilot.pause()
        assert "replicas" not in _generated(app)
        await pilot.click("#rep-embed-dec")  # floored at 1
        await pilot.pause()
        assert "replicas" not in _generated(app)


@pytest.mark.asyncio
async def test_no_replica_stepper_for_chat(monkeypatch):
    """Non-replicated roles (chat) have no replica stepper."""
    from textual.css.query import NoMatches

    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        with pytest.raises(NoMatches):
            app.screen.query_one("#rep-chat-inc", Button)


@pytest.mark.asyncio
async def test_preview_uses_edited_spec(monkeypatch):
    """ctrl+r resolves the placement built from the toggles, not a stale value."""
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.providers.fleet.placement_spec import PlacementSpec
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())
    calls: list[object] = []

    def _preview(spec):  # type: ignore[no-untyped-def]
        calls.append(spec)
        return _make_view()

    monkeypatch.setattr(screen_mod, "preview_placement", _preview)

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#dev-chat-3")  # chat now on CUDA1 + CUDA3
        await pilot.pause()
        await pilot.press("ctrl+r")
        await pilot.pause()
        await pilot.pause()

    assert calls and isinstance(calls[0], PlacementSpec)
    assert calls[0].roles[WorkerRole.CHAT].devices == (1, 3)


@pytest.mark.asyncio
async def test_apply_uses_edited_spec(monkeypatch):
    """ctrl+s applies the placement built from the toggles."""
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.providers.fleet.placement_spec import PlacementSpec
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())
    calls: list[object] = []
    monkeypatch.setattr(
        screen_mod, "set_placement", lambda spec: calls.append(spec) or _make_view(manual=True)
    )

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#dev-embed-3")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()

    assert calls and isinstance(calls[0], PlacementSpec)
    assert calls[0].roles[WorkerRole.EMBED].devices == (0, 3)


@pytest.mark.asyncio
async def test_clear_calls_set_placement_none(monkeypatch):
    """ctrl+x restores auto placement via set_placement(None)."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())
    calls: list[object] = []
    monkeypatch.setattr(
        screen_mod, "set_placement", lambda spec: calls.append(spec) or _make_view()
    )

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+x")
        await pilot.pause()
        await pilot.pause()

    assert calls == [None]


@pytest.mark.asyncio
async def test_preview_error_notifies(monkeypatch):
    """A PlacementError from preview surfaces as a notification, not a crash."""
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.providers.fleet.placement_spec import PlacementError

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    def _boom(spec):  # type: ignore[no-untyped-def]
        raise PlacementError("chat needs 70 GiB but device 0 has 40 GiB free")

    monkeypatch.setattr(screen_mod, "preview_placement", _boom)

    notes: list[str] = []
    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app.screen, "notify", lambda msg, **k: notes.append(msg))
        await pilot.press("ctrl+r")
        await pilot.pause()
        await pilot.pause()

    assert any("40 GiB free" in n for n in notes)


@pytest.mark.asyncio
async def test_apply_error_notifies(monkeypatch):
    """A PlacementError from set_placement surfaces as a notification."""
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.providers.fleet.placement_spec import PlacementError

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    def _oom(spec):  # type: ignore[no-untyped-def]
        raise PlacementError("oom")

    monkeypatch.setattr(screen_mod, "set_placement", _oom)

    notes: list[str] = []
    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app.screen, "notify", lambda msg, **k: notes.append(msg))
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()

    assert any("oom" in n for n in notes)


@pytest.mark.asyncio
async def test_unplaceable_warns(monkeypatch):
    """A role that does not fit is surfaced as a warning on load."""
    from lilbee.cli.tui.screens import placement as screen_mod
    from lilbee.providers.roles import WorkerRole

    view = _make_view(unplaceable=(WorkerRole.VISION,))
    monkeypatch.setattr(screen_mod, "get_placement", lambda: view)

    notes: list[str] = []
    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        app.screen.notify = lambda msg, **k: notes.append(msg)  # type: ignore[method-assign]
        app.screen._load_placement()
        await pilot.pause()

    assert any("vision" in n.lower() for n in notes)


@pytest.mark.asyncio
async def test_apply_disables_editor_while_running(monkeypatch):
    """ctrl+s sets applying=True and disables the editor until set_placement returns."""
    from textual.containers import Vertical

    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())
    gate = threading.Event()
    monkeypatch.setattr(screen_mod, "set_placement", lambda spec: gate.wait())

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        screen = app.screen
        assert screen.applying is True
        assert screen.query_one(screen_mod._EDITOR_ID, Vertical).disabled is True
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.applying is False


@pytest.mark.asyncio
async def test_apply_ignored_while_applying(monkeypatch):
    """A second ctrl+s while applying is a no-op (single-flight guard)."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())
    calls: list[object] = []
    gate = threading.Event()

    def _blocking(spec):  # type: ignore[no-untyped-def]
        calls.append(spec)
        gate.wait()

    monkeypatch.setattr(screen_mod, "set_placement", _blocking)

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_go_back_binding(monkeypatch):
    """q pops the screen."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, screen_mod.PlacementScreen)
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.asyncio
async def test_go_back_single_screen(monkeypatch):
    """action_go_back with a single-item screen_stack calls switch_view('Chat')."""
    from lilbee.cli.tui.screens import placement as screen_mod

    monkeypatch.setattr(screen_mod, "get_placement", lambda: _make_view())

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        screen = app.screen
        called: list[str] = []
        monkeypatch.setattr(type(app), "screen_stack", property(lambda self: [screen]))
        monkeypatch.setattr(app, "switch_view", lambda v: called.append(v))
        screen.action_go_back()
        await pilot.pause()

    assert called == ["Chat"]


@pytest.mark.asyncio
async def test_load_placement_error_notifies(monkeypatch):
    """An exception from get_placement on load shows a notification."""
    from lilbee.cli.tui.screens import placement as screen_mod

    def _boom():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(screen_mod, "get_placement", _boom)
    notes: list[str] = []

    app = PlacementTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        screen = app.screen
        screen.notify = lambda msg, **k: notes.append(msg)  # type: ignore[method-assign]
        screen._load_placement()
        await pilot.pause()

    assert any("probe failed" in n for n in notes)
