"""Tests for the Fleet view: FleetScreen hosting FleetBody.

These drive the real widgets (GPU toggle pills, replica steppers, key
bindings) rather than poking private state, so the input path is actually
exercised.
"""

from __future__ import annotations

import threading

import pytest

from lilbee.cli.tui.widgets.fleet_body import FleetPill
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


class FleetTestApp(LilbeeAppHost):
    """Minimal app host for FleetScreen tests."""

    CSS = ""

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.fleet import FleetScreen

        self.push_screen(FleetScreen())


def _generated(app) -> str:  # type: ignore[no-untyped-def]
    """The equivalent-spec JSON the editor state produces.

    The on-screen spec readout was removed for a cleaner drawer; the underlying
    ``_spec_from_editor`` it rendered is what these tests actually assert on.
    """
    from lilbee.cli.tui.widgets.fleet_body import FleetBody
    from lilbee.providers.fleet.placement_spec import PlacementError

    body = app.screen.query_one(FleetBody)
    try:
        spec = body._spec_from_editor()
    except PlacementError as exc:
        return str(exc)
    return spec.to_json() if spec else "(auto)"


async def _toggle_device(pilot, selector: str, *, expect_on: bool = True) -> None:  # type: ignore[no-untyped-def]
    """Activate a GPU toggle and wait until the press has been applied.

    ``pilot.click`` resolves the target's screen coordinates up front, so under
    parallel load it can fire before layout settles and miss the pill
    entirely. ``FleetPill.press`` posts ``FleetPill.Pressed`` directly (no
    coordinates), and the handler flips the ``on`` class in the same step that
    mutates the device set -- so wait for that class as the post-condition.
    """
    pill = pilot.app.screen.query_one(selector, FleetPill)
    pill.press()
    for _ in range(100):
        await pilot.pause()
        if pill.has_class("on") == expect_on:
            return
    raise AssertionError(f"{selector} did not reach on={expect_on}")  # pragma: no cover


async def _step_until_generated(pilot, selector: str, app, predicate) -> None:  # type: ignore[no-untyped-def]
    """Activate a control and wait until the generated spec satisfies ``predicate``.

    The replica stepper has no ``on`` class to watch, so synchronise on the
    equivalent-spec text it drives. Uses ``FleetPill.press`` for the same
    coordinate-free reason as ``_toggle_device``.
    """
    pilot.app.screen.query_one(selector, FleetPill).press()
    for _ in range(100):
        await pilot.pause()
        if predicate(_generated(app)):
            return
    raise AssertionError(
        f"{selector}: generated never matched: {_generated(app)!r}"
    )  # pragma: no cover


def _make_view_with_skipped():
    from lilbee.app.placement import (
        GpuInfo,
        PlacementView,
        RolePlacementView,
        SkippedRole,
    )
    from lilbee.providers.roles import WorkerRole

    return PlacementView(
        gpus=(GpuInfo(0, "MTL", "MTL0", "Apple M3", 24 * GIB, 20 * GIB),),
        roles=(RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0,), None, 1),),
        unplaceable=(),
        manual=False,
        spec_json=None,
        skipped_not_installed=(SkippedRole(WorkerRole.CHAT, "Qwen/Qwen3-4B-GGUF/Q4.gguf"),),
    )


@pytest.mark.asyncio
async def test_skipped_role_shows_not_downloaded_note(monkeypatch):
    """A configured role skipped for a missing model shows a legible 'not downloaded'
    line, not an unexplained empty table."""
    from textual.widgets import Static

    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", _make_view_with_skipped)

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        note = app.screen.query_one("#placement-skipped", Static)
        assert note.display is True
        rendered = str(note.render())
        assert "chat" in rendered
        assert "not downloaded" in rendered
        assert "Qwen3-4B" in rendered


@pytest.mark.asyncio
async def test_no_skipped_note_when_all_models_installed(monkeypatch):
    """With every configured model installed the note stays hidden."""
    from textual.widgets import Static

    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        assert app.screen.query_one("#placement-skipped", Static).display is False


@pytest.mark.asyncio
async def test_fleet_view_shows_live_rows_with_role_badges(monkeypatch):
    """Fleet view mounts FleetBody, which pushes role badges into GpuFleetPanel."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        panel = app.screen.query_one(GpuFleetPanel)
        # chat is on device 1; role badge for device 1 must start with "chat"
        assert panel._roles[1].startswith("chat")


@pytest.mark.asyncio
async def test_screen_lists_gpus_with_roles(monkeypatch):
    """The live GPU table renders one row per GPU and shows each card's role."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel, GpuStat

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    stats = {i: GpuStat(i, None, 10 * GIB, 47 * GIB) for i in range(4)}
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: stats)

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        panel = app.screen.query_one(GpuFleetPanel)
        panel._request_stats()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = str(panel.render())
        assert rendered.count("CUDA") == 4  # one row per GPU
        assert "embed" in rendered  # CUDA0's role
        assert "chat" in rendered  # CUDA1's role


@pytest.mark.asyncio
async def test_toggle_device_updates_spec(monkeypatch):
    """Clicking a GPU toggle for a role updates the generated spec live."""
    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await _toggle_device(pilot, "#dev-embed-2")  # add CUDA2 to embed
        assert '"embed": {"devices": [0, 2]}' in _generated(app)


def _make_split_view():  # type: ignore[no-untyped-def]
    """A view whose chat role carries a manual, uneven tensor split across CUDA1+2."""
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.providers.roles import WorkerRole

    return PlacementView(
        gpus=tuple(
            GpuInfo(i, "CUDA", f"CUDA{i}", "NVIDIA A40", 44 * GIB, 44 * GIB) for i in range(4)
        ),
        roles=(RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (1, 2), (3, 1), 1),),
        unplaceable=(),
        manual=True,
        spec_json=None,
    )


@pytest.mark.asyncio
async def test_editor_preserves_manual_tensor_split_when_unedited(monkeypatch):
    """A loaded manual split is re-emitted verbatim, so re-applying an untouched
    placement doesn't fall back to an even split that OOMs unequal cards."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.fleet_body import FleetBody
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(fbm, "get_placement", _make_split_view)
    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        spec = app.screen.query_one(FleetBody)._spec_from_editor()
        assert spec.roles[WorkerRole.CHAT].tensor_split == (3, 1)


@pytest.mark.asyncio
async def test_editing_devices_clears_stale_tensor_split(monkeypatch):
    """Toggling a role's devices drops the loaded split: its length no longer fits
    the new card set, so the planner re-derives a capacity split."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.fleet_body import FleetBody
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(fbm, "get_placement", _make_split_view)
    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await _toggle_device(pilot, "#dev-chat-3")  # chat now on CUDA1+2+3
        spec = app.screen.query_one(FleetBody)._spec_from_editor()
        assert spec.roles[WorkerRole.CHAT].devices == (1, 2, 3)
        assert spec.roles[WorkerRole.CHAT].tensor_split is None


@pytest.mark.asyncio
async def test_cannot_remove_last_device(monkeypatch):
    """A role must keep at least one GPU; toggling its only device is a no-op."""
    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await _toggle_device(pilot, "#dev-embed-0", expect_on=True)  # only device -> stays
        assert '"embed": {"devices": [0]}' in _generated(app)


@pytest.mark.asyncio
async def test_replica_stepper(monkeypatch):
    """The +/- stepper changes replicas (floored at 1) for a replicated role."""
    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await _step_until_generated(pilot, "#rep-embed-inc", app, lambda g: '"replicas": 2' in g)
        assert '"replicas": 2' in _generated(app)
        await _step_until_generated(pilot, "#rep-embed-dec", app, lambda g: "replicas" not in g)
        assert "replicas" not in _generated(app)
        app.screen.query_one("#rep-embed-dec", FleetPill).press()  # floored at 1 (stays omitted)
        await pilot.pause()
        assert "replicas" not in _generated(app)


@pytest.mark.asyncio
async def test_no_replica_stepper_for_chat(monkeypatch):
    """Non-replicated roles (chat) have no replica stepper."""
    from textual.css.query import NoMatches

    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        with pytest.raises(NoMatches):
            app.screen.query_one("#rep-chat-inc", FleetPill)


@pytest.mark.asyncio
async def test_preview_uses_edited_spec(monkeypatch):
    """ctrl+r resolves the placement built from the toggles, not a stale value."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.providers.fleet.placement_spec import PlacementSpec
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    calls: list[object] = []

    def _preview(spec):  # type: ignore[no-untyped-def]
        calls.append(spec)
        return _make_view()

    monkeypatch.setattr(fbm, "preview_placement", _preview)

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await _toggle_device(pilot, "#dev-chat-3")  # chat now on CUDA1 + CUDA3
        await pilot.press("ctrl+r")
        await pilot.pause()
        await pilot.pause()

    assert calls and isinstance(calls[0], PlacementSpec)
    assert calls[0].roles[WorkerRole.CHAT].devices == (1, 3)


@pytest.mark.asyncio
async def test_apply_uses_edited_spec(monkeypatch):
    """ctrl+s applies the placement built from the toggles."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.providers.fleet.placement_spec import PlacementSpec
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    calls: list[object] = []
    monkeypatch.setattr(
        fbm, "set_placement", lambda spec: calls.append(spec) or _make_view(manual=True)
    )

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await _toggle_device(pilot, "#dev-embed-3")
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()

    assert calls and isinstance(calls[0], PlacementSpec)
    assert calls[0].roles[WorkerRole.EMBED].devices == (0, 3)


@pytest.mark.asyncio
async def test_clear_calls_set_placement_none(monkeypatch):
    """ctrl+x restores auto placement via set_placement(None)."""
    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    calls: list[object] = []
    monkeypatch.setattr(fbm, "set_placement", lambda spec: calls.append(spec) or _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+x")
        await pilot.pause()
        await pilot.pause()

    assert calls == [None]


@pytest.mark.asyncio
async def test_preview_error_notifies(monkeypatch):
    """A PlacementError from preview surfaces as a notification, not a crash."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.providers.fleet.placement_spec import PlacementError

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    def _boom(spec):  # type: ignore[no-untyped-def]
        raise PlacementError("chat needs 70 GiB but device 0 has 40 GiB free")

    monkeypatch.setattr(fbm, "preview_placement", _boom)

    notes: list[str] = []
    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        monkeypatch.setattr(body, "notify", lambda msg, **k: notes.append(msg))
        await pilot.press("ctrl+r")
        await pilot.pause()
        await pilot.pause()

    assert any("40 GiB free" in n for n in notes)


@pytest.mark.asyncio
async def test_apply_error_notifies(monkeypatch):
    """A PlacementError from set_placement surfaces as a notification."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.providers.fleet.placement_spec import PlacementError

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    def _oom(spec):  # type: ignore[no-untyped-def]
        raise PlacementError("oom")

    monkeypatch.setattr(fbm, "set_placement", _oom)

    notes: list[str] = []
    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        monkeypatch.setattr(body, "notify", lambda msg, **k: notes.append(msg))
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()

    assert any("oom" in n for n in notes)


@pytest.mark.asyncio
async def test_unplaceable_warns(monkeypatch):
    """A role that does not fit is surfaced as a warning on load."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.providers.roles import WorkerRole

    view = _make_view(unplaceable=(WorkerRole.VISION,))
    monkeypatch.setattr(fbm, "get_placement", lambda: view)

    notes: list[str] = []
    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        body = app.screen.query_one("FleetBody")
        body.notify = lambda msg, **k: notes.append(msg)  # type: ignore[method-assign]
        body._load_worker()
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("vision" in n.lower() for n in notes)


@pytest.mark.asyncio
async def test_apply_disables_editor_while_running(monkeypatch):
    """ctrl+s sets applying=True and disables the editor until set_placement returns."""
    from textual.containers import Vertical

    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.fleet_body import _EDITOR_ID

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    gate = threading.Event()
    monkeypatch.setattr(fbm, "set_placement", lambda spec: gate.wait())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        assert body.applying is True
        assert body.query_one(_EDITOR_ID, Vertical).disabled is True
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert body.applying is False


@pytest.mark.asyncio
async def test_apply_ignored_while_applying(monkeypatch):
    """A second ctrl+s while applying is a no-op (single-flight guard)."""
    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    calls: list[object] = []
    gate = threading.Event()

    def _blocking(spec):  # type: ignore[no-untyped-def]
        calls.append(spec)
        gate.wait()

    monkeypatch.setattr(fbm, "set_placement", _blocking)

    app = FleetTestApp()
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
    """q routes back to Chat via the guarded switch_view."""
    from lilbee.cli.tui.screens.fleet import FleetScreen
    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, FleetScreen)
        called: list[str] = []
        monkeypatch.setattr(app, "switch_view", lambda v: called.append(v))
        await pilot.press("q")
        await pilot.pause()

    assert called == ["Chat"]


@pytest.mark.asyncio
async def test_load_placement_error_notifies(monkeypatch):
    """An exception from get_placement on load shows a notification."""
    from lilbee.cli.tui.widgets import fleet_body as fbm

    def _boom():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(fbm, "get_placement", _boom)
    notes: list[str] = []

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        body = app.screen.query_one("FleetBody")
        body.notify = lambda msg, **k: notes.append(msg)  # type: ignore[method-assign]
        body._load_worker()
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("probe failed" in n for n in notes)


@pytest.mark.asyncio
async def test_remove_device_when_multiple_devices(monkeypatch):
    """Toggling off a GPU when a role owns two GPUs removes it and updates the spec."""
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.providers.roles import WorkerRole

    def _two_device_view():
        return PlacementView(
            gpus=tuple(GpuInfo(i, "CUDA", f"CUDA{i}", "A40", 44 * GIB, 44 * GIB) for i in range(4)),
            roles=(
                RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0, 3), None, 1),
                RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (1,), None, 1),
            ),
            unplaceable=(),
            manual=False,
            spec_json=None,
        )

    monkeypatch.setattr(fbm, "get_placement", _two_device_view)

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        assert '"embed": {"devices": [0, 3]}' in _generated(app)
        await _toggle_device(pilot, "#dev-embed-3", expect_on=False)
        assert '"embed": {"devices": [0]}' in _generated(app)


@pytest.mark.asyncio
async def test_unrecognized_button_id_is_noop(monkeypatch):
    """A pill press with an unrecognized ID is silently ignored."""
    from unittest.mock import MagicMock

    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        before = _generated(app)
        pill = MagicMock(spec=FleetPill)
        pill.id = "some-other-pill"
        event = FleetPill.Pressed(pill)
        body = app.screen.query_one("FleetBody")
        body.on_fleet_pill_pressed(event)
        await pilot.pause()
        assert _generated(app) == before


@pytest.mark.asyncio
async def test_spec_from_editor_returns_none_with_no_edits(monkeypatch):
    """_spec_from_editor returns None when the edits dict is empty."""
    from lilbee.app.placement import GpuInfo, PlacementView
    from lilbee.cli.tui.widgets import fleet_body as fbm

    def _empty_view():
        return PlacementView(
            gpus=(GpuInfo(0, "CUDA", "CUDA0", "A40", 44 * GIB, 44 * GIB),),
            roles=(),
            unplaceable=(),
            manual=False,
            spec_json=None,
        )

    monkeypatch.setattr(fbm, "get_placement", _empty_view)

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        assert body._spec_from_editor() is None


@pytest.mark.asyncio
async def test_preview_raises_placement_error_from_spec(monkeypatch):
    """ctrl+r shows a notification when _spec_from_editor raises PlacementError."""
    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    notes: list[str] = []

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        monkeypatch.setattr(body, "notify", lambda msg, **k: notes.append(msg))
        next(iter(body._edits.values())).devices.clear()
        await pilot.press("ctrl+r")
        await pilot.pause()

    assert any("needs at least one GPU" in n or "GPU" in n for n in notes)


@pytest.mark.asyncio
async def test_apply_raises_placement_error_from_spec(monkeypatch):
    """ctrl+s shows a notification when _spec_from_editor raises PlacementError."""
    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    notes: list[str] = []

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        monkeypatch.setattr(body, "notify", lambda msg, **k: notes.append(msg))
        next(iter(body._edits.values())).devices.clear()
        await pilot.press("ctrl+s")
        await pilot.pause()

    assert any("needs at least one GPU" in n or "GPU" in n for n in notes)


@pytest.mark.asyncio
async def test_clear_ignored_while_applying(monkeypatch):
    """ctrl+x is a no-op when apply is in progress (single-flight guard)."""
    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    calls: list[object] = []
    monkeypatch.setattr(fbm, "set_placement", lambda spec: calls.append(spec))

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        body.applying = True
        await pilot.press("ctrl+x")
        await pilot.pause()
        body.applying = False

    assert calls == []


@pytest.mark.asyncio
async def test_clear_worker_error_notifies(monkeypatch):
    """An exception from set_placement(None) in _clear_worker shows a notification."""
    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    def _boom(spec):  # type: ignore[no-untyped-def]
        raise RuntimeError("disk full")

    monkeypatch.setattr(fbm, "set_placement", _boom)
    notes: list[str] = []

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        monkeypatch.setattr(body, "notify", lambda msg, **k: notes.append(msg))
        await pilot.press("ctrl+x")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("disk full" in n for n in notes)


@pytest.mark.asyncio
async def test_apply_shows_rebuilding_status(monkeypatch):
    """While an apply reloads the fleet the state segment shows 'Rebuilding fleet…'
    instead of a silent, still-'edited' idle screen, and clears once it's ready."""
    from textual.widgets import Static

    from lilbee.cli.tui import messages as tui_msg
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.fleet_body import _STATE_ID

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    gate = threading.Event()
    monkeypatch.setattr(fbm, "set_placement", lambda spec: gate.wait())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await _toggle_device(pilot, "#dev-embed-3")  # a dirty draft: state is 'edited'
        state = app.screen.query_one(_STATE_ID, Static)
        assert state.has_class("-edited")
        await pilot.press("ctrl+s")
        await pilot.pause()
        # mid-rebuild: the stale 'edited' label is gone, a rebuilding status shows
        assert state.has_class("-rebuilding")
        assert not state.has_class("-edited")
        assert tui_msg.FLEET_STATE_REBUILDING in str(state.render())
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not state.has_class("-rebuilding")


@pytest.mark.asyncio
async def test_reset_shows_rebuilding_then_clears_draft(monkeypatch):
    """Reset-to-auto drops the stale 'edited' draft immediately (rebuilding status),
    then returns to the read-only auto view with a fresh draft."""
    from textual.widgets import Static

    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.fleet_body import _STATE_ID, FleetBody
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())
    gate = threading.Event()
    monkeypatch.setattr(fbm, "set_placement", lambda spec: gate.wait())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await _toggle_device(pilot, "#dev-embed-2")  # dirty draft: embed on {0, 2}
        state = app.screen.query_one(_STATE_ID, Static)
        assert state.has_class("-edited")
        await pilot.press("ctrl+x")  # reset to auto
        await pilot.pause()
        assert state.has_class("-rebuilding")
        assert not state.has_class("-edited")
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not state.has_class("-rebuilding")
        assert not state.has_class("-edited")
        # a fresh draft off the live auto view, not the stale {0, 2} edit
        assert app.screen.query_one(FleetBody)._edits[WorkerRole.EMBED].devices == {0}


@pytest.mark.asyncio
async def test_failed_change_restores_live_view(monkeypatch):
    """A failed apply doesn't strand the rejected draft: the view re-renders from the
    live placement and drops the 'edited'/'rebuilding' state."""
    from textual.widgets import Static

    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.fleet_body import _STATE_ID, FleetBody
    from lilbee.providers.fleet.placement_spec import PlacementError
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    def _oom(spec):  # type: ignore[no-untyped-def]
        raise PlacementError("oom")

    monkeypatch.setattr(fbm, "set_placement", _oom)
    notes: list[str] = []

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one(FleetBody)
        monkeypatch.setattr(body, "notify", lambda msg, **k: notes.append(msg))
        await _toggle_device(pilot, "#dev-embed-2")  # dirty draft
        await pilot.press("ctrl+s")
        await app.workers.wait_for_complete()
        await pilot.pause()
        state = app.screen.query_one(_STATE_ID, Static)
        assert not state.has_class("-edited")
        assert not state.has_class("-rebuilding")
        # the rejected draft is discarded; the editor reflects the live placement
        assert body._edits[WorkerRole.EMBED].devices == {0}
    assert any("oom" in n for n in notes)


@pytest.mark.asyncio
async def test_reload_read_error_after_change_notifies(monkeypatch):
    """If re-reading the live placement after an applied change fails, the error
    surfaces and the editor doesn't hang in the applying state."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.fleet_body import FleetBody

    monkeypatch.setattr(fbm, "set_placement", lambda spec: _make_view(manual=True))
    reads = {"n": 0}

    def _get():
        reads["n"] += 1
        if reads["n"] == 1:
            return _make_view()  # initial load populates the editor
        raise RuntimeError("probe failed on reload")

    monkeypatch.setattr(fbm, "get_placement", _get)
    notes: list[str] = []

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()  # finish the initial load
        body = app.screen.query_one(FleetBody)
        monkeypatch.setattr(body, "notify", lambda msg, **k: notes.append(msg))
        await pilot.press("ctrl+s")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert body.applying is False
    assert any("probe failed on reload" in n for n in notes)


@pytest.mark.asyncio
async def test_spec_from_editor_errors_on_empty_devices(monkeypatch):
    """_spec_from_editor raises PlacementError when a role is left with no GPUs."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.providers.fleet.placement_spec import PlacementError

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        next(iter(body._edits.values())).devices.clear()
        with pytest.raises(PlacementError, match="at least one GPU"):
            body._spec_from_editor()


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (
            "Qwen/Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf",
            "Qwen3-235B-A22B",
        ),
        ("Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf", "Qwen3-Embedding-8B"),
        ("Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B"),
        ("bartowski/Llama-3.3-70B.gguf", "Llama-3.3-70B"),
        ("plain-name-Q6_K.gguf", "plain-name"),
        ("solo-model-Q4_K_M-00001-of-00003.gguf", "solo-model"),
    ],
)
def test_clean_model_name(ref: str, expected: str) -> None:
    from lilbee.cli.tui.widgets.fleet_body import _clean_model_name

    assert _clean_model_name(ref) == expected


def _make_big_view(n: int):  # type: ignore[no-untyped-def]
    """A fleet larger than one page: chat split across all n, embed copied to all."""
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.providers.roles import WorkerRole

    return PlacementView(
        gpus=tuple(
            GpuInfo(i, "CUDA", f"CUDA{i}", "NVIDIA A100", 80 * GIB, 80 * GIB) for i in range(n)
        ),
        roles=(
            RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", tuple(range(n)), None, 1),
            RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", tuple(range(n)), None, n),
        ),
        unplaceable=(),
        manual=True,
        spec_json=None,
    )


async def _wait_for(pilot, selector: str) -> None:  # type: ignore[no-untyped-def]
    """Pause until ``selector`` exists on the screen."""
    from textual.css.query import NoMatches

    for _ in range(100):
        await pilot.pause()
        try:
            pilot.app.screen.query_one(selector, FleetPill)
            return
        except NoMatches:
            continue
    raise AssertionError(f"{selector} never appeared")  # pragma: no cover


@pytest.mark.asyncio
async def test_placement_grid_paginates_large_fleet(monkeypatch):
    """A fleet past one page shows a pager; pg-next/pg-prev move the visible page."""
    from textual.css.query import NoMatches

    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_big_view(10))

    app = FleetTestApp()
    async with app.run_test(size=(160, 44)) as pilot:
        await pilot.pause()
        # page 1 shows GPUs 0-7; 8-9 are on page 2
        app.screen.query_one("#dev-chat-0", FleetPill)
        with pytest.raises(NoMatches):
            app.screen.query_one("#dev-chat-9", FleetPill)
        # advance to page 2
        app.screen.query_one("#pg-next", FleetPill).press()
        await _wait_for(pilot, "#dev-chat-9")
        app.screen.query_one("#dev-chat-8", FleetPill)
        with pytest.raises(NoMatches):
            app.screen.query_one("#dev-chat-0", FleetPill)
        # back to page 1
        app.screen.query_one("#pg-prev", FleetPill).press()
        await _wait_for(pilot, "#dev-chat-0")


def _make_view_rerank():  # type: ignore[no-untyped-def]
    """A view with rerank enabled (a single pinned instance on one card)."""
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.providers.roles import WorkerRole

    return PlacementView(
        gpus=tuple(
            GpuInfo(i, "CUDA", f"CUDA{i}", "NVIDIA A40", 44 * GIB, 44 * GIB) for i in range(4)
        ),
        roles=(
            RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (0, 1), None, 1),
            RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0,), None, 1),
            RolePlacementView(WorkerRole.RERANK, "org/rerank.gguf", (0,), None, 1),
        ),
        unplaceable=(),
        manual=True,
        spec_json=None,
    )


@pytest.mark.asyncio
async def test_rerank_is_single_select(monkeypatch):
    """rerank is a single pinned instance: picking another card moves it, never adds."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.fleet_body import FleetBody
    from lilbee.providers.roles import WorkerRole

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view_rerank())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one(FleetBody)
        assert body._edits[WorkerRole.RERANK].devices == {0}
        # single-instance roles carry the 'single' kind and a 'one card' tag
        assert app.screen.query_one("#dev-rerank-0", FleetPill).has_class("single")
        assert any("one card" in str(lbl.render()) for lbl in app.screen.query(".role-tag"))
        # picking GPU 2 MOVES rerank there; it must not become {0, 2}
        app.screen.query_one("#dev-rerank-2", FleetPill).press()
        for _ in range(100):
            await pilot.pause()
            if body._edits[WorkerRole.RERANK].devices == {2}:
                break
        assert body._edits[WorkerRole.RERANK].devices == {2}
        assert app.screen.query_one("#dev-rerank-2", FleetPill).has_class("on")
        assert not app.screen.query_one("#dev-rerank-0", FleetPill).has_class("on")
