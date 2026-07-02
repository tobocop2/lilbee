"""Tests for the Fleet view: FleetScreen hosting FleetBody.

These drive the real widgets (GPU toggle Buttons, replica steppers, key
bindings) rather than poking private state, so the input path is actually
exercised.
"""

from __future__ import annotations

import threading

import pytest
from textual.widgets import Button, Static

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
    from textual.widgets import Static as _Static

    from lilbee.cli.tui.widgets.fleet_body import _GENERATED_ID

    return str(app.screen.query_one(_GENERATED_ID, _Static).render())


async def _toggle_device(pilot, selector: str, *, expect_on: bool = True) -> None:  # type: ignore[no-untyped-def]
    """Activate a GPU toggle and wait until the press has been applied.

    ``pilot.click`` resolves the target's screen coordinates up front, so under
    parallel load it can fire before layout settles and miss the button
    entirely. ``Button.press`` posts ``Button.Pressed`` directly (no
    coordinates), and the handler flips the ``on`` class in the same step that
    mutates the device set -- so wait for that class as the post-condition.
    """
    button = pilot.app.screen.query_one(selector, Button)
    button.press()
    for _ in range(100):
        await pilot.pause()
        if button.has_class("on") == expect_on:
            return
    raise AssertionError(f"{selector} did not reach on={expect_on}")  # pragma: no cover


async def _step_until_generated(pilot, selector: str, app, predicate) -> None:  # type: ignore[no-untyped-def]
    """Activate a control and wait until the generated spec satisfies ``predicate``.

    The replica stepper has no ``on`` class to watch, so synchronise on the
    equivalent-spec text it drives. Uses ``Button.press`` for the same
    coordinate-free reason as ``_toggle_device``.
    """
    pilot.app.screen.query_one(selector, Button).press()
    for _ in range(100):
        await pilot.pause()
        if predicate(_generated(app)):
            return
    raise AssertionError(
        f"{selector}: generated never matched: {_generated(app)!r}"
    )  # pragma: no cover


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
        app.screen.query_one("#rep-embed-dec", Button).press()  # floored at 1 (stays omitted)
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
            app.screen.query_one("#rep-chat-inc", Button)


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
    """A button press with an unrecognized ID is silently ignored."""
    from unittest.mock import MagicMock

    from textual.widgets import Button as TxtButton

    from lilbee.cli.tui.widgets import fleet_body as fbm

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        before = _generated(app)
        btn = MagicMock(spec=TxtButton)
        btn.id = "some-other-button"
        event = TxtButton.Pressed(btn)
        body = app.screen.query_one("FleetBody")
        body.on_button_pressed(event)
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
async def test_refresh_generated_shows_error_on_empty_devices(monkeypatch):
    """_refresh_generated shows a red error when _spec_from_editor raises PlacementError."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets.fleet_body import _GENERATED_ID

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view())

    app = FleetTestApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        body = app.screen.query_one("FleetBody")
        next(iter(body._edits.values())).devices.clear()
        body._refresh_generated()
        await pilot.pause()
        gen_text = str(body.query_one(_GENERATED_ID, Static).render())
        assert "needs at least one GPU" in gen_text or "GPU" in gen_text


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
