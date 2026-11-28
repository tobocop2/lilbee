"""Tests for the FleetDrawer: ctrl+g docks placement beside a live screen."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Input

from tests._lilbee_app_test_host import LilbeeAppHost, await_chat

GIB = 1024**3


def _make_view(*, manual: bool = False, gpus: int = 2):  # type: ignore[no-untyped-def]
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.providers.roles import WorkerRole

    return PlacementView(
        gpus=tuple(
            GpuInfo(i, "CUDA", f"CUDA{i}", "NVIDIA A40", 44 * GIB, 44 * GIB) for i in range(gpus)
        ),
        roles=(
            RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0,), None, 1, "llama.cpp"),
            RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (gpus - 1,), None, 1, "llama.cpp"),
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
async def test_drawer_renders_probe_failure_instead_of_probing_forever(monkeypatch) -> None:
    """A placement load failure lands in the GPU panel as a named error (bb-0yf0);
    the old behavior left 'probing GPUs…' up for the life of the session."""
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer
    from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel
    from lilbee.providers.base import ProviderError

    def _wedged():  # type: ignore[no-untyped-def]
        raise ProviderError("The GPU device probe did not respond within 60s")

    monkeypatch.setattr(fbm, "get_placement", _wedged)
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})

    app = _DrawerApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await app.workers.wait_for_complete()
        await pilot.pause()
        panel = app.screen.query_one(FleetDrawer).query_one(GpuFleetPanel)
        rendered = str(panel.render())
        assert "GPU device probe" in rendered
        assert "probing" not in rendered


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


class _RecordingFleetApp(LilbeeAppHost):
    """Host that records PlacementReloading messages bubbled from its FleetBody."""

    CSS = ""

    def __init__(self) -> None:
        super().__init__()
        self.reload_events: list[bool] = []

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.fleet_body import FleetBody

        yield FleetBody()

    def on_fleet_body_placement_reloading(self, event) -> None:
        self.reload_events.append(event.active)


@pytest.mark.asyncio
async def test_fleet_body_applying_emits_reloading_message(_patched):
    """Toggling `applying` posts PlacementReloading so the chat screen can hold
    submissions during the reload."""
    from lilbee.cli.tui.widgets.fleet_body import FleetBody

    app = _RecordingFleetApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        body = app.screen.query_one(FleetBody)
        app.reload_events.clear()
        body.applying = True
        await pilot.pause()
        body.applying = False
        await pilot.pause()
        assert app.reload_events == [True, False]


def _gate_apply(monkeypatch, returned):  # type: ignore[no-untyped-def]
    """Hold each apply until the returned event is set; record every spec applied."""
    import threading

    from lilbee.cli.tui.widgets import fleet_body as fbm

    gate = threading.Event()
    calls: list[object] = []

    def _set(spec):  # type: ignore[no-untyped-def]
        calls.append(spec)
        gate.wait()
        return returned

    monkeypatch.setattr(fbm, "set_placement", _set)
    monkeypatch.setattr(fbm, "wait_chat_ready", lambda: True)
    return gate, calls


async def _open_drawer_and_focus(app, pilot, selector: str):  # type: ignore[no-untyped-def]
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    await pilot.press("ctrl+g")
    await pilot.pause()
    await pilot.pause()
    app.screen.query_one(FleetDrawer).query_one(selector).focus()
    await pilot.pause()


async def _finish_apply(app, pilot, gate) -> None:  # type: ignore[no-untyped-def]
    gate.set()
    await app.workers.wait_for_complete()
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_focus_returns_to_the_editor_control_after_apply(_patched, monkeypatch):
    """Apply disables the editor, which drops the focus; the end of the apply gives
    it back to the control that had it, so the drawer keys work at once."""
    gate, calls = _gate_apply(monkeypatch, _make_view(manual=True))
    app = _DrawerApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _open_drawer_and_focus(app, pilot, "#dev-embed-0")
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert app.focused is None
        await _finish_apply(app, pilot, gate)
        assert app.focused is not None
        assert app.focused.id == "dev-embed-0"
        await pilot.press("ctrl+s")
        await _finish_apply(app, pilot, gate)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_focus_goes_to_apply_when_the_control_is_gone(_patched, monkeypatch):
    """A control the new placement no longer shows hands the focus to the Apply pill."""
    gate, _ = _gate_apply(monkeypatch, _make_view(manual=True, gpus=1))
    app = _DrawerApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _open_drawer_and_focus(app, pilot, "#dev-embed-1")
        await pilot.press("ctrl+s")
        await pilot.pause()
        await _finish_apply(app, pilot, gate)
        assert not app.screen.query("#dev-embed-1")
        assert app.focused is not None
        assert app.focused.id == "cmd-apply"


@pytest.mark.asyncio
async def test_focus_moved_during_apply_stays_where_it_is(_patched, monkeypatch):
    """Focus the user gives the chat prompt during the apply is not taken back."""
    gate, _ = _gate_apply(monkeypatch, _make_view(manual=True))
    app = _DrawerApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _open_drawer_and_focus(app, pilot, "#dev-embed-0")
        await pilot.press("ctrl+s")
        await pilot.pause()
        app.query_one("#probe-input", Input).focus()
        await pilot.pause()
        await _finish_apply(app, pilot, gate)
        assert app.focused is app.query_one("#probe-input", Input)


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
        await app.workers.wait_for_complete()
        await pilot.pause()
        # The drawer's keys are focus-scoped by design (typing in chat must not
        # trigger them). Focus a toggle inside, then drive the priority
        # bindings themselves so a child swallowing ctrl+r/s/x fails this test.
        from lilbee.cli.tui.widgets.fleet_body import FleetPill

        drawer.query(FleetPill).first().focus()
        await pilot.pause()
        await pilot.press("ctrl+r")
        await pilot.press("ctrl+s")
        await pilot.press("ctrl+x")
        await pilot.pause()
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
            RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0,), None, 1, "llama.cpp"),
            RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (1, 2), None, 1, "llama.cpp"),
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
            mock.patch("lilbee.cli.tui.app.chat_ready", return_value=True),
            mock.patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
            mock.patch(f"{cs}._embedding_ready", return_value=True),
        ):
            app = LilbeeApp()
            async with app.run_test(size=(140, 40)) as pilot:
                await await_chat(app, pilot)
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
@pytest.mark.parametrize("insert_mode", [True, False], ids=["insert", "normal"])
async def test_chat_screen_leaves_the_focus_to_the_drawer_after_apply(
    monkeypatch, insert_mode: bool
) -> None:
    """On the real chat screen, in either mode, the drawer control gets the focus back
    after Apply: the chat input does not take it when the reload hold ends."""
    from unittest import mock

    from lilbee.app.services import set_services
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.widgets import fleet_body as fbm
    from lilbee.cli.tui.widgets import gpu_fleet_panel as gfp

    monkeypatch.setattr(fbm, "get_placement", lambda: _make_view3())
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})
    gate, calls = _gate_apply(monkeypatch, _make_view3())
    svc = mock.MagicMock()
    svc.provider.list_models.return_value = []
    svc.searcher._embedder.embedding_available.return_value = True
    set_services(svc)
    try:
        cs = "lilbee.cli.tui.screens.chat.ChatScreen"
        with (
            mock.patch("lilbee.cli.tui.app.chat_ready", return_value=True),
            mock.patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
            mock.patch(f"{cs}._embedding_ready", return_value=True),
        ):
            app = LilbeeApp()
            async with app.run_test(size=(140, 40)) as pilot:
                await await_chat(app, pilot)
                await pilot.pause()
                if not insert_mode:
                    await pilot.press("escape")
                    await pilot.pause()
                await _open_drawer_and_focus(app, pilot, "#dev-embed-0")
                assert app.screen._insert_mode is insert_mode
                await pilot.press("ctrl+s")
                await pilot.pause()
                await _finish_apply(app, pilot, gate)
                assert app.focused is not None
                assert app.focused.id == "dev-embed-0"
                assert app.screen._insert_mode is insert_mode
    finally:
        set_services(None)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_reload_the_drawer_does_not_refocus_gives_the_input_the_focus() -> None:
    """With the focus outside the drawer, the end of the reload hold still gives
    the chat input the focus in insert mode, as before."""
    from unittest import mock

    from lilbee.app.services import set_services
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.widgets.fleet_body import FleetBody

    svc = mock.MagicMock()
    svc.provider.list_models.return_value = []
    svc.searcher._embedder.embedding_available.return_value = True
    set_services(svc)
    try:
        cs = "lilbee.cli.tui.screens.chat.ChatScreen"
        with (
            mock.patch("lilbee.cli.tui.app.chat_ready", return_value=True),
            mock.patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
            mock.patch(f"{cs}._embedding_ready", return_value=True),
        ):
            app = LilbeeApp()
            async with app.run_test(size=(140, 40)) as pilot:
                await await_chat(app, pilot)
                screen = app.screen
                screen.on_fleet_body_placement_reloading(FleetBody.PlacementReloading(True))
                await pilot.pause()
                screen.set_focus(None)
                screen.on_fleet_body_placement_reloading(FleetBody.PlacementReloading(False))
                await pilot.pause()
                assert screen._insert_mode is True
                assert app.focused is screen._chat_input
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


async def test_single_gpu_keeps_the_grid_for_the_cpu_choice(monkeypatch) -> None:
    """One card leaves only the cpu choice to arrange: the note, the grid and the commands show."""
    from textual.widgets import Label

    import lilbee.cli.tui.widgets.fleet_body as fbm
    import lilbee.cli.tui.widgets.gpu_fleet_panel as gfp
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.providers.roles import WorkerRole

    GIB = 1024**3
    view = PlacementView(
        gpus=(GpuInfo(0, "CUDA", "CUDA0", "NVIDIA A40", 44 * GIB, 44 * GIB),),
        roles=(
            RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0,), None, 1, "llama.cpp"),
            RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (0,), None, 1, "llama.cpp"),
        ),
        manual=False,
        unplaceable=(),
        spec_json=None,
    )
    monkeypatch.setattr(fbm, "get_placement", lambda: view)
    monkeypatch.setattr(gfp, "probe_gpu_stats", lambda devices: {})
    app = _DrawerApp()
    async with app.run_test() as pilot:
        await pilot.press("ctrl+g")
        await pilot.pause()
        body = app.query_one(fbm.FleetBody)
        # The card pill and the cpu and auto choices of both roles stay on offer.
        assert sorted(str(pill.render()) for pill in body.query(".dev-toggle")) == [
            " 0 ",
            " 0 ",
            " auto ",
            " auto ",
            " cpu ",
            " cpu ",
        ]
        assert body.query_one("#placement-commands").display is True
        note = body.query_one(".single-gpu-note", Label)
        assert "One graphics card" in str(note.render())
