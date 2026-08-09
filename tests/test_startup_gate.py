"""The startup gate holds the screen only while the services container builds."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.app.placement import chat_engine_ready
from lilbee.app.services import set_services
from lilbee.cli.tui import messages as msg
from lilbee.providers.roles import WorkerRole


@pytest.fixture
def _no_services():
    """Leave the singleton empty, as it is before the container is built."""
    set_services(None)
    yield
    set_services(None)


def test_chat_engine_not_ready_before_services_exist(_no_services):
    """The pre-services window must read as not-ready, not as ready."""
    assert chat_engine_ready() is False


def test_chat_engine_ready_tracks_the_chat_role():
    """Readiness is the chat role's own readiness, asked positively."""
    services = mock.MagicMock()
    services.provider.role_ready.return_value = True
    set_services(services)
    try:
        assert chat_engine_ready() is True
        services.provider.role_ready.assert_called_once_with(WorkerRole.CHAT)
    finally:
        set_services(None)


def test_chat_engine_not_ready_while_the_role_is_cold():
    """A spawned-but-cold chat role still cannot serve a prompt."""
    services = mock.MagicMock()
    services.provider.role_ready.return_value = False
    set_services(services)
    try:
        assert chat_engine_ready() is False
    finally:
        set_services(None)


def _xterm_to_hex(index: int) -> str:
    """The 6x6x6 colour-cube hex for an xterm-256 index in 16..231."""
    levels = (0, 95, 135, 175, 215, 255)
    offset = index - 16
    red = levels[offset // 36]
    green = levels[(offset % 36) // 6]
    blue = levels[offset % 6]
    return f"#{red:02x}{green:02x}{blue:02x}"


def _gate_stylesheet() -> str:
    import pathlib

    return (
        pathlib.Path(__file__).resolve().parents[1] / "src/lilbee/cli/tui/screens/startup_gate.tcss"
    ).read_text()


def test_gate_colours_match_the_shared_rose_palette():
    """Every hex in the stylesheet must be an xterm index from runtime/bee_logo."""
    from lilbee.runtime.bee_logo import ROSE_BRIGHT_XTERM, ROSE_DIM_XTERM, ROSE_MID_XTERM

    stylesheet = _gate_stylesheet()
    for index in (ROSE_BRIGHT_XTERM, ROSE_MID_XTERM, ROSE_DIM_XTERM):
        assert _xterm_to_hex(index) in stylesheet, f"xterm {index} missing from the gate CSS"


def test_gate_stylesheet_has_no_colour_outside_the_palette():
    """A hand-picked hex would drift from the bootstrap and the splash."""
    import re

    from lilbee.runtime.bee_logo import ROSE_BRIGHT_XTERM, ROSE_DIM_XTERM, ROSE_MID_XTERM

    allowed = {_xterm_to_hex(i) for i in (ROSE_BRIGHT_XTERM, ROSE_MID_XTERM, ROSE_DIM_XTERM)}
    found = set(re.findall(r"#[0-9a-fA-F]{6}", _gate_stylesheet()))
    assert found <= allowed, f"colours outside the palette: {sorted(found - allowed)}"


def _gate_with_app():
    """A gate whose call_from_thread runs inline, so the worker body is testable.

    ``_stopping`` needs a live worker context and a mounted screen, neither of
    which a direct call to the worker body has, so it is stubbed to "keep going".
    """
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    app.call_from_thread.side_effect = lambda fn, *a: fn(*a)
    app.screen = gate  # the gate owns the screen, as it does in production
    return gate, app


async def test_gate_release_reveals_chat():
    """A built container hands the screen to chat."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    app.screen = gate
    with (
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
    ):
        gate._release()
    app.reveal_chat.assert_called_once_with()


async def test_gate_releases_once_services_build_even_with_a_cold_engine(monkeypatch):
    """The gate waits for the container, never for the model load.

    The engine loads in the background after the handover; a prompt that arrives
    first waits inside its own answer bubble. Holding the screen for the load
    would re-block every launch that the lazy path exists to unblock.
    """
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: False)

    provider = mock.MagicMock()
    provider.role_ready.return_value = False  # engine is stone cold
    services = mock.MagicMock()
    services.provider = provider
    monkeypatch.setattr(gate_mod, "get_services", lambda: services)
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    app.reveal_chat.assert_called_once_with()
    provider.role_ready.assert_not_called()


async def test_gate_failure_surfaces_the_error_and_still_reveals_chat():
    """A failed start must not lock the user out of Catalog and Settings."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    app.screen = gate
    with (
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
    ):
        gate._fail("no such file")
    notified = [c.args[0] for c in app.notify.call_args_list if c.args]
    assert msg.STARTUP_FAILED.format(error="no such file") in notified
    app.reveal_chat.assert_called_once_with()


async def test_gate_reveals_chat_when_setup_is_required(monkeypatch):
    """With no models installed the wizard owns the flow, so the gate steps aside."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: True)
    built = mock.Mock()
    monkeypatch.setattr(gate_mod, "get_services", built)
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    app.reveal_chat.assert_called_once_with()
    built.assert_not_called()


async def test_gate_surfaces_a_failure_to_build_services(monkeypatch):
    """A container that cannot be built must show the error, not hang the screen."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: False)
    monkeypatch.setattr(gate_mod, "get_services", mock.Mock(side_effect=OSError("disk gone")))
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    notified = [c.args[0] for c in app.notify.call_args_list if c.args]
    assert msg.STARTUP_FAILED.format(error="disk gone") in notified
    app.reveal_chat.assert_called_once_with()


async def test_marshal_skips_the_ui_hop_once_the_app_is_tearing_down():
    """A gate worker that outlives the app must not touch a shutting-down UI thread."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "_stopping", return_value=True),
    ):
        gate._marshal(gate._release)
    app.call_from_thread.assert_not_called()
    app.reveal_chat.assert_not_called()


async def test_stopping_is_true_when_the_gate_is_unmounted():
    """An unmounted gate must report that the worker should stop."""
    from textual.worker import Worker

    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    worker = mock.MagicMock(spec=Worker)
    worker.is_cancelled = False
    with mock.patch("lilbee.cli.tui.screens.startup_gate.get_current_worker", return_value=worker):
        assert gate._stopping() is True


async def test_stopping_is_true_when_the_worker_is_cancelled():
    """A cancelled worker must stop even while the gate is still mounted."""
    from textual.worker import Worker

    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    worker = mock.MagicMock(spec=Worker)
    worker.is_cancelled = True
    with (
        mock.patch("lilbee.cli.tui.screens.startup_gate.get_current_worker", return_value=worker),
        mock.patch.object(type(gate), "is_mounted", new=mock.PropertyMock(return_value=True)),
    ):
        assert gate._stopping() is True


@pytest.mark.first_run
async def test_first_run_hands_over_when_setup_is_required(tmp_path, monkeypatch):
    """Regression: an unmounted gate must not be mistaken for a torn-down one.

    ``push_screen`` returns an AwaitMount. Left unawaited, the boot worker could
    reach ``_stopping`` before the gate mounted, silently drop the handover, and
    strand a first-run user on the loading screen forever. A first run lands on
    setup rather than on chat, so the assertion is that the gate steps aside at
    all, not which screen takes its place.
    """
    import asyncio
    import time

    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.startup_gate import StartupGate
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "lancedb_dir", tmp_path / "missing")  # forces needs_setup

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and isinstance(app.screen, StartupGate):
            await pilot.pause()
            await asyncio.sleep(0.02)
        assert not isinstance(app.screen, StartupGate)


async def test_gate_composes_and_styles_its_widgets(monkeypatch):
    """Render the gate for real: a bad CSS selector must fail here, not at launch.

    Every other gate test stubs query_one, so nothing else parses the stylesheet
    or mounts the ProgressBar's component classes.
    """
    from textual.app import App, ComposeResult
    from textual.widgets import ProgressBar, Static

    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate
    from lilbee.runtime.bee_logo import ROSE_BRIGHT_XTERM

    monkeypatch.setattr(StartupGate, "start_boot", lambda self: None)

    class _Host(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

        def on_mount(self) -> None:
            self.push_screen(StartupGate())

    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        gate = app.screen
        assert isinstance(gate, StartupGate)

        bar = gate.query_one("#gate-bar", ProgressBar)
        status = gate.query_one("#gate-status", Static)
        logo = gate.query_one("#gate-logo", Static)

        assert bar.show_percentage is False  # the build has no byte signal to fake
        assert msg.STARTUP_PREPARING in str(status.render())
        assert logo.styles.color.hex.lower() == _xterm_to_hex(ROSE_BRIGHT_XTERM)
        assert gate_mod._LOGO.splitlines()[1].strip().startswith("@@@")


async def test_gate_does_not_steal_a_screen_pushed_over_it(monkeypatch):
    """Regression: reveal_chat switches the *current* screen, whichever it is.

    A slow build let another screen open above the gate; when the build finished
    the gate replaced that screen instead of itself, throwing the user out of it.
    """
    from textual.app import ComposeResult
    from textual.screen import Screen
    from textual.widgets import Static

    from lilbee.cli.tui.screens.startup_gate import StartupGate

    class _Other(Screen[None]):
        def compose(self) -> ComposeResult:
            yield Static("other")

    gate = StartupGate()
    app = mock.MagicMock()
    app.screen = _Other()  # something else owns the screen now
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "query_one"),
    ):
        gate._release()
    app.reveal_chat.assert_not_called()


async def test_gate_hands_over_when_it_still_owns_the_screen():
    """The ordinary path: the gate is on top, so it reveals chat."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    app.screen = gate
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "query_one"),
    ):
        gate._release()
    app.reveal_chat.assert_called_once_with()


async def test_built_services_defer_the_handover_off_on_mount(monkeypatch):
    """Regression: switching screens inside on_mount stalled Textual.

    start_boot runs at the tail of LilbeeApp.on_mount. Calling reveal_chat straight
    from there switches screens mid-mount; defer it a frame instead.
    """
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    monkeypatch.setattr(gate_mod, "peek_services", lambda: mock.MagicMock())

    deferred: list = []
    released: list = []
    monkeypatch.setattr(gate, "call_after_refresh", deferred.append, raising=False)
    monkeypatch.setattr(gate, "_release", lambda: released.append(True), raising=False)

    gate.start_boot()

    assert released == [], "the handover must not run inside on_mount"
    assert deferred == [gate._release]


async def test_start_boot_builds_services_when_none_exist(monkeypatch):
    """Without a container the gate takes the worker path, not the fast path."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    monkeypatch.setattr(gate_mod, "peek_services", lambda: None)

    worker_calls: list = []
    deferred: list = []
    monkeypatch.setattr(gate, "_boot_worker", lambda: worker_calls.append(True), raising=False)
    monkeypatch.setattr(gate, "call_after_refresh", deferred.append, raising=False)

    gate.start_boot()

    assert worker_calls == [True]
    assert deferred == []


async def test_boot_worker_canonicalizes_before_the_setup_check(monkeypatch):
    """Canonicalization can swap a stale ref to a working one, which is exactly
    what decides whether setup is needed, so it must run first."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    order: list[str] = []
    app.canonicalize_persisted_models.side_effect = lambda: order.append("canonicalize")
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: order.append("setup") or True)
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    assert order == ["canonicalize", "setup"]


async def test_boot_worker_surfaces_a_canonicalization_failure(monkeypatch):
    """A crash while settling the refs shows the error instead of hanging the gate."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    app.canonicalize_persisted_models.side_effect = OSError("registry unreadable")
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: False)
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    notified = [c.args[0] for c in app.notify.call_args_list if c.args]
    assert msg.STARTUP_FAILED.format(error="registry unreadable") in notified
    app.reveal_chat.assert_called_once_with()


async def test_gate_mount_retires_the_splash_then_repaints(monkeypatch):
    """The splash animates over the blank alt-screen until the gate paints;
    the dismissal then repaints anything a final frame touched."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    order: list[str] = []
    monkeypatch.setattr("lilbee.runtime.splash.dismiss", lambda: order.append("dismiss"))
    monkeypatch.setattr(gate, "_repaint", lambda: order.append("refresh"), raising=False)
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._retire_splash.__wrapped__(gate)
    assert order == ["dismiss", "refresh"]
