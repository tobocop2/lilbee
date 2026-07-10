"""The startup gate holds the screen until the chat engine can serve a prompt."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.app.placement import chat_engine_ready
from lilbee.app.services import set_services
from lilbee.catalog.formatting import display_label_for_ref
from lilbee.cli.tui import messages as msg
from lilbee.providers.roles import WorkerRole
from lilbee.providers.warm_progress import WarmPhase, WarmProgress


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


def test_gate_colours_match_the_shared_amber_palette():
    """Every hex in the stylesheet must be an xterm index from runtime/bee_logo."""
    from lilbee.runtime.bee_logo import AMBER_BRIGHT_XTERM, AMBER_DIM_XTERM, AMBER_MID_XTERM

    stylesheet = _gate_stylesheet()
    for index in (AMBER_BRIGHT_XTERM, AMBER_MID_XTERM, AMBER_DIM_XTERM):
        assert _xterm_to_hex(index) in stylesheet, f"xterm {index} missing from the gate CSS"


def test_gate_stylesheet_has_no_colour_outside_the_palette():
    """A hand-picked hex would drift from the bootstrap and the splash."""
    import re

    from lilbee.runtime.bee_logo import AMBER_BRIGHT_XTERM, AMBER_DIM_XTERM, AMBER_MID_XTERM

    allowed = {_xterm_to_hex(i) for i in (AMBER_BRIGHT_XTERM, AMBER_MID_XTERM, AMBER_DIM_XTERM)}
    found = set(re.findall(r"#[0-9a-fA-F]{6}", _gate_stylesheet()))
    assert found <= allowed, f"colours outside the palette: {sorted(found - allowed)}"


def _snapshot(phase: WarmPhase, **kwargs) -> WarmProgress:
    return WarmProgress(phase=phase, **kwargs)


async def test_gate_shows_a_byte_bar_while_reading_weights():
    """READING_WEIGHTS carries byte counts, so the bar is determinate."""
    from textual.widgets import ProgressBar

    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    with mock.patch.object(StartupGate, "query_one") as query:
        bar, status = mock.MagicMock(spec=ProgressBar), mock.MagicMock()
        query.side_effect = lambda selector, _kind=None: bar if "bar" in selector else status
        gate._apply_snapshot(
            _snapshot(
                WarmPhase.READING_WEIGHTS, bytes_done=512, bytes_total=1024, model_ref="a/b/c.gguf"
            )
        )
    bar.update.assert_called_once_with(total=1024, progress=512)


async def test_gate_bar_is_indeterminate_while_loading_the_engine():
    """The VRAM upload emits no byte signal, so the bar must not fake one."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    with mock.patch.object(StartupGate, "query_one") as query:
        bar, status = mock.MagicMock(), mock.MagicMock()
        query.side_effect = lambda selector, _kind=None: bar if "bar" in selector else status
        gate._apply_snapshot(_snapshot(WarmPhase.LOADING_ENGINE))
    bar.update.assert_called_once_with(total=None)
    status.update.assert_called_once_with(msg.STARTUP_LOADING_ENGINE)


async def test_gate_release_reveals_chat():
    """Reaching ready hands the screen to chat."""
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


async def test_gate_releases_when_nothing_is_warming(monkeypatch):
    """No fleet, no warm, or an uninstalled model must not hold the screen forever."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    monkeypatch.setattr(gate_mod, "_WARM_START_GRACE_S", 0.0)

    provider = mock.MagicMock()
    provider.role_ready.return_value = False
    provider.warm_progress.return_value = None
    provider.warm_pending.return_value = False
    services = mock.MagicMock()
    services.provider = provider
    set_services(services)
    try:
        gate = StartupGate()
        app = mock.MagicMock()
        app.screen = gate
        with (
            mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
            mock.patch.object(gate_mod, "needs_setup", return_value=False),
            mock.patch.object(StartupGate, "_stopping", return_value=False),
        ):
            app.call_from_thread.side_effect = lambda fn, *a: fn(*a)
            with mock.patch.object(StartupGate, "query_one"):
                StartupGate._boot_worker.__wrapped__(gate)
        app.reveal_chat.assert_called_once_with()
    finally:
        set_services(None)


async def test_gate_failure_surfaces_the_error_and_still_reveals_chat():
    """A failed load must not lock the user out of Catalog and Settings."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    app.screen = gate
    status = mock.MagicMock()
    with (
        mock.patch.object(StartupGate, "query_one", return_value=status),
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
    ):
        gate._fail("no such file")
    notified = [c.args[0] for c in app.notify.call_args_list if c.args]
    assert msg.STARTUP_FAILED.format(error="no such file") in notified
    assert msg.STARTUP_FAILED_HINT in notified
    app.reveal_chat.assert_called_once_with()


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


async def test_gate_reveals_chat_when_setup_is_required(monkeypatch):
    """With no models installed the wizard owns the flow, so the gate steps aside."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: True)
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    app.reveal_chat.assert_called_once_with()


async def test_gate_surfaces_a_failure_to_build_services(monkeypatch):
    """A container that cannot be built must show the error, not hang the screen."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: False)
    monkeypatch.setattr(gate_mod, "get_services", mock.Mock(side_effect=OSError("disk gone")))
    status = mock.MagicMock()
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "query_one", return_value=status),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    notified = [c.args[0] for c in app.notify.call_args_list if c.args]
    assert msg.STARTUP_FAILED.format(error="disk gone") in notified
    app.reveal_chat.assert_called_once_with()


async def test_gate_does_not_wait_when_eager_start_is_disabled(monkeypatch):
    """With eager start off nothing warms, so holding the screen would never end."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate
    from lilbee.core.config import cfg

    gate, app = _gate_with_app()
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: False)
    monkeypatch.setattr(cfg, "worker_pool_eager_start", False)
    provider = mock.MagicMock()
    provider.role_ready.return_value = False
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


async def test_gate_renders_progress_then_fails_on_an_error_phase(monkeypatch):
    """An in-flight warm renders, and a later ERROR phase surfaces the reason."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate
    from lilbee.core.config import cfg

    gate, app = _gate_with_app()
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: False)
    monkeypatch.setattr(cfg, "worker_pool_eager_start", True)
    monkeypatch.setattr(gate_mod, "_POLL_INTERVAL_S", 0.0)

    provider = mock.MagicMock()
    provider.role_ready.return_value = False
    provider.warm_progress.side_effect = [
        _snapshot(WarmPhase.READING_WEIGHTS, bytes_done=1, bytes_total=2, model_ref="a/b/c.gguf"),
        _snapshot(WarmPhase.ERROR, error="out of memory"),
    ]
    services = mock.MagicMock()
    services.provider = provider
    monkeypatch.setattr(gate_mod, "get_services", lambda: services)
    set_services(services)
    try:
        status = mock.MagicMock()
        with (
            mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
            mock.patch.object(StartupGate, "query_one", return_value=status),
            mock.patch.object(StartupGate, "_stopping", return_value=False),
        ):
            StartupGate._boot_worker.__wrapped__(gate)
        status.update.assert_any_call(
            msg.STARTUP_READING_WEIGHTS.format(name=display_label_for_ref("a/b/c.gguf"))
        )
        notified = [c.args[0] for c in app.notify.call_args_list if c.args]
        assert msg.STARTUP_FAILED.format(error="out of memory") in notified
    finally:
        set_services(None)


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


async def test_first_run_hands_over_when_setup_is_required(tmp_path, monkeypatch):
    """Regression: an unmounted gate must not be mistaken for a torn-down one.

    ``push_screen`` returns an AwaitMount. Left unawaited, the boot worker could
    reach ``_stopping`` before the gate mounted, silently drop the handover, and
    strand a first-run user on the loading screen forever.
    """
    import asyncio
    import time

    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "lancedb_dir", tmp_path / "missing")  # forces needs_setup
    revealed = mock.MagicMock()
    monkeypatch.setattr(LilbeeApp, "reveal_chat", revealed)

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not revealed.called:
            await pilot.pause()
            await asyncio.sleep(0.02)

    revealed.assert_called_once_with()


async def test_gate_holds_while_a_warm_is_pending_but_unreported(monkeypatch):
    """Regression: the fleet spawns for seconds before the tracker stamps a phase.

    Releasing on "warm_progress() is None" dropped the user into chat while
    llama-server had not been launched, which is the bug this screen exists for.
    """
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate
    from lilbee.core.config import cfg

    gate, app = _gate_with_app()
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: False)
    monkeypatch.setattr(cfg, "worker_pool_eager_start", True)
    monkeypatch.setattr(gate_mod, "_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(gate_mod, "_WARM_START_GRACE_S", 0.0)

    provider = mock.MagicMock()
    provider.warm_progress.return_value = None
    # Pending for the first polls, then the role comes up ready.
    provider.warm_pending.side_effect = [True] * 5 + [False]
    provider.role_ready.side_effect = [False] * 5 + [True]

    services = mock.MagicMock()
    services.provider = provider
    monkeypatch.setattr(gate_mod, "get_services", lambda: services)
    set_services(services)
    try:
        with (
            mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
            mock.patch.object(StartupGate, "query_one"),
            mock.patch.object(StartupGate, "_stopping", return_value=False),
        ):
            StartupGate._boot_worker.__wrapped__(gate)
    finally:
        set_services(None)

    # A zero grace would have released on the first poll had warm_pending been ignored.
    assert provider.warm_pending.call_count >= 5
    app.reveal_chat.assert_called_once_with()


async def test_gate_releases_when_no_warm_is_pending_and_none_reported(monkeypatch):
    """A remote or uninstalled chat model never warms, so the gate must step aside."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate
    from lilbee.core.config import cfg

    gate, app = _gate_with_app()
    monkeypatch.setattr(gate_mod, "needs_setup", lambda: False)
    monkeypatch.setattr(cfg, "worker_pool_eager_start", True)
    monkeypatch.setattr(gate_mod, "_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(gate_mod, "_WARM_START_GRACE_S", 0.0)

    provider = mock.MagicMock()
    provider.warm_progress.return_value = None
    provider.warm_pending.return_value = False
    provider.role_ready.return_value = False

    services = mock.MagicMock()
    services.provider = provider
    monkeypatch.setattr(gate_mod, "get_services", lambda: services)
    set_services(services)
    try:
        with (
            mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
            mock.patch.object(StartupGate, "query_one"),
            mock.patch.object(StartupGate, "_stopping", return_value=False),
        ):
            StartupGate._boot_worker.__wrapped__(gate)
    finally:
        set_services(None)
    app.reveal_chat.assert_called_once_with()


async def test_gate_composes_and_styles_its_widgets(monkeypatch):
    """Render the gate for real: a bad CSS selector must fail here, not at launch.

    Every other gate test stubs query_one, so nothing else parses the stylesheet
    or mounts the ProgressBar's component classes.
    """
    from textual.app import App, ComposeResult
    from textual.widgets import ProgressBar, Static

    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate
    from lilbee.runtime.bee_logo import AMBER_BRIGHT_XTERM

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

        assert bar.show_percentage is False  # no placeholder before a byte signal
        assert msg.STARTUP_PREPARING in str(status.render())
        assert logo.styles.color.hex.lower() == _xterm_to_hex(AMBER_BRIGHT_XTERM)
        assert gate_mod._LOGO.splitlines()[1].strip().startswith("@@@")


async def test_gate_does_not_steal_a_screen_pushed_over_it(monkeypatch):
    """Regression: reveal_chat switches the *current* screen, whichever it is.

    A slow warm let another screen open above the gate; when the warm finished the
    gate replaced that screen instead of itself, throwing the user out of it.
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


async def test_ready_fleet_defers_the_handover_off_on_mount(monkeypatch):
    """Regression: switching screens inside on_mount stalled Textual.

    start_boot runs at the tail of LilbeeApp.on_mount. Calling reveal_chat straight
    from there switches screens mid-mount; defer it a frame instead.
    """
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    monkeypatch.setattr(gate_mod, "chat_engine_ready", lambda: True)

    deferred: list = []
    released: list = []
    monkeypatch.setattr(gate, "call_after_refresh", deferred.append, raising=False)
    monkeypatch.setattr(gate, "_release", lambda: released.append(True), raising=False)

    gate.start_boot()

    assert released == [], "the handover must not run inside on_mount"
    assert deferred == [gate._release]


async def test_gate_tip_points_at_the_warm_setting_only_when_off(monkeypatch):
    """The tip renders during a cold load and disappears once warm is enabled."""
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    from lilbee.cli.tui.screens.startup_gate import StartupGate
    from lilbee.core.config import cfg

    monkeypatch.setattr(StartupGate, "start_boot", lambda self: None)

    class _Host(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

        def on_mount(self) -> None:
            self.push_screen(StartupGate())

    monkeypatch.setattr(cfg, "keep_engine_warm", False)
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tip = app.screen.query_one("#gate-tip", Static)
        assert msg.STARTUP_WARM_TIP in str(tip.render())

    monkeypatch.setattr(cfg, "keep_engine_warm", True)
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tip = app.screen.query_one("#gate-tip", Static)
        assert msg.STARTUP_WARM_TIP not in str(tip.render())
