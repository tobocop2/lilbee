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
    The app lands in ``_ui_app`` exactly as ``on_mount`` leaves it: the workers
    read the captured reference, never the ``app`` property.
    """
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    app.call_from_thread.side_effect = lambda fn, *a: fn(*a)
    app.screen = gate  # the gate owns the screen, as it does in production
    gate._ui_app = app
    return gate, app


async def test_boot_worker_never_resolves_app_from_the_thread():
    """Regression: the boot thread has no Textual app context, so resolving
    ``self.app`` there raised NoActiveAppError mid-boot on Windows first
    launches and left a dead loading screen. The worker must run on the
    reference captured at mount, even when the property cannot resolve."""
    from textual._context import NoActiveAppError

    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(side_effect=NoActiveAppError())),
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    app.settle_landing.assert_called_once_with()
    app.reveal_landing.assert_called_once_with()


async def test_mount_captures_the_app_for_the_workers():
    """on_mount runs on the UI thread, the one place ``self.app`` always
    resolves, so that is where the workers' reference is captured."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    with (
        mock.patch.object(type(gate), "app", new=mock.PropertyMock(return_value=app)),
        mock.patch.object(StartupGate, "_retire_splash"),
    ):
        gate.on_mount()
    assert gate._ui_app is app


async def test_gate_release_reveals_chat():
    """A built container hands the screen to the landing view."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    app.screen = gate
    gate._ui_app = app
    with mock.patch.object(StartupGate, "query_one"):
        gate._release()
    app.reveal_landing.assert_called_once_with()


async def test_gate_releases_once_services_build_even_with_a_cold_engine(monkeypatch):
    """The gate waits for the container, never for the model load.

    The engine loads in the background after the handover; a prompt that arrives
    first waits inside its own answer bubble. Holding the screen for the load
    would re-block every launch that the lazy path exists to unblock.
    """
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    order: list[str] = []
    app.adopt_services.side_effect = lambda: order.append("adopt")
    app.reveal_landing.side_effect = lambda: order.append("reveal")
    with (
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    # The container, then the handover -- nothing in between asks the engine
    # whether it has finished loading.
    assert order == ["adopt", "reveal"]


async def test_gate_failure_surfaces_the_error_and_still_reveals_chat():
    """A failed start must not lock the user out of Catalog and Settings."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    app.screen = gate
    gate._ui_app = app
    with mock.patch.object(StartupGate, "query_one"):
        gate._fail("no such file")
    notified = [c.args[0] for c in app.notify.call_args_list if c.args]
    assert msg.STARTUP_FAILED.format(error="no such file") in notified
    app.reveal_landing.assert_called_once_with()


async def test_gate_builds_the_container_even_with_nothing_configured(monkeypatch):
    """An unconfigured app still gets its container and its handover.

    The catalog landing needs the container (HF client, downloads), and the
    per-role empty states are the voice for "pick a model" -- withholding the
    handover would strand the user on the gate.
    """
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    with (
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    app.settle_landing.assert_called_once_with()
    app.adopt_services.assert_called_once_with()
    app.reveal_landing.assert_called_once_with()


async def test_gate_settles_setup_before_it_hands_over(monkeypatch):
    """An already built container still waits on the setup answer.

    Nothing is left to build for a second TUI in the same process, which is the
    one case that used to skip the worker entirely. Handing over first and
    asking afterwards flashes up the very chat screen the answer withholds.
    """
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    order: list[str] = []
    app.settle_landing.side_effect = lambda: order.append("settle")
    app.reveal_landing.side_effect = lambda: order.append("reveal")
    monkeypatch.setattr(gate_mod, "peek_services", lambda: mock.MagicMock())
    with (
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    assert order == ["settle", "reveal"]
    # Nothing to settle in the refs: that work belongs to the build it skipped.
    app.canonicalize_persisted_models.assert_not_called()


async def test_gate_surfaces_a_failure_to_build_services(monkeypatch):
    """A container that cannot be built must show the error, not hang the screen."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    app.adopt_services.side_effect = OSError("disk gone")
    with (
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    notified = [c.args[0] for c in app.notify.call_args_list if c.args]
    assert msg.STARTUP_FAILED.format(error="disk gone") in notified
    app.reveal_landing.assert_called_once_with()


async def test_marshal_skips_the_ui_hop_once_the_app_is_tearing_down():
    """A gate worker that outlives the app must not touch a shutting-down UI thread."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    gate._ui_app = app
    with mock.patch.object(StartupGate, "_stopping", return_value=True):
        gate._marshal(gate._release)
    app.call_from_thread.assert_not_called()
    app.reveal_landing.assert_not_called()


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


async def test_first_run_hands_over_when_nothing_resolves(tmp_path, monkeypatch):
    """Regression: an unmounted gate must not be mistaken for a torn-down one.

    ``push_screen`` returns an AwaitMount. Left unawaited, the boot worker could
    reach ``_stopping`` before the gate mounted, silently drop the handover, and
    strand a first-run user on the loading screen forever. A first run lands on
    the catalog rather than on chat, so the assertion is that the gate steps
    aside at all, not which screen takes its place.
    """
    import asyncio
    import time

    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    monkeypatch.setattr("lilbee.cli.tui.app.chat_ready", lambda: False)
    monkeypatch.setattr("lilbee.cli.tui.app.embedding_ready", lambda: False)
    monkeypatch.setattr(LilbeeApp, "adopt_services", lambda self: None)

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and isinstance(app.screen, StartupGate):
            await pilot.pause()
            await asyncio.sleep(0.02)
        assert not isinstance(app.screen, StartupGate)


async def test_the_container_is_built_behind_the_gate(monkeypatch):
    """The app must not build the services container on its mount path.

    ``get_services`` constructs every singleton and spawns the role servers.
    Building it from ``on_mount`` blocks the first frame with the work the gate
    exists to hide, and leaves the gate with nothing to wait for: its boot
    worker, and the canonicalization that only runs there, never run at all.
    """
    import threading

    from lilbee.app.services import peek_services, set_services
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.startup_gate import StartupGate
    from tests._async_wait import wait_until

    set_services(None)
    container = mock.MagicMock()
    monkeypatch.setattr("lilbee.cli.tui.app.get_services", lambda: container)
    watched: list[object] = []
    monkeypatch.setattr("lilbee.cli.tui.app.watch_pool", lambda **kw: watched.append(kw))
    # An app with nothing to set up, so the gate runs its whole boot.
    monkeypatch.setattr("lilbee.cli.tui.app.chat_ready", lambda: True)
    monkeypatch.setattr("lilbee.cli.tui.app.embedding_ready", lambda: True)

    at_start_boot: list[object] = []
    real_start_boot = StartupGate.start_boot

    def _spy_start_boot(self):
        at_start_boot.append(peek_services())
        return real_start_boot(self)

    monkeypatch.setattr(StartupGate, "start_boot", _spy_start_boot)

    canonicalized: list[str] = []
    monkeypatch.setattr(
        LilbeeApp,
        "canonicalize_persisted_models",
        lambda self: canonicalized.append(threading.current_thread().name),
    )

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(
            pilot,
            lambda: bool(canonicalized) and bool(watched),
            max_pauses=200,
        )

    assert at_start_boot == [None], "the mount path built the container ahead of the gate"
    assert canonicalized, "the gate's boot worker never ran, so the refs were never settled"
    assert canonicalized[0] != threading.main_thread().name, "boot work ran on the UI thread"
    assert watched, "the app never watched the pool, so a services reset would drop its hints"


async def test_the_gate_opens_the_library_on_a_thread_the_interpreter_joins(monkeypatch):
    """A quit during the open waits for it, so it cannot hit CPython gh-87135.

    The daemon flag decides nothing here: an executor worker inherits it from
    the thread that grew the pool, and the interpreter joins every executor
    worker at exit (``concurrent.futures.thread._python_exit``) either way.
    """
    import threading
    from concurrent.futures import thread as executor_threads

    from lilbee.app.services import set_services
    from lilbee.cli.tui.app import LilbeeApp
    from tests._async_wait import wait_until

    set_services(None)
    opened_on: list[threading.Thread] = []
    container = mock.MagicMock()
    container.store.get_db.side_effect = lambda: opened_on.append(threading.current_thread())
    monkeypatch.setattr("lilbee.cli.tui.app.get_services", lambda: container)
    monkeypatch.setattr("lilbee.cli.tui.screens.startup_gate.get_services", lambda: container)
    monkeypatch.setattr("lilbee.cli.tui.app.chat_ready", lambda: True)
    monkeypatch.setattr("lilbee.cli.tui.app.embedding_ready", lambda: True)
    monkeypatch.setattr(LilbeeApp, "canonicalize_persisted_models", lambda self: None)

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: bool(opened_on), max_pauses=200)

    # The pool registers a worker just after starting it, so ask once the open is done.
    assert [t in executor_threads._threads_queues for t in opened_on] == [True]


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
    """Regression: reveal_landing switches the *current* screen, whichever it is.

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
    gate._ui_app = app
    with mock.patch.object(StartupGate, "query_one"):
        gate._release()
    app.reveal_landing.assert_not_called()


async def test_gate_hands_over_when_it_still_owns_the_screen():
    """The ordinary path: the gate is on top, so it reveals chat."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    app.screen = gate
    gate._ui_app = app
    with mock.patch.object(StartupGate, "query_one"):
        gate._release()
    app.reveal_landing.assert_called_once_with()


@pytest.mark.parametrize("container", [None, "built"])
async def test_start_boot_always_takes_the_worker(monkeypatch, container):
    """Regression: switching screens inside on_mount stalled Textual.

    start_boot runs at the tail of LilbeeApp.on_mount, so it must not hand over
    inline. There is no longer a fast path for an already built container
    either: the setup answer decides whether the handover happens at all, and
    it is not a question the UI thread can afford to ask.
    """
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    services = None if container is None else mock.MagicMock()
    monkeypatch.setattr(gate_mod, "peek_services", lambda: services)

    worker_calls: list = []
    deferred: list = []
    released: list = []
    monkeypatch.setattr(gate, "_boot_worker", lambda: worker_calls.append(True), raising=False)
    monkeypatch.setattr(gate, "call_after_refresh", deferred.append, raising=False)
    monkeypatch.setattr(gate, "_release", lambda: released.append(True), raising=False)

    gate.start_boot()

    assert worker_calls == [True]
    assert released == [], "the handover must not run inside on_mount"
    assert deferred == []


async def test_boot_worker_canonicalizes_before_the_setup_check(monkeypatch):
    """Canonicalization can swap a stale ref to a working one, which is exactly
    what decides whether setup is needed, so it must run first."""
    from lilbee.cli.tui.screens import startup_gate as gate_mod
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, app = _gate_with_app()
    order: list[str] = []
    monkeypatch.setattr(gate_mod, "peek_services", lambda: None)
    app.canonicalize_persisted_models.side_effect = lambda: order.append("canonicalize")
    app.settle_landing.side_effect = lambda: order.append("setup")
    with (
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
    monkeypatch.setattr(gate_mod, "peek_services", lambda: None)
    with (
        mock.patch.object(StartupGate, "query_one"),
        mock.patch.object(StartupGate, "_stopping", return_value=False),
    ):
        StartupGate._boot_worker.__wrapped__(gate)
    notified = [c.args[0] for c in app.notify.call_args_list if c.args]
    assert msg.STARTUP_FAILED.format(error="registry unreadable") in notified
    app.reveal_landing.assert_called_once_with()


async def test_gate_mount_retires_the_splash_then_repaints(monkeypatch):
    """The splash animates over the blank alt-screen until the gate paints;
    the dismissal then repaints anything a final frame touched."""
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate, _app = _gate_with_app()
    order: list[str] = []
    monkeypatch.setattr("lilbee.runtime.splash.dismiss", lambda: order.append("dismiss"))
    monkeypatch.setattr(gate, "_repaint", lambda: order.append("refresh"), raising=False)
    with mock.patch.object(StartupGate, "_stopping", return_value=False):
        StartupGate._retire_splash.__wrapped__(gate)
    assert order == ["dismiss", "refresh"]


def _app_held_at_the_gate(monkeypatch, tmp_path):
    """A LilbeeApp whose gate never hands over, with no services to build."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(StartupGate, "start_boot", lambda self: None)
    monkeypatch.setattr("lilbee.cli.tui.app.chat_ready", lambda: False)
    monkeypatch.setattr("lilbee.cli.tui.app.embedding_ready", lambda: False)
    monkeypatch.setattr(LilbeeApp, "adopt_services", lambda self: None)
    return LilbeeApp()


async def test_keys_typed_at_the_gate_reach_the_prompt_not_a_view_binding(monkeypatch, tmp_path):
    """Regression: a key typed while the gate shows fell to the app's view bindings.

    "t" opened Tasks, and the gate then gave up its handover because it no longer
    owned the screen. The app now keeps printable keys for the prompt.
    """
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    app = _app_held_at_the_gate(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, StartupGate)
        await pilot.press("t", "h", "i", "enter")
        await pilot.pause()
        assert isinstance(app.screen, StartupGate)
        assert app.active_view != "Tasks"
        assert app.take_type_ahead() == "thi"


async def test_text_pasted_at_the_gate_is_kept_like_typed_keys(monkeypatch, tmp_path):
    """A paste is one Paste event, not Key events; the app keeps its text too."""
    from textual import events

    from lilbee.cli.tui.screens.startup_gate import StartupGate

    app = _app_held_at_the_gate(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("a")
        app.post_message(events.Paste("bc d"))
        await pilot.pause()
        assert isinstance(app.screen, StartupGate)
        assert app.take_type_ahead() == "abc d"


async def test_keys_typed_over_a_screen_above_the_gate_reach_that_screen(
    monkeypatch, tmp_path
):
    """Only the gate and the chat it hands to keep keys; a screen above them gets its own."""
    from textual import events
    from textual.screen import Screen

    keys: list[str] = []

    class _Above(Screen[None]):
        def on_key(self, event: events.Key) -> None:
            keys.append(event.key)

    app = _app_held_at_the_gate(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(_Above())
        await pilot.press("x")
        await pilot.pause()
        assert keys == ["x"]
        assert app.take_type_ahead() == ""


async def test_the_prompt_takes_the_kept_text_once():
    """Taking the text ends the keeping, so a later key goes to its widget."""
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    app._type_ahead = ["a", "b"]
    assert app.take_type_ahead() == "ab"
    assert app.take_type_ahead() == ""
    assert app._type_ahead is None


async def test_pressing_the_sync_key_at_the_gate_starts_no_sync(monkeypatch, tmp_path):
    """S is a priority binding: left live, it swaps chat in under the gate.

    The gate then finds another screen on top and skips its handover. At the
    gate the app keeps the key as typed text.
    """
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(StartupGate, "start_boot", lambda self: None)
    monkeypatch.setattr("lilbee.cli.tui.app.chat_ready", lambda: False)
    monkeypatch.setattr("lilbee.cli.tui.app.embedding_ready", lambda: False)
    monkeypatch.setattr(LilbeeApp, "adopt_services", lambda self: None)
    syncs: list[str] = []
    monkeypatch.setattr(LilbeeApp, "action_run_sync", lambda self: syncs.append("sync"))

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, StartupGate)
        await pilot.press("a", "S", "b")
        await pilot.pause()
        gate = app.screen
        assert isinstance(gate, StartupGate)
        assert syncs == []
        assert app.take_type_ahead() == "aSb"


@pytest.mark.parametrize(
    ("key", "typed"),
    [("left_square_bracket", "]["), ("c", "]c"), ("escape", "]")],
)
async def test_no_view_opens_before_chat_is_installed(monkeypatch, tmp_path, key, typed):
    """The view keys reach no view while chat loads: the way back from one is chat.

    The app keeps them as typed text, as it keeps the sync key.
    """
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(LilbeeApp, "_load_chat_screen", lambda self, gate: None)
    monkeypatch.setattr(LilbeeApp, "adopt_services", lambda self: None)

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for pressed in ("right_square_bracket", key):
            await pilot.press(pressed)
            await pilot.pause()
        assert isinstance(app.screen, StartupGate)
        assert app._type_ahead == list(typed)
        app.switch_view("Chat")
        await pilot.pause()
        assert isinstance(app.screen, StartupGate)


async def test_a_host_with_no_gate_opens_a_view_with_no_chat_installed():
    """Only the gate waits for chat: a host that pushes its own first screen keeps its views."""
    from textual.screen import Screen

    from tests._async_wait import wait_until
    from tests._lilbee_app_test_host import LilbeeAppHost

    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(Screen())
        assert not app.is_screen_installed("chat")
        app.switch_view("Tasks")
        assert await wait_until(pilot, lambda: app.active_view == "Tasks", timeout=5.0)


@pytest.fixture
def failed_install_app(monkeypatch, tmp_path):
    """Build an app whose chat screen fails to install, with the list of boots it starts."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    def _build(*, cold: bool, error: Exception) -> tuple[LilbeeApp, list[StartupGate]]:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "data"))
        if cold:
            monkeypatch.setattr(LilbeeApp, "_load_chat_screen", LilbeeApp._chat_import_worker)
        monkeypatch.setattr(ChatScreen, "__init__", mock.Mock(side_effect=error))
        boots: list[StartupGate] = []
        monkeypatch.setattr(StartupGate, "_boot_worker", lambda self: boots.append(self))
        return LilbeeApp(), boots

    return _build


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (RuntimeError("boom"), "boom"),
        (ValueError("boom"), "boom"),
        (ValueError(), "ValueError"),
    ],
)
@pytest.mark.parametrize("cold", [True, False])
async def test_the_app_leaves_the_gate_when_chat_cannot_be_installed(
    failed_install_app, monkeypatch, cold, error, reason
):
    """No view opens without chat, so a failed install ends the app and says why."""
    import asyncio

    from lilbee.cli.tui.app import LilbeeApp

    app, _boots = failed_install_app(cold=cold, error=error)
    messages: list[object] = []
    leave = LilbeeApp.exit

    def _exit(self, *args, **kwargs):
        messages.append(kwargs.get("message"))
        leave(self, *args, **kwargs)

    monkeypatch.setattr(LilbeeApp, "exit", _exit)

    await asyncio.wait_for(app.run_async(headless=True, size=(120, 40)), timeout=20)
    assert app.return_code == 1
    assert messages == [msg.CHAT_STACK_FAILED.format(error=reason)]


@pytest.mark.parametrize("cold", [True, False])
async def test_no_boot_starts_when_chat_cannot_be_installed(failed_install_app, cold):
    """The boot builds the services and the role servers, which an app that leaves must not."""
    import asyncio

    app, boots = failed_install_app(cold=cold, error=ValueError("boom"))
    await asyncio.wait_for(app.run_async(headless=True, size=(120, 40)), timeout=20)
    assert boots == []


def _headless_run_tui(monkeypatch, auto_pilot=None) -> mock.Mock:
    """Make run_tui's own app run headless through Textual; return the straggler exit."""
    from lilbee.cli import tui
    from lilbee.cli.tui.app import LilbeeApp

    run_async = LilbeeApp.run_async
    monkeypatch.setattr(
        LilbeeApp,
        "run_async",
        lambda self: run_async(self, headless=True, size=(120, 40), auto_pilot=auto_pilot),
    )
    monkeypatch.setattr("lilbee.cli.sync.shutdown_executor", lambda: None)
    monkeypatch.setattr(tui, "reset_services_on_exit", lambda: None)
    straggler_exit = mock.Mock()
    monkeypatch.setattr(tui, "exit_when_stragglers_would_hang", straggler_exit)
    return straggler_exit


@pytest.mark.parametrize("cold", [True, False])
def test_the_shell_gets_code_1_when_chat_cannot_be_installed(failed_install_app, monkeypatch, cold):
    """The command ends with the app's return code, and a wedged thread exits with it too."""
    from lilbee.cli.tui import run_tui

    failed_install_app(cold=cold, error=RuntimeError("boom"))
    straggler_exit = _headless_run_tui(monkeypatch)
    with pytest.raises(SystemExit) as exited:
        run_tui()
    assert exited.value.code == 1
    straggler_exit.assert_called_once_with(1)


def test_a_quit_from_the_gate_returns_code_0(monkeypatch, tmp_path):
    """Ctrl-C quits through the app's own exit; the command returns with no error."""
    from lilbee.cli.tui import run_tui
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(StartupGate, "start_boot", lambda self: None)
    monkeypatch.setattr(LilbeeApp, "adopt_services", lambda self: None)
    codes: list[int | None] = []

    async def _quit(pilot) -> None:
        await pilot.press("ctrl+c")

    leave = LilbeeApp.exit

    def _exit(self, *args, **kwargs):
        leave(self, *args, **kwargs)
        codes.append(self.return_code)

    monkeypatch.setattr(LilbeeApp, "exit", _exit)
    straggler_exit = _headless_run_tui(monkeypatch, auto_pilot=_quit)
    set_services(mock.MagicMock())
    try:
        run_tui()
    finally:
        set_services(None)
    assert codes == [0]
    straggler_exit.assert_called_once_with(0)


async def test_a_view_opens_once_chat_is_installed(monkeypatch, tmp_path):
    """The refusal ends with the install, also while the gate still shows."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.startup_gate import StartupGate
    from tests._async_wait import wait_until

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(StartupGate, "start_boot", lambda self: None)
    monkeypatch.setattr(LilbeeApp, "adopt_services", lambda self: None)

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        assert await wait_until(pilot, lambda: app.is_screen_installed("chat"), timeout=5.0)
        assert isinstance(app.screen, StartupGate)
        app.switch_view("Tasks")
        assert await wait_until(pilot, lambda: app.active_view == "Tasks", timeout=5.0)
        assert not isinstance(app.screen, StartupGate)


async def test_gate_release_reveals_the_landing():
    from lilbee.cli.tui.screens.startup_gate import StartupGate

    gate = StartupGate()
    app = mock.MagicMock()
    app.screen = gate
    gate._ui_app = app
    with mock.patch.object(StartupGate, "query_one"):
        gate._release()
    app.reveal_landing.assert_called_once_with()


def test_landing_on_the_catalog_drops_the_typed_keys():
    """Keys typed at the gate are for the prompt; a catalog landing has none."""
    from lilbee.cli.tui.app import LilbeeApp

    app = mock.MagicMock()
    app.chat_is_ready = False
    app._type_ahead = ["h", "i"]
    LilbeeApp.reveal_landing(app)
    assert app._type_ahead is None
    app.switch_view.assert_called_once_with(msg.CATALOG_VIEW)
