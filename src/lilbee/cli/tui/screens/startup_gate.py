"""Startup splash: the lilbee wordmark while the services build and the library opens."""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import monotonic

from textual import events, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import ProgressBar, Static
from textual.worker import get_current_worker

from lilbee.app.services import get_services, peek_services, report_library_upgrade
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.runtime.bee_logo import BEE_LINES
from lilbee.runtime.progress import (
    UPGRADE_HEADLINE,
    EventType,
    LibraryUpgradeEvent,
    ProgressEvent,
    UpgradePhase,
)

log = logging.getLogger(__name__)

_LOGO = "\n".join(BEE_LINES)
# How often the status line redraws the elapsed time of a library upgrade.
_UPGRADE_TICK_SECONDS = 1.0
# How long the finished upgrade's toast stays on the landing view.
_UPGRADE_DONE_TOAST_SECONDS = 6


class StartupGate(Screen[None]):
    """Holds the screen until readiness settles, then hands over to the landing view.

    The engine itself loads in the background after the handover; a prompt sent
    before it is ready waits inside its own answer bubble with live progress.
    """

    CSS_PATH = "startup_gate.tcss"

    # Lilbee always hosts screens on a LilbeeApp, so narrowing the type lets
    # reveal_landing resolve without reflection.
    app: LilbeeApp  # type: ignore[assignment]

    # The app reference every worker and UI hop uses: ``self.app`` reads a
    # contextvar that plain threads never carry and then walks ``_parent``,
    # which raises NoActiveAppError the moment the gate detaches. Captured
    # once in on_mount, on the UI thread, where resolution is certain.
    _ui_app: LilbeeApp

    def __init__(self) -> None:
        super().__init__()
        # The latest event of a library upgrade that runs, when it began, and the
        # timer that redraws its time; no timer while none runs.
        self._upgrade_event = LibraryUpgradeEvent(phase=UpgradePhase.WAITING)
        self._upgrade_started = 0.0
        self._upgrade_timer: Timer | None = None
        # Printable keys pressed while the gate shows; the prompt receives them.
        self._typed: list[str] = []

    def on_key(self, event: events.Key) -> None:
        """Keep printable keys for the prompt so no view binding takes them."""
        if event.is_printable and event.character:
            self._typed.append(event.character)
            event.prevent_default()
            event.stop()

    def compose(self) -> ComposeResult:
        with Vertical(id="gate-body"):
            yield Static(_LOGO, id="gate-logo")
            yield ProgressBar(total=None, show_eta=False, show_percentage=False, id="gate-bar")
            yield Static(msg.STARTUP_PREPARING, id="gate-status")

    def on_mount(self) -> None:
        """Capture the app for the workers, then retire the launcher's splash.

        The splash animates over the blank alt-screen right up to this moment,
        so the wordmark never leaves the terminal. Dismissal waits on the
        subprocess, so it runs off-thread; the refresh afterwards repaints
        anything a final splash frame may have touched.
        """
        self._ui_app = self.app
        self._retire_splash()

    @work(thread=True, name="splash_retire", exit_on_error=False)
    def _retire_splash(self) -> None:
        from lilbee.runtime.splash import dismiss

        dismiss()
        self._marshal(self._repaint)

    def _repaint(self) -> None:
        """Repaint anything a final splash frame may have scribbled over."""
        self.refresh()

    def start_boot(self) -> None:
        """Work out what the app can serve, off the UI thread, then hand over.

        One path even when the container is already built (a second TUI in the
        same process, a test host): the setup answer still decides whether the
        handover happens. The thread hop also keeps the screen switch out of
        on_mount, which stalls Textual.
        """
        self._boot_worker()

    @work(thread=True, name="startup_gate", exit_on_error=False)
    def _boot_worker(self) -> None:
        """Settle the refs, settle readiness, build the container, hand over.

        Canonicalization runs first: it can swap a stale ref for a working one,
        or adopt an installed model into an unconfigured role, which decides
        the landing view. An already built container skips it.

        ``settle_landing`` blocks until the app has recorded the answer, so
        the handover cannot read a readiness flag that has yet to be written.

        Building the container spawns the role servers, so it belongs on this
        thread, behind the loading bar, and so does opening the library: an open
        that upgrades it shows each step in the status line.
        """
        report_library_upgrade(self._show_upgrade)
        try:
            if peek_services() is None:
                self._ui_app.canonicalize_persisted_models()
            self._ui_app.settle_landing()
            self._ui_app.adopt_services()
            get_services().store.get_db()
        except Exception as exc:
            # Any failure to prepare the app leaves the user with no engine.
            # Show it and hand them the rest of the TUI rather than a dead screen.
            log.exception("startup gate could not prepare the app")
            self._marshal(self._fail, str(exc))
            return
        self._marshal(self._release)

    def _show_upgrade(self, _event_type: EventType, event: ProgressEvent) -> None:
        """Hand a library upgrade event to the UI thread, from the opening thread."""
        # The reporter takes the whole progress union; the store passes it upgrade events only.
        if isinstance(event, LibraryUpgradeEvent) and self.is_mounted:
            self._ui_app.call_from_thread(self._show_event, event)

    def _show_event(self, event: LibraryUpgradeEvent) -> None:
        """Show *event* under the bar; while the upgrade runs its time redraws each second.

        The finished line is also a toast, so it outlives the gate's handover.
        """
        if not event.running:
            self._stop_ticking()
            self._set_status(event.text)
            if event.phase is UpgradePhase.DONE:
                self._ui_app.notify(event.text, timeout=_UPGRADE_DONE_TOAST_SECONDS)
            return
        self._upgrade_event = event
        self._upgrade_started = monotonic() - event.elapsed_s
        if self._upgrade_timer is None:
            self._upgrade_timer = self.set_interval(_UPGRADE_TICK_SECONDS, self._tick_upgrade)
        self._tick_upgrade()

    def _tick_upgrade(self) -> None:
        """Redraw the upgrade's headline and its progress line, timed to now."""
        line = self._upgrade_event.line_at(monotonic() - self._upgrade_started)
        self._set_status(f"{UPGRADE_HEADLINE}\n{line}")

    def _stop_ticking(self) -> None:
        """Stop redrawing the upgrade's time; safe when it never started."""
        if self._upgrade_timer is not None:
            self._upgrade_timer.stop()
            self._upgrade_timer = None

    def _set_status(self, text: str) -> None:
        """Put *text* in the status line under the bar."""
        for status in self.query("#gate-status").results(Static):
            status.update(text)

    def _stopping(self) -> bool:
        """True once the worker was cancelled or the gate left the screen."""
        worker = get_current_worker()
        return worker.is_cancelled or not self.is_mounted

    def _marshal(self, callback: Callable[..., None], *args: object) -> None:
        """Hop to the UI thread, unless the app is already tearing down."""
        if self._stopping():
            return
        self._ui_app.call_from_thread(callback, *args)

    def _fail(self, error: str) -> None:
        """Surface a failed start and hand the user to the rest of the TUI to fix it."""
        self._stop_ticking()
        self._ui_app.notify(msg.STARTUP_FAILED.format(error=error), severity="error", timeout=8)
        self._release()

    def _release(self) -> None:
        """Hand the screen to the landing view, unless something else has taken it.

        reveal_landing switches whatever screen is on top, so a gate that resolved
        after another screen opened above it would replace that screen instead of
        itself. No widget lookup here either: the gate can resolve before compose
        has mounted its children, and a missed query would strand the user.
        """
        if self._ui_app.screen is not self:
            return
        self._ui_app.type_ahead = "".join(self._typed)
        self._ui_app.reveal_landing()
