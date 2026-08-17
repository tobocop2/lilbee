"""Startup splash: the lilbee wordmark while the services container builds."""

from __future__ import annotations

import logging
from collections.abc import Callable

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import ProgressBar, Static
from textual.worker import get_current_worker

from lilbee.app.services import peek_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.runtime.bee_logo import BEE_LINES

log = logging.getLogger(__name__)

_LOGO = "\n".join(BEE_LINES)


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
        thread, behind the loading bar.
        """
        try:
            if peek_services() is None:
                self._ui_app.canonicalize_persisted_models()
            self._ui_app.settle_landing()
            self._ui_app.adopt_services()
        except Exception as exc:
            # Any failure to prepare the app leaves the user with no engine.
            # Show it and hand them the rest of the TUI rather than a dead screen.
            log.exception("startup gate could not prepare the app")
            self._marshal(self._fail, str(exc))
            return
        self._marshal(self._release)

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
        self._ui_app.reveal_landing()
