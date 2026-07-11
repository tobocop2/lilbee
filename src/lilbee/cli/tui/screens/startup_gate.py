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

from lilbee.app.services import get_services, peek_services
from lilbee.app.setup_state import needs_setup
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.runtime.bee_logo import BEE_LINES

log = logging.getLogger(__name__)

_LOGO = "\n".join(BEE_LINES)


class StartupGate(Screen[None]):
    """Holds the screen until the app can serve, then hands over to chat.

    The engine itself loads in the background after the handover; a prompt sent
    before it is ready waits inside its own answer bubble with live progress.
    """

    CSS_PATH = "startup_gate.tcss"

    # Lilbee always hosts screens on a LilbeeApp, so narrowing the type lets
    # reveal_chat resolve without reflection.
    app: LilbeeApp  # type: ignore[assignment]

    def compose(self) -> ComposeResult:
        with Vertical(id="gate-body"):
            yield Static(_LOGO, id="gate-logo")
            yield ProgressBar(total=None, show_eta=False, show_percentage=False, id="gate-bar")
            yield Static(msg.STARTUP_PREPARING, id="gate-status")

    def on_mount(self) -> None:
        """Retire the launcher's splash now that Textual is painting.

        The splash animates over the blank alt-screen right up to this moment,
        so the wordmark never leaves the terminal. Dismissal waits on the
        subprocess, so it runs off-thread; the refresh afterwards repaints
        anything a final splash frame may have touched.
        """
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
        """Reveal chat at once when services already exist, else build them off-thread.

        A container that is already built (a second TUI in the same process, a
        test host) has nothing to wait for, so painting a loading screen would be
        a lie. The handover is deferred a frame because start_boot runs at the
        tail of the app's on_mount, and switching screens from inside on_mount
        stalls Textual.
        """
        if peek_services() is not None:
            self.call_after_refresh(self._release)
            return
        self._boot_worker()

    @work(thread=True, name="startup_gate", exit_on_error=False)
    def _boot_worker(self) -> None:
        """Settle the model refs, build the services container, then hand over.

        Canonicalization runs first (it can swap a stale ref to a working
        fallback, which decides whether setup is needed) and runs here rather
        than on the mount path: its disk reads and server probes would sit
        between the terminal handover and the TUI's first frame.
        """
        try:
            self.app.canonicalize_persisted_models()
            if needs_setup():
                self._marshal(self._release)
                return
            get_services()
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
        self.app.call_from_thread(callback, *args)

    def _fail(self, error: str) -> None:
        """Surface a failed start and hand the user to the rest of the TUI to fix it."""
        self.app.notify(msg.STARTUP_FAILED.format(error=error), severity="error", timeout=8)
        self._release()

    def _release(self) -> None:
        """Hand the screen to chat, unless something else has taken it.

        reveal_chat switches whatever screen is on top, so a gate that resolved
        after another screen opened above it would replace that screen instead of
        itself. No widget lookup here either: the gate can resolve before compose
        has mounted its children, and a missed query would strand the user.
        """
        if self.app.screen is not self:
            return
        self.app.reveal_chat()
