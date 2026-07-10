"""Blocking startup screen: the lilbee wordmark and the chat model's real load progress."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import ProgressBar, Static
from textual.worker import get_current_worker

from lilbee.app.placement import ACTIVE_WARM_PHASES, chat_engine_ready
from lilbee.app.services import get_services
from lilbee.app.setup_state import needs_setup
from lilbee.catalog.formatting import display_label_for_ref
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.core.config import cfg
from lilbee.providers.warm_progress import WarmPhase, WarmProgress
from lilbee.runtime.bee_logo import BEE_LINES

log = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.1
# Covers get_services kicking its warm on a separate thread before a phase is stamped.
_WARM_START_GRACE_S = 3.0
_LOGO = "\n".join(BEE_LINES)


class StartupGate(Screen[None]):
    """Holds the screen until the chat engine can answer, or has failed trying."""

    CSS_PATH = "startup_gate.tcss"

    # Lilbee always hosts screens on a LilbeeApp, so narrowing the type lets
    # reveal_chat resolve without reflection.
    app: LilbeeApp  # type: ignore[assignment]

    def compose(self) -> ComposeResult:
        with Vertical(id="gate-body"):
            yield Static(_LOGO, id="gate-logo")
            yield ProgressBar(total=None, show_eta=False, id="gate-bar")
            yield Static(msg.STARTUP_PREPARING, id="gate-status")

    def start_boot(self) -> None:
        """Reveal chat at once when a prompt can already be served, else warm off-thread.

        A fleet that is already up (a second TUI against a live engine) has nothing
        to wait for, so painting a loading screen would be a lie.
        """
        if chat_engine_ready():
            self._release()
            return
        self._boot_worker()

    @work(thread=True, name="startup_gate", exit_on_error=False)
    def _boot_worker(self) -> None:
        """Build services (which warms the fleet), then track the warm to completion."""
        if needs_setup():
            self._marshal(self._release)
            return

        try:
            provider = get_services().provider
        except Exception as exc:
            # Any failure to build the container leaves the user with no engine.
            # Show it and hand them the rest of the TUI rather than a dead screen.
            log.exception("startup gate could not build the services container")
            self._marshal(self._fail, str(exc))
            return

        if not cfg.worker_pool_eager_start:
            self._marshal(self._release)
            return

        # The grace covers get_services kicking its warm on a separate thread. With
        # no fleet, no warm, or a model that was never installed, nothing will ever
        # stamp a phase, and holding the screen forever would be worse than the
        # blank terminal this gate replaces.
        grace_deadline = time.monotonic() + _WARM_START_GRACE_S
        while not chat_engine_ready():
            if self._stopping():
                return
            snapshot = provider.warm_progress()
            if snapshot is not None and snapshot.phase is WarmPhase.ERROR:
                self._marshal(self._fail, snapshot.error or "")
                return
            warm_in_flight = snapshot is not None and snapshot.phase in ACTIVE_WARM_PHASES
            if warm_in_flight:
                self._marshal(self._apply_snapshot, snapshot)
            elif time.monotonic() > grace_deadline:
                break
            time.sleep(_POLL_INTERVAL_S)

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

    def _apply_snapshot(self, snapshot: WarmProgress) -> None:
        """Reflect one warm snapshot onto the bar and the status line."""
        bar = self.query_one("#gate-bar", ProgressBar)
        status = self.query_one("#gate-status", Static)
        if snapshot.phase is WarmPhase.READING_WEIGHTS and snapshot.bytes_total:
            bar.update(total=snapshot.bytes_total, progress=snapshot.bytes_done)
            status.update(msg.STARTUP_READING_WEIGHTS.format(name=_model_label(snapshot)))
            return
        # No byte signal outside the read phase, so the bar stays indeterminate.
        bar.update(total=None)
        status.update(
            msg.STARTUP_LOADING_ENGINE
            if snapshot.phase is WarmPhase.LOADING_ENGINE
            else msg.STARTUP_PREPARING
        )

    def _fail(self, error: str) -> None:
        """Surface a failed load and hand the user to the rest of the TUI to fix it."""
        self.app.notify(msg.STARTUP_FAILED.format(error=error), severity="error", timeout=8)
        self.app.notify(msg.STARTUP_FAILED_HINT, severity="error", timeout=8)
        self._release()

    def _release(self) -> None:
        """Reveal the chat screen.

        No widget lookup here: the gate can resolve before compose has mounted its
        children, and a missed query would strand the user on the loading screen.
        """
        self.app.reveal_chat()


def _model_label(snapshot: WarmProgress) -> str:
    """The name to show while reading weights."""
    return display_label_for_ref(snapshot.model_ref) if snapshot.model_ref else ""
