"""The line a command shows on stderr while a first open upgrades the library."""

from __future__ import annotations

from contextlib import ExitStack
from time import monotonic

from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lilbee.core.config import cfg
from lilbee.runtime.console import PlainConsole
from lilbee.runtime.progress import UPGRADE_HEADLINE, EventType, LibraryUpgradeEvent, ProgressEvent


class UpgradeLine:
    """Shows library upgrade events on *console*: a live line on a terminal, else plain lines.

    The live line is a Rich spinner, whose own refresh reads the elapsed time
    again at each redraw. With *live* off, under ``--json`` or piped, each event is a line.
    """

    def __init__(self, console: PlainConsole, *, live: bool = True) -> None:
        self._console = console
        self.live = live
        self._event: LibraryUpgradeEvent | None = None
        self._started = 0.0
        # Holds Rich's status spinner while a live line shows.
        self._status: ExitStack | None = None

    def __call__(self, _event_type: EventType, event: ProgressEvent) -> None:
        # The reporter takes the whole progress union; the store passes it upgrade events only.
        if not isinstance(event, LibraryUpgradeEvent):
            return
        if event.running:
            self._show_running(event)
        else:
            self._show_end(event)

    def __rich__(self) -> Text:
        """The progress line of the latest event, timed to now."""
        event = self._event
        return Text("" if event is None else event.line_at(monotonic() - self._started))

    def _show_running(self, event: LibraryUpgradeEvent) -> None:
        if self._event is None:
            self._write(UPGRADE_HEADLINE)
        self._event = event
        self._started = monotonic() - event.elapsed_s
        if not (self.live and self._console.is_terminal and not cfg.json_mode):
            self._write(event.text)
        elif self._status is None:
            # Rich's own status spinner redirects stdout into the console; a print to
            # stdout stays on stdout, so the same spinner is built on a Live that does not.
            live = Live(
                Spinner("dots", text=self, style="status.spinner"),
                console=self._console,
                refresh_per_second=12.5,
                transient=True,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._status = ExitStack()
            self._status.enter_context(live)

    def _show_end(self, event: LibraryUpgradeEvent) -> None:
        if self._status is not None:
            self._status.close()
            self._status = None
        self._event = None
        self._write(event.text)

    def _write(self, line: str) -> None:
        self._console.print(Text(line), soft_wrap=True, highlight=False)
