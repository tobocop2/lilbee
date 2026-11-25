"""The line a command shows on stderr while a first open upgrades the library."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from time import monotonic

from rich.progress import ProgressColumn, SpinnerColumn, Task
from rich.text import Text

from lilbee.core.config import cfg
from lilbee.runtime.console import PlainConsole, live_progress
from lilbee.runtime.progress import UPGRADE_HEADLINE, EventType, LibraryUpgradeEvent, ProgressEvent


class _LineColumn(ProgressColumn):
    """A progress column that shows *line* at each redraw, as literal text."""

    def __init__(self, line: Callable[[], Text]) -> None:
        super().__init__()
        self._line = line

    def render(self, task: Task) -> Text:
        return self._line()


class UpgradeLine:
    """Shows library upgrade events on *console*: a live line on a terminal, else plain lines.

    The live line is a spinner and the latest line, which Rich's own refresh reads
    again at each redraw. With *live* off, under ``--json`` or piped, each event is a line.
    """

    def __init__(self, console: PlainConsole, *, live: bool = True) -> None:
        self._console = console
        self.live = live
        self._event: LibraryUpgradeEvent | None = None
        self._started = 0.0
        # Holds the live progress display while a live line shows.
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
            progress = live_progress(
                SpinnerColumn("dots", style="status.spinner"),
                _LineColumn(self.__rich__),
                console=self._console,
                transient=True,
                refresh_per_second=12.5,
            )
            self._status = ExitStack()
            self._status.enter_context(progress)
            progress.add_task("", total=None)

    def _show_end(self, event: LibraryUpgradeEvent) -> None:
        if self._status is not None:
            self._status.close()
            self._status = None
        self._event = None
        self._write(event.text)

    def _write(self, line: str) -> None:
        self._console.print(Text(line), soft_wrap=True, highlight=False)
