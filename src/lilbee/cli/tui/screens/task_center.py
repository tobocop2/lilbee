"""Task Center screen -- flight-deck-style background task monitor.

Each task renders as a ``TaskRow`` with a three-line body (title +
type, detail + percent, block-char bar) and a thick left rail in the
state's color. On the active row the rail pulses at ~1 Hz, which is
the only motion in the screen beyond the bar filling.

The render path is poll-based: ``_poll`` runs on the main thread at
10 Hz, reads the shared ``TaskQueue``, and reconciles rows in place by
task_id. There's no subscriber chain; tasks owned by the controller
write into the lock-protected queue from worker threads and the poll
picks them up next tick.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Label

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.task_queue import Task, TaskStatus
from lilbee.cli.tui.widgets.task_row import TaskRow

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

log = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 0.1


class TaskCenter(Screen[None]):
    """Live view of active + queued + recently completed tasks."""

    CSS_PATH = "task_center.tcss"
    AUTO_FOCUS = "#task-rows"
    HELP = "Background task monitor.\n\nPress r to refresh, c to cancel the focused task."

    app: LilbeeApp

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "go_back", "Back", show=False),
        Binding("r", "refresh_tasks", "Refresh", show=True),
        Binding("c", "cancel_task", "Cancel", show=True),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        yield Label(msg.TASK_CENTER_TITLE, id="task-center-title")
        yield Label("", id="task-center-counts")
        yield VerticalScroll(id="task-rows")
        yield Label(
            f"{msg.TASK_CENTER_EMPTY_HEADLINE}\n{msg.TASK_CENTER_EMPTY_DETAIL}",
            id="task-center-empty",
        )
        yield Label(msg.TASK_CENTER_HINT, id="task-center-hint")
        yield TaskBar()
        yield ViewTabs()
        yield Footer()

    def action_go_back(self) -> None:
        """Return to Chat (or pop if we're on a detached test app)."""
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.app.switch_view("Chat")
        else:
            self.app.pop_screen()

    def on_mount(self) -> None:
        self._tick: int = 0
        self._rows: dict[str, TaskRow] = {}
        self._poll()
        self.set_interval(_POLL_INTERVAL_SECONDS, self._poll)

    def action_refresh_tasks(self) -> None:
        """Manual refresh (r). No-op beyond forcing an immediate poll."""
        self._poll()

    def action_cancel_task(self) -> None:
        """Cancel the task whose row currently has focus.

        Falls back to the first active task if no row has focus.
        """
        focused = self.focused
        if isinstance(focused, TaskRow) and focused.id:
            self.app.task_bar.queue.cancel(focused.id.removeprefix("task-"))
            return
        active = self.app.task_bar.queue.active_task
        if active is not None:
            self.app.task_bar.queue.cancel(active.task_id)

    def action_cursor_down(self) -> None:
        self.focus_next()

    def action_cursor_up(self) -> None:
        self.focus_previous()

    def _all_tasks(self) -> list[Task]:
        """Tasks in display order: active first, then queued, then history."""
        queue = self.app.task_bar.queue
        return queue.active_tasks + queue.queued_tasks + list(reversed(queue.history))

    def _poll(self) -> None:
        """10 Hz reconciliation: add new rows, update existing, remove stale."""
        self._tick += 1
        container = self.query_one("#task-rows", VerticalScroll)
        tasks = self._all_tasks()
        seen: set[str] = set()
        for task in tasks:
            seen.add(task.task_id)
            row = self._rows.get(task.task_id)
            if row is None:
                row = TaskRow(task_id=task.task_id)
                self._rows[task.task_id] = row
                container.mount(row)
            row.update(task, self._tick)
        for tid in list(self._rows):
            if tid not in seen:
                row = self._rows.pop(tid)
                try:
                    row.remove()
                except Exception:
                    log.debug("Row %s already removed", tid, exc_info=True)
        self._update_counts(tasks)
        empty = self.query_one("#task-center-empty", Label)
        empty.display = not tasks

    def _update_counts(self, tasks: list[Task]) -> None:
        """Top-right status strip: N running · M queued · K done."""
        counts_label = self.query_one("#task-center-counts", Label)
        active = sum(1 for t in tasks if t.status == TaskStatus.ACTIVE)
        queued = sum(1 for t in tasks if t.status == TaskStatus.QUEUED)
        done = sum(1 for t in tasks if t.status == TaskStatus.DONE)
        counts_label.update(msg.TASK_CENTER_COUNTS.format(active=active, queued=queued, done=done))
