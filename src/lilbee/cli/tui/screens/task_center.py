"""Task center screen — monitor background operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Static

from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.task_queue import STATUS_ICONS, Task, TaskStatus

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

log = logging.getLogger(__name__)

_STATUS_COLORS: dict[TaskStatus, str] = {
    TaskStatus.ACTIVE: "$primary",
    TaskStatus.DONE: "$success",
    TaskStatus.FAILED: "$error",
    TaskStatus.QUEUED: "$text-muted",
    TaskStatus.CANCELLED: "$text-muted",
}


def _status_icon(status: TaskStatus) -> str:
    """Return a unicode icon for the given task status."""
    return STATUS_ICONS.get(status, "?")


def _status_pill(status: TaskStatus) -> str:
    """Return a pill-formatted status badge."""
    color = _STATUS_COLORS.get(status, "$text-muted")
    badge = pill(status.value, color, "$text")
    return str(badge)


class TaskCenter(Screen[None]):
    """View for monitoring active, queued, and recent background tasks."""

    CSS_PATH = "task_center.tcss"

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
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        yield Static("Background Tasks", id="task-center-title")
        yield DataTable(id="task-table", cursor_type="row")
        yield Static("", id="task-detail")
        yield ViewTabs()
        yield Footer()

    def action_go_back(self) -> None:
        """Go back to the Chat screen."""
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.app.switch_view("Chat")
        else:
            self.app.pop_screen()

    def on_mount(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.add_columns("Status", "Name", "Type", "Progress")
        self._refresh_tasks()
        self.app.task_bar.queue.subscribe(self._on_queue_change)

    def on_unmount(self) -> None:
        self.app.task_bar.queue.unsubscribe(self._on_queue_change)

    def _on_queue_change(self) -> None:
        """Called when task queue changes - refresh the display."""
        try:
            self._refresh_tasks()
        except Exception:
            log.debug("Queue change refresh failed", exc_info=True)

    def action_refresh_tasks(self) -> None:
        """Refresh the task list."""
        self._refresh_tasks()

    def action_cancel_task(self) -> None:
        """Cancel the currently selected task."""
        table = self.query_one("#task-table", DataTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        if row_key.value is not None:
            self.app.task_bar.queue.cancel(row_key.value)
        self._refresh_tasks()

    def action_cursor_down(self) -> None:
        """Move cursor down in the task table."""
        self.query_one("#task-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up in the task table."""
        self.query_one("#task-table", DataTable).action_cursor_up()

    @on(DataTable.RowHighlighted, "#task-table")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show detail for the highlighted row."""
        key = event.row_key.value if event.row_key is not None else None
        self._show_task_detail(key)

    def _show_task_detail(self, task_id: str | None) -> None:
        """Update the detail panel with info about the given task."""
        detail = self.query_one("#task-detail", Static)
        if task_id is None:
            detail.update("")
            return
        task = self._find_task(task_id)
        if task is None:
            detail.update("")
            return
        text = f"{task.name} ({task.task_type}) — {task.progress}%"
        if task.detail:
            text += f"\n{task.detail}"
        detail.update(text)

    def _find_task(self, task_id: str) -> Task | None:
        """Look up a task by ID across all queue lists."""
        for task in self._all_tasks():
            if task.task_id == task_id:
                return task
        return None

    def _all_tasks(self) -> list[Task]:
        """Gather all tasks: active, queued, and history."""
        queue = self.app.task_bar.queue
        return queue.active_tasks + queue.queued_tasks + list(reversed(queue.history))

    def _refresh_tasks(self) -> None:
        """Populate task table from TaskBar's queue."""
        table = self.query_one("#task-table", DataTable)
        table.clear()
        for task in self._all_tasks():
            self._add_task_row(table, task)

    def _add_task_row(self, table: DataTable, task: Task) -> None:
        """Add a single task as a row in the data table."""
        status_badge = _status_pill(task.status)
        table.add_row(
            status_badge,
            task.name,
            task.task_type,
            f"{task.progress}%",
            key=task.task_id,
        )
