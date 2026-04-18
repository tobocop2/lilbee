"""Task center screen — monitor background operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from rich.console import RenderableType
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Static

from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.task_queue import STATUS_ICONS, Task, TaskStatus
from lilbee.cli.tui.widgets.progress_cell import indeterminate_cell, progress_cell

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

log = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 0.1

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
    AUTO_FOCUS = "#task-table"
    HELP = "Background task monitor.\n\nUse r to refresh, c to cancel the selected task."

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
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        yield Static("Background Tasks", id="task-center-title")
        yield DataTable(id="task-table", cursor_type="row")
        yield Static("", id="task-detail")
        yield TaskBar()
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
        cols = table.add_columns("Status", "Name", "Type", "Progress")
        self._col_status, self._col_name, self._col_type, self._col_progress = cols
        self._tick: int = 0
        self._row_task_ids: set[str] = set()
        self._poll()
        self.set_interval(_POLL_INTERVAL_SECONDS, self._poll)

    def action_refresh_tasks(self) -> None:
        """Refresh the task list."""
        self._poll()

    def action_cancel_task(self) -> None:
        """Cancel the currently selected task."""
        table = self.query_one("#task-table", DataTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        if row_key.value is not None:
            self.app.task_bar.queue.cancel(row_key.value)
        self._poll()

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
        text = f"{task.name} ({task.task_type})"
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

    def _poll(self) -> None:
        """Reconcile DataTable rows with the task queue in-place at 10 Hz."""
        self._tick += 1
        table = self.query_one("#task-table", DataTable)
        tasks = self._all_tasks()
        seen: set[str] = set()
        for task in tasks:
            seen.add(task.task_id)
            if task.task_id in self._row_task_ids:
                self._update_row_cells(table, task)
            else:
                self._add_row(table, task)
                self._row_task_ids.add(task.task_id)
        stale = self._row_task_ids - seen
        for tid in stale:
            try:
                table.remove_row(tid)
            except Exception:
                log.debug("Row %s already removed", tid, exc_info=True)
            self._row_task_ids.discard(tid)
        if table.row_count > 0:
            try:
                row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
                if row_key.value is not None:
                    self._show_task_detail(row_key.value)
            except Exception:
                log.debug("Failed to refresh detail panel", exc_info=True)

    def _add_row(self, table: DataTable, task: Task) -> None:
        """Append a new row for the given task."""
        table.add_row(
            _status_pill(task.status),
            task.name,
            task.task_type,
            self._progress_renderable(task),
            key=task.task_id,
        )

    def _update_row_cells(self, table: DataTable, task: Task) -> None:
        """Update cells in place for an existing row."""
        try:
            table.update_cell(task.task_id, self._col_status, _status_pill(task.status))
            table.update_cell(task.task_id, self._col_name, task.name)
            table.update_cell(task.task_id, self._col_type, task.task_type)
            table.update_cell(task.task_id, self._col_progress, self._progress_renderable(task))
        except Exception:
            log.debug("Failed to update row %s", task.task_id, exc_info=True)

    def _progress_renderable(self, task: Task) -> RenderableType:
        """Render progress cell for a task (block bar or indeterminate pulse)."""
        if task.indeterminate:
            return indeterminate_cell(self._tick)
        return progress_cell(task.progress)
