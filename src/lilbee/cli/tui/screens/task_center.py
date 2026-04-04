"""Task center screen — monitor background operations."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from lilbee.cli.tui.task_queue import TaskStatus
from lilbee.cli.tui.widgets.nav_bar import NavBar


class TaskCenter(Screen[None]):
    """View for monitoring active, queued, and recent background tasks."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "pop_screen", "Back", show=False),
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
        Binding("r", "refresh_tasks", "Refresh", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Background Tasks", id="task-center-title")
        yield DataTable(id="task-table")
        yield Static("", id="task-detail")
        yield Footer()
        yield NavBar(id="global-nav-bar")

    def on_mount(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.add_columns("Status", "Name", "Type", "Progress")
        table.cursor_type = "row"
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Populate table from TaskBar's queue."""
        table = self.query_one("#task-table", DataTable)
        table.clear()

        task_bar = getattr(self.app, "_task_bar", None)
        if task_bar is None:
            table.add_row("—", "No task bar available", "", "")
            return

        queue = task_bar.queue
        active = queue.active_task
        queued = queue.queued_tasks

        if not active and not queued:
            table.add_row("—", "No active tasks", "", "")
            self.query_one("#task-detail", Static).update("All quiet.")
            return

        if active:
            status_icon = _status_icon(active.status)
            pct = f"{active.progress}%" if active.progress > 0 else "..."
            detail = active.detail or ""
            table.add_row(
                status_icon,
                active.name,
                active.task_type,
                f"{pct} {detail}",
                key=active.task_id,
            )

        for task in queued:
            table.add_row(
                _status_icon(task.status),
                task.name,
                task.task_type,
                "queued",
                key=task.task_id,
            )

    def action_refresh_tasks(self) -> None:
        self._refresh_table()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_cursor_down(self) -> None:
        self.query_one("#task-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#task-table", DataTable).action_cursor_up()


def _status_icon(status: TaskStatus) -> str:
    """Map task status to a display icon."""
    return {
        TaskStatus.QUEUED: "⏳",
        TaskStatus.ACTIVE: "▶",
        TaskStatus.DONE: "✓",
        TaskStatus.FAILED: "✗",
        TaskStatus.CANCELLED: "⊘",
    }.get(status, "?")
