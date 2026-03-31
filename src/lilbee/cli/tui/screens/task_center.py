"""TaskCenter screen — shows all background tasks (active, queued, history)."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Header, Static

from lilbee.cli.tui.task_queue import TaskStatus
from lilbee.cli.tui.widgets.task_bar import TaskBar


class TaskCenter(Screen):
    """Screen showing all background tasks — active, queued, and history."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "pop_screen", "Back", show=False),
    ]

    def __init__(self, task_bar: TaskBar | None = None) -> None:
        super().__init__()
        self._task_bar = task_bar

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="task-center-content"):
            yield Static("Tasks", id="tc-title")
            yield Static("", id="tc-active")
            yield Static("", id="tc-queued")
            yield Static("", id="tc-history")

    def on_mount(self) -> None:
        self._refresh_display()

    def _get_task_bar(self) -> TaskBar:
        """Get task bar from app or use cached reference."""
        if self._task_bar is None:
            self._task_bar = getattr(self.app, "_task_bar", None)
        if self._task_bar is None:
            self._task_bar = self.app.query_one("#global-nav-bar", TaskBar)
        return self._task_bar

    def _refresh_display(self) -> None:
        """Refresh task display."""
        task_bar = self._get_task_bar()
        if task_bar is None:
            return

        queue = task_bar.queue
        active = queue.active_task
        queued = queue.queued_tasks

        active_label = self.query_one("#tc-active", Static)
        queued_label = self.query_one("#tc-queued", Static)

        if active:
            icon = "\u25b6" if active.status == TaskStatus.ACTIVE else "\u2713"
            detail = f" — {active.detail}" if active.detail else ""
            active_label.update(
                f"[bold]{icon} {active.name}[/bold]{detail}\n\n"
                f"[bold]Queued:[/bold]"
            )
        else:
            active_label.update("[bold]No active tasks[/bold]\n\n[bold]Queued:[/bold]")

        if queued:
            lines = []
            for i, t in enumerate(queued, 1):
                lines.append(f"  {i}. {t.name} ({t.task_type})")
            queued_label.update("\n".join(lines))
        else:
            queued_label.update("  (none)")


class TaskCenterPlaceholder(Screen):
    """Placeholder when TaskBar is not available."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "pop_screen", "Back", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("No tasks available", id="tc-placeholder")
