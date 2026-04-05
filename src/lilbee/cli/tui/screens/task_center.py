"""Task center screen — monitor background operations."""

from __future__ import annotations

from typing import ClassVar, Callable

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Collapsible, Footer, Header, ProgressBar, Static

from lilbee.cli.tui.task_queue import Task, TaskStatus
from lilbee.cli.tui.widgets.nav_bar import NavBar


class TaskCenter(Screen[None]):
    """View for monitoring active, queued, and recent background tasks."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "pop_screen", "Back", show=False),
        Binding("r", "refresh_tasks", "Refresh", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield NavBar(id="global-nav-bar")
        yield Header()
        yield Static("Background Tasks", id="task-center-title")
        yield VerticalScroll(id="task-list")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_tasks()
        self._subscribe_to_queue()

    def on_unmount(self) -> None:
        self._unsubscribe_from_queue()

    def _subscribe_to_queue(self) -> None:
        """Subscribe to task bar queue changes for real-time updates."""
        task_bar = getattr(self.app, "_task_bar", None)
        if task_bar is not None:
            task_bar.queue.subscribe(self._on_queue_change)

    def _unsubscribe_from_queue(self) -> None:
        """Unsubscribe from task bar queue changes."""
        task_bar = getattr(self.app, "_task_bar", None)
        if task_bar is not None:
            task_bar.queue.unsubscribe(self._on_queue_change)

    def _on_queue_change(self) -> None:
        """Called when task queue changes - refresh the display."""
        try:
            self._refresh_tasks_safe()
        except Exception:
            pass

    def _refresh_tasks_safe(self) -> None:
        """Safely refresh tasks, catching any errors."""
        try:
            self._refresh_tasks()
        except Exception:
            pass

    def action_refresh_tasks(self) -> None:
        """Refresh the task list."""
        self._refresh_tasks()

    def _refresh_tasks(self) -> None:
        """Populate task list from TaskBar's queue."""
        task_list = self.query_one("#task-list", VerticalScroll)
        task_list.remove_children()

        task_bar = getattr(self.app, "_task_bar", None)
        if task_bar is None:
            task_list.mount(Static("No task bar available"))
            return

        queue = task_bar.queue
        active_list = queue.active_tasks
        queued = queue.queued_tasks
        history = queue.history

        if not active_list and not queued and not history:
            task_list.mount(Static("All quiet. No background tasks."))
            return

        for task in active_list:
            self._add_task_widget(task_list, task)

        for task in queued:
            self._add_task_widget(task_list, task)

        for task in reversed(history):
            self._add_task_widget(task_list, task)

    def _add_task_widget(self, container: VerticalScroll, task: Task) -> None:
        """Add a collapsible task widget with progress bar."""
        title = f"{task.name} ({task.progress}%)"

        collapsible = Collapsible(
            title=title,
            collapsed=True,
            id=f"task-{task.task_id}",
        )

        # Mount collapsible to container FIRST, then add children
        container.mount(collapsible)

        # Now add children to the already-mounted collapsible
        collapsible.mount(Static(f"Type: {task.task_type}"))
        collapsible.mount(Static(f"Status: {task.status.value}"))

        if task.detail:
            collapsible.mount(Static(f"Detail: {task.detail}"))

        if task.status == TaskStatus.ACTIVE:
            progress = ProgressBar(
                total=100,
                show_eta=False,
                id=f"task-progress-{task.task_id}",
            )
            progress.update(progress=task.progress)
            collapsible.mount(progress)
