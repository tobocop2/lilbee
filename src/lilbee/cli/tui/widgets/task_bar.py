"""TaskBar widget — unified panel showing active + queued background tasks."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ProgressBar, Static

from lilbee.cli.tui.task_queue import STATUS_ICONS, Task, TaskQueue, TaskStatus

log = logging.getLogger(__name__)

_DONE_FLASH_SECONDS = 1.0
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.1


class TaskBar(Static):
    """Docked panel showing active and queued background tasks.

    Auto-hides when no tasks are present. Max ~5 lines tall.
    """

    DEFAULT_CSS = """
    TaskBar {
        dock: bottom;
        height: auto;
        max-height: 5;
        padding: 0 1;
    }
    TaskBar .task-active-label {
        height: 1;
    }
    TaskBar ProgressBar {
        height: 1;
    }
    TaskBar .task-queued-label {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._queue = TaskQueue(on_change=self._on_queue_change)
        self._spinner_index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="task-bar-content"):
            yield Label("", id="task-active-label", classes="task-active-label")
            yield ProgressBar(total=100, show_eta=False, id="task-progress")
            yield Label("", id="task-queued-label", classes="task-queued-label")

    def on_mount(self) -> None:
        self._refresh_display()
        self.set_interval(_SPINNER_INTERVAL, self._tick_spinner)

    @property
    def queue(self) -> TaskQueue:
        """Expose the queue for external use."""
        return self._queue

    def add_task(self, name: str, task_type: str, fn: Callable[[], None] | None = None) -> str:
        """Add a task to the queue. Returns task_id.

        The *fn* is stored but not started here -- the caller is responsible
        for running the work (typically via @work) and calling update/complete.
        """
        task_id = self._queue.enqueue(fn or (lambda: None), name, task_type)
        self._refresh_display()
        return task_id

    def update_task(self, task_id: str, progress: int, detail: str = "") -> None:
        """Update progress (0-100) and optional detail text."""
        self._queue.update_task(task_id, progress, detail)
        self._refresh_display()

    def complete_task(self, task_id: str) -> None:
        """Mark task done, show brief 'done' flash, then remove."""
        task = self._queue.get_task(task_id)
        task_type = task.task_type if task else None
        self._queue.complete_task(task_id)
        self._refresh_display()
        self.set_timer(_DONE_FLASH_SECONDS, lambda: self._remove_and_advance(task_id, task_type))

    def fail_task(self, task_id: str, detail: str = "") -> None:
        """Mark task failed, show briefly, then remove."""
        task = self._queue.get_task(task_id)
        task_type = task.task_type if task else None
        self._queue.fail_task(task_id, detail)
        self._refresh_display()
        self.set_timer(_DONE_FLASH_SECONDS, lambda: self._remove_and_advance(task_id, task_type))

    def cancel_task(self, task_id: str) -> None:
        """Cancel and remove a task."""
        task = self._queue.get_task(task_id)
        task_type = task.task_type if task else None
        self._queue.cancel(task_id)
        self._queue.remove_task(task_id)
        self._try_advance(task_type)
        self._refresh_display()

    def _remove_and_advance(self, task_id: str, task_type: str | None) -> None:
        self._queue.remove_task(task_id)
        self._try_advance(task_type)
        self._refresh_display()

    def _try_advance(self, task_type: str | None = None) -> None:
        """Advance the queue for a specific type, then try all other types too."""
        if task_type:
            self._queue.advance(task_type)
        # Also advance any other types that may have pending tasks
        while self._queue.advance() is not None:
            pass

    def _tick_spinner(self) -> None:
        """Advance the spinner frame and refresh if there are active tasks."""
        if self._queue.active_tasks:
            self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
            self._refresh_display()

    def _on_queue_change(self) -> None:
        """Called by TaskQueue when state changes (may be from worker thread)."""
        with contextlib.suppress(Exception):
            self.app.call_from_thread(self._refresh_display)

    def _refresh_display(self) -> None:
        """Update widget contents based on queue state."""
        active_list = self._queue.active_tasks
        queued = self._queue.queued_tasks

        if not active_list and not queued:
            self.display = False
            return

        self.display = True

        active_label = self.query_one("#task-active-label", Label)
        progress_bar = self.query_one("#task-progress", ProgressBar)
        queued_label = self.query_one("#task-queued-label", Label)

        if active_list:
            primary = active_list[0]
            self._render_active_task(primary, active_label, progress_bar)
        else:
            active_label.display = False
            progress_bar.display = False

        if queued:
            names = ", ".join(f"{t.name} ({t.task_type})" for t in queued[:3])
            suffix = f" +{len(queued) - 3} more" if len(queued) > 3 else ""
            queued_label.update(f"   Queued: {names}{suffix}")
            queued_label.display = True
        else:
            queued_label.display = False

        self.refresh()

    def _render_active_task(self, task: Task, label: Label, progress_bar: ProgressBar) -> None:
        """Render a single active task into the label and progress bar."""
        if task.status == TaskStatus.ACTIVE:
            icon = _SPINNER_FRAMES[self._spinner_index]
        else:
            icon = self._status_icon(task.status)
        detail = f"  {task.detail}" if task.detail else ""
        label.update(f" {icon} {task.name}{detail}")
        label.display = True
        progress_bar.update(total=100, progress=task.progress)
        progress_bar.display = True

    @staticmethod
    def _status_icon(status: TaskStatus) -> str:
        return STATUS_ICONS.get(status, "▸")
