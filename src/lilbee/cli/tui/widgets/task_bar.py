"""TaskBar widget — unified panel showing active + queued background tasks."""

from __future__ import annotations

import contextlib
import logging

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ProgressBar, Static

from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

log = logging.getLogger(__name__)

_DONE_FLASH_SECONDS = 1.0
_SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
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

    def add_task(self, name: str, task_type: str, fn: object = None) -> str:
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
        self._queue.complete_task(task_id)
        self._refresh_display()
        self.set_timer(_DONE_FLASH_SECONDS, lambda: self._remove_and_advance(task_id))

    def fail_task(self, task_id: str, detail: str = "") -> None:
        """Mark task failed, show briefly, then remove."""
        self._queue.fail_task(task_id, detail)
        self._refresh_display()
        self.set_timer(_DONE_FLASH_SECONDS, lambda: self._remove_and_advance(task_id))

    def cancel_task(self, task_id: str) -> None:
        """Cancel and remove a task."""
        self._queue.cancel(task_id)
        self._queue.remove_task(task_id)
        self._try_advance()
        self._refresh_display()

    def _remove_and_advance(self, task_id: str) -> None:
        self._queue.remove_task(task_id)
        self._try_advance()
        self._refresh_display()

    def _try_advance(self) -> None:
        """If no active task, advance the queue to the next item."""
        task = self._queue.advance()
        if task:
            self._refresh_display()

    def _tick_spinner(self) -> None:
        """Advance the spinner frame and refresh if there's an active task."""
        if self._queue.active_task is not None:
            self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
            self._refresh_display()

    def _on_queue_change(self) -> None:
        """Called by TaskQueue when state changes (may be from worker thread)."""
        with contextlib.suppress(Exception):
            self.app.call_from_thread(self._refresh_display)

    def _refresh_display(self) -> None:
        """Update widget contents based on queue state."""
        active = self._queue.active_task
        queued = self._queue.queued_tasks
        total_tasks = len(queued) + (1 if active else 0)

        if not active and not queued:
            self.display = False
            return

        self.display = True

        active_label = self.query_one("#task-active-label", Label)
        progress_bar = self.query_one("#task-progress", ProgressBar)
        queued_label = self.query_one("#task-queued-label", Label)

        if active:
            if active.status == TaskStatus.ACTIVE:
                spinner = _SPINNER_FRAMES[self._spinner_index]
                type_icon = self._task_type_icon(active.task_type)
                icon = f"{spinner} {type_icon}"
            else:
                icon = self._status_icon(active.status)
            detail = f"  {active.detail}" if active.detail else ""
            active_label.update(f" {icon} {active.name}{detail}")
            active_label.display = True
            progress_bar.update(total=100, progress=active.progress)
            progress_bar.display = True
        else:
            active_label.display = False
            progress_bar.display = False

        if queued:
            parts = []
            for t in queued[:3]:
                type_icon = self._task_type_icon(t.task_type)
                parts.append(f"{type_icon} {t.name}")
            names = ", ".join(parts)
            suffix = f" +{len(queued) - 3} more" if len(queued) > 3 else ""
            badge = f" [{total_tasks} tasks]" if total_tasks > 1 else ""
            queued_label.update(f"   Queued{badge}: {names}{suffix}")
            queued_label.display = True
        else:
            queued_label.display = False

    @staticmethod
    def _task_type_icon(task_type: str) -> str:
        """Get icon for task type."""
        icons = {
            "download": "\u2b73",
            "sync": "\u21bb",
            "crawl": "\u1f310",
            "add": "\u2795",
        }
        return icons.get(task_type, "\u25b6")

    @staticmethod
    def _status_icon(status: TaskStatus) -> str:
        icons = {
            TaskStatus.ACTIVE: "\u25b8",
            TaskStatus.DONE: "\u2713",
            TaskStatus.FAILED: "\u2717",
            TaskStatus.CANCELLED: "\u2212",
            TaskStatus.QUEUED: "\u2022",
        }
        return icons.get(status, "\u25b8")
