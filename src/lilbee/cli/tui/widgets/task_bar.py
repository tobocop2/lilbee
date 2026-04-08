"""TaskBar widget — browser-style download panels with per-task progress bars.

Each active/completing task gets its own row with a label and progress bar.
Panels appear when tasks start and disappear shortly after completion.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Label, ProgressBar, Static

from lilbee.cli.tui.task_queue import STATUS_ICONS, Task, TaskQueue, TaskStatus

log = logging.getLogger(__name__)

_DONE_FLASH_SECONDS = 1.0
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.1
_MAX_VISIBLE_PANELS = 5


class _TaskPanel(Static):
    """A single task's label + progress bar row."""

    DEFAULT_CSS = """
    _TaskPanel {
        height: auto;
        max-height: 3;
        padding: 0 1;
    }
    _TaskPanel .task-panel-label {
        height: 1;
    }
    _TaskPanel ProgressBar {
        height: 1;
    }
    _TaskPanel.task-done .task-panel-label {
        color: $success;
    }
    _TaskPanel.task-failed .task-panel-label {
        color: $error;
    }
    """

    def __init__(self, task_id: str, **kwargs: object) -> None:
        super().__init__(id=f"task-panel-{task_id}", **kwargs)  # type: ignore[arg-type]
        self.task_id = task_id

    def compose(self) -> ComposeResult:
        yield Label("", classes="task-panel-label")
        yield ProgressBar(total=100, show_eta=False)


class TaskBar(Static):
    """Docked panel showing browser-style download progress bars.

    Each task gets its own panel with a label and progress bar.
    Panels auto-hide when complete. Auto-hides when no tasks are present.
    """

    DEFAULT_CSS = """
    TaskBar {
        dock: bottom;
        height: auto;
        max-height: 20;
        padding: 0;
    }
    TaskBar .task-queued-label {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._queue = TaskQueue(on_change=self._on_queue_change)
        self._spinner_index = 0
        self._panels: dict[str, _TaskPanel] = {}

    def compose(self) -> ComposeResult:
        yield Vertical(id="task-bar-panels")
        yield Label("", id="task-queued-label", classes="task-queued-label")

    def on_mount(self) -> None:
        self._refresh_display()
        self.set_interval(_SPINNER_INTERVAL, self._tick_spinner)

    @property
    def queue(self) -> TaskQueue:
        """Expose the queue for external use."""
        return self._queue

    def add_task(self, name: str, task_type: str, fn: Callable[[], None] | None = None) -> str:
        """Add a task to the queue. Returns task_id."""
        task_id = self._queue.enqueue(fn or (lambda: None), name, task_type)
        self._refresh_display()
        return task_id

    def update_task(self, task_id: str, progress: int, detail: str = "") -> None:
        """Update progress (0-100) and optional detail text."""
        self._queue.update_task(task_id, progress, detail)
        self._refresh_display()

    def complete_task(self, task_id: str) -> None:
        """Mark task done, flash briefly, then remove panel."""
        task = self._queue.get_task(task_id)
        task_type = task.task_type if task else None
        self._queue.complete_task(task_id)
        panel = self._panels.get(task_id)
        if panel:
            panel.add_class("task-done")
            self._render_task_panel(task_id, task)
        self._refresh_display()
        self.set_timer(_DONE_FLASH_SECONDS, lambda: self._dismiss_panel(task_id, task_type))

    def fail_task(self, task_id: str, detail: str = "") -> None:
        """Mark task failed, flash briefly, then remove panel."""
        task = self._queue.get_task(task_id)
        task_type = task.task_type if task else None
        self._queue.fail_task(task_id, detail)
        panel = self._panels.get(task_id)
        if panel:
            panel.add_class("task-failed")
            self._render_task_panel(task_id, task)
        self._refresh_display()
        self.set_timer(_DONE_FLASH_SECONDS, lambda: self._dismiss_panel(task_id, task_type))

    def cancel_task(self, task_id: str) -> None:
        """Cancel and immediately remove a task panel."""
        task = self._queue.get_task(task_id)
        task_type = task.task_type if task else None
        self._queue.cancel(task_id)
        self._queue.remove_task(task_id)
        self._remove_panel(task_id)
        self._try_advance(task_type)
        self._refresh_display()

    def _dismiss_panel(self, task_id: str, task_type: str | None) -> None:
        """Remove a completed/failed panel after the flash period."""
        self._queue.remove_task(task_id)
        self._remove_panel(task_id)
        self._try_advance(task_type)
        self._refresh_display()

    def _remove_panel(self, task_id: str) -> None:
        """Unmount and forget a task panel."""
        panel = self._panels.pop(task_id, None)
        if panel:
            panel.remove()

    def _ensure_panel(self, task_id: str) -> _TaskPanel:
        """Get or create a panel for a task."""
        if task_id not in self._panels:
            panel = _TaskPanel(task_id)
            self._panels[task_id] = panel
            container = self.query_one("#task-bar-panels", Vertical)
            container.mount(panel)
        return self._panels[task_id]

    def _try_advance(self, task_type: str | None = None) -> None:
        """Advance the queue for a specific type, then try all other types too."""
        if task_type:
            self._queue.advance(task_type)
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
        """Update all task panels based on current queue state."""
        active_list = self._queue.active_tasks
        queued = self._queue.queued_tasks

        if not active_list and not queued and not self._panels:
            self.display = False
            return

        self.display = True

        for task in active_list[:_MAX_VISIBLE_PANELS]:
            self._ensure_panel(task.task_id)
            self._render_task_panel(task.task_id, task)

        queued_label = self.query_one("#task-queued-label", Label)
        if queued:
            names = ", ".join(t.name for t in queued[:3])
            suffix = f" +{len(queued) - 3} more" if len(queued) > 3 else ""
            queued_label.update(f"   Queued: {names}{suffix}")
            queued_label.display = True
        else:
            queued_label.display = False

        self.refresh()

    def _render_task_panel(self, task_id: str, task: Task | None) -> None:
        """Render a task's current state into its panel."""
        panel = self._panels.get(task_id)
        if not panel or not task:
            return

        if task.status == TaskStatus.ACTIVE:
            icon = _SPINNER_FRAMES[self._spinner_index]
        else:
            icon = self._status_icon(task.status)

        detail = f"  {task.detail}" if task.detail else ""
        try:
            label = panel.query_one(".task-panel-label", Label)
            label.update(f" {icon} {task.name}{detail}")
            progress_bar = panel.query_one(ProgressBar)
            progress_bar.update(total=100, progress=task.progress)
        except NoMatches:
            pass  # panel children not yet composed — next refresh will catch up

    @staticmethod
    def _status_icon(status: TaskStatus) -> str:
        return STATUS_ICONS.get(status, "▸")
