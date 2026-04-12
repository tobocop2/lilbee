"""TaskBar widget and controller.

The TaskBar is a browser-style docked panel that shows per-task progress bars.
State ownership is split so the bar can render on every screen:

- `TaskBarController` lives on the app (`app.task_bar`) and owns the single
  `TaskQueue`. Callers enqueue/update/complete/fail tasks through it.
- `TaskBar` is a stateless view widget composed by each Screen. It subscribes
  to the shared queue and re-renders when the queue changes.

This lets progress stay visible as the user navigates between screens,
because each screen has its own `TaskBar` instance bound to the same queue.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Label, ProgressBar, Static

from lilbee.cli.tui.task_queue import STATUS_ICONS, Task, TaskQueue, TaskStatus

if TYPE_CHECKING:
    from textual.app import App

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
        self._last_progress: int = -1
        self._last_indeterminate: bool | None = None

    def compose(self) -> ComposeResult:
        yield Label("", classes="task-panel-label")
        yield ProgressBar(total=100, show_eta=False)


class TaskBarController:
    """App-level coordinator for background tasks.

    Owns the shared `TaskQueue` and schedules the brief post-completion flash
    period before finished tasks are removed. `TaskBar` view widgets subscribe
    to `queue.on_change` and re-render whenever this controller mutates state.
    """

    def __init__(self, app: App[None]) -> None:
        self._app = app
        self.queue = TaskQueue()

    def add_task(
        self,
        name: str,
        task_type: str,
        fn: Callable[[], None] | None = None,
        *,
        indeterminate: bool = False,
    ) -> str:
        """Enqueue a task. Returns the new task_id."""
        return self.queue.enqueue(
            fn or (lambda: None), name, task_type, indeterminate=indeterminate
        )

    def update_task(
        self,
        task_id: str,
        progress: int,
        detail: str = "",
        *,
        indeterminate: bool | None = None,
    ) -> None:
        """Update progress and detail text for a task.
        When *indeterminate* is True the panel renders a pulsing bar instead
        of a percentage, so long-running phases without a reliable percent
        don't falsely claim to be finished.
        """
        self.queue.update_task(task_id, progress, detail, indeterminate=indeterminate)

    def complete_task(self, task_id: str) -> None:
        """Mark a task done; keep it visible for a brief flash, then remove."""
        self.queue.complete_task(task_id)
        self._app.set_timer(_DONE_FLASH_SECONDS, lambda: self._dismiss(task_id))

    def fail_task(self, task_id: str, detail: str = "") -> None:
        """Mark a task failed; keep it visible for a brief flash, then remove."""
        self.queue.fail_task(task_id, detail)
        self._app.set_timer(_DONE_FLASH_SECONDS, lambda: self._dismiss(task_id))

    def cancel_task(self, task_id: str) -> None:
        """Cancel and immediately remove a task."""
        task = self.queue.get_task(task_id)
        task_type = task.task_type if task else None
        self.queue.cancel(task_id)
        self.queue.remove_task(task_id)
        self._advance_all(task_type)

    def _dismiss(self, task_id: str) -> None:
        """Final cleanup after the flash period: remove and advance the queue."""
        task = self.queue.get_task(task_id)
        task_type = task.task_type if task else None
        self.queue.remove_task(task_id)
        self._advance_all(task_type)

    def _advance_all(self, task_type: str | None) -> None:
        """Try to advance the freed type first, then any other idle type."""
        if task_type:
            self.queue.advance(task_type)
        while self.queue.advance() is not None:
            pass


class TaskBar(Static):
    """Docked view of the app's TaskBarController.

    Each Screen composes its own `TaskBar` instance. All instances subscribe
    to the app-level `TaskBarController.queue` and render its displayable
    tasks, so progress is visible regardless of which screen is on top.
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
        self._spinner_index = 0
        self._panels: dict[str, _TaskPanel] = {}

    def compose(self) -> ComposeResult:
        yield Vertical(id="task-bar-panels")
        yield Label("", id="task-queued-label", classes="task-queued-label")

    def on_mount(self) -> None:
        self.queue.subscribe(self._on_queue_change)
        self._refresh_display()
        self.set_interval(_SPINNER_INTERVAL, self._tick_spinner)

    def on_unmount(self) -> None:
        self.queue.unsubscribe(self._on_queue_change)

    @property
    def _controller(self) -> TaskBarController:
        """Return the app's TaskBarController, creating one if missing.
        LilbeeApp wires up a controller in its `__init__`. Bare test harnesses
        that instantiate a TaskBar on a plain `App` (without a controller) get
        one lazily attached so every screen in the app still shares state. In
        production this branch is a bug indicator: log a warning so accidental
        misuse surfaces instead of silently regressing the per-screen fix.
        """
        controller = getattr(self.app, "task_bar", None)
        if not isinstance(controller, TaskBarController):
            log.warning(
                "TaskBar mounted on %s without a TaskBarController; creating one lazily",
                type(self.app).__name__,
            )
            controller = TaskBarController(self.app)
            self.app.task_bar = controller  # type: ignore[attr-defined]
        return controller

    @property
    def queue(self) -> TaskQueue:
        """Expose the shared queue for callers that iterate or advance it."""
        return self._controller.queue

    def add_task(
        self,
        name: str,
        task_type: str,
        fn: Callable[[], None] | None = None,
        *,
        indeterminate: bool = False,
    ) -> str:
        """Enqueue a task via the app's controller. Returns the task_id."""
        return self._controller.add_task(name, task_type, fn, indeterminate=indeterminate)

    def update_task(
        self,
        task_id: str,
        progress: int,
        detail: str = "",
        *,
        indeterminate: bool | None = None,
    ) -> None:
        self._controller.update_task(task_id, progress, detail, indeterminate=indeterminate)

    def complete_task(self, task_id: str) -> None:
        self._controller.complete_task(task_id)

    def fail_task(self, task_id: str, detail: str = "") -> None:
        self._controller.fail_task(task_id, detail)

    def cancel_task(self, task_id: str) -> None:
        self._controller.cancel_task(task_id)

    @staticmethod
    def _status_icon(status: TaskStatus) -> str:
        return STATUS_ICONS.get(status, "▸")

    def _on_queue_change(self) -> None:
        """Queue callback. May fire on either the main or a worker thread."""
        if threading.current_thread() is threading.main_thread():
            with contextlib.suppress(Exception):
                self._refresh_display()
            return
        with contextlib.suppress(Exception):
            self.app.call_from_thread(self._refresh_display)

    def _tick_spinner(self) -> None:
        if self.queue.active_tasks:
            self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
            self._refresh_display()

    def _refresh_display(self) -> None:
        """Rebuild panels from the shared queue's displayable tasks."""
        queue = self.queue
        displayable = queue.displayable_tasks[:_MAX_VISIBLE_PANELS]
        queued = queue.queued_tasks

        if not displayable and not queued:
            self.display = False
            self._clear_panels()
            return

        self.display = True

        visible_ids = {task.task_id for task in displayable}
        for tid in list(self._panels.keys()):
            if tid not in visible_ids:
                panel = self._panels.pop(tid)
                panel.remove()

        for task in displayable:
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

    def _clear_panels(self) -> None:
        for panel in self._panels.values():
            panel.remove()
        self._panels.clear()

    def _ensure_panel(self, task_id: str) -> _TaskPanel:
        if task_id not in self._panels:
            panel = _TaskPanel(task_id)
            self._panels[task_id] = panel
            container = self.query_one("#task-bar-panels", Vertical)
            container.mount(panel)
        return self._panels[task_id]

    def _render_task_panel(self, task_id: str, task: Task | None) -> None:
        panel = self._panels.get(task_id)
        if not panel or task is None:
            return

        panel.set_class(task.status == TaskStatus.DONE, "task-done")
        panel.set_class(task.status == TaskStatus.FAILED, "task-failed")

        if task.status == TaskStatus.ACTIVE:
            icon = _SPINNER_FRAMES[self._spinner_index]
        else:
            icon = self._status_icon(task.status)

        detail = f"  {task.detail}" if task.detail else ""
        try:
            label = panel.query_one(".task-panel-label", Label)
            label.update(f" {icon} {task.name}{detail}")
            is_indeterminate = task.indeterminate and task.status == TaskStatus.ACTIVE
            progress_changed = (
                is_indeterminate != panel._last_indeterminate
                or task.progress != panel._last_progress
            )
            if progress_changed:
                progress_bar = panel.query_one(ProgressBar)
                if is_indeterminate:
                    progress_bar.update(total=None, progress=0)
                else:
                    progress_bar.update(total=100, progress=task.progress)
                panel._last_progress = task.progress
                panel._last_indeterminate = is_indeterminate
        except NoMatches:
            pass  # panel children not yet composed, next refresh will catch up
