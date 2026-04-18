"""TaskBar widget and controller.

The TaskBar is a slim 1-line status indicator docked at the bottom of every
screen. It shows a count of active/queued tasks and directs users to the
Task Center (``t``) for detailed progress. Full progress panels with spinners
and progress bars live only in the Task Center screen.

State ownership is split so the bar can render on every screen:

- ``TaskBarController`` lives on the app (``app.task_bar``) and owns the
  single ``TaskQueue``. Callers enqueue/update/complete/fail tasks through it.
- ``TaskBar`` is a stateless view widget composed by each Screen. It polls the
  shared queue at 10 Hz on the main event loop and re-renders in place; no
  thread marshaling or subscriber callbacks are involved in the render path.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.widgets import Label, Static

from lilbee.cli.tui.task_queue import TaskQueue

if TYPE_CHECKING:
    from textual.app import App

log = logging.getLogger(__name__)

_DONE_FLASH_SECONDS = 2.0
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_POLL_INTERVAL_SECONDS = 0.1


class TaskBarController:
    """App-level owner of the shared TaskQueue.

    The controller is attached as ``app.task_bar`` during ``LilbeeApp.__init__``.
    All task lifecycle methods (add/update/complete/fail/cancel) go through here
    so every ``TaskBar`` widget sees the same state.
    """

    def __init__(self, app: App[Any]) -> None:
        self.app = app
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
        progress: float,
        detail: str = "",
        *,
        indeterminate: bool | None = None,
    ) -> None:
        """Update progress and detail text for a task."""
        self.queue.update_task(task_id, progress, detail, indeterminate=indeterminate)

    def complete_task(self, task_id: str) -> None:
        """Mark a task done; keep it visible for a brief flash, then remove."""
        self.queue.complete_task(task_id)
        self.app.set_timer(_DONE_FLASH_SECONDS, lambda: self._dismiss(task_id))

    def fail_task(self, task_id: str, detail: str = "") -> None:
        """Mark a task as failed; flash, then remove."""
        self.queue.fail_task(task_id, detail)
        self.app.set_timer(_DONE_FLASH_SECONDS, lambda: self._dismiss(task_id))

    def cancel_task(self, task_id: str) -> None:
        self.queue.cancel(task_id)
        self._dismiss(task_id)

    def _dismiss(self, task_id: str) -> None:
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
    """Slim 1-line status indicator for background tasks.

    Shows a compact summary when tasks are active and hides when idle.
    Detailed progress (spinners, progress bars, task panels) lives in
    the Task Center screen, accessible via ``t``.
    """

    DEFAULT_CSS = """
    TaskBar {
        dock: bottom;
        height: 1;
        max-height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._spinner_index = 0

    def compose(self) -> ComposeResult:
        yield Label("", id="task-status-label")

    def on_mount(self) -> None:
        self._refresh_display()
        self.set_interval(_POLL_INTERVAL_SECONDS, self._tick)

    @property
    def _controller(self) -> TaskBarController:
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
        progress: float,
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

    def _tick(self) -> None:
        """Poll the shared queue at 10 Hz and re-render."""
        if self.queue.active_tasks:
            self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Rebuild the 1-line status label from the shared queue."""
        queue = self.queue
        active = queue.active_tasks
        queued = queue.queued_tasks

        if not active and not queued:
            self.display = False
            return

        self.display = True
        spinner = _SPINNER_FRAMES[self._spinner_index]
        parts: list[str] = []

        if active:
            count = len(active)
            task = active[0]
            pct = f" {task.progress:.1f}%" if not task.indeterminate else ""
            if count == 1:
                detail = f" {task.detail}" if task.detail else ""
                parts.append(f"{spinner} {task.name}{pct}{detail}")
            else:
                parts.append(f"{spinner} {count} tasks running")

        if queued:
            parts.append(f"{len(queued)} queued")

        summary = " | ".join(parts)
        from lilbee.cli.tui import messages as msg

        label_text = f" {summary}  {msg.TASKBAR_HINT}"

        with contextlib.suppress(Exception):
            label = self.query_one("#task-status-label", Label)
            label.update(label_text)
