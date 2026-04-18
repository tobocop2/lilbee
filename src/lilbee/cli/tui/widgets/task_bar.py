"""TaskBar widget and controller.

The TaskBar is a slim 1-line status indicator docked at the bottom of every
screen. It shows a count of active/queued tasks and directs users to the
Task Center (``t``) for detailed progress. Full progress panels with spinners
and progress bars live only in the Task Center screen.

State ownership is split so the bar can render on every screen:

- ``TaskBarController`` lives on the app (``app.task_bar``) and owns the
  single ``TaskQueue``. Every long-running operation in the app should be
  submitted to the controller via ``start_task`` (or the typed
  ``start_download`` specialization) so it survives any screen navigation.
- ``TaskBar`` is a stateless view widget composed by each Screen. It polls the
  shared queue at 10 Hz on the main event loop and re-renders in place; no
  thread marshaling or subscriber callbacks are involved in the render path.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.widgets import Label, Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus, TaskType
from lilbee.cli.tui.thread_safe import call_from_thread

if TYPE_CHECKING:
    from textual.app import App

    from lilbee.catalog import CatalogModel

log = logging.getLogger(__name__)

_DONE_FLASH_SECONDS = 2.0
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_POLL_INTERVAL_SECONDS = 0.1
_DOWNLOAD_CONCURRENCY = 2


class TaskCancelled(Exception):
    """Raised inside a ``ProgressReporter.update`` call to abort the task.

    The worker's target function receives a ``ProgressReporter`` whose
    ``check_cancelled()`` / ``update()`` both raise ``TaskCancelled`` when
    the task has been cancelled from the UI. The worker can let the
    exception propagate and the controller handles it uniformly.
    """


# Back-compat alias: legacy tests/imports still reference this name.
_DownloadCancelled = TaskCancelled


class TaskOutcome(StrEnum):
    """How a task terminated. Passed from worker thread to finalizer."""

    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressReporter:
    """Thread-safe handle a worker uses to report progress and check cancellation.

    The worker only sees this object; it never touches ``self.app``,
    ``call_from_thread``, or any screen. Writes to the lock-protected
    ``TaskQueue`` so updates survive any UI navigation.
    """

    def __init__(self, controller: TaskBarController, task_id: str) -> None:
        self._controller = controller
        self._task_id = task_id

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def cancelled(self) -> bool:
        task = self._controller.queue.get_task(self._task_id)
        return task is not None and task.status == TaskStatus.CANCELLED

    def check_cancelled(self) -> None:
        """Raise ``TaskCancelled`` if the task was cancelled from the UI."""
        if self.cancelled:
            raise TaskCancelled

    def update(
        self, progress: float, detail: str = "", *, indeterminate: bool | None = None
    ) -> None:
        """Write a progress snapshot to the shared queue.

        Raises ``TaskCancelled`` first if the UI cancelled the task, so
        callers can use ``update`` as both a progress write and a cancel
        checkpoint.
        """
        self.check_cancelled()
        self._controller.queue.update_task(
            self._task_id, progress, detail, indeterminate=indeterminate
        )


TaskTarget = Callable[[ProgressReporter], None]


class TaskBarController:
    """App-level owner of the shared TaskQueue + all long-running work.

    The controller is attached as ``app.task_bar`` during
    ``LilbeeApp.__init__``. All task lifecycle methods
    (add/update/complete/fail/cancel) go through here so every ``TaskBar``
    widget sees the same state, and every long-running op is spawned by
    this controller — never by a screen that may dismiss mid-flight.
    """

    def __init__(self, app: App[Any]) -> None:
        self.app = app
        self.queue = TaskQueue(capacity={TaskType.DOWNLOAD.value: _DOWNLOAD_CONCURRENCY})
        # task_id -> (target, on_success). Worker looks up its target here
        # so we don't capture in a closure that outlives the task.
        self._task_targets: dict[str, tuple[TaskTarget, Callable[[], None] | None]] = {}

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

    def start_task(
        self,
        name: str,
        task_type: TaskType,
        target: TaskTarget,
        *,
        indeterminate: bool = False,
        on_success: Callable[[], None] | None = None,
    ) -> str:
        """Enqueue a task, spawn its worker, return task_id.

        The *target* receives a ``ProgressReporter`` as its only argument.
        It should periodically call ``reporter.update(percent, detail)`` and
        may call ``reporter.check_cancelled()`` to cooperatively abort.

        On success (target returns normally) the queue marks the task DONE
        and ``on_success`` (if provided) runs after on the same worker
        thread. On ``TaskCancelled`` the task is marked CANCELLED. On any
        other exception the task is marked FAILED with ``str(exc)`` as
        detail. In all cases the row is dismissed after a 2-second flash.

        Per-type capacity in ``TaskQueue`` (download=2, everything else=1)
        controls concurrency: a second sync queues behind the first, but a
        third download waits until one of the two active downloads finishes.
        """
        task_id = self.queue.enqueue(
            lambda: None, name, task_type.value, indeterminate=indeterminate
        )
        self._task_targets[task_id] = (target, on_success)
        self._try_start_next(task_type.value)
        return task_id

    def _try_start_next(self, task_type: str) -> None:
        """Promote queued tasks of this type into any free capacity slots."""
        while (task := self.queue.advance(task_type)) is not None:
            self._spawn_task_worker(task.task_id)

    def _spawn_task_worker(self, task_id: str) -> None:
        """Start a daemon thread for the task. Safe to call from any thread."""
        if task_id not in self._task_targets:
            return
        thread = threading.Thread(
            target=self._run_task_worker,
            args=(task_id,),
            daemon=True,
            name=f"task-{task_id}",
        )
        thread.start()

    def _run_task_worker(self, task_id: str) -> None:
        """Body of the daemon worker thread."""
        entry = self._task_targets.get(task_id)
        if entry is None:
            return
        target, on_success = entry
        task = self.queue.get_task(task_id)
        task_type = task.task_type if task is not None else None
        reporter = ProgressReporter(self, task_id)
        try:
            target(reporter)
        except TaskCancelled:
            log.info("Task %s cancelled", task_id)
            self._post_finalize(task_id, TaskOutcome.CANCELLED, "", task_type)
        except Exception as exc:
            log.warning("Task %s failed: %s", task_id, exc)
            self._post_finalize(task_id, TaskOutcome.FAILED, str(exc), task_type)
        else:
            self._post_finalize(task_id, TaskOutcome.DONE, "", task_type)
            if on_success is not None:
                try:
                    on_success()
                except Exception:
                    log.warning("on_success for %s raised", task_id, exc_info=True)
        finally:
            self._task_targets.pop(task_id, None)

    def _post_finalize(
        self, task_id: str, outcome: TaskOutcome, detail: str, task_type: str | None
    ) -> None:
        """Marshal finalization back to the main thread.

        Main-thread execution matters because ``set_timer`` (used for the
        flash-then-remove cycle) isn't safe from workers. ``call_from_thread``
        targets ``self.app`` — the App is long-lived; screens are not.
        """
        call_from_thread(self.app, self._finalize_task, task_id, outcome, detail, task_type)

    def _finalize_task(
        self, task_id: str, outcome: TaskOutcome, detail: str, task_type: str | None
    ) -> None:
        """Mark the queue state, schedule the flash, promote next queued task.

        Runs on the main thread. Atomically: we free the active slot, set
        a 2 s flash timer, then advance the queue so any pending task of
        the same type promotes immediately.
        """
        if outcome is TaskOutcome.DONE:
            self.queue.complete_task(task_id)
        elif outcome is TaskOutcome.FAILED:
            self.queue.fail_task(task_id, detail)
        elif outcome is TaskOutcome.CANCELLED:
            self.queue.cancel(task_id)
        self.app.set_timer(_DONE_FLASH_SECONDS, lambda: self._dismiss(task_id))
        if task_type:
            self._try_start_next(task_type)

    def start_download(self, model: CatalogModel) -> str:
        """Enqueue a model download and spawn a background worker.

        Thin specialization of ``start_task`` that wires the HuggingFace
        ``download_model`` API and translates ``PermissionError`` into a
        friendly "repo requires login" message — gated repos are a common
        failure mode and the raw exception text is opaque.
        """
        return self.start_task(
            model.display_name,
            TaskType.DOWNLOAD,
            lambda reporter: _download_target(reporter, model),
        )


def _download_target(reporter: ProgressReporter, model: CatalogModel) -> None:
    """``start_task`` target for a HuggingFace model download.

    Kept at module scope (not as a controller method) so it can be unit-
    tested without spinning up a controller. Translates
    ``PermissionError`` into the gated-repo friendly message so every call
    site (wizard, catalog, chat) gets consistent error UX.
    """
    from lilbee.catalog import DownloadProgress, download_model, make_download_callback

    def _on_progress(p: DownloadProgress) -> None:
        reporter.update(p.percent, f"{model.display_name}: {p.detail}")

    callback = make_download_callback(_on_progress)
    try:
        download_model(model, on_progress=callback)
    except PermissionError as exc:
        raise RuntimeError(msg.CATALOG_GATED_REPO.format(name=model.display_name)) from exc


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
        label_text = f" {summary}  {msg.TASKBAR_HINT}"

        with contextlib.suppress(Exception):
            label = self.query_one("#task-status-label", Label)
            label.update(label_text)
