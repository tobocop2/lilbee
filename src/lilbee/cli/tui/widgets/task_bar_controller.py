"""TaskBarController and the per-task ProgressReporter."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from textual.app import App

from lilbee.catalog.formatting import download_task_name
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus, TaskType
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.crawler import bootstrap_chromium, chromium_installed
from lilbee.runtime import asyncio_loop
from lilbee.runtime.cancellation import TaskCancelledError
from lilbee.runtime.progress import EventType, SetupProgressEvent

if TYPE_CHECKING:
    from lilbee.catalog import CatalogModel

log = logging.getLogger(__name__)

_DOWNLOAD_CONCURRENCY = 2
_BYTES_PER_MB = 1024 * 1024


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
        """Raise ``TaskCancelledError`` if the task was cancelled from the UI."""
        if self.cancelled:
            raise TaskCancelledError

    def update(
        self, progress: float, detail: str = "", *, indeterminate: bool | None = None
    ) -> None:
        """Write a progress snapshot to the shared queue.

        Raises ``TaskCancelledError`` first if the UI cancelled the task, so
        callers can use ``update`` as both a progress write and a cancel
        checkpoint.
        """
        self.check_cancelled()
        self._controller.queue.update_task(
            self._task_id, progress, detail, indeterminate=indeterminate
        )


TaskTarget = Callable[[ProgressReporter], None]


def _chromium_bootstrap_target(reporter: ProgressReporter) -> None:
    """Worker target for the SETUP task: run bootstrap_chromium with progress forwarding.

    Module-level so ``TaskBarController.ensure_chromium`` stays short and
    tests can stub the target in isolation.
    """

    def _forward(event_type: EventType, data: Any) -> None:
        if event_type != EventType.SETUP_PROGRESS:
            return
        if not isinstance(data, SetupProgressEvent):
            return
        total = data.total_bytes or 0
        pct = int(data.downloaded_bytes * 100 / total) if total > 0 else 0
        mb = data.downloaded_bytes // _BYTES_PER_MB
        if total > 0:
            detail = msg.SETUP_CHROMIUM_DETAIL.format(done=mb, total=total // _BYTES_PER_MB)
        else:
            detail = msg.SETUP_CHROMIUM_DETAIL_UNKNOWN.format(done=mb)
        reporter.update(pct, detail)

    asyncio_loop.run(bootstrap_chromium(on_progress=_forward))


class TaskBarController:
    """App-level owner of the shared TaskQueue + all long-running work.

    The controller is attached as ``app.task_bar`` during
    ``LilbeeApp.__init__``. All task lifecycle methods
    (add/update/complete/fail/cancel) go through here so every ``TaskBar``
    widget sees the same state, and every long-running op is spawned by
    this controller: never by a screen that may dismiss mid-flight.
    """

    def __init__(self, app: App[Any]) -> None:
        self.app = app
        self.queue = TaskQueue(capacity={TaskType.DOWNLOAD.value: _DOWNLOAD_CONCURRENCY})
        # task_id -> (target, on_success). Worker looks up its target here
        # so we don't capture in a closure that outlives the task.
        self._task_targets: dict[str, tuple[TaskTarget, Callable[[], None] | None]] = {}
        # Number of files in documents/ that are out of date with the store.
        # Set by start_detect_pending; read by TaskBar to render the
        # "N docs to sync · S to sync" hint when no live tasks are running.
        # Atomic int writes are safe under the GIL; the bar polls at 10 Hz.
        self.pending_sync_count: int = 0
        self._detect_thread: threading.Thread | None = None
        # Roles whose worker is currently in the spawn window (1-3 s cold
        # start). Surfaced as a single TaskBar hint instead of one toast
        # per role so the chat screen isn't drowned in implementation
        # detail on first prompt.
        self.spawning_roles: set[str] = set()

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
        """Mark a task done. Row lingers in history until the user clears it."""
        task_type = self._task_type_of(task_id)
        self.queue.complete_task(task_id)
        self._after_done_hooks(task_type)
        self._advance_all(task_type)

    def fail_task(self, task_id: str, detail: str = "") -> None:
        """Mark a task failed. Row lingers in history until the user clears it."""
        self.queue.fail_task(task_id, detail)
        self._advance_all(self._task_type_of(task_id))

    def cancel_task(self, task_id: str) -> None:
        """Mark a task cancelled. Row lingers in history until the user clears it."""
        task_type = self._task_type_of(task_id)
        self.queue.cancel(task_id)
        self._advance_all(task_type)

    def _after_done_hooks(self, task_type: str | None) -> None:
        """Side effects triggered by a DONE completion.

        Callable from both the direct ``complete_task`` convenience and
        the worker-thread ``_finalize_task`` path so every success route
        stays in sync. Does NOT advance the queue; each caller picks the
        advance strategy that fits its context (``_advance_all`` vs
        ``_try_start_next``).
        """
        if task_type == TaskType.DOWNLOAD.value:
            self._notify_model_installed()

    def _task_type_of(self, task_id: str) -> str | None:
        task = self.queue.get_task(task_id)
        return task.task_type if task else None

    def _advance_all(self, task_type: str | None) -> None:
        """Try to advance the freed type first, then any other idle type."""
        if task_type:
            self.queue.advance(task_type)
        while self.queue.advance() is not None:
            pass

    def downloading_label_for(self, ref: str) -> str | None:
        """Return the task name if *ref*'s download is queued or active, else None.

        ``ref`` is a model reference (catalog repo id or native GGUF
        ref); the helper maps it to the canonical
        :attr:`CatalogModel.display_name` and matches against
        in-flight DOWNLOAD tasks. The returned label is suitable for
        embedding in a user-facing toast.
        """
        label = download_task_name(ref)
        if not label:
            return None
        for task in self.queue.active_tasks + self.queue.queued_tasks:
            if task.task_type == TaskType.DOWNLOAD.value and task.name == label:
                return task.name
        return None

    def set_pending_sync(self, count: int) -> None:
        """Update the pending-sync count surfaced in the TaskBar hint."""
        self.pending_sync_count = max(count, 0)

    def clear_pending_sync(self) -> None:
        """Drop the pending hint. Called when sync starts so the bar shows live progress instead."""
        self.pending_sync_count = 0

    def mark_role_spawning(self, role: str) -> None:
        """Add *role* to the set of workers whose pool process is starting."""
        self.spawning_roles.add(role)

    def mark_role_spawned(self, role: str) -> None:
        """Drop *role* from the spawn-in-progress set; harmless if already absent."""
        self.spawning_roles.discard(role)

    def start_detect_pending(self) -> None:
        """Run the cheap sync-detection (filesystem walk + hash compare) on a daemon thread.

        Writes the result via ``set_pending_sync``. No-op if a detect job
        is already running. Errors are logged and silently swallowed: a
        failed detect just leaves the previous count in place rather
        than blocking the UI.
        """
        if self._detect_thread is not None and self._detect_thread.is_alive():
            return
        thread = threading.Thread(
            target=self._run_detect_pending, daemon=True, name="detect-pending"
        )
        self._detect_thread = thread
        thread.start()

    def _run_detect_pending(self) -> None:
        # Local import: lilbee.data.ingest pulls in lancedb + the embedder
        # transitively; the TUI shouldn't pay for that just to import the
        # task bar widget.
        from lilbee.data.ingest import detect_pending

        try:
            count = detect_pending()
        except Exception:
            log.warning("detect_pending failed", exc_info=True)
            return
        self.set_pending_sync(count)

    def ensure_chromium(self, on_ready: Callable[[], None]) -> None:
        """Kick off a Chromium bootstrap if missing, then call ``on_ready``.

        If Chromium is already installed, ``on_ready`` runs immediately on
        the caller's thread. Otherwise a single SETUP task is enqueued
        that runs ``bootstrap_chromium``; on success the controller
        invokes ``on_ready`` on the worker thread via the task's
        ``on_success`` hook. On failure the SETUP task surfaces as FAILED
        and ``on_ready`` is NOT called (the follow-up work shouldn't
        proceed against a missing browser).

        bb-wq8g: the on_ready hook is how callers like ``_do_crawl`` chain
        their real work behind the one-time bootstrap.
        """
        if chromium_installed():
            on_ready()
            return

        self.start_task(
            msg.SETUP_CHROMIUM_NAME,
            TaskType.SETUP,
            _chromium_bootstrap_target,
            indeterminate=False,
            on_success=on_ready,
        )

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
        thread. On ``TaskCancelledError`` the task is marked CANCELLED. On any
        other exception the task is marked FAILED with ``str(exc)`` as
        detail. Rows linger in the Task Center under their final status
        until the user presses capital ``C`` to clear; the bottom bar
        flashes the outcome once and then hides when idle.

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
        except TaskCancelledError:
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
        targets ``self.app``: the App is long-lived; screens are not.
        """
        call_from_thread(self.app, self._finalize_task, task_id, outcome, detail, task_type)

    def _finalize_task(
        self, task_id: str, outcome: TaskOutcome, detail: str, task_type: str | None
    ) -> None:
        """Mark the queue state, refresh dependents, promote next queued task.

        Runs on the main thread. Atomically: free the active slot, notify
        anything downstream that needs a repaint (e.g. model dropdowns
        after a download lands), and advance the queue. Rows stay in
        history; the bottom bar flash expires on its own. Users clear
        finished rows from the Task Center manually.
        """
        if outcome is TaskOutcome.DONE:
            self.queue.complete_task(task_id)
            self._after_done_hooks(task_type)
        elif outcome is TaskOutcome.FAILED:
            self.queue.fail_task(task_id, detail)
        elif outcome is TaskOutcome.CANCELLED:
            self.queue.cancel(task_id)
        if task_type:
            self._try_start_next(task_type)

    def _notify_model_installed(self) -> None:
        """Refresh any ChatScreen's ModelBar so the new model is selectable.

        The dropdowns are built once on mount from the registry; without
        this nudge, a freshly-downloaded model only appears after the
        user reopens the screen. NoMatches and similar query errors are
        silently skipped so a transient "bar not mounted yet" doesn't
        crash the finalize path; anything else is logged so a real
        failure surfaces in debug output.
        """
        # Late import to avoid a circular (ChatScreen imports this module).
        from textual.css.query import QueryError

        from lilbee.cli.tui.screens.chat import ChatScreen

        for screen in self.app.screen_stack:
            # screen_stack is typed Screen[Any]; narrow at runtime to
            # locate the one screen that owns the ModelBar.
            if isinstance(screen, ChatScreen):
                try:
                    screen.refresh_model_bar()
                except QueryError:
                    log.debug("ModelBar not mounted yet; skipping refresh", exc_info=True)
                break

    def start_download(
        self, model: CatalogModel, *, on_success: Callable[[], None] | None = None
    ) -> str:
        """Enqueue a download; ``on_success`` runs on the worker thread once the file is on disk."""
        return self.start_task(
            model.display_name,
            TaskType.DOWNLOAD,
            lambda reporter: _download_target(reporter, model),
            on_success=on_success,
        )


def _download_target(reporter: ProgressReporter, model: CatalogModel) -> None:
    """``start_task`` target for a HuggingFace model download.

    Translates ``PermissionError`` into the gated-repo friendly message so
    every call site (wizard, catalog, chat) gets consistent error UX.
    """
    from lilbee.app.models import pull_model_data
    from lilbee.catalog import DownloadProgress
    from lilbee.catalog.types import ModelSource

    def _on_progress(p: DownloadProgress) -> None:
        reporter.update(p.percent, f"{model.display_name}: {p.detail}")

    try:
        pull_model_data(model.ref, ModelSource.NATIVE, on_update=_on_progress)
    except PermissionError as exc:
        raise RuntimeError(msg.CATALOG_GATED_REPO.format(name=model.display_name)) from exc
