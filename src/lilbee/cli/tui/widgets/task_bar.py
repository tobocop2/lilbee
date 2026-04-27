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
from textual.timer import Timer
from textual.widgets import Label, Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus, TaskType
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.crawler import bootstrap_chromium, chromium_installed
from lilbee.runtime import asyncio_loop
from lilbee.runtime.cancellation import TaskCancelled
from lilbee.runtime.progress import EventType, SetupProgressEvent

if TYPE_CHECKING:
    from textual.app import App

    from lilbee.catalog import CatalogModel

log = logging.getLogger(__name__)

_DONE_FLASH_SECONDS = 2.0
_POLL_INTERVAL_SECONDS = 0.1
_DOWNLOAD_CONCURRENCY = 2

# Pulsing-dot cadence: on/off flip at half of this tick count.
# 10 Hz poll x 5 = 500 ms per half cycle, which is a 1 Hz dot pulse,
# matching the active-row rail pulse in the Task Center.
_DOT_PULSE_HALF_TICKS = 5
_DOT_GLYPH = "●"


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


_BYTES_PER_MB = 1024 * 1024


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
        thread. On ``TaskCancelled`` the task is marked CANCELLED. On any
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

    # NOTE: no ``dock: bottom`` here. TaskBar is always mounted inside a
    # ``BottomBars`` container that owns the dock; multiple dock-bottom
    # siblings overlap at the same row in Textual (see BottomBars docstring).
    DEFAULT_CSS = """
    TaskBar {
        height: 1;
        max-height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._tick_count = 0
        # Timestamp (tick count) at which the current flash started.
        # None when no flash is active. The 2 s completion/failure
        # flash holds the coloured dot + summary past queue drain.
        self._flash_until_tick: int | None = None
        self._flash_outcome: TaskStatus | None = None
        # Task ids we've already flashed on. Task Center rows linger in
        # history after DONE/FAILED/CANCELLED so the user can review
        # recent work; without this gate the bar would re-flash the same
        # task every poll because ``history[-1]`` keeps matching.
        self._flashed_ids: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Label("", id="task-status-label")

    def on_mount(self) -> None:
        self._refresh_display()
        # Capture the handle so we can cancel the poll on unmount. Without
        # this, a screen push/pop cycle leaves the previous TaskBar's
        # interval firing against a detached widget, racing with the new
        # TaskBar and occasionally setting ``display=False`` on the live
        # instance (bb-3uzp).
        self._interval: Timer | None = self.set_interval(_POLL_INTERVAL_SECONDS, self._tick)

    def on_unmount(self) -> None:
        interval = getattr(self, "_interval", None)
        if interval is not None:
            interval.stop()
            self._interval = None

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
        self._tick_count += 1
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Rebuild the 1-line status label from the shared queue.

        Visual language:
        - Leading ``●`` pulses ``$primary`` <-> ``$primary-lighten-2`` at 1 Hz
          when anything is active. Dim ``$text-muted`` when only queued tasks
          remain, ``$success`` during a completion flash, ``$error`` during
          a failure flash.
        - The text either reads ``{name} {pct}`` (one active, zero queued),
          ``{N} tasks running`` (plural), ``{N} queued`` (throttle mode),
          or the flash copy.
        - Right-aligned muted-italic ``Press t for Tasks`` hint.
        """
        queue = self.queue
        active = queue.active_tasks
        queued = queue.queued_tasks
        history = queue.history

        # Drop flashed-id entries for tasks the user has cleared from
        # history. Without this prune, the set grows unbounded over a
        # long session even though any id not in history can't re-flash.
        if self._flashed_ids:
            live_ids = {t.task_id for t in history}
            self._flashed_ids &= live_ids

        in_flash = self._flash_until_tick is not None and self._tick_count <= self._flash_until_tick
        if not in_flash:
            self._flash_until_tick = None
            self._flash_outcome = None
            # Flash on the freshest completion that hasn't been flashed
            # yet. History now persists (rows show as DONE in Task
            # Center until cleared), so we must gate by task_id instead
            # of "history is non-empty".
            if not active and not queued and history:
                last = history[-1]
                if last.task_id not in self._flashed_ids and last.status in (
                    TaskStatus.DONE,
                    TaskStatus.FAILED,
                ):
                    self._flashed_ids.add(last.task_id)
                    self._flash_until_tick = self._tick_count + int(
                        _DONE_FLASH_SECONDS / _POLL_INTERVAL_SECONDS
                    )
                    self._flash_outcome = last.status

        if not active and not queued and not in_flash and self._flash_outcome is None:
            self.display = False
            return

        self.display = True
        dot_color, summary = self._compose_segments(active, queued)
        hint = f"[i dim]{self._hint_copy()}[/]"
        dot = f"[{dot_color}]{_DOT_GLYPH}[/]"
        label_text = f" {dot}  {summary}    {hint}"

        with contextlib.suppress(Exception):
            label = self.query_one("#task-status-label", Label)
            label.update(label_text)

    def _hint_copy(self) -> str:
        """Return the right-aligned hint, context-aware.

        When a chat ``Input`` (or similar) is focused the ``t`` keypress is
        eaten before the app-level binding fires, so the user needs
        ``Esc then t``. Every other screen (wizard grid, catalog,
        settings, task center) lets ``t`` bubble, so a shorter ``Press t
        for Tasks`` is accurate and easier to scan.
        """
        from textual.widgets import Input

        try:
            focused = self.app.focused
        except Exception:
            return msg.TASKBAR_HINT
        if isinstance(focused, Input):
            return msg.TASKBAR_HINT_INPUT
        return msg.TASKBAR_HINT

    def _compose_segments(self, active: list, queued: list) -> tuple[str, str]:
        """Return (dot color, text summary) for the current state."""
        # Pulsing even/odd cadence, shared with TaskRow's rail pulse.
        on_beat = (self._tick_count // _DOT_PULSE_HALF_TICKS) % 2 == 0

        if self._flash_outcome == TaskStatus.DONE:
            return "$success", msg.TASKBAR_ALL_DONE
        if self._flash_outcome == TaskStatus.FAILED:
            count = sum(1 for t in self.queue.history if t.status == TaskStatus.FAILED)
            key = msg.TASKBAR_FAILED if count == 1 else msg.TASKBAR_FAILED_PLURAL
            return "$error", key.format(count=count)

        parts: list[str] = []
        if active:
            count = len(active)
            task = active[0]
            if count == 1 and not queued:
                pct = "" if task.indeterminate else f"  [b]{task.progress:.1f}%[/b]"
                parts.append(f"[b]{task.name}[/b]{pct}")
            else:
                key = msg.TASKBAR_ONE if count == 1 else msg.TASKBAR_MULTIPLE
                parts.append(key.format(count=count))
                parts.append(f"[b]{task.name}[/b]")
        if queued:
            parts.append(f"[dim]{msg.TASKBAR_QUEUED_COUNT.format(count=len(queued))}[/dim]")

        dot_color = ("$primary" if on_beat else "$primary-lighten-2") if active else "$text-muted"
        return dot_color, "  ·  ".join(parts)
