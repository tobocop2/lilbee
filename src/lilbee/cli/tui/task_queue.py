"""Per-type concurrent task queue for background operations (downloads, syncs, crawls).

Each task type (download, sync, crawl) gets its own independent queue, so a long
download does not block a sync from starting. Within a type, tasks run sequentially.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

log = logging.getLogger(__name__)


class TaskStatus(StrEnum):
    """Lifecycle states for a queued task."""

    QUEUED = "queued"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


STATUS_ICONS: dict[TaskStatus, str] = {
    TaskStatus.QUEUED: "⏳",
    TaskStatus.ACTIVE: "▶",
    TaskStatus.DONE: "✓",
    TaskStatus.FAILED: "✗",
    TaskStatus.CANCELLED: "⊘",
}


@dataclass
class Task:
    """A single unit of work in the queue."""

    task_id: str
    name: str
    task_type: str
    fn: Callable[[], None]
    status: TaskStatus = TaskStatus.QUEUED
    progress: int = 0
    detail: str = ""


class TaskQueue:
    """Per-type concurrent task queue.

    Thread-safe. Each task type (download, sync, crawl, etc.) has its own
    independent FIFO queue. One task per type can be active simultaneously,
    so a download does not block a sync.

    Callers receive a *task_id* they can use to update progress, cancel, or
    query status.
    """

    def __init__(self, *, on_change: Callable[[], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._queues: dict[str, list[str]] = {}
        self._active_ids: dict[str, str] = {}
        self._on_change: list[Callable[[], None]] = []
        if on_change:
            self._on_change.append(on_change)
        self._history: list[Task] = []

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Subscribe to task queue changes. Callback is called on any queue update."""
        with self._lock:
            if callback not in self._on_change:
                self._on_change.append(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        """Unsubscribe from task queue changes."""
        with self._lock:
            if callback in self._on_change:
                self._on_change.remove(callback)

    @property
    def active_task(self) -> Task | None:
        """Return the first active task (for backward compat). Prefer active_tasks."""
        with self._lock:
            for tid in self._active_ids.values():
                task = self._tasks.get(tid)
                if task:
                    return task
            return None

    @property
    def active_tasks(self) -> list[Task]:
        """Return all currently active tasks (one per type)."""
        with self._lock:
            tasks = []
            for tid in self._active_ids.values():
                task = self._tasks.get(tid)
                if task:
                    tasks.append(task)
            return tasks

    @property
    def queued_tasks(self) -> list[Task]:
        with self._lock:
            result: list[Task] = []
            for tids in self._queues.values():
                for tid in tids:
                    task = self._tasks.get(tid)
                    if task:
                        result.append(task)
            return result

    @property
    def history(self) -> list[Task]:
        with self._lock:
            return list(self._history)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            has_active = len(self._active_ids) > 0
            has_queued = any(len(q) > 0 for q in self._queues.values())
            return not has_active and not has_queued

    def get_task(self, task_id: str) -> Task | None:
        """Look up a task by ID. Returns None if not found."""
        with self._lock:
            return self._tasks.get(task_id)

    def enqueue(self, fn: Callable[[], None], name: str, task_type: str) -> str:
        """Add a task to the per-type queue. Returns a task_id."""
        task_id = uuid.uuid4().hex[:8]
        task = Task(task_id=task_id, name=name, task_type=task_type, fn=fn)
        with self._lock:
            self._tasks[task_id] = task
            self._queues.setdefault(task_type, []).append(task_id)
        self._notify()
        return task_id

    def update_task(self, task_id: str, progress: int, detail: str = "") -> None:
        """Update progress and detail text for a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.progress = progress
                task.detail = detail
            # Notify while still holding lock to ensure consistency
            self._notify()

    def complete_task(self, task_id: str) -> None:
        """Mark a task as done and remove it from tracking."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.DONE
                task.progress = 100
                self._history.append(task)
                self._remove_from_active_locked(task_id, task.task_type)
                self._remove_from_queue_locked(task_id, task.task_type)
        self._notify()

    def fail_task(self, task_id: str, detail: str = "") -> None:
        """Mark a task as failed and remove it."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.FAILED
                task.detail = detail
                self._history.append(task)
                self._remove_from_active_locked(task_id, task.task_type)
                self._remove_from_queue_locked(task_id, task.task_type)
        self._notify()

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued or active task. Returns True if found."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.status = TaskStatus.CANCELLED
            self._remove_from_active_locked(task_id, task.task_type)
            self._remove_from_queue_locked(task_id, task.task_type)
        self._notify()
        return True

    def advance(self, task_type: str | None = None) -> Task | None:
        """Pop the next queued task and mark it active.

        If *task_type* is given, only advance that type's queue.
        If omitted, advance any type that has no active task.
        Returns None if nothing can advance.
        """
        with self._lock:
            types = [task_type] if task_type else list(self._queues.keys())
            for tt in types:
                if tt in self._active_ids:
                    continue
                queue = self._queues.get(tt, [])
                if not queue:
                    continue
                tid = queue.pop(0)
                task = self._tasks.get(tid)
                if task:
                    task.status = TaskStatus.ACTIVE
                    self._active_ids[tt] = tid
                    return task
            return None

    def remove_task(self, task_id: str) -> None:
        """Remove a completed/failed/cancelled task from tracking entirely."""
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                self._remove_from_active_locked(task_id, task.task_type)
                self._remove_from_queue_locked(task_id, task.task_type)
        self._notify()

    def _remove_from_active_locked(self, task_id: str, task_type: str) -> None:
        """Remove a task from active tracking. Caller must hold _lock."""
        if self._active_ids.get(task_type) == task_id:
            del self._active_ids[task_type]

    def _remove_from_queue_locked(self, task_id: str, task_type: str) -> None:
        """Remove a task from its type queue. Caller must hold _lock."""
        queue = self._queues.get(task_type)
        if queue:
            self._queues[task_type] = [tid for tid in queue if tid != task_id]

    def _notify(self) -> None:
        for callback in self._on_change:
            callback()
