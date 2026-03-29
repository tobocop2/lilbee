"""Sequential task queue for background operations (downloads, syncs, crawls)."""

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
    """FIFO queue that runs one task at a time.

    Thread-safe: tasks are enqueued from any thread, but only one executes
    at a time. Callers receive a *task_id* they can use to update progress,
    cancel, or query status.
    """

    def __init__(self, *, on_change: Callable[[], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._queue: list[str] = []
        self._active_id: str | None = None
        self._on_change = on_change

    @property
    def active_task(self) -> Task | None:
        with self._lock:
            if self._active_id:
                return self._tasks.get(self._active_id)
            return None

    @property
    def queued_tasks(self) -> list[Task]:
        with self._lock:
            return [self._tasks[tid] for tid in self._queue if tid in self._tasks]

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._active_id is None and len(self._queue) == 0

    def enqueue(self, fn: Callable[[], None], name: str, task_type: str) -> str:
        """Add a task to the queue. Returns a task_id."""
        task_id = uuid.uuid4().hex[:8]
        task = Task(task_id=task_id, name=name, task_type=task_type, fn=fn)
        with self._lock:
            self._tasks[task_id] = task
            self._queue.append(task_id)
        self._notify()
        return task_id

    def update_task(self, task_id: str, progress: int, detail: str = "") -> None:
        """Update progress and detail text for a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.progress = progress
                task.detail = detail
        self._notify()

    def complete_task(self, task_id: str) -> None:
        """Mark a task as done and remove it from tracking."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.DONE
                task.progress = 100
            if self._active_id == task_id:
                self._active_id = None
            self._queue = [tid for tid in self._queue if tid != task_id]
        self._notify()

    def fail_task(self, task_id: str, detail: str = "") -> None:
        """Mark a task as failed and remove it."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.FAILED
                task.detail = detail
            if self._active_id == task_id:
                self._active_id = None
            self._queue = [tid for tid in self._queue if tid != task_id]
        self._notify()

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued or active task. Returns True if found."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.status = TaskStatus.CANCELLED
            if self._active_id == task_id:
                self._active_id = None
            self._queue = [tid for tid in self._queue if tid != task_id]
        self._notify()
        return True

    def advance(self) -> Task | None:
        """Pop the next queued task and mark it active. Returns None if empty."""
        with self._lock:
            if self._active_id is not None:
                return None
            if not self._queue:
                return None
            task_id = self._queue.pop(0)
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.ACTIVE
                self._active_id = task_id
            return task

    def remove_task(self, task_id: str) -> None:
        """Remove a completed/failed/cancelled task from tracking entirely."""
        with self._lock:
            self._tasks.pop(task_id, None)
            self._queue = [tid for tid in self._queue if tid != task_id]
            if self._active_id == task_id:
                self._active_id = None
        self._notify()

    def _notify(self) -> None:
        if self._on_change:
            self._on_change()
