"""Background crawl task management — start, track, and query crawl operations."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from lilbee.progress import DetailedProgressCallback, EventType

log = logging.getLogger(__name__)

# Maximum completed tasks to retain in memory before evicting oldest.
_MAX_COMPLETED_TASKS = 100


class TaskStatus(StrEnum):
    """Lifecycle states for a crawl task."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class CrawlTask:
    """Tracks a single crawl operation."""

    task_id: str
    url: str
    depth: int
    max_pages: int
    status: TaskStatus = TaskStatus.PENDING
    pages_crawled: int = 0
    pages_total: int | None = None
    error: str | None = None
    started_at: str = ""
    finished_at: str = ""
    _async_task: asyncio.Task[None] | None = field(default=None, repr=False, init=False)


# In-memory registry of active/completed tasks
_active_tasks: dict[str, CrawlTask] = {}


def now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def make_progress_updater(task: CrawlTask) -> DetailedProgressCallback:
    """Return a progress callback that updates task fields from crawl events."""

    def _on_progress(event_type: EventType, data: dict[str, Any]) -> None:
        if event_type == EventType.CRAWL_PAGE:
            task.pages_crawled = data.get("current", task.pages_crawled)
            task.pages_total = data.get("total", task.pages_total)

    return _on_progress


async def run_crawl(task: CrawlTask) -> None:
    """Execute crawl, save results, and trigger sync."""
    from lilbee.crawler import crawl_and_save

    task.status = TaskStatus.RUNNING
    task.started_at = now_iso()
    progress = make_progress_updater(task)

    try:
        paths = await crawl_and_save(
            task.url,
            depth=task.depth,
            max_pages=task.max_pages,
            on_progress=progress,
        )
        task.status = TaskStatus.DONE
        task.pages_crawled = task.pages_crawled or len(paths)
        log.info("Crawl complete: %s → %d files", task.url, len(paths))
    except Exception as exc:
        task.status = TaskStatus.FAILED
        task.error = str(exc)
        log.warning("Crawl failed: %s — %s", task.url, exc)
    finally:
        task.finished_at = now_iso()


def _evict_completed() -> None:
    """Remove oldest completed tasks when the limit is exceeded."""
    done_statuses = (TaskStatus.DONE, TaskStatus.FAILED)
    completed = [(tid, t) for tid, t in _active_tasks.items() if t.status in done_statuses]
    excess = len(completed) - _MAX_COMPLETED_TASKS
    if excess <= 0:
        return
    completed.sort(key=lambda pair: pair[1].finished_at)
    for tid, _ in completed[:excess]:
        del _active_tasks[tid]


def start_crawl(
    url: str,
    depth: int = 0,
    max_pages: int = 0,
) -> str:
    """Create a crawl task and launch it as an asyncio background task.

    Returns the task_id for status polling.
    """
    _evict_completed()
    task_id = uuid.uuid4().hex[:12]
    task = CrawlTask(
        task_id=task_id,
        url=url,
        depth=depth,
        max_pages=max_pages,
    )
    _active_tasks[task_id] = task
    task._async_task = asyncio.create_task(run_crawl(task))
    return task_id


def get_task(task_id: str) -> CrawlTask | None:
    """Look up a crawl task by ID."""
    return _active_tasks.get(task_id)


def list_tasks() -> list[CrawlTask]:
    """Return all tracked crawl tasks (active and completed)."""
    return list(_active_tasks.values())


def clear_tasks() -> None:
    """Remove all tasks from the registry (for testing)."""
    _active_tasks.clear()
