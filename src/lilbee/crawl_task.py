"""Background crawl task management — start, track, and query crawl operations."""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

log = logging.getLogger(__name__)


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


# In-memory registry of active/completed tasks
_active_tasks: dict[str, CrawlTask] = {}


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _make_progress_updater(task: CrawlTask):  # type: ignore[no-untyped-def]
    """Return a progress callback that updates task fields."""

    def _on_progress(crawled: int, total: int, _url: str) -> None:
        task.pages_crawled = crawled
        task.pages_total = total

    return _on_progress


async def _run_crawl(task: CrawlTask) -> None:
    """Execute crawl, save results, and trigger sync."""
    from lilbee.crawler import crawl_and_save

    task.status = TaskStatus.RUNNING
    task.started_at = _now_iso()
    progress = _make_progress_updater(task)

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
        task.finished_at = _now_iso()


def start_crawl(
    url: str,
    depth: int = 0,
    max_pages: int = 0,
) -> str:
    """Create a crawl task and launch it as an asyncio background task.

    Returns the task_id for status polling.
    """
    task_id = uuid.uuid4().hex[:12]
    task = CrawlTask(
        task_id=task_id,
        url=url,
        depth=depth,
        max_pages=max_pages,
    )
    _active_tasks[task_id] = task
    # Store reference to prevent garbage collection (RUF006)
    task._async_task = asyncio.create_task(_run_crawl(task))  # type: ignore[attr-defined]
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
