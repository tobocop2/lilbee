"""Tests for crawl task management."""

from unittest.mock import AsyncMock, patch

import pytest

from lilbee.config import cfg
from lilbee.crawl_task import (
    _MAX_COMPLETED_TASKS,
    CrawlTask,
    TaskStatus,
    _active_tasks,
    clear_tasks,
    get_task,
    list_tasks,
    make_progress_updater,
    now_iso,
    run_crawl,
    start_crawl,
)
from lilbee.progress import EventType


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths and clear task registry for every test."""
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    clear_tasks()
    yield tmp_path
    clear_tasks()
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestTaskStatus:
    def test_enum_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.DONE == "done"
        assert TaskStatus.FAILED == "failed"


class TestCrawlTask:
    def test_creation(self):
        task = CrawlTask(
            task_id="abc123",
            url="https://example.com",
            depth=2,
            max_pages=50,
        )
        assert task.status == TaskStatus.PENDING
        assert task.pages_crawled == 0
        assert task.pages_total is None
        assert task.error is None

    def test_default_timestamps(self):
        task = CrawlTask(task_id="t1", url="https://example.com", depth=0, max_pages=10)
        assert task.started_at == ""
        assert task.finished_at == ""


class TestNowIso:
    def test_returns_string(self):
        result = now_iso()
        assert isinstance(result, str)
        assert "T" in result


class TestMakeProgressUpdater:
    def test_updates_task_fields_on_crawl_page(self):
        task = CrawlTask(task_id="t1", url="https://example.com", depth=1, max_pages=10)
        updater = make_progress_updater(task)
        updater(
            EventType.CRAWL_PAGE, {"current": 5, "total": 10, "url": "https://example.com/page5"}
        )
        assert task.pages_crawled == 5
        assert task.pages_total == 10

    def test_ignores_non_crawl_page_events(self):
        task = CrawlTask(task_id="t1", url="https://example.com", depth=1, max_pages=10)
        updater = make_progress_updater(task)
        updater(EventType.CRAWL_START, {"url": "https://example.com", "depth": 1})
        assert task.pages_crawled == 0
        assert task.pages_total is None


class TestRunCrawl:
    @patch("lilbee.crawl_task.crawl_and_save", new_callable=AsyncMock)
    async def test_success(self, mock_crawl):
        from pathlib import Path

        mock_crawl.return_value = [Path("/tmp/a.md"), Path("/tmp/b.md")]
        task = CrawlTask(task_id="t1", url="https://example.com", depth=1, max_pages=10)

        await run_crawl(task)
        assert task.status == TaskStatus.DONE
        assert task.started_at != ""
        assert task.finished_at != ""
        assert task.pages_crawled == 2

    @patch("lilbee.crawl_task.crawl_and_save", new_callable=AsyncMock)
    async def test_failure(self, mock_crawl):
        mock_crawl.side_effect = RuntimeError("network error")
        task = CrawlTask(task_id="t1", url="https://example.com", depth=0, max_pages=10)

        await run_crawl(task)
        assert task.status == TaskStatus.FAILED
        assert "network error" in task.error
        assert task.finished_at != ""


class TestTaskRegistry:
    @patch("lilbee.crawl_task.run_crawl", new_callable=AsyncMock)
    async def test_start_and_get(self, mock_run):
        task_id = start_crawl("https://example.com", depth=1, max_pages=10)
        assert task_id is not None
        task = get_task(task_id)
        assert task is not None
        assert task.url == "https://example.com"
        assert task.depth == 1
        assert task.max_pages == 10

    def test_get_nonexistent(self):
        assert get_task("nonexistent") is None

    @patch("lilbee.crawl_task.run_crawl", new_callable=AsyncMock)
    async def test_list_tasks(self, mock_run):
        start_crawl("https://a.com")
        start_crawl("https://b.com")
        tasks = list_tasks()
        assert len(tasks) == 2

    @patch("lilbee.crawl_task.run_crawl", new_callable=AsyncMock)
    async def test_clear_tasks(self, mock_run):
        start_crawl("https://example.com")
        assert len(list_tasks()) == 1
        clear_tasks()
        assert len(list_tasks()) == 0

    @patch("lilbee.crawl_task.run_crawl", new_callable=AsyncMock)
    async def test_concurrent_tasks(self, mock_run):
        id1 = start_crawl("https://a.com")
        id2 = start_crawl("https://b.com")
        assert id1 != id2
        t1 = get_task(id1)
        t2 = get_task(id2)
        assert t1.url == "https://a.com"
        assert t2.url == "https://b.com"


class TestEviction:
    @patch("lilbee.crawl_task.run_crawl", new_callable=AsyncMock)
    async def test_evicts_oldest_completed_tasks(self, mock_run):
        """When completed tasks exceed _MAX_COMPLETED_TASKS, oldest are evicted."""
        for i in range(_MAX_COMPLETED_TASKS + 5):
            task = CrawlTask(
                task_id=f"t{i:04d}",
                url=f"https://example.com/{i}",
                depth=0,
                max_pages=10,
                status=TaskStatus.DONE,
                finished_at=f"2026-01-01T00:{i:02d}:00+00:00",
            )
            _active_tasks[task.task_id] = task

        start_crawl("https://example.com/new")
        completed = [t for t in _active_tasks.values() if t.status == TaskStatus.DONE]
        assert len(completed) <= _MAX_COMPLETED_TASKS
