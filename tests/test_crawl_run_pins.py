"""Pin what run_crawl hands to the crawler and what it leaves on the task.

A mutation run of the crawl-sync logging change left these values unread: the settings
passed to the crawler, the progress callback, the running state, the finish stamps of a
cancelled, finished and failed crawl, and the cleanup of the login and the async task.
"""

import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler.models import NO_LOGIN, CrawlLogin
from lilbee.crawler.task import CrawlTask, TaskStatus, run_crawl
from lilbee.runtime.progress import CrawlPageEvent, EventType

_LOGIN = CrawlLogin(authorization="Bearer secret-token")


def _task(**overrides) -> CrawlTask:
    fields = {
        "task_id": "t1",
        "url": "https://example.com",
        "depth": 3,
        "max_pages": 7,
        "render_mode": CrawlRenderMode.BROWSER,
        "include_subdomains": True,
        "login": _LOGIN,
    }
    fields.update(overrides)
    return CrawlTask(**fields)


def _is_stamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    datetime.fromisoformat(value)
    return True


@patch("lilbee.data.ingest.sync", new_callable=AsyncMock)
@patch("lilbee.crawler.task.crawl_and_save", new_callable=AsyncMock)
async def test_the_crawler_gets_every_setting_of_the_task(mock_crawl, _sync):
    mock_crawl.return_value = [Path("a.md")]
    task = _task()
    await run_crawl(task)
    call = mock_crawl.await_args
    assert call.args == ("https://example.com",)
    kwargs = call.kwargs
    assert kwargs["depth"] == 3
    assert kwargs["max_pages"] == 7
    assert kwargs["cancel"] is task.cancel
    assert kwargs["include_subdomains"] is True
    assert kwargs["render_mode"] is CrawlRenderMode.BROWSER
    assert kwargs["login"] is _LOGIN


@patch("lilbee.data.ingest.sync", new_callable=AsyncMock)
@patch("lilbee.crawler.task.crawl_and_save", new_callable=AsyncMock)
async def test_the_progress_callback_of_the_crawler_updates_the_task(mock_crawl, _sync):
    mock_crawl.return_value = []
    task = _task()
    await run_crawl(task)
    on_progress = mock_crawl.await_args.kwargs["on_progress"]
    on_progress(EventType.CRAWL_PAGE, CrawlPageEvent(current=4, total=9, url="https://example.com"))
    assert (task.pages_crawled, task.pages_total) == (4, 9)


@patch("lilbee.data.ingest.sync", new_callable=AsyncMock)
async def test_a_crawl_in_flight_is_running_and_has_a_start_stamp(_sync):
    seen = {}

    async def crawl(*_args, **_kwargs):
        seen["status"] = task.status
        seen["started_at"] = task.started_at
        seen["login"] = task.login
        return []

    task = _task()
    with patch("lilbee.crawler.task.crawl_and_save", side_effect=crawl):
        await run_crawl(task)
    assert seen["status"] is TaskStatus.RUNNING
    assert _is_stamp(seen["started_at"])
    assert seen["login"] is _LOGIN


@patch("lilbee.data.ingest.sync", new_callable=AsyncMock)
@patch("lilbee.crawler.task.crawl_and_save", new_callable=AsyncMock)
async def test_a_finished_crawl_has_a_finish_stamp(mock_crawl, _sync):
    mock_crawl.return_value = []
    task = _task()
    await run_crawl(task)
    assert task.status is TaskStatus.DONE
    assert _is_stamp(task.finished_at)


@patch("lilbee.crawler.task.crawl_and_save", new_callable=AsyncMock)
async def test_a_failed_crawl_has_a_finish_stamp(mock_crawl):
    mock_crawl.side_effect = RuntimeError("down")
    task = _task()
    await run_crawl(task)
    assert task.status is TaskStatus.FAILED
    assert _is_stamp(task.finished_at)


async def test_a_cancelled_crawl_keeps_its_pages_and_has_a_finish_stamp():
    cancel = threading.Event()

    async def crawl(*_args, **_kwargs):
        cancel.set()
        return [Path("a.md"), Path("b.md"), Path("c.md")]

    task = _task(cancel=cancel)
    with (
        patch("lilbee.crawler.task.crawl_and_save", side_effect=crawl),
        patch("lilbee.data.ingest.sync", new_callable=AsyncMock) as sync,
    ):
        await run_crawl(task)
    assert task.status is TaskStatus.CANCELLED
    assert task.pages_crawled == 3
    assert _is_stamp(task.finished_at)
    sync.assert_not_awaited()


@pytest.mark.parametrize("fails", [False, True], ids=["done", "failed"])
@patch("lilbee.data.ingest.sync", new_callable=AsyncMock)
@patch("lilbee.crawler.task.crawl_and_save", new_callable=AsyncMock)
async def test_a_crawl_drops_its_login_and_its_async_task(mock_crawl, _sync, fails):
    if fails:
        mock_crawl.side_effect = RuntimeError("down")
    else:
        mock_crawl.return_value = []
    task = _task()
    task._async_task = object()  # type: ignore[assignment]
    await run_crawl(task)
    assert task.login is NO_LOGIN
    assert task._async_task is None
