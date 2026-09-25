"""End-to-end crawls through the TUI `/crawl` dialog and typed command."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Checkbox, Input

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.task_center import TaskCenter
from lilbee.cli.tui.task_queue import Task, TaskStatus, TaskType
from lilbee.cli.tui.widgets.crawl_dialog import CrawlDialog
from lilbee.cli.tui.widgets.task_row import TaskRow
from lilbee.core.config import cfg
from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import require_valid_crawl_url
from tests.integration import _crawl_site as site_mod
from tests.integration._crawl_site import (
    NAMED_HOST,
    SUB_HOST,
    CrawlSite,
    crawl_sandbox,
    full_crawl_only,
    needs_crawler,
    needs_named_hosts,
    require_chromium,
    saved_pages,
    windows_proactor_loop,
)
from tests.integration.test_tui_integration import (
    _IntegrationChatApp,
    _join_task_workers,
    _submit_slash,
)

pytestmark = [pytest.mark.slow, needs_crawler, full_crawl_only]

# Wall-clock budgets sized for Windows runners, where a browser crawl and its teardown are slow.
_CRAWL_WAIT_S = 120.0
_DIALOG_WAIT_S = 10.0
_CANCEL_AFTER_PAGES = 3
_SETTLE_S = 2.0
_POLL_S = 0.05
PRIVATE_URL = "http://10.0.0.1/"
RECURSIVE_DEPTH = 1
SLASH_DEPTH = 1
SLASH_MAX_PAGES = 5
MIN_PAGES_BEYOND_SEED = 2
SEED_ONLY_DEPTH = 0
CONFIG_MAX_DEPTH = 1
FAILED_PAGE_COUNT = 2
TIMEOUT_S = 2


class _CrawlApp(_IntegrationChatApp):
    """The chat app host that also records every notification it shows."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def notify(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self.messages.append(str(message))
        super().notify(message, *args, **kwargs)


@contextmanager
def _tui_sandbox(tmp_path: Path) -> Iterator[Path]:
    with crawl_sandbox(tmp_path) as sandbox_root:
        try:
            yield sandbox_root
        finally:
            _join_task_workers()


@pytest.fixture
def root(tmp_path: Path) -> Iterator[Path]:
    with _tui_sandbox(tmp_path) as sandbox_root:
        yield sandbox_root


@pytest.fixture
def browser_root(tmp_path: Path) -> Iterator[Path]:
    """The sandbox for a browser-mode crawl, with its crawl loop able to start Chromium."""
    with windows_proactor_loop(), _tui_sandbox(tmp_path) as sandbox_root:
        yield sandbox_root


async def _wait_for(pilot: Pilot[Any], predicate: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await pilot.pause()
        if predicate():
            return True
        await asyncio.sleep(_POLL_S)
    return predicate()


def _workers_idle() -> bool:
    """True once no task-bar worker (the crawl, then its follow-up sync) is still running."""
    return not any(t.name.startswith("task-") and t.is_alive() for t in threading.enumerate())


def _crawl_task(app: _CrawlApp) -> Task | None:
    queue = app.task_bar.queue
    tasks = queue.active_tasks + queue.queued_tasks + queue.history
    return next((t for t in tasks if t.task_type == TaskType.CRAWL), None)


def _crawl_finished(app: _CrawlApp) -> bool:
    task = _crawl_task(app)
    return task is not None and task.status is not TaskStatus.ACTIVE


async def _submit_crawl_dialog(
    pilot: Pilot[Any],
    app: _CrawlApp,
    url: str,
    *,
    recursive: bool,
    browser: bool = False,
    depth: str = "",
) -> None:
    """Open ``/crawl``, fill the dialog, and submit it with enter."""
    await _submit_slash(pilot, app, "/crawl")
    assert await _wait_for(pilot, lambda: isinstance(app.screen, CrawlDialog), _DIALOG_WAIT_S)
    dialog = app.screen
    dialog.query_one("#crawl-url-input", Input).value = url
    dialog.query_one("#crawl-recursive-checkbox", Checkbox).value = recursive
    dialog.query_one("#crawl-browser-checkbox", Checkbox).value = browser
    dialog.query_one("#crawl-depth-input", Input).value = depth
    dialog.query_one("#crawl-url-input", Input).focus()
    await pilot.press("enter")
    assert await _wait_for(pilot, lambda: _crawl_task(app) is not None, _DIALOG_WAIT_S)


Submit = Callable[[Pilot[Any], "_CrawlApp"], Awaitable[None]]


def _dialog(url: str, *, recursive: bool, browser: bool = False) -> Submit:
    """Submit *url* through the ``/crawl`` dialog."""
    depth = str(RECURSIVE_DEPTH) if recursive else ""

    async def submit(pilot: Pilot[Any], app: _CrawlApp) -> None:
        await _submit_crawl_dialog(
            pilot, app, url, recursive=recursive, browser=browser, depth=depth
        )

    return submit


def _slash(command: str) -> Submit:
    """Submit a typed ``/crawl`` command with its arguments."""

    async def submit(pilot: Pilot[Any], app: _CrawlApp) -> None:
        await _submit_slash(pilot, app, command)
        assert await _wait_for(pilot, lambda: _crawl_task(app) is not None, _DIALOG_WAIT_S)

    return submit


async def _run_crawl(root: Path, url: str, submit: Submit) -> tuple[_CrawlApp, dict[str, str]]:
    """Start a crawl with *submit*, wait for its task row to finish, and check it is done."""
    app = _CrawlApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await submit(pilot, app)
        assert await _wait_for(pilot, lambda: _crawl_finished(app), _CRAWL_WAIT_S)
        pages = saved_pages(root)
        saved_count = len(list((root / "documents" / "_web").rglob("*.md")))
        task = _crawl_task(app)
        assert task is not None
        assert task.status is TaskStatus.DONE
        assert msg.CMD_CRAWL_SUCCESS.format(count=saved_count, url=url) in app.messages
        app.push_screen(TaskCenter())
        await pilot.pause()
        row = app.screen.query_one(f"#task-{task.task_id}", TaskRow)
        assert row.has_class("-done")
        assert await _wait_for(pilot, _workers_idle, _CRAWL_WAIT_S)
    return app, pages


async def test_single_page_crawl_saves_only_the_seed(root: Path, crawl_site: CrawlSite) -> None:
    url = crawl_site.url("/")
    _app, pages = await _run_crawl(root, url, _dialog(url, recursive=False))
    assert set(pages) == {"index.md"}
    assert site_mod.HOME_TEXT in pages["index.md"]


async def test_recursive_crawl_saves_the_linked_pages(root: Path, crawl_site: CrawlSite) -> None:
    url = crawl_site.url("/rel/child.html")
    _app, pages = await _run_crawl(root, url, _dialog(url, recursive=True))
    assert set(pages) == {"rel/child.md", "rel/sibling.md"}
    assert site_mod.SIBLING_TEXT in pages["rel/sibling.md"]


async def test_browser_mode_crawl_saves_the_linked_pages(
    browser_root: Path, crawl_site: CrawlSite
) -> None:
    require_chromium()
    url = crawl_site.url("/rel/child.html")
    _app, pages = await _run_crawl(browser_root, url, _dialog(url, recursive=True, browser=True))
    assert set(pages) == {"rel/child.md", "rel/sibling.md"}
    assert site_mod.SIBLING_TEXT in pages["rel/sibling.md"]


async def test_cancel_stops_the_crawl(root: Path, crawl_site: CrawlSite) -> None:
    app = _CrawlApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit_crawl_dialog(pilot, app, crawl_site.url("/slow/"), recursive=True)
        started = await _wait_for(
            pilot, lambda: len(saved_pages(root)) >= _CANCEL_AFTER_PAGES, _CRAWL_WAIT_S
        )
        assert started, sorted(saved_pages(root))
        app.push_screen(TaskCenter())
        await pilot.pause()
        await pilot.press("c")
        task = _crawl_task(app)
        assert task is not None
        assert await _wait_for(pilot, lambda: task.status is TaskStatus.CANCELLED, _DIALOG_WAIT_S)
        row = app.screen.query_one(f"#task-{task.task_id}", TaskRow)
        assert await _wait_for(pilot, lambda: row.has_class("-cancelled"), _DIALOG_WAIT_S)
        assert await _wait_for(pilot, _workers_idle, _CRAWL_WAIT_S)
    settled = set(saved_pages(root))
    settled_at = time.monotonic()
    await asyncio.sleep(_SETTLE_S)
    assert set(saved_pages(root)) == settled
    assert crawl_site.requested_paths("/slow/p", since=settled_at) == []
    assert len(settled) < site_mod.SLOW_PAGE_COUNT // 2, sorted(settled)


async def test_typed_crawl_command_honours_its_depth(root: Path, crawl_site: CrawlSite) -> None:
    url = crawl_site.url("/a/")
    _app, pages = await _run_crawl(root, url, _slash(f"/crawl {url} --depth {SLASH_DEPTH}"))
    assert set(pages) == {"a/index.md", "a/b/index.md"}


async def test_typed_crawl_command_honours_its_max_pages(root: Path, crawl_site: CrawlSite) -> None:
    url = crawl_site.url("/wide/")
    command = f"/crawl {url} --depth {SLASH_DEPTH} --max-pages {SLASH_MAX_PAGES}"
    _app, pages = await _run_crawl(root, url, _slash(command))
    assert MIN_PAGES_BEYOND_SEED <= len(pages) <= SLASH_MAX_PAGES, sorted(pages)
    assert len(pages) < site_mod.WIDE_PAGE_COUNT


async def test_typed_render_flag_selects_browser_mode(
    browser_root: Path, crawl_site: CrawlSite
) -> None:
    require_chromium()
    url = crawl_site.url("/js/")
    command = f"/crawl {url} --depth {SEED_ONLY_DEPTH} --render {CrawlRenderMode.BROWSER.value}"
    _app, pages = await _run_crawl(browser_root, url, _slash(command))
    assert site_mod.SCRIPT_TEXT in pages["js/index.md"]


@needs_named_hosts
async def test_typed_include_subdomains_flag_follows_the_subdomain(
    root: Path, crawl_site: CrawlSite
) -> None:
    url = crawl_site.url("/", host=NAMED_HOST)
    command = f"/crawl {url} --depth {SLASH_DEPTH} --include-subdomains"
    await _run_crawl(root, url, _slash(command))
    assert site_mod.SUB_TEXT in saved_pages(root, SUB_HOST)["index.md"]


async def test_typed_crawl_command_rejects_a_private_address(root: Path) -> None:
    app = _CrawlApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit_slash(pilot, app, f"/crawl {PRIVATE_URL}")
        assert await _wait_for(pilot, lambda: bool(app.messages), _DIALOG_WAIT_S)
        with pytest.raises(ValueError) as refusal:
            require_valid_crawl_url(PRIVATE_URL)
        assert str(refusal.value) in app.messages, app.messages
        assert _crawl_task(app) is None
    assert saved_pages(root) == {}


async def test_config_max_depth_applies_without_a_flag(root: Path, crawl_site: CrawlSite) -> None:
    cfg.crawl_max_depth = CONFIG_MAX_DEPTH
    url = crawl_site.url("/a/")
    _app, pages = await _run_crawl(root, url, _slash(f"/crawl {url}"))
    assert set(pages) == {"a/index.md", "a/b/index.md"}


async def test_config_render_mode_applies_without_a_flag(
    browser_root: Path, crawl_site: CrawlSite
) -> None:
    require_chromium()
    cfg.crawl_render_mode = CrawlRenderMode.BROWSER
    url = crawl_site.url("/js/")
    _app, pages = await _run_crawl(
        browser_root, url, _slash(f"/crawl {url} --depth {SEED_ONLY_DEPTH}")
    )
    assert site_mod.SCRIPT_TEXT in pages["js/index.md"]


async def test_failed_pages_are_reported_and_the_crawl_completes(
    root: Path, crawl_site: CrawlSite
) -> None:
    cfg.crawl_timeout = TIMEOUT_S
    url = crawl_site.url("/timeout/")
    app, pages = await _run_crawl(root, url, _slash(f"/crawl {url} --depth {RECURSIVE_DEPTH}"))
    assert set(pages) == {"timeout/index.md", "timeout/fast-a/index.md", "timeout/fast-b/index.md"}
    notice = msg.CMD_CRAWL_PAGES_FAILED.format(count=FAILED_PAGE_COUNT, reason="")
    assert any(message.startswith(notice) for message in app.messages), app.messages
