"""TaskBarController owns all long-running work.

The worker thread writes progress directly to the shared TaskQueue
(lock-protected), so progress survives any screen navigation. Completion
is posted back to the main thread via ``call_from_thread(app, ...)`` so
the 2-second flash-then-remove cycle actually runs — the previous
direct-queue-update path leaked task state indefinitely.

Tests cover both the generic ``start_task`` API and the typed
``start_download`` specialization.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Footer

from lilbee.catalog import CatalogModel
from lilbee.cli.tui.task_queue import TaskStatus, TaskType
from lilbee.cli.tui.widgets.task_bar import (
    ProgressReporter,
    TaskBarController,
    TaskCancelled,
    _DownloadCancelled,
)


def _make_model(name: str = "test", display: str = "Test Model") -> CatalogModel:
    return CatalogModel(
        name=name,
        tag="7b",
        display_name=display,
        hf_repo=f"org/{name}-7b",
        gguf_filename="test.gguf",
        size_gb=4.0,
        min_ram_gb=8.0,
        description="",
        featured=False,
        downloads=0,
        task="chat",
    )


class _Host(App[None]):
    """Minimal host so TaskBarController can bind to an App."""

    def compose(self) -> ComposeResult:
        yield Footer()


def _wait_until(predicate, timeout: float = 3.0) -> None:
    """Spin on the predicate so tests don't sleep longer than necessary."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting for predicate {predicate!r}")


@pytest.mark.asyncio
async def test_start_task_runs_target_with_reporter_and_marks_done() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        events: list[tuple[float, str]] = []

        def _target(reporter: ProgressReporter) -> None:
            reporter.update(25.0, "one")
            reporter.update(75.0, "two")
            events.append((75.0, "two"))

        task_id = controller.start_task("demo", TaskType.SYNC, _target)
        _wait_until(lambda: events)
        # give the main thread time to finalize
        for _ in range(20):
            await pilot.pause()
            task = controller.queue.get_task(task_id)
            if task is not None and task.status == TaskStatus.DONE:
                break
        task = controller.queue.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.DONE
        # Task stays visible during the 2 s flash window.


@pytest.mark.asyncio
async def test_start_task_failure_marks_failed_with_detail() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)

        def _target(reporter: ProgressReporter) -> None:
            raise RuntimeError("boom")

        task_id = controller.start_task("demo", TaskType.SYNC, _target)
        for _ in range(30):
            await pilot.pause()
            task = controller.queue.get_task(task_id)
            if task is not None and task.status == TaskStatus.FAILED:
                break
        task = controller.queue.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.FAILED
        assert "boom" in task.detail


@pytest.mark.asyncio
async def test_start_task_cancelled_marks_cancelled() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        arrived = threading.Event()

        def _target(reporter: ProgressReporter) -> None:
            arrived.set()
            # Spin on the reporter's cancel check so we exit promptly.
            for _ in range(50):
                reporter.check_cancelled()
                time.sleep(0.02)

        task_id = controller.start_task("demo", TaskType.SYNC, _target)
        assert arrived.wait(timeout=2.0)
        controller.queue.cancel(task_id)
        for _ in range(50):
            await pilot.pause()
            task = controller.queue.get_task(task_id)
            if task is not None and task.status == TaskStatus.CANCELLED:
                break
        task = controller.queue.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_on_success_runs_after_target_success() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        on_success_ran = threading.Event()

        controller.start_task(
            "demo",
            TaskType.SYNC,
            lambda reporter: None,
            on_success=on_success_ran.set,
        )
        # Worker spawn + finalize marshal back via app.call_from_thread, which
        # blocks the worker until the main thread drains. pilot.pause yields
        # to that loop; loop until the on_success flag flips.
        for _ in range(30):
            await pilot.pause()
            if on_success_ran.is_set():
                break
        assert on_success_ran.is_set()


@pytest.mark.asyncio
async def test_on_success_not_called_if_target_raises() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        ran = threading.Event()

        def _target(reporter: ProgressReporter) -> None:
            raise RuntimeError("nope")

        task_id = controller.start_task("demo", TaskType.SYNC, _target, on_success=ran.set)
        for _ in range(20):
            await pilot.pause()
            task = controller.queue.get_task(task_id)
            if task is not None and task.status == TaskStatus.FAILED:
                break
        assert not ran.is_set()


@pytest.mark.asyncio
async def test_progress_reporter_update_writes_to_queue() -> None:
    app = _Host()
    async with app.run_test():
        controller = TaskBarController(app)
        task_id = controller.queue.enqueue(
            lambda: None, "demo", TaskType.SYNC.value, indeterminate=False
        )
        controller.queue.advance(TaskType.SYNC.value)
        reporter = ProgressReporter(controller, task_id)

        reporter.update(42.5, "half")
        task = controller.queue.get_task(task_id)
        assert task is not None
        assert task.progress == 42.5
        assert task.detail == "half"


@pytest.mark.asyncio
async def test_progress_reporter_update_raises_when_cancelled() -> None:
    app = _Host()
    async with app.run_test():
        controller = TaskBarController(app)
        task_id = controller.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        controller.queue.advance(TaskType.SYNC.value)
        controller.queue.cancel(task_id)
        reporter = ProgressReporter(controller, task_id)

        with pytest.raises(TaskCancelled):
            reporter.update(0, "nope")


def test_download_cancelled_is_alias_for_task_cancelled() -> None:
    """Legacy import path must keep working."""
    assert _DownloadCancelled is TaskCancelled


@pytest.mark.asyncio
async def test_start_download_enqueues_under_download_type() -> None:
    """start_download delegates to start_task with TaskType.DOWNLOAD."""
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        release = [False]

        def fake_download(model, on_progress=None):
            while not release[0]:
                time.sleep(0.01)

        with patch("lilbee.catalog.download_model", side_effect=fake_download):
            task_id = controller.start_download(_make_model())
            _wait_until(lambda: controller.queue.get_task(task_id).status == TaskStatus.ACTIVE)
            task = controller.queue.get_task(task_id)
            assert task is not None
            assert task.task_type == TaskType.DOWNLOAD.value
            release[0] = True
            for _ in range(30):
                await pilot.pause()
                if controller.queue.get_task(task_id).status == TaskStatus.DONE:
                    break
            assert controller.queue.get_task(task_id).status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_start_download_progress_flows_through_queue() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)

        def fake_download(model, on_progress=None):
            on_progress(10, 100)
            on_progress(60, 100)
            on_progress(100, 100)

        with patch("lilbee.catalog.download_model", side_effect=fake_download):
            task_id = controller.start_download(_make_model())
            for _ in range(30):
                await pilot.pause()
                if controller.queue.get_task(task_id).status == TaskStatus.DONE:
                    break
        task = controller.queue.get_task(task_id)
        assert task is not None
        assert task.progress == 100
        assert task.status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_start_download_permission_error_gets_gated_repo_message() -> None:
    """S3: PermissionError is rewritten to the gated-repo friendly detail."""
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)

        def fake_download(model, on_progress=None):
            raise PermissionError("401 Unauthorized")

        with patch("lilbee.catalog.download_model", side_effect=fake_download):
            task_id = controller.start_download(_make_model(display="Gated Model"))
            for _ in range(30):
                await pilot.pause()
                if controller.queue.get_task(task_id).status == TaskStatus.FAILED:
                    break
        task = controller.queue.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.FAILED
        # Message comes from messages.CATALOG_GATED_REPO; just check the model name is in it.
        assert "Gated Model" in task.detail


@pytest.mark.asyncio
async def test_finalize_removes_task_after_flash(monkeypatch) -> None:
    """B1: finished tasks must not leak. After the 2 s flash, the task is removed."""
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        release = [False]

        def fake_download(model, on_progress=None):
            while not release[0]:
                time.sleep(0.01)

        # Shorten the flash so the test doesn't wait 2 s real time.
        original_set_timer = app.set_timer

        def fast_timer(_delay, callback, *args, **kwargs):
            return original_set_timer(0.05, callback, *args, **kwargs)

        monkeypatch.setattr(app, "set_timer", fast_timer)

        with patch("lilbee.catalog.download_model", side_effect=fake_download):
            task_id = controller.start_download(_make_model())
            _wait_until(lambda: controller.queue.get_task(task_id).status == TaskStatus.ACTIVE)
            release[0] = True
            for _ in range(40):
                await pilot.pause()
                if controller.queue.get_task(task_id) is None:
                    break

        assert controller.queue.get_task(task_id) is None, (
            "task was not removed after flash; B1 regression"
        )
        assert controller.queue.is_empty


@pytest.mark.asyncio
async def test_concurrent_downloads_two_active_one_queued() -> None:
    """Capacity 2: submit 3, first two go ACTIVE, third stays QUEUED."""
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        release = [False]

        def fake_download(model, on_progress=None):
            while not release[0]:
                time.sleep(0.01)

        with patch("lilbee.catalog.download_model", side_effect=fake_download):
            ids = [controller.start_download(_make_model(f"m{i}")) for i in range(3)]
            _wait_until(lambda: len(controller.queue.active_tasks) == 2)

            active_ids = {t.task_id for t in controller.queue.active_tasks}
            queued_ids = {t.task_id for t in controller.queue.queued_tasks}
            assert len(active_ids) == 2
            assert len(queued_ids) == 1
            assert active_ids | queued_ids == set(ids)

            release[0] = True
            for _ in range(50):
                await pilot.pause()
                if all(controller.queue.get_task(tid).status == TaskStatus.DONE for tid in ids):
                    break
        assert all(controller.queue.get_task(tid).status == TaskStatus.DONE for tid in ids)


@pytest.mark.asyncio
async def test_spawn_worker_without_target_is_noop() -> None:
    """Defensive: _spawn_task_worker with unknown task_id does nothing."""
    app = _Host()
    async with app.run_test():
        controller = TaskBarController(app)
        controller._spawn_task_worker("unknown-task-id")  # must not raise
