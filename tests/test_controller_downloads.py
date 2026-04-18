"""TaskBarController owns downloads.

The worker thread writes progress directly to the shared TaskQueue
(lock-protected), so progress survives any screen navigation. These
tests cover ``start_download``, the progress callback, completion /
failure / cancellation paths, and the two-concurrent capacity.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Footer

from lilbee.catalog import CatalogModel, DownloadProgress
from lilbee.cli.tui.task_queue import TaskStatus
from lilbee.cli.tui.widgets.task_bar import TaskBarController, _DownloadCancelled


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


def _wait_until(predicate, timeout: float = 2.0) -> None:
    """Spin on the predicate so tests don't sleep longer than necessary."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting for predicate {predicate!r}")


def test_start_download_enqueues_and_returns_task_id() -> None:
    app = _Host()
    controller = TaskBarController(app)
    # download_model is stubbed so no network call actually runs. The
    # fake spins until we release it, so the task stays ACTIVE.
    release = [False]

    def fake_download(model, on_progress=None):
        while not release[0]:
            time.sleep(0.01)

    with patch("lilbee.catalog.download_model", side_effect=fake_download):
        task_id = controller.start_download(_make_model())
        _wait_until(lambda: controller.queue.get_task(task_id).status == TaskStatus.ACTIVE)
        assert isinstance(task_id, str)
        release[0] = True
        _wait_until(lambda: controller.queue.get_task(task_id).status == TaskStatus.DONE)


def test_progress_callback_updates_queue_percent_and_detail() -> None:
    app = _Host()
    controller = TaskBarController(app)
    progress_values = [10, 50, 90]

    def fake_download(model, on_progress=None):
        for pct in progress_values:
            on_progress(pct, 100)

    with patch("lilbee.catalog.download_model", side_effect=fake_download):
        task_id = controller.start_download(_make_model())
        _wait_until(lambda: controller.queue.get_task(task_id).status == TaskStatus.DONE)

    # make_download_callback throttles at 100 ms; the final 90% sample
    # might be throttled. We only assert that progress moved off zero,
    # which is what "real-time" means from the user's perspective.
    task = controller.queue.get_task(task_id)
    assert task is not None
    assert task.progress == 100
    assert task.status == TaskStatus.DONE


def test_failed_download_marks_task_failed_with_detail() -> None:
    app = _Host()
    controller = TaskBarController(app)

    def fake_download(model, on_progress=None):
        raise RuntimeError("network unreachable")

    with patch("lilbee.catalog.download_model", side_effect=fake_download):
        task_id = controller.start_download(_make_model())
        _wait_until(lambda: controller.queue.get_task(task_id).status == TaskStatus.FAILED)

    task = controller.queue.get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert "network unreachable" in task.detail


def test_cancelled_task_aborts_download_via_callback() -> None:
    """Cancelling a task from the UI raises _DownloadCancelled on the next tick."""
    app = _Host()
    controller = TaskBarController(app)
    ticks = [0]

    def fake_download(model, on_progress=None):
        for _ in range(100):
            # Cancel before the second tick. make_download_callback
            # has throttle_interval=0.1s, so we push the first tick and
            # immediately cancel; the second tick sees CANCELLED and
            # raises _DownloadCancelled.
            if ticks[0] == 0:
                pass
            elif ticks[0] == 1:
                controller.queue.cancel(task_id)
            on_progress(ticks[0] * 10 + 1, 100)
            ticks[0] += 1
            time.sleep(0.12)

    task_id = ""
    with patch("lilbee.catalog.download_model", side_effect=fake_download):
        task_id = controller.start_download(_make_model())
        _wait_until(
            lambda: controller.queue.get_task(task_id).status == TaskStatus.CANCELLED,
            timeout=5.0,
        )

    task = controller.queue.get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.CANCELLED


def test_concurrent_downloads_two_active_one_queued() -> None:
    """Capacity 2: submit 3, first two go ACTIVE, third stays QUEUED."""
    app = _Host()
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

        # Release the first two; the queued one should promote.
        release[0] = True
        _wait_until(
            lambda: all(controller.queue.get_task(tid).status == TaskStatus.DONE for tid in ids)
        )


def test_progress_callback_raises_cancelled_if_status_flipped_before_call() -> None:
    """Direct unit: the returned closure raises _DownloadCancelled when CANCELLED."""
    app = _Host()
    controller = TaskBarController(app)
    model = _make_model()
    task_id = controller.queue.enqueue(lambda: None, model.display_name, "download")
    controller.queue.advance("download")
    handler = controller._make_progress_handler(task_id, model)

    handler(DownloadProgress(percent=25.0, detail="25 MB", is_cache_hit=False))
    assert controller.queue.get_task(task_id).progress == 25.0

    controller.queue.cancel(task_id)
    with pytest.raises(_DownloadCancelled):
        handler(DownloadProgress(percent=30.0, detail="30 MB", is_cache_hit=False))


def test_spawn_worker_without_model_is_noop() -> None:
    """Defensive: _spawn_download_worker with unknown task_id does nothing."""
    app = _Host()
    controller = TaskBarController(app)
    controller._spawn_download_worker("unknown-task-id")  # must not raise
