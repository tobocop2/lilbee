"""Asking twice for one download is a double keypress, not two downloads."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest
from textual.app import ComposeResult
from textual.widgets import Footer

from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask
from lilbee.cli.tui.task_queue import TERMINAL_STATUSES, TaskQueue, TaskStatus
from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController
from tests._lilbee_app_test_host import LilbeeAppHost


class _Host(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Footer()


def _noop() -> None:
    return None


def _live(q: TaskQueue) -> int:
    """Tasks the queue still owns, active or waiting."""
    return len(q.active_tasks) + len(q.queued_tasks)


def _model(repo: str = "unsloth/Qwen3-14B-GGUF", filename: str = "Q4_K_M.gguf") -> CatalogModel:
    return CatalogModel(
        hf_repo=repo,
        gguf_filename=filename,
        size_gb=9.0,
        min_ram_gb=12.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
    )


def test_second_enqueue_of_same_key_adds_nothing() -> None:
    q = TaskQueue()
    first = q.enqueue(_noop, "Qwen3 14B", "download", dedupe_key="k")
    second = q.enqueue(_noop, "Qwen3 14B", "download", dedupe_key="k")

    assert first == second
    assert _live(q) == 1


def test_queued_not_yet_active_duplicate_is_also_refused() -> None:
    """The bug was reachable while the first pull was still waiting for a slot."""
    q = TaskQueue(capacity={"download": 1})
    q.enqueue(_noop, "a", "download", dedupe_key="a")
    q.advance("download")
    behind = q.enqueue(_noop, "b", "download", dedupe_key="b")
    again = q.enqueue(_noop, "b", "download", dedupe_key="b")

    assert behind == again
    assert q.get_task(behind).status == TaskStatus.QUEUED
    assert _live(q) == 2


def test_dedupe_is_per_task_type() -> None:
    """A sync and a download that happen to share a key are unrelated work."""
    q = TaskQueue()
    d = q.enqueue(_noop, "x", "download", dedupe_key="same")
    s = q.enqueue(_noop, "x", "sync", dedupe_key="same")

    assert d != s
    assert _live(q) == 2


def test_finished_download_can_be_requested_again() -> None:
    """Dedupe must not become a permanent ban: a completed or cancelled pull is
    re-runnable, which is how a user retries a failed download."""
    q = TaskQueue()
    first = q.enqueue(_noop, "a", "download", dedupe_key="k")
    q.advance("download")
    q.complete_task(first)

    second = q.enqueue(_noop, "a", "download", dedupe_key="k")
    assert second != first


def test_no_key_means_no_dedupe() -> None:
    q = TaskQueue()
    a = q.enqueue(_noop, "a", "sync")
    b = q.enqueue(_noop, "a", "sync")

    assert a != b


def test_find_pending_reports_the_live_task() -> None:
    q = TaskQueue()
    assert q.find_pending("download", "k") is None
    tid = q.enqueue(_noop, "a", "download", dedupe_key="k")
    found = q.find_pending("download", "k")

    assert found is not None
    assert found.task_id == tid


def test_download_key_separates_quants_of_one_repo() -> None:
    """Two quants of a repo are different files and may both be wanted."""
    from lilbee.cli.tui.widgets.task_bar_controller import download_key

    assert download_key(_model(filename="Q4_K_M.gguf")) != download_key(
        _model(filename="Q8_0.gguf")
    )
    assert download_key(_model()) == download_key(_model())


@pytest.mark.asyncio
async def test_controller_refuses_a_second_pull_of_the_same_model() -> None:
    """The production path: two start_download calls for one model.

    Covers what the unit tests above cannot, that start_download threads a key
    through and that the duplicate never gets a worker of its own.
    """
    app = _Host()
    async with app.run_test():
        controller = TaskBarController(app)
        release = [False]

        def fake_download(model, on_progress=None, on_complete=None):
            while not release[0]:
                time.sleep(0.01)

        with patch("lilbee.catalog.download_model", side_effect=fake_download):
            model = _model()
            first = controller.start_download(model)
            second = controller.start_download(model)
            third = controller.start_download(_model())  # equal by value, same key

            assert first == second == third
            assert _live(controller.queue) == 1
            assert controller.pending_download(model) is not None

            # Let the single worker finish; conftest fails the test if one leaks.
            release[0] = True
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                task = controller.queue.get_task(first)
                if task is not None and task.status in TERMINAL_STATUSES:
                    break
                await asyncio.sleep(0.02)
