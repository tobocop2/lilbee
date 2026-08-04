"""Cancelling a xet download has to stop the transfer, not just mark the row.

xet drives the progress callback from a thread it owns, so the TaskCancelledError
lilbee raises there is swallowed and the download runs to completion behind a row
that says cancelled. Measured: the HTTP path aborts in 1.8s after 39 callbacks;
the xet path reported the file fully written.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog import download as dl
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask
from lilbee.runtime.cancellation import TaskCancelledError


def _entry() -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/x-GGUF",
        gguf_filename="x.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
    )


def test_abort_calls_through_to_the_xet_session(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    module = type(sys)("huggingface_hub.utils._xet")
    module.abort_xet_session = lambda: called.append("abort")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._xet", module)

    dl.abort_active_download()

    assert called == ["abort"]


def test_abort_is_a_no_op_without_xet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the xet path needs it; HTTP cancels by raising through the callback."""
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._xet", None)

    dl.abort_active_download()  # must not raise


def test_an_aborted_transfer_reads_as_cancelled_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hf_xet reports the abort as a bare RuntimeError. Left untranslated the row
    says failed, which reads as a broken download rather than the user's own
    keypress."""
    config = dl.DownloadConfig(
        repo_id="acme/x-GGUF", filename="x.gguf", token=None, cache_dir=str(tmp_path)
    )

    def _aborted(**_kw: Any) -> str:
        raise RuntimeError("Operation cancelled: Task cancelled: task 19 was cancelled")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _aborted)

    with pytest.raises(TaskCancelledError):
        dl._hf_download_or_translate(_entry(), config)


async def test_cancelling_a_running_download_aborts_the_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row alone is not the fix: without the abort the bytes keep arriving."""
    from lilbee.cli.tui.task_queue import TaskStatus, TaskType
    from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController
    from tests._lilbee_app_test_host import LilbeeAppHost

    aborted: list[str] = []
    monkeypatch.setattr(dl, "abort_active_download", lambda: aborted.append("abort"))

    app = LilbeeAppHost()
    async with app.run_test():
        controller = TaskBarController(app)
        # Straight to the queue: advance() promotes to ACTIVE without the
        # controller spawning a worker that would outlive the test.
        task_id = controller.queue.enqueue(lambda: None, "Downloading x", TaskType.DOWNLOAD.value)
        controller.queue.advance(TaskType.DOWNLOAD.value)
        task = controller.queue.get_task(task_id)
        assert task is not None and task.status is TaskStatus.ACTIVE

        controller.cancel_task(task_id)

    assert aborted == ["abort"], "a running download was cancelled without aborting it"


async def test_cancelling_a_queued_download_leaves_the_running_one_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The abort is session-wide, so firing it for a row that never started
    would kill whichever transfer is actually running."""
    from lilbee.cli.tui.task_queue import TaskType
    from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController
    from tests._lilbee_app_test_host import LilbeeAppHost

    aborted: list[str] = []
    monkeypatch.setattr(dl, "abort_active_download", lambda: aborted.append("abort"))

    app = LilbeeAppHost()
    async with app.run_test():
        controller = TaskBarController(app)
        queued = controller.queue.enqueue(lambda: None, "Downloading y", TaskType.DOWNLOAD.value)

        controller.cancel_task(queued)

    assert aborted == [], "cancelling a queued row aborted the active transfer"


def test_an_unrelated_runtime_error_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cancellation branch must not swallow real failures."""
    config = dl.DownloadConfig(
        repo_id="acme/x-GGUF", filename="x.gguf", token=None, cache_dir=str(tmp_path)
    )

    def _boom(**_kw: Any) -> str:
        raise RuntimeError("something genuinely broken")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)

    with pytest.raises(RuntimeError, match="something genuinely broken"):
        dl._hf_download_or_translate(_entry(), config)
