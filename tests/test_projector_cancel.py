"""A task-bar cancel ends the projector fetch of a pull while it waits to try again."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lilbee.app import models as app_models
from lilbee.catalog import download as dl
from lilbee.catalog.models import CatalogModel
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.task_queue import TaskStatus
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import await_chat
from tests.test_tui_engine_choice import _ChatApp

_REPO = "acme/SmolVLM2-GGUF"


def _model() -> CatalogModel:
    return CatalogModel(
        hf_repo=_REPO,
        gguf_filename="m.gguf",
        size_gb=1.0,
        min_ram_gb=2,
        description="d",
        featured=False,
        downloads=0,
        task="chat",
    )


async def test_cancelling_a_pull_during_the_projector_backoff_ends_it_before_the_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[str] = []

    def failing_transfer(entry: CatalogModel, config: dl.DownloadConfig) -> Path:
        requests.append(config.filename)
        raise dl._TransientDownloadError("connection reset")

    model = _model()
    monkeypatch.setattr(dl, "_models_dir", lambda: tmp_path)
    monkeypatch.setattr(dl, "_resolve_mmproj_filename", lambda *a: "mmproj.gguf")
    monkeypatch.setattr(dl, "fetch_remote_file", lambda *a: dl.RemoteFile(size=1, blob="b"))
    monkeypatch.setattr(dl, "_require_disk_space", lambda *a: None)
    monkeypatch.setattr(dl, "_hf_download_or_translate", failing_transfer)
    monkeypatch.setattr(dl, "_RETRY_BACKOFF_SECONDS", 60)
    monkeypatch.setattr("lilbee.catalog.resolve_pull_target", lambda ref: model)
    monkeypatch.setattr(app_models, "_vision_projector_missing", lambda ref: True)
    monkeypatch.setattr(
        app_models.get_services().model_manager.__class__,
        "is_installed",
        lambda self, ref, source: True,
    )
    app = _ChatApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        app.task_bar.start_download(model)
        assert await wait_until(pilot, lambda: len(requests) == 1, timeout=10)
        started = time.monotonic()
        app.task_bar.cancel_task(app.task_bar.queue.active_tasks[0].task_id)
        assert await wait_until(
            pilot,
            lambda: bool(app.task_bar.queue.history) and not app.task_bar._workers,
            timeout=10,
        )
        (row,) = app.task_bar.queue.history
        assert row.status is TaskStatus.CANCELLED
        assert time.monotonic() - started < 5
    assert requests == ["mmproj.gguf"]
    assert list(tmp_path.rglob("*")) == []
