"""Catalog refuses to re-request a pull that is already queued or running."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask
from lilbee.cli.tui.task_queue import TaskQueue, TaskType
from lilbee.cli.tui.widgets.task_bar_controller import download_key
from tests._lilbee_app_test_host import LilbeeAppHost


def _model() -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/already-GGUF",
        gguf_filename="model.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
    )


class _App(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host")


async def test_install_of_a_pending_model_notifies_and_does_not_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second install press must stop at the message, never reaching the queue."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()

        model = _model()
        # A real pending row, so the screen reads the same state the user saw.
        queue = TaskQueue()
        queue.enqueue(
            lambda: None,
            model.display_name,
            TaskType.DOWNLOAD.value,
            dedupe_key=download_key(model),
        )
        pilot.app.task_bar.queue = queue

        enqueued: list[str] = []
        monkeypatch.setattr(screen, "_enqueue_download", lambda m: enqueued.append(m.hf_repo))
        notices: list[str] = []
        monkeypatch.setattr(screen, "notify", lambda message, **_kw: notices.append(str(message)))

        screen._install_model(model)

        assert enqueued == []
        assert any("already downloading" in n for n in notices)


async def test_install_of_an_unqueued_model_still_enqueues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not swallow the normal path: an empty queue installs."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()

        pilot.app.task_bar.queue = TaskQueue()

        enqueued: list[str] = []
        monkeypatch.setattr(screen, "_enqueue_download", lambda m: enqueued.append(m.hf_repo))
        monkeypatch.setattr(screen, "notify", lambda message, **_kw: None)

        screen._install_model(_model())

        assert enqueued == ["acme/already-GGUF"]
