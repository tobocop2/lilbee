"""Catalog refuses to re-request a pull that is already queued or running."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.catalog.download import ListingFailedError, ListingOfflineError
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask
from lilbee.cli.tui.task_queue import TaskQueue, TaskType
from lilbee.cli.tui.widgets.task_bar_controller import download_key
from tests._lilbee_app_test_host import LilbeeAppHost


def _live(q: TaskQueue) -> int:
    return len(q.active_tasks) + len(q.queued_tasks)


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
        # Otherwise this asserts on the runner's free space, not on the guard.
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.catalog.disk_shortfall", lambda *_a, **_kw: None
        )

        screen._install_model(_model())

        assert enqueued == ["acme/already-GGUF"]


async def test_a_pull_too_big_for_the_disk_never_becomes_a_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two presses on an over-large model must leave the Task Center empty.

    The worker-side check refuses on the download's first instruction, so the
    task is already terminal by the next keypress and dedupe cannot stop a
    second identical failed row. Refusing before enqueue is what keeps them out.
    """
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        pilot.app.task_bar.queue = TaskQueue()

        big = dataclasses.replace(_model(), size_gb=24.2)

        enqueued: list[str] = []
        monkeypatch.setattr(screen, "_enqueue_download", lambda m: enqueued.append(m.hf_repo))
        notices: list[str] = []
        monkeypatch.setattr(screen, "notify", lambda message, **_kw: notices.append(str(message)))
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.catalog.preflight",
            lambda *_a, **_kw: (
                "Not enough disk space for acme/already-GGUF: needs 24.2 GB, 0.2 GB free."
            ),
        )

        screen._install_model(big)
        screen._install_model(big)

        assert enqueued == []
        assert _live(pilot.app.task_bar.queue) == 0
        assert len(notices) == 2
        assert all("Not enough disk space" in n for n in notices)


async def test_the_disk_check_charges_the_pull_plan_not_the_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row's size approximates off the parameter count.

    A refusal is a decision, so it runs against the bytes HuggingFace reports
    for the files the download itself fetches, rather than the browse figure.
    """
    from lilbee.catalog import download
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        pilot.app.task_bar.queue = TaskQueue()

        plan = download.PullPlan(
            "acme/already-GGUF",
            (download.PlannedFile("acme-Q4_K_M.gguf", 9_000_000_000, None, None),),
        )
        monkeypatch.setattr("lilbee.cli.tui.screens.catalog.plan_pull", lambda *_a: plan)
        asked: list[int] = []
        monkeypatch.setattr(
            download, "disk_shortfall", lambda _dir, _repo, needed: asked.append(needed) or None
        )
        monkeypatch.setattr(screen, "_enqueue_download", lambda _m: None)
        monkeypatch.setattr(screen, "notify", lambda message, **_kw: None)

        screen._install_model(dataclasses.replace(_model(), size_gb=4.3))

        assert asked == [9_000_000_000]


@pytest.mark.parametrize("error", [ListingOfflineError, ListingFailedError])
async def test_an_unlistable_repo_leaves_the_rows_own_figure(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    """Without the listing, the approximation is the only figure there is, and it still guards."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        pilot.app.task_bar.queue = TaskQueue()

        def _unlistable(*_a: object) -> object:
            raise error("offline")

        asked: list[int] = []
        monkeypatch.setattr("lilbee.cli.tui.screens.catalog.plan_pull", _unlistable)
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.catalog.disk_shortfall",
            lambda _dir, _repo, needed: asked.append(needed) or None,
        )
        enqueued: list[str] = []
        monkeypatch.setattr(screen, "_enqueue_download", lambda m: enqueued.append(m.hf_repo))
        monkeypatch.setattr(screen, "notify", lambda message, **_kw: None)

        screen._install_model(dataclasses.replace(_model(), size_gb=4.0))

        assert asked == [int(4.0 * 1024**3)]
        assert enqueued == ["acme/already-GGUF"]


async def test_a_plan_the_download_would_refuse_is_shown_and_not_enqueued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gated repo, an empty selection or a missing shard fails the download at once."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        pilot.app.task_bar.queue = TaskQueue()

        def _gated(*_a: object) -> object:
            raise PermissionError("acme/already-GGUF requires HuggingFace authentication.")

        monkeypatch.setattr("lilbee.cli.tui.screens.catalog.plan_pull", _gated)
        enqueued: list[str] = []
        notices: list[tuple[str, str]] = []
        monkeypatch.setattr(screen, "_enqueue_download", lambda m: enqueued.append(m.hf_repo))
        monkeypatch.setattr(
            screen,
            "notify",
            lambda message, severity="information", **_kw: notices.append((message, severity)),
        )

        screen._install_model(_model())

        assert enqueued == []
        assert notices == [("acme/already-GGUF requires HuggingFace authentication.", "error")]


async def test_a_plan_the_disk_already_holds_reports_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lilbee.catalog import download
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        pilot.app.task_bar.queue = TaskQueue()

        held = download.PlannedFile("model.gguf", 1, None, tmp_path / "model.gguf")
        plan = download.PullPlan("acme/already-GGUF", (held,))
        monkeypatch.setattr("lilbee.cli.tui.screens.catalog.plan_pull", lambda *_a: plan)
        enqueued: list[str] = []
        notices: list[str] = []
        monkeypatch.setattr(screen, "_enqueue_download", lambda m: enqueued.append(m.hf_repo))
        monkeypatch.setattr(screen, "notify", lambda message, **_kw: notices.append(message))

        screen._install_model(_model())

        assert enqueued == []
        assert len(notices) == 1 and "already installed" in notices[0].lower()


async def test_enqueue_download_notifies_a_bracketed_name_literally() -> None:
    """The model's display name is derived from a user-chosen HF repo id."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()

        pilot.app.task_bar.start_download = lambda *_a, **_kw: None
        notified: list[str] = []
        screen.notify = lambda message, **_kw: notified.append(message)

        model = dataclasses.replace(_model(), hf_repo="acme/note[draft]-GGUF")
        screen._enqueue_download(model)

        assert len(notified) == 1
        assert "note[draft]" in notified[0]


async def test_unsupported_confirm_enqueue_notifies_a_bracketed_name_literally() -> None:
    """The unsupported-arch confirm path's queued toast carries the same bracketed name.

    ``_enqueue_download`` has two toast sites for the same message: one inline
    for a normal download, one inside the confirm dialog's callback for an
    unsupported architecture.
    """
    from lilbee.catalog.types import ModelCompat
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()

        pilot.app.task_bar.start_download = lambda *_a, **_kw: None
        notified: list[str] = []
        screen.notify = lambda message, **_kw: notified.append(message)

        with patch.object(screen.app, "push_screen") as mock_push:
            model = dataclasses.replace(
                _model(), hf_repo="acme/note[draft]-GGUF", compat=ModelCompat.UNSUPPORTED
            )
            screen._enqueue_download(model)
            on_confirm = mock_push.call_args[0][1]
            on_confirm(True)

        assert len(notified) == 1
        assert "note[draft]" in notified[0]
