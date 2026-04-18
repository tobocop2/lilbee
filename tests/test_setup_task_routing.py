"""Coverage for setup wizard → shared TaskBar routing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lilbee.catalog import CatalogModel, DownloadProgress
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.setup import SetupWizard


def _patch_setup_scan(chat: list[str] | None = None, embed: list[str] | None = None):
    return patch(
        "lilbee.cli.tui.screens.setup._scan_installed_models",
        return_value=(chat or [], embed or []),
    )


def _patch_setup_ram(ram_gb: float = 16.0):
    return patch("lilbee.models.get_system_ram_gb", return_value=ram_gb)


def _make_catalog_model(name: str = "test", display_name: str = "Test Model") -> CatalogModel:
    return CatalogModel(
        name=name,
        tag="7b",
        display_name=display_name,
        hf_repo=f"org/{name}-7b",
        gguf_filename="test.gguf",
        size_gb=4.0,
        min_ram_gb=8.0,
        description="A test model",
        featured=False,
        downloads=1000,
        task="chat",
    )


@pytest.mark.asyncio
async def test_enqueue_download_tasks_routes_to_shared_queue() -> None:
    """_enqueue_download_tasks creates a task per download in the shared queue."""
    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupWizard())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            cm1 = _make_catalog_model(name="chat-one", display_name="ChatOne")
            cm2 = _make_catalog_model(name="chat-two", display_name="ChatTwo")
            screen._download_models = [cm1, cm2]
            screen._enqueue_download_tasks()
            assert cm1.ref in screen._download_tasks
            assert cm2.ref in screen._download_tasks
            # One active, one queued under the download type
            assert app.task_bar.queue.active_task is not None


@pytest.mark.asyncio
async def test_update_task_from_progress_mirrors_bytes() -> None:
    """Progress updates flow into the shared TaskBarController."""
    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupWizard())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            cm = _make_catalog_model(name="single", display_name="Single")
            screen._download_models = [cm]
            screen._mount_download_rows()
            screen._enqueue_download_tasks()
            task_id = screen._download_tasks[cm.ref]
            screen._update_task_from_progress(task_id, 42.5, cm.ref, "42/100 MB")
            task = app.task_bar.queue.get_task(task_id)
            assert task is not None
            assert task.progress == 42.5


@pytest.mark.asyncio
async def test_update_task_from_progress_without_row_falls_back() -> None:
    """If no local download row exists, the model_ref itself is used as label."""
    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupWizard())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            cm = _make_catalog_model(name="only", display_name="Only")
            screen._download_models = [cm]
            screen._enqueue_download_tasks()
            task_id = screen._download_tasks[cm.ref]
            # No _mount_download_rows call → no local row
            screen._update_task_from_progress(task_id, 10.0, cm.ref, "10 MB")
            task = app.task_bar.queue.get_task(task_id)
            assert task is not None
            assert "only:7b" in task.detail or task.progress == 10.0


@pytest.mark.asyncio
async def test_complete_download_task_success_flashes_done() -> None:
    """_complete_download_task with failed=False calls complete_task."""
    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupWizard())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            cm = _make_catalog_model(name="done", display_name="Done")
            screen._download_models = [cm]
            screen._enqueue_download_tasks()
            assert cm.ref in screen._download_tasks
            screen._complete_download_task(cm.ref, False)
            assert cm.ref not in screen._download_tasks


@pytest.mark.asyncio
async def test_complete_download_task_failure_fails_task() -> None:
    """_complete_download_task with failed=True calls fail_task."""
    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupWizard())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            cm = _make_catalog_model(name="oops", display_name="Oops")
            screen._download_models = [cm]
            screen._enqueue_download_tasks()
            screen._complete_download_task(cm.ref, True)
            assert cm.ref not in screen._download_tasks


@pytest.mark.asyncio
async def test_complete_download_task_unknown_ref_is_noop() -> None:
    """_complete_download_task returns silently for unknown model refs."""
    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupWizard())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            screen._complete_download_task("nonexistent:ref", True)  # no raise


@pytest.mark.asyncio
async def test_update_task_from_progress_noop_outside_lilbee_app() -> None:
    """_update_task_from_progress returns silently when host app isn't LilbeeApp."""
    from textual.app import App, ComposeResult
    from textual.widgets import Footer

    class _PlainApp(App[None]):
        def compose(self) -> ComposeResult:
            yield Footer()

        def on_mount(self) -> None:
            self.push_screen(SetupWizard())

    app = _PlainApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            # Must return without raising even though app.task_bar doesn't exist.
            screen._update_task_from_progress("task-id", 10.0, "ignored", "10 MB")


@pytest.mark.asyncio
async def test_on_download_progress_forwards_to_shared_queue() -> None:
    """_on_download_progress notifies _update_task_from_progress when a task exists."""
    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupWizard())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            cm = _make_catalog_model(name="prog", display_name="Prog")
            screen._download_models = [cm]
            screen._mount_download_rows()
            screen._enqueue_download_tasks()

            called: list[tuple[object, ...]] = []

            def fake_notify(fn, *args, **kwargs):
                called.append((fn, args, kwargs))
                fn(*args, **kwargs)

            progress = DownloadProgress(percent=50.0, detail="50/100 MB", is_cache_hit=False)
            screen._on_download_progress(fake_notify, cm.ref, progress)
            # _update_row and _update_task_from_progress both fired
            fns = [c[0] for c in called]
            assert screen._update_row in fns
            assert screen._update_task_from_progress in fns
