"""Tests for the poll-based Task Center render path."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.coordinate import Coordinate
from textual.widgets import DataTable

from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.task_center import TaskCenter

_PROGRESS_COLUMN_INDEX = 3


def _progress_cell_text(table: DataTable, row_index: int = 0) -> str:
    renderable = table.get_cell_at(Coordinate(row_index, _PROGRESS_COLUMN_INDEX))
    return renderable.plain


@pytest.mark.asyncio
async def test_poll_adds_new_rows_for_enqueued_tasks() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        app.task_bar.add_task("Download X", "download")
        app.task_bar.queue.advance("download")
        await pilot.pause(delay=0.15)
        table = screen.query_one("#task-table", DataTable)
        assert table.row_count == 1


@pytest.mark.asyncio
async def test_poll_updates_progress_cell_in_place() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.add_task("Download Y", "download")
        app.task_bar.queue.advance("download")
        app.task_bar.update_task(tid, 37.5, "10/25 MB")
        await pilot.pause(delay=0.15)
        table = screen.query_one("#task-table", DataTable)
        assert "37.5%" in _progress_cell_text(table)


@pytest.mark.asyncio
async def test_poll_removes_stale_rows() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.add_task("Ephemeral", "download")
        app.task_bar.queue.advance("download")
        await pilot.pause(delay=0.15)
        table = screen.query_one("#task-table", DataTable)
        assert table.row_count == 1
        app.task_bar.queue.remove_task(tid)
        await pilot.pause(delay=0.15)
        assert table.row_count == 0


@pytest.mark.asyncio
async def test_poll_renders_indeterminate_for_indeterminate_tasks() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.add_task("Working", "sync", indeterminate=True)
        app.task_bar.queue.advance("sync")
        app.task_bar.update_task(tid, 0, "starting", indeterminate=True)
        await pilot.pause(delay=0.15)
        table = screen.query_one("#task-table", DataTable)
        assert "%" not in _progress_cell_text(table)


@pytest.mark.asyncio
async def test_poll_survives_remove_row_exception() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.add_task("Gone", "sync")
        app.task_bar.queue.advance("sync")
        await pilot.pause(delay=0.15)
        app.task_bar.queue.remove_task(tid)
        table = screen.query_one("#task-table", DataTable)
        with patch.object(table, "remove_row", side_effect=RuntimeError("boom")):
            await pilot.pause(delay=0.15)  # must not raise


@pytest.mark.asyncio
async def test_poll_survives_update_cell_exception() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.add_task("Flaky", "sync")
        app.task_bar.queue.advance("sync")
        await pilot.pause(delay=0.15)
        app.task_bar.update_task(tid, 50.0, "halfway")
        table = screen.query_one("#task-table", DataTable)
        with patch.object(table, "update_cell", side_effect=RuntimeError("boom")):
            await pilot.pause(delay=0.15)  # must not raise


@pytest.mark.asyncio
async def test_poll_survives_detail_refresh_exception() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        app.task_bar.add_task("Irrelevant", "sync")
        app.task_bar.queue.advance("sync")
        await pilot.pause(delay=0.15)
        table = screen.query_one("#task-table", DataTable)
        with patch.object(table, "coordinate_to_cell_key", side_effect=RuntimeError("boom")):
            await pilot.pause(delay=0.15)  # must not raise


def test_shrink_hf_download_chunk_size_missing_module() -> None:
    """The chunk-size shrink helper ignores missing huggingface_hub gracefully."""
    import builtins

    from lilbee import _shrink_hf_download_chunk_size

    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object):
        if name == "huggingface_hub":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=blocked_import):
        _shrink_hf_download_chunk_size()  # must not raise even on ImportError
