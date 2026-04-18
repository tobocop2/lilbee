"""Tests for the poll-based Task Center render path.

After the Bucket 3 redesign the screen is a ``VerticalScroll`` of
``TaskRow`` widgets, not a ``DataTable``. These tests cover the
reconciliation loop, the counts strip, and the ``huggingface_hub``
chunk-size helper's import-error path.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.task_center import TaskCenter
from lilbee.cli.tui.task_queue import TaskType
from lilbee.cli.tui.widgets.task_row import TaskRow


@pytest.mark.asyncio
async def test_poll_mounts_new_row_for_enqueued_task() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        # Let the 10 Hz poll see the new task.
        for _ in range(5):
            await pilot.pause(delay=0.1)
            if tid in screen._rows:
                break
        assert tid in screen._rows
        assert isinstance(screen._rows[tid], TaskRow)


@pytest.mark.asyncio
async def test_poll_updates_existing_row() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        for _ in range(5):
            await pilot.pause(delay=0.1)
            if tid in screen._rows:
                break
        app.task_bar.queue.update_task(tid, 42.0, "halfway")
        for _ in range(5):
            await pilot.pause(delay=0.1)
        # Row survived the update; bar now reads 42.0%.
        row = screen._rows[tid]
        bar = row.query_one("#row-bar")
        assert "42.0%" in str(bar._Static__content)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_poll_removes_rows_for_removed_tasks() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        for _ in range(5):
            await pilot.pause(delay=0.1)
            if tid in screen._rows:
                break
        app.task_bar.queue.remove_task(tid)
        for _ in range(5):
            await pilot.pause(delay=0.1)
            if tid not in screen._rows:
                break
        assert tid not in screen._rows


@pytest.mark.asyncio
async def test_poll_updates_counts_strip() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        app.task_bar.queue.enqueue(lambda: None, "a", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        for _ in range(5):
            await pilot.pause(delay=0.1)
        counts = str(
            screen.query_one("#task-center-counts")._Static__content  # type: ignore[attr-defined]
        )
        assert "1 running" in counts


@pytest.mark.asyncio
async def test_action_cancel_hits_active_when_no_focus() -> None:
    """``c`` with no row focused cancels the first active task."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        screen.action_cancel_task()
        task = app.task_bar.queue.get_task(tid)
        assert task is not None
        assert task.status.value == "cancelled"


@pytest.mark.asyncio
async def test_refresh_action_is_safe_on_empty_queue() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        screen.action_refresh_tasks()  # must not raise


@pytest.mark.asyncio
async def test_cursor_actions_move_focus() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        # These should not raise even when there are no rows.
        screen.action_cursor_down()
        screen.action_cursor_up()


@pytest.mark.asyncio
async def test_go_back_switches_to_chat_on_lilbee_app() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        screen.action_go_back()
        for _ in range(5):
            await pilot.pause()
            if not isinstance(app.screen, TaskCenter):
                break
        assert not isinstance(app.screen, TaskCenter)


@pytest.mark.asyncio
async def test_action_cancel_task_uses_focused_task_row() -> None:
    """``c`` with a TaskRow focused cancels that specific task."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        for _ in range(5):
            await pilot.pause(delay=0.1)
            if tid in screen._rows:
                break
        row = screen._rows[tid]
        row.focus()
        await pilot.pause()
        screen.action_cancel_task()
        task = app.task_bar.queue.get_task(tid)
        assert task is not None
        assert task.status.value == "cancelled"


@pytest.mark.asyncio
async def test_poll_swallows_row_remove_exception() -> None:
    """If a row's ``remove`` raises during reconciliation, the poll survives."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        tid = app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        for _ in range(5):
            await pilot.pause(delay=0.1)
            if tid in screen._rows:
                break
        row = screen._rows[tid]
        with patch.object(row, "remove", side_effect=RuntimeError("boom")):
            app.task_bar.queue.remove_task(tid)
            # Poll should catch the exception and still drop the row from _rows.
            screen._poll()
        assert tid not in screen._rows


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
