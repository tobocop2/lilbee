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
        # bb-18y3: counts strip carries a rotating spinner glyph while any
        # task is active so the header visibly moves.
        from lilbee.cli.tui.screens.task_center import _COUNTS_SPINNER_FRAMES

        assert any(frame in counts for frame in _COUNTS_SPINNER_FRAMES)


@pytest.mark.asyncio
async def test_poll_counts_strip_has_no_spinner_when_idle() -> None:
    """bb-18y3: spinner only shows while tasks are active, not when idle."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        counts = str(
            screen.query_one("#task-center-counts")._Static__content  # type: ignore[attr-defined]
        )
        from lilbee.cli.tui.screens.task_center import _COUNTS_SPINNER_FRAMES

        assert not any(frame in counts for frame in _COUNTS_SPINNER_FRAMES)


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
async def test_clear_history_action_drops_finished_rows() -> None:
    """Shift+C clears DONE/FAILED/CANCELLED rows and leaves active ones alone."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        done_id = app.task_bar.queue.enqueue(lambda: None, "done", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        app.task_bar.queue.complete_task(done_id)
        screen._poll()
        await pilot.pause()
        assert done_id in screen._rows
        screen.action_clear_history()
        await pilot.pause()
        assert done_id not in screen._rows


@pytest.mark.asyncio
async def test_empty_state_visibility_follows_queue() -> None:
    """Empty headline shows when the queue is empty; #task-rows hides.

    Regression test for bb-xd7m: the empty-state Label and the row
    scroll share the same 1fr slot. Toggling ``display`` on both so
    exactly one is visible keeps the headline centred instead of
    floating under a ghost scroll.
    """
    from textual.containers import VerticalScroll

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        rows = screen.query_one("#task-rows", VerticalScroll)
        empty = screen.query_one("#task-center-empty")

        # Empty queue: rows hidden, empty state visible.
        assert empty.display is True
        assert rows.display is False

        tid = app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        for _ in range(5):
            await pilot.pause(delay=0.1)
            if tid in screen._rows:
                break
        # With a task: rows shown, empty state hidden.
        assert empty.display is False
        assert rows.display is True


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
async def test_initial_focus_lands_on_first_active_row() -> None:
    """entering Task Center focuses the first active/queued row,
    not whatever DONE row happens to be at position 0."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        done_id = app.task_bar.queue.enqueue(lambda: None, "old", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        app.task_bar.queue.complete_task(done_id)
        active_id = app.task_bar.queue.enqueue(lambda: None, "live", TaskType.CRAWL.value)
        app.task_bar.queue.advance(TaskType.CRAWL.value)
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        # Both rows must have mounted, and the active row gets focus.
        for _ in range(5):
            await pilot.pause(delay=0.1)
            if (
                active_id in screen._rows
                and done_id in screen._rows
                and isinstance(app.focused, TaskRow)
            ):
                break
        assert isinstance(app.focused, TaskRow)
        assert app.focused is screen._rows[active_id]


@pytest.mark.asyncio
async def test_initial_focus_prefers_queued_when_no_active() -> None:
    """if the only live row is QUEUED, focus lands there instead
    of on a DONE history row."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        done_id = app.task_bar.queue.enqueue(lambda: None, "old", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        app.task_bar.queue.complete_task(done_id)
        # Enqueue but do NOT advance so the task stays QUEUED.
        queued_id = app.task_bar.queue.enqueue(lambda: None, "next", TaskType.CRAWL.value)
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        for _ in range(5):
            await pilot.pause(delay=0.1)
            if queued_id in screen._rows and isinstance(app.focused, TaskRow):
                break
        assert isinstance(app.focused, TaskRow)
        assert app.focused is screen._rows[queued_id]


@pytest.mark.asyncio
async def test_initial_focus_noop_when_no_tasks() -> None:
    """with an empty queue there is no row to focus -- the
    on_mount focus step must not raise."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        # No rows, no crash. AUTO_FOCUS falls back to the scroll container.
        assert screen._rows == {}


@pytest.mark.asyncio
async def test_initial_focus_falls_back_when_only_history_present() -> None:
    """no active/queued work means _focus_initial_row is a no-op
    and AUTO_FOCUS's row-1 landing (a history row) still holds. The
    cancel action is a no-op on that focused terminal row."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        done_id = app.task_bar.queue.enqueue(lambda: None, "old", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        app.task_bar.queue.complete_task(done_id)
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        for _ in range(5):
            await pilot.pause(delay=0.1)
            if done_id in screen._rows:
                break
        screen._rows[done_id].focus()
        await pilot.pause()
        screen.action_cancel_task()
        task = app.task_bar.queue.get_task(done_id)
        assert task is not None
        # Status stays DONE: cancel is a no-op on terminal rows.
        assert task.status.value == "done"


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
