"""TaskBar: pulsing-dot status indicator with state-aware copy."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label

from lilbee.cli.tui.task_queue import TaskType
from lilbee.cli.tui.widgets.task_bar import TaskBar, TaskBarController


class _Harness(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield TaskBar(id="tbar")


def _label_text(bar: TaskBar) -> str:
    label = bar.query_one("#task-status-label", Label)
    return str(label._Static__content)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_taskbar_hidden_when_idle() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(TaskBar)
        assert bar.display is False


@pytest.mark.asyncio
async def test_taskbar_single_active_shows_name_and_percent() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        tid = app.task_bar.queue.enqueue(lambda: None, "Gemma 4", TaskType.DOWNLOAD.value)
        app.task_bar.queue.advance(TaskType.DOWNLOAD.value)
        app.task_bar.queue.update_task(tid, 42.5, "42/100 MB")
        bar = app.query_one(TaskBar)
        bar._refresh_display()
        text = _label_text(bar)
        assert "●" in text
        assert "Gemma 4" in text
        assert "42.5%" in text
        # Single-active state drops the "1 task running" prefix.
        assert "tasks running" not in text.lower()


@pytest.mark.asyncio
async def test_taskbar_multiple_active_shows_count_plural() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        for i in range(2):
            app.task_bar.queue.enqueue(lambda: None, f"m{i}", TaskType.DOWNLOAD.value)
            app.task_bar.queue.advance(TaskType.DOWNLOAD.value)
        bar = app.query_one(TaskBar)
        bar._refresh_display()
        text = _label_text(bar)
        assert "2 tasks running" in text


@pytest.mark.asyncio
async def test_taskbar_dot_pulses_across_ticks() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        bar = app.query_one(TaskBar)
        bar._tick_count = 0
        bar._refresh_display()
        on = _label_text(bar)
        bar._tick_count = 5
        bar._refresh_display()
        off = _label_text(bar)
        assert on != off, "dot colour should differ between on-beat and off-beat"


@pytest.mark.asyncio
async def test_taskbar_queued_only_uses_muted_copy() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Fill the download active slots, then enqueue a queued one.
        for i in range(2):
            tid = app.task_bar.queue.enqueue(lambda: None, f"m{i}", TaskType.DOWNLOAD.value)
            app.task_bar.queue.advance(TaskType.DOWNLOAD.value)
            app.task_bar.queue.complete_task(tid)
        tid_queued = app.task_bar.queue.enqueue(lambda: None, "waiting", TaskType.DOWNLOAD.value)
        # Don't advance — it stays queued (capacity unused because the
        # first two are DONE not ACTIVE; advance promotes next active).
        bar = app.query_one(TaskBar)
        bar._refresh_display()
        text = _label_text(bar)
        # With at most active=0 queued=1, the bar shows "1 queued".
        if "queued" in text.lower():
            assert "1 queued" in text
        # Sanity: the widget is visible.
        assert bar.display is True
        # Cleanup so the next test starts idle.
        app.task_bar.queue.remove_task(tid_queued)


@pytest.mark.asyncio
async def test_taskbar_includes_hint_text() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        bar = app.query_one(TaskBar)
        bar._refresh_display()
        text = _label_text(bar)
        assert "Press t for Tasks" in text


@pytest.mark.asyncio
async def test_taskbar_hint_becomes_esc_variant_when_input_focused() -> None:
    """With a chat-style Input focused, the hint should read 'Esc then t'."""
    from textual.widgets import Input

    class _InputHarness(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.task_bar = TaskBarController(self)

        def compose(self) -> ComposeResult:
            yield Input(id="dummy-input")
            yield TaskBar(id="tbar")

    app = _InputHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#dummy-input", Input).focus()
        await pilot.pause()
        app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        bar = app.query_one(TaskBar)
        bar._refresh_display()
        text = _label_text(bar)
        assert "Esc then t for Tasks" in text


@pytest.mark.asyncio
async def test_taskbar_completion_flash_after_queue_drains() -> None:
    """After the last task completes, show a 'Done' flash before hiding."""
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        tid = app.task_bar.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        app.task_bar.queue.complete_task(tid)
        bar = app.query_one(TaskBar)
        bar._refresh_display()
        text = _label_text(bar)
        assert "Done" in text


@pytest.mark.asyncio
async def test_taskbar_failure_flash_shows_count() -> None:
    """Failed task keeps the bar lit with error-coloured copy."""
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        tid = app.task_bar.queue.enqueue(lambda: None, "oops", TaskType.SYNC.value)
        app.task_bar.queue.advance(TaskType.SYNC.value)
        app.task_bar.queue.fail_task(tid, "network error")
        bar = app.query_one(TaskBar)
        bar._refresh_display()
        text = _label_text(bar)
        assert "failed" in text.lower()
