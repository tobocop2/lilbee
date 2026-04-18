"""TaskRow widget: three-line renderer used by Task Center."""

from __future__ import annotations

import time

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll

from lilbee.cli.tui.task_queue import Task, TaskStatus
from lilbee.cli.tui.widgets.task_row import TaskRow, _format_elapsed


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="host")


def _task(
    *,
    status: TaskStatus = TaskStatus.ACTIVE,
    progress: float = 0.0,
    detail: str = "",
    indeterminate: bool = False,
    started_at: float | None = None,
    name: str = "demo",
    task_type: str = "download",
) -> Task:
    return Task(
        task_id="t1",
        name=name,
        task_type=task_type,
        fn=lambda: None,
        status=status,
        progress=progress,
        detail=detail,
        indeterminate=indeterminate,
        started_at=started_at,
    )


def test_format_elapsed_shows_mmss_after_start() -> None:
    started = time.monotonic() - 65
    assert _format_elapsed(started, TaskStatus.ACTIVE) == "01:05"


def test_format_elapsed_returns_queued_label_for_queued_task() -> None:
    assert _format_elapsed(None, TaskStatus.QUEUED) == "queued"


def test_format_elapsed_empty_for_active_without_started_at() -> None:
    assert _format_elapsed(None, TaskStatus.ACTIVE) == ""


@pytest.mark.asyncio
async def test_update_applies_active_class_and_bar_percent() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        host = app.query_one("#host", VerticalScroll)
        row = TaskRow("t1")
        await host.mount(row)
        for _ in range(3):
            await pilot.pause()
        row.update(_task(status=TaskStatus.ACTIVE, progress=42.0, detail="halfway"), 0)
        assert row.has_class("-active")
        bar_widget = row.query_one("#row-bar")
        bar = str(bar_widget._Static__content)  # type: ignore[attr-defined]
        assert "42.0%" in bar
        # On tick 0 the pulse is on (even beat).
        assert row.has_class("-pulse")


@pytest.mark.asyncio
async def test_update_pulse_toggles_on_active_across_ticks() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        host = app.query_one("#host", VerticalScroll)
        row = TaskRow("t1")
        await host.mount(row)
        for _ in range(3):
            await pilot.pause()
        row.update(_task(status=TaskStatus.ACTIVE), 0)
        on_beat = row.has_class("-pulse")
        row.update(_task(status=TaskStatus.ACTIVE), 5)
        off_beat = row.has_class("-pulse")
        assert on_beat != off_beat, "rail should pulse between even/odd 5-tick halves"


@pytest.mark.asyncio
async def test_update_never_pulses_queued_task() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        host = app.query_one("#host", VerticalScroll)
        row = TaskRow("t1")
        await host.mount(row)
        for _ in range(3):
            await pilot.pause()
        row.update(_task(status=TaskStatus.QUEUED), 0)
        row.update(_task(status=TaskStatus.QUEUED), 5)
        assert not row.has_class("-pulse")
        assert row.has_class("-queued")


@pytest.mark.asyncio
async def test_update_uses_indeterminate_cell_when_flagged() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        host = app.query_one("#host", VerticalScroll)
        row = TaskRow("t1")
        await host.mount(row)
        for _ in range(3):
            await pilot.pause()
        row.update(_task(indeterminate=True, status=TaskStatus.ACTIVE), 0)
        bar_widget = row.query_one("#row-bar")
        bar = str(bar_widget._Static__content)  # type: ignore[attr-defined]
        assert "%" not in bar


@pytest.mark.asyncio
async def test_update_shows_state_classes_for_every_status() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        host = app.query_one("#host", VerticalScroll)
        row = TaskRow("t1")
        await host.mount(row)
        for _ in range(3):
            await pilot.pause()
        for status, cls in [
            (TaskStatus.QUEUED, "-queued"),
            (TaskStatus.ACTIVE, "-active"),
            (TaskStatus.DONE, "-done"),
            (TaskStatus.FAILED, "-failed"),
            (TaskStatus.CANCELLED, "-cancelled"),
        ]:
            row.update(_task(status=status), 0)
            assert row.has_class(cls), f"row missing {cls} after update with status {status}"


@pytest.mark.asyncio
async def test_flash_completed_adds_class() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        host = app.query_one("#host", VerticalScroll)
        row = TaskRow("t1")
        await host.mount(row)
        for _ in range(3):
            await pilot.pause()
        row.flash_completed()
        assert row.has_class("-just-completed")
