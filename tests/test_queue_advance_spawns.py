"""Promoting a task has to start it.

``queue.advance`` only marks the next task ACTIVE. A promotion that skips
``_spawn_task_worker`` leaves a row rendering as a live download with no thread
behind it, and the queue never drains.
"""

from __future__ import annotations

import collections
import threading

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.cli.tui.task_queue import TaskStatus, TaskType
from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController
from tests._lilbee_app_test_host import LilbeeAppHost


class _Host(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host")


@pytest.mark.timeout(90)
@pytest.mark.parametrize("finish", ["cancel_task", "complete_task", "fail_task"])
async def test_finishing_a_task_starts_exactly_one_successor(
    monkeypatch: pytest.MonkeyPatch, finish: str
) -> None:
    """All three terminal paths share ``_advance_all``, so all three must promote.

    Counting runs per task also pins the opposite failure: a successor spawned
    twice would run two transfers for one row.
    """
    monkeypatch.setattr(
        "lilbee.cli.tui.widgets.task_bar_controller.abort_active_download",
        lambda: None,
        raising=False,
    )
    runs: collections.Counter[str] = collections.Counter()
    gate = threading.Event()

    def target(name: str):
        def _run(reporter: object) -> None:
            runs[name] += 1
            gate.wait(timeout=20)

        return _run

    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        ids = {
            name: controller.start_task(name, TaskType.DOWNLOAD, target(name), dedupe_key=name)
            for name in ("A", "B", "C")
        }

        for _ in range(60):
            if runs:
                break
            await pilot.pause()
        assert dict(runs) == {"A": 1}, f"expected only A running, got {dict(runs)}"
        assert controller.queue.get_task(ids["B"]).status is TaskStatus.QUEUED  # ty: ignore

        getattr(controller, finish)(ids["A"])

        for _ in range(120):
            if len(runs) > 1:
                break
            await pilot.pause()
        gate.set()
        for _ in range(40):
            await pilot.pause()

    assert runs["B"] == 1, f"successor never started after {finish}: {dict(runs)}"
    assert all(count == 1 for count in runs.values()), f"task ran twice: {dict(runs)}"


@pytest.mark.timeout(90)
async def test_the_successor_waits_for_the_cancelled_worker_to_exit() -> None:
    """Promoting on the cancel keypress overlaps the two transfers, and the xet
    abort is session-wide, so the successor dies with the cancelled one. The
    worker's own exit is what may advance the queue."""
    started: list[str] = []
    release = threading.Event()

    def target(name: str):
        def _run(reporter: object) -> None:
            started.append(name)
            release.wait(timeout=30)

        return _run

    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        first = controller.start_task("A", TaskType.DOWNLOAD, target("A"), dedupe_key="a")
        controller.start_task("B", TaskType.DOWNLOAD, target("B"), dedupe_key="b")

        for _ in range(60):
            if started:
                break
            await pilot.pause()
        assert started == ["A"]

        controller.cancel_task(first)
        for _ in range(40):
            await pilot.pause()
        assert started == ["A"], f"B started while A was still running: {started}"

        release.set()
        for _ in range(120):
            if len(started) > 1:
                break
            await pilot.pause()

    assert started == ["A", "B"], f"B never ran after A exited: {started}"
