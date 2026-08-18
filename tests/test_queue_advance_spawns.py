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

# The download slots plus one queued row, so promotion is observable.
_SUBMITTED = 5
_SLOTS = 4
_NAMES = [f"task-{index}" for index in range(_SUBMITTED)]


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
            for name in _NAMES
        }

        for _ in range(60):
            if len(runs) == _SLOTS:
                break
            await pilot.pause()
        assert set(runs) == set(_NAMES[:_SLOTS]), f"expected a full set of slots, got {dict(runs)}"
        last = _NAMES[-1]
        assert controller.queue.get_task(ids[last]).status is TaskStatus.QUEUED  # ty: ignore

        getattr(controller, finish)(ids[_NAMES[0]])

        for _ in range(120):
            if last in runs:
                break
            await pilot.pause()
        gate.set()
        for _ in range(40):
            await pilot.pause()

    assert runs[last] == 1, f"successor never started after {finish}: {dict(runs)}"
    assert all(count == 1 for count in runs.values()), f"task ran twice: {dict(runs)}"


@pytest.mark.timeout(90)
async def test_the_successor_waits_for_the_cancelled_worker_to_exit() -> None:
    """Promotion happens at worker exit: the queued row must not start while the
    cancelled worker is still winding down its transfer."""
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
        ids = {
            name: controller.start_task(name, TaskType.DOWNLOAD, target(name), dedupe_key=name)
            for name in _NAMES
        }

        for _ in range(60):
            if len(started) == _SLOTS:
                break
            await pilot.pause()
        assert set(started) == set(_NAMES[:_SLOTS])

        last = _NAMES[-1]
        controller.cancel_task(ids[_NAMES[0]])
        for _ in range(40):
            await pilot.pause()
        assert last not in started, f"{last} started while its predecessors ran: {started}"

        release.set()
        for _ in range(120):
            if last in started:
                break
            await pilot.pause()

    assert last in started, f"{last} never ran after the workers exited: {started}"


@pytest.mark.timeout(90)
async def test_cancelling_a_queued_row_leaves_the_queue_draining() -> None:
    """A queued row holds no slot, so cancelling it advances nothing, and the
    rows behind it must still run once an active one finishes."""
    names = [*_NAMES, "task-behind"]
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
        ids = {
            name: controller.start_task(name, TaskType.DOWNLOAD, target(name), dedupe_key=name)
            for name in names
        }

        for _ in range(60):
            if len(started) == _SLOTS:
                break
            await pilot.pause()
        assert set(started) == set(names[:_SLOTS])

        controller.cancel_task(ids[names[_SLOTS]])  # queued, never ran
        release.set()

        for _ in range(160):
            if names[-1] in started:
                break
            await pilot.pause()

    assert names[_SLOTS] not in started, f"the cancelled row ran anyway: {started}"
    assert names[-1] in started, f"queue did not drain past the cancelled row: {started}"
