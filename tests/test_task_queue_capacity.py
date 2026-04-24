"""Per-type capacity for TaskQueue: default 1, configurable per type."""

from __future__ import annotations

from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus


def _noop() -> None:
    return None


def test_default_capacity_is_one_active_per_type() -> None:
    """With no capacity override, only one task per type can be active."""
    q = TaskQueue()
    t1 = q.enqueue(_noop, "a", "sync")
    t2 = q.enqueue(_noop, "b", "sync")

    assert q.advance("sync") is not None
    assert q.advance("sync") is None
    assert q.get_task(t1).status == TaskStatus.ACTIVE
    assert q.get_task(t2).status == TaskStatus.QUEUED


def test_capacity_two_allows_two_active() -> None:
    """Capacity 2 for download allows two tasks of that type to be active at once."""
    q = TaskQueue(capacity={"download": 2})
    q.enqueue(_noop, "a", "download")
    q.enqueue(_noop, "b", "download")

    first = q.advance("download")
    second = q.advance("download")

    assert first is not None
    assert second is not None
    assert len(q.active_tasks) == 2


def test_advance_beyond_capacity_returns_none() -> None:
    """A third download stays QUEUED while the first two are active."""
    q = TaskQueue(capacity={"download": 2})
    ids = [q.enqueue(_noop, f"m{i}", "download") for i in range(3)]

    q.advance("download")
    q.advance("download")
    assert q.advance("download") is None

    statuses = [q.get_task(tid).status for tid in ids]
    assert statuses.count(TaskStatus.ACTIVE) == 2
    assert statuses.count(TaskStatus.QUEUED) == 1


def test_completing_active_frees_slot_for_next_advance() -> None:
    """Once an active task finishes, the next queued task can advance."""
    q = TaskQueue(capacity={"download": 2})
    a = q.enqueue(_noop, "a", "download")
    q.enqueue(_noop, "b", "download")
    c = q.enqueue(_noop, "c", "download")

    q.advance("download")
    q.advance("download")
    assert q.advance("download") is None

    q.complete_task(a)
    promoted = q.advance("download")

    assert promoted is not None
    assert promoted.task_id == c
    assert q.get_task(c).status == TaskStatus.ACTIVE


def test_is_empty_false_while_any_active_slot_has_task() -> None:
    """After all active sets are populated, is_empty is False."""
    q = TaskQueue(capacity={"download": 2})
    q.enqueue(_noop, "a", "download")
    q.advance("download")
    assert q.is_empty is False


def test_active_task_returns_one_of_the_active_set() -> None:
    """active_task returns any one active task (back-compat property)."""
    q = TaskQueue(capacity={"download": 2})
    a = q.enqueue(_noop, "a", "download")
    b = q.enqueue(_noop, "b", "download")
    q.advance("download")
    q.advance("download")

    task = q.active_task
    assert task is not None
    assert task.task_id in {a, b}


def test_active_tasks_returns_all_active() -> None:
    """active_tasks returns the full union of active tasks across types."""
    q = TaskQueue(capacity={"download": 2})
    q.enqueue(_noop, "a", "download")
    q.enqueue(_noop, "b", "download")
    q.enqueue(_noop, "c", "sync")
    q.advance("download")
    q.advance("download")
    q.advance("sync")

    names = {t.name for t in q.active_tasks}
    assert names == {"a", "b", "c"}


def test_advance_without_task_type_respects_capacity() -> None:
    """advance() with no type argument also respects per-type capacity."""
    q = TaskQueue(capacity={"download": 2})
    q.enqueue(_noop, "a", "download")
    q.enqueue(_noop, "b", "download")
    q.enqueue(_noop, "c", "download")

    assert q.advance() is not None
    assert q.advance() is not None
    assert q.advance() is None  # capacity 2 reached
