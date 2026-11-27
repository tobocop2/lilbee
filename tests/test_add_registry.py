"""The registry that maps the id of a running add to the run a cancel stops."""

from __future__ import annotations

import pytest

from lilbee.runtime.add_registry import AddIdRunningError, AddRegistry


class _Run:
    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


def test_a_reserved_id_is_held_until_its_release():
    registry, run = AddRegistry(), _Run()
    release = registry.reserve("a1", run.stop)
    assert registry.cancel("a1") is True
    release()
    assert registry.cancel("a1") is False
    assert run.stops == 1


def test_a_cancel_of_an_id_that_does_not_run_stops_nothing():
    registry, run = AddRegistry(), _Run()
    registry.reserve("a1", run.stop)
    assert registry.cancel("other") is False
    assert run.stops == 0


def test_an_add_with_no_id_holds_nothing_a_cancel_can_reach():
    registry, run = AddRegistry(), _Run()
    release = registry.reserve(None, run.stop)
    assert registry._stops == {}
    release()
    assert run.stops == 0


def test_a_second_add_with_a_running_id_is_refused_and_keeps_the_first():
    registry, first = AddRegistry(), _Run()
    registry.reserve("a1", first.stop)
    with pytest.raises(AddIdRunningError, match="'a1' is already running"):
        registry.reserve("a1", _Run().stop)
    assert registry.cancel("a1") is True
    assert first.stops == 1


def test_a_second_release_leaves_the_next_holder_of_the_id_alone():
    registry, first, second = AddRegistry(), _Run(), _Run()
    release_first = registry.reserve("a1", first.stop)
    release_first()
    registry.reserve("a1", second.stop)
    release_first()
    assert registry.cancel("a1") is True
    assert (first.stops, second.stops) == (0, 1)
