"""The registry that maps the id of a running add to the run a cancel stops."""

from __future__ import annotations

import pytest

from lilbee.runtime.add_registry import AddIdRunningError, AddRegistry


class _Run:
    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


def test_an_id_runs_while_its_block_runs_and_not_after():
    registry = AddRegistry()
    with registry.running("a1", _Run().stop):
        assert registry.is_running("a1")
    assert not registry.is_running("a1")


def test_an_id_is_free_again_after_a_block_that_raised():
    registry = AddRegistry()
    with pytest.raises(RuntimeError), registry.running("a1", _Run().stop):
        raise RuntimeError("the add failed")
    assert not registry.is_running("a1")


def test_a_cancel_calls_the_stop_of_its_id():
    registry, run = AddRegistry(), _Run()
    with registry.running("a1", run.stop):
        assert registry.cancel("a1") is True
    assert run.stops == 1


def test_a_cancel_of_an_id_that_does_not_run_stops_nothing():
    registry, run = AddRegistry(), _Run()
    with registry.running("a1", run.stop):
        assert registry.cancel("other") is False
    assert registry.cancel("a1") is False
    assert run.stops == 0


def test_an_add_with_no_id_holds_nothing_a_cancel_can_reach():
    registry, run = AddRegistry(), _Run()
    with registry.running(None, run.stop):
        assert registry._stops == {}
    assert run.stops == 0


def test_a_second_add_with_a_running_id_is_refused_and_keeps_the_first():
    registry, first = AddRegistry(), _Run()
    with registry.running("a1", first.stop):
        with (
            pytest.raises(AddIdRunningError, match="'a1' is already running"),
            registry.running("a1", _Run().stop),
        ):
            pass
        assert registry.cancel("a1") is True
    assert first.stops == 1
