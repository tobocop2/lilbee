"""A dead swap's rebuild runs as fleet work and its outcome reaches the health warning."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.provider import RELOAD_FAILED_NOTICE, FleetProvider
from lilbee.providers.roles import WorkerRole
from tests.test_engine_write_applied import _local, _wait_for, fleet
from tests.test_fleet_work_orders import _Caller

__all__ = ["fleet"]

_ROLE = WorkerRole.CHAT
_NO_BINARY = ProviderError("The llama-server binary is missing.")


def _warnings(services: Any) -> list[str]:
    return [w.message for w in services.provider.health_warnings()]


def _kill_swap(local: FleetProvider) -> None:
    """The role's swap process has died and its pool is empty."""
    with local._lock:
        swap = local._swaps[local._role_group[_ROLE]]
        local._clients[_ROLE] = []
    swap.is_live = lambda: False


def _missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def _plan(*, engines: Any) -> Any:
        raise _NO_BINARY

    monkeypatch.setattr(planning, "plan_all_launches", _plan)


def test_a_failed_rebuild_shows_in_the_health_warning(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    _kill_swap(local)
    _missing_binary(monkeypatch)
    assert RELOAD_FAILED_NOTICE not in _warnings(fleet)
    with pytest.raises(ProviderError, match="binary is missing"):
        local._require_clients(_ROLE)
    assert local.reload_notice() == RELOAD_FAILED_NOTICE
    assert RELOAD_FAILED_NOTICE in _warnings(fleet)
    with pytest.raises(ProviderError, match="binary is missing"):
        local._require_clients(_ROLE)
    assert RELOAD_FAILED_NOTICE in _warnings(fleet)


def test_two_calls_on_a_dead_swap_start_one_rebuild(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    _kill_swap(local)
    passes: list[int] = []
    entered = threading.Event()
    gate = threading.Event()
    revived = local._clients

    def _pass(force: frozenset[WorkerRole] = frozenset()) -> None:
        passes.append(len(passes))
        with local._lock:
            # The pass has dropped the group: a caller no longer waits for the reload.
            local._reload_owed = frozenset()
            local._reload_done.notify_all()
        entered.set()
        assert gate.wait(10.0)
        with local._lock:
            local._swaps[local._role_group[_ROLE]].is_live = lambda: True
            revived[_ROLE] = [MagicMock()]

    monkeypatch.setattr(local, "_reload_pass", _pass)
    handed: list[Any] = []
    real_start = local._start_fleet_work

    def _recorded() -> Any:
        work = real_start()
        handed.append(work)
        return work

    monkeypatch.setattr(local, "_start_fleet_work", _recorded)
    first = _Caller(lambda: local._require_clients(_ROLE))
    assert entered.wait(10.0)
    second = _Caller(lambda: local._require_clients(_ROLE))
    # Both callers hold the work of their rebuild before the first pass ends.
    _wait_for(lambda: len(handed) == 2)
    gate.set()
    first.join()
    second.join()
    assert (first.heard, second.heard) == (None, None)
    assert handed[0] is handed[1]
    assert len(passes) == 1


def test_a_rebuild_that_succeeds_clears_the_failure(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    _kill_swap(local)
    real_plan = planning.plan_all_launches
    _missing_binary(monkeypatch)
    with pytest.raises(ProviderError, match="binary is missing"):
        local._require_clients(_ROLE)
    assert RELOAD_FAILED_NOTICE in _warnings(fleet)
    monkeypatch.setattr(planning, "plan_all_launches", real_plan)
    assert local._require_clients(_ROLE)
    assert RELOAD_FAILED_NOTICE not in _warnings(fleet)
