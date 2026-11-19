"""A failed fleet pass or build that leaves no group still shows a health notice."""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.provider import NO_ENGINE_NOTICE, RELOAD_FAILED_NOTICE, FleetProvider
from lilbee.providers.roles import WorkerRole
from tests.test_engine_write_applied import _local, _ProcessSwap, fleet

__all__ = ["fleet"]

_REFUSED = ProviderError("engine refused", provider="llama-server")


def _warnings(services: Any) -> list[str]:
    return [w.message for w in services.provider.health_warnings()]


def _fail_a_pass_that_drops_every_group(local: FleetProvider) -> None:
    _ProcessSwap.start_failure = _REFUSED
    with local._lock:
        local._forced_roles.update({WorkerRole.CHAT, WorkerRole.EMBED})
    with pytest.raises(ProviderError, match="engine refused"):
        local.replan_fleet(wait=True)
    assert not local._swaps


def test_a_failed_pass_that_drops_every_group_shows_the_no_engine_notice(fleet: Any) -> None:
    local = _local(fleet)
    _fail_a_pass_that_drops_every_group(local)
    assert local.reload_notice() == NO_ENGINE_NOTICE
    notice = local.reload_notice()
    assert notice == "The last engine start failed; no local engine group is running."
    assert NO_ENGINE_NOTICE in _warnings(fleet)
    assert RELOAD_FAILED_NOTICE not in _warnings(fleet)


def test_the_no_engine_notice_clears_when_a_later_build_succeeds(fleet: Any) -> None:
    local = _local(fleet)
    _fail_a_pass_that_drops_every_group(local)
    assert local.reload_notice() == NO_ENGINE_NOTICE
    _ProcessSwap.start_failure = None
    assert local._require_clients(WorkerRole.CHAT)
    assert local.reload_notice() is None


def test_a_failed_build_with_no_group_shows_the_no_engine_notice(fleet: Any) -> None:
    local = _local(fleet)
    local._drop_swap_refs()
    _ProcessSwap.start_failure = _REFUSED
    with pytest.raises(ProviderError, match="engine refused"):
        local._require_clients(WorkerRole.CHAT)
    assert not local._swaps
    assert local.reload_notice() == NO_ENGINE_NOTICE


def test_a_group_dropped_after_an_old_failure_does_not_show_that_failure(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)

    def _no_plan(*, engines: Any) -> None:
        raise _REFUSED

    monkeypatch.setattr(planning, "plan_all_launches", _no_plan)
    with pytest.raises(ProviderError, match="engine refused"):
        local.replan_fleet(wait=True)
    assert local._swaps
    assert local.reload_notice() == RELOAD_FAILED_NOTICE
    with local._lock:
        for group in list(local._swaps):
            local._drop_group(group)
    assert not local._swaps
    assert local.reload_notice() is None


def test_a_failure_that_lands_after_shutdown_shows_no_notice(fleet: Any) -> None:
    local = _local(fleet)
    local.shutdown()
    with local._lock:
        local._end_fleet_work(_REFUSED)
    assert local._reload_failed
    assert local.reload_notice() is None


def test_a_connection_blip_after_an_old_failure_shows_no_notice(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)

    def _no_plan(*, engines: Any) -> None:
        raise _REFUSED

    monkeypatch.setattr(planning, "plan_all_launches", _no_plan)
    with pytest.raises(ProviderError, match="engine refused"):
        local.replan_fleet(wait=True)
    assert local.reload_notice() == RELOAD_FAILED_NOTICE
    local._drop_swap_refs()
    assert not local._swaps
    assert local.reload_notice() is None


def test_a_shut_down_provider_shows_no_notice_after_a_failure(fleet: Any) -> None:
    local = _local(fleet)
    _fail_a_pass_that_drops_every_group(local)
    local.shutdown()
    assert local.reload_notice() is None
