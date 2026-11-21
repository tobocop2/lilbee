"""The errors a request hears when its role has no server: the exact text, the provider and the kind."""

from __future__ import annotations

import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from tests import test_fleet_reload_plan as plan_tests
from tests.test_fleet_reload_gap import _provider
from tests.test_fleet_reload_plan import (
    _DRAIN_LIMIT_S,
    _START_REFUSED,
    _EngineSwap,
    _ask,
    _fail_first_starts,
    _join_fleet_threads,
    _time_out_a_chat,
)

# The fixtures are shared by assignment: pytest finds them by module attribute.
mlx_engine_fleet = plan_tests.mlx_engine_fleet
real_planner = plan_tests.real_planner

_FAILED_START_TEXT = (
    f"The chat model server was restarted and did not start again: {_START_REFUSED}\n"
    "Send your request again to try the start once more. If it fails again, "
    "read the engine log."
)


def test_a_chat_after_a_restart_that_did_not_start_hears_the_exact_failed_start_error(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    _fail_first_starts(monkeypatch, 2, ProviderErrorKind.SERVER)
    _time_out_a_chat(provider)
    _join_fleet_threads()

    heard = _ask(provider)
    _join_fleet_threads()

    assert isinstance(heard, ProviderError)
    assert str(heard) == _FAILED_START_TEXT
    assert heard.provider == "llama-server"
    assert heard.kind is ProviderErrorKind.SERVER


@pytest.mark.parametrize("stuck", [False, True], ids=["reload", "stuck-restart"])
def test_a_request_that_waits_out_a_restart_hears_an_error_from_the_llama_server(
    monkeypatch: pytest.MonkeyPatch, stuck: bool
) -> None:
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 0.05)
    provider = _provider(monkeypatch, owed=frozenset(WorkerRole))
    if stuck:
        provider._stuck_roles = frozenset({WorkerRole.EMBED})

    with pytest.raises(ProviderError) as heard:
        provider._require_clients(WorkerRole.EMBED)

    assert heard.value.provider == "llama-server"
