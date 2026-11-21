"""What a reload pass leaves behind: no stuck roles, and the preload of the roles it restarted."""

from __future__ import annotations

import threading

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from tests import test_fleet_reload_plan as plan_tests
from tests.test_fleet_reload_plan import (
    _JOIN_S,
    _OTHER_CHAT_REF,
    _EngineSwap,
    _fail_first_starts,
    _join_fleet_threads,
)

# The fixtures are shared by assignment: pytest finds them by module attribute.
engine_fleet = plan_tests.engine_fleet
real_planner = plan_tests.real_planner

CHAT = WorkerRole.CHAT
EMBED = WorkerRole.EMBED
FleetFixture = tuple[FleetProvider, list[_EngineSwap]]

_RELOAD_TIMEOUT_TEXT = (
    "The embed model server is still restarting after a settings change. "
    "Wait for the restart to finish, then try again."
)


@pytest.mark.parametrize("start_fails", [False, True], ids=["pass-succeeds", "pass-fails"])
def test_a_request_after_a_reload_pass_for_a_role_with_no_server_hears_the_reload_timeout(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch, start_fails: bool
) -> None:
    provider, _swaps = engine_fleet
    if start_fails:
        _fail_first_starts(monkeypatch, 1)
    cfg.chat_model = _OTHER_CHAT_REF
    provider.reload_role(CHAT)
    _join_fleet_threads()
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 0.05)
    with provider._lock:
        provider._clients.pop(EMBED, None)
        provider._reload_owed = frozenset({EMBED})

    with pytest.raises(ProviderError) as heard:
        provider._require_clients(EMBED)

    assert str(heard.value) == _RELOAD_TIMEOUT_TEXT


def test_a_reload_on_a_shared_engine_preloads_the_roles_the_rebind_restarted(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    for swap in provider._swaps.values():
        swap.bound = True
    monkeypatch.setattr(provider, "_rebind_or_overflow", lambda: [CHAT])
    preloaded: list[frozenset[WorkerRole]] = []
    done = threading.Event()

    def preload(*, roles: frozenset[WorkerRole]) -> None:
        preloaded.append(roles)
        done.set()

    monkeypatch.setattr(provider, "_preload_roles", preload)
    cfg.chat_model = _OTHER_CHAT_REF

    provider.reload_role(CHAT)
    _join_fleet_threads()

    assert done.wait(_JOIN_S)
    assert preloaded == [frozenset({CHAT})]
