"""A fleet hold lives exactly as long as the groups and retired clients that use its dir."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.fleet.swap_manager import SwapGroup
from lilbee.providers.profiles import LLAMA
from lilbee.types import WorkerRole

_DIR = Path("/engine/a")
_OTHER = Path("/engine/b")


def _client() -> LlamaServerClient:
    return LlamaServerClient(
        "http://127.0.0.1:1", "chat-0", profile=LLAMA.request_profile(WorkerRole.CHAT)
    )


def _provider(*dirs: Path) -> tuple[FleetProvider, dict[Path, MagicMock]]:
    p = FleetProvider()
    holds = {d: MagicMock() for d in dirs}
    p._engine_holds = dict(holds)
    return p, holds


def _group() -> SwapGroup:
    return next(iter(SwapGroup))


def _assert_nothing_outlives(p: FleetProvider, holds: dict[Path, MagicMock]) -> None:
    assert p._dir_leases == {}
    assert p._engine_holds == {}
    for hold in holds.values():
        hold.release_and_check_last.assert_called_once()


def test_the_hold_is_released_when_the_last_group_on_the_dir_ends() -> None:
    p, holds = _provider(_DIR)
    with p._lock:
        p._set_group_dir(_group(), _DIR)
    assert p._engine_holds == holds
    with p._lock:
        p._unset_group_dir(_group())
    _assert_nothing_outlives(p, holds)


def test_moving_a_group_to_a_new_dir_releases_only_the_old_dir() -> None:
    p, holds = _provider(_DIR, _OTHER)
    with p._lock:
        p._set_group_dir(_group(), _DIR)
        p._set_group_dir(_group(), _OTHER)
    assert p._engine_holds == {_OTHER: holds[_OTHER]}
    holds[_DIR].release_and_check_last.assert_called_once()
    holds[_OTHER].release_and_check_last.assert_not_called()


def test_rebinding_a_group_in_the_same_dir_never_drops_the_hold() -> None:
    p, holds = _provider(_DIR)
    with p._lock:
        p._set_group_dir(_group(), _DIR)
        p._set_group_dir(_group(), _DIR)
    assert p._dir_leases == {_DIR: 1}
    holds[_DIR].release_and_check_last.assert_not_called()


def test_a_retired_client_with_a_request_in_flight_keeps_the_hold_after_its_group_ends() -> None:
    p, holds = _provider(_DIR)
    client = _client()
    client.reserve()
    with p._lock:
        p._set_group_dir(_group(), _DIR)
        p._retire_clients([client], _DIR)
        p._unset_group_dir(_group())
    holds[_DIR].release_and_check_last.assert_not_called()
    client.release()
    _assert_nothing_outlives(p, holds)
    client.close()


def test_an_idle_retired_client_keeps_no_hold() -> None:
    p, holds = _provider(_DIR)
    client = _client()
    with p._lock:
        p._set_group_dir(_group(), _DIR)
        p._retire_clients([client], _DIR)
        p._unset_group_dir(_group())
    _assert_nothing_outlives(p, holds)
    client.close()


def test_closing_a_retired_client_ends_its_lease_once() -> None:
    p, holds = _provider(_DIR)
    client = _client()
    client.reserve()
    with p._lock:
        p._retire_clients([client], _DIR)
    p._close_client(client)
    client.release()
    _assert_nothing_outlives(p, holds)


def test_giving_a_client_back_to_dispatch_ends_its_lease() -> None:
    p, _holds = _provider(_DIR)
    client = _client()
    client.reserve()
    with p._lock:
        p._set_group_dir(_group(), _DIR)
        p._hold_dir_for(client, _DIR)
        p._retiring_clients = [client]
        p._role_group[WorkerRole.CHAT] = _group()
    p._give_back_clients({WorkerRole.CHAT: [client]})
    assert p._dir_leases == {_DIR: 1}
    client.release()
    assert p._dir_leases == {_DIR: 1}
    client.close()


def test_a_dir_whose_start_runs_keeps_its_hold() -> None:
    p, holds = _provider(_DIR)
    p._starting_dirs.add(_DIR)
    with p._lock:
        p._set_group_dir(_group(), _DIR)
        p._unset_group_dir(_group())
    assert p._engine_holds == holds
    holds[_DIR].release_and_check_last.assert_not_called()


def _connection_error() -> ProviderError:
    return ProviderError("down", kind=ProviderErrorKind.CONNECTION)


def test_a_rediscover_keeps_the_holds_through_the_retry_and_releases_the_dir_it_left(
    monkeypatch,
) -> None:
    p, holds = _provider(_DIR)
    with p._lock:
        p._set_group_dir(_group(), _DIR)
    seen: list[bool] = []
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _connection_error()
        seen.append(not holds[_DIR].release_and_check_last.called)
        with p._lock:
            p._set_group_dir(_group(), _OTHER)
        return "ok"

    assert p._with_rediscover(_call) == "ok"
    assert seen == [True]
    holds[_DIR].release_and_check_last.assert_called_once()
    assert p._dir_leases == {_OTHER: 1}


def test_a_rediscover_that_rebinds_the_dir_keeps_its_hold() -> None:
    p, holds = _provider(_DIR)
    with p._lock:
        p._set_group_dir(_group(), _DIR)
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _connection_error()
        with p._lock:
            p._set_group_dir(_group(), _DIR)
        return "ok"

    assert p._with_rediscover(_call) == "ok"
    holds[_DIR].release_and_check_last.assert_not_called()


def test_a_failed_rediscover_retry_still_ends_its_leases() -> None:
    p, holds = _provider(_DIR)
    with p._lock:
        p._set_group_dir(_group(), _DIR)

    def _call() -> str:
        raise _connection_error()

    with pytest.raises(ProviderError):
        p._with_rediscover(_call)
    with p._lock:
        p._unset_group_dir(_group())
    _assert_nothing_outlives(p, holds)


def test_a_transport_error_rediscover_also_scopes_its_leases() -> None:
    p, _holds = _provider(_DIR)
    with p._lock:
        p._set_group_dir(_group(), _DIR)

    def _call() -> str:
        raise httpx.ConnectError("refused")

    with pytest.raises(httpx.ConnectError):
        p._with_rediscover(_call)
    assert p._dir_leases == {}


def test_the_hold_sweep_machinery_is_gone() -> None:
    assert not hasattr(prov_mod, "_HOLD_SWEEP_THREAD")
    assert not hasattr(FleetProvider, "_release_unused_holds")
    assert not hasattr(FleetProvider(), "_open_rediscovers")
