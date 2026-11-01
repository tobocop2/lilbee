"""A request that lands while a reload restarts its role's server waits for the new server."""

from __future__ import annotations

import threading
import time

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole


def _provider(*, reloading: bool) -> FleetProvider:
    provider = FleetProvider.__new__(FleetProvider)
    provider._lock = threading.RLock()
    provider._reload_done = threading.Condition(provider._lock)
    provider._reloading = reloading
    provider._clients = {}
    provider._role_group = {}
    provider._swaps = {}
    provider._skipped_unusable_ctx = {}
    provider._ensure_fleet = lambda: True
    return provider


def test_embed_request_waits_for_the_reload_that_is_restarting_the_server() -> None:
    provider = _provider(reloading=True)
    client = object()

    def finish_reload() -> None:
        time.sleep(0.3)
        with provider._lock:
            provider._clients[WorkerRole.EMBED] = [client]
            provider._reloading = False
            provider._reload_done.notify_all()

    threading.Thread(target=finish_reload, daemon=True).start()
    assert provider._require_clients(WorkerRole.EMBED) == [client]


def test_request_raises_when_the_reload_ends_without_a_server_for_the_role() -> None:
    provider = _provider(reloading=True)

    def finish_reload() -> None:
        time.sleep(0.2)
        with provider._lock:
            provider._reloading = False
            provider._reload_done.notify_all()

    threading.Thread(target=finish_reload, daemon=True).start()
    with pytest.raises(ProviderError, match="No embed model server is running"):
        provider._require_clients(WorkerRole.EMBED)


def test_request_stops_waiting_when_the_reload_never_ends(monkeypatch) -> None:
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 0.2)
    provider = _provider(reloading=True)
    started = time.monotonic()
    with pytest.raises(ProviderError, match="No embed model server is running"):
        provider._require_clients(WorkerRole.EMBED)
    assert time.monotonic() - started < 5


def test_request_does_not_wait_when_no_reload_runs(monkeypatch) -> None:
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 30.0)
    provider = _provider(reloading=False)
    started = time.monotonic()
    with pytest.raises(ProviderError, match="No embed model server is running"):
        provider._require_clients(WorkerRole.EMBED)
    assert time.monotonic() - started < 5
