"""A request that lands while a reload restarts its role's server waits for that server."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from tests.test_fleet_provider import _fake_client, _fake_launch

_JOIN_S = 10.0
_STRESS_ROUNDS = 100


class _SignalCondition(threading.Condition):
    """A condition that counts the requests that block on it and signals each one."""

    def __init__(self, lock: threading.RLock) -> None:
        super().__init__(lock)
        self.blocked = 0
        self.entered = threading.Event()
        self.expected = 1
        self.timeouts = 0

    def wait(self, timeout=None):  # type: ignore[no-untyped-def]
        woken = super().wait(timeout)
        if not woken:
            self.timeouts += 1
        return woken

    def wait_for(self, predicate, timeout=None):  # type: ignore[no-untyped-def]
        if predicate():
            return True
        self.blocked += 1
        if self.blocked >= self.expected:
            self.entered.set()
        return super().wait_for(predicate, timeout)


def _provider(monkeypatch, *, reloading: bool, owed: frozenset[WorkerRole]) -> FleetProvider:
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    provider = FleetProvider()
    provider._reload_done = _SignalCondition(provider._lock)
    provider._reloading = reloading
    provider._reload_owed = owed
    monkeypatch.setattr(provider, "_ensure_fleet", lambda: True)
    return provider


def _adopt_embed(provider: FleetProvider) -> None:
    swap = MagicMock()
    swap.endpoint.return_value = "http://127.0.0.1:1"
    swap.log_path = Path("/engine/logs/llama-swap-fake.log")
    with provider._lock:
        provider._adopt_group(SwapGroup.EMBED, swap, [_fake_launch(WorkerRole.EMBED)])


def _run(target) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_request_returns_when_its_role_is_adopted_not_when_the_reload_ends(monkeypatch) -> None:
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 5.0)
    provider = _provider(monkeypatch, reloading=True, owed=frozenset(WorkerRole))
    gate: _SignalCondition = provider._reload_done  # type: ignore[assignment]

    def adopt_when_request_blocks() -> None:
        assert gate.entered.wait(_JOIN_S)
        _adopt_embed(provider)

    adopter = _run(adopt_when_request_blocks)
    clients = provider._require_clients(WorkerRole.EMBED)
    adopter.join(_JOIN_S)
    assert len(clients) == 1
    assert provider._reloading is True
    assert gate.timeouts == 0


def test_role_the_reload_does_not_bring_back_fails_at_once(monkeypatch) -> None:
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 30.0)
    provider = _provider(monkeypatch, reloading=True, owed=frozenset({WorkerRole.EMBED}))
    gate: _SignalCondition = provider._reload_done  # type: ignore[assignment]
    with pytest.raises(ProviderError, match="No rerank model server is running"):
        provider._require_clients(WorkerRole.RERANK)
    assert gate.blocked == 0


def test_request_waits_while_the_reload_plan_leaves_the_role_open(monkeypatch) -> None:
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 30.0)
    provider = _provider(monkeypatch, reloading=True, owed=frozenset(WorkerRole))
    gate: _SignalCondition = provider._reload_done  # type: ignore[assignment]

    def publish_plan_when_request_blocks() -> None:
        assert gate.entered.wait(_JOIN_S)
        with provider._lock:
            provider._reload_owed = frozenset({WorkerRole.EMBED})
            provider._reload_done.notify_all()

    publisher = _run(publish_plan_when_request_blocks)
    with pytest.raises(ProviderError, match="No rerank model server is running"):
        provider._require_clients(WorkerRole.RERANK)
    publisher.join(_JOIN_S)
    assert gate.blocked == 1


def test_request_raises_when_the_reload_ends_without_a_server_for_the_role(monkeypatch) -> None:
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 30.0)
    provider = _provider(monkeypatch, reloading=True, owed=frozenset(WorkerRole))
    gate: _SignalCondition = provider._reload_done  # type: ignore[assignment]

    def end_reload_when_request_blocks() -> None:
        assert gate.entered.wait(_JOIN_S)
        with provider._lock:
            provider._reloading = False
            provider._reload_owed = frozenset()
            provider._reload_done.notify_all()

    ender = _run(end_reload_when_request_blocks)
    with pytest.raises(ProviderError, match="No embed model server is running"):
        provider._require_clients(WorkerRole.EMBED)
    ender.join(_JOIN_S)


def test_timeout_names_the_problem_and_the_fix(monkeypatch) -> None:
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 0.05)
    provider = _provider(monkeypatch, reloading=True, owed=frozenset(WorkerRole))
    with pytest.raises(
        ProviderError,
        match=re.escape(
            "The embed model server is still restarting after a settings change. "
            "Wait for the restart to finish, then try again."
        ),
    ):
        provider._require_clients(WorkerRole.EMBED)


def test_timeout_returns_a_server_that_came_up_without_a_wake_up(monkeypatch) -> None:
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 0.2)
    provider = _provider(monkeypatch, reloading=True, owed=frozenset(WorkerRole))
    gate: _SignalCondition = provider._reload_done  # type: ignore[assignment]
    client = _fake_client()

    def add_client_without_notify() -> None:
        assert gate.entered.wait(_JOIN_S)
        with provider._lock:
            provider._clients[WorkerRole.EMBED] = [client]

    adder = _run(add_client_without_notify)
    assert provider._require_clients(WorkerRole.EMBED) == [client]
    adder.join(_JOIN_S)


def test_request_does_not_wait_when_no_reload_runs(monkeypatch) -> None:
    provider = _provider(monkeypatch, reloading=False, owed=frozenset())
    gate: _SignalCondition = provider._reload_done  # type: ignore[assignment]
    with pytest.raises(ProviderError, match="No embed model server is running"):
        provider._require_clients(WorkerRole.EMBED)
    assert gate.blocked == 0


def _stress_round(monkeypatch) -> None:
    waiters_per_role = 4
    provider = _provider(monkeypatch, reloading=True, owed=frozenset(WorkerRole))
    gate: _SignalCondition = provider._reload_done  # type: ignore[assignment]
    gate.expected = 2 * waiters_per_role
    outcomes: list[tuple[WorkerRole, object]] = []
    outcomes_lock = threading.Lock()

    def request(role: WorkerRole) -> None:
        try:
            result: object = provider._require_clients(role)
        except ProviderError as exc:
            result = exc
        with outcomes_lock:
            outcomes.append((role, result))

    def reload_run() -> None:
        assert gate.entered.wait(_JOIN_S)
        with provider._lock:
            provider._reload_owed = frozenset({WorkerRole.EMBED})
            provider._reload_done.notify_all()
        _adopt_embed(provider)
        with provider._lock:
            provider._reloading = False
            provider._reload_owed = frozenset()
            provider._reload_done.notify_all()

    threads = [_run(reload_run)]
    for role in (WorkerRole.EMBED, WorkerRole.RERANK):
        threads += [_run(lambda r=role: request(r)) for _ in range(waiters_per_role)]
    for thread in threads:
        thread.join(_JOIN_S)
        assert not thread.is_alive()
    embed = [r for role, r in outcomes if role is WorkerRole.EMBED]
    rerank = [r for role, r in outcomes if role is WorkerRole.RERANK]
    assert gate.timeouts == 0
    assert len(embed) == len(rerank) == waiters_per_role
    assert all(isinstance(r, list) and len(r) == 1 for r in embed)
    assert all(
        isinstance(r, ProviderError) and "No rerank model server is running" in str(r)
        for r in rerank
    )


def test_concurrent_requests_each_end_on_their_own_roles_outcome(monkeypatch) -> None:
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", 5.0)
    for _ in range(_STRESS_ROUNDS):
        _stress_round(monkeypatch)


def test_a_new_reload_owes_every_role_until_its_plan_is_known(monkeypatch) -> None:
    provider = _provider(monkeypatch, reloading=False, owed=frozenset({WorkerRole.EMBED}))
    provider._swaps = {SwapGroup.EMBED: MagicMock()}
    monkeypatch.setattr(provider, "_reload_blocking", lambda: None)
    provider._dispatch_reload("test-reload", wait=True)
    assert provider._reload_owed == frozenset(WorkerRole)


def test_a_rerun_pass_owes_every_role_until_its_plan_is_known(monkeypatch) -> None:
    provider = _provider(monkeypatch, reloading=True, owed=frozenset(WorkerRole))
    owed_at_pass_start: list[frozenset[WorkerRole]] = []

    def pass_that_queues_a_rerun(force=frozenset()) -> None:
        owed_at_pass_start.append(provider._reload_owed)
        with provider._lock:
            provider._reload_owed = frozenset({WorkerRole.EMBED})
            if len(owed_at_pass_start) == 1:
                provider._reload_pending = True

    monkeypatch.setattr(provider, "_reload_pass", pass_that_queues_a_rerun)
    provider._reload_blocking()
    assert owed_at_pass_start == [frozenset(WorkerRole), frozenset(WorkerRole)]
    assert provider._reload_owed == frozenset()
    assert provider._reloading is False
