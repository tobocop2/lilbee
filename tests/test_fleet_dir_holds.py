"""A fleet dir hold lives exactly as long as the users of its dir.

The users are a group bound in the dir, a retired client with a request in
flight, a rediscover retry, a start, a reload pass, a settings change and a
shutdown. The last user to end releases the hold on its own thread and never
waits on the provider lock, so a stream that a finalizer closes cannot deadlock.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole

_DIR = Path("/engine/a")
_OTHER = Path("/engine/b")
_JOIN_S = 10.0


def _client() -> LlamaServerClient:
    return LlamaServerClient(
        "http://127.0.0.1:1", "chat-0", profile=LLAMA.request_profile(WorkerRole.CHAT)
    )


def _group() -> SwapGroup:
    return next(iter(SwapGroup))


def _provider(*dirs: Path) -> tuple[FleetProvider, dict[Path, MagicMock]]:
    p = FleetProvider()
    holds = {d: MagicMock() for d in dirs}
    p._engine_holds = dict(holds)
    return p, holds


def _bound(p: FleetProvider, group: SwapGroup, engine_dir: Path) -> None:
    """Record that *group* runs in *engine_dir*, by whichever path the tree has.

    The fallback lets the behaviour tests run red on a tree without dir uses.
    """
    with p._lock:
        setter = getattr(p, "_set_group_dir", None)
        if setter is not None:
            setter(group, engine_dir)
        else:
            p._group_dirs[group] = engine_dir


def _provider_with_busy_chat() -> tuple[FleetProvider, MagicMock, LlamaServerClient]:
    """A provider whose chat group runs in _DIR with one request in flight."""
    p, holds = _provider(_DIR)
    group = _group()
    client = _client()
    with p._lock:
        p._swaps[group] = MagicMock()
        p._role_group[WorkerRole.CHAT] = group
        p._clients[WorkerRole.CHAT] = [client]
    _bound(p, group, _DIR)
    client.reserve()
    return p, holds[_DIR], client


def _assert_released(p: FleetProvider, holds: dict[Path, MagicMock]) -> None:
    assert p._dir_users == {}
    assert p._engine_holds == {}
    for hold in holds.values():
        hold.release_and_check_last.assert_called_once()


def _connection_error() -> ProviderError:
    return ProviderError("down", kind=ProviderErrorKind.CONNECTION)


def test_the_last_request_releases_the_hold_on_its_own_thread(monkeypatch) -> None:
    started: list[str] = []
    real_start = threading.Thread.start

    def recording_start(self: threading.Thread) -> None:
        started.append(self.name)
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", recording_start)
    p, hold, client = _provider_with_busy_chat()
    p._drop_swap_refs()  # the rediscover drop: the group ends, the client retires busy
    hold.release_and_check_last.assert_not_called()
    client.release()
    assert started == [], f"a helper thread started: {started}"
    hold.release_and_check_last.assert_called_once()
    assert _DIR not in p._engine_holds


def test_ending_the_last_request_never_waits_on_the_provider_lock() -> None:
    # Thread U holds the provider lock and waits on something the releasing thread
    # does after its release, as the vision dispatcher's pick waits on the
    # condition that a finalizer's thread can hold.
    p, _hold, client = _provider_with_busy_chat()
    p._drop_swap_refs()
    holding = threading.Event()
    released = threading.Event()

    def lock_holder() -> None:
        with p._lock:
            holding.set()
            released.wait(timeout=5)

    holder = threading.Thread(target=lock_holder, daemon=True)
    holder.start()
    assert holding.wait(timeout=5)
    ender = threading.Thread(target=lambda: (client.release(), released.set()), daemon=True)
    ender.start()
    ender_done = released.wait(timeout=2)
    released.set()
    holder.join(timeout=_JOIN_S)
    ender.join(timeout=_JOIN_S)
    assert ender_done, "the release waited on the provider lock held by another thread"


def test_a_reload_that_stops_every_group_of_its_dir_keeps_the_dir_held() -> None:
    p, hold, client = _provider_with_busy_chat()
    client.release()  # drained: no request in flight
    group = p._role_group[WorkerRole.CHAT]
    drained = {WorkerRole.CHAT: p._clients[WorkerRole.CHAT]}
    p._stop_groups({group}, set(), frozenset(), drained, {})
    # The start phase of the same pass binds a new group in _DIR next, so the dir
    # must still be held here.
    hold.release_and_check_last.assert_not_called()
    assert _DIR in p._engine_holds


def test_a_reload_stop_that_raises_gives_back_the_uses_it_took() -> None:
    p, holds = _provider(_DIR)
    _bound(p, SwapGroup.CHAT, _DIR)
    swap = MagicMock()
    swap.shutdown.side_effect = OSError("stop failed")
    with pytest.raises(OSError, match="stop failed"):
        p._stop_groups({SwapGroup.CHAT}, set(), frozenset(), {}, {SwapGroup.CHAT: swap})
    assert p._dir_users == {}
    holds[_DIR].release_and_check_last.assert_called_once()


def test_the_last_use_releases_the_hold_and_stops_no_engine(monkeypatch) -> None:
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    p, holds = _provider(_DIR, _OTHER)
    _bound(p, SwapGroup.CHAT, _DIR)
    _bound(p, SwapGroup.EMBED, _OTHER)
    with p._lock:
        p._drop_group(SwapGroup.CHAT)
    assert p._engine_holds == {_OTHER: holds[_OTHER]}
    holds[_DIR].release_and_check_last.assert_called_once()
    holds[_OTHER].release_and_check_last.assert_not_called()
    assert stopped == []


def test_moving_a_group_to_a_new_dir_releases_only_the_old_dir() -> None:
    p, holds = _provider(_DIR, _OTHER)
    _bound(p, _group(), _DIR)
    _bound(p, _group(), _OTHER)
    assert p._engine_holds == {_OTHER: holds[_OTHER]}
    holds[_DIR].release_and_check_last.assert_called_once()
    holds[_OTHER].release_and_check_last.assert_not_called()


def test_binding_a_group_again_in_the_same_dir_never_drops_the_hold() -> None:
    p, holds = _provider(_DIR)
    _bound(p, _group(), _DIR)
    _bound(p, _group(), _DIR)
    assert p._dir_users == {_DIR: 1}
    holds[_DIR].release_and_check_last.assert_not_called()


def test_a_retired_client_with_a_request_in_flight_keeps_the_dir_after_its_group_ends() -> None:
    p, holds = _provider(_DIR)
    client = _client()
    client.reserve()
    _bound(p, _group(), _DIR)
    with p._lock:
        p._retire_clients([client], _DIR)
        p._drop_group(_group())
    holds[_DIR].release_and_check_last.assert_not_called()
    client.release()
    _assert_released(p, holds)
    client.close()


def test_an_idle_retired_client_keeps_no_hold() -> None:
    p, holds = _provider(_DIR)
    client = _client()
    _bound(p, _group(), _DIR)
    with p._lock:
        p._retire_clients([client], _DIR)
        p._drop_group(_group())
    _assert_released(p, holds)
    client.close()


def test_closing_a_retired_client_ends_its_use_once() -> None:
    p, holds = _provider(_DIR)
    client = _client()
    client.reserve()
    other = _client()
    with p._lock:
        p._retire_clients([client], _DIR)
    client.release()
    _assert_released(p, holds)
    # The next retire closes the idle client; its use ended already.
    with p._lock:
        p._retire_clients([other], None)
    assert p._client_uses == {}
    assert p._dir_users == {}
    other.close()


def test_giving_a_client_back_to_dispatch_ends_its_use() -> None:
    p, holds = _provider(_DIR)
    client = _client()
    client.reserve()
    _bound(p, _group(), _DIR)
    with p._lock:
        p._role_group[WorkerRole.CHAT] = _group()
        p._clients[WorkerRole.CHAT] = [client]
        drained: dict[WorkerRole, list[LlamaServerClient]] = {}
        p._take_out_of_dispatch({_group()}, drained, {})
    assert p._dir_users == {_DIR: 2}
    p._give_back_clients(drained)
    assert p._dir_users == {_DIR: 1}
    client.release()
    assert p._dir_users == {_DIR: 1}
    holds[_DIR].release_and_check_last.assert_not_called()
    client.close()


def test_a_rediscover_keeps_the_holds_through_the_retry_and_releases_the_dir_it_left() -> None:
    p, holds = _provider(_DIR)
    _bound(p, _group(), _DIR)
    seen: list[bool] = []
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _connection_error()
        seen.append(holds[_DIR].release_and_check_last.called)
        _bound(p, _group(), _OTHER)
        return "ok"

    assert p._with_rediscover(_call) == "ok"
    assert seen == [False]
    holds[_DIR].release_and_check_last.assert_called_once()
    assert p._dir_users == {_OTHER: 1}


def test_a_rediscover_that_binds_the_same_dir_keeps_its_hold() -> None:
    p, holds = _provider(_DIR)
    _bound(p, _group(), _DIR)
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _connection_error()
        _bound(p, _group(), _DIR)
        return "ok"

    assert p._with_rediscover(_call) == "ok"
    holds[_DIR].release_and_check_last.assert_not_called()
    assert p._dir_users == {_DIR: 1}


def test_a_failed_rediscover_retry_still_ends_its_uses() -> None:
    p, holds = _provider(_DIR)
    _bound(p, _group(), _DIR)

    def _call() -> str:
        raise _connection_error()

    with pytest.raises(ProviderError):
        p._with_rediscover(_call)
    _assert_released(p, holds)


def test_a_transport_error_rediscover_also_ends_its_uses() -> None:
    p, holds = _provider(_DIR)
    _bound(p, _group(), _DIR)

    def _call() -> str:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError, match="not responding"):
        p._with_rediscover(_call)
    _assert_released(p, holds)


def test_a_client_that_goes_idle_during_a_retry_keeps_the_dir_until_the_retry_ends() -> None:
    # Thread B ends a retired client's last request while thread A's retry runs.
    p, hold, client = _provider_with_busy_chat()
    calls = {"n": 0}
    held_after_idle: list[bool] = []

    def _call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _connection_error()
        ender = threading.Thread(target=client.release, daemon=True)
        ender.start()
        ender.join(_JOIN_S)
        held_after_idle.append(_DIR in p._engine_holds)
        return "ok"

    assert p._with_rediscover(_call) == "ok"
    assert held_after_idle == [True]
    hold.release_and_check_last.assert_called_once()
    assert p._engine_holds == {}


def test_a_start_that_fails_on_a_held_dir_gives_its_use_back(monkeypatch) -> None:
    # The dir was held before the start, so the start did not take it freshly.
    p, holds = _provider(_DIR)
    _bound(p, _group(), _DIR)

    def _spawn_after_the_group_left(_dir: Path) -> bool:
        with p._lock:
            p._drop_group(_group())
        return False

    monkeypatch.setattr(p, "_plan_and_spawn", _spawn_after_the_group_left)
    assert p._start_and_hold(_DIR) is False
    assert p._engine_holds == {}
    holds[_DIR].release_and_check_last.assert_called_once()


def test_the_last_group_use_that_ends_during_a_start_into_the_dir_leaves_it_held(
    monkeypatch,
) -> None:
    p, holds = _provider(_DIR)
    _bound(p, SwapGroup.CHAT, _DIR)
    held_during_start: list[bool] = []

    def _spawn(_dir: Path) -> bool:
        with p._lock:
            p._drop_group(SwapGroup.CHAT)
        held_during_start.append(_DIR in p._engine_holds)
        _bound(p, SwapGroup.EMBED, _DIR)
        return True

    monkeypatch.setattr(p, "_plan_and_spawn", _spawn)
    monkeypatch.setattr(prov_mod, "_watch_bound_engine", lambda _d: None)
    assert p._start_and_hold(_DIR) is True
    assert held_during_start == [True]
    assert p._dir_users == {_DIR: 1}
    holds[_DIR].release_and_check_last.assert_not_called()


def test_ending_the_last_use_never_waits_on_the_fleet_work() -> None:
    class _Work:
        waits = 0

        def wait(self, timeout: float | None = None) -> bool:
            self.waits += 1
            return True

    p, holds = _provider(_DIR)
    work = _Work()
    p._fleet_work = work  # type: ignore[assignment]
    _bound(p, _group(), _DIR)
    with p._lock:
        p._drop_group(_group())
    assert work.waits == 0
    _assert_released(p, holds)


def test_ending_the_last_use_starts_no_thread_even_when_none_can_start(monkeypatch) -> None:
    def _no_threads(self: threading.Thread) -> None:
        raise RuntimeError("can't create new thread at interpreter shutdown")

    p, holds = _provider(_DIR)
    _bound(p, _group(), _DIR)
    monkeypatch.setattr(threading.Thread, "start", _no_threads)
    with p._lock:
        p._drop_group(_group())
    _assert_released(p, holds)


def test_a_settings_change_keeps_the_dir_held_until_the_release_step_stops_it(
    monkeypatch, tmp_path: Path
) -> None:
    # The change drops every group first; the release step must still find the
    # hold, so the last user stops the engine it leaves.
    from lilbee.runtime.engine_lock import hold_user_lock

    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(prov_mod, "keep_warm_requested", lambda _d: False)
    p = FleetProvider()
    p._engine_holds = {tmp_path: hold_user_lock(tmp_path)}
    with p._lock:
        p._swaps[_group()] = MagicMock()
    _bound(p, _group(), tmp_path)
    monkeypatch.setattr(p, "_acquire_engine", lambda _root: False)

    assert p._rebind_or_overflow() == []

    assert stopped == [tmp_path]
    assert p._engine_holds == {}
    assert p._dir_users == {}
