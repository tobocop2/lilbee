"""The MLX restart and its helpers: the message, the error, the flags and the spawn arguments."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from unittest.mock import MagicMock

import httpx
import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.fleet.swap_manager import EngineLifetime
from lilbee.providers.roles import WorkerRole
from lilbee.runtime.daemon_call import DaemonCall
from tests import test_fleet_reload_plan as plan_tests
from tests.test_fleet_reload_plan import (
    _DRAIN_LIMIT_S,
    _JOIN_S,
    _QUESTION,
    _EngineSwap,
    _join_fleet_threads,
    _time_out_a_chat,
)

# The fixtures are shared by assignment: pytest finds them by module attribute.
engine_fleet = plan_tests.engine_fleet
mlx_engine_fleet = plan_tests.mlx_engine_fleet
real_planner = plan_tests.real_planner

CHAT = WorkerRole.CHAT
EMBED = WorkerRole.EMBED
FleetFixture = tuple[FleetProvider, list[_EngineSwap]]

_RESTART_TEXT = (
    "The chat model server stopped answering and was restarted. Send your request again."
)
_EMBED_RESTART_TEXT = (
    "The embed model server stopped answering and was restarted. Send your request again."
)


def _connection_error() -> ProviderError:
    return ProviderError("down", kind=ProviderErrorKind.CONNECTION)


def _release_first_engines(swaps: list[_EngineSwap]) -> None:
    """The engines of the first fleet hold each chat request; let them answer."""
    for swap in swaps:
        swap.engine.release.set()


def _hold_the_plan(
    monkeypatch: pytest.MonkeyPatch, reached: threading.Event, release: threading.Event
) -> None:
    """Hold the next plan call, which runs after the stuck servers drained, until *release*."""
    real_plan = planning_mod.plan_all_launches

    def held(**kwargs: object) -> object:
        reached.set()
        assert release.wait(_JOIN_S)
        return real_plan(**kwargs)

    monkeypatch.setattr(planning_mod, "plan_all_launches", held)


def _fail_first_open(client: object, error: Exception) -> list[int]:
    """Make the first stream open on *client* fail with *error*; later opens are real."""
    opens: list[int] = []
    real_open = client.chat_stream_items  # type: ignore[attr-defined]

    def open_stream(*args: object, **kwargs: object) -> object:
        opens.append(1)
        if len(opens) == 1:
            raise error
        return real_open(*args, **kwargs)

    client.chat_stream_items = open_stream  # type: ignore[attr-defined]
    return opens


def test_a_stream_that_a_restart_ends_has_the_exact_message_provider_and_kind(
    mlx_engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    old = provider._swap_for(CHAT)
    stream = provider.chat(_QUESTION, stream=True)
    try:
        assert old.engine.started.wait(_JOIN_S)
        _time_out_a_chat(provider)
        with pytest.raises(ProviderError) as caught:
            list(stream)
        _join_fleet_threads()
    finally:
        stream.close()

    assert str(caught.value) == _RESTART_TEXT
    assert caught.value.provider == "llama-server"
    assert caught.value.kind is ProviderErrorKind.CONNECTION


def test_a_stream_open_that_fails_on_a_restarted_server_rediscovers_nothing(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart owns the old server: the retry waits for the new one, no refs are dropped."""
    provider, swaps = engine_fleet
    _release_first_engines(swaps)
    drop = MagicMock()
    monkeypatch.setattr(provider, "_drop_swap_refs", drop)
    client = provider._clients[CHAT][0]
    client.restart_message = _RESTART_TEXT
    opens = _fail_first_open(client, httpx.ConnectError("refused"))

    stream = provider.chat(_QUESTION, stream=True)
    frames = list(stream)

    assert frames
    assert len(opens) == 2
    drop.assert_not_called()


def test_a_stream_open_that_fails_on_a_server_nobody_restarted_rediscovers(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, swaps = engine_fleet
    _release_first_engines(swaps)
    drop = MagicMock()
    monkeypatch.setattr(provider, "_drop_swap_refs", drop)
    monkeypatch.setattr(provider, "_release_unused_holds", lambda: None)
    client = provider._clients[CHAT][0]
    opens = _fail_first_open(client, httpx.ConnectError("refused"))

    frames = list(provider.chat(_QUESTION, stream=True))

    assert frames
    assert len(opens) == 2
    drop.assert_called_once_with()


def test_a_role_that_the_pass_no_longer_forces_leaves_the_forced_pass_quietly(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    real_pass = provider._reload_pass

    def pass_that_clears_the_forced_roles(force: frozenset[WorkerRole] = frozenset()) -> None:
        real_pass(force=force)
        with provider._lock:
            provider._forced_roles.clear()

    monkeypatch.setattr(provider, "_reload_pass", pass_that_clears_the_forced_roles)
    work = provider._force_restart(CHAT, stopped_answering=False)
    assert work is not None
    provider._join_fleet_work(work)
    _join_fleet_threads()

    assert provider._forced_roles == {}


class _FlakyLock:
    """A lock whose next entry raises, then behaves as the lock it wraps."""

    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self.fail_next = False

    def __enter__(self) -> bool:
        if self.fail_next:
            self.fail_next = False
            raise OSError("the lock failed")
        return self._lock.__enter__()

    def __exit__(self, *exc: object) -> None:
        self._lock.__exit__(*exc)


def test_a_rediscover_that_never_opened_releases_no_count(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    monkeypatch.setattr(provider, "_release_unused_holds", lambda: None)
    flaky = _FlakyLock(provider._lock)
    monkeypatch.setattr(provider, "_lock", flaky)
    flaky.fail_next = True

    def call() -> str:
        raise _connection_error()

    with pytest.raises(OSError, match="the lock failed"):
        provider._with_rediscover(call)

    assert provider._open_rediscovers == 0


def test_two_overlapping_rediscovers_count_two(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    monkeypatch.setattr(provider, "_drop_swap_refs", lambda: None)
    monkeypatch.setattr(provider, "_release_unused_holds", lambda: None)
    release = threading.Event()
    inside = [threading.Event(), threading.Event()]
    results: list[str] = []

    def retrying(index: int) -> Callable[[], str]:
        attempts: list[int] = []

        def call() -> str:
            attempts.append(1)
            if len(attempts) == 1:
                raise _connection_error()
            inside[index].set()
            assert release.wait(_JOIN_S)
            return "answered"

        return call

    def caller(call: Callable[[], str]) -> Callable[[], None]:
        return lambda: results.append(provider._with_rediscover(call))

    callers = [
        threading.Thread(
            target=caller(retrying(i)),
            daemon=True,
        )
        for i in range(2)
    ]
    for caller in callers:
        caller.start()
    try:
        assert all(event.wait(_JOIN_S) for event in inside)
        counted = provider._open_rediscovers
    finally:
        release.set()
        for caller in callers:
            caller.join(_JOIN_S)

    assert counted == 2
    assert results == ["answered", "answered"]
    assert provider._open_rediscovers == 0


def test_a_timeout_on_a_role_with_no_clients_restarts_nothing(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    restart = MagicMock()
    monkeypatch.setattr(provider, "_force_restart", restart)
    monkeypatch.setattr(provider, "_drop_swap_refs", lambda: None)
    monkeypatch.setattr(provider, "_release_unused_holds", lambda: None)
    with provider._lock:
        provider._clients.pop(CHAT)
    attempts: list[int] = []

    def call() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("slow", request=httpx.Request("POST", "http://engine/chat"))
        return "answered"

    assert provider._with_rediscover(call, role=CHAT) == "answered"
    restart.assert_not_called()


def test_a_dead_server_rebuild_does_not_mark_the_server_as_one_that_stopped_answering(
    engine_fleet: FleetFixture,
) -> None:
    provider, _swaps = engine_fleet
    old_client = provider._clients[CHAT][0]

    provider._rebuild_role(CHAT)
    _join_fleet_threads()

    assert old_client.restart_message is None
    assert provider._swap_for(CHAT) is not None


def test_a_forced_restart_that_did_not_stop_answering_is_stored_as_false(
    engine_fleet: FleetFixture,
) -> None:
    provider, _swaps = engine_fleet
    with provider._lock:
        work = provider._force_restart(CHAT, stopped_answering=False)
        stored = dict(provider._forced_roles)
    assert work is not None
    provider._join_fleet_work(work)
    _join_fleet_threads()

    assert stored == {CHAT: False}
    assert stored[CHAT] is False


def test_a_reload_takes_the_not_installed_reasons_from_the_new_plan(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    real_plan = planning_mod.plan_all_launches

    def plan_with_a_skipped_role(**kwargs: object) -> object:
        plan = real_plan(**kwargs)
        return dataclasses.replace(plan, skipped_not_installed={WorkerRole.RERANK: "reranker-ref"})

    monkeypatch.setattr(planning_mod, "plan_all_launches", plan_with_a_skipped_role)
    provider._skipped_not_installed = {WorkerRole.VISION: "stale-ref"}
    provider.reload_role(WorkerRole.VISION, wait=True)
    _join_fleet_threads()

    assert provider._skipped_not_installed == {WorkerRole.RERANK: "reranker-ref"}


@pytest.mark.parametrize("hold_warm", [True, False])
def test_a_restarted_group_spawns_with_the_origin_lifetime_and_warm_hold(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch, hold_warm: bool
) -> None:
    provider, _swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_engine_lifetime", lambda _dir: EngineLifetime.BUILDER)
    provider._hold_warm_for_session = hold_warm

    provider._rebuild_role(CHAT)
    _join_fleet_threads()

    swap = provider._swap_for(CHAT)
    assert swap.origin == prov_mod._plan_origin()
    assert swap.origin is not None
    assert swap.origin.config_root == str(cfg.data_root)
    assert swap.lifetime is EngineLifetime.BUILDER
    assert swap.ttl_seconds == (0 if hold_warm else cfg.engine_idle_ttl_minutes * 60)


def test_a_restart_that_did_not_stop_answering_keeps_the_clients_in_dispatch(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    reached = threading.Event()
    release = threading.Event()
    _hold_the_plan(monkeypatch, reached, release)
    client = provider._clients[CHAT][0]
    work = provider._force_restart(CHAT, stopped_answering=False)
    try:
        assert reached.wait(_JOIN_S)
        with provider._lock:
            in_dispatch = list(provider._clients.get(CHAT, []))
            retired = list(provider._retiring_clients)
    finally:
        release.set()
    assert work is not None
    provider._join_fleet_work(work)
    _join_fleet_threads()

    assert in_dispatch == [client]
    assert client not in retired


def test_a_restart_of_a_role_with_no_server_group_drains_and_restarts_nothing(
    engine_fleet: FleetFixture,
) -> None:
    provider, _swaps = engine_fleet
    chat_swap = provider._swap_for(CHAT)
    chat_client = provider._clients[CHAT][0]
    assert WorkerRole.VISION not in provider._role_group

    work = provider._force_restart(WorkerRole.VISION, stopped_answering=True)
    assert work is not None
    provider._join_fleet_work(work)
    _join_fleet_threads()

    assert provider._swap_for(CHAT) is chat_swap
    assert provider._clients[CHAT] == [chat_client]


def test_a_stalled_role_with_no_clients_restarts_its_group(engine_fleet: FleetFixture) -> None:
    provider, _swaps = engine_fleet
    old = provider._swap_for(CHAT)
    with provider._lock:
        provider._clients.pop(CHAT)

    work = provider._force_restart(CHAT, stopped_answering=True)
    assert work is not None
    provider._join_fleet_work(work)
    _join_fleet_threads()

    assert old.shutdowns == 1
    assert provider._swap_for(CHAT) is not old
    assert provider._clients.get(CHAT)


def _restart_with_drained(
    provider: FleetProvider,
    drained: dict[WorkerRole, list[object]],
    owned: dict[object, object] | None = None,
) -> list[WorkerRole]:
    """Run the restart step with *drained* and *owned* as the pass hands them over."""
    with provider._lock:
        running = set(provider._swaps)
        old = dict(provider._launches)
    return provider._restart_changed(
        provider._reload_dir(),
        running,
        old,
        frozenset(),
        drained,  # type: ignore[arg-type]
        {} if owned is None else owned,  # type: ignore[arg-type]
    )


def test_a_drained_role_restarts_its_group_and_its_clients_hear_the_restart(
    engine_fleet: FleetFixture,
) -> None:
    provider, _swaps = engine_fleet
    chat_swap = provider._swap_for(CHAT)
    embed_swap = provider._swap_for(EMBED)
    embed_client = provider._clients[EMBED][0]
    with provider._lock:
        drained = {EMBED: provider._clients.pop(EMBED)}
        owned = {provider._role_group[EMBED]: embed_swap}

    restarted = _restart_with_drained(provider, drained, owned)

    assert restarted == [EMBED]
    assert embed_swap.shutdowns == 1
    assert provider._swap_for(EMBED) is not embed_swap
    assert provider._swap_for(CHAT) is chat_swap
    assert embed_client.restart_message == _EMBED_RESTART_TEXT


def test_a_drained_role_that_left_the_fleet_restarts_nothing(engine_fleet: FleetFixture) -> None:
    provider, _swaps = engine_fleet
    chat_swap = provider._swap_for(CHAT)
    embed_swap = provider._swap_for(EMBED)

    restarted = _restart_with_drained(provider, {WorkerRole.VISION: []})

    assert restarted == []
    assert provider._swap_for(CHAT) is chat_swap
    assert provider._swap_for(EMBED) is embed_swap


def test_a_drained_group_restarts_after_a_rediscover_cleared_the_role_map(
    engine_fleet: FleetFixture,
) -> None:
    """The pass restarts the group it took, though the fleet no longer maps the role to it."""
    provider, swaps = engine_fleet
    embed_swap = provider._swap_for(EMBED)
    reload_dir = provider._reload_dir()
    with provider._lock:
        running = set(provider._swaps)
        old = dict(provider._launches)
        drained = {EMBED: provider._clients.pop(EMBED)}
        owned = {provider._role_group[EMBED]: embed_swap}
    provider._drop_swap_refs()

    restarted = provider._restart_changed(
        reload_dir,
        running,
        old,
        frozenset(),
        drained,
        owned,  # type: ignore[arg-type]
    )

    assert restarted == [EMBED]
    assert embed_swap.shutdowns == 1
    assert provider._swap_for(EMBED) is swaps[-1]
    assert swaps[-1] is not embed_swap


def _plan_without_chat(monkeypatch: pytest.MonkeyPatch, *, then_fail: bool = False) -> list[int]:
    """Make each plan leave out the chat launches, so a reload drops the chat group.

    With *then_fail*, every plan after the first raises. Returns one entry per plan.
    """
    real_plan = planning_mod.plan_all_launches
    plans: list[int] = []

    def no_chat(**kwargs: object) -> object:
        plans.append(1)
        if then_fail and len(plans) > 1:
            raise ProviderError("The engine binary is missing.")
        plan = real_plan(**kwargs)
        kept = tuple(launch for launch in plan.launches if launch.role is not CHAT)
        return dataclasses.replace(plan, launches=kept)

    monkeypatch.setattr(planning_mod, "plan_all_launches", no_chat)
    return plans


def _rediscover_during_the_drain(
    provider: FleetProvider, monkeypatch: pytest.MonkeyPatch, release: threading.Event
) -> tuple[list[int], threading.Thread]:
    """Start a request thread's rediscover when the pass drains a client; then release the hold.

    The request fails once while the drain runs, so its rediscover starts inside the drain.
    Returns the calls of that request and its thread.
    """
    real_drain = provider._drain_clients
    calls: list[int] = []
    failed = threading.Event()

    def fail_once() -> str:
        calls.append(1)
        if len(calls) == 1:
            failed.set()
            raise _connection_error()
        return "retried"

    request = threading.Thread(target=provider._with_rediscover, args=(fail_once,))

    def drain(clients: list[object], until: float) -> None:
        if clients and not request.ident:
            request.start()
            assert failed.wait(_JOIN_S)
            release.set()
        real_drain(clients, until)  # type: ignore[arg-type]

    monkeypatch.setattr(provider, "_drain_clients", drain)
    return calls, request


def test_a_rediscover_during_the_drain_still_stops_the_group_the_new_plan_drops(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", plan_tests._LONG_DRAIN_S)
    old_chat = provider._swap_for(CHAT)
    answers: list[object] = []
    caller = plan_tests._held_chat(provider, answers)
    calls, request = _rediscover_during_the_drain(provider, monkeypatch, old_chat.engine.release)
    _plan_without_chat(monkeypatch)

    provider.reload_role(CHAT)
    caller.join(_JOIN_S)
    request.join(_JOIN_S)
    _join_fleet_threads()

    assert calls == [1, 1]
    assert [getattr(answer, "text", answer) for answer in answers] == ["the answer"]
    assert old_chat.shutdowns == 1
    assert old_chat.engine.stopped.is_set()


def test_a_rediscover_during_a_stalled_drain_still_stops_the_group_the_new_plan_drops(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stalled server drains before the plan; the pass stops it though a rediscover ran."""
    provider, _swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", plan_tests._LONG_DRAIN_S)
    old_chat = provider._swap_for(CHAT)
    answers: list[object] = []
    caller = plan_tests._held_chat(provider, answers)
    calls, request = _rediscover_during_the_drain(provider, monkeypatch, old_chat.engine.release)
    _plan_without_chat(monkeypatch)

    provider._force_restart(CHAT, stopped_answering=True)
    caller.join(_JOIN_S)
    request.join(_JOIN_S)
    _join_fleet_threads()

    assert calls == [1, 1]
    assert old_chat.shutdowns == 1


def test_a_rediscover_in_a_failed_pass_waits_until_the_pass_gave_the_server_back(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan after the drain fails; the fleet still maps the server at the give-back."""
    provider, _swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", plan_tests._LONG_DRAIN_S)
    old_chat = provider._swap_for(CHAT)
    answers: list[object] = []
    caller = plan_tests._held_chat(provider, answers)
    calls, request = _rediscover_during_the_drain(provider, monkeypatch, old_chat.engine.release)
    plans = _plan_without_chat(monkeypatch, then_fail=True)
    real_give_back = provider._give_back_clients
    mapped_at_give_back: list[bool] = []

    def give_back(drained: dict[WorkerRole, list[object]]) -> None:
        mapped_at_give_back.append(provider._swap_for(CHAT) is old_chat)
        real_give_back(drained)  # type: ignore[arg-type]

    monkeypatch.setattr(provider, "_give_back_clients", give_back)

    provider.reload_role(CHAT)
    caller.join(_JOIN_S)
    request.join(_JOIN_S)
    _join_fleet_threads()

    assert calls == [1, 1]
    assert len(plans) == 2
    assert mapped_at_give_back == [True]
    assert old_chat.shutdowns == 0


def test_a_failed_pass_without_a_rediscover_gives_the_server_back_running(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan after the drain fails; the fleet still maps the taken server, so it serves on."""
    provider, _swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", plan_tests._LONG_DRAIN_S)
    old_chat = provider._swap_for(CHAT)
    old_clients = list(provider._clients[CHAT])
    answers: list[object] = []
    caller = plan_tests._held_chat(provider, answers)
    real_drain = provider._drain_clients

    def drain(clients: list[object], until: float) -> None:
        # The first loop turn drains nothing; the held chat must still run at the first plan.
        if clients:
            old_chat.engine.release.set()
        real_drain(clients, until)  # type: ignore[arg-type]

    monkeypatch.setattr(provider, "_drain_clients", drain)
    plans = _plan_without_chat(monkeypatch, then_fail=True)

    provider.reload_role(CHAT)
    caller.join(_JOIN_S)
    _join_fleet_threads()

    assert len(plans) == 2
    assert provider._swap_for(CHAT) is old_chat
    assert old_chat.shutdowns == 0
    assert provider._clients[CHAT] == old_clients


# How long the drain waits for a request thread's rediscover to clear the fleet's map.
_REDISCOVER_WAIT_S = 2.0


def test_a_rediscover_on_a_request_thread_during_the_drain_leaves_the_kept_group_serving(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request thread rediscovers while the pass drains chat; embed, which it keeps, serves."""
    provider, _swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", plan_tests._LONG_DRAIN_S)
    old_chat = provider._swap_for(CHAT)
    old_embed = provider._swap_for(EMBED)
    answers: list[object] = []
    caller = plan_tests._held_chat(provider, answers)
    dropped = threading.Event()
    real_drop = provider._drop_swap_refs

    def drop(**kwargs: bool) -> None:
        real_drop(**kwargs)
        dropped.set()

    calls: list[int] = []
    failed = threading.Event()

    def fail_once() -> str:
        calls.append(1)
        if len(calls) == 1:
            failed.set()
            raise _connection_error()
        return "retried"

    retried: list[str] = []
    request = threading.Thread(target=lambda: retried.append(provider._with_rediscover(fail_once)))
    real_drain = provider._drain_clients
    in_drain: list[bool] = []

    def drain(clients: list[object], until: float) -> None:
        if clients and not request.ident:
            request.start()
            # The request reached its rediscover while the pass drains chat.
            in_drain.append(failed.wait(_JOIN_S))
            dropped.wait(_REDISCOVER_WAIT_S)
            old_chat.engine.release.set()
        real_drain(clients, until)  # type: ignore[arg-type]

    monkeypatch.setattr(provider, "_drop_swap_refs", drop)
    monkeypatch.setattr(provider, "_drain_clients", drain)

    # The stalled chat restarts on an unchanged plan, so the pass keeps embed.
    provider._force_restart(CHAT, stopped_answering=True)
    caller.join(_JOIN_S)
    request.join(_JOIN_S)
    _join_fleet_threads()

    assert in_drain == [True]
    assert retried == ["retried"]
    assert old_chat.shutdowns == 1
    embed_running = old_embed.shutdowns == 0
    try:
        clients = provider._require_clients(EMBED)
    except ProviderError as err:
        pytest.fail(f"no embed server; old embed server running={embed_running}: {err}")
    assert clients


def test_a_rediscover_still_waiting_at_the_fleet_work_limit_raises_and_keeps_the_map(
    engine_fleet: FleetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fleet work that outlasts the limit: the drop raises the still-starting error, maps intact."""
    provider, _swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_FLEET_WAIT_S", 0.0)
    old_embed = provider._swap_for(EMBED)
    gate = threading.Event()

    def hold() -> None:
        gate.wait(_JOIN_S)

    with provider._lock:
        provider._fleet_work = DaemonCall(hold, name="held-fleet-work")

    try:
        with pytest.raises(ProviderError, match="still starting") as raised:
            provider._drop_swap_refs()
    finally:
        gate.set()
        with provider._lock:
            provider._fleet_work = None

    assert raised.value.kind is ProviderErrorKind.STARTING
    assert provider._swap_for(EMBED) is old_embed
    assert provider._clients[EMBED]
