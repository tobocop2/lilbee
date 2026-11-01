"""Each order of a write or a read against fleet work, through the provider, with no sleep."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from typer.testing import CliRunner

from lilbee.app import engine_service
from lilbee.cli import app as cli_app
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import ingest_warmth
from lilbee.providers.fleet import provider as fleet_provider
from lilbee.providers.fleet.provider import RELOAD_FAILED_NOTICE, RELOADING_NOTICE, FleetProvider
from lilbee.providers.roles import WorkerRole
from tests.test_engine_real_fleet import GGUF_CHAT
from tests.test_engine_write_applied import (
    _local,
    _ProcessSwap,
    _wait_for,
    fleet,
)

__all__ = ["fleet"]

_WAIT_S = 10.0
_FAILED = ProviderError("engine did not start", provider="llama-server")
_STILL_STARTING = "The engines are still starting after 0 s."
_MLX = "mlx"
_LLAMA = "llama.cpp"


def _chat_argv(local: FleetProvider) -> list[str]:
    launches = {launch.role: launch.argv for group in local._launches.values() for launch in group}
    return list(launches[WorkerRole.CHAT])


@pytest.fixture
def passes(fleet: Any, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts each reload pass the fleet work runs."""
    local = _local(fleet)
    seen: list[int] = []
    real = local._reload_pass

    def _counted(*args: Any, **kwargs: Any) -> None:
        seen.append(len(seen))
        real(*args, **kwargs)

    monkeypatch.setattr(local, "_reload_pass", _counted)
    return seen


class _Caller:
    """Runs one call on a thread and keeps what it heard, or None when it returned."""

    def __init__(self, call: Callable[[], Any]) -> None:
        self.heard: BaseException | None = None
        self._call = call
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._call()
        except (ProviderError, engine_service.EngineNotAppliedError) as exc:
            self.heard = exc

    def join(self) -> None:
        self._thread.join(_WAIT_S)
        assert not self._thread.is_alive()


def _write(engine: str) -> _Caller:
    return _Caller(lambda: engine_service.prefer(engine))


def _cold(local: FleetProvider) -> None:
    """Drop every group and stop its fake engine, which a dropped reference would orphan."""
    local.invalidate_load_cache()
    for swap in _ProcessSwap.live.values():
        swap.shutdown()


def _cold_with_a_gate(local: FleetProvider) -> threading.Event:
    """No group up and no work; the next start waits at the returned gate."""
    _cold(local)
    _ProcessSwap.starting.clear()
    gate = threading.Event()
    _ProcessSwap.start_gate = gate
    return gate


def _build_at_the_gate(local: FleetProvider) -> tuple[threading.Event, _Caller]:
    """A first build, with its plan made, held at the gate, and the read that started it."""
    gate = _cold_with_a_gate(local)
    reader = _Caller(local._ensure_fleet)
    assert _ProcessSwap.starting.wait(_WAIT_S)
    assert local._swaps == {}
    return gate, reader


def _pass_at_the_gate(local: FleetProvider, engine: str = _MLX) -> tuple[threading.Event, _Caller]:
    """A pass that has stopped every group, held at the gate, and the write that started it."""
    _ProcessSwap.starting.clear()
    gate = threading.Event()
    _ProcessSwap.start_gate = gate
    first = _write(engine)
    assert _ProcessSwap.starting.wait(_WAIT_S)
    assert local._swaps == {}
    return gate, first


def _work_threads() -> int:
    """Live fleet-work threads; a preload thread comes and goes on its own and is not counted."""
    return sum(thread.name == "fleet-work" for thread in threading.enumerate())


def _settle(local: FleetProvider) -> None:
    work = local._fleet_work
    if work is not None:
        assert work.wait(_WAIT_S)
    assert local._fleet_work is None


@pytest.fixture
def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test value for the bound: a caller answers at once when work still runs."""
    monkeypatch.setattr(fleet_provider, "_FLEET_WAIT_S", 0)


def test_the_warm_up_waits_for_the_build_past_the_bound(
    fleet: Any, no_wait: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    failures: list[str] = []
    monkeypatch.setattr(local, "_fail_warm_unless_ready", failures.append)
    monkeypatch.setattr(local, "_preload_roles", lambda roles=None: None)
    gate = _cold_with_a_gate(local)
    warm = _Caller(local._warm_up_blocking)
    assert _ProcessSwap.starting.wait(_WAIT_S)
    gate.set()
    warm.join()
    assert failures == []
    assert local._swaps != {}


def test_o1_a_write_during_the_first_build_reaches_the_engines(
    fleet: Any, passes: list[int]
) -> None:
    local = _local(fleet)
    gate, reader = _build_at_the_gate(local)
    writer = _write(_MLX)
    _wait_for(lambda: local._writes_wanted > local._writes_done)
    gate.set()
    reader.join()
    writer.join()
    assert (reader.heard, writer.heard) == (None, None)
    assert len(passes) == 1
    assert "--mlx" in _chat_argv(local)
    assert local.reload_notice() is None


def test_o2_a_write_while_every_group_stops_reaches_the_engines(
    fleet: Any, passes: list[int]
) -> None:
    local = _local(fleet)
    gate, first = _pass_at_the_gate(local)
    second = _write(_LLAMA)
    _wait_for(lambda: local._writes_wanted == local._writes_done + 2)
    gate.set()
    first.join()
    second.join()
    assert (first.heard, second.heard) == (None, None)
    assert len(passes) == 2
    assert "--mlx" not in _chat_argv(local)


def test_o3_a_write_that_joins_a_failing_pass_hears_it(fleet: Any, passes: list[int]) -> None:
    local = _local(fleet)
    _ProcessSwap.start_failure = _FAILED
    gate, first = _pass_at_the_gate(local)
    second = _write(_LLAMA)
    _wait_for(lambda: local._writes_wanted == local._writes_done + 2)
    gate.set()
    first.join()
    second.join()
    for caller in (first, second):
        assert isinstance(caller.heard, engine_service.EngineNotAppliedError)
        assert "saved, but the running server could not apply it" in str(caller.heard)
        assert "engine did not start" in str(caller.heard)
    assert len(passes) == 2
    assert local._swaps == {}


def test_o4_a_write_after_the_work_ends_runs_its_own_pass(fleet: Any, passes: list[int]) -> None:
    local = _local(fleet)
    engine_service.prefer(_MLX)
    assert local._fleet_work is None
    engine_service.prefer(_LLAMA)
    assert len(passes) == 2
    assert "--mlx" not in _chat_argv(local)


def test_o5_a_write_after_a_failed_pass_with_no_group_shows_no_notice(
    fleet: Any, passes: list[int]
) -> None:
    local = _local(fleet)
    _ProcessSwap.start_failure = _FAILED
    with pytest.raises(engine_service.EngineNotAppliedError, match="engine did not start"):
        engine_service.prefer(_MLX)
    assert local._swaps == {}
    assert len(passes) == 1
    _ProcessSwap.start_failure = None
    assert engine_service.prefer(_LLAMA) == _LLAMA
    assert len(passes) == 1
    assert local.reload_notice() is None


def test_o6_a_write_with_no_group_and_no_work_starts_no_thread(fleet: Any) -> None:
    local = _local(fleet)
    _cold(local)
    _ProcessSwap.starting.clear()
    assert engine_service.prefer(_MLX) == _MLX
    assert _work_threads() == 0
    assert local._fleet_work is None
    assert _ProcessSwap.starting.is_set() is False
    assert local._ensure_fleet() is True
    assert "--mlx" in _chat_argv(local)


@pytest.mark.parametrize(
    ("engine", "groups_left"),
    [(_MLX, 0), (_LLAMA, 2)],
    ids=["none_left", "every_group_left"],
)
def test_o7_a_write_after_dead_groups_drop_applies(
    fleet: Any, passes: list[int], engine: str, groups_left: int
) -> None:
    local = _local(fleet)
    if groups_left == 0:
        _cold(local)
    else:
        assert len(local._swaps) == groups_left
    assert engine_service.prefer(engine) == engine
    assert len(passes) == (1 if groups_left else 0)
    assert len(local._swaps) == groups_left


def test_o7_a_write_during_a_real_drop_joins_the_loop(
    fleet: Any, passes: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    entered, leave = threading.Event(), threading.Event()
    release = local._release_engines

    def _held_release(**kwargs: Any) -> None:
        entered.set()
        assert leave.wait(_WAIT_S)
        release(**kwargs)

    monkeypatch.setattr(local, "_release_engines", _held_release)
    local.drop_loaded_models_async()
    assert entered.wait(_WAIT_S)
    work = local._fleet_work
    assert work is not None
    assert local._swaps == {}
    writer = _write(_MLX)
    _wait_for(lambda: local._writes_wanted != local._writes_done)
    leave.set()
    writer.join()
    assert writer.heard is None
    assert passes == []
    assert local._fleet_work is None
    assert local._swaps == {}
    assert local._ensure_fleet() is True
    assert "--mlx" in _chat_argv(local)


def test_a_drop_while_every_group_stops_reaches_the_new_fleet(
    fleet: Any, passes: list[int]
) -> None:
    local = _local(fleet)
    gate, writer = _pass_at_the_gate(local)
    assert local._swaps == {}
    local.drop_loaded_models_async()
    assert local._drop_wanted is True
    gate.set()
    writer.join()
    assert writer.heard is None
    assert len(passes) == 1
    _settle(local)
    assert local._drop_wanted is False
    assert local._swaps == {}


def test_a_read_after_a_drop_waits_for_the_drop_and_builds_again(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    old = dict(local._swaps)
    leave = threading.Event()
    release = local._release_engines

    def _held_release(**kwargs: Any) -> None:
        assert leave.wait(_WAIT_S)
        release(**kwargs)

    monkeypatch.setattr(local, "_release_engines", _held_release)
    local.invalidate_load_cache()
    reader = _Caller(local._ensure_fleet)
    leave.set()
    reader.join()
    assert reader.heard is None
    assert set(local._swaps) == set(old)
    assert all(local._swaps[group] is not old[group] for group in old)


def test_a_read_while_a_drop_is_wanted_never_answers_from_the_old_groups(fleet: Any) -> None:
    local = _local(fleet)
    old = dict(local._swaps)
    with local._lock:
        local._drop_wanted = True
    assert local._ensure_fleet() is True
    assert local._drop_wanted is False
    assert set(local._swaps) == set(old)
    assert all(local._swaps[group] is not old[group] for group in old)


def test_a_failed_forced_pass_keeps_the_role_for_the_rerun(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    role = next(iter(local._role_group))
    local._forced_roles.add(role)

    def _fail(force: frozenset[WorkerRole]) -> None:
        raise _FAILED

    monkeypatch.setattr(local, "_reload_pass", _fail)
    with pytest.raises(ProviderError, match="engine did not start"):
        local._forced_pass()
    assert local._forced_roles == {role}


def test_a_dead_group_rebuild_runs_as_a_step(fleet: Any, passes: list[int]) -> None:
    local = _local(fleet)
    role = next(iter(local._role_group))
    old = local._swaps[local._role_group[role]]
    local._rebuild_role(role)
    assert len(passes) == 1
    assert local._swaps[local._role_group[role]] is not old
    assert local._forced_roles == set()
    assert local._fleet_work is None
    assert _work_threads() == 0


def test_a_drop_after_a_failed_pass_still_runs(fleet: Any, passes: list[int]) -> None:
    local = _local(fleet)
    gate, writer = _pass_at_the_gate(local)
    _ProcessSwap.start_failure = _FAILED
    local.drop_loaded_models_async()
    gate.set()
    writer.join()
    assert writer.heard is None
    _settle(local)
    assert local._drop_wanted is False
    assert local._swaps == {}


def test_a_rebuild_and_a_drop_after_the_latch_start_nothing(fleet: Any) -> None:
    local = _local(fleet)
    role = next(iter(local._role_group))
    with local._lock:
        local._shut_down = True
    local._rebuild_role(role)
    local.invalidate_load_cache()
    assert local._forced_roles == set()
    assert local._drop_wanted is False
    assert local._fleet_work is None


def test_a_drop_with_nothing_held_starts_no_thread(fleet: Any) -> None:
    local = _local(fleet)
    _cold(local)
    local._engine_holds = {}
    local.invalidate_load_cache()
    assert local._drop_wanted is False
    assert local._fleet_work is None


@pytest.mark.parametrize(
    ("groups_left", "expected_passes"),
    [(True, 1), (False, 0)],
    ids=["group_left_pass_ran", "groups_gone"],
)
def test_o8_a_write_that_joins_at_the_latch_is_answered(
    fleet: Any, passes: list[int], groups_left: bool, expected_passes: int
) -> None:
    local = _local(fleet)
    if groups_left:
        engine_service.prefer(_MLX)
    else:
        _cold(local)
    with local._lock:
        local._shut_down = True
    assert engine_service.prefer(_LLAMA) == _LLAMA
    assert local._fleet_work is None
    assert _work_threads() == 0
    assert len(passes) == expected_passes


def test_o8_the_loop_ends_at_the_latch_with_a_write_outstanding(
    fleet: Any, passes: list[int]
) -> None:
    local = _local(fleet)
    gate, first = _pass_at_the_gate(local)
    second = _write(_LLAMA)
    _wait_for(lambda: local._writes_wanted == local._writes_done + 2)
    with local._lock:
        local._shut_down = True
    gate.set()
    first.join()
    second.join()
    assert (first.heard, second.heard) == (None, None)
    assert len(passes) == 1


def test_s1_a_write_during_a_stuck_build_answers_saved_at_the_bound(
    fleet: Any, passes: list[int], no_wait: None
) -> None:
    local = _local(fleet)
    gate, reader = _build_at_the_gate(local)
    with pytest.raises(engine_service.EngineNotAppliedError, match=_STILL_STARTING) as heard:
        engine_service.prefer(_MLX)
    assert "saved, but the running server could not apply it" in str(heard.value)
    assert passes == []
    gate.set()
    reader.join()
    _settle(local)


def test_s2_a_write_no_wait_returns_at_once_during_a_stuck_build(fleet: Any) -> None:
    local = _local(fleet)
    gate = _cold_with_a_gate(local)
    work = threading.Thread(target=local._ensure_fleet, kwargs={"bounded": False})
    work.start()
    assert _ProcessSwap.starting.wait(_WAIT_S)
    running = local._fleet_work
    local.reload_placement(wait=False)  # the gate is shut, so a wait here would never end
    assert _work_threads() == 1
    assert local._fleet_work is running
    assert local.reload_notice() == RELOADING_NOTICE
    gate.set()
    work.join(_WAIT_S)
    _settle(local)
    assert local.reload_notice() is None


def test_s3_a_read_during_a_stuck_build_answers_at_the_bound(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    gate = _cold_with_a_gate(local)
    monkeypatch.setattr(fleet_provider, "_FLEET_WAIT_S", 0)
    with pytest.raises(ProviderError, match=_STILL_STARTING):
        local._ensure_fleet()
    assert _ProcessSwap.starting.wait(_WAIT_S)
    with pytest.raises(ProviderError, match=_STILL_STARTING):
        local._require_clients(WorkerRole.CHAT)
    gate.set()
    _settle(local)


def test_s3_a_read_with_a_group_up_never_waits_on_work(fleet: Any, no_wait: None) -> None:
    local = _local(fleet)
    gate = threading.Event()
    _ProcessSwap.start_gate = gate
    with pytest.raises(engine_service.EngineNotAppliedError, match=_STILL_STARTING):
        engine_service.use_for_role(WorkerRole.CHAT, GGUF_CHAT, _MLX)
    assert _ProcessSwap.starting.wait(_WAIT_S)
    assert local._swaps != {}
    assert local._fleet_work is not None
    assert local._ensure_fleet() is True
    gate.set()
    _settle(local)


def test_s4_two_writes_during_a_stuck_build_each_hear_the_bound(
    fleet: Any, passes: list[int], no_wait: None
) -> None:
    local = _local(fleet)
    gate, reader = _build_at_the_gate(local)
    for engine in (_MLX, _LLAMA):
        with pytest.raises(engine_service.EngineNotAppliedError, match=_STILL_STARTING):
            engine_service.prefer(engine)
    gate.set()
    reader.join()
    _settle(local)
    assert len(passes) == 1
    assert "--mlx" not in _chat_argv(local)


def test_s5_a_build_that_ends_after_the_bound_still_applies_the_write(
    fleet: Any, passes: list[int], no_wait: None
) -> None:
    local = _local(fleet)
    gate, reader = _build_at_the_gate(local)
    with pytest.raises(engine_service.EngineNotAppliedError, match=_STILL_STARTING):
        engine_service.prefer(_MLX)
    assert local.reload_notice() == RELOADING_NOTICE
    gate.set()
    reader.join()
    _settle(local)
    assert len(passes) == 1
    assert "--mlx" in _chat_argv(local)
    assert local.reload_notice() is None


def test_s6_a_build_that_ends_with_no_group_runs_no_pass(
    fleet: Any, passes: list[int], no_wait: None
) -> None:
    local = _local(fleet)
    _ProcessSwap.start_failure = _FAILED
    gate, reader = _build_at_the_gate(local)
    with pytest.raises(engine_service.EngineNotAppliedError, match=_STILL_STARTING):
        engine_service.prefer(_MLX)
    gate.set()
    reader.join()
    _settle(local)
    assert passes == []
    assert local._swaps == {}
    assert local.reload_notice() is None


def test_s7_a_write_during_a_stuck_pass_answers_at_the_bound(
    fleet: Any, passes: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    gate, first = _pass_at_the_gate(local)
    monkeypatch.setattr(fleet_provider, "_FLEET_WAIT_S", 0)
    with pytest.raises(engine_service.EngineNotAppliedError, match=_STILL_STARTING):
        engine_service.prefer(_LLAMA)
    monkeypatch.setattr(fleet_provider, "_FLEET_WAIT_S", _WAIT_S)
    gate.set()
    first.join()
    _settle(local)
    assert len(passes) == 2


def test_f1_a_write_that_joins_a_failing_build_hears_it(fleet: Any, passes: list[int]) -> None:
    local = _local(fleet)
    _ProcessSwap.start_failure = _FAILED
    gate, reader = _build_at_the_gate(local)
    writer = _write(_MLX)
    _wait_for(lambda: local._writes_wanted > local._writes_done)
    gate.set()
    reader.join()
    writer.join()
    assert isinstance(writer.heard, engine_service.EngineNotAppliedError)
    assert "engine did not start" in str(writer.heard)
    assert passes == []
    assert local._swaps == {}
    _ProcessSwap.start_failure = None
    _ProcessSwap.start_gate = None
    assert local._ensure_fleet() is True
    assert "--mlx" in _chat_argv(local)


def test_f2_a_read_that_joins_a_failing_build_hears_the_same_failure(fleet: Any) -> None:
    local = _local(fleet)
    _ProcessSwap.start_failure = _FAILED
    gate, first = _build_at_the_gate(local)
    work = local._fleet_work
    second = _Caller(local._ensure_fleet)
    _wait_for(lambda: local._build_wanted)
    assert local._fleet_work is work
    gate.set()
    first.join()
    second.join()
    assert first.heard is _FAILED
    assert second.heard is _FAILED


def test_f3_a_failed_pass_with_a_later_write_runs_again_and_joiners_hear_the_last(
    fleet: Any, passes: list[int]
) -> None:
    local = _local(fleet)
    _ProcessSwap.start_failure = _FAILED
    gate, first = _pass_at_the_gate(local)
    second = _write(_LLAMA)
    _wait_for(lambda: local._writes_wanted == local._writes_done + 2)
    real_start = _ProcessSwap.start

    def _second_pass_starts(self: _ProcessSwap, launches: list[Any], **kwargs: Any) -> None:
        if len(passes) == 2:
            _ProcessSwap.start_failure = None
        real_start(self, launches, **kwargs)

    _ProcessSwap.start = _second_pass_starts  # type: ignore[method-assign]
    try:
        gate.set()
        first.join()
        second.join()
    finally:
        _ProcessSwap.start = real_start  # type: ignore[method-assign]
    assert (first.heard, second.heard) == (None, None)
    assert len(passes) == 2
    assert "--mlx" not in _chat_argv(local)
    assert local.reload_notice() is None


def test_f4_an_older_failure_does_not_reach_a_write_that_joins_a_good_pass(
    fleet: Any, passes: list[int]
) -> None:
    local = _local(fleet)
    _ProcessSwap.start_failure = _FAILED
    with pytest.raises(engine_service.EngineNotAppliedError):
        engine_service.use_for_role(WorkerRole.CHAT, GGUF_CHAT, _MLX)
    assert local.reload_notice() == RELOAD_FAILED_NOTICE
    _ProcessSwap.start_failure = None
    assert engine_service.prefer(_MLX) == _MLX
    assert len(passes) == 2
    assert local.reload_notice() is None


def test_a_build_keeps_the_context_of_the_caller_that_started_it(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    _cold(local)
    ttls: list[Any] = []
    real_start = _ProcessSwap.start

    def _record(self: _ProcessSwap, launches: list[Any], **kwargs: Any) -> None:
        ttls.append(kwargs["ttl_seconds"])
        real_start(self, launches, **kwargs)

    monkeypatch.setattr(_ProcessSwap, "start", _record)
    monkeypatch.setattr(
        fleet_provider,
        "_engine_lifetime",
        lambda _d: fleet_provider.EngineLifetime.BUILDER,
    )
    with ingest_warmth.keep_fleet_warm():
        assert local._ensure_fleet() is True
    assert ttls
    assert set(ttls) == {0}


def test_a_build_step_picked_before_a_shutdown_starts_nothing(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(fleet)
    _cold(local)
    _ProcessSwap.starting.clear()
    pick = local._next_fleet_step

    def _pick_then_latch() -> Callable[[], None] | None:
        step = pick()
        local._shut_down = True
        return step

    monkeypatch.setattr(local, "_next_fleet_step", _pick_then_latch)
    reader = _Caller(local._ensure_fleet)
    reader.join()
    assert reader.heard is None
    assert _ProcessSwap.starting.is_set() is False
    assert local._swaps == {}
    assert local._fleet_work is None


def test_engine_roles_and_the_engine_list_while_cold_start_nothing(fleet: Any) -> None:
    local = _local(fleet)
    _cold(local)
    processes = len(_ProcessSwap.live)
    _ProcessSwap.starting.clear()
    assert CliRunner().invoke(cli_app, ["--json", "engine", "list"]).exit_code == 0
    assert CliRunner().invoke(cli_app, ["--json", "engine", "roles"]).exit_code == 0
    assert engine_service.roles()
    assert _work_threads() == 0
    assert _ProcessSwap.starting.is_set() is False
    assert len(_ProcessSwap.live) == processes
    assert local._fleet_work is None


@pytest.fixture(autouse=True)
def _no_model_load_and_an_open_gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A load request racing a stopped fake engine would rebuild it; a shut gate holds teardown."""
    monkeypatch.setattr(FleetProvider, "_preload_roles", lambda self, roles=None: None)
    yield
    gate = _ProcessSwap.start_gate
    if gate is not None:
        gate.set()
