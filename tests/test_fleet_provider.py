"""Tests for FleetProvider routing and llama-swap lifecycle."""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.provider import FleetProvider, _least_in_flight
from lilbee.providers.roles import RerankMode, WorkerRole

_GB = 1024**3


def _fake_client(in_flight: int = 0) -> MagicMock:
    client = MagicMock()
    client.in_flight = in_flight
    return client


def _fake_launch(
    role: WorkerRole,
    *,
    model: str = "",
    slots: int = 1,
    ctx: int = 0,
    weights_bytes: int = 0,
    replica: int = 0,
) -> MagicMock:
    launch = MagicMock()
    launch.role = role
    launch.model = model or f"model-{role.value}"
    launch.slots = slots
    launch.ctx = ctx
    launch.weights_bytes = weights_bytes
    launch.replica = replica
    return launch


class _FakeSwap:
    """A stand-in SwapManager recording lifecycle calls; ready roles are settable."""

    def __init__(self) -> None:
        self.started: list[list] = []
        self.reaps = 0
        self.reloads = 0
        self.shutdowns = 0
        self.ready: set[WorkerRole] = set()
        self.running = True
        self.bound = False  # bound to a shared engine (set by the adopt/bind path)
        self.bound_lifetime = True  # spawned with the crash-orphan death binding

    def reap_stale(self) -> None:
        self.reaps += 1

    def start(self, launches: list, **kwargs: object) -> None:
        self.started.append(launches)
        self.bound_lifetime = bool(kwargs.get("bind_lifetime", True))
        self.running = True

    def endpoint(self) -> str:
        return "http://fake-endpoint"

    def is_live(self) -> bool:
        # Default: the fake swap is considered live so existing tests that have
        # an empty client pool still raise ProviderError, not trigger a rebuild.
        return True

    def role_ready(self, role: WorkerRole) -> bool:
        return role in self.ready

    def reload(self, launches: list) -> None:
        self.reloads += 1

    def shutdown(self) -> None:
        self.shutdowns += 1
        self.running = False


@pytest.fixture(autouse=True)
def _no_real_probe(monkeypatch, tmp_path_factory):
    """No test in this module may probe real hardware, resolve real binaries, or
    touch the real per-user machine engine slot.

    capture_plan_probe and placeable_total_vram both resolve the engine binary
    and spawn device probes; on a host without the bundled engine (CI) they raise,
    and on a dev box they silently probe the real GPUs. machine_engine_dir would
    otherwise let parallel tests collide on one real directory. Tests that
    exercise these override the stubs with their own recorders; placeability is
    stubbed true so a configured role is placeable unless a test says otherwise.
    """
    monkeypatch.setattr(planning_mod, "capture_plan_probe", lambda: None)
    monkeypatch.setattr(planning_mod, "assert_engine_probeable", lambda: None)
    monkeypatch.setattr(planning_mod, "placeable_total_vram", lambda: 0)
    monkeypatch.setattr(planning_mod, "role_model_placeable", lambda _role, _ref, _vram: True)
    mslot = tmp_path_factory.mktemp("machine-slot")
    monkeypatch.setattr(prov_mod, "machine_engine_dir", lambda: mslot)


def _install_engine(
    monkeypatch, tmp_path: Path, *, launches: list, swap: _FakeSwap | None = None
) -> _FakeSwap:
    """Patch the swap, client, planner, and ladder so _ensure_fleet builds fakes.

    The machine slot points at a fresh dir under the test's tmp_path, so tests
    never touch the real per-user engine slot and leave nothing behind, and
    stop_engine is inert (fakes own no processes).
    """
    import tempfile

    swap = swap or _FakeSwap()
    machine = Path(tempfile.mkdtemp(prefix="lilbee-test-slot-", dir=tmp_path))
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _data_dir, _group: swap)
    monkeypatch.setattr(prov_mod, "machine_engine_dir", lambda: machine)
    monkeypatch.setattr(prov_mod, "engine_pin", lambda: "test-pin")
    monkeypatch.setattr(prov_mod, "state_is_healthy", lambda _state: True)
    monkeypatch.setattr(prov_mod, "stop_engine", lambda _d: None)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _data_dir, **_kw: None)
    monkeypatch.setattr(planning_mod, "capture_plan_probe", lambda: None)
    monkeypatch.setattr(
        prov_mod, "LlamaServerClient", lambda _endpoint, _model, **_kw: _fake_client()
    )
    monkeypatch.setattr(
        planning_mod, "plan_all_launches", lambda: planning_mod.FleetPlan(tuple(launches))
    )
    return swap


def _provider_with_clients(clients: dict[WorkerRole, list[MagicMock]]) -> FleetProvider:
    """A provider with a fake swap already up and a client pool per role (no real start)."""
    p = FleetProvider()
    # Non-empty so _ensure_fleet short-circuits; roles without clients still error.
    roles = list(clients) or [WorkerRole.CHAT]
    p._role_group = {role: SwapGroup(role.value) for role in roles}
    p._swaps = {SwapGroup(role.value): _FakeSwap() for role in roles}
    p._clients = {role: list(cs) for role, cs in clients.items() if cs}
    return p


def test_warm_up_pool_rechecks_warming_after_readiness_probe(monkeypatch) -> None:
    # The readiness probe runs off the lock; if a concurrent warm starts during
    # it, the re-check under the lock abandons this dispatch so no second warm
    # thread spawns.
    p = FleetProvider()
    group = SwapGroup(WorkerRole.CHAT.value)
    p._role_group = {WorkerRole.CHAT: group}
    p._swaps = {group: _FakeSwap()}  # fleet up, role cold
    kicked = _stub_warm_thread(monkeypatch)

    def _probe_then_a_warm_wins_the_race() -> bool:
        p._warming = True  # a sibling warm started while we were probing
        return False

    monkeypatch.setattr(p, "_roles_ready", _probe_then_a_warm_wins_the_race)
    p.warm_up_pool()

    assert not kicked, "the re-check must abandon this dispatch, not start a second warm"


def _stub_warm_thread(monkeypatch) -> list[dict]:
    """Replace the warm-up daemon thread with a recorder; return the kicked list."""
    kicked: list[dict] = []
    monkeypatch.setattr(
        prov_mod.threading,
        "Thread",
        lambda **kw: SimpleNamespace(start=lambda: kicked.append(kw)),
    )
    return kicked


def test_warm_up_pool_rewarms_when_fleet_up_but_role_unloaded(monkeypatch) -> None:
    # llama-swap idle-unloads a model (ttl) by stopping only the llama-server
    # child; the swap handle stays in _swaps while the role goes cold. A prompt
    # sent into that gap must re-warm so llama-swap reloads on demand, not bounce
    # forever on a stale "swap exists therefore ready" assumption.
    p = FleetProvider()
    group = SwapGroup(WorkerRole.CHAT.value)
    p._role_group = {WorkerRole.CHAT: group}
    p._swaps = {group: _FakeSwap()}  # ready set empty -> role_ready False
    kicked = _stub_warm_thread(monkeypatch)

    p.warm_up_pool()

    assert kicked, "warm_up_pool must re-warm a cold role even while its swap is up"
    assert p._warming is True


def test_least_in_flight_picks_minimum() -> None:
    busy, idle = _fake_client(5), _fake_client(1)
    assert _least_in_flight([busy, idle]) is idle


def test_chat_routes_to_chat_server() -> None:
    from lilbee.providers.base import ChatResult, FinishReason

    client = _fake_client()
    result = ChatResult(text="from-server", tool_calls=(), finish_reason=FinishReason.STOP)
    client.chat_result.return_value = result
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    assert p.chat([{"role": "user", "content": "hi"}]) == result
    client.chat_result.assert_called_once()


def test_chat_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})  # no chat client, no in-process fallback
    with pytest.raises(ProviderError, match="No chat model server is running"):
        p.chat([{"role": "user", "content": "hi"}])


def test_chat_translates_options_before_routing_to_server() -> None:
    # The engine must apply the same option translation as in-process: num_predict
    # -> max_tokens (the server doesn't read num_predict) and drop the load-only
    # num_ctx. A raw passthrough would silently ignore the generation length.
    from lilbee.providers.base import ChatResult, FinishReason

    client = _fake_client(0)
    client.chat_result.return_value = ChatResult(
        text="ok", tool_calls=(), finish_reason=FinishReason.STOP
    )
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    p.chat(
        [{"role": "user", "content": "hi"}],
        options={"num_predict": 64, "temperature": 0.2, "num_ctx": 8192},
    )
    sent = client.chat_result.call_args.kwargs["options"]
    assert sent["max_tokens"] == 64
    assert sent["temperature"] == 0.2
    assert "num_predict" not in sent
    assert "num_ctx" not in sent


def test_embed_routes_to_server_when_present() -> None:
    client = _fake_client()
    client.embed.return_value = [[0.1]]
    p = _provider_with_clients({WorkerRole.EMBED: [client]})
    assert p.embed(["a"]) == [[0.1]]


def test_count_tokens_routes_to_embed_server() -> None:
    client = _fake_client()
    client.count_tokens.return_value = 9
    p = _provider_with_clients({WorkerRole.EMBED: [client]})
    assert p.count_tokens("hello") == 9
    client.count_tokens.assert_called_once_with("hello")


def test_count_tokens_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})
    with pytest.raises(ProviderError):
        p.count_tokens("hello")


def test_embed_routes_to_least_busy_replica() -> None:
    # Data-parallel replicas: a request goes to the idlest replica in the pool.
    busy, idle = _fake_client(5), _fake_client(1)
    idle.embed.return_value = [[0.2]]
    p = _provider_with_clients({WorkerRole.EMBED: [busy, idle]})
    assert p.embed(["a"]) == [[0.2]]
    idle.embed.assert_called_once()
    busy.embed.assert_not_called()


def test_adopt_group_builds_a_client_per_replica(monkeypatch, tmp_path: Path) -> None:
    launches = [_fake_launch(WorkerRole.EMBED), _fake_launch(WorkerRole.EMBED)]
    _install_engine(monkeypatch, tmp_path, launches=launches)
    p = FleetProvider()
    p._ensure_fleet()
    assert len(p._clients[WorkerRole.EMBED]) == 2  # one client per replica launch


def test_ensure_fleet_records_skipped_not_installed_from_plan(monkeypatch, tmp_path: Path) -> None:
    # The plan reports a configured-but-missing chat model; _plan_and_spawn records
    # it so the warm finalizer can fail chat with a named reason. The installed embed
    # role still starts.
    _install_engine(monkeypatch, tmp_path, launches=[_fake_launch(WorkerRole.EMBED)])
    monkeypatch.setattr(
        planning_mod,
        "plan_all_launches",
        lambda: planning_mod.FleetPlan(
            (_fake_launch(WorkerRole.EMBED),),
            skipped_not_installed={WorkerRole.CHAT: "org/repo/missing-chat.gguf"},
        ),
    )
    p = FleetProvider()
    p._ensure_fleet()
    assert p._skipped_not_installed == {WorkerRole.CHAT: "org/repo/missing-chat.gguf"}


def test_reload_propagates_a_probe_failure_on_the_resurrect_path(monkeypatch) -> None:
    # A reload with nothing running recaptures the probe like a first build; a
    # wedged probe there must fail loud, not silently leave the box empty. The
    # raise lands before the stop phase, so no running group is torn down.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)

    def _wedged() -> None:
        raise ProviderError(
            "device probe wedged", provider="llama-server", kind=ProviderErrorKind.CONNECTION
        )

    monkeypatch.setattr(planning_mod, "capture_plan_probe", _wedged)
    p = FleetProvider()  # no swaps: the resurrect path recaptures the probe
    with pytest.raises(ProviderError) as excinfo:
        p._reload_pass()
    assert excinfo.value.kind is ProviderErrorKind.CONNECTION


def test_reload_stays_quiet_when_the_engine_binary_is_missing(monkeypatch) -> None:
    # A missing binary on the reload resurrect path aborts quietly (nothing to
    # serve), mirroring the build path rather than raising.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)

    def _no_binary() -> None:
        raise ProviderError(
            "no engine binary", provider="llama-server", kind=ProviderErrorKind.NOT_FOUND
        )

    monkeypatch.setattr(planning_mod, "capture_plan_probe", _no_binary)
    p = FleetProvider()
    p._reload_pass()  # must not raise
    assert p._swaps == {}


def test_drop_group_prunes_its_dir_entry() -> None:
    # A dir entry must not outlive its group: a stale entry makes _reload_dir see
    # two dirs and pick one arbitrarily, splitting the provider across dirs.
    p = FleetProvider()
    group = SwapGroup.CHAT
    p._swaps = {group: _FakeSwap()}
    p._group_dirs = {group: Path("/slot/a")}
    p._drop_group(group)
    assert group not in p._group_dirs


def test_drop_swap_refs_clears_the_dir_map() -> None:
    # A full teardown leaves no group, so it must leave no dir map either.
    p = FleetProvider()
    p._group_dirs = {SwapGroup.CHAT: Path("/slot/a"), SwapGroup.EMBED: Path("/slot/b")}
    p._drop_swap_refs()
    assert p._group_dirs == {}


def test_restart_engines_releases_and_prunes_holds(monkeypatch, tmp_path: Path) -> None:
    # A config-change restart must release membership and clear the hold map: a
    # stale hold would later evict a foreign engine that claimed the dir, and keep
    # it falsely live so its real last user could never reap it.
    from lilbee.runtime.engine_lock import hold_user_lock, live_users_exist

    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    p = FleetProvider()
    p._engine_holds = {tmp_path: hold_user_lock(tmp_path)}
    assert live_users_exist(tmp_path) is True  # we hold membership

    p._release_engines(config_changed=True)

    assert p._engine_holds == {}  # hold map pruned
    assert live_users_exist(tmp_path) is False  # membership released, not left stale
    assert stopped == [tmp_path]  # engine stopped for the config change


def test_reload_of_a_bound_engine_reacquires_instead_of_duplicating(monkeypatch) -> None:
    # A provider bound to another process's engine owns none of its groups. A model
    # change must not restart the group "in place": a bound manager's shutdown only
    # detaches, so restarting spawns a SECOND full fleet into the shared slot, sized
    # blind against the incumbent's resident VRAM (an OOM on a small-VRAM box). The
    # binder must drop its binding and re-run the acquisition ladder instead.
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
    built: list = []

    def _record_manager(data_dir, group):
        built.append((data_dir, group))
        return _FakeSwap()

    monkeypatch.setattr(prov_mod, "SwapManager", _record_manager)
    monkeypatch.setattr(
        planning_mod,
        "plan_all_launches",
        lambda: planning_mod.FleetPlan((_fake_launch(WorkerRole.CHAT),)),
    )
    reacquired: list[bool] = []

    def _fake_acquire(_root) -> bool:
        reacquired.append(True)
        return False  # rebound/overflowed off the shared slot; nothing to preload

    p = FleetProvider()
    monkeypatch.setattr(p, "_acquire_engine", _fake_acquire)
    bound_swap = _FakeSwap()
    bound_swap.bound = True
    group = SwapGroup.CHAT
    p._swaps = {group: bound_swap}
    p._role_group = {WorkerRole.CHAT: group}
    p._group_dirs = {group: prov_mod.machine_engine_dir()}

    p._reload_pass()

    assert reacquired == [True]  # re-acquired through the ladder, not restarted in place
    assert bound_swap.shutdowns == 1  # the binding was dropped (detached)
    assert built == []  # no duplicate fleet spawned into the shared slot


def test_reload_of_a_bound_engine_preloads_the_reacquired_roles(monkeypatch) -> None:
    # When the re-acquire lands (rebinds or overflows), the reload warms the roles
    # it now serves off-thread, just like an owned restart does.
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
    kicked = _stub_warm_thread(monkeypatch)

    def _fake_acquire(_root) -> bool:
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}  # ladder adopted the role
        return True

    p = FleetProvider()
    monkeypatch.setattr(p, "_acquire_engine", _fake_acquire)
    monkeypatch.setattr(p, "_release_engines", lambda: None)
    bound_swap = _FakeSwap()
    bound_swap.bound = True
    p._swaps = {SwapGroup.CHAT: bound_swap}
    p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}

    p._reload_pass()

    assert kicked and kicked[0]["name"] == "fleet-reload-warm"  # roles warmed off-thread


def test_ensure_fleet_refused_after_shutdown(monkeypatch, tmp_path: Path) -> None:
    """bb-dpp source guard: once shut down (and likely discarded by reset_services),
    a lingering warm-up/reload thread's _ensure_fleet must not spawn a new llama-swap
    on the dead provider -- that is exactly the duplicate that leaks on teardown."""
    swap = _install_engine(monkeypatch, tmp_path, launches=[_fake_launch(WorkerRole.CHAT)])
    p = FleetProvider()
    p._shutdown_swap()  # latches _shut_down (and reaps via a fresh SwapManager)
    assert p._ensure_fleet() is False
    assert swap.started == []  # no swap started after shutdown


def test_adopt_group_retires_old_clients_without_closing(monkeypatch, tmp_path: Path) -> None:
    # Re-adopting (a reload) must not close old clients in place (a
    # reader may still hold one); they are retired for deferred close.
    launch = _fake_launch(WorkerRole.EMBED)
    swap = _install_engine(monkeypatch, tmp_path, launches=[launch])
    p = FleetProvider()
    old = [_fake_client(), _fake_client()]
    p._clients = {WorkerRole.EMBED: old}

    with p._lock:
        p._adopt_group(SwapGroup.EMBED, swap, [launch])

    assert p._retiring_clients == old  # retired, not closed yet
    for client in old:
        client.close.assert_not_called()
    assert p._clients[WorkerRole.EMBED][0] not in old  # fresh client adopted


def test_retire_closes_prior_idle_generation_at_next_reload() -> None:
    # The prior reload's clients are closed at the
    # next reload, by when their readers (in_flight==0) have finished.
    p = FleetProvider()
    prior = [_fake_client(in_flight=0), _fake_client(in_flight=0)]
    p._retiring_clients = list(prior)
    current_old = [_fake_client(in_flight=0)]

    with p._lock:
        p._retire_clients(current_old)

    for client in prior:
        client.close.assert_called_once_with()  # prior idle generation closed
    assert p._retiring_clients == current_old  # this reload's clients now pending


def test_retire_keeps_busy_prior_client_for_a_later_reload() -> None:
    p = FleetProvider()
    busy = _fake_client(in_flight=1)  # a reader is still mid-request on it
    p._retiring_clients = [busy]

    with p._lock:
        p._retire_clients([])

    busy.close.assert_not_called()  # not closed while in flight
    assert busy in p._retiring_clients  # retained for the next reload


def test_drop_swap_refs_closes_retiring_clients() -> None:
    p = FleetProvider()
    retiring = _fake_client(in_flight=1)  # even a busy one is closed at shutdown
    live = _fake_client()
    p._clients = {WorkerRole.EMBED: [live]}
    p._retiring_clients = [retiring]

    p._drop_swap_refs(close_all=True)

    live.close.assert_called_once_with()
    retiring.close.assert_called_once_with()
    assert p._retiring_clients == []


def test_adopt_group_threads_rerank_mode(monkeypatch) -> None:
    launch = _fake_launch(WorkerRole.RERANK)
    launch.rerank_mode = RerankMode.LLM
    captured: dict[str, object] = {}

    def _capture(_endpoint, _model, **kw):
        captured["rerank_mode"] = kw.get("rerank_mode")
        return _fake_client()

    monkeypatch.setattr(prov_mod, "SwapManager", lambda _data_dir, _group: _FakeSwap())
    monkeypatch.setattr(prov_mod, "LlamaServerClient", _capture)
    monkeypatch.setattr(
        prov_mod,
        "_configured_model_for",
        lambda role: "m-rerank" if role is WorkerRole.RERANK else "",
    )
    monkeypatch.setattr(planning_mod, "role_model_placeable", lambda _role, _ref, _vram: True)
    monkeypatch.setattr(
        planning_mod, "plan_all_launches", lambda: planning_mod.FleetPlan((launch,))
    )
    FleetProvider()._ensure_fleet()
    assert captured["rerank_mode"] is RerankMode.LLM


def test_adopt_group_gives_embed_client_cold_load_deadline_only(monkeypatch) -> None:
    # The EMBED client waits out a still-warming replica for the full cold-load
    # budget (so a cold-start burst never drops files); rerank/chat/vision keep the
    # short interactive attempt cap (deadline None).
    from lilbee.providers.fleet.swap_config import cold_load_timeout_s

    weights = 6_000_000_000
    launches = [
        _fake_launch(WorkerRole.EMBED, weights_bytes=weights),
        _fake_launch(WorkerRole.RERANK, weights_bytes=weights),
    ]
    for launch in launches:
        launch.rerank_mode = None
    captured: dict[WorkerRole, object] = {}

    def _capture(_endpoint, model, **kw):
        captured[model] = kw.get("embed_busy_deadline_s")
        return _fake_client()

    for launch in launches:
        launch.model_id = launch.role
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _data_dir, _group: _FakeSwap())
    monkeypatch.setattr(prov_mod, "LlamaServerClient", _capture)
    monkeypatch.setattr(
        planning_mod, "plan_all_launches", lambda: planning_mod.FleetPlan(tuple(launches))
    )
    FleetProvider()._ensure_fleet()
    assert captured[WorkerRole.EMBED] == cold_load_timeout_s(weights)
    assert captured[WorkerRole.RERANK] is None


def test_embed_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})
    with pytest.raises(ProviderError, match="No embed model server is running"):
        p.embed(["a"])


def test_rerank_routes_to_engine() -> None:
    client = _fake_client()
    client.rerank.return_value = [0.9, 0.1]
    p = _provider_with_clients({WorkerRole.RERANK: [client]})
    assert p.rerank("q", ["a", "b"]) == [0.9, 0.1]
    client.rerank.assert_called_once_with("q", ["a", "b"])


def test_rerank_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})
    with pytest.raises(ProviderError, match="No rerank model server is running"):
        p.rerank("q", ["a"])


def test_fit_chat_context_passthrough_when_ctx_unknown() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    msgs = [{"role": "user", "content": "hi"}]
    # _chat_ctx is None until a chat launch is adopted: no windowing, same list.
    assert p._fit_chat_context(msgs, None, None, "m") is msgs


def test_fit_chat_context_windows_overlong_history() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p._chat_ctx = 2048
    msgs: list[dict] = [{"role": "system", "content": "s"}]
    for _ in range(30):
        msgs.append({"role": "user", "content": "x" * 500})
        msgs.append({"role": "assistant", "content": "y" * 500})
    msgs.append({"role": "user", "content": "final"})
    out = p._fit_chat_context(msgs, None, None, "m")
    assert len(out) < len(msgs)
    assert out[0]["role"] == "system"
    assert out[-1]["content"] == "final"


def test_fit_chat_context_raises_context_overflow_when_unfixable() -> None:
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p._chat_ctx = 64
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "x" * 9000}]
    with pytest.raises(ProviderError) as excinfo:
        p._fit_chat_context(msgs, None, None, "qwen")
    assert excinfo.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW


def test_fit_chat_context_clamps_a_greedy_output_reservation() -> None:
    """A num_predict the window cannot honor shrinks to the floor, not an error.

    Agent clients reserve output from their own (under-counting) prompt
    estimate; a prompt that fits the window with the default generation room
    must be served, with llama-server stopping at the context edge if the
    generation runs long.
    """
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p._chat_ctx = 2000
    msgs = [{"role": "user", "content": "x" * 2000}]
    out = p._fit_chat_context(msgs, None, {"num_predict": 1900}, "m")
    assert out[-1]["content"] == "x" * 2000


def test_fit_chat_context_keeps_history_a_greedy_reservation_would_evict() -> None:
    """The clamp is a policy, not a rescue: it must fire before history is lost.

    An agent that reserves most of the window on every call left a prompt
    budget of a few dozen tokens. The final turn still squeezed in, so the fit
    "succeeded" and the reservation was honored verbatim -- silently dropping
    the entire conversation on exactly the clients the clamp exists for.
    """
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p._chat_ctx = 8192
    msgs: list[dict] = [{"role": "system", "content": "s"}]
    for i in range(10):
        msgs.append({"role": "user", "content": f"turn {i} " + "x" * 200})
        msgs.append({"role": "assistant", "content": "y" * 200})
    msgs.append({"role": "user", "content": "final"})

    greedy = p._fit_chat_context(msgs, None, {"num_predict": 8000}, "m")

    # The same history the default reservation preserves, not a bare final turn.
    assert greedy == p._fit_chat_context(msgs, None, None, "m")
    assert len(greedy) > 2


def test_fit_chat_context_honors_a_small_output_reservation() -> None:
    """Capping the reserve at the default must not shrink a modest request."""
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p._chat_ctx = 8192
    msgs: list[dict] = [{"role": "system", "content": "s"}]
    for i in range(200):
        msgs.append({"role": "user", "content": f"turn {i} " + "x" * 200})
        msgs.append({"role": "assistant", "content": "y" * 200})
    msgs.append({"role": "user", "content": "final"})

    # A caller wanting only 16 tokens out earns more prompt room, not less.
    small = p._fit_chat_context(msgs, None, {"num_predict": 16}, "m")
    default = p._fit_chat_context(msgs, None, None, "m")
    assert len(small) > len(default)


def test_fit_chat_context_raises_when_even_the_floor_cannot_fit() -> None:
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p._chat_ctx = 2000
    # far beyond ctx minus the floor reserve: no reservation shrink can save it
    msgs = [{"role": "user", "content": "x" * 40_000}]
    with pytest.raises(ProviderError) as excinfo:
        p._fit_chat_context(msgs, None, {"num_predict": 1900}, "m")
    assert excinfo.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW


def test_vision_ocr_routes_to_engine_for_configured_model(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    client = _fake_client()
    client.chat.return_value = "ocr text"
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    assert p.vision_ocr(b"png", "org/repo/v.gguf") == "ocr text"


def test_vision_ocr_model_override_raises(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    p = _provider_with_clients({WorkerRole.VISION: [_fake_client()]})
    # override != the server's configured vision model -> hard error (no fallback)
    with pytest.raises(ProviderError, match="serves the configured vision model"):
        p.vision_ocr(b"png", "org/repo/other.gguf")


def test_chat_model_override_raises(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr(cfg, "chat_model", "org/repo/configured.gguf")
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    with pytest.raises(ProviderError, match="serves the configured chat model"):
        p.chat([{"role": "user", "content": "hi"}], model="org/repo/other.gguf")


def test_chat_model_override_error_is_bad_request_kind(monkeypatch) -> None:
    """The mismatch is a client error; BAD_REQUEST maps it to a 400 envelope, not a 500."""
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr(cfg, "chat_model", "org/repo/configured.gguf")
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    with pytest.raises(ProviderError) as excinfo:
        p.chat([{"role": "user", "content": "hi"}], model="org/repo/other.gguf")
    assert excinfo.value.kind is ProviderErrorKind.BAD_REQUEST


def test_vision_call_returns_text() -> None:
    client = _fake_client()
    client.chat.return_value = "OCR text"
    assert prov_mod._vision_call(client, [{"role": "user", "content": "x"}], None) == "OCR text"


def test_vision_call_enforces_timeout() -> None:
    client = _fake_client()
    client.chat_bounded.return_value = "OCR text"
    assert prov_mod._vision_call(client, [{"role": "user", "content": "x"}], 5.0) == "OCR text"
    client.chat_bounded.assert_called_once()
    client.chat.assert_not_called()  # the timed path streams via chat_bounded, not chat


def test_vision_call_caps_output_tokens(monkeypatch) -> None:
    # A runaway OCR page can loop to tens of thousands of chars and dominate a
    # scan's time; the call must cap generation at cfg.vision_ocr_max_tokens.
    monkeypatch.setattr(cfg, "vision_ocr_max_tokens", 4096)
    client = _fake_client()
    client.chat.return_value = "OCR text"
    prov_mod._vision_call(client, [{"role": "user", "content": "x"}], None)
    assert client.chat.call_args.kwargs["options"] == {"max_tokens": 4096}


def test_vision_ocr_retries_busy_then_succeeds(monkeypatch) -> None:
    # A 429 mid-ingest is transient (cold replicas, slot contention); the vision
    # path must back off and retry so the page is ingested, not dropped.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    busy = ProviderError("busy", provider="llama-server", kind=ProviderErrorKind.RATE_LIMIT)
    client = _fake_client()
    client.chat.side_effect = [busy, busy, "ocr text"]
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    assert p.vision_ocr(b"png", "org/repo/v.gguf") == "ocr text"
    assert client.chat.call_count == 3


def test_vision_ocr_retries_past_attempt_cap_until_deadline(monkeypatch) -> None:
    # On a deep OCR queue the drain time exceeds the fixed attempt budget, so a
    # 429'd image must keep retrying until its own deadline rather than being
    # dropped after _VISION_BUSY_RETRIES attempts (bb-z34 regression).
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    monkeypatch.setattr("lilbee.providers.fleet.client.time.monotonic", lambda: 0.0)
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    busy = ProviderError("busy", provider="llama-server", kind=ProviderErrorKind.RATE_LIMIT)
    client = _fake_client()
    # timeout > 0 routes through the bounded (chat_bounded) path.
    client.chat_bounded.side_effect = [busy] * (prov_mod._VISION_BUSY_RETRIES + 5) + ["ocr text"]
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    assert p.vision_ocr(b"png", "org/repo/v.gguf", timeout=300.0) == "ocr text"
    assert client.chat_bounded.call_count == prov_mod._VISION_BUSY_RETRIES + 6


def test_vision_ocr_gives_up_when_deadline_passes(monkeypatch) -> None:
    # A fleet that never frees a slot fails the page once its deadline passes.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    clock = {"t": 0.0}
    monkeypatch.setattr("lilbee.providers.fleet.client.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("lilbee.providers.fleet.provider.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    monkeypatch.setattr(cfg, "vision_load_budget_s", 0.0)  # deadline == the 20s timeout

    def _busy(*_a, **_k):
        clock["t"] += 5.0  # each attempt advances the clock toward the deadline
        raise ProviderError("busy", provider="llama-server", kind=ProviderErrorKind.RATE_LIMIT)

    client = _fake_client()
    client.chat_bounded.side_effect = _busy
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    with pytest.raises(ProviderError) as excinfo:
        p.vision_ocr(b"png", "org/repo/v.gguf", timeout=20.0)
    assert excinfo.value.kind is ProviderErrorKind.RATE_LIMIT


def test_vision_ocr_deadline_passed_before_generation_raises_timeout(monkeypatch) -> None:
    # A slot that only frees after the image's deadline has already passed yields a
    # user-facing timeout, not the internal budget-exhausted signal.
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    monkeypatch.setattr(cfg, "vision_load_budget_s", 0.0)  # deadline == the 20s timeout
    ticks = iter([0.0, 100.0])  # deadline computed at 0; the slot frees at 100 (past 20)
    monkeypatch.setattr("lilbee.providers.fleet.provider.time.monotonic", lambda: next(ticks))
    client = _fake_client()
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    with pytest.raises(ProviderError, match="timed out waiting for a free vision slot"):
        p.vision_ocr(b"png", "org/repo/v.gguf", timeout=20.0)
    client.chat.assert_not_called()  # the deadline lapsed before any generation ran


def test_vision_ocr_does_not_retry_non_busy_errors(monkeypatch) -> None:
    # A genuine extraction failure must surface immediately, not burn the budget.
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    client = _fake_client()
    client.chat.side_effect = ProviderError("boom", provider="llama-server")
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    with pytest.raises(ProviderError, match="boom"):
        p.vision_ocr(b"png", "org/repo/v.gguf")
    assert client.chat.call_count == 1


def test_vision_pool_pairs_each_replica_with_its_fitted_slots(monkeypatch) -> None:
    # Dispatch admits per replica at the servers' fitted --parallel slots: planning
    # can fit fewer slots than vision_ocr_concurrency asks for, and admitting more
    # than the real slot count oversubscribes that server into a 429 storm.
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 16)
    first_client, second_client = _fake_client(), _fake_client()
    p = _provider_with_clients({WorkerRole.VISION: [first_client, second_client]})
    p._role_group[WorkerRole.VISION] = SwapGroup.VISION
    p._launches[SwapGroup.VISION] = (
        _fake_launch(WorkerRole.VISION, slots=3),
        _fake_launch(WorkerRole.VISION, slots=2, replica=1),
    )
    assert p._vision_pool() == [
        prov_mod._VisionReplica(first_client, 3),
        prov_mod._VisionReplica(second_client, 2),
    ]


def test_vision_pool_falls_back_to_configured_slots(monkeypatch) -> None:
    # Without a launch snapshot (mid-reload) the configured per-server ceiling holds.
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
    client = _fake_client()
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    assert p._vision_pool() == [prov_mod._VisionReplica(client, 4)]


def test_vision_pool_falls_back_on_launch_client_mismatch(monkeypatch) -> None:
    # A launch snapshot that no longer matches the client pool (mid-reload race)
    # must not misassign slot counts; the configured ceiling applies instead.
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
    clients = [_fake_client(), _fake_client()]
    p = _provider_with_clients({WorkerRole.VISION: clients})
    p._role_group[WorkerRole.VISION] = SwapGroup.VISION
    p._launches[SwapGroup.VISION] = (_fake_launch(WorkerRole.VISION, slots=3),)
    assert [replica.slots for replica in p._vision_pool()] == [4, 4]


def test_vision_slot_capacity_sums_fitted_launch_slots() -> None:
    # The ingest fan-out sizes to this: the sum of the running servers' fitted slots.
    p = _provider_with_clients({WorkerRole.VISION: [_fake_client(), _fake_client()]})
    p._role_group[WorkerRole.VISION] = SwapGroup.VISION
    p._launches[SwapGroup.VISION] = (
        _fake_launch(WorkerRole.VISION, slots=3),
        _fake_launch(WorkerRole.VISION, slots=2, replica=1),
    )
    assert p.vision_slot_capacity() == 5


def test_vision_slot_capacity_none_before_fleet_up() -> None:
    # No launch snapshot yet: the fan-out keeps its own estimate.
    assert FleetProvider().vision_slot_capacity() is None


def test_vision_dispatcher_caps_each_replica_at_its_slots() -> None:
    # The ingest fan-out can launch far more concurrent OCR requests than the
    # servers have slots; the dispatcher assigns a request to a replica only while
    # that replica has a free slot, so no server can ever be oversubscribed by
    # lilbee's own traffic. All callers still run.
    import threading
    import time

    dispatcher = prov_mod._VisionDispatcher()
    replica_a, replica_b = _fake_client(), _fake_client()
    pool = [prov_mod._VisionReplica(replica_a, 2), prov_mod._VisionReplica(replica_b, 1)]

    lock = threading.Lock()
    live: dict[int, int] = {}
    peaks: dict[int, int] = {}
    ran = 0

    def _hold() -> None:
        nonlocal ran
        with dispatcher.slot(pool) as client:
            key = id(client)
            with lock:
                live[key] = live.get(key, 0) + 1
                peaks[key] = max(peaks.get(key, 0), live[key])
            time.sleep(0.02)
            with lock:
                live[key] -= 1
                ran += 1

    threads = [threading.Thread(target=_hold) for _ in range(9)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert ran == 9  # every caller ran, queued in-process instead of 429ing
    assert peaks[id(replica_a)] <= 2 and peaks[id(replica_b)] <= 1  # per-replica cap
    assert set(peaks) == {id(replica_a), id(replica_b)}  # both replicas pulled work


def test_vision_dispatcher_blocks_on_a_full_pool_until_a_slot_frees() -> None:
    # Every slot held up front, so the waiter's first pick returns None and it
    # parks on the condition's timed re-poll; freeing a slot must wake it. Filling
    # the pool before the waiter starts makes the block deterministic rather than
    # relying on threads racing into a full pool.
    import threading

    dispatcher = prov_mod._VisionDispatcher()
    only = _fake_client()
    pool = [prov_mod._VisionReplica(only, 1)]

    acquired = threading.Event()

    def _wait_for_slot() -> None:
        with dispatcher.slot(pool):
            acquired.set()

    with dispatcher.slot(pool) as held:
        assert held is only  # the pool's one slot is now taken
        waiter = threading.Thread(target=_wait_for_slot)
        waiter.start()
        # The waiter cannot get in while the slot is held; it is parked in the
        # condition wait, not spinning on a busy pool.
        assert not acquired.wait(timeout=0.2)
    # Slot released here; release notifies the waiter, which wakes and acquires.
    assert acquired.wait(timeout=5.0)
    waiter.join(timeout=5.0)
    assert not waiter.is_alive()


def test_vision_dispatcher_prefers_replica_with_most_free_slots() -> None:
    # Balanced routing drains fastest: the next request goes to the replica with
    # the most free slots, not to whichever raced ahead.
    dispatcher = prov_mod._VisionDispatcher()
    roomy, tight = _fake_client(), _fake_client()
    pool = [prov_mod._VisionReplica(tight, 1), prov_mod._VisionReplica(roomy, 3)]
    with dispatcher.slot(pool) as first, dispatcher.slot(pool) as second:
        assert first is roomy  # 3 free beats 1
        assert second is roomy  # still 2 free vs 1
        with dispatcher.slot(pool) as third:
            assert third is tight  # 1 free each; the earlier pool entry wins the tie


def test_vision_dispatcher_skips_unhealthy_replica() -> None:
    # A replica marked unhealthy takes no new assignments; its slots are dead
    # weight until the half-open cooldown re-admits it.
    dispatcher = prov_mod._VisionDispatcher()
    dead, alive = _fake_client(), _fake_client()
    dead.healthy = False
    pool = [prov_mod._VisionReplica(dead, 4), prov_mod._VisionReplica(alive, 1)]
    with dispatcher.slot(pool) as client:
        assert client is alive


def test_vision_dispatcher_falls_back_when_all_unhealthy() -> None:
    # With every replica unhealthy the dispatcher still assigns (mirrors
    # _least_in_flight): the call surfaces the error instead of queueing forever,
    # and a success restores the replica.
    dispatcher = prov_mod._VisionDispatcher()
    dead = _fake_client()
    dead.healthy = False
    pool = [prov_mod._VisionReplica(dead, 1)]
    with dispatcher.slot(pool) as client:
        assert client is dead


def test_dispatch_vision_fails_over_and_marks_health() -> None:
    # A connection-dead replica is marked unhealthy and the request retries once
    # on another replica, mirroring _call_with_failover.
    import httpx as _httpx

    from lilbee.providers.base import ProviderError

    dead, alive = _fake_client(), _fake_client()
    dead.chat.side_effect = _httpx.ConnectError("refused")
    alive.chat.return_value = "ocr text"
    pool = [prov_mod._VisionReplica(dead, 2), prov_mod._VisionReplica(alive, 1)]
    result = prov_mod._dispatch_vision(pool, lambda c: c.chat([], options={}, stream=False))
    assert result == "ocr text"
    dead.mark_unhealthy.assert_called_once()
    alive.mark_healthy.assert_called_once()

    # With no other replica the failure surfaces as the no-replica error.
    lone = _fake_client()
    lone.chat.side_effect = _httpx.ConnectError("refused")
    with pytest.raises(ProviderError, match="no healthy replica"):
        prov_mod._dispatch_vision(
            [prov_mod._VisionReplica(lone, 1)],
            lambda c: c.chat([], options={}, stream=False),
        )


def test_dispatch_vision_marks_second_replica_unhealthy_on_retry_failure() -> None:
    # The failover leg stamps health too: a second dead replica is marked
    # unhealthy and the failure propagates.
    import httpx as _httpx

    dead_a, dead_b = _fake_client(), _fake_client()
    dead_a.chat.side_effect = _httpx.ConnectError("refused")
    dead_b.chat.side_effect = _httpx.ConnectError("refused")
    pool = [prov_mod._VisionReplica(dead_a, 2), prov_mod._VisionReplica(dead_b, 1)]
    with pytest.raises(_httpx.ConnectError):
        prov_mod._dispatch_vision(pool, lambda c: c.chat([], options={}, stream=False))
    dead_a.mark_unhealthy.assert_called_once()
    dead_b.mark_unhealthy.assert_called_once()


def test_chat_streams_from_server() -> None:
    client = _fake_client(0)
    client.chat_stream_items.return_value = iter(["a", "b"])
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    assert list(p.chat([{"role": "user", "content": "hi"}], stream=True)) == ["a", "b"]
    client.chat_stream_items.assert_called_once()


def test_chat_stream_with_no_frames_yields_nothing() -> None:
    # Priming must pass an immediately-exhausted stream through, not error on it.
    client = _fake_client(0)
    client.chat_stream_items.return_value = iter([])
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    assert list(p.chat([{"role": "user", "content": "hi"}], stream=True)) == []


def test_chat_stream_close_before_iteration_releases_the_request() -> None:
    closed: list[bool] = []

    def _frames():
        try:
            yield "a"
            yield "b"
        finally:
            closed.append(True)

    client = _fake_client(0)
    client.chat_stream_items.return_value = _frames()
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    stream = p.chat([{"role": "user", "content": "hi"}], stream=True)
    stream.close()  # truncated before consuming anything
    assert closed == [True]  # the source stream (and its request slot) was released


def test_supports_rerank_always_true() -> None:
    # llama-server reranks any cross-encoder GGUF via --pooling rank.
    assert FleetProvider().supports_rerank() is True


def test_list_chat_models_empty() -> None:
    # The local engine has no frontier-provider catalog.
    assert FleetProvider().list_chat_models("openai") == []


def test_pull_model_not_supported() -> None:
    with pytest.raises(NotImplementedError, match="cannot pull"):
        FleetProvider().pull_model("org/repo/m.gguf")


def test_list_models_reads_registry(monkeypatch) -> None:
    services = MagicMock()
    manifest_a, manifest_b = MagicMock(), MagicMock()
    manifest_a.ref, manifest_b.ref = "z/repo/b.gguf", "a/repo/a.gguf"
    services.registry.list_installed.return_value = [manifest_a, manifest_b]
    monkeypatch.setattr("lilbee.app.services.get_services", lambda: services)
    assert FleetProvider().list_models() == ["a/repo/a.gguf", "z/repo/b.gguf"]


def test_show_model_reads_gguf_headers(monkeypatch) -> None:
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _m: Path("/m/x.gguf")
    )
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"architecture": "llama"}
    )
    assert FleetProvider().show_model("org/repo/x.gguf") == {"architecture": "llama"}


def test_show_model_returns_none_when_unresolved(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    def _raise(_m: str) -> Path:
        raise ProviderError("not installed")

    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _raise)
    assert FleetProvider().show_model("org/repo/x.gguf") is None


def test_get_capabilities_rerank_ref(monkeypatch) -> None:
    monkeypatch.setattr("lilbee.catalog.is_rerank_ref", lambda _m: True)
    assert FleetProvider().get_capabilities("org/repo/rerank.gguf") == ["rerank"]


def test_get_capabilities_vision_when_mmproj_present(monkeypatch) -> None:
    monkeypatch.setattr("lilbee.catalog.is_rerank_ref", lambda _m: False)
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _m: Path("/m/v.gguf")
    )
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.find_mmproj_for_model", lambda _p: Path("/m/mmproj.gguf")
    )
    assert FleetProvider().get_capabilities("org/repo/v.gguf") == ["completion", "vision"]


def test_get_capabilities_completion_only_without_mmproj(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr("lilbee.catalog.is_rerank_ref", lambda _m: False)
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _m: Path("/m/c.gguf")
    )

    def _no_mmproj(_p: Path) -> Path:
        raise ProviderError("no mmproj")

    monkeypatch.setattr("lilbee.providers.gguf_meta.find_mmproj_for_model", _no_mmproj)
    assert FleetProvider().get_capabilities("org/repo/c.gguf") == ["completion"]


_TOOL_AWARE_TEMPLATE = "{% if tools %}{{ tool_calls }}{% endif %}"
_PLAIN_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


@pytest.fixture()
def _clear_tools_cache():
    """``_supports_tools_cached`` is module-level lru_cache; clear it per test so
    a True/False from one case can't leak into the next via a shared path key."""
    prov_mod._supports_tools_cached.cache_clear()
    yield
    prov_mod._supports_tools_cached.cache_clear()


def test_supports_tools_true_for_tool_aware_template(monkeypatch, tmp_path, _clear_tools_cache):
    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: gguf)
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata",
        lambda _p: {"chat_template": _TOOL_AWARE_TEMPLATE},
    )
    assert FleetProvider().supports_tools("org/repo/chat.gguf") is True


def test_supports_tools_false_for_plain_template(monkeypatch, tmp_path, _clear_tools_cache):
    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: gguf)
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata",
        lambda _p: {"chat_template": _PLAIN_TEMPLATE},
    )
    assert FleetProvider().supports_tools("org/repo/chat.gguf") is False


def test_supports_tools_false_when_metadata_unreadable(monkeypatch, tmp_path, _clear_tools_cache):
    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: gguf)
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: None)
    assert FleetProvider().supports_tools("org/repo/chat.gguf") is False


def test_supports_tools_false_when_template_missing(monkeypatch, tmp_path, _clear_tools_cache):
    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: gguf)
    # Metadata present but no chat_template key -> template is not a str.
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"architecture": "llama"}
    )
    assert FleetProvider().supports_tools("org/repo/chat.gguf") is False


def test_supports_tools_false_when_model_unresolved(monkeypatch, _clear_tools_cache):
    from lilbee.providers.base import ProviderError

    def _raise(_m: str) -> Path:
        raise ProviderError("not installed")

    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _raise)
    assert FleetProvider().supports_tools("org/repo/missing.gguf") is False


def test_supports_tools_tolerates_unstattable_path(monkeypatch, _clear_tools_cache):
    """A resolved path whose ``stat()`` fails still probes (mtime falls back to 0)."""
    bogus = Path("/does/not/exist/chat.gguf")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: bogus)
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata",
        lambda _p: {"chat_template": _TOOL_AWARE_TEMPLATE},
    )
    assert FleetProvider().supports_tools("org/repo/chat.gguf") is True


# --- llama-swap lifecycle ----------------------------------------------------


def test_ensure_fleet_starts_once_and_builds_clients(monkeypatch, tmp_path: Path) -> None:
    launches = [_fake_launch(WorkerRole.CHAT, slots=4, ctx=32768), _fake_launch(WorkerRole.EMBED)]
    swap = _install_engine(monkeypatch, tmp_path, launches=launches)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert len(swap.started) == 2  # one start per placed role group
    assert set(p._clients) == {WorkerRole.CHAT, WorkerRole.EMBED}  # one client per placed role
    assert p._chat_slots == 4  # chat capacity / ctx taken from the chat launch
    assert p._chat_ctx == 32768
    p._ensure_fleet()  # second call reuses the running groups
    assert len(swap.started) == 2


def test_ensure_fleet_defaults_chat_slots_without_chat_launch(monkeypatch, tmp_path: Path) -> None:
    _install_engine(monkeypatch, tmp_path, launches=[_fake_launch(WorkerRole.EMBED)])
    p = FleetProvider()
    p._ensure_fleet()
    assert p._chat_slots == 1  # no chat launch -> default capacity
    assert p._chat_ctx is None


class _OrderedReapSwap(_FakeSwap):
    """A fake swap appending reap events to a shared order log."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def reap_stale(self) -> None:
        super().reap_stale()
        self._order.append("reap")


def _ordered_planner(order: list[str], launches: list) -> object:
    """A plan_all_launches stand-in appending to the shared order log."""

    def _plan() -> planning_mod.FleetPlan:
        order.append("plan")
        return planning_mod.FleetPlan(tuple(launches))

    return _plan


def test_ensure_fleet_reaps_stale_swaps_before_planning(monkeypatch) -> None:
    # An OOM-survivor llama-swap holds VRAM; reaping after planning would let
    # the device probe see artificially reduced free memory and misplace.
    order: list[str] = []
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _FakeSwap())
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: order.append("reap"))
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    monkeypatch.setattr(
        planning_mod, "plan_all_launches", _ordered_planner(order, [_fake_launch(WorkerRole.CHAT)])
    )
    FleetProvider()._ensure_fleet()
    # A placeable-set plan precedes the reap (it reads total VRAM, reap-independent);
    # the sizing plan that sees free VRAM still runs after the reap.
    assert order[-2:] == ["reap", "plan"]


def test_reload_pass_reaps_stale_swaps_before_planning(monkeypatch) -> None:
    order: list[str] = []
    swap = _FakeSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: swap)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: order.append("reap"))
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    monkeypatch.setattr(
        planning_mod, "plan_all_launches", _ordered_planner(order, [_fake_launch(WorkerRole.CHAT)])
    )
    p = FleetProvider()
    p._swaps = {SwapGroup.CHAT: swap}
    p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
    p._reload_pass()
    assert order == ["reap", "plan"]


def test_ensure_fleet_spawns_nothing_when_no_models(monkeypatch, tmp_path: Path) -> None:
    # No configured/installed model -> no launches -> no swap process at all
    # (matches the old supervisor, which spawned nothing for an empty launch set).
    started = {"swaps": 0}

    class _CountingSwap(_FakeSwap):
        def start(self, launches: list, **_kw: object) -> None:
            started["swaps"] += 1
            super().start(launches)

    _install_engine(monkeypatch, tmp_path, launches=[], swap=_CountingSwap())
    p = FleetProvider()
    assert p._ensure_fleet() is False
    assert started["swaps"] == 0  # never started
    assert p._swaps == {}
    assert p._clients == {}


def test_ensure_fleet_returns_none_when_engine_binary_unavailable(monkeypatch) -> None:
    """plan_all_launches raising NOT_FOUND (no engine binary) yields no swap."""
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr(prov_mod, "SwapManager", lambda _data_dir, _group: _FakeSwap())

    def _no_binary() -> list:
        raise ProviderError("llama-server binary not found.", kind=ProviderErrorKind.NOT_FOUND)

    monkeypatch.setattr(planning_mod, "plan_all_launches", _no_binary)
    p = FleetProvider()
    assert p._ensure_fleet() is False
    assert p._swaps == {}


def _captured_client_kwargs(monkeypatch, launch) -> dict:
    """Build the engine around *launch* and return the client constructor kwargs."""
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _FakeSwap())
    monkeypatch.setattr(
        planning_mod, "plan_all_launches", lambda: planning_mod.FleetPlan((launch,))
    )
    captured: list[dict] = []

    def _capture(_endpoint, _model, **kwargs):
        captured.append(kwargs)
        return _fake_client()

    monkeypatch.setattr(prov_mod, "LlamaServerClient", _capture)
    FleetProvider()._ensure_fleet()
    return captured[0]


def test_clients_get_token_cap_and_cold_load_timeout(monkeypatch) -> None:
    # Each role's client carries the launch token_cap (embed/rerank input truncation,
    # the in-process backstop) and a timeout long enough for a cold upstream load,
    # matching the old supervisor's client construction.
    launch = _fake_launch(WorkerRole.EMBED)
    launch.token_cap = 2048
    kwargs = _captured_client_kwargs(monkeypatch, launch)
    assert kwargs["token_cap"] == 2048
    assert kwargs["timeout"] == prov_mod._REQUEST_TIMEOUT_FLOOR_S


def test_small_model_client_keeps_the_floor_timeout(monkeypatch) -> None:
    launch = _fake_launch(WorkerRole.CHAT, weights_bytes=4 * _GB)
    kwargs = _captured_client_kwargs(monkeypatch, launch)
    assert kwargs["timeout"] == prov_mod._REQUEST_TIMEOUT_FLOOR_S


def test_giant_model_client_timeout_covers_its_cold_load(monkeypatch) -> None:
    # A split-GGUF giant loads longer than the fixed floor; the client request
    # timeout must ride out the cold load llama-swap itself is willing to wait
    # for, or the first chat marks the replica unhealthy mid-load.
    from lilbee.providers.fleet import swap_config as swap_config_mod

    launch = _fake_launch(WorkerRole.CHAT, weights_bytes=300 * _GB)
    kwargs = _captured_client_kwargs(monkeypatch, launch)
    health_timeout = swap_config_mod._health_check_timeout_s([launch])
    assert health_timeout > prov_mod._REQUEST_TIMEOUT_FLOOR_S
    assert kwargs["timeout"] >= health_timeout
    assert kwargs["timeout"] == health_timeout + prov_mod._REQUEST_TIMEOUT_GENERATION_MARGIN_S


def test_chat_starts_swap_on_first_use(monkeypatch) -> None:
    from lilbee.providers.base import ChatResult, FinishReason

    captured: dict[str, MagicMock] = {}

    def _make_client(_endpoint, model, **_kw):
        client = _fake_client()
        client.chat_result.return_value = ChatResult(
            text="ok", tool_calls=(), finish_reason=FinishReason.STOP
        )
        captured[model] = client
        return client

    swap = _FakeSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: swap)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", _make_client)
    monkeypatch.setattr(
        planning_mod,
        "plan_all_launches",
        lambda: planning_mod.FleetPlan((_fake_launch(WorkerRole.CHAT),)),
    )
    p = FleetProvider()
    assert p.chat([{"role": "user", "content": "hi"}]).text == "ok"
    assert len(swap.started) == 1  # routing the first chat started the swap
    assert swap.bound_lifetime is True  # bound by default so a crash cannot orphan it


def test_keep_engine_warm_starts_the_swap_unbound(monkeypatch, tmp_path: Path) -> None:
    """A warm engine must outlive lilbee, so it is spawned without the death binding."""
    monkeypatch.setattr(cfg, "keep_engine_warm", True)
    swap = _install_engine(monkeypatch, tmp_path, launches=[_fake_launch(WorkerRole.CHAT)])

    FleetProvider()._ensure_fleet()

    assert swap.started  # the fleet was built
    assert swap.bound_lifetime is False  # keep_engine_warm: not bound to this process


def test_concurrent_first_requests_start_swap_once(monkeypatch) -> None:
    starts = {"n": 0}

    class _SlowSwap(_FakeSwap):
        def start(self, launches: list, **_kw: object) -> None:
            starts["n"] += 1
            time.sleep(0.05)  # widen the race window
            super().start(launches)

    swap = _SlowSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: swap)

    def _make_client(_endpoint, _model, **_kw):
        client = _fake_client()
        client.chat_result.return_value = "ok"
        return client

    monkeypatch.setattr(prov_mod, "LlamaServerClient", _make_client)
    monkeypatch.setattr(
        planning_mod,
        "plan_all_launches",
        lambda: planning_mod.FleetPlan((_fake_launch(WorkerRole.CHAT),)),
    )
    p = FleetProvider()
    barrier = threading.Barrier(8)

    def _hit() -> None:
        # Bounded wait so a thread that dies under heavy CI load can't deadlock the
        # barrier and hang the test; the single-flight assertion holds regardless.
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=10.0)
        p.chat([{"role": "user", "content": "hi"}])

    threads = [threading.Thread(target=_hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
    assert starts["n"] == 1  # single-flight: 8 concurrent first-requests start one swap


def test_shutdown_drops_refs_and_closes_clients() -> None:
    # Engine processes are stopped through stop_engine (last-out), never by
    # signalling tracked swaps: shutdown's provider-side job is refs and clients.
    client = _fake_client()
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    p.shutdown()
    client.close.assert_called_once()
    assert p._swaps == {}
    assert p._shut_down is True


def test_invalidate_load_cache_drops_swap_refs() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p.invalidate_load_cache()
    assert p._swaps == {}


def test_invalidate_load_cache_leaves_provider_reusable(monkeypatch, tmp_path: Path) -> None:
    """A cache drop is not terminal: the next use rebuilds the swap."""
    swap = _install_engine(monkeypatch, tmp_path, launches=[_fake_launch(WorkerRole.CHAT)])
    p = FleetProvider()
    assert p._ensure_fleet() is True
    p.invalidate_load_cache()
    assert p._swaps == {}
    assert p._ensure_fleet() is True  # rebuilt with current cfg, not refused
    assert p._swaps.get(WorkerRole.CHAT) is swap


def test_drop_loaded_models_async_leaves_provider_reusable(monkeypatch, tmp_path: Path) -> None:
    """The off-thread drop used by settings changes must not latch shutdown.

    app.settings routes num_ctx/kv_cache_type changes here while retaining the
    provider; a latched flag would refuse every later chat/embed/rerank call
    until process restart.
    """
    swap = _install_engine(monkeypatch, tmp_path, launches=[_fake_launch(WorkerRole.CHAT)])
    p = FleetProvider()
    assert p._ensure_fleet() is True
    p.drop_loaded_models_async()
    assert _wait_until(lambda: p._swaps == {})
    assert p._ensure_fleet() is True  # rebuilt with current cfg, not refused
    assert p._swaps.get(WorkerRole.CHAT) is swap


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll *predicate* until true or *timeout*; generous so xdist load can't flake it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_warm_up_pool_starts_swap_off_thread(monkeypatch, tmp_path: Path) -> None:
    # The eager warm-up at TUI mount must not block the caller; it dispatches a
    # background start thread and returns immediately.
    started = threading.Event()
    release = threading.Event()

    class _SlowSwap(_FakeSwap):
        def start(self, launches: list, **_kw: object) -> None:
            started.set()
            release.wait(timeout=5.0)
            super().start(launches)

    swap = _install_engine(
        monkeypatch, tmp_path, launches=[_fake_launch(WorkerRole.CHAT)], swap=_SlowSwap()
    )
    p = FleetProvider()
    p.warm_up_pool()
    assert started.wait(timeout=5.0)  # start runs on a background thread
    assert p._swaps == {}  # warm_up_pool returned before start completed
    release.set()
    assert _wait_until(lambda: p._swaps.get(WorkerRole.CHAT) is swap)


def test_warm_up_pool_single_flight_does_not_double_start(monkeypatch, tmp_path: Path) -> None:
    starts = {"n": 0}
    in_start = threading.Event()
    release = threading.Event()

    class _GatedSwap(_FakeSwap):
        def start(self, launches: list, **_kw: object) -> None:
            starts["n"] += 1
            in_start.set()
            release.wait(timeout=5.0)
            super().start(launches)

    swap = _install_engine(
        monkeypatch, tmp_path, launches=[_fake_launch(WorkerRole.CHAT)], swap=_GatedSwap()
    )
    p = FleetProvider()
    p.warm_up_pool()
    assert in_start.wait(timeout=5.0)  # first start genuinely in flight
    p.warm_up_pool()  # second call while warming: must not start a second swap
    release.set()
    assert _wait_until(lambda: p._swaps.get(WorkerRole.CHAT) is swap)
    assert starts["n"] == 1


def test_warm_up_pool_noop_when_swap_already_up_and_ready(monkeypatch, tmp_path: Path) -> None:
    # A fully-loaded fleet (swap up AND its role ready) short-circuits: no swap
    # restart and no re-warm thread. The cold-role counterpart, where the swap is
    # up but the model idle-unloaded, is covered by the rewarm test above.
    starts = {"n": 0}
    swap = _install_engine(monkeypatch, tmp_path, launches=[])
    monkeypatch.setattr(swap, "start", lambda launches: starts.__setitem__("n", starts["n"] + 1))
    ready_swap = _FakeSwap()
    ready_swap.ready = {WorkerRole.CHAT}
    p = FleetProvider()
    p._swaps = {SwapGroup.CHAT: ready_swap}  # up and loaded
    p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
    kicked = _stub_warm_thread(monkeypatch)
    p.warm_up_pool()
    assert starts["n"] == 0  # no start dispatched
    assert not kicked  # no re-warm thread dispatched
    assert p._warming is False


def test_warm_up_blocking_logs_and_clears_guard_on_failure(monkeypatch, caplog) -> None:
    def _boom() -> list:
        raise RuntimeError("plan failed")

    monkeypatch.setattr(planning_mod, "plan_all_launches", _boom)
    p = FleetProvider()
    with caplog.at_level("WARNING", logger="lilbee.providers.fleet.provider"):
        p._warm_up_blocking()  # runs the body synchronously for the assertion
    assert p._swaps == {}
    assert p._warming is False  # guard cleared so a later warm-up can retry
    assert "warm-up failed" in caplog.text.lower()
    # The handled failure must not carry a traceback: a WARNING with exc_info
    # reads like a crash for a condition the next real call recovers from.
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings and all(r.exc_info is None for r in warnings)


def test_cancel_inference_severs_chat_and_retiring_streams() -> None:
    """The cancel reaches every chat replica, including a client a reload
    already retired: the swap's settings write can retire the busy client
    before the cancel lands."""
    active, retired, embed = _fake_client(), _fake_client(), _fake_client()
    p = _provider_with_clients({WorkerRole.CHAT: [active], WorkerRole.EMBED: [embed]})
    p._retiring_clients = [retired]
    p.cancel_inference()
    active.abort_streams.assert_called_once_with()
    retired.abort_streams.assert_called_once_with()
    embed.abort_streams.assert_not_called()


def test_warm_up_blocking_stamps_error_with_the_real_reason(monkeypatch) -> None:
    """A warm that dies before the chat warm begins still surfaces its reason.

    The prompt path reads this via chat_warm_error(); without the stamp the
    user got a generic "not ready" bounce with no explanation.
    """
    from lilbee.providers.warm_progress import WarmPhase

    def _boom() -> list:
        raise RuntimeError("engine exited before it was ready")

    monkeypatch.setattr(planning_mod, "plan_all_launches", _boom)
    p = FleetProvider()
    p._warm_up_blocking()
    snap = p.warm_progress()
    assert snap is not None
    assert snap.phase is WarmPhase.ERROR
    assert "engine exited before it was ready" in (snap.error or "")


def test_warm_up_blocking_surfaces_a_probe_failure(monkeypatch) -> None:
    """A planning ProviderError (e.g. a wedged GPU probe) must not be swallowed.

    The bb-0yf0 silent hang: the old catch logged every planning ProviderError at
    DEBUG as "binary unavailable" and cleared the warm tracker, so the fleet sat
    never-ready with no error anywhere. The failure now reaches the tracker (and
    with it health's chat_status/chat_error and the TUI's warm line).
    """
    from lilbee.providers.base import ProviderError, ProviderErrorKind
    from lilbee.providers.warm_progress import WarmPhase

    def _wedged() -> list:
        raise ProviderError(
            "The GPU device probe (llama-server --list-devices) did not respond within 60s",
            kind=ProviderErrorKind.SERVER,
        )

    monkeypatch.setattr(planning_mod, "plan_all_launches", _wedged)
    p = FleetProvider()
    p._warm_up_blocking()
    snap = p.warm_progress()
    assert snap is not None
    assert snap.phase is WarmPhase.ERROR
    assert "GPU device probe" in (snap.error or "")
    assert p._warming is False  # guard cleared so a later warm-up can retry


def test_ensure_fleet_propagates_a_probe_failure_to_on_demand_callers(monkeypatch) -> None:
    # A chat/embed call that triggers the build must see the real reason (the
    # route layer maps it to a 503), not a generic "no server running".
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr(prov_mod, "reap_stale", lambda _data_dir, **_kw: None)
    # The placeable-set query plans too; give it a fleet so the build proceeds to
    # the wedged clean-box snapshot, which is what must surface.
    monkeypatch.setattr(
        planning_mod,
        "plan_all_launches",
        lambda: planning_mod.FleetPlan((_fake_launch(WorkerRole.CHAT),)),
    )

    def _wedged() -> None:
        raise ProviderError("The GPU device probe did not respond", kind=ProviderErrorKind.SERVER)

    monkeypatch.setattr(planning_mod, "capture_plan_probe", _wedged)
    p = FleetProvider()
    with pytest.raises(ProviderError, match="GPU device probe"):
        p._ensure_fleet()


def test_ensure_fleet_stays_quiet_when_the_binary_is_missing(monkeypatch) -> None:
    # A host without the engine binary legitimately serves nothing; only the
    # NOT_FOUND binary resolution keeps the quiet no-fleet path.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr(prov_mod, "reap_stale", lambda _data_dir, **_kw: None)

    def _no_binary() -> None:
        raise ProviderError("llama-server binary not found.", kind=ProviderErrorKind.NOT_FOUND)

    # A missing binary surfaces first when the placeable set is planned; nothing
    # is placeable, so the ladder serves nothing without ever reaching a build.
    monkeypatch.setattr(planning_mod, "plan_all_launches", _no_binary)
    monkeypatch.setattr(planning_mod, "capture_plan_probe", _no_binary)
    p = FleetProvider()
    assert p._ensure_fleet() is False
    assert p._swaps == {}


def test_plan_and_spawn_serves_nothing_if_the_binary_vanishes_before_the_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    # The placeable-set plan and _can_build_engine both confirmed the binary, so
    # this NOT_FOUND guard fires only if it is removed before the clean-box
    # snapshot; it must serve nothing, not raise.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    def _no_binary() -> None:
        raise ProviderError("binary vanished", kind=ProviderErrorKind.NOT_FOUND)

    monkeypatch.setattr(planning_mod, "capture_plan_probe", _no_binary)

    assert FleetProvider()._plan_and_spawn(tmp_path) is False


def test_warm_up_blocking_reports_starting_before_fleet_spawn(monkeypatch) -> None:
    """The tracker stamps STARTING before the spawn so the task bar shows life
    during the whole spawn/health window instead of a dead gap."""
    from lilbee.providers.warm_progress import WarmPhase

    p = FleetProvider()
    seen: list[WarmPhase] = []

    def _observe_fleet() -> None:
        snap = p.warm_progress()
        assert snap is not None
        seen.append(snap.phase)

    monkeypatch.setattr(p, "_ensure_fleet", _observe_fleet)
    monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
    p._warm_up_blocking()
    assert seen == [WarmPhase.STARTING]


def test_warm_up_blocking_clears_stamp_when_chat_absent_for_other_reasons(monkeypatch) -> None:
    """No chat instance placed for a non-install reason (a remote-routed chat has
    no local server to warm): the early STARTING stamp is dropped so the warm line
    cannot spin forever, and no spurious failure is stamped."""
    p = FleetProvider()
    monkeypatch.setattr(p, "_ensure_fleet", lambda: None)
    monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
    p._warm_up_blocking()
    assert p.warm_progress() is None


def test_warm_up_blocking_fails_when_chat_model_not_installed(monkeypatch) -> None:
    """Chat model configured but not installed: the warm ends in a terminal ERROR
    naming the cause, so the prompt path renders 'failed to load' instead of an
    endless 'not ready, send again' bounce that retrying can never resolve."""
    from lilbee.providers.warm_progress import WarmPhase

    p = FleetProvider()

    def _plan_skips_chat() -> None:
        # _ensure_fleet records what the plan left unplaced; a not-installed chat
        # model is skipped here, as _plan_and_spawn would record it.
        p._skipped_not_installed = {WorkerRole.CHAT: "Qwen/Qwen3-8B-GGUF"}

    monkeypatch.setattr(p, "_ensure_fleet", _plan_skips_chat)
    monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
    p._warm_up_blocking()
    snap = p.warm_progress()
    assert snap is not None
    assert snap.phase is WarmPhase.ERROR
    assert (snap.error or "") == "chat model Qwen3 8B is not installed"


def test_warm_up_blocking_swallows_interpreter_shutdown_race(monkeypatch, caplog) -> None:
    # A fast CLI exit tears down the interpreter mid-warm; the pool submit then
    # raises RuntimeError. During finalization this must be dropped quietly, not
    # logged as a scary WARNING traceback.
    def _shutdown_race() -> list:
        raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    monkeypatch.setattr(planning_mod, "plan_all_launches", _shutdown_race)
    monkeypatch.setattr("lilbee.providers.fleet.provider.sys.is_finalizing", lambda: True)
    p = FleetProvider()
    with caplog.at_level("WARNING", logger="lilbee.providers.fleet.provider"):
        p._warm_up_blocking()
    assert p._warming is False
    assert "warm-up failed" not in caplog.text.lower()  # no WARNING on shutdown


def test_preload_roles_warms_each_role_and_fires_listeners(monkeypatch) -> None:
    chat, embed = _fake_client(), _fake_client()
    embed.embed.return_value = [[0.1]]
    p = _provider_with_clients({WorkerRole.CHAT: [chat], WorkerRole.EMBED: [embed]})
    # Keep the unit hermetic: the chat warm path's shard prewarm reads cfg.chat_model
    # off disk, which this test isn't exercising.
    monkeypatch.setattr(p, "_prewarm_chat_weights", lambda: None)
    spawning: list[WorkerRole] = []
    spawned: list[WorkerRole] = []
    p.add_spawn_listener(on_spawning=spawning.append, on_spawned=spawned.append)
    p._preload_roles()
    chat.chat.assert_called_once()  # chat warmed with a minimal completion
    embed.embed.assert_called_once()  # embed warmed
    assert set(spawning) == {WorkerRole.CHAT, WorkerRole.EMBED}
    assert set(spawned) == {WorkerRole.CHAT, WorkerRole.EMBED}


def test_preload_roles_skips_failing_role(monkeypatch) -> None:
    embed = _fake_client()
    embed.embed.side_effect = RuntimeError("not loaded")
    p = _provider_with_clients({WorkerRole.EMBED: [embed]})
    p._preload_roles()  # a per-role warm failure must not raise
    embed.embed.assert_called_once()


def test_preload_roles_warms_roles_concurrently(monkeypatch) -> None:
    # The light roles must not queue behind the chat load: with the chat warm
    # blocked mid-flight, the embed warm still completes.
    chat_started = threading.Event()
    release_chat = threading.Event()
    embed_warmed = threading.Event()

    chat, embed = _fake_client(), _fake_client()

    def _blocked_chat(*args, **kwargs) -> MagicMock:
        chat_started.set()
        release_chat.wait(timeout=5.0)
        return MagicMock()

    chat.chat.side_effect = _blocked_chat
    embed.embed.side_effect = lambda *a, **k: (embed_warmed.set(), [[0.1]])[1]
    p = _provider_with_clients({WorkerRole.CHAT: [chat], WorkerRole.EMBED: [embed]})
    monkeypatch.setattr(p, "_prewarm_chat_weights", lambda: None)
    preload = threading.Thread(target=p._preload_roles, daemon=True)
    preload.start()
    assert chat_started.wait(timeout=5.0)
    assert embed_warmed.wait(timeout=5.0)  # embed warmed while chat still loading
    release_chat.set()
    preload.join(timeout=5.0)
    assert not preload.is_alive()


def _pinned_launch(role: WorkerRole, devices: str, replica: int = 0) -> InstanceLaunch:
    return InstanceLaunch(
        role=role,
        argv=[],
        env_overrides={"CUDA_VISIBLE_DEVICES": devices},
        model=f"{role.value}-{replica}",
        replica=replica,
    )


def test_warm_chains_serializes_shared_device_chat_last() -> None:
    sets = {
        WorkerRole.CHAT: frozenset({"CUDA_VISIBLE_DEVICES=0"}),
        WorkerRole.EMBED: frozenset({"CUDA_VISIBLE_DEVICES=0"}),
    }
    chains = prov_mod._warm_chains([WorkerRole.CHAT, WorkerRole.EMBED], sets)
    assert chains == [[WorkerRole.EMBED, WorkerRole.CHAT]]


def test_warm_chains_keeps_disjoint_devices_parallel() -> None:
    sets = {
        WorkerRole.CHAT: frozenset({"CUDA_VISIBLE_DEVICES=0"}),
        WorkerRole.EMBED: frozenset({"CUDA_VISIBLE_DEVICES=1"}),
    }
    chains = prov_mod._warm_chains([WorkerRole.CHAT, WorkerRole.EMBED], sets)
    assert {tuple(c) for c in chains} == {(WorkerRole.EMBED,), (WorkerRole.CHAT,)}


def test_warm_chains_unpinned_role_stays_parallel() -> None:
    # No pinning info (Metal, or a launch without visibility env) keeps today's
    # concurrent warm; only proven device sharing serializes.
    sets = {WorkerRole.CHAT: frozenset({"CUDA_VISIBLE_DEVICES=0"})}
    chains = prov_mod._warm_chains([WorkerRole.CHAT, WorkerRole.EMBED], sets)
    assert {tuple(c) for c in chains} == {(WorkerRole.EMBED,), (WorkerRole.CHAT,)}


def test_warm_chains_transitive_overlap_merges() -> None:
    sets = {
        WorkerRole.CHAT: frozenset({"CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=1"}),
        WorkerRole.EMBED: frozenset({"CUDA_VISIBLE_DEVICES=1", "CUDA_VISIBLE_DEVICES=2"}),
        WorkerRole.RERANK: frozenset({"CUDA_VISIBLE_DEVICES=2"}),
    }
    chains = prov_mod._warm_chains([WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK], sets)
    assert chains == [[WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.CHAT]]


def test_role_device_sets_unions_replicas_and_skips_unpinned() -> None:
    launches = [
        _pinned_launch(WorkerRole.EMBED, "0", replica=0),
        _pinned_launch(WorkerRole.EMBED, "1", replica=1),
        InstanceLaunch(role=WorkerRole.VISION, argv=[], env_overrides={}, model="vision-0"),
    ]
    sets = prov_mod._role_device_sets(launches)
    assert sets == {
        WorkerRole.EMBED: frozenset({"CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=1"})
    }


def test_preload_serializes_roles_sharing_a_device(monkeypatch) -> None:
    # Chat and embed pinned to the same card must warm one at a time, embed
    # first: concurrent loads on a shared device race each other for VRAM and
    # the loser OOMs its first attempt.
    order: list[str] = []
    chat, embed = _fake_client(), _fake_client()
    chat.chat.side_effect = lambda *a, **k: (order.append("chat"), MagicMock())[1]
    embed.embed.side_effect = lambda *a, **k: (order.append("embed"), [[0.1]])[1]
    p = _provider_with_clients({WorkerRole.CHAT: [chat], WorkerRole.EMBED: [embed]})
    p._launches = {
        SwapGroup.CHAT: (_pinned_launch(WorkerRole.CHAT, "0"),),
        SwapGroup.EMBED: (_pinned_launch(WorkerRole.EMBED, "0"),),
    }
    monkeypatch.setattr(p, "_prewarm_chat_weights", lambda: None)
    p._preload_roles()
    assert order == ["embed", "chat"]


def test_preload_chain_gives_every_role_its_warm_attempt(monkeypatch) -> None:
    # An unexpected error warming one chain role (a listener blowing up) must
    # not rob the roles behind it of their warm; it surfaces after the chain.
    chat, embed = _fake_client(), _fake_client()
    embed.embed.return_value = [[0.1]]
    p = _provider_with_clients({WorkerRole.CHAT: [chat], WorkerRole.EMBED: [embed]})
    p._launches = {
        SwapGroup.CHAT: (_pinned_launch(WorkerRole.CHAT, "0"),),
        SwapGroup.EMBED: (_pinned_launch(WorkerRole.EMBED, "0"),),
    }
    monkeypatch.setattr(p, "_prewarm_chat_weights", lambda: None)

    def _blowing_listener(role: WorkerRole) -> None:
        if role is WorkerRole.EMBED:
            raise RuntimeError("listener broke")

    p.add_spawn_listener(on_spawning=_blowing_listener)
    with pytest.raises(RuntimeError, match="listener broke"):
        p._preload_roles()
    chat.chat.assert_called_once()  # chat still warmed behind the failed embed


def _registry_with_shards(monkeypatch, shards: list[Path]) -> None:
    registry = MagicMock()
    registry.shard_paths.return_value = shards
    monkeypatch.setattr(prov_mod, "ModelRegistry", MagicMock(return_value=registry))


def test_prewarm_skips_shards_already_paged_this_boot(monkeypatch, tmp_path) -> None:
    shard = tmp_path / "model.gguf"
    shard.write_bytes(b"x" * 64)
    _registry_with_shards(monkeypatch, [shard])
    monkeypatch.setattr(prov_mod, "_PREWARMED_SHARDS", set())
    p = _provider_with_clients({})
    p._prewarm_chat_weights()  # first pass reads and records the shard

    opens = {"n": 0}
    real_open = Path.open

    def _counting_open(self, *args, **kwargs):
        if self == shard:
            opens["n"] += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _counting_open)
    p._prewarm_chat_weights()  # second pass: cache is hot, no re-read
    assert opens["n"] == 0


def test_prewarm_rereads_when_a_shard_changed(monkeypatch, tmp_path) -> None:
    shard = tmp_path / "model.gguf"
    shard.write_bytes(b"x" * 64)
    _registry_with_shards(monkeypatch, [shard])
    monkeypatch.setattr(prov_mod, "_PREWARMED_SHARDS", set())
    p = _provider_with_clients({})
    p._prewarm_chat_weights()
    shard.write_bytes(b"y" * 128)  # new size + mtime -> new prewarm identity

    opens = {"n": 0}
    real_open = Path.open

    def _counting_open(self, *args, **kwargs):
        if self == shard:
            opens["n"] += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _counting_open)
    p._prewarm_chat_weights()
    assert opens["n"] == 1  # changed shard is read again


def test_warm_role_dispatches_per_role() -> None:
    chat, embed, rerank, vision = (_fake_client() for _ in range(4))
    prov_mod._warm_role(WorkerRole.CHAT, chat)
    prov_mod._warm_role(WorkerRole.EMBED, embed)
    prov_mod._warm_role(WorkerRole.RERANK, rerank)
    prov_mod._warm_role(WorkerRole.VISION, vision)  # vision warms lazily -> no call
    chat.chat.assert_called_once()
    embed.embed.assert_called_once_with([prov_mod._WARM_PROMPT])
    rerank.rerank.assert_called_once()
    vision.chat.assert_not_called()
    vision.embed.assert_not_called()


def test_role_ready_false_without_swap() -> None:
    assert FleetProvider().role_ready(WorkerRole.CHAT) is False


def test_role_ready_reflects_swap_running_state() -> None:
    p = FleetProvider()
    swap = _FakeSwap()
    swap.ready = {WorkerRole.CHAT}
    p._swaps = {SwapGroup.CHAT: swap}
    p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
    assert p.role_ready(WorkerRole.CHAT) is True
    assert p.role_ready(WorkerRole.EMBED) is False


def test_drop_loaded_models_async_tears_down_off_thread() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p.drop_loaded_models_async()
    # The off-thread worker drops the refs; the engine restart itself goes
    # through stop_engine on the held dirs (none in this fixture).
    assert _wait_until(lambda: p._swaps == {})


def test_drop_loaded_models_async_noop_without_swap() -> None:
    p = FleetProvider()  # _swap is None
    p.drop_loaded_models_async()  # must not raise or spawn a thread
    assert p._swaps == {}


def test_apply_fleet_gpu_env_honors_gpu_devices_pin(monkeypatch) -> None:
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", "0")
    # apply_fleet_gpu_env writes these vars in place; monkeypatch.delenv does not
    # track app-side additions, so restore the pre-apply snapshot to avoid leaking
    # the pin (CUDA_VISIBLE_DEVICES=0) into later tests that read os.environ.
    snapshot = {name: os.environ.get(name) for name in _GPU_VISIBLE_ENV_VARS}
    try:
        gpu_env.apply_fleet_gpu_env()
        # Every backend except the AMD pair, which is written to one var only:
        # ROCr filters before HIP re-indexes within the survivors.
        for name in _GPU_VISIBLE_ENV_VARS:
            if name == "ROCR_VISIBLE_DEVICES":
                assert name not in os.environ
                continue
            assert os.environ[name] == "0"
    finally:
        for name, value in snapshot.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_apply_fleet_gpu_env_clears_empty_cuda_visible_devices(monkeypatch) -> None:
    # SkyPilot / Docker GPU setups expose the GPU via NVIDIA_VISIBLE_DEVICES but export an
    # empty CUDA_VISIBLE_DEVICES, which hides the GPU from CUDA ("no CUDA-capable device").
    # The runtime's marker is what makes the empty value a contradiction rather than an
    # instruction, so the test has to set it: without it, an empty mask means CPU.
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", None)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    gpu_env.apply_fleet_gpu_env()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_apply_fleet_gpu_env_keeps_another_vendors_empty_visible_devices(monkeypatch) -> None:
    # The NVIDIA container runtime's marker speaks for NVIDIA only. An empty Vulkan
    # or AMD mask is somebody fencing that backend off, and clearing it on the
    # strength of an unrelated vendor's marker overrides a deliberate choice.
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", None)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "")
    gpu_env.apply_fleet_gpu_env()
    assert os.environ["GGML_VK_VISIBLE_DEVICES"] == ""


def test_apply_fleet_gpu_env_keeps_nonempty_cuda_visible_devices(monkeypatch) -> None:
    # An explicit index pin (and the conventional "-1" CPU opt-out) is user intent, not an
    # orchestration artifact: leave it alone.
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", None)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    gpu_env.apply_fleet_gpu_env()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_apply_fleet_gpu_env_pin_replaces_empty_cuda_visible_devices(monkeypatch) -> None:
    # Clearing the empty var before the pin runs is what lets a cfg.gpu_devices pin take
    # effect: setdefault would otherwise treat the present-but-empty value as already set.
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", "0")
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    gpu_env.apply_fleet_gpu_env()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_apply_fleet_gpu_env_keeps_empty_cuda_visible_devices_without_gpu(monkeypatch) -> None:
    # With no GPU present, an empty CUDA_VISIBLE_DEVICES is honored as genuine CPU intent.
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", None)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    gpu_env.apply_fleet_gpu_env()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_routing_provider_local_engine_is_fleet() -> None:
    # The fleet is the sole local engine now: AUTO routes local refs through
    # RoutingProvider, whose local engine is a FleetProvider (a single machine
    # is a fleet-of-one). Construction is side-effect-free; no swap is started.
    from lilbee.providers.routing_provider import RoutingProvider

    assert isinstance(RoutingProvider()._get_local(), FleetProvider)


def test_get_capabilities_unresolved_model_returns_completion(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    def _raise(_m: str):
        raise ProviderError("not found")

    monkeypatch.setattr("lilbee.catalog.is_rerank_ref", lambda _m: False)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _raise)
    assert FleetProvider().get_capabilities("missing/model.gguf") == ["completion"]


class TestLifecycleMethods:
    def test_cancel_inference_is_noop(self) -> None:
        # llama-server stops on client disconnect; cancel has nothing to flip.
        assert _provider_with_clients({}).cancel_inference() is None

    def test_reload_role_noop_when_swap_not_up(self, monkeypatch) -> None:
        spawned = {"thread": False}
        monkeypatch.setattr("threading.Thread", lambda *a, **k: spawned.__setitem__("thread", True))
        p = FleetProvider()  # _swap is None
        p.reload_role(WorkerRole.EMBED)
        assert spawned["thread"] is False  # no background restart dispatched

    def test_reload_role_dispatches_background_restart(self) -> None:
        done = threading.Event()
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}  # non-None so reload dispatches
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._reload_blocking = lambda: done.set()  # type: ignore[method-assign]
        p.reload_role(WorkerRole.EMBED)
        assert done.wait(timeout=2.0)  # the spawned thread ran the blocking restart

    def test_reload_blocking_restarts_swap_and_readopts(self, monkeypatch) -> None:
        launches = [_fake_launch(WorkerRole.CHAT, slots=2, ctx=4096)]
        monkeypatch.setattr(
            planning_mod, "plan_all_launches", lambda: planning_mod.FleetPlan(tuple(launches))
        )
        monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
        fresh = _FakeSwap()
        monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: fresh)
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
        p = FleetProvider()
        stale = _FakeSwap()
        p._swaps = {SwapGroup.CHAT: stale}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._reload_blocking()
        # The chat launches changed (old running set unknown -> differs), so the
        # old group was stopped and a fresh one started and adopted.
        assert stale.shutdowns == 1
        assert len(fresh.started) == 1
        assert p._swaps[WorkerRole.CHAT] is fresh
        assert p._chat_slots == 2  # capacity re-adopted from the new launch set
        assert set(p._clients) == {WorkerRole.CHAT}

    def test_reload_blocking_noop_when_swap_cleared(self, monkeypatch) -> None:
        monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: planning_mod.FleetPlan(()))
        p = FleetProvider()  # _swap stays None
        p._reload_blocking()  # must not raise

    def test_reload_role_wait_runs_synchronously(self, monkeypatch) -> None:
        spawned = {"thread": False}
        monkeypatch.setattr("threading.Thread", lambda *a, **k: spawned.__setitem__("thread", True))
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        ran = {"blocking": False}
        p._reload_blocking = lambda: ran.__setitem__("blocking", True)  # type: ignore[method-assign]
        p.reload_role(WorkerRole.CHAT, wait=True)
        assert spawned["thread"] is False  # no background thread; ran in the caller's
        assert ran["blocking"] is True  # reload ran synchronously before returning

    def test_reload_role_wait_propagates_failure(self) -> None:
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}

        def boom() -> None:
            raise RuntimeError("reload failed")

        p._reload_blocking = boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="reload failed"):
            p.reload_role(WorkerRole.CHAT, wait=True)

    def test_reload_role_wait_blocks_until_in_flight_done(self) -> None:
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        with p._lock:
            p._reloading = True  # simulate a reload already in flight
        returned = threading.Event()

        def waiter() -> None:
            p.reload_role(WorkerRole.CHAT, wait=True)  # sets pending, waits on the cond
            returned.set()

        t = threading.Thread(target=waiter)
        t.start()
        assert not returned.wait(timeout=0.3)  # blocked while the in-flight reload runs
        assert p._reload_pending is True  # the waiter queued its pass
        with p._lock:  # the in-flight reload finishes
            p._reloading = False
            p._reload_done.notify_all()
        assert returned.wait(timeout=2.0)  # the waiter woke and returned
        t.join(timeout=2.0)

    def test_add_spawn_listener_stores_callbacks(self) -> None:
        p = FleetProvider()

        def on_spawning(_r: WorkerRole) -> None: ...

        def on_spawned(_r: WorkerRole) -> None: ...

        p.add_spawn_listener(on_spawning=on_spawning, on_spawned=on_spawned)
        assert p._on_spawning is on_spawning
        assert p._on_spawned is on_spawned


class TestChatWithTools:
    def test_routes_to_chat_server(self) -> None:
        from lilbee.providers.base import ChatToolResult, ToolCall

        client = _fake_client(0)
        client.chat_tools.return_value = ChatToolResult(
            content="", tool_calls=[ToolCall("c1", "f", "{}")]
        )
        p = _provider_with_clients({WorkerRole.CHAT: [client]})
        result = p.chat_with_tools(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice="auto",
        )
        assert result.tool_calls[0].name == "f"
        client.chat_tools.assert_called_once()
        assert client.chat_tools.call_args.kwargs["tool_choice"] == "auto"

    def test_without_server_raises(self) -> None:
        from lilbee.providers.base import ProviderError

        p = _provider_with_clients({})
        with pytest.raises(ProviderError, match="No chat model server is running"):
            p.chat_with_tools([{"role": "user", "content": "hi"}], tools=[])

    def test_model_override_raises(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        monkeypatch.setattr(cfg, "chat_model", "org/repo/configured.gguf")
        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        with pytest.raises(ProviderError, match="serves the configured chat model"):
            p.chat_with_tools([], tools=[], model="org/repo/other.gguf")


class TestChatCapacityAndCtxGetters:
    """max_concurrent_chats / served_chat_ctx read the chat launch once the swap is up."""

    def test_max_concurrent_chats_defaults_to_one_before_swap(self) -> None:
        assert FleetProvider().max_concurrent_chats() == 1

    def test_max_concurrent_chats_reads_chat_slots_when_up(self) -> None:
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._chat_slots = 4
        assert p.max_concurrent_chats() == 4

    def test_served_chat_ctx_is_none_before_swap(self) -> None:
        assert FleetProvider().served_chat_ctx() is None

    def test_served_chat_ctx_reads_chat_ctx_when_up(self) -> None:
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._chat_ctx = 32768
        assert p.served_chat_ctx() == 32768


class _FakeReplica:
    """A minimal client double with real health/in-flight state for routing tests."""

    def __init__(self, *, in_flight: int = 0, fail: Exception | None = None) -> None:
        self.in_flight = in_flight
        self.healthy = True
        self.fail = fail
        self.calls = 0

    def mark_unhealthy(self) -> None:
        self.healthy = False

    def mark_healthy(self) -> None:
        self.healthy = True

    def reserve(self) -> None:
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight -= 1

    def close(self) -> None:
        """Rediscovery retires the pool; a real client's close releases httpx."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return [[0.1]] * len(texts)

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return [0.5] * len(candidates)

    def chat(
        self, messages: object, options: object = None, stream: bool = False, **_kw: object
    ) -> str:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return "ocr text"


class TestReserveSpreadsLoad:
    def test_reserve_at_selection_spreads_concurrent_picks(self) -> None:
        # Every picker reserves before the next selects, so four idle replicas
        # each get exactly one request instead of all landing on the first
        # (the thundering herd that left cards idle on the 8x A100 fleet).
        replicas = [_FakeReplica() for _ in range(4)]
        picked = [prov_mod._reserve_least_in_flight(replicas) for _ in range(4)]
        assert {id(c) for c in picked} == {id(c) for c in replicas}
        assert all(r.in_flight == 1 for r in replicas)

    def test_call_with_failover_releases_the_reservation(self) -> None:
        replicas = [_FakeReplica(), _FakeReplica()]
        prov_mod._call_with_failover(replicas, lambda c: c.embed(["x"]))
        assert all(r.in_flight == 0 for r in replicas)  # reservation released

    def test_failover_releases_both_reservations(self) -> None:
        import httpx

        bad = _FakeReplica(fail=httpx.ConnectError("refused"))
        good = _FakeReplica()
        prov_mod._call_with_failover([bad, good], lambda c: c.embed(["x"]))
        assert bad.in_flight == 0 and good.in_flight == 0  # both released
        assert good.calls == 1 and not bad.healthy  # retried onto the healthy one

    def test_concurrent_dispatch_does_not_pile_on_one_replica(self) -> None:
        import threading

        replicas = [_FakeReplica() for _ in range(4)]
        # Each request holds its reservation at the barrier until all 8 have been
        # reserved, so every reservation is live while the others are selecting.
        # With the atomic reserve the 8 requests spread evenly (2 per replica);
        # without it the herd piled onto whichever replica read idlest first.
        overlap = threading.Barrier(8)

        def _dispatch() -> None:
            prov_mod._call_with_failover(replicas, lambda c: (overlap.wait(), c.embed(["x"]))[1])

        threads = [threading.Thread(target=_dispatch) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r.calls == 2 for r in replicas)  # perfectly spread, no herd
        assert all(r.in_flight == 0 for r in replicas)  # all released


class TestReplicaHealthRouting:
    def test_least_in_flight_skips_unhealthy_clients(self) -> None:
        dead, busy_but_alive = _FakeReplica(in_flight=0), _FakeReplica(in_flight=9)
        dead.mark_unhealthy()
        assert prov_mod._least_in_flight([dead, busy_but_alive]) is busy_but_alive

    def test_least_in_flight_falls_back_when_all_unhealthy(self) -> None:
        only = _FakeReplica()
        only.mark_unhealthy()
        assert prov_mod._least_in_flight([only]) is only

    def test_embed_fails_over_once_to_a_healthy_replica(self) -> None:
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        alive = _FakeReplica(in_flight=5)  # busier, picked only after failover
        p = _provider_with_clients({WorkerRole.EMBED: [dead, alive]})
        assert p.embed(["a", "b"]) == [[0.1], [0.1]]
        assert dead.healthy is False  # marked out of the pool
        assert alive.calls == 1

    def test_dead_replica_stops_receiving_traffic_after_failover(self) -> None:
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        alive = _FakeReplica(in_flight=5)
        p = _provider_with_clients({WorkerRole.EMBED: [dead, alive]})
        p.embed(["a"])
        dead_calls_after_failover = dead.calls
        p.embed(["b"])  # routed straight to the healthy replica now
        assert dead.calls == dead_calls_after_failover

    def test_all_dead_surfaces_provider_error(self, monkeypatch, tmp_path) -> None:
        # All replicas dead now triggers one engine rediscovery; when the
        # rebuild serves nothing (as here), that emptiness is what surfaces.
        import httpx as _httpx

        from lilbee.providers.base import ProviderError

        _install_ladder(monkeypatch, tmp_path, launches=[])
        only = _FakeReplica(fail=_httpx.ConnectError("refused"))
        p = _provider_with_clients({WorkerRole.EMBED: [only]})
        with pytest.raises(ProviderError, match="No embed model server"):
            p.embed(["a"])

    def test_vision_ocr_fails_over_to_healthy_replica(self, monkeypatch) -> None:
        """Vision OCR uses the failover path like embed/rerank: a dead replica is
        marked unhealthy and the call retries on a live one, instead of the dead
        replica being re-picked and the error swallowed to empty text (bb-7jg1.5)."""
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        alive = _FakeReplica(in_flight=5)  # busier, picked only after failover
        monkeypatch.setattr(cfg, "vision_model", "org/vis/model.gguf")
        p = _provider_with_clients({WorkerRole.VISION: [dead, alive]})
        assert p.vision_ocr(b"\x89PNG", "") == "ocr text"
        assert dead.healthy is False
        assert alive.calls == 1

    def test_successful_call_restores_an_unhealthy_replica(self) -> None:
        only = _FakeReplica()
        only.mark_unhealthy()
        p = _provider_with_clients({WorkerRole.EMBED: [only]})
        assert p.embed(["a"]) == [[0.1]]
        assert only.healthy is True

    def test_model_level_errors_propagate_without_failover(self) -> None:
        from lilbee.providers.base import ProviderError

        broken = _FakeReplica(fail=ProviderError("bad input", provider="llama-server"))
        sibling = _FakeReplica(in_flight=5)
        p = _provider_with_clients({WorkerRole.EMBED: [broken, sibling]})
        with pytest.raises(ProviderError, match="bad input"):
            p.embed(["a"])
        assert broken.healthy is True  # not a connection failure, stays in the pool
        assert sibling.calls == 0

    def test_rerank_fails_over_to_a_healthy_replica(self) -> None:
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        alive = _FakeReplica(in_flight=5)
        p = _provider_with_clients({WorkerRole.RERANK: [dead, alive]})
        assert p.rerank("q", ["c"]) == [0.5]
        assert alive.calls == 1

    def test_failover_probes_a_cooling_replica_when_none_healthy(self) -> None:
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        cooling = _FakeReplica(in_flight=5)
        cooling.mark_unhealthy()  # the only sibling is mid cool-down
        p = _provider_with_clients({WorkerRole.EMBED: [dead, cooling]})
        assert p.embed(["a"]) == [[0.1]]  # probed instead of "no healthy replica"
        assert cooling.calls == 1
        assert cooling.healthy is True  # the successful probe restored it

    def test_retry_connection_failure_marks_the_second_replica_unhealthy(self) -> None:
        import httpx as _httpx

        from lilbee.providers.base import ProviderError

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        also_dead = _FakeReplica(in_flight=5, fail=_httpx.ConnectError("refused"))
        p = _provider_with_clients({WorkerRole.EMBED: [dead, also_dead]})
        # An exhausted pool triggers one engine rediscovery; with no engine to
        # rebuild here, that surfaces the no-server error, not the transport one.
        with pytest.raises(ProviderError, match="No embed model server"):
            p.embed(["a"])
        assert dead.healthy is False
        assert also_dead.healthy is False  # the retry target is taken out too

    def test_retry_model_error_propagates_without_marking(self) -> None:
        import httpx as _httpx

        from lilbee.providers.base import ProviderError

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        broken = _FakeReplica(in_flight=5, fail=ProviderError("bad input", provider="llama-server"))
        p = _provider_with_clients({WorkerRole.EMBED: [dead, broken]})
        with pytest.raises(ProviderError, match="bad input"):
            p.embed(["a"])
        assert broken.healthy is True  # not a connection failure, stays in the pool

    def test_cooled_down_replica_gets_traffic_again_and_recovers(self, monkeypatch) -> None:
        import httpx as _httpx

        from lilbee.providers.fleet import client as client_mod
        from lilbee.providers.fleet.client import LlamaServerClient

        clock = {"now": 0.0}
        monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock["now"])
        calls: dict[str, int] = {"recovered": 0, "sibling": 0}

        def _handler(name: str):
            def handler(_request: _httpx.Request) -> _httpx.Response:
                calls[name] += 1
                vector = base64.b64encode(struct.pack("<f", 0.25)).decode()
                return _httpx.Response(200, json={"data": [{"embedding": vector}]})

            return handler

        def _real_client(name: str) -> LlamaServerClient:
            http = _httpx.Client(
                transport=_httpx.MockTransport(_handler(name)), base_url="http://gpu"
            )
            return LlamaServerClient("http://gpu", name, http=http)

        recovered, sibling = _real_client("recovered"), _real_client("sibling")
        recovered.mark_unhealthy()
        p = _provider_with_clients({WorkerRole.EMBED: [recovered, sibling]})
        p.embed(["a"])  # within the cool-down: routed to the sibling only
        assert (calls["recovered"], calls["sibling"]) == (0, 1)
        clock["now"] = client_mod._UNHEALTHY_RETRY_S
        p.embed(["b"])  # cooled down: the replica is routable again (the probe)
        assert calls["recovered"] == 1
        assert recovered.healthy is True  # the successful probe restored it


class TestVisionTimeout:
    def test_deadline_signal_maps_to_vision_timeout_error(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.fleet.client import ChatDeadlineError

        client = _fake_client(0)
        client.chat_bounded.side_effect = ChatDeadlineError("deadline", provider="llama-server")
        with pytest.raises(ProviderError, match="Vision OCR timed out after 12s") as excinfo:
            prov_mod._vision_call(client, [{"role": "user", "content": "x"}], 12.0)
        # A timeout is not a connection failure, so failover must not retry it.
        assert not prov_mod.is_connection_failure(excinfo.value)

    def test_bounded_call_passes_the_deadline_to_chat_bounded(self) -> None:
        client = _fake_client(0)
        client.chat_bounded.return_value = "text"
        assert prov_mod._vision_call(client, [{"role": "user", "content": "x"}], 9.0) == "text"
        assert client.chat_bounded.call_args.kwargs["deadline_s"] == 9.0


class TestReloadSingleFlight:
    def test_concurrent_reload_calls_dispatch_one_reload(self, monkeypatch) -> None:
        threads: list[object] = []

        class _RecordingThread:
            def __init__(self, *, target, name, daemon) -> None:
                self.target = target
                threads.append(self)

            def start(self) -> None:
                return None  # held un-run so the second call sees the in-flight guard

        monkeypatch.setattr(prov_mod.threading, "Thread", _RecordingThread)
        p = _provider_with_clients({})
        p.reload_role(WorkerRole.CHAT)
        p.reload_role(WorkerRole.EMBED)  # racing call while the first is in flight
        assert len(threads) == 1
        assert p._reload_pending is True  # queued for the in-flight thread, not dropped

    def test_reload_requested_mid_flight_runs_a_second_pass(self, monkeypatch) -> None:
        plans: list[int] = []

        def _plan() -> planning_mod.FleetPlan:
            plans.append(len(plans))
            # fresh object -> differs -> restarts
            return planning_mod.FleetPlan((_fake_launch(WorkerRole.CHAT),))

        monkeypatch.setattr(planning_mod, "plan_all_launches", _plan)
        monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _FakeSwap())
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
        monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
        p = FleetProvider()
        monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._reloading = True  # an in-flight reload that already snapshotted its plan
        p.reload_role(WorkerRole.EMBED)  # the second settings change arrives mid-flight
        assert p._reload_pending is True
        p._reload_blocking()  # the in-flight thread runs to completion
        assert len(plans) == 2  # one pass per request: the change was applied
        assert p._reloading is False
        assert p._reload_pending is False

    def test_fresh_reload_clears_a_stale_pending_flag(self, monkeypatch) -> None:
        threads: list[object] = []

        class _RecordingThread:
            def __init__(self, *, target, name, daemon) -> None:
                threads.append(self)

            def start(self) -> None:
                return None

        monkeypatch.setattr(prov_mod.threading, "Thread", _RecordingThread)
        p = _provider_with_clients({})
        p._reload_pending = True  # left over; the fresh pass plans from current cfg
        p.reload_role(WorkerRole.CHAT)
        assert len(threads) == 1
        assert p._reload_pending is False

    def test_reload_pass_failure_clears_guards_and_propagates(self, monkeypatch) -> None:
        class _ExplodingSwap(_FakeSwap):
            def start(self, launches: list, **_kw: object) -> None:
                raise RuntimeError("respawn failed")

        plans: list[int] = []

        def _plan() -> planning_mod.FleetPlan:
            plans.append(len(plans))
            return planning_mod.FleetPlan((_fake_launch(WorkerRole.CHAT),))

        monkeypatch.setattr(planning_mod, "plan_all_launches", _plan)
        monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _ExplodingSwap())
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._reloading = True
        p._reload_pending = True
        with pytest.raises(RuntimeError, match="respawn failed"):
            p._reload_blocking()
        assert len(plans) == 2  # the pending pass still ran before the failure surfaced
        assert p._reloading is False
        assert p._reload_pending is False

    def test_failed_pass_still_applies_the_pending_change(self, monkeypatch) -> None:
        built: list[_FakeSwap] = []

        class _FlakyFirstSwap(_FakeSwap):
            def start(self, launches: list, **_kw: object) -> None:
                if len(built) == 1:  # only the first fresh manager fails its spawn
                    raise RuntimeError("first pass failed")
                super().start(launches)

        def _factory(_d: object, _g: object) -> _FakeSwap:
            built.append(_FlakyFirstSwap())
            return built[-1]

        monkeypatch.setattr(prov_mod, "SwapManager", _factory)
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
        monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
        monkeypatch.setattr(
            planning_mod,
            "plan_all_launches",
            lambda: planning_mod.FleetPlan((_fake_launch(WorkerRole.CHAT),)),
        )
        p = FleetProvider()
        monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._reloading = True
        p._reload_pending = True  # a settings change arrived during the failing pass
        p._reload_blocking()  # must not raise: the pending pass succeeded
        assert len(built) == 2
        assert p._swaps.get(WorkerRole.CHAT) is built[-1]  # adopted by the successful pass
        assert p._reloading is False
        assert p._reload_pending is False

    def test_final_pass_failure_drops_the_dead_swap(self, monkeypatch) -> None:
        class _ExplodingSwap(_FakeSwap):
            def start(self, launches: list, **_kw: object) -> None:
                self.running = False  # the failed restart tore the process down
                raise RuntimeError("respawn failed")

        monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _ExplodingSwap())
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
        monkeypatch.setattr(
            planning_mod,
            "plan_all_launches",
            lambda: planning_mod.FleetPlan((_fake_launch(WorkerRole.CHAT),)),
        )
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._reloading = True
        with pytest.raises(RuntimeError, match="respawn failed"):
            p._reload_blocking()
        assert p._swaps == {}  # the next call rebuilds instead of hitting a dead swap

    def test_planning_failure_keeps_a_live_swap(self, monkeypatch) -> None:
        swap = _FakeSwap()
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: swap}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._reloading = True

        def _broken_plan() -> list:
            raise RuntimeError("no devices")

        monkeypatch.setattr(planning_mod, "plan_all_launches", _broken_plan)
        with pytest.raises(RuntimeError, match="no devices"):
            p._reload_blocking()
        assert p._swaps.get(WorkerRole.CHAT) is swap  # still running and serving the old config
        assert swap.shutdowns == 0

    def test_reload_clears_the_guard_when_done(self, monkeypatch) -> None:
        swap = _FakeSwap()
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: swap}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._reloading = True
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
        monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: planning_mod.FleetPlan(()))
        p._reload_blocking()
        assert swap.shutdowns == 1  # nothing planned -> the running group stops
        assert p._reloading is False
        p.reload_role(WorkerRole.CHAT)  # guard released -> a new reload can dispatch

    def test_reload_blocking_noops_when_swap_already_gone(self, monkeypatch) -> None:
        # Nothing running and nothing planned: the pass replans (a resurrect
        # would start whatever the fresh plan holds), finds nothing, and the
        # guard is still released.
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
        monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: planning_mod.FleetPlan(()))
        p = FleetProvider()
        p._reloading = True
        p._reload_blocking()
        assert p._reloading is False
        assert p._swaps == {}

    def test_reload_racing_shutdown_serializes_and_leaks_nothing(self, monkeypatch) -> None:
        order: list[str] = []
        gate = threading.Event()

        class _OrderedSwap(_FakeSwap):
            def shutdown(self) -> None:
                order.append("shutdown")
                super().shutdown()

        swap = _OrderedSwap()
        p = FleetProvider()
        p._swaps = {SwapGroup.CHAT: swap}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._reloading = True

        reload_entered = threading.Event()

        def _slow_plan() -> planning_mod.FleetPlan:
            reload_entered.set()
            gate.wait(5.0)
            return planning_mod.FleetPlan(())

        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
        monkeypatch.setattr(prov_mod, "stop_engine", lambda _d: None)
        monkeypatch.setattr(planning_mod, "plan_all_launches", _slow_plan)
        reloader = threading.Thread(target=p._reload_blocking)
        reloader.start()
        assert reload_entered.wait(5.0)  # the reload holds the build lock first
        shutter = threading.Thread(target=p._shutdown_swap)
        shutter.start()
        time.sleep(0.05)
        assert order == []  # shutdown is blocked behind the in-flight reload
        gate.set()
        reloader.join(timeout=5.0)
        shutter.join(timeout=5.0)
        # The reload's stop phase ran (nothing planned -> group stops); the
        # terminal shutdown then dropped refs, serialized on the build lock.
        assert order == ["shutdown"]
        assert p._swaps == {}  # the shutdown's state cleanup still landed


class TestWarmProgressTracking:
    """The chat-role warm path drives the WarmProgress tracker for launchers."""

    @staticmethod
    def _patch_registry(monkeypatch, registry: MagicMock) -> None:
        # ModelRegistry is imported at the fleet module top, so patch it there.
        monkeypatch.setattr(prov_mod, "ModelRegistry", lambda _dir: registry)

    def test_prewarm_reads_shards_and_reports_full_byte_progress(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        shard_a = tmp_path / "model-00001.gguf"
        shard_b = tmp_path / "model-00002.gguf"
        shard_a.write_bytes(b"a" * 4096)
        shard_b.write_bytes(b"b" * 2048)
        registry = MagicMock()
        registry.shard_paths.return_value = [shard_a, shard_b]
        self._patch_registry(monkeypatch, registry)
        monkeypatch.setattr(cfg, "chat_model", "repo/model-00001.gguf")

        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        p._warm_tracker.begin("repo/model-00001.gguf")
        p._prewarm_chat_weights()

        snap = p._warm_tracker.snapshot()
        assert snap.phase is WarmPhase.READING_WEIGHTS
        assert snap.bytes_done == 4096 + 2048
        assert snap.bytes_total == 4096 + 2048
        assert snap.detail == "shard 2/2"  # multi-shard detail string

    def test_prewarm_single_shard_omits_detail(self, monkeypatch, tmp_path: Path) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        shard = tmp_path / "solo.gguf"
        shard.write_bytes(b"x" * 1024)
        registry = MagicMock()
        registry.shard_paths.return_value = [shard]
        self._patch_registry(monkeypatch, registry)
        monkeypatch.setattr(cfg, "chat_model", "repo/solo.gguf")

        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        p._warm_tracker.begin("repo/solo.gguf")
        p._prewarm_chat_weights()
        snap = p._warm_tracker.snapshot()
        assert snap.phase is WarmPhase.READING_WEIGHTS
        assert snap.bytes_done == 1024
        assert snap.detail is None  # single shard: no "shard N/M" label

    def test_prewarm_skips_unresolvable_ref(self, monkeypatch) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        registry = MagicMock()
        registry.shard_paths.side_effect = KeyError("not registered")
        self._patch_registry(monkeypatch, registry)
        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        p._warm_tracker.begin("ghost/model.gguf")
        p._prewarm_chat_weights()
        # Returned before the read phase: the tracker stays at STARTING.
        assert p._warm_tracker.snapshot().phase is WarmPhase.STARTING

    def test_prewarm_skips_when_shards_are_empty(self, monkeypatch) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        registry = MagicMock()
        registry.shard_paths.return_value = []  # nothing to size -> total 0
        self._patch_registry(monkeypatch, registry)
        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        p._warm_tracker.begin("repo/m.gguf")
        p._prewarm_chat_weights()
        assert p._warm_tracker.snapshot().phase is WarmPhase.STARTING

    def test_prewarm_tolerates_an_unreadable_shard_mid_read(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        good = tmp_path / "good.gguf"
        good.write_bytes(b"a" * 2048)
        # A directory stats fine (so sizing succeeds) but open('rb') raises
        # IsADirectoryError (an OSError), exercising the inner read guard.
        unreadable = tmp_path / "broken.gguf"
        unreadable.mkdir()
        registry = MagicMock()
        registry.shard_paths.return_value = [good, unreadable]
        self._patch_registry(monkeypatch, registry)
        monkeypatch.setattr(cfg, "chat_model", "repo/m.gguf")

        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        p._warm_tracker.begin("repo/m.gguf")
        p._prewarm_chat_weights()  # must not raise
        snap = p._warm_tracker.snapshot()
        assert snap.phase is WarmPhase.READING_WEIGHTS
        assert snap.bytes_done == 2048  # only the readable shard counted

    def test_preload_marks_chat_ready_when_a_warm_request_returns(self, monkeypatch) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        monkeypatch.setattr(p, "_prewarm_chat_weights", lambda: None)
        monkeypatch.setattr(cfg, "chat_model", "repo/m.gguf")
        p._preload_roles()  # the fake client's warm request returns -> ready
        assert p._warm_tracker.snapshot().phase is WarmPhase.READY

    def test_preload_marks_chat_failed_when_every_warm_request_raises(self, monkeypatch) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        client = _fake_client()
        client.chat.side_effect = RuntimeError("engine never came up")
        p = _provider_with_clients({WorkerRole.CHAT: [client]})
        monkeypatch.setattr(p, "_prewarm_chat_weights", lambda: None)
        monkeypatch.setattr(cfg, "chat_model", "repo/m.gguf")
        p._preload_roles()
        snap = p._warm_tracker.snapshot()
        assert snap.phase is WarmPhase.ERROR
        assert snap.error

    def test_warm_progress_snapshot_exposed(self, monkeypatch) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        assert p.warm_progress() is None  # nothing warming yet
        p._warm_tracker.begin("repo/m.gguf")
        p._warm_tracker.ready()
        assert p.warm_progress().phase is WarmPhase.READY


def test_require_clients_reprobes_dead_swap(monkeypatch) -> None:
    """Empty pool + dead swap triggers a one-shot rebuild; clients are returned after."""
    from unittest import mock

    p = FleetProvider()
    p._clients = {}
    dead = mock.Mock()
    dead.is_live.return_value = False
    p._swaps = {SwapGroup.CHAT: dead}
    p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
    rebuilt = {"called": False}

    def fake_rebuild(role: WorkerRole) -> None:
        rebuilt["called"] = True
        p._clients = {WorkerRole.CHAT: [_fake_client()]}

    monkeypatch.setattr(p, "_rebuild_role", fake_rebuild, raising=False)
    clients = p._require_clients(WorkerRole.CHAT)
    assert rebuilt["called"] is True
    assert len(clients) == 1


def test_require_clients_no_reprobe_when_swap_none(monkeypatch) -> None:
    """Empty pool + no swap at all (unconfigured role) still raises, no rebuild."""
    from lilbee.providers.base import ProviderError

    rebuilt = {"called": False}

    def fake_rebuild(role: WorkerRole) -> None:
        rebuilt["called"] = True

    p = FleetProvider()
    p._swaps = {}
    p._clients = {}
    monkeypatch.setattr(p, "_rebuild_role", fake_rebuild, raising=False)
    with pytest.raises(ProviderError, match="No chat model server is running"):
        p._require_clients(WorkerRole.CHAT)
    assert rebuilt["called"] is False


def test_require_clients_no_reprobe_when_swap_live(monkeypatch) -> None:
    """Empty pool + live swap (real misconfiguration) still raises, no rebuild."""
    from unittest import mock

    from lilbee.providers.base import ProviderError

    rebuilt = {"called": False}

    def fake_rebuild(role: WorkerRole) -> None:
        rebuilt["called"] = True

    p = FleetProvider()
    live = mock.Mock()
    live.is_live.return_value = True
    p._swaps = {SwapGroup.CHAT: live}
    p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
    p._clients = {}
    monkeypatch.setattr(p, "_rebuild_role", fake_rebuild, raising=False)
    with pytest.raises(ProviderError, match="No chat model server is running"):
        p._require_clients(WorkerRole.CHAT)
    assert rebuilt["called"] is False


def test_rebuild_role_restarts_only_that_role(monkeypatch) -> None:
    """A dead group's rebuild replaces just that role's swap; the live one stays."""
    fresh = _FakeSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: fresh)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    chat_launch, embed_launch = _fake_launch(WorkerRole.CHAT), _fake_launch(WorkerRole.EMBED)
    monkeypatch.setattr(
        planning_mod,
        "plan_all_launches",
        lambda: planning_mod.FleetPlan((chat_launch, embed_launch)),
    )
    p = FleetProvider()
    monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
    dead, live = _FakeSwap(), _FakeSwap()
    p._swaps = {WorkerRole.EMBED: dead, WorkerRole.CHAT: live}
    # The running launches match the plan, so nothing restarts on its own; the
    # force set is what replaces the dead embed group.
    p._launches = {
        WorkerRole.EMBED: (embed_launch,),
        WorkerRole.CHAT: (chat_launch,),
    }
    p._rebuild_role(WorkerRole.EMBED)
    assert dead.shutdowns == 1  # the dead group was torn down...
    assert p._swaps[WorkerRole.EMBED] is fresh  # ...and replaced from the fresh plan
    assert p._swaps[WorkerRole.CHAT] is live  # the healthy group was never touched
    assert live.shutdowns == 0


def test_co_tenant_roles_share_one_swap_process(monkeypatch, tmp_path: Path) -> None:
    """Chat and vision must land in the same llama-swap process; only a shared process
    can evict one to load the other. Separate processes would hold both resident."""
    groups: list[SwapGroup] = []

    def _factory(_d: object, group: SwapGroup) -> _FakeSwap:
        groups.append(group)
        return _FakeSwap()

    chat, vision = _fake_launch(WorkerRole.CHAT), _fake_launch(WorkerRole.VISION)
    embed = _fake_launch(WorkerRole.EMBED)
    import tempfile

    monkeypatch.setattr(prov_mod, "SwapManager", _factory)
    monkeypatch.setattr(
        prov_mod,
        "machine_engine_dir",
        lambda: Path(tempfile.mkdtemp(prefix="lilbee-slot-", dir=tmp_path)),
    )
    monkeypatch.setattr(prov_mod, "engine_pin", lambda: "test-pin")
    monkeypatch.setattr(prov_mod, "state_is_healthy", lambda _s: True)
    monkeypatch.setattr(prov_mod, "stop_engine", lambda _d: None)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
    monkeypatch.setattr(planning_mod, "capture_plan_probe", lambda: None)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    monkeypatch.setattr(
        planning_mod,
        "plan_all_launches",
        lambda: planning_mod.FleetPlan(
            (chat, vision, embed),
            co_tenants=frozenset({WorkerRole.CHAT, WorkerRole.VISION}),
        ),
    )

    p = FleetProvider()
    assert p._ensure_fleet() is True

    assert sorted(groups) == sorted([SwapGroup.CO_TENANT, SwapGroup.EMBED])
    assert p._role_group[WorkerRole.CHAT] is SwapGroup.CO_TENANT
    assert p._role_group[WorkerRole.VISION] is SwapGroup.CO_TENANT
    assert p._role_group[WorkerRole.EMBED] is SwapGroup.EMBED
    # Both roles keep their own client pool even though they share a process.
    assert len(p._clients[WorkerRole.CHAT]) == 1
    assert len(p._clients[WorkerRole.VISION]) == 1
    # The shared group's launches are filtered down to the role's own replicas.
    assert p._role_launches(WorkerRole.VISION) == (vision,)


def test_ensure_fleet_partial_failure_tears_down_started_groups(monkeypatch) -> None:
    """A later group failing to start must stop the groups already started, so a
    half-built fleet never leaks past the failure."""
    built: list[_FakeSwap] = []

    class _SecondExplodes(_FakeSwap):
        def start(self, launches: list, **_kw: object) -> None:
            if len(built) > 1:
                raise RuntimeError("second group failed")
            super().start(launches)

    def _factory(_d: object, _g: object) -> _FakeSwap:
        built.append(_SecondExplodes())
        return built[-1]

    monkeypatch.setattr(prov_mod, "SwapManager", _factory)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    monkeypatch.setattr(
        planning_mod,
        "plan_all_launches",
        lambda: planning_mod.FleetPlan(
            (_fake_launch(WorkerRole.CHAT), _fake_launch(WorkerRole.EMBED))
        ),
    )
    p = FleetProvider()
    with pytest.raises(RuntimeError, match="second group failed"):
        p._ensure_fleet()
    assert built[0].shutdowns == 1  # the group that did start was torn down
    assert p._swaps == {}


def test_drop_dead_swaps_drops_only_dead_groups() -> None:
    p = FleetProvider()
    dead, live = _FakeSwap(), _FakeSwap()
    dead.running = False
    p._swaps = {WorkerRole.EMBED: dead, WorkerRole.CHAT: live}
    p._drop_dead_swaps()
    assert set(p._swaps) == {WorkerRole.CHAT}  # the live group is untouched


def test_reload_placement_dispatches_the_diff_reload(monkeypatch) -> None:
    passes: list[bool] = []
    p = FleetProvider()
    p._swaps = {SwapGroup.CHAT: _FakeSwap()}
    p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
    monkeypatch.setattr(p, "_reload_pass", lambda force=frozenset(): passes.append(True))
    p.reload_placement(wait=True)
    assert passes == [True]


def test_reload_pass_refuses_after_terminal_shutdown(monkeypatch) -> None:
    """A reload queued behind a terminal shutdown must not resurrect the fleet."""
    plans: list[int] = []
    monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: plans.append(1) or [])
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
    p = FleetProvider()
    p._shut_down = True
    p._reload_pass(force=frozenset((WorkerRole.CHAT,)))
    assert plans == []  # returned before planning; nothing can spawn


class TestPlanProbeLifecycle:
    """The provider owns the plan snapshot: captured on clean-box builds only."""

    def _wire(self, monkeypatch, tmp_path: Path, order: list[str]) -> None:
        import tempfile

        monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _FakeSwap())
        monkeypatch.setattr(
            prov_mod,
            "machine_engine_dir",
            lambda: Path(tempfile.mkdtemp(prefix="lilbee-slot-", dir=tmp_path)),
        )
        monkeypatch.setattr(prov_mod, "engine_pin", lambda: "test-pin")
        monkeypatch.setattr(prov_mod, "state_is_healthy", lambda _s: True)
        monkeypatch.setattr(prov_mod, "stop_engine", lambda _d: None)
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: order.append("reap"))
        monkeypatch.setattr(planning_mod, "capture_plan_probe", lambda: order.append("capture"))
        monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
        monkeypatch.setattr(
            planning_mod,
            "plan_all_launches",
            lambda: (
                order.append("plan") or planning_mod.FleetPlan((_fake_launch(WorkerRole.CHAT),))
            ),
        )

    def test_first_build_snapshots_after_reaping(self, monkeypatch, tmp_path: Path) -> None:
        # Capture must follow the reap (a dead owner's servers still hold VRAM
        # before it) and precede planning (the plan sizes against the snapshot).
        order: list[str] = []
        self._wire(monkeypatch, tmp_path, order)
        FleetProvider()._ensure_fleet()
        # The clean-box snapshot and the sizing plan it feeds still follow the reap;
        # a placeable-set plan (total-VRAM, reap-independent) may precede it.
        assert order[-3:] == ["reap", "capture", "plan"]
        assert "capture" not in order[: order.index("reap")]

    def test_reload_with_a_loaded_fleet_reuses_the_snapshot(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        # THE #474 follow-up regression guard: re-planning while our own fleet
        # holds VRAM must not re-probe (which would shrink ctx/slots and diff
        # every launch); the boot snapshot stays the sizing basis.
        order: list[str] = []
        self._wire(monkeypatch, tmp_path, order)
        p = FleetProvider()
        monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
        p._swaps = {SwapGroup.CHAT: _FakeSwap()}
        p._role_group = {WorkerRole.CHAT: SwapGroup.CHAT}
        p._reload_pass()
        assert "capture" not in order
        assert "plan" in order

    def test_resurrect_reload_recaptures_the_clean_box(self, monkeypatch, tmp_path: Path) -> None:
        order: list[str] = []
        self._wire(monkeypatch, tmp_path, order)
        p = FleetProvider()
        monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
        p._reload_pass(force=frozenset((WorkerRole.CHAT,)))  # nothing running
        assert order.index("capture") < order.index("plan")

    def test_full_teardown_clears_the_snapshot(self, monkeypatch) -> None:
        cleared: list[bool] = []
        monkeypatch.setattr(planning_mod, "clear_plan_probe", lambda: cleared.append(True))
        FleetProvider()._drop_swap_refs()
        assert cleared == [True]


class TestWarmFailureIsSurfaced:
    """A model that cannot load must not fail silently at debug level."""

    def test_warm_failure_logs_the_underlying_error_at_warning(self, caplog) -> None:
        p = FleetProvider()
        client = _fake_client()
        client.chat.side_effect = RuntimeError("unknown model architecture: qwen35moe")

        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.provider"):
            warmed = p._warm_role_clients(WorkerRole.CHAT, [client])

        assert warmed is False
        assert "unknown model architecture: qwen35moe" in caplog.text
        assert "chat" in caplog.text

    def test_successful_warm_logs_nothing(self, caplog) -> None:
        p = FleetProvider()
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.provider"):
            assert p._warm_role_clients(WorkerRole.EMBED, [_fake_client()]) is True
        assert caplog.text == ""


class TestChatLoadFailureMessage:
    """The launcher and the TUI must see the engine's real reason, not a generic one."""

    def test_failure_message_carries_the_engine_error(self) -> None:
        p = FleetProvider()
        p._warm_errors[WorkerRole.CHAT] = "unknown model architecture: qwen35moe"
        assert "unknown model architecture: qwen35moe" in p._chat_load_failure()

    def test_failure_message_falls_back_when_the_engine_said_nothing(self) -> None:
        assert "did not finish loading" in FleetProvider()._chat_load_failure()


class TestWarmErrorsClearedOnTeardown:
    def test_dropping_the_fleet_forgets_its_load_failures(self) -> None:
        # The errors describe servers that no longer exist; a rebuilt fleet must not
        # inherit them.
        p = FleetProvider()
        p._warm_errors[WorkerRole.CHAT] = "unknown model architecture: qwen35moe"
        p._drop_swap_refs()
        assert p._warm_errors == {}
        assert "did not finish loading" in p._chat_load_failure()


class TestLaunchStateRoundTrip:
    def test_round_trips_every_field(self) -> None:
        launch = InstanceLaunch(
            role=WorkerRole.RERANK,
            argv=["/bin/llama-server", "--flag"],
            env_overrides={"CUDA_VISIBLE_DEVICES": "1"},
            model="o/r-GGUF/r.gguf",
            token_cap=512,
            weights_bytes=123,
            slots=2,
            ctx=4096,
            replica=1,
            rerank_mode=RerankMode.LLM,
            est_vram_bytes=7 * 1024**3,
            est_vram_by_device={"CUDA0": 4 * 1024**3, "CUDA1": 3 * 1024**3},
        )
        rebuilt = InstanceLaunch.from_state(launch.to_state())
        assert rebuilt == launch

    def test_a_state_file_written_before_the_estimate_existed_still_loads(self) -> None:
        # Every field carries a default on the way in, so an engine recorded by an
        # older lilbee is readable rather than a hard failure on upgrade.
        payload = {
            "role": "chat",
            "argv": ["/bin/llama-server"],
            "model": "o/c-GGUF/c.gguf",
        }
        rebuilt = InstanceLaunch.from_state(payload)
        assert rebuilt.est_vram_bytes == 0
        assert rebuilt.est_vram_by_device == {}


# ── The engine acquisition ladder ───────────────────────────────────


def _engine_state_file(engine_dir: Path, group: str, *, pin: str, model: str, role: str) -> Path:
    """A live engine's state record as another process would have written it."""
    import json

    from lilbee.providers.fleet import swap_manager as sm

    engine_dir.mkdir(parents=True, exist_ok=True)
    path = engine_dir / sm._state_filename(999_999, group)
    path.write_text(
        json.dumps(
            {
                "pid": 999_998,
                "member_ports": [4000],
                "proxy_port": 4100,
                "launches": [
                    {
                        "role": role,
                        "argv": ["/bin/llama-server"],
                        "env_overrides": {},
                        "model": model,
                    }
                ],
                "engine_pin": pin,
            }
        )
    )
    return path


class _BindableSwap(_FakeSwap):
    """A fake swap that can also bind to an existing engine record."""

    def __init__(self) -> None:
        super().__init__()
        self.binds: list[object] = []
        self.bind_result = True
        self.bound = False

    def bind(self, state) -> bool:
        self.binds.append(state)
        if self.bind_result:
            self.bound = True
        return self.bind_result


def _install_ladder(
    monkeypatch,
    tmp_path: Path,
    *,
    launches: list,
    swap: _BindableSwap | None = None,
    pin: str = "pin-a",
):
    """Point the ladder at a tmp machine dir with controllable fakes."""
    swap = swap or _BindableSwap()
    machine = tmp_path / "machine-slot"
    built_dirs: list[Path] = []

    def _swap_factory(data_dir, _group):
        built_dirs.append(Path(data_dir))
        return swap

    monkeypatch.setattr(prov_mod, "SwapManager", _swap_factory)
    monkeypatch.setattr(prov_mod, "machine_engine_dir", lambda: machine)
    monkeypatch.setattr(prov_mod, "engine_pin", lambda: pin)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _data_dir, **_kw: None)
    monkeypatch.setattr(prov_mod, "stop_engine", lambda _data_dir: None)
    monkeypatch.setattr(prov_mod, "state_is_healthy", lambda _state: True)
    monkeypatch.setattr(planning_mod, "capture_plan_probe", lambda: None)
    # Everything the test configures is placeable and buildable unless overridden.
    monkeypatch.setattr(planning_mod, "placeable_total_vram", lambda: 0)
    monkeypatch.setattr(planning_mod, "role_model_placeable", lambda _role, _ref, _vram: True)
    monkeypatch.setattr(prov_mod, "_can_build_engine", lambda _wanted: True)
    monkeypatch.setattr(
        prov_mod, "LlamaServerClient", lambda _endpoint, _model, **_kw: _fake_client()
    )
    monkeypatch.setattr(
        planning_mod, "plan_all_launches", lambda: planning_mod.FleetPlan(tuple(launches))
    )
    return swap, machine, built_dirs


def _chat_launch() -> InstanceLaunch:
    return InstanceLaunch(
        role=WorkerRole.CHAT, argv=["/bin/llama-server"], env_overrides={}, model="m-chat"
    )


def test_ladder_never_kills_a_live_engine_on_a_probe_failure(monkeypatch, tmp_path: Path) -> None:
    """A busy engine whose proxy probe transiently fails must survive.

    Membership (a live user lock), not the HTTP health probe, decides whether an
    engine may be replaced. With every probe failing (fd exhaustion, host thrash)
    but a live user present, the ladder must not reap or stop the engine; it
    overflows to the private dir instead.
    """
    _swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    reaped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(prov_mod, "reap_stale", lambda d, **_k: reaped.append(Path(d)))
    monkeypatch.setattr(prov_mod, "state_is_healthy", lambda _state: False)  # every probe fails
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    from lilbee.runtime.engine_lock import hold_user_lock

    _engine_state_file(machine, "chat", pin="pin-a", model="m-chat", role="chat")
    holder = hold_user_lock(machine, pid=999_888)  # the engine is in live use
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert machine not in stopped and machine not in reaped  # the live engine is untouched
    holder.release_and_check_last()


def test_ladder_leaves_a_warm_engine_it_cannot_replace(monkeypatch, tmp_path: Path) -> None:
    """A process that can serve nothing must not stop an engine left warm.

    keep_engine_warm leaves the machine engine running with no live users. A
    process whose models are all unplaceable (or whose binary is missing) would
    otherwise stop that warm engine and then spawn nothing.
    """
    _swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(prov_mod, "_placeable_wanted", lambda: set())  # nothing placeable
    monkeypatch.setattr(prov_mod, "_can_build_engine", lambda _wanted: False)
    _engine_state_file(machine, "chat", pin="pin-a", model="m-warm", role="chat")
    p = FleetProvider()
    assert p._ensure_fleet() is False  # serves nothing...
    assert stopped == []  # ...but never stops the warm engine


def test_ladder_binds_when_a_configured_role_is_unplaceable(monkeypatch, tmp_path: Path) -> None:
    """A configured-but-unplaceable role must not keep the engine restarting.

    The engine serves the installed chat model; embed is configured but not
    installed, so a fresh plan omits it. wanted must reflect that placeable set
    so bind matches and the engine is never stopped and rebuilt (the storm).
    """
    swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(
        prov_mod,
        "_configured_model_for",
        lambda role: {"chat": "m-chat", "embed": "m-embed-missing"}.get(role.value, ""),
    )
    # embed's model is not installed, so the planner would drop it; chat is fine.
    monkeypatch.setattr(
        planning_mod,
        "role_model_placeable",
        lambda role, _ref, _vram: role is WorkerRole.CHAT,
    )
    _engine_state_file(machine, "chat", pin="pin-a", model="m-chat", role="chat")
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert len(swap.binds) == 1  # bound to the running engine...
    assert swap.started == [] and stopped == []  # ...never stopped or rebuilt


def test_ladder_binds_to_a_matching_machine_engine(monkeypatch, tmp_path: Path) -> None:
    swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    _engine_state_file(machine, "chat", pin="pin-a", model="m-chat", role="chat")
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert len(swap.binds) == 1
    assert swap.started == []  # bound, never built
    # Binding must take membership too, or a peer's clean exit stops the engine
    # this process is actively using -- the multi-process bug the ladder exists
    # to prevent. Only the build path asserted this before.
    assert list((machine / "engine-users").glob("*.lock"))


def test_ladder_builds_into_the_machine_dir_when_slot_is_empty(monkeypatch, tmp_path: Path) -> None:
    swap, machine, built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert swap.started  # built
    assert built and built[0] == machine  # into the machine slot
    users = list((machine / "engine-users").glob("*.lock"))
    assert len(users) == 1  # holding membership


def test_ladder_overflows_to_private_dir_on_pin_mismatch(monkeypatch, tmp_path: Path) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    swap, machine, built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    _engine_state_file(machine, "chat", pin="pin-OTHER", model="m-chat", role="chat")
    holder = hold_user_lock(machine, pid=999_888)  # the incumbent is in live use
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert swap.binds == []  # incompatible: never bound
    assert swap.started  # built instead
    assert built and built[0] == tmp_path / "root" / "data" / "engine"
    holder.release_and_check_last()


def test_ladder_private_path_stops_short_when_it_cannot_build(monkeypatch, tmp_path: Path) -> None:
    """Overflow to the private dir also refuses to build when nothing is buildable,
    rather than stopping a private incumbent it cannot replace."""
    from lilbee.runtime.engine_lock import hold_user_lock

    swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[])
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    monkeypatch.setattr(prov_mod, "_can_build_engine", lambda _wanted: False)
    _engine_state_file(machine, "chat", pin="pin-OTHER", model="m-chat", role="chat")
    holder = hold_user_lock(machine, pid=999_888)  # live foreign incumbent forces overflow
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is False  # can't build, so the private path serves nothing
    assert swap.started == []
    holder.release_and_check_last()


def test_ladder_replaces_an_unused_incompatible_machine_engine(monkeypatch, tmp_path: Path) -> None:
    """A wrong-shape incumbent nobody holds a user lock on is replaced in place.

    A fleet built while only some configured models were installed serves a
    partial contract; once its builder exits, overflowing around it would
    poison the machine slot for every later arrival.
    """
    swap, machine, built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    _engine_state_file(machine, "chat", pin="pin-OTHER", model="m-chat", role="chat")
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert swap.binds == []  # incompatible: never bound
    assert stopped == [machine]  # the unused incumbent was stopped...
    assert built and built[0] == machine  # ...and the slot rebuilt, not overflowed


def test_last_out_stops_the_engine(monkeypatch, tmp_path: Path) -> None:
    _swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", False, raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    p.shutdown()
    assert stopped == [machine]
    assert not list((machine / "engine-users").glob("*.lock"))


def test_not_last_leaves_the_engine(monkeypatch, tmp_path: Path) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    _swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", False, raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    peer = hold_user_lock(machine, pid=555_555)
    p.shutdown()
    assert stopped == []
    peer.release_and_check_last()


def _bindable_machine(monkeypatch, tmp_path: Path):
    """A machine slot holding an engine any compatible provider can bind."""
    swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    _engine_state_file(machine, "chat", pin="pin-a", model="m-chat", role="chat")
    return swap, machine


def test_a_default_config_peer_leaving_last_keeps_a_warm_users_engine(
    monkeypatch, tmp_path: Path
) -> None:
    """The machine slot is shared; whose config decides must not be exit order."""
    from lilbee.runtime.engine_lock import request_keep_warm

    swap, _machine = _bindable_machine(monkeypatch, tmp_path)
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))

    # The warm user is a sibling installation, seeded under its own config root.
    request_keep_warm(_machine, tmp_path / "peer-install")
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", False, raising=False)
    plain = FleetProvider()
    assert plain._ensure_fleet() is True
    assert swap.started == []  # bound the one machine engine

    plain.shutdown()  # last out, and its own config says stop
    assert stopped == []


def test_a_warm_peer_leaving_first_still_keeps_the_engine(monkeypatch, tmp_path: Path) -> None:
    """The opt-in belongs to the engine, so it outlives the process that made it."""
    from lilbee.runtime.engine_lock import request_keep_warm

    _swap, _machine = _bindable_machine(monkeypatch, tmp_path)
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))

    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", False, raising=False)
    plain = FleetProvider()
    assert plain._ensure_fleet() is True
    # A sibling installation opted in and has already left: only its mark
    # remains, which is exactly what "outlive me" means.
    request_keep_warm(_machine, tmp_path / "peer-install")

    plain.shutdown()  # last out, reading a config that says stop
    assert stopped == []


def test_opting_in_after_binding_still_keeps_the_engine(monkeypatch, tmp_path: Path) -> None:
    """The setting doesn't affect the load, so a flip never re-acquires."""
    _swap, _machine = _bindable_machine(monkeypatch, tmp_path)
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", False, raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True

    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", True, raising=False)
    p.shutdown()
    assert stopped == []


def test_a_config_change_leaves_an_engine_other_users_are_serving(
    monkeypatch, tmp_path: Path
) -> None:
    """One member's settings change must not interrupt every peer's requests."""
    from lilbee.runtime.engine_lock import hold_user_lock

    _swap, machine = _bindable_machine(monkeypatch, tmp_path)
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    p = FleetProvider()
    assert p._ensure_fleet() is True
    peer = hold_user_lock(machine, pid=555_555)

    p.invalidate_load_cache()

    assert stopped == []  # the peer is mid-request; we rebind or overflow on next use
    assert p._engine_holds == {}  # our membership is gone either way
    peer.release_and_check_last()


def test_a_config_change_stops_an_engine_no_one_else_is_using(monkeypatch, tmp_path: Path) -> None:
    """Nothing keeps a stale-config engine resident once its last user leaves."""
    _swap, machine = _bindable_machine(monkeypatch, tmp_path)
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    # Even an explicit warm opt-in does not preserve an engine built for a
    # configuration that no longer exists.
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", True, raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True

    p.invalidate_load_cache()

    assert stopped == [machine]


def test_stopping_an_engine_forgets_its_keep_warm_optin(tmp_path: Path) -> None:
    """The mark describes one engine instance; the next one starts unmarked."""
    from lilbee.providers.fleet.swap_manager import stop_engine
    from lilbee.runtime.engine_lock import keep_warm_requested, request_keep_warm

    request_keep_warm(tmp_path, tmp_path / "cfgroot")
    assert keep_warm_requested(tmp_path) is True
    stop_engine(tmp_path)
    assert keep_warm_requested(tmp_path) is False


def test_warm_leaves_the_engine_even_when_last(monkeypatch, tmp_path: Path) -> None:
    _swap, _machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", True, raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    p.shutdown()
    assert stopped == []


def test_turning_warm_off_mid_session_stops_the_engine(monkeypatch, tmp_path: Path) -> None:
    """A withdrawn opt-in must not keep the engine: the mark is this user's, not the engine's.

    Without withdrawal the mark outlives the setting, and because the mark is
    what suppresses ``stop_engine`` -- the only thing that clears it -- the
    engine stays resident on every later run too.
    """
    _swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", True, raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", False, raising=False)
    p.shutdown()
    assert stopped == [machine]


def test_a_prior_runs_keep_warm_is_reclaimed_when_the_setting_goes_off(
    monkeypatch, tmp_path: Path
) -> None:
    """A mark left by an earlier run of this install (new pid, same config root)

    is reclaimed on a later warm-off run, so the engine stops rather than staying
    warm forever. The opt-in is keyed by config root precisely so a restart can
    reach its own prior mark.
    """
    from lilbee.runtime.engine_lock import request_keep_warm

    _swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", False, raising=False)
    # An earlier run of this same installation opted in, then exited.
    request_keep_warm(machine, prov_mod.cfg.data_root)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    restarted_pid = os.getpid() + 4242
    monkeypatch.setattr(os, "getpid", lambda: restarted_pid)  # this run is a new process
    p.shutdown()
    assert stopped == [machine]


def test_warm_on_pins_weights_resident(monkeypatch) -> None:
    # keep_engine_warm means the model stays loaded, so the idle unload is off.
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", True, raising=False)
    monkeypatch.setattr(prov_mod.cfg, "engine_idle_ttl_minutes", 7, raising=False)
    assert prov_mod._warm_ttl_seconds() == 0


def test_ttl_applies_with_warm_off(monkeypatch) -> None:
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", False, raising=False)
    monkeypatch.setattr(prov_mod.cfg, "engine_idle_ttl_minutes", 7, raising=False)
    assert prov_mod._warm_ttl_seconds() == 420


def test_shutdown_latches_regardless_of_warm(monkeypatch, tmp_path: Path) -> None:
    _swap, _machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    monkeypatch.setattr(prov_mod.cfg, "keep_engine_warm", True, raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    p.shutdown()
    assert p._shut_down is True


def test_config_change_restarts_the_engine_and_releases_membership(
    monkeypatch, tmp_path: Path
) -> None:
    _swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    p = FleetProvider()
    assert p._ensure_fleet() is True
    p.invalidate_load_cache()
    assert stopped == [machine]  # the shared engine restarts for everyone
    # Membership is released, not kept: a retained hold would let a later config
    # change stop a foreign engine that claimed the slot, and keep it falsely live.
    # The next use re-runs the ladder and re-acquires membership in whatever dir it
    # rebuilds into.
    assert not list((machine / "engine-users").glob("*.lock"))
    assert p._engine_holds == {}
    assert p._shut_down is False  # provider reusable; next use rebuilds


def test_ladder_binds_in_the_private_dir_when_machine_is_incompatible(
    monkeypatch, tmp_path: Path
) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    _engine_state_file(machine, "chat", pin="pin-OTHER", model="m-chat", role="chat")
    holder = hold_user_lock(machine, pid=999_888)  # the incumbent is in live use
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    private = tmp_path / "root" / "data" / "engine"
    _engine_state_file(private, "chat", pin="pin-a", model="m-chat", role="chat")
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert len(swap.binds) == 1  # bound in the overflow dir
    assert swap.started == []
    assert list((private / "engine-users").glob("*.lock"))  # membership in the bound dir
    holder.release_and_check_last()


def test_ladder_replaces_an_unused_incompatible_private_engine(monkeypatch, tmp_path: Path) -> None:
    """The private dir gets the same replacement rule: no stacked second fleet."""
    from lilbee.runtime.engine_lock import hold_user_lock

    _swap, machine, built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    _engine_state_file(machine, "chat", pin="pin-OTHER", model="m-chat", role="chat")
    holder = hold_user_lock(machine, pid=999_888)  # machine incumbent is in live use
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    private = tmp_path / "root" / "data" / "engine"
    _engine_state_file(private, "chat", pin="pin-OTHER", model="m-chat", role="chat")
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert stopped == [private]  # unused private incumbent stopped, machine's left alone
    assert built and built[0] == private
    holder.release_and_check_last()


def test_ladder_does_not_kill_a_live_incompatible_overflow_engine(
    monkeypatch, tmp_path: Path
) -> None:
    """The overflow dir never evicts or stacks on a live in-use incompatible engine.

    With the machine slot held by one live setup and the overflow dir held by
    another, there is nowhere further to go; the ladder serves nothing rather than
    kill the overflow incumbent or load a second fleet's weights beside it.
    """
    from lilbee.runtime.engine_lock import hold_user_lock

    _swap, machine, built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    _engine_state_file(machine, "chat", pin="pin-OTHER", model="m-chat", role="chat")
    machine_holder = hold_user_lock(machine, pid=999_888)  # machine incumbent in live use
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    private = tmp_path / "root" / "data" / "engine"
    _engine_state_file(private, "chat", pin="pin-OTHER", model="m-chat", role="chat")
    private_holder = hold_user_lock(private, pid=999_777)  # overflow incumbent in live use too

    p = FleetProvider()
    assert p._ensure_fleet() is False  # nowhere to serve
    assert stopped == []  # neither live incumbent was killed
    assert built == []  # no second fleet stacked into the overflow dir

    machine_holder.release_and_check_last()
    private_holder.release_and_check_last()


def test_ladder_returns_false_when_overflow_has_nothing_to_serve(
    monkeypatch, tmp_path: Path
) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[])
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    _engine_state_file(machine, "chat", pin="pin-OTHER", model="m-chat", role="chat")
    holder = hold_user_lock(machine, pid=999_888)  # the incumbent is in live use
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is False
    assert swap.binds == [] and swap.started == []
    holder.release_and_check_last()


def test_ladder_ignores_groups_serving_only_unwanted_models(monkeypatch, tmp_path: Path) -> None:
    swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    # A pin-matching group serving a model nobody asked for: no bind, and with
    # no live users it is replaced along with the rest of the unused engine.
    _engine_state_file(machine, "embed", pin="pin-a", model="m-unrelated", role="embed")
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert swap.binds == []
    assert stopped == [machine]  # unused wrong-shape incumbent replaced
    assert swap.started  # built fresh in the machine slot


def test_ladder_rolls_back_earlier_binds_when_a_later_one_fails(
    monkeypatch, tmp_path: Path
) -> None:
    class _SecondBindFails(_BindableSwap):
        def __init__(self) -> None:
            super().__init__()
            self.results = [True, False]
            self.shutdowns_after_bind = 0

        def bind(self, state) -> bool:
            self.binds.append(state)
            return self.results.pop(0)

        def shutdown(self) -> None:
            self.shutdowns_after_bind += 1
            super().shutdown()

    swap = _SecondBindFails()
    # The placeable set (from the plan) is what the ladder tries to bind, so the
    # plan carries both roles the state files below serve.
    _swap, machine, _built = _install_ladder(
        monkeypatch,
        tmp_path,
        launches=[
            _fake_launch(WorkerRole.CHAT, model="m-chat"),
            _fake_launch(WorkerRole.EMBED, model="m-embed"),
        ],
        swap=swap,
        pin="pin-a",
    )
    monkeypatch.setattr(
        prov_mod,
        "_configured_model_for",
        lambda role: {"chat": "m-chat", "embed": "m-embed"}.get(role.value, ""),
    )
    _engine_state_file(machine, "chat", pin="pin-a", model="m-chat", role="chat")
    _engine_state_file(machine, "embed", pin="pin-a", model="m-embed", role="embed")
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    p = FleetProvider()
    p._ensure_fleet()
    assert len(swap.binds) == 2  # first bound, second refused
    assert swap.shutdowns_after_bind >= 1  # the earlier bind was rolled back


def test_ladder_rebuilds_partially_dead_compatible_machine_slot_in_place(
    monkeypatch, tmp_path: Path
) -> None:
    """A pin-equal incumbent missing a wanted group is rebuilt in place, not overflowed.

    The healthy groups serve only wanted models, so the slot is this
    contract's own engine and its live members are waiting for exactly this
    rebuild; overflowing around it would load duplicate weights.
    """
    from lilbee.runtime.engine_lock import hold_user_lock

    _swap, machine, built = _install_ladder(
        monkeypatch, tmp_path, launches=[_chat_launch(), _embed_launch()]
    )
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(
        prov_mod,
        "_configured_model_for",
        lambda role: {"chat": "m-chat", "embed": "m-embed"}.get(role.value, ""),
    )
    # Only the chat group survives; the wanted embed group has no live state.
    _engine_state_file(machine, "chat", pin="pin-a", model="m-chat", role="chat")
    holder = hold_user_lock(machine, pid=999_888)  # a member (e.g. serve) is still live
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert stopped == [machine]  # the partially dead engine was stopped...
    assert built and built[0] == machine  # ...and rebuilt in the machine slot
    holder.release_and_check_last()


def test_bindable_group_refuses_an_undecodable_contract_on_its_own(monkeypatch) -> None:
    """The bind path handles an undecodable contract itself, not by call ordering.

    Previously the bare decode was safe only because contract_matches ran first and
    its except clause proved decodability; reordering or dropping that guard turned
    a non-match into an unhandled exception in the ladder. With the guard stubbed
    permissive, an undecodable contract must still yield "not bindable" rather than
    raise.
    """
    state = object()  # opaque: the decode is what is under test
    monkeypatch.setattr(prov_mod, "contract_matches", lambda *_a, **_k: True)  # guard removed
    monkeypatch.setattr(prov_mod, "decoded_launches", lambda _s: None)  # undecodable record

    assert prov_mod._bindable_group(state, "pin-a", {(WorkerRole.CHAT, "m")}) is None


def test_ladder_probes_each_group_once_per_pass(monkeypatch, tmp_path: Path) -> None:
    """Bind eligibility and replaceability read one snapshot, not two probe passes.

    Every probe runs under the cross-process build lock that gates every other
    lilbee start, and two passes could also disagree about an engine that died
    between them.
    """
    from lilbee.runtime.engine_lock import hold_user_lock

    _swap, machine = _bindable_machine(monkeypatch, tmp_path)
    probed: list[int | None] = []
    real_healthy = prov_mod.state_is_healthy

    def counting_probe(state):
        probed.append(state.proxy_port)
        return real_healthy(state)

    monkeypatch.setattr(prov_mod, "state_is_healthy", counting_probe)
    # Want a model the recorded engine does not serve, so the bind fails and the
    # replaceability check runs too -- the path that used to re-probe. A live
    # peer keeps that check from short-circuiting on membership.
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-other" if role is WorkerRole.CHAT else ""
    )
    peer = hold_user_lock(machine, pid=555_555)

    p = FleetProvider()
    try:
        p._ensure_fleet()
    finally:
        peer.release_and_check_last()

    assert probed  # the ladder really did probe
    assert len(probed) == len(set(probed))  # and no port twice in one pass


def test_ladder_skips_the_shared_slot_without_kernel_arbitrated_locks(
    monkeypatch, tmp_path: Path
) -> None:
    """Membership is only meaningful when the kernel releases locks on death.

    On a filesystem where flock is unavailable, probing a live member's lock
    destroys it, so the shared slot would look free while another setup serves
    from it. The ladder must keep to this config root's own dir instead.
    """
    swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    _engine_state_file(machine, "chat", pin="pin-a", model="m-chat", role="chat")
    monkeypatch.setattr(prov_mod, "kernel_arbitrates_locks", lambda _d: False)

    p = FleetProvider()
    assert p._ensure_fleet() is True

    assert swap.binds == []  # never bound the shared engine
    assert machine not in p._engine_holds  # and took no membership in it


def test_healthy_groups_ours_is_false_with_no_healthy_group() -> None:
    # The vacuous case: an empty snapshot is "not ours" (live membership owns
    # that branch); the helper must not claim it.
    assert prov_mod._healthy_groups_ours({}, "pin-a", {(WorkerRole.CHAT, "m-chat")}) is False


def test_ladder_overflows_when_a_live_used_pin_equal_group_serves_unwanted_models(
    monkeypatch, tmp_path: Path
) -> None:
    """Pin-equal but serving a model outside the contract: protected while in use."""
    from lilbee.runtime.engine_lock import hold_user_lock

    _swap, machine, built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    _engine_state_file(machine, "embed", pin="pin-a", model="m-unrelated", role="embed")
    holder = hold_user_lock(machine, pid=999_888)  # that foreign fleet is in live use
    monkeypatch.setattr(prov_mod.cfg, "data_root", tmp_path / "root", raising=False)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert built and built[0] == tmp_path / "root" / "data" / "engine"  # overflowed
    holder.release_and_check_last()


# ── Retry-rediscover: a vanished engine gets one ladder re-run ──────


def _embed_launch() -> InstanceLaunch:
    return InstanceLaunch(
        role=WorkerRole.EMBED, argv=["/bin/llama-server"], env_overrides={}, model="m-embed"
    )


def _connection_error() -> Exception:
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    return ProviderError("refused", provider="llama-server", kind=ProviderErrorKind.CONNECTION)


def test_placeable_wanted_drops_roles_the_plan_does_not_place(monkeypatch) -> None:
    # A role dropped by co-placement gets no launch, so it is not wanted; bind
    # then matches a running engine instead of rebuilding it every start. Chat is
    # configured but the plan below places only embed+rerank (chat could not
    # co-tenant), so chat is excluded despite being configured.
    models = {"chat": "m-chat", "embed": "m-embed", "rerank": "m-rerank"}
    monkeypatch.setattr(prov_mod, "_configured_model_for", lambda role: models.get(role.value, ""))
    monkeypatch.setattr(planning_mod, "role_model_placeable", lambda _role, _ref, _vram: True)
    monkeypatch.setattr(
        planning_mod,
        "plan_all_launches",
        lambda: planning_mod.FleetPlan(
            (
                _fake_launch(WorkerRole.EMBED, model="m-embed"),
                _fake_launch(WorkerRole.RERANK, model="m-rerank"),
            )
        ),
    )

    assert prov_mod._placeable_wanted() == {
        (WorkerRole.EMBED, "m-embed"),
        (WorkerRole.RERANK, "m-rerank"),
    }  # chat placed nowhere -> not wanted


def test_placeable_wanted_is_empty_without_an_engine_binary(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    def _no_binary() -> planning_mod.FleetPlan:
        raise ProviderError("no engine binary", kind=ProviderErrorKind.NOT_FOUND)

    monkeypatch.setattr(planning_mod, "plan_all_launches", _no_binary)

    assert prov_mod._placeable_wanted() == set()


def test_placeable_wanted_propagates_a_real_planning_failure(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    def _wedged() -> planning_mod.FleetPlan:
        raise ProviderError("wedged probe", kind=ProviderErrorKind.SERVER)

    monkeypatch.setattr(planning_mod, "plan_all_launches", _wedged)

    with pytest.raises(ProviderError, match="wedged"):
        prov_mod._placeable_wanted()


def test_release_holds_drops_membership_without_stopping_engines(monkeypatch) -> None:
    # The rediscover path drops this process's memberships so it does not count
    # itself a live user and overflow; it must never stop an engine to do so.
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    p = FleetProvider()
    hold = MagicMock()
    p._engine_holds = {Path("/e1"): hold, Path("/e2"): MagicMock()}

    p._release_holds()

    assert p._engine_holds == {}
    hold.release_and_check_last.assert_called_once()
    assert stopped == []


def test_rediscover_releases_holds_before_the_retry(monkeypatch) -> None:
    # A retained self-hold makes the machine slot look in use, overflowing the
    # rebuild to a private engine; rediscover must release it before retrying.
    p = FleetProvider()
    order: list[str] = []
    monkeypatch.setattr(p, "_drop_swap_refs", lambda **_k: order.append("drop"))
    monkeypatch.setattr(p, "_release_holds", lambda: order.append("release"))
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _connection_error()
        order.append("retry")
        return "ok"

    assert p._with_rediscover(_call) == "ok"
    assert order == ["drop", "release", "retry"]  # holds dropped before the retry


def test_rediscover_does_not_close_a_client_a_reader_is_using() -> None:
    # _with_rediscover reaches _drop_swap_refs on any connection blip. A client
    # another thread is mid-embed or mid-stream on must be retired, not closed
    # underneath it; only idle clients close.
    busy, idle = _fake_client(in_flight=1), _fake_client(in_flight=0)
    p = _provider_with_clients({WorkerRole.CHAT: [busy], WorkerRole.EMBED: [idle]})

    p._drop_swap_refs()

    busy.close.assert_not_called()  # still streaming: severing it would kill the read
    assert busy in p._retiring_clients  # kept for a later pass
    idle.close.assert_not_called()  # retired this pass; closes on the next one


def test_connection_failure_rediscovers_once_and_recovers(monkeypatch, tmp_path: Path) -> None:
    _swap, _machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_embed_launch()])
    clients: list[MagicMock] = []

    def _client_factory(_endpoint, _model, **_kw):
        client = _fake_client()
        if not clients:
            client.embed.side_effect = _connection_error()
        else:
            client.embed.return_value = [[1.0]]
        clients.append(client)
        return client

    monkeypatch.setattr(prov_mod, "LlamaServerClient", _client_factory)
    p = FleetProvider()
    assert p.embed(["hello"]) == [[1.0]]
    assert len(clients) == 2  # first pool failed, rediscovery built a second


def test_persistent_connection_failure_raises_after_one_retry(monkeypatch, tmp_path: Path) -> None:
    from lilbee.providers.base import ProviderError

    _swap, _machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_embed_launch()])
    calls: list[int] = []

    def _client_factory(_endpoint, _model, **_kw):
        client = _fake_client()
        client.embed.side_effect = _connection_error()
        calls.append(1)
        return client

    monkeypatch.setattr(prov_mod, "LlamaServerClient", _client_factory)
    p = FleetProvider()
    with pytest.raises(ProviderError):
        p.embed(["hello"])
    assert len(calls) == 2  # exactly one rediscovery, then the error surfaces


def test_non_connection_errors_do_not_rediscover(monkeypatch, tmp_path: Path) -> None:
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    _swap, _machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_embed_launch()])
    calls: list[int] = []

    def _client_factory(_endpoint, _model, **_kw):
        client = _fake_client()
        client.embed.side_effect = ProviderError(
            "boom", provider="llama-server", kind=ProviderErrorKind.SERVER
        )
        calls.append(1)
        return client

    monkeypatch.setattr(prov_mod, "LlamaServerClient", _client_factory)
    p = FleetProvider()
    with pytest.raises(ProviderError):
        p.embed(["hello"])
    assert len(calls) == 1  # no ladder re-run for non-connection failures


def _chat_ladder(monkeypatch, tmp_path: Path, client_factory) -> FleetProvider:
    """A ladder-built provider whose chat clients come from *client_factory*."""
    _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    monkeypatch.setattr(prov_mod, "LlamaServerClient", client_factory)
    return FleetProvider()


def test_chat_transport_failure_rediscovers_once_and_recovers(monkeypatch, tmp_path: Path) -> None:
    """A dead llama-swap proxy (raw ConnectError, no HTTP status) still rediscovers.

    A SIGKILLed proxy surfaces as httpx.ConnectError, not as a ProviderError
    carrying a status; rediscovery must classify both as connection failures.
    """
    import httpx

    clients: list[MagicMock] = []

    def _client_factory(_endpoint, _model, **_kw):
        client = _fake_client()
        if not clients:
            client.chat_result.side_effect = httpx.ConnectError("refused")
        else:
            client.chat_result.return_value = "recovered"
        clients.append(client)
        return client

    p = _chat_ladder(monkeypatch, tmp_path, _client_factory)
    assert p.chat([{"role": "user", "content": "hi"}]) == "recovered"
    assert len(clients) == 2  # first pool failed, rediscovery built a second


def test_chat_stream_open_failure_rediscovers_once_and_streams(monkeypatch, tmp_path: Path) -> None:
    """A stream whose open dies on a dead proxy rediscovers before the first frame."""
    import httpx

    def _dead_stream():
        raise httpx.ConnectError("refused")
        yield  # pragma: no cover - unreachable; makes this a generator

    clients: list[MagicMock] = []

    def _client_factory(_endpoint, _model, **_kw):
        client = _fake_client()
        if not clients:
            client.chat_stream_items.return_value = _dead_stream()
        else:
            client.chat_stream_items.return_value = iter(["a", "b"])
        clients.append(client)
        return client

    p = _chat_ladder(monkeypatch, tmp_path, _client_factory)
    assert list(p.chat([{"role": "user", "content": "hi"}], stream=True)) == ["a", "b"]
    assert len(clients) == 2  # the dead pool was dropped and rebuilt


def test_chat_with_tools_transport_failure_rediscovers_once(monkeypatch, tmp_path: Path) -> None:
    """chat_with_tools gets the same transport-failure rediscovery as chat."""
    import httpx

    clients: list[MagicMock] = []

    def _client_factory(_endpoint, _model, **_kw):
        client = _fake_client()
        if not clients:
            client.chat_tools.side_effect = httpx.ConnectError("refused")
        else:
            client.chat_tools.return_value = "tools-recovered"
        clients.append(client)
        return client

    p = _chat_ladder(monkeypatch, tmp_path, _client_factory)
    assert p.chat_with_tools([{"role": "user", "content": "hi"}], tools=[]) == "tools-recovered"
    assert len(clients) == 2


def test_can_build_engine_false_when_nothing_placeable() -> None:
    assert prov_mod._can_build_engine(set()) is False


def test_can_build_engine_false_when_binary_unresolvable(monkeypatch) -> None:
    # A NOT_FOUND probe failure (no engine binary) is the quiet serve-nothing case.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    def _boom():
        raise ProviderError(
            "no engine binary", provider="llama-server", kind=ProviderErrorKind.NOT_FOUND
        )

    monkeypatch.setattr(planning_mod, "assert_engine_probeable", _boom)
    assert prov_mod._can_build_engine({(WorkerRole.CHAT, "m")}) is False


def test_can_build_engine_true_with_a_placeable_model_and_binary(monkeypatch) -> None:
    monkeypatch.setattr(planning_mod, "assert_engine_probeable", lambda: None)
    assert prov_mod._can_build_engine({(WorkerRole.CHAT, "m")}) is True


def test_can_build_engine_false_when_the_probe_raises_oserror(monkeypatch) -> None:
    # A probe OSError (a device node vanished, a broken pipe) is not a viable
    # build; stand down quietly rather than propagate a raw OS error.
    def _oops() -> None:
        raise OSError("device node gone")

    monkeypatch.setattr(planning_mod, "assert_engine_probeable", _oops)
    assert prov_mod._can_build_engine({(WorkerRole.CHAT, "m")}) is False


def test_can_build_engine_propagates_a_wedged_probe(monkeypatch) -> None:
    # A wedged GPU probe / unusable CUDA runtime (not a missing binary) must fail
    # loud here, before any caller stops a replaceable incumbent.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    def _wedged():
        raise ProviderError(
            "cuda init failed", provider="llama-server", kind=ProviderErrorKind.CONNECTION
        )

    monkeypatch.setattr(planning_mod, "assert_engine_probeable", _wedged)
    with pytest.raises(ProviderError) as excinfo:
        prov_mod._can_build_engine({(WorkerRole.CHAT, "m")})
    assert excinfo.value.kind is ProviderErrorKind.CONNECTION


def test_ladder_does_not_kill_a_replaceable_incumbent_when_the_probe_is_wedged(
    monkeypatch, tmp_path: Path
) -> None:
    """A wedged device probe fails loud BEFORE the stop, sparing the incumbent.

    The build precondition captures the probe, so a wedged GPU probe raises before
    the ladder stops the replaceable incumbent that other members may still hold.
    Were the probe left to _plan_and_spawn (after the stop), the incumbent would be
    killed and the raised error would then skip the overflow, leaving zero engines.
    """
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    real_can_build = prov_mod._can_build_engine  # captured before _install_ladder stubs it
    _swap, machine, _built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    monkeypatch.setattr(prov_mod, "_can_build_engine", real_can_build)  # exercise the real gate
    monkeypatch.setattr(prov_mod, "state_is_healthy", lambda _state: False)  # bind fails
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))

    def _wedged() -> None:
        raise ProviderError(
            "cuda init failed", provider="llama-server", kind=ProviderErrorKind.CONNECTION
        )

    monkeypatch.setattr(planning_mod, "assert_engine_probeable", _wedged)
    # A recorded incumbent with no live users: replaceable, so without the pre-stop
    # gate the ladder would stop it and then hit the wedge in _plan_and_spawn.
    _engine_state_file(machine, "chat", pin="pin-a", model="m-chat", role="chat")
    p = FleetProvider()

    with pytest.raises(ProviderError) as excinfo:
        p._ensure_fleet()

    assert excinfo.value.kind is ProviderErrorKind.CONNECTION  # failed loud
    assert stopped == []  # the incumbent was never stopped


def test_ladder_clears_an_unprobeable_incumbent_before_build(monkeypatch, tmp_path: Path) -> None:
    """A recorded but unprobeable no-user engine is stopped before build, not
    double-built beside. The stop is keyed on the state file, not the probe."""
    _swap, machine, built = _install_ladder(monkeypatch, tmp_path, launches=[_chat_launch()])
    stopped: list[Path] = []
    monkeypatch.setattr(prov_mod, "stop_engine", lambda d: stopped.append(Path(d)))
    monkeypatch.setattr(prov_mod, "state_is_healthy", lambda _state: False)  # unprobeable
    monkeypatch.setattr(
        prov_mod, "_configured_model_for", lambda role: "m-chat" if role is WorkerRole.CHAT else ""
    )
    # A recorded engine with no live users (keep_warm orphan): probe fails but the
    # state file exists, so it must be stopped before the fresh build.
    _engine_state_file(machine, "chat", pin="pin-a", model="m-chat", role="chat")
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert stopped == [machine]  # cleared the recorded incumbent...
    assert built and built[0] == machine  # ...then built fresh in the same slot


def test_shutdown_does_not_hang_on_a_wedged_build(monkeypatch) -> None:
    """A wedged engine start must not make process exit unreachable.

    Every _shut_down check runs after the build lock is taken, so a build that
    never finishes used to hold shutdown forever rather than for a bounded time.
    """
    import threading

    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_BUILD_LOCK_WAIT_S", 0.2)
    released: list[str] = []
    monkeypatch.setattr(p, "_release_engines", lambda **_kw: released.append("released"))

    p._build_lock.acquire()  # stand in for a builder that never returns
    try:
        done = threading.Event()
        threading.Thread(target=lambda: (p.shutdown(), done.set()), daemon=True).start()
        assert done.wait(timeout=10), "shutdown blocked on the build lock"
    finally:
        p._build_lock.release()

    assert released == ["released"]  # teardown ran rather than being skipped
    assert p._shut_down is True  # and the latch is set for any queued thread


def test_shutdown_latches_before_taking_the_build_lock() -> None:
    """A queued warm thread can only bail early if the flag is already set."""
    provider = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    seen: list[bool] = []

    class _SpyLock:
        def __init__(self, inner) -> None:
            self._inner = inner

        def acquire(self, *args, **kwargs):
            # What a warm or reload thread would observe on getting the lock next.
            seen.append(provider._shut_down)
            return self._inner.acquire(*args, **kwargs)

        def release(self) -> None:
            self._inner.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_exc) -> None:
            self.release()

    provider._build_lock = _SpyLock(provider._build_lock)  # type: ignore[assignment]
    provider.shutdown()
    assert seen == [True]

    def test_no_detached_states_means_no_adoption(self, tmp_path) -> None:
        assert FleetProvider()._try_adopt_detached(tmp_path) is False


class TestAReplicaThatCannotLoadLeavesTheRoutingPool:
    """A device that enumerates but cannot allocate fails at model load and
    nowhere else.

    Left healthy, its replica keeps winning the least-in-flight pick, so every
    request goes to the one instance that cannot serve while a sibling on a
    working card sits idle.
    """

    def _clients(self, failing: set[int]):
        from lilbee.providers.fleet.client import LlamaServerClient

        made = []
        for i in range(2):
            client = mock.MagicMock(spec=LlamaServerClient)
            client.healthy = True
            client.index = i
            made.append(client)
        return made, failing

    def test_the_failing_replica_is_marked_unhealthy(self, monkeypatch) -> None:
        from lilbee.providers.fleet import provider as provider_mod

        clients, _ = self._clients({0})

        def _warm(_role, client):
            if client.index == 0:
                raise RuntimeError("failed to allocate on device 0")

        monkeypatch.setattr(provider_mod, "_warm_role", _warm)
        fleet = provider_mod.FleetProvider.__new__(provider_mod.FleetProvider)
        fleet._warm_errors = {}

        assert fleet._warm_role_clients(WorkerRole.EMBED, clients) is True

        clients[0].mark_unhealthy.assert_called_once()
        clients[1].mark_healthy.assert_called_once()
        clients[1].mark_unhealthy.assert_not_called()

    def test_a_replica_that_loads_is_returned_to_the_pool(self, monkeypatch) -> None:
        """A previously-failed replica whose device recovered must route again."""
        from lilbee.providers.fleet import provider as provider_mod

        clients, _ = self._clients(set())
        monkeypatch.setattr(provider_mod, "_warm_role", lambda _r, _c: None)
        fleet = provider_mod.FleetProvider.__new__(provider_mod.FleetProvider)
        fleet._warm_errors = {}

        fleet._warm_role_clients(WorkerRole.EMBED, clients)

        for client in clients:
            client.mark_healthy.assert_called_once()
            client.mark_unhealthy.assert_not_called()

    def test_every_replica_failing_still_reports_the_role_unwarmed(self, monkeypatch) -> None:
        from lilbee.providers.fleet import provider as provider_mod

        clients, _ = self._clients({0, 1})

        def _boom(_role, _client):
            raise RuntimeError("no device could allocate")

        monkeypatch.setattr(provider_mod, "_warm_role", _boom)
        fleet = provider_mod.FleetProvider.__new__(provider_mod.FleetProvider)
        fleet._warm_errors = {}

        assert fleet._warm_role_clients(WorkerRole.EMBED, clients) is False
        assert "no device could allocate" in fleet._warm_errors[WorkerRole.EMBED]
        for client in clients:
            client.mark_unhealthy.assert_called_once()
