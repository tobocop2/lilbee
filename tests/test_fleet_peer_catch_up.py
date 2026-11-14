"""A peer on a new config value waits for the slot's builder to rebuild rather than overflow.

The owner of the machine slot has written config.toml but not yet rebuilt. A peer of the
same config root takes the new value first. The plan, the ladder, the reload pass and the
cross-process build lock (a file lock under tmp_path) are real; the processes and the
health probe are the fakes of test_fleet_settings_bind.py. The order is set with Events.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lilbee.app import settings as appset
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.base import EngineAdapter
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.provider import FleetProvider, _launches_by_group
from lilbee.providers.fleet.swap_manager import PlanOrigin, SwapState
from lilbee.providers.reload_keys import FLEET_RELOAD_KEYS
from lilbee.runtime.engine_lock import private_engine_dir
from tests._fleet_engines import engines_for_all_roles
from tests.test_fleet_provider import _FakeSwap
from tests.test_fleet_reload_plan import real_planner
from tests.test_fleet_settings_bind import (
    _call_failing_once_then_binding,
    _peer_mid_call_over_an_owner_restart,
    _record,
)

__all__ = ["real_planner"]
_WAIT_S = 10.0
_States = dict[SwapGroup, SwapState]


@dataclasses.dataclass
class _Gap:
    peer: FleetProvider
    owner: FleetProvider
    states: _States
    built: list[Path]
    waits: threading.Event
    old: _States


def _gap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, root: str | None = None) -> _Gap:
    """Owner and peer on flash_attention on; the slot records the origin of that file."""
    persistent_settings.update_values(cfg.data_root, {"flash_attention": True})
    peer, states, built = _peer_mid_call_over_an_owner_restart(
        monkeypatch, tmp_path, flash_attention=True
    )
    origin = prov_mod._plan_origin()
    if root is not None:
        origin = dataclasses.replace(origin, config_root=root)
    for group, state in list(states.items()):
        states[group] = dataclasses.replace(state, origin=origin)

    class _Swap(_FakeSwap):
        def __init__(self, group: SwapGroup) -> None:
            super().__init__()
            self._group = group

        def start(self, launches, **kwargs) -> None:
            super().start(launches, **kwargs)
            _record(states, self._group, launches)
            states[self._group] = dataclasses.replace(states[self._group], origin=self.origin)

        def bind(self, _state: SwapState) -> bool:
            self.bound = True
            return True

    monkeypatch.setattr(prov_mod, "SwapManager", lambda _dir, group: _Swap(group))
    owner = FleetProvider()
    monkeypatch.setattr(owner, "_preload_roles", lambda roles=None: None)
    monkeypatch.setattr(owner, "_reload_dir", lambda: tmp_path)
    with owner._lock:
        for group, launches in _launches_by_group(
            planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA))
        ).items():
            owner._adopt_group(group, _FakeSwap(), list(launches))
            owner._group_dirs[group] = tmp_path
    waits = threading.Event()
    real_wait = prov_mod._await_slot_change

    def noted_wait(engine_dir: Path, before: _States, deadline: float) -> None:
        waits.set()
        real_wait(engine_dir, before, deadline)

    monkeypatch.setattr(prov_mod, "_await_slot_change", noted_wait)
    monkeypatch.setattr(prov_mod, "_OWNER_CATCH_UP_POLL_S", 0.001)
    monkeypatch.setattr(prov_mod, "_OWNER_CATCH_UP_WAIT_S", _WAIT_S)
    return _Gap(peer, owner, states, built, waits, dict(states))


def _write_flash_attention_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The owner's write: config.toml and the shared in-process cfg take the new value."""
    persistent_settings.update_values(cfg.data_root, {"flash_attention": False})
    monkeypatch.setattr(cfg, "flash_attention", False)


def _peer_takes_the_value(monkeypatch: pytest.MonkeyPatch, peer: FleetProvider, path: str) -> None:
    if path == "retry":
        monkeypatch.setattr(peer, "_config_refresh", lambda: None, raising=False)
        assert peer._with_rediscover(_call_failing_once_then_binding(peer)) is True
    else:
        peer.replan_fleet(wait=True)


def _owner_held_until(monkeypatch: pytest.MonkeyPatch, release: threading.Event) -> threading.Event:
    """Hold the owner's reload inside its own build lock, before the slot's build lock."""
    held = threading.Event()
    refresh = planning_mod.refresh_plan_devices

    def gated_refresh(engine: EngineAdapter) -> None:
        if threading.current_thread().name == "owner-reload":
            held.set()
            release.wait(_WAIT_S)
        refresh(engine)

    monkeypatch.setattr(planning_mod, "refresh_plan_devices", gated_refresh)
    return held


def _start_owner(gap: _Gap) -> threading.Thread:
    thread = threading.Thread(
        target=gap.owner.replan_fleet, kwargs={"wait": True}, name="owner-reload"
    )
    thread.start()
    return thread


def _assert_peer_bound_to_the_rebuilt_slot(gap: _Gap) -> None:
    assert gap.built == []
    assert gap.states != gap.old
    assert set(gap.peer._swaps) == set(gap.states)
    assert all(swap.bound for swap in gap.peer._swaps.values())


@pytest.mark.parametrize("path", ["retry", "follow"])
def test_a_peer_in_the_gap_waits_for_the_owner_and_binds(
    real_planner, monkeypatch, tmp_path: Path, path: str
) -> None:
    gap = _gap(monkeypatch, tmp_path)
    held = _owner_held_until(monkeypatch, gap.waits)
    _write_flash_attention_off(monkeypatch)
    owner_thread = _start_owner(gap)
    assert held.wait(_WAIT_S)
    try:
        _peer_takes_the_value(monkeypatch, gap.peer, path)
    finally:
        gap.waits.set()
        owner_thread.join(_WAIT_S)
    assert not owner_thread.is_alive()
    _assert_peer_bound_to_the_rebuilt_slot(gap)


@pytest.mark.parametrize("path", ["retry", "follow"])
def test_control_when_the_owner_builds_first_the_peer_binds_with_no_wait(
    real_planner, monkeypatch, tmp_path: Path, path: str
) -> None:
    gap = _gap(monkeypatch, tmp_path)
    _write_flash_attention_off(monkeypatch)
    gap.owner.replan_fleet(wait=True)
    _peer_takes_the_value(monkeypatch, gap.peer, path)
    assert not gap.waits.is_set()
    _assert_peer_bound_to_the_rebuilt_slot(gap)


def test_control_an_owner_that_dies_in_the_wait_has_its_slot_replaced_at_once(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    gap = _gap(monkeypatch, tmp_path)
    monkeypatch.setattr(prov_mod, "_OWNER_CATCH_UP_WAIT_S", 3600.0)
    monkeypatch.setattr(
        prov_mod,
        "live_users_exist",
        lambda d, **_kw: Path(d) == tmp_path and not gap.waits.is_set(),
    )
    _write_flash_attention_off(monkeypatch)
    _peer_takes_the_value(monkeypatch, gap.peer, "follow")
    assert gap.waits.is_set()
    assert gap.built == [tmp_path]  # replaced in place; no private fleet


def test_control_an_owner_that_never_takes_the_change_lets_the_peer_overflow_after_the_bound(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    gap = _gap(monkeypatch, tmp_path)
    monkeypatch.setattr(prov_mod, "_OWNER_CATCH_UP_WAIT_S", 0.02)
    _write_flash_attention_off(monkeypatch)
    _peer_takes_the_value(monkeypatch, gap.peer, "follow")
    assert gap.waits.is_set()
    assert gap.built == [private_engine_dir(cfg.data_root)]


def _another_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Gap:
    gap = _gap(monkeypatch, tmp_path, root=str(tmp_path / "another-root"))
    _write_flash_attention_off(monkeypatch)
    return gap


def _peer_is_stale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Gap:
    gap = _gap(monkeypatch, tmp_path)
    monkeypatch.setattr(cfg, "flash_attention", False)  # in memory only; the file stays on
    return gap


def _unreadable_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Gap:
    gap = _gap(monkeypatch, tmp_path)
    _write_flash_attention_off(monkeypatch)
    monkeypatch.setattr(prov_mod, "read_persisted", lambda _root: (None, None))
    return gap


@pytest.mark.parametrize("setup", [_another_root, _peer_is_stale, _unreadable_file])
def test_control_a_peer_with_no_owed_rebuild_overflows_at_once(
    real_planner, monkeypatch, tmp_path: Path, setup: Callable[..., _Gap]
) -> None:
    gap = setup(monkeypatch, tmp_path)
    _peer_takes_the_value(monkeypatch, gap.peer, "follow")
    assert not gap.waits.is_set()
    assert gap.built == [private_engine_dir(cfg.data_root)]


def _bind_fails_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """The peer plans flash_attention off, so the slot's fleet is not bindable by it."""
    monkeypatch.setattr(cfg, "flash_attention", False)


def test_a_placement_write_leaves_a_same_root_peer_with_no_wait(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    """An owner that adopts a placement write does not rebuild, so the peer must not wait."""
    assert "placement" not in FLEET_RELOAD_KEYS
    gap = _gap(monkeypatch, tmp_path)
    before = prov_mod._plan_origin()
    persistent_settings.update_values(cfg.data_root, {"placement": '{"chat": {"devices": [0]}}'})
    _bind_fails_in_memory(monkeypatch)
    assert prov_mod._plan_origin() == before
    _peer_takes_the_value(monkeypatch, gap.peer, "follow")
    assert not gap.waits.is_set()
    assert gap.built == [private_engine_dir(cfg.data_root)]


# One valid value per reload key, unlike the built-in; a new reload key fails the walk below.
_RELOAD_KEY_VALUES: dict[str, object] = {
    "num_ctx": 4096,
    "num_ctx_max": 8192,
    "chat_n_ctx_target": 6144,
    "kv_cache_type": "f16",
    "n_gpu_layers": 7,
    "cpu_moe": True,
    "n_cpu_moe": 3,
    "flash_attention": False,
    "chat_model": "org/Walk-GGUF/walk-chat-Q4_K_M.gguf",
    "embedding_model": "org/Walk-GGUF/walk-embed-Q4_K_M.gguf",
    "vision_model": "org/Walk-GGUF/walk-vision-Q4_K_M.gguf",
    "reranker_model": "org/Walk-GGUF/walk-rerank-Q4_K_M.gguf",
    "reranker_type": "llm",
    "chunk_size": 256,
    "ocr": "off",
}
_OTHER_KEY_VALUES: dict[str, object] = {"top_k": 3, "placement": '{"chat": {"devices": [0]}}'}


def _owner_rebuilds_on(monkeypatch: pytest.MonkeyPatch, key: str) -> bool:
    """Whether an owner that adopts a change to *key* reloads a role or replans its fleet."""
    services = SimpleNamespace(provider=MagicMock(), reload_role=MagicMock())
    monkeypatch.setattr("lilbee.app.services.peek_services", lambda: services)
    monkeypatch.setattr("lilbee.data.extract.backends.sync_xberg_backend", lambda *_a: None)
    appset._invalidate_caches({key})
    return services.reload_role.called or services.provider.replan_fleet.called


@pytest.mark.parametrize("key", [*sorted(FLEET_RELOAD_KEYS), *_OTHER_KEY_VALUES])
def test_a_key_is_in_the_plan_view_iff_a_change_makes_the_owner_rebuild(
    monkeypatch, key: str
) -> None:
    assert _RELOAD_KEY_VALUES.keys() == FLEET_RELOAD_KEYS
    before = prov_mod._plan_origin()
    value = (_RELOAD_KEY_VALUES | _OTHER_KEY_VALUES)[key]
    persistent_settings.update_values(cfg.data_root, {key: value})
    assert key in (persistent_settings.persisted_values(cfg.data_root) or {})  # the write took
    view_moved = prov_mod._plan_origin() != before
    rebuilds = _owner_rebuilds_on(monkeypatch, key)
    assert view_moved is rebuilds is (key in FLEET_RELOAD_KEYS)


def test_a_write_of_a_key_no_plan_reads_leaves_the_peer_with_no_wait(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    gap = _gap(monkeypatch, tmp_path)
    before = prov_mod._plan_origin()
    persistent_settings.update_values(cfg.data_root, {"top_k": 3})
    _bind_fails_in_memory(monkeypatch)
    assert prov_mod._plan_origin() == before
    _peer_takes_the_value(monkeypatch, gap.peer, "follow")
    assert not gap.waits.is_set()
    assert gap.built == [private_engine_dir(cfg.data_root)]


def test_one_bind_attempt_waits_within_one_bound_over_both_dirs(
    monkeypatch, tmp_path: Path
) -> None:
    """The slot and the private dir both owe a rebuild and no owner follows: one bound, not two."""
    clock = [0.0]

    def advance(seconds: float) -> None:
        clock[0] += seconds

    fake_time = SimpleNamespace(monotonic=lambda: clock[0], sleep=advance)
    monkeypatch.setattr(prov_mod, "time", fake_time)
    monkeypatch.setattr(prov_mod, "_OWNER_CATCH_UP_WAIT_S", 30.0)
    monkeypatch.setattr(prov_mod, "_OWNER_CATCH_UP_POLL_S", 0.5)
    monkeypatch.setattr(prov_mod, "engine_pin", lambda: "pin")
    monkeypatch.setattr(
        prov_mod,
        "_placeable_demand",
        lambda _engine, _engines: SimpleNamespace(
            skipped_not_installed={}, skipped_unusable_ctx={}
        ),
    )
    monkeypatch.setattr(prov_mod, "machine_engine_dir", lambda: tmp_path / "slot")
    monkeypatch.setattr(prov_mod, "kernel_arbitrates_locks", lambda _dir: True)
    monkeypatch.setattr(prov_mod, "live_users_exist", lambda _dir, **_kw: True)
    monkeypatch.setattr(prov_mod, "_healthy_states", lambda _dir: {})
    peer = FleetProvider()
    passes: list[tuple[Path, bool]] = []

    def owed_pass(engine_dir: Path, *_a, may_wait: bool, **_kw) -> prov_mod._SlotOwed | None:
        passes.append((engine_dir, may_wait))
        return prov_mod._SlotOwed({}) if may_wait else None

    monkeypatch.setattr(peer, "_acquire_in_dir_locked", owed_pass)
    assert peer._acquire_engine(cfg.data_root) is False
    assert clock[0] == 30.0
    private = private_engine_dir(cfg.data_root)
    assert passes == [(tmp_path / "slot", True), (tmp_path / "slot", False), (private, False)]


def test_a_reload_that_keeps_every_group_restamps_them_with_the_file(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    gap = _gap(monkeypatch, tmp_path)
    kept = dict(gap.owner._swaps)
    # A plan key the reload finds no launch change for: only the file moves.
    persistent_settings.update_values(cfg.data_root, {"kv_cache_type": "f16"})
    gap.owner.replan_fleet(wait=True)
    now = prov_mod._plan_origin()
    assert gap.owner._swaps == kept  # nothing restarted
    assert all(swap.origin == now for swap in kept.values())
    assert isinstance(now, PlanOrigin) and '"kv_cache_type": "f16"' in (now.fleet_view or "")


def test_a_restamp_that_fails_to_write_does_not_fail_the_reload(
    real_planner, monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gap = _gap(monkeypatch, tmp_path)
    kept = dict(gap.owner._swaps)
    failures: list[str] = []
    for swap in kept.values():
        real_restamp = swap.restamp

        def restamp_failing_once(origin: PlanOrigin, _real: Callable = real_restamp) -> None:
            if not failures:
                failures.append("denied")
                raise PermissionError("state file is open in a peer")
            _real(origin)

        monkeypatch.setattr(swap, "restamp", restamp_failing_once)
    persistent_settings.update_values(cfg.data_root, {"kv_cache_type": "f16"})
    with caplog.at_level("WARNING", logger=prov_mod.log.name):
        gap.owner.replan_fleet(wait=True)
    assert failures == ["denied"]
    assert gap.owner._swaps == kept  # the reload finished and restarted nothing
    assert "Could not record the new config" in caplog.text
    persistent_settings.update_values(cfg.data_root, {"kv_cache_type": "q8_0"})
    gap.owner.replan_fleet(wait=True)
    now = prov_mod._plan_origin()
    assert all(swap.origin == now for swap in kept.values())
    assert '"kv_cache_type": "q8_0"' in (now.fleet_view or "")
