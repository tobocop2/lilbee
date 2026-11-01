"""No process stops a group that has a live user other than itself.

The plan, the ladder, the contract, the reload pass and the user locks are real. Only the
process layer and the health probe are fake. A peer is a user lock under another pid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.providers import engine_params, gguf_meta
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet import swap_manager as swap_mod
from lilbee.providers.fleet.binary import engine_pin
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.provider import FleetProvider, _launches_by_group
from lilbee.providers.fleet.swap_manager import SwapManager, SwapState
from lilbee.providers.roles import WorkerRole
from lilbee.runtime.engine_lock import hold_user_lock, live_users_exist
from tests.test_fleet_provider import _fake_client, _FakeSwap
from tests.test_fleet_reload_plan import (
    _NOTES_CHUNK_SIZE,
    _apply_settings_write,
    real_planner,
)

__all__ = ["real_planner"]

_PEER_PID = 4_000_001
_RERANK_REF = "BAAI/rr-GGUF/rr-Q8_0.gguf"
_SLOT = "slot"
_PRIVATE = "private"


class _Box:
    """The engine dirs of one machine: what each dir records, and each stop."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "config-root"
        self.dirs = {name: tmp_path / name for name in (_SLOT, _PRIVATE)}
        for path in self.dirs.values():
            path.mkdir()
        self.slot = self.dirs[_SLOT]
        self.private = self.dirs[_PRIVATE]
        self.records: dict[str, dict[SwapGroup, SwapState]] = {_SLOT: {}, _PRIVATE: {}}
        self.stopped: list[str] = []
        self.next_pid = 100

    def record(self, engine_dir: Path, group: SwapGroup, launches: list[InstanceLaunch]) -> None:
        """Write the state a real start writes, under a new process id."""
        self.next_pid += 1
        self.records[engine_dir.name][group] = SwapState(
            pid=self.next_pid,
            pgid=1,
            launches=tuple(launch.to_state() for launch in launches),
            engine_pin=engine_pin(),
        )

    def seed_fleet(self, engine_dir: Path) -> dict[SwapGroup, int]:
        """Start the current plan in *engine_dir* as another process; return its pids."""
        for group, launches in _launches_by_group(planning_mod.plan_all_launches()).items():
            self.record(engine_dir, group, list(launches))
        return self.pids(engine_dir)

    def pids(self, engine_dir: Path) -> dict[SwapGroup, int]:
        return {group: state.pid for group, state in self.records[engine_dir.name].items()}

    def stop_engine(self, engine_dir: Path) -> list[str]:
        self.stopped.append(engine_dir.name)
        self.records[engine_dir.name] = {}
        return []


@pytest.fixture
def box(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Box:
    machine = _Box(tmp_path)

    class _ProcessSwap(_FakeSwap):
        def __init__(self, engine_dir: Path, group: SwapGroup) -> None:
            super().__init__()
            self._dir = engine_dir
            self._group = group

        def bind(self, _state: SwapState) -> bool:
            self.bound = True
            return True

        def start(self, launches: list[InstanceLaunch], **kwargs: object) -> None:
            super().start(launches, **kwargs)
            machine.record(self._dir, self._group, launches)

        def shutdown(self) -> None:
            super().shutdown()
            if not self.bound:
                machine.records[self._dir.name].pop(self._group, None)

    monkeypatch.setattr(prov_mod, "machine_engine_dir", lambda: machine.slot)
    monkeypatch.setattr(prov_mod, "private_engine_dir", lambda _root: machine.private)
    monkeypatch.setattr(prov_mod, "kernel_arbitrates_locks", lambda _dir: True)
    monkeypatch.setattr(prov_mod, "_watch_bound_engine", lambda _dir: None)
    monkeypatch.setattr(prov_mod, "_can_build_engine", lambda _wanted: True)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    monkeypatch.setattr(prov_mod, "SwapManager", _ProcessSwap)
    monkeypatch.setattr(prov_mod, "stop_engine", machine.stop_engine)
    monkeypatch.setattr(prov_mod, "engine_record_exists", lambda d: bool(machine.records[d.name]))
    monkeypatch.setattr(prov_mod, "_healthy_states", lambda d: dict(machine.records[d.name]))
    monkeypatch.setattr(cfg, "keep_engine_warm", False)
    return machine


def _pick_reranker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Configure a reranker that the real planner places."""
    resolve = engine_params.resolve_model_path
    read = gguf_meta.read_gguf_metadata
    weights = tmp_path / "rr.gguf"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(
        engine_params,
        "resolve_model_path",
        lambda ref: weights if ref == _RERANK_REF else resolve(ref),
    )
    monkeypatch.setattr(
        gguf_meta,
        "read_gguf_metadata",
        lambda path: (
            {"general.architecture": "bert", "bert.context_length": "512"}
            if Path(path) == weights
            else read(path)
        ),
    )
    monkeypatch.setattr(cfg, "reranker_model", _RERANK_REF)


def _process(monkeypatch: pytest.MonkeyPatch) -> FleetProvider:
    provider = FleetProvider()
    monkeypatch.setattr(provider, "_preload_roles", lambda roles=None: None)
    return provider


def _assert_serves_all_from_private(provider: FleetProvider, box: _Box) -> None:
    """The process owns a private fleet with a client for each planned role."""
    assert set(provider._group_dirs.values()) == {box.private}
    assert not any(swap.bound for swap in provider._swaps.values())
    assert list(provider._engine_holds) == [box.private]
    planned = {launch.role for launch in planning_mod.plan_all_launches().launches}
    assert set(provider._clients) == planned
    assert all(provider._clients[role] for role in planned)
    assert set(box.records[_PRIVATE]) == set(provider._swaps)


def test_a_process_that_adds_a_role_leaves_a_live_peer_running(
    real_planner, box: _Box, monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    peer_pids = box.seed_fleet(box.slot)
    peer_hold = hold_user_lock(box.slot, pid=_PEER_PID)
    _pick_reranker(monkeypatch, tmp_path)
    joiner = _process(monkeypatch)
    with caplog.at_level(logging.INFO, logger=prov_mod.log.name):
        assert joiner._acquire_engine(box.root) is True
    assert f"The engine at {box.slot} has a live user" in caplog.text
    assert box.stopped == []
    assert box.pids(box.slot) == peer_pids
    assert WorkerRole.RERANK in joiner._clients
    _assert_serves_all_from_private(joiner, box)
    joiner.shutdown()
    assert peer_hold.release_and_check_last() is True


def test_a_setting_write_that_adds_a_role_leaves_a_live_peer_running(
    real_planner, box: _Box, monkeypatch, tmp_path: Path
) -> None:
    peer_pids = box.seed_fleet(box.slot)
    peer_hold = hold_user_lock(box.slot, pid=_PEER_PID)
    joiner = _process(monkeypatch)
    assert joiner._acquire_engine(box.root) is True
    assert all(swap.bound for swap in joiner._swaps.values())
    _pick_reranker(monkeypatch, tmp_path)
    _apply_settings_write(monkeypatch, joiner, {"reranker_model"})
    assert box.stopped == []
    assert box.pids(box.slot) == peer_pids
    assert WorkerRole.RERANK in joiner._clients
    _assert_serves_all_from_private(joiner, box)
    joiner.shutdown()
    assert peer_hold.release_and_check_last() is True


def test_a_dead_group_beside_a_live_peer_leaves_the_healthy_groups_running(
    real_planner, box: _Box, monkeypatch
) -> None:
    box.seed_fleet(box.slot)
    del box.records[_SLOT][SwapGroup.CHAT]
    peer_pids = box.pids(box.slot)
    peer_hold = hold_user_lock(box.slot, pid=_PEER_PID)
    joiner = _process(monkeypatch)
    assert joiner._acquire_engine(box.root) is True
    assert box.stopped == []
    assert box.pids(box.slot) == peer_pids
    _assert_serves_all_from_private(joiner, box)
    joiner.shutdown()
    assert peer_hold.release_and_check_last() is True


def test_a_full_private_dir_with_a_live_user_serves_nothing_and_stops_nothing(
    real_planner, box: _Box, monkeypatch, tmp_path: Path
) -> None:
    slot_pids = box.seed_fleet(box.slot)
    private_pids = box.seed_fleet(box.private)
    holds = [hold_user_lock(d, pid=_PEER_PID) for d in (box.slot, box.private)]
    _pick_reranker(monkeypatch, tmp_path)
    joiner = _process(monkeypatch)
    assert joiner._acquire_engine(box.root) is False
    assert box.stopped == []
    assert (box.pids(box.slot), box.pids(box.private)) == (slot_pids, private_pids)
    _assert_every_request_names_the_engines_in_use(joiner)
    for hold in holds:
        hold.release_and_check_last()


def test_a_dead_group_in_a_private_dir_with_a_live_user_names_the_cause(
    real_planner, box: _Box, monkeypatch, tmp_path: Path
) -> None:
    box.seed_fleet(box.slot)
    box.seed_fleet(box.private)
    del box.records[_PRIVATE][SwapGroup.CHAT]
    holds = [hold_user_lock(d, pid=_PEER_PID) for d in (box.slot, box.private)]
    _pick_reranker(monkeypatch, tmp_path)
    joiner = _process(monkeypatch)
    assert joiner._acquire_engine(box.root) is False
    assert box.stopped == []
    _assert_every_request_names_the_engines_in_use(joiner)
    for hold in holds:
        hold.release_and_check_last()


def test_a_later_acquisition_that_succeeds_drops_the_engines_in_use_message(
    real_planner, box: _Box, monkeypatch, tmp_path: Path
) -> None:
    box.seed_fleet(box.slot)
    box.seed_fleet(box.private)
    holds = [hold_user_lock(d, pid=_PEER_PID) for d in (box.slot, box.private)]
    _pick_reranker(monkeypatch, tmp_path)
    joiner = _process(monkeypatch)
    assert joiner._acquire_engine(box.root) is False
    assert joiner._engine_refusal == prov_mod._ENGINES_IN_USE_MESSAGE
    for hold in holds:
        hold.release_and_check_last()
    assert joiner._acquire_engine(box.root) is True
    assert joiner._engine_refusal is None
    joiner.shutdown()


def _assert_every_request_names_the_engines_in_use(joiner: FleetProvider) -> None:
    """Each role's request fails with the engine-in-use cause, never the install line."""
    message = prov_mod._ENGINES_IN_USE_MESSAGE
    assert "LILBEE_ENGINE_DIR" in message
    for role in (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK):
        with pytest.raises(ProviderError) as raised:
            joiner._require_clients(role)
        assert str(raised.value) == message
        assert "installed and configured" not in str(raised.value)


def test_with_no_live_user_a_new_process_replaces_the_stale_fleet(
    real_planner, box: _Box, monkeypatch, tmp_path: Path
) -> None:
    stale_pids = box.seed_fleet(box.slot)
    _pick_reranker(monkeypatch, tmp_path)
    joiner = _process(monkeypatch)
    assert joiner._acquire_engine(box.root) is True
    assert box.stopped == [_SLOT]
    assert set(joiner._group_dirs.values()) == {box.slot}
    assert not set(box.pids(box.slot).values()) & set(stale_pids.values())
    assert SwapGroup.RERANK in box.records[_SLOT]
    assert box.records[_PRIVATE] == {}
    joiner.shutdown()


def test_an_overflowed_process_that_writes_a_setting_leaves_the_owner_running(
    real_planner, box: _Box, monkeypatch, tmp_path: Path
) -> None:
    peer_pids = box.seed_fleet(box.slot)
    peer_hold = hold_user_lock(box.slot, pid=_PEER_PID)
    _pick_reranker(monkeypatch, tmp_path)
    joiner = _process(monkeypatch)
    assert joiner._acquire_engine(box.root) is True
    moved = box.pids(box.private)[SwapGroup.EMBED]
    kept = box.pids(box.private)[SwapGroup.CHAT]
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    _apply_settings_write(monkeypatch, joiner, {"chunk_size"})
    assert box.stopped == []
    assert box.pids(box.slot) == peer_pids
    assert box.pids(box.private)[SwapGroup.EMBED] != moved
    assert box.pids(box.private)[SwapGroup.CHAT] == kept
    joiner.shutdown()
    assert box.pids(box.slot) == peer_pids
    assert peer_hold.release_and_check_last() is True


def test_an_evicted_owner_that_writes_a_setting_leaves_the_new_owner_running(
    real_planner, box: _Box, monkeypatch
) -> None:
    evicted = _process(monkeypatch)
    assert evicted._acquire_engine(box.root) is True
    assert set(evicted._group_dirs.values()) == {box.slot}
    # Another process took the slot, as after `lilbee engine stop` and a new build.
    new_owner_pids = box.seed_fleet(box.slot)
    new_owner_hold = hold_user_lock(box.slot, pid=_PEER_PID)
    new_owner_chat = box.records[_SLOT][SwapGroup.CHAT]
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    _apply_settings_write(monkeypatch, evicted, {"chunk_size"})
    assert box.stopped == []
    assert box.records[_SLOT][SwapGroup.CHAT] is new_owner_chat
    assert box.pids(box.slot)[SwapGroup.CHAT] == new_owner_pids[SwapGroup.CHAT]
    assert live_users_exist(box.slot) is True
    evicted._engine_holds = {}
    assert new_owner_hold.release_and_check_last() is True


def test_a_start_kills_only_the_swap_at_this_process_config(
    box: _Box, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kill by config path cannot reach a group that another process started."""
    own_config = SwapManager(box.slot, SwapGroup.EMBED)._config_path
    other_config = box.slot / swap_mod._config_filename(_PEER_PID, SwapGroup.EMBED.value)
    assert own_config != other_config
    own = MagicMock()
    own.cmdline.return_value = ["llama-swap", "-config", str(own_config)]
    other = MagicMock()
    other.cmdline.return_value = ["llama-swap", "-config", str(other_config)]
    monkeypatch.setattr(swap_mod, "_processes_named", lambda _name: iter([own, other]))
    assert swap_mod._swaps_for_config(own_config) == [own]


def test_a_subset_process_binds_and_its_exit_leaves_the_owner_running(
    real_planner, box: _Box, monkeypatch, tmp_path: Path
) -> None:
    _pick_reranker(monkeypatch, tmp_path)
    owner_pids = box.seed_fleet(box.slot)
    assert SwapGroup.RERANK in owner_pids
    owner_hold = hold_user_lock(box.slot, pid=_PEER_PID)
    monkeypatch.setattr(cfg, "reranker_model", "")
    subset = _process(monkeypatch)
    assert subset._acquire_engine(box.root) is True
    assert all(swap.bound for swap in subset._swaps.values())
    assert WorkerRole.RERANK not in subset._clients
    assert list(subset._engine_holds) == [box.slot]
    subset.shutdown()
    assert box.stopped == []
    assert box.pids(box.slot) == owner_pids
    assert owner_hold.release_and_check_last() is True


def test_a_subset_process_that_leaves_last_stops_the_engine(
    real_planner, box: _Box, monkeypatch, tmp_path: Path
) -> None:
    _pick_reranker(monkeypatch, tmp_path)
    box.seed_fleet(box.slot)
    monkeypatch.setattr(cfg, "reranker_model", "")
    subset = _process(monkeypatch)
    assert subset._acquire_engine(box.root) is True
    subset.shutdown()
    assert box.stopped == [_SLOT]


def _warming_process(monkeypatch: pytest.MonkeyPatch, chat_ref: str) -> FleetProvider:
    """A process whose chat warm has started, with every local role refused."""
    provider = FleetProvider()
    monkeypatch.setattr(
        prov_mod,
        "_configured_model_for",
        lambda role: chat_ref if role is WorkerRole.CHAT else "",
    )
    monkeypatch.setattr(prov_mod, "_unusable_engine_reason", lambda: None)
    provider._engine_refusal = prov_mod._ENGINES_IN_USE_MESSAGE
    provider._warm_tracker.begin(chat_ref)
    return provider


def test_a_remote_chat_warm_ignores_the_engines_in_use_refusal(monkeypatch) -> None:
    provider = _warming_process(monkeypatch, "openai/gpt-x")
    provider._finalize_warm_if_chat_never_ran()
    assert provider._warm_tracker.snapshot() is None


def test_a_local_chat_warm_names_the_engines_in_use_refusal(monkeypatch) -> None:
    from lilbee.providers.warm_progress import WarmPhase

    provider = _warming_process(monkeypatch, "Org/chat-GGUF/chat-Q4_K_M.gguf")
    provider._finalize_warm_if_chat_never_ran()
    snapshot = provider._warm_tracker.snapshot()
    assert snapshot is not None
    assert snapshot.phase is WarmPhase.ERROR
    assert snapshot.error == prov_mod._ENGINES_IN_USE_MESSAGE


def test_a_remote_chat_model_still_refuses_an_embed_request_with_the_cause(
    real_planner, box: _Box, monkeypatch, tmp_path: Path
) -> None:
    box.seed_fleet(box.slot)
    box.seed_fleet(box.private)
    holds = [hold_user_lock(d, pid=_PEER_PID) for d in (box.slot, box.private)]
    _pick_reranker(monkeypatch, tmp_path)
    monkeypatch.setattr(cfg, "chat_model", "openai/gpt-x")
    joiner = _process(monkeypatch)
    assert joiner._acquire_engine(box.root) is False
    with pytest.raises(ProviderError) as raised:
        joiner._require_clients(WorkerRole.EMBED)
    assert str(raised.value) == prov_mod._ENGINES_IN_USE_MESSAGE
    for hold in holds:
        hold.release_and_check_last()
