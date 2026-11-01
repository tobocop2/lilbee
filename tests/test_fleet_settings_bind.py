"""A setting write leaves every running group bindable by a new process on the new config.

The plan, the launch diff, the contract and the bind ladder are real. Only the process and
the health probe are fake.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.app.settings import _reload_changed_roles
from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers import engine_params, gguf_meta
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.binary import engine_pin
from lilbee.providers.fleet.contract import contract_matches
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.provider import FleetProvider, _launches_by_group
from lilbee.providers.fleet.swap_manager import SwapState
from lilbee.providers.roles import RerankMode, WorkerRole
from tests.test_fleet_provider import _fake_client, _FakeSwap
from tests.test_fleet_reload_plan import (
    _CHAT_REF,
    _NOTES_CHUNK_OVERLAP,
    _NOTES_CHUNK_SIZE,
    _apply_settings_write,
    _join_fleet_threads,
    real_planner,
)

__all__ = ["real_planner"]

_OTHER_CHAT_REF = "Qwen/Qwen3-1B-GGUF/Qwen3-1B-Q8_0.gguf"
_VISION_REF = "ggml-org/vision-GGUF/vision-Q4_K_M.gguf"
_States = dict[SwapGroup, SwapState]


def _record(states: _States, group: SwapGroup, launches: list[InstanceLaunch]) -> None:
    """Write the state a real start writes: the launches and the current engine pin."""
    states[group] = SwapState(
        pid=len(states) + 1,
        pgid=1,
        launches=tuple(launch.to_state() for launch in launches),
        engine_pin=engine_pin(),
    )


def _owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[FleetProvider, _States]:
    """A provider that owns the fleet; each start records state as the real manager does."""
    states: _States = {}

    class _RecordingSwap(_FakeSwap):
        def __init__(self, group: SwapGroup) -> None:
            super().__init__()
            self._group = group

        def start(self, launches: list[InstanceLaunch], **kwargs: object) -> None:
            super().start(launches, **kwargs)
            _record(states, self._group, launches)

    monkeypatch.setattr(prov_mod, "SwapManager", lambda _dir, group: _RecordingSwap(group))
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _dir, **_kw: None)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    provider = FleetProvider()
    monkeypatch.setattr(provider, "_preload_roles", lambda roles=None: None)
    monkeypatch.setattr(provider, "_reload_dir", lambda: tmp_path)
    with provider._lock:
        for group, launches in _launches_by_group(planning_mod.plan_all_launches()).items():
            provider._adopt_group(group, _FakeSwap(), list(launches))
            provider._group_dirs[group] = tmp_path
            _record(states, group, list(launches))
    return provider, states


def _new_process_ladder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, states: _States
) -> tuple[bool | None, FleetProvider]:
    """Run the real ladder as a new process on the current config, with live users present."""
    monkeypatch.setattr(prov_mod, "_healthy_states", lambda _dir: dict(states))
    monkeypatch.setattr(prov_mod, "live_users_exist", lambda _dir: True)
    monkeypatch.setattr(prov_mod, "_watch_bound_engine", lambda _dir: None)
    monkeypatch.setattr(prov_mod, "_can_build_engine", lambda _wanted: True)
    monkeypatch.setattr(prov_mod, "stop_engine", lambda _dir: pytest.fail("stopped the fleet"))

    class _BindingSwap(_FakeSwap):
        def bind(self, _state: SwapState) -> bool:
            self.bound = True
            return True

    monkeypatch.setattr(prov_mod, "SwapManager", lambda _dir, _group: _BindingSwap())
    peer = FleetProvider()
    monkeypatch.setattr(peer, "_hold_membership", lambda _dir: None)
    monkeypatch.setattr(peer, "_plan_and_spawn", lambda _dir: pytest.fail("built a second fleet"))
    result = peer._acquire_in_dir(
        tmp_path, engine_pin(), prov_mod._placeable_demand(), is_overflow=False
    )
    return result, peer


def _assert_new_process_binds_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, states: _States
) -> None:
    result, peer = _new_process_ladder(monkeypatch, tmp_path, states)
    assert result is True
    assert set(peer._swaps) == set(states)
    assert all(swap.bound for swap in peer._swaps.values())


def test_a_new_process_binds_with_no_write(real_planner, monkeypatch, tmp_path: Path) -> None:
    _provider, states = _owner(monkeypatch, tmp_path)
    _assert_new_process_binds_all(monkeypatch, tmp_path, states)


def test_a_new_process_binds_after_a_chunk_size_write(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider, states = _owner(monkeypatch, tmp_path)
    kept_chat = states[SwapGroup.CHAT]
    moved_embed = states[SwapGroup.EMBED]
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    _apply_settings_write(monkeypatch, provider, {"chunk_size"})
    assert states[SwapGroup.CHAT] is kept_chat
    assert states[SwapGroup.EMBED] is not moved_embed
    _assert_new_process_binds_all(monkeypatch, tmp_path, states)


def test_a_new_process_binds_after_a_profile_apply(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider, states = _owner(monkeypatch, tmp_path)
    kept_chat = states[SwapGroup.CHAT]
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    cfg.chunk_overlap = _NOTES_CHUNK_OVERLAP
    cfg.ocr = "off"
    _apply_settings_write(monkeypatch, provider, {"chunk_size", "chunk_overlap", "ocr"})
    assert states[SwapGroup.CHAT] is kept_chat
    _assert_new_process_binds_all(monkeypatch, tmp_path, states)


def test_a_new_process_binds_after_a_model_role_write(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    resolve = engine_params.resolve_model_path
    chat_path = resolve(_CHAT_REF)
    monkeypatch.setattr(
        engine_params,
        "resolve_model_path",
        lambda ref: chat_path if ref == _OTHER_CHAT_REF else resolve(ref),
    )
    provider, states = _owner(monkeypatch, tmp_path)
    kept_embed = states[SwapGroup.EMBED]
    moved_chat = states[SwapGroup.CHAT]
    cfg.chat_model = _OTHER_CHAT_REF
    provider.reload_role(WorkerRole.CHAT, wait=True)
    _join_fleet_threads()
    assert states[SwapGroup.EMBED] is kept_embed
    assert states[SwapGroup.CHAT] is not moved_chat
    _assert_new_process_binds_all(monkeypatch, tmp_path, states)


@pytest.mark.parametrize(
    ("key", "value"),
    [("flash_attention", True), ("cpu_moe", True), ("reranker_type", "llm")],
)
def test_a_setting_no_launch_uses_keeps_every_group_and_the_bind(
    real_planner, monkeypatch, tmp_path: Path, key: str, value: object
) -> None:
    provider, states = _owner(monkeypatch, tmp_path)
    before = dict(states)
    monkeypatch.setattr(cfg, key, value)
    _apply_settings_write(monkeypatch, provider, {key})
    assert states == before
    _assert_new_process_binds_all(monkeypatch, tmp_path, states)


def test_a_setting_a_launch_uses_restarts_only_that_group(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider, states = _owner(monkeypatch, tmp_path)
    kept_embed = states[SwapGroup.EMBED]
    moved_chat = states[SwapGroup.CHAT]
    monkeypatch.setattr(cfg, "flash_attention", False)
    _apply_settings_write(monkeypatch, provider, {"flash_attention"})
    assert states[SwapGroup.EMBED] is kept_embed
    assert states[SwapGroup.CHAT] is not moved_chat
    _assert_new_process_binds_all(monkeypatch, tmp_path, states)


def test_a_peer_on_the_old_chunk_size_never_binds_the_moved_window(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider, states = _owner(monkeypatch, tmp_path)
    old_chunk_size = cfg.chunk_size
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    _apply_settings_write(monkeypatch, provider, {"chunk_size"})
    cfg.chunk_size = old_chunk_size
    result, peer = _new_process_ladder(monkeypatch, tmp_path, states)
    assert result is None
    assert not peer._swaps


def test_a_new_build_never_binds_a_kept_group(real_planner, monkeypatch, tmp_path: Path) -> None:
    provider, states = _owner(monkeypatch, tmp_path)
    monkeypatch.setattr("lilbee.providers.fleet.binary.engine_build_id", lambda: "upgraded")
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    _apply_settings_write(monkeypatch, provider, {"chunk_size"})
    result, _peer = _new_process_ladder(monkeypatch, tmp_path, states)
    assert result is None


def test_an_ocr_change_with_a_vision_model_stops_vision_and_keeps_the_rest_bindable(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    resolve = engine_params.resolve_model_path
    read = gguf_meta.read_gguf_metadata
    vision_path = tmp_path / "vision.gguf"
    vision_path.write_bytes(b"weights")
    monkeypatch.setattr(
        engine_params,
        "resolve_model_path",
        lambda ref: vision_path if ref == _VISION_REF else resolve(ref),
    )
    monkeypatch.setattr(
        gguf_meta,
        "read_gguf_metadata",
        lambda path: {"general.architecture": "clip"} if Path(path) == vision_path else read(path),
    )
    monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _ref: vision_path)
    monkeypatch.setattr(cfg, "vision_model", _VISION_REF)
    provider, states = _owner(monkeypatch, tmp_path)
    assert SwapGroup.VISION in states
    kept = {group: states[group] for group in (SwapGroup.CHAT, SwapGroup.EMBED)}
    cfg.ocr = "off"
    _apply_settings_write(monkeypatch, provider, {"ocr"})
    assert SwapGroup.VISION not in provider._swaps
    del states[SwapGroup.VISION]
    assert states == kept
    _assert_new_process_binds_all(monkeypatch, tmp_path, states)


@pytest.mark.parametrize(
    ("key", "value", "moved", "kept"),
    [
        ("chunk_size", 126, WorkerRole.EMBED, WorkerRole.CHAT),
        ("chunk_size", 126, WorkerRole.RERANK, WorkerRole.VISION),
        ("flash_attention", False, WorkerRole.CHAT, WorkerRole.EMBED),
        ("flash_attention", False, WorkerRole.VISION, WorkerRole.RERANK),
        ("kv_cache_type", KvCacheType.F16, WorkerRole.CHAT, WorkerRole.VISION),
        ("placement", '{"chat": {"devices": [1]}}', WorkerRole.CHAT, WorkerRole.EMBED),
        ("n_gpu_layers", 10, WorkerRole.CHAT, WorkerRole.EMBED),
    ],
)
def test_a_setting_moves_only_the_settings_of_the_roles_that_use_it(
    monkeypatch, key: str, value: object, moved: WorkerRole, kept: WorkerRole
) -> None:
    before = {role: planning_mod.launch_settings(role, None, None) for role in (moved, kept)}
    monkeypatch.setattr(cfg, key, value)
    assert planning_mod.launch_settings(moved, None, None) != before[moved]
    assert planning_mod.launch_settings(kept, None, None) == before[kept]


def test_the_rerank_mode_moves_the_settings() -> None:
    role = WorkerRole.RERANK
    assert planning_mod.launch_settings(role, RerankMode.LLM, None) != (
        planning_mod.launch_settings(role, None, None)
    )


@pytest.mark.parametrize("role", list(WorkerRole))
def test_gpu_devices_moves_every_role_and_ctx_sizing_moves_none(monkeypatch, role) -> None:
    before = planning_mod.launch_settings(role, None, None)
    monkeypatch.setattr(cfg, "num_ctx", 4096)
    monkeypatch.setattr(cfg, "num_ctx_max", 16384)
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 49152)
    assert planning_mod.launch_settings(role, None, None) == before
    monkeypatch.setattr(cfg, "gpu_devices", "1")
    assert planning_mod.launch_settings(role, None, None) != before


def test_expert_offload_moves_the_settings_only_for_a_moe_model(monkeypatch) -> None:
    moe = {"expert_count": "8"}
    before_plain = planning_mod.launch_settings(WorkerRole.CHAT, None, None)
    before_moe = planning_mod.launch_settings(WorkerRole.CHAT, None, moe)
    monkeypatch.setattr(cfg, "cpu_moe", True)
    assert planning_mod.launch_settings(WorkerRole.CHAT, None, None) == before_plain
    assert planning_mod.launch_settings(WorkerRole.CHAT, None, moe) != before_moe


def test_an_llm_reranker_carries_flash_attention(monkeypatch) -> None:
    before = planning_mod.launch_settings(WorkerRole.RERANK, RerankMode.LLM, None)
    pooled = planning_mod.launch_settings(WorkerRole.RERANK, RerankMode.CROSS_ENCODER, None)
    monkeypatch.setattr(cfg, "flash_attention", False)
    assert planning_mod.launch_settings(WorkerRole.RERANK, RerankMode.LLM, None) != before
    assert planning_mod.launch_settings(WorkerRole.RERANK, RerankMode.CROSS_ENCODER, None) == pooled


def test_the_contract_refuses_a_launch_with_other_settings() -> None:
    launch = InstanceLaunch(WorkerRole.CHAT, ["x"], {}, "m", settings="a")
    state = SwapState(pid=1, pgid=1, launches=(launch.to_state(),), engine_pin="pin")
    assert contract_matches(state, (), "pin", {WorkerRole.CHAT: "a"})
    assert not contract_matches(state, (), "pin", {WorkerRole.CHAT: "b"})
    assert contract_matches(state, (), "pin", {WorkerRole.EMBED: "b"})


def test_the_restart_log_names_each_launch_field_that_moved() -> None:
    before = InstanceLaunch(WorkerRole.EMBED, ["x"], {}, "m", ctx=2048)
    after = InstanceLaunch(WorkerRole.EMBED, ["x"], {}, "m", ctx=1536)
    assert prov_mod._launch_changes((before,), (after,)) == "ctx 2048 -> 1536"
    assert prov_mod._launch_changes((before,), (before,)) == "no launch field moved"
    assert prov_mod._launch_changes((), (after,)) == "replicas 0 -> 1"


@pytest.mark.parametrize(
    ("keys", "roles", "replans"),
    [
        ({"chunk_size"}, 0, 1),
        ({"chunk_size", "num_ctx", "flash_attention"}, 0, 1),
        ({"chunk_size", "chat_model"}, 1, 0),
    ],
)
def test_a_settings_write_runs_one_reload(monkeypatch, keys, roles: int, replans: int) -> None:
    services = MagicMock()
    monkeypatch.setattr("lilbee.app.services.peek_services", lambda: services)
    _reload_changed_roles(keys)
    assert services.reload_role.call_count == roles
    assert services.provider.replan_fleet.call_count == replans


def test_a_thread_that_cannot_start_leaves_no_reload_in_flight(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider, _states = _owner(monkeypatch, tmp_path)

    def refuse(_thread: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    with pytest.raises(RuntimeError, match="start new thread"):
        provider.replan_fleet()
    assert provider._reloading is False
    assert provider._reload_owed == frozenset()
