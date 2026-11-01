"""A settings write restarts only the group whose launch changed, and no request sees a gap."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lilbee.app.settings import _reload_changed_roles
from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice
from lilbee.providers.fleet.provider import FleetProvider, _launches_by_group
from lilbee.providers.roles import WorkerRole
from tests.test_fleet_provider import _fake_client, _FakeSwap
from tests.test_fleet_reload_gap import _SignalCondition

_JOIN_S = 10.0
_GB = 1024**3
_EMBED_REF = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
_NOTES_CHUNK_SIZE = 384
_NOTES_CHUNK_OVERLAP = 64
_EMBED_WINDOW_AT_512 = 2048
_EMBED_WINDOW_AT_384 = 1544


@pytest.fixture
def real_planner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The real planner, with fakes only at the engine binary, device probe and GGUF read."""
    files = {_EMBED_REF: tmp_path / "embed.gguf", _CHAT_REF: tmp_path / "chat.gguf"}
    for path in files.values():
        path.write_bytes(b"weights")
    meta = {
        files[_EMBED_REF]: {
            "general.architecture": "nomic-bert",
            "nomic-bert.context_length": "2048",
        },
        files[_CHAT_REF]: {"general.architecture": "qwen3", "qwen3.context_length": "32768"},
    }
    device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
    monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/no/llama-server"))
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref: files[ref])
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata", lambda path: meta[Path(path)]
    )
    monkeypatch.setattr(
        planning_mod,
        "probe_devices",
        lambda _binary: DeviceProbe([device], "Available devices:\n", spoke_protocol=True),
    )
    monkeypatch.setattr(cfg, "embedding_model", _EMBED_REF)
    monkeypatch.setattr(cfg, "chat_model", _CHAT_REF)
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "ocr", "auto")
    monkeypatch.setattr(cfg, "chunk_size", 512)
    monkeypatch.setattr(cfg, "chunk_overlap", 100)
    planning_mod.clear_plan_probe()
    planning_mod.clear_read_device_cache()
    planning_mod.capture_plan_probe()
    yield
    planning_mod.clear_plan_probe()
    planning_mod.clear_read_device_cache()


def _running_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FleetProvider:
    """A provider serving the current real plan, with fake servers and fake clients."""
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _dir, _group: _FakeSwap())
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _dir, **_kw: None)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    provider = FleetProvider()
    monkeypatch.setattr(provider, "_preload_roles", lambda roles=None: None)
    with provider._lock:
        for group, launches in _launches_by_group(planning_mod.plan_all_launches()).items():
            provider._adopt_group(group, _FakeSwap(), list(launches))
            provider._group_dirs[group] = tmp_path
    return provider


def _join_fleet_threads() -> None:
    """Wait for every background thread the fleet started, so a late teardown is seen."""
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and thread.name.startswith("fleet"):
            thread.join(_JOIN_S)
            assert not thread.is_alive()


def _apply_settings_write(monkeypatch, provider: FleetProvider, keys: set[str]) -> None:
    """Run the settings reload path for *keys* against *provider* and wait for it to settle."""
    services = SimpleNamespace(provider=provider, reload_role=provider.reload_role)
    monkeypatch.setattr("lilbee.app.services.peek_services", lambda: services)
    _reload_changed_roles(keys)
    _join_fleet_threads()


def test_an_ocr_change_leaves_every_launch_identical(real_planner) -> None:
    before = planning_mod.plan_all_launches().launches
    cfg.ocr = "off"
    assert planning_mod.plan_all_launches().launches == before


def test_a_chunk_size_change_moves_only_the_embed_window(real_planner) -> None:
    before = {launch.role: launch for launch in planning_mod.plan_all_launches().launches}
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    after = {launch.role: launch for launch in planning_mod.plan_all_launches().launches}
    assert before[WorkerRole.CHAT] == after[WorkerRole.CHAT]
    assert before[WorkerRole.EMBED].ctx == _EMBED_WINDOW_AT_512
    assert after[WorkerRole.EMBED].ctx == _EMBED_WINDOW_AT_384


def test_reload_for_ocr_keeps_the_embed_client_without_a_drop(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider = _running_provider(monkeypatch, tmp_path)
    embed_client = provider._clients[WorkerRole.EMBED][0]
    drop = MagicMock(wraps=provider._drop_group)
    monkeypatch.setattr(provider, "_drop_group", drop)
    cfg.ocr = "off"
    provider.reload_role(WorkerRole.VISION, wait=True)
    assert provider._clients[WorkerRole.EMBED][0] is embed_client
    drop.assert_not_called()


def test_a_preset_write_restarts_embed_and_keeps_chat_serving(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider = _running_provider(monkeypatch, tmp_path)
    chat_client = provider._clients[WorkerRole.CHAT][0]
    embed_client = provider._clients[WorkerRole.EMBED][0]
    teardown = MagicMock(wraps=provider._drop_swap_refs)
    monkeypatch.setattr(provider, "_drop_swap_refs", teardown)
    cfg.ocr = "off"
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    cfg.chunk_overlap = _NOTES_CHUNK_OVERLAP
    _apply_settings_write(monkeypatch, provider, {"chunk_size", "chunk_overlap", "ocr"})
    teardown.assert_not_called()
    assert provider._clients[WorkerRole.CHAT][0] is chat_client
    assert provider._require_clients(WorkerRole.EMBED)[0] is not embed_client


def test_a_chunk_size_write_keeps_the_chat_server(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider = _running_provider(monkeypatch, tmp_path)
    chat_client = provider._clients[WorkerRole.CHAT][0]
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    _apply_settings_write(monkeypatch, provider, {"chunk_size"})
    assert provider._clients[WorkerRole.CHAT][0] is chat_client
    assert provider._role_launches(WorkerRole.EMBED)[0].ctx == _EMBED_WINDOW_AT_384


def test_a_request_during_a_real_embed_restart_gets_the_new_server(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider = _running_provider(monkeypatch, tmp_path)
    gate = _SignalCondition(provider._lock)
    provider._reload_done = gate
    old_client = provider._clients[WorkerRole.EMBED][0]
    starting = threading.Event()
    release = threading.Event()

    class _HeldSwap(_FakeSwap):
        def start(self, *args, **kwargs) -> None:
            starting.set()
            assert release.wait(_JOIN_S)
            super().start(*args, **kwargs)

    monkeypatch.setattr(prov_mod, "SwapManager", lambda _dir, _group: _HeldSwap())
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    provider.reload_role(WorkerRole.VISION)
    assert starting.wait(_JOIN_S)

    results: list[object] = []
    request = threading.Thread(
        target=lambda: results.append(provider._require_clients(WorkerRole.EMBED)), daemon=True
    )
    request.start()
    assert gate.entered.wait(_JOIN_S)
    release.set()
    request.join(_JOIN_S)
    _join_fleet_threads()
    assert len(results) == 1
    assert results[0][0] is not old_client  # type: ignore[index]
