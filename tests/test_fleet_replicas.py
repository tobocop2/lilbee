"""Unit tests for the shared replica-count resolver and cached GPU probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import devices
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.planning import DeviceReading, clear_read_device_cache
from lilbee.providers.fleet.replicas import gpu_device_count, resolve_replica_count
from lilbee.providers.roles import EngineBackend, WorkerRole


def _card(index: int) -> FleetDevice:
    return FleetDevice("CUDA", index, f"card {index}", 8 << 30, 8 << 30)


def test_auto_resolves_to_one_replica_per_gpu(monkeypatch) -> None:
    # 0 (the default) means auto: one replica per enumerated GPU.
    monkeypatch.setattr(cfg, "embed_replicas", 0)
    monkeypatch.setattr(cfg, "vision_replicas", 0)
    assert resolve_replica_count(WorkerRole.EMBED, 4) == 4
    assert resolve_replica_count(WorkerRole.VISION, 3) == 3


def test_auto_floors_at_one_replica_when_gpu_less(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "embed_replicas", 0)
    monkeypatch.setattr(cfg, "vision_replicas", 0)
    assert resolve_replica_count(WorkerRole.EMBED, 0) == 1
    assert resolve_replica_count(WorkerRole.VISION, 0) == 1


def test_explicit_replica_count_wins_over_device_count(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "embed_replicas", 2)
    monkeypatch.setattr(cfg, "vision_replicas", 5)
    assert resolve_replica_count(WorkerRole.EMBED, 8) == 2
    assert resolve_replica_count(WorkerRole.VISION, 1) == 5


def test_non_replicated_roles_always_run_one(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "embed_replicas", 0)
    monkeypatch.setattr(cfg, "vision_replicas", 0)
    assert resolve_replica_count(WorkerRole.CHAT, 8) == 1
    assert resolve_replica_count(WorkerRole.RERANK, 8) == 1


def test_gpu_device_count_returns_probe_length(monkeypatch) -> None:
    clear_read_device_cache()
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server", lambda: object()
    )
    monkeypatch.setattr(
        "lilbee.providers.fleet.planning.resolve_devices",
        lambda _b, engine: [_card(0), _card(1), _card(2)],
    )
    assert gpu_device_count(LLAMA) == 3


def test_gpu_device_count_floors_at_one_when_no_devices(monkeypatch) -> None:
    clear_read_device_cache()
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server", lambda: object()
    )
    monkeypatch.setattr("lilbee.providers.fleet.planning.resolve_devices", lambda _b, engine: [])
    assert gpu_device_count(LLAMA) == 1


def test_gpu_device_count_falls_back_when_engine_missing(monkeypatch) -> None:
    """No engine binary resolves auto to one replica instead of raising, so the
    ingest concurrency sizing degrades gracefully on an engine-less host."""
    from lilbee.providers.base import ProviderError

    clear_read_device_cache()

    def _raise() -> object:
        raise ProviderError("llama-server binary not found")

    monkeypatch.setattr("lilbee.providers.fleet.engines.llama.resolve_llama_server", _raise)
    assert gpu_device_count(LLAMA) == 1
    clear_read_device_cache()


def test_gpu_device_count_is_cached(monkeypatch, tmp_path: Path) -> None:
    clear_read_device_cache()
    calls = {"n": 0}

    def _probe(_b: object, engine) -> DeviceReading:
        calls["n"] += 1
        return DeviceReading(devices=[_card(0), _card(1)], backend=EngineBackend.CUDA)

    binary = tmp_path / "llama-server"
    binary.write_bytes(b"launcher")
    monkeypatch.setattr("lilbee.providers.fleet.engines.llama.resolve_llama_server", lambda: binary)
    monkeypatch.setattr("lilbee.providers.fleet.planning._read_devices", _probe)
    assert gpu_device_count(LLAMA) == 2
    assert gpu_device_count(LLAMA) == 2
    assert calls["n"] == 1  # cached: probed once, not per call
    clear_read_device_cache()


def test_gpu_device_count_sizes_a_gpu_less_host_when_the_probe_hits_a_missing_dll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe's own missing-DLL error, not a stand-in, reaches the catch that sizes the host."""
    clear_read_device_cache()
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server",
        lambda: Path("/bin/llama-server"),
    )
    monkeypatch.setattr(devices, "_run_list_devices", lambda _binary, _timeout: ("", 0xC0000135))
    assert gpu_device_count(LLAMA) == 1
    clear_read_device_cache()
