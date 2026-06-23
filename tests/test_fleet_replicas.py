"""Unit tests for the shared replica-count resolver and cached GPU probe."""

from __future__ import annotations

from lilbee.core.config import cfg
from lilbee.providers.fleet.replicas import gpu_device_count, resolve_replica_count
from lilbee.providers.roles import WorkerRole


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
    gpu_device_count.cache_clear()
    monkeypatch.setattr("lilbee.providers.fleet.binary.resolve_llama_server", lambda: object())
    monkeypatch.setattr(
        "lilbee.providers.fleet.planning.resolve_devices", lambda _b: [object(), object(), object()]
    )
    assert gpu_device_count() == 3


def test_gpu_device_count_floors_at_one_when_no_devices(monkeypatch) -> None:
    gpu_device_count.cache_clear()
    monkeypatch.setattr("lilbee.providers.fleet.binary.resolve_llama_server", lambda: object())
    monkeypatch.setattr("lilbee.providers.fleet.planning.resolve_devices", lambda _b: [])
    assert gpu_device_count() == 1


def test_gpu_device_count_is_cached(monkeypatch) -> None:
    gpu_device_count.cache_clear()
    calls = {"n": 0}

    def _probe(_b: object) -> list[object]:
        calls["n"] += 1
        return [object(), object()]

    monkeypatch.setattr("lilbee.providers.fleet.binary.resolve_llama_server", lambda: object())
    monkeypatch.setattr("lilbee.providers.fleet.planning.resolve_devices", _probe)
    assert gpu_device_count() == 2
    assert gpu_device_count() == 2
    assert calls["n"] == 1  # cached: probed once, not per call
    gpu_device_count.cache_clear()
