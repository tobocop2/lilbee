"""Sizing of the dedicated ingest offload pool.

Extraction (PDF rasterization + OCR) runs on this pool, so its width caps how
many documents rasterize in parallel to feed the GPU OCR/embed slots on a
full-corpus run. OCR ingest is GPU-bound, so these lock in a default capped at
``min(32, cpu_count + 4)`` (a worker sweep showed throughput flat/declining past
~32) that stays operator-overridable via ``LILBEE_INGEST_MAX_WORKERS``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.core.config import cfg
from lilbee.data import offload
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import replicas

_REAL_RESOLVE_REPLICA_COUNT = replicas.resolve_replica_count


@pytest.fixture(autouse=True)
def _pin_sizing_inputs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Drop the env override, zero the inflight floor, stub the fleet probe to one replica.

    ``max_workers`` floors its CPU-derived result at ``ingest_max_inflight``, or at the
    probed embed-replica count when that is zero, so leaving either ambient makes these
    assertions depend on what ran earlier.
    """
    monkeypatch.delenv("LILBEE_INGEST_MAX_WORKERS", raising=False)
    monkeypatch.setattr(cfg, "ingest_max_inflight", 0)
    monkeypatch.setattr(replicas, "resolve_replica_count", lambda *_a, **_k: 1)
    yield


def test_default_caps_at_32_on_a_big_box(monkeypatch: pytest.MonkeyPatch) -> None:
    # OCR ingest is GPU-bound; a worker sweep showed throughput flat/declining past
    # ~32, so the default caps there instead of scaling to the vCPU count.
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 128)
    assert offload.max_workers() == 32


def test_default_keeps_headroom_on_a_small_box(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)
    assert offload.max_workers() == 12


def test_default_falls_back_when_cpu_count_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offload.os, "cpu_count", lambda: None)
    assert offload.max_workers() == 8


def test_env_override_wins_over_the_capped_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("LILBEE_INGEST_MAX_WORKERS", "200")
    assert offload.max_workers() == 200


@pytest.mark.parametrize("bad", ["nan", "0", "-4", "  "])
def test_invalid_env_override_is_ignored(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("LILBEE_INGEST_MAX_WORKERS", bad)
    assert offload.max_workers() == 12


async def test_the_ceiling_matches_the_pool_that_actually_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controller sizes its permit ceiling from max_workers() while the run's pool
    is already running. Re-reading the env per call would report a ceiling the
    live pool does not have -- promising threads that are not there."""
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)

    @offload.owns_ingest_pool
    async def run() -> tuple[int, int]:
        pool = offload._run_pool.get()
        assert pool is not None
        monkeypatch.setenv("LILBEE_INGEST_MAX_WORKERS", "200")
        return offload.max_workers(), pool._max_workers

    ceiling, pool_size = await run()
    assert ceiling == pool_size == 12


def test_the_shared_pool_is_sized_without_probing_the_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared pool is built on the event loop, so its size must not wait on a probe."""
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)
    monkeypatch.delenv("LILBEE_INGEST_MAX_WORKERS", raising=False)
    probes: list[str] = []
    monkeypatch.setattr(offload, "fleet_inflight_target", lambda: probes.append("probe") or 64)
    assert offload._ingest_executor()._max_workers == 12
    assert probes == []


def test_the_shared_pool_keeps_the_worker_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LILBEE_INGEST_MAX_WORKERS", "5")
    assert offload._ingest_executor()._max_workers == 5


def test_embed_inflight_target_is_zero_when_the_device_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed = False

    def _wedged(*_args: object, **_kwargs: object) -> int:
        nonlocal probed
        probed = True
        raise ProviderError("The device probe did not respond within 30 s")

    monkeypatch.setattr(replicas, "resolve_replica_count", _wedged)
    assert offload.embed_inflight_target() == 0
    # A single-replica stub also returns 0, so pin that the raising probe ran.
    assert probed


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


def _fleet_of_two_cards_after_a_failed_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> _Clock:
    """Wire the real sizing path to a device probe that fails once, then finds two cards."""
    from lilbee.app.services import set_services
    from lilbee.providers.fleet import planning
    from lilbee.providers.fleet.devices import FleetDevice

    binary = tmp_path / "llama-server"
    binary.write_bytes(b"engine")
    engine = SimpleNamespace(binary=lambda: binary)
    services = SimpleNamespace(
        engines=SimpleNamespace(for_role=lambda _role: SimpleNamespace(adapter=engine))
    )
    set_services(services)
    monkeypatch.setattr(replicas, "resolve_replica_count", _REAL_RESOLVE_REPLICA_COUNT)
    monkeypatch.setattr(cfg, "embed_replicas", 0)
    cards = [FleetDevice("CUDA", i, "gpu", 24 * 1024**3, 23 * 1024**3) for i in range(2)]
    outcomes: list[object] = [ProviderError("the device probe did not respond"), cards]

    def _probe(_binary: Path, _engine: object) -> planning.DeviceReading:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return planning.DeviceReading(list(outcome), planning.EngineBackend.UNKNOWN)

    clock = _Clock()
    monkeypatch.setattr(planning, "_read_devices", _probe)
    monkeypatch.setattr(planning, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(
        planning,
        "_read_device_cache",
        planning._ReadDeviceCache(
            planning._DEVICE_PROBE_TTL_S, planning._DEVICE_PROBE_FAILURE_TTL_S
        ),
    )
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)
    return clock


def test_a_failed_probe_gives_the_cpu_default_only_until_the_failure_window_ends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lilbee.providers.fleet import planning

    clock = _fleet_of_two_cards_after_a_failed_probe(monkeypatch, tmp_path)
    assert offload.max_workers() == 12
    clock.now += planning._DEVICE_PROBE_FAILURE_TTL_S - 1
    assert offload.max_workers() == 12
    clock.now += 2
    assert offload.max_workers() == 16


def test_a_recovered_sizing_is_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clock = _fleet_of_two_cards_after_a_failed_probe(monkeypatch, tmp_path)
    assert offload.max_workers() == 12
    clock.now += 61
    assert offload.max_workers() == 16
    clock.now += 600  # a later probe would find a failure, but the sizing is kept
    assert offload.max_workers() == 16


async def test_a_run_keeps_the_pool_size_it_was_built_with(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A probe that recovers during a run does not lift the ceiling above the live pool."""
    clock = _fleet_of_two_cards_after_a_failed_probe(monkeypatch, tmp_path)

    @offload.owns_ingest_pool
    async def run() -> int:
        clock.now += 61
        return offload.max_workers()

    assert await run() == 12
    assert offload.max_workers() == 16


def test_embed_inflight_target_raises_an_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _bug(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("replica knob is not a number")

    monkeypatch.setattr(replicas, "resolve_replica_count", _bug)
    with pytest.raises(RuntimeError, match="replica knob is not a number"):
        offload.embed_inflight_target()


def test_the_pool_ceiling_raises_when_the_fleet_probe_has_a_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _bug(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("replica knob is not a number")

    monkeypatch.setattr(replicas, "resolve_replica_count", _bug)
    with pytest.raises(RuntimeError, match="replica knob is not a number"):
        offload.max_workers()


async def test_a_fleet_probe_bug_reaches_the_user_as_a_sync_error_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.server import handlers

    def _bug(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("replica knob is not a number")

    monkeypatch.setattr(replicas, "resolve_replica_count", _bug)
    events = [event async for event in handlers.sync_stream()]
    errors = [event for event in events if event.startswith("event: error")]
    assert len(errors) == 1
    assert "replica knob is not a number" in errors[0]
    assert not [event for event in events if event.startswith("event: done")]
