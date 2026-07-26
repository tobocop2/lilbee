"""Sizing of the dedicated ingest offload pool.

Extraction (PDF rasterization + OCR) runs on this pool, so its width caps how
many documents rasterize in parallel to feed the GPU OCR/embed slots on a
full-corpus run. OCR ingest is GPU-bound, so these lock in a default capped at
``min(32, cpu_count + 4)`` (a worker sweep showed throughput flat/declining past
~32) that stays operator-overridable via ``LILBEE_INGEST_MAX_WORKERS``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from lilbee.core.config import cfg
from lilbee.data.ingest import offload
from lilbee.providers.fleet import replicas


@pytest.fixture(autouse=True)
def _pin_sizing_inputs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Drop the env override, zero the inflight floor, stub the fleet probe to one replica.

    ``max_workers`` floors its CPU-derived result at ``ingest_max_inflight``, or at the
    probed embed-replica count when that is zero, so leaving either ambient makes these
    assertions depend on what ran earlier. The cache is cleared around each test so the
    pinned inputs are what the cached value reflects.
    """
    monkeypatch.delenv("LILBEE_INGEST_MAX_WORKERS", raising=False)
    monkeypatch.setattr(cfg, "ingest_max_inflight", 0)
    monkeypatch.setattr(replicas, "resolve_replica_count", lambda *_a, **_k: 1)
    offload.max_workers.cache_clear()
    yield
    offload.max_workers.cache_clear()


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


def test_the_ceiling_matches_the_pool_that_actually_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controller sizes its permit ceiling from max_workers() while the pool
    is already running. Re-reading the env per call would report a ceiling the
    live pool does not have -- promising threads that are not there."""
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)
    offload._ingest_executor.cache_clear()
    try:
        pool = offload._ingest_executor()
        monkeypatch.setenv("LILBEE_INGEST_MAX_WORKERS", "200")
        assert offload.max_workers() == pool._max_workers
    finally:
        pool.shutdown(wait=False)
        offload._ingest_executor.cache_clear()


def test_embed_inflight_target_is_zero_when_the_fleet_cannot_be_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fleet may not be resolvable yet (probe raises); admission sizing must
    # fall back to the CPU-bound quota rather than crash the ingest run.
    probed = False

    def _raise(*_args: object, **_kwargs: object) -> int:
        nonlocal probed
        probed = True
        raise RuntimeError("fleet not resolvable yet")

    monkeypatch.setattr(replicas, "resolve_replica_count", _raise)
    assert offload.embed_inflight_target() == 0
    # A single-replica stub also returns 0, so pin that the raising probe ran.
    assert probed
