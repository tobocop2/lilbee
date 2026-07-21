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

from lilbee.data.ingest import offload


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("LILBEE_INGEST_MAX_WORKERS", raising=False)
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
