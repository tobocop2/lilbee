"""Sizing of the dedicated ingest offload pool.

Extraction (PDF rasterization + OCR) runs on this pool, so its width caps how
many documents rasterize in parallel to feed the GPU OCR/embed slots on a
full-corpus run. OCR ingest is GPU-bound, so these lock in a default capped at
``min(32, cpu_count + 4)`` (a worker sweep showed throughput flat/declining past
~32) that stays operator-overridable via ``LILBEE_INGEST_MAX_WORKERS``.
"""

from __future__ import annotations

import pytest

from lilbee.data.ingest import offload


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LILBEE_INGEST_MAX_WORKERS", raising=False)


def test_default_caps_at_32_on_a_big_box(monkeypatch: pytest.MonkeyPatch) -> None:
    # OCR ingest is GPU-bound; a worker sweep showed throughput flat/declining past
    # ~32, so the default caps there instead of scaling to the vCPU count.
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 128)
    assert offload._max_workers() == 32


def test_default_keeps_headroom_on_a_small_box(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)
    assert offload._max_workers() == 12


def test_default_falls_back_when_cpu_count_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offload.os, "cpu_count", lambda: None)
    assert offload._max_workers() == 8


def test_env_override_wins_over_the_capped_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("LILBEE_INGEST_MAX_WORKERS", "200")
    assert offload._max_workers() == 200


@pytest.mark.parametrize("bad", ["nan", "0", "-4", "  "])
def test_invalid_env_override_is_ignored(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("LILBEE_INGEST_MAX_WORKERS", bad)
    assert offload._max_workers() == 12
