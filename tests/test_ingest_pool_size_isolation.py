"""Tests do not share the cached ingest pool size.

The two tests run in file order in one worker. The second expects a CPU count
the first did not use, so it fails when the first test's result stays cached.
"""

from __future__ import annotations

import pytest

from lilbee.data import offload


@pytest.fixture(autouse=True)
def _pin_other_sizing_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LILBEE_INGEST_MAX_WORKERS", raising=False)
    monkeypatch.setattr(offload, "embed_inflight_target", lambda: 0)


def test_a_small_box_sizes_the_pool_from_its_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 8)
    assert offload.max_workers() == 12


def test_a_big_box_after_a_small_one_sizes_the_pool_from_its_own_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 128)
    assert offload.max_workers() == 32
