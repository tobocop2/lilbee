"""The ingest admission is built from the sizing result, in both modes.

The sizing gives a baseline (the static slot count), the fleet devices and the pool's
worker count, which is the adaptive ceiling. Each figure must reach the piece that
uses it, and the figures are chosen apart so a swapped one shows.
"""

from __future__ import annotations

import asyncio
import logging
import types
from collections.abc import Sequence
from typing import ClassVar

import pytest

from lilbee.data.ingest import pipeline
from lilbee.data.ingest.adaptive import ResizableGate

BASELINE = 3
CEILING = 7
CARD = object()


class _RecordingController:
    """An adaptive controller that records its arguments and ends at once."""

    built: ClassVar[list[_RecordingController]] = []

    def __init__(self, gate: ResizableGate, profile: object, sampler: object, *_a: object, **kw):
        self.gate = gate
        self.profile = profile
        self.sampler = sampler
        self.permit_min = kw["permit_min"]
        self.permit_max = kw["permit_max"]
        self.built.append(self)

    async def run(self) -> None:
        return None


@pytest.fixture
def adaptive(monkeypatch: pytest.MonkeyPatch) -> list[Sequence[object]]:
    """Switch the adaptive profile on; return the device lists the sampler factory received."""
    sampled: list[Sequence[object]] = []
    _RecordingController.built = []
    monkeypatch.setattr(
        pipeline, "profile_for", lambda _mode: types.SimpleNamespace(name="adaptive")
    )
    monkeypatch.setattr(
        pipeline,
        "make_signal_sampler",
        lambda devices: sampled.append(list(devices)) or (lambda: None),
    )
    monkeypatch.setattr(pipeline, "AdaptiveController", _RecordingController)
    return sampled


async def _slots(semaphore: asyncio.Semaphore) -> int:
    """How many holders the semaphore admits before it blocks."""
    held = 0
    while not semaphore.locked():
        await semaphore.acquire()
        held += 1
    return held


async def test_the_adaptive_gate_starts_at_the_baseline_below_the_ceiling(adaptive) -> None:
    gate, _window, task = pipeline._build_admission(BASELINE, [0], [CARD], CEILING)
    assert isinstance(gate, ResizableGate)
    assert gate.limit == BASELINE
    assert task is not None
    await task


async def test_the_adaptive_gate_starts_at_the_ceiling_below_the_baseline(adaptive) -> None:
    gate, _window, task = pipeline._build_admission(CEILING + 2, [0], [CARD], CEILING)
    assert isinstance(gate, ResizableGate)
    assert gate.limit == CEILING
    assert task is not None
    await task


async def test_the_adaptive_window_is_twice_the_ceiling(adaptive) -> None:
    _gate, window, task = pipeline._build_admission(BASELINE, [0], [CARD], CEILING)
    assert window == CEILING * 2
    assert task is not None
    await task


async def test_the_controller_gets_the_ceiling_the_gate_and_the_devices(adaptive) -> None:
    pages = [0]
    gate, _window, task = pipeline._build_admission(BASELINE, pages, [CARD], CEILING)
    (controller,) = _RecordingController.built
    assert controller.permit_max == CEILING
    assert controller.permit_min == 1
    assert controller.gate is gate
    assert adaptive == [[CARD]]
    assert task is not None
    await task


async def test_the_adaptive_start_and_ceiling_are_logged(
    adaptive, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=pipeline.log.name):
        _gate, _window, task = pipeline._build_admission(BASELINE, [0], [CARD], CEILING)
    assert task is not None
    await task
    assert [record.getMessage() for record in caplog.records] == [
        f"Adaptive ingest concurrency (adaptive): start {BASELINE}, max {CEILING}"
    ]


async def test_a_profile_with_no_device_keeps_the_static_semaphore(adaptive) -> None:
    semaphore, window, task = pipeline._build_admission(BASELINE, [0], [], CEILING)
    assert isinstance(semaphore, asyncio.Semaphore)
    assert await _slots(semaphore) == BASELINE
    assert window == BASELINE * 2
    assert task is None
    assert _RecordingController.built == []


async def test_a_device_with_no_profile_keeps_the_static_semaphore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "profile_for", lambda _mode: None)
    semaphore, window, task = pipeline._build_admission(BASELINE, [0], [CARD], CEILING)
    assert isinstance(semaphore, asyncio.Semaphore)
    assert await _slots(semaphore) == BASELINE
    assert window == BASELINE * 2
    assert task is None


def test_the_sizing_returns_the_baseline_the_fleet_devices_and_the_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[object] = []
    engine = object()
    services = types.SimpleNamespace(
        engines=types.SimpleNamespace(for_role=lambda _role: types.SimpleNamespace(adapter=engine))
    )
    monkeypatch.setattr(pipeline, "get_services", lambda: services)
    monkeypatch.setattr(pipeline, "profile_for", lambda _mode: object())
    monkeypatch.setattr(
        pipeline, "enumerate_fleet_devices", lambda adapter: asked.append(adapter) or [CARD]
    )
    monkeypatch.setattr(pipeline, "_max_concurrent", lambda: BASELINE)
    monkeypatch.setattr(pipeline, "max_workers", lambda: CEILING)
    assert pipeline._admission_sizing() == (BASELINE, [CARD], CEILING)
    assert asked == [engine]


def test_the_sizing_without_a_profile_lists_no_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "profile_for", lambda _mode: None)
    monkeypatch.setattr(pipeline, "_max_concurrent", lambda: BASELINE)
    monkeypatch.setattr(pipeline, "max_workers", lambda: CEILING)
    assert pipeline._admission_sizing() == (BASELINE, [], CEILING)
