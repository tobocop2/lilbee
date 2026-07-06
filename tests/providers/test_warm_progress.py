"""Tests for the chat-model warm-progress tracker."""

from __future__ import annotations

from lilbee.providers.warm_progress import WarmPhase, WarmProgress, WarmProgressTracker


def test_snapshot_is_none_before_begin() -> None:
    assert WarmProgressTracker().snapshot() is None


def test_reading_reports_byte_progress() -> None:
    tracker = WarmProgressTracker()
    tracker.begin("repo/Model-Q4.gguf")
    tracker.reading(25, 100, detail="shard 1/3")
    snap = tracker.snapshot()
    assert snap is not None
    assert snap.phase is WarmPhase.READING_WEIGHTS
    assert snap.model_ref == "repo/Model-Q4.gguf"
    assert snap.bytes_done == 25
    assert snap.bytes_total == 100
    assert snap.detail == "shard 1/3"
    assert snap.elapsed_s >= 0.0


def test_phase_transitions_carry_model_ref() -> None:
    tracker = WarmProgressTracker()
    tracker.begin("m")
    tracker.reading(150, 100)
    tracker.loading_engine("on 3 GPUs")
    engine = tracker.snapshot()
    assert engine.phase is WarmPhase.LOADING_ENGINE
    assert engine.detail == "on 3 GPUs"
    assert engine.model_ref == "m"  # model carried across phases


def test_ready_and_error_phases() -> None:
    tracker = WarmProgressTracker()
    tracker.begin("m")
    tracker.ready()
    assert tracker.snapshot().phase is WarmPhase.READY
    tracker.fail("out of memory")
    failed = tracker.snapshot()
    assert failed.phase is WarmPhase.ERROR
    assert failed.error == "out of memory"


def test_progress_model_defaults() -> None:
    snap = WarmProgress(phase=WarmPhase.STARTING)
    assert snap.model_ref is None
    assert snap.bytes_done == 0
    assert snap.bytes_total == 0
    assert snap.error is None
