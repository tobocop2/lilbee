"""Tests for the chat-model warm-progress tracker and its fraction logic."""

from __future__ import annotations

from lilbee.providers.warm_progress import WarmPhase, WarmProgress, WarmProgressTracker


def test_snapshot_is_none_before_begin() -> None:
    assert WarmProgressTracker().snapshot() is None


def test_reading_reports_true_byte_fraction() -> None:
    tracker = WarmProgressTracker()
    tracker.begin("repo/Model-Q4.gguf")
    tracker.reading(25, 100, detail="shard 1/3")
    snap = tracker.snapshot()
    assert snap is not None
    assert snap.phase is WarmPhase.READING_WEIGHTS
    assert snap.model_ref == "repo/Model-Q4.gguf"
    assert snap.detail == "shard 1/3"
    assert snap.fraction == 0.25
    assert snap.elapsed_s >= 0.0


def test_fraction_is_capped_and_indeterminate_off_read_phase() -> None:
    tracker = WarmProgressTracker()
    tracker.begin("m")
    tracker.reading(150, 100)  # over-count clamps to a full bar, never > 1
    assert tracker.snapshot().fraction == 1.0
    tracker.loading_engine("on 3 GPUs")
    engine = tracker.snapshot()
    assert engine.phase is WarmPhase.LOADING_ENGINE
    assert engine.fraction is None  # no byte signal -> indeterminate
    assert engine.model_ref == "m"  # model carried across phases


def test_ready_and_error_phases() -> None:
    tracker = WarmProgressTracker()
    tracker.begin("m")
    tracker.ready()
    assert tracker.snapshot().phase is WarmPhase.READY
    assert tracker.snapshot().fraction == 1.0
    tracker.fail("out of memory")
    failed = tracker.snapshot()
    assert failed.phase is WarmPhase.ERROR
    assert failed.error == "out of memory"
    assert failed.fraction is None


def test_starting_phase_has_no_fraction() -> None:
    snap = WarmProgress(phase=WarmPhase.STARTING)
    assert snap.fraction is None


def test_zero_total_read_is_indeterminate() -> None:
    # A registry that can't size the shards yields total 0; the bar must not
    # divide by zero and must stay indeterminate.
    snap = WarmProgress(phase=WarmPhase.READING_WEIGHTS, bytes_done=0, bytes_total=0)
    assert snap.fraction is None
