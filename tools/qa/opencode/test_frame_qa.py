"""Unit tests for the frame-QA timeline assertions (no tesseract / PNGs needed)."""

from __future__ import annotations

from frame_qa import validate_texts


def _checks(report) -> dict[str, bool]:
    return {c.name: c.ok for c in report.checks}


def _stream(text: str, start: int, count: int, step: int = 2) -> list[tuple[int, str]]:
    """A run of distinct streaming frames (token index keeps each frame unique)."""
    return [(start + i * step, f"{text} tok{i}") for i in range(count)]


def _good_small_timeline() -> list[tuple[int, str]]:
    # A healthy reel: prompt typed, tool dispatched, grounded answer streamed.
    return [
        (0, "lilbee launch opencode"),
        (4, "search the indexed godot 4 class reference for the astargrid2d class"),
        (8, "calling lilbee_search get_id_path astargrid2d"),
        *_stream("get_id_path returns a packedvector2array of grid coordinates", 12, 12),
        (40, "get_id_path returns a packedvector2array local $0.00"),
    ]


def test_clean_small_reel_passes() -> None:
    report = validate_texts("qwen3", "small", _good_small_timeline())
    assert report.ok, _checks(report)


def test_error_frame_fails_even_with_answer() -> None:
    timeline = [*_good_small_timeline(), (42, "chat failed: no models installed")]
    report = validate_texts("qwen3", "small", timeline)
    assert not report.ok
    assert _checks(report)["no_error_frame"] is False


def test_missing_dispatch_fails() -> None:
    # Model answered from memory: prompt + answer present, but never tool-called.
    timeline = [
        (0, "lilbee launch opencode"),
        (4, "search the indexed godot 4 reference for the astargrid2d class"),
        *_stream("get_id_path returns a packedvector2array", 8, 12),
    ]
    report = validate_texts("qwen3", "small", timeline)
    assert not report.ok
    assert _checks(report)["dispatch_visible"] is False


def test_ungrounded_answer_fails_answer_check() -> None:
    timeline = [
        (0, "lilbee launch opencode"),
        (4, "search the indexed godot 4 reference for the astargrid2d class"),
        (8, "calling lilbee_search"),
        *_stream("i could not find get_id_path return type in the reference sorry", 12, 12),
    ]
    report = validate_texts("qwen3", "small", timeline)
    assert not report.ok
    assert _checks(report)["answer_visible"] is False


def test_dead_screen_fails() -> None:
    # Two frames far apart: a boot reel that never showed real work.
    timeline = [(0, "lilbee launch opencode loading"), (200, "lilbee launch opencode loading")]
    report = validate_texts("qwen3", "small", timeline)
    assert not report.ok
    assert _checks(report)["not_dead_screen"] is False


def test_sparse_frames_fail_not_dead_screen() -> None:
    # Real work shown, but only a few unique frames over a long span (low ratio):
    # the camera mostly filmed a frozen screen.
    timeline = [
        (0, "lilbee launch opencode"),
        (5, "search astargrid2d get_id_path"),
        (10, "calling lilbee_search"),
        (200, "get_id_path returns a packedvector2array"),
    ]
    report = validate_texts("qwen3", "small", timeline)
    assert _checks(report)["not_dead_screen"] is False


def test_coder_tier_requires_code_on_camera() -> None:
    timeline = [
        (0, "lilbee launch opencode"),
        (6, "write ./level_generator.gd a procedural level generator"),
        (12, "calling lilbee_search set_cell tilemap"),
        *_stream("extends node2d func generate() -> void: set_cell(0, pos)", 18, 20),
        (60, "wrote level_generator.gd local $0.00"),
    ]
    report = validate_texts("qwen3-coder", "coder", timeline)
    assert report.ok, _checks(report)
