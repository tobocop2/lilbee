"""Tests for the launcher-side warm progress renderer and its SSE parsing."""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest import mock

import httpx

from lilbee.cli.launchers import warm_render
from lilbee.providers.warm_progress import WarmPhase, WarmProgress


def _sse(progress: WarmProgress) -> str:
    return f"data: {json.dumps(progress.model_dump())}"


class _FakeStream:
    """Stand-in for ``httpx.stream(...)`` as a context manager over SSE lines."""

    def __init__(self, lines: list[str], *, raise_on_enter: Exception | None = None) -> None:
        self._lines = lines
        self._raise = raise_on_enter

    def __enter__(self) -> _FakeStream:
        if self._raise is not None:
            raise self._raise
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self) -> Iterator[str]:
        yield from self._lines


def _patch_stream(stream: _FakeStream):
    return mock.patch.object(warm_render.httpx, "stream", return_value=stream)


def test_short_model_strips_repo_and_suffix() -> None:
    ref = "unsloth/GLM-4.5-Air-GGUF/GLM-4.5-Air-Q2_K.gguf"
    assert warm_render._short_model(ref) == "GLM-4.5-Air-Q2_K"
    assert warm_render._short_model(None) == "chat model"


def test_render_returns_true_when_stream_reaches_ready() -> None:
    lines = [
        _sse(WarmProgress(phase=WarmPhase.READING_WEIGHTS, bytes_done=5, bytes_total=10)),
        _sse(WarmProgress(phase=WarmPhase.LOADING_ENGINE)),
        _sse(WarmProgress(phase=WarmPhase.READY, bytes_total=10)),
        "data: [DONE]",
    ]
    with _patch_stream(_FakeStream(lines)):
        assert warm_render.render_warm("http://x", 5.0) is True


def test_render_returns_false_on_error_phase() -> None:
    lines = [_sse(WarmProgress(phase=WarmPhase.ERROR, error="no vram"))]
    with _patch_stream(_FakeStream(lines)):
        assert warm_render.render_warm("http://x", 5.0) is False


def test_render_returns_false_when_stream_ends_without_ready() -> None:
    lines = [_sse(WarmProgress(phase=WarmPhase.LOADING_ENGINE)), "data: [DONE]"]
    with _patch_stream(_FakeStream(lines)):
        assert warm_render.render_warm("http://x", 5.0) is False


def test_render_returns_none_when_stream_unavailable() -> None:
    # An older server without the endpoint -> connection error -> caller falls back.
    stream = _FakeStream([], raise_on_enter=httpx.ConnectError("refused"))
    with _patch_stream(stream):
        assert warm_render.render_warm("http://x", 5.0) is None


def test_iter_skips_non_data_and_malformed_lines() -> None:
    lines = [
        "event: warm",
        "",
        "data: not-json",
        'data: {"no_phase": 1}',
        _sse(WarmProgress(phase=WarmPhase.READY)),
    ]
    with _patch_stream(_FakeStream(lines)):
        events = list(warm_render._iter_warm_events("http://x", 5.0))
    assert [e.phase for e in events] == [WarmPhase.READY]
