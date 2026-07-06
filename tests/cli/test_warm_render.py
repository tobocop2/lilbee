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


def test_model_label_uses_canonical_label_with_fallback() -> None:
    ref = "unsloth/GLM-4.5-Air-GGUF/GLM-4.5-Air-Q2_K.gguf"
    # Reuses the project's canonical ref-to-label helper, so it matches the rest
    # of the UI (cleaned repo name) rather than a bespoke filename.
    assert warm_render._model_label(ref) == warm_render.display_label_for_ref(ref)
    assert warm_render._model_label(None) == "chat model"


def test_render_walks_all_phases_to_ready() -> None:
    # Includes STARTING so the indeterminate-preparing branch of _apply is covered.
    lines = [
        _sse(WarmProgress(phase=WarmPhase.STARTING)),
        _sse(
            WarmProgress(
                phase=WarmPhase.READING_WEIGHTS, bytes_done=5, bytes_total=10, detail="shard 1/2"
            )
        ),
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
        "data:",  # a data line with an empty payload (e.g. a keep-alive)
        "data: not-json",
        'data: {"no_phase": 1}',
        _sse(WarmProgress(phase=WarmPhase.READY)),
    ]
    with _patch_stream(_FakeStream(lines)):
        events = list(warm_render._iter_warm_events("http://x", 5.0))
    assert [e.phase for e in events] == [WarmPhase.READY]
