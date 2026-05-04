"""Tests for the shared worker bootstrap helpers in :mod:`worker_runtime`.

The per-role workers (embed, chat, rerank, vision) all dispatch through
``_dispatch_kind`` and bootstrap through ``run_worker``, so the
ping/shutdown/unknown-kind behaviour is centralized here. Per-role
handler tests live next to their handler in the worker-specific test
files.
"""

from __future__ import annotations

from typing import Any

from lilbee.providers.worker.worker_runtime import (
    WorkerLoopState,
    _dispatch_kind,
    stream_window,
)


class _RecordingConn:
    """In-process stand-in for multiprocessing.Connection.send."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)


def _state() -> WorkerLoopState:
    return WorkerLoopState(session=object())


def test_dispatch_kind_handles_shutdown_returns_false() -> None:
    conn = _RecordingConn()
    assert _dispatch_kind(conn, "shutdown", None, _state(), {}, "embed") is False
    assert conn.sent == [("ack", None)]


def test_dispatch_kind_handles_ping_continues_loop() -> None:
    conn = _RecordingConn()
    assert _dispatch_kind(conn, "ping", None, _state(), {}, "embed") is True
    assert conn.sent == [("pong", None)]


def test_dispatch_kind_drops_ping_during_stream_window() -> None:
    """Ping arriving while a stream is in flight must not emit a pong frame.

    The parent's stream consumer would read that pong out of band and
    raise ``ProtocolError: streamed unexpected kind 'pong'``. The next
    health tick re-pings once the stream completes.
    """
    conn = _RecordingConn()
    state = _state()
    state.stream_in_flight = True
    assert _dispatch_kind(conn, "ping", None, state, {}, "chat") is True
    assert conn.sent == []


def test_dispatch_kind_routes_to_role_handler() -> None:
    conn = _RecordingConn()
    seen: list[tuple[Any, Any, Any]] = []

    def _handler(c: Any, payload: Any, state: WorkerLoopState) -> None:
        seen.append((c, payload, state))

    state = WorkerLoopState(session="session-marker")
    assert _dispatch_kind(conn, "embed", ["x"], state, {"embed": _handler}, "embed") is True
    assert len(seen) == 1
    assert seen[0][1] == ["x"]
    assert seen[0][2] is state
    assert seen[0][2].session == "session-marker"


def test_dispatch_kind_unknown_emits_serialized_value_error() -> None:
    conn = _RecordingConn()
    assert _dispatch_kind(conn, "totally_unknown", None, _state(), {}, "embed") is True
    assert len(conn.sent) == 1
    kind, payload = conn.sent[0]
    assert kind == "error"
    assert payload.type_name == "ValueError"
    assert "totally_unknown" in payload.message
    assert "embed worker" in payload.message


def test_stream_window_flips_and_clears_flag() -> None:
    state = _state()
    assert state.stream_in_flight is False
    with stream_window(state):
        assert state.stream_in_flight is True
    assert state.stream_in_flight is False


def test_stream_window_clears_flag_on_exception() -> None:
    state = _state()
    try:
        with stream_window(state):
            assert state.stream_in_flight is True
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert state.stream_in_flight is False
