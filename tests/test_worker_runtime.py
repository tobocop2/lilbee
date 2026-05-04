"""Tests for the shared worker bootstrap helpers in :mod:`worker_runtime`.

The per-role workers (embed, chat, rerank, vision) all dispatch through
``_handle_data_frame`` / ``_handle_health_frame`` and bootstrap through
``run_worker``. Per-role handler tests live next to their handler in
the worker-specific test files.
"""

from __future__ import annotations

from typing import Any

from lilbee.providers.worker.worker_runtime import (
    WorkerLoopState,
    _handle_data_frame,
    _handle_health_frame,
)


class _RecordingConn:
    """In-process stand-in for multiprocessing.Connection."""

    def __init__(self, incoming: list[tuple[str, Any]] | None = None) -> None:
        self._incoming = list(incoming or [])
        self.sent: list[tuple[str, Any]] = []

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)

    def recv(self) -> tuple[str, Any]:
        return self._incoming.pop(0)


def _state() -> WorkerLoopState:
    return WorkerLoopState(session=object())


def test_handle_data_frame_shutdown_returns_false() -> None:
    """Shutdown frame on the data pipe acks and stops the worker loop."""
    conn = _RecordingConn(incoming=[("shutdown", None)])
    assert _handle_data_frame(conn, _state(), {}, "embed") is False
    assert conn.sent == [("ack", None)]


def test_handle_data_frame_routes_to_role_handler() -> None:
    """Recognized role kinds get dispatched to their handler with the live state."""
    conn = _RecordingConn(incoming=[("embed", ["x"])])
    seen: list[tuple[Any, Any, WorkerLoopState]] = []

    def _handler(c: Any, payload: Any, state: WorkerLoopState) -> None:
        seen.append((c, payload, state))

    state = WorkerLoopState(session="session-marker")
    assert _handle_data_frame(conn, state, {"embed": _handler}, "embed") is True
    assert len(seen) == 1
    assert seen[0][1] == ["x"]
    assert seen[0][2] is state


def test_handle_data_frame_unknown_emits_serialized_value_error() -> None:
    """Unknown kinds reply with a serialized ValueError naming the role + kind."""
    conn = _RecordingConn(incoming=[("totally_unknown", None)])
    assert _handle_data_frame(conn, _state(), {}, "embed") is True
    assert len(conn.sent) == 1
    kind, payload = conn.sent[0]
    assert kind == "error"
    assert payload.type_name == "ValueError"
    assert "totally_unknown" in payload.message
    assert "embed worker" in payload.message


def test_handle_data_frame_eof_returns_false() -> None:
    """Pipe EOF on the data side stops the worker loop cleanly."""

    class _EOFConn:
        def recv(self) -> tuple[str, Any]:
            raise EOFError

        def send(self, message: tuple[str, Any]) -> None:
            raise AssertionError("worker must not send after EOF")

    assert _handle_data_frame(_EOFConn(), _state(), {}, "embed") is False


def test_handle_health_frame_pings_pong() -> None:
    """A ping on the health pipe replies with pong."""
    conn = _RecordingConn(incoming=[("ping", None)])
    _handle_health_frame(conn, "embed")
    assert conn.sent == [("pong", None)]


def test_handle_health_frame_eof_is_silent() -> None:
    """EOF on the health pipe drops without raising; the data side handles shutdown."""

    class _EOFConn:
        def recv(self) -> tuple[str, Any]:
            raise EOFError

        def send(self, message: tuple[str, Any]) -> None:
            raise AssertionError("worker must not send on EOF")

    _handle_health_frame(_EOFConn(), "embed")  # must not raise


def test_handle_health_frame_unexpected_kind_logs_and_drops(caplog) -> None:
    """Anything other than ping on the health pipe logs a warning and drops."""
    conn = _RecordingConn(incoming=[("garbage", None)])
    with caplog.at_level("WARNING", logger="lilbee.providers.worker.worker_runtime"):
        _handle_health_frame(conn, "embed")
    assert conn.sent == []
    assert any("unexpected health-pipe kind" in rec.message for rec in caplog.records)


def test_run_worker_dispatches_data_then_health_then_shutdown(monkeypatch) -> None:
    """End-to-end: run_worker loops over data + health, drains both, exits on shutdown."""
    from collections import deque

    from lilbee.providers.worker import worker_runtime
    from lilbee.providers.worker.transport import RoleConfig

    monkeypatch.setattr(
        "lilbee.providers.worker.worker_runtime.redirect_stdio_to_devnull", lambda: None
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.worker_runtime.configure_worker_logging", lambda _r: None
    )

    class _FakeConn:
        def __init__(self, incoming: list[tuple[str, Any]]) -> None:
            self._incoming = deque(incoming)
            self.sent: list[tuple[str, Any]] = []
            self.closed = False

        def recv(self) -> tuple[str, Any]:
            return self._incoming.popleft()

        def send(self, message: tuple[str, Any]) -> None:
            self.sent.append(message)

        def close(self) -> None:
            self.closed = True

    data = _FakeConn(incoming=[("embed", ["x"]), ("shutdown", None)])
    health = _FakeConn(incoming=[("ping", None)])

    # First wait returns no ready conns (timeout, exercises the continue
    # branch). Then the embed (data), the ping (health), and the shutdown.
    sequence: deque[list[Any]] = deque([[], [data], [health], [data]])

    def _fake_wait(_conns: Any, timeout: float) -> list[Any]:
        return sequence.popleft() if sequence else []

    monkeypatch.setattr("lilbee.providers.worker.worker_runtime.wait", _fake_wait)

    seen_payloads: list[Any] = []

    def _embed_handler(conn: Any, payload: Any, _state: WorkerLoopState) -> None:
        seen_payloads.append(payload)
        conn.send(("result", [[1.0]]))

    closed_sessions: list[bool] = []

    class _Session:
        def close(self) -> None:
            closed_sessions.append(True)

    role_config = RoleConfig(
        role="embed", model_path=__import__("pathlib").Path("/nope"), mode="embed"
    )
    worker_runtime.run_worker(
        data,
        health,
        None,
        role_config,
        session_factory=lambda *_a: _Session(),
        kind_handlers={"embed": _embed_handler},
    )
    assert seen_payloads == [["x"]]
    assert ("result", [[1.0]]) in data.sent
    assert ("ack", None) in data.sent
    assert ("pong", None) in health.sent
    assert closed_sessions == [True]
    assert data.closed
    assert health.closed
