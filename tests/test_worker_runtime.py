"""Tests for the shared worker bootstrap helpers in :mod:`worker_runtime`.

The per-role workers (embed, chat, rerank, vision) all dispatch through
``_handle_data_frame`` and bootstrap through ``run_worker``. Per-role
handler tests live next to their handler in the worker-specific test
files.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from lilbee.providers.worker.worker_runtime import (
    Reply,
    WorkerLoopState,
    _handle_data_frame,
    _heartbeat_loop,
)


class _RecordingConn:
    """In-process stand-in for multiprocessing.Connection."""

    def __init__(self, incoming: list[tuple[int, str, Any]] | None = None) -> None:
        self._incoming = list(incoming or [])
        self.sent: list[tuple[int, str, Any]] = []

    def send(self, message: tuple[int, str, Any]) -> None:
        self.sent.append(message)

    def recv(self) -> tuple[int, str, Any]:
        return self._incoming.pop(0)


def _state() -> WorkerLoopState:
    return WorkerLoopState(session=object())


def test_handle_data_frame_shutdown_returns_false() -> None:
    """Shutdown frame on the data pipe acks (echoing call_id) and stops the loop."""
    conn = _RecordingConn(incoming=[(7, "shutdown", None)])
    assert _handle_data_frame(conn, _state(), {}, "embed") is False
    assert conn.sent == [(7, "ack", None)]


def test_handle_data_frame_routes_to_role_handler() -> None:
    """Recognized role kinds get a Reply bound to the request's call_id."""
    conn = _RecordingConn(incoming=[(42, "embed", ["x"])])
    seen: list[tuple[Any, Any, WorkerLoopState]] = []

    def _handler(reply: Reply, payload: Any, state: WorkerLoopState) -> None:
        seen.append((reply, payload, state))
        reply.send("result", [[1.0]])

    state = WorkerLoopState(session="session-marker")
    assert _handle_data_frame(conn, state, {"embed": _handler}, "embed") is True
    assert len(seen) == 1
    assert seen[0][1] == ["x"]
    assert seen[0][2] is state
    assert conn.sent == [(42, "result", [[1.0]])]


def test_handle_data_frame_unknown_emits_serialized_value_error() -> None:
    """Unknown kinds reply with a serialized ValueError tagged with the call_id."""
    conn = _RecordingConn(incoming=[(99, "totally_unknown", None)])
    assert _handle_data_frame(conn, _state(), {}, "embed") is True
    assert len(conn.sent) == 1
    call_id, kind, payload = conn.sent[0]
    assert call_id == 99
    assert kind == "error"
    assert payload.type_name == "ValueError"
    assert "totally_unknown" in payload.message
    assert "embed worker" in payload.message


def test_handle_data_frame_eof_returns_false() -> None:
    """Pipe EOF on the data side stops the worker loop cleanly."""

    class _EOFConn:
        def recv(self) -> tuple[int, str, Any]:
            raise EOFError

        def send(self, message: tuple[int, str, Any]) -> None:
            raise AssertionError("worker must not send after EOF")

    assert _handle_data_frame(_EOFConn(), _state(), {}, "embed") is False


def test_heartbeat_loop_pongs_pings() -> None:
    """The heartbeat thread responds to ping with pong on its own pipe."""

    class _EOFAfterFirst(_RecordingConn):
        """RecordingConn that raises EOF after the first recv, to exit the loop."""

        def recv(self) -> tuple[int, str, Any]:
            if self._incoming:
                return self._incoming.pop(0)
            raise EOFError

    health = _EOFAfterFirst(incoming=[(0, "ping", None)])
    _heartbeat_loop(health, "embed")
    assert health.sent == [(0, "pong", None)]


def test_heartbeat_loop_drops_unexpected_kind(caplog) -> None:
    """Anything other than ping on the health pipe logs a warning and drops."""

    class _Pump:
        def __init__(self) -> None:
            self._pending = [(0, "garbage", None)]
            self.sent: list[tuple[int, str, Any]] = []

        def recv(self) -> tuple[int, str, Any]:
            if self._pending:
                return self._pending.pop(0)
            raise EOFError

        def send(self, message: tuple[int, str, Any]) -> None:
            self.sent.append(message)

    health = _Pump()
    with caplog.at_level("WARNING", logger="lilbee.providers.worker.worker_runtime"):
        _heartbeat_loop(health, "embed")
    assert health.sent == []
    assert any("unexpected health-pipe kind" in rec.message for rec in caplog.records)


def test_heartbeat_loop_returns_on_send_error() -> None:
    """Pong send that fails (parent already closed) exits the loop quietly."""

    class _BadSend:
        def __init__(self) -> None:
            self._pending = [(0, "ping", None)]

        def recv(self) -> tuple[int, str, Any]:
            if self._pending:
                return self._pending.pop(0)
            raise EOFError

        def send(self, _message: tuple[int, str, Any]) -> None:
            raise BrokenPipeError("parent gone")

    _heartbeat_loop(_BadSend(), "embed")  # must not raise


def test_run_worker_dispatches_data_then_shutdown(monkeypatch) -> None:
    """End-to-end: run_worker reads data frames, dispatches handler, exits on shutdown."""
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
        def __init__(self, incoming: list[tuple[int, str, Any]]) -> None:
            self._incoming = deque(incoming)
            self.sent: list[tuple[int, str, Any]] = []
            self.closed = False

        def recv(self) -> tuple[int, str, Any]:
            if self._incoming:
                return self._incoming.popleft()
            raise EOFError

        def send(self, message: tuple[int, str, Any]) -> None:
            self.sent.append(message)

        def close(self) -> None:
            self.closed = True

    data = _FakeConn(incoming=[(11, "embed", ["x"]), (0, "shutdown", None)])
    health = _FakeConn(incoming=[])  # heartbeat thread will get EOF and exit

    seen_payloads: list[Any] = []
    handler_started = threading.Event()

    def _embed_handler(reply: Reply, payload: Any, _state: WorkerLoopState) -> None:
        seen_payloads.append(payload)
        reply.send("result", [[1.0]])
        handler_started.set()

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
    assert handler_started.is_set()
    # Brief wait so the heartbeat daemon thread can observe the EOF on health.
    time.sleep(0.05)
    assert seen_payloads == [["x"]]
    assert (11, "result", [[1.0]]) in data.sent
    assert (0, "ack", None) in data.sent
    assert closed_sessions == [True]
    assert data.closed
    assert health.closed
