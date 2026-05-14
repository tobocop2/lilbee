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
    """In-process stand-in for multiprocessing.Connection.

    ``poll`` is wired so the data loop test cases return True immediately
    when an incoming frame is queued.
    """

    def __init__(self, incoming: list[tuple[str, Any]] | None = None) -> None:
        self._incoming = list(incoming or [])
        self.sent: list[tuple[str, Any]] = []

    def poll(self, _timeout: float = 0.0) -> bool:
        return bool(self._incoming)

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)

    def recv(self) -> tuple[str, Any]:
        return self._incoming.pop(0)


def _state() -> WorkerLoopState:
    return WorkerLoopState(session=object())


def test_handle_data_frame_routes_to_role_handler() -> None:
    """Recognized role kinds get a Reply that sends on the same data pipe."""
    conn = _RecordingConn(incoming=[("embed", ["x"])])
    seen: list[tuple[Any, Any, WorkerLoopState]] = []

    def _handler(reply: Reply, payload: Any, state: WorkerLoopState) -> None:
        seen.append((reply, payload, state))
        reply.send("result", [[1.0]])

    state = WorkerLoopState(session="session-marker")
    assert _handle_data_frame(conn, state, {"embed": _handler}, "embed", threading.Event()) is True
    assert len(seen) == 1
    assert seen[0][1] == ["x"]
    assert seen[0][2] is state
    assert conn.sent == [("result", [[1.0]])]


def test_handle_data_frame_unknown_emits_serialized_value_error() -> None:
    """Unknown kinds reply with a serialized ValueError on the data pipe."""
    conn = _RecordingConn(incoming=[("totally_unknown", None)])
    assert _handle_data_frame(conn, _state(), {}, "embed", threading.Event()) is True
    assert len(conn.sent) == 1
    kind, payload = conn.sent[0]
    assert kind == "error"
    assert payload.type_name == "ValueError"
    assert "totally_unknown" in payload.message
    assert "embed worker" in payload.message


def test_handle_data_frame_eof_returns_false() -> None:
    """Pipe EOF on the data side stops the worker loop cleanly."""

    class _EOFConn:
        def poll(self, _timeout: float = 0.0) -> bool:
            return True

        def recv(self) -> tuple[str, Any]:
            raise EOFError

        def send(self, message: tuple[str, Any]) -> None:
            raise AssertionError("worker must not send after EOF")

    assert _handle_data_frame(_EOFConn(), _state(), {}, "embed", threading.Event()) is False


def test_handle_data_frame_idle_poll_returns_on_shutdown_event() -> None:
    """When no frame is pending and shutdown_event fires, the loop exits."""

    class _IdleConn:
        polls: int = 0

        def poll(self, _timeout: float = 0.0) -> bool:
            _IdleConn.polls += 1
            return False

        def recv(self) -> tuple[str, Any]:
            raise AssertionError("recv must not be called when poll returns False")

        def send(self, message: tuple[str, Any]) -> None:
            raise AssertionError("worker must not send during idle shutdown")

    event = threading.Event()
    event.set()
    assert _handle_data_frame(_IdleConn(), _state(), {}, "embed", event) is False
    assert _IdleConn.polls == 1


def test_heartbeat_loop_pongs_pings() -> None:
    """The heartbeat thread responds to ping with pong on its own pipe."""

    class _EOFAfterFirst(_RecordingConn):
        """RecordingConn that raises EOF after the first recv, to exit the loop."""

        def recv(self) -> tuple[str, Any]:
            if self._incoming:
                return self._incoming.pop(0)
            raise EOFError

    health = _EOFAfterFirst(incoming=[("ping", None)])
    event = threading.Event()
    _heartbeat_loop(health, "embed", event)
    assert health.sent == [("pong", None)]
    assert event.is_set()


def test_heartbeat_loop_acks_shutdown_and_sets_event() -> None:
    """SHUTDOWN on the health pipe acks back and signals the data loop to stop."""

    class _Pump:
        def __init__(self) -> None:
            self._pending = [("shutdown", None)]
            self.sent: list[tuple[str, Any]] = []

        def recv(self) -> tuple[str, Any]:
            if self._pending:
                return self._pending.pop(0)
            raise EOFError

        def send(self, message: tuple[str, Any]) -> None:
            self.sent.append(message)

    health = _Pump()
    event = threading.Event()
    _heartbeat_loop(health, "embed", event)
    assert health.sent == [("ack", None)]
    assert event.is_set()


def test_heartbeat_loop_shutdown_survives_send_error() -> None:
    """If the ack send fails because the parent already closed, exit quietly."""

    class _Pump:
        def __init__(self) -> None:
            self._pending = [("shutdown", None)]

        def recv(self) -> tuple[str, Any]:
            if self._pending:
                return self._pending.pop(0)
            raise EOFError

        def send(self, _message: tuple[str, Any]) -> None:
            raise BrokenPipeError("parent gone")

    event = threading.Event()
    _heartbeat_loop(_Pump(), "embed", event)  # must not raise
    assert event.is_set()


def test_heartbeat_loop_drops_unexpected_kind(caplog) -> None:
    """Anything other than ping or shutdown on the health pipe logs a warning."""

    class _Pump:
        def __init__(self) -> None:
            self._pending = [("garbage", None)]
            self.sent: list[tuple[str, Any]] = []

        def recv(self) -> tuple[str, Any]:
            if self._pending:
                return self._pending.pop(0)
            raise EOFError

        def send(self, message: tuple[str, Any]) -> None:
            self.sent.append(message)

    health = _Pump()
    event = threading.Event()
    with caplog.at_level("WARNING", logger="lilbee.providers.worker.worker_runtime"):
        _heartbeat_loop(health, "embed", event)
    assert health.sent == []
    assert event.is_set()  # set on EOF after the warning frame
    assert any("unexpected health-pipe kind" in rec.message for rec in caplog.records)


def test_heartbeat_loop_returns_on_ping_send_error() -> None:
    """Pong send that fails (parent already closed) exits the loop quietly."""

    class _BadSend:
        def __init__(self) -> None:
            self._pending = [("ping", None)]

        def recv(self) -> tuple[str, Any]:
            if self._pending:
                return self._pending.pop(0)
            raise EOFError

        def send(self, _message: tuple[str, Any]) -> None:
            raise BrokenPipeError("parent gone")

    event = threading.Event()
    _heartbeat_loop(_BadSend(), "embed", event)  # must not raise
    assert event.is_set()


class _FakeDataConn:
    """Module-level fake data conn for the run_worker integration test."""

    def __init__(self, incoming: list[tuple[str, Any]]) -> None:
        from collections import deque

        self._incoming = deque(incoming)
        self.sent: list[tuple[str, Any]] = []
        self.closed = False

    def poll(self, _timeout: float = 0.0) -> bool:
        return bool(self._incoming)

    def recv(self) -> tuple[str, Any]:
        if self._incoming:
            return self._incoming.popleft()
        raise EOFError

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)

    def close(self) -> None:
        self.closed = True


class _GatedHealthConn:
    """Health pipe that delivers SHUTDOWN only after a gating event fires.

    Gating eliminates the race between the heartbeat daemon thread and
    the main data loop so the test deterministically observes that
    run_worker dispatched the data frame before exiting on shutdown.
    """

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate
        self._sent_shutdown = False
        self.sent: list[tuple[str, Any]] = []
        self.closed = False

    def recv(self) -> tuple[str, Any]:
        if not self._sent_shutdown:
            self._gate.wait(timeout=2.0)
            self._sent_shutdown = True
            return ("shutdown", None)
        raise EOFError

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)

    def close(self) -> None:
        self.closed = True


class _RecordingSession:
    """Session stub that records ``close()`` for assertion in run_worker tests."""

    def __init__(self, log: list[bool]) -> None:
        self._log = log

    def close(self) -> None:
        self._log.append(True)


def test_run_worker_dispatches_data_then_health_shutdown(monkeypatch) -> None:
    """run_worker reads data frames, dispatches handler, exits on health shutdown."""
    from lilbee.providers.worker import worker_runtime
    from lilbee.providers.worker.transport import RoleConfig

    monkeypatch.setattr(
        "lilbee.providers.worker.worker_runtime.redirect_stdio_to_devnull", lambda: None
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.worker_runtime.configure_worker_logging", lambda _r: None
    )

    data = _FakeDataConn(incoming=[("embed", ["x"])])
    handler_done = threading.Event()
    health = _GatedHealthConn(gate=handler_done)

    seen_payloads: list[Any] = []

    def _embed_handler(reply: Reply, payload: Any, _state: WorkerLoopState) -> None:
        seen_payloads.append(payload)
        reply.send("result", [[1.0]])
        handler_done.set()

    closed_sessions: list[bool] = []
    role_config = RoleConfig(
        role="embed", model_path=__import__("pathlib").Path("/nope"), mode="embed"
    )
    worker_runtime.run_worker(
        data,
        health,
        None,
        role_config,
        session_factory=lambda *_a: _RecordingSession(closed_sessions),
        kind_handlers={"embed": _embed_handler},
    )
    assert handler_done.is_set()
    # Brief wait so heartbeat daemon thread is observably finished.
    time.sleep(0.05)
    assert seen_payloads == [["x"]]
    assert ("result", [[1.0]]) in data.sent
    assert ("ack", None) in health.sent
    assert closed_sessions == [True]
    assert data.closed
    assert health.closed
