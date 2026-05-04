"""Tests for the shared worker bootstrap helpers in :mod:`worker_runtime`.

The per-role workers (embed, chat, rerank, vision) all dispatch through
``_dispatch_kind`` and bootstrap through ``run_worker``, so the
ping/shutdown/unknown-kind behaviour is centralized here. Per-role
handler tests live next to their handler in the worker-specific test
files.
"""

from __future__ import annotations

from typing import Any

from lilbee.providers.worker.worker_runtime import _dispatch_kind


class _RecordingConn:
    """In-process stand-in for multiprocessing.Connection.send."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)


def test_dispatch_kind_handles_shutdown_returns_false() -> None:
    conn = _RecordingConn()
    assert _dispatch_kind(conn, "shutdown", None, object(), {}, "embed") is False
    assert conn.sent == [("ack", None)]


def test_dispatch_kind_handles_ping_continues_loop() -> None:
    conn = _RecordingConn()
    assert _dispatch_kind(conn, "ping", None, object(), {}, "embed") is True
    assert conn.sent == [("pong", None)]


def test_dispatch_kind_routes_to_role_handler() -> None:
    conn = _RecordingConn()
    seen: list[tuple[Any, Any, Any]] = []

    def _handler(c: Any, payload: Any, session: Any) -> None:
        seen.append((c, payload, session))

    assert _dispatch_kind(conn, "embed", ["x"], "session-marker", {"embed": _handler}, "embed") is (
        True
    )
    assert len(seen) == 1
    assert seen[0][1] == ["x"]
    assert seen[0][2] == "session-marker"


def test_dispatch_kind_unknown_emits_serialized_value_error() -> None:
    conn = _RecordingConn()
    assert _dispatch_kind(conn, "totally_unknown", None, object(), {}, "embed") is True
    assert len(conn.sent) == 1
    kind, payload = conn.sent[0]
    assert kind == "error"
    assert payload.type_name == "ValueError"
    assert "totally_unknown" in payload.message
    assert "embed worker" in payload.message
