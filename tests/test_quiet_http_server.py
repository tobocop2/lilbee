"""The quiet test server hides a client hang-up and still reports a real handler error."""

from __future__ import annotations

import contextlib
import io
import socket
import threading
from http.server import BaseHTTPRequestHandler

import pytest

from tests._quiet_http_server import QuietHTTPServer


class _Handler(BaseHTTPRequestHandler):
    failure: type[Exception] = BrokenPipeError

    def do_GET(self) -> None:
        raise self.failure("client went away")

    def log_message(self, *args: object) -> None:
        """Silence the access log."""


def _serve_one_request(failure: type[Exception], monkeypatch: pytest.MonkeyPatch) -> str:
    """Send one request to a server whose handler raises *failure*; return what it printed."""
    handler = type("_Failing", (_Handler,), {"failure": failure})
    server = QuietHTTPServer(("127.0.0.1", 0), handler)
    # Non-daemon workers, so server_close() below waits for the handler to finish.
    server.daemon_threads = False
    printed = io.StringIO()
    monkeypatch.setattr("sys.stderr", printed)
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    with contextlib.closing(socket.create_connection(server.server_address)) as client:
        client.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        client.settimeout(5)
        with contextlib.suppress(OSError):
            client.recv(1)
    thread.join(timeout=5)
    # server_close() joins the worker thread that ran the handler.
    server.server_close()
    return printed.getvalue()


@pytest.mark.parametrize("failure", [BrokenPipeError, ConnectionAbortedError, ConnectionResetError])
def test_a_client_hang_up_prints_nothing(
    failure: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _serve_one_request(failure, monkeypatch) == ""


def test_any_other_handler_error_still_prints(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "RuntimeError" in _serve_one_request(RuntimeError, monkeypatch)
