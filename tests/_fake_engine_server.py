"""A fake engine server whose chat requests wait for the test and die when the engine stops."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler

from tests._quiet_http_server import QuietHTTPServer

_HELD_WAIT_S = 30.0
_FIRST_FRAME = {"choices": [{"delta": {"content": "first"}}]}
_FINISH_FRAME = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
_ANSWER = {"choices": [{"message": {"content": "the answer"}, "finish_reason": "stop"}]}


def _chunk(text: str) -> bytes:
    data = text.encode()
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def _sse(frame: dict[str, object]) -> bytes:
    return _chunk(f"data: {json.dumps(frame)}\n\n")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _EngineHTTPServer

    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        engine = self.server.engine
        if body.get("tools"):
            # The client's template probe carries tools; it is not a chat the test holds.
            self._send_answer()
            return
        engine.count_request()
        if body.get("stream"):
            self._stream()
        else:
            self._answer()

    def _held_until_released(self) -> bool:
        """Wait for the test to release the request; False when the engine stopped first."""
        engine = self.server.engine
        engine.started.set()
        engine.release.wait(_HELD_WAIT_S)
        return not engine.stopped.is_set()

    def _abort(self) -> None:
        """Drop the connection the way a killed process does: no closing bytes."""
        self.close_connection = True
        self.connection.shutdown(socket.SHUT_RDWR)

    def _answer(self) -> None:
        if not self._held_until_released():
            self._abort()
            return
        self.server.engine.count_completed()
        self._send_answer()

    def _send_answer(self) -> None:
        body = json.dumps(_ANSWER).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.write(_sse(_FIRST_FRAME))
        self.wfile.flush()
        if not self._held_until_released():
            self._abort()
            return
        self.server.engine.count_completed()
        self.wfile.write(_sse(_FINISH_FRAME) + _chunk("data: [DONE]\n\n") + b"0\r\n\r\n")


class _EngineHTTPServer(QuietHTTPServer):
    engine: FakeEngine


class FakeEngine:
    """One engine process: it serves chat on a loopback port and counts what reaches it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.stopped = threading.Event()
        self._lock = threading.Lock()
        self._requests = 0
        self._completed = 0
        self._server = _EngineHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.engine = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def count_request(self) -> None:
        with self._lock:
            self._requests += 1

    def count_completed(self) -> None:
        with self._lock:
            self._completed += 1

    @property
    def requests(self) -> int:
        """Chat requests that reached this engine."""
        with self._lock:
            return self._requests

    @property
    def completed(self) -> int:
        """Chat requests this engine began a complete answer for."""
        with self._lock:
            return self._completed

    def stop(self) -> None:
        """Kill the engine: held requests drop their connections and the port closes."""
        self.stopped.set()
        self.release.set()
        self._server.shutdown()
        self._server.server_close()
