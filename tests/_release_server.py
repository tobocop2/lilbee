"""A loopback HTTP server that stands in for a GitHub release download."""

from __future__ import annotations

import http.server
import socket
import struct
import threading
from collections.abc import Iterator
from contextlib import contextmanager

REDIRECT_PATH = "/old"


class ReleaseServer:
    """Serves ``files`` by path with Range support; ``/old`` redirects to ``/f``.

    ``fail_next`` answers that many requests with ``fail_status``. ``reset_next`` sends half
    of the body and then resets the connection. ``close_next`` closes the connection before
    it sends a status line. ``silent_next`` accepts a request and sends nothing. ``stall_next``
    sends the first half of the body and then goes quiet on the open socket, so a resume past
    the half gets nothing. Both hold the socket until
    the server stops.
    """

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[dict[str, str]] = []
        self.fail_next = 0
        self.fail_status = 503
        self.reset_next = 0
        self.close_next = 0
        self.silent_next = 0
        self.stall_next = 0
        self.stop = threading.Event()
        self.url = ""

    def handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                owner.requests.append({k.lower(): v for k, v in self.headers.items()})
                if owner.fail_next:
                    owner.fail_next -= 1
                    self.send_response(owner.fail_status)
                    self.end_headers()
                    return
                if owner.silent_next:
                    owner.silent_next -= 1
                    owner.stop.wait()
                    return
                if owner.close_next:
                    owner.close_next -= 1
                    self.close_connection = True
                    self.connection.close()
                    return
                if self.path == REDIRECT_PATH:
                    self.send_response(302)
                    self.send_header("Location", "/f")
                    self.end_headers()
                    return
                body = owner.files.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                start = int(self.headers.get("Range", "bytes=0-")[6:].split("-")[0])
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(body) - start))
                self.end_headers()
                if owner.reset_next or owner.stall_next:
                    reset = bool(owner.reset_next)
                    owner.reset_next -= reset
                    owner.stall_next -= not reset
                    end = start + (len(body) - start) // 2 if reset else len(body) // 2
                    self.wfile.write(body[start:end])
                    self.wfile.flush()
                    if reset:
                        linger = struct.pack("ii", 1, 0)
                        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
                        self.close_connection = True
                        self.connection.close()
                    else:
                        owner.stop.wait()
                    return
                self.wfile.write(body[start:])

        return Handler


@contextmanager
def release_server() -> Iterator[ReleaseServer]:
    owner = ReleaseServer()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), owner.handler())
    owner.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield owner
    finally:
        owner.stop.set()
        server.shutdown()
        server.server_close()
        thread.join()
