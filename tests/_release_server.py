"""A loopback HTTP server that stands in for a GitHub release download."""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator
from contextlib import contextmanager

REDIRECT_PATH = "/old"


class ReleaseServer:
    """Serves ``files`` by path with Range support; ``/old`` redirects to ``/f``."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[dict[str, str]] = []
        self.fail_next = 0
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
                    self.send_response(503)
                    self.end_headers()
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
        server.shutdown()
        server.server_close()
        thread.join()
