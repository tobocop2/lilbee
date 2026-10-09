"""A loopback server that answers every request from recorded responses and from nothing else."""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

from tools.qa.crawl_parity.corpus import (
    ORIGIN_PLACEHOLDER,
    Alternate,
    AlternateWhen,
    Corpus,
    Record,
    Response,
)

LOOPBACK = "127.0.0.1"
OTHER_HOST = "localhost"
OTHER_ORIGIN_PLACEHOLDER = b"{{OTHER_ORIGIN}}"
UNRECORDED_STATUS = 599
NO_BODY_STATUSES = frozenset({204, 304})
MS_PER_SECOND = 1000


@dataclass(frozen=True)
class Request:
    """One request the server answered."""

    method: str
    path: str
    status: int
    user_agent: str
    cookie: str
    at: float


class _Server(ThreadingHTTPServer):
    """A server that stays quiet when a client hangs up and reports every other error."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The hook gets no exception argument; the one being handled is the only source.
        if not isinstance(sys.exception(), ConnectionError):
            super().handle_error(request, client_address)


class Replay:
    """Serves one corpus on 127.0.0.1 and keeps a log of what was asked for."""

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus
        self._lock = threading.Lock()
        self._attempts: Counter[str] = Counter()
        self._requests: list[Request] = []
        self._extra_delay_ms = 0.0
        self._patient = False
        self._server = _Server((LOOPBACK, 0), self._handler_class())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> Replay:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    @property
    def port(self) -> int:
        """The port the server listens on."""
        return int(self._server.server_address[1])

    @property
    def origin(self) -> str:
        """The origin of the site under crawl."""
        return f"http://{LOOPBACK}:{self.port}"

    @property
    def other_origin(self) -> str:
        """The origin that stands for every other host."""
        return f"http://{OTHER_HOST}:{self.port}"

    def url(self, path: str) -> str:
        """The replay URL of a corpus path."""
        return self.origin + path

    def reset(self) -> None:
        """Forget the request log and the request counts, as before a new run."""
        with self._lock:
            self._attempts.clear()
            self._requests.clear()

    def delay_every_response(self, milliseconds: float) -> None:
        """Wait this long before every answer; the speed self-test plants a slowdown with it."""
        with self._lock:
            self._extra_delay_ms = milliseconds

    def answer_as_to_a_patient_reader(self, patient: bool) -> None:
        """Skip the answers a record gives only to its first requests: a reader tries again."""
        with self._lock:
            self._patient = patient

    def requests(self) -> list[Request]:
        """The requests answered since the last reset."""
        with self._lock:
            return list(self._requests)

    def unrecorded(self) -> list[str]:
        """The paths asked for since the last reset that no record answers."""
        return [request.path for request in self.requests() if request.status == UNRECORDED_STATUS]

    def _find(self, path: str) -> Record | None:
        records = self._corpus.records
        return records.get(path) or records.get(urlsplit(path).path)

    def _use_alternate(self, alternate: Alternate, path: str, cookie: str) -> bool:
        if alternate.when is AlternateWhen.NO_COOKIE:
            return alternate.cookie not in cookie
        if self._patient:
            return False
        with self._lock:
            self._attempts[path] += 1
            return self._attempts[path] <= alternate.count

    def _substitute(self, data: bytes) -> bytes:
        return data.replace(ORIGIN_PLACEHOLDER, self.origin.encode()).replace(
            OTHER_ORIGIN_PLACEHOLDER, self.other_origin.encode()
        )

    def answer(self, path: str, cookie: str) -> tuple[Response, float]:
        """The response for *path* and the delay before it in milliseconds."""
        record = self._find(path)
        if record is None:
            return Response(
                UNRECORDED_STATUS, (("Content-Type", "text/plain"),), b"not recorded"
            ), 0
        response = record.response
        if record.alternate and self._use_alternate(record.alternate, path, cookie):
            response = record.alternate.response
        headers = tuple(
            (name, self._substitute(value.encode()).decode()) for name, value in response.headers
        )
        body = self._substitute(response.body) if record.substitute_origin else response.body
        return Response(response.status, headers, body), record.delay_ms + self._extra_delay_ms

    def _log(self, request: Request) -> None:
        with self._lock:
            self._requests.append(request)

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        replay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                """Requests go to the replay log, not to stderr."""

            def _serve(self, send_body: bool) -> None:
                cookie = self.headers.get("Cookie", "")
                response, delay_ms = replay.answer(self.path, cookie)
                if delay_ms:
                    time.sleep(delay_ms / MS_PER_SECOND)
                self.send_response(response.status)
                for name, value in response.headers:
                    self.send_header(name, value)
                has_body = response.status not in NO_BODY_STATUSES
                if has_body:
                    self.send_header("Content-Length", str(len(response.body)))
                self.end_headers()
                if has_body and send_body:
                    self.wfile.write(response.body)
                replay._log(
                    Request(
                        self.command,
                        self.path,
                        response.status,
                        self.headers.get("User-Agent", ""),
                        cookie,
                        time.time(),
                    )
                )

            def do_GET(self) -> None:
                self._serve(send_body=True)

            def do_HEAD(self) -> None:
                self._serve(send_body=False)

        return Handler
