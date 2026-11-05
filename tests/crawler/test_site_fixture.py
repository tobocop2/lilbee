"""The fixture site the end-to-end crawl tests serve keeps its handler errors off stderr.

The CLI tests run in this process and read the CLI's stdout and stderr as one string. A
client that resets its connection is normal for a crawler. The stock server prints a traceback
for it, and that print lands in the middle of the CLI's JSON line.
"""

from __future__ import annotations

import socket
import struct
import threading
from http.server import ThreadingHTTPServer

import pytest

from tests.integration._crawl_site import LOOPBACK, RunningSite

_HANDLER_DONE_SECONDS = 10


def _reset_connection(port: int) -> None:
    """Open a connection to *port* and close it with a reset, as a crawler that gives up does."""
    client = socket.create_connection((LOOPBACK, port))
    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    client.close()


def _wait_for_a_handled_connection(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """An event that is set once the server has finished with a connection, error or not.

    The server closes a connection after it reports the error, so this marks the point after
    which a report can no longer arrive.
    """
    done = threading.Event()
    shut_down = ThreadingHTTPServer.shutdown_request

    def _marked(self: ThreadingHTTPServer, request: object) -> None:
        shut_down(self, request)  # type: ignore[arg-type]
        done.set()

    monkeypatch.setattr(ThreadingHTTPServer, "shutdown_request", _marked)
    return done


class TestFixtureSiteErrors:
    def test_a_client_that_resets_its_connection_prints_nothing(
        self, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        done = _wait_for_a_handled_connection(monkeypatch)
        with RunningSite() as site:
            _reset_connection(site.port)
            assert done.wait(_HANDLER_DONE_SECONDS)
        assert capfd.readouterr().err == ""

    @pytest.mark.parametrize(
        "error",
        [
            ConnectionResetError("reset by peer"),
            ConnectionAbortedError(10053, "aborted by the software in your host machine"),
            BrokenPipeError("broken pipe"),
        ],
        ids=["reset", "aborted", "broken-pipe"],
    )
    def test_every_closed_connection_error_the_platforms_raise_prints_nothing(
        self,
        error: ConnectionError,
        capfd: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        done = _wait_for_a_handled_connection(monkeypatch)

        def _closed(self: object, *args: object) -> None:
            raise error

        monkeypatch.setattr("http.server.BaseHTTPRequestHandler.handle_one_request", _closed)
        with RunningSite() as site:
            _reset_connection(site.port)
            assert done.wait(_HANDLER_DONE_SECONDS)
        assert capfd.readouterr().err == ""

    def test_an_error_that_is_not_a_closed_connection_is_still_printed(
        self, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        done = _wait_for_a_handled_connection(monkeypatch)

        def _broken(self: object, *args: object) -> None:
            raise ValueError("the handler is broken")

        monkeypatch.setattr("http.server.BaseHTTPRequestHandler.handle_one_request", _broken)
        with RunningSite() as site:
            _reset_connection(site.port)
            assert done.wait(_HANDLER_DONE_SECONDS)
        err = capfd.readouterr().err
        assert "ValueError: the handler is broken" in err
