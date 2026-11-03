"""A threaded test web server that stays silent when a client hangs up."""

from __future__ import annotations

import sys
from http.server import ThreadingHTTPServer
from typing import Any


class QuietHTTPServer(ThreadingHTTPServer):
    """A server whose handler threads print nothing when the client closes the connection.

    The crawler closes a connection while the site still reads or writes it. socketserver then
    prints a traceback to ``sys.stderr``. An in-process CLI test reads that stream as the
    command's output, so the traceback lands in front of the JSON line and the line no longer
    starts with a brace. Any other handler error still prints.
    """

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)
