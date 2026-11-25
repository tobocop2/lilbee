"""The REST route that does what a command in an error message does.

The messages in the engine and model code name a lilbee command, because the CLI and the TUI
show them as they are. An HTTP caller has no command line. The server adds the route where an
error becomes an HTTP response, with `name_routes`.
"""

from __future__ import annotations

from typing import NamedTuple


class _Route(NamedTuple):
    command: str
    method: str
    path: str
    note: str = ""


# There is no entry for `lilbee model rm`: the delete route takes one path segment, so it cannot
# remove a model whose ref holds a slash.
_ROUTES: tuple[_Route, ...] = (
    _Route("lilbee model pull", "POST", "/api/models/pull"),
    _Route("lilbee model list", "GET", "/api/models/installed"),
    _Route(
        "lilbee engine install", "POST", "/api/engines/install", ", which needs allow_http_engines"
    ),
    _Route("lilbee engine list", "GET", "/api/engines"),
    _Route("lilbee setup crawler", "POST", "/setup/crawler"),
)


def name_routes(text: str) -> str:
    """*text* with a sentence for each command it names, giving the route that does the same.

    A command whose route the text already names gets no sentence.
    """
    sentences = [
        f"Over HTTP, call {r.method} {r.path}{r.note} in place of '{r.command}'."
        for r in _ROUTES
        if r.command in text and r.path not in text
    ]
    return " ".join([text, *sentences])
