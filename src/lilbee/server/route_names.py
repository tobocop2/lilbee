"""The REST route and the MCP tool that do what a command in an error message does.

The messages in the engine and model code name a lilbee command, because the CLI and the TUI
show them as they are. An HTTP caller has no command line, and an MCP client has none either.
The server adds the route where an error becomes an HTTP response, with `name_routes`. The MCP
server adds the tool where it returns an error or a notice, with `name_tools`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple


class _Route(NamedTuple):
    command: str
    method: str
    path: str
    note: str = ""
    tool: str = ""
    tool_note: str = ""


# There is no entry for `lilbee model rm`: the delete route takes one path segment, so it cannot
# remove a model whose ref holds a slash. The crawler setup has a route and no MCP tool.
_ROUTES: tuple[_Route, ...] = (
    _Route("lilbee model pull", "POST", "/api/models/pull", tool="model_pull"),
    _Route("lilbee model list", "GET", "/api/models/installed", tool="model_list"),
    _Route(
        "lilbee engine install",
        "POST",
        "/api/engines/install",
        ", which needs allow_http_engines",
        tool="engine_install",
        tool_note=", which needs mcp_engines_enabled",
    ),
    _Route(
        "lilbee engine list",
        "GET",
        "/api/engines",
        tool="engine_list",
        tool_note=", which needs mcp_engines_enabled",
    ),
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


def name_tools(text: str) -> str:
    """*text* with a sentence for each command it names, giving the MCP tool that does the same.

    A command with no MCP tool, or whose tool the text already names, gets no sentence.
    """
    sentences = [
        f"Over MCP, call the {r.tool} tool{r.tool_note} in place of '{r.command}'."
        for r in _ROUTES
        if r.tool and r.command in text and r.tool not in text
    ]
    return " ".join([text, *sentences])


def name_damaged_folders(payload: dict[str, Any], namer: Callable[[str], str]) -> dict[str, Any]:
    """The engine list *payload* with *namer* applied to the error of each damaged folder."""
    return {
        **payload,
        "damaged": [{**row, "error": namer(row["error"])} for row in payload["damaged"]],
    }
