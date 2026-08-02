"""The lilbee_search MCP surface must steer models toward retrieval and away
from guessing a wiki scope on corpora that have no wiki."""

from __future__ import annotations

import asyncio

import lilbee.mcp_server as mcp_server
from lilbee.mcp_server import build_mcp_server


def _search_description(server) -> str:
    """The search tool description as shipped on the wire, whitespace collapsed."""
    tools = asyncio.run(server.list_tools())
    desc = next(t.description for t in tools if t.name == "search")
    assert isinstance(desc, str)
    return " ".join(desc.split())


def test_server_instructions_direct_models_to_lilbee_search() -> None:
    # Built-in web/file tools out-compete lilbee_search unless the always-loaded
    # context tells the model to prefer retrieval; the server instructions are
    # that context.
    instructions = build_mcp_server().instructions or ""
    assert "lilbee_search" in instructions
    assert "Prefer it over" in instructions


def test_search_description_prefers_retrieval_over_builtin_tools() -> None:
    desc = _search_description(build_mcp_server())
    assert "prefer it over web-fetch or file-read" in desc


def test_search_description_guides_scope_selection() -> None:
    desc = _search_description(build_mcp_server())
    # The default is recommended and wiki is gated on actually having a wiki.
    assert '"both"' in desc
    assert "wiki" in desc.lower()


def test_scope_hint_warns_when_corpus_has_no_wiki(monkeypatch) -> None:
    # When wiki generation is off, advertise only raw/both so the model stops
    # guessing scope="wiki" (which would silently fall back to the full pool).
    monkeypatch.setattr(mcp_server.cfg, "wiki", False)
    assert "No wiki layer here" in _search_description(build_mcp_server())


def test_scope_hint_absent_when_wiki_enabled(monkeypatch) -> None:
    monkeypatch.setattr(mcp_server.cfg, "wiki", True)
    assert "No wiki layer here" not in _search_description(build_mcp_server())


def test_scope_hint_tracks_config_on_a_live_server(monkeypatch) -> None:
    # The hint is computed per list_tools call, so a config change (vault
    # switch, settings reload) is reflected without any re-tune step.
    server = build_mcp_server()
    monkeypatch.setattr(mcp_server.cfg, "wiki", False)
    assert "No wiki layer here" in _search_description(server)
    monkeypatch.setattr(mcp_server.cfg, "wiki", True)
    assert "No wiki layer here" not in _search_description(server)
