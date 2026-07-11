"""The lilbee_search MCP surface must steer models toward retrieval and away
from guessing a wiki scope on corpora that have no wiki."""

from __future__ import annotations

import lilbee.mcp_server as mcp_server
from lilbee.mcp_server import _tune_search_scope_for_corpus, mcp


def _search_description() -> str:
    """The search tool description with whitespace collapsed, since the docstring
    wraps across indented lines on the wire."""
    desc = mcp._tool_manager._tools["search"].description
    assert isinstance(desc, str)
    return " ".join(desc.split())


def test_server_instructions_direct_models_to_lilbee_search() -> None:
    # Built-in web/file tools out-compete lilbee_search unless the always-loaded
    # context tells the model to prefer retrieval; the FastMCP instructions are
    # that context.
    instructions = mcp.instructions or ""
    assert "lilbee_search" in instructions
    assert "Prefer it over" in instructions


def test_search_description_prefers_retrieval_over_builtin_tools() -> None:
    desc = _search_description()
    assert "prefer it over web-fetch or file-read" in desc


def test_search_description_guides_scope_selection() -> None:
    desc = _search_description()
    # The default is recommended and wiki is gated on actually having a wiki.
    assert '"both"' in desc
    assert "wiki" in desc.lower()


def test_scope_hint_warns_when_corpus_has_no_wiki(monkeypatch) -> None:
    # When wiki generation is off, advertise only raw/both so the model stops
    # guessing scope="wiki" (which would silently fall back to the full pool).
    info = mcp._tool_manager._tools["search"]
    original = info.description
    monkeypatch.setattr(mcp_server.cfg, "wiki", False)
    try:
        _tune_search_scope_for_corpus()
        assert isinstance(info.description, str)
        assert "No wiki layer here" in info.description
    finally:
        info.description = original


def test_scope_hint_absent_when_wiki_enabled(monkeypatch) -> None:
    info = mcp._tool_manager._tools["search"]
    original = info.description
    monkeypatch.setattr(mcp_server.cfg, "wiki", True)
    try:
        _tune_search_scope_for_corpus()
        assert "No wiki layer here" not in (info.description or "")
    finally:
        info.description = original


def test_scope_hint_is_stripped_when_wiki_turns_on(monkeypatch) -> None:
    # A config reload that enables wiki must remove a previously-added hint so
    # the model is told the wiki scope is now available (the reversible branch).
    info = mcp._tool_manager._tools["search"]
    original = info.description
    try:
        monkeypatch.setattr(mcp_server.cfg, "wiki", False)
        _tune_search_scope_for_corpus()
        assert "No wiki layer here" in info.description

        monkeypatch.setattr(mcp_server.cfg, "wiki", True)
        _tune_search_scope_for_corpus()
        assert "No wiki layer here" not in info.description
    finally:
        info.description = original


def test_tune_is_noop_when_search_tool_absent() -> None:
    # If the search tool isn't registered, tuning returns early instead of
    # dereferencing a missing tool.
    tools = mcp._tool_manager._tools
    saved = tools.pop("search")
    try:
        _tune_search_scope_for_corpus()  # must not raise; returns early
    finally:
        tools["search"] = saved
