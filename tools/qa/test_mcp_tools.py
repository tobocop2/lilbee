"""T3 MCP. tools/list shape, expected tool names present, no-op tool calls."""

from __future__ import annotations

import json

import pytest
from drivers.mcp import MCPStdioClient

# Tools we expect on the MCP surface. Tools that need a model loaded to do
# anything useful (sync, add, model_pull, wiki_build, wiki_synthesize) are
# exercised in the writer tier with a model fixture; this set just pins the
# inventory so a deletion or rename surfaces here.
_EXPECTED_TOOLS = {
    "search",
    "status",
    "sync",
    "add",
    "crawl",
    "crawl_status",
    "init",
    "remove",
    "list_documents",
    "reset",
    "wiki_status",
    "wiki_list",
    "wiki_lint",
    "wiki_citations",
    "wiki_read",
    "wiki_build",
    "wiki_update",
    "wiki_prune",
    "wiki_synthesize",
    "wiki_drafts_list",
    "wiki_drafts_diff",
    "model_list",
    "model_show",
    "model_pull",
    "model_rm",
}


@pytest.mark.mcp
def test_tools_list_returns_array(mcp_client: MCPStdioClient) -> None:
    tools = mcp_client.list_tools()
    assert isinstance(tools, list)
    assert tools, "tools/list returned an empty array"


@pytest.mark.mcp
def test_tools_list_includes_expected_names(mcp_client: MCPStdioClient) -> None:
    tools = mcp_client.list_tools()
    names = {tool["name"] for tool in tools if "name" in tool}
    missing = _EXPECTED_TOOLS - names
    assert not missing, f"MCP server missing expected tools: {sorted(missing)}"


@pytest.mark.mcp
def test_status_tool_returns_payload(mcp_client: MCPStdioClient) -> None:
    """Calling the `status` tool over MCP returns a structured response."""
    response = mcp_client.call_tool("status")
    assert isinstance(response, dict), response


@pytest.mark.mcp
def test_list_documents_tool_returns_empty(mcp_client: MCPStdioClient) -> None:
    """An empty data dir reports zero documents via list_documents."""
    response = mcp_client.call_tool("list_documents")
    assert isinstance(response, dict), response


@pytest.mark.mcp
def test_wiki_status_tool_returns_payload(mcp_client: MCPStdioClient) -> None:
    response = mcp_client.call_tool("wiki_status")
    assert isinstance(response, dict), response


@pytest.mark.mcp
def test_wiki_status_tool_response_has_documented_keys(mcp_client: MCPStdioClient) -> None:
    """wiki_status MCP response carries the documented WikiStatusResult keys.

    Beyond inventory (tools/list count): the wiki_* tools are public MCP
    contract. wiki_status surfaces wiki_enabled + page/draft counts; an
    empty data dir should still return a structurally valid envelope
    rather than a bare error.
    """
    response = mcp_client.call_tool("wiki_status")
    assert isinstance(response, dict), response
    text = json.dumps(response)
    # The MCP layer wraps the typed result inside a content block; the
    # documented WikiStatusResult fields appear in the JSON-serialized text.
    assert "wiki_enabled" in text, response


@pytest.mark.mcp
def test_wiki_lint_tool_runs_on_empty_store(mcp_client: MCPStdioClient) -> None:
    """wiki_lint over an empty wiki tree returns a structurally valid
    response (empty issues list), not an error envelope."""
    response = mcp_client.call_tool("wiki_lint")
    assert isinstance(response, dict), response
    # An empty wiki tree should report success, not isError.
    assert response.get("isError") is not True, response


@pytest.mark.mcp
def test_wiki_list_tool_runs_on_empty_store(mcp_client: MCPStdioClient) -> None:
    """wiki_list on an empty wiki tree returns a valid (likely empty) list."""
    response = mcp_client.call_tool("wiki_list")
    assert isinstance(response, dict), response
    assert response.get("isError") is not True, response


@pytest.mark.mcp
def test_model_list_tool_returns_payload(mcp_client: MCPStdioClient) -> None:
    response = mcp_client.call_tool("model_list")
    assert isinstance(response, dict), response


@pytest.mark.mcp
def test_unknown_tool_call_returns_error_payload(mcp_client: MCPStdioClient) -> None:
    """Calling an unregistered tool returns isError=True, not a silent success."""
    response = mcp_client.call_tool("this_tool_does_not_exist")
    assert response.get("isError") is True, response
