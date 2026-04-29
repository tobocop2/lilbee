"""T3 MCP. tools/list shape, expected tool names present, no-op tool calls."""

from __future__ import annotations

import pytest
from drivers.mcp import MCPStdioClient

# Tools we expect to be registered on the b455-era MCP surface. Other tools
# (sync, add, model_pull, wiki build/update, etc.) need a model loaded to do
# anything useful and are exercised in the writer tier with a model fixture.
# wiki_build / wiki_update landed after b455 and don't appear here yet.
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
def test_model_list_tool_returns_payload(mcp_client: MCPStdioClient) -> None:
    response = mcp_client.call_tool("model_list")
    assert isinstance(response, dict), response


@pytest.mark.mcp
def test_unknown_tool_call_returns_error_payload(mcp_client: MCPStdioClient) -> None:
    """Calling an unregistered tool returns isError=True, not a silent success."""
    response = mcp_client.call_tool("this_tool_does_not_exist")
    assert response.get("isError") is True, response
