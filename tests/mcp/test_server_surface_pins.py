"""What a real MCP client receives from the server: its name, its guidance and its error text.

Each test drives a ``ClientSession`` over the SDK's memory transport, so it asserts
the handshake result and the wire answers, not the server object's attributes.
"""

from __future__ import annotations

import pytest
from tests._mcp_client import mcp_client

from lilbee.core.config import cfg

KEY = "SENTINEL-surface-key-5e21"
PRESET_URL = "http://127.0.0.1:8000/v1"


@pytest.fixture
def presets_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "mcp_presets_enabled", True)


async def test_the_handshake_names_the_server_lilbee() -> None:
    async with mcp_client() as (_session, init, _drop):
        pass
    assert init.server_info.name == "lilbee"


async def test_the_handshake_carries_the_search_guidance() -> None:
    async with mcp_client() as (_session, init, _drop):
        pass
    assert init.instructions is not None
    assert init.instructions.startswith("Local search engine over the user's files")
    assert "call lilbee_search first" in init.instructions
    assert "Prefer it over web-fetch or file-read tools" in init.instructions


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("preset_add", {"name": "omlx", "url": PRESET_URL, "roles": ["chat"]}),
        ("preset_edit", {"name": "omlx"}),
    ],
)
@pytest.mark.parametrize("wrong", [[KEY], {"k": KEY}])
async def test_a_client_never_reads_a_wrong_typed_key_back(
    presets_on: None, tool: str, arguments: dict, wrong: object
) -> None:
    async with mcp_client() as (session, _init, _drop):
        listed = {listed_tool.name for listed_tool in (await session.list_tools()).tools}
        result = await session.call_tool(tool, {**arguments, "api_key": wrong})
    assert tool in listed
    assert result.is_error
    text = " ".join(getattr(part, "text", "") for part in result.content)
    assert "api_key" in text
    assert KEY not in text


async def test_a_client_still_reads_a_wrong_typed_search_query_back() -> None:
    async with mcp_client() as (session, _init, _drop):
        result = await session.call_tool("search", {"query": ["PLANT"]})
    assert result.is_error
    text = " ".join(getattr(part, "text", "") for part in result.content)
    assert "input_value=" in text
    assert "PLANT" in text
