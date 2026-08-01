"""End-to-end conformance: a real MCP client against the real lilbee server.

The rest of the MCP suite calls the server's methods in-process or drives the
mount with hand-built JSON. This drives a genuine ``ClientSession`` over the
SDK's memory transport instead, so the handshake, the tools/list wire encoding
and the call/response envelopes all run through the client half of the SDK.
That is the layer the mcp 2.x migration moved, and it is the layer an agent
actually speaks.

Every tool exercised here is chosen to need no services: an engine start in the
unit suite would leak llama-server processes onto the developer's machine.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from mcp.shared.memory import create_client_server_memory_streams

from lilbee.mcp_server import _offload_sync, build_mcp_server
from mcp import ClientSession

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Present on every build regardless of config gating (wiki, sessions and memory
# tools come and go with settings, so they are not asserted here).
_CORE_TOOLS = frozenset(
    {"search", "sync", "add", "remove", "status", "list_documents", "settings_list"}
)


@asynccontextmanager
async def _client(server: Any = None) -> AsyncIterator[tuple[ClientSession, Any]]:
    """A connected client session and the result of its initialize handshake."""
    server = server if server is not None else build_mcp_server()
    lowlevel = server._lowlevel_server
    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        anyio.create_task_group() as tg,
    ):

        async def _serve() -> None:
            await lowlevel.run(
                server_streams[0],
                server_streams[1],
                lowlevel.create_initialization_options(),
                raise_exceptions=False,
            )

        tg.start_soon(_serve)
        async with ClientSession(client_streams[0], client_streams[1]) as session:
            yield session, await session.initialize()
        tg.cancel_scope.cancel()


async def test_the_handshake_identifies_lilbee() -> None:
    async with _client() as (_session, init):
        pass
    assert init.server_info.name == "lilbee"


async def test_the_core_tools_reach_a_real_client() -> None:
    async with _client() as (session, _init):
        names = {tool.name for tool in (await session.list_tools()).tools}
    assert names >= _CORE_TOOLS


async def test_the_wire_schema_stays_trimmed_through_the_client() -> None:
    """The strip runs in list_tools, so only a real client sees the wire form.

    Guards the schema attribute the 2.x rename moved: reading the wrong one
    would surface here as untrimmed titles rather than as an import error.
    """
    async with _client() as (session, _init):
        tools = (await session.list_tools()).tools
    for tool in tools:
        schema = tool.input_schema
        assert "title" not in schema, f"{tool.name}: top-level title on the wire"
        for prop_name, prop in schema.get("properties", {}).items():
            assert "title" not in prop, f"{tool.name}.{prop_name}: property title on the wire"
            assert "default" not in prop, f"{tool.name}.{prop_name}: default on the wire"


async def test_tool_descriptions_arrive_flattened() -> None:
    async with _client() as (session, _init):
        tools = (await session.list_tools()).tools
    for tool in tools:
        if tool.description:
            assert "\n  " not in tool.description, f"{tool.name}: unflattened indentation"


async def test_a_tool_call_round_trips_a_result() -> None:
    async with _client() as (session, _init):
        result = await session.call_tool("settings_list", {})
    assert not result.is_error
    assert result.content


async def test_a_refused_call_round_trips_the_error_envelope() -> None:
    """The envelope is a normal result, not a protocol error, so agents can read it."""
    async with _client() as (session, _init):
        result = await session.call_tool("search", {"query": "   "})
    assert not result.is_error
    assert "query must not be empty" in str(result.content)


async def test_an_unknown_tool_is_a_protocol_error() -> None:
    async with _client() as (session, _init):
        result = await session.call_tool("no_such_tool", {})
    assert result.is_error


@pytest.mark.xfail(
    reason="offloaded sync handlers ignore cancellation; the work runs to completion",
    strict=True,
)
async def test_a_cancelled_sync_tool_stops_working() -> None:
    """Pins that cancelling an agent's tool call does not stop the work.

    Sync handlers run on a worker thread through anyio.to_thread.run_sync, which
    neither interrupts the thread nor returns early, so a cancelled extraction
    keeps going. A pure-async handler does honour the cancellation, so the SDK
    and the protocol are not what break this. Flip to a passing test when the
    MCP tools take the cancel token the ingest pipeline already accepts.
    """
    state = {"ticks": 0}

    def slow(seconds: float) -> str:
        for _ in range(int(seconds * 10)):
            time.sleep(0.1)
            state["ticks"] += 1
        return "done"

    server = build_mcp_server()
    server.add_tool(_offload_sync(slow), name="slow")
    async with _client(server) as (session, _init):
        with anyio.move_on_after(0.3):
            await session.call_tool("slow", {"seconds": 1.0})
        at_cancel = state["ticks"]
        await anyio.sleep(1.2)
    assert state["ticks"] == at_cancel, "work continued after the client cancelled"
