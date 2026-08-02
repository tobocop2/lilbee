"""Unit tests for the MCP sync-handler offload mechanism."""

from __future__ import annotations

import inspect
import threading

import pytest

from lilbee import mcp_server as m


def test_offload_passes_async_handlers_through() -> None:
    async def handler() -> int:
        return 1

    assert m._offload_sync(handler) is handler


async def test_offload_runs_sync_handler_off_the_event_loop() -> None:
    main_thread = threading.current_thread()
    seen: dict[str, threading.Thread] = {}

    def probe(value: int) -> int:
        seen["thread"] = threading.current_thread()
        return value * 2

    wrapped = m._offload_sync(probe)
    assert inspect.iscoroutinefunction(wrapped)

    result = await wrapped(21)

    assert result == 42
    assert seen["thread"] is not main_thread


def test_offload_preserves_name_and_signature() -> None:
    def handler(query: str, top_k: int = 5) -> dict[str, int]:
        return {"n": top_k}

    wrapped = m._offload_sync(handler)

    assert wrapped.__name__ == "handler"
    assert list(inspect.signature(wrapped).parameters) == ["query", "top_k"]


def test_registered_tool_keeps_sync_callable_for_in_process_use() -> None:
    assert not inspect.iscoroutinefunction(m.search)


async def test_registered_tool_schema_preserves_parameters() -> None:
    tools = {tool.name: tool for tool in await m.build_mcp_server().list_tools()}

    properties = tools["search"].input_schema["properties"]

    assert {"query", "top_k", "scope"} <= set(properties)


async def _wire_names() -> set[str]:
    return {t.name for t in await m.build_mcp_server().list_tools()}


async def test_tool_if_false_gate_leaves_function_off_the_wire_but_callable() -> None:
    decorator = m._tool_if(lambda: False)

    def sample_unique_tool_name() -> int:
        return 7

    assert decorator(sample_unique_tool_name) is sample_unique_tool_name
    assert "sample_unique_tool_name" not in await _wire_names()


async def test_tool_if_true_gate_registers_and_returns_original() -> None:
    decorator = m._tool_if(lambda: True)

    def sample_unique_tool_name(value: int) -> int:
        return value

    returned = decorator(sample_unique_tool_name)

    assert returned is sample_unique_tool_name
    assert "sample_unique_tool_name" in await _wire_names()


async def test_tool_if_gate_is_evaluated_per_build() -> None:
    # The gate reflects config at server-build time, not import time, so a
    # server built after a settings change carries the current tool surface.
    enabled = False

    def gated_probe_tool() -> int:
        return 7

    m._tool_if(lambda: enabled)(gated_probe_tool)

    assert "gated_probe_tool" not in await _wire_names()
    enabled = True
    assert "gated_probe_tool" in await _wire_names()


@pytest.fixture(autouse=True)
def _drop_test_tool() -> None:
    """Drop registrations the tests add so later-built servers keep a stable schema."""
    registered = len(m._REGISTRATIONS)
    yield
    del m._REGISTRATIONS[registered:]
