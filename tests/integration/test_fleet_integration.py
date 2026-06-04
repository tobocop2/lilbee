"""End-to-end client test: a real llama-server stub subprocess + real httpx.

Spawns the stub server in its own process and drives a real LlamaServerClient
against it over real sockets (the same OpenAI surface llama-swap proxies), then
tears it down. The llama-swap lifecycle itself is covered by test_fleet_swap_manager
and validated against the real binary.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.swap_manager import _pick_free_port

_STUB = Path(__file__).parent / "_llama_server_stub.py"


@pytest.fixture
def stub_client() -> Iterator[LlamaServerClient]:
    """A LlamaServerClient pointed at a freshly spawned stub llama-server."""
    port = _pick_free_port()
    proc = subprocess.Popen([sys.executable, str(_STUB), "--port", str(port)])
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base}/health", timeout=1.0).status_code == httpx.codes.OK:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise RuntimeError("stub server did not become ready")
        yield LlamaServerClient(base, "chat")
    finally:
        proc.terminate()
        proc.wait(timeout=10.0)


def test_client_chat_plain_stream_and_tools(stub_client: LlamaServerClient) -> None:
    assert stub_client.chat([{"role": "user", "content": "hi"}]) == "stub-chat"
    streamed = "".join(stub_client.chat([{"role": "user", "content": "hi"}], stream=True))
    assert streamed == "stub-chat"
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    tool_result = stub_client.chat_tools([{"role": "user", "content": "call it"}], tools=tools)
    assert tool_result.tool_calls[0].name == "lookup"


def test_client_embeds_over_real_http(stub_client: LlamaServerClient) -> None:
    embeds = stub_client.embed(["a", "b"])
    assert len(embeds) == 2
    assert embeds[0] == [0.5, 0.5]
