"""Tests for MCP placement tools (get/preview/set/clear)."""

import pytest

import lilbee.mcp_server as mcp_server
from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
from lilbee.providers.roles import WorkerRole

GIB = 1024**3


def _view(manual=False):
    return PlacementView(
        gpus=(GpuInfo(0, "CUDA", "CUDA0", "NVIDIA A100", 80 * GIB, 72 * GIB),),
        roles=(RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (0,), None, 1),),
        unplaceable=(),
        manual=manual,
        spec_json=None,
    )


def test_get_placement_tool(monkeypatch):
    monkeypatch.setattr(mcp_server, "get_placement", lambda: _view())
    out = mcp_server.get_placement_tool()
    assert out["gpus"][0]["label"] == "CUDA0"
    assert out["manual"] is False


def test_get_gpus_tool(monkeypatch):
    """get_gpus exposes the detected GPUs to agents, matching HTTP /api/gpus (bb-z9w)."""
    monkeypatch.setattr(mcp_server, "get_placement", lambda: _view())
    out = mcp_server.get_gpus_tool()
    assert [g["label"] for g in out["gpus"]] == ["CUDA0"]
    assert out["gpus"][0]["free_bytes"] == 72 * GIB


def test_get_gpus_tool_returns_error_on_provider_failure(monkeypatch):
    from lilbee.providers.base import ProviderError

    def _boom():
        raise ProviderError("engine binary missing", provider="llama-server")

    monkeypatch.setattr(mcp_server, "get_placement", _boom)
    out = mcp_server.get_gpus_tool()
    assert out["error"] == "engine binary missing"


def test_set_placement_tool(monkeypatch):
    seen = {}

    def _capture(spec):
        seen["spec"] = spec
        return _view(True)

    monkeypatch.setattr(mcp_server, "set_placement", _capture)
    out = mcp_server.set_placement_tool({"chat": {"devices": [0]}})
    assert seen["spec"].roles[WorkerRole.CHAT].devices == (0,)
    assert out["manual"] is True


def test_set_placement_tool_unfit_returns_error(monkeypatch):
    from lilbee.providers.fleet.placement_spec import PlacementError

    def boom(spec):
        raise PlacementError("chat needs 70 GiB but device 0 has 40 GiB free")

    monkeypatch.setattr(mcp_server, "set_placement", boom)
    out = mcp_server.set_placement_tool({"chat": {"devices": [0]}})
    assert "error" in out
    assert "40 GiB free" in out["error"]


def test_set_placement_tool_empty_spec_does_not_clear(monkeypatch):
    """set_placement({}) builds an empty spec (rejected downstream), never a clear."""
    from lilbee.providers.fleet.placement_spec import PlacementSpec

    seen = {}

    def _capture(spec):
        seen["spec"] = spec
        return _view(True)

    monkeypatch.setattr(mcp_server, "set_placement", _capture)
    mcp_server.set_placement_tool({})
    assert isinstance(seen["spec"], PlacementSpec)
    assert seen["spec"].roles == {}


def test_clear_placement_tool(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        mcp_server, "set_placement", lambda spec: seen.setdefault("spec", spec) or _view()
    )
    mcp_server.clear_placement_tool()
    assert seen["spec"] is None


def test_set_placement_tool_refused_on_http(monkeypatch):
    """On the shared HTTP transport, set_placement is refused like the REST route."""
    called = {}
    monkeypatch.setattr(
        mcp_server, "set_placement", lambda spec: called.setdefault("spec", spec) or _view(True)
    )
    monkeypatch.setattr(mcp_server.cfg, "allow_http_placement", False)
    mcp_server.set_http_mounted(True)
    try:
        out = mcp_server.set_placement_tool({"chat": {"devices": [0]}})
    finally:
        mcp_server.set_http_mounted(False)
    assert "allow_http_placement" in out["error"]
    assert not called  # the fleet was never rebuilt


def test_clear_placement_tool_refused_on_http(monkeypatch):
    """clear_placement also rebuilds the fleet, so it shares the HTTP gate."""
    called = {}
    monkeypatch.setattr(
        mcp_server, "set_placement", lambda spec: called.setdefault("spec", spec) or _view()
    )
    monkeypatch.setattr(mcp_server.cfg, "allow_http_placement", False)
    mcp_server.set_http_mounted(True)
    try:
        out = mcp_server.clear_placement_tool()
    finally:
        mcp_server.set_http_mounted(False)
    assert "allow_http_placement" in out["error"]
    assert not called


def test_set_placement_tool_allowed_on_http_when_opted_in(monkeypatch):
    """allow_http_placement opts a single-client deployment into HTTP placement."""
    monkeypatch.setattr(mcp_server, "set_placement", lambda spec: _view(True))
    monkeypatch.setattr(mcp_server.cfg, "allow_http_placement", True)
    mcp_server.set_http_mounted(True)
    try:
        out = mcp_server.set_placement_tool({"chat": {"devices": [0]}})
    finally:
        mcp_server.set_http_mounted(False)
    assert out["manual"] is True


def test_preview_placement_tool_auto(monkeypatch):
    monkeypatch.setattr(mcp_server, "preview_placement", lambda spec=None: _view())
    out = mcp_server.preview_placement_tool()
    assert out["gpus"][0]["label"] == "CUDA0"
    assert out["manual"] is False


def test_preview_placement_tool_with_spec(monkeypatch):
    seen = {}

    def _capture(spec=None):
        seen["spec"] = spec
        return _view()

    monkeypatch.setattr(mcp_server, "preview_placement", _capture)
    out = mcp_server.preview_placement_tool({"chat": {"devices": [0]}})
    assert seen["spec"].roles[WorkerRole.CHAT].devices == (0,)
    assert "gpus" in out


def test_preview_placement_tool_error(monkeypatch):
    from lilbee.providers.fleet.placement_spec import PlacementError

    def boom(spec=None):
        raise PlacementError("no GPUs available")

    monkeypatch.setattr(mcp_server, "preview_placement", boom)
    out = mcp_server.preview_placement_tool({"chat": {"devices": [99]}})
    assert "error" in out
    assert "no GPUs available" in out["error"]


def test_get_placement_tool_provider_error(monkeypatch):
    """get_placement surfaces ProviderError as a structured error, not a raw exception."""
    from lilbee.providers.base import ProviderError

    def boom():
        raise ProviderError("llama-server binary not found")

    monkeypatch.setattr(mcp_server, "get_placement", boom)
    out = mcp_server.get_placement_tool()
    assert "error" in out
    assert "llama-server" in out["error"]


def test_clear_placement_tool_provider_error(monkeypatch):
    """clear_placement surfaces ProviderError as a structured error."""
    from lilbee.providers.base import ProviderError

    def boom(spec):
        raise ProviderError("no engine")

    monkeypatch.setattr(mcp_server, "set_placement", boom)
    out = mcp_server.clear_placement_tool()
    assert out.get("error") == "no engine"


@pytest.mark.asyncio
async def test_placement_tools_wire_names():
    """Placement tools must be registered under clean wire names (no _tool suffix)."""
    tools = await mcp_server.mcp.list_tools()
    names = {t.name for t in tools}
    assert "get_placement" in names
    assert "preview_placement" in names
    assert "set_placement" in names
    assert "clear_placement" in names
    assert "get_placement_tool" not in names
    assert "preview_placement_tool" not in names
    assert "set_placement_tool" not in names
    assert "clear_placement_tool" not in names
