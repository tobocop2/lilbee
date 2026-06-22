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


def test_clear_placement_tool(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        mcp_server, "set_placement", lambda spec: seen.setdefault("spec", spec) or _view()
    )
    mcp_server.clear_placement_tool()
    assert seen["spec"] is None


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
