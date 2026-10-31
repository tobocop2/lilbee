"""Tests for the MCP preset tools, run against the real service and a temp config."""

import json

import pytest

import lilbee.mcp_server as mcp_server
from lilbee.app.presets import engines_refused_message
from lilbee.core import settings
from lilbee.core.config import Config, cfg

KEY = "SENTINEL-mcp-key-9b04"
LOCAL = "http://127.0.0.1:8000/v1"
OTHER = "http://127.0.0.1:9000/v1"


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.server_presets = None
    cfg.allow_http_engines = False
    cfg.mcp_presets_enabled = True
    yield
    mcp_server.set_http_mounted(False)
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def http_mounted():
    mcp_server.set_http_mounted(True)


def _stored() -> list[dict]:
    raw = settings.load(cfg.data_root).get("server_presets")
    return json.loads(raw) if raw else []


PRESET_TOOLS = {"preset_list", "preset_add", "preset_edit", "preset_remove"}


def _wire_names() -> set[str]:
    return {tool.name for tool in mcp_server.build_mcp_server()._tool_manager.list_tools()}


def test_the_four_tools_are_registered_by_name_when_the_setting_is_on():
    assert _wire_names() >= PRESET_TOOLS


def test_the_tools_are_off_the_wire_by_default():
    assert Config.model_fields["mcp_presets_enabled"].default is False
    cfg.mcp_presets_enabled = Config.model_fields["mcp_presets_enabled"].default
    names = _wire_names()
    assert names, "the server registered no tools at all"
    assert names.isdisjoint(PRESET_TOOLS)
    assert "search" in names


def test_settings_set_turns_the_tools_on_after_a_rebuild():
    cfg.mcp_presets_enabled = False
    assert _wire_names().isdisjoint(PRESET_TOOLS)
    out = mcp_server.settings_set({"mcp_presets_enabled": True})
    assert out["updated"] == ["mcp_presets_enabled"]
    assert cfg.mcp_presets_enabled is True
    assert _wire_names() >= PRESET_TOOLS


def test_settings_set_off_refuses_every_tool_of_a_live_server():
    mcp_server.settings_set({"mcp_presets_enabled": False})
    calls = {
        "preset_list": lambda: mcp_server.preset_list_tool(),
        "preset_add": lambda: mcp_server.preset_add_tool("omlx", LOCAL, ["chat"]),
        "preset_edit": lambda: mcp_server.preset_edit_tool("omlx", url=OTHER),
        "preset_remove": lambda: mcp_server.preset_remove_tool("omlx"),
    }
    assert set(calls) == PRESET_TOOLS
    for name, call in calls.items():
        assert call() == {"error": mcp_server.PRESETS_DISABLED_ERROR}, name
    assert _stored() == []


def test_list_empty():
    assert mcp_server.preset_list_tool() == {"presets": []}


def test_add_returns_the_view_and_never_the_key():
    out = mcp_server.preset_add_tool("omlx", LOCAL, ["chat"], api_key=KEY)
    assert out == {
        "name": "omlx",
        "base_url": LOCAL,
        "roles": ["chat"],
        "has_key": True,
        "valid": True,
        "rerank_wire": "v1",
    }
    assert KEY not in json.dumps(mcp_server.preset_list_tool())


def test_add_refusal_is_the_service_text():
    out = mcp_server.preset_add_tool("omlx", LOCAL, ["chatt"])
    assert out["error"].startswith("Unknown role 'chatt'")
    assert _stored() == []


def test_edit_changes_roles_and_wire():
    mcp_server.preset_add_tool("omlx", LOCAL, ["chat"])
    out = mcp_server.preset_edit_tool("omlx", roles=["rerank"], rerank_wire="tei")
    assert (out["roles"], out["rerank_wire"]) == (["rerank"], "tei")


def test_edit_clear_key():
    mcp_server.preset_add_tool("omlx", LOCAL, ["chat"], api_key=KEY)
    assert mcp_server.preset_edit_tool("omlx", clear_key=True)["has_key"] is False


def test_edit_missing_preset():
    assert mcp_server.preset_edit_tool("nope", url=LOCAL) == {
        "error": "No server preset is named 'nope'."
    }


def test_remove():
    mcp_server.preset_add_tool("omlx", LOCAL, ["chat"])
    assert mcp_server.preset_remove_tool("omlx") == {"removed": "omlx"}
    assert _stored() == []


def test_remove_missing_preset():
    assert "error" in mcp_server.preset_remove_tool("nope")


def test_a_locked_config_file_returns_the_write_failure_text(monkeypatch):
    def locked(*args, **kwargs):
        raise OSError("locked")

    monkeypatch.setattr(settings, "mutate_value", locked)
    assert "locked" in mcp_server.preset_add_tool("omlx", LOCAL, ["chat"])["error"]


def test_each_write_is_refused_on_http_when_the_flag_is_off(http_mounted):
    cfg.allow_http_engines = True
    mcp_server.preset_add_tool("omlx", LOCAL, ["chat"], api_key=KEY)
    cfg.allow_http_engines = False
    before = _stored()
    answers = [
        mcp_server.preset_add_tool("second", LOCAL, ["chat"]),
        mcp_server.preset_edit_tool("omlx", roles=["embed"]),
        mcp_server.preset_remove_tool("omlx"),
    ]
    assert answers == [{"error": engines_refused_message()}] * 3
    assert _stored() == before


def test_list_is_open_on_http_when_the_flag_is_off(http_mounted):
    assert mcp_server.preset_list_tool() == {"presets": []}


def test_writes_are_open_off_http_when_the_flag_is_off():
    assert "error" not in mcp_server.preset_add_tool("omlx", LOCAL, ["chat"])
