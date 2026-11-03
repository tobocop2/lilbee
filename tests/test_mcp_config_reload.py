"""A stdio MCP server takes the config.toml another process wrote before each tool call."""

from __future__ import annotations

import os
from typing import Any

import pytest
from mcp.types import CallToolResult

from lilbee.app import services as svc_mod
from lilbee.app.config_follow import ConfigFileFollower
from lilbee.core import settings as persistent_settings
from lilbee.core.config import CONFIG_FILE_NAME, cfg
from lilbee.core.config.enums import LlmProvider
from lilbee.mcp_server import LilbeeMCP, build_mcp_server, main
from tests import _mcp_client
from tests._mcp_client import mcp_client
from tests.conftest import make_mock_services

_MTIME_NS = 1_700_000_000_000_000_000


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path / "project"
    cfg.data_dir = cfg.data_root / "data"
    cfg.documents_dir = cfg.data_root / "documents"
    cfg.lancedb_dir = cfg.data_dir / "lancedb"
    cfg.data_root.mkdir(parents=True)
    svc_mod.set_services(make_mock_services())
    yield tmp_path
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _stdio_server() -> LilbeeMCP:
    server = build_mcp_server()
    server.follow_config_file()
    return server


def _write_from_another_process(key: str, value: Any, tick: int = 1) -> None:
    """Write config.toml the way the CLI does, with an mtime no earlier write had."""
    persistent_settings.set_value(cfg.data_root, key, value)
    stamp = _MTIME_NS + tick * 1_000_000_000
    os.utime(cfg.data_root / CONFIG_FILE_NAME, ns=(stamp, stamp))


async def _top_k_seen_by_the_tool(server: LilbeeMCP) -> Any:
    result = await server.call_tool("settings_get", {"key": "top_k"})
    assert isinstance(result, CallToolResult), result
    assert not result.is_error, result.content
    payload = result.structured_content
    assert payload is not None, result.content
    return payload["setting"]["value"]


async def test_a_write_between_two_tool_calls_reaches_the_second_call():
    server = _stdio_server()
    first = await _top_k_seen_by_the_tool(server)
    assert first == cfg.top_k
    _write_from_another_process("top_k", first + 5)
    assert await _top_k_seen_by_the_tool(server) == first + 5
    assert cfg.top_k == first + 5


async def test_a_second_write_is_taken_after_the_first():
    server = _stdio_server()
    start = cfg.top_k
    _write_from_another_process("top_k", start + 2, tick=1)
    assert await _top_k_seen_by_the_tool(server) == start + 2
    _write_from_another_process("top_k", start + 9, tick=2)
    assert await _top_k_seen_by_the_tool(server) == start + 9


async def test_a_server_that_does_not_follow_the_file_keeps_its_values():
    server = build_mcp_server()
    start = cfg.top_k
    _write_from_another_process("top_k", start + 5)
    assert await _top_k_seen_by_the_tool(server) == start


async def test_a_file_that_does_not_parse_keeps_the_values_and_the_tool_answers():
    server = _stdio_server()
    start = cfg.top_k
    _write_from_another_process("top_k", start + 3)
    assert await _top_k_seen_by_the_tool(server) == start + 3
    (cfg.data_root / CONFIG_FILE_NAME).write_text("top_k = [unclosed\n", encoding="utf-8")
    stamp = _MTIME_NS + 5_000_000_000
    os.utime(cfg.data_root / CONFIG_FILE_NAME, ns=(stamp, stamp))
    assert await _top_k_seen_by_the_tool(server) == start + 3
    (cfg.data_root / CONFIG_FILE_NAME).write_text(f"top_k = {start + 8}\n", encoding="utf-8")
    stamp = _MTIME_NS + 9_000_000_000
    os.utime(cfg.data_root / CONFIG_FILE_NAME, ns=(stamp, stamp))
    assert await _top_k_seen_by_the_tool(server) == start + 8


async def test_a_reload_that_raises_keeps_the_values_and_the_tool_answers(monkeypatch, caplog):
    server = _stdio_server()
    start = cfg.top_k

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk went away")

    monkeypatch.setattr("lilbee.app.config_follow.adopt_external_changes", broken)
    assert await _top_k_seen_by_the_tool(server) == start
    assert await _top_k_seen_by_the_tool(server) == start
    failures = [r for r in caplog.records if "Could not take" in r.getMessage()]
    assert len(failures) == 1
    _write_from_another_process("top_k", start + 4)
    assert await _top_k_seen_by_the_tool(server) == start
    failures = [r for r in caplog.records if "Could not take" in r.getMessage()]
    assert len(failures) == 2


async def test_no_config_file_takes_no_lock_and_creates_no_file(tmp_path, monkeypatch):
    cfg.data_root = tmp_path / "never-made"
    cfg.data_dir = cfg.data_root / "data"
    locks: list[Any] = []
    real_lock = persistent_settings._config_write_lock

    def spy(root: Any) -> Any:
        locks.append(root)
        return real_lock(root)

    monkeypatch.setattr(persistent_settings, "_config_write_lock", spy)
    server = _stdio_server()
    start = cfg.top_k
    assert await _top_k_seen_by_the_tool(server) == start
    assert await _top_k_seen_by_the_tool(server) == start
    assert locks == []
    assert not cfg.data_root.exists()


async def test_an_unchanged_file_is_not_read_again(monkeypatch):
    _write_from_another_process("top_k", cfg.top_k + 1)
    server = _stdio_server()
    reads: list[Any] = []
    real = persistent_settings.read_persisted

    def spy(root: Any) -> Any:
        reads.append(root)
        return real(root)

    monkeypatch.setattr(persistent_settings, "read_persisted", spy)
    await _top_k_seen_by_the_tool(server)
    await _top_k_seen_by_the_tool(server)
    assert reads == []


async def test_a_provider_switch_in_the_file_waits_for_a_restart(caplog):
    cfg.llm_provider = LlmProvider.AUTO
    server = _stdio_server()
    before = cfg.llm_provider
    _write_from_another_process("llm_provider", LlmProvider.REMOTE.value)
    await _top_k_seen_by_the_tool(server)
    assert cfg.llm_provider == before
    assert [r.getMessage() for r in caplog.records if "restart" in r.getMessage()] == [
        "config.toml now sets llm_provider; restart this process to use it."
    ]


async def test_every_registered_tool_passes_the_config_check(monkeypatch):
    server = _stdio_server()
    names = [tool.name for tool in await server.list_tools()]
    assert len(names) > 20
    checks: list[int] = []
    dispatched: list[str] = []
    follower = server._config_follower
    assert isinstance(follower, ConfigFileFollower)
    real_take = follower.take

    async def counting_take() -> None:
        checks.append(len(dispatched))
        await real_take()

    async def stub_dispatch(name: str, *_args: Any, **_kwargs: Any) -> str:
        dispatched.append(name)
        return "ran"

    monkeypatch.setattr(follower, "take", counting_take)
    monkeypatch.setattr(server._tool_manager, "call_tool", stub_dispatch)
    for name in names:
        assert await server.call_tool(name, {}) == "ran"
    assert dispatched == names
    assert checks == list(range(len(names)))


async def test_a_tool_called_over_the_wire_takes_the_file_first(monkeypatch):
    server = _stdio_server()
    monkeypatch.setattr(_mcp_client, "build_mcp_server", lambda: server)
    start = cfg.top_k
    async with mcp_client() as (session, _init, _drop):
        first = await session.call_tool("settings_get", {"key": "top_k"})
        assert first.structured_content is not None
        assert first.structured_content["setting"]["value"] == start
        _write_from_another_process("top_k", start + 6)
        second = await session.call_tool("settings_get", {"key": "top_k"})
    assert second.structured_content is not None
    assert second.structured_content["setting"]["value"] == start + 6


def test_the_stdio_entry_point_serves_a_server_that_follows_the_file(monkeypatch):
    built = build_mcp_server()
    served: list[LilbeeMCP] = []

    async def fake_serve(server: LilbeeMCP) -> None:
        served.append(server)

    monkeypatch.setattr("lilbee.mcp_server.build_mcp_server", lambda: built)
    monkeypatch.setattr("lilbee.mcp_server._serve_stdio", fake_serve)
    monkeypatch.setattr("lilbee.mcp_server.get_services", lambda: None)
    main()
    assert served == [built]
    assert isinstance(built._config_follower, ConfigFileFollower)


def test_the_http_mount_builds_a_server_that_does_not_follow_the_file():
    assert build_mcp_server()._config_follower is None
