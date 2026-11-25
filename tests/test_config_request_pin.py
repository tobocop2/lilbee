"""A request reads one Config: the one published by its config check, plus its own writes."""

from __future__ import annotations

import asyncio
import contextvars
import os
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from lilbee.app import settings as app_settings
from lilbee.app.config_follow import ConfigFileFollower
from lilbee.app.services import refresh_config, set_config_refresh, set_services
from lilbee.core import settings as persistent_settings
from lilbee.core.config import CONFIG_FILE_NAME, Config, active_config, cfg, config_scope
from lilbee.core.config.context import published_config_scope, request_config_scope
from lilbee.core.config.model import commit_config, published_config
from lilbee.mcp_server import LilbeeMCP, build_mcp_server
from lilbee.server.app import _ConfigFileReload
from tests.conftest import make_mock_services

_MTIME_NS = 1_700_000_000_000_000_000


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    root = tmp_path / "project"
    root.mkdir()
    cfg.data_root = root
    cfg.data_dir = root / "data"
    cfg.documents_dir = root / "documents"
    cfg.lancedb_dir = cfg.data_dir / "lancedb"
    monkeypatch.setenv("LILBEE_DATA", str(root))
    set_services(make_mock_services())
    yield root
    set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _config_path() -> Path:
    return cfg.data_root / CONFIG_FILE_NAME


def _write_from_another_process(key: str, value: Any, tick: int) -> None:
    """Write config.toml the way the CLI does, with an mtime no earlier write had."""
    persistent_settings.set_value(cfg.data_root, key, value)
    stamp = _MTIME_NS + tick * 1_000_000_000
    os.utime(_config_path(), ns=(stamp, stamp))


def _another_process_moves_top_k_and_chunk_size() -> None:
    _write_from_another_process("top_k", 13, tick=2)
    _write_from_another_process("chunk_size", cfg.chunk_size + 64, tick=3)


async def _run_request(app: Any, middleware: _ConfigFileReload | None = None) -> None:
    middleware = middleware or _ConfigFileReload()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request"}

    async def send(message: Any) -> None:
        return None

    await middleware.handle({"type": "http", "path": "/api/x"}, receive, send, app)


def test_a_request_reads_one_config_while_another_request_takes_a_change():
    _write_from_another_process("top_k", 21, tick=1)
    cfg.top_k = 21
    start_chunk_size = cfg.chunk_size
    seen: list[tuple[int, int]] = []
    middleware = _ConfigFileReload()  # the server's one follower, shared by every request

    async def read_twice(scope: Any, receive: Any, send: Any) -> None:
        seen.append((cfg.top_k, cfg.chunk_size))
        _another_process_moves_top_k_and_chunk_size()
        # another request's check on the same follower, as a concurrent request runs it
        await asyncio.to_thread(middleware._follower.take_now)
        seen.append((cfg.top_k, cfg.chunk_size))

    asyncio.run(_run_request(read_twice, middleware))
    assert seen == [(21, start_chunk_size), (21, start_chunk_size)]
    assert (cfg.top_k, cfg.chunk_size) == (13, start_chunk_size + 64)


def test_a_request_reads_its_own_write():
    seen: list[int] = []

    async def write_then_read(scope: Any, receive: Any, send: Any) -> None:
        await asyncio.to_thread(app_settings.apply_settings_update, {"top_k": 17})
        seen.append(cfg.top_k)

    asyncio.run(_run_request(write_then_read))
    assert seen == [17]
    assert cfg.top_k == 17


def test_a_task_a_request_starts_keeps_the_request_config():
    cfg.top_k = 21
    seen: list[int] = []

    async def start_a_sync(scope: Any, receive: Any, send: Any) -> None:
        async def sync() -> None:
            seen.append(cfg.top_k)
            cfg.top_k = 5

        with published_config_scope():
            cfg.top_k = 8  # another request's change, published meanwhile
        await asyncio.create_task(sync())
        seen.append(cfg.top_k)

    asyncio.run(_run_request(start_a_sync))
    assert seen == [21, 5]
    assert cfg.top_k == 5


async def test_a_stdio_tool_reads_one_config_while_another_check_takes_a_change(monkeypatch):
    _write_from_another_process("top_k", 21, tick=1)
    cfg.top_k = 21
    server: LilbeeMCP = build_mcp_server()
    server.follow_config_file()
    follower = server._config_follower
    assert follower is not None
    seen: list[int] = []

    async def read_twice(name: str, *_args: Any, **_kwargs: Any) -> str:
        seen.append(cfg.top_k)
        _another_process_moves_top_k_and_chunk_size()
        await asyncio.to_thread(follower.take_now)
        seen.append(cfg.top_k)
        return "ran"

    monkeypatch.setattr(server._tool_manager, "call_tool", read_twice)
    assert await server.call_tool("settings_get", {}) == "ran"
    assert seen == [21, 21]
    assert cfg.top_k == 13


async def test_an_http_mounted_tool_binds_no_pin_of_its_own(monkeypatch):
    server: LilbeeMCP = build_mcp_server()
    pins: list[Config] = []

    async def record(name: str, *_args: Any, **_kwargs: Any) -> str:
        pins.append(cfg.model_copy())
        return "ran"

    monkeypatch.setattr(server._tool_manager, "call_tool", record)
    cfg.top_k = 9
    with request_config_scope():  # the HTTP middleware's pin
        with published_config_scope():
            cfg.top_k = 4  # another request's change, published meanwhile
        assert await server.call_tool("settings_get", {}) == "ran"
    assert pins[0].top_k == 9


async def test_the_rag_stream_thread_reads_the_request_config(monkeypatch):
    from lilbee.server.handlers import rag

    async def no_frames() -> list[str]:
        return []

    async def no_sources(*_args: Any) -> AsyncIterator[str]:
        for frame in ():
            yield frame

    seen: list[str] = []
    searcher = rag.get_services().searcher
    monkeypatch.setattr(searcher, "search_unavailable", lambda: False)
    monkeypatch.setattr(searcher, "skip_retrieval", lambda: False)
    monkeypatch.setattr(rag, "_warming_frames", no_frames)
    monkeypatch.setattr(rag, "_emit_sources_and_memories", no_sources)
    monkeypatch.setattr(
        rag, "_resolve_stream_context", lambda *_a, **_k: ([], [{"role": "user"}], [])
    )
    monkeypatch.setattr(rag, "_run_llm_stream", lambda *_args: seen.append(cfg.chat_model))
    cfg.chat_model = "pinned/chat"
    pinned = cfg.chat_model
    with request_config_scope():
        with published_config_scope():
            cfg.chat_model = "later/chat"  # another request's change, published meanwhile
        frames = [frame async for frame in rag.ask_stream("q")]
    assert cfg.chat_model != pinned
    assert seen == [pinned], frames


def test_a_refresh_inside_a_request_reads_the_new_file_value():
    # The fleet calls refresh_config before its last retry, to plan on the newest settings.
    _write_from_another_process("top_k", 12, tick=1)
    cfg.top_k = 12
    follower = ConfigFileFollower()
    set_config_refresh(follower.take_now)
    try:
        with request_config_scope():
            _write_from_another_process("top_k", 19, tick=2)
            refresh_config()
            seen = cfg.top_k
    finally:
        set_config_refresh(None)
    assert (cfg.top_k, seen) == (19, 19)


@pytest.mark.parametrize("step", [5, 6], ids=["same_as_other", "different"])
def test_a_request_reads_its_own_write_also_of_a_value_another_request_published(step):
    start = cfg.top_k
    with request_config_scope():
        contextvars.Context().run(setattr, cfg, "top_k", start + 5)  # another request
        assert cfg.top_k == start
        cfg.top_k = start + step
        assert cfg.top_k == start + step


def test_dict_of_cfg_lists_the_fields_it_reads():
    assert dict(cfg) == dict(published_config())


def test_a_write_publishes_a_new_config_and_leaves_the_old_one_whole():
    before = published_config()
    old_top_k = before.top_k
    cfg.top_k = old_top_k + 1
    after = published_config()
    assert after is not before
    assert before.top_k == old_top_k
    assert after.top_k == old_top_k + 1
    assert "top_k" in after.model_fields_set


def test_a_write_that_moves_nothing_publishes_nothing():
    before = published_config()
    cfg.top_k = before.top_k
    assert published_config() is before


def test_a_refused_write_publishes_nothing():
    before = published_config()
    with pytest.raises(ValueError):
        cfg.top_k = "many"
    assert published_config() is before


def test_a_write_from_another_thread_during_a_commit_keeps_its_value():
    start_overlap = cfg.chunk_overlap

    def change(draft: Config) -> None:
        worker = threading.Thread(target=setattr, args=(cfg, "chunk_overlap", start_overlap + 1))
        worker.start()
        worker.join()
        draft.top_k = 31

    commit_config(change)
    assert (cfg.top_k, cfg.chunk_overlap) == (31, start_overlap + 1)
    assert {"top_k", "chunk_overlap"} <= published_config().model_fields_set


def test_a_library_scope_keeps_its_config_when_cfg_is_written():
    own = cfg.model_copy(update={"top_k": 3})
    with config_scope(own):
        cfg.top_k = 44
        assert active_config() is own
        assert cfg.top_k == 44
    assert own.top_k == 3


def test_a_request_scope_reads_through_cfg():
    with request_config_scope():
        assert active_config() is cfg


def test_patch_object_restores_a_field():
    start = cfg.top_k
    with mock.patch.object(cfg, "top_k", start + 2):
        assert cfg.top_k == start + 2
    assert cfg.top_k == start


def test_cfg_shows_the_config_it_reads():
    assert repr(cfg) == repr(published_config())
