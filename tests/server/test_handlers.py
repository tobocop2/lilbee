"""Tests for the /api/add endpoint and SSE progress streaming."""

import asyncio
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litestar.testing import AsyncTestClient

from lilbee.config import cfg
from lilbee.server.handlers import MAX_ADD_FILES


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path):
    """Redirect config paths to temp dir for every test."""
    snapshot = replace(cfg)
    docs = tmp_path / "documents"
    docs.mkdir()
    cfg.documents_dir = docs
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield docs
    for f in fields(cfg):
        setattr(cfg, f.name, getattr(snapshot, f.name))


def _fake_embed_batch(texts: list[str], **kwargs: Any) -> list[list[float]]:
    return [[0.1] * 768 for _ in texts]


def _make_kreuzberg_result(text: str = "Some extracted text. " * 20, num_chunks: int = 1):
    chunks = []
    for i in range(num_chunks):
        chunk_text = text[i * len(text) // num_chunks : (i + 1) * len(text) // num_chunks]
        chunk = mock.MagicMock()
        chunk.content = chunk_text
        chunk.metadata = {"chunk_index": i}
        chunks.append(chunk)
    result = mock.MagicMock()
    result.chunks = chunks
    result.content = text
    return result


def _parse_sse_events(body: bytes) -> list[tuple[str, dict]]:
    """Parse raw SSE bytes into a list of (event_type, data_dict) tuples."""
    events = []
    current_event = ""
    current_data = ""
    for line in body.decode().split("\n"):
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            current_data = line[6:]
        elif line == "" and current_event:
            events.append((current_event, json.loads(current_data)))
            current_event = ""
            current_data = ""
    return events


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
@mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result())
class TestAddEndpoint:
    async def test_add_single_file(self, _kf, _eb, _vm, isolated_env, tmp_path):
        """POST /api/add with a valid file streams SSE events and adds it."""
        from lilbee.server.litestar_app import create_app

        src = tmp_path / "input.txt"
        src.write_text("Hello world content for testing.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": [str(src)]})

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        event_types = [e[0] for e in events]
        assert "file_start" in event_types
        assert "file_done" in event_types
        assert "done" in event_types
        assert "summary" in event_types

    async def test_add_nonexistent_file_in_errors(self, _kf, _eb, _vm, isolated_env, tmp_path):
        """Nonexistent paths appear in the summary errors list."""
        from lilbee.server.litestar_app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": ["/no/such/file.txt"]})

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        summary = next(d for t, d in events if t == "summary")
        assert "/no/such/file.txt" in summary["errors"]

    async def test_add_with_force_flag(self, _kf, _eb, _vm, isolated_env, tmp_path):
        """The force flag allows overwriting existing files."""
        from lilbee.server.litestar_app import create_app

        src = tmp_path / "dup.txt"
        src.write_text("Version 1")
        (isolated_env / "dup.txt").write_text("Existing")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": [str(src)], "force": True})

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        summary = next(d for t, d in events if t == "summary")
        assert "dup.txt" in summary["copied"]

    async def test_done_event_has_correct_fields(self, _kf, _eb, _vm, isolated_env, tmp_path):
        """The done event includes added, updated, removed, failed counts."""
        from lilbee.server.litestar_app import create_app

        src = tmp_path / "doc.txt"
        src.write_text("Content for done event testing.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": [str(src)]})

        events = _parse_sse_events(resp.content)
        done_data = next(d for t, d in events if t == "done")
        assert "added" in done_data
        assert "updated" in done_data
        assert "removed" in done_data
        assert "failed" in done_data

    async def test_file_start_has_total_and_current(self, _kf, _eb, _vm, isolated_env, tmp_path):
        """file_start event includes total_files and current_file."""
        from lilbee.server.litestar_app import create_app

        src = tmp_path / "progress.txt"
        src.write_text("Progress tracking test.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": [str(src)]})

        events = _parse_sse_events(resp.content)
        file_start = next(d for t, d in events if t == "file_start")
        assert file_start["total_files"] >= 1
        assert file_start["current_file"] >= 1

    async def test_add_with_vision_model(self, _kf, _eb, _vm, isolated_env, tmp_path):
        """Vision model parameter is temporarily set on cfg during sync."""
        from lilbee.server.litestar_app import create_app

        src = tmp_path / "doc.txt"
        src.write_text("Content for vision model test.")
        original_vision = cfg.vision_model

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add",
                json={"paths": [str(src)], "vision_model": "test-vision"},
            )

        assert resp.status_code == 201
        # Vision model should be restored after the call
        assert cfg.vision_model == original_vision


class TestAddValidation:
    async def test_empty_paths_returns_400(self, isolated_env):
        """POST /api/add with empty paths list returns 400."""
        from lilbee.server.litestar_app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": []})
        assert resp.status_code == 400

    async def test_missing_paths_returns_400(self, isolated_env):
        """POST /api/add without paths key returns 400."""
        from lilbee.server.litestar_app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"force": True})
        assert resp.status_code == 400

    async def test_too_many_files_returns_400(self, isolated_env):
        """POST /api/add with >100 paths returns 400."""
        from lilbee.server.litestar_app import create_app

        paths = [f"/fake/file_{i}.txt" for i in range(MAX_ADD_FILES + 1)]
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": paths})
        assert resp.status_code == 400

    async def test_exactly_max_files_accepted(self, isolated_env, tmp_path):
        """POST /api/add with exactly 100 paths is accepted (paths can be nonexistent)."""
        from lilbee.server.litestar_app import create_app

        paths = [f"/fake/file_{i}.txt" for i in range(MAX_ADD_FILES)]
        with (
            mock.patch("lilbee.embedder.validate_model"),
            mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch),
            mock.patch(
                "kreuzberg.extract_file",
                new_callable=AsyncMock,
                return_value=_make_kreuzberg_result(),
            ),
        ):
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post("/api/add", json={"paths": paths})
        # Should be 200 (all files nonexistent, but request is valid)
        assert resp.status_code == 201


class TestMakeSseCallback:
    async def test_callback_enqueues_formatted_sse(self):
        """The SSE callback formats events correctly."""
        from lilbee.server.handlers import _make_sse_callback

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        callback = _make_sse_callback(queue)
        callback("file_start", {"file": "test.txt", "total_files": 1, "current_file": 1})
        item = queue.get_nowait()
        assert item is not None
        assert item.startswith("event: file_start\n")
        assert '"file": "test.txt"' in item

    async def test_callback_from_thread_uses_threadsafe(self):
        """When called from a worker thread, uses call_soon_threadsafe."""
        import concurrent.futures

        from lilbee.server.handlers import _make_sse_callback

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        callback = _make_sse_callback(queue)

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = loop.run_in_executor(pool, callback, "embed", {"chunk": 1})
            await future

        # Give the event loop a tick to process the call_soon_threadsafe
        await asyncio.sleep(0)
        item = queue.get_nowait()
        assert item is not None
        assert "embed" in item


class TestSseGenerator:
    async def test_generator_yields_until_sentinel(self):
        """The SSE generator yields items and stops at None."""
        from lilbee.server.handlers import _sse_generator

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        queue.put_nowait("event: test\ndata: {}\n\n")
        queue.put_nowait(None)

        chunks = []
        async for chunk in _sse_generator(queue):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0] == b"event: test\ndata: {}\n\n"


class TestCreateApp:
    def test_app_has_add_route(self):
        """The Litestar app registers the /api/add route."""
        from lilbee.server.litestar_app import create_app

        app = create_app()
        paths = [r.path for r in app.routes]
        assert "/api/add" in paths
