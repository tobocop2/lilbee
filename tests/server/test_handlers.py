"""Tests for the /api/add endpoint and SSE progress streaming."""

import asyncio
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litestar.testing import AsyncTestClient

from lilbee.config import cfg
from lilbee.server import auth as _auth_mod
from lilbee.server.handlers import MAX_ADD_FILES
from lilbee.services import set_services
from tests.server.conftest import parse_sse_events as _parse_sse_events


def _auth_headers() -> dict[str, str]:
    """Return Authorization header using the current session token."""
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path):
    """Redirect config paths to temp dir for every test."""
    snapshot = cfg.model_copy()
    docs = tmp_path / "documents"
    docs.mkdir()
    cfg.documents_dir = docs
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.concept_graph = False
    yield docs
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def mock_svc():
    """Inject mock Services so handlers never touch real backends."""
    from tests.conftest import make_mock_services

    embedder = mock.MagicMock()
    embedder.embed.return_value = [0.1] * 768
    embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]
    embedder.validate_model.return_value = None
    services = make_mock_services(embedder=embedder)
    set_services(services)
    yield services
    set_services(None)


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


@mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result())
class TestAddEndpoint:
    async def test_add_single_file(self, mock_extract_file, isolated_env, tmp_path):
        """POST /api/add with a valid file streams SSE events and adds it."""
        from lilbee.server.app import create_app

        src = tmp_path / "input.txt"
        src.write_text("Hello world content for testing.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        event_types = [e[0] for e in events]
        assert "file_start" in event_types
        assert "file_done" in event_types
        assert "done" in event_types
        assert "summary" in event_types

    async def test_add_nonexistent_file_in_errors(self, mock_extract_file, isolated_env, tmp_path):
        """Nonexistent paths appear in the summary errors list."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": ["/no/such/file.txt"]}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        summary = next(d for t, d in events if t == "summary")
        assert "/no/such/file.txt" in summary["errors"]

    async def test_add_with_force_flag(self, mock_extract_file, isolated_env, tmp_path):
        """The force flag allows overwriting existing files."""
        from lilbee.server.app import create_app

        src = tmp_path / "dup.txt"
        src.write_text("Version 1")
        (isolated_env / "dup.txt").write_text("Existing")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)], "force": True}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        summary = next(d for t, d in events if t == "summary")
        assert "dup.txt" in summary["copied"]

    async def test_done_event_has_correct_fields(self, mock_extract_file, isolated_env, tmp_path):
        """The done event includes added, updated, removed, failed counts."""
        from lilbee.server.app import create_app

        src = tmp_path / "doc.txt"
        src.write_text("Content for done event testing.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
            )

        events = _parse_sse_events(resp.content)
        done_data = next(d for t, d in events if t == "done")
        assert "added" in done_data
        assert "updated" in done_data
        assert "removed" in done_data
        assert "failed" in done_data

    async def test_file_start_has_total_and_current(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """file_start event includes total_files and current_file."""
        from lilbee.server.app import create_app

        src = tmp_path / "progress.txt"
        src.write_text("Progress tracking test.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
            )

        events = _parse_sse_events(resp.content)
        file_start = next(d for t, d in events if t == "file_start")
        assert file_start["total_files"] >= 1
        assert file_start["current_file"] >= 1

    async def test_add_with_enable_ocr(self, mock_extract_file, isolated_env, tmp_path):
        """enable_ocr parameter is temporarily set on cfg during sync."""
        from lilbee.server.app import create_app

        src = tmp_path / "doc.txt"
        src.write_text("Content for enable_ocr test.")
        original_ocr = cfg.enable_ocr

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add",
                json={"paths": [str(src)], "enable_ocr": True},
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        # enable_ocr should be restored after the call
        assert cfg.enable_ocr == original_ocr


class TestAddValidation:
    async def test_empty_paths_returns_400(self, isolated_env):
        """POST /api/add with empty paths list returns 400."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": []}, headers=_auth_headers())
        assert resp.status_code == 400

    async def test_missing_paths_returns_400(self, isolated_env):
        """POST /api/add without paths key returns 400."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"force": True}, headers=_auth_headers())
        assert resp.status_code == 400

    async def test_too_many_files_returns_400(self, isolated_env):
        """POST /api/add with >100 paths returns 400."""
        from lilbee.server.app import create_app

        paths = [f"/fake/file_{i}.txt" for i in range(MAX_ADD_FILES + 1)]
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": paths}, headers=_auth_headers())
        assert resp.status_code == 400

    async def test_exactly_max_files_accepted(self, isolated_env, tmp_path):
        """POST /api/add with exactly 100 paths is accepted (paths can be nonexistent)."""
        from lilbee.server.app import create_app

        paths = [f"/fake/file_{i}.txt" for i in range(MAX_ADD_FILES)]
        with mock.patch(
            "kreuzberg.extract_file",
            new_callable=AsyncMock,
            return_value=_make_kreuzberg_result(),
        ):
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post("/api/add", json={"paths": paths}, headers=_auth_headers())
        # Should be 200 (all files nonexistent, but request is valid)
        assert resp.status_code == 201


class TestSseStreamCallback:
    async def test_callback_enqueues_formatted_sse(self):
        """The SSE callback formats events correctly."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()
        sse.callback("file_start", {"file": "test.txt", "total_files": 1, "current_file": 1})
        item = sse.queue.get_nowait()
        assert item is not None
        assert item.startswith("event: file_start\n")
        assert '"file": "test.txt"' in item

    async def test_callback_from_thread_uses_threadsafe(self):
        """When called from a worker thread, uses call_soon_threadsafe."""
        import concurrent.futures

        from lilbee.server.handlers import SseStream

        sse = SseStream()

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = loop.run_in_executor(pool, sse.callback, "embed", {"chunk": 1})
            await future

        # Give the event loop a tick to process the call_soon_threadsafe
        await asyncio.sleep(0)
        item = sse.queue.get_nowait()
        assert item is not None
        assert "embed" in item


class TestOptionsPassthrough:
    """Verify generation options are extracted from request body and passed through."""

    async def test_ask_passes_options(self, isolated_env):
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/ask",
                json={"question": "test", "options": {"temperature": 0.3}},
                headers=_auth_headers(),
            )
        assert resp.status_code == 201
        body = resp.json()
        assert "answer" in body

    async def test_chat_passes_options(self, isolated_env):
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "question": "test",
                    "history": [],
                    "options": {"seed": 42},
                },
                headers=_auth_headers(),
            )
        assert resp.status_code == 201
        body = resp.json()
        assert "answer" in body

    async def test_ask_without_options(self, isolated_env):
        """Request without options field still works."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/ask", json={"question": "test"}, headers=_auth_headers())
        assert resp.status_code == 201


class TestCreateApp:
    def test_app_has_add_route(self):
        """The Litestar app registers the /api/add route."""
        from lilbee.server.app import create_app

        app = create_app()
        paths = [r.path for r in app.routes]
        assert "/api/add" in paths
