"""Tests for the MCP server tools."""

from dataclasses import fields, replace
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from lilbee.config import cfg
from lilbee.ingest import SyncResult


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all MCP tests."""
    snapshot = replace(cfg)

    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"

    yield tmp_path

    for f in fields(cfg):
        setattr(cfg, f.name, getattr(snapshot, f.name))


@pytest.fixture(autouse=True)
def _skip_model_validation():
    """MCP tests never need real Ollama model validation."""
    with mock.patch("lilbee.embedder.validate_model"):
        yield


_SYNC_NOOP = SyncResult()


class TestClean:
    def test_strips_vector(self):
        from lilbee.mcp import clean

        result = clean({"source": "a.pdf", "vector": [0.1], "chunk": "hi"})
        assert "vector" not in result
        assert result["source"] == "a.pdf"

    def test_renames_distance(self):
        from lilbee.mcp import clean

        result = clean({"_distance": 0.42, "chunk": "hi"})
        assert result["distance"] == 0.42
        assert "_distance" not in result


class TestLilbeeSearch:
    @mock.patch("lilbee.query.search_context")
    def test_returnscleaned_results(self, mock_search):
        from lilbee.mcp import lilbee_search

        mock_search.return_value = [
            {"source": "doc.pdf", "chunk": "content", "_distance": 0.3, "vector": [0.1] * 768},
        ]
        results = lilbee_search("test query", top_k=3)
        assert len(results) == 1
        assert "vector" not in results[0]
        assert results[0]["distance"] == 0.3
        mock_search.assert_called_once_with("test query", top_k=3)

    @mock.patch("lilbee.query.search_context", return_value=[])
    def test_empty_results(self, _search):
        from lilbee.mcp import lilbee_search

        assert lilbee_search("nothing") == []


class TestLilbeeStatus:
    def test_empty_status(self):
        from lilbee.mcp import lilbee_status

        result = lilbee_status()
        assert "config" in result
        assert result["sources"] == []
        assert result["total_chunks"] == 0

    def test_with_sources(self):
        from lilbee.mcp import lilbee_status
        from lilbee.store import upsert_source

        upsert_source("test.pdf", "abc123", 10)
        result = lilbee_status()
        assert len(result["sources"]) == 1
        assert result["sources"][0]["filename"] == "test.pdf"
        assert result["total_chunks"] == 10


class TestLilbeeSync:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_sync_empty(self, _sync):
        from lilbee.mcp import lilbee_sync

        result = await lilbee_sync()
        assert result["added"] == []
        assert result["unchanged"] == 0

    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(added=["test.txt"]),
    )
    async def test_sync_with_file(self, _sync):
        from lilbee.mcp import lilbee_sync

        (cfg.documents_dir / "test.txt").write_text("Hello world content.")
        result = await lilbee_sync()
        assert "test.txt" in result["added"]


class TestLilbeeReset:
    def test_reset_clears_everything(self):
        from lilbee.mcp import lilbee_reset

        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        (cfg.documents_dir / "doc.txt").write_text("content")
        (cfg.data_dir / "db_file").write_text("data")

        result = lilbee_reset()
        assert result["command"] == "reset"
        assert result["deleted_docs"] == 1
        assert result["deleted_data"] == 1
        assert list(cfg.documents_dir.iterdir()) == []
        assert list(cfg.data_dir.iterdir()) == []

    def test_reset_empty_dirs(self):
        from lilbee.mcp import lilbee_reset

        result = lilbee_reset()
        assert result["command"] == "reset"
        assert result["deleted_docs"] == 0
        assert result["deleted_data"] == 0


class TestLilbeeAdd:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_single_file(self, _sync, tmp_path):
        from lilbee.mcp import lilbee_add

        src = tmp_path / "test.txt"
        src.write_text("hello world")

        result = await lilbee_add([str(src)])

        assert result["command"] == "add"
        assert "test.txt" in result["copied"]
        assert result["errors"] == []
        assert result["skipped"] == []
        assert (cfg.documents_dir / "test.txt").read_text() == "hello world"
        _sync.assert_awaited_once()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_nonexistent_path(self, _sync, tmp_path):
        from lilbee.mcp import lilbee_add

        result = await lilbee_add(["/no/such/path.txt"])

        assert "/no/such/path.txt" in result["errors"]
        assert result["copied"] == []

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_existing_no_force(self, _sync, tmp_path):
        from lilbee.mcp import lilbee_add

        (cfg.documents_dir / "exist.txt").write_text("old")
        src = tmp_path / "exist.txt"
        src.write_text("new")

        result = await lilbee_add([str(src)])

        assert "exist.txt" in result["skipped"]
        assert result["copied"] == []
        assert (cfg.documents_dir / "exist.txt").read_text() == "old"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_existing_with_force(self, _sync, tmp_path):
        from lilbee.mcp import lilbee_add

        (cfg.documents_dir / "exist.txt").write_text("old")
        src = tmp_path / "exist.txt"
        src.write_text("new")

        result = await lilbee_add([str(src)], force=True)

        assert "exist.txt" in result["copied"]
        assert result["skipped"] == []
        assert (cfg.documents_dir / "exist.txt").read_text() == "new"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_directory(self, _sync, tmp_path):
        from lilbee.mcp import lilbee_add

        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("a")

        result = await lilbee_add([str(src_dir)])

        assert "mydir" in result["copied"]
        assert (cfg.documents_dir / "mydir" / "a.txt").read_text() == "a"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_with_vision_model(self, mock_sync, tmp_path):
        from lilbee.mcp import lilbee_add

        src = tmp_path / "scan.pdf"
        src.write_bytes(b"%PDF-fake")

        await lilbee_add([str(src)], vision_model="test-vision:latest")

        # vision_model should be restored after the call
        assert getattr(cfg, "vision_model", "") == ""

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=RuntimeError("boom"))
    async def test_add_vision_model_restored_on_error(self, _sync, tmp_path):
        from lilbee.mcp import lilbee_add

        src = tmp_path / "file.txt"
        src.write_text("content")

        with pytest.raises(RuntimeError, match="boom"):
            await lilbee_add([str(src)], vision_model="test-vision:latest")

        assert getattr(cfg, "vision_model", "") == ""

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_empty_paths(self, _sync):
        from lilbee.mcp import lilbee_add

        result = await lilbee_add([])

        assert result["copied"] == []
        assert result["skipped"] == []
        assert result["errors"] == []


class TestMain:
    @mock.patch("lilbee.mcp.mcp")
    def test_main_calls_run(self, mock_mcp):
        from lilbee.mcp import main

        main()
        mock_mcp.run.assert_called_once()


class TestMcpSubcommand:
    @mock.patch("lilbee.mcp.main")
    def test_mcp_subcommand(self, mock_main):
        from typer.testing import CliRunner

        from lilbee.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["mcp"])
        assert result.exit_code == 0
        mock_main.assert_called_once()
