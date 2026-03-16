"""Tests for the MCP server tools."""

from unittest import mock
from unittest.mock import AsyncMock

import pytest

from lilbee.config import cfg
from lilbee.ingest import SyncResult
from lilbee.mcp import (
    clean,
    lilbee_add,
    lilbee_init,
    lilbee_reset,
    lilbee_search,
    lilbee_status,
    lilbee_sync,
    main,
)
from lilbee.store import SearchChunk


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all MCP tests."""
    snapshot = cfg.model_copy()

    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"

    yield tmp_path

    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def _skip_model_validation():
    """MCP tests never need real Ollama model validation."""
    with mock.patch("lilbee.embedder.validate_model"):
        yield


_SYNC_NOOP = SyncResult()


class TestClean:
    def test_strips_vector(self):
        chunk = SearchChunk(
            source="a.pdf",
            content_type="text",
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk="hi",
            chunk_index=0,
            vector=[0.1],
            distance=0.5,
        )
        result = clean(chunk)
        assert "vector" not in result
        assert result["source"] == "a.pdf"

    def test_has_distance(self):
        chunk = SearchChunk(
            source="a.pdf",
            content_type="text",
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk="hi",
            chunk_index=0,
            vector=[0.1],
            distance=0.42,
        )
        result = clean(chunk)
        assert result["distance"] == 0.42


class TestLilbeeSearch:
    @mock.patch("lilbee.query.search_context")
    def test_returnscleaned_results(self, mock_search):
        mock_search.return_value = [
            SearchChunk(
                source="doc.pdf",
                content_type="pdf",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="content",
                chunk_index=0,
                vector=[0.1] * 768,
                distance=0.3,
            ),
        ]
        results = lilbee_search("test query", top_k=3)
        assert len(results) == 1
        assert "vector" not in results[0]
        assert results[0]["distance"] == 0.3
        mock_search.assert_called_once_with("test query", top_k=3)

    @mock.patch("lilbee.query.search_context", return_value=[])
    def test_empty_results(self, _search):
        assert lilbee_search("nothing") == []


class TestLilbeeStatus:
    def test_empty_status(self):
        result = lilbee_status()
        assert "config" in result
        assert result["sources"] == []
        assert result["total_chunks"] == 0

    def test_with_sources(self):
        from lilbee.store import upsert_source

        upsert_source("test.pdf", "abc123", 10)
        result = lilbee_status()
        assert len(result["sources"]) == 1
        assert result["sources"][0]["filename"] == "test.pdf"
        assert result["total_chunks"] == 10

    def test_status_includes_vision_model_when_set(self):
        cfg.vision_model = "test-vision:latest"
        result = lilbee_status()
        assert result["config"]["vision_model"] == "test-vision:latest"

    def test_status_excludes_vision_model_when_empty(self):
        cfg.vision_model = ""
        result = lilbee_status()
        assert "vision_model" not in result["config"]


class TestLilbeeSync:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_sync_empty(self, _sync):
        result = await lilbee_sync()
        assert result["added"] == []
        assert result["unchanged"] == 0

    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(added=["test.txt"]),
    )
    async def test_sync_with_file(self, _sync):
        (cfg.documents_dir / "test.txt").write_text("Hello world content.")
        result = await lilbee_sync()
        assert "test.txt" in result["added"]


class TestLilbeeReset:
    def test_reset_clears_everything(self):
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
        result = lilbee_reset()
        assert result["command"] == "reset"
        assert result["deleted_docs"] == 0
        assert result["deleted_data"] == 0


class TestLilbeeInit:
    def test_init_creates_structure(self, tmp_path):
        result = lilbee_init(str(tmp_path))
        root = tmp_path / ".lilbee"
        assert result["command"] == "init"
        assert result["created"] is True
        assert root.is_dir()
        assert (root / "documents").is_dir()
        assert (root / "data").is_dir()
        assert (root / ".gitignore").read_text() == "data/\n"

    def test_init_already_exists(self, tmp_path):
        (tmp_path / ".lilbee").mkdir()
        result = lilbee_init(str(tmp_path))
        assert result["created"] is False

    def test_init_default_cwd(self, tmp_path):
        with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            result = lilbee_init()
        assert result["created"] is True
        assert (tmp_path / ".lilbee" / "documents").is_dir()


class TestLilbeeAdd:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_single_file(self, _sync, tmp_path):
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
        result = await lilbee_add(["/no/such/path.txt"])

        assert "/no/such/path.txt" in result["errors"]
        assert result["copied"] == []

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_existing_no_force(self, _sync, tmp_path):
        (cfg.documents_dir / "exist.txt").write_text("old")
        src = tmp_path / "exist.txt"
        src.write_text("new")

        result = await lilbee_add([str(src)])

        assert "exist.txt" in result["skipped"]
        assert result["copied"] == []
        assert (cfg.documents_dir / "exist.txt").read_text() == "old"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_existing_with_force(self, _sync, tmp_path):
        (cfg.documents_dir / "exist.txt").write_text("old")
        src = tmp_path / "exist.txt"
        src.write_text("new")

        result = await lilbee_add([str(src)], force=True)

        assert "exist.txt" in result["copied"]
        assert result["skipped"] == []
        assert (cfg.documents_dir / "exist.txt").read_text() == "new"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_directory(self, _sync, tmp_path):
        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("a")

        result = await lilbee_add([str(src_dir)])

        assert "mydir" in result["copied"]
        assert (cfg.documents_dir / "mydir" / "a.txt").read_text() == "a"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_with_vision_model(self, mock_sync, tmp_path):
        src = tmp_path / "scan.pdf"
        src.write_bytes(b"%PDF-fake")
        original_vision = cfg.vision_model

        await lilbee_add([str(src)], vision_model="test-vision:latest")

        # vision_model should be restored after the call
        assert cfg.vision_model == original_vision

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=RuntimeError("boom"))
    async def test_add_vision_model_restored_on_error(self, _sync, tmp_path):
        src = tmp_path / "file.txt"
        src.write_text("content")
        original_vision = cfg.vision_model

        with pytest.raises(RuntimeError, match="boom"):
            await lilbee_add([str(src)], vision_model="test-vision:latest")

        assert cfg.vision_model == original_vision

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_empty_paths(self, _sync):
        result = await lilbee_add([])

        assert result["copied"] == []
        assert result["skipped"] == []
        assert result["errors"] == []


class TestMain:
    @mock.patch("lilbee.mcp.mcp")
    def test_main_calls_run(self, mock_mcp):
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
