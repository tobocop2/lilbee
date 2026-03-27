"""Tests for the MCP server tools."""

from unittest import mock
from unittest.mock import AsyncMock

import pytest

from lilbee.config import cfg
from lilbee.crawl_task import clear_tasks
from lilbee.ingest import SyncResult
from lilbee.mcp import (
    clean,
    lilbee_add,
    lilbee_crawl,
    lilbee_crawl_status,
    lilbee_init,
    lilbee_list_documents,
    lilbee_remove,
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
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"

    yield tmp_path

    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def _no_dns():
    """Bypass SSRF DNS resolution in all MCP tests."""
    with mock.patch(
        "lilbee.crawler.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("93.184.216.34", 0))],
    ):
        yield


@pytest.fixture(autouse=True)
def _skip_model_validation():
    """MCP tests never need real model validation."""
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
    def test_empty_results(self, mock_search):
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
    async def test_sync_empty(self, mock_sync):
        result = await lilbee_sync()
        assert result["added"] == []
        assert result["unchanged"] == 0

    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(added=["test.txt"]),
    )
    async def test_sync_with_file(self, mock_sync):
        (cfg.documents_dir / "test.txt").write_text("Hello world content.")
        result = await lilbee_sync()
        assert "test.txt" in result["added"]


class TestLilbeeRemove:
    @mock.patch("lilbee.mcp.get_sources")
    @mock.patch("lilbee.mcp.delete_source")
    @mock.patch("lilbee.mcp.delete_by_source")
    def test_removes_known_file(self, mock_del, mock_del_src, mock_sources):
        mock_sources.return_value = [{"filename": "a.md"}]
        result = lilbee_remove(["a.md"])
        assert result["removed"] == ["a.md"]
        assert result["not_found"] == []

    @mock.patch("lilbee.mcp.get_sources")
    def test_not_found(self, mock_sources):
        mock_sources.return_value = []
        result = lilbee_remove(["missing.md"])
        assert result["not_found"] == ["missing.md"]

    @mock.patch("lilbee.mcp.get_sources")
    @mock.patch("lilbee.mcp.delete_source")
    @mock.patch("lilbee.mcp.delete_by_source")
    def test_delete_files_removes_from_disk(self, mock_del, mock_del_src, mock_sources):
        mock_sources.return_value = [{"filename": "a.md"}]
        f = cfg.documents_dir / "a.md"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("content")
        result = lilbee_remove(["a.md"], delete_files=True)
        assert result["removed"] == ["a.md"]
        assert not f.exists()


class TestLilbeeListDocuments:
    @mock.patch("lilbee.mcp.get_sources")
    def test_returns_documents(self, mock_sources):
        mock_sources.return_value = [{"filename": "a.md", "chunk_count": 3}]
        result = lilbee_list_documents()
        assert result["total"] == 1
        assert result["documents"][0]["filename"] == "a.md"

    @mock.patch("lilbee.mcp.get_sources")
    def test_empty(self, mock_sources):
        mock_sources.return_value = []
        result = lilbee_list_documents()
        assert result["total"] == 0


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
        with mock.patch("pathlib.Path.home", return_value=tmp_path.parent):
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
        with mock.patch("pathlib.Path.home", return_value=tmp_path.parent):
            result = lilbee_init(str(tmp_path))
        assert result["created"] is False

    def test_init_default_cwd(self, tmp_path):
        with (
            mock.patch("pathlib.Path.cwd", return_value=tmp_path),
            mock.patch("pathlib.Path.home", return_value=tmp_path.parent),
        ):
            result = lilbee_init()
        assert result["created"] is True
        assert (tmp_path / ".lilbee" / "documents").is_dir()

    def test_init_outside_home_rejected(self, tmp_path):
        with mock.patch("pathlib.Path.home", return_value=tmp_path / "fakehome"):
            result = lilbee_init(str(tmp_path))
        assert "error" in result


class TestLilbeeAdd:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_single_file(self, mock_sync, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("hello world")

        result = await lilbee_add([str(src)])

        assert result["command"] == "add"
        assert "test.txt" in result["copied"]
        assert result["errors"] == []
        assert result["skipped"] == []
        assert (cfg.documents_dir / "test.txt").read_text() == "hello world"
        mock_sync.assert_awaited_once()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_nonexistent_path(self, mock_sync, tmp_path):
        result = await lilbee_add(["/no/such/path.txt"])

        assert "/no/such/path.txt" in result["errors"]
        assert result["copied"] == []

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_existing_no_force(self, mock_sync, tmp_path):
        (cfg.documents_dir / "exist.txt").write_text("old")
        src = tmp_path / "exist.txt"
        src.write_text("new")

        result = await lilbee_add([str(src)])

        assert "exist.txt" in result["skipped"]
        assert result["copied"] == []
        assert (cfg.documents_dir / "exist.txt").read_text() == "old"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_existing_with_force(self, mock_sync, tmp_path):
        (cfg.documents_dir / "exist.txt").write_text("old")
        src = tmp_path / "exist.txt"
        src.write_text("new")

        result = await lilbee_add([str(src)], force=True)

        assert "exist.txt" in result["copied"]
        assert result["skipped"] == []
        assert (cfg.documents_dir / "exist.txt").read_text() == "new"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_directory(self, mock_sync, tmp_path):
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
    async def test_add_vision_model_restored_on_error(self, mock_sync, tmp_path):
        src = tmp_path / "file.txt"
        src.write_text("content")
        original_vision = cfg.vision_model

        with pytest.raises(RuntimeError, match="boom"):
            await lilbee_add([str(src)], vision_model="test-vision:latest")

        assert cfg.vision_model == original_vision

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_empty_paths(self, mock_sync):
        result = await lilbee_add([])

        assert result["copied"] == []
        assert result["skipped"] == []
        assert result["errors"] == []


class TestMain:
    @mock.patch("lilbee.mcp.mcp")
    def test_main_calls_run(self, mock_mcp):
        main()
        mock_mcp.run.assert_called_once()


class TestLilbeeAddWithUrls:
    async def test_add_url_without_crawler(self, isolated_env):
        """Adding URLs when crawl4ai not installed returns error."""
        with mock.patch("lilbee.crawler.crawler_available", return_value=False):
            result = await lilbee_add(paths=["https://example.com"])
            assert "error" in result
            assert "pip install" in result["error"].lower()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    async def test_add_url(self, mock_crawl, mock_sync, isolated_env):
        """URLs in paths list are routed to the crawler."""
        from pathlib import Path

        mock_crawl.return_value = [Path(str(isolated_env / "documents" / "_web" / "page.md"))]
        result = await lilbee_add(paths=["https://example.com"])
        assert result["crawled"] == 1
        mock_crawl.assert_awaited_once()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    async def test_add_mixed_urls_and_paths(self, mock_crawl, mock_sync, isolated_env):
        """Mixed URLs and paths: URLs crawled, nonexistent paths reported."""
        mock_crawl.return_value = []
        result = await lilbee_add(paths=["https://example.com", "/nonexistent"])
        assert result["crawled"] == 0
        assert "/nonexistent" in result["errors"]

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    async def test_add_url_with_vision(self, mock_crawl, mock_sync, isolated_env):
        """Vision model is temporarily applied during sync."""
        mock_crawl.return_value = []
        old_vision = cfg.vision_model
        await lilbee_add(paths=["https://example.com"], vision_model="test-vision:latest")
        # Vision model should be restored after sync
        assert cfg.vision_model == old_vision

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_url_ssrf_rejected(self, mock_sync, isolated_env):
        """Private IP URLs are rejected with an error, not crawled."""
        with mock.patch(
            "lilbee.crawler.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("127.0.0.1", 0))],
        ):
            result = await lilbee_add(paths=["http://evil.test/steal"])
        assert result["crawled"] == 0
        assert any("evil.test" in e for e in result["errors"])


class TestLilbeeCrawl:
    @mock.patch("lilbee.mcp.start_crawl", return_value="abc123")
    def test_returns_task_id(self, mock_start, isolated_env):
        """Non-blocking crawl returns a task_id immediately."""
        result = lilbee_crawl(url="https://example.com")
        assert result["status"] == "started"
        assert result["task_id"] == "abc123"
        assert result["url"] == "https://example.com"
        mock_start.assert_called_once_with("https://example.com", depth=0, max_pages=50)

    @mock.patch("lilbee.mcp.start_crawl", return_value="def456")
    def test_passes_depth_and_max_pages(self, mock_start, isolated_env):
        """Depth and max_pages are forwarded to start_crawl."""
        result = lilbee_crawl(url="https://example.com", depth=2, max_pages=10)
        assert result["task_id"] == "def456"
        mock_start.assert_called_once_with("https://example.com", depth=2, max_pages=10)

    def test_rejects_invalid_url(self):
        result = lilbee_crawl(url="ftp://bad.com")
        assert "error" in result

    def test_crawler_not_installed(self):
        """Returns error when crawl4ai is not installed."""
        with mock.patch("lilbee.crawler.crawler_available", return_value=False):
            result = lilbee_crawl(url="https://example.com")
            assert "error" in result
            assert "pip install" in result["error"].lower()


class TestLilbeeCrawlStatus:
    @mock.patch("lilbee.mcp.get_task")
    def test_returns_task_state(self, mock_get_task, isolated_env):
        """Status returns current task state."""
        from lilbee.crawl_task import CrawlTask, TaskStatus

        mock_get_task.return_value = CrawlTask(
            task_id="abc123",
            url="https://example.com",
            depth=0,
            max_pages=50,
            status=TaskStatus.RUNNING,
            pages_crawled=3,
        )
        status = lilbee_crawl_status("abc123")
        assert status["url"] == "https://example.com"
        assert status["status"] == "running"
        assert status["pages_crawled"] == 3

    def test_not_found(self):
        """Unknown task_id returns an error."""
        clear_tasks()
        result = lilbee_crawl_status("nonexistent")
        assert "error" in result


class TestMcpSubcommand:
    @mock.patch("lilbee.mcp.main")
    def test_mcp_subcommand(self, mock_main):
        from typer.testing import CliRunner

        from lilbee.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["mcp"])
        assert result.exit_code == 0
        mock_main.assert_called_once()
