"""Tests for the MCP server tools."""

from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

import lilbee.services as svc_mod
from lilbee.config import cfg
from lilbee.crawl_task import clear_tasks
from lilbee.ingest import SyncResult
from lilbee.mcp import (
    add,
    clean,
    crawl,
    crawl_status,
    init,
    list_documents,
    main,
    remove,
    reset,
    search,
    status,
    sync,
    wiki_citations,
    wiki_lint,
    wiki_list,
    wiki_prune,
    wiki_read,
    wiki_status,
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
def mock_svc():
    """Provide a mock Services container for all MCP tests."""
    from tests.conftest import make_mock_services

    searcher = MagicMock()
    searcher.search.return_value = []
    services = make_mock_services(searcher=searcher)
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


@pytest.fixture(autouse=True)
def _no_dns():
    """Bypass SSRF DNS resolution in all MCP tests."""
    with mock.patch(
        "lilbee.crawler.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("93.184.216.34", 0))],
    ):
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


class TestSearch:
    def test_returns_cleaned_results(self, mock_svc):
        mock_svc.searcher.search.return_value = [
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
        results = search("test query", top_k=3)
        assert len(results) == 1
        assert "vector" not in results[0]
        assert results[0]["distance"] == 0.3
        mock_svc.searcher.search.assert_called_once_with("test query", top_k=3)

    def test_empty_results(self, mock_svc):
        mock_svc.searcher.search.return_value = []
        assert search("nothing") == []

    def test_empty_query_returns_error(self, mock_svc):
        result = search("", top_k=3)
        assert result == {"error": "query must not be empty"}
        mock_svc.searcher.search.assert_not_called()

    def test_whitespace_query_returns_error(self, mock_svc):
        result = search("   ", top_k=3)
        assert result == {"error": "query must not be empty"}
        mock_svc.searcher.search.assert_not_called()

    def test_embedding_failure_returns_error(self, mock_svc):
        mock_svc.searcher.search.side_effect = RuntimeError("embed failed")
        result = search("test", top_k=3)
        assert "error" in result
        assert "embed failed" in result["error"]

    def test_filters_irrelevant_results(self, mock_svc):
        """Results with distance > max_distance are excluded."""
        cfg.max_distance = 0.8
        mock_svc.searcher.search.return_value = [
            SearchChunk(
                source="good.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="relevant",
                chunk_index=0,
                vector=[0.1],
                distance=0.5,
            ),
            SearchChunk(
                source="bad.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="irrelevant",
                chunk_index=0,
                vector=[0.1],
                distance=0.95,
            ),
        ]
        results = search("test")
        assert len(results) == 1
        assert results[0]["source"] == "good.md"

    def test_keeps_hybrid_results_without_distance(self, mock_svc):
        """Hybrid/RRF results with distance=None are not filtered."""
        mock_svc.searcher.search.return_value = [
            SearchChunk(
                source="hybrid.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="hybrid result",
                chunk_index=0,
                vector=[0.1],
                distance=None,
            ),
        ]
        results = search("test")
        assert len(results) == 1
        assert results[0]["source"] == "hybrid.md"


class TestStatus:
    def test_empty_status(self, mock_svc):
        result = status()
        assert "config" in result
        assert result["sources"] == []
        assert result["total_chunks"] == 0

    def test_with_sources(self, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {
                "filename": "test.pdf",
                "file_hash": "abc123",
                "chunk_count": 10,
                "ingested_at": "2026-01-01T00:00:00",
            }
        ]
        result = status()
        assert len(result["sources"]) == 1
        assert result["sources"][0]["filename"] == "test.pdf"
        assert result["total_chunks"] == 10

    def test_status_includes_enable_ocr_when_set(self):
        cfg.enable_ocr = True
        result = status()
        assert result["config"]["enable_ocr"] is True

    def test_status_enable_ocr_none_by_default(self):
        cfg.enable_ocr = None
        result = status()
        assert result["config"]["enable_ocr"] is None


class TestSync:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_sync_empty(self, mock_sync):
        result = await sync()
        assert result["added"] == []
        assert result["unchanged"] == 0

    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(added=["test.txt"]),
    )
    async def test_sync_with_file(self, mock_sync):
        (cfg.documents_dir / "test.txt").write_text("Hello world content.")
        result = await sync()
        assert "test.txt" in result["added"]


class TestRemove:
    def test_removes_known_file(self, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(removed=["a.md"], not_found=[])
        result = remove(["a.md"])
        assert result["removed"] == ["a.md"]
        assert result["not_found"] == []

    def test_not_found(self, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=[], not_found=["missing.md"]
        )
        result = remove(["missing.md"])
        assert result["not_found"] == ["missing.md"]

    def test_delete_files_removes_from_disk(self, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(removed=["a.md"], not_found=[])
        result = remove(["a.md"], delete_files=True)
        assert result["removed"] == ["a.md"]

    def test_delete_files_path_traversal_skipped(self, mock_svc):
        """Path traversal names are caught and skipped during delete_files."""
        from lilbee.store import RemoveResult

        traversal_name = "../../etc/passwd"
        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=[traversal_name], not_found=[]
        )
        result = remove([traversal_name], delete_files=True)
        assert result["removed"] == [traversal_name]


class TestListDocuments:
    def test_returns_documents(self, mock_svc):
        mock_svc.store.get_sources.return_value = [{"filename": "a.md", "chunk_count": 3}]
        result = list_documents()
        assert result["total"] == 1
        assert result["documents"][0]["filename"] == "a.md"

    def test_empty(self, mock_svc):
        mock_svc.store.get_sources.return_value = []
        result = list_documents()
        assert result["total"] == 0


class TestReset:
    def test_reset_requires_confirm(self):
        result = reset()
        assert "error" in result
        assert "confirm" in result["error"]

    def test_reset_confirm_false(self):
        result = reset(confirm=False)
        assert "error" in result

    def test_reset_clears_everything(self):
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        (cfg.documents_dir / "doc.txt").write_text("content")
        (cfg.data_dir / "db_file").write_text("data")

        result = reset(confirm=True)
        assert result["command"] == "reset"
        assert result["deleted_docs"] == 1
        assert result["deleted_data"] == 1
        assert list(cfg.documents_dir.iterdir()) == []
        assert list(cfg.data_dir.iterdir()) == []

    def test_reset_empty_dirs(self):
        result = reset(confirm=True)
        assert result["command"] == "reset"
        assert result["deleted_docs"] == 0
        assert result["deleted_data"] == 0


class TestInit:
    def test_init_creates_structure(self, tmp_path):
        target = tmp_path / "project"
        target.mkdir()
        result = init(str(target))
        root = target / ".lilbee"
        assert result["command"] == "init"
        assert result["created"] is True
        assert root.is_dir()
        assert (root / "documents").is_dir()
        assert (root / "data").is_dir()
        assert (root / ".gitignore").read_text() == "data/\n"

    def test_init_already_exists(self, tmp_path):
        (tmp_path / ".lilbee").mkdir()
        result = init(str(tmp_path))
        assert result["created"] is False

    def test_init_default_cwd(self, tmp_path):
        with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            result = init()
        assert result["created"] is True
        assert (tmp_path / ".lilbee" / "documents").is_dir()

    def test_init_no_home_dir_restriction(self, tmp_path):
        """Init works outside home directory (BEE-o2e)."""
        target = tmp_path / "anywhere"
        target.mkdir()
        result = init(str(target))
        assert result["created"] is True
        assert (target / ".lilbee").is_dir()

    def test_init_switches_config(self, tmp_path):
        """After init, cfg points to the new project KB (BEE-xlu)."""
        target = tmp_path / "myproject"
        target.mkdir()
        init(str(target))
        root = target / ".lilbee"
        assert cfg.documents_dir == root / "documents"
        assert cfg.data_dir == root / "data"
        assert cfg.lancedb_dir == root / "data" / "lancedb"
        assert cfg.data_root == target

    def test_init_existing_also_switches_config(self, tmp_path):
        """Init on existing .lilbee/ still switches config context."""
        root = tmp_path / ".lilbee"
        root.mkdir()
        init(str(tmp_path))
        assert cfg.documents_dir == root / "documents"
        assert cfg.data_root == tmp_path

    def test_init_normalizes_model_tags(self, tmp_path):
        """Init adds :latest to bare model names (BEE-bhe)."""
        cfg.chat_model = "qwen3"
        cfg.embedding_model = "nomic-embed-text"
        target = tmp_path / "proj"
        target.mkdir()
        init(str(target))
        assert cfg.chat_model == "qwen3:latest"
        assert cfg.embedding_model == "nomic-embed-text:latest"

    def test_init_preserves_tagged_models(self, tmp_path):
        """Init does not alter models that already have a tag."""
        cfg.chat_model = "qwen3:0.6b"
        cfg.embedding_model = "nomic-embed-text:v1.5"
        target = tmp_path / "proj"
        target.mkdir()
        init(str(target))
        assert cfg.chat_model == "qwen3:0.6b"
        assert cfg.embedding_model == "nomic-embed-text:v1.5"


class TestAdd:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_single_file(self, mock_sync, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("hello world")

        result = await add([str(src)])

        assert result["command"] == "add"
        assert "test.txt" in result["copied"]
        assert result["errors"] == []
        assert result["skipped"] == []
        assert (cfg.documents_dir / "test.txt").read_text() == "hello world"
        mock_sync.assert_awaited_once()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_nonexistent_path(self, mock_sync, tmp_path):
        result = await add(["/no/such/path.txt"])

        assert "/no/such/path.txt" in result["errors"]
        assert result["copied"] == []

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_nonexistent_has_warning(self, mock_sync):
        """Nonexistent paths produce a warning field (BEE-dlj)."""
        result = await add(["/no/such/path.txt"])
        assert "warning" in result

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_existing_no_force(self, mock_sync, tmp_path):
        (cfg.documents_dir / "exist.txt").write_text("old")
        src = tmp_path / "exist.txt"
        src.write_text("new")

        result = await add([str(src)])

        assert "exist.txt" in result["skipped"]
        assert result["copied"] == []
        assert (cfg.documents_dir / "exist.txt").read_text() == "old"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_existing_with_force(self, mock_sync, tmp_path):
        (cfg.documents_dir / "exist.txt").write_text("old")
        src = tmp_path / "exist.txt"
        src.write_text("new")

        result = await add([str(src)], force=True)

        assert "exist.txt" in result["copied"]
        assert result["skipped"] == []
        assert (cfg.documents_dir / "exist.txt").read_text() == "new"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_directory(self, mock_sync, tmp_path):
        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("a")

        result = await add([str(src_dir)])

        assert "mydir" in result["copied"]
        assert (cfg.documents_dir / "mydir" / "a.txt").read_text() == "a"

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_with_enable_ocr(self, mock_sync, tmp_path):
        src = tmp_path / "scan.pdf"
        src.write_bytes(b"%PDF-fake")
        original_ocr = cfg.enable_ocr

        await add([str(src)], enable_ocr=True)

        # enable_ocr should be restored after the call
        assert cfg.enable_ocr == original_ocr

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=RuntimeError("boom"))
    async def test_add_enable_ocr_restored_on_error(self, mock_sync, tmp_path):
        src = tmp_path / "file.txt"
        src.write_text("content")
        original_ocr = cfg.enable_ocr

        with pytest.raises(RuntimeError, match="boom"):
            await add([str(src)], enable_ocr=True)

        assert cfg.enable_ocr == original_ocr

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_empty_paths(self, mock_sync):
        result = await add([])

        assert result["copied"] == []
        assert result["skipped"] == []
        assert result["errors"] == []

    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(failed=["bad.md"]),
    )
    async def test_add_sync_failures_has_warning(self, mock_sync, tmp_path):
        """Sync failures produce a warning field (BEE-wmn)."""
        src = tmp_path / "bad.md"
        src.write_text("content")
        result = await add([str(src)])
        assert "warning" in result


class TestMain:
    @mock.patch("lilbee.mcp.mcp")
    def test_main_calls_run(self, mock_mcp):
        main()
        mock_mcp.run.assert_called_once()


class TestAddWithUrls:
    async def test_add_url_without_crawler(self, isolated_env):
        """Adding URLs when crawl4ai not installed returns error."""
        with mock.patch("lilbee.crawler.crawler_available", return_value=False):
            result = await add(paths=["https://example.com"])
            assert "error" in result
            assert "pip install" in result["error"].lower()

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    async def test_add_url(self, mock_crawl, mock_sync, _mock_avail, isolated_env):
        """URLs in paths list are routed to the crawler."""
        from pathlib import Path

        mock_crawl.return_value = [Path(str(isolated_env / "documents" / "_web" / "page.md"))]
        result = await add(paths=["https://example.com"])
        assert result["crawled"] == 1
        mock_crawl.assert_awaited_once()

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    async def test_add_mixed_urls_and_paths(self, mock_crawl, mock_sync, _mock_avail, isolated_env):
        """Mixed URLs and paths: URLs crawled, nonexistent paths reported."""
        mock_crawl.return_value = []
        result = await add(paths=["https://example.com", "/nonexistent"])
        assert result["crawled"] == 0
        assert "/nonexistent" in result["errors"]

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    async def test_add_url_with_enable_ocr(self, mock_crawl, mock_sync, isolated_env):
        """enable_ocr is temporarily applied during sync."""
        mock_crawl.return_value = []
        old_ocr = cfg.enable_ocr
        with mock.patch("lilbee.crawler.crawler_available", return_value=True):
            await add(paths=["https://example.com"], enable_ocr=True)
        assert cfg.enable_ocr == old_ocr

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_url_ssrf_rejected(self, mock_sync, _mock_avail, isolated_env):
        """Private IP URLs are rejected with an error, not crawled."""
        with mock.patch(
            "lilbee.crawler.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("127.0.0.1", 0))],
        ):
            result = await add(paths=["http://evil.test/steal"])
        assert result["crawled"] == 0
        assert any("evil.test" in e for e in result["errors"])


class TestCrawl:
    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.mcp.start_crawl", return_value="abc123")
    def test_returns_task_id(self, mock_start, _mock_avail, isolated_env):
        """Non-blocking crawl returns a task_id immediately."""
        result = crawl(url="https://example.com")
        assert result["status"] == "started"
        assert result["task_id"] == "abc123"
        assert result["url"] == "https://example.com"
        mock_start.assert_called_once_with("https://example.com", depth=0, max_pages=50)

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.mcp.start_crawl", return_value="def456")
    def test_passes_depth_and_max_pages(self, mock_start, _mock_avail, isolated_env):
        """Depth and max_pages are forwarded to start_crawl."""
        result = crawl(url="https://example.com", depth=2, max_pages=10)
        assert result["task_id"] == "def456"
        mock_start.assert_called_once_with("https://example.com", depth=2, max_pages=10)

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    def test_rejects_invalid_url(self, _mock_avail):
        result = crawl(url="ftp://bad.com")
        assert "error" in result

    def test_crawler_not_installed(self):
        """Returns error when crawl4ai is not installed."""
        with mock.patch("lilbee.crawler.crawler_available", return_value=False):
            result = crawl(url="https://example.com")
            assert "error" in result
            assert "pip install" in result["error"].lower()


class TestCrawlStatus:
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
        result = crawl_status("abc123")
        assert result["url"] == "https://example.com"
        assert result["status"] == "running"
        assert result["pages_crawled"] == 3

    def test_not_found(self):
        """Unknown task_id returns an error."""
        clear_tasks()
        result = crawl_status("nonexistent")
        assert "error" in result


class TestWikiLint:
    def test_lint_all_pages(self, mock_svc, tmp_path):
        cfg.data_root = tmp_path
        cfg.wiki_dir = "wiki"
        cfg.wiki = True
        wiki_dir = tmp_path / "wiki" / "summaries"
        wiki_dir.mkdir(parents=True)
        (wiki_dir / "doc.md").write_text("Unmarked claim.\n")
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = wiki_lint()
        assert result["command"] == "wiki_lint"
        assert result["total"] >= 1

    def test_lint_single_page(self, mock_svc, tmp_path):
        cfg.data_root = tmp_path
        cfg.wiki_dir = "wiki"
        cfg.wiki = True
        wiki_dir = tmp_path / "wiki" / "summaries"
        wiki_dir.mkdir(parents=True)
        (wiki_dir / "doc.md").write_text(
            "> Cited.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n"
        )
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = wiki_lint(wiki_source="wiki/summaries/doc.md")
        assert result["total"] == 0

    def test_lint_no_wiki_dir(self, mock_svc, tmp_path):
        cfg.data_root = tmp_path
        cfg.wiki_dir = "wiki"
        result = wiki_lint()
        assert result["total"] == 0


class TestWikiCitations:
    def test_returns_citations(self, mock_svc):
        mock_svc.store.get_citations_for_wiki.return_value = [
            {
                "wiki_source": "wiki/summaries/doc.md",
                "wiki_chunk_index": 0,
                "citation_key": "src1",
                "claim_type": "fact",
                "source_filename": "doc.md",
                "source_hash": "abc",
                "page_start": 0,
                "page_end": 0,
                "line_start": 1,
                "line_end": 10,
                "excerpt": "text",
                "created_at": "2026-01-01",
            }
        ]
        result = wiki_citations("wiki/summaries/doc.md")
        assert result["command"] == "wiki_citations"
        assert result["total"] == 1
        assert result["citations"][0]["citation_key"] == "src1"

    def test_no_citations(self, mock_svc):
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = wiki_citations("wiki/summaries/missing.md")
        assert result["total"] == 0


class TestWikiStatus:
    def test_no_wiki_dir(self, tmp_path, mock_svc):
        cfg.data_root = tmp_path
        cfg.wiki_dir = "wiki"
        cfg.wiki = True
        result = wiki_status()
        assert result["wiki_enabled"] is True
        assert result["pages"] == 0

    def test_with_pages(self, tmp_path, mock_svc):
        cfg.data_root = tmp_path
        cfg.wiki_dir = "wiki"
        cfg.wiki = True
        (tmp_path / "wiki" / "summaries").mkdir(parents=True)
        (tmp_path / "wiki" / "summaries" / "a.md").write_text("content")
        (tmp_path / "wiki" / "drafts").mkdir(parents=True)
        (tmp_path / "wiki" / "drafts" / "b.md").write_text("content")
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = wiki_status()
        assert result["summaries"] == 1
        assert result["drafts"] == 1
        assert result["pages"] == 2


class TestWikiPrune:
    def test_prune_no_pages(self, mock_svc, tmp_path):
        cfg.wiki_dir = "wiki"
        cfg.wiki = True
        result = wiki_prune()
        assert result["command"] == "wiki_prune"
        assert result["archived"] == 0
        assert result["flagged"] == 0
        assert result["records"] == []


class TestWikiList:
    def test_wiki_disabled(self):
        cfg.wiki = False
        result = wiki_list()
        assert "error" in result
        assert result["error"] == "wiki not enabled"

    def test_empty_wiki(self, isolated_env):
        cfg.wiki = True
        cfg.data_root = isolated_env
        cfg.wiki_dir = "wiki"
        result = wiki_list()
        assert result["command"] == "wiki_list"
        assert result["pages"] == []
        assert result["total"] == 0

    def test_with_pages(self, isolated_env):
        cfg.wiki = True
        cfg.data_root = isolated_env
        cfg.wiki_dir = "wiki"
        wiki_root = isolated_env / "wiki"
        summaries = wiki_root / "summaries"
        summaries.mkdir(parents=True)
        (summaries / "doc-a.md").write_text(
            "---\ntitle: Doc A\nsources: [x.md]\n---\n# Doc A\n", encoding="utf-8"
        )
        synthesis = wiki_root / "synthesis"
        synthesis.mkdir(parents=True)
        (synthesis / "typing.md").write_text("# Typing\n", encoding="utf-8")
        result = wiki_list()
        assert result["total"] == 2
        slugs = {p["slug"] for p in result["pages"]}
        assert "summaries/doc-a" in slugs
        assert "synthesis/typing" in slugs


class TestWikiRead:
    def test_wiki_disabled(self):
        cfg.wiki = False
        result = wiki_read("summaries/test")
        assert "error" in result
        assert result["error"] == "wiki not enabled"

    def test_existing_page(self, isolated_env):
        cfg.wiki = True
        cfg.data_root = isolated_env
        cfg.wiki_dir = "wiki"
        wiki_root = isolated_env / "wiki"
        summaries = wiki_root / "summaries"
        summaries.mkdir(parents=True)
        (summaries / "my-doc.md").write_text(
            "---\ntitle: My Doc\nsources: [a.txt]\n---\n# My Doc\nBody.\n", encoding="utf-8"
        )
        result = wiki_read("summaries/my-doc")
        assert result["command"] == "wiki_read"
        assert result["slug"] == "summaries/my-doc"
        assert result["title"] == "My Doc"
        assert "Body." in result["content"]
        assert result["frontmatter"]["title"] == "My Doc"

    def test_missing_page(self, isolated_env):
        cfg.wiki = True
        cfg.data_root = isolated_env
        cfg.wiki_dir = "wiki"
        result = wiki_read("summaries/nope")
        assert "error" in result
        assert "not found" in result["error"]

    def test_path_traversal(self, isolated_env):
        cfg.wiki = True
        cfg.data_root = isolated_env
        cfg.wiki_dir = "wiki"
        result = wiki_read("../../etc/passwd")
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
