"""Tests for the MCP server tools."""

import asyncio
import os
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

import lilbee.app.services as svc_mod
from lilbee.core.config import cfg
from lilbee.crawler.task import clear_tasks
from lilbee.data.ingest import SyncResult
from lilbee.data.store import SearchChunk, Store
from lilbee.mcp_server import (
    add,
    catalog_browse,
    crawl,
    crawl_status,
    export_dataset,
    import_dataset,
    init,
    list_documents,
    main,
    remove,
    reset,
    search,
    settings_get,
    settings_list,
    settings_reset,
    settings_set,
    status,
    sync,
    wiki_build,
    wiki_citations,
    wiki_drafts_diff,
    wiki_drafts_list,
    wiki_lint,
    wiki_list,
    wiki_prune,
    wiki_read,
    wiki_status,
    wiki_synthesize,
    wiki_update,
)
from lilbee.runtime.progress import EmbedEvent, EventType, FileStartEvent, noop_callback
from lilbee.wiki.shared import WIKI_DISABLED_ERROR


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
        "lilbee.crawler.url_filter.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("93.184.216.34", 0))],
    ):
        yield


_SYNC_NOOP = SyncResult()
_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"


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
        mock_svc.searcher.search.assert_called_once_with("test query", top_k=3, chunk_type=None)

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

    def test_scope_wiki_filters_to_wiki_chunks(self, mock_svc):
        mock_svc.searcher.search.return_value = []
        search("q", top_k=3, scope="wiki")
        mock_svc.searcher.search.assert_called_once_with("q", top_k=3, chunk_type="wiki")

    def test_scope_raw_filters_to_raw_chunks(self, mock_svc):
        mock_svc.searcher.search.return_value = []
        search("q", top_k=3, scope="raw")
        mock_svc.searcher.search.assert_called_once_with("q", top_k=3, chunk_type="raw")

    def test_scope_both_means_no_filter(self, mock_svc):
        mock_svc.searcher.search.return_value = []
        search("q", top_k=3, scope="both")
        mock_svc.searcher.search.assert_called_once_with("q", top_k=3, chunk_type=None)

    def test_invalid_scope_falls_back_to_both(self, mock_svc, caplog):
        """Smaller models echo prose like 'indexed docs' back as scope; we
        warn and run the search against the default scope instead of
        hard-erroring, so the model's intent ('do a search') still resolves.
        """
        import logging

        mock_svc.searcher.search.return_value = []
        with caplog.at_level(logging.WARNING, logger="lilbee.mcp_server"):
            result = search("q", top_k=3, scope="bogus")
        assert result == []
        mock_svc.searcher.search.assert_called_once()
        # chunk_type=None means "both" -> no filter at the data layer.
        call_kwargs = mock_svc.searcher.search.call_args.kwargs
        assert call_kwargs["chunk_type"] is None
        assert any("unknown scope" in record.message for record in caplog.records)

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

    def test_omitted_top_k_falls_back_to_cfg(self, mock_svc):
        """settings_set top_k must govern search calls that omit the arg."""
        cfg.top_k = 15
        mock_svc.searcher.search.return_value = []
        search("test")
        mock_svc.searcher.search.assert_called_once_with("test", top_k=15, chunk_type=None)

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

    def test_status_exposes_all_four_model_roles(self):
        """MCP status must expose vision + reranker slots so plugin clients see them."""
        result = status()
        assert "chat_model" in result["config"]
        assert "embedding_model" in result["config"]
        assert "vision_model" in result["config"]
        assert "reranker_model" in result["config"]

    def test_status_exposes_memory_tuning_settings(self):
        """MCP status reports the dynamic-ctx tuning knobs so clients can read them."""
        result = status()
        for key in (
            "num_ctx",
            "num_ctx_max",
            "chat_n_ctx_target",
            "flash_attention",
            "kv_cache_type",
            "n_gpu_layers",
        ):
            assert key in result["config"], f"missing {key} in MCP status"
        # Enum value is stringified (kv_cache_type is a StrEnum, not raw text)
        assert isinstance(result["config"]["kv_cache_type"], str)


class TestSync:
    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_sync_empty(self, mock_sync):
        result = await sync()
        assert result["added"] == []
        assert result["unchanged"] == 0

    @mock.patch(
        "lilbee.data.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(added=["test.txt"]),
    )
    async def test_sync_with_file(self, mock_sync):
        (cfg.documents_dir / "test.txt").write_text("Hello world content.")
        result = await sync()
        assert "test.txt" in result["added"]

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_sync_passes_recovery_flags(self, mock_sync):
        await sync(force_rebuild=True, retry_skipped=True)
        _, kwargs = mock_sync.call_args
        assert kwargs["force_rebuild"] is True
        assert kwargs["retry_skipped"] is True

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_sync_recovery_flags_default_false(self, mock_sync):
        await sync()
        _, kwargs = mock_sync.call_args
        assert kwargs["force_rebuild"] is False
        assert kwargs["retry_skipped"] is False


class TestRemove:
    def test_removes_known_file(self, mock_svc):
        from lilbee.data.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(removed=["a.md"], not_found=[])
        result = remove(["a.md"])
        assert result["removed"] == ["a.md"]
        assert result["not_found"] == []

    def test_not_found(self, mock_svc):
        from lilbee.data.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=[], not_found=["missing.md"]
        )
        result = remove(["missing.md"])
        assert result["not_found"] == ["missing.md"]

    def test_delete_files_removes_from_disk(self, mock_svc):
        from lilbee.data.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(removed=["a.md"], not_found=[])
        result = remove(["a.md"], delete_files=True)
        assert result["removed"] == ["a.md"]

    def test_delete_files_path_traversal_skipped(self, mock_svc):
        """Path traversal names are caught and skipped during delete_files."""
        from lilbee.data.store import RemoveResult

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

    def test_reset_drops_cached_store(self):
        """reset must invalidate the cached Store so a follow-up sees the empty data dir."""
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        (cfg.documents_dir / "doc.txt").write_text("content")

        with mock.patch("lilbee.mcp_server.reset_store") as mock_reset_store:
            result = reset(confirm=True)
            assert result["command"] == "reset"
            mock_reset_store.assert_called_once()


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

    def test_bare_name_tag_rejected_after_init(self, tmp_path):
        """The cfg validator rejects bare ``name:tag`` shapes."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must be a HuggingFace ref"):
            cfg.chat_model = "qwen3:0.6b"

    def test_init_exports_lilbee_data_env(self, tmp_path, monkeypatch):
        """``init`` exports ``LILBEE_DATA`` so MCP-spawned workers can write logs.

        Parity with ``cli/app.py::_apply_data_root``. Without this, worker
        subprocesses spawned during an MCP session (chat / embed / vision)
        re-import lilbee with a fresh cfg, lose the project's data root,
        and silently drop their log files on the floor. On Windows this
        turns a worker heap-corruption crash into an opaque "subprocess
        exited unexpectedly" with no diagnostic trail.
        """
        target = tmp_path / "myproject"
        target.mkdir()
        monkeypatch.delenv("LILBEE_DATA", raising=False)

        init(str(target))
        assert os.environ.get("LILBEE_DATA") == str(target)

    def test_init_overlays_per_root_config_toml(self, tmp_path):
        """init() must re-read the project base's config.toml, the same fix as
        the CLI's --data-dir entry point. Without this, switching the MCP
        session to a project that has its own model preferences silently
        keeps the previously-active models."""
        cfg.chat_model = "ollama/stale-global:latest"
        cfg.embedding_model = "ollama/stale-embed:latest"
        target = tmp_path / "myproject"
        target.mkdir()
        (target / "config.toml").write_text(
            'chat_model = "ollama/qwen3:4b"\nembedding_model = "ollama/nomic-embed-text:v1.5"\n'
        )

        init(str(target))

        assert cfg.chat_model == "ollama/qwen3:4b"
        assert cfg.embedding_model == "ollama/nomic-embed-text:v1.5"


class TestAdd:
    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
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

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_nonexistent_path(self, mock_sync, tmp_path):
        result = await add(["/no/such/path.txt"])

        assert "/no/such/path.txt" in result["errors"]
        assert result["copied"] == []

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_nonexistent_has_warning(self, mock_sync):
        """Nonexistent paths produce a warning field (BEE-dlj)."""
        result = await add(["/no/such/path.txt"])
        assert "warning" in result

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_existing_no_force(self, mock_sync, tmp_path):
        (cfg.documents_dir / "exist.txt").write_text("old")
        src = tmp_path / "exist.txt"
        src.write_text("new")

        result = await add([str(src)])

        assert "exist.txt" in result["skipped"]
        assert result["copied"] == []
        assert (cfg.documents_dir / "exist.txt").read_text() == "old"

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_existing_with_force(self, mock_sync, tmp_path):
        (cfg.documents_dir / "exist.txt").write_text("old")
        src = tmp_path / "exist.txt"
        src.write_text("new")

        result = await add([str(src)], force=True)

        assert "exist.txt" in result["copied"]
        assert result["skipped"] == []
        assert (cfg.documents_dir / "exist.txt").read_text() == "new"

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_directory(self, mock_sync, tmp_path):
        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("a")

        result = await add([str(src_dir)])

        assert "mydir" in result["copied"]
        assert (cfg.documents_dir / "mydir" / "a.txt").read_text() == "a"

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_with_enable_ocr(self, mock_sync, tmp_path):
        src = tmp_path / "scan.pdf"
        src.write_bytes(b"%PDF-fake")
        original_ocr = cfg.enable_ocr

        await add([str(src)], enable_ocr=True)

        # enable_ocr should be restored after the call
        assert cfg.enable_ocr == original_ocr

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, side_effect=RuntimeError("boom"))
    async def test_add_enable_ocr_restored_on_error(self, mock_sync, tmp_path):
        src = tmp_path / "file.txt"
        src.write_text("content")
        original_ocr = cfg.enable_ocr

        with pytest.raises(RuntimeError, match="boom"):
            await add([str(src)], enable_ocr=True)

        assert cfg.enable_ocr == original_ocr

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_empty_paths(self, mock_sync):
        result = await add([])

        assert result["copied"] == []
        assert result["skipped"] == []
        assert result["errors"] == []

    @mock.patch(
        "lilbee.data.ingest.sync",
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
    @mock.patch("lilbee.mcp_server.mcp")
    def test_main_calls_run(self, mock_mcp):
        main()
        mock_mcp.run.assert_called_once()

    @mock.patch("lilbee.mcp_server.mcp")
    @mock.patch("lilbee.mcp_server.get_services", side_effect=RuntimeError("boom"))
    def test_main_preload_failure_does_not_block_server(self, mock_get_services, mock_mcp):
        """A crash in the pre-warm path falls through to mcp.run() instead of aborting."""
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
    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    async def test_add_url(self, mock_crawl, mock_sync, _mock_avail, isolated_env):
        """URLs in paths list are routed to the crawler."""
        from pathlib import Path

        mock_crawl.return_value = [Path(str(isolated_env / "documents" / "_web" / "page.md"))]
        result = await add(paths=["https://example.com"])
        assert result["crawled"] == 1
        mock_crawl.assert_awaited_once()

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    async def test_add_mixed_urls_and_paths(self, mock_crawl, mock_sync, _mock_avail, isolated_env):
        """Mixed URLs and paths: URLs crawled, nonexistent paths reported."""
        mock_crawl.return_value = []
        result = await add(paths=["https://example.com", "/nonexistent"])
        assert result["crawled"] == 0
        assert "/nonexistent" in result["errors"]

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    async def test_add_url_with_enable_ocr(self, mock_crawl, mock_sync, isolated_env):
        """enable_ocr is temporarily applied during sync."""
        mock_crawl.return_value = []
        old_ocr = cfg.enable_ocr
        with mock.patch("lilbee.crawler.crawler_available", return_value=True):
            await add(paths=["https://example.com"], enable_ocr=True)
        assert cfg.enable_ocr == old_ocr

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    async def test_add_url_ssrf_rejected(self, mock_sync, _mock_avail, isolated_env):
        """Private IP URLs are rejected with an error, not crawled."""
        with mock.patch(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("127.0.0.1", 0))],
        ):
            result = await add(paths=["http://evil.test/steal"])
        assert result["crawled"] == 0
        assert any("evil.test" in e for e in result["errors"])


class TestCrawl:
    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.mcp_server.start_crawl", return_value="abc123")
    def test_returns_task_id(self, mock_start, _mock_avail, isolated_env):
        """Non-blocking crawl returns a task_id immediately."""
        result = crawl(url="https://example.com")
        assert result["status"] == "started"
        assert result["task_id"] == "abc123"
        assert result["url"] == "https://example.com"
        mock_start.assert_called_once_with("https://example.com", depth=None, max_pages=None)

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.mcp_server.start_crawl", return_value="def456")
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
    @mock.patch("lilbee.mcp_server.get_task")
    def test_returns_task_state(self, mock_get_task, isolated_env):
        """Status returns current task state."""
        from lilbee.crawler.task import CrawlTask, TaskStatus

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


class TestWikiBuildTool:
    def test_wiki_disabled_returns_error(self, mock_svc, tmp_path):
        cfg.wiki = False
        assert wiki_build() == {"error": WIKI_DISABLED_ERROR}

    def test_returns_build_summary(self, mock_svc, tmp_path, monkeypatch):
        cfg.wiki = True
        result_dict = {"paths": [str(tmp_path / "p.md")], "entities": 7, "count": 1}
        monkeypatch.setattr("lilbee.wiki.run_full_build", lambda *a, **kw: result_dict)
        result = wiki_build()
        assert result["command"] == "wiki_build"
        assert result["count"] == 1
        assert result["entities"] == 7


class TestWikiUpdateTool:
    def test_wiki_disabled_returns_error(self, mock_svc, tmp_path):
        cfg.wiki = False
        assert wiki_update() == {"error": WIKI_DISABLED_ERROR}

    def test_returns_build_summary(self, mock_svc, tmp_path, monkeypatch):
        cfg.wiki = True
        result_dict = {"paths": [], "entities": 0, "count": 0}
        monkeypatch.setattr("lilbee.wiki.run_full_build", lambda *a, **kw: result_dict)
        result = wiki_update()
        assert result["command"] == "wiki_update"
        assert result["count"] == 0


class TestWikiSynthesizeTool:
    def test_wiki_disabled_returns_error(self, mock_svc, tmp_path):
        cfg.wiki = False
        assert wiki_synthesize() == {"error": WIKI_DISABLED_ERROR}

    def test_returns_synthesis_paths(self, mock_svc, tmp_path, monkeypatch):
        cfg.wiki = True
        paths = [tmp_path / "wiki" / "synthesis" / "typing.md"]
        monkeypatch.setattr(
            "lilbee.wiki.generation.generate_synthesis_pages", lambda *a, **kw: paths
        )
        result = wiki_synthesize()
        assert result["command"] == "wiki_synthesize"
        assert result["count"] == 1
        assert result["paths"] == [str(paths[0])]

    def test_no_clusters_returns_empty(self, mock_svc, tmp_path, monkeypatch):
        cfg.wiki = True
        monkeypatch.setattr("lilbee.wiki.generation.generate_synthesis_pages", lambda *a, **kw: [])
        result = wiki_synthesize()
        assert result["count"] == 0
        assert result["paths"] == []


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
        assert result["error"] == WIKI_DISABLED_ERROR

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
        assert result["error"] == WIKI_DISABLED_ERROR

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
    @mock.patch("lilbee.mcp_server.main")
    def test_mcp_subcommand(self, mock_main):
        from typer.testing import CliRunner

        from lilbee.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["mcp"])
        assert result.exit_code == 0
        mock_main.assert_called_once()


class TestWikiDraftsMcp:
    """Drafts MCP tools surface the same data as the CLI list/diff."""

    def test_wiki_drafts_list_returns_entries(self, isolated_env):
        cfg.wiki = True
        cfg.data_root = isolated_env
        cfg.wiki_dir = "wiki"
        drafts_dir = isolated_env / "wiki" / "drafts"
        drafts_dir.mkdir(parents=True)
        (drafts_dir / "x.md").write_text(
            "---\nfaithfulness_score: 0.7\n---\n\nbody\n", encoding="utf-8"
        )
        result = wiki_drafts_list()
        assert result["command"] == "wiki_drafts_list"
        assert result["total"] == 1
        assert result["drafts"][0]["slug"] == "x"

    def test_wiki_drafts_diff_returns_unified_diff(self, isolated_env):
        cfg.wiki = True
        cfg.data_root = isolated_env
        cfg.wiki_dir = "wiki"
        wiki_root = isolated_env / "wiki"
        (wiki_root / "summaries").mkdir(parents=True)
        (wiki_root / "summaries" / "x.md").write_text("old\n", encoding="utf-8")
        (wiki_root / "drafts").mkdir(parents=True)
        (wiki_root / "drafts" / "x.md").write_text("new\n", encoding="utf-8")
        result = wiki_drafts_diff("x")
        assert result["command"] == "wiki_drafts_diff"
        assert result["slug"] == "x"
        assert "-old" in result["diff"]
        assert "+new" in result["diff"]

    def test_wiki_drafts_diff_missing_returns_error(self, isolated_env):
        cfg.wiki = True
        cfg.data_root = isolated_env
        cfg.wiki_dir = "wiki"
        result = wiki_drafts_diff("missing")
        assert "error" in result

    def test_wiki_drafts_diff_traversal_slug_generic_error_no_leak(self, isolated_env):
        cfg.wiki = True
        cfg.data_root = isolated_env
        cfg.wiki_dir = "wiki"
        (isolated_env / "wiki" / "drafts").mkdir(parents=True)
        result = wiki_drafts_diff("../../secret")
        assert result == {"error": "invalid draft slug"}
        assert str(isolated_env) not in str(result)


class TestSettingsMcp:
    """MCP settings_* tools read and write through the canonical write boundary."""

    def test_settings_list_returns_metadata(self, isolated_env):
        cfg.data_root = isolated_env
        result = settings_list()
        assert result["command"] == "settings_list"
        assert result["total"] > 0
        keys = {entry["key"] for entry in result["settings"]}
        assert {"top_k", "chunk_size", "max_distance"}.issubset(keys)
        top_k = next(e for e in result["settings"] if e["key"] == "top_k")
        assert top_k["type"] == "int"
        assert top_k["group"] == "Retrieval"
        assert top_k["help"]
        assert top_k["reindex_required"] is False

    def test_settings_list_filters_by_group(self, isolated_env):
        cfg.data_root = isolated_env
        result = settings_list(group="Retrieval")
        groups = {entry["group"] for entry in result["settings"]}
        assert groups == {"Retrieval"}
        assert any(entry["key"] == "top_k" for entry in result["settings"])

    def test_settings_get_returns_value(self, isolated_env):
        cfg.data_root = isolated_env
        cfg.top_k = 7
        result = settings_get("top_k")
        assert result["command"] == "settings_get"
        assert result["setting"]["key"] == "top_k"
        assert result["setting"]["value"] == 7

    def test_settings_get_unknown_key_returns_error(self, isolated_env):
        cfg.data_root = isolated_env
        result = settings_get("bogus")
        assert "error" in result

    def test_settings_set_persists_writable_ints(self, isolated_env):
        cfg.data_root = isolated_env
        result = settings_set({"top_k": 11, "chunk_size": 256})
        assert result["command"] == "settings_set"
        assert set(result["updated"]) == {"top_k", "chunk_size"}
        assert result["reindex_required"] is True
        assert cfg.top_k == 11
        assert cfg.chunk_size == 256
        persisted = (isolated_env / "config.toml").read_text(encoding="utf-8")
        assert "top_k" in persisted
        assert "chunk_size" in persisted

    def test_settings_set_pre_validates_chunk_size(self, isolated_env):
        cfg.data_root = isolated_env
        cfg.top_k = 5
        result = settings_set({"top_k": 12, "chunk_size": 1})
        assert "error" in result
        assert cfg.top_k == 5

    def test_settings_set_rolls_back_when_pydantic_rejects_second_field(self, isolated_env):
        cfg.data_root = isolated_env
        cfg.top_k = 5
        cfg.temperature = 0.6
        result = settings_set({"top_k": 9, "temperature": -1.0})
        assert "error" in result
        assert cfg.top_k == 5, "first field must roll back when second fails pydantic"
        assert cfg.temperature == 0.6

    def test_settings_set_rejects_unknown_key(self, isolated_env):
        cfg.data_root = isolated_env
        result = settings_set({"not_a_real_field": 1})
        assert "error" in result

    def test_settings_set_clears_nullable_field(self, isolated_env):
        cfg.data_root = isolated_env
        settings_set({"max_tokens": 1024})
        persisted = (isolated_env / "config.toml").read_text(encoding="utf-8")
        assert "max_tokens" in persisted
        settings_set({"max_tokens": None})
        assert cfg.max_tokens is None
        persisted = (isolated_env / "config.toml").read_text(encoding="utf-8")
        assert "max_tokens" not in persisted

    def test_settings_set_invalidates_model_arch_cache(self, isolated_env):
        cfg.data_root = isolated_env
        with mock.patch("lilbee.modelhub.model_info.invalidate_cache") as mock_invalidate:
            settings_set({"chat_model": _CHAT_REF})
        mock_invalidate.assert_called_once()

    def test_settings_reset_restores_default(self, isolated_env):
        cfg.data_root = isolated_env
        cfg.top_k = 99
        result = settings_reset(["top_k"])
        assert result["command"] == "settings_reset"
        assert "top_k" in result["updated"]
        from lilbee.app.settings import get_setting

        assert cfg.top_k == get_setting("top_k").default

    def test_settings_reset_unknown_key_returns_error(self, isolated_env):
        cfg.data_root = isolated_env
        result = settings_reset(["nope"])
        assert "error" in result

    def test_settings_reset_nullable_with_non_none_default_restores_value(self, isolated_env):
        """A nullable field whose pydantic default is non-None resets to that value, not None."""
        cfg.data_root = isolated_env
        cfg.temperature = 0.9
        result = settings_reset(["temperature"])
        assert "error" not in result
        from lilbee.app.settings import get_setting

        assert cfg.temperature == get_setting("temperature").default

    def test_settings_reset_refuses_path_sentinel_field(self, isolated_env):
        """documents_dir has no resettable default; resetting must error instead of corrupting."""
        cfg.data_root = isolated_env
        result = settings_reset(["documents_dir"])
        assert "error" in result
        assert "documents_dir" in result["error"]

    def test_settings_list_unknown_group_returns_error(self, isolated_env):
        cfg.data_root = isolated_env
        result = settings_list(group="not-a-group")
        assert "error" in result

    def test_settings_list_filter_is_case_insensitive(self, isolated_env):
        cfg.data_root = isolated_env
        lower = settings_list(group="retrieval")
        upper = settings_list(group="Retrieval")
        assert lower["total"] == upper["total"]
        assert lower["total"] > 0

    def test_settings_list_excludes_write_only_api_keys(self, isolated_env):
        """API keys (write_only) must not appear in settings_list to avoid secret leak."""
        cfg.data_root = isolated_env
        cfg.openai_api_key = "sk-secret"
        result = settings_list()
        keys = {entry["key"] for entry in result["settings"]}
        assert "openai_api_key" not in keys
        assert "hf_token" not in keys

    def test_settings_get_refuses_write_only_field(self, isolated_env):
        """settings_get must refuse write_only fields so API keys cannot be read back."""
        cfg.data_root = isolated_env
        cfg.openai_api_key = "sk-secret"
        result = settings_get("openai_api_key")
        assert "error" in result
        assert "write-only" in result["error"].lower()

    def test_settings_set_rejects_overlap_at_or_above_chunk_size(self, isolated_env):
        """chunk_overlap must stay below chunk_size; the boundary catches the pair upfront."""
        cfg.data_root = isolated_env
        cfg.chunk_size = 512
        result = settings_set({"chunk_overlap": 1024})
        assert "error" in result

    def test_settings_set_rejects_overlap_equal_to_chunk_size(self, isolated_env):
        """chunk_overlap == chunk_size is also rejected (strict less-than)."""
        cfg.data_root = isolated_env
        cfg.chunk_size = 512
        result = settings_set({"chunk_overlap": 512})
        assert "error" in result

    def test_settings_set_writes_hf_token_and_keeps_it_out_of_reads(self, isolated_env):
        """hf_token is write_only: settings_set succeeds, but it stays out of every read."""
        cfg.data_root = isolated_env
        result = settings_set({"hf_token": "hf_abc123"})
        assert "command" in result and result["command"] == "settings_set"
        assert cfg.hf_token == "hf_abc123"
        keys = {entry["key"] for entry in settings_list()["settings"]}
        assert "hf_token" not in keys
        assert "error" in settings_get("hf_token")

    def test_list_settings_accepts_settinggroup_enum(self, isolated_env):
        """list_settings accepts a SettingGroup enum directly, not just a string."""
        cfg.data_root = isolated_env
        from lilbee.app.settings import list_settings
        from lilbee.app.settings_map import SettingGroup

        infos = list_settings(group=SettingGroup.RETRIEVAL)
        assert infos
        assert all(info.group == SettingGroup.RETRIEVAL for info in infos)

    def test_setting_default_handles_pydantic_undefined(self, isolated_env):
        """_setting_default returns None when the pydantic field has no default."""
        cfg.data_root = isolated_env
        from unittest.mock import MagicMock, patch

        from pydantic_core import PydanticUndefined

        from lilbee.app.settings import _setting_default

        field_info = MagicMock()
        field_info.default_factory = None
        field_info.default = PydanticUndefined
        with patch("lilbee.app.settings.Config.model_fields", {"top_k": field_info}):
            assert _setting_default("top_k") is None

    def test_is_nullable_returns_false_for_model_role_field(self, isolated_env):
        """Model role fields are not in WRITABLE_CONFIG_FIELDS; _is_nullable returns False."""
        cfg.data_root = isolated_env
        from lilbee.app.settings import _is_nullable

        assert _is_nullable("chat_model") is False
        assert _is_nullable("embedding_model") is False

    def test_settings_get_wire_format_nullable_matches_boundary(self, isolated_env):
        """settings_get's MCP wire format reports nullable=False for model role fields."""
        cfg.data_root = isolated_env
        for role in ("chat_model", "embedding_model", "vision_model", "reranker_model"):
            result = settings_get(role)
            assert result["setting"]["nullable"] is False, role

    def test_settings_list_includes_lilbee_name_and_show_path(self, isolated_env):
        """The two new naming knobs are reachable over MCP."""
        cfg.data_root = isolated_env
        keys = {entry["key"] for entry in settings_list()["settings"]}
        assert "lilbee_name" in keys
        assert "show_lilbee_path" in keys
        assert settings_get("show_lilbee_path")["setting"]["type"] == "bool"

    def test_settings_set_rolls_back_on_disk_failure(self, isolated_env):
        """OSError from the TOML write reverts cfg before re-raising."""
        from lilbee.app.settings import apply_settings_update

        cfg.data_root = isolated_env
        cfg.top_k = 5
        with (
            mock.patch(
                "lilbee.app.settings.persistent_settings.update_values",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            apply_settings_update({"top_k": 11})
        assert cfg.top_k == 5

    def test_settings_reset_clears_nullable_with_none_default(self, isolated_env):
        """Resetting a nullable field whose pydantic default is None clears the entry."""
        cfg.data_root = isolated_env
        cfg.seed = 42
        settings_reset(["seed"])
        assert cfg.seed is None


class TestLilbeeLabel:
    """app.status.lilbee_label is the friendly name the TUI status bar shows."""

    def test_user_alias_wins(self, isolated_env):
        from lilbee.app.status import lilbee_label

        cfg.data_root = isolated_env
        cfg.lilbee_name = "my-project"
        cfg.show_lilbee_path = False
        assert lilbee_label() == "my-project"

    def test_path_toggle_shows_full_absolute(self, isolated_env):
        from lilbee.app.status import lilbee_label

        cfg.data_root = isolated_env
        cfg.lilbee_name = ""
        cfg.show_lilbee_path = True
        assert lilbee_label() == str(isolated_env.resolve())

    def test_project_local_label_strips_dotlilbee(self, isolated_env):
        """A project-local ``.lilbee`` data_root labels as the project path, not '.lilbee'."""
        from lilbee.app.status import lilbee_label
        from lilbee.core.system import LOCAL_ROOT_DIRNAME

        project = isolated_env / "my-godot-game"
        project.mkdir()
        cfg.data_root = project / LOCAL_ROOT_DIRNAME
        cfg.lilbee_name = ""
        cfg.show_lilbee_path = False
        label = lilbee_label()
        assert ".lilbee" not in label
        assert "my-godot-game" in label

    def test_global_data_dir_label(self, isolated_env):
        from lilbee.app.status import lilbee_label
        from lilbee.core.system import default_data_dir

        cfg.data_root = default_data_dir()
        cfg.lilbee_name = ""
        cfg.show_lilbee_path = False
        assert lilbee_label() == "global"

    def test_show_path_expands_global_to_default_data_dir(self, isolated_env):
        """F4 on global swaps 'global' for the actual platform-default path."""
        from lilbee.app.status import lilbee_label
        from lilbee.core.system import default_data_dir

        cfg.data_root = default_data_dir()
        cfg.lilbee_name = ""
        cfg.show_lilbee_path = True
        assert lilbee_label() == str(default_data_dir())

    def test_whitespace_alias_falls_through(self, isolated_env):
        """A whitespace-only alias is stripped so the path chain takes over."""
        import os

        from lilbee.app.status import lilbee_label

        cfg.data_root = isolated_env
        cfg.lilbee_name = "   "
        cfg.show_lilbee_path = False
        assert cfg.lilbee_name == ""
        label = lilbee_label()
        assert os.sep in label or label.startswith("~") or label == "global"

    def test_global_label_matches_when_data_root_has_trailing_slash(self, isolated_env):
        """A trailing-slash variant of the platform default still resolves to 'global'."""
        from pathlib import Path

        from lilbee.app.status import lilbee_label
        from lilbee.core.system import default_data_dir

        cfg.data_root = Path(f"{default_data_dir()}/")
        cfg.lilbee_name = ""
        cfg.show_lilbee_path = False
        assert lilbee_label() == "global"

    def test_home_paths_collapse_to_tilde(self, isolated_env, tmp_path, monkeypatch):
        """Compact labels substitute ``~`` for $HOME using the native separator."""
        import os
        from pathlib import Path

        from lilbee.app.status import lilbee_label

        # Path.home() reads HOME on POSIX and USERPROFILE on Windows; patch
        # the resolver directly so the test works on both.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        project = tmp_path / "code" / "lilbee-mcp-settings"
        project.mkdir(parents=True)
        cfg.data_root = project / ".lilbee"
        cfg.lilbee_name = ""
        cfg.show_lilbee_path = False
        label = lilbee_label()
        assert label.startswith(f"~{os.sep}")
        assert "lilbee-mcp-settings" in label

    def test_long_path_truncates_from_left(self, isolated_env):
        """Paths over LILBEE_LABEL_MAX_LEN truncate with a leading ellipsis."""
        from lilbee.app.status import LILBEE_LABEL_MAX_LEN, lilbee_label

        deep = isolated_env
        for chunk in ("aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "leaf"):
            deep = deep / chunk
        deep.mkdir(parents=True)
        cfg.data_root = deep
        cfg.lilbee_name = ""
        cfg.show_lilbee_path = False
        label = lilbee_label()
        assert label.startswith("…")
        assert len(label) <= LILBEE_LABEL_MAX_LEN
        assert "leaf" in label

    def test_pathological_leaf_still_respects_max_len(self, isolated_env):
        """A single huge leaf segment gets an internal ellipsis to honor the hard cap."""
        from lilbee.app.status import LILBEE_LABEL_MAX_LEN, lilbee_label

        # No mkdir; the label helper only reads the path, doesn't stat it.
        cfg.data_root = isolated_env / ("z" * 500)
        cfg.lilbee_name = ""
        cfg.show_lilbee_path = False
        label = lilbee_label()
        assert len(label) <= LILBEE_LABEL_MAX_LEN
        assert "…" in label

    def test_path_toggle_overrides_truncation(self, isolated_env):
        """show_lilbee_path returns the full untruncated absolute path."""
        from lilbee.app.status import LILBEE_LABEL_MAX_LEN, lilbee_label

        deep = isolated_env
        for chunk in ("aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "leaf"):
            deep = deep / chunk
        deep.mkdir(parents=True)
        cfg.data_root = deep
        cfg.lilbee_name = ""
        cfg.show_lilbee_path = True
        label = lilbee_label()
        assert label == str(deep.resolve())
        assert len(label) > LILBEE_LABEL_MAX_LEN

    def test_compact_path_exactly_home_returns_tilde(self, tmp_path, monkeypatch):
        """When the project path IS $HOME with no subdir, compact rendering is just '~'."""
        from pathlib import Path

        from lilbee.app.status import _compact_path

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert _compact_path(str(tmp_path)) == "~"

    def test_truncate_leaf_helper_handles_tight_budget(self):
        """``_truncate_leaf`` keeps a head + tail when budget allows, else just ellipsis."""
        from lilbee.app.status import _truncate_leaf

        assert _truncate_leaf("short", 10) == "short"
        out = _truncate_leaf("a" * 100, 9)
        assert len(out) == 9
        assert "…" in out
        assert _truncate_leaf("anything", 1) == "…"


class TestCatalogBrowseMcp:
    """MCP catalog_browse exposes the featured catalog + HF for any model role."""

    def test_browse_returns_featured_embedding_models(self, isolated_env, mock_svc):
        """task=embedding + featured=true returns the curated embedding catalog."""
        # featured=True skips the live HF call; the real HfClient never runs.
        result = catalog_browse(task="embedding", featured=True, limit=5)
        assert result["command"] == "catalog_browse"
        assert result["total"] >= 1
        for entry in result["models"]:
            assert entry["task"] == "embedding"
            assert "ref" in entry
            assert "size_gb" in entry
            assert entry["featured"] is True

    def test_browse_returns_featured_vision_and_rerank(self, isolated_env, mock_svc):
        """The vision and rerank slots both have at least one curated entry."""
        for task in ("vision", "rerank"):
            result = catalog_browse(task=task, featured=True)
            assert result["total"] >= 1, f"no featured models for {task}"

    def test_browse_rejects_invalid_task(self, isolated_env, mock_svc):
        """An unknown task name returns the uniform error envelope."""
        result = catalog_browse(task="not-a-task", featured=True)
        assert "error" in result

    def test_browse_rejects_invalid_size(self, isolated_env, mock_svc):
        """An unknown size bucket returns the uniform error envelope."""
        result = catalog_browse(size="huge", featured=True)
        assert "error" in result

    def test_browse_rejects_invalid_sort(self, isolated_env, mock_svc):
        """An unknown sort key returns the uniform error envelope."""
        result = catalog_browse(sort="random", featured=True)
        assert "error" in result

    def test_browse_omits_chat_model_when_filtered_to_chat_only_role(self, isolated_env, mock_svc):
        """task=rerank never returns chat models even with no other filters."""
        result = catalog_browse(task="rerank", featured=True)
        assert all(m["task"] == "rerank" for m in result["models"])

    def test_browse_forwards_get_catalog_value_error(self, isolated_env, mock_svc):
        """A ValueError from get_catalog surfaces as the uniform error envelope."""
        with mock.patch("lilbee.catalog.query.get_catalog", side_effect=ValueError("bad filter")):
            result = catalog_browse(task="embedding", featured=True)
        assert result == {"error": "bad filter"}


class TestToolsSchemaSize:
    """Schema budget: keep the per-request OpenAI tools schema under a ceiling
    so any model with ``n_ctx >= ~16K`` has room for system + history + content
    after the tools schema is rendered. the original 20K-token schema on
    Qwen3-8B making 51% of the context unusable; trimming docstrings and
    gating the wiki / crawler tools dropped that significantly. A higher
    number doesn't fail builds, it forces a deliberate cap bump that
    reviewers can scrutinise.
    """

    def test_tool_if_true_returns_mcp_tool_decorator(self) -> None:
        """``_tool_if(True)`` returns a real decorator; ``_tool_if(False)``
        returns a pass-through so the function stays importable but isn't on
        the MCP wire. Cleans up after itself so the test doesn't pollute the
        shared FastMCP server with a sentinel tool.
        """
        from lilbee.mcp_server import _tool_if
        from lilbee.mcp_server import mcp as _mcp

        sentinel_name = "_schema_size_test_sentinel"

        def _schema_size_test_sentinel() -> None: ...

        gated_off = _tool_if(False)(_schema_size_test_sentinel)
        assert gated_off is _schema_size_test_sentinel
        assert sentinel_name not in _mcp._tool_manager._tools

        gated_on = _tool_if(True)(_schema_size_test_sentinel)
        try:
            assert callable(gated_on)
            assert sentinel_name in _mcp._tool_manager._tools
        finally:
            _mcp._tool_manager._tools.pop(sentinel_name, None)

    async def test_wiki_tools_not_registered_when_wiki_disabled(self) -> None:
        """``cfg.wiki=False`` (the default) keeps wiki MCP tools off the wire."""
        from lilbee.core.config import cfg as _cfg
        from lilbee.mcp_server import mcp as _mcp

        assert _cfg.wiki is False
        tools = await _mcp.list_tools()
        wiki_tool_names = [t.name for t in tools if t.name.startswith("wiki_")]
        assert wiki_tool_names == []

    async def test_default_tools_schema_under_budget(self) -> None:
        """Default schema (wiki off, crawler off unless the extra is installed)
        must stay under 6 KB so small-context (16K) chat models keep room
        for the user's actual content. _strip_schema_noise removes title /
        default / null-arm-anyOf / additionalProperties=true noise, hitting
        ~5.2 KB today.
        """
        import json as _json

        from lilbee.mcp_server import mcp as _mcp

        tools = await _mcp.list_tools()
        payload = [
            {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
            for t in tools
        ]
        total_bytes = len(_json.dumps(payload))
        # Bumped from 6_000 when the always-on export_dataset / import_dataset tools
        # landed; their docstrings were trimmed first. ~1.6K tokens still leaves a
        # 16K-context model ample room for system + history + content.
        ceiling = 7_000
        assert total_bytes <= ceiling, (
            f"Default OpenAI tools schema is {total_bytes} bytes, exceeds "
            f"{ceiling}. Each MCP tool's docstring becomes the schema "
            "description; trim verbose Args sections before bumping the cap."
        )

    async def test_no_title_noise_in_input_schema(self) -> None:
        """``_strip_schema_noise`` keeps FastMCP-auto-generated title fields
        off the wire. Adding a tool whose schema contains a title fails here.
        """
        from lilbee.mcp_server import mcp as _mcp

        tools = await _mcp.list_tools()
        for t in tools:
            assert "title" not in t.inputSchema, f"{t.name}: top-level title leaked into schema"
            for pname, pdef in t.inputSchema.get("properties", {}).items():
                if isinstance(pdef, dict):
                    assert "title" not in pdef, (
                        f"{t.name}.{pname}: per-property title leaked into schema"
                    )

    async def test_no_default_value_noise_in_input_schema(self) -> None:
        """The model picks tools from name + description; ``default`` values
        on properties are server-side trivia that cost tokens at every dispatch.
        """
        from lilbee.mcp_server import mcp as _mcp

        tools = await _mcp.list_tools()
        for t in tools:
            for pname, pdef in t.inputSchema.get("properties", {}).items():
                if isinstance(pdef, dict):
                    assert "default" not in pdef, (
                        f"{t.name}.{pname}: default leaked into schema; "
                        "remove via _strip_property_noise"
                    )

    async def test_nullable_anyof_collapsed_in_input_schema(self) -> None:
        """``T | None`` should serialize as ``{type: T}``, not a two-arm
        ``anyOf`` with a null branch the model gains nothing from.
        """
        from lilbee.mcp_server import mcp as _mcp

        tools = await _mcp.list_tools()
        for t in tools:
            for pname, pdef in t.inputSchema.get("properties", {}).items():
                if isinstance(pdef, dict):
                    arms = pdef.get("anyOf")
                    if isinstance(arms, list):
                        null_arms = [
                            a for a in arms if isinstance(a, dict) and a.get("type") == "null"
                        ]
                        assert not null_arms, (
                            f"{t.name}.{pname}: nullable anyOf branch leaked "
                            "into schema; collapse via _strip_property_noise"
                        )

    async def test_no_redundant_additional_properties_true(self) -> None:
        """``additionalProperties: true`` is JSON Schema's default; serializing
        it explicitly is bytes for nothing.
        """
        from lilbee.mcp_server import mcp as _mcp

        tools = await _mcp.list_tools()
        for t in tools:
            for pname, pdef in t.inputSchema.get("properties", {}).items():
                if isinstance(pdef, dict):
                    assert pdef.get("additionalProperties") is not True, (
                        f"{t.name}.{pname}: additionalProperties=true leaked "
                        "into schema; remove via _strip_property_noise"
                    )


class _StubEmbedder:
    truncated_total = 0

    def embed_batch(self, texts, **_kwargs):
        return [[0.1] * cfg.embedding_dim for _ in texts]


@pytest.fixture()
def dataset_store(tmp_path):
    """Real store wired into services for the dataset tools."""
    from tests.conftest import make_mock_services

    store = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb"}))
    store.add_page_texts(
        [{"source": "doc.pdf", "page": 1, "text": "page one", "content_type": "pdf"}]
    )
    store.upsert_source("doc.pdf", "h", 1)
    svc_mod.set_services(make_mock_services(store=store, embedder=_StubEmbedder()))
    yield store
    svc_mod.set_services(None)


class TestExportDataset:
    def test_writes_file(self, dataset_store, tmp_path):
        out = tmp_path / "pages.parquet"
        result = export_dataset(str(out))
        assert result == {
            "command": "export",
            "format": "parquet",
            "output": str(out),
            "pages": 1,
            "sources": 1,
        }
        assert out.exists()

    def test_error_envelope(self, dataset_store, tmp_path):
        result = export_dataset(str(tmp_path / "pages.parquet"), source="missing.pdf")
        assert "Source not found" in result["error"]


class _ProgressStubEmbedder:
    """Stub embedder that forwards one non-embed and one embed progress event."""

    truncated_total = 0

    def embed_batch(self, texts, source="", on_progress=noop_callback, **_kwargs):
        on_progress(
            EventType.FILE_START, FileStartEvent(file=source, total_files=1, current_file=1)
        )
        on_progress(
            EventType.EMBED, EmbedEvent(file=source, chunk=len(texts), total_chunks=len(texts))
        )
        return [[0.1] * cfg.embedding_dim for _ in texts]


class TestImportDataset:
    async def test_round_trip(self, dataset_store, tmp_path):
        out = tmp_path / "pages.jsonl"
        assert "error" not in export_dataset(str(out))
        result = await import_dataset(str(out))
        assert result["command"] == "import"
        assert result["sources"] == ["doc.pdf"]
        assert result["pages"] == 1

    async def test_error_envelope(self, dataset_store, tmp_path):
        result = await import_dataset(str(tmp_path / "nope.parquet"))
        assert "Dataset not found" in result["error"]

    async def test_reports_embed_progress_with_ctx(self, dataset_store, tmp_path):
        from tests.conftest import make_mock_services

        out = tmp_path / "pages.jsonl"
        assert "error" not in export_dataset(str(out))
        svc_mod.set_services(
            make_mock_services(store=dataset_store, embedder=_ProgressStubEmbedder())
        )
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()

        result = await import_dataset(str(out), ctx=ctx)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert "error" not in result
        assert ctx.report_progress.await_count == 1
        kwargs = ctx.report_progress.await_args_list[0].kwargs
        assert kwargs["progress"] == kwargs["total"]
        assert kwargs["message"] == "doc.pdf"

    async def test_no_ctx_skips_progress(self, dataset_store, tmp_path):
        from tests.conftest import make_mock_services

        out = tmp_path / "pages.jsonl"
        assert "error" not in export_dataset(str(out))
        svc_mod.set_services(
            make_mock_services(store=dataset_store, embedder=_ProgressStubEmbedder())
        )
        result = await import_dataset(str(out))
        assert "error" not in result
