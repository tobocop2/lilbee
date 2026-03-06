"""Tests for the MCP server tools."""

from unittest import mock

import pytest

import lilbee.config as cfg
import lilbee.store as store_mod


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all MCP tests."""
    orig_docs, orig_db, orig_data = cfg.DOCUMENTS_DIR, cfg.LANCEDB_DIR, cfg.DATA_DIR

    cfg.DOCUMENTS_DIR = tmp_path / "documents"
    cfg.DOCUMENTS_DIR.mkdir()
    cfg.DATA_DIR = tmp_path / "data"
    cfg.LANCEDB_DIR = tmp_path / "data" / "lancedb"
    store_mod.LANCEDB_DIR = cfg.LANCEDB_DIR

    yield tmp_path

    cfg.DOCUMENTS_DIR = orig_docs
    cfg.DATA_DIR = orig_data
    cfg.LANCEDB_DIR = orig_db
    store_mod.LANCEDB_DIR = orig_db


@pytest.fixture(autouse=True)
def _skip_model_validation():
    """MCP tests never need real Ollama model validation."""
    with mock.patch("lilbee.embedder.validate_model"):
        yield


class TestClean:
    def test_strips_vector(self):
        from lilbee.mcp import _clean

        result = _clean({"source": "a.pdf", "vector": [0.1], "chunk": "hi"})
        assert "vector" not in result
        assert result["source"] == "a.pdf"

    def test_renames_distance(self):
        from lilbee.mcp import _clean

        result = _clean({"_distance": 0.42, "chunk": "hi"})
        assert result["distance"] == 0.42
        assert "_distance" not in result


class TestLilbeeSearch:
    @mock.patch("lilbee.query.search_context")
    def test_returns_cleaned_results(self, mock_search):
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


class TestLilbeeAsk:
    @mock.patch("lilbee.query.ask_raw")
    def test_returns_answer_and_sources(self, mock_ask):
        from lilbee.mcp import lilbee_ask
        from lilbee.query import AskResult

        mock_ask.return_value = AskResult(
            answer="5 quarts",
            sources=[{"source": "manual.pdf", "_distance": 0.2, "vector": [0.1], "chunk": "oil"}],
        )
        result = lilbee_ask("oil capacity?")
        assert result["answer"] == "5 quarts"
        assert len(result["sources"]) == 1
        assert "vector" not in result["sources"][0]


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
    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_sync_empty(self, _e, _eb):
        from lilbee.mcp import lilbee_sync

        result = lilbee_sync()
        assert result["added"] == []
        assert result["unchanged"] == 0

    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_sync_with_file(self, _e, _eb):
        from lilbee.mcp import lilbee_sync

        (cfg.DOCUMENTS_DIR / "test.txt").write_text("Hello world content.")
        result = lilbee_sync()
        assert "test.txt" in result["added"]


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
