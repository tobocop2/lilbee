"""Tests for the CLI interface using typer's test runner."""

from unittest import mock

import pytest
from typer.testing import CliRunner

from lilbee.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all CLI tests."""
    import lilbee.config as cfg
    import lilbee.store as store_mod

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


class TestStatus:
    def test_empty_status(self):
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "No documents indexed" in result.output

    def test_status_shows_paths(self):
        result = runner.invoke(app, ["status"])
        assert "Documents:" in result.output
        assert "Database:" in result.output

    def test_status_shows_models(self):
        result = runner.invoke(app, ["status"])
        assert "Chat model:" in result.output
        assert "Embeddings:" in result.output

    def test_status_with_indexed_docs(self, isolated_env):
        from lilbee.store import upsert_source

        upsert_source("test.pdf", "abc123", 10)
        result = runner.invoke(app, ["status"])
        assert "test.pdf" in result.output
        assert "10" in result.output


class TestSync:
    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_sync_empty(self, _e, _eb):
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Added: 0" in result.output

    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_sync_with_file(self, _e, _eb, isolated_env):
        import lilbee.config as cfg

        (cfg.DOCUMENTS_DIR / "test.txt").write_text("Hello world content.")
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Added: 1" in result.output


class TestRebuild:
    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_rebuild_empty(self, _e, _eb):
        result = runner.invoke(app, ["rebuild"])
        assert result.exit_code == 0
        assert "Rebuilt:" in result.output


class TestAsk:
    @mock.patch("lilbee.query.ask_stream")
    @mock.patch(
        "lilbee.ingest.sync",
        return_value={"added": [], "updated": [], "removed": [], "unchanged": 0},
    )
    def test_ask_prints_response(self, _sync, mock_stream):
        mock_stream.return_value = iter(["Hello", " world"])
        result = runner.invoke(app, ["ask", "test question"])
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream")
    @mock.patch(
        "lilbee.ingest.sync",
        return_value={"added": [], "updated": [], "removed": [], "unchanged": 0},
    )
    def test_ask_with_model_flag(self, _sync, mock_stream):
        mock_stream.return_value = iter(["answer"])
        result = runner.invoke(app, ["ask", "question", "--model", "llama3"])
        assert result.exit_code == 0


class TestDataDirFlag:
    def test_status_with_data_dir(self, tmp_path):
        custom = tmp_path / "custom"
        custom.mkdir()
        (custom / "documents").mkdir()
        result = runner.invoke(app, ["status", "--data-dir", str(custom)])
        assert result.exit_code == 0

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_sync_with_data_dir(self, _e, _eb, tmp_path):
        custom = tmp_path / "custom"
        custom.mkdir()
        (custom / "documents").mkdir()
        result = runner.invoke(app, ["sync", "--data-dir", str(custom)])
        assert result.exit_code == 0


class TestAutoSync:
    @mock.patch(
        "lilbee.ingest.sync",
        return_value={"added": ["new.pdf"], "updated": [], "removed": [], "unchanged": 0},
    )
    @mock.patch("lilbee.query.ask_stream", return_value=iter(["answer"]))
    def test_auto_sync_prints_summary(self, _stream, _sync):
        result = runner.invoke(app, ["ask", "test"])
        assert result.exit_code == 0
        assert "Synced:" in result.output


class TestChat:
    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello"]))
    @mock.patch(
        "lilbee.ingest.sync",
        return_value={"added": [], "updated": [], "removed": [], "unchanged": 0},
    )
    def test_chat_quit(self, _sync, _stream):
        result = runner.invoke(app, ["chat"], input="question\nquit\n")
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello"]))
    @mock.patch(
        "lilbee.ingest.sync",
        return_value={"added": [], "updated": [], "removed": [], "unchanged": 0},
    )
    def test_chat_exit(self, _sync, _stream):
        result = runner.invoke(app, ["chat"], input="exit\n")
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello"]))
    @mock.patch(
        "lilbee.ingest.sync",
        return_value={"added": [], "updated": [], "removed": [], "unchanged": 0},
    )
    def test_chat_empty_input_skipped(self, _sync, _stream):
        result = runner.invoke(app, ["chat"], input="\nquit\n")
        assert result.exit_code == 0

    @mock.patch(
        "lilbee.ingest.sync",
        return_value={"added": [], "updated": [], "removed": [], "unchanged": 0},
    )
    def test_chat_eof_exits(self, _sync):
        # Empty input simulates EOF
        result = runner.invoke(app, ["chat"], input="")
        assert result.exit_code == 0


class TestApplyOverrides:
    def test_data_dir_override(self, tmp_path):
        import lilbee.config as cfg
        from lilbee.cli import _apply_overrides

        _apply_overrides(data_dir=tmp_path)
        assert tmp_path / "documents" == cfg.DOCUMENTS_DIR

    def test_model_override(self):
        import lilbee.config as cfg
        from lilbee.cli import _apply_overrides

        _apply_overrides(model="phi3")
        assert cfg.CHAT_MODEL == "phi3"

    def test_none_values_are_noop(self):
        import lilbee.config as cfg
        from lilbee.cli import _apply_overrides

        original_model = cfg.CHAT_MODEL
        _apply_overrides(data_dir=None, model=None)
        assert original_model == cfg.CHAT_MODEL


class TestMainModule:
    def test_python_m_lilbee_runs(self):
        """Ensure `python -m lilbee` invokes the CLI app."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "lilbee" in result.output
