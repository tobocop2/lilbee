"""Tests for the document sync engine (mocked — no live server needed)."""

from pathlib import Path
from unittest import mock

import pytest

import lilbee.config as cfg
import lilbee.store as store_mod


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths to temp dir for every test."""
    docs = tmp_path / "documents"
    docs.mkdir()
    data = tmp_path / "data" / "lancedb"

    orig_docs, orig_db, orig_data = cfg.DOCUMENTS_DIR, cfg.LANCEDB_DIR, cfg.DATA_DIR

    cfg.DOCUMENTS_DIR = docs
    cfg.DATA_DIR = tmp_path / "data"
    cfg.LANCEDB_DIR = data
    store_mod.LANCEDB_DIR = data

    yield docs

    cfg.DOCUMENTS_DIR = orig_docs
    cfg.DATA_DIR = orig_data
    cfg.LANCEDB_DIR = orig_db
    store_mod.LANCEDB_DIR = orig_db


def _fake_embed_batch(texts):
    return [[0.1] * 768 for _ in texts]


def _fake_embed(text):
    return [0.1] * 768


@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
class TestSync:
    def test_empty_documents_dir(self, _eb, _e, isolated_env):
        from lilbee.ingest import sync

        result = sync()
        assert result == {"added": [], "updated": [], "removed": [], "unchanged": 0, "failed": []}

    def test_ingest_text_file(self, _eb, _e, isolated_env):
        (isolated_env / "test.txt").write_text("Hello world. This is a test document.")
        from lilbee.ingest import sync

        result = sync()
        assert "test.txt" in result["added"]

    def test_ingest_markdown_file(self, _eb, _e, isolated_env):
        (isolated_env / "readme.md").write_text("# Title\n\nSome markdown content.")
        from lilbee.ingest import sync

        assert "readme.md" in sync()["added"]

    def test_ingest_html_file(self, _eb, _e, isolated_env):
        (isolated_env / "page.html").write_text("<p>Content</p>")
        from lilbee.ingest import sync

        assert "page.html" in sync()["added"]

    def test_ingest_rst_file(self, _eb, _e, isolated_env):
        (isolated_env / "doc.rst").write_text("Title\n=====\n\nContent.")
        from lilbee.ingest import sync

        assert "doc.rst" in sync()["added"]

    def test_modified_file_reingested(self, _eb, _e, isolated_env):
        f = isolated_env / "changing.txt"
        f.write_text("Version 1")
        from lilbee.ingest import sync

        sync()
        f.write_text("Version 2 — different content now")
        assert "changing.txt" in sync()["updated"]

    def test_deleted_file_removed(self, _eb, _e, isolated_env):
        f = isolated_env / "temp.txt"
        f.write_text("Temporary")
        from lilbee.ingest import sync

        sync()
        f.unlink()
        assert "temp.txt" in sync()["removed"]

    def test_unchanged_file_skipped(self, _eb, _e, isolated_env):
        (isolated_env / "stable.txt").write_text("I stay the same")
        from lilbee.ingest import sync

        sync()
        result = sync()
        assert result["unchanged"] == 1
        assert result["added"] == []

    def test_unsupported_extension_skipped(self, _eb, _e, isolated_env):
        (isolated_env / "data.xlsx").write_bytes(b"binary data")
        from lilbee.ingest import sync

        assert sync()["added"] == []

    def test_hidden_files_skipped(self, _eb, _e, isolated_env):
        (isolated_env / ".hidden").write_text("secret")
        from lilbee.ingest import sync

        assert sync()["added"] == []

    def test_subdirectory_files_ingested(self, _eb, _e, isolated_env):
        sub = isolated_env / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested content")
        from lilbee.ingest import sync

        assert any("nested.txt" in f for f in sync()["added"])

    def test_code_file_ingested(self, _eb, _e, isolated_env):
        (isolated_env / "example.py").write_text("def hello():\n    print('hi')\n")
        from lilbee.ingest import sync

        assert "example.py" in sync()["added"]

    def test_force_rebuild_clears_and_reingests(self, _eb, _e, isolated_env):
        (isolated_env / "keep.txt").write_text("I survive rebuilds")
        from lilbee.ingest import sync

        sync()
        result = sync(force_rebuild=True)
        assert "keep.txt" in result["added"]

    def test_ingest_pdf(self, _eb, _e, isolated_env):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = isolated_env / "test.pdf"
        c = canvas.Canvas(str(pdf), pagesize=letter)
        c.drawString(72, 700, "Oil capacity is 5 quarts.")
        c.showPage()
        c.save()

        from lilbee.ingest import sync

        assert "test.pdf" in sync()["added"]

    def test_nonexistent_documents_dir(self, _eb, _e, isolated_env, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        cfg.DOCUMENTS_DIR = nonexistent
        from lilbee.ingest import sync

        result = sync()
        assert result == {"added": [], "updated": [], "removed": [], "unchanged": 0, "failed": []}
        assert nonexistent.exists()  # Directory was auto-created

    def test_ingest_error_logged_not_raised(self, _eb, _e, isolated_env):
        """A file that fails ingestion is logged but doesn't crash sync."""
        from unittest.mock import patch

        (isolated_env / "good.txt").write_text("This is fine.")
        (isolated_env / "bad.txt").write_text("This will fail.")

        from lilbee.ingest import sync

        orig_ingest = __import__("lilbee.ingest", fromlist=["_ingest_file"])._ingest_file

        def _failing_ingest(path, name, content_type):
            if "bad" in name:
                raise RuntimeError("simulated failure")
            return orig_ingest(path, name, content_type)

        with patch("lilbee.ingest._ingest_file", side_effect=_failing_ingest):
            result = sync()
        # good.txt was added, bad.txt failed
        assert "good.txt" in result["added"]
        assert "bad.txt" not in result["added"]
        assert "bad.txt" in result["failed"]

    def test_ingest_error_on_update_tracked_as_failed(self, _eb, _e, isolated_env):
        """A file that fails re-ingestion on update goes to failed, not updated."""
        from unittest.mock import patch

        f = isolated_env / "flaky.txt"
        f.write_text("Version 1")

        from lilbee.ingest import sync

        sync()  # First ingest succeeds

        f.write_text("Version 2 — will fail")

        orig_ingest = __import__("lilbee.ingest", fromlist=["_ingest_file"])._ingest_file

        def _failing_ingest(path, name, content_type):
            if "flaky" in name:
                raise RuntimeError("simulated failure on update")
            return orig_ingest(path, name, content_type)

        with patch("lilbee.ingest._ingest_file", side_effect=_failing_ingest):
            result = sync()
        assert "flaky.txt" not in result["updated"]
        assert "flaky.txt" in result["failed"]


class TestIngestHelpers:
    """Cover edge cases in _ingest_text, _ingest_code, _ingest_pdf."""

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    def test_ingest_text_empty_chunks(self, _eb, isolated_env):
        """Text file that produces no chunks returns empty list."""
        from lilbee.ingest import _ingest_text

        # File with only whitespace
        f = isolated_env / "empty.txt"
        f.write_text("   ")
        result = _ingest_text(f, "empty.txt")
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    def test_ingest_code_empty_chunks(self, _eb, isolated_env):
        """Code file that produces no chunks returns empty list."""
        from unittest.mock import patch

        from lilbee.ingest import _ingest_code

        f = isolated_env / "empty.py"
        f.write_text("")
        with patch("lilbee.code_chunker.chunk_code", return_value=[]):
            result = _ingest_code(f, "empty.py")
            assert result == []

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    def test_ingest_pdf_empty_pages(self, _eb, isolated_env):
        """PDF that produces no page chunks returns empty list."""
        from unittest.mock import patch

        from lilbee.ingest import _ingest_pdf

        f = isolated_env / "empty.pdf"
        f.write_bytes(b"fake")
        with (
            patch("pymupdf4llm.to_markdown", return_value=[{"metadata": {"page": 0}, "text": ""}]),
            patch("lilbee.chunker.chunk_pages", return_value=[]),
        ):
            result = _ingest_pdf(f, "empty.pdf")
            assert result == []


class TestDiscoverFiles:
    def test_nonexistent_dir_returns_empty(self, isolated_env, tmp_path):
        cfg.DOCUMENTS_DIR = tmp_path / "does_not_exist"
        from lilbee.ingest import _discover_files

        assert _discover_files() == {}


class TestClassifyFile:
    def test_pdf(self):
        from lilbee.ingest import _classify_file

        assert _classify_file(Path("doc.pdf")) == "pdf"

    def test_text_types(self):
        from lilbee.ingest import _classify_file

        assert _classify_file(Path("f.md")) == "text"
        assert _classify_file(Path("f.txt")) == "text"
        assert _classify_file(Path("f.html")) == "text"
        assert _classify_file(Path("f.rst")) == "text"

    def test_code_types(self):
        from lilbee.ingest import _classify_file

        assert _classify_file(Path("f.py")) == "code"
        assert _classify_file(Path("f.js")) == "code"
        assert _classify_file(Path("f.go")) == "code"

    def test_unsupported(self):
        from lilbee.ingest import _classify_file

        assert _classify_file(Path("f.xlsx")) is None
        assert _classify_file(Path("f.zip")) is None


class TestFileHash:
    def test_deterministic(self, tmp_path):
        from lilbee.ingest import _file_hash

        f = tmp_path / "test.txt"
        f.write_text("hello")
        h1 = _file_hash(f)
        h2 = _file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        from lilbee.ingest import _file_hash

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert _file_hash(f1) != _file_hash(f2)
