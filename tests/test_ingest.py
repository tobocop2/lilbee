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


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
class TestSync:
    def test_empty_documents_dir(self, _eb, _e, _vm, isolated_env):
        from lilbee.ingest import sync

        result = sync()
        assert result == {"added": [], "updated": [], "removed": [], "unchanged": 0, "failed": []}

    def test_ingest_text_file(self, _eb, _e, _vm, isolated_env):
        (isolated_env / "test.txt").write_text("Hello world. This is a test document.")
        from lilbee.ingest import sync

        result = sync()
        assert "test.txt" in result["added"]

    def test_quiet_mode_suppresses_progress(self, _eb, _e, _vm, isolated_env):
        (isolated_env / "quiet.txt").write_text("Quiet mode test content.")
        from lilbee.ingest import sync

        result = sync(quiet=True)
        assert "quiet.txt" in result["added"]

    def test_ingest_markdown_file(self, _eb, _e, _vm, isolated_env):
        (isolated_env / "readme.md").write_text("# Title\n\nSome markdown content.")
        from lilbee.ingest import sync

        assert "readme.md" in sync()["added"]

    def test_ingest_html_file(self, _eb, _e, _vm, isolated_env):
        (isolated_env / "page.html").write_text("<p>Content</p>")
        from lilbee.ingest import sync

        assert "page.html" in sync()["added"]

    def test_ingest_rst_file(self, _eb, _e, _vm, isolated_env):
        (isolated_env / "doc.rst").write_text("Title\n=====\n\nContent.")
        from lilbee.ingest import sync

        assert "doc.rst" in sync()["added"]

    def test_modified_file_reingested(self, _eb, _e, _vm, isolated_env):
        f = isolated_env / "changing.txt"
        f.write_text("Version 1")
        from lilbee.ingest import sync

        sync()
        f.write_text("Version 2 — different content now")
        assert "changing.txt" in sync()["updated"]

    def test_deleted_file_removed(self, _eb, _e, _vm, isolated_env):
        f = isolated_env / "temp.txt"
        f.write_text("Temporary")
        from lilbee.ingest import sync

        sync()
        f.unlink()
        assert "temp.txt" in sync()["removed"]

    def test_unchanged_file_skipped(self, _eb, _e, _vm, isolated_env):
        (isolated_env / "stable.txt").write_text("I stay the same")
        from lilbee.ingest import sync

        sync()
        result = sync()
        assert result["unchanged"] == 1
        assert result["added"] == []

    def test_unsupported_extension_skipped(self, _eb, _e, _vm, isolated_env):
        (isolated_env / "data.zip").write_bytes(b"binary data")
        from lilbee.ingest import sync

        assert sync()["added"] == []

    def test_hidden_files_skipped(self, _eb, _e, _vm, isolated_env):
        (isolated_env / ".hidden").write_text("secret")
        from lilbee.ingest import sync

        assert sync()["added"] == []

    def test_subdirectory_files_ingested(self, _eb, _e, _vm, isolated_env):
        sub = isolated_env / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested content")
        from lilbee.ingest import sync

        assert any("nested.txt" in f for f in sync()["added"])

    def test_code_file_ingested(self, _eb, _e, _vm, isolated_env):
        (isolated_env / "example.py").write_text("def hello():\n    print('hi')\n")
        from lilbee.ingest import sync

        assert "example.py" in sync()["added"]

    def test_force_rebuild_clears_and_reingests(self, _eb, _e, _vm, isolated_env):
        (isolated_env / "keep.txt").write_text("I survive rebuilds")
        from lilbee.ingest import sync

        sync()
        result = sync(force_rebuild=True)
        assert "keep.txt" in result["added"]

    def test_ingest_pdf(self, _eb, _e, _vm, isolated_env):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = isolated_env / "test.pdf"
        c = canvas.Canvas(str(pdf), pagesize=letter)
        c.drawString(72, 700, "Oil capacity is 5 quarts.")
        c.showPage()
        c.save()

        from lilbee.ingest import sync

        assert "test.pdf" in sync()["added"]

    def test_nonexistent_documents_dir(self, _eb, _e, _vm, isolated_env, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        cfg.DOCUMENTS_DIR = nonexistent
        from lilbee.ingest import sync

        result = sync()
        assert result == {"added": [], "updated": [], "removed": [], "unchanged": 0, "failed": []}
        assert nonexistent.exists()  # Directory was auto-created

    def test_ingest_error_logged_not_raised(self, _eb, _e, _vm, isolated_env):
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

    def test_ingest_error_on_update_tracked_as_failed(self, _eb, _e, _vm, isolated_env):
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

    def test_ingest_error_in_quiet_mode(self, _eb, _e, _vm, isolated_env):
        """Quiet-mode error handling works the same as non-quiet."""
        from unittest.mock import patch

        (isolated_env / "bad.txt").write_text("Will fail in quiet mode.")
        from lilbee.ingest import sync

        with patch("lilbee.ingest._ingest_file", side_effect=RuntimeError("boom")):
            result = sync(quiet=True)
        assert "bad.txt" in result["failed"]
        assert "bad.txt" not in result["added"]

    def test_ingest_error_on_update_quiet_mode(self, _eb, _e, _vm, isolated_env):
        """Quiet-mode update failure tracks in failed list."""
        from unittest.mock import patch

        f = isolated_env / "qflaky.txt"
        f.write_text("Version 1")
        from lilbee.ingest import sync

        sync()  # First ingest succeeds
        f.write_text("Version 2 — fail quietly")

        orig = __import__("lilbee.ingest", fromlist=["_ingest_file"])._ingest_file

        def _fail(path, name, ct):
            if "qflaky" in name:
                raise RuntimeError("quiet fail")
            return orig(path, name, ct)

        with patch("lilbee.ingest._ingest_file", side_effect=_fail):
            result = sync(quiet=True)
        assert "qflaky.txt" in result["failed"]
        assert "qflaky.txt" not in result["updated"]


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

    def test_skips_hidden_directories(self, isolated_env):
        from lilbee.ingest import _discover_files

        hidden = isolated_env / ".git"
        hidden.mkdir()
        (hidden / "config.txt").write_text("git config")
        (isolated_env / "visible.txt").write_text("visible")

        found = _discover_files()
        assert "visible.txt" in found
        assert not any(".git" in name for name in found)

    def test_skips_node_modules(self, isolated_env):
        from lilbee.ingest import _discover_files

        nm = isolated_env / "node_modules"
        nm.mkdir()
        (nm / "pkg.txt").write_text("npm package")
        (isolated_env / "app.txt").write_text("app code")

        found = _discover_files()
        assert "app.txt" in found
        assert not any("node_modules" in name for name in found)

    def test_skips_pycache(self, isolated_env):
        from lilbee.ingest import _discover_files

        pc = isolated_env / "__pycache__"
        pc.mkdir()
        (pc / "mod.py").write_text("cached")
        (isolated_env / "main.py").write_text("def main(): pass")

        found = _discover_files()
        assert "main.py" in found
        assert not any("__pycache__" in name for name in found)

    def test_skips_custom_ignore_via_env(self, isolated_env):
        from unittest import mock as _mock

        custom = isolated_env / "generated"
        custom.mkdir()
        (custom / "output.txt").write_text("generated output")
        (isolated_env / "source.txt").write_text("real source")

        with _mock.patch.dict("os.environ", {"LILBEE_IGNORE": "generated"}):
            import importlib

            import lilbee.config

            # Save and restore DOCUMENTS_DIR since reload resets it
            saved_docs = cfg.DOCUMENTS_DIR
            importlib.reload(lilbee.config)
            cfg.DOCUMENTS_DIR = saved_docs
            cfg.IGNORE_DIRS = lilbee.config.IGNORE_DIRS
            cfg.is_ignored_dir = lilbee.config.is_ignored_dir

            from lilbee.ingest import _discover_files

            found = _discover_files()

        assert "source.txt" in found
        assert not any("generated" in name for name in found)


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

        assert _classify_file(Path("f.zip")) is None
        assert _classify_file(Path("f.exe")) is None


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


class TestClassifyNewFormats:
    def test_office_formats(self):
        from lilbee.ingest import _classify_file

        assert _classify_file(Path("doc.docx")) == "docx"
        assert _classify_file(Path("sheet.xlsx")) == "xlsx"
        assert _classify_file(Path("slides.pptx")) == "pptx"

    def test_epub(self):
        from lilbee.ingest import _classify_file

        assert _classify_file(Path("book.epub")) == "epub"

    def test_image_formats(self):
        from lilbee.ingest import _classify_file

        assert _classify_file(Path("photo.png")) == "image"
        assert _classify_file(Path("photo.jpg")) == "image"
        assert _classify_file(Path("photo.jpeg")) == "image"
        assert _classify_file(Path("scan.tiff")) == "image"
        assert _classify_file(Path("scan.tif")) == "image"
        assert _classify_file(Path("img.bmp")) == "image"
        assert _classify_file(Path("img.webp")) == "image"

    def test_data_formats(self):
        from lilbee.ingest import _classify_file

        assert _classify_file(Path("data.csv")) == "data"
        assert _classify_file(Path("data.tsv")) == "data"


class TestDiscoverNewFormats:
    def test_new_extensions_discovered(self, isolated_env):
        from lilbee.ingest import _discover_files

        for ext in [".docx", ".xlsx", ".pptx", ".epub", ".png", ".csv", ".tsv"]:
            (isolated_env / f"test{ext}").write_bytes(b"dummy")

        found = _discover_files()
        for ext in [".docx", ".xlsx", ".pptx", ".epub", ".png", ".csv", ".tsv"]:
            assert f"test{ext}" in found


class TestIngestDocx:
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    def test_ingest_docx(self, _eb, isolated_env):
        from unittest.mock import MagicMock, patch

        from lilbee.ingest import _ingest_docx

        mock_doc = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "Hello from a DOCX document with enough text to form a chunk."
        mock_doc.paragraphs = [mock_para]
        mock_doc.tables = []

        f = isolated_env / "test.docx"
        f.write_bytes(b"fake")
        with patch("docx.Document", return_value=mock_doc):
            result = _ingest_docx(f, "test.docx")
        assert len(result) >= 1
        assert result[0]["content_type"] == "docx"

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    def test_ingest_docx_with_tables(self, _eb, isolated_env):
        from unittest.mock import MagicMock, patch

        from lilbee.ingest import _ingest_docx

        mock_doc = MagicMock()
        mock_doc.paragraphs = []

        mock_cell1 = MagicMock()
        mock_cell1.text = "Header A"
        mock_cell2 = MagicMock()
        mock_cell2.text = "Header B"
        mock_row = MagicMock()
        mock_row.cells = [mock_cell1, mock_cell2]
        mock_table = MagicMock()
        mock_table.rows = [mock_row]
        mock_doc.tables = [mock_table]

        f = isolated_env / "table.docx"
        f.write_bytes(b"fake")
        with patch("docx.Document", return_value=mock_doc):
            result = _ingest_docx(f, "table.docx")
        assert len(result) >= 1

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    def test_ingest_docx_empty(self, _eb, isolated_env):
        from unittest.mock import MagicMock, patch

        from lilbee.ingest import _ingest_docx

        mock_doc = MagicMock()
        mock_doc.paragraphs = []
        mock_doc.tables = []

        f = isolated_env / "empty.docx"
        f.write_bytes(b"fake")
        with patch("docx.Document", return_value=mock_doc):
            result = _ingest_docx(f, "empty.docx")
        assert result == []


class TestIngestXlsx:
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    def test_ingest_xlsx(self, _eb, isolated_env):
        from unittest.mock import MagicMock, patch

        from lilbee.ingest import _ingest_xlsx

        mock_ws = MagicMock()
        mock_ws.iter_rows.return_value = [("Name", "Age"), ("Alice", 30)]
        mock_wb = MagicMock()
        mock_wb.sheetnames = ["Sheet1"]
        mock_wb.__getitem__ = lambda self, key: mock_ws

        f = isolated_env / "test.xlsx"
        f.write_bytes(b"fake")
        with patch("openpyxl.load_workbook", return_value=mock_wb):
            result = _ingest_xlsx(f, "test.xlsx")
        assert len(result) >= 1
        assert result[0]["content_type"] == "xlsx"

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    def test_ingest_xlsx_empty(self, _eb, isolated_env):
        from unittest.mock import MagicMock, patch

        from lilbee.ingest import _ingest_xlsx

        mock_ws = MagicMock()
        mock_ws.iter_rows.return_value = []
        mock_wb = MagicMock()
        mock_wb.sheetnames = ["Sheet1"]
        mock_wb.__getitem__ = lambda self, key: mock_ws

        f = isolated_env / "empty.xlsx"
        f.write_bytes(b"fake")
        with patch("openpyxl.load_workbook", return_value=mock_wb):
            result = _ingest_xlsx(f, "empty.xlsx")
        assert result == []


class TestIngestPptx:
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    def test_ingest_pptx(self, _eb, isolated_env):
        from unittest.mock import MagicMock, PropertyMock, patch

        from lilbee.ingest import _ingest_pptx

        mock_shape = MagicMock()
        mock_shape.has_text_frame = True
        mock_shape.text_frame.text = "Slide title with enough content for chunking."
        mock_slide = MagicMock()
        mock_slide.shapes = [mock_shape]
        mock_prs = MagicMock()
        type(mock_prs).slides = PropertyMock(return_value=[mock_slide])

        f = isolated_env / "test.pptx"
        f.write_bytes(b"fake")
        with patch("pptx.Presentation", return_value=mock_prs):
            result = _ingest_pptx(f, "test.pptx")
        assert len(result) >= 1
        assert result[0]["content_type"] == "pptx"

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    def test_ingest_pptx_empty(self, _eb, isolated_env):
        from unittest.mock import MagicMock, PropertyMock, patch

        from lilbee.ingest import _ingest_pptx

        mock_prs = MagicMock()
        type(mock_prs).slides = PropertyMock(return_value=[])

        f = isolated_env / "empty.pptx"
        f.write_bytes(b"fake")
        with patch("pptx.Presentation", return_value=mock_prs):
            result = _ingest_pptx(f, "empty.pptx")
        assert result == []


class TestIngestEpub:
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    def test_ingest_epub(self, _eb, isolated_env):
        from unittest.mock import MagicMock, patch

        from lilbee.ingest import _ingest_epub

        mock_item = MagicMock()
        mock_item.get_content.return_value = b"<p>Chapter content with text.</p>"
        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = [mock_item]

        f = isolated_env / "test.epub"
        f.write_bytes(b"fake")
        with patch("ebooklib.epub.read_epub", return_value=mock_book):
            result = _ingest_epub(f, "test.epub")
        assert len(result) >= 1
        assert result[0]["content_type"] == "epub"

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    def test_ingest_epub_empty(self, _eb, isolated_env):
        from unittest.mock import MagicMock, patch

        from lilbee.ingest import _ingest_epub

        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = []

        f = isolated_env / "empty.epub"
        f.write_bytes(b"fake")
        with patch("ebooklib.epub.read_epub", return_value=mock_book):
            result = _ingest_epub(f, "empty.epub")
        assert result == []


class TestIngestImage:
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    def test_ingest_image(self, _eb, isolated_env):
        from unittest.mock import MagicMock, patch

        from lilbee.ingest import _ingest_image

        f = isolated_env / "test.png"
        f.write_bytes(b"fake")
        with (
            patch("PIL.Image.open", return_value=MagicMock()),
            patch("pytesseract.image_to_string", return_value="OCR text from image."),
        ):
            result = _ingest_image(f, "test.png")
        assert len(result) >= 1
        assert result[0]["content_type"] == "image"

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    def test_ingest_image_text_but_empty_chunks(self, _eb, isolated_env):
        """OCR returns text but chunk_text produces nothing (e.g., only whitespace tokens)."""
        from unittest.mock import MagicMock, patch

        from lilbee.ingest import _ingest_image

        f = isolated_env / "tiny.png"
        f.write_bytes(b"fake")
        with (
            patch("PIL.Image.open", return_value=MagicMock()),
            patch("pytesseract.image_to_string", return_value="x"),
            patch("lilbee.ingest.chunk_text", return_value=[]),
        ):
            result = _ingest_image(f, "tiny.png")
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    def test_ingest_image_no_text(self, _eb, isolated_env):
        from unittest.mock import MagicMock, patch

        from lilbee.ingest import _ingest_image

        f = isolated_env / "blank.png"
        f.write_bytes(b"fake")
        with (
            patch("PIL.Image.open", return_value=MagicMock()),
            patch("pytesseract.image_to_string", return_value="   "),
        ):
            result = _ingest_image(f, "blank.png")
        assert result == []


class TestIngestCsvTsv:
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    def test_ingest_csv(self, _eb, isolated_env):
        from lilbee.ingest import _ingest_data

        f = isolated_env / "test.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        result = _ingest_data(f, "test.csv")
        assert len(result) >= 1
        assert result[0]["content_type"] == "data"

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    def test_ingest_tsv(self, _eb, isolated_env):
        from lilbee.ingest import _ingest_data

        f = isolated_env / "test.tsv"
        f.write_text("name\tage\nAlice\t30\nBob\t25\n")
        result = _ingest_data(f, "test.tsv")
        assert len(result) >= 1
        assert result[0]["content_type"] == "data"

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    def test_ingest_csv_empty(self, _eb, isolated_env):
        from lilbee.ingest import _ingest_data

        f = isolated_env / "empty.csv"
        f.write_text("")
        result = _ingest_data(f, "empty.csv")
        assert result == []
