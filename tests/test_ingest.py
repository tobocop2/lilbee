"""Tests for the document sync engine (mocked — no live server needed)."""

from dataclasses import fields, replace
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from lilbee.config import cfg


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths to temp dir for every test."""
    snapshot = replace(cfg)

    docs = tmp_path / "documents"
    docs.mkdir()
    cfg.documents_dir = docs
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"

    yield docs

    for f in fields(cfg):
        setattr(cfg, f.name, getattr(snapshot, f.name))


def _fake_embed_batch(texts):
    return [[0.1] * 768 for _ in texts]


def _fake_embed(text):
    return [0.1] * 768


def _make_kreuzberg_result(text="Some extracted text. " * 20, num_chunks=1, has_pages=False):
    """Build a mock kreuzberg ExtractionResult."""
    chunks = []
    for i in range(num_chunks):
        chunk_text = text[i * len(text) // num_chunks : (i + 1) * len(text) // num_chunks]
        metadata = {
            "byte_start": 0,
            "byte_end": len(chunk_text),
            "chunk_index": i,
            "total_chunks": num_chunks,
            "token_count": None,
        }
        if has_pages:
            metadata["first_page"] = i + 1
            metadata["last_page"] = i + 1
        chunk = mock.MagicMock()
        chunk.content = chunk_text
        chunk.metadata = metadata
        chunks.append(chunk)

    result = mock.MagicMock()
    result.chunks = chunks
    result.content = text
    return result


def _make_empty_result():
    """Build a mock kreuzberg ExtractionResult with no chunks."""
    result = mock.MagicMock()
    result.chunks = []
    result.content = ""
    return result


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
@mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result())
class TestSync:
    async def test_empty_documents_dir(self, _kf, _eb, _e, _vm, isolated_env):
        from lilbee.ingest import SyncResult, sync

        result = await sync()
        assert result == SyncResult()

    async def test_ingest_text_file(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "test.txt").write_text("Hello world. This is a test document.")
        from lilbee.ingest import sync

        result = await sync()
        assert "test.txt" in result.added

    async def test_quiet_mode_suppresses_progress(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "quiet.txt").write_text("Quiet mode test content.")
        from lilbee.ingest import sync

        result = await sync(quiet=True)
        assert "quiet.txt" in result.added

    async def test_ingest_markdown_file(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "readme.md").write_text("# Title\n\nSome markdown content.")
        from lilbee.ingest import sync

        assert "readme.md" in (await sync()).added

    async def test_ingest_html_file(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "page.html").write_text("<p>Content</p>")
        from lilbee.ingest import sync

        assert "page.html" in (await sync()).added

    async def test_ingest_rst_file(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "doc.rst").write_text("Title\n=====\n\nContent.")
        from lilbee.ingest import sync

        assert "doc.rst" in (await sync()).added

    async def test_modified_file_reingested(self, _kf, _eb, _e, _vm, isolated_env):
        f = isolated_env / "changing.txt"
        f.write_text("Version 1")
        from lilbee.ingest import sync

        await sync()
        f.write_text("Version 2 — different content now")
        assert "changing.txt" in (await sync()).updated

    async def test_deleted_file_removed(self, _kf, _eb, _e, _vm, isolated_env):
        f = isolated_env / "temp.txt"
        f.write_text("Temporary")
        from lilbee.ingest import sync

        await sync()
        f.unlink()
        assert "temp.txt" in (await sync()).removed

    async def test_unchanged_file_skipped(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "stable.txt").write_text("I stay the same")
        from lilbee.ingest import sync

        await sync()
        result = await sync()
        assert result.unchanged == 1
        assert result.added == []

    async def test_unsupported_extension_skipped(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "data.zip").write_bytes(b"binary data")
        from lilbee.ingest import sync

        assert (await sync()).added == []

    async def test_hidden_files_skipped(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / ".hidden").write_text("secret")
        from lilbee.ingest import sync

        assert (await sync()).added == []

    async def test_subdirectory_files_ingested(self, _kf, _eb, _e, _vm, isolated_env):
        sub = isolated_env / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested content")
        from lilbee.ingest import sync

        assert any("nested.txt" in f for f in (await sync()).added)

    async def test_code_file_ingested(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "example.py").write_text("def hello():\n    print('hi')\n")
        from lilbee.ingest import sync

        assert "example.py" in (await sync()).added

    async def test_force_rebuild_clears_and_reingests(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "keep.txt").write_text("I survive rebuilds")
        from lilbee.ingest import sync

        await sync()
        result = await sync(force_rebuild=True)
        assert "keep.txt" in result.added

    async def test_ingest_pdf(self, _kf, _eb, _e, _vm, isolated_env):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = isolated_env / "test.pdf"
        c = canvas.Canvas(str(pdf), pagesize=letter)
        c.drawString(72, 700, "Oil capacity is 5 quarts.")
        c.showPage()
        c.save()

        from lilbee.ingest import sync

        assert "test.pdf" in (await sync()).added

    async def test_nonexistent_documents_dir(self, _kf, _eb, _e, _vm, isolated_env, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        cfg.documents_dir = nonexistent
        from lilbee.ingest import SyncResult, sync

        result = await sync()
        assert result == SyncResult()
        assert nonexistent.exists()  # Directory was auto-created

    async def test_ingest_error_logged_not_raised(self, _kf, _eb, _e, _vm, isolated_env):
        """A file that fails ingestion is logged but doesn't crash sync."""
        from unittest.mock import patch

        (isolated_env / "good.txt").write_text("This is fine.")
        (isolated_env / "bad.txt").write_text("This will fail.")

        from lilbee.ingest import sync

        orig_ingest = __import__("lilbee.ingest", fromlist=["_ingest_file"])._ingest_file

        async def _failing_ingest(path, name, content_type):
            if "bad" in name:
                raise RuntimeError("simulated failure")
            return await orig_ingest(path, name, content_type)

        with patch("lilbee.ingest._ingest_file", side_effect=_failing_ingest):
            result = await sync()
        # good.txt was added, bad.txt failed
        assert "good.txt" in result.added
        assert "bad.txt" not in result.added
        assert "bad.txt" in result.failed

    async def test_ingest_error_on_update_tracked_as_failed(self, _kf, _eb, _e, _vm, isolated_env):
        """A file that fails re-ingestion on update goes to failed, not updated."""
        from unittest.mock import patch

        f = isolated_env / "flaky.txt"
        f.write_text("Version 1")

        from lilbee.ingest import sync

        await sync()  # First ingest succeeds

        f.write_text("Version 2 — will fail")

        orig_ingest = __import__("lilbee.ingest", fromlist=["_ingest_file"])._ingest_file

        async def _failing_ingest(path, name, content_type):
            if "flaky" in name:
                raise RuntimeError("simulated failure on update")
            return await orig_ingest(path, name, content_type)

        with patch("lilbee.ingest._ingest_file", side_effect=_failing_ingest):
            result = await sync()
        assert "flaky.txt" not in result.updated
        assert "flaky.txt" in result.failed

    async def test_ingest_error_in_quiet_mode(self, _kf, _eb, _e, _vm, isolated_env):
        """Quiet-mode error handling works the same as non-quiet."""
        from unittest.mock import patch

        (isolated_env / "bad.txt").write_text("Will fail in quiet mode.")
        from lilbee.ingest import sync

        async def _fail(*args):
            raise RuntimeError("boom")

        with patch("lilbee.ingest._ingest_file", side_effect=_fail):
            result = await sync(quiet=True)
        assert "bad.txt" in result.failed
        assert "bad.txt" not in result.added

    async def test_ingest_error_on_update_quiet_mode(self, _kf, _eb, _e, _vm, isolated_env):
        """Quiet-mode update failure tracks in failed list."""
        from unittest.mock import patch

        f = isolated_env / "qflaky.txt"
        f.write_text("Version 1")
        from lilbee.ingest import sync

        await sync()  # First ingest succeeds
        f.write_text("Version 2 — fail quietly")

        orig = __import__("lilbee.ingest", fromlist=["_ingest_file"])._ingest_file

        async def _fail(path, name, ct):
            if "qflaky" in name:
                raise RuntimeError("quiet fail")
            return await orig(path, name, ct)

        with patch("lilbee.ingest._ingest_file", side_effect=_fail):
            result = await sync(quiet=True)
        assert "qflaky.txt" in result.failed
        assert "qflaky.txt" not in result.updated


class TestIngestHelpers:
    """Cover edge cases in ingest_document and ingest_code_sync."""

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_empty_result())
    async def testingest_document_empty_chunks(self, _kf, _eb, isolated_env):
        """Document that produces no chunks returns empty list."""
        from lilbee.ingest import ingest_document

        f = isolated_env / "empty.txt"
        f.write_text("   ")
        result = await ingest_document(f, "empty.txt", "text")
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    async def test_ingest_code_empty_chunks(self, _eb, isolated_env):
        """Code file that produces no chunks returns empty list."""
        from unittest.mock import patch

        from lilbee.ingest import ingest_code_sync

        f = isolated_env / "empty.py"
        f.write_text("")
        with patch("lilbee.code_chunker.chunk_code", return_value=[]):
            result = ingest_code_sync(f, "empty.py")
            assert result == []

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def testingest_document_pdf_with_pages(self, mock_kf, _eb, isolated_env):
        """PDF document returns records with page metadata."""
        mock_kf.return_value = _make_kreuzberg_result(
            text="Page 1 content. " * 10 + "Page 2 content. " * 10,
            num_chunks=2,
            has_pages=True,
        )
        from lilbee.ingest import ingest_document

        f = isolated_env / "test.pdf"
        f.write_bytes(b"fake")
        result = await ingest_document(f, "test.pdf", "pdf")
        assert len(result) == 2
        assert result[0]["page_start"] == 1
        assert result[1]["page_start"] == 2


class TestCancellation:
    @mock.patch("lilbee.embedder.validate_model")
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch(
        "kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_cancelled_error_propagates(self, _kf, _eb, _vm, isolated_env):
        """CancelledError in _process_one is re-raised, not swallowed."""
        import asyncio

        async def _cancel(*args):
            raise asyncio.CancelledError()

        with mock.patch("lilbee.ingest._ingest_file", side_effect=_cancel):
            from lilbee.ingest import ingest_batch

            added = ["cancel.txt"]
            with pytest.raises(asyncio.CancelledError):
                await ingest_batch(
                    [("cancel.txt", isolated_env / "cancel.txt", "text")],
                    added,
                    [],
                    [],
                    quiet=True,
                )


class TestDiscoverFiles:
    def test_nonexistent_dir_returns_empty(self, isolated_env, tmp_path):
        cfg.documents_dir = tmp_path / "does_not_exist"
        from lilbee.ingest import discover_files

        assert discover_files() == {}

    def test_skips_hidden_directories(self, isolated_env):
        from lilbee.ingest import discover_files

        hidden = isolated_env / ".git"
        hidden.mkdir()
        (hidden / "config.txt").write_text("git config")
        (isolated_env / "visible.txt").write_text("visible")

        found = discover_files()
        assert "visible.txt" in found
        assert not any(".git" in name for name in found)

    def test_skips_node_modules(self, isolated_env):
        from lilbee.ingest import discover_files

        nm = isolated_env / "node_modules"
        nm.mkdir()
        (nm / "pkg.txt").write_text("npm package")
        (isolated_env / "app.txt").write_text("app code")

        found = discover_files()
        assert "app.txt" in found
        assert not any("node_modules" in name for name in found)

    def test_skips_pycache(self, isolated_env):
        from lilbee.ingest import discover_files

        pc = isolated_env / "__pycache__"
        pc.mkdir()
        (pc / "mod.py").write_text("cached")
        (isolated_env / "main.py").write_text("def main(): pass")

        found = discover_files()
        assert "main.py" in found
        assert not any("__pycache__" in name for name in found)

    def test_skips_custom_ignore_via_env(self, isolated_env):
        from lilbee.ingest import discover_files

        custom = isolated_env / "generated"
        custom.mkdir()
        (custom / "output.txt").write_text("generated output")
        (isolated_env / "source.txt").write_text("real source")

        cfg.ignore_dirs = cfg.ignore_dirs | frozenset({"generated"})
        found = discover_files()

        assert "source.txt" in found
        assert not any("generated" in name for name in found)


class TestClassifyFile:
    def test_pdf(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("doc.pdf")) == "pdf"

    def test_text_types(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("f.md")) == "text"
        assert classify_file(Path("f.txt")) == "text"
        assert classify_file(Path("f.html")) == "text"
        assert classify_file(Path("f.rst")) == "text"

    def test_code_types(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("f.py")) == "code"
        assert classify_file(Path("f.js")) == "code"
        assert classify_file(Path("f.go")) == "code"

    def test_unsupported(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("f.zip")) is None
        assert classify_file(Path("f.exe")) is None


class TestFileHash:
    def test_deterministic(self, tmp_path):
        from lilbee.ingest import file_hash

        f = tmp_path / "test.txt"
        f.write_text("hello")
        h1 = file_hash(f)
        h2 = file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        from lilbee.ingest import file_hash

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert file_hash(f1) != file_hash(f2)


class TestApplyResultZeroChunks:
    def test_zero_chunks_not_recorded_as_added(self):
        from lilbee.ingest import _apply_result, _IngestResult

        added = ["scanned.pdf"]
        updated: list[str] = []
        failed: list[str] = []
        result = _IngestResult("scanned.pdf", Path("scanned.pdf"), chunk_count=0, error=None)
        _apply_result(result, added, updated, failed)
        assert "scanned.pdf" not in added
        assert "scanned.pdf" not in failed

    def test_zero_chunks_not_recorded_as_updated(self):
        from lilbee.ingest import _apply_result, _IngestResult

        added: list[str] = []
        updated = ["scanned.pdf"]
        failed: list[str] = []
        result = _IngestResult("scanned.pdf", Path("scanned.pdf"), chunk_count=0, error=None)
        _apply_result(result, added, updated, failed)
        assert "scanned.pdf" not in updated
        assert "scanned.pdf" not in failed

    def test_nonzero_chunks_recorded(self):
        from lilbee.ingest import _apply_result, _IngestResult

        added = ["doc.pdf"]
        updated: list[str] = []
        failed: list[str] = []
        result = _IngestResult("doc.pdf", Path("doc.pdf"), chunk_count=5, error=None)
        with (
            mock.patch("lilbee.ingest.store") as mock_store,
            mock.patch("lilbee.ingest.file_hash", return_value="abc123"),
        ):
            _apply_result(result, added, updated, failed)
        mock_store.upsert_source.assert_called_once()
        assert "doc.pdf" in added


class TestSyncResultStr:
    def test_str_no_failures(self):
        from lilbee.ingest import SyncResult

        result = SyncResult(
            added=["a.txt"], updated=["b.txt"], removed=["c.txt"], unchanged=2, failed=[]
        )
        text = str(result)
        assert "Added: 1" in text
        assert "Updated: 1" in text
        assert "Removed: 1" in text
        assert "Unchanged: 2" in text
        assert "Failed: 0" in text

    def test_str_with_failures(self):
        from lilbee.ingest import SyncResult

        result = SyncResult(failed=["x.txt", "y.txt"])
        text = str(result)
        assert "Failed: 2" in text
        assert "[red]x.txt[/red]" in text
        assert "[red]y.txt[/red]" in text

    def test_repr_matches_str(self):
        from lilbee.ingest import SyncResult

        result = SyncResult(added=["a.txt"])
        assert repr(result) == str(result)


class TestClassifyNewFormats:
    def test_office_formats(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("doc.docx")) == "docx"
        assert classify_file(Path("sheet.xlsx")) == "xlsx"
        assert classify_file(Path("slides.pptx")) == "pptx"

    def test_epub(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("book.epub")) == "epub"

    def test_image_formats(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("photo.png")) == "image"
        assert classify_file(Path("photo.jpg")) == "image"
        assert classify_file(Path("photo.jpeg")) == "image"
        assert classify_file(Path("scan.tiff")) == "image"
        assert classify_file(Path("scan.tif")) == "image"
        assert classify_file(Path("img.bmp")) == "image"
        assert classify_file(Path("img.webp")) == "image"

    def test_data_formats(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("data.csv")) == "data"
        assert classify_file(Path("data.tsv")) == "data"


class TestDiscoverNewFormats:
    def test_new_extensions_discovered(self, isolated_env):
        from lilbee.ingest import discover_files

        for ext in [".docx", ".xlsx", ".pptx", ".epub", ".png", ".csv", ".tsv"]:
            (isolated_env / f"test{ext}").write_bytes(b"dummy")

        found = discover_files()
        for ext in [".docx", ".xlsx", ".pptx", ".epub", ".png", ".csv", ".tsv"]:
            assert f"test{ext}" in found


class TestKreuzbergConfig:
    def test_pdf_gets_page_config(self):
        from lilbee.ingest import kreuzberg_config

        config = kreuzberg_config("pdf")
        assert config.pages is not None

    def test_pdf_no_markdown_output(self):
        from lilbee.ingest import kreuzberg_config

        config = kreuzberg_config("pdf")
        assert getattr(config, "output_format", None) != "markdown"

    def test_non_pdf_no_page_config(self):
        from lilbee.ingest import kreuzberg_config

        config = kreuzberg_config("text")
        assert config.pages is None

    def test_chunking_config_set(self):
        from lilbee.ingest import kreuzberg_config

        config = kreuzberg_config("text")
        assert config.chunking is not None

    @pytest.mark.parametrize("content_type", ["text", "docx", "xlsx", "pptx", "epub", "image"])
    def test_non_pdf_gets_markdown_output(self, content_type):
        from lilbee.ingest import kreuzberg_config

        config = kreuzberg_config(content_type)
        assert config.output_format == "markdown"


class TestClassifyStructuredFormats:
    def test_xml_classified_as_xml(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("data.xml")) == "xml"

    def test_json_classified_as_json(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("data.json")) == "json"

    def test_jsonl_classified_as_json(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("data.jsonl")) == "json"

    def test_yaml_classified_as_text(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("config.yaml")) == "text"

    def test_yml_classified_as_text(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("config.yml")) == "text"

    def test_csv_still_classified_as_data(self):
        from lilbee.ingest import classify_file

        assert classify_file(Path("data.csv")) == "data"


def _fake_preprocess(path: Path) -> str:
    return f"Preprocessed content from {path.name}. " * 20


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
@mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result())
class TestSyncStructuredFormats:
    @mock.patch("lilbee.preprocessors.preprocess_xml", side_effect=_fake_preprocess)
    async def test_xml_file_ingested(self, _px, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "data.xml").write_text("<root><item>value</item></root>")
        from lilbee.ingest import sync

        result = await sync()
        assert "data.xml" in result.added

    @mock.patch("lilbee.preprocessors.preprocess_json", side_effect=_fake_preprocess)
    async def test_json_file_ingested(self, _pj, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "data.json").write_text('{"key": "value"}')
        from lilbee.ingest import sync

        result = await sync()
        assert "data.json" in result.added

    @mock.patch("lilbee.preprocessors.preprocess_json", side_effect=_fake_preprocess)
    async def test_jsonl_file_ingested(self, _pj, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "data.jsonl").write_text('{"key": "value"}\n{"key2": "value2"}')
        from lilbee.ingest import sync

        result = await sync()
        assert "data.jsonl" in result.added

    @mock.patch("lilbee.preprocessors.preprocess_csv", side_effect=_fake_preprocess)
    async def test_csv_file_ingested_via_preprocessor(self, _pc, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "data.csv").write_text("name,age\nAlice,30\nBob,25")
        from lilbee.ingest import sync

        result = await sync()
        assert "data.csv" in result.added


class TestHasMeaningfulText:
    def test_empty_chunks_returns_false(self):
        from lilbee.ingest import _has_meaningful_text

        result = mock.MagicMock(chunks=[])
        assert _has_meaningful_text(result) is False

    def test_no_chunks_attr_returns_false(self):
        from lilbee.ingest import _has_meaningful_text

        result = object()
        assert _has_meaningful_text(result) is False

    def test_short_text_returns_false(self):
        from lilbee.ingest import _has_meaningful_text

        chunk = mock.MagicMock()
        chunk.content = "short"
        result = mock.MagicMock(chunks=[chunk])
        assert _has_meaningful_text(result) is False

    def test_meaningful_text_returns_true(self):
        from lilbee.ingest import _has_meaningful_text

        chunk = mock.MagicMock()
        chunk.content = "A" * 100
        result = mock.MagicMock(chunks=[chunk])
        assert _has_meaningful_text(result) is True


class TestVisionFallback:
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_empty_result())
    async def test_vision_fallback_called_for_empty_pdf(self, _kf, _eb, isolated_env):
        """When PDF extraction is empty and vision_model is set, fall back to vision."""
        cfg.vision_model = "test-vision"
        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        vision_pages = [(1, "Vision extracted text. " * 10)]
        with mock.patch(
            "lilbee.vision.extract_pdf_vision", return_value=vision_pages
        ) as mock_vision:
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "scanned.pdf", "pdf")
        mock_vision.assert_called_once_with(f, "test-vision")
        assert len(result) > 0
        assert result[0]["content_type"] == "pdf"
        assert result[0]["page_start"] == 1

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_empty_result())
    async def test_vision_fallback_not_called_without_model(self, _kf, _eb, isolated_env):
        """When vision_model is empty, no fallback occurs."""
        cfg.vision_model = ""
        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        with mock.patch("lilbee.vision.extract_pdf_vision") as mock_vision:
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "scanned.pdf", "pdf")
        mock_vision.assert_not_called()
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_vision_fallback_not_called_for_non_pdf(self, mock_kf, _eb, isolated_env):
        """Vision fallback only triggers for PDF content type."""
        mock_kf.return_value = _make_empty_result()
        cfg.vision_model = "test-vision"
        f = isolated_env / "doc.txt"
        f.write_text("")

        with mock.patch("lilbee.vision.extract_pdf_vision") as mock_vision:
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "doc.txt", "text")
        mock_vision.assert_not_called()
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_empty_result())
    async def test_vision_fallback_empty_vision_text_returns_empty(self, _kf, _eb, isolated_env):
        """When vision also returns empty text, return empty list."""
        cfg.vision_model = "test-vision"
        f = isolated_env / "blank.pdf"
        f.write_bytes(b"fake pdf")

        with mock.patch("lilbee.vision.extract_pdf_vision", return_value=[]):
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "blank.pdf", "pdf")
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_no_vision_fallback_when_text_meaningful(self, mock_kf, _eb, isolated_env):
        """When kreuzberg produces meaningful text, no vision fallback."""
        mock_kf.return_value = _make_kreuzberg_result(
            text="Meaningful PDF content. " * 20, num_chunks=1, has_pages=True
        )
        cfg.vision_model = "test-vision"
        f = isolated_env / "good.pdf"
        f.write_bytes(b"fake pdf")

        with mock.patch("lilbee.vision.extract_pdf_vision") as mock_vision:
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "good.pdf", "pdf")
        mock_vision.assert_not_called()
        assert len(result) > 0

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_empty_result())
    async def test_vision_fallback_no_chunks_returns_empty(self, _kf, _eb, isolated_env):
        """When vision text produces no chunks, return empty list."""
        cfg.vision_model = "test-vision"
        f = isolated_env / "nochunks.pdf"
        f.write_bytes(b"fake pdf")

        with (
            mock.patch("lilbee.vision.extract_pdf_vision", return_value=[(1, "Some text")]),
            mock.patch("lilbee.ingest.chunk_text", return_value=[]),
        ):
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "nochunks.pdf", "pdf")
        assert result == []


class TestIngestStructuredEdgeCases:
    async def test_empty_preprocessed_text_returns_empty(self, isolated_env):
        from lilbee.ingest import _PREPROCESSORS, ingest_structured

        with mock.patch.dict(_PREPROCESSORS, {"xml": lambda _: "   "}):
            result = await ingest_structured(isolated_env / "e.xml", "e.xml", "xml")
        assert result == []

    async def test_no_chunks_returns_empty(self, isolated_env):
        from lilbee.ingest import _PREPROCESSORS, ingest_structured

        with (
            mock.patch.dict(_PREPROCESSORS, {"xml": lambda _: "some content here"}),
            mock.patch("lilbee.ingest.chunk_text", return_value=[]),
        ):
            result = await ingest_structured(isolated_env / "s.xml", "s.xml", "xml")
        assert result == []
