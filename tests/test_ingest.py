"""Tests for the document sync engine (mocked — no live server needed)."""

from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from lilbee.config import cfg


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths to temp dir for every test."""
    snapshot = cfg.model_copy()

    docs = tmp_path / "documents"
    docs.mkdir()
    cfg.documents_dir = docs
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.concept_graph = False

    yield docs

    # Reset store singleton so next test gets fresh connection
    import lilbee.store as store_mod

    store_mod._default_store = None

    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _fake_embed_batch(texts, **kwargs):
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
    async def test_empty_documents_dir(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        from lilbee.ingest import SyncResult, sync

        result = await sync()
        assert result == SyncResult()

    async def test_ingest_text_file(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "test.txt").write_text("Hello world. This is a test document.")
        from lilbee.ingest import sync

        result = await sync()
        assert "concept_test2.txt" in result.added

    async def test_quiet_mode_suppresses_progress(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "quiet.txt").write_text("Quiet mode test content.")
        from lilbee.ingest import sync

        result = await sync(quiet=True)
        assert "quiet.txt" in result.added

    async def test_on_progress_callback_quiet(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "cb.txt").write_text("Callback test.")
        from lilbee.ingest import sync

        events: list[tuple[str, dict]] = []
        result = await sync(quiet=True, on_progress=lambda t, d: events.append((t, d)))
        assert "cb.txt" in result.added
        event_types = [t for t, _ in events]
        assert "file_start" in event_types
        assert "file_done" in event_types
        assert "done" in event_types
        file_done = next(d for t, d in events if t == "file_done")
        assert file_done["file"] == "cb.txt"
        assert file_done["status"] == "ok"

    async def test_on_progress_callback_with_progress_bar(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "cb2.txt").write_text("Callback with progress bar.")
        from lilbee.ingest import sync

        events: list[tuple[str, dict]] = []
        result = await sync(quiet=False, on_progress=lambda t, d: events.append((t, d)))
        assert "cb2.txt" in result.added
        event_types = [t for t, _ in events]
        assert "file_done" in event_types
        file_done = next(d for t, d in events if t == "file_done")
        assert file_done["status"] == "ok"

    async def test_ingest_markdown_file(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "readme.md").write_text("# Title\n\nSome markdown content.")
        from lilbee.ingest import sync

        assert "readme.md" in (await sync()).added

    async def test_ingest_html_file(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "page.html").write_text("<p>Content</p>")
        from lilbee.ingest import sync

        assert "page.html" in (await sync()).added

    async def test_ingest_rst_file(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "doc.rst").write_text("Title\n=====\n\nContent.")
        from lilbee.ingest import sync

        assert "doc.rst" in (await sync()).added

    async def test_modified_file_reingested(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        f = isolated_env / "changing.txt"
        f.write_text("Version 1")
        from lilbee.ingest import sync

        await sync()
        f.write_text("Version 2 — different content now")
        assert "changing.txt" in (await sync()).updated

    async def test_deleted_file_removed(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        f = isolated_env / "temp.txt"
        f.write_text("Temporary")
        from lilbee.ingest import sync

        await sync()
        f.unlink()
        assert "temp.txt" in (await sync()).removed

    async def test_unchanged_file_skipped(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "stable.txt").write_text("I stay the same")
        from lilbee.ingest import sync

        await sync()
        result = await sync()
        assert result.unchanged == 1
        assert result.added == []

    async def test_unsupported_extension_skipped(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "data.zip").write_bytes(b"binary data")
        from lilbee.ingest import sync

        assert (await sync()).added == []

    async def test_hidden_files_skipped(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / ".hidden").write_text("secret")
        from lilbee.ingest import sync

        assert (await sync()).added == []

    async def test_subdirectory_files_ingested(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        sub = isolated_env / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested content")
        from lilbee.ingest import sync

        assert any("nested.txt" in f for f in (await sync()).added)

    async def test_code_file_ingested(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "example.py").write_text("def hello():\n    print('hi')\n")
        from lilbee.ingest import sync

        assert "example.py" in (await sync()).added

    async def test_force_rebuild_clears_and_reingests(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        (isolated_env / "keep.txt").write_text("I survive rebuilds")
        from lilbee.ingest import sync

        await sync()
        result = await sync(force_rebuild=True)
        assert "keep.txt" in result.added

    async def test_ingest_pdf(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = isolated_env / "test.pdf"
        c = canvas.Canvas(str(pdf), pagesize=letter)
        c.drawString(72, 700, "Oil capacity is 5 quarts.")
        c.showPage()
        c.save()

        from lilbee.ingest import sync

        assert "test.pdf" in (await sync()).added

    async def test_nonexistent_documents_dir(
        self,
        mock_extract_file,
        mock_embed_batch,
        mock_embed,
        mock_validate_model,
        isolated_env,
        tmp_path,
    ):
        nonexistent = tmp_path / "nonexistent"
        cfg.documents_dir = nonexistent
        from lilbee.ingest import SyncResult, sync

        result = await sync()
        assert result == SyncResult()
        assert nonexistent.exists()  # Directory was auto-created

    async def test_ingest_error_logged_not_raised(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        """A file that fails ingestion is logged but doesn't crash sync."""
        from unittest.mock import patch

        (isolated_env / "good.txt").write_text("This is fine.")
        (isolated_env / "bad.txt").write_text("This will fail.")

        from lilbee.ingest import sync

        orig_ingest = __import__("lilbee.ingest", fromlist=["_ingest_file"])._ingest_file

        async def _failing_ingest(path, name, content_type, **kwargs):
            if "bad" in name:
                raise RuntimeError("simulated failure")
            return await orig_ingest(path, name, content_type, **kwargs)

        with patch("lilbee.ingest._ingest_file", side_effect=_failing_ingest):
            result = await sync()
        # good.txt was added, bad.txt failed
        assert "good.txt" in result.added
        assert "bad.txt" not in result.added
        assert "bad.txt" in result.failed

    async def test_ingest_error_on_update_tracked_as_failed(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        """A file that fails re-ingestion on update goes to failed, not updated."""
        from unittest.mock import patch

        f = isolated_env / "flaky.txt"
        f.write_text("Version 1")

        from lilbee.ingest import sync

        await sync()  # First ingest succeeds

        f.write_text("Version 2 — will fail")

        orig_ingest = __import__("lilbee.ingest", fromlist=["_ingest_file"])._ingest_file

        async def _failing_ingest(path, name, content_type, **kwargs):
            if "flaky" in name:
                raise RuntimeError("simulated failure on update")
            return await orig_ingest(path, name, content_type, **kwargs)

        with patch("lilbee.ingest._ingest_file", side_effect=_failing_ingest):
            result = await sync()
        assert "flaky.txt" not in result.updated
        assert "flaky.txt" in result.failed

    async def test_ingest_error_in_quiet_mode(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
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

    async def test_ingest_error_on_update_quiet_mode(
        self, mock_extract_file, mock_embed_batch, mock_embed, mock_validate_model, isolated_env
    ):
        """Quiet-mode update failure tracks in failed list."""
        from unittest.mock import patch

        f = isolated_env / "qflaky.txt"
        f.write_text("Version 1")
        from lilbee.ingest import sync

        await sync()  # First ingest succeeds
        f.write_text("Version 2 — fail quietly")

        orig = __import__("lilbee.ingest", fromlist=["_ingest_file"])._ingest_file

        async def _fail(path, name, ct, **kwargs):
            if "qflaky" in name:
                raise RuntimeError("quiet fail")
            return await orig(path, name, ct, **kwargs)

        with patch("lilbee.ingest._ingest_file", side_effect=_fail):
            result = await sync(quiet=True)
        assert "qflaky.txt" in result.failed
        assert "qflaky.txt" not in result.updated


class TestIngestHelpers:
    """Cover edge cases in ingest_document and ingest_code_sync."""

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_empty_result())
    async def testingest_document_empty_chunks(
        self, mock_extract_file, mock_embed_batch, isolated_env
    ):
        """Document that produces no chunks returns empty list."""
        from lilbee.ingest import ingest_document

        f = isolated_env / "empty.txt"
        f.write_text("   ")
        result = await ingest_document(f, "empty.txt", "text")
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    async def test_ingest_code_empty_chunks(self, mock_embed_batch, isolated_env):
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
    async def testingest_document_pdf_with_pages(self, mock_kf, mock_embed_batch, isolated_env):
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
    async def test_cancelled_error_propagates(
        self, mock_extract_file, mock_embed_batch, mock_validate_model, isolated_env
    ):
        """CancelledError in _process_one is re-raised, not swallowed."""
        import asyncio

        async def _cancel(*args, **kwargs):
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

    @pytest.mark.parametrize(
        "ignored_dir, child_file",
        [
            (".git", "config.txt"),
            ("node_modules", "pkg.txt"),
            ("__pycache__", "mod.py"),
        ],
    )
    def test_skips_ignored_directories(self, ignored_dir, child_file, isolated_env):
        from lilbee.ingest import discover_files

        d = isolated_env / ignored_dir
        d.mkdir()
        (d / child_file).write_text("content")
        (isolated_env / "visible.txt").write_text("visible")

        found = discover_files()
        assert "visible.txt" in found
        assert not any(ignored_dir in name for name in found)

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
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("doc.pdf", "pdf"),
            ("f.md", "text"),
            ("f.txt", "text"),
            ("f.html", "text"),
            ("f.rst", "text"),
            ("f.py", "code"),
            ("f.js", "code"),
            ("f.go", "code"),
            ("f.zip", None),
            ("f.exe", None),
        ],
    )
    def test_classify(self, filename, expected):
        from lilbee.ingest import classify_file

        assert classify_file(Path(filename)) == expected


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
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("doc.docx", "docx"),
            ("sheet.xlsx", "xlsx"),
            ("slides.pptx", "pptx"),
            ("book.epub", "epub"),
            ("photo.png", "image"),
            ("photo.jpg", "image"),
            ("photo.jpeg", "image"),
            ("scan.tiff", "image"),
            ("scan.tif", "image"),
            ("img.bmp", "image"),
            ("img.webp", "image"),
            ("data.csv", "data"),
            ("data.tsv", "data"),
        ],
    )
    def test_classify(self, filename, expected):
        from lilbee.ingest import classify_file

        assert classify_file(Path(filename)) == expected


class TestDiscoverNewFormats:
    def test_new_extensions_discovered(self, isolated_env):
        from lilbee.ingest import discover_files

        for ext in [".docx", ".xlsx", ".pptx", ".epub", ".png", ".csv", ".tsv"]:
            (isolated_env / f"test{ext}").write_bytes(b"dummy")

        found = discover_files()
        for ext in [".docx", ".xlsx", ".pptx", ".epub", ".png", ".csv", ".tsv"]:
            assert f"test{ext}" in found


class TestExtractionConfig:
    def test_pdf_gets_page_config(self):
        from lilbee.ingest import extraction_config

        config = extraction_config("pdf")
        assert config.pages is not None

    def test_pdf_no_markdown_output(self):
        from lilbee.ingest import extraction_config

        config = extraction_config("pdf")
        assert getattr(config, "output_format", None) != "markdown"

    def test_non_pdf_no_page_config(self):
        from lilbee.ingest import extraction_config

        config = extraction_config("text")
        assert config.pages is None

    def test_chunking_config_set(self):
        from lilbee.ingest import extraction_config

        config = extraction_config("text")
        assert config.chunking is not None

    @pytest.mark.parametrize("content_type", ["text", "docx", "xlsx", "pptx", "epub", "image"])
    def test_non_pdf_gets_markdown_output(self, content_type):
        from lilbee.ingest import extraction_config

        config = extraction_config(content_type)
        assert config.output_format == "markdown"


class TestClassifyStructuredFormats:
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("data.xml", "xml"),
            ("data.json", "json"),
            ("data.jsonl", "json"),
            ("config.yaml", "text"),
            ("config.yml", "text"),
            ("data.csv", "data"),
        ],
    )
    def test_classify(self, filename, expected):
        from lilbee.ingest import classify_file

        assert classify_file(Path(filename)) == expected


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
@mock.patch("kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result())
class TestSyncStructuredFormats:
    async def test_xml_file_ingested(
        self,
        mock_extract_file,
        mock_embed_batch,
        mock_embed,
        mock_validate_model,
        isolated_env,
    ):
        (isolated_env / "data.xml").write_text("<root><item>value</item></root>")
        from lilbee.ingest import sync

        result = await sync()
        assert "data.xml" in result.added

    async def test_json_file_ingested(
        self,
        mock_extract_file,
        mock_embed_batch,
        mock_embed,
        mock_validate_model,
        isolated_env,
    ):
        (isolated_env / "data.json").write_text('{"key": "value"}')
        from lilbee.ingest import sync

        result = await sync()
        assert "data.json" in result.added

    async def test_jsonl_file_ingested(
        self,
        mock_extract_file,
        mock_embed_batch,
        mock_embed,
        mock_validate_model,
        isolated_env,
    ):
        (isolated_env / "data.jsonl").write_text('{"key": "value"}\n{"key2": "value2"}')
        from lilbee.ingest import sync

        result = await sync()
        assert "data.jsonl" in result.added

    async def test_csv_file_ingested(
        self,
        mock_extract_file,
        mock_embed_batch,
        mock_embed,
        mock_validate_model,
        isolated_env,
    ):
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
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_vision_fallback_called_for_empty_pdf(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """When PDF extraction is empty and vision_model is set, fall back to vision."""
        cfg.vision_model = "test-vision"
        cfg.vision_timeout = 45.0
        # Called twice: initial extraction + Tesseract OCR (both empty)
        empty = _make_empty_result()
        mock_kf.side_effect = [empty, empty]

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        vision_pages = [(1, "Vision extracted text. " * 10)]
        with mock.patch(
            "lilbee.ingest.extract_pdf_vision", return_value=vision_pages
        ) as mock_vision:
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "scanned.pdf", "pdf", quiet=True)
        mock_vision.assert_called_once_with(
            f, "test-vision", quiet=True, timeout=45.0, on_progress=mock.ANY
        )
        assert len(result) > 0
        assert result[0]["content_type"] == "pdf"
        assert result[0]["page_start"] == 1

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_vision_fallback_quiet_false_by_default(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """Without quiet=True, vision fallback passes quiet=False."""
        cfg.vision_model = "test-vision"
        cfg.vision_timeout = 120.0
        empty = _make_empty_result()
        mock_kf.side_effect = [empty, empty]

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        vision_pages = [(1, "Vision extracted text. " * 10)]
        with mock.patch(
            "lilbee.ingest.extract_pdf_vision", return_value=vision_pages
        ) as mock_vision:
            from lilbee.ingest import ingest_document

            await ingest_document(f, "scanned.pdf", "pdf")
        mock_vision.assert_called_once_with(
            f, "test-vision", quiet=False, timeout=120.0, on_progress=mock.ANY
        )

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_ingest_file_threads_quiet_to_vision(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """quiet=True flows from _ingest_file through ingest_document to vision."""
        cfg.vision_model = "test-vision"
        cfg.vision_timeout = 120.0
        empty = _make_empty_result()
        mock_kf.return_value = empty

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        vision_pages = [(1, "Vision extracted text. " * 10)]
        with mock.patch(
            "lilbee.ingest.extract_pdf_vision", return_value=vision_pages
        ) as mock_vision:
            from lilbee.ingest import _ingest_file

            await _ingest_file(f, "scanned.pdf", "pdf", quiet=True)
        mock_vision.assert_called_once_with(
            f, "test-vision", quiet=True, timeout=120.0, on_progress=mock.ANY
        )

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_vision_fallback_not_called_without_model(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """When vision_model is empty, no fallback occurs (Tesseract tried first)."""
        mock_kf.return_value = _make_empty_result()
        cfg.vision_model = ""
        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        with mock.patch("lilbee.ingest.extract_pdf_vision") as mock_vision:
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "scanned.pdf", "pdf")
        mock_vision.assert_not_called()
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_vision_fallback_not_called_for_non_pdf(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """Vision fallback only triggers for PDF content type."""
        mock_kf.return_value = _make_empty_result()
        cfg.vision_model = "test-vision"
        f = isolated_env / "doc.txt"
        f.write_text("")

        with mock.patch("lilbee.ingest.extract_pdf_vision") as mock_vision:
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "doc.txt", "text")
        mock_vision.assert_not_called()
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_vision_fallback_empty_vision_text_returns_empty(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """When vision also returns empty text, return empty list."""
        cfg.vision_model = "test-vision"
        empty = _make_empty_result()
        mock_kf.side_effect = [empty, empty]

        f = isolated_env / "blank.pdf"
        f.write_bytes(b"fake pdf")

        with mock.patch("lilbee.ingest.extract_pdf_vision", return_value=[]):
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "blank.pdf", "pdf")
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_no_vision_fallback_when_text_meaningful(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """When kreuzberg produces meaningful text, no vision fallback."""
        mock_kf.return_value = _make_kreuzberg_result(
            text="Meaningful PDF content. " * 20, num_chunks=1, has_pages=True
        )
        cfg.vision_model = "test-vision"
        f = isolated_env / "good.pdf"
        f.write_bytes(b"fake pdf")

        with mock.patch("lilbee.ingest.extract_pdf_vision") as mock_vision:
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "good.pdf", "pdf")
        mock_vision.assert_not_called()
        assert len(result) > 0

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_vision_fallback_no_chunks_returns_empty(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """When vision text produces no chunks, return empty list."""
        cfg.vision_model = "test-vision"
        empty = _make_empty_result()
        mock_kf.side_effect = [empty, empty]

        f = isolated_env / "nochunks.pdf"
        f.write_bytes(b"fake pdf")

        with (
            mock.patch("lilbee.ingest.extract_pdf_vision", return_value=[(1, "Some text")]),
            mock.patch("lilbee.ingest.chunk_text", return_value=[]),
        ):
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "nochunks.pdf", "pdf")
        assert result == []


class TestTesseractOcrMiddleTier:
    """Tests for the Tesseract OCR tier between text extraction and vision fallback."""

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_tesseract_ocr_succeeds_skips_vision(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """When Tesseract OCR produces meaningful text, vision is not called."""
        cfg.vision_model = ""
        empty = _make_empty_result()
        ocr_result = _make_kreuzberg_result(
            text="Tesseract extracted text. " * 20, num_chunks=1, has_pages=True
        )
        mock_kf.side_effect = [empty, ocr_result]

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        with mock.patch("lilbee.ingest.extract_pdf_vision") as mock_vision:
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "scanned.pdf", "pdf")
        mock_vision.assert_not_called()
        assert len(result) > 0
        assert result[0]["content_type"] == "pdf"

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_tesseract_ocr_fails_falls_through_to_vision(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """When Tesseract OCR also yields < 50 chars, fall through to vision."""
        cfg.vision_model = "test-vision"
        cfg.vision_timeout = 120.0
        empty = _make_empty_result()
        # Called twice: initial extraction + Tesseract OCR (both empty)
        mock_kf.side_effect = [empty, empty]

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        vision_pages = [(1, "Vision extracted text. " * 10)]
        with mock.patch(
            "lilbee.ingest.extract_pdf_vision", return_value=vision_pages
        ) as mock_vision:
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "scanned.pdf", "pdf")
        mock_vision.assert_called_once()
        assert len(result) > 0

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_tesseract_exception_falls_through(self, mock_kf, mock_embed_batch, isolated_env):
        """When Tesseract is not installed (raises exception), fall through gracefully."""
        cfg.vision_model = ""
        empty = _make_empty_result()
        mock_kf.side_effect = [empty, RuntimeError("tesseract not found")]

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.ingest import ingest_document

        result = await ingest_document(f, "scanned.pdf", "pdf")
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_non_pdf_skips_tesseract_ocr(self, mock_kf, mock_embed_batch, isolated_env):
        """Non-PDF files never attempt Tesseract OCR retry."""
        mock_kf.return_value = _make_empty_result()
        cfg.vision_model = ""

        f = isolated_env / "doc.txt"
        f.write_text("")

        from lilbee.ingest import ingest_document

        await ingest_document(f, "doc.txt", "text")
        # Only one call to extract_file (the initial extraction, no OCR retry)
        assert mock_kf.call_count == 1

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_vision_explicit_skips_tesseract(self, mock_kf, mock_embed_batch, isolated_env):
        """When force_vision=True, Tesseract OCR tier is skipped entirely."""
        cfg.vision_model = "test-vision"
        cfg.vision_timeout = 120.0
        empty = _make_empty_result()
        mock_kf.return_value = empty

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        vision_pages = [(1, "Vision extracted text. " * 10)]
        with mock.patch("lilbee.ingest.extract_pdf_vision", return_value=vision_pages):
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "scanned.pdf", "pdf", force_vision=True)
        # extract_file called only once (initial extraction), not twice (no OCR retry)
        assert mock_kf.call_count == 1
        assert len(result) > 0

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_tesseract_ocr_empty_no_vision_warns(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """When Tesseract fails and no vision model, warning mentions Tesseract."""
        cfg.vision_model = ""
        empty = _make_empty_result()
        mock_kf.side_effect = [empty, empty]

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.ingest import ingest_document

        result = await ingest_document(f, "scanned.pdf", "pdf")
        assert result == []

    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch("kreuzberg.extract_file", new_callable=AsyncMock)
    async def test_configured_vision_model_skips_tesseract(
        self, mock_kf, mock_embed_batch, isolated_env
    ):
        """With vision_model set, Tesseract is skipped — vision takes precedence."""
        cfg.vision_model = "test-vision"
        cfg.vision_timeout = 120.0
        empty = _make_empty_result()
        mock_kf.return_value = empty

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        vision_pages = [(1, "Vision extracted text. " * 10)]
        with mock.patch("lilbee.ingest.extract_pdf_vision", return_value=vision_pages):
            from lilbee.ingest import ingest_document

            result = await ingest_document(f, "scanned.pdf", "pdf")
        # extract_file called only once (initial extraction), Tesseract skipped
        assert mock_kf.call_count == 1
        assert len(result) > 0


class TestSharedProgress:
    @mock.patch("lilbee.embedder.validate_model")
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch(
        "kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_contextvar_set_during_progress(
        self, mock_extract_file, mock_embed_batch, mock_validate_model, isolated_env
    ):
        """shared_progress contextvar is set inside _collect_results_with_progress."""
        from lilbee.progress import shared_progress

        (isolated_env / "a.txt").write_text("Content for shared progress test.")

        captured: list[tuple] = []

        # Use on_progress callback to capture the contextvar value during ingestion
        def capture_progress(event_type, data):
            val = shared_progress.get(None)
            if val is not None and val not in captured:
                captured.append(val)

        from lilbee.ingest import sync

        await sync(quiet=False, on_progress=capture_progress)
        assert len(captured) > 0, "shared_progress was never set during progress bar"
        progress_obj, task_id = captured[0]
        assert progress_obj is not None
        assert task_id is not None

    @mock.patch("lilbee.embedder.validate_model")
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch(
        "kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_contextvar_not_set_in_quiet_mode(
        self, mock_extract_file, mock_embed_batch, mock_validate_model, isolated_env
    ):
        """shared_progress contextvar is NOT set in quiet mode (no progress bar)."""
        from lilbee.progress import shared_progress

        (isolated_env / "b.txt").write_text("Content for quiet mode test.")

        captured: list[object] = []

        def capture_progress(event_type, data):
            val = shared_progress.get(None)
            if val is not None:
                captured.append(val)

        from lilbee.ingest import sync

        await sync(quiet=True, on_progress=capture_progress)
        assert len(captured) == 0, "shared_progress should not be set in quiet mode"

    @mock.patch("lilbee.embedder.validate_model")
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch(
        "kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_contextvar_reset_after_progress(
        self, mock_extract_file, mock_embed_batch, mock_validate_model, isolated_env
    ):
        """shared_progress is reset to None after _collect_results_with_progress completes."""
        from lilbee.progress import shared_progress

        (isolated_env / "c.txt").write_text("Content for reset test.")
        from lilbee.ingest import sync

        await sync(quiet=False)
        assert shared_progress.get(None) is None


class TestOcrExtractionConfig:
    def test_ocr_config_has_tesseract_backend(self):
        from lilbee.ingest import ocr_extraction_config

        config = ocr_extraction_config()
        assert config.ocr is not None
        assert config.ocr.backend == "tesseract"

    def test_ocr_config_has_page_config(self):
        from lilbee.ingest import ocr_extraction_config

        config = ocr_extraction_config()
        assert config.pages is not None

    def test_ocr_config_has_chunking(self):
        from lilbee.ingest import ocr_extraction_config

        config = ocr_extraction_config()
        assert config.chunking is not None


class TestIngestMarkdownEdgeCases:
    async def test_empty_markdown_returns_empty(self, isolated_env):
        from lilbee.ingest import ingest_markdown

        md = isolated_env / "empty.md"
        md.write_text("   ")
        result = await ingest_markdown(md, "empty.md")
        assert result == []

    async def test_no_chunks_returns_empty(self, isolated_env):
        from lilbee.ingest import ingest_markdown

        md = isolated_env / "blank.md"
        md.write_text("some text")
        with mock.patch("lilbee.ingest.chunk_text", return_value=[]):
            result = await ingest_markdown(md, "blank.md")
        assert result == []

    async def test_frontmatter_only_returns_empty(self, isolated_env):
        from lilbee.ingest import ingest_markdown

        md = isolated_env / "fm_only.md"
        md.write_text("---\ntitle: Just Frontmatter\ntags: [test]\n---\n")
        result = await ingest_markdown(md, "fm_only.md")
        assert result == []


class TestIngestDocumentEdgeCases:
    async def test_empty_extraction_returns_empty(self, isolated_env):
        """Structured formats now go through kreuzberg — empty result yields no chunks."""
        from lilbee.ingest import ingest_document

        empty_result = mock.MagicMock(chunks=[])
        mock_extract = AsyncMock(return_value=empty_result)
        with mock.patch("kreuzberg.extract_file", mock_extract):
            result = await ingest_document(isolated_env / "e.xml", "e.xml", "xml")
        assert result == []

    async def test_no_chunks_returns_empty(self, isolated_env):
        from lilbee.ingest import ingest_document

        no_chunks_result = mock.MagicMock(chunks=[])
        mock_extract = AsyncMock(return_value=no_chunks_result)
        with mock.patch("kreuzberg.extract_file", mock_extract):
            result = await ingest_document(isolated_env / "s.xml", "s.xml", "xml")
        assert result == []


class TestChunkViaKreuzberg:
    def test_empty_returns_empty(self):
        from lilbee.chunk import chunk_text

        assert chunk_text("") == []

    def test_returns_chunks(self):
        from lilbee.chunk import chunk_text

        result = chunk_text("Some text that should be chunked.")
        assert len(result) >= 1


class TestConceptIndexing:
    @mock.patch("lilbee.embedder.validate_model")
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch(
        "kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_concept_extraction_called_during_ingest(
        self, mock_extract_file, mock_embed_batch, mock_validate_model, isolated_env
    ):
        """When concept_graph is enabled, extraction is called after ingest."""
        cfg.concept_graph = True
        (isolated_env / "concept_test1.txt").write_text("Content for concepts test one.")

        with (
            mock.patch("lilbee.concepts.extract_concepts_batch", return_value=[["test"]]) as m_ext,
            mock.patch("lilbee.concepts.build_from_chunks"),
            mock.patch("lilbee.concepts.get_graph", return_value=True),
            mock.patch("lilbee.concepts.rebuild_clusters"),
        ):
            from lilbee.ingest import sync

            await sync(quiet=True)
        m_ext.assert_called()

    @mock.patch("lilbee.embedder.validate_model")
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch(
        "kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_concept_disabled_skips_extraction(
        self, mock_extract_file, mock_embed_batch, mock_validate_model, isolated_env
    ):
        """When concept_graph is disabled, extraction is not called."""
        cfg.concept_graph = False
        (isolated_env / "concept_test2.txt").write_text("Some test content.")

        with mock.patch("lilbee.concepts.extract_concepts_batch") as m_ext:
            from lilbee.ingest import sync

            await sync(quiet=True)
        m_ext.assert_not_called()

    @mock.patch("lilbee.embedder.validate_model")
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch(
        "kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_concept_failure_does_not_break_ingest(
        self, mock_extract_file, mock_embed_batch, mock_validate_model, isolated_env
    ):
        """When concept extraction raises, ingest still succeeds."""
        cfg.concept_graph = True
        (isolated_env / "concept_test2.txt").write_text("Some test content.")

        with (
            mock.patch(
                "lilbee.concepts.extract_concepts_batch", side_effect=RuntimeError("spacy broke")
            ),
            mock.patch("lilbee.concepts.get_graph", return_value=True),
            mock.patch("lilbee.concepts.rebuild_clusters"),
        ):
            from lilbee.ingest import sync

            result = await sync(quiet=True)
        assert "concept_test2.txt" in result.added

    @mock.patch("lilbee.embedder.validate_model")
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch(
        "kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_cluster_rebuild_called_after_sync(
        self, mock_extract_file, mock_embed_batch, mock_validate_model, isolated_env
    ):
        """After sync completes, rebuild_clusters is called."""
        cfg.concept_graph = True
        (isolated_env / "concept_test4.txt").write_text("Some test content.")

        with (
            mock.patch("lilbee.concepts.extract_concepts_batch", return_value=[["test"]]),
            mock.patch("lilbee.concepts.build_from_chunks"),
            mock.patch("lilbee.concepts.get_graph", return_value=True),
            mock.patch("lilbee.concepts.rebuild_clusters") as mock_rebuild,
        ):
            from lilbee.ingest import sync

            await sync(quiet=True)
        mock_rebuild.assert_called()

    @mock.patch("lilbee.embedder.validate_model")
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch(
        "kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_cluster_rebuild_failure_does_not_break_sync(
        self, mock_extract_file, mock_embed_batch, mock_validate_model, isolated_env
    ):
        """When cluster rebuild raises, sync still succeeds."""
        cfg.concept_graph = True
        (isolated_env / "rebuild_test.txt").write_text("Content for rebuild test.")

        with (
            mock.patch("lilbee.concepts.extract_concepts_batch", return_value=[["test"]]),
            mock.patch("lilbee.concepts.build_from_chunks"),
            mock.patch("lilbee.concepts.get_graph", return_value=True),
            mock.patch(
                "lilbee.concepts.rebuild_clusters",
                side_effect=RuntimeError("leiden broke"),
            ),
        ):
            from lilbee.ingest import sync

            result = await sync(quiet=True)
        assert "rebuild_test.txt" in result.added

    @mock.patch("lilbee.embedder.validate_model")
    @mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
    @mock.patch(
        "kreuzberg.extract_file", new_callable=AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_graph_none_skips_indexing(
        self, mock_extract_file, mock_embed_batch, mock_validate_model, isolated_env
    ):
        """When get_graph() returns None, concept indexing is skipped gracefully."""
        cfg.concept_graph = True
        (isolated_env / "graph_none_test.txt").write_text("Content for graph none test.")

        with (
            mock.patch("lilbee.concepts.extract_concepts_batch", return_value=[["test"]]),
            mock.patch("lilbee.concepts.build_from_chunks"),
            mock.patch("lilbee.concepts.get_graph", return_value=False),
        ):
            from lilbee.ingest import sync

            result = await sync(quiet=True)
        assert "graph_none_test.txt" in result.added
