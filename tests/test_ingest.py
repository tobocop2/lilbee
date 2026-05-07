"""Tests for the document sync engine (mocked: no live server needed)."""

import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest

import lilbee.core.services as svc_mod
from lilbee.core.config import cfg


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
    import lilbee.data.store as store_mod

    store_mod._store = None

    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def mock_svc():
    """Provide a mock Services container for all ingest tests."""
    from tests.conftest import make_mock_services

    _sources: dict[str, dict] = {}
    store = MagicMock()
    store.search.return_value = []
    store.bm25_probe.return_value = []
    store.get_sources.side_effect = lambda: list(_sources.values())
    store.add_chunks.side_effect = lambda records: len(records)

    def _upsert(fn, fh, cc, source_type="document"):
        from datetime import UTC, datetime

        _sources[fn] = {
            "filename": fn,
            "file_hash": fh,
            "chunk_count": cc,
            "ingested_at": datetime.now(UTC).isoformat(),
            "source_type": source_type,
        }

    store.upsert_source.side_effect = _upsert
    store.delete_source.side_effect = lambda fn: _sources.pop(fn, None)
    store.delete_by_source.return_value = None
    store.drop_all.side_effect = lambda: _sources.clear()
    store.ensure_fts_index.return_value = None
    embedder = MagicMock()
    embedder.embed.side_effect = lambda text, **kw: [0.1] * 768
    embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]
    searcher = MagicMock()
    services = make_mock_services(store=store, embedder=embedder, searcher=searcher)
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


def _make_kreuzberg_result(
    text="Some extracted text. " * 20,
    num_chunks=1,
    has_pages=False,
    document=None,
):
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
    result.document = document
    return result


def _make_empty_result():
    """Build a mock kreuzberg ExtractionResult with no chunks."""
    result = mock.MagicMock()
    result.chunks = []
    result.content = ""
    result.document = None
    return result


@mock.patch("kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result())
class TestSync:
    async def test_empty_documents_dir(self, mock_extract_file, isolated_env):
        from lilbee.data.ingest import SyncResult, sync

        result = await sync()
        assert result == SyncResult()

    async def test_ingest_text_file(self, mock_extract_file, isolated_env):
        (isolated_env / "test.txt").write_text("Hello world. This is a test document.")
        from lilbee.data.ingest import sync

        result = await sync()
        assert "test.txt" in result.added
        mock_extract_file.assert_called()
        assert any("test.txt" in str(call) for call in mock_extract_file.call_args_list)

    async def test_quiet_mode_suppresses_progress(self, mock_extract_file, isolated_env):
        (isolated_env / "quiet.txt").write_text("Quiet mode test content.")
        from lilbee.data.ingest import sync

        result = await sync(quiet=True)
        assert "quiet.txt" in result.added

    async def test_on_progress_callback_quiet(self, mock_extract_file, isolated_env):
        (isolated_env / "cb.txt").write_text("Callback test.")
        from lilbee.data.ingest import sync

        events: list[tuple] = []
        result = await sync(quiet=True, on_progress=lambda t, d: events.append((t, d)))
        assert "cb.txt" in result.added
        event_types = [t for t, _ in events]
        assert "file_start" in event_types
        assert "file_done" in event_types
        assert "done" in event_types
        # ExtractEvent fires once per file before the embed phase so subscribers
        # can show "extracted N pages" before the bar starts to tick at chunk
        # granularity. Without this a 44MB PDF sat silently for many minutes.
        assert "extract" in event_types
        extract = next(d for t, d in events if t == "extract")
        assert extract.file == "cb.txt"
        assert extract.total_pages >= 1
        file_done = next(d for t, d in events if t == "file_done")
        assert file_done.file == "cb.txt"
        assert file_done.status == "ok"

    async def test_on_progress_callback_with_progress_bar(self, mock_extract_file, isolated_env):
        (isolated_env / "cb2.txt").write_text("Callback with progress bar.")
        from lilbee.data.ingest import sync

        events: list[tuple] = []
        result = await sync(quiet=False, on_progress=lambda t, d: events.append((t, d)))
        assert "cb2.txt" in result.added
        event_types = [t for t, _ in events]
        assert "file_done" in event_types
        file_done = next(d for t, d in events if t == "file_done")
        assert file_done.status == "ok"

    async def test_ingest_markdown_file(self, mock_extract_file, isolated_env):
        (isolated_env / "readme.md").write_text("# Title\n\nSome markdown content.")
        from lilbee.data.ingest import sync

        assert "readme.md" in (await sync()).added

    async def test_ingest_html_file(self, mock_extract_file, isolated_env):
        (isolated_env / "page.html").write_text("<p>Content</p>")
        from lilbee.data.ingest import sync

        assert "page.html" in (await sync()).added

    async def test_ingest_rst_file(self, mock_extract_file, isolated_env):
        (isolated_env / "doc.rst").write_text("Title\n=====\n\nContent.")
        from lilbee.data.ingest import sync

        assert "doc.rst" in (await sync()).added

    async def test_modified_file_reingested(self, mock_extract_file, isolated_env):
        f = isolated_env / "changing.txt"
        f.write_text("Version 1")
        from lilbee.data.ingest import sync

        await sync()
        f.write_text("Version 2, different content now")
        assert "changing.txt" in (await sync()).updated

    async def test_deleted_file_removed(self, mock_extract_file, isolated_env):
        f = isolated_env / "temp.txt"
        f.write_text("Temporary")
        from lilbee.data.ingest import sync

        await sync()
        f.unlink()
        assert "temp.txt" in (await sync()).removed

    async def test_unchanged_file_skipped(self, mock_extract_file, isolated_env):
        (isolated_env / "stable.txt").write_text("I stay the same")
        from lilbee.data.ingest import sync

        await sync()
        result = await sync()
        assert result.unchanged == 1
        assert result.added == []

    async def test_unsupported_extension_skipped(self, mock_extract_file, isolated_env):
        (isolated_env / "data.zip").write_bytes(b"binary data")
        from lilbee.data.ingest import sync

        assert (await sync()).added == []

    async def test_hidden_files_skipped(self, mock_extract_file, isolated_env):
        (isolated_env / ".hidden").write_text("secret")
        from lilbee.data.ingest import sync

        assert (await sync()).added == []

    async def test_subdirectory_files_ingested(self, mock_extract_file, isolated_env):
        sub = isolated_env / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested content")
        from lilbee.data.ingest import sync

        assert any("nested.txt" in f for f in (await sync()).added)

    async def test_code_file_ingested(self, mock_extract_file, isolated_env):
        (isolated_env / "example.py").write_text("def hello():\n    print('hi')\n")
        from lilbee.data.ingest import sync

        assert "example.py" in (await sync()).added

    async def test_force_rebuild_clears_and_reingests(self, mock_extract_file, isolated_env):
        (isolated_env / "keep.txt").write_text("I survive rebuilds")
        from lilbee.data.ingest import sync

        await sync()
        result = await sync(force_rebuild=True)
        assert "keep.txt" in result.added

    async def test_ingest_pdf(self, mock_extract_file, isolated_env):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = isolated_env / "test.pdf"
        c = canvas.Canvas(str(pdf), pagesize=letter)
        c.drawString(72, 700, "Oil capacity is 5 quarts.")
        c.showPage()
        c.save()

        from lilbee.data.ingest import sync

        assert "test.pdf" in (await sync()).added

    async def test_nonexistent_documents_dir(
        self,
        mock_extract_file,
        isolated_env,
        tmp_path,
    ):
        nonexistent = tmp_path / "nonexistent"
        cfg.documents_dir = nonexistent
        from lilbee.data.ingest import SyncResult, sync

        result = await sync()
        assert result == SyncResult()
        assert nonexistent.exists()  # Directory was auto-created

    async def test_ingest_error_logged_not_raised(self, mock_extract_file, isolated_env):
        """A file that fails ingestion is logged but doesn't crash sync."""
        from unittest.mock import patch

        (isolated_env / "good.txt").write_text("This is fine.")
        (isolated_env / "bad.txt").write_text("This will fail.")

        from lilbee.data.ingest import sync
        from lilbee.data.ingest.pipeline import _ingest_file as orig_ingest

        async def _failing_ingest(path, name, content_type, **kwargs):
            if "bad" in name:
                raise RuntimeError("simulated failure")
            return await orig_ingest(path, name, content_type, **kwargs)

        with patch("lilbee.data.ingest.pipeline._ingest_file", side_effect=_failing_ingest):
            result = await sync()
        # good.txt was added, bad.txt failed
        assert "good.txt" in result.added
        assert "bad.txt" not in result.added
        assert "bad.txt" in result.failed

    async def test_ingest_error_on_update_tracked_as_failed(self, mock_extract_file, isolated_env):
        """A file that fails re-ingestion on update goes to failed, not updated."""
        from unittest.mock import patch

        f = isolated_env / "flaky.txt"
        f.write_text("Version 1")

        from lilbee.data.ingest import sync

        await sync()  # First ingest succeeds

        f.write_text("Version 2, will fail")

        from lilbee.data.ingest.pipeline import _ingest_file as orig_ingest

        async def _failing_ingest(path, name, content_type, **kwargs):
            if "flaky" in name:
                raise RuntimeError("simulated failure on update")
            return await orig_ingest(path, name, content_type, **kwargs)

        with patch("lilbee.data.ingest.pipeline._ingest_file", side_effect=_failing_ingest):
            result = await sync()
        assert "flaky.txt" not in result.updated
        assert "flaky.txt" in result.failed

    async def test_ingest_error_in_quiet_mode(self, mock_extract_file, isolated_env):
        """Quiet-mode error handling works the same as non-quiet."""
        from unittest.mock import patch

        (isolated_env / "bad.txt").write_text("Will fail in quiet mode.")
        from lilbee.data.ingest import sync

        async def _fail(*args):
            raise RuntimeError("boom")

        with patch("lilbee.data.ingest.pipeline._ingest_file", side_effect=_fail):
            result = await sync(quiet=True)
        assert "bad.txt" in result.failed
        assert "bad.txt" not in result.added

    async def test_ingest_error_on_update_quiet_mode(self, mock_extract_file, isolated_env):
        """Quiet-mode update failure tracks in failed list."""
        from unittest.mock import patch

        f = isolated_env / "qflaky.txt"
        f.write_text("Version 1")
        from lilbee.data.ingest import sync

        await sync()  # First ingest succeeds
        f.write_text("Version 2, fail quietly")

        from lilbee.data.ingest.pipeline import _ingest_file as orig

        async def _fail(path, name, ct, **kwargs):
            if "qflaky" in name:
                raise RuntimeError("quiet fail")
            return await orig(path, name, ct, **kwargs)

        with patch("lilbee.data.ingest.pipeline._ingest_file", side_effect=_fail):
            result = await sync(quiet=True)
        assert "qflaky.txt" in result.failed
        assert "qflaky.txt" not in result.updated


@mock.patch("kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result())
class TestSyncCancellation:
    """Tests for cancel support and atomic per-file delete in sync."""

    async def test_cancel_stops_file_discovery(self, mock_extract_file, isolated_env):
        """Setting cancel before sync starts prevents any file processing."""
        import threading

        (isolated_env / "a.txt").write_text("file a")
        (isolated_env / "b.txt").write_text("file b")
        from lilbee.data.ingest import sync

        cancel = threading.Event()
        cancel.set()
        result = await sync(quiet=True, cancel=cancel)
        # Cancel was set before the loop, so no files should be processed
        assert result.added == []
        assert result.unchanged == 0

    async def test_cancel_during_ingest_batch(self, mock_extract_file, isolated_env, mock_svc):
        """Cancel set mid-batch raises CancelledError for pending files."""
        import asyncio
        import threading

        from lilbee.data.ingest import ingest_batch

        (isolated_env / "a.txt").write_text("file a")
        (isolated_env / "b.txt").write_text("file b")

        cancel = threading.Event()
        cancel.set()

        added = ["a.txt", "b.txt"]
        files = [
            ("a.txt", isolated_env / "a.txt", "text", "hash_a", False),
            ("b.txt", isolated_env / "b.txt", "text", "hash_b", False),
        ]
        with pytest.raises(asyncio.CancelledError):
            await ingest_batch(files, added, [], [], [], quiet=True, cancel=cancel)

    async def test_atomic_delete_for_modified_file(self, mock_extract_file, isolated_env, mock_svc):
        """Modified file: old chunks deleted in _process_one, not in sync() loop."""
        from lilbee.data.ingest import sync

        f = isolated_env / "doc.txt"
        f.write_text("Version 1")
        await sync(quiet=True)

        # Reset call tracking
        mock_svc.store.delete_by_source.reset_mock()

        f.write_text("Version 2, modified content")
        await sync(quiet=True)

        # delete_by_source should be called (from _process_one), not from the sync loop
        mock_svc.store.delete_by_source.assert_called_with("doc.txt")

    async def test_cancel_preserves_old_chunks_for_modified_file(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """If cancel is set before a modified file is processed, old chunks remain."""
        import threading

        from lilbee.data.ingest import sync

        f = isolated_env / "doc.txt"
        f.write_text("Version 1")
        await sync(quiet=True)

        mock_svc.store.delete_by_source.reset_mock()

        f.write_text("Version 2, modified content")
        cancel = threading.Event()
        cancel.set()
        await sync(quiet=True, cancel=cancel)

        # Old chunks should NOT have been deleted since cancel was set before processing
        mock_svc.store.delete_by_source.assert_not_called()


class TestIngestHelpers:
    """Cover edge cases in ingest_document and ingest_code_sync."""

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_empty_result())
    async def testingest_document_empty_chunks(self, mock_extract_file, isolated_env):
        """Document that produces no chunks returns empty list."""
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "empty.txt"
        f.write_text("   ")
        result = await ingest_document(f, "empty.txt", "text")
        assert result == []

    async def test_ingest_code_empty_chunks(self, isolated_env):
        """Code file that produces no chunks returns empty list."""
        from unittest.mock import patch

        from lilbee.data.ingest import ingest_code_sync

        f = isolated_env / "empty.py"
        f.write_text("")
        with patch("lilbee.data.code_chunker.chunk_code", return_value=[]):
            result = ingest_code_sync(f, "empty.py")
            assert result == []

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def testingest_document_pdf_with_pages(self, mock_kf, isolated_env):
        """PDF document returns records with page metadata."""
        mock_kf.return_value = _make_kreuzberg_result(
            text="Page 1 content. " * 10 + "Page 2 content. " * 10,
            num_chunks=2,
            has_pages=True,
        )
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "test.pdf"
        f.write_bytes(b"fake")
        result = await ingest_document(f, "test.pdf", "pdf")
        assert len(result) == 2
        assert result[0]["page_start"] == 1
        assert result[1]["page_start"] == 2


class TestCancellation:
    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_cancelled_error_propagates(self, mock_extract_file, isolated_env):
        """CancelledError in _process_one is re-raised, not swallowed."""
        import asyncio

        async def _cancel(*args, **kwargs):
            raise asyncio.CancelledError()

        with mock.patch("lilbee.data.ingest.pipeline._ingest_file", side_effect=_cancel):
            from lilbee.data.ingest import ingest_batch

            added = ["cancel.txt"]
            with pytest.raises(asyncio.CancelledError):
                await ingest_batch(
                    [("cancel.txt", isolated_env / "cancel.txt", "text", "abc123", False)],
                    added,
                    [],
                    [],
                    [],
                    quiet=True,
                )


class TestDiscoverFiles:
    def test_nonexistent_dir_returns_empty(self, isolated_env, tmp_path):
        cfg.documents_dir = tmp_path / "does_not_exist"
        from lilbee.data.ingest import discover_files

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
        from lilbee.data.ingest import discover_files

        d = isolated_env / ignored_dir
        d.mkdir()
        (d / child_file).write_text("content")
        (isolated_env / "visible.txt").write_text("visible")

        found = discover_files()
        assert "visible.txt" in found
        assert not any(ignored_dir in name for name in found)

    def test_skips_custom_ignore_via_env(self, isolated_env):
        from lilbee.data.ingest import discover_files

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
        from lilbee.data.ingest import classify_file

        assert classify_file(Path(filename)) == expected


class TestFileHash:
    def test_deterministic(self, tmp_path):
        from lilbee.data.ingest import file_hash

        f = tmp_path / "test.txt"
        f.write_text("hello")
        h1 = file_hash(f)
        h2 = file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        from lilbee.data.ingest import file_hash

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert file_hash(f1) != file_hash(f2)


class TestApplyResultZeroChunks:
    def test_zero_chunks_not_recorded_as_added(self):
        from lilbee.data.ingest.pipeline import _apply_result
        from lilbee.data.ingest.types import _IngestResult

        added = ["scanned.pdf"]
        updated: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        result = _IngestResult("scanned.pdf", Path("scanned.pdf"), chunk_count=0, error=None)
        _apply_result(result, added, updated, failed, skipped)
        assert "scanned.pdf" not in added
        assert "scanned.pdf" not in failed
        assert "scanned.pdf" in skipped

    def test_zero_chunks_not_recorded_as_updated(self):
        from lilbee.data.ingest.pipeline import _apply_result
        from lilbee.data.ingest.types import _IngestResult

        added: list[str] = []
        updated = ["scanned.pdf"]
        failed: list[str] = []
        skipped: list[str] = []
        result = _IngestResult("scanned.pdf", Path("scanned.pdf"), chunk_count=0, error=None)
        _apply_result(result, added, updated, failed, skipped)
        assert "scanned.pdf" not in updated
        assert "scanned.pdf" not in failed
        assert "scanned.pdf" in skipped

    def test_nonzero_chunks_recorded(self, mock_svc):
        from lilbee.data.ingest.pipeline import _apply_result
        from lilbee.data.ingest.types import _IngestResult

        added = ["doc.pdf"]
        updated: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        result = _IngestResult("doc.pdf", Path("doc.pdf"), chunk_count=5, error=None)
        with mock.patch("lilbee.data.ingest.pipeline.file_hash", return_value="abc123"):
            _apply_result(result, added, updated, failed, skipped)
        mock_svc.store.upsert_source.assert_called_once()
        assert "doc.pdf" in added
        assert "doc.pdf" not in skipped

    def test_ingest_error_logs_warning_not_exception(self, caplog):
        """ingest errors must not log at exception level.

        The logger's exception() call routes the full traceback through the
        stderr bridge into the TUI chat pane, even though the functional path
        already surfaces the failure via SyncResult.failed. Dropping to
        warning() keeps the noise out of chat while leaving the error
        reachable at DEBUG level.
        """
        import logging

        from lilbee.data.ingest.pipeline import _apply_result
        from lilbee.data.ingest.types import _IngestResult

        added: list[str] = []
        updated: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        err = RuntimeError("embedder bogus:bogus not installed")
        result = _IngestResult("qa-fail.md", Path("qa-fail.md"), chunk_count=0, error=err)
        with caplog.at_level(logging.DEBUG, logger="lilbee.data.ingest.pipeline"):
            _apply_result(result, added, updated, failed, skipped)
        levels = [r.levelno for r in caplog.records if r.name == "lilbee.data.ingest.pipeline"]
        assert logging.WARNING in levels
        # Exception-level records carry exc_info; warning call should not.
        warning_records = [
            r
            for r in caplog.records
            if r.name == "lilbee.data.ingest.pipeline" and r.levelno == logging.WARNING
        ]
        assert warning_records
        assert warning_records[0].exc_info is None
        assert "qa-fail.md" in failed


class TestSyncResultStr:
    def test_str_no_failures(self):
        from lilbee.data.ingest import SyncResult

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
        from lilbee.data.ingest import SyncResult

        result = SyncResult(failed=["x.txt", "y.txt"])
        text = str(result)
        assert "Failed: 2" in text
        assert "[red]x.txt[/red]" in text
        assert "[red]y.txt[/red]" in text

    def test_str_with_skipped(self):
        from lilbee.data.ingest import SyncResult

        result = SyncResult(skipped=["scan.pdf", "scan2.pdf"])
        text = str(result)
        assert "Skipped: 2" in text
        assert "[yellow]scan.pdf[/yellow]" in text
        assert "[yellow]scan2.pdf[/yellow]" in text

    def test_repr_matches_str(self):
        from lilbee.data.ingest import SyncResult

        result = SyncResult(added=["a.txt"])
        assert "SyncResult" in repr(result)
        assert "added=1" in repr(result)


class TestCollectResultsSkipped:
    """Verify _collect_results emits BATCH_PROGRESS with status=skipped for 0-chunk files."""

    async def test_zero_chunks_emits_skipped_status(self):
        import asyncio

        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.ingest.types import _IngestResult
        from lilbee.runtime.progress import BatchProgressEvent, EventType

        captured: list[tuple[EventType, BatchProgressEvent]] = []

        def on_progress(event_type: object, data: object) -> None:
            if event_type == EventType.BATCH_PROGRESS and isinstance(data, BatchProgressEvent):
                captured.append((event_type, data))

        async def _zero_chunk_result() -> _IngestResult:
            return _IngestResult("scan.pdf", Path("scan.pdf"), chunk_count=0, error=None)

        task = asyncio.ensure_future(_zero_chunk_result())
        added: list[str] = []
        updated: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        await _collect_results([task], added, updated, failed, skipped, on_progress=on_progress)
        assert len(captured) == 1
        assert captured[0][1].status == "skipped"
        assert "scan.pdf" in skipped


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
        from lilbee.data.ingest import classify_file

        assert classify_file(Path(filename)) == expected


class TestDiscoverSymlinkEscape:
    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks require admin on Windows")
    def test_symlink_escaping_docs_dir_skipped(self, isolated_env):
        """Symlinks that resolve outside the documents directory are skipped."""
        import os

        from lilbee.data.ingest import discover_files

        outside_file = isolated_env.parent / "secret.txt"
        outside_file.write_text("secret data")
        link_path = isolated_env / "escaped.txt"
        os.symlink(outside_file, link_path)

        found = discover_files()
        assert "escaped.txt" not in found
        outside_file.unlink()

    def test_path_validation_failure_skipped(self, isolated_env):
        """Files that fail path validation are skipped (covers Windows too)."""
        from unittest.mock import patch

        from lilbee.data.ingest import discover_files

        (isolated_env / "normal.md").write_text("# Normal")

        original = __import__(
            "lilbee.data.ingest.discovery", fromlist=["validate_path_within"]
        ).validate_path_within

        def strict_validate(path, base):
            if "normal" in str(path):
                raise ValueError("blocked")
            return original(path, base)

        with patch(
            "lilbee.data.ingest.discovery.validate_path_within", side_effect=strict_validate
        ):
            found = discover_files()
        assert "normal.md" not in found


class TestDiscoverNewFormats:
    def test_new_extensions_discovered(self, isolated_env):
        from lilbee.data.ingest import discover_files

        for ext in [".docx", ".xlsx", ".pptx", ".epub", ".png", ".csv", ".tsv"]:
            (isolated_env / f"test{ext}").write_bytes(b"dummy")

        found = discover_files()
        for ext in [".docx", ".xlsx", ".pptx", ".epub", ".png", ".csv", ".tsv"]:
            assert f"test{ext}" in found


class TestExtractionConfig:
    def test_paginated_has_page_config(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.PAGINATED)
        assert config.pages is not None

    def test_paginated_no_markdown_output(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.PAGINATED)
        assert getattr(config, "output_format", None) != "markdown"

    def test_markdown_has_no_page_config(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.MARKDOWN)
        assert config.pages is None

    def test_markdown_has_chunking(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.MARKDOWN)
        assert config.chunking is not None

    def test_markdown_sets_output_format(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.MARKDOWN)
        assert config.output_format == "markdown"

    def test_paginated_ocr_has_tesseract(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.PAGINATED_OCR)
        assert config.pages is not None
        assert config.ocr is not None
        assert config.ocr.backend == "tesseract"

    @pytest.mark.parametrize(
        "content_type, expected_mode_name",
        [
            ("pdf", "PAGINATED"),
            ("text", "MARKDOWN"),
            ("docx", "MARKDOWN"),
            ("xlsx", "MARKDOWN"),
            ("pptx", "MARKDOWN"),
            ("epub", "MARKDOWN"),
            ("image", "MARKDOWN"),
            ("code", "MARKDOWN"),
        ],
    )
    def test_content_type_to_mode(self, content_type, expected_mode_name):
        from lilbee.data.ingest import ExtractMode, content_type_to_mode

        assert content_type_to_mode(content_type) is getattr(ExtractMode, expected_mode_name)

    def test_topic_threshold_propagates_to_every_mode(self, monkeypatch):
        """Every ExtractMode carries the semantic chunking config, not just PDF."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "semantic_chunking", True)
        monkeypatch.setattr(cfg, "topic_threshold", 0.42)
        for mode in ExtractMode:
            config = extraction_config(mode)
            assert config.chunking.chunker_type == "semantic"
            assert config.chunking.topic_threshold == pytest.approx(0.42, abs=1e-5)


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
        from lilbee.data.ingest import classify_file

        assert classify_file(Path(filename)) == expected


@mock.patch("kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result())
class TestSyncStructuredFormats:
    async def test_xml_file_ingested(
        self,
        mock_extract_file,
        isolated_env,
    ):
        (isolated_env / "data.xml").write_text("<root><item>value</item></root>")
        from lilbee.data.ingest import sync

        result = await sync()
        assert "data.xml" in result.added

    async def test_json_file_ingested(
        self,
        mock_extract_file,
        isolated_env,
    ):
        (isolated_env / "data.json").write_text('{"key": "value"}')
        from lilbee.data.ingest import sync

        result = await sync()
        assert "data.json" in result.added

    async def test_jsonl_file_ingested(
        self,
        mock_extract_file,
        isolated_env,
    ):
        (isolated_env / "data.jsonl").write_text('{"key": "value"}\n{"key2": "value2"}')
        from lilbee.data.ingest import sync

        result = await sync()
        assert "data.jsonl" in result.added

    async def test_csv_file_ingested(
        self,
        mock_extract_file,
        isolated_env,
    ):
        (isolated_env / "data.csv").write_text("name,age\nAlice,30\nBob,25")
        from lilbee.data.ingest import sync

        result = await sync()
        assert "data.csv" in result.added


class TestHasMeaningfulText:
    def test_empty_chunks_returns_false(self):
        from lilbee.data.ingest.extract import _has_meaningful_text

        result = mock.MagicMock(chunks=[])
        assert _has_meaningful_text(result) is False

    def test_short_text_returns_false(self):
        from lilbee.data.ingest.extract import _has_meaningful_text

        chunk = mock.MagicMock()
        chunk.content = "short"
        result = mock.MagicMock(chunks=[chunk])
        assert _has_meaningful_text(result) is False

    def test_meaningful_text_returns_true(self):
        from lilbee.data.ingest.extract import _has_meaningful_text

        chunk = mock.MagicMock()
        chunk.content = "A" * 100
        result = mock.MagicMock(chunks=[chunk])
        assert _has_meaningful_text(result) is True


def _vision_pages(*pairs):
    """Build PageText-like NamedTuples for stubbing provider.pdf_ocr."""
    from lilbee.vision import PageText

    return [PageText(page, text) for page, text in pairs]


class TestVisionFallback:
    """OCR fallback dispatches to ``provider.pdf_ocr`` (the persistent worker)."""

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_vision_fallback_called_for_empty_pdf(self, mock_kf, isolated_env, mock_svc):
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        cfg.ocr_timeout = 45.0
        cfg.enable_ocr = True
        mock_kf.return_value = _make_empty_result()
        mock_svc.provider.pdf_ocr.return_value = _vision_pages((1, "Vision extracted text. " * 10))

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        result = await ingest_document(f, "scanned.pdf", "pdf", quiet=True)
        mock_svc.provider.pdf_ocr.assert_called_once_with(
            f,
            backend="vision",
            model="org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf",
            per_page_timeout_s=45.0,
            quiet=True,
            on_progress=mock.ANY,
        )
        assert len(result) > 0
        assert result[0]["content_type"] == "pdf"
        assert result[0]["page_start"] == 1

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_vision_fallback_quiet_false_by_default(self, mock_kf, isolated_env, mock_svc):
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        cfg.ocr_timeout = 120.0
        cfg.enable_ocr = True
        mock_kf.return_value = _make_empty_result()
        mock_svc.provider.pdf_ocr.return_value = _vision_pages((1, "Vision extracted text. " * 10))

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        await ingest_document(f, "scanned.pdf", "pdf")
        mock_svc.provider.pdf_ocr.assert_called_once_with(
            f,
            backend="vision",
            model="org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf",
            per_page_timeout_s=120.0,
            quiet=False,
            on_progress=mock.ANY,
        )

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_vision_pool_not_called_when_ocr_disabled(self, mock_kf, isolated_env, mock_svc):
        """``cfg.enable_ocr=False`` skips the vision pool path entirely.

        Tesseract OCR runs inline (via the patched ``extract_file_sync``)
        rather than through ``provider.pdf_ocr``, so the assertion is
        that the pool was never reached.
        """
        mock_kf.return_value = _make_empty_result()
        cfg.enable_ocr = False
        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        result = await ingest_document(f, "scanned.pdf", "pdf")
        mock_svc.provider.pdf_ocr.assert_not_called()
        assert result == []

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_vision_fallback_not_called_for_non_pdf(self, mock_kf, isolated_env, mock_svc):
        mock_kf.return_value = _make_empty_result()
        cfg.enable_ocr = True
        f = isolated_env / "doc.txt"
        f.write_text("")

        from lilbee.data.ingest import ingest_document

        result = await ingest_document(f, "doc.txt", "text")
        mock_svc.provider.pdf_ocr.assert_not_called()
        assert result == []

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_vision_fallback_empty_vision_text_returns_empty(
        self, mock_kf, isolated_env, mock_svc
    ):
        cfg.enable_ocr = True
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        mock_kf.return_value = _make_empty_result()
        mock_svc.provider.pdf_ocr.return_value = []

        f = isolated_env / "blank.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        result = await ingest_document(f, "blank.pdf", "pdf")
        assert result == []

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_no_vision_fallback_when_text_meaningful(self, mock_kf, isolated_env, mock_svc):
        mock_kf.return_value = _make_kreuzberg_result(
            text="Meaningful PDF content. " * 20, num_chunks=1, has_pages=True
        )
        cfg.enable_ocr = True
        f = isolated_env / "good.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        result = await ingest_document(f, "good.pdf", "pdf")
        mock_svc.provider.pdf_ocr.assert_not_called()
        assert len(result) > 0

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_vision_fallback_no_chunks_returns_empty(self, mock_kf, isolated_env, mock_svc):
        cfg.enable_ocr = True
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        mock_kf.return_value = _make_empty_result()
        mock_svc.provider.pdf_ocr.return_value = _vision_pages((1, "Some text"))

        f = isolated_env / "nochunks.pdf"
        f.write_bytes(b"fake pdf")

        with mock.patch("lilbee.data.ingest.extract.chunk_text", return_value=[]):
            from lilbee.data.ingest import ingest_document

            result = await ingest_document(f, "nochunks.pdf", "pdf")
        assert result == []

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_vision_fallback_bypasses_semantic_chunking(
        self, mock_kf, isolated_env, mock_svc
    ):
        """Per-page OCR text gets ``chunk_text(use_semantic=False)`` to skip the
        embedding round-trip per single-topic page."""
        cfg.enable_ocr = True
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        mock_kf.return_value = _make_empty_result()
        mock_svc.provider.pdf_ocr.return_value = _vision_pages(
            (1, "page one text"), (2, "page two text")
        )

        f = isolated_env / "bypass.pdf"
        f.write_bytes(b"fake pdf")

        with mock.patch(
            "lilbee.data.ingest.extract.chunk_text", return_value=["chunk"]
        ) as mock_chunk:
            from lilbee.data.ingest import ingest_document

            await ingest_document(f, "bypass.pdf", "pdf")

        assert mock_chunk.call_count == 2, "chunk_text called once per OCR page"
        for call in mock_chunk.call_args_list:
            assert call.kwargs.get("use_semantic") is False


class TestShouldRunOcrAutoDetect:
    def test_auto_detect_vision_model_set(self, isolated_env):
        """When enable_ocr is None, _should_run_ocr reflects cfg.vision_model."""
        cfg.enable_ocr = None
        cfg.vision_model = "noctrex/LightOnOCR-2-1B-GGUF/lightonocr-Q4_K_M.gguf"
        from lilbee.data.ingest.extract import _should_run_ocr

        assert _should_run_ocr() is True

    def test_auto_detect_vision_model_empty(self, isolated_env):
        """When enable_ocr is None and cfg.vision_model is empty, returns False."""
        cfg.enable_ocr = None
        cfg.vision_model = ""
        from lilbee.data.ingest.extract import _should_run_ocr

        assert _should_run_ocr() is False

    def test_force_on_with_empty_vision_model(self, isolated_env):
        """enable_ocr=True still returns True; caller may fall back to Tesseract."""
        cfg.enable_ocr = True
        cfg.vision_model = ""
        from lilbee.data.ingest.extract import _should_run_ocr

        assert _should_run_ocr() is True

    def test_force_off(self, isolated_env):
        """enable_ocr=False always returns False regardless of vision_model."""
        cfg.enable_ocr = False
        cfg.vision_model = "noctrex/LightOnOCR-2-1B-GGUF/lightonocr-Q4_K_M.gguf"
        from lilbee.data.ingest.extract import _should_run_ocr

        assert _should_run_ocr() is False


class TestOcrFallbackBackendDispatch:
    """``_handle_scanned_pdf_fallback`` routes to the right backend on the pool."""

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_vision_backend_when_ocr_enabled_and_model_set(
        self, mock_kf, isolated_env, mock_svc
    ):
        cfg.enable_ocr = True
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        cfg.ocr_timeout = 60.0
        mock_kf.return_value = _make_empty_result()
        mock_svc.provider.pdf_ocr.return_value = _vision_pages((1, "Vision text. " * 10))

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        result = await ingest_document(f, "scanned.pdf", "pdf")
        assert mock_svc.provider.pdf_ocr.call_args.kwargs["backend"] == "vision"
        assert mock_svc.provider.pdf_ocr.call_args.kwargs["per_page_timeout_s"] == 60.0
        assert len(result) > 0

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_tesseract_backend_when_no_vision_model(self, mock_kf, isolated_env, mock_svc):
        """Without a vision model the scanned PDF runs Tesseract inline (no pool)."""
        cfg.enable_ocr = True
        cfg.vision_model = ""
        cfg.tesseract_timeout = 30.0
        # First call: pre-extract that flags the PDF as scanned. Second
        # call: the inline Tesseract fallback that produces real chunks.
        mock_kf.side_effect = [
            _make_empty_result(),
            _make_kreuzberg_result(text="Tesseract OCR text. " * 10, num_chunks=1, has_pages=True),
        ]

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        result = await ingest_document(f, "scanned.pdf", "pdf")
        # Pool pdf_ocr is not used for tesseract.
        mock_svc.provider.pdf_ocr.assert_not_called()
        assert mock_kf.call_count == 2
        assert len(result) > 0

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_tesseract_returning_empty_logs_skip(
        self, mock_kf, isolated_env, mock_svc, caplog
    ):
        """Tesseract returning no pages produces a user-facing skip warning."""
        cfg.enable_ocr = False
        cfg.vision_model = ""
        # Both pre-extract and tesseract fallback return empty.
        mock_kf.side_effect = [_make_empty_result(), _make_empty_result()]

        f = isolated_env / "blank.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        with caplog.at_level("WARNING", logger="lilbee.data.ingest.extract"):
            result = await ingest_document(f, "blank.pdf", "pdf")
        assert result == []
        assert "Skipped blank.pdf" in caplog.text

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_pool_exception_returns_empty(self, mock_kf, isolated_env, mock_svc):
        """Worker error in pdf_ocr is logged and surfaces as an empty result."""
        cfg.enable_ocr = True
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        mock_kf.return_value = _make_empty_result()
        mock_svc.provider.pdf_ocr.side_effect = RuntimeError("pool died")

        f = isolated_env / "broken.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        result = await ingest_document(f, "broken.pdf", "pdf", quiet=True)
        assert result == []

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_tesseract_no_timeout_runs_uncapped(self, mock_kf, isolated_env, mock_svc):
        """``cfg.tesseract_timeout == 0`` skips ``asyncio.wait_for`` and runs uncapped."""
        cfg.enable_ocr = False
        cfg.vision_model = ""
        cfg.tesseract_timeout = 0
        mock_kf.side_effect = [
            _make_empty_result(),
            _make_kreuzberg_result(text="Tesseract OCR text. " * 10, num_chunks=1, has_pages=True),
        ]

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        result = await ingest_document(f, "scanned.pdf", "pdf")
        assert mock_kf.call_count == 2
        assert len(result) > 0

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_tesseract_timeout_logs_and_returns_empty(
        self, mock_kf, isolated_env, mock_svc, caplog
    ):
        """Tesseract exceeding the wall-clock cap logs a warning and skips the file."""
        cfg.enable_ocr = False
        cfg.vision_model = ""
        cfg.tesseract_timeout = 30.0
        # Pre-extract returns empty; tesseract fallback's wait_for fires.
        mock_kf.side_effect = [_make_empty_result(), _make_kreuzberg_result()]

        async def _instant_timeout(coro, *, timeout):
            coro.close()
            raise TimeoutError

        with mock.patch("asyncio.wait_for", side_effect=_instant_timeout):
            f = isolated_env / "scanned.pdf"
            f.write_bytes(b"fake pdf")

            from lilbee.data.ingest import ingest_document

            with caplog.at_level("WARNING", logger="lilbee.data.ingest.extract"):
                result = await ingest_document(f, "scanned.pdf", "pdf")
        assert result == []
        assert "Tesseract OCR exceeded" in caplog.text

    @mock.patch("kreuzberg.extract_file_sync", new_callable=Mock)
    async def test_tesseract_extract_exception_logs_and_returns_empty(
        self, mock_kf, isolated_env, mock_svc, caplog
    ):
        """A kreuzberg backend crash logs a warning and returns no chunks."""
        cfg.enable_ocr = False
        cfg.vision_model = ""
        cfg.tesseract_timeout = 30.0
        # Pre-extract OK; tesseract fallback raises.
        mock_kf.side_effect = [_make_empty_result(), RuntimeError("tesseract binary missing")]

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")

        from lilbee.data.ingest import ingest_document

        with caplog.at_level("WARNING", logger="lilbee.data.ingest.extract"):
            result = await ingest_document(f, "scanned.pdf", "pdf")
        assert result == []
        assert "OCR via tesseract backend failed" in caplog.text


class TestSharedProgress:
    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_contextvar_set_during_progress(self, mock_extract_file, isolated_env):
        """shared_progress contextvar is set inside _collect_results_with_progress."""
        from lilbee.runtime.progress import shared_progress

        (isolated_env / "a.txt").write_text("Content for shared progress test.")

        captured: list[tuple] = []

        # Use on_progress callback to capture the contextvar value during ingestion
        def capture_progress(event_type, data):
            val = shared_progress.get(None)
            if val is not None and val not in captured:
                captured.append(val)

        from lilbee.data.ingest import sync

        await sync(quiet=False, on_progress=capture_progress)
        assert len(captured) > 0, "shared_progress was never set during progress bar"
        progress_obj, task_id = captured[0]
        assert progress_obj is not None
        assert task_id is not None

    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_contextvar_not_set_in_quiet_mode(self, mock_extract_file, isolated_env):
        """shared_progress contextvar is NOT set in quiet mode (no progress bar)."""
        from lilbee.runtime.progress import shared_progress

        (isolated_env / "b.txt").write_text("Content for quiet mode test.")

        captured: list[object] = []

        def capture_progress(event_type, data):
            val = shared_progress.get(None)
            if val is not None:
                captured.append(val)

        from lilbee.data.ingest import sync

        await sync(quiet=True, on_progress=capture_progress)
        assert len(captured) == 0, "shared_progress should not be set in quiet mode"

    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_contextvar_reset_after_progress(self, mock_extract_file, isolated_env):
        """shared_progress is reset to None after _collect_results_with_progress completes."""
        from lilbee.runtime.progress import shared_progress

        (isolated_env / "c.txt").write_text("Content for reset test.")
        from lilbee.data.ingest import sync

        await sync(quiet=False)
        assert shared_progress.get(None) is None


class TestIngestMarkdownEdgeCases:
    async def test_empty_markdown_returns_empty(self, isolated_env):
        from lilbee.data.ingest import ingest_markdown

        md = isolated_env / "empty.md"
        md.write_text("   ")
        result = await ingest_markdown(md, "empty.md")
        assert result == []

    async def test_no_chunks_returns_empty(self, isolated_env):
        from lilbee.data.ingest import ingest_markdown

        md = isolated_env / "blank.md"
        md.write_text("some text")
        with mock.patch("lilbee.data.ingest.extract.chunk_text", return_value=[]):
            result = await ingest_markdown(md, "blank.md")
        assert result == []

    async def test_frontmatter_only_produces_chunks(self, isolated_env):
        from lilbee.data.ingest import ingest_markdown

        md = isolated_env / "fm_only.md"
        md.write_text("---\ntitle: Just Frontmatter\ntags: [test]\n---\n")
        result = await ingest_markdown(md, "fm_only.md")
        assert len(result) > 0, "Frontmatter content should be indexed"


class TestIngestDocumentEdgeCases:
    async def test_empty_extraction_returns_empty(self, isolated_env):
        """Structured formats now go through kreuzberg: empty result yields no chunks."""
        from lilbee.data.ingest import ingest_document

        empty_result = mock.MagicMock(chunks=[])
        mock_extract = Mock(return_value=empty_result)
        with mock.patch("kreuzberg.extract_file_sync", mock_extract):
            result = await ingest_document(isolated_env / "e.xml", "e.xml", "xml")
        assert result == []

    async def test_no_chunks_returns_empty(self, isolated_env):
        from lilbee.data.ingest import ingest_document

        no_chunks_result = mock.MagicMock(chunks=[])
        mock_extract = Mock(return_value=no_chunks_result)
        with mock.patch("kreuzberg.extract_file_sync", mock_extract):
            result = await ingest_document(isolated_env / "s.xml", "s.xml", "xml")
        assert result == []


class TestChunkViaKreuzberg:
    def test_empty_returns_empty(self):
        from lilbee.data.chunk import chunk_text

        assert chunk_text("") == []

    def test_returns_chunks(self):
        from lilbee.data.chunk import chunk_text

        result = chunk_text("Some text that should be chunked.")
        assert len(result) >= 1


class TestConceptIndexing:
    @pytest.fixture(autouse=True)
    def _mock_concepts_available(self):
        with mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True):
            yield

    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_concept_extraction_called_during_ingest(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """When concept_graph is enabled, extraction is called after ingest."""
        cfg.concept_graph = True
        (isolated_env / "concept_test1.txt").write_text("Content for concepts test one.")

        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts_batch.return_value = [["test"]]
        from lilbee.data.ingest import sync

        await sync(quiet=True)
        mock_svc.concepts.extract_concepts_batch.assert_called()

    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_concept_disabled_skips_extraction(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """When concept_graph is disabled, extraction is not called."""
        cfg.concept_graph = False
        (isolated_env / "concept_test2.txt").write_text("Some test content.")

        from lilbee.data.ingest import sync

        await sync(quiet=True)
        mock_svc.concepts.extract_concepts_batch.assert_not_called()

    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_concept_failure_does_not_break_ingest(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """When concept extraction raises, ingest still succeeds."""
        cfg.concept_graph = True
        (isolated_env / "concept_test2.txt").write_text("Some test content.")

        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts_batch.side_effect = RuntimeError("spacy broke")
        from lilbee.data.ingest import sync

        result = await sync(quiet=True)
        assert "concept_test2.txt" in result.added

    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_cluster_rebuild_called_after_sync(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """After sync completes, rebuild_clusters is called."""
        cfg.concept_graph = True
        (isolated_env / "concept_test4.txt").write_text("Some test content.")

        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts_batch.return_value = [["test"]]
        from lilbee.data.ingest import sync

        await sync(quiet=True)
        mock_svc.concepts.rebuild_clusters.assert_called()

    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_cluster_rebuild_failure_does_not_break_sync(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """When cluster rebuild raises, sync still succeeds."""
        cfg.concept_graph = True
        (isolated_env / "rebuild_test.txt").write_text("Content for rebuild test.")

        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts_batch.return_value = [["test"]]
        mock_svc.concepts.rebuild_clusters.side_effect = RuntimeError("leiden broke")
        from lilbee.data.ingest import sync

        result = await sync(quiet=True)
        assert "rebuild_test.txt" in result.added

    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_graph_none_skips_indexing(self, mock_extract_file, isolated_env, mock_svc):
        """When get_graph() returns None, concept indexing is skipped gracefully."""
        cfg.concept_graph = True
        (isolated_env / "graph_none_test.txt").write_text("Content for graph none test.")

        mock_svc.concepts.get_graph.return_value = False
        from lilbee.data.ingest import sync

        result = await sync(quiet=True)
        assert "graph_none_test.txt" in result.added

    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_concepts_unavailable_skips_rebuild(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """When concepts_available() returns False, _rebuild_concept_clusters is a no-op."""
        cfg.concept_graph = True
        (isolated_env / "unavail_rebuild.txt").write_text("Content for unavailable test.")

        with mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=False):
            from lilbee.data.ingest import sync

            result = await sync(quiet=True)
        assert "unavail_rebuild.txt" in result.added
        mock_svc.concepts.rebuild_clusters.assert_not_called()

    @mock.patch(
        "kreuzberg.extract_file_sync", new_callable=Mock, return_value=_make_kreuzberg_result()
    )
    async def test_concepts_unavailable_skips_indexing(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """When concepts_available() returns False, _index_concepts is a no-op."""
        cfg.concept_graph = True
        (isolated_env / "unavail_index.txt").write_text("Content for unavailable index test.")

        with mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=False):
            from lilbee.data.ingest import sync

            result = await sync(quiet=True)
        assert "unavail_index.txt" in result.added
        mock_svc.concepts.extract_concepts_batch.assert_not_called()


class TestUnsupportedFileInSync:
    async def test_classify_none_raises_value_error(self, isolated_env, mock_svc):
        """When classify_file returns None for a discovered file, sync raises ValueError."""
        mystery = isolated_env / "mystery.bin"
        mystery.write_bytes(b"\x00\x01\x02")

        # discover_files includes the file, but classify_file returns None
        with (
            mock.patch(
                "lilbee.data.ingest.pipeline.discover_files",
                return_value={"mystery.bin": mystery},
            ),
            mock.patch("lilbee.data.ingest.pipeline.classify_file", return_value=None),
        ):
            from lilbee.data.ingest import sync

            with pytest.raises(ValueError, match="Unsupported file slipped through"):
                await sync(quiet=True)
