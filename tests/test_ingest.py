"""Tests for the document sync engine (mocked: no live server needed)."""

import asyncio
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

import lilbee.app.services as svc_mod
from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths to temp dir for every test."""
    snapshot = cfg.model_copy()

    docs = tmp_path / "documents"
    docs.mkdir()
    cfg.documents_dir = docs
    cfg.data_root = tmp_path
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
    store.add_chunks.side_effect = len

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

    def _write_batch(items):
        # Mirror the real write: each batched doc upserts a source record so
        # multi-sync tests see it via get_sources().
        for it in items:
            _upsert(it.source, it.file_hash, len(it.records))
        return sum(len(it.records) for it in items)

    store.write_chunks_batch.side_effect = _write_batch
    store.delete_source.side_effect = lambda fn: _sources.pop(fn, None)
    store.delete_by_source.return_value = None

    def _remove_documents(names, **_kw):
        return [_sources.pop(n, None) for n in names]

    store.remove_documents.side_effect = _remove_documents
    store.drop_all.side_effect = lambda: _sources.clear()
    store.ensure_fts_index.return_value = None
    embedder = MagicMock()
    embedder.embed.side_effect = lambda text, **kw: [0.1] * 768
    embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]
    # Real int so sync()'s truncated_total delta is 0, not a coerced MagicMock.
    embedder.truncated_total = 0
    searcher = MagicMock()
    services = make_mock_services(store=store, embedder=embedder, searcher=searcher)
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


def _install_real_store():
    """Swap the autouse mock store for a real LanceDB-backed one."""
    from lilbee.data.store import Store
    from tests.conftest import make_mock_services

    store = Store(cfg)
    svc_mod.set_services(make_mock_services(store=store))
    return store


def _real_ingest_result(name, *, file_hash, page_text="page one of a.pdf", stat=None):
    """A store-schema _IngestResult with one chunk and one page-text row."""
    from lilbee.data.ingest.types import _IngestResult
    from lilbee.data.store import PageTextRecord

    records = [
        {
            "source": name,
            "content_type": "pdf",
            "chunk_type": "raw",
            "page_start": 1,
            "page_end": 1,
            "line_start": 0,
            "line_end": 0,
            "chunk": f"{name} {file_hash} chunk",
            "chunk_index": 0,
            "vector": [0.1] * cfg.embedding_dim,
        }
    ]
    return _IngestResult(
        name=name,
        path=Path(name),
        chunk_count=1,
        error=None,
        file_hash=file_hash,
        records=records,
        needs_cleanup=True,
        page_texts=[PageTextRecord(source=name, page=1, text=page_text, content_type="pdf")],
        stat=stat,
    )


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
        chunk = mock.MagicMock()
        chunk.content = chunk_text
        chunk.metadata = mock.MagicMock(
            byte_start=0,
            byte_end=len(chunk_text),
            chunk_index=i,
            total_chunks=num_chunks,
            token_count=None,
            first_page=(i + 1) if has_pages else None,
            last_page=(i + 1) if has_pages else None,
        )
        chunks.append(chunk)

    result = mock.MagicMock()
    result.chunks = chunks
    result.content = text
    result.document = document
    result.pages = (
        [mock.MagicMock(page_number=i + 1, content=chunks[i].content) for i in range(num_chunks)]
        if has_pages
        else []
    )
    return result


def _make_empty_result():
    """Build a mock kreuzberg ExtractionResult with no chunks."""
    result = mock.MagicMock()
    result.chunks = []
    result.content = ""
    result.document = None
    return result


@mock.patch(
    "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
)
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
        (isolated_env / "data.exe").write_bytes(b"binary data")
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

    async def test_many_small_files_batch_into_one_write(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        # Bulk ingest must amortize the LanceDB write: many small files below the
        # flush threshold land in a single write_chunks_batch, not one per file.
        for i in range(5):
            (isolated_env / f"f{i}.txt").write_text(f"content {i}")
        from lilbee.data.ingest import sync

        result = await sync(quiet=True)
        assert len(result.added) == 5
        mock_svc.store.write_chunks_batch.assert_called_once()
        items = mock_svc.store.write_chunks_batch.call_args.args[0]
        assert sorted(it.source for it in items) == [f"f{i}.txt" for i in range(5)]

    async def test_concept_writes_batched_per_flush(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        # Concept rows ride the flush: one write_concept_records call carries
        # every flushed file's merged rows, not one store write per file.
        from lilbee.data.ingest import sync
        from lilbee.data.store import ConceptRecords

        cfg.concept_graph = True
        for i in range(3):
            (isolated_env / f"f{i}.txt").write_text(f"content {i}")
        mock_svc.concepts.extract_concepts_batch.side_effect = lambda texts: [
            ["alpha", "beta"] for _ in texts
        ]
        mock_svc.concepts.build_concept_records.side_effect = lambda chunk_ids, lists: (
            ConceptRecords(
                nodes=[{"concept": chunk_ids[0][0], "cluster_id": 0, "degree": 1}],
                edges=[],
                chunk_concepts=[],
            )
        )

        with mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True):
            result = await sync(quiet=True)
        assert len(result.added) == 3
        mock_svc.concepts.write_concept_records.assert_called_once()
        merged = mock_svc.concepts.write_concept_records.call_args.args[0]
        assert sorted(n["concept"] for n in merged.nodes) == ["f0.txt", "f1.txt", "f2.txt"]

    async def test_midstream_flush_when_chunk_threshold_crossed(
        self, mock_extract_file, isolated_env, mock_svc, monkeypatch
    ):
        # A long ingest must not hold every chunk in memory until the end: once the
        # buffer crosses the flush threshold it writes mid-stream, so a low threshold
        # forces more than the single final write.
        from lilbee.data.ingest import pipeline, sync

        monkeypatch.setattr(pipeline, "_WRITE_FLUSH_CHUNKS", 1)
        for i in range(3):
            (isolated_env / f"f{i}.txt").write_text(f"content {i}")

        result = await sync(quiet=True)
        assert len(result.added) == 3
        assert mock_svc.store.write_chunks_batch.call_count >= 2

    def test_flush_writes_marks_files_failed_on_write_error(self, _mock_extract_file):
        # A failed batch write moves every buffered file to ``failed`` and out of
        # added/updated, since none of its chunks persisted; the buffer still clears.
        # ``_mock_extract_file`` is the class-level kreuzberg patch, unused here.
        from lilbee.app.services import get_services
        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.types import _IngestResult

        get_services().store.write_chunks_batch.side_effect = RuntimeError("disk full")
        result = _IngestResult(
            name="a.txt",
            path=Path("a.txt"),
            chunk_count=1,
            error=None,
            file_hash="h",
            records=[{"text": "x"}],
            needs_cleanup=True,
        )

        buffer = [result]
        added: dict[str, None] = {"a.txt": None}
        updated: dict[str, None] = {}
        failed: dict[str, None] = {}
        flush_failed: set[str] = set()
        pipeline._flush_writes(buffer, added, updated, failed, {}, flush_failed)

        assert added == {}
        assert list(failed) == ["a.txt"]
        assert flush_failed == {"a.txt"}
        assert buffer == []

    def test_flush_failure_drops_page_text_only_file_from_skipped(self, _mock_extract_file):
        # A page-text-only file is pre-marked skipped at classification but still
        # buffered; if the flush fails it must end up in failed only, never both.
        from lilbee.app.services import get_services
        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.types import _IngestResult

        get_services().store.write_chunks_batch.side_effect = RuntimeError("disk full")
        result = _IngestResult(
            name="blank.pdf",
            path=Path("blank.pdf"),
            chunk_count=0,
            error=None,
            file_hash="h",
            records=[],
            needs_cleanup=True,
            page_texts=[{"source": "blank.pdf", "page": 1, "text": " ", "content_type": "pdf"}],
        )
        skipped: dict[str, None] = {"blank.pdf": None}
        failed: dict[str, None] = {}
        pipeline._flush_writes([result], {}, {}, failed, skipped, set())

        assert list(failed) == ["blank.pdf"]
        assert "blank.pdf" not in skipped  # not double-listed

    def test_flush_writes_retries_once_on_lock_timeout(self, _mock_extract_file, monkeypatch):
        # A LockTimeoutError on the first write is retried after a short backoff;
        # the second attempt succeeds, so the files stay recorded as written.
        from lilbee.app.services import get_services
        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.types import _IngestResult
        from lilbee.runtime.lock import LockTimeoutError

        sleeps: list[float] = []
        monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
        store = get_services().store
        store.write_chunks_batch.side_effect = [LockTimeoutError("busy"), 1]
        result = _IngestResult(
            name="a.txt",
            path=Path("a.txt"),
            chunk_count=1,
            error=None,
            file_hash="h",
            records=[{"text": "x"}],
            needs_cleanup=True,
        )

        added: dict[str, None] = {"a.txt": None}
        failed: dict[str, None] = {}
        flush_failed: set[str] = set()
        pipeline._flush_writes([result], added, {}, failed, {}, flush_failed)

        assert store.write_chunks_batch.call_count == 2
        assert sleeps == [pipeline._FLUSH_RETRY_DELAY_SECONDS]
        assert list(added) == ["a.txt"]
        assert failed == {}
        assert flush_failed == set()

    def test_flush_writes_lock_timeout_twice_marks_flush_failed(
        self, _mock_extract_file, monkeypatch
    ):
        # Both attempts time out: the file is failed AND flush_failed (transient,
        # retried next sync), never skip-marked.
        from lilbee.app.services import get_services
        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.types import _IngestResult
        from lilbee.runtime.lock import LockTimeoutError

        monkeypatch.setattr(pipeline.time, "sleep", lambda _s: None)
        get_services().store.write_chunks_batch.side_effect = LockTimeoutError("busy")
        result = _IngestResult(
            name="a.txt",
            path=Path("a.txt"),
            chunk_count=1,
            error=None,
            file_hash="h",
            records=[{"text": "x"}],
            needs_cleanup=True,
        )

        added: dict[str, None] = {"a.txt": None}
        failed: dict[str, None] = {}
        flush_failed: set[str] = set()
        pipeline._flush_writes([result], added, {}, failed, {}, flush_failed)

        assert list(failed) == ["a.txt"]
        assert flush_failed == {"a.txt"}

    async def test_flush_failure_does_not_write_skip_marker(self, _mock_extract_file, isolated_env):
        # End-to-end: a hard flush failure records the file as failed but leaves
        # no skip marker, so the next sync re-plans it; an extraction failure
        # still gets a marker.
        from lilbee.app.services import get_services
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import load_skip_markers

        (isolated_env / "doc.txt").write_text("content here")
        get_services().store.write_chunks_batch.side_effect = RuntimeError("disk full")

        result = await sync(quiet=True)
        assert "doc.txt" in result.failed
        assert "doc.txt" not in load_skip_markers(cfg.data_root)

        # Next sync re-plans the file (no marker, hash unchanged but never stored).
        get_services().store.write_chunks_batch.side_effect = None
        retry = await sync(quiet=True)
        assert "doc.txt" in retry.added

    def test_flush_path_persists_page_texts_through_cleanup(self, _mock_extract_file):
        # Real store: the flush's cleanup delete runs in the same transaction as
        # the page-text write, so a needs_cleanup re-ingest must not wipe the
        # page texts it just wrote.
        from lilbee.data.ingest import pipeline

        store = _install_real_store()
        result = _real_ingest_result("a.pdf", file_hash="h1")
        pipeline._flush_writes([result], {"a.pdf": None}, {}, {}, {}, set())
        assert [row["text"] for row in store.get_page_texts("a.pdf")] == ["page one of a.pdf"]

        replanned = _real_ingest_result("a.pdf", file_hash="h2", page_text="page one, edited")
        pipeline._flush_writes([replanned], {"a.pdf": None}, {}, {}, {}, set())
        assert [row["text"] for row in store.get_page_texts("a.pdf")] == ["page one, edited"]
        sources = {s["filename"]: s for s in store.get_sources()}
        assert sources["a.pdf"]["file_hash"] == "h2"

    def test_page_text_write_failure_keeps_source_row_at_old_hash_and_stat(
        self, _mock_extract_file
    ):
        # Real store: a page-text failure inside the batched transaction must
        # leave the source row at the old hash/stat so the file replans next sync.
        import lilbee.data.store.core as core_mod
        from lilbee.core.config import PAGE_TEXTS_TABLE
        from lilbee.data.ingest import pipeline
        from lilbee.data.store import SourceStat, source_stat

        store = _install_real_store()
        old_stat = SourceStat(11, 22, 33)
        first = _real_ingest_result("a.pdf", file_hash="h1", stat=old_stat)
        pipeline._flush_writes([first], {"a.pdf": None}, {}, {}, {}, set())

        real_ensure = core_mod.ensure_table

        def _failing_ensure(db, name, schema):
            if name == PAGE_TEXTS_TABLE:
                raise RuntimeError("page table corrupt")
            return real_ensure(db, name, schema)

        replanned = _real_ingest_result(
            "a.pdf", file_hash="h2", page_text="edited", stat=SourceStat(99, 88, 77)
        )
        added: dict[str, None] = {"a.pdf": None}
        failed: dict[str, None] = {}
        flush_failed: set[str] = set()
        with mock.patch.object(core_mod, "ensure_table", _failing_ensure):
            pipeline._flush_writes([replanned], added, {}, failed, {}, flush_failed)

        assert added == {}
        assert list(failed) == ["a.pdf"]
        assert flush_failed == {"a.pdf"}
        record = {s["filename"]: s for s in store.get_sources()}["a.pdf"]
        assert record["file_hash"] == "h1"
        assert source_stat(record) == old_stat

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
        from lilbee.data.ingest.pipeline import _produce_records as orig_ingest

        async def _failing_ingest(path, name, content_type, **kwargs):
            if "bad" in name:
                raise RuntimeError("simulated failure")
            return await orig_ingest(path, name, content_type, **kwargs)

        with patch("lilbee.data.ingest.pipeline._produce_records", side_effect=_failing_ingest):
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

        from lilbee.data.ingest.pipeline import _produce_records as orig_ingest

        async def _failing_ingest(path, name, content_type, **kwargs):
            if "flaky" in name:
                raise RuntimeError("simulated failure on update")
            return await orig_ingest(path, name, content_type, **kwargs)

        with patch("lilbee.data.ingest.pipeline._produce_records", side_effect=_failing_ingest):
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

        with patch("lilbee.data.ingest.pipeline._produce_records", side_effect=_fail):
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

        from lilbee.data.ingest.pipeline import _produce_records as orig

        async def _fail(path, name, ct, **kwargs):
            if "qflaky" in name:
                raise RuntimeError("quiet fail")
            return await orig(path, name, ct, **kwargs)

        with patch("lilbee.data.ingest.pipeline._produce_records", side_effect=_fail):
            result = await sync(quiet=True)
        assert "qflaky.txt" in result.failed
        assert "qflaky.txt" not in result.updated


@mock.patch(
    "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
)
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

        from lilbee.data.ingest.types import FileToProcess

        added = {"a.txt": None, "b.txt": None}
        files = [
            FileToProcess("a.txt", isolated_env / "a.txt", "text", "hash_a", False),
            FileToProcess("b.txt", isolated_env / "b.txt", "text", "hash_b", False),
        ]
        with pytest.raises(asyncio.CancelledError):
            await ingest_batch(files, added, {}, {}, {}, quiet=True, cancel=cancel)

    async def test_cancel_in_batch_still_flushes_completed_sibling(self, isolated_env, mock_svc):
        """A cancel landing in the same done-batch as a genuinely completed file must
        not drop that file: it is buffered and flushed before the cancel propagates
        (bb-ziks.21). Prior code re-raised the first CancelledError from fut.result()
        and abandoned the sibling."""
        from lilbee.data.ingest import pipeline

        async def _ok():
            return _real_ingest_result("good.pdf", file_hash="h-good")

        async def _cancelled():
            raise asyncio.CancelledError

        added: dict[str, None] = {}
        flushed: list[str] = []

        def _record_flush(buffer, _added, _updated, _failed, _skipped, _flush_failed):
            flushed.extend(r.name for r in buffer)

        # `done` is a set, so the order the loop sees the two completed futures is
        # not guaranteed. Force the cancelled future to be processed FIRST so the
        # test deterministically fails on the old code (its CancelledError aborted
        # the loop before the completed sibling) and passes only with the fix.
        real_wait = pipeline.asyncio.wait

        async def _cancel_first_wait(fs, **kwargs):
            done, pending = await real_wait(fs, **kwargs)

            def _is_cancel(fut):
                return fut.cancelled() or isinstance(
                    fut.exception() if not fut.cancelled() else None, asyncio.CancelledError
                )

            return sorted(done, key=lambda f: 0 if _is_cancel(f) else 1), pending

        with (
            mock.patch.object(pipeline.asyncio, "wait", _cancel_first_wait),
            mock.patch.object(pipeline, "_flush_writes", side_effect=_record_flush),
            mock.patch.object(pipeline, "_purge_emptied_sources"),
            pytest.raises(asyncio.CancelledError),
        ):
            await pipeline._collect_results(
                iter([_ok(), _cancelled()]),
                total=2,
                added=added,
                updated={},
                failed={},
                skipped={},
                window=2,
            )

        assert "good.pdf" in flushed

    async def test_atomic_delete_for_modified_file(self, mock_extract_file, isolated_env, mock_svc):
        """Modified file: its old chunks are cleaned up in the same batched write."""
        from lilbee.data.ingest import sync

        f = isolated_env / "doc.txt"
        f.write_text("Version 1")
        await sync(quiet=True)

        mock_svc.store.write_chunks_batch.reset_mock()

        f.write_text("Version 2, modified content")
        await sync(quiet=True)

        # The batched write carries a cleanup item for the modified source, so the
        # delete and the new chunks land in one transaction.
        mock_svc.store.write_chunks_batch.assert_called_once()
        items = mock_svc.store.write_chunks_batch.call_args.args[0]
        doc_item = next(it for it in items if it.source == "doc.txt")
        assert doc_item.needs_cleanup is True

    async def test_cancel_preserves_old_chunks_for_modified_file(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """If cancel is set before a modified file is processed, no write happens."""
        import threading

        from lilbee.data.ingest import sync

        f = isolated_env / "doc.txt"
        f.write_text("Version 1")
        await sync(quiet=True)

        mock_svc.store.write_chunks_batch.reset_mock()

        f.write_text("Version 2, modified content")
        cancel = threading.Event()
        cancel.set()
        await sync(quiet=True, cancel=cancel)

        # Cancel fired before the file was processed, so nothing was written and
        # the old chunks were never deleted.
        mock_svc.store.write_chunks_batch.assert_not_called()


class TestIngestHelpers:
    """Cover edge cases in ingest_document and ingest_code_sync."""

    @mock.patch(
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_empty_result()
    )
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

    async def test_ingest_code_header_uses_relative_source_name(self, isolated_env, mock_svc):
        """ingest_code_sync threads the relative source_name into the chunk header so
        the indexed content never carries the host's absolute path (bb-ziks.19).
        Reverting to chunk_code(path) would emit the file's basename instead."""
        from unittest.mock import patch

        from lilbee.data.ingest import ingest_code_sync

        class _FakeMeta:
            symbols_defined = ("alpha",)

        class _FakeChunk:
            content = "def alpha():\n    return 1\n"
            start_line = 0
            end_line = 2
            metadata = _FakeMeta()

        class _FakeResult:
            chunks = (_FakeChunk(),)

        f = isolated_env / "abs_module.py"
        f.write_text("def alpha():\n    return 1\n")
        with (
            patch("lilbee.data.code_chunker._ensure_language", return_value=True),
            patch("lilbee.data.code_chunker.process", return_value=_FakeResult()),
        ):
            records = ingest_code_sync(f, "pkg/mod.py")
        assert records
        joined = "\n".join(r["chunk"] for r in records)
        assert "# File: pkg/mod.py" in joined
        assert str(f) not in joined  # absolute path must never leak into content

    @mock.patch("kreuzberg.extract_file", new_callable=mock.AsyncMock)
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
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_cancelled_error_propagates(self, mock_extract_file, isolated_env):
        """CancelledError in _process_one is re-raised, not swallowed."""
        import asyncio

        async def _cancel(*args, **kwargs):
            raise asyncio.CancelledError()

        with mock.patch("lilbee.data.ingest.pipeline._produce_records", side_effect=_cancel):
            from lilbee.data.ingest import ingest_batch
            from lilbee.data.ingest.types import FileToProcess

            added = {"cancel.txt": None}
            entry = FileToProcess(
                "cancel.txt", isolated_env / "cancel.txt", "text", "abc123", False
            )
            with pytest.raises(asyncio.CancelledError):
                await ingest_batch([entry], added, {}, {}, {}, quiet=True)

    @mock.patch(
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_task_cancelled_error_does_not_orphan_siblings(
        self, mock_extract_file, isolated_env
    ):
        """TaskCancelledError from on_progress must not leak past _process_one.

        Regression test for the firehose where reporter.check_cancelled() raised
        inside the progress callback, fell through the broad ``except Exception``
        in ``_process_one``, then re-entered ``on_progress(FILE_DONE)`` and
        raised again. The second raise leaked out of _process_one, _collect_results
        exited on the first failure, and every sibling task ended up logging
        "Task exception was never retrieved". After the fix the entire batch
        winds down cleanly as a single asyncio.CancelledError.
        """
        import asyncio

        from lilbee.data.ingest import ingest_batch
        from lilbee.runtime.cancellation import TaskCancelledError

        # The callback flips on the second invocation so the first file makes
        # progress (which exercises the FILE_DONE re-entry path inside the error
        # handler), then every subsequent event raises.
        calls = [0]

        def on_progress(event_type, data):
            calls[0] += 1
            if calls[0] >= 2:
                raise TaskCancelledError

        # Three files queued concurrently; first will receive FILE_START then
        # error out, the other two get cancelled by _collect_results.
        from lilbee.data.ingest.types import FileToProcess

        files = [
            FileToProcess(f"f{i}.txt", isolated_env / f"f{i}.txt", "text", f"hash_{i}", False)
            for i in range(3)
        ]
        for entry in files:
            entry.path.write_text("hello world")
        added: dict[str, None] = {}
        failed: dict[str, None] = {}
        skipped: dict[str, None] = {}

        # Should raise asyncio.CancelledError (not TaskCancelledError) so the
        # surrounding try/except in _do_sync catches it cleanly.
        with pytest.raises(asyncio.CancelledError):
            await ingest_batch(
                files, added, {}, failed, skipped, quiet=True, on_progress=on_progress
            )


class TestSkipMarkerLifecycle:
    """A file that produces no chunks gets a skip marker; the next sync skips it
    until the file changes or retry_skipped / force_rebuild clears the marker."""

    @staticmethod
    def _zero_chunks(*_args, **_kwargs) -> list:
        # Simulate "OCR found no usable text": no records produced, so the file
        # is recorded as skipped.
        return []

    async def test_failed_file_is_skipped_on_next_sync(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import load_skip_markers

        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline._produce_records", side_effect=self._zero_chunks
        ):
            first = await sync(quiet=True)
            assert "scanned.pdf" in first.skipped
            assert "scanned.pdf" in load_skip_markers(cfg.data_root)
            # Second sync: same content, same hash -> skipped, not retried.
            second = await sync(quiet=True)
            assert "scanned.pdf" not in second.skipped
            assert "scanned.pdf" not in second.added

    async def test_retry_skipped_clears_markers(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync

        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline._produce_records", side_effect=self._zero_chunks
        ):
            await sync(quiet=True)
            # retry_skipped drops the marker so the file is attempted again.
            retried = await sync(quiet=True, retry_skipped=True)
            assert "scanned.pdf" in retried.skipped  # attempted again (still 0 chunks)

    async def test_force_rebuild_also_clears_markers(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync

        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline._produce_records", side_effect=self._zero_chunks
        ):
            await sync(quiet=True)
            rebuilt = await sync(quiet=True, force_rebuild=True)
            assert "scanned.pdf" in rebuilt.skipped  # attempted again after the wipe


class TestZeroChunkPageTextPersistence:
    """A zero-chunk file with page texts persists both and stops replanning."""

    @staticmethod
    async def _pages_no_chunks(
        path, source_name, content_type, *, quiet=False, on_progress=None, page_texts_out=None
    ):
        if page_texts_out is not None:
            page_texts_out.append(
                {"source": source_name, "page": 1, "text": " ", "content_type": "pdf"}
            )
        return []

    async def test_pages_and_source_row_persist_and_replan_stops(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync

        (isolated_env / "blank.pdf").write_bytes(b"%PDF-1.4 whitespace only")
        with mock.patch(
            "lilbee.data.ingest.pipeline._produce_records", side_effect=self._pages_no_chunks
        ):
            first = await sync(quiet=True)
            # 0 searchable chunks: reported as skipped, not added ...
            assert "blank.pdf" not in first.added
            assert "blank.pdf" in first.skipped
            # ... but its page text and source row still persist (one atomic write).
            items = mock_svc.store.write_chunks_batch.call_args.args[0]
            item = next(it for it in items if it.source == "blank.pdf")
            assert item.records == []
            assert [page["page"] for page in item.page_texts] == [1]
            # Next sync: the hash matches the persisted source row -> unchanged.
            mock_svc.store.write_chunks_batch.reset_mock()
            second = await sync(quiet=True)
            assert second.unchanged == 1
            assert "blank.pdf" not in second.added
            assert "blank.pdf" not in second.skipped
            mock_svc.store.write_chunks_batch.assert_not_called()

    async def test_zero_chunk_files_still_trigger_the_flush_threshold(self, isolated_env, mock_svc):
        # Each zero-chunk file counts one unit toward the flush threshold, so a
        # run of them cannot grow the write buffer without bound.
        import lilbee.data.ingest.pipeline as pipeline_mod
        from lilbee.data.ingest import sync

        for i in range(3):
            (isolated_env / f"blank{i}.pdf").write_bytes(b"%PDF-1.4 whitespace only")
        with (
            mock.patch(
                "lilbee.data.ingest.pipeline._produce_records",
                side_effect=self._pages_no_chunks,
            ),
            mock.patch.object(pipeline_mod, "_WRITE_FLUSH_CHUNKS", 2),
            mock.patch.object(
                pipeline_mod, "_flush_writes", wraps=pipeline_mod._flush_writes
            ) as flush_spy,
        ):
            result = await sync(quiet=True)
        # Zero searchable chunks -> reported skipped, but still buffered/flushed.
        assert len(result.skipped) == 3
        assert not result.added
        assert flush_spy.call_count >= 2


class TestDetectPending:
    """Cheap detection: filesystem walk + hash compare against the sources table."""

    def test_returns_zero_when_documents_dir_missing(self, isolated_env, tmp_path):
        cfg.documents_dir = tmp_path / "does_not_exist"
        from lilbee.data.ingest import detect_pending

        assert detect_pending() == 0

    def test_counts_added_updated_and_removed(self, isolated_env, mock_svc):
        from lilbee.data.ingest import detect_pending, file_hash

        new_file = isolated_env / "new.md"
        new_file.write_text("brand new content")

        existing = isolated_env / "existing.md"
        existing.write_text("v1")
        mock_svc.store.upsert_source("existing.md", file_hash(existing), 1, source_type="document")
        existing.write_text("v2")  # hash now diverges from store

        # A row in the store with no matching disk file = removed.
        mock_svc.store.upsert_source("gone.md", "deadbeef", 1, source_type="document")

        # 1 added (new.md) + 1 updated (existing.md) + 1 removed (gone.md) = 3
        assert detect_pending() == 3

    def test_returns_zero_when_in_sync(self, isolated_env, mock_svc):
        from lilbee.data.ingest import detect_pending, file_hash

        f = isolated_env / "synced.md"
        f.write_text("stable")
        mock_svc.store.upsert_source("synced.md", file_hash(f), 1, source_type="document")

        assert detect_pending() == 0

    def test_imported_source_not_counted_as_removed(self, isolated_env, mock_svc):
        from lilbee.data.ingest import detect_pending
        from lilbee.data.store import SourceType

        # Detached import with no backing file: must not show up as pending removal.
        mock_svc.store.upsert_source("shared.pdf", "", 4, source_type=SourceType.IMPORTED)

        assert detect_pending() == 0


class TestStatShortCircuit:
    """Planning hashes only files whose stored (size, mtime) drifted."""

    # A stat capture clearly later than the file's mtime, so the racily-clean
    # tie-breaker does not force a hash in tests of the clean path.
    _CAPTURE_MARGIN_NS = 1_000_000_000

    @classmethod
    def _record(cls, name: str, fhash: str, path: Path | None = None) -> dict:
        from lilbee.data.store import SOURCE_STAT_UNKNOWN

        record = {
            "filename": name,
            "file_hash": fhash,
            "ingested_at": "",
            "chunk_count": 1,
            "source_type": "document",
            "size_bytes": SOURCE_STAT_UNKNOWN,
            "mtime_ns": SOURCE_STAT_UNKNOWN,
            "stat_captured_ns": SOURCE_STAT_UNKNOWN,
        }
        if path is not None:
            st = path.stat()
            record["size_bytes"] = st.st_size
            record["mtime_ns"] = st.st_mtime_ns
            record["stat_captured_ns"] = st.st_mtime_ns + cls._CAPTURE_MARGIN_NS
        return record

    def test_matching_stat_skips_hashing(self, isolated_env, monkeypatch):
        from lilbee.data.ingest import file_hash, pipeline

        f = isolated_env / "stable.txt"
        f.write_text("stable content")
        record = self._record("stable.txt", file_hash(f), f)

        hash_calls: list[Path] = []

        def _counting_hash(path: Path) -> str:
            hash_calls.append(path)
            return file_hash(path)

        monkeypatch.setattr(pipeline, "file_hash", _counting_hash)
        plan = pipeline._plan_file_changes({"stable.txt": f}, {"stable.txt": record}, cancel=None)
        assert hash_calls == []
        assert plan.unchanged == 1
        assert plan.files_to_process == []
        assert plan.stat_backfills == []

    def test_racily_clean_mtime_at_capture_time_is_hashed(self, isolated_env, monkeypatch):
        # A same-size edit landing in the same mtime tick as the recorded
        # capture cannot be proven unseen by the stat alone; it must be hashed.
        from lilbee.data.ingest import file_hash, pipeline

        f = isolated_env / "racy.txt"
        f.write_text("racy content!!")
        record = self._record("racy.txt", file_hash(f), f)
        record["stat_captured_ns"] = f.stat().st_mtime_ns  # capture tied with mtime

        hash_calls: list[Path] = []

        def _counting_hash(path: Path) -> str:
            hash_calls.append(path)
            return file_hash(path)

        monkeypatch.setattr(pipeline, "file_hash", _counting_hash)
        plan = pipeline._plan_file_changes({"racy.txt": f}, {"racy.txt": record}, cancel=None)
        assert hash_calls == [f]
        assert plan.unchanged == 1  # content matched after the forced hash
        assert len(plan.stat_backfills) == 1  # the fresh capture re-arms the short-circuit

    def test_unknown_capture_time_is_hashed(self, isolated_env, monkeypatch):
        # A row with stat columns but no capture time (pre-capture format) cannot
        # apply the racily-clean tie-breaker; hash once and backfill.
        from lilbee.data.ingest import file_hash, pipeline

        f = isolated_env / "precapture.txt"
        f.write_text("pre-capture row")
        record = self._record("precapture.txt", file_hash(f), f)
        del record["stat_captured_ns"]

        hash_calls: list[Path] = []

        def _counting_hash(path: Path) -> str:
            hash_calls.append(path)
            return file_hash(path)

        monkeypatch.setattr(pipeline, "file_hash", _counting_hash)
        plan = pipeline._plan_file_changes(
            {"precapture.txt": f}, {"precapture.txt": record}, cancel=None
        )
        assert hash_calls == [f]
        assert plan.unchanged == 1
        assert len(plan.stat_backfills) == 1

    def test_changed_mtime_rehashes(self, isolated_env, monkeypatch):
        import os

        from lilbee.data.ingest import file_hash, pipeline

        f = isolated_env / "touched.txt"
        f.write_text("same content")
        record = self._record("touched.txt", file_hash(f), f)
        os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 1_000_000_000))

        hash_calls: list[Path] = []

        def _counting_hash(path: Path) -> str:
            hash_calls.append(path)
            return file_hash(path)

        monkeypatch.setattr(pipeline, "file_hash", _counting_hash)
        plan = pipeline._plan_file_changes({"touched.txt": f}, {"touched.txt": record}, cancel=None)
        # mtime drifted, so the hash ran; content matched, so the fresh stat
        # is queued for backfill and the file stays unchanged.
        assert hash_calls == [f]
        assert plan.unchanged == 1
        assert len(plan.stat_backfills) == 1
        assert plan.stat_backfills[0].record["filename"] == "touched.txt"
        assert plan.stat_backfills[0].stat.mtime_ns == f.stat().st_mtime_ns

    def test_missing_stat_fields_backfill(self, isolated_env):
        from lilbee.data.ingest import file_hash, pipeline

        f = isolated_env / "legacy.txt"
        f.write_text("legacy row content")
        # Legacy row: no usable stat columns.
        record = self._record("legacy.txt", file_hash(f))

        plan = pipeline._plan_file_changes({"legacy.txt": f}, {"legacy.txt": record}, cancel=None)
        assert plan.unchanged == 1
        assert len(plan.stat_backfills) == 1
        assert plan.stat_backfills[0].stat.size_bytes == f.stat().st_size

    def test_disk_stat_returns_none_on_oserror(self, isolated_env):
        from lilbee.data.ingest.pipeline import _disk_stat

        assert _disk_stat(isolated_env / "does-not-exist.txt") is None

    def test_flush_failure_without_tracker_still_fails_files(self, mock_svc):
        # _flush_writes tolerates a missing flush_failed tracker (default None).
        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.types import _IngestResult

        mock_svc.store.write_chunks_batch.side_effect = RuntimeError("disk full")
        result = _IngestResult(
            name="a.txt",
            path=Path("a.txt"),
            chunk_count=1,
            error=None,
            file_hash="h",
            records=[{"text": "x"}],
            needs_cleanup=True,
        )
        failed: dict[str, None] = {}
        pipeline._flush_writes([result], {"a.txt": None}, {}, failed, {})
        assert list(failed) == ["a.txt"]

    def test_changed_content_reprocessed_with_stat(self, isolated_env):
        from lilbee.data.ingest import pipeline

        f = isolated_env / "edited.txt"
        f.write_text("new content entirely")
        record = self._record("edited.txt", "old-hash")

        plan = pipeline._plan_file_changes({"edited.txt": f}, {"edited.txt": record}, cancel=None)
        assert [e.name for e in plan.files_to_process] == ["edited.txt"]
        entry = plan.files_to_process[0]
        assert entry.stat is not None
        assert entry.stat.size_bytes == f.stat().st_size
        assert list(plan.updated) == ["edited.txt"]

    async def test_sync_runs_planning_off_the_event_loop(self, isolated_env, mock_svc, monkeypatch):
        import threading

        from lilbee.data.ingest import pipeline, sync

        (isolated_env / "doc.txt").write_text("content")
        loop_thread = threading.get_ident()
        plan_threads: list[int] = []
        real_plan = pipeline._plan_file_changes

        def _spy_plan(*args, **kwargs):
            plan_threads.append(threading.get_ident())
            return real_plan(*args, **kwargs)

        monkeypatch.setattr(pipeline, "_plan_file_changes", _spy_plan)
        with mock.patch(
            "kreuzberg.extract_file",
            new_callable=mock.AsyncMock,
            return_value=_make_kreuzberg_result(),
        ):
            await sync(quiet=True)
        assert plan_threads
        assert all(t != loop_thread for t in plan_threads)

    async def test_sync_backfills_stats_via_store(self, isolated_env, mock_svc):
        from lilbee.data.ingest import file_hash, sync

        f = isolated_env / "legacy.txt"
        f.write_text("legacy content")
        # Tracked at the right hash but with no stat columns (legacy store row).
        mock_svc.store.upsert_source("legacy.txt", file_hash(f), 1, source_type="document")

        with mock.patch(
            "kreuzberg.extract_file",
            new_callable=mock.AsyncMock,
            return_value=_make_kreuzberg_result(),
        ):
            result = await sync(quiet=True)
        assert result.unchanged == 1
        mock_svc.store.update_source_stats.assert_called_once()
        backfills = mock_svc.store.update_source_stats.call_args.args[0]
        assert backfills[0].record["filename"] == "legacy.txt"
        assert backfills[0].stat.size_bytes == f.stat().st_size


class TestRemovableSources:
    """Only document sources whose file is gone are removable; imports persist."""

    def _src(self, filename, source_type):
        return {
            "filename": filename,
            "file_hash": "",
            "ingested_at": "",
            "chunk_count": 1,
            "source_type": source_type,
        }

    def test_keeps_imported_drops_missing_document(self):
        from pathlib import Path

        from lilbee.data.ingest.pipeline import _removable_sources
        from lilbee.data.store import SourceType

        sources = [
            self._src("gone.md", SourceType.DOCUMENT),
            self._src("present.md", SourceType.DOCUMENT),
            self._src("shared.pdf", SourceType.IMPORTED),
        ]
        disk_files = {"present.md": Path("present.md")}
        assert _removable_sources(sources, disk_files) == ["gone.md"]


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
            ("f.md", "md"),
            ("f.txt", "txt"),
            ("f.html", "html"),
            ("f.rst", "rst"),
            ("f.py", "code"),
            ("f.js", "code"),
            ("f.go", "code"),
            ("f.zip", "zip"),
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


class TestClassifyResult:
    def test_records_reason_for_failed_and_skipped_files(self):
        # The optional reasons map captures WHY a file was skipped (the exception
        # for a failure, "no text" for a zero-chunk file) so a report can show it.
        from lilbee.data.ingest.pipeline import _classify_result
        from lilbee.data.ingest.types import _IngestResult

        reasons: dict[str, str] = {}
        _classify_result(
            _IngestResult("err.pdf", Path("err.pdf"), 0, error=TimeoutError("OCR timed out")),
            {},
            {},
            {},
            {},
            reasons,
        )
        _classify_result(
            _IngestResult("blank.tiff", Path("blank.tiff"), chunk_count=0, error=None),
            {},
            {},
            {},
            {},
            reasons,
        )
        assert reasons["err.pdf"] == "TimeoutError: OCR timed out"
        assert reasons["blank.tiff"] == "no text extracted (0 chunks)"

    def test_zero_chunks_not_recorded_as_added(self):
        from lilbee.data.ingest.pipeline import _classify_result
        from lilbee.data.ingest.types import _IngestResult
        from lilbee.runtime.progress import BatchStatus

        added: dict[str, None] = {"scanned.pdf": None}
        updated: dict[str, None] = {}
        failed: dict[str, None] = {}
        skipped: dict[str, None] = {}
        result = _IngestResult("scanned.pdf", Path("scanned.pdf"), chunk_count=0, error=None)
        assert _classify_result(result, added, updated, failed, skipped) is BatchStatus.SKIPPED
        assert "scanned.pdf" not in added
        assert "scanned.pdf" not in failed
        assert "scanned.pdf" in skipped

    def test_zero_chunks_not_recorded_as_updated(self):
        from lilbee.data.ingest.pipeline import _classify_result
        from lilbee.data.ingest.types import _IngestResult
        from lilbee.runtime.progress import BatchStatus

        added: dict[str, None] = {}
        updated: dict[str, None] = {"scanned.pdf": None}
        failed: dict[str, None] = {}
        skipped: dict[str, None] = {}
        result = _IngestResult("scanned.pdf", Path("scanned.pdf"), chunk_count=0, error=None)
        assert _classify_result(result, added, updated, failed, skipped) is BatchStatus.SKIPPED
        assert "scanned.pdf" not in updated
        assert "scanned.pdf" not in failed
        assert "scanned.pdf" in skipped

    def test_zero_chunks_with_page_texts_persists_but_counts_skipped(self):
        # Whitespace-only OCR: no searchable chunks, so it is reported as skipped
        # , but the pages and source row must still persist (it stops
        # replanning), so the status stays INGESTED for the batched flush.
        from lilbee.data.ingest.pipeline import _classify_result
        from lilbee.data.ingest.types import _IngestResult
        from lilbee.runtime.progress import BatchStatus

        added: dict[str, None] = {"blank.pdf": None}
        skipped: dict[str, None] = {}
        result = _IngestResult(
            "blank.pdf",
            Path("blank.pdf"),
            chunk_count=0,
            error=None,
            page_texts=[{"source": "blank.pdf", "page": 1, "text": " ", "content_type": "pdf"}],
        )
        assert _classify_result(result, added, {}, {}, skipped) is BatchStatus.INGESTED
        assert "blank.pdf" not in added  # not counted as added: search can't see it
        assert "blank.pdf" in skipped

    def test_nonzero_chunks_stay_for_flush(self):
        # A successful file is reported INGESTED and left in added; persistence
        # (the source upsert) is the batched flush's job, not classification's.
        from lilbee.data.ingest.pipeline import _classify_result
        from lilbee.data.ingest.types import _IngestResult
        from lilbee.runtime.progress import BatchStatus

        added: dict[str, None] = {"doc.pdf": None}
        updated: dict[str, None] = {}
        failed: dict[str, None] = {}
        skipped: dict[str, None] = {}
        result = _IngestResult("doc.pdf", Path("doc.pdf"), chunk_count=5, error=None)
        assert _classify_result(result, added, updated, failed, skipped) is BatchStatus.INGESTED
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

        from lilbee.data.ingest.pipeline import _classify_result
        from lilbee.data.ingest.types import _IngestResult

        added: dict[str, None] = {}
        updated: dict[str, None] = {}
        failed: dict[str, None] = {}
        skipped: dict[str, None] = {}
        err = RuntimeError("embedder bogus:bogus not installed")
        result = _IngestResult("qa-fail.md", Path("qa-fail.md"), chunk_count=0, error=err)
        with caplog.at_level(logging.DEBUG, logger="lilbee.data.ingest.pipeline"):
            _classify_result(result, added, updated, failed, skipped)
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
        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.ingest.types import _IngestResult
        from lilbee.runtime.progress import BatchProgressEvent, EventType

        captured: list[tuple[EventType, BatchProgressEvent]] = []

        def on_progress(event_type: object, data: object) -> None:
            if event_type == EventType.BATCH_PROGRESS and isinstance(data, BatchProgressEvent):
                captured.append((event_type, data))

        async def _zero_chunk_result() -> _IngestResult:
            return _IngestResult("scan.pdf", Path("scan.pdf"), chunk_count=0, error=None)

        added: dict[str, None] = {}
        updated: dict[str, None] = {}
        failed: dict[str, None] = {}
        skipped: dict[str, None] = {}
        await _collect_results(
            iter([_zero_chunk_result()]),
            1,
            added,
            updated,
            failed,
            skipped,
            window=2,
            on_progress=on_progress,
        )
        assert len(captured) == 1
        assert captured[0][1].status == "skipped"
        assert "scan.pdf" in skipped

    async def test_zero_text_update_purges_prior_index_entry(self, mock_svc):
        # An already-indexed file edited to extract to nothing must have its old
        # Chunks/source row removed, not left orphaned in search.
        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.ingest.types import _IngestResult

        async def _emptied() -> _IngestResult:
            return _IngestResult(
                "notes.md", Path("notes.md"), chunk_count=0, error=None, needs_cleanup=True
            )

        await _collect_results(iter([_emptied()]), 1, {}, {}, {}, {}, window=2)
        mock_svc.store.remove_documents.assert_called_once_with(["notes.md"])

    async def test_zero_text_without_cleanup_does_not_purge(self, mock_svc):
        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.ingest.types import _IngestResult

        async def _emptied() -> _IngestResult:
            return _IngestResult(
                "fresh.md", Path("fresh.md"), chunk_count=0, error=None, needs_cleanup=False
            )

        await _collect_results(iter([_emptied()]), 1, {}, {}, {}, {}, window=2)
        mock_svc.store.remove_documents.assert_not_called()

    def test_purge_emptied_sources_noop_on_empty(self, mock_svc):
        from lilbee.data.ingest.pipeline import _purge_emptied_sources

        _purge_emptied_sources([])
        mock_svc.store.remove_documents.assert_not_called()

    async def test_error_cancels_still_running_siblings(self):
        """When one task raises, _collect_results' finally cancels the in-flight ones."""
        import asyncio

        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.ingest.types import _IngestResult

        started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def _boom() -> _IngestResult:
            await started.wait()
            raise RuntimeError("ingest blew up")

        async def _never_finishes() -> _IngestResult:
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise
            raise AssertionError("sibling task should have been cancelled")

        added: dict[str, None] = {}
        failed: dict[str, None] = {}
        skipped: dict[str, None] = {}
        with pytest.raises(RuntimeError, match="ingest blew up"):
            await _collect_results(
                iter([_boom(), _never_finishes()]), 2, added, {}, failed, skipped, window=2
            )
        assert sibling_cancelled.is_set()

    async def test_finally_path_flush_failure_still_cancels_siblings(self, monkeypatch):
        """A flush raising on the way out must not strand the in-flight siblings."""
        import asyncio

        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.types import _IngestResult

        started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def _boom() -> _IngestResult:
            await started.wait()
            raise RuntimeError("ingest blew up")

        async def _never_finishes() -> _IngestResult:
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise
            raise AssertionError("sibling task should have been cancelled")

        def _exploding_flush(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("flush blew up")

        monkeypatch.setattr(pipeline, "_flush_writes", _exploding_flush)
        with pytest.raises(RuntimeError, match="flush blew up"):
            await pipeline._collect_results(
                iter([_boom(), _never_finishes()]), 2, {}, {}, {}, {}, window=2
            )
        assert sibling_cancelled.is_set()


class TestTaskWindow:
    """The ingest pump keeps at most ``window`` tasks alive at once."""

    async def test_in_flight_high_water_stays_at_window(self):
        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.ingest.types import _IngestResult

        total = 50
        window = 4
        created = 0
        finished = 0
        high_water = 0

        async def _one(i: int) -> _IngestResult:
            nonlocal finished, high_water
            high_water = max(high_water, created - finished)
            await asyncio.sleep(0)
            finished += 1
            return _IngestResult(f"f{i}.txt", Path(f"f{i}.txt"), chunk_count=0, error=None)

        def _pending():
            nonlocal created
            for i in range(total):
                created += 1
                yield _one(i)

        skipped: dict[str, None] = {}
        await _collect_results(_pending(), total, {}, {}, {}, skipped, window=window)
        assert len(skipped) == total
        assert high_water <= window
        assert created == total

    async def test_cancel_mid_stream_flushes_buffer_and_stops(self, mock_svc):
        # Files completed before the cancel are flushed in the finally; files
        # never pulled from the iterator are never started.
        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.ingest.types import _IngestResult

        created: list[int] = []

        async def _ok(i: int) -> _IngestResult:
            return _IngestResult(
                f"f{i}.txt",
                Path(f"f{i}.txt"),
                chunk_count=1,
                error=None,
                file_hash="h",
                records=[{"text": "x"}],
            )

        async def _cancelled(i: int) -> _IngestResult:
            raise asyncio.CancelledError

        def _pending():
            for i in range(10):
                created.append(i)
                yield _ok(i) if i == 0 else _cancelled(i)

        added: dict[str, None] = {"f0.txt": None}
        with pytest.raises(asyncio.CancelledError):
            await _collect_results(_pending(), 10, added, {}, {}, {}, window=1)
        # The window kept task creation bounded: only f0 and the cancelling f1 started.
        assert created == [0, 1]
        # The completed file's chunks were flushed on the way out.
        items = mock_svc.store.write_chunks_batch.call_args.args[0]
        assert [it.source for it in items] == ["f0.txt"]


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
            ("data.csv", "csv"),
            ("data.tsv", "tsv"),
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
        assert config.get("pages") is not None

    def test_paginated_no_markdown_output(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.PAGINATED)
        assert getattr(config, "output_format", None) != "markdown"

    def test_markdown_has_no_page_config(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.MARKDOWN)
        assert config.get("pages") is None

    def test_markdown_has_chunking(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.MARKDOWN)
        assert config.get("chunking") is not None

    def test_markdown_sets_output_format(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.MARKDOWN)
        assert config["output_format"] == "markdown"

    def test_paginated_has_tesseract_ocr_backend(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.PAGINATED)
        assert config.get("pages") is not None
        assert config.get("ocr") is not None
        # No vision model configured -> kreuzberg's tesseract backend OCRs scanned pages.
        assert config["ocr"].backend == "tesseract"
        # kreuzberg 5.x errors on image OCR with an empty language list; lilbee must
        # set an explicit default to preserve 4.x behavior (regression guard).
        assert config["ocr"].language == ["eng"]

    def test_tesseract_ocr_language_from_config(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "ocr_language", ["deu", "fra"])
        config = extraction_config(ExtractMode.PAGINATED)
        assert config["ocr"].language == ["deu", "fra"]

    @pytest.mark.parametrize(
        "content_type, expected_mode_name",
        [
            ("pdf", "PAGINATED"),
            ("text", "MARKDOWN"),
            ("docx", "MARKDOWN"),
            ("xlsx", "MARKDOWN"),
            ("pptx", "MARKDOWN"),
            ("epub", "MARKDOWN"),
            ("image", "PAGINATED"),
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
            assert config["chunking"].chunker_type == "semantic"
            assert config["chunking"].topic_threshold == pytest.approx(0.42, abs=1e-5)


class TestClassifyKreuzbergParityFormats:
    """Formats kreuzberg extracts that the map must not silently drop."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("anim.gif", "image"),
            ("scan.jp2", "image"),
            ("scan.j2k", "image"),
            ("scan.j2c", "image"),
            ("scan.jpx", "image"),
            ("page.htm", "htm"),
            ("memo.doc", "doc"),
            ("deck.ppt", "ppt"),
            ("ledger.xls", "xls"),
            ("letter.rtf", "rtf"),
            ("memo.odt", "odt"),
            ("ledger.ods", "ods"),
            ("mail.eml", "eml"),
            ("mail.msg", "msg"),
            ("table.dbf", "dbf"),
            # Containers are supported too (kreuzberg extracts their combined text).
            ("archive.pst", "pst"),
            ("archive.7z", "7z"),
        ],
    )
    def test_classify(self, filename, expected):
        from lilbee.data.ingest import classify_file

        assert classify_file(Path(filename)) == expected


class TestClassifyStructuredFormats:
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("data.xml", "xml"),
            ("data.json", "json"),
            ("data.jsonl", "jsonl"),
            ("config.yaml", "yaml"),
            ("config.yml", "yml"),
            ("data.csv", "csv"),
        ],
    )
    def test_classify(self, filename, expected):
        from lilbee.data.ingest import classify_file

        assert classify_file(Path(filename)) == expected


@mock.patch(
    "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
)
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


class TestOcrOverrideContextVar:
    """Per-request OCR overrides use a ContextVar, never a global cfg mutation."""

    def test_override_does_not_mutate_global_cfg(self, isolated_env):
        from lilbee.data.ingest.extract import ocr_override

        cfg.enable_ocr = False
        cfg.ocr_timeout = 11.0
        with ocr_override(enable_ocr=True, ocr_timeout=99.0):
            pass
        assert cfg.enable_ocr is False
        assert cfg.ocr_timeout == 11.0

    def test_effective_values_reflect_override_then_revert(self, isolated_env):
        from lilbee.data.ingest.extract import (
            _effective_enable_ocr,
            _effective_ocr_timeout,
            ocr_override,
        )

        cfg.enable_ocr = False
        cfg.ocr_timeout = 11.0
        with ocr_override(enable_ocr=True, ocr_timeout=99.0):
            assert _effective_enable_ocr() is True
            assert _effective_ocr_timeout() == 99.0
        assert _effective_enable_ocr() is False
        assert _effective_ocr_timeout() == 11.0

    def test_none_arguments_keep_cfg_defaults(self, isolated_env):
        from lilbee.data.ingest.extract import _effective_enable_ocr, ocr_override

        cfg.enable_ocr = True
        with ocr_override(enable_ocr=None, ocr_timeout=None):
            assert _effective_enable_ocr() is True

    def test_overrides_isolated_across_contexts(self, isolated_env):
        # Two copied contexts must each see only their own override; this is the
        # concurrency guarantee that a global cfg mutation could not give.
        import contextvars

        from lilbee.data.ingest.extract import _effective_ocr_timeout, ocr_override

        cfg.ocr_timeout = 5.0
        seen: dict[str, float] = {}

        def run_with(value: float, key: str) -> None:
            with ocr_override(ocr_timeout=value):
                seen[key] = _effective_ocr_timeout()

        contextvars.copy_context().run(run_with, 30.0, "a")
        contextvars.copy_context().run(run_with, 70.0, "b")
        assert seen == {"a": 30.0, "b": 70.0}
        assert _effective_ocr_timeout() == 5.0  # parent context untouched

    def test_temporary_ocr_config_delegates_without_global_mutation(self, isolated_env):
        from lilbee.app.ingest import temporary_ocr_config
        from lilbee.data.ingest.extract import _effective_ocr_timeout

        cfg.ocr_timeout = 8.0
        with temporary_ocr_config(ocr_timeout=42.0):
            assert _effective_ocr_timeout() == 42.0
            assert cfg.ocr_timeout == 8.0
        assert _effective_ocr_timeout() == 8.0

    async def test_override_propagates_into_to_thread_worker(self, isolated_env):
        # The fix relies on asyncio.to_thread copying the calling context, which
        # is how the override reaches the extract worker that actually OCRs.
        import asyncio

        from lilbee.data.ingest.extract import _effective_ocr_timeout, ocr_override

        cfg.ocr_timeout = 5.0
        with ocr_override(ocr_timeout=88.0):
            seen = await asyncio.to_thread(_effective_ocr_timeout)
        assert seen == 88.0
        assert await asyncio.to_thread(_effective_ocr_timeout) == 5.0


class TestPhaseProgressCallback:
    """The non-quiet Rich bar advances once per file, so a single multi-page file
    would freeze at "0/1" through its whole OCR + embed phase. The phase-progress
    wrapper drives the bar's description off EXTRACT (OCR page) and EMBED (chunk)
    events so the row visibly moves while one file is being worked."""

    def test_extract_event_updates_bar_description(self):
        from lilbee.data.ingest.pipeline import _phase_progress_callback
        from lilbee.runtime.progress import EventType, ExtractEvent

        progress = MagicMock()
        cb = _phase_progress_callback(progress, "task-1", lambda *_: None)
        cb(EventType.EXTRACT, ExtractEvent(file="scan.pdf", page=3, total_pages=12))
        desc = progress.update.call_args.kwargs["description"]
        assert "scan.pdf" in desc
        assert "3/12" in desc

    def test_embed_event_updates_bar_description(self):
        from lilbee.data.ingest.pipeline import _phase_progress_callback
        from lilbee.runtime.progress import EmbedEvent, EventType

        progress = MagicMock()
        cb = _phase_progress_callback(progress, "task-1", lambda *_: None)
        cb(EventType.EMBED, EmbedEvent(file="scan.pdf", chunk=8, total_chunks=40))
        desc = progress.update.call_args.kwargs["description"]
        assert "Embedding" in desc
        assert "scan.pdf" in desc
        assert "8/40" in desc

    def test_events_forward_to_chained_callback(self):
        from lilbee.data.ingest.pipeline import _phase_progress_callback
        from lilbee.runtime.progress import EmbedEvent, EventType, ExtractEvent

        seen: list[object] = []
        cb = _phase_progress_callback(MagicMock(), "t", lambda et, d: seen.append((et, d)))
        extract = ExtractEvent(file="x", page=1, total_pages=2)
        embed = EmbedEvent(file="x", chunk=1, total_chunks=2)
        cb(EventType.EXTRACT, extract)
        cb(EventType.EMBED, embed)
        assert seen == [(EventType.EXTRACT, extract), (EventType.EMBED, embed)]

    def test_unrelated_event_does_not_touch_bar(self):
        from lilbee.data.ingest.pipeline import _phase_progress_callback
        from lilbee.runtime.progress import EventType, FileStartEvent

        progress = MagicMock()
        seen: list[object] = []
        cb = _phase_progress_callback(progress, "t", lambda et, d: seen.append(et))
        start = FileStartEvent(file="x", total_files=1, current_file=1)
        cb(EventType.FILE_START, start)
        progress.update.assert_not_called()  # description only changes for EXTRACT/EMBED
        assert seen == [EventType.FILE_START]  # still forwarded to the chain


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


class TestPageTextAccumulator:
    """`page_texts_out` captures clean per-page text for the export dataset."""

    @mock.patch("kreuzberg.extract_file", new_callable=mock.AsyncMock)
    async def test_pdf_pages_captured(self, mock_kf, isolated_env):
        mock_kf.return_value = _make_kreuzberg_result(num_chunks=2, has_pages=True)
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "test.pdf"
        f.write_bytes(b"fake")
        pages: list = []
        await ingest_document(f, "test.pdf", "pdf", page_texts_out=pages)
        assert [p["page"] for p in pages] == [1, 2]
        assert all(p["content_type"] == "pdf" for p in pages)

    @mock.patch("kreuzberg.extract_file", new_callable=mock.AsyncMock)
    async def test_non_paginated_doc_captured_as_page_zero(self, mock_kf, isolated_env):
        mock_kf.return_value = _make_kreuzberg_result(text="Plain body. " * 10, has_pages=False)
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "note.txt"
        f.write_text("body")
        pages: list = []
        await ingest_document(f, "note.txt", "text", page_texts_out=pages)
        assert len(pages) == 1
        assert pages[0]["page"] == 0
        assert pages[0]["content_type"] == "text"
        assert "Plain body." in pages[0]["text"]

    async def test_markdown_captured_as_page_zero(self, isolated_env):
        from lilbee.data.ingest import ingest_markdown

        md = isolated_env / "doc.md"
        md.write_text("# Title\n\nSome markdown body text here.")
        pages: list = []
        await ingest_markdown(md, "doc.md", page_texts_out=pages)
        assert len(pages) == 1
        assert pages[0]["page"] == 0
        assert "markdown body" in pages[0]["text"]

    @mock.patch("kreuzberg.extract_file", new_callable=mock.AsyncMock)
    async def test_ocr_pages_captured(self, mock_kf, isolated_env, mock_svc):
        # kreuzberg OCRs scanned pages in-pass; the OCR'd text arrives in result.pages.
        cfg.enable_ocr = True
        cfg.vision_model = ""
        mock_kf.return_value = _make_kreuzberg_result(
            text="OCR page text. " * 10, num_chunks=2, has_pages=True
        )
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "scanned.pdf"
        f.write_bytes(b"fake pdf")
        pages: list = []
        await ingest_document(f, "scanned.pdf", "pdf", page_texts_out=pages)
        assert {p["page"] for p in pages} == {1, 2}
        assert all(p["content_type"] == "pdf" for p in pages)


class TestIngestDocumentEdgeCases:
    async def test_empty_extraction_returns_empty(self, isolated_env):
        """Structured formats now go through kreuzberg: empty result yields no chunks."""
        from lilbee.data.ingest import ingest_document

        empty_result = mock.MagicMock(chunks=[])
        mock_extract = mock.AsyncMock(return_value=empty_result)
        with mock.patch("kreuzberg.extract_file", mock_extract):
            result = await ingest_document(isolated_env / "e.xml", "e.xml", "xml")
        assert result == []

    async def test_no_chunks_returns_empty(self, isolated_env):
        from lilbee.data.ingest import ingest_document

        no_chunks_result = mock.MagicMock(chunks=[])
        mock_extract = mock.AsyncMock(return_value=no_chunks_result)
        with mock.patch("kreuzberg.extract_file", mock_extract):
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
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
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
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
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
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
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
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_concept_write_failure_does_not_fail_files(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """A failing batched concept write is logged; the files still ingest."""
        from lilbee.data.store import ConceptRecords

        cfg.concept_graph = True
        (isolated_env / "concept_write.txt").write_text("Content for write test.")

        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts_batch.return_value = [["test"]]
        mock_svc.concepts.build_concept_records.return_value = ConceptRecords([], [], [])
        mock_svc.concepts.write_concept_records.side_effect = RuntimeError("disk full")
        from lilbee.data.ingest import sync

        result = await sync(quiet=True)
        assert "concept_write.txt" in result.added
        assert result.failed == []

    @mock.patch(
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
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
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
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
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
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
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
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
        "kreuzberg.extract_file", new_callable=mock.AsyncMock, return_value=_make_kreuzberg_result()
    )
    async def test_concepts_unavailable_skips_indexing(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """When concepts_available() returns False, _build_concept_records is a no-op."""
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


class TestRemoveDocumentsDurably:
    def test_writes_skip_marker_for_kept_file(self, isolated_env, mock_svc):
        """A durable delete keeps the file but skip-marks it so sync won't re-ingest."""
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import load_skip_markers, load_skip_reasons
        from lilbee.data.store.types import RemoveResult

        doc = isolated_env / "keep.txt"
        doc.write_text("content")
        mock_svc.store.remove_documents.side_effect = None
        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=["keep.txt"], not_found=[]
        )

        remove_documents_durably(["keep.txt"])

        assert doc.exists()  # non-destructive: file stays on disk
        assert load_skip_markers(cfg.data_root)["keep.txt"] == file_hash(doc)
        assert "keep.txt" in load_skip_reasons(cfg.data_root)

    def test_no_marker_when_nothing_removed(self, isolated_env, mock_svc):
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest.skip_marker import load_skip_markers
        from lilbee.data.store.types import RemoveResult

        mock_svc.store.remove_documents.side_effect = None
        mock_svc.store.remove_documents.return_value = RemoveResult(removed=[], not_found=["gone"])
        remove_documents_durably(["gone"])
        assert load_skip_markers(cfg.data_root) == {}


class TestOcrConfigSelection:
    """extraction_config picks the OCR backend from enable_ocr + vision_model."""

    def test_disabled_when_enable_ocr_false(self, isolated_env):
        from lilbee.data.ingest import ExtractMode, extraction_config

        cfg.enable_ocr = False
        config = extraction_config(ExtractMode.PAGINATED)
        assert config["ocr"].enabled is False

    def test_vision_backend_with_token_when_model_set(self, isolated_env):
        from lilbee.data.ingest import ExtractMode, extraction_config

        cfg.enable_ocr = None
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        config = extraction_config(ExtractMode.PAGINATED, ocr_token="tok-123")
        assert config["ocr"].backend == "lilbee-vision"
        assert "tok-123" in config["ocr"].backend_options


class TestChunkAndEmbedPagesEmpty:
    async def test_empty_page_texts_returns_empty(self):
        from lilbee.data.ingest.extract import chunk_and_embed_pages
        from lilbee.runtime.progress import noop_callback

        assert await chunk_and_embed_pages([], "s", "pdf", noop_callback) == []

    async def test_whitespace_pages_yield_no_chunks(self, mock_svc):
        from lilbee.data.ingest.extract import chunk_and_embed_pages
        from lilbee.runtime.progress import noop_callback

        assert await chunk_and_embed_pages([(1, "   ")], "s", "pdf", noop_callback) == []


class TestIngestDocumentOcrPath:
    @mock.patch("kreuzberg.extract_file", new_callable=mock.AsyncMock)
    async def test_empty_pdf_warns_no_usable_text(self, mock_kf, isolated_env, mock_svc, caplog):
        mock_kf.return_value = mock.MagicMock(chunks=[])
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        result = await ingest_document(f, "scan.pdf", "pdf")
        assert result == []
        assert "no usable text" in caplog.text

    @mock.patch("kreuzberg.extract_file", new_callable=mock.AsyncMock)
    async def test_streams_per_page_progress_ticks(self, mock_kf, isolated_env, mock_svc):
        import json

        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        result_obj = _make_kreuzberg_result(num_chunks=1, has_pages=True)

        async def fake_extract(path, *, config):
            # Simulate kreuzberg calling the registered backend once per scanned page.
            from lilbee.data.ingest.vision_ocr_backend import ocr_requests

            token = json.loads(config["ocr"].backend_options)["req"]
            ctx = ocr_requests.get(token)
            ctx.on_page()
            ctx.on_page()
            return result_obj

        mock_kf.side_effect = fake_extract
        from lilbee.data.ingest import ingest_document

        seen: list[int] = []

        def on_prog(_et, ev):
            seen.append(ev.page)

        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        await ingest_document(f, "scan.pdf", "pdf", on_progress=on_prog)
        assert 1 in seen and 2 in seen  # two per-page running-count ticks
