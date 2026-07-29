"""Tests for the document sync engine (mocked: no live server needed)."""

import asyncio
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from xberg import Metadata

import lilbee.app.services as svc_mod
from lilbee.core.config import cfg
from tests.conftest import make_pdf


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
    embedder.embed.side_effect = lambda text, **kw: np.full(768, 0.1, dtype=np.float32)
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


def _feed(coros):
    """A _ResultFeed over one already-planned shard of file coroutines."""
    from lilbee.data.ingest.pipeline import _ResultFeed
    from tests.conftest import one_shard

    return _ResultFeed(one_shard(coros))


def _lazy_feed(coros):
    """A _ResultFeed that plans one file at a time, as a live plan stream does."""
    from lilbee.data.ingest.pipeline import _ResultFeed

    async def _shards():
        for coro in coros:
            yield [coro]

    return _ResultFeed(_shards())


def _real_ingest_result(name, *, file_hash, page_text="page one of a.pdf", stat=None):
    """A store-schema _IngestResult with one chunk and one page-text row."""
    from lilbee.data.store import PageTextRecord
    from lilbee.data.types import _IngestResult

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


def _make_xberg_result(
    text="Some extracted text. " * 20,
    num_chunks=1,
    has_pages=False,
    document=None,
    tables=None,
    metadata=None,
):
    """Build a mock xberg ExtractionResult."""
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
    result.tables = tables if tables is not None else []
    result.metadata = metadata if metadata is not None else Metadata()
    result.pages = (
        [mock.MagicMock(page_number=i + 1, content=chunks[i].content) for i in range(num_chunks)]
        if has_pages
        else []
    )
    return result


def _make_table(markdown="| h1 | h2 |\n|---|---|\n| a | b |", page_number=1):
    """Build a mock xberg Table carrying its markdown serialization."""
    table = mock.MagicMock()
    table.markdown = markdown
    table.page_number = page_number
    return table


def _make_empty_result():
    """Build a mock xberg ExtractionResult with no chunks."""
    result = mock.MagicMock()
    result.chunks = []
    result.content = ""
    result.document = None
    result.tables = []
    result.pages = []
    result.metadata = Metadata()
    return result


def test_reconcile_missing_flags_only_silent_drops():
    from pathlib import Path

    from lilbee.data.ingest.pipeline import _reconcile_missing

    disk = {n: Path(n) for n in ("a.pdf", "b.pdf", "c.pdf", "d.pdf")}
    # a indexed, b failed, c skipped, d dropped with no signal.
    missing = _reconcile_missing(disk, [{"filename": "a.pdf"}], failed=["b.pdf"], skipped=["c.pdf"])
    assert missing == ["d.pdf"]


@mock.patch(
    "lilbee.data.extract.xberg.aextract_document",
    new_callable=mock.AsyncMock,
    return_value=_make_xberg_result(),
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

    async def test_pool_is_sized_on_unindexed_files_not_the_whole_corpus(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        """An incremental sync over an indexed corpus must not reach for a pool.

        The streamed plan has no total up front, so the pool decision is sized on
        a pre-diff proxy. Sizing it on every file on disk would spawn workers for
        a two-file sync of a large indexed corpus.
        """
        from lilbee.data.ingest import pipeline as pipeline_mod
        from lilbee.data.ingest import sync

        for i in range(3):
            (isolated_env / f"c{i}.txt").write_text(f"body {i}")
        await sync()  # index them, so the next pass sees them as unchanged

        sized_on: list[int] = []
        monkeypatch.setattr(
            pipeline_mod,
            "resolve_process_count",
            lambda count: sized_on.append(count) or 1,
        )
        (isolated_env / "new.txt").write_text("only this one is new")
        await sync()

        # One new file, three already indexed: the corpus is 4.
        assert sized_on == [1]

    async def test_sync_uses_batch_extraction_when_enabled(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        """With the toggle on, sync extracts through extract_batch, not single extract."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync

        monkeypatch.setattr(cfg, "batch_extraction", True)
        (isolated_env / "d.txt").write_text("Hello world. Batch extraction test document.")

        async def fake_batch(inputs, _config):
            res = mock.MagicMock()
            res.results = [_make_xberg_result() for _ in inputs]
            res.errors = []
            return res

        with mock.patch("xberg.extract_batch", side_effect=fake_batch) as mock_batch:
            result = await sync()

        assert "d.txt" in result.added
        mock_batch.assert_called()
        mock_extract_file.assert_not_called()

    async def test_moved_file_relocates_without_reingest(self, mock_extract_file, isolated_env):
        import shutil

        from lilbee.app.services import get_services
        from lilbee.data.ingest import sync

        (isolated_env / "a.txt").write_text("Hello world. This document will move.")
        first = await sync()
        assert "a.txt" in first.added

        # Move it: same content, new key. Sync must relocate, not re-ingest.
        (isolated_env / "sub").mkdir()
        shutil.move(str(isolated_env / "a.txt"), str(isolated_env / "sub" / "a.txt"))
        mock_extract_file.reset_mock()
        store = get_services().store
        store.relocate_sources.reset_mock()

        second = await sync()

        assert second.relocated == ["sub/a.txt"]
        assert second.added == []
        assert second.removed == []  # a move is not a removal
        mock_extract_file.assert_not_called()  # no re-extraction/re-embedding
        store.relocate_sources.assert_called_once()
        moves = store.relocate_sources.call_args.args[0]
        assert [(old, new) for old, new, _stat in moves] == [("a.txt", "sub/a.txt")]

    async def test_adaptive_mode_ingests_and_stops_controller(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        (isolated_env / "adaptive.txt").write_text("Adaptive-mode content for ingest.")
        from lilbee.data.ingest import pipeline, sync
        from lilbee.data.ingest.adaptive import Signals

        monkeypatch.setenv("LILBEE_INGEST_CONCURRENCY", "adaptive-conservative")
        # A fake fleet + sampler so the adaptive path engages without real GPU probing.
        monkeypatch.setattr(pipeline, "enumerate_fleet_devices", lambda: [object()])
        monkeypatch.setattr(
            pipeline,
            "make_signal_sampler",
            lambda _devices: lambda t: Signals(t, 50.0, 60.0, 50.0, 0.5),
        )
        result = await sync(quiet=True)
        assert "adaptive.txt" in result.added

    async def test_spaced_filename_ingests_without_silent_drop(
        self, mock_extract_file, isolated_env, caplog
    ):
        # Regression for the silent drop of files with spaces in their names.
        (isolated_env / "Request No. 1.txt").write_text("Maintenance request approved, page one.")
        import logging

        from lilbee.data.ingest import sync

        with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"):
            result = await sync(quiet=True)
        assert "Request No. 1.txt" in result.added
        assert not any("reconciliation" in r.getMessage().lower() for r in caplog.records)

    async def test_reconciliation_warns_on_silent_drop(
        self, mock_extract_file, isolated_env, monkeypatch, caplog
    ):
        # A file discovered but never indexed, failed, or skipped is a silent drop.
        (isolated_env / "ghost.txt").write_text("This file gets dropped by a broken ingest stage.")
        import logging

        from lilbee.data.ingest import pipeline, sync

        async def _noop_ingest(*_a, **_k):
            return None

        monkeypatch.setattr(pipeline, "ingest_stream", _noop_ingest)
        with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"):
            await sync(quiet=True)
        assert any(
            "reconciliation" in r.getMessage().lower() and "ghost.txt" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.parametrize("auto_update", [True, False])
    async def test_wiki_hook_runs_only_when_auto_update_is_on(
        self, mock_extract_file, isolated_env, monkeypatch, auto_update
    ):
        """Enabling the wiki never generates by itself; auto-update is the opt-in."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync

        (isolated_env / "wikified.txt").write_text("Content the wiki hook would summarize.")
        monkeypatch.setattr(cfg, "wiki", True)
        monkeypatch.setattr(cfg, "wiki_auto_update", auto_update)
        hook = mock.AsyncMock()
        monkeypatch.setattr("lilbee.wiki.ingest.incremental_update", hook)

        await sync(quiet=True)

        assert hook.called is auto_update

    @pytest.mark.parametrize("auto_update", [False, True], ids=["off", "on"])
    async def test_index_refresh_runs_regardless_of_auto_update(
        self, mock_extract_file, isolated_env, monkeypatch, auto_update
    ):
        """The index spends no LLM call and is what lets a page appear in the
        browse tree as soon as its document lands, so it is not gated on the
        setting that governs generation."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync

        (isolated_env / "indexed.txt").write_text("Content the index would name entities in.")
        monkeypatch.setattr(cfg, "wiki", True)
        monkeypatch.setattr(cfg, "wiki_auto_update", auto_update)
        monkeypatch.setattr("lilbee.wiki.ingest.incremental_update", mock.AsyncMock())
        # A real signature, not a MagicMock: the hook calls this through
        # to_ingest_thread, which forwards its arguments, and a permissive mock
        # accepts an arity the real function rejects.
        calls: list[tuple] = []

        def fake_refresh(store, config=None, *, sources=None):
            calls.append((store, config, sources))
            return {}

        monkeypatch.setattr("lilbee.wiki.stubs.refresh_stub_index", fake_refresh)

        await sync(quiet=True)

        assert len(calls) == 1
        assert calls[0][2] is not None

    async def test_index_refresh_is_skipped_when_the_wiki_is_off(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync

        (isolated_env / "unindexed.txt").write_text("Content.")
        monkeypatch.setattr(cfg, "wiki", False)
        calls: list[tuple] = []

        def fake_refresh(store, config=None, *, sources=None):
            calls.append((store, config, sources))
            return {}

        monkeypatch.setattr("lilbee.wiki.stubs.refresh_stub_index", fake_refresh)

        await sync(quiet=True)

        assert calls == []

    async def test_a_failing_index_refresh_does_not_stop_the_sync(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        """The ingest already succeeded; a wiki failure must not swallow it."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync

        (isolated_env / "resilient.txt").write_text("Content.")
        monkeypatch.setattr(cfg, "wiki", True)
        monkeypatch.setattr(cfg, "wiki_auto_update", False)
        monkeypatch.setattr(
            "lilbee.wiki.stubs.refresh_stub_index",
            mock.MagicMock(side_effect=RuntimeError("spacy exploded")),
        )

        result = await sync(quiet=True)

        assert "resilient.txt" in result.added

    async def test_wiki_hook_receives_the_config_the_gate_read(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        """The auto-update gate and the regeneration must consult one config."""
        from lilbee.core.config import active_config, cfg
        from lilbee.data.ingest import sync

        (isolated_env / "wikified.txt").write_text("Content the wiki hook would summarize.")
        monkeypatch.setattr(cfg, "wiki", True)
        monkeypatch.setattr(cfg, "wiki_auto_update", True)
        hook = mock.AsyncMock()
        monkeypatch.setattr("lilbee.wiki.ingest.incremental_update", hook)

        await sync(quiet=True)

        assert hook.call_args.args[1] is active_config()

    async def test_wiki_failure_does_not_skip_post_ingest_verification(
        self, mock_extract_file, isolated_env, monkeypatch, caplog
    ):
        """A wiki exception must not abort the entity pass or the silent-drop guard."""
        import logging

        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync

        (isolated_env / "wikified.txt").write_text("Content the wiki hook chokes on.")
        monkeypatch.setattr(cfg, "wiki", True)
        monkeypatch.setattr(cfg, "wiki_auto_update", True)
        monkeypatch.setattr(
            "lilbee.wiki.ingest.incremental_update",
            mock.AsyncMock(side_effect=RuntimeError("embedder down")),
        )
        entities = mock.MagicMock()
        monkeypatch.setattr("lilbee.retrieval.entities.lifecycle.ensure_entities", entities)

        with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"):
            result = await sync(quiet=True)

        assert "wikified.txt" in result.added
        assert any("Wiki auto-update failed" in r.getMessage() for r in caplog.records)
        entities.assert_called_once()

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

    async def test_deleted_file_stays_indexed(self, mock_extract_file, isolated_env):
        # A vanished file is a dead path-link, not a removal: sync leaves it in
        # the index (searchable), and the user only discovers it is gone when
        # they try to open it.
        f = isolated_env / "temp.txt"
        f.write_text("Temporary")
        from lilbee.app.services import get_services
        from lilbee.data.ingest import sync

        await sync()
        f.unlink()
        result = await sync()
        assert result.removed == []
        assert "temp.txt" in {s["filename"] for s in get_services().store.get_sources()}

    async def test_registered_root_reindexes_edited_file(self, mock_extract_file, isolated_env):
        # Editing a file inside a registered root re-ingests it in place: it is
        # reported updated (not added) and the store's recorded hash reflects the
        # edit, proving a real reindex rather than a no-op.
        from lilbee.app.services import get_services
        from lilbee.core.config import cfg
        from lilbee.data.ingest import file_hash, sync

        corpus = isolated_env.parent / "corpus"
        corpus.mkdir()
        (corpus / "f.txt").write_text("first version of the content")
        cfg.linked_roots = {"corpus": str(corpus)}

        first = await sync()
        assert "corpus/f.txt" in first.added

        (corpus / "f.txt").write_text("a second, edited version of the content")
        second = await sync()
        assert "corpus/f.txt" in second.updated
        assert "corpus/f.txt" not in second.added
        rec = next(s for s in get_services().store.get_sources() if s["filename"] == "corpus/f.txt")
        assert rec["file_hash"] == file_hash(corpus / "f.txt")

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
        # ``_mock_extract_file`` is the class-level xberg patch, unused here.
        from lilbee.app.services import get_services
        from lilbee.data.ingest import pipeline
        from lilbee.data.types import _IngestResult

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
        from lilbee.data.types import _IngestResult

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
        from lilbee.data.types import _IngestResult
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
        from lilbee.data.types import _IngestResult
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
        from lilbee.data.ingest.pipeline import produce_records as orig_ingest

        async def _failing_ingest(path, name, content_type, **kwargs):
            if "bad" in name:
                raise RuntimeError("simulated failure")
            return await orig_ingest(path, name, content_type, **kwargs)

        with patch("lilbee.data.ingest.pipeline.produce_records", side_effect=_failing_ingest):
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

        from lilbee.data.ingest.pipeline import produce_records as orig_ingest

        async def _failing_ingest(path, name, content_type, **kwargs):
            if "flaky" in name:
                raise RuntimeError("simulated failure on update")
            return await orig_ingest(path, name, content_type, **kwargs)

        with patch("lilbee.data.ingest.pipeline.produce_records", side_effect=_failing_ingest):
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

        with patch("lilbee.data.ingest.pipeline.produce_records", side_effect=_fail):
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

        from lilbee.data.ingest.pipeline import produce_records as orig

        async def _fail(path, name, ct, **kwargs):
            if "qflaky" in name:
                raise RuntimeError("quiet fail")
            return await orig(path, name, ct, **kwargs)

        with patch("lilbee.data.ingest.pipeline.produce_records", side_effect=_fail):
            result = await sync(quiet=True)
        assert "qflaky.txt" in result.failed
        assert "qflaky.txt" not in result.updated


@mock.patch(
    "lilbee.data.extract.xberg.aextract_document",
    new_callable=mock.AsyncMock,
    return_value=_make_xberg_result(),
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

    async def test_worker_mode_releases_the_pool_and_routes_every_file(
        self, mock_extract_file, isolated_env, mock_svc, monkeypatch
    ):
        """ingest_stream in worker mode must shut the pool down, happy path included.

        Drives the whole worker dispatch (shard -> batch -> collect -> flush) with the
        pool faked out, which is the only place the shard slotting, _collect_from_worker
        and the pool teardown run together.
        """
        from concurrent.futures import ThreadPoolExecutor

        from lilbee.data.ingest import ingest_stream, workers
        from lilbee.data.ingest import pipeline as pipeline_mod
        from lilbee.data.types import FileToProcess
        from tests.conftest import one_shard

        shutdowns: list[dict] = []
        order: list[str] = []

        class RecordingPool(ThreadPoolExecutor):
            def shutdown(self, wait=True, **kwargs):
                shutdowns.append(kwargs)
                super().shutdown(wait=wait)

        monkeypatch.setattr(pipeline_mod, "resolve_process_count", lambda count: 2)
        monkeypatch.setattr(pipeline_mod, "warm_parent_engine", lambda: order.append("warm"))
        built: dict = {}

        def fake_build_pool(n, cfg, inflight):
            order.append("pool")
            built["processes"], built["inflight"] = n, inflight
            return RecordingPool(max_workers=2)

        monkeypatch.setattr(pipeline_mod, "build_pool", fake_build_pool)
        monkeypatch.setattr(
            workers,
            "run_batch",
            lambda batch: [
                workers.WorkerOutcome(name=f.name, records=[], page_texts=[]) for f in batch
            ],
        )

        (isolated_env / "w1.txt").write_text("one")
        (isolated_env / "w2.txt").write_text("two")
        files = [
            FileToProcess("w1.txt", isolated_env / "w1.txt", "text", "h1", False),
            FileToProcess("w2.txt", isolated_env / "w2.txt", "text", "h2", False),
        ]
        added = {"w1.txt": None, "w2.txt": None}
        skipped: dict[str, None] = {}

        await ingest_stream(
            one_shard(files), added, {}, {}, skipped, quiet=True, unindexed_files=len(files)
        )

        # Ordering is the fix for the A/B that embedded 0 of 50k: a pool built
        # first spawns attach-only workers against an engine nothing has started.
        assert order == ["warm", "pool"]
        assert built["processes"] == 2
        assert built["inflight"] > 0  # admission is passed through, not defaulted
        assert shutdowns == [{"cancel_futures": True}]
        # Zero chunks is a skip, not an add: the worker produced no searchable text.
        assert set(skipped) == {"w1.txt", "w2.txt"}
        assert added == {}

    async def test_cancel_during_ingest_stream(self, mock_extract_file, isolated_env, mock_svc):
        """Cancel set mid-batch raises CancelledError for pending files."""
        import asyncio
        import threading

        from lilbee.data.ingest import ingest_stream
        from tests.conftest import one_shard

        (isolated_env / "a.txt").write_text("file a")
        (isolated_env / "b.txt").write_text("file b")

        cancel = threading.Event()
        cancel.set()

        from lilbee.data.types import FileToProcess

        added = {"a.txt": None, "b.txt": None}
        files = [
            FileToProcess("a.txt", isolated_env / "a.txt", "text", "hash_a", False),
            FileToProcess("b.txt", isolated_env / "b.txt", "text", "hash_b", False),
        ]
        with pytest.raises(asyncio.CancelledError):
            await ingest_stream(
                one_shard(files), added, {}, {}, {}, quiet=True, cancel=cancel, unindexed_files=0
            )

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
                _feed([_ok(), _cancelled()]),
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


class TestOcrForceRequested:
    """cfg-independent LILBEE_OCR_FORCE lever that OCRs every page, text layer or not."""

    def test_false_when_env_unset(self, monkeypatch):
        from lilbee.data.extract.document import _ocr_force_requested

        monkeypatch.delenv("LILBEE_OCR_FORCE", raising=False)
        assert _ocr_force_requested() is False

    @pytest.mark.parametrize("value", ["1", "true", "YES"])
    def test_true_when_env_set(self, monkeypatch, value):
        from lilbee.data.extract.document import _ocr_force_requested

        monkeypatch.setenv("LILBEE_OCR_FORCE", value)
        assert _ocr_force_requested() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
    def test_false_for_non_truthy_values(self, monkeypatch, value):
        from lilbee.data.extract.document import _ocr_force_requested

        monkeypatch.setenv("LILBEE_OCR_FORCE", value)
        assert _ocr_force_requested() is False


class TestIngestHelpers:
    """Cover edge cases in ingest_document and ingest_code_sync."""

    @mock.patch(
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_empty_result(),
    )
    async def testingest_document_empty_chunks(self, mock_extract_file, isolated_env):
        """Document that produces no chunks returns empty list."""
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "empty.txt"
        f.write_text("   ")
        result, _ = await ingest_document(f, "empty.txt", "text")
        assert result == []

    async def test_ingest_code_empty_chunks(self, isolated_env):
        """Code file that produces no chunks returns empty list."""
        from unittest.mock import patch

        from lilbee.data.ingest import ingest_code_sync

        f = isolated_env / "empty.py"
        f.write_text("")
        with patch("lilbee.data.extract.code_chunker.chunk_code", return_value=[]):
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
            patch("lilbee.data.extract.code_chunker._ensure_language", return_value=True),
            patch("lilbee.data.extract.code_chunker.process", return_value=_FakeResult()),
        ):
            records = ingest_code_sync(f, "pkg/mod.py")
        assert records
        joined = "\n".join(r["chunk"] for r in records)
        assert "# File: pkg/mod.py" in joined
        assert str(f) not in joined  # absolute path must never leak into content

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def testingest_document_pdf_with_pages(self, mock_kf, isolated_env):
        """PDF document returns records with page metadata."""
        mock_kf.return_value = _make_xberg_result(
            text="Page 1 content. " * 10 + "Page 2 content. " * 10,
            num_chunks=2,
            has_pages=True,
        )
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "test.pdf"
        f.write_bytes(b"fake")
        result, _ = await ingest_document(f, "test.pdf", "pdf")
        assert len(result) == 2
        assert result[0]["page_start"] == 1
        assert result[1]["page_start"] == 2


class TestCancellation:
    @mock.patch(
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
    )
    async def test_cancelled_error_propagates(self, mock_extract_file, isolated_env):
        """CancelledError in _process_one is re-raised, not swallowed."""
        import asyncio

        async def _cancel(*args, **kwargs):
            raise asyncio.CancelledError()

        with mock.patch("lilbee.data.ingest.pipeline.produce_records", side_effect=_cancel):
            from lilbee.data.ingest import ingest_stream
            from lilbee.data.types import FileToProcess
            from tests.conftest import one_shard

            added = {"cancel.txt": None}
            entry = FileToProcess(
                "cancel.txt", isolated_env / "cancel.txt", "text", "abc123", False
            )
            with pytest.raises(asyncio.CancelledError):
                await ingest_stream(
                    one_shard([entry]), added, {}, {}, {}, quiet=True, unindexed_files=0
                )

    @mock.patch(
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
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

        from lilbee.data.ingest import ingest_stream
        from lilbee.runtime.cancellation import TaskCancelledError
        from tests.conftest import one_shard

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
        from lilbee.data.types import FileToProcess

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
            await ingest_stream(
                one_shard(files),
                added,
                {},
                failed,
                skipped,
                quiet=True,
                on_progress=on_progress,
                unindexed_files=0,
            )


class TestSkipMarkerLifecycle:
    """A file that produces no chunks gets a skip marker; the next sync skips it
    until the file changes or retry_skipped / force_rebuild clears the marker."""

    @staticmethod
    def _zero_chunks(*_args, **_kwargs):
        # Simulate "OCR found no usable text": no records produced, so the file
        # is recorded as skipped.
        from lilbee.data.store import SourceMeta

        return [], SourceMeta()

    async def test_failed_file_is_skipped_on_next_sync(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import load_skip_markers

        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_chunks
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
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_chunks
        ):
            await sync(quiet=True)
            # retry_skipped drops the marker so the file is attempted again.
            retried = await sync(quiet=True, retry_skipped=True)
            assert "scanned.pdf" in retried.skipped  # attempted again (still 0 chunks)

    async def test_force_rebuild_also_clears_markers(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync

        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_chunks
        ):
            await sync(quiet=True)
            rebuilt = await sync(quiet=True, force_rebuild=True)
            assert "scanned.pdf" in rebuilt.skipped  # attempted again after the wipe


class TestZeroChunkPageTextPersistence:
    """A zero-chunk file with page texts persists both and stops replanning."""

    @staticmethod
    async def _pages_no_chunks(
        path,
        source_name,
        content_type,
        *,
        quiet=False,
        on_progress=None,
        page_texts_out=None,
    ):
        from lilbee.data.store import SourceMeta

        if page_texts_out is not None:
            page_texts_out.append(
                {"source": source_name, "page": 1, "text": " ", "content_type": "pdf"}
            )
        return [], SourceMeta()

    async def test_pages_and_source_row_persist_and_replan_stops(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync

        (isolated_env / "blank.pdf").write_bytes(b"%PDF-1.4 whitespace only")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._pages_no_chunks
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
                "lilbee.data.ingest.pipeline.produce_records",
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


class TestPlanProgress:
    def test_logs_periodic_progress_at_warning(self, caplog, monkeypatch):
        import logging

        from lilbee.data.ingest import pipeline

        # Log on every tick so a fast unit test still exercises the periodic path.
        monkeypatch.setattr(pipeline, "_PLAN_LOG_INTERVAL_S", 0.0)
        progress = pipeline._PlanProgress(total=3)
        # Capture at WARNING, the default LILBEE_LOG_LEVEL: an info line is
        # filtered here exactly as it is under a real headless `lilbee sync`, so
        # this fails if the progress ever drops back below warning and goes
        # silent in the very case it exists to make observable.
        with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"):
            progress.tick()
            progress.tick()
            progress.tick()
        records = [r for r in caplog.records if "Planning: examined" in r.getMessage()]
        assert records
        assert all(r.levelno >= logging.WARNING for r in records)
        assert any("3/3" in r.getMessage() for r in records)

    def test_silent_below_the_interval(self, caplog, monkeypatch):
        import logging

        from lilbee.data.ingest import pipeline

        # A short sync (ticks faster than the interval) stays quiet -- no noise.
        monkeypatch.setattr(pipeline, "_PLAN_LOG_INTERVAL_S", 3600.0)
        progress = pipeline._PlanProgress(total=100)
        with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"):
            for _ in range(50):
                progress.tick()
        assert not any("Planning:" in r.getMessage() for r in caplog.records)


class TestScanProgress:
    def test_logs_periodic_scan_progress_at_warning(self, caplog, monkeypatch):
        import logging

        from lilbee.data.ingest import discovery

        # Log on every tick so a fast unit test still exercises the periodic path.
        monkeypatch.setattr(discovery, "_SCAN_LOG_INTERVAL_S", 0.0)
        progress = discovery._ScanProgress()
        # WARNING, the default level: the discovery walk runs before the plan
        # pass and is otherwise silent, so its heartbeat must survive the default
        # or a headless sync over a large tree looks hung during the walk.
        with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.discovery"):
            progress.tick(matched=True)
            progress.tick(matched=False)
        records = [r for r in caplog.records if "Scanning for files" in r.getMessage()]
        assert records
        assert all(r.levelno >= logging.WARNING for r in records)
        # examined counts every visited file; matched only the supported ones.
        assert any("examined 2, matched 1" in r.getMessage() for r in records)

    def test_silent_below_the_interval(self, caplog, monkeypatch):
        import logging

        from lilbee.data.ingest import discovery

        # A quick walk (ticks faster than the interval) stays quiet -- no noise.
        monkeypatch.setattr(discovery, "_SCAN_LOG_INTERVAL_S", 3600.0)
        progress = discovery._ScanProgress()
        with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.discovery"):
            for _ in range(50):
                progress.tick(matched=True)
        assert not any("Scanning for files" in r.getMessage() for r in caplog.records)


class TestDetectPending:
    """Cheap detection: filesystem walk + hash compare against the sources table."""

    def test_returns_zero_when_documents_dir_missing(self, isolated_env, tmp_path):
        cfg.documents_dir = tmp_path / "does_not_exist"
        from lilbee.data.ingest import detect_pending

        assert detect_pending() == 0

    def test_counts_added_and_updated_not_absent(self, isolated_env, mock_svc):
        from lilbee.data.ingest import detect_pending, file_hash

        new_file = isolated_env / "new.md"
        new_file.write_text("brand new content")

        existing = isolated_env / "existing.md"
        existing.write_text("v1")
        mock_svc.store.upsert_source("existing.md", file_hash(existing), 1, source_type="document")
        existing.write_text("v2")  # hash now diverges from store

        # A row whose disk file is gone is NOT pending: sync leaves it indexed,
        # so it is not counted as work to do.
        mock_svc.store.upsert_source("gone.md", "deadbeef", 1, source_type="document")

        # 1 added (new.md) + 1 updated (existing.md); gone.md does not count.
        assert detect_pending() == 2

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

    def test_parallel_plan_matches_serial(self, isolated_env, monkeypatch):
        # Many brand-new files (no existing sources): the parallel planning pass
        # must produce the same added set, order, and counts as the serial pass.
        from lilbee.data.ingest import pipeline

        names = [f"doc{i:03d}.txt" for i in range(50)]
        disk: dict[str, Path] = {}
        for n in names:
            p = isolated_env / n
            p.write_text(f"content of {n}")
            disk[n] = p

        monkeypatch.setattr(pipeline, "_plan_workers", lambda: 8)
        parallel = pipeline._plan_file_changes(disk, {}, cancel=None)
        monkeypatch.setattr(pipeline, "_plan_workers", lambda: 1)
        serial = pipeline._plan_file_changes(disk, {}, cancel=None)

        assert [f.name for f in parallel.files_to_process] == [
            f.name for f in serial.files_to_process
        ]
        assert list(parallel.added) == list(serial.added) == names
        assert parallel.unchanged == serial.unchanged == 0

    @pytest.mark.parametrize("workers", [1, 4])
    def test_cancelled_plan_returns_partial(self, isolated_env, monkeypatch, workers):
        # Both planning paths stop on a set cancel: the serial one (single CPU,
        # and what detect_pending falls back to) and the pooled one.
        import threading

        from lilbee.data.ingest import pipeline

        disk: dict[str, Path] = {}
        for i in range(10):
            p = isolated_env / f"c{i}.txt"
            p.write_text("x")
            disk[p.name] = p

        cancel = threading.Event()
        cancel.set()
        monkeypatch.setattr(pipeline, "_plan_workers", lambda: workers)
        plan = pipeline._plan_file_changes(disk, {}, cancel=cancel)
        assert plan.files_to_process == []

    def test_parallel_plan_cancels_pending_work_midpass(self, isolated_env, monkeypatch):
        import threading

        from lilbee.data.ingest import pipeline

        disk: dict[str, Path] = {}
        for i in range(30):
            p = isolated_env / f"m{i:02d}.txt"
            p.write_text(str(i))
            disk[p.name] = p

        cancel = threading.Event()
        monkeypatch.setattr(pipeline, "_plan_workers", lambda: 4)
        original = pipeline._classify_file_change
        seen = {"n": 0}

        def _spy(name, path, record, markers):
            seen["n"] += 1
            if seen["n"] >= 3:
                cancel.set()  # trip cancel partway through the pass
            return original(name, path, record, markers)

        monkeypatch.setattr(pipeline, "_classify_file_change", _spy)
        plan = pipeline._plan_file_changes(disk, {}, cancel=cancel)

        # A mid-pass cancel drops queued work rather than hashing all 30 files.
        assert len(plan.files_to_process) < 30

    def test_plan_workers_config_override_beats_auto(self, monkeypatch):
        import types

        from lilbee.data.ingest import pipeline

        monkeypatch.setattr(
            pipeline, "active_config", lambda: types.SimpleNamespace(ingest_workers=3)
        )
        assert pipeline._plan_workers() == 3

        monkeypatch.setattr(
            pipeline, "active_config", lambda: types.SimpleNamespace(ingest_workers=0)
        )
        monkeypatch.setattr(pipeline, "available_cpu_count", lambda: 9)
        assert pipeline._plan_workers() == 9

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
        from lilbee.data.types import _IngestResult

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
        real_plan = pipeline._plan_items

        def _spy_plan(*args, **kwargs):
            plan_threads.append(threading.get_ident())
            return real_plan(*args, **kwargs)

        monkeypatch.setattr(pipeline, "_plan_items", _spy_plan)
        with mock.patch(
            "lilbee.data.extract.xberg.aextract_document",
            new_callable=mock.AsyncMock,
            return_value=_make_xberg_result(),
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
            "lilbee.data.extract.xberg.aextract_document",
            new_callable=mock.AsyncMock,
            return_value=_make_xberg_result(),
        ):
            result = await sync(quiet=True)
        assert result.unchanged == 1
        mock_svc.store.update_source_stats.assert_called_once()
        backfills = mock_svc.store.update_source_stats.call_args.args[0]
        assert backfills[0].record["filename"] == "legacy.txt"
        assert backfills[0].stat.size_bytes == f.stat().st_size

    async def test_sync_refuses_without_an_embedding_model(self, isolated_env, mock_svc):
        """Ingest cannot degrade the way search can: warning and continuing pays the
        full parse and OCR cost for every file and then fails to embed all of them,
        so the run must stop before any of that work happens."""
        from lilbee.data.ingest import sync

        (isolated_env / "doc.txt").write_text("content")
        mock_svc.embedder.validate_model.return_value = False
        with pytest.raises(RuntimeError, match="embedding model"):
            await sync(quiet=True)


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

    def test_absent_set_keeps_imported_drops_missing_document(self):
        from pathlib import Path

        from lilbee.data.ingest.pipeline import _absent_sources
        from lilbee.data.store import SourceType

        # The absent set drives move detection only (a missing file is never
        # removed on its own). An import has no backing file, so it is excluded.
        sources = [
            self._src("gone.md", SourceType.DOCUMENT),
            self._src("present.md", SourceType.DOCUMENT),
            self._src("shared.pdf", SourceType.IMPORTED),
        ]
        disk_files = {"present.md": Path("present.md")}
        assert _absent_sources(sources, disk_files) == ["gone.md"]


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
        from lilbee.data.types import _IngestResult

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
        from lilbee.data.types import _IngestResult
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
        from lilbee.data.types import _IngestResult
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
        from lilbee.data.types import _IngestResult
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
        from lilbee.data.types import _IngestResult
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
        from lilbee.data.types import _IngestResult

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
        from lilbee.data.types import _IngestResult
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
            _feed([_zero_chunk_result()]),
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
        from lilbee.data.types import _IngestResult

        async def _emptied() -> _IngestResult:
            return _IngestResult(
                "notes.md", Path("notes.md"), chunk_count=0, error=None, needs_cleanup=True
            )

        await _collect_results(_feed([_emptied()]), {}, {}, {}, {}, window=2)
        mock_svc.store.remove_documents.assert_called_once_with(["notes.md"])

    async def test_zero_text_without_cleanup_does_not_purge(self, mock_svc):
        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.types import _IngestResult

        async def _emptied() -> _IngestResult:
            return _IngestResult(
                "fresh.md", Path("fresh.md"), chunk_count=0, error=None, needs_cleanup=False
            )

        await _collect_results(_feed([_emptied()]), {}, {}, {}, {}, window=2)
        mock_svc.store.remove_documents.assert_not_called()

    def test_purge_emptied_sources_noop_on_empty(self, mock_svc):
        from lilbee.data.ingest.pipeline import _purge_emptied_sources

        _purge_emptied_sources([])
        mock_svc.store.remove_documents.assert_not_called()

    async def test_error_cancels_still_running_siblings(self):
        """When one task raises, _collect_results' finally cancels the in-flight ones."""
        import asyncio

        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.types import _IngestResult

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
                _feed([_boom(), _never_finishes()]), added, {}, failed, skipped, window=2
            )
        assert sibling_cancelled.is_set()

    async def test_finally_path_flush_failure_still_cancels_siblings(self, monkeypatch):
        """A flush raising on the way out must not strand the in-flight siblings."""
        import asyncio

        from lilbee.data.ingest import pipeline
        from lilbee.data.types import _IngestResult

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
                _feed([_boom(), _never_finishes()]), {}, {}, {}, {}, window=2
            )
        assert sibling_cancelled.is_set()


class TestStreamedPlan:
    """The plan pass is sharded and overlapped with ingest, not a barrier before it."""

    @staticmethod
    def _small_shards(monkeypatch, size=2):
        from lilbee.data.ingest import pipeline

        monkeypatch.setattr(pipeline, "_PLAN_SHARD_MIN_FILES", size)
        monkeypatch.setattr(pipeline, "_PLAN_SHARD_MAX_FILES", size)

    @staticmethod
    async def _sync_with_extraction():
        from lilbee.data.ingest import sync

        with mock.patch(
            "lilbee.data.extract.xberg.aextract_document",
            new_callable=mock.AsyncMock,
            return_value=_make_xberg_result(),
        ):
            return await sync(quiet=True)

    def test_shard_bounds_ramp_and_cover_every_file(self):
        from lilbee.data.ingest.pipeline import (
            _PLAN_SHARD_MAX_FILES,
            _PLAN_SHARD_MIN_FILES,
            _shard_bounds,
        )

        assert list(_shard_bounds(0)) == []
        total = _PLAN_SHARD_MAX_FILES * 4
        bounds = list(_shard_bounds(total))
        # Small first shard, doubling, capped -- and contiguous over the corpus.
        assert bounds[0] == (0, _PLAN_SHARD_MIN_FILES)
        assert bounds[1][1] - bounds[1][0] == 2 * _PLAN_SHARD_MIN_FILES
        assert max(hi - lo for lo, hi in bounds) == _PLAN_SHARD_MAX_FILES
        assert [lo for lo, _ in bounds[1:]] == [hi for _, hi in bounds[:-1]]
        assert bounds[-1][1] == total

    async def test_ingest_starts_before_the_corpus_is_planned(
        self, isolated_env, monkeypatch, mock_svc
    ):
        # Planning the last file blocks until a file from the first shard has been
        # extracted. A plan-everything-then-ingest pass can never satisfy that, so
        # this deadlocks (and fails on the wait) unless planning overlaps ingest.
        import threading

        from lilbee.data.ingest import pipeline, sync

        names = [f"doc{i}.txt" for i in range(6)]
        for name in names:
            (isolated_env / name).write_text(f"content of {name}")
        self._small_shards(monkeypatch)

        extracted = threading.Event()
        real_hash = pipeline.file_hash

        def _blocking_hash(path):
            if path.name == names[-1]:
                assert extracted.wait(timeout=10), "planning never overlapped ingest"
            return real_hash(path)

        def _extract(*_args, **_kwargs):
            extracted.set()
            return _make_xberg_result()

        monkeypatch.setattr(pipeline, "file_hash", _blocking_hash)
        with mock.patch(
            "lilbee.data.extract.xberg.aextract_document",
            new_callable=mock.AsyncMock,
            side_effect=_extract,
        ):
            result = await asyncio.wait_for(sync(quiet=True), timeout=60)
        assert sorted(result.added) == sorted(names)

    async def test_sharded_plan_accumulates_unchanged_and_backfills(
        self, isolated_env, monkeypatch, mock_svc
    ):
        # Counts and stat backfills are per shard, so they have to add up across
        # the whole run: four legacy rows over two shards, all unchanged.
        from lilbee.data.ingest import file_hash

        names = [f"legacy{i}.txt" for i in range(4)]
        for name in names:
            path = isolated_env / name
            path.write_text(f"legacy content of {name}")
            # Tracked at the right hash but with no stat columns, so each file
            # hashes clean and queues a backfill.
            mock_svc.store.upsert_source(name, file_hash(path), 1, source_type="document")
        self._small_shards(monkeypatch)

        result = await self._sync_with_extraction()
        assert result.unchanged == 4
        assert result.added == []
        backfilled = [
            b.record["filename"]
            for call in mock_svc.store.update_source_stats.call_args_list
            for b in call.args[0]
        ]
        assert sorted(backfilled) == names
        assert mock_svc.store.update_source_stats.call_count == 2

    async def test_shard_backfill_retries_once_on_lock_timeout(
        self, isolated_env, monkeypatch, mock_svc
    ):
        # A shard's backfill now lands while ingest is flushing, so it can lose the
        # store lock to a flush; it takes the same one-shot retry the flush does.
        from lilbee.data.ingest import file_hash, pipeline
        from lilbee.runtime.lock import LockTimeoutError

        path = isolated_env / "legacy.txt"
        path.write_text("legacy content")
        mock_svc.store.upsert_source("legacy.txt", file_hash(path), 1, source_type="document")
        sleeps: list[float] = []
        monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
        mock_svc.store.update_source_stats.side_effect = [LockTimeoutError("busy"), None]

        result = await self._sync_with_extraction()
        assert result.unchanged == 1
        assert mock_svc.store.update_source_stats.call_count == 2
        assert sleeps == [pipeline._FLUSH_RETRY_DELAY_SECONDS]

    async def test_move_pairs_across_shard_boundaries(self, isolated_env, monkeypatch, mock_svc):
        # The absent-source pool is built once for the run, so a file that moved
        # still pairs with its old key when the two land in different shards.
        from lilbee.data.ingest import file_hash

        for i in range(4):
            (isolated_env / f"doc{i}.txt").write_text(f"content of doc{i}")
        moved = isolated_env / "zmoved.txt"
        moved.write_text("the relocated document")
        mock_svc.store.upsert_source("old/moved.txt", file_hash(moved), 1, source_type="document")
        self._small_shards(monkeypatch)

        result = await self._sync_with_extraction()
        assert result.relocated == ["zmoved.txt"]
        assert "zmoved.txt" not in result.added
        mock_svc.store.relocate_sources.assert_called_once()
        assert mock_svc.store.relocate_sources.call_args.args[0][0][:2] == (
            "old/moved.txt",
            "zmoved.txt",
        )

    async def test_cancel_stops_planning_the_rest_of_the_corpus(
        self, isolated_env, monkeypatch, mock_svc
    ):
        # A cancel during ingest stops the planner feeding it, instead of hashing
        # every remaining file first.
        import threading

        from lilbee.data.ingest import pipeline, sync

        names = [f"doc{i:02d}.txt" for i in range(40)]
        for name in names:
            (isolated_env / name).write_text(f"content of {name}")
        self._small_shards(monkeypatch)

        cancel = threading.Event()
        classified: list[str] = []
        real_classify = pipeline._classify_file_change

        def _spy_classify(name, path, record, markers):
            classified.append(name)
            return real_classify(name, path, record, markers)

        def _extract(*_args, **_kwargs):
            cancel.set()
            return _make_xberg_result()

        monkeypatch.setattr(pipeline, "_classify_file_change", _spy_classify)
        with (
            mock.patch(
                "lilbee.data.extract.xberg.aextract_document",
                new_callable=mock.AsyncMock,
                side_effect=_extract,
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await asyncio.wait_for(sync(quiet=True, cancel=cancel), timeout=60)
        assert len(classified) < len(names)

    async def test_plan_shards_break_between_shards_is_deterministic(
        self, isolated_env, monkeypatch, mock_svc
    ):
        # The shard loop's break fires when a cancel is observed between shards.
        # Driving it through sync relies on a cancel-during-extract race that is
        # not portable across platforms; drive _plan_shards directly with a cancel
        # already set, over >1 shard, so the break is hit without any timing.
        import threading

        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.pipeline import _plan_shards, _StreamedPlan

        disk = {f"doc{i}.txt": isolated_env / f"doc{i}.txt" for i in range(4)}
        for path in disk.values():
            path.write_text("content")
        monkeypatch.setattr(pipeline, "_PLAN_SHARD_MIN_FILES", 1)
        monkeypatch.setattr(pipeline, "_PLAN_SHARD_MAX_FILES", 1)

        cancel = threading.Event()
        cancel.set()  # shard 0 plans nothing, ahead drops to None, shard 1 breaks
        state = _StreamedPlan()
        shards = _plan_shards(disk, {}, {}, [], state, cancel)
        yielded = [shard async for shard in shards]
        assert yielded == []  # the break stopped planning the remaining shards
        assert state.planned == 0

    async def test_cancel_with_nothing_left_to_admit_still_raises(self, isolated_env, mock_svc):
        # The last admitted file finishes and the planner has stopped, so ingest
        # returns without any file raising; sync still reports the run cancelled.
        import threading

        from lilbee.data.ingest import sync

        (isolated_env / "only.txt").write_text("the one file in this corpus")
        cancel = threading.Event()

        def _extract(*_args, **_kwargs):
            cancel.set()
            return _make_xberg_result()

        with (
            mock.patch(
                "lilbee.data.extract.xberg.aextract_document",
                new_callable=mock.AsyncMock,
                side_effect=_extract,
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await sync(quiet=True, cancel=cancel)
        # Cancelled before the marker pass: the file is not skip-marked.
        from lilbee.data.ingest.skip_marker import load_skip_markers

        assert load_skip_markers(cfg.data_root) == {}

    async def test_feed_close_discards_a_shard_that_landed_late(self):
        # A shard landing between the prefetch and the close still owns unstarted
        # coroutines; closing the feed has to close them, not leak them.
        from lilbee.data.ingest.pipeline import _ResultFeed
        from lilbee.data.types import _IngestResult

        planned: list = []

        async def _file(name) -> _IngestResult:
            raise AssertionError("an unstarted file must never run")

        async def _shards():
            coro = _file("a.txt")
            planned.append(coro)
            yield [coro]

        feed = _ResultFeed(_shards())
        assert await feed.take(wait=False) is None  # starts the prefetch
        await asyncio.sleep(0)  # let the shard land while nothing is consuming
        await feed.aclose()
        assert planned[0].cr_frame is None  # closed, not leaked

    async def test_feed_does_not_wait_on_a_shard_that_is_not_ready(self):
        from lilbee.data.ingest.pipeline import _ResultFeed
        from lilbee.data.types import _IngestResult

        gate = asyncio.Event()

        async def _file(name) -> _IngestResult:
            return _IngestResult(name, Path(name), chunk_count=0, error=None)

        async def _shards():
            yield [_file("a.txt")]
            await gate.wait()
            yield [_file("b.txt")]

        feed = _ResultFeed(_shards())
        first = await feed.take(wait=True)
        assert first is not None
        first.close()
        assert feed.planned == 1
        # The second shard is still being planned: the collector is handed None
        # rather than being blocked behind it.
        assert await feed.take(wait=False) is None
        gate.set()
        second = await feed.take(wait=True)
        assert second is not None
        second.close()
        assert feed.planned == 2
        await feed.aclose()

    async def test_collector_wakeups_stay_bounded_per_file(self, monkeypatch, mock_svc):
        # With every window slot busy, an already-planned shard must not wake the
        # collector on every pass: it has nothing to admit until a file finishes.
        from lilbee.data.ingest import pipeline
        from lilbee.data.types import _IngestResult

        async def _slow(name) -> _IngestResult:
            await asyncio.sleep(0.05)
            return _IngestResult(name, Path(name), chunk_count=0, error=None)

        async def _shards():
            yield [_slow("a.txt")]
            yield [_slow("b.txt")]

        waits = 0
        real_wait = pipeline.asyncio.wait

        async def _counting_wait(fs, **kwargs):
            nonlocal waits
            waits += 1
            return await real_wait(fs, **kwargs)

        monkeypatch.setattr(pipeline.asyncio, "wait", _counting_wait)
        await pipeline._collect_results(pipeline._ResultFeed(_shards()), {}, {}, {}, {}, window=1)
        # One wait per file, plus at most a couple for the shard landings.
        assert waits <= 6


class TestTaskWindow:
    """The ingest pump keeps at most ``window`` tasks alive at once."""

    async def test_in_flight_high_water_stays_at_window(self):
        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.types import _IngestResult

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
        await _collect_results(_lazy_feed(_pending()), {}, {}, {}, skipped, window=window)
        assert len(skipped) == total
        assert high_water <= window
        assert created == total

    async def test_cancel_mid_stream_flushes_buffer_and_stops(self, mock_svc):
        # Files completed before the cancel are flushed in the finally; files
        # never pulled from the iterator are never started.
        from lilbee.data.ingest.pipeline import _collect_results
        from lilbee.data.types import _IngestResult

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
            await _collect_results(_lazy_feed(_pending()), added, {}, {}, {}, window=1)
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


class TestDiscoverRegisteredRoots:
    def test_registered_dir_root_indexed_under_label(self, isolated_env, tmp_path):
        """A registered directory root indexes its files keyed under the label."""
        from lilbee.data.ingest import discover_files

        source = tmp_path / "corpus"
        source.mkdir()
        (source / "doc.md").write_text("# Doc")
        cfg.linked_roots = {"corpus": str(source)}

        found = discover_files()
        assert found["corpus/doc.md"] == source / "doc.md"

    def test_registered_file_root_indexed_by_label(self, isolated_env, tmp_path):
        """A registered single-file root indexes as one entry keyed by the label."""
        from lilbee.data.ingest import discover_files

        outside_file = tmp_path / "linked.md"
        outside_file.write_text("# Linked")
        cfg.linked_roots = {"linked.md": str(outside_file)}

        found = discover_files()
        assert found["linked.md"] == outside_file

    def test_owned_files_and_roots_coexist(self, isolated_env, tmp_path):
        """documents_dir files and a registered root are both discovered."""
        from lilbee.data.ingest import discover_files

        (isolated_env / "owned.md").write_text("# Owned")
        source = tmp_path / "corpus"
        source.mkdir()
        (source / "doc.md").write_text("# Doc")
        cfg.linked_roots = {"corpus": str(source)}

        found = discover_files()
        assert "owned.md" in found
        assert "corpus/doc.md" in found

    def test_vanished_root_contributes_nothing(self, isolated_env, tmp_path):
        """A registered root whose path is gone yields nothing, not an error."""
        from lilbee.data.ingest import discover_files

        cfg.linked_roots = {"gone": str(tmp_path / "gone")}

        found = discover_files()
        assert "gone" not in found
        assert not any(name.startswith("gone/") for name in found)

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks require admin on Windows")
    def test_directory_symlink_inside_root_is_not_followed(self, isolated_env, tmp_path):
        """os.walk runs with followlinks=False, so a nested dir symlink is not descended."""
        import os

        from lilbee.data.ingest import discover_files

        source = tmp_path / "corpus"
        source.mkdir()
        (source / "doc.md").write_text("# Doc")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.md").write_text("secret")
        os.symlink(outside, source / "escape")
        cfg.linked_roots = {"corpus": str(source)}

        found = discover_files()
        assert "corpus/doc.md" in found
        assert not any("secret.md" in name for name in found)


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
        # No vision model configured -> xberg's tesseract backend OCRs scanned pages.
        assert config["ocr"].backend == "tesseract"
        # xberg 5.x errors on image OCR with an empty language list; lilbee must
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

    def test_table_extraction_off_sets_no_pdf_options(self, monkeypatch):
        """Table and layout both off leave pdf_options unset."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", False)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config.get("pdf_options") is None

    def test_table_extraction_sets_pdf_options_and_table_chunking(self, monkeypatch):
        """The flag turns on xberg table recognition and header-repeating splits."""
        from xberg import TableChunkingMode

        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "table_extraction", True)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config["pdf_options"].extract_tables is True
        assert config["chunking"].table_chunking == TableChunkingMode.REPEAT_HEADER

    def test_table_extraction_markdown_mode_skips_pdf_options(self, monkeypatch):
        """Non-paginated formats have no PdfConfig but still chunk tables with headers."""
        from xberg import TableChunkingMode

        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "table_extraction", True)
        config = extraction_config(ExtractMode.MARKDOWN)
        assert config.get("pdf_options") is None
        assert config["chunking"].table_chunking == TableChunkingMode.REPEAT_HEADER

    def test_layout_detection_off_sets_no_layout(self, monkeypatch):
        """With layout detection off, no layout config or pdf_options."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", False)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config.get("layout") is None
        assert config.get("pdf_options") is None

    def test_layout_detection_sets_layout_and_reading_order(self, monkeypatch):
        """The flag enables layout detection, reading order, and margin stripping."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", True)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config["layout"] is not None
        assert config["use_layout_for_markdown"] is True
        pdf = config["pdf_options"]
        assert pdf.reading_order is True
        assert pdf.top_margin_fraction == pytest.approx(0.05)
        assert pdf.bottom_margin_fraction == pytest.approx(0.05)
        # extract_tables stays at xberg's default (True); table INDEXING is
        # gated separately by cfg.table_extraction in _document_tables.
        assert pdf.extract_tables is True

    def test_table_model_defaults_to_slanet_auto(self, monkeypatch):
        """The layout config carries the configured table model (docling-parity default)."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", True)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config["layout"].table_model == "slanet_auto"

    def test_table_model_override(self, monkeypatch):
        """A different table model flows through to the layout config."""
        from lilbee.core.config import cfg
        from lilbee.core.config.enums import TableModel
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", True)
        monkeypatch.setattr(cfg, "table_model", TableModel.TATR)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config["layout"].table_model == "tatr"

    def test_layout_detection_markdown_mode_unchanged(self, monkeypatch):
        """Layout detection is visual-format work; non-paginated formats skip it."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", True)
        config = extraction_config(ExtractMode.MARKDOWN)
        assert config.get("layout") is None
        assert config.get("pdf_options") is None

    def test_layout_and_tables_combine_in_one_pdf_config(self, monkeypatch):
        """Both flags land in the same PdfConfig."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "table_extraction", True)
        monkeypatch.setattr(cfg, "layout_detection", True)
        config = extraction_config(ExtractMode.PAGINATED)
        pdf = config["pdf_options"]
        assert pdf.extract_tables is True
        assert pdf.reading_order is True


class TestTableModelCouplingWarning:
    """table_model only runs under layout detection; warn when it is silently unused."""

    def test_warns_when_table_extraction_on_without_layout(self, monkeypatch, caplog):
        from lilbee.core.config import cfg
        from lilbee.data.extract.document import warn_if_table_model_ignored

        monkeypatch.setattr(cfg, "table_extraction", True)
        monkeypatch.setattr(cfg, "layout_detection", False)
        with caplog.at_level("WARNING"):
            warn_if_table_model_ignored()
        assert any(
            "table_model" in r.getMessage() and "layout_detection" in r.getMessage()
            for r in caplog.records
        )

    def test_no_warning_when_layout_on(self, monkeypatch, caplog):
        from lilbee.core.config import cfg
        from lilbee.data.extract.document import warn_if_table_model_ignored

        monkeypatch.setattr(cfg, "table_extraction", True)
        monkeypatch.setattr(cfg, "layout_detection", True)
        with caplog.at_level("WARNING"):
            warn_if_table_model_ignored()
        assert not any("table_model" in r.getMessage() for r in caplog.records)

    def test_no_warning_for_default_config(self, caplog):
        """Out-of-box (table_extraction off): no spam for users not doing table work."""
        from lilbee.data.extract.document import warn_if_table_model_ignored

        with caplog.at_level("WARNING"):
            warn_if_table_model_ignored()
        assert not any("table_model" in r.getMessage() for r in caplog.records)


class TestBatchExtractionRouting:
    """cfg.batch_extraction routes ingest_document through the coalescer."""

    def test_make_extract_batcher_off_by_default(self):
        from lilbee.data.extract.document import make_extract_batcher

        assert make_extract_batcher() is None

    def test_make_extract_batcher_on(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.extract.batch import ExtractBatcher
        from lilbee.data.extract.document import make_extract_batcher

        monkeypatch.setattr(cfg, "batch_extraction", True)
        monkeypatch.setattr(cfg, "batch_extraction_size", 5)
        batcher = make_extract_batcher()
        assert isinstance(batcher, ExtractBatcher)
        assert batcher._size == 5

    async def test_ingest_document_extracts_through_the_active_batcher(self, isolated_env):
        """With a batcher active, extraction goes through the batch call, not single extract."""
        from lilbee.data.extract.batch import (
            ExtractBatcher,
            reset_active_batcher,
            set_active_batcher,
        )
        from lilbee.data.extract.document import _ocr_config, extraction_config
        from lilbee.data.ingest import ingest_document

        async def batch_fn(items, config):
            return [_make_xberg_result() for _ in items]

        batcher = ExtractBatcher(
            size=1, config_fn=extraction_config, ocr_fn=_ocr_config, batch_fn=batch_fn
        )
        with mock.patch(
            "lilbee.data.extract.xberg.aextract_document",
            side_effect=AssertionError("single-file extract must not run under batch mode"),
        ):
            token = set_active_batcher(batcher)
            try:
                f = isolated_env / "d.pdf"
                f.write_bytes(b"fake")
                records, _ = await ingest_document(f, "d.pdf", "pdf")
            finally:
                await batcher.close()
                reset_active_batcher(token)
        assert len(records) >= 1

    async def test_batch_extraction_detects_pdf_by_filename(self, isolated_env):
        """Regression: the batch path must let xberg detect the PDF from its filename,
        not pass lilbee's bare content_type ('pdf') as a MIME (xberg rejects it)."""
        from lilbee.data.extract.batch import (
            ExtractBatcher,
            reset_active_batcher,
            set_active_batcher,
        )
        from lilbee.data.extract.document import _ocr_config, extraction_config
        from lilbee.data.extract.xberg import aextract_batch
        from lilbee.data.ingest import ingest_document

        batcher = ExtractBatcher(
            size=1, config_fn=extraction_config, ocr_fn=_ocr_config, batch_fn=aextract_batch
        )
        f = isolated_env / "doc.pdf"
        f.write_bytes(make_pdf(pages=1))
        token = set_active_batcher(batcher)
        try:
            records, _ = await ingest_document(f, "doc.pdf", "pdf")
        finally:
            await batcher.close()
            reset_active_batcher(token)
        assert records, "batch extraction produced no records for a PDF"


class TestTableChunks:
    """Table indexing in ingest_document, driven by cfg.table_extraction."""

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_config_off_ignores_tables(self, mock_kf, isolated_env):
        """With the flag off, result tables are not indexed: current behavior."""
        mock_kf.return_value = _make_xberg_result(tables=[_make_table()])
        from lilbee.data.ingest import ingest_document
        from lilbee.data.store import ChunkType

        f = isolated_env / "test.pdf"
        f.write_bytes(b"fake")
        records, _ = await ingest_document(f, "test.pdf", "pdf")
        assert len(records) == 1
        assert all(r["chunk_type"] == ChunkType.RAW for r in records)

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_tables_indexed_as_table_chunks(self, mock_kf, isolated_env, monkeypatch):
        """Each table becomes its own chunk: markdown text, page span, TABLE type."""
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "table_extraction", True)
        tables = [
            _make_table(markdown="| a | b |\n|---|---|\n| 1 | 2 |", page_number=2),
            _make_table(markdown="| x |\n|---|\n| y |", page_number=5),
        ]
        mock_kf.return_value = _make_xberg_result(tables=tables)
        from lilbee.data.ingest import ingest_document
        from lilbee.data.store import ChunkType

        f = isolated_env / "test.pdf"
        f.write_bytes(b"fake")
        records, _ = await ingest_document(f, "test.pdf", "pdf")
        assert len(records) == 3
        table_records = [r for r in records if r["chunk_type"] == ChunkType.TABLE]
        assert len(table_records) == 2
        assert table_records[0]["chunk"] == tables[0].markdown
        assert table_records[0]["page_start"] == 2
        assert table_records[0]["page_end"] == 2
        assert table_records[1]["page_start"] == 5
        # Indices continue after the content chunks so (source, chunk_index) stays unique.
        assert [r["chunk_index"] for r in records] == [0, 1, 2]
        assert all(r["vector"] for r in table_records)
        assert all(r["content_type"] == "pdf" for r in table_records)

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_spanning_cell_markdown_passes_through_verbatim(
        self, mock_kf, isolated_env, monkeypatch
    ):
        """xberg's serialization of spanning cells is indexed exactly as produced."""
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "table_extraction", True)
        spanned = "| h1 | h1 | h2 |\n|---|---|---|\n| merged | merged | v |"
        mock_kf.return_value = _make_xberg_result(tables=[_make_table(markdown=spanned)])
        from lilbee.data.ingest import ingest_document
        from lilbee.data.store import ChunkType

        f = isolated_env / "test.pdf"
        f.write_bytes(b"fake")
        records, _ = await ingest_document(f, "test.pdf", "pdf")
        assert records[-1]["chunk_type"] == ChunkType.TABLE
        assert records[-1]["chunk"] == spanned

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_blank_table_markdown_skipped(self, mock_kf, isolated_env, monkeypatch):
        """A table with no usable serialization contributes no chunk."""
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "table_extraction", True)
        mock_kf.return_value = _make_xberg_result(tables=[_make_table(markdown="   ")])
        from lilbee.data.ingest import ingest_document
        from lilbee.data.store import ChunkType

        f = isolated_env / "test.pdf"
        f.write_bytes(b"fake")
        records, _ = await ingest_document(f, "test.pdf", "pdf")
        assert len(records) == 1
        assert records[0]["chunk_type"] == ChunkType.RAW

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_tables_without_text_chunks_still_indexed(
        self, mock_kf, isolated_env, monkeypatch
    ):
        """A document whose only content is tables is not dropped as empty."""
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "table_extraction", True)
        doc = _make_empty_result()
        doc.tables = [_make_table(page_number=1)]
        mock_kf.return_value = doc
        from lilbee.data.ingest import ingest_document
        from lilbee.data.store import ChunkType

        f = isolated_env / "test.pdf"
        f.write_bytes(b"fake")
        records, _ = await ingest_document(f, "test.pdf", "pdf")
        assert len(records) == 1
        assert records[0]["chunk_type"] == ChunkType.TABLE
        assert records[0]["chunk_index"] == 0


class TestClassifyXbergParityFormats:
    """Formats xberg extracts that the map must not silently drop."""

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
            # Containers are supported too (xberg extracts their combined text).
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
    "lilbee.data.extract.xberg.aextract_document",
    new_callable=mock.AsyncMock,
    return_value=_make_xberg_result(),
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
        from lilbee.data.extract.document import ocr_override

        cfg.enable_ocr = False
        cfg.ocr_timeout = 11.0
        with ocr_override(enable_ocr=True, ocr_timeout=99.0):
            pass
        assert cfg.enable_ocr is False
        assert cfg.ocr_timeout == 11.0

    def test_effective_values_reflect_override_then_revert(self, isolated_env):
        from lilbee.data.extract.document import (
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
        from lilbee.data.extract.document import _effective_enable_ocr, ocr_override

        cfg.enable_ocr = True
        with ocr_override(enable_ocr=None, ocr_timeout=None):
            assert _effective_enable_ocr() is True

    def test_overrides_isolated_across_contexts(self, isolated_env):
        # Two copied contexts must each see only their own override; this is the
        # concurrency guarantee that a global cfg mutation could not give.
        import contextvars

        from lilbee.data.extract.document import _effective_ocr_timeout, ocr_override

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
        from lilbee.data.extract.document import _effective_ocr_timeout

        cfg.ocr_timeout = 8.0
        with temporary_ocr_config(ocr_timeout=42.0):
            assert _effective_ocr_timeout() == 42.0
            assert cfg.ocr_timeout == 8.0
        assert _effective_ocr_timeout() == 8.0

    async def test_override_propagates_into_to_thread_worker(self, isolated_env):
        # The fix relies on asyncio.to_thread copying the calling context, which
        # is how the override reaches the extract worker that actually OCRs.
        import asyncio

        from lilbee.data.extract.document import _effective_ocr_timeout, ocr_override

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
        result, _ = await ingest_markdown(md, "empty.md")
        assert result == []

    async def test_no_chunks_returns_empty(self, isolated_env):
        from lilbee.data.ingest import ingest_markdown

        md = isolated_env / "blank.md"
        md.write_text("some text")
        with mock.patch("lilbee.data.extract.document.chunk_text", return_value=[]):
            result, _ = await ingest_markdown(md, "blank.md")
        assert result == []

    async def test_frontmatter_only_produces_chunks(self, isolated_env):
        from lilbee.data.ingest import ingest_markdown

        md = isolated_env / "fm_only.md"
        md.write_text("---\ntitle: Just Frontmatter\ntags: [test]\n---\n")
        result, _ = await ingest_markdown(md, "fm_only.md")
        assert len(result) > 0, "Frontmatter content should be indexed"


class TestPageTextAccumulator:
    """`page_texts_out` captures clean per-page text for the export dataset."""

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_pdf_pages_captured(self, mock_kf, isolated_env):
        mock_kf.return_value = _make_xberg_result(num_chunks=2, has_pages=True)
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "test.pdf"
        f.write_bytes(b"fake")
        pages: list = []
        await ingest_document(f, "test.pdf", "pdf", page_texts_out=pages)
        assert [p["page"] for p in pages] == [1, 2]
        assert all(p["content_type"] == "pdf" for p in pages)

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_non_paginated_doc_captured_as_page_zero(self, mock_kf, isolated_env):
        mock_kf.return_value = _make_xberg_result(text="Plain body. " * 10, has_pages=False)
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

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_ocr_pages_captured(self, mock_kf, isolated_env, mock_svc):
        # xberg OCRs scanned pages in-pass; the OCR'd text arrives in result.pages.
        cfg.enable_ocr = True
        cfg.vision_model = ""
        mock_kf.return_value = _make_xberg_result(
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
        """Structured formats now go through xberg: empty result yields no chunks."""
        from lilbee.data.ingest import ingest_document

        (isolated_env / "e.xml").write_bytes(b"<x/>")  # ingest_document reads the file bytes
        empty_result = mock.MagicMock(chunks=[], metadata=Metadata())
        mock_extract = mock.AsyncMock(return_value=empty_result)
        with mock.patch("lilbee.data.extract.xberg.aextract_document", mock_extract):
            result, _ = await ingest_document(isolated_env / "e.xml", "e.xml", "xml")
        assert result == []

    async def test_no_chunks_returns_empty(self, isolated_env):
        from lilbee.data.ingest import ingest_document

        (isolated_env / "s.xml").write_bytes(b"<x/>")  # ingest_document reads the file bytes
        no_chunks_result = mock.MagicMock(chunks=[], metadata=Metadata())
        mock_extract = mock.AsyncMock(return_value=no_chunks_result)
        with mock.patch("lilbee.data.extract.xberg.aextract_document", mock_extract):
            result, _ = await ingest_document(isolated_env / "s.xml", "s.xml", "xml")
        assert result == []


class TestChunkViaXberg:
    def test_empty_returns_empty(self):
        from lilbee.data.extract.chunk import chunk_text

        assert chunk_text("") == []

    def test_returns_chunks(self):
        from lilbee.data.extract.chunk import chunk_text

        result = chunk_text("Some text that should be chunked.")
        assert len(result) >= 1


class TestConceptIndexing:
    @pytest.fixture(autouse=True)
    def _mock_concepts_available(self):
        with mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True):
            yield

    @mock.patch(
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
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
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
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
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
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
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
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
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
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
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
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
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
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
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
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
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_make_xberg_result(),
    )
    async def test_concepts_unavailable_skips_indexing(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """When concepts_available() returns False, build_concept_records is a no-op."""
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


class TestRemoveDropsFromWikiIndex:
    """A removed document's skip marker keeps it out of later syncs, so no
    refresh would ever revisit its entries: removal has to drop them itself."""

    def test_removed_documents_leave_the_index(self, isolated_env, monkeypatch):
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.core.config import cfg
        from lilbee.wiki.entity_extractor import EntityKind
        from lilbee.wiki.stubs import WikiStub, load_stub_index, save_stub_index

        monkeypatch.setattr(cfg, "wiki", True)
        monkeypatch.setattr(cfg, "wiki_entity_min_mentions", 1)
        (isolated_env / "gone.txt").write_text("content")
        save_stub_index(
            {
                "ford": WikiStub(
                    slug="ford",
                    label="Ford",
                    kind=EntityKind.ENTITY,
                    type_hint="PERSON",
                    source_mentions=(("gone.txt", 2),),
                    chunk_refs=(("gone.txt", 0),),
                )
            }
        )
        monkeypatch.setattr(
            "lilbee.app.ingest.get_services",
            lambda: mock.MagicMock(
                store=mock.MagicMock(
                    remove_documents=mock.MagicMock(
                        return_value=mock.MagicMock(removed=["gone.txt"], not_found=[])
                    )
                )
            ),
        )

        remove_documents_durably(["gone.txt"])

        assert load_stub_index() == {}

    def test_the_hook_is_skipped_when_the_wiki_is_off(self, isolated_env, monkeypatch):
        from lilbee.app import ingest as ingest_mod
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "wiki", False)
        called: list[set] = []
        monkeypatch.setattr(
            "lilbee.wiki.stubs.drop_sources_from_index", lambda names: called.append(names)
        )
        ingest_mod._forget_removed_from_wiki_index(["gone.txt"])
        assert called == []

    def test_an_index_failure_does_not_fail_the_removal(self, isolated_env, monkeypatch):
        """The removal already succeeded; a wiki failure must not surface as one."""
        from lilbee.app import ingest as ingest_mod
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "wiki", True)
        monkeypatch.setattr(
            "lilbee.wiki.stubs.drop_sources_from_index",
            mock.MagicMock(side_effect=RuntimeError("index unwritable")),
        )
        ingest_mod._forget_removed_from_wiki_index(["gone.txt"])

    def test_removing_nothing_touches_no_index(self, isolated_env, monkeypatch):
        from lilbee.app import ingest as ingest_mod
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "wiki", True)
        called: list[set] = []
        monkeypatch.setattr(
            "lilbee.wiki.stubs.drop_sources_from_index", lambda names: called.append(names)
        )
        ingest_mod._forget_removed_from_wiki_index([])
        assert called == []


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
        assert config["ocr"].backend_options["req"] == "tok-123"

    def test_force_ocr_off_by_default(self, isolated_env, monkeypatch):
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.delenv("LILBEE_OCR_FORCE", raising=False)
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        config = extraction_config(ExtractMode.PAGINATED)
        assert config.get("force_ocr") is False

    @pytest.mark.parametrize("mode_name", ["PAGINATED", "MARKDOWN"])
    def test_force_ocr_set_when_env_and_vision_model(self, isolated_env, monkeypatch, mode_name):
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setenv("LILBEE_OCR_FORCE", "1")
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        config = extraction_config(getattr(ExtractMode, mode_name))
        assert config["force_ocr"] is True

    def test_force_ocr_ignored_without_vision_model(self, isolated_env, monkeypatch):
        # Tesseract path: force is a vision/GPU re-OCR lever, so it stays off here.
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setenv("LILBEE_OCR_FORCE", "1")
        cfg.vision_model = ""
        config = extraction_config(ExtractMode.PAGINATED)
        assert config["ocr"].backend == "tesseract"
        assert config.get("force_ocr") is False

    def test_force_ocr_ignored_when_ocr_disabled(self, isolated_env, monkeypatch):
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setenv("LILBEE_OCR_FORCE", "1")
        cfg.enable_ocr = False
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        config = extraction_config(ExtractMode.PAGINATED)
        assert config["ocr"].enabled is False
        assert config.get("force_ocr") is False


class _CountingVisionBackend:
    """Minimal xberg custom OCR backend registered as 'lilbee-vision' that records
    how many pages were sent to OCR. Used to observe whether force_ocr defeats
    xberg's text-layer short-circuit."""

    def __init__(self) -> None:
        self.calls = 0

    def name(self):
        from lilbee.data.types import OcrBackendName

        return OcrBackendName.LILBEE_VISION

    def version(self):
        return "0"

    def supported_languages(self):
        return []

    def supports_language(self, _lang):
        return True

    def initialize(self): ...

    def shutdown(self): ...

    def backend_type(self):
        from xberg import OcrBackendType

        return OcrBackendType.CUSTOM

    def supports_table_detection(self):
        return False

    def supports_document_processing(self):
        return False

    def emits_structured_markdown(self):
        return False

    def process_image(self, _image_bytes, _config):
        from xberg import ExtractedDocument

        from lilbee.data.types import MARKDOWN_MIME

        self.calls += 1
        return ExtractedDocument(content="OCR-TEXT", mime_type=MARKDOWN_MIME)

    def process_image_file(self, _path, config):
        return self.process_image(b"", config)

    def process_document(self, _path, _config):
        raise NotImplementedError


class TestForceOcrRoutesToBackend:
    """Behavioral contract against real xberg: LILBEE_OCR_FORCE must OCR every page
    of a born-digital PDF through the registered vision backend, defeating xberg's
    text-layer short-circuit. This is the guard that a future xberg bump can't
    silently disable forced re-OCR (the way rc25's short-circuit did)."""

    @staticmethod
    def _extract(pdf: bytes, config) -> tuple[int, str]:
        """Extract under a fake 'lilbee-vision' backend; return (ocr_calls, content)."""
        from xberg import register_ocr_backend, unregister_ocr_backend

        from lilbee.data.extract.xberg import extract_document
        from lilbee.data.types import OcrBackendName

        backend = _CountingVisionBackend()
        register_ocr_backend(backend)
        try:
            doc = extract_document(pdf, "application/pdf", filename="born.pdf", config=config)
            return backend.calls, doc.content or ""
        finally:
            unregister_ocr_backend(OcrBackendName.LILBEE_VISION)

    def test_native_text_skips_ocr_without_force(self, isolated_env, monkeypatch):
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.delenv("LILBEE_OCR_FORCE", raising=False)
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        config = extraction_config(ExtractMode.PAGINATED)
        calls, content = self._extract(make_pdf(pages=2), config)
        assert calls == 0
        assert "clean native text layer" in content

    def test_force_ocrs_every_page(self, isolated_env, monkeypatch):
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setenv("LILBEE_OCR_FORCE", "1")
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        config = extraction_config(ExtractMode.PAGINATED)
        calls, content = self._extract(make_pdf(pages=2), config)
        assert calls == 2  # one OCR call per page, text layer notwithstanding
        assert "OCR-TEXT" in content
        assert "clean native text layer" not in content


class TestChunkAndEmbedPagesEmpty:
    async def test_empty_page_texts_returns_empty(self):
        from lilbee.data.extract.document import chunk_and_embed_pages
        from lilbee.runtime.progress import noop_callback

        assert await chunk_and_embed_pages([], "s", "pdf", noop_callback) == []

    async def test_whitespace_pages_yield_no_chunks(self, mock_svc):
        from lilbee.data.extract.document import chunk_and_embed_pages
        from lilbee.runtime.progress import noop_callback

        assert await chunk_and_embed_pages([(1, "   ")], "s", "pdf", noop_callback) == []


class TestIngestDocumentOcrPath:
    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_empty_pdf_warns_no_usable_text(self, mock_kf, isolated_env, mock_svc, caplog):
        mock_kf.return_value = mock.MagicMock(chunks=[], metadata=Metadata())
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        result, _ = await ingest_document(f, "scan.pdf", "pdf")
        assert result == []
        assert "no usable text" in caplog.text

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_streams_per_page_progress_ticks(self, mock_kf, isolated_env, mock_svc):
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        result_obj = _make_xberg_result(num_chunks=1, has_pages=True)

        async def fake_extract(data, *, filename=None, config):
            # Simulate xberg calling the registered backend once per scanned page.
            from lilbee.data.extract.backends.vision_ocr import ocr_requests

            token = config["ocr"].backend_options["req"]
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


class TestTitleStamping:
    """Every produced record carries the document title; the source row its metadata.

    Guards the title path end to end: xberg's typed Metadata is folded into
    SourceMeta, stamped on every chunk row, and written to the source row. This is
    the test that catches titles silently breaking, which would poison title_search.
    """

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_extracted_metadata_flows_to_records_and_source_row(
        self, mock_kf, isolated_env, mock_svc
    ):
        mock_kf.return_value = _make_xberg_result(
            metadata=Metadata(
                title="Extracted Title",
                authors=["Ada", "Grace"],
                created_at="2020-01-01",
            )
        )
        (isolated_env / "report_2021.txt").write_text("body text")
        from lilbee.data.ingest import sync

        await sync(quiet=True)
        items = mock_svc.store.write_chunks_batch.call_args.args[0]
        item = next(it for it in items if it.source == "report_2021.txt")
        assert item.meta.title == "Extracted Title"
        assert item.meta.authors == "Ada, Grace"
        assert item.meta.created_at == "2020-01-01"
        assert all(r["title"] == "Extracted Title" for r in item.records)

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_bare_string_author_is_one_author_not_characters(
        self, mock_kf, isolated_env, mock_svc
    ):
        # xberg annotates authors as list[str] | None but does not enforce it: a PDF
        # /Author field arrives as a bare str. Joining it naively yields "A, d, a".
        mock_kf.return_value = _make_xberg_result(metadata=Metadata(authors="Ada Lovelace"))
        (isolated_env / "paper.txt").write_text("body text")
        from lilbee.data.ingest import sync

        await sync(quiet=True)
        items = mock_svc.store.write_chunks_batch.call_args.args[0]
        item = next(it for it in items if it.source == "paper.txt")
        assert item.meta.authors == "Ada Lovelace"

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_missing_metadata_falls_back_to_stem(self, mock_kf, isolated_env, mock_svc):
        mock_kf.return_value = _make_xberg_result()
        (isolated_env / "annual_wildlife_survey.txt").write_text("body text")
        from lilbee.data.ingest import sync

        await sync(quiet=True)
        items = mock_svc.store.write_chunks_batch.call_args.args[0]
        item = next(it for it in items if it.source == "annual_wildlife_survey.txt")
        assert item.meta.title == "annual wildlife survey"
        assert item.meta.authors == ""
        assert all(r["title"] == "annual wildlife survey" for r in item.records)

    async def test_markdown_title_uses_the_h1_heading(self, isolated_env, mock_svc):
        (isolated_env / "meeting_notes.md").write_text("# Project Kickoff\n\nSome content here.")
        from lilbee.data.ingest import sync

        await sync(quiet=True)
        items = mock_svc.store.write_chunks_batch.call_args.args[0]
        item = next(it for it in items if it.source == "meeting_notes.md")
        assert item.meta.title == "Project Kickoff"
        assert all(r["title"] == "Project Kickoff" for r in item.records)

    async def test_markdown_without_h1_falls_back_to_stem(self, isolated_env, mock_svc):
        (isolated_env / "meeting_notes.md").write_text("Some content, no heading.")
        from lilbee.data.ingest import sync

        await sync(quiet=True)
        items = mock_svc.store.write_chunks_batch.call_args.args[0]
        item = next(it for it in items if it.source == "meeting_notes.md")
        assert item.meta.title == "meeting notes"


class TestDetectMoves:
    def _entry(self, name, fhash):
        from lilbee.data.types import FileToProcess

        return FileToProcess(name, Path(name), "text", fhash, needs_cleanup=True, stat=None)

    def _record(self, name, fhash):
        return {"filename": name, "file_hash": fhash}

    def test_matches_new_file_to_removed_source_by_hash(self):
        from lilbee.data.ingest import pipeline

        entry = self._entry("new/a.txt", "h1")
        files = [entry]
        added = {"new/a.txt": None}
        to_remove = ["old/a.txt"]
        existing = {"old/a.txt": self._record("old/a.txt", "h1")}

        moves = pipeline._detect_moves(files, added, pipeline._MovePool(to_remove, existing))
        assert len(moves) == 1
        assert moves[0].old == "old/a.txt"
        assert moves[0].new == "new/a.txt"

    def test_changed_content_is_not_a_move(self):
        from lilbee.data.ingest import pipeline

        files = [self._entry("new/a.txt", "h2")]
        added = {"new/a.txt": None}
        existing = {"old/a.txt": self._record("old/a.txt", "h1")}

        moves = pipeline._detect_moves(files, added, pipeline._MovePool(["old/a.txt"], existing))
        assert moves == []

    def test_update_is_not_a_move(self):
        from lilbee.data.ingest import pipeline

        # Same name (an update, not in `added`) is never a move even on hash match.
        files = [self._entry("a.txt", "h1")]
        existing = {"a.txt": self._record("a.txt", "h1")}
        moves = pipeline._detect_moves(files, {}, pipeline._MovePool([], existing))
        assert moves == []

    def test_duplicate_hash_pairs_one_to_one(self):
        from lilbee.data.ingest import pipeline

        files = [self._entry("new/a.txt", "h1"), self._entry("new/b.txt", "h1")]
        added = {"new/a.txt": None, "new/b.txt": None}
        existing = {
            "old/a.txt": self._record("old/a.txt", "h1"),
            "old/b.txt": self._record("old/b.txt", "h1"),
        }
        moves = pipeline._detect_moves(
            files, added, pipeline._MovePool(["old/a.txt", "old/b.txt"], existing)
        )
        assert {m.new for m in moves} == {"new/a.txt", "new/b.txt"}
        assert {m.old for m in moves} == {"old/a.txt", "old/b.txt"}


class TestSyncResultRender:
    def test_str_includes_relocated_line(self):
        from lilbee.data.types import SyncResult

        r = SyncResult(added=[], updated=[], removed=[], unchanged=0, relocated=["a.md", "b.md"])
        assert "Relocated: 2" in str(r)


class TestResolveSourcePath:
    def test_owned_key_resolves_under_documents_dir(self, isolated_env):
        from lilbee.data.ingest.discovery import resolve_source_path

        assert resolve_source_path("notes/a.md") == cfg.documents_dir / "notes/a.md"

    def test_dir_root_key_resolves_under_the_root(self, isolated_env, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path

        root = tmp_path / "corpus"
        cfg.linked_roots = {"corpus": str(root)}
        assert resolve_source_path("corpus/sub/a.pdf") == root / "sub" / "a.pdf"

    def test_file_root_key_resolves_to_the_file(self, isolated_env, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path

        f = tmp_path / "report.pdf"
        cfg.linked_roots = {"report.pdf": str(f)}
        assert resolve_source_path("report.pdf") == f

    def test_checked_resolver_rejects_escaping_key(self, isolated_env, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path_checked

        root = tmp_path / "corpus"
        root.mkdir()
        cfg.linked_roots = {"corpus": str(root)}
        assert resolve_source_path_checked("corpus/a.pdf") == (root / "a.pdf").resolve()
        assert resolve_source_path_checked("corpus/../../etc/passwd") is None


class TestSourceLabelTaken:
    def test_true_for_live_registered_root(self, isolated_env, tmp_path):
        from lilbee.app.ingest import source_label_taken

        root = tmp_path / "corpus"
        root.mkdir()
        cfg.linked_roots = {"corpus": str(root)}
        assert source_label_taken("corpus") is True

    def test_false_for_dangling_registered_root(self, isolated_env, tmp_path):
        from lilbee.app.ingest import source_label_taken

        cfg.linked_roots = {"corpus": str(tmp_path / "gone")}
        assert source_label_taken("corpus") is False

    def test_true_for_owned_documents_entry(self, isolated_env):
        from lilbee.app.ingest import source_label_taken

        (cfg.documents_dir / "owned.md").write_text("x")
        assert source_label_taken("owned.md") is True

    def test_false_for_unknown_name(self, isolated_env):
        from lilbee.app.ingest import source_label_taken

        assert source_label_taken("nope") is False


class TestRegisteredRootHelpers:
    def test_unregister_roots_skips_nested_and_removes_top_level(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources, unregister_roots
        from lilbee.core.config import cfg

        source = tmp_path / "corpus"
        source.mkdir()
        register_sources([source])  # persists the root, mirroring real usage

        # A nested name is not a root and is left alone; the label is un-registered.
        removed = unregister_roots(["corpus/a.txt", "corpus"])

        assert removed == ["corpus"]
        assert "corpus" not in cfg.linked_roots
        assert source.exists()  # source bytes untouched

    def test_unregister_roots_ignores_unknown_label(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources, unregister_roots
        from lilbee.core.config import cfg

        source = tmp_path / "corpus"
        source.mkdir()
        register_sources([source])
        assert unregister_roots(["not-a-root"]) == []
        assert "corpus" in cfg.linked_roots

    def test_unregister_empty_is_a_noop(self, isolated_env):
        from lilbee.app.ingest import unregister_roots

        assert unregister_roots([]) == []

    def test_register_empty_is_a_noop(self, isolated_env):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        result = register_sources([])
        assert result.registered == []
        assert result.skipped == []
        assert cfg.linked_roots == {}


class TestRegisterSources:
    def test_registers_dir_root_by_basename(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        result = register_sources([corpus])
        assert result.registered == ["corpus"]
        assert cfg.linked_roots == {"corpus": str(corpus.resolve())}

    def test_reregistering_same_path_is_idempotent(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        register_sources([corpus])
        result = register_sources([corpus])
        assert result.registered == []
        assert result.skipped == ["corpus"]
        assert cfg.linked_roots == {"corpus": str(corpus.resolve())}

    def test_label_collision_needs_force(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        one = tmp_path / "a" / "corpus"
        one.mkdir(parents=True)
        two = tmp_path / "b" / "corpus"
        two.mkdir(parents=True)
        register_sources([one])
        # Same basename, different live path: skipped without force.
        result = register_sources([two])
        assert result.registered == []
        assert result.skipped == ["corpus"]
        assert cfg.linked_roots == {"corpus": str(one.resolve())}
        # force overwrites the label.
        forced = register_sources([two], force=True)
        assert forced.registered == ["corpus"]
        assert cfg.linked_roots == {"corpus": str(two.resolve())}

    def test_dangling_root_relinks_without_force(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core import settings
        from lilbee.core.config import cfg

        # A root registered to a path that no longer exists (the source moved):
        # re-adding the new location re-points the label, no force needed.
        settings.set_value(
            cfg.data_root, "linked_roots", {"corpus": str(tmp_path / "old" / "corpus")}
        )
        moved = tmp_path / "new" / "corpus"
        moved.mkdir(parents=True)
        result = register_sources([moved])
        assert result.registered == ["corpus"]
        assert cfg.linked_roots == {"corpus": str(moved.resolve())}

    def test_path_inside_documents_dir_left_to_owned_walk(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        inside = cfg.documents_dir / "sub"
        inside.mkdir()
        result = register_sources([inside])
        assert result.registered == []
        assert result.skipped == ["sub"]
        assert cfg.linked_roots == {}

    def test_registration_persists_to_config_toml(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core import settings
        from lilbee.core.config import cfg

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        register_sources([corpus])
        persisted = settings.load(cfg.data_root)
        assert persisted.get("linked_roots") == {"corpus": str(corpus.resolve())}

    def test_register_merges_with_concurrently_persisted_root(self, isolated_env, tmp_path):
        # Another process persisted a root to config.toml while our in-memory view
        # is stale; register must merge, not clobber it (no lost update).
        from lilbee.app.ingest import register_sources
        from lilbee.core import settings
        from lilbee.core.config import cfg

        other = tmp_path / "a" / "other"
        other.mkdir(parents=True)
        settings.set_value(cfg.data_root, "linked_roots", {"other": str(other)})
        cfg.linked_roots = {}  # stale snapshot: does not know about "other"

        mine = tmp_path / "b" / "mine"
        mine.mkdir(parents=True)
        register_sources([mine])

        expected = {"other": str(other), "mine": str(mine.resolve())}
        assert settings.load(cfg.data_root)["linked_roots"] == expected
        assert cfg.linked_roots == expected

    def test_rejects_root_nested_under_existing_root(self, isolated_env, tmp_path):
        # Registering a child of an existing root would walk (and index) the same
        # file under two keys; it is skipped.
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        corpus = tmp_path / "corpus"
        (corpus / "papers").mkdir(parents=True)
        register_sources([corpus])
        result = register_sources([corpus / "papers"])
        assert result.registered == []
        assert result.skipped == ["papers"]
        assert cfg.linked_roots == {"corpus": str(corpus.resolve())}

    def test_rejects_root_containing_existing_root(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        child = tmp_path / "data" / "corpus"
        child.mkdir(parents=True)
        register_sources([child])
        result = register_sources([tmp_path / "data"])  # a parent of the existing root
        assert result.registered == []
        assert cfg.linked_roots == {"corpus": str(child.resolve())}

    def test_rejects_root_that_is_ancestor_of_documents_dir(self, isolated_env, tmp_path):
        # documents_dir lives under tmp_path; registering tmp_path would re-index
        # every owned file a second time under the parent label.
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        result = register_sources([cfg.documents_dir.parent])
        assert result.registered == []
        assert cfg.linked_roots == {}

    def test_force_never_shadows_owned_documents_entry(self, isolated_env, tmp_path):
        # An owned top-level entry must never be shadowed by a same-named root,
        # even under --force, or resolve_source_path would disagree with discovery.
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        (cfg.documents_dir / "reports").mkdir(parents=True)
        external = tmp_path / "reports"
        external.mkdir()
        result = register_sources([external], force=True)
        assert result.registered == []
        assert result.skipped == ["reports"]
        assert cfg.linked_roots == {}
