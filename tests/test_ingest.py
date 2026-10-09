"""Tests for the document sync engine (mocked: no live server needed)."""

import asyncio
import contextlib
import json
import os
import sys
import threading
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import rich.progress
from xberg import Metadata

import lilbee.app.services as svc_mod
from lilbee.app.ingest import RegisterResult
from lilbee.core.config import cfg
from lilbee.core.config.enums import OcrMode
from lilbee.data.types import ExtractMode, OcrBackendName
from lilbee.runtime.progress import OcrBackendUsed
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
        from lilbee.data.store.types import RemoveResult

        removed = [n for n in names if n in _sources]
        not_found = [n for n in names if n not in _sources]
        for name in removed:
            _sources.pop(name, None)
        return RemoveResult(removed=removed, not_found=not_found)

    store.remove_documents.side_effect = _remove_documents

    def _relocate(moves):
        # Mirror the real re-key: a move takes its first candidate that holds a row.
        taken = {}
        for move in moves:
            old = next((name for name in move.candidates if name in _sources), None)
            if old is not None:
                _sources[move.new] = {**_sources.pop(old), "filename": move.new}
                taken[move.new] = old
        return taken

    store.relocate_sources.side_effect = _relocate
    store.drop_all.side_effect = lambda: _sources.clear()
    store.ensure_fts_index.return_value = None
    store.get_meta.return_value = None
    store.index_mismatch.return_value = None
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
    from tests.conftest import one_plan_batch

    return _ResultFeed(one_plan_batch(coros))


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


def _member(path, mime_type, result):
    """One archive child as xberg reports it: path, MIME type, full result."""
    return mock.MagicMock(path=path, mime_type=mime_type, result=result)


def _make_archive_result(children):
    """An archive's extraction result: a listing the index ignores, plus its members."""
    result = mock.MagicMock()
    result.chunks = []
    result.content = "ZIP Archive"
    result.tables = []
    result.pages = []
    result.metadata = Metadata()
    result.children = children
    return result


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
    result.children = None
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
    missing = _reconcile_missing(
        disk, [{"filename": "a.pdf"}], failed=["b.pdf"], skipped=["c.pdf"], held=[]
    )
    assert missing == ["d.pdf"]


def test_reconcile_missing_accounts_for_persisted_skip_markers():
    """A file an earlier run marked is in neither this run's failed nor skipped
    lists, so without the marker set it reads as a silent drop every sync."""
    from pathlib import Path

    from lilbee.data.ingest.pipeline import _reconcile_missing

    disk = {n: Path(n) for n in ("held.md", "dropped.md")}
    missing = _reconcile_missing(disk, [], failed=[], skipped=[], held={"held.md": "deadbeef"})
    assert missing == ["dropped.md"]


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

    async def test_sync_reports_an_index_built_with_another_embedder(
        self, mock_extract_file, isolated_env, mock_svc
    ):
        """An unchanged corpus never reaches the write gate, so the sync used to
        finish green over an index search refuses. The result names the drift
        and the index is left as it is: a rebuild is the caller's call."""
        from lilbee.data.ingest import sync
        from lilbee.data.store import EmbeddingModelMismatchError

        mock_svc.store.index_mismatch.return_value = EmbeddingModelMismatchError(
            persisted_model="old/embed-GGUF/old.gguf",
            persisted_dim=768,
            current_model="new/embed-GGUF/new.gguf",
            current_dim=384,
        )

        result = await sync()

        assert result.index_mismatch is not None
        assert result.index_mismatch.persisted_model == "old/embed-GGUF/old.gguf"
        assert result.index_mismatch.current_model == "new/embed-GGUF/new.gguf"
        assert result.index_mismatch.adoptable is False
        assert "old/embed-GGUF/old.gguf" in str(result)
        mock_svc.store.drop_all.assert_not_called()

    async def test_ingest_text_file(self, mock_extract_file, isolated_env):
        (isolated_env / "test.txt").write_text("Hello world. This is a test document.")
        from lilbee.data.ingest import sync

        result = await sync()
        assert "test.txt" in result.added
        mock_extract_file.assert_called()
        assert any("test.txt" in str(call) for call in mock_extract_file.call_args_list)

    async def test_sync_uses_batch_extraction_when_enabled(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        """With the toggle on, sync extracts through extract_batch, not single extract."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync

        monkeypatch.setattr(cfg, "batch_extraction", True)
        (isolated_env / "d.txt").write_text("Hello world. Batch extraction test document.")

        async def fake_batch(inputs, _config, _on_progress):
            res = mock.MagicMock()
            res.results = [_make_xberg_result() for _ in inputs]
            res.errors = []
            return res

        with mock.patch("xberg.progress.extract_batch", side_effect=fake_batch) as mock_batch:
            result = await sync()

        assert "d.txt" in result.added
        mock_batch.assert_called()
        mock_extract_file.assert_not_called()

    async def test_a_move_subtracts_the_old_name_from_the_wiki_index(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        """The index is keyed by source name. Without the old key the move
        leaves it there forever: its mentions double-count and its dead chunk
        refs occupy the per-subject cap ahead of live evidence."""
        import shutil

        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync

        (isolated_env / "a.txt").write_text("Hello world. This document will move.")
        await sync()

        monkeypatch.setattr(cfg, "wiki", True)
        seen: list[set] = []

        def fake_refresh(store, config=None, *, sources=None):
            seen.append(sources or set())
            return {}

        monkeypatch.setattr("lilbee.wiki.stubs.refresh_stub_index", fake_refresh)
        (isolated_env / "sub").mkdir()
        shutil.move(str(isolated_env / "a.txt"), str(isolated_env / "sub" / "a.txt"))

        await sync()

        assert seen, "the post-sync refresh did not run"
        assert {"a.txt", "sub/a.txt"} <= seen[-1]

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
        assert [(move.candidates, move.new) for move in moves] == [(("a.txt",), "sub/a.txt")]

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

    async def test_archive_members_are_indexed_as_their_own_sources(
        self, mock_extract_file, isolated_env, caplog
    ):
        """Each member is a source named archive/member; the archive itself has no chunks."""
        import logging

        from lilbee.app.services import get_services
        from lilbee.data.ingest import sync

        mock_extract_file.return_value = _make_archive_result(
            [
                _member(
                    "report.pdf",
                    "application/pdf",
                    _make_xberg_result(num_chunks=2, has_pages=True),
                ),
                _member("notes.txt", "text/plain", _make_xberg_result(num_chunks=1)),
            ]
        )
        (isolated_env / "docs.zip").write_bytes(b"PK\x03\x04")

        with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"):
            result = await sync(quiet=True)

        assert result.added == ["docs.zip"]
        assert result.skipped == []
        rows = {s["filename"]: s["chunk_count"] for s in get_services().store.get_sources()}
        assert rows == {"docs.zip": 0, "docs.zip/report.pdf": 2, "docs.zip/notes.txt": 1}
        assert not any("reconciliation" in r.getMessage().lower() for r in caplog.records)

    async def test_member_past_the_chunk_cap_holds_the_archive_out_and_names_it(
        self, mock_extract_file, isolated_env, monkeypatch, mock_svc
    ):
        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import load_skip_reasons

        monkeypatch.setattr(cfg, "max_chunks_per_file", 2)
        mock_extract_file.return_value = _make_archive_result(
            [_member("runs.csv", "text/csv", _make_xberg_result(num_chunks=5))]
        )
        (isolated_env / "runs.gz").write_bytes(b"\x1f\x8b")

        result = await sync(quiet=True)

        assert result.skipped == ["runs.gz"]
        assert result.added == []
        mock_svc.embedder.embed_batch.assert_not_called()
        reason = load_skip_reasons(cfg.data_root)["runs.gz"]
        assert reason.startswith("runs.gz/runs.csv: 5 chunks exceed the per-file limit of 2")

    async def test_chunk_explosion_is_skipped_before_it_is_embedded(
        self, mock_extract_file, isolated_env, monkeypatch, mock_svc
    ):
        """A file past the cap is refused with its count, and never reaches the embedder."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import load_skip_reasons
        from lilbee.runtime.progress import BatchStatus, EventType

        monkeypatch.setattr(cfg, "max_chunks_per_file", 2)
        mock_extract_file.return_value = _make_xberg_result(num_chunks=5)
        (isolated_env / "huge.txt").write_text(
            "A document that chunks far past the cap.", encoding="utf-8"
        )

        events: list = []
        result = await sync(quiet=True, on_progress=lambda t, d: events.append((t, d)))

        assert result.skipped == ["huge.txt"]
        assert result.added == []
        mock_svc.embedder.embed_batch.assert_not_called()
        assert any(
            t is EventType.BATCH_PROGRESS and d.status is BatchStatus.SKIPPED for t, d in events
        )
        reason = load_skip_reasons(cfg.data_root)["huge.txt"]
        assert "5 chunks exceed the per-file limit of 2" in reason

    async def test_raising_the_chunk_cap_lets_the_file_in(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        """The cap is a setting: lift it, retry the skipped file, and it indexes."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import sync

        monkeypatch.setattr(cfg, "max_chunks_per_file", 2)
        mock_extract_file.return_value = _make_xberg_result(num_chunks=5)
        (isolated_env / "huge.txt").write_text(
            "A document that chunks far past the cap.", encoding="utf-8"
        )
        assert (await sync(quiet=True)).skipped == ["huge.txt"]

        monkeypatch.setattr(cfg, "max_chunks_per_file", 0)
        second = await sync(quiet=True, retry_skipped=True)

        assert second.added == ["huge.txt"]
        assert second.skipped == []

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
        """The auto-update gate and the regeneration must consult one config.

        Run under a bound scope, so the scoped config and the process-global
        are different objects. Without a scope active_config() returns the
        global itself, and a hook handed the global instead of the config the
        gate read would satisfy the assertion. The library API binds a scope
        around the whole pipeline, so this is the shape a Lilbee(config=...)
        caller actually runs in."""
        from lilbee.core.config import cfg, config_scope
        from lilbee.data.ingest import sync

        (isolated_env / "wikified.txt").write_text("Content the wiki hook would summarize.")
        monkeypatch.setattr(cfg, "wiki", False)
        scoped = cfg.model_copy()
        scoped.wiki = True
        scoped.wiki_auto_update = True
        hook = mock.AsyncMock()
        monkeypatch.setattr("lilbee.wiki.ingest.incremental_update", hook)
        refresh = mock.MagicMock(return_value={})
        monkeypatch.setattr("lilbee.wiki.stubs.refresh_stub_index", refresh)

        with config_scope(scoped):
            await sync(quiet=True)

        # Both calls, not just the regeneration. The index refresh is the one
        # that always runs when the wiki is on, since regeneration sits behind
        # wiki_auto_update, and it falls back to the process-global when handed
        # None.
        assert refresh.call_args.args[1] is scoped
        assert hook.call_args.args[1] is scoped

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
        assert "sync_done" in event_types
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

    async def test_cli_bar_shows_the_per_file_events_of_a_sync(
        self, mock_extract_file, isolated_env
    ):
        """The bar of a non-quiet sync renders the OCR-start, page and chunk events a file emits."""
        from rich.progress import Progress

        from lilbee.data.ingest import sync
        from lilbee.data.types import DocumentRecords, OcrReport, SourceMeta
        from lilbee.runtime.progress import (
            EmbedEvent,
            EventType,
            ExtractEvent,
            OcrStartEvent,
        )

        (isolated_env / "scan.pdf").write_bytes(b"%PDF-1.4 scanned")

        async def fake_ingest_document(path, source_name, content_type, *, on_progress, **_kw):
            on_progress(EventType.OCR_START, OcrStartEvent(file=source_name, total_pages=8))
            on_progress(
                EventType.EXTRACT,
                ExtractEvent(
                    file=source_name,
                    page=8,
                    total_pages=8,
                    ocr_backend=OcrBackendUsed.TESSERACT,
                ),
            )
            on_progress(EventType.EMBED, EmbedEvent(file=source_name, chunk=3, total_chunks=5))
            return DocumentRecords(
                [], SourceMeta(title="scan"), OcrReport(backend=OcrBackendUsed.TESSERACT)
            )

        descriptions: list[str] = []
        totals: set[float | None] = set()
        real_update = Progress.update

        def spy_update(self, task_id, **kwargs):
            if "description" in kwargs:
                descriptions.append(kwargs["description"])
            totals.add(self.tasks[0].total)
            return real_update(self, task_id, **kwargs)

        forwarded: list[object] = []
        with (
            mock.patch(
                "lilbee.data.ingest.pipeline.ingest_document", side_effect=fake_ingest_document
            ),
            mock.patch.object(Progress, "update", spy_update),
        ):
            result = await sync(quiet=False, on_progress=lambda et, _d: forwarded.append(et))

        mock_extract_file.assert_not_called()  # the stub replaced the whole extraction
        assert result.skipped == ["scan.pdf"]  # the stub's zero records mark the file skipped
        assert totals == {1}  # the bar measures the one-file corpus

        assert descriptions == [
            "Tesseract OCR on scan.pdf (8 pages in the file)",
            "Tesseract OCR scan.pdf (page 8/8)",
            "Embedding scan.pdf (3/5)",
            "Ingested scan.pdf",
        ]
        # The caller's own callback still receives every per-file event.
        per_file = (EventType.OCR_START, EventType.EXTRACT, EventType.EMBED)
        assert tuple(et for et in forwarded if et in per_file) == per_file

    @pytest.mark.parametrize("quiet", [True, False])
    async def test_sync_draws_the_bar_only_when_not_quiet(
        self, mock_extract_file, quiet, isolated_env, capsys
    ):
        """A quiet sync (JSON output, TUI) writes nothing; a non-quiet one draws the bar."""
        (isolated_env / "bar.txt").write_text("Bar or no bar.", encoding="utf-8")
        from lilbee.data.ingest import sync

        result = await sync(quiet=quiet)

        assert "bar.txt" in result.added
        assert (capsys.readouterr().out == "") is quiet

    @pytest.mark.parametrize("quiet", [True, False])
    async def test_the_bar_never_uses_rich_s_global_console(
        self, mock_extract_file, quiet, isolated_env, capsys, monkeypatch
    ):
        """Rich's global console fixes its terminal detection when first built; a bar skips it."""

        def _global_console():
            raise AssertionError("the ingest bar asked for rich's global console")

        monkeypatch.setattr(rich.progress, "get_console", _global_console)
        (isolated_env / "bar.txt").write_text("Bar or no bar.", encoding="utf-8")
        from lilbee.data.ingest import sync

        result = await sync(quiet=quiet)

        assert "bar.txt" in result.added
        assert (capsys.readouterr().out == "") is quiet

    async def test_batch_progress_measures_the_corpus_not_the_plan_so_far(
        self, mock_extract_file, isolated_env
    ):
        # Two files indexed, one then edited. The re-sync replans only the edited
        # file, so a total taken from the plan reads 1/1 and says nothing about
        # the corpus. The total is the files on disk, and the untouched file
        # counts as done because it is.
        (isolated_env / "a.txt").write_text("first")
        (isolated_env / "b.txt").write_text("second")
        from lilbee.data.ingest import sync

        await sync(quiet=True)
        (isolated_env / "b.txt").write_text("second, edited")
        events: list[tuple] = []
        await sync(quiet=True, on_progress=lambda t, d: events.append((t, d)))
        batch = [d for t, d in events if t == "batch_progress"]
        assert [(d.current, d.total) for d in batch] == [(2, 2)]

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
class TestSyncDropsNewlyIgnored:
    """A .lilbeeignore pattern added after ingest removes what it now excludes."""

    async def _ingest_two(self, isolated_env):
        from lilbee.data.ingest import sync

        (isolated_env / "keep.txt").write_text("a document worth keeping", encoding="utf-8")
        (isolated_env / "drop.txt").write_text("a document about to be excluded", encoding="utf-8")
        result = await sync()
        assert sorted(result.added) == ["drop.txt", "keep.txt"]

    async def test_a_pattern_added_after_ingest_leaves_the_source_indexed(
        self, mock_extract_file, isolated_env
    ):
        """The patterns govern what sync takes in; they never drop what it already has."""
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        await self._ingest_two(isolated_env)
        (isolated_env / IGNORE_FILENAME).write_text("drop.txt\n", encoding="utf-8")

        result = await sync()
        assert result.removed == []
        store = svc_mod.get_services().store
        assert {s["filename"] for s in store.get_sources()} == {"drop.txt", "keep.txt"}
        store.remove_documents.assert_not_called()

    async def test_a_refused_format_already_indexed_is_removed(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        """A drawing an earlier version indexed leaves the index on the next sync."""
        from lilbee.data.ingest import discovery, sync

        (isolated_env / "keep.txt").write_text("a document worth keeping", encoding="utf-8")
        (isolated_env / "logo.svg").write_text("<svg/>", encoding="utf-8")
        discovery.supported_extension_map.cache_clear()
        monkeypatch.setattr(discovery, "excluded_extension_reasons", lambda: {})
        first = await sync()
        monkeypatch.undo()
        discovery.supported_extension_map.cache_clear()
        assert sorted(first.added) == ["keep.txt", "logo.svg"]

        result = await sync()
        assert result.removed == ["logo.svg"]
        assert result.skipped == ["logo.svg"]
        store = svc_mod.get_services().store
        assert {s["filename"] for s in store.get_sources()} == {"keep.txt"}

    async def test_prune_ignored_drops_the_source_on_request(self, mock_extract_file, isolated_env):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        await self._ingest_two(isolated_env)
        (isolated_env / IGNORE_FILENAME).write_text("drop.txt\n", encoding="utf-8")

        result = await sync(prune_ignored=True)
        assert result.removed == ["drop.txt"]
        assert {s["filename"] for s in svc_mod.get_services().store.get_sources()} == {"keep.txt"}

    async def test_a_removal_only_sync_rebuilds_the_clusters(
        self, mock_extract_file, isolated_env, monkeypatch
    ):
        """A removed source leaves stale concept nodes behind unless Leiden runs again."""
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        monkeypatch.setattr(cfg, "concept_graph", True)
        concepts = svc_mod.get_services().concepts
        concepts.get_graph.return_value = True
        await self._ingest_two(isolated_env)
        concepts.rebuild_clusters.reset_mock()
        (isolated_env / IGNORE_FILENAME).write_text("drop.txt\n", encoding="utf-8")

        with mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True):
            result = await sync(prune_ignored=True)

        assert result.removed == ["drop.txt"]
        assert result.added == [] and result.updated == []
        concepts.rebuild_clusters.assert_called_once_with()

    async def test_a_shard_worker_leaves_removal_to_the_parent(
        self, mock_extract_file, isolated_env
    ):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.ignore import IGNORE_FILENAME
        from lilbee.data.types import ShardId

        await self._ingest_two(isolated_env)
        (isolated_env / IGNORE_FILENAME).write_text("drop.txt\n", encoding="utf-8")

        # A worker sees one slice of the corpus but the whole sources table, so
        # running this pass per worker would race k ways over sources it does
        # not own. Every shard must leave the index untouched here.
        store = svc_mod.get_services().store
        for index in range(2):
            result = await sync(
                prune_ignored=True, shard=ShardId(index=index, count=2, records_root=cfg.data_root)
            )
            assert result.removed == []
        store.remove_documents.assert_not_called()
        assert {s["filename"] for s in store.get_sources()} == {"drop.txt", "keep.txt"}


@mock.patch(
    "lilbee.data.extract.xberg.aextract_document",
    new_callable=mock.AsyncMock,
    return_value=_make_xberg_result(),
)
class TestSyncCancellation:
    """Tests for cancel support and atomic per-file delete in sync."""

    async def test_cancel_stops_file_discovery(
        self, mock_extract_file, isolated_env, mock_svc, caplog
    ):
        """A cancel set before sync starts processes no file and raises the cancel."""
        import asyncio
        import threading

        (isolated_env / "a.txt").write_text("file a")
        (isolated_env / "b.txt").write_text("file b")
        from lilbee.data.ingest import sync

        cancel = threading.Event()
        cancel.set()
        with caplog.at_level("WARNING"), pytest.raises(asyncio.CancelledError):
            await sync(quiet=True, cancel=cancel)
        mock_svc.store.write_chunks_batch.assert_not_called()
        # The unplanned corpus is not reconciled as a silent drop.
        assert "Sync reconciliation" not in caplog.text

    async def test_cancel_during_ingest_stream(self, mock_extract_file, isolated_env, mock_svc):
        """Cancel set mid-batch raises CancelledError for pending files."""
        import asyncio
        import threading

        from lilbee.data.ingest import ingest_stream
        from tests.conftest import one_plan_batch

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
            await ingest_stream(one_plan_batch(files), added, {}, {}, {}, quiet=True, cancel=cancel)

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
        import asyncio
        import threading

        from lilbee.data.ingest import sync

        f = isolated_env / "doc.txt"
        f.write_text("Version 1")
        await sync(quiet=True)

        mock_svc.store.write_chunks_batch.reset_mock()

        f.write_text("Version 2, modified content")
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await sync(quiet=True, cancel=cancel)

        # Cancel fired before the file was processed, so nothing was written and
        # the old chunks were never deleted.
        mock_svc.store.write_chunks_batch.assert_not_called()


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
        result, _, _ = await ingest_document(f, "empty.txt", "text")
        assert result == []

    async def test_ingest_code_empty_chunks(self, isolated_env, mock_svc):
        """A code file with no chunks returns early, without reaching the embedder."""
        from unittest.mock import patch

        from lilbee.data.ingest import ingest_code_sync

        f = isolated_env / "empty.py"
        f.write_text("")
        with patch("lilbee.data.ingest.code.chunk_code", return_value=[]):
            result = ingest_code_sync(f, "empty.py")
        assert result == []
        mock_svc.embedder.embed_batch.assert_not_called()

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
        result, _, _ = await ingest_document(f, "test.pdf", "pdf")
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
            from tests.conftest import one_plan_batch

            added = {"cancel.txt": None}
            entry = FileToProcess(
                "cancel.txt", isolated_env / "cancel.txt", "text", "abc123", False
            )
            with pytest.raises(asyncio.CancelledError):
                await ingest_stream(one_plan_batch([entry]), added, {}, {}, {}, quiet=True)

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
        from tests.conftest import one_plan_batch

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
                one_plan_batch(files),
                added,
                {},
                failed,
                skipped,
                quiet=True,
                on_progress=on_progress,
            )


class TestSkipMarkerLifecycle:
    """A file that produces no chunks gets a skip marker; the next sync skips it
    until the file changes or retry_skipped / force_rebuild clears the marker."""

    @staticmethod
    def _zero_chunks(*_args, **_kwargs):
        # Simulate "OCR found no usable text": no records produced, so the file
        # is recorded as skipped.
        from lilbee.data.store import SourceMeta
        from lilbee.data.types import DocumentRecords

        return DocumentRecords([], SourceMeta())

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

    async def test_held_out_file_is_reported_apart_from_unchanged(self, isolated_env, mock_svc):
        """A marker-held file must not be tallied as unchanged: it is not indexed,
        and 'Unchanged: N' would present it as a file the index already holds."""
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.pipeline import produce_records as orig

        async def _zero_for_the_scan(path, name, content_type, **kwargs):
            if name == "scanned.pdf":
                return self._zero_chunks()
            return await orig(path, name, content_type, **kwargs)

        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        (isolated_env / "readable.txt").write_text("plenty of text", encoding="utf-8")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=_zero_for_the_scan
        ):
            first = await sync(quiet=True)
            assert "readable.txt" in first.added
            second = await sync(quiet=True)

        assert [held.filename for held in second.held_out] == ["scanned.pdf"]
        assert second.held_out[0].reason == "no text extracted (0 chunks)"
        assert second.unchanged == 1  # readable.txt only; the held file is not counted here
        assert "Held out: 1" in str(second)
        assert "no text extracted (0 chunks)" in str(second)

    async def test_held_out_file_does_not_trip_the_reconciliation_guard(
        self, isolated_env, mock_svc, caplog
    ):
        """The marker accounts for the file, so the data-loss warning must stay
        silent rather than naming the same file on every sync."""
        import logging

        from lilbee.data.ingest import sync

        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_chunks
        ):
            await sync(quiet=True)
            with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"):
                await sync(quiet=True)

        assert "possible silent drop" not in caplog.text

    async def test_stale_marker_does_not_hide_a_silent_drop(self, isolated_env, mock_svc, caplog):
        """A marker for an older hash accounts for nothing once the file changes.
        Only the files this run held out are subtracted, so an edited file that
        then vanishes without a failure is still reported."""
        import logging

        from lilbee.data.ingest import sync

        scan = isolated_env / "scanned.pdf"
        scan.write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_chunks
        ):
            await sync(quiet=True)

        async def _drop_everything(plan_batches, *_args, **_kwargs):
            async for _batch in plan_batches:
                pass

        scan.write_bytes(b"%PDF-1.4 edited since it was marked")
        with (
            mock.patch("lilbee.data.ingest.pipeline.ingest_stream", side_effect=_drop_everything),
            caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"),
        ):
            result = await sync(quiet=True)

        assert result.held_out == []
        assert "possible silent drop" in caplog.text
        assert "scanned.pdf" in caplog.text

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

    async def test_retry_skipped_leaves_a_removal_out(self, isolated_env, mock_svc):
        """retry-skipped retries failed files only; a removed source stays removed."""
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import SkipKind, load_skip_kinds

        (isolated_env / "gone.txt").write_text("removed by the user", encoding="utf-8")
        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_for("scanned.pdf")
        ):
            await sync(quiet=True)
            remove_documents_durably(["gone.txt"])
            retried = await sync(quiet=True, retry_skipped=True)

        assert "scanned.pdf" in retried.skipped
        assert "gone.txt" not in [*retried.added, *retried.updated, *retried.skipped]
        assert load_skip_kinds(cfg.data_root) == {
            "gone.txt": SkipKind.REMOVED,
            "scanned.pdf": SkipKind.FAILED,
        }

    async def test_force_rebuild_also_clears_markers(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync

        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_chunks
        ):
            await sync(quiet=True)
            rebuilt = await sync(quiet=True, force_rebuild=True)
            assert "scanned.pdf" in rebuilt.skipped  # attempted again after the wipe

    async def test_a_failure_is_recorded_as_failed(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import SkipKind, load_skip_kinds

        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_chunks
        ):
            await sync(quiet=True)

        assert load_skip_kinds(cfg.data_root) == {"scanned.pdf": SkipKind.FAILED}

    async def test_a_removal_stays_removed_and_is_not_reported_held_out(
        self, isolated_env, mock_svc
    ):
        """The sync keeps a removal's kind and leaves it out of the held-out list."""
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import SkipKind, load_skip_kinds

        (isolated_env / "gone.txt").write_text("removed later", encoding="utf-8")
        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_for("scanned.pdf")
        ):
            await sync(quiet=True)
            remove_documents_durably(["gone.txt"])
            second = await sync(quiet=True)

        assert "gone.txt" not in second.added
        assert [held.filename for held in second.held_out] == ["scanned.pdf"]
        assert load_skip_kinds(cfg.data_root) == {
            "gone.txt": SkipKind.REMOVED,
            "scanned.pdf": SkipKind.FAILED,
        }

    async def test_an_edited_removal_that_fails_becomes_a_failure(self, isolated_env, mock_svc):
        """A removed file edited on disk is retried; failing then records it as failed."""
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import SkipKind, load_skip_kinds

        doc = isolated_env / "gone.txt"
        doc.write_text("removed later", encoding="utf-8")
        await sync(quiet=True)
        remove_documents_durably(["gone.txt"])
        doc.write_text("edited after the removal", encoding="utf-8")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_chunks
        ):
            result = await sync(quiet=True)

        assert "gone.txt" in result.skipped
        assert load_skip_kinds(cfg.data_root) == {"gone.txt": SkipKind.FAILED}

    async def test_a_sync_keeps_the_stored_kind_of_a_marker_it_did_not_touch(
        self, isolated_env, mock_svc
    ):
        """The stored kind wins over the reason: a removal with any reason stays a removal."""
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import (
            SkipKind,
            SkipRecords,
            load_skip_kinds,
            write_skip_records,
        )

        kept = isolated_env / "kept.md"
        kept.write_text("removed by the user", encoding="utf-8")
        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 no text")
        write_skip_records(
            cfg.data_root,
            SkipRecords(
                markers={"kept.md": file_hash(kept)},
                reasons={"kept.md": "custom note"},
                kinds={"kept.md": SkipKind.REMOVED},
            ),
        )
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records", side_effect=self._zero_for("scanned.pdf")
        ):
            result = await sync(quiet=True)

        assert "scanned.pdf" in result.failed + result.skipped
        assert load_skip_kinds(cfg.data_root) == {
            "kept.md": SkipKind.REMOVED,
            "scanned.pdf": SkipKind.FAILED,
        }
        assert [held.filename for held in result.held_out] == []

    @classmethod
    def _zero_for(cls, *targets: str):
        from lilbee.data.ingest.pipeline import produce_records as orig

        async def _produce(path, name, content_type, **kwargs):
            if name in targets:
                return cls._zero_chunks()
            return await orig(path, name, content_type, **kwargs)

        return _produce


class TestSyncMergesItsSkipRecords:
    """A writer that changes the skip records while a sync runs keeps its change.

    Each writer runs in the gap between the sync's read of the records and its
    write-back, and the sync's own verdict for a file that produced no chunks
    still lands. A reset in that gap is refused instead.
    """

    @staticmethod
    def _writer_in_the_gap(writer):
        from lilbee.data.ingest import pipeline

        real = pipeline.ingest_stream

        async def _run_writer_then_ingest(*args, **kwargs):
            writer()
            return await real(*args, **kwargs)

        return mock.patch.object(pipeline, "ingest_stream", _run_writer_then_ingest)

    async def _sync_holding_out(self, scan: Path, writer) -> None:
        from lilbee.data.ingest import sync

        scan.write_bytes(b"%PDF-1.4 not really text")
        with (
            mock.patch(
                "lilbee.data.ingest.pipeline.produce_records",
                side_effect=TestSkipMarkerLifecycle._zero_chunks,
            ),
            self._writer_in_the_gap(writer),
        ):
            result = await sync(quiet=True)
        assert result.skipped == [scan.name]

    async def test_a_file_that_now_ingests_drops_its_record(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import (
            load_skip_markers,
            load_skip_reasons,
            write_skip_markers,
            write_skip_reasons,
        )

        (isolated_env / "fixed.txt").write_text("readable now", encoding="utf-8")
        write_skip_markers(cfg.data_root, {"fixed.txt": "hash-of-the-unreadable-version"})
        write_skip_reasons(cfg.data_root, {"fixed.txt": "no text extracted (0 chunks)"})

        assert (await sync(quiet=True)).added == ["fixed.txt"]

        assert load_skip_markers(cfg.data_root) == {}
        assert load_skip_reasons(cfg.data_root) == {}

    async def test_a_reset_during_a_sync_is_refused(self, isolated_env, mock_svc):
        from lilbee.app.reset import perform_reset
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import (
            load_skip_markers,
            load_skip_reasons,
            write_skip_markers,
            write_skip_reasons,
        )
        from lilbee.runtime.lock import ResetRefusedError

        write_skip_markers(cfg.data_root, {"older.txt": "abc"})
        write_skip_reasons(cfg.data_root, {"older.txt": "held before the sync"})
        scan = isolated_env / "scanned.pdf"
        refused: list[Exception] = []

        def _reset() -> None:
            whole = (
                r"^A sync, an import, an add or a wiki build is running on this library\. "
                r"Reset again when it finishes\.$"
            )
            with pytest.raises(ResetRefusedError, match=whole) as caught:
                perform_reset()
            refused.append(caught.value)

        await self._sync_holding_out(scan, _reset)

        assert len(refused) == 1
        assert scan.exists()
        assert load_skip_markers(cfg.data_root) == {
            "older.txt": "abc",
            "scanned.pdf": file_hash(scan),
        }
        assert load_skip_reasons(cfg.data_root) == {
            "older.txt": "held before the sync",
            "scanned.pdf": "no text extracted (0 chunks)",
        }
        assert perform_reset().deleted_docs == 1
        assert load_skip_markers(cfg.data_root) == {}

    async def test_a_delete_during_a_sync_stays_deleted(self, isolated_env, mock_svc):
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import (
            REMOVED_SKIP_REASON,
            SkipKind,
            load_skip_kinds,
            load_skip_markers,
            load_skip_reasons,
        )

        stay = isolated_env / "stay.txt"
        stay.write_text("indexed before the delete", encoding="utf-8")
        assert "stay.txt" in (await sync(quiet=True)).added
        scan = isolated_env / "scanned.pdf"

        await self._sync_holding_out(scan, lambda: remove_documents_durably(["stay.txt"]))

        assert load_skip_markers(cfg.data_root) == {
            "stay.txt": file_hash(stay),
            "scanned.pdf": file_hash(scan),
        }
        assert load_skip_reasons(cfg.data_root) == {
            "stay.txt": REMOVED_SKIP_REASON,
            "scanned.pdf": "no text extracted (0 chunks)",
        }
        assert load_skip_kinds(cfg.data_root) == {
            "stay.txt": SkipKind.REMOVED,
            "scanned.pdf": SkipKind.FAILED,
        }

    async def test_an_add_rolled_back_during_a_sync_stays_rolled_back(
        self, isolated_env, mock_svc, tmp_path
    ):
        from lilbee.app.ingest import register_sources
        from lilbee.cli.tui.screens.chat_helpers import unregister_added_roots
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import (
            load_skip_markers,
            load_skip_reasons,
            write_skip_markers,
            write_skip_reasons,
        )

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_bytes(b"")
        register_sources([corpus])
        write_skip_markers(cfg.data_root, {"corpus/a.txt": file_hash(corpus / "a.txt")})
        write_skip_reasons(cfg.data_root, {"corpus/a.txt": "no text extracted (0 chunks)"})
        scan = isolated_env / "scanned.pdf"

        await self._sync_holding_out(scan, lambda: unregister_added_roots(["corpus"]))

        assert load_skip_markers(cfg.data_root) == {"scanned.pdf": file_hash(scan)}
        assert load_skip_reasons(cfg.data_root) == {"scanned.pdf": "no text extracted (0 chunks)"}
        assert cfg.linked_roots == {}


class TestFanoutReadsTheCorpusSkipRecords:
    """A multi-worker sync holds out and records files exactly as a single-process one does.

    The workers run as threads over the one mock store, so the real fan-out path
    runs from ``sync`` through each worker's own ``sync`` of its slice.
    """

    @pytest.fixture(params=[False, True], ids=["one-process", "fan-out"])
    def fan_out(self, request, monkeypatch, mock_svc):
        from lilbee.data.ingest import fanout, pipeline
        from tests.test_ingest_fanout import FakeContext

        if request.param:
            monkeypatch.setattr(pipeline, "plan_fanout", lambda: fanout.shard_specs(cfg, 2, 2))
            monkeypatch.setattr(fanout.multiprocessing, "get_context", lambda _k: FakeContext())
            monkeypatch.setattr(fanout, "_FINAL_DRAIN_S", 0.0)
            monkeypatch.setattr(fanout, "_apply_shard_env", lambda spec: None)
            monkeypatch.setattr(
                "lilbee.providers.fleet.child_guard.bind_lifetime_to_parent", lambda pid: None
            )
            monkeypatch.setattr("lilbee.app.services.build_services", lambda config: mock_svc)
        else:
            monkeypatch.setattr(pipeline, "plan_fanout", list)
        return request.param

    async def test_a_removed_file_is_not_ingested_again(self, isolated_env, fan_out):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import SkipKind, load_skip_kinds, mark_removed

        (isolated_env / "gone.txt").write_text("removed by the user", encoding="utf-8")
        (isolated_env / "kept.txt").write_text("still wanted", encoding="utf-8")
        mark_removed(cfg.data_root, {"gone.txt": file_hash(isolated_env / "gone.txt")})

        result = await sync(quiet=True)

        assert result.added == ["kept.txt"]
        assert load_skip_kinds(cfg.data_root) == {"gone.txt": SkipKind.REMOVED}

    async def test_a_failure_is_recorded_for_the_corpus(self, isolated_env, fan_out):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import SkipKind, load_skip_kinds

        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        (isolated_env / "kept.txt").write_text("still wanted", encoding="utf-8")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records",
            side_effect=TestSkipMarkerLifecycle._zero_for("scanned.pdf"),
        ):
            first = await sync(quiet=True)
            second = await sync(quiet=True)

        assert first.added == ["kept.txt"]
        assert load_skip_kinds(cfg.data_root) == {"scanned.pdf": SkipKind.FAILED}
        assert [held.filename for held in second.held_out] == ["scanned.pdf"]

    async def test_retry_skipped_retries_a_failure_and_keeps_a_removal(self, isolated_env, fan_out):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import (
            SkipKind,
            load_skip_kinds,
            mark_removed,
            update_skip_records,
        )

        gone = isolated_env / "gone.txt"
        fixed = isolated_env / "fixed.txt"
        gone.write_text("removed by the user", encoding="utf-8")
        fixed.write_text("readable now", encoding="utf-8")
        mark_removed(cfg.data_root, {"gone.txt": file_hash(gone)})

        def _fail(records):
            records.markers["fixed.txt"] = file_hash(fixed)
            records.kinds["fixed.txt"] = SkipKind.FAILED

        update_skip_records(cfg.data_root, _fail)

        result = await sync(quiet=True, retry_skipped=True)

        assert result.added == ["fixed.txt"]
        assert load_skip_kinds(cfg.data_root) == {"gone.txt": SkipKind.REMOVED}

    async def test_a_rebuild_clears_the_records_once_and_keeps_new_verdicts(
        self, isolated_env, fan_out, monkeypatch
    ):
        """A worker clearing the shared records would erase a sibling's fresh verdict."""
        from lilbee.data.ingest import pipeline, sync
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import SkipKind, load_skip_kinds, mark_removed

        cleared = []
        real_clear = pipeline.clear_skip_markers
        monkeypatch.setattr(
            pipeline, "clear_skip_markers", lambda root: cleared.append(root) or real_clear(root)
        )
        gone = isolated_env / "gone.txt"
        gone.write_text("removed, then restored by the rebuild", encoding="utf-8")
        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        mark_removed(cfg.data_root, {"gone.txt": file_hash(gone)})
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records",
            side_effect=TestSkipMarkerLifecycle._zero_for("scanned.pdf"),
        ):
            result = await sync(quiet=True, force_rebuild=True)

        assert cleared == [cfg.data_root]
        assert result.added == ["gone.txt"]
        assert load_skip_kinds(cfg.data_root) == {"scanned.pdf": SkipKind.FAILED}

    async def test_the_data_root_ignore_file_keeps_a_file_out(self, isolated_env, fan_out):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        (cfg.data_root / IGNORE_FILENAME).write_text("drop.txt\n", encoding="utf-8")
        (isolated_env / "drop.txt").write_text("excluded by the library", encoding="utf-8")
        (isolated_env / "kept.txt").write_text("still wanted", encoding="utf-8")

        result = await sync(quiet=True)

        assert result.added == ["kept.txt"]

    async def test_prune_ignored_drops_what_the_data_root_file_excludes(
        self, isolated_env, fan_out
    ):
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        (isolated_env / "drop.txt").write_text("excluded after ingest", encoding="utf-8")
        (isolated_env / "kept.txt").write_text("still wanted", encoding="utf-8")
        await sync(quiet=True)
        (cfg.data_root / IGNORE_FILENAME).write_text("drop.txt\n", encoding="utf-8")

        result = await sync(quiet=True, prune_ignored=True)

        assert result.removed == ["drop.txt"]

    async def test_a_held_records_lock_leaves_the_records_unwritten_and_says_so(
        self, isolated_env, fan_out, monkeypatch
    ):
        from filelock import FileLock

        from lilbee.data.ingest import skip_marker, sync
        from lilbee.data.ingest.skip_marker import SKIP_MARKER_FILENAME, load_skip_kinds

        monkeypatch.setattr(skip_marker, "_RECORDS_LOCK_TIMEOUT_S", 0.05)
        (isolated_env / "scanned.pdf").write_bytes(b"%PDF-1.4 not really text")
        (isolated_env / "kept.txt").write_text("still wanted", encoding="utf-8")
        holder = FileLock(str(cfg.data_root / SKIP_MARKER_FILENAME) + ".lock")
        holder.acquire()
        try:
            with mock.patch(
                "lilbee.data.ingest.pipeline.produce_records",
                side_effect=TestSkipMarkerLifecycle._zero_for("scanned.pdf"),
            ):
                result = await sync(quiet=True)
        finally:
            holder.release()

        assert load_skip_kinds(cfg.data_root) == {}
        assert result.added == ["kept.txt"]
        assert result.skip_records_error is not None
        assert SKIP_MARKER_FILENAME + ".lock" in result.skip_records_error
        assert result.skip_records_error in str(result)


class TestStatusExposesTheIndexEmbedder:
    """A client can tell a stale index from the configured model before the
    first search refuses it."""

    def test_status_exposes_the_embedder_that_built_the_index(self, mock_svc):
        from lilbee.app.status import gather_status

        mock_svc.store.get_meta.return_value = {
            "embedding_model": "old/embed-GGUF/old.gguf",
            "embedding_dim": 768,
            "schema_version": 2,
            "updated_at": "2026-09-09T00:00:00+00:00",
        }

        status = gather_status()

        assert status.index is not None
        assert status.index.embedding_model == "old/embed-GGUF/old.gguf"
        assert status.index.embedding_dim == 768

    def test_status_has_no_index_section_before_the_first_sync(self, mock_svc):
        from lilbee.app.status import gather_status

        mock_svc.store.get_meta.return_value = None
        assert gather_status().index is None


class TestStatusSaysWhatHappensToScannedPages:
    """Status carries one scanned-pages line: skipped, or read by which engine."""

    _VISION = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"

    @pytest.mark.parametrize(
        ("ocr", "vision_model", "expected"),
        [
            (OcrMode.AUTO, _VISION, f"read by {_VISION}"),
            (OcrMode.AUTO, "", "read by Tesseract (eng+deu)"),
            (OcrMode.ALL, _VISION, f"every page read by {_VISION} (ocr = all)"),
            (OcrMode.ALL, "", "every page read by Tesseract (eng+deu) (ocr = all)"),
            (OcrMode.OFF, _VISION, "skipped (ocr = off)"),
            (OcrMode.OFF, "", "skipped (ocr = off)"),
        ],
    )
    def test_the_line_follows_ocr_and_the_engine(self, mock_svc, ocr, vision_model, expected):
        from lilbee.app.status import gather_status

        mock_svc.store.get_meta.return_value = None
        cfg.vision_model = vision_model
        cfg.ocr = ocr
        cfg.ocr_language = ["eng", "deu"]
        assert gather_status().ocr_note == expected

    def test_status_config_reports_the_mode(self, mock_svc):
        from lilbee.app.status import gather_status

        mock_svc.store.get_meta.return_value = None
        cfg.ocr = OcrMode.ALL
        assert gather_status().config.ocr is OcrMode.ALL


class TestOcrBackendChosen:
    """ocr = off reads no page with either engine; otherwise a set vision model wins."""

    @pytest.mark.parametrize(
        ("ocr", "vision_model", "expected"),
        [
            (OcrMode.OFF, "org/V-GGUF/v.gguf", OcrBackendUsed.NONE),
            (OcrMode.OFF, "", OcrBackendUsed.NONE),
            (OcrMode.AUTO, "org/V-GGUF/v.gguf", OcrBackendUsed.VISION),
            (OcrMode.ALL, "org/V-GGUF/v.gguf", OcrBackendUsed.VISION),
            (OcrMode.AUTO, "", OcrBackendUsed.TESSERACT),
            (OcrMode.ALL, "", OcrBackendUsed.TESSERACT),
        ],
    )
    def test_chosen(self, ocr, vision_model, expected):
        assert OcrBackendUsed.chosen(ocr, vision_model) is expected


class TestStatusReportsHeldOutFiles:
    """/api/status and `lilbee status --json` read the skip sidecars, so a client
    can show which files are held out of the index and why."""

    def test_status_lists_held_out_files_with_reasons(self):
        from lilbee.app.status import gather_status
        from lilbee.data.ingest.skip_marker import (
            DEFAULT_SKIP_REASON,
            write_skip_markers,
            write_skip_reasons,
        )

        write_skip_markers(cfg.data_root, {"scan.pdf": "deadbeef", "blank.md": "cafef00d"})
        write_skip_reasons(cfg.data_root, {"scan.pdf": "OCR timed out after 300s"})

        status = gather_status()

        assert [(s.filename, s.reason) for s in status.skipped] == [
            ("blank.md", DEFAULT_SKIP_REASON),
            ("scan.pdf", "OCR timed out after 300s"),
        ]
        assert status.skipped_total == 2

    def test_status_caps_the_list_but_reports_the_real_total(self, monkeypatch):
        from lilbee.app import status as status_mod
        from lilbee.data.ingest.skip_marker import write_skip_markers

        monkeypatch.setattr(status_mod, "STATUS_SKIPPED_LIMIT", 2)
        write_skip_markers(cfg.data_root, {f"scan{i}.pdf": "hash" for i in range(5)})

        status = status_mod.gather_status()

        assert [s.filename for s in status.skipped] == ["scan0.pdf", "scan1.pdf"]
        assert status.skipped_total == 5

    def test_status_leaves_removed_sources_out(self):
        """A removed source is not a library problem, so status does not list it."""
        from lilbee.app.status import gather_status
        from lilbee.data.ingest.skip_marker import (
            REMOVED_SKIP_REASON,
            SkipKind,
            SkipRecords,
            write_skip_records,
        )

        write_skip_records(
            cfg.data_root,
            SkipRecords(
                markers={"scan.pdf": "h1", "gone.txt": "h2", "old.md": "h3"},
                reasons={"old.md": REMOVED_SKIP_REASON},
                kinds={"scan.pdf": SkipKind.FAILED, "gone.txt": SkipKind.REMOVED},
            ),
        )

        status = gather_status()

        assert [s.filename for s in status.skipped] == ["scan.pdf"]
        assert status.skipped_total == 1

    def test_status_is_empty_when_nothing_is_held_out(self):
        from lilbee.app.status import gather_status

        status = gather_status()

        assert status.skipped == []
        assert status.skipped_total == 0


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
        cancel=None,
    ):
        from lilbee.data.store import SourceMeta
        from lilbee.data.types import DocumentRecords

        if page_texts_out is not None:
            page_texts_out.append(
                {"source": source_name, "page": 1, "text": " ", "content_type": "pdf"}
            )
        return DocumentRecords([], SourceMeta())

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


class TestExcludedFormats:
    """Archives are containers whose members ingest; drawings are refused and say so."""

    @pytest.mark.parametrize("name", ["runs.gz", "corpus.tgz", "docs.zip", "b.tar", "a.7z"])
    def test_archives_are_containers_not_refusals(self, name):
        from lilbee.data.ingest.discovery import (
            archive_content_types,
            excluded_extension_reasons,
            supported_extension_map,
        )

        suffix = Path(name).suffix
        assert suffix not in excluded_extension_reasons()
        assert supported_extension_map()[suffix] == suffix.lstrip(".")
        assert suffix.lstrip(".") in archive_content_types()

    @pytest.mark.parametrize("suffix", [".md", ".pdf", ".epub", ".docx", ".txt"])
    def test_documents_are_not_refused(self, suffix):
        """epub is application/epub+zip: packaged as a zip, but a book."""
        from lilbee.data.ingest.discovery import archive_content_types, supported_extension_map

        assert suffix in supported_extension_map()
        assert supported_extension_map()[suffix] not in archive_content_types()

    def test_svg_is_refused_as_a_drawing(self):
        from lilbee.data.ingest.discovery import (
            ExclusionReason,
            excluded_extension_reasons,
            supported_extension_map,
        )

        assert excluded_extension_reasons()[".svg"] == ExclusionReason.VECTOR_GRAPHIC
        assert ".svg" not in supported_extension_map()

    @pytest.mark.parametrize("suffix", [".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".webm"])
    def test_audio_and_video_are_refused_without_transcription(self, suffix):
        """xberg errors on audio and video unless a transcription model is configured."""
        from lilbee.data.ingest.discovery import (
            ExclusionReason,
            excluded_extension_reasons,
            supported_extension_map,
        )

        assert excluded_extension_reasons()[suffix] == ExclusionReason.NEEDS_TRANSCRIPTION
        assert suffix not in supported_extension_map()

    def test_scan_keeps_archives_and_refuses_drawings(self, isolated_env):
        from lilbee.data.ingest.discovery import ExclusionReason, discover_corpus

        (isolated_env / "note.md").write_text("# Note", encoding="utf-8")
        (isolated_env / "runs.gz").write_bytes(b"\x1f\x8b")
        (isolated_env / "docs.zip").write_bytes(b"PK\x03\x04")
        (isolated_env / "logo.svg").write_text("<svg/>", encoding="utf-8")

        scan = discover_corpus()
        assert set(scan.files) == {"note.md", "runs.gz", "docs.zip"}
        assert scan.excluded == {"logo.svg": ExclusionReason.VECTOR_GRAPHIC}

    def test_discover_files_hides_the_refused_ones(self, isolated_env):
        from lilbee.data.ingest import discover_files

        (isolated_env / "note.md").write_text("# Note", encoding="utf-8")
        (isolated_env / "logo.svg").write_text("<svg/>", encoding="utf-8")

        assert set(discover_files()) == {"note.md"}

    def test_registered_drawing_root_is_refused(self, isolated_env, tmp_path):
        """A single-file root pointing at a refused format is refused like any other file."""
        from lilbee.data.ingest.discovery import ExclusionReason, discover_corpus

        drawing = tmp_path / "linked.svg"
        drawing.write_text("<svg/>", encoding="utf-8")
        cfg.linked_roots = {"linked.svg": str(drawing)}

        scan = discover_corpus()
        assert scan.files == {}
        assert scan.excluded == {"linked.svg": ExclusionReason.VECTOR_GRAPHIC}

    def test_shard_owns_refused_files_too(self, isolated_env):
        """Every key a shard owns is reported by that shard, refusals included."""
        from lilbee.data.ingest.discovery import discover_corpus
        from lilbee.data.types import ShardId

        for i in range(12):
            (isolated_env / f"a{i}.md").write_text("# Note", encoding="utf-8")
            (isolated_env / f"a{i}.svg").write_text("<svg/>", encoding="utf-8")

        slices = [
            discover_corpus(ShardId(index=i, count=3, records_root=cfg.data_root)) for i in range(3)
        ]
        assert sum(len(s.files) for s in slices) == 12
        assert sum(len(s.excluded) for s in slices) == 12

    def test_refused_files_do_not_count_toward_the_fanout_threshold(self, isolated_env):
        from lilbee.data.ingest.discovery import corpus_has_at_least

        for i in range(5):
            (isolated_env / f"a{i}.svg").write_text("<svg/>", encoding="utf-8")
        (isolated_env / "note.md").write_text("# Note", encoding="utf-8")

        assert corpus_has_at_least(1)
        assert not corpus_has_at_least(2)


class TestArchiveMimeRule:
    @pytest.mark.parametrize(
        "mime",
        [
            "application/gzip",
            "application/zip",
            "application/x-tar",
            "application/x-7z-compressed",
        ],
    )
    def test_container_mime_types_are_archives(self, mime):
        from lilbee.data.ingest.discovery import _is_archive_mime

        assert _is_archive_mime(mime)

    @pytest.mark.parametrize(
        "mime",
        [
            "application/epub+zip",
            "application/pdf",
            "text/markdown",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ],
    )
    def test_document_mime_types_are_not_archives(self, mime):
        from lilbee.data.ingest.discovery import _is_archive_mime

        assert not _is_archive_mime(mime)


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

    def test_str_keeps_bracketed_names_unescaped(self):
        """``str()`` is public through the API; it must not add escape backslashes."""
        from lilbee.data.ingest import SyncResult

        text = str(SyncResult(skipped=["C:\\n\\[a].pdf"]))
        assert "  [yellow]C:\\n\\[a].pdf[/yellow]" in text

    def test_rich_output_highlights_counts(self):
        from lilbee.data.ingest import SyncResult

        rendered = SyncResult(added=["a"]).__rich__()
        assert any(rendered.plain[span.start : span.end] == "1" for span in rendered.spans)

    def test_repr_matches_str(self):
        from lilbee.data.ingest import SyncResult

        result = SyncResult(added=["a.txt"])
        assert "SyncResult" in repr(result)
        assert "added=1" in repr(result)


class TestSyncResultRich:
    """``__rich__`` is what ``console.print(result)`` renders; filenames must survive brackets."""

    def test_bracketed_filenames_render_literally(self):
        from rich.console import Console

        from lilbee.data.ingest import SyncResult
        from lilbee.data.types import SkippedSource

        result = SyncResult(
            skipped=["note[draft].pdf"],
            failed=["bad[x].md"],
            relocated=["moved.md"],
            held_out=[SkippedSource(filename="C:\\scans\\[red]scan.pdf", reason="no [/x] text")],
        )
        console = Console(force_terminal=False, width=200)
        with console.capture() as capture:
            console.print(result)
        rendered = capture.get()
        assert "note[draft].pdf" in rendered
        assert "bad[x].md" in rendered
        assert "Relocated: 1" in rendered
        assert "C:\\scans\\[red]scan.pdf: no [/x] text" in rendered

    def test_bracketed_index_mismatch_message_renders_literally(self):
        from rich.console import Console

        from lilbee.data.ingest import SyncResult
        from lilbee.data.store import IndexMismatch

        result = SyncResult(
            index_mismatch=IndexMismatch(
                persisted_model="old[embedder]",
                persisted_dim=384,
                current_model="new-embedder",
                current_dim=768,
                adoptable=False,
                message="index built with [old-embedder]",
            )
        )
        console = Console(force_terminal=False, width=200)
        with console.capture() as capture:
            console.print(result)
        assert "index built with [old-embedder]" in capture.get()


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


class TestCancelDuringThresholdFlush:
    """A cancel landing on a threshold flush waits for it, so the buffer is written once."""

    @pytest.mark.parametrize("cancels", [1, 2])
    async def test_the_buffer_is_flushed_once(self, monkeypatch, cancels):
        import asyncio
        import threading

        from lilbee.data.ingest import pipeline
        from lilbee.data.types import _IngestResult

        monkeypatch.setattr(pipeline, "_WRITE_FLUSH_CHUNKS", 1)
        entered = threading.Event()
        release = threading.Event()
        guard = threading.Lock()
        active: list[int] = [0]
        overlaps: list[int] = []
        flushed: list[list[str]] = []

        def _slow_flush(buffer: list[_IngestResult]) -> None:
            with guard:
                active[0] += 1
                overlaps.append(active[0])
                flushed.append([r.name for r in buffer])
            entered.set()
            release.wait(5)
            with guard:
                active[0] -= 1

        async def _done() -> _IngestResult:
            return _IngestResult("a.txt", Path("a.txt"), chunk_count=1, error=None)

        async def _never() -> _IngestResult:
            await asyncio.sleep(3600)
            raise AssertionError("the sibling should have been cancelled")

        monkeypatch.setattr(pipeline, "_flush_batch", _slow_flush)
        task = asyncio.create_task(
            pipeline._collect_results(_feed([_done(), _never()]), {}, {}, {}, {}, window=2)
        )
        assert await asyncio.to_thread(entered.wait, 5)
        for _ in range(cancels):
            task.cancel()
            await asyncio.sleep(0.2)  # time for a second flush to start, if one would
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert flushed == [["a.txt"]]
        assert overlaps == [1]

    def test_the_exit_drain_waits_for_a_flush_in_progress(self, monkeypatch):
        """The drain at TUI exit cancels every task on the loop; the write still ends first."""
        import asyncio
        import threading

        from lilbee.data.ingest import pipeline
        from lilbee.data.types import _IngestResult
        from lilbee.runtime import asyncio_loop

        monkeypatch.setattr(pipeline, "_WRITE_FLUSH_CHUNKS", 1)
        entered = threading.Event()
        release = threading.Event()
        flushed: list[list[str]] = []
        ended: list[type[BaseException]] = []

        def _slow_flush(buffer: list[_IngestResult]) -> None:
            flushed.append([r.name for r in buffer])
            entered.set()
            release.wait(5)

        async def _done() -> _IngestResult:
            return _IngestResult("a.txt", Path("a.txt"), chunk_count=1, error=None)

        async def _never() -> _IngestResult:
            await asyncio.sleep(3600)
            raise AssertionError("the sibling should have been cancelled")

        def _worker() -> None:
            feed = _feed([_done(), _never()])
            try:
                asyncio_loop.run(pipeline._collect_results(feed, {}, {}, {}, {}, window=2))
            except BaseException as exc:
                ended.append(type(exc))

        monkeypatch.setattr(pipeline, "_flush_batch", _slow_flush)
        worker = threading.Thread(target=_worker)
        worker.start()
        assert entered.wait(5)
        drain = threading.Thread(target=asyncio_loop.shutdown)
        drain.start()
        worker.join(0.5)  # time for the sync to resume beside the write, if it would
        release.set()
        worker.join(10)
        drain.join(15)
        assert ended == [asyncio.CancelledError]
        assert flushed == [["a.txt"]]  # a sync that resumed early flushes the buffer again

    async def test_a_write_that_fails_under_a_cancel_leaves_as_the_cancel(
        self, monkeypatch, caplog
    ):
        import asyncio
        import logging
        import threading

        from lilbee.data.ingest import pipeline

        entered = threading.Event()
        release = threading.Event()

        def _failing_write(*_args: object) -> None:
            entered.set()
            release.wait(5)
            raise OSError("disk full")

        monkeypatch.setattr(pipeline, "_flush_writes", _failing_write)
        task = asyncio.create_task(pipeline._flush_to_end([], {}, {}, {}, {}, None))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)  # the cancel reaches the waiting flush before the write fails
        release.set()
        with (
            caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"),
            pytest.raises(asyncio.CancelledError) as stopped,
        ):
            await task
        assert str(stopped.value.__cause__) == "disk full"
        logged = [r.exc_info[1] for r in caplog.records if r.exc_info]
        assert [str(exc) for exc in logged] == ["disk full"]

    @pytest.mark.parametrize("flush_at", [1, 10_000], ids=["threshold_flush", "final_flush"])
    @pytest.mark.parametrize("stopping", [True, False], ids=["cancel_set", "no_cancel"])
    async def test_a_store_write_that_fails_under_a_set_cancel_leaves_as_its_cause(
        self, monkeypatch, caplog, flush_at, stopping
    ):
        """The sync's cancel signal, with no cancelled task, still takes the failed write."""
        import asyncio
        import logging
        import threading

        from lilbee.data.ingest import pipeline
        from lilbee.data.types import _IngestResult

        cancel = threading.Event()

        def _disk_full(_buffer: list[_IngestResult]) -> None:
            if stopping:
                cancel.set()  # the Ctrl+C lands in the write
            raise OSError("disk full")

        async def _done() -> _IngestResult:
            return _IngestResult("a.txt", Path("a.txt"), chunk_count=1, error=None)

        monkeypatch.setattr(pipeline, "_WRITE_FLUSH_CHUNKS", flush_at)
        monkeypatch.setattr(pipeline, "_flush_batch", _disk_full)
        failed: dict[str, None] = {}
        flush_failed: set[str] = set()
        collect = pipeline._collect_results(
            _feed([_done()]), {}, {}, failed, {}, window=1, flush_failed=flush_failed, cancel=cancel
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"):
            if stopping:
                with pytest.raises(asyncio.CancelledError) as stopped:
                    await collect
                assert str(stopped.value.__cause__) == "disk full"
            else:
                await collect  # a failed write with no cancel is tracked, and the sync goes on
        assert failed == {"a.txt": None}
        assert flush_failed == {"a.txt"}
        # The tracked failure is logged once, where it was tracked.
        assert [r.getMessage() for r in caplog.records] == ["Failed to write a.txt: disk full"]

    async def test_a_second_cancel_waits_for_the_final_flush(self, monkeypatch):
        """The sync does not end while the flush on its way out is still writing."""
        import asyncio
        import threading

        from lilbee.data.ingest import pipeline
        from lilbee.data.types import _IngestResult

        entered = threading.Event()
        release = threading.Event()
        written: list[str] = []

        def _slow_flush(buffer: list[_IngestResult]) -> None:
            entered.set()
            release.wait(5)
            written.extend(r.name for r in buffer)

        async def _done() -> _IngestResult:
            return _IngestResult("a.txt", Path("a.txt"), chunk_count=1, error=None)

        async def _never() -> _IngestResult:
            await asyncio.sleep(3600)
            raise AssertionError("the sibling should have been cancelled")

        buffered = asyncio.Event()
        monkeypatch.setattr(pipeline, "_flush_batch", _slow_flush)
        task = asyncio.create_task(
            pipeline._collect_results(
                _feed([_done(), _never()]),
                {},
                {},
                {},
                {},
                window=2,
                on_progress=lambda *_args: buffered.set(),
            )
        )
        await asyncio.wait_for(buffered.wait(), 5)  # a.txt is buffered below the threshold
        task.cancel()
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()  # lands on the final flush
        await asyncio.sleep(0.2)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert written == ["a.txt"]


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
            _plan_batch_bounds,
        )

        assert list(_plan_batch_bounds(0)) == []
        total = _PLAN_SHARD_MAX_FILES * 4
        bounds = list(_plan_batch_bounds(total))
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

    async def test_a_relocation_only_sync_leaves_the_clusters_alone(
        self, isolated_env, monkeypatch, mock_svc
    ):
        """A move changes no concept co-occurrence, so Leiden has nothing new to see."""
        from lilbee.data.ingest import file_hash

        monkeypatch.setattr(cfg, "concept_graph", True)
        mock_svc.concepts.get_graph.return_value = True
        moved = isolated_env / "zmoved.txt"
        moved.write_text("the relocated document", encoding="utf-8")
        mock_svc.store.upsert_source("old/moved.txt", file_hash(moved), 1, source_type="document")

        with mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True):
            result = await self._sync_with_extraction()

        assert result.relocated == ["zmoved.txt"]
        assert result.added == [] and result.updated == []
        mock_svc.concepts.rebuild_clusters.assert_not_called()
        (move,) = mock_svc.store.relocate_sources.call_args.args[0]
        assert (move.candidates, move.new) == (("old/moved.txt",), "zmoved.txt")

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
        # not portable across platforms; drive _plan_batches directly with a cancel
        # already set, over >1 shard, so the break is hit without any timing.
        import threading

        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.pipeline import _plan_batches, _StreamedPlan

        disk = {f"doc{i}.txt": isolated_env / f"doc{i}.txt" for i in range(4)}
        for path in disk.values():
            path.write_text("content")
        monkeypatch.setattr(pipeline, "_PLAN_SHARD_MIN_FILES", 1)
        monkeypatch.setattr(pipeline, "_PLAN_SHARD_MAX_FILES", 1)

        cancel = threading.Event()
        cancel.set()  # shard 0 plans nothing, ahead drops to None, shard 1 breaks
        state = _StreamedPlan()
        shards = _plan_batches(disk, {}, {}, pipeline._MovePool([], {}), state, cancel)
        yielded = [shard async for shard in shards]
        assert yielded == []  # the break stopped planning the remaining shards
        assert state.planned == 0

    async def test_every_plan_batch_runs_on_the_plan_driver_thread(
        self, isolated_env, monkeypatch, mock_svc
    ):
        import threading

        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.pipeline import _plan_batches, _StreamedPlan

        disk = {f"doc{i}.txt": isolated_env / f"doc{i}.txt" for i in range(3)}
        for path in disk.values():
            path.write_text("content", encoding="utf-8")
        monkeypatch.setattr(pipeline, "_PLAN_SHARD_MIN_FILES", 1)
        monkeypatch.setattr(pipeline, "_PLAN_SHARD_MAX_FILES", 1)
        real_plan_items = pipeline._plan_items
        planned_on: list[str] = []

        def _plan_items(*args, **kwargs):
            planned_on.append(threading.current_thread().name)
            return real_plan_items(*args, **kwargs)

        monkeypatch.setattr(pipeline, "_plan_items", _plan_items)
        moves = pipeline._MovePool([], {})
        batches = _plan_batches(disk, {}, {}, moves, _StreamedPlan(), None)
        shards = [shard async for shard in batches]
        assert [entry.name for shard in shards for entry in shard] == sorted(disk)
        assert planned_on == ["lilbee-plan-driver_0"] * 3

    async def test_cancel_with_nothing_left_to_admit_still_raises(self, isolated_env, mock_svc):
        # The last admitted file finishes and the planner has stopped, so ingest
        # returns without any file raising; sync still reports the run cancelled.
        import threading

        from lilbee.data.ingest import sync

        (isolated_env / "only.txt").write_text("the one file in this corpus")
        cancel = threading.Event()
        embedded: list[int] = []

        def _embed_then_cancel(texts, **_kwargs):
            # After extraction, so the file finishes instead of raising.
            cancel.set()
            embedded.append(len(texts))
            return [[0.1] * 768 for _ in texts]

        mock_svc.embedder.embed_batch.side_effect = _embed_then_cancel
        with (
            mock.patch(
                "lilbee.data.extract.xberg.aextract_document",
                new_callable=mock.AsyncMock,
                return_value=_make_xberg_result(),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await sync(quiet=True, cancel=cancel)
        assert embedded, "the cancel must land after extraction, in the embed step"
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

    def test_registering_a_refused_file_reports_the_reason(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources

        drawing = tmp_path / "logo.svg"
        drawing.write_text("<svg/>", encoding="utf-8")
        result = register_sources([drawing])
        assert result.refused == ["logo.svg: vector graphic, not a document"]
        assert result.registered == []
        assert cfg.linked_roots == {}

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

    def test_paginated_has_tesseract_ocr_backend(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        config = extraction_config(ExtractMode.PAGINATED)
        assert config.pages is not None
        assert config.ocr is not None
        # No vision model configured -> xberg's tesseract backend OCRs scanned pages.
        assert config.ocr.backend == "tesseract"
        # xberg 5.x errors on image OCR with an empty language list; lilbee must
        # set an explicit default to preserve 4.x behavior (regression guard).
        assert config.ocr.language == ["eng"]

    def test_extraction_is_uncapped_by_default(self):
        """Shipped defaults leave no per-file cap on extraction.

        Unset, xberg applies its own 600s limit, and a long file dies at ten
        minutes with no lilbee setting able to raise it.
        """
        from lilbee.data.ingest import ExtractMode, extraction_config

        for mode in (ExtractMode.PAGINATED, ExtractMode.MARKDOWN):
            assert extraction_config(mode).extraction_timeout_secs is None

    def test_extraction_timeout_from_config(self, monkeypatch):
        """cfg.extraction_timeout sets xberg's per-file extraction cap.

        Left unset, xberg applies its own 600s default and no lilbee surface can
        reach it, so a long file dies at ten minutes with nothing to tune.
        """
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "extraction_timeout", 45)
        for mode in (ExtractMode.PAGINATED, ExtractMode.MARKDOWN):
            assert extraction_config(mode).extraction_timeout_secs == 45

    def test_extraction_timeout_zero_lifts_the_cap(self, monkeypatch):
        """0 means no cap, matching ocr_timeout."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "extraction_timeout", 0)
        for mode in (ExtractMode.PAGINATED, ExtractMode.MARKDOWN):
            assert extraction_config(mode).extraction_timeout_secs is None

    def test_tesseract_ocr_language_from_config(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "ocr_language", ["deu", "fra"])
        config = extraction_config(ExtractMode.PAGINATED)
        assert config.ocr.language == ["deu", "fra"]

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
            assert config.chunking.chunker_type == "semantic"
            assert config.chunking.topic_threshold == pytest.approx(0.42, abs=1e-5)

    def test_table_extraction_off_sets_no_pdf_options(self, monkeypatch):
        """Table and layout both off leave pdf_options unset."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", False)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config.pdf_options is None

    def test_table_extraction_sets_pdf_options_and_table_chunking(self, monkeypatch):
        """The flag turns on xberg table recognition and header-repeating splits."""
        from xberg import TableChunkingMode

        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "table_extraction", True)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config.pdf_options.extract_tables is True
        assert config.chunking.table_chunking == TableChunkingMode.REPEAT_HEADER

    def test_table_extraction_markdown_mode_skips_pdf_options(self, monkeypatch):
        """Non-paginated formats have no PdfConfig but still chunk tables with headers."""
        from xberg import TableChunkingMode

        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "table_extraction", True)
        config = extraction_config(ExtractMode.MARKDOWN)
        assert config.pdf_options is None
        assert config.chunking.table_chunking == TableChunkingMode.REPEAT_HEADER

    def test_layout_detection_off_sets_no_layout(self, monkeypatch):
        """With layout detection off, no layout config or pdf_options."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", False)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config.layout is None
        assert config.pdf_options is None

    def test_layout_detection_sets_layout_and_reading_order(self, monkeypatch):
        """The flag enables layout detection, reading order, and margin stripping."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", True)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config.layout is not None
        assert config.use_layout_for_markdown is True
        pdf = config.pdf_options
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
        assert config.layout.table_model == "slanet_auto"

    def test_table_model_override(self, monkeypatch):
        """A different table model flows through to the layout config."""
        from lilbee.core.config import cfg
        from lilbee.core.config.enums import TableModel
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", True)
        monkeypatch.setattr(cfg, "table_model", TableModel.TATR)
        config = extraction_config(ExtractMode.PAGINATED)
        assert config.layout.table_model == "tatr"

    def test_layout_detection_markdown_mode_unchanged(self, monkeypatch):
        """Layout detection is visual-format work; non-paginated formats skip it."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "layout_detection", True)
        config = extraction_config(ExtractMode.MARKDOWN)
        assert config.layout is None
        assert config.pdf_options is None

    def test_layout_and_tables_combine_in_one_pdf_config(self, monkeypatch):
        """Both flags land in the same PdfConfig."""
        from lilbee.core.config import cfg
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "table_extraction", True)
        monkeypatch.setattr(cfg, "layout_detection", True)
        config = extraction_config(ExtractMode.PAGINATED)
        pdf = config.pdf_options
        assert pdf.extract_tables is True
        assert pdf.reading_order is True


class TestPageCountConfig:
    def test_disables_ocr_page_bodies_and_quality_processing(self):
        """The metadata-only pass reads structure only, nothing that renders a page."""
        from lilbee.data.extract.document import _page_count_config

        config = _page_count_config()
        assert config.disable_ocr is True
        assert config.ocr is None
        assert config.pages.extract_pages is False
        assert config.enable_quality_processing is False


class TestProbePages:
    async def test_real_xberg_counts_pages_without_an_ocr_block(self):
        """The OCR-free config still reads the page count from a real PDF."""
        from lilbee.data.extract.document import _probe_pages

        assert (await _probe_pages(make_pdf(pages=3), "three.pdf")).pages == 3

    async def test_returns_the_probes_page_count(self):
        """The count comes from the metadata-only document's own counts, not a guess."""
        from lilbee.data.extract.document import _probe_pages

        probe = mock.MagicMock(counts=mock.MagicMock(pages=7), metadata=Metadata())
        with mock.patch(
            "lilbee.data.extract.xberg.aextract_document",
            new_callable=mock.AsyncMock,
            return_value=probe,
        ):
            assert (await _probe_pages(b"pdf bytes", "f.pdf")).pages == 7

    async def test_falls_back_to_an_empty_probe_when_it_raises(self, caplog):
        """A probe failure returns 0 pages and no scans, not an exception."""
        import logging

        from lilbee.data.extract.document import _PageProbe, _probe_pages

        with (
            mock.patch(
                "lilbee.data.extract.xberg.aextract_document",
                new_callable=mock.AsyncMock,
                side_effect=RuntimeError("bad pdf"),
            ),
            caplog.at_level(logging.DEBUG, logger="lilbee.data.extract.document"),
        ):
            assert await _probe_pages(b"pdf bytes", "f.pdf") == _PageProbe()
        assert "f.pdf" in caplog.text

    async def test_real_scanned_pdf_reports_scanned_pages(self):
        """A real image-only PDF has scanned pages; a real text PDF has none."""
        from pathlib import Path

        from lilbee.data.extract.document import _probe_pages

        scan = Path(__file__).parent / "integration" / "fixtures" / "scanned_maintenance.pdf"
        scanned = await _probe_pages(scan.read_bytes(), scan.name)
        text = await _probe_pages(make_pdf(pages=3), "three.pdf")
        assert scanned.pages == 1 and scanned.has_scanned_pages is True
        assert text.pages == 3 and text.has_scanned_pages is False


class TestScannedPages:
    def test_no_format_metadata_has_no_scanned_pages(self):
        from lilbee.data.extract.document import _scanned_pages

        assert _scanned_pages(mock.MagicMock(metadata=Metadata())) == []

    def test_non_pdf_format_has_no_scanned_pages(self):
        """An image's format metadata carries no PDF block."""
        from lilbee.data.extract.document import _scanned_pages

        metadata = mock.MagicMock(format=mock.MagicMock(pdf=None))
        assert _scanned_pages(mock.MagicMock(metadata=metadata)) == []

    def test_a_pdf_without_a_scanned_page_list_has_none(self):
        from lilbee.data.extract.document import _scanned_pages

        metadata = mock.MagicMock(format=mock.MagicMock(pdf=mock.MagicMock(scanned_pages=None)))
        assert _scanned_pages(mock.MagicMock(metadata=metadata)) == []


class TestOcrPageSelection:
    """cfg.ocr_strategy and cfg.force_ocr_pages reach xberg's ExtractionConfig."""

    @pytest.fixture(autouse=True)
    def _ocr_on(self, monkeypatch):
        monkeypatch.setattr(cfg, "ocr", OcrMode.AUTO)

    def test_auto_sends_the_auto_strategy_and_no_forced_pages(self):
        from lilbee.data.ingest import ExtractMode, extraction_config

        for mode in ExtractMode:
            config = extraction_config(mode)
            assert config.ocr_strategy.mode == "auto"
            assert config.force_ocr_pages is None

    def test_scanned_pages_carries_the_configured_confidence_in_both_modes(self, monkeypatch):
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "ocr_strategy", "scanned_pages")
        monkeypatch.setattr(cfg, "ocr_scan_confidence", 0.5)
        for mode in ExtractMode:
            strategy = json.loads(str(extraction_config(mode).ocr_strategy))
            assert strategy == {"mode": "scanned_pages", "min_confidence": 0.5}

    def test_forced_pages_reach_both_modes(self, monkeypatch):
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "force_ocr_pages", [1, 3])
        for mode in ExtractMode:
            assert extraction_config(mode).force_ocr_pages == [1, 3]

    def test_ocr_off_sends_auto_and_no_forced_pages(self, monkeypatch):
        """xberg rejects scanned_pages when OCR is disabled."""
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "ocr", OcrMode.OFF)
        monkeypatch.setattr(cfg, "ocr_strategy", "scanned_pages")
        monkeypatch.setattr(cfg, "force_ocr_pages", [2])
        for mode in ExtractMode:
            config = extraction_config(mode)
            assert config.ocr_strategy.mode == "auto"
            assert config.force_ocr_pages is None

    def test_per_request_ocr_off_sends_auto_and_no_forced_pages(self, monkeypatch):
        """The HTTP/MCP ocr=off override also drops the page selection."""
        from lilbee.data.extract.document import ocr_override
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "ocr_strategy", "scanned_pages")
        monkeypatch.setattr(cfg, "force_ocr_pages", [2])
        with ocr_override(ocr=OcrMode.OFF):
            for mode in ExtractMode:
                config = extraction_config(mode)
                assert config.ocr_strategy.mode == "auto"
                assert config.force_ocr_pages is None

    def test_vision_with_ocr_off_drops_the_page_selection(self, monkeypatch):
        """ocr = off stops the vision model too, so no page selection applies."""
        from lilbee.data.extract.document import ocr_override
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "vision_model", "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf")
        monkeypatch.setattr(cfg, "ocr_strategy", "scanned_pages")
        monkeypatch.setattr(cfg, "force_ocr_pages", [2])
        with ocr_override(ocr=OcrMode.OFF):
            for mode in ExtractMode:
                config = extraction_config(mode)
                assert config.ocr_strategy.mode == "auto"
                assert config.force_ocr_pages is None

    @pytest.mark.parametrize("ocr", list(OcrMode))
    def test_real_xberg_accepts_the_built_config(self, monkeypatch, ocr):
        from lilbee.data.extract.xberg import extract_document
        from lilbee.data.ingest import ExtractMode, extraction_config

        monkeypatch.setattr(cfg, "ocr", ocr)
        monkeypatch.setattr(cfg, "ocr_strategy", "scanned_pages")
        monkeypatch.setattr(cfg, "force_ocr_pages", [1])
        for mode in ExtractMode:
            doc = extract_document(
                b"plain text", "text/plain", filename="a.txt", config=extraction_config(mode)
            )
            assert "plain text" in doc.content


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
                records, _, _ = await ingest_document(f, "d.pdf", "pdf")
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
            records, _, _ = await ingest_document(f, "doc.pdf", "pdf")
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
        records, _, _ = await ingest_document(f, "test.pdf", "pdf")
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
        records, _, _ = await ingest_document(f, "test.pdf", "pdf")
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
        records, _, _ = await ingest_document(f, "test.pdf", "pdf")
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
        records, _, _ = await ingest_document(f, "test.pdf", "pdf")
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
        records, _, _ = await ingest_document(f, "test.pdf", "pdf")
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
            ("scan.jpc", "image"),
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
            # A mail store is a document set xberg reads; an archive is refused.
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

        cfg.ocr = OcrMode.OFF
        cfg.ocr_timeout = 11.0
        with ocr_override(ocr=OcrMode.ALL, ocr_timeout=99.0):
            pass
        assert cfg.ocr is OcrMode.OFF
        assert cfg.ocr_timeout == 11.0

    def test_effective_values_reflect_override_then_revert(self, isolated_env):
        from lilbee.data.extract.document import (
            _effective_ocr_mode,
            _effective_ocr_timeout,
            ocr_override,
        )

        cfg.ocr = OcrMode.OFF
        cfg.ocr_timeout = 11.0
        with ocr_override(ocr=OcrMode.ALL, ocr_timeout=99.0):
            assert _effective_ocr_mode() is OcrMode.ALL
            assert _effective_ocr_timeout() == 99.0
        assert _effective_ocr_mode() is OcrMode.OFF
        assert _effective_ocr_timeout() == 11.0

    def test_none_arguments_keep_cfg_defaults(self, isolated_env):
        from lilbee.data.extract.document import _effective_ocr_mode, ocr_override

        cfg.ocr = OcrMode.ALL
        with ocr_override(ocr=None, ocr_timeout=None):
            assert _effective_ocr_mode() is OcrMode.ALL

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
        from lilbee.runtime.progress import EventType, ExtractEvent, OcrBackendUsed

        progress = MagicMock()
        cb = _phase_progress_callback(progress, "task-1", lambda *_: None)
        event = ExtractEvent(
            file="scan.pdf", page=3, total_pages=12, ocr_backend=OcrBackendUsed.TESSERACT
        )
        cb(EventType.EXTRACT, event)
        desc = progress.update.call_args.kwargs["description"]
        assert desc == "Tesseract OCR scan.pdf (page 3/12)"

    def test_ocr_start_event_updates_bar_description(self):
        from lilbee.data.ingest.pipeline import _phase_progress_callback
        from lilbee.runtime.progress import EventType, OcrStartEvent

        progress = MagicMock()
        cb = _phase_progress_callback(progress, "task-1", lambda *_: None)
        cb(EventType.OCR_START, OcrStartEvent(file="scan.pdf", total_pages=212))
        desc = progress.update.call_args.kwargs["description"]
        assert desc == "Tesseract OCR on scan.pdf (212 pages in the file)"

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
        from lilbee.runtime.progress import EmbedEvent, EventType, ExtractEvent, OcrBackendUsed

        seen: list[object] = []
        cb = _phase_progress_callback(MagicMock(), "t", lambda et, d: seen.append((et, d)))
        extract = ExtractEvent(file="x", page=1, total_pages=2, ocr_backend=OcrBackendUsed.NONE)
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


class TestExcludedLogLine:
    def test_a_long_list_is_summarized(self, caplog):
        """A vault of logos must not bury the sync output in one line per file."""
        import logging

        from lilbee.data.ingest.discovery import ExclusionReason
        from lilbee.data.ingest.pipeline import _log_excluded

        excluded = {f"logo{i}.svg": ExclusionReason.VECTOR_GRAPHIC for i in range(8)}
        with caplog.at_level(logging.WARNING, logger="lilbee.data.ingest.pipeline"):
            _log_excluded(excluded)

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "Skipped 8 file(s), vector graphic, not a document" in message
        assert "and 3 more" in message


class TestChunkLimit:
    """The per-file chunk cap, and the three ingest paths that honour it."""

    def test_a_count_under_the_limit_passes(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import enforce_chunk_limit

        monkeypatch.setattr(cfg, "max_chunks_per_file", 10)
        enforce_chunk_limit(10)

    def test_a_count_over_the_limit_raises_with_both_numbers(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import ChunkLimitError, enforce_chunk_limit

        monkeypatch.setattr(cfg, "max_chunks_per_file", 10)
        with pytest.raises(ChunkLimitError) as excinfo:
            enforce_chunk_limit(11)
        assert excinfo.value.count == 11
        assert excinfo.value.limit == 10
        assert "11 chunks exceed the per-file limit of 10" in str(excinfo.value)

    def test_zero_lifts_the_limit(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import enforce_chunk_limit

        monkeypatch.setattr(cfg, "max_chunks_per_file", 0)
        enforce_chunk_limit(1_000_000)

    async def test_markdown_over_the_limit_is_refused(self, isolated_env, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import ChunkLimitError
        from lilbee.data.ingest import ingest_markdown

        monkeypatch.setattr(cfg, "max_chunks_per_file", 2)
        md = isolated_env / "huge.md"
        md.write_text("# Heading\n\nbody", encoding="utf-8")
        with (
            mock.patch("lilbee.data.extract.document.chunk_text", return_value=["a", "b", "c"]),
            pytest.raises(ChunkLimitError),
        ):
            await ingest_markdown(md, "huge.md")

    def test_code_over_the_limit_is_refused(self, isolated_env, monkeypatch, mock_svc):
        from lilbee.core.config import cfg
        from lilbee.data.extract.chunk import ChunkLimitError
        from lilbee.data.extract.code_chunker import CodeChunk
        from lilbee.data.ingest import ingest_code_sync

        monkeypatch.setattr(cfg, "max_chunks_per_file", 1)
        f = isolated_env / "huge.py"
        f.write_text("x = 1\n", encoding="utf-8")
        chunks = [
            CodeChunk(chunk=f"c{i}", chunk_index=i, line_start=i, line_end=i) for i in range(2)
        ]
        with (
            mock.patch("lilbee.data.ingest.code.chunk_code", return_value=chunks),
            pytest.raises(ChunkLimitError),
        ):
            ingest_code_sync(f, "huge.py")
        mock_svc.embedder.embed_batch.assert_not_called()


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
        cfg.ocr = OcrMode.AUTO
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
            result, _, _ = await ingest_document(isolated_env / "e.xml", "e.xml", "xml")
        assert result == []

    async def test_no_chunks_returns_empty(self, isolated_env):
        from lilbee.data.ingest import ingest_document

        (isolated_env / "s.xml").write_bytes(b"<x/>")  # ingest_document reads the file bytes
        no_chunks_result = mock.MagicMock(chunks=[], metadata=Metadata())
        mock_extract = mock.AsyncMock(return_value=no_chunks_result)
        with mock.patch("lilbee.data.extract.xberg.aextract_document", mock_extract):
            result, _, _ = await ingest_document(isolated_env / "s.xml", "s.xml", "xml")
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
        mock_svc.concepts.rebuild_clusters.assert_called_once_with({"concept_test4.txt"}, set())

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

        # the scan includes the file, but classify_file returns None
        from lilbee.data.ingest.discovery import CorpusScan

        with (
            mock.patch(
                "lilbee.data.ingest.pipeline.discover_corpus",
                return_value=CorpusScan({"mystery.bin": mystery}, {}),
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
        ingest_mod.forget_removed_from_wiki_index(["gone.txt"])
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
        ingest_mod.forget_removed_from_wiki_index(["gone.txt"])

    def test_removing_nothing_touches_no_index(self, isolated_env, monkeypatch):
        from lilbee.app import ingest as ingest_mod
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "wiki", True)
        called: list[set] = []
        monkeypatch.setattr(
            "lilbee.wiki.stubs.drop_sources_from_index", lambda names: called.append(names)
        )
        ingest_mod.forget_removed_from_wiki_index([])
        assert called == []


class TestForgetMissingFromWikiIndex:
    def test_a_failure_to_read_the_index_is_logged_and_does_not_raise(
        self, isolated_env, monkeypatch, caplog
    ):
        from lilbee.app import ingest as ingest_mod
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "wiki", True)
        monkeypatch.setattr(
            "lilbee.wiki.stubs.load_stub_index",
            mock.MagicMock(side_effect=RuntimeError("index unreadable")),
        )
        with caplog.at_level("WARNING", logger=ingest_mod.log.name):
            ingest_mod.forget_missing_from_wiki_index()
        assert "Failed to drop missing documents from the wiki index" in caplog.text

    def test_with_the_wiki_off_the_index_is_not_read(self, isolated_env, monkeypatch):
        from lilbee.app import ingest as ingest_mod
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "wiki", False)
        read = mock.MagicMock()
        monkeypatch.setattr("lilbee.wiki.stubs.load_stub_index", read)
        ingest_mod.forget_missing_from_wiki_index()
        read.assert_not_called()


class TestRemoveDocumentsDurably:
    def test_writes_skip_marker_for_kept_file(self, isolated_env, mock_svc):
        """A durable delete keeps the file but skip-marks it so sync won't re-ingest."""
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import (
            SkipKind,
            load_skip_kinds,
            load_skip_markers,
            load_skip_reasons,
        )
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
        assert load_skip_kinds(cfg.data_root) == {"keep.txt": SkipKind.REMOVED}

    def test_no_marker_when_nothing_removed(self, isolated_env, mock_svc):
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest.skip_marker import load_skip_markers
        from lilbee.data.store.types import RemoveResult

        mock_svc.store.remove_documents.side_effect = None
        mock_svc.store.remove_documents.return_value = RemoveResult(removed=[], not_found=["gone"])
        remove_documents_durably(["gone"])
        assert load_skip_markers(cfg.data_root) == {}

    def test_marker_lands_under_the_active_config_data_root(self, isolated_env, mock_svc, tmp_path):
        """A caller running against its own Config gets the marker where it reads it.

        The library API removes inside a ``config_scope``; a marker written to the
        process-global data root instead would leave that caller's next sync
        re-ingesting the document it just removed.
        """
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.core.config.context import config_scope
        from lilbee.data.ingest.skip_marker import load_skip_markers

        scoped_root = tmp_path / "scoped"
        (scoped_root / "documents").mkdir(parents=True)
        (scoped_root / "documents" / "keep.txt").write_text("content", encoding="utf-8")
        scoped = cfg.model_copy(
            update={"data_root": scoped_root, "documents_dir": scoped_root / "documents"}
        )
        mock_svc.store.upsert_source("keep.txt", "hash", 1)

        with config_scope(scoped):
            remove_documents_durably(["keep.txt"])

        assert "keep.txt" in load_skip_markers(scoped_root)
        assert load_skip_markers(cfg.data_root) == {}


class TestRemovalHoldsAcrossSyncs:
    """A removal has to keep holding through syncs that cannot see the file."""

    async def test_removal_survives_a_sync_with_the_root_unavailable(
        self, isolated_env, mock_svc, tmp_path
    ):
        """A root offline for one sync must not erase the markers holding its removals."""
        from lilbee.app.ingest import register_sources, remove_documents_durably
        from lilbee.data.ingest import detect_pending, sync

        ext = tmp_path / "ext"
        ext.mkdir()
        (ext / "a.txt").write_text("external content a", encoding="utf-8")
        (ext / "b.txt").write_text("external content b", encoding="utf-8")
        register_sources([ext])
        await sync(quiet=True)

        remove_documents_durably(["ext/a.txt"])
        assert detect_pending() == 0

        # The root goes away for one sync: an unmounted volume, a moved folder,
        # a share that had not come back yet.
        ext.rename(tmp_path / "ext-away")
        await sync(quiet=True)
        (tmp_path / "ext-away").rename(ext)

        assert detect_pending() == 0

    async def test_a_sync_keeps_the_reason_for_a_marker_it_did_not_touch(
        self, isolated_env, mock_svc
    ):
        """The reasons sidecar must still explain every marker still in force."""
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import load_skip_reasons

        (isolated_env / "gone.txt").write_text("removed later", encoding="utf-8")
        (isolated_env / "stay.txt").write_text("stays indexed", encoding="utf-8")
        await sync(quiet=True)
        remove_documents_durably(["gone.txt"])

        (isolated_env / "new.txt").write_text("brand new", encoding="utf-8")
        await sync(quiet=True)

        assert "gone.txt" in load_skip_reasons(cfg.data_root)


class TestRemovingHeldOutFiles:
    """A file an ingestion failure holds out is a remove target, on the one removal rule."""

    @staticmethod
    async def _fail_under_corpus(tmp_path, *failing: str):
        """Register ``corpus`` with good.txt plus each failing file, and sync it."""
        from lilbee.app.ingest import register_sources
        from lilbee.data.ingest import sync

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "good.txt").write_text("plenty of text", encoding="utf-8")
        for name in failing:
            (corpus / name).write_text(f"unreadable {name}", encoding="utf-8")
        register_sources([corpus])
        producer = TestSkipMarkerLifecycle._zero_for(*(f"corpus/{n}" for n in failing))
        with mock.patch("lilbee.data.ingest.pipeline.produce_records", side_effect=producer):
            await sync(quiet=True)
        return corpus

    async def test_a_failure_under_a_directory_root_stays_out_after_removal(
        self, isolated_env, mock_svc, tmp_path
    ):
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import SkipKind, load_skip_kinds, load_skip_markers

        corpus = await self._fail_under_corpus(tmp_path, "bad.txt")
        assert load_skip_kinds(cfg.data_root) == {"corpus/bad.txt": SkipKind.FAILED}

        result = remove_documents_durably(["corpus/bad.txt"])
        again = await sync(quiet=True)  # nothing patched: the file would ingest now

        assert result.removed == ["corpus/bad.txt"]
        assert result.not_found == []
        assert load_skip_kinds(cfg.data_root) == {"corpus/bad.txt": SkipKind.REMOVED}
        assert load_skip_markers(cfg.data_root) == {"corpus/bad.txt": file_hash(corpus / "bad.txt")}
        assert "corpus/bad.txt" not in again.added
        assert "corpus/bad.txt" not in _indexed(mock_svc)
        assert again.held_out == []
        assert "corpus" in cfg.linked_roots
        assert "corpus/good.txt" in _indexed(mock_svc)

    async def test_a_folder_covers_the_failures_beneath_it(self, isolated_env, mock_svc, tmp_path):
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import SkipKind, load_skip_kinds

        corpus = await self._fail_under_corpus(tmp_path)
        sub = corpus / "sub"
        sub.mkdir()
        (sub / "bad.txt").write_text("unreadable", encoding="utf-8")
        with mock.patch(
            "lilbee.data.ingest.pipeline.produce_records",
            side_effect=TestSkipMarkerLifecycle._zero_for("corpus/sub/bad.txt"),
        ):
            await sync(quiet=True)

        result = remove_documents_durably(["corpus/sub"])

        assert result.removed == ["corpus/sub/bad.txt"]
        assert load_skip_kinds(cfg.data_root) == {"corpus/sub/bad.txt": SkipKind.REMOVED}

    def test_an_unreachable_failure_keeps_its_hash_as_a_removal(self, isolated_env, mock_svc):
        """A failed file that is not on disk now stays out when it comes back."""
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest.skip_marker import (
            SkipKind,
            load_skip_kinds,
            load_skip_markers,
            write_skip_markers,
        )

        write_skip_markers(cfg.data_root, {"away.pdf": "h1"})

        result = remove_documents_durably(["away.pdf"])

        assert result.removed == ["away.pdf"]
        assert load_skip_markers(cfg.data_root) == {"away.pdf": "h1"}
        assert load_skip_kinds(cfg.data_root) == {"away.pdf": SkipKind.REMOVED}

    def test_an_unreachable_failure_keeps_the_hash_recorded_when_it_is_marked(
        self, isolated_env, mock_svc
    ):
        """A marker another writer changes while the removal runs is the hash the removal keeps."""
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest.discovery import resolve_source_path
        from lilbee.data.ingest.skip_marker import load_skip_markers, write_skip_markers

        written = ["h1"]
        write_skip_markers(cfg.data_root, {"away.pdf": written[-1]})

        def _rewritten_meanwhile(name: str) -> Path:
            written.append(f"h{len(written) + 1}")
            write_skip_markers(cfg.data_root, {"away.pdf": written[-1]})
            return resolve_source_path(name)

        with mock.patch("lilbee.app.ingest.resolve_source_path", side_effect=_rewritten_meanwhile):
            result = remove_documents_durably(["away.pdf"])

        assert result.removed == ["away.pdf"]
        assert len(written) > 2
        assert load_skip_markers(cfg.data_root) == {"away.pdf": written[-1]}

    def test_a_failed_single_file_root_is_forgotten(self, isolated_env, mock_svc, tmp_path):
        """A single-file root is un-registered and its records dropped, not converted."""
        from lilbee.app.ingest import register_sources, remove_documents_durably
        from lilbee.data.ingest.skip_marker import (
            load_skip_kinds,
            load_skip_markers,
            write_skip_markers,
        )

        scan = tmp_path / "scan.pdf"
        scan.write_bytes(b"%PDF-1.4")
        register_sources([scan])
        write_skip_markers(cfg.data_root, {"scan.pdf": "h1"})

        result = remove_documents_durably(["scan.pdf"])

        assert result.removed == ["scan.pdf"]
        assert load_skip_markers(cfg.data_root) == {}
        assert load_skip_kinds(cfg.data_root) == {}
        assert "scan.pdf" not in cfg.linked_roots
        assert scan.exists()

    async def test_removing_a_root_drops_the_records_under_it(
        self, isolated_env, mock_svc, tmp_path
    ):
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest.skip_marker import (
            load_skip_markers,
            load_skip_reasons,
            write_skip_markers,
        )

        await self._fail_under_corpus(tmp_path, "bad.txt")
        write_skip_markers(
            cfg.data_root, load_skip_markers(cfg.data_root) | {"elsewhere.txt": "h9"}
        )

        result = remove_documents_durably(["corpus"])

        assert result.removed == ["corpus/good.txt", "corpus/bad.txt"]
        assert "corpus" not in cfg.linked_roots
        assert load_skip_markers(cfg.data_root) == {"elsewhere.txt": "h9"}
        assert load_skip_reasons(cfg.data_root) == {}

    async def test_a_root_whose_files_all_failed_can_be_removed(
        self, isolated_env, mock_svc, tmp_path
    ):
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest.skip_marker import load_skip_markers

        corpus = await self._fail_under_corpus(tmp_path, "bad.txt")
        mock_svc.store.remove_documents(["corpus/good.txt"])  # nothing of it indexed

        result = remove_documents_durably(["corpus"])

        assert result.removed == ["corpus/bad.txt"]
        assert result.not_found == []
        assert "corpus" not in cfg.linked_roots
        assert load_skip_markers(cfg.data_root) == {}
        assert (corpus / "bad.txt").exists()

    def test_removing_a_root_marks_no_owned_file_of_the_same_path(
        self, isolated_env, mock_svc, tmp_path
    ):
        """A name under the removed root never marks the documents_dir file it now resolves to."""
        from lilbee.app.ingest import register_sources, remove_documents_durably
        from lilbee.data.ingest.skip_marker import load_skip_markers

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("linked", encoding="utf-8")
        register_sources([corpus])
        owned = isolated_env / "corpus" / "a.txt"
        owned.parent.mkdir()
        owned.write_text("owned", encoding="utf-8")
        mock_svc.store.upsert_source("corpus/a.txt", "hash", 1)

        result = remove_documents_durably(["corpus"])

        assert result.removed == ["corpus/a.txt"]
        assert "corpus" not in cfg.linked_roots
        assert load_skip_markers(cfg.data_root) == {}

    @pytest.mark.parametrize("name", ["empty", "empty/"])
    def test_an_empty_root_can_be_removed(self, isolated_env, mock_svc, tmp_path, name):
        """A shell completes a directory with a trailing slash; the root is still removed."""
        from lilbee.app.ingest import register_sources, remove_documents_durably

        empty = tmp_path / "empty"
        empty.mkdir()
        register_sources([empty])

        result = remove_documents_durably([name, "nope/"])

        assert result.removed == [name]
        assert result.not_found == ["nope/"]
        assert "empty" not in cfg.linked_roots

    def test_a_removed_source_is_not_a_target_again(self, isolated_env, mock_svc):
        """Only failures are held-out targets; a removal already holds and is not found."""
        from lilbee.app.ingest import remove_documents_durably
        from lilbee.data.ingest.skip_marker import (
            SkipKind,
            load_skip_kinds,
            mark_removed,
            write_skip_markers,
        )

        mark_removed(cfg.data_root, {"gone.txt": "h1"})
        write_skip_markers(cfg.data_root, {"gone.txt": "h1", "x.txt": "h2"})

        result = remove_documents_durably(["gone.txt", "nope.txt"])

        assert result.removed == []
        assert result.not_found == ["gone.txt", "nope.txt"]
        assert load_skip_kinds(cfg.data_root)["gone.txt"] is SkipKind.REMOVED

    def test_removable_names_lists_indexed_then_failures(self, isolated_env, mock_svc):
        from lilbee.app.ingest import removable_names
        from lilbee.data.ingest.skip_marker import SkipKind, SkipRecords, write_skip_records

        mock_svc.store.upsert_source("a.txt", "hash", 1)
        write_skip_records(
            cfg.data_root,
            SkipRecords(
                markers={"a.txt": "h0", "b.pdf": "h1", "gone.txt": "h2"},
                kinds={"gone.txt": SkipKind.REMOVED},
            ),
        )

        assert removable_names() == ["a.txt", "b.pdf"]


class TestOcrConfigSelection:
    """extraction_config picks the OCR backend from ocr + vision_model."""

    _VISION = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"

    @pytest.mark.parametrize("vision_model", ["", _VISION])
    @pytest.mark.parametrize("mode", [ExtractMode.PAGINATED, ExtractMode.MARKDOWN])
    def test_no_ocr_block_when_ocr_is_off(self, isolated_env, mode, vision_model):
        from lilbee.data.ingest import extraction_config

        cfg.ocr = OcrMode.OFF
        cfg.vision_model = vision_model
        config = extraction_config(mode, ocr_token="tok-123")
        assert config.ocr is None
        assert config.disable_ocr is True
        assert config.force_ocr is False

    @pytest.mark.parametrize("mode", [ExtractMode.PAGINATED, ExtractMode.MARKDOWN])
    def test_tesseract_reads_when_no_vision_model_is_set(self, isolated_env, mode):
        from lilbee.data.ingest import extraction_config

        cfg.ocr = OcrMode.AUTO
        cfg.vision_model = ""
        config = extraction_config(mode)
        assert config.ocr.backend == OcrBackendName.TESSERACT
        assert config.disable_ocr is False

    def test_vision_backend_with_token_when_model_set(self, isolated_env):
        from lilbee.data.ingest import ExtractMode, extraction_config

        cfg.ocr = OcrMode.AUTO
        cfg.vision_model = self._VISION
        config = extraction_config(ExtractMode.PAGINATED, ocr_token="tok-123")
        assert config.ocr.backend == "lilbee-vision"
        assert json.loads(config.ocr.backend_options)["req"] == "tok-123"

    def test_force_ocr_off_under_auto(self, isolated_env):
        from lilbee.data.ingest import ExtractMode, extraction_config

        cfg.ocr = OcrMode.AUTO
        cfg.vision_model = self._VISION
        assert extraction_config(ExtractMode.PAGINATED).force_ocr is False

    @pytest.mark.parametrize(
        ("vision_model", "backend"),
        [(_VISION, OcrBackendName.LILBEE_VISION), ("", OcrBackendName.TESSERACT)],
    )
    @pytest.mark.parametrize("mode", [ExtractMode.PAGINATED, ExtractMode.MARKDOWN])
    def test_ocr_all_forces_ocr_for_either_engine(self, isolated_env, mode, vision_model, backend):
        from lilbee.data.ingest import extraction_config

        cfg.ocr = OcrMode.ALL
        cfg.vision_model = vision_model
        config = extraction_config(mode)
        assert config.ocr.backend == backend
        assert config.force_ocr is True

    def test_a_per_request_all_forces_ocr_over_the_setting(self, isolated_env):
        from lilbee.data.extract.document import ocr_override
        from lilbee.data.ingest import ExtractMode, extraction_config

        cfg.ocr = OcrMode.AUTO
        cfg.vision_model = ""
        with ocr_override(ocr=OcrMode.ALL):
            assert extraction_config(ExtractMode.PAGINATED).force_ocr is True
        assert extraction_config(ExtractMode.PAGINATED).force_ocr is False


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
    """Behavioral contract against real xberg: ocr = all must OCR every page of a
    born-digital PDF through the registered vision backend, defeating xberg's
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

    def test_native_text_skips_ocr_without_force(self, isolated_env):
        from lilbee.data.ingest import ExtractMode, extraction_config

        cfg.ocr = OcrMode.AUTO
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        config = extraction_config(ExtractMode.PAGINATED)
        calls, content = self._extract(make_pdf(pages=2), config)
        assert calls == 0
        assert "clean native text layer" in content

    def test_ocr_all_reads_every_page(self, isolated_env):
        from lilbee.data.ingest import ExtractMode, extraction_config

        cfg.ocr = OcrMode.ALL
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        config = extraction_config(ExtractMode.PAGINATED)
        calls, content = self._extract(make_pdf(pages=2), config)
        assert calls == 2  # one OCR call per page, text layer notwithstanding
        assert "OCR-TEXT" in content
        assert "clean native text layer" not in content


_SHORT_TEXT_LINES = ("a perfectly clean native text layer.",)
_LONG_TEXT_LINES = ("a long and perfectly clean native text layer with many words.",) * 4


# Small print with characters Tesseract confuses: OCR of this line returns other characters.
_CONFUSABLE_LINE = "ref l1O0-Il|5S8B: 0.00 1,000.50"
_SMALL_PRINT_POINTS = 5
_BODY_POINTS = 12
# xberg's floor of non-blank characters for a page to count as having usable text.
_USABLE_TEXT_CHARS = 32
_SCAN_SIZE = (1240, 1754)
_HALF_SCAN_SIZE = (1240, 877)


def _page_line(number: int, line: str) -> str:
    """The text a T page of ``_layout_pdf`` draws for *line*."""
    return f"Page {number} with {line}"


def _scan_line(number: int, row: int) -> str:
    """The text an S page of ``_layout_pdf`` shows on *row* of its image."""
    return f"Scanned page {number} line {row} with words"


def _non_blank(text: str) -> int:
    return len("".join(text.split()))


def _draw_scan(pdf, number: int, size: tuple[int, int]) -> None:
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib.utils import ImageReader

    scan = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(scan)
    font = ImageFont.load_default(size=36)
    for row in range(size[1] // 110):
        draw.text((100, 60 + row * 100), _scan_line(number, row), fill="black", font=font)
    pdf.drawImage(ImageReader(scan), 0, 0, width=595, height=842 * size[1] // _SCAN_SIZE[1])


def _draw_text_page(pdf, number: int, text_lines: tuple[str, ...]) -> None:
    for row, line in enumerate(text_lines):
        pdf.drawString(72, 720 - row * 20, _page_line(number, line))


def _draw_scanned_page(pdf, number: int, _text_lines: tuple[str, ...]) -> None:
    _draw_scan(pdf, number, _SCAN_SIZE)


def _draw_image_and_text_page(pdf, number: int, text_lines: tuple[str, ...]) -> None:
    _draw_scan(pdf, number, _HALF_SCAN_SIZE)
    _draw_text_page(pdf, number, text_lines)


def _draw_page_number_page(pdf, number: int, _text_lines: tuple[str, ...]) -> None:
    pdf.drawString(290, 30, str(number))


def _draw_whitespace_page(pdf, _number: int, _text_lines: tuple[str, ...]) -> None:
    pdf.drawString(72, 720, "     ")


def _draw_hidden_text_page(pdf, number: int, text_lines: tuple[str, ...]) -> None:
    pdf.setFillColorRGB(1, 1, 1)
    _draw_text_page(pdf, number, text_lines)


# T: a text layer. S: an image of text. B: an image of text under a text layer.
# N: a page number only. W: whitespace only. H: a text layer drawn in white, so no OCR sees it.
_PAGE_DRAWERS = {
    "T": _draw_text_page,
    "S": _draw_scanned_page,
    "B": _draw_image_and_text_page,
    "N": _draw_page_number_page,
    "W": _draw_whitespace_page,
    "H": _draw_hidden_text_page,
}


def _layout_pdf(
    layout: str,
    text_lines: tuple[str, ...] = _SHORT_TEXT_LINES,
    points: int = _BODY_POINTS,
) -> bytes:
    """A PDF with one page per letter of *layout*, drawn by ``_PAGE_DRAWERS``."""
    import io

    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    for number, kind in enumerate(layout, start=1):
        pdf.setFont("Helvetica", points)
        _PAGE_DRAWERS[kind](pdf, number, text_lines)
        pdf.showPage()
    pdf.save()
    return buf.getvalue()


def _mixed_pdf() -> bytes:
    """A three-page PDF whose page 2 is an image of text with no text layer."""
    return _layout_pdf("TST")


class TestTesseractUnderRealXberg:
    """ocr modes against real xberg with Tesseract, no vision model."""

    @pytest.fixture(autouse=True)
    def _tesseract(self, isolated_env, tessdata_ready):
        cfg.vision_model = ""

    @staticmethod
    def _extract(pdf: bytes, ocr: OcrMode):
        from lilbee.data.extract.xberg import extract_document
        from lilbee.data.ingest import ExtractMode, extraction_config

        cfg.ocr = ocr
        config = extraction_config(ExtractMode.PAGINATED)
        return extract_document(pdf, "application/pdf", filename="f.pdf", config=config)

    def _ocr_pages(self, pdf: bytes, ocr: OcrMode) -> list[int]:
        pages = self._extract(pdf, ocr).pages
        return [page.page_number for page in pages if page.ocr_confidence is not None]

    def test_ocr_all_rereads_a_born_digital_pdf(self):
        assert self._ocr_pages(make_pdf(pages=2), OcrMode.AUTO) == []
        assert self._ocr_pages(make_pdf(pages=2), OcrMode.ALL) == [1, 2]

    def test_auto_reads_only_the_scanned_page_and_off_reads_none(self):
        assert self._ocr_pages(_mixed_pdf(), OcrMode.AUTO) == [2]
        assert self._ocr_pages(_mixed_pdf(), OcrMode.OFF) == []

    @pytest.mark.parametrize(
        ("layout", "text_lines", "expected"),
        [
            pytest.param("SST", _LONG_TEXT_LINES, [1, 2], id="scans-then-a-paragraph"),
            pytest.param("SST", _SHORT_TEXT_LINES, [1, 2], id="scans-then-one-line"),
            pytest.param("TSS", _SHORT_TEXT_LINES, [2, 3], id="one-line-then-scans"),
            pytest.param("TTSS", _SHORT_TEXT_LINES, [3, 4], id="half-scans"),
            pytest.param("SSS", _SHORT_TEXT_LINES, [1, 2, 3], id="every-page-scanned"),
            pytest.param("TTT", _SHORT_TEXT_LINES, [], id="no-page-scanned"),
            pytest.param("SBT", _LONG_TEXT_LINES, [1], id="image-and-text-on-one-page"),
            pytest.param("TNT", _LONG_TEXT_LINES, [2], id="page-number-only"),
            pytest.param("TWT", _LONG_TEXT_LINES, [2], id="whitespace-only"),
        ],
    )
    def test_auto_reads_only_the_pages_without_usable_text(self, layout, text_lines, expected):
        assert self._ocr_pages(_layout_pdf(layout, text_lines), OcrMode.AUTO) == expected

    @pytest.mark.parametrize(
        ("chars", "expected"),
        [(_USABLE_TEXT_CHARS - 1, [1, 2, 3]), (_USABLE_TEXT_CHARS, [1, 3])],
    )
    def test_a_page_has_usable_text_from_32_non_blank_characters(self, chars, expected):
        line = "x" * (chars - _non_blank(_page_line(2, "")))
        assert _non_blank(_page_line(2, line)) == chars
        assert self._ocr_pages(_layout_pdf("STS", (line,)), OcrMode.AUTO) == expected

    def test_ocr_text_replaces_a_text_layer_too_short_to_be_usable(self):
        line = "x" * (_USABLE_TEXT_CHARS - 1 - _non_blank(_page_line(2, "")))
        usable = "y" * (_USABLE_TEXT_CHARS - _non_blank(_page_line(3, "")))
        doc = self._extract(_layout_pdf("SHT", (line,)), OcrMode.AUTO)
        kept = self._extract(_layout_pdf("SHT", (usable,)), OcrMode.AUTO)
        assert _page_line(2, line) not in doc.pages[1].content
        assert _page_line(2, usable) in kept.pages[1].content
        assert _page_line(3, usable) in kept.pages[2].content

    @pytest.mark.parametrize("layout", ["STS", "SST", "TSS"])
    async def test_auto_keeps_native_text_and_adds_the_ocr_text_of_scanned_pages(
        self, isolated_env, mock_svc, layout
    ):
        """The text page is one small-print line that OCR misreads, so reading it fails this."""
        from lilbee.data.ingest import ingest_document

        cfg.ocr = OcrMode.AUTO
        f = isolated_env / "mixed.pdf"
        f.write_bytes(_layout_pdf(layout, (_CONFUSABLE_LINE,), points=_SMALL_PRINT_POINTS))
        pages: list = []
        await ingest_document(f, "mixed.pdf", "pdf", page_texts_out=pages)
        assert [page["page"] for page in pages] == [1, 2, 3]
        texts = {page["page"]: " ".join(page["text"].split()) for page in pages}
        native = {n: _page_line(n, _CONFUSABLE_LINE) for n in (1, 2, 3) if layout[n - 1] == "T"}
        scanned = {n: _scan_line(n, 0) for n in (1, 2, 3) if layout[n - 1] == "S"}
        assert {n: texts[n] for n in native} == native
        assert all(first_row in texts[n] for n, first_row in scanned.items())
        whole = " ".join(texts[n] for n in (1, 2, 3))
        assert [whole.count(part) for part in (*native.values(), *scanned.values())] == [1, 1, 1]

    @pytest.mark.parametrize(("ocr", "warned"), [(OcrMode.OFF, True), (OcrMode.AUTO, False)])
    async def test_a_mixed_pdf_under_ocr_off_names_its_skipped_pages(
        self, isolated_env, mock_svc, caplog, ocr, warned
    ):
        from lilbee.data.ingest import ingest_document

        cfg.ocr = ocr
        f = isolated_env / "mixed.pdf"
        f.write_bytes(_mixed_pdf())
        with caplog.at_level("WARNING", logger="lilbee.data.extract.document"):
            records = await ingest_document(f, "mixed.pdf", "pdf")
        assert records.records  # the text pages are indexed either way
        expected = "Indexed mixed.pdf without its scanned pages 2: OCR is off (ocr = off)"
        assert (expected in caplog.text) is warned


class TestVisionOcrStopsOnCancel:
    """A set cancel stops the vision model before each page, against real xberg."""

    _PAGES = 4

    @pytest.fixture(autouse=True)
    def _forced_vision(self, isolated_env):
        cfg.ocr = OcrMode.ALL
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"

    @staticmethod
    @contextlib.contextmanager
    def _vision_backend(calls: list[bytes], set_on_call: threading.Event | None):
        from xberg import register_ocr_backend, unregister_ocr_backend

        from lilbee.data.extract.backends.vision_ocr import VisionOcrBackend

        def _ocr(image_bytes, _model, _prompt, *, timeout, cancel):
            calls.append(image_bytes)
            if set_on_call is not None:
                set_on_call.set()
            return "OCR-TEXT"

        backend = VisionOcrBackend(ocr_fn=_ocr, model_ref_fn=lambda: cfg.vision_model)
        register_ocr_backend(backend)
        try:
            yield
        finally:
            unregister_ocr_backend(OcrBackendName.LILBEE_VISION)

    async def _extract(self, tmp_path, cancel, calls: list[bytes], *, cancel_on_call=False) -> None:
        from lilbee.data.extract.document import _extract_document

        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(make_pdf(pages=self._PAGES))
        with (
            self._vision_backend(calls, cancel if cancel_on_call else None),
            mock.patch(
                "lilbee.data.extract.backends.vision_ocr.resolve_ocr_prompt",
                return_value="OCR",
            ),
        ):
            await _extract_document(
                pdf, "scan.pdf", "pdf", ExtractMode.PAGINATED, lambda *_a: None, cancel
            )

    async def test_every_page_reaches_the_model_without_a_cancel(self, tmp_path):
        calls: list[bytes] = []
        await self._extract(tmp_path, threading.Event(), calls)
        assert len(calls) == self._PAGES

    async def test_a_set_cancel_sends_no_page_to_the_model(self, tmp_path):
        cancel = threading.Event()
        cancel.set()
        calls: list[bytes] = []
        with pytest.raises(RuntimeError, match="TaskCancelledError"):
            await self._extract(tmp_path, cancel, calls)
        assert calls == []

    @pytest.mark.parametrize("archived", [False, True])
    async def test_a_cancelled_sync_sends_no_page_to_the_model(
        self, isolated_env, mock_svc, archived
    ):
        """The sync's cancel reaches vision OCR for a file and for an archive member."""
        import zipfile

        from lilbee.data.ingest import sync
        from lilbee.runtime.progress import EventType

        mock_svc.provider.vision_slot_capacity.return_value = 1
        pdf = make_pdf(pages=self._PAGES)
        if archived:
            with zipfile.ZipFile(isolated_env / "scans.zip", "w") as archive:
                archive.writestr("scan.pdf", pdf)
        else:
            (isolated_env / "scan.pdf").write_bytes(pdf)
        cancel = threading.Event()

        def _on_progress(event_type, _data) -> None:
            if event_type is EventType.FILE_START:
                cancel.set()

        uncancelled: list[bytes] = []
        with (
            self._vision_backend(uncancelled, None),
            mock.patch(
                "lilbee.data.extract.backends.vision_ocr.resolve_ocr_prompt",
                return_value="OCR",
            ),
        ):
            await sync(quiet=True, force_rebuild=True, cancel=threading.Event())
        assert len(uncancelled) == self._PAGES  # the fixture does reach the vision model

        calls: list[bytes] = []
        with (
            self._vision_backend(calls, None),
            mock.patch(
                "lilbee.data.extract.backends.vision_ocr.resolve_ocr_prompt",
                return_value="OCR",
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await sync(quiet=True, force_rebuild=True, on_progress=_on_progress, cancel=cancel)
        assert cancel.is_set()
        assert calls == []

    async def test_a_cancel_during_extraction_returns_no_partial_document(self, tmp_path):
        cancel = threading.Event()
        calls: list[bytes] = []
        with pytest.raises(asyncio.CancelledError):
            await self._extract(tmp_path, cancel, calls, cancel_on_call=True)
        assert calls  # a page was read before the cancel, so xberg had a document to return


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
        result, _, _ = await ingest_document(f, "scan.pdf", "pdf")
        assert result == []
        assert "no usable text" in caplog.text

    @staticmethod
    async def _ingest_reporting(mock_kf, isolated_env, reports) -> list[tuple[int, int, str]]:
        """Ingest scan.pdf while the extraction reports *reports* as xberg OCR page events.

        Each report is (page, total, completed, backend). Returns the EXTRACT
        events seen while the extraction ran, which excludes the per-file summary.
        """
        from xberg import ProgressEvent

        from lilbee.data.ingest import ingest_document
        from lilbee.runtime.progress import EventType

        seen: list[tuple[int, int, str]] = []
        during_extraction: list[tuple[int, int, str]] = []

        async def fake_extract(data, *, filename=None, config, on_progress=None):
            if on_progress is None:
                return mock.MagicMock(counts=mock.MagicMock(pages=0), metadata=Metadata())
            for page, total, completed, backend in reports:
                on_progress(ProgressEvent("ocr_page", page, total, completed, backend))
            during_extraction.extend(seen)
            return _make_xberg_result(num_chunks=1, has_pages=True)

        def on_prog(event_type, ev):
            if event_type is EventType.EXTRACT:
                seen.append((ev.page, ev.total_pages, ev.ocr_backend))

        mock_kf.side_effect = fake_extract
        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        await ingest_document(f, "scan.pdf", "pdf", on_progress=on_prog)
        return during_extraction

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_a_page_reported_twice_counts_once(self, mock_kf, isolated_env, mock_svc):
        """xberg reports page 5 twice without advancing ``completed``; so does lilbee."""
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        vision = OcrBackendName.LILBEE_VISION
        ticks = await self._ingest_reporting(
            mock_kf,
            isolated_env,
            [(5, 9, 1, vision), (5, 9, 1, vision), (2, 9, 2, vision)],
        )
        assert [(page, total) for page, total, _ in ticks] == [(1, 9), (1, 9), (2, 9)]

    @pytest.mark.parametrize(
        ("vision_model", "expected"),
        [
            pytest.param(
                "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf", OcrBackendUsed.VISION, id="vision"
            ),
            pytest.param("", OcrBackendUsed.TESSERACT, id="tesseract"),
        ],
    )
    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_page_ticks_carry_xbergs_total_and_the_configured_backend(
        self, mock_kf, vision_model, expected, isolated_env, mock_svc
    ):
        """The backend comes from lilbee's configuration, whatever name xberg's event carries."""
        cfg.vision_model = vision_model
        cfg.ocr = OcrMode.AUTO
        ticks = await self._ingest_reporting(mock_kf, isolated_env, [(4, 9, 1, "paddle-ocr")])
        assert ticks == [(1, 9, expected)]

    @pytest.mark.parametrize("content_type", ["pdf", "image"])
    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_a_pdf_or_image_extraction_hands_xberg_a_progress_callback(
        self, mock_kf, content_type, isolated_env, mock_svc
    ):
        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        mock_kf.return_value = _make_xberg_result(num_chunks=1, has_pages=True)
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "scan.bin"
        f.write_bytes(b"x")
        await ingest_document(f, "scan.bin", content_type)
        mock_kf.assert_awaited_once()
        assert callable(mock_kf.await_args.kwargs["on_progress"])

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_page_count_probe_sends_no_ocr_block(self, mock_kf, isolated_env, mock_svc):
        """The probe is metadata-only: it carries no OCR block and sets disable_ocr."""
        cfg.vision_model = ""
        cfg.ocr = OcrMode.AUTO
        probe_result = mock.MagicMock(counts=mock.MagicMock(pages=5))
        configs = []

        async def fake_extract(data, *, filename=None, config, on_progress=None):
            configs.append(config)
            if config.disable_ocr:
                return probe_result
            return _make_xberg_result(num_chunks=1, has_pages=True)

        mock_kf.side_effect = fake_extract
        from lilbee.data.ingest import ingest_document
        from lilbee.data.types import OcrBackendName

        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        await ingest_document(f, "scan.pdf", "pdf", on_progress=lambda *_: None)
        probes = [c for c in configs if c.disable_ocr]
        extractions = [c for c in configs if not c.disable_ocr]
        assert len(probes) == 1 and len(extractions) == 1
        assert probes[0].ocr is None
        assert extractions[0].ocr.backend == OcrBackendName.TESSERACT


_VISION_MODEL = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"


class _CancelAtFirstPage:
    """A progress callback that sets its cancel at the first OCR page event.

    It raises on every event from then on, as the callback of a cancelled sync does.
    """

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.events = 0

    def __call__(self, event_type, _data) -> None:
        from lilbee.runtime.cancellation import TaskCancelledError
        from lilbee.runtime.progress import EventType

        self.events += 1
        if event_type is EventType.EXTRACT:
            self.cancel.set()
        if self.cancel.is_set():
            raise TaskCancelledError


class TestOcrCancel:
    """A cancel that lands at an OCR page event stops the vision OCR of that extraction."""

    @staticmethod
    def _vision_backend():
        from lilbee.data.extract.backends.vision_ocr import VisionOcrBackend

        calls: list[bytes] = []

        def ocr_fn(image_bytes, _model, _prompt, *, timeout, cancel):
            calls.append(image_bytes)
            return "OCR-TEXT"

        return VisionOcrBackend(ocr_fn=ocr_fn, model_ref_fn=lambda: _VISION_MODEL), calls

    @staticmethod
    def _ocr_pages_like_xberg(backend, ocr_config, on_page, pages: int) -> None:
        """OCR *pages* pages the way xberg does: a raise fails one page or one report only."""
        from xberg import ProgressEvent

        for page in range(1, pages + 1):
            try:
                backend.process_image(b"PNG", ocr_config)
                on_page(ProgressEvent("ocr_page", page, pages, page, ocr_config.backend))
            except Exception:  # noqa: S112  # xberg isolates a failed page and a raising callback
                continue

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_cancel_on_the_first_page_event_skips_the_ocr_of_later_pages(
        self, mock_kf, isolated_env, mock_svc
    ):
        from lilbee.data.ingest import ingest_document

        cfg.vision_model = _VISION_MODEL
        backend, calls = self._vision_backend()

        async def fake_extract(data, *, filename=None, config, on_progress=None):
            self._ocr_pages_like_xberg(backend, config.ocr, on_progress, pages=5)
            return _make_xberg_result(num_chunks=1, has_pages=True)

        mock_kf.side_effect = fake_extract
        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        on_progress = _CancelAtFirstPage()
        with pytest.raises(asyncio.CancelledError):
            await ingest_document(
                f, "scan.pdf", "pdf", on_progress=on_progress, cancel=on_progress.cancel
            )
        assert len(calls) == 1
        # The extraction ends as cancelled before the per-file summary event.
        assert on_progress.events == 1

    async def test_cancel_on_the_first_page_event_skips_later_pages_in_a_batch(
        self, isolated_env, mock_svc
    ):
        from lilbee.data.extract.batch import (
            ExtractBatcher,
            reset_active_batcher,
            set_active_batcher,
        )
        from lilbee.data.extract.document import _ocr_config, extraction_config
        from lilbee.data.ingest import ingest_document

        cfg.vision_model = _VISION_MODEL
        backend, calls = self._vision_backend()

        async def batch_fn(items, config):
            for item in items:
                self._ocr_pages_like_xberg(backend, item.ocr, item.on_progress, pages=5)
            return [_make_xberg_result(num_chunks=1, has_pages=True) for _ in items]

        batcher = ExtractBatcher(
            size=1, config_fn=extraction_config, ocr_fn=_ocr_config, batch_fn=batch_fn
        )
        token = set_active_batcher(batcher)
        try:
            f = isolated_env / "scan.pdf"
            f.write_bytes(b"x")
            on_progress = _CancelAtFirstPage()
            with pytest.raises(asyncio.CancelledError):
                await ingest_document(
                    f, "scan.pdf", "pdf", on_progress=on_progress, cancel=on_progress.cancel
                )
        finally:
            await batcher.close()
            reset_active_batcher(token)
        assert len(calls) == 1

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_without_a_cancel_every_page_reaches_the_ocr_function(
        self, mock_kf, isolated_env, mock_svc
    ):
        from lilbee.data.ingest import ingest_document

        cfg.vision_model = _VISION_MODEL
        backend, calls = self._vision_backend()

        async def fake_extract(data, *, filename=None, config, on_progress=None):
            self._ocr_pages_like_xberg(backend, config.ocr, on_progress, pages=5)
            return _make_xberg_result(num_chunks=1, has_pages=True)

        mock_kf.side_effect = fake_extract
        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        records, _, _ = await ingest_document(f, "scan.pdf", "pdf", cancel=threading.Event())
        assert len(calls) == 5
        assert len(records) == 1

    @pytest.mark.parametrize("cancelled", [True, False])
    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_a_sync_records_a_failed_extraction_only_without_a_cancel(
        self, mock_kf, cancelled, isolated_env, mock_svc
    ):
        """xberg fails an extraction whose every page was refused; a cancel is not a failure."""
        from lilbee.data.ingest import sync

        mock_svc.provider.vision_slot_capacity.return_value = 1
        cfg.vision_model = _VISION_MODEL
        backend, _ = self._vision_backend()

        async def fake_extract(data, *, filename=None, config, on_progress=None):
            self._ocr_pages_like_xberg(backend, config.ocr, on_progress, pages=2)
            raise RuntimeError("OCR failed on all 2 page(s)")

        mock_kf.side_effect = fake_extract
        (isolated_env / "scan.pdf").write_bytes(b"x")
        if cancelled:
            on_progress = _CancelAtFirstPage()
            with pytest.raises(asyncio.CancelledError):
                await sync(quiet=True, on_progress=on_progress, cancel=on_progress.cancel)
        else:
            result = await sync(quiet=True, cancel=threading.Event())
            assert result.failed == ["scan.pdf"]

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_an_extraction_that_fails_without_a_cancel_keeps_its_error(
        self, mock_kf, isolated_env, mock_svc
    ):
        from lilbee.data.ingest import ingest_document

        cfg.vision_model = _VISION_MODEL
        mock_kf.side_effect = RuntimeError("corrupt file")
        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        with pytest.raises(RuntimeError, match="corrupt file"):
            await ingest_document(f, "scan.pdf", "pdf")

    async def test_real_xberg_stops_calling_the_vision_model_after_a_cancel(
        self, isolated_env, mock_svc
    ):
        """Real xberg: a cancel on the first page event leaves most pages un-OCR'd."""
        from xberg import register_ocr_backend, unregister_ocr_backend

        from lilbee.data.ingest import ingest_document
        from lilbee.runtime.cpu import cpu_quota

        # xberg OCRs one page per thread at once, so that many pages can start before the cancel.
        pages = 4 * max(cpu_quota(), 8)
        cfg.ocr = OcrMode.ALL
        cfg.vision_model = _VISION_MODEL
        backend, calls = self._vision_backend()
        f = isolated_env / "scan.pdf"
        f.write_bytes(make_pdf(pages=pages))
        on_progress = _CancelAtFirstPage()
        register_ocr_backend(backend)
        try:
            with pytest.raises(asyncio.CancelledError):
                await ingest_document(
                    f, "scan.pdf", "pdf", on_progress=on_progress, cancel=on_progress.cancel
                )
        finally:
            unregister_ocr_backend(OcrBackendName.LILBEE_VISION)
        assert 1 <= len(calls) < pages // 2


class TestTesseractOcrStartEvent:
    """A scanned file under Tesseract gets one OCR_START event before its extraction."""

    @staticmethod
    def _probe(pages: int, scanned: list[int]) -> mock.MagicMock:
        from xberg import FormatMetadata, PdfMetadata

        fmt = FormatMetadata.from_pdf(PdfMetadata(page_count=pages, scanned_pages=scanned))
        return mock.MagicMock(counts=mock.MagicMock(pages=pages), metadata=Metadata(format=fmt))

    async def _ingest(self, mock_kf, isolated_env, probe) -> list[tuple[object, object]]:
        order: list[tuple[object, object]] = []

        async def fake_extract(data, *, filename=None, config, on_progress=None):
            if config.disable_ocr:
                return probe
            order.append(("extract", None))
            return _make_xberg_result(num_chunks=1, has_pages=True)

        mock_kf.side_effect = fake_extract
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        await ingest_document(
            f, "scan.pdf", "pdf", on_progress=lambda et, ev: order.append((et, ev))
        )
        return order

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_scanned_pdf_announces_ocr_once_before_extraction(
        self, mock_kf, isolated_env, mock_svc
    ):
        from lilbee.runtime.progress import EventType, OcrStartEvent

        cfg.vision_model = ""
        cfg.ocr = OcrMode.AUTO
        order = await self._ingest(mock_kf, isolated_env, self._probe(8, [1, 2, 3]))
        starts = [ev for et, ev in order if et is EventType.OCR_START]
        assert starts == [OcrStartEvent(file="scan.pdf", total_pages=8)]
        kinds = [et for et, _ in order]
        assert kinds.index(EventType.OCR_START) < kinds.index("extract")
        # The per-file "extracted N pages" event follows the extraction and names Tesseract.
        extracts = [ev for et, ev in order if et is EventType.EXTRACT]
        assert [ev.ocr_backend for ev in extracts] == [OcrBackendUsed.TESSERACT]

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_extract_event_names_no_backend_when_no_page_was_ocrd(
        self, mock_kf, isolated_env, mock_svc
    ):
        """A text-layer PDF under Tesseract reports its pages as extracted, not OCR'd."""
        from lilbee.runtime.progress import EventType

        cfg.vision_model = ""
        cfg.ocr = OcrMode.AUTO
        probe = self._probe(2, [])
        text_pdf = _make_xberg_result(num_chunks=2, has_pages=True)
        for page in text_pdf.pages:
            page.ocr_confidence = None

        async def fake_extract(data, *, filename=None, config, on_progress=None):
            return probe if config.disable_ocr else text_pdf

        mock_kf.side_effect = fake_extract
        from lilbee.data.ingest import ingest_document

        order: list[tuple[object, object]] = []
        f = isolated_env / "text.pdf"
        f.write_bytes(b"x")
        await ingest_document(
            f, "text.pdf", "pdf", on_progress=lambda et, ev: order.append((et, ev))
        )
        extracts = [ev for et, ev in order if et is EventType.EXTRACT]
        assert [(ev.page, ev.ocr_backend) for ev in extracts] == [(2, OcrBackendUsed.NONE)]

    @pytest.mark.parametrize(
        ("vision_model", "ocr", "scanned"),
        [
            pytest.param("", OcrMode.AUTO, [], id="text-pdf-needs-no-ocr"),
            pytest.param(
                "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf",
                OcrMode.AUTO,
                [1, 2],
                id="vision-ticks-per-page",
            ),
            pytest.param("", OcrMode.OFF, [1, 2], id="ocr-off"),
        ],
    )
    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_announces_nothing_when_tesseract_does_not_ocr(
        self, mock_kf, vision_model, ocr, scanned, isolated_env, mock_svc
    ):
        """No OCR_START unless Tesseract is the backend and the probe found a scanned page."""
        from lilbee.runtime.progress import EventType

        cfg.vision_model = vision_model
        cfg.ocr = ocr
        order = await self._ingest(mock_kf, isolated_env, self._probe(8, scanned))
        assert "extract" in [et for et, _ in order]
        assert EventType.OCR_START not in [et for et, _ in order]

    @pytest.mark.parametrize(
        ("vision_model", "ocr", "content_type", "probes"),
        [
            pytest.param("", OcrMode.AUTO, "pdf", 1, id="tesseract-pdf-is-probed"),
            pytest.param("", OcrMode.AUTO, "image", 1, id="tesseract-image-is-probed"),
            pytest.param("", OcrMode.AUTO, "text", 0, id="tesseract-unpaginated"),
            pytest.param(
                "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf", OcrMode.AUTO, "pdf", 0, id="vision"
            ),
            pytest.param("", OcrMode.OFF, "pdf", 0, id="ocr-off"),
        ],
    )
    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_only_a_tesseract_pdf_or_image_pays_for_the_page_probe(
        self, mock_kf, vision_model, ocr, content_type, probes, isolated_env, mock_svc
    ):
        """The metadata-only pass runs only where its result decides the OCR_START event."""
        cfg.vision_model = vision_model
        cfg.ocr = ocr
        configs = []

        async def fake_extract(data, *, filename=None, config, on_progress=None):
            configs.append(config)
            return _make_xberg_result(num_chunks=1, has_pages=True)

        mock_kf.side_effect = fake_extract
        from lilbee.data.ingest import ingest_document

        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        await ingest_document(f, "scan.pdf", content_type)
        assert len(configs) == probes + 1
        assert [c.pages.extract_pages for c in configs[:probes]] == [False] * probes


def _ocr_sent(config) -> tuple[str | None, bool]:
    """The OCR backend an xberg config carries (None: no OCR block) and its disable_ocr."""
    return (config.ocr.backend if config.ocr is not None else None, config.disable_ocr)


class TestOcrOffSendsNoOcrBlock:
    """With OCR off, the extraction xberg receives has no OCR block and disable_ocr set.

    xberg auto-OCRs a PDF with no text layer unless disable_ocr is set.
    """

    @pytest.mark.parametrize(
        ("ocr", "expected"),
        [(OcrMode.OFF, (None, True)), (OcrMode.AUTO, (OcrBackendName.TESSERACT, False))],
    )
    async def test_single_file_extraction(self, isolated_env, ocr, expected):
        from lilbee.data.ingest import ingest_document

        cfg.vision_model = ""
        cfg.ocr = ocr
        extractions = []

        async def fake_extract(_input, config, _on_progress):
            if config.pages.extract_pages:
                extractions.append(config)
            return mock.MagicMock(results=[_make_xberg_result(has_pages=True)])

        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        with mock.patch("xberg.progress.extract", side_effect=fake_extract):
            await ingest_document(f, "scan.pdf", "pdf")
        assert [_ocr_sent(c) for c in extractions] == [expected]

    @pytest.mark.parametrize(
        ("ocr", "expected"),
        [(OcrMode.OFF, (None, True)), (OcrMode.AUTO, (OcrBackendName.TESSERACT, False))],
    )
    async def test_batch_extraction(self, isolated_env, ocr, expected):
        from lilbee.data.extract.batch import reset_active_batcher, set_active_batcher
        from lilbee.data.extract.document import make_extract_batcher
        from lilbee.data.ingest import ingest_document

        cfg.vision_model = ""
        cfg.ocr = ocr
        cfg.batch_extraction = True
        sent = []

        async def fake_batch(inputs, config, _on_progress):
            sent.append((config, [i.config for i in inputs]))
            return mock.MagicMock(results=[_make_xberg_result() for _ in inputs], errors=[])

        f = isolated_env / "scan.pdf"
        f.write_bytes(b"x")
        batcher = make_extract_batcher()
        with mock.patch("xberg.progress.extract_batch", side_effect=fake_batch):
            token = set_active_batcher(batcher)
            try:
                await ingest_document(f, "scan.pdf", "pdf")
            finally:
                await batcher.close()
                reset_active_batcher(token)
        [(batch_config, file_configs)] = sent
        assert _ocr_sent(batch_config) == expected
        file_ocr = [None if c is None else _ocr_sent(c)[0] for c in file_configs]
        assert file_ocr == [expected[0]]


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
        assert [(move.candidates, move.new) for move in moves] == [(("old/a.txt",), "new/a.txt")]

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

    def test_files_of_one_hash_are_each_offered_every_candidate_in_name_order(self):
        from lilbee.data.ingest import pipeline

        files = [self._entry("new/a.txt", "h1"), self._entry("new/b.txt", "h1")]
        added = {"new/a.txt": None, "new/b.txt": None}
        existing = {
            "old/a.txt": self._record("old/a.txt", "h1"),
            "old/b.txt": self._record("old/b.txt", "h1"),
        }
        pool = pipeline._MovePool(["old/b.txt", "old/a.txt"], existing)
        moves = pipeline._detect_moves(files, added, pool)
        assert [move.new for move in moves] == ["new/a.txt", "new/b.txt"]
        assert {move.candidates for move in moves} == {("old/a.txt", "old/b.txt")}


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

    def test_false_when_target_is_the_registered_source(self, isolated_env, tmp_path):
        """Re-adding the exact path already registered under the label is not a
        collision: register_sources treats it as a no-op, so no dialog either."""
        from lilbee.app.ingest import source_label_taken

        src = tmp_path / "report.pdf"
        src.write_bytes(b"x")
        cfg.linked_roots = {"report.pdf": str(src.resolve())}
        assert source_label_taken("report.pdf", src) is False

    def test_true_when_target_differs_from_registered_source(self, isolated_env, tmp_path):
        """A different file whose basename collides with a live root still counts."""
        from lilbee.app.ingest import source_label_taken

        registered = tmp_path / "a" / "report.pdf"
        registered.parent.mkdir()
        registered.write_bytes(b"x")
        cfg.linked_roots = {"report.pdf": str(registered.resolve())}
        other = tmp_path / "b" / "report.pdf"
        other.parent.mkdir()
        other.write_bytes(b"y")
        assert source_label_taken("report.pdf", other) is True


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
        assert result.name_taken == []
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

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (RegisterResult(registered=["corpus"]), True),
            (RegisterResult(tracked=["corpus"]), True),
            (RegisterResult(overlapping=["papers"]), True),
            (RegisterResult(name_taken=["corpus"]), False),
            (RegisterResult(), False),
            (RegisterResult(refused=["logo.svg: vector graphic, not a document"]), False),
        ],
    )
    def test_reached_corpus_says_whether_a_sync_has_anything_to_index(self, result, expected):
        """Every add surface asks this before running a whole-vault sync."""
        assert result.reached_corpus is expected

    def test_outside_corpus_names_a_taken_label_and_a_refused_file(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources

        held, taken = tmp_path / "a" / "held", tmp_path / "b" / "held"
        logo = tmp_path / "logo.svg"
        for folder in (held, taken):
            folder.mkdir(parents=True)
        logo.write_text("<svg/>", encoding="utf-8")
        result = register_sources([held, taken, logo, held])
        assert result.registered == ["held"]
        assert result.tracked == ["held"]
        assert result.outside_corpus == ["held", "logo.svg"]

    def test_reregistering_same_path_is_idempotent(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        register_sources([corpus])
        result = register_sources([corpus])
        assert result.registered == []
        assert result.tracked == ["corpus"]  # already tracked, not a collision
        assert result.name_taken == []
        assert cfg.linked_roots == {"corpus": str(corpus.resolve())}

    def test_label_collision_needs_force(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        one = tmp_path / "a" / "corpus"
        one.mkdir(parents=True)
        two = tmp_path / "b" / "corpus"
        two.mkdir(parents=True)
        register_sources([one])
        # Same basename, different live path: name taken without force.
        result = register_sources([two])
        assert result.registered == []
        assert result.name_taken == ["corpus"]
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
        assert result.tracked == ["sub"]  # owned by the documents dir already
        assert result.name_taken == []
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
        assert result.overlapping == ["papers"]
        assert result.overlapping_inside == ["papers"]
        assert (result.containing, result.outside_corpus) == ([], [])
        assert result.name_taken == []
        assert result.reached_corpus is True
        assert cfg.linked_roots == {"corpus": str(corpus.resolve())}

    def test_label_held_by_another_source_is_name_taken_and_not_in_the_corpus(
        self, isolated_env, tmp_path
    ):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        one = tmp_path / "a" / "corpus"
        one.mkdir(parents=True)
        register_sources([one])
        result = register_sources([tmp_path / "b" / "corpus"])
        assert result.name_taken == ["corpus"]
        assert result.overlapping == []
        assert result.reached_corpus is False
        assert cfg.linked_roots == {"corpus": str(one.resolve())}

    def test_a_root_containing_an_existing_root_takes_it_in(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        child = tmp_path / "data" / "corpus"
        child.mkdir(parents=True)
        register_sources([child])
        result = register_sources([tmp_path / "data"])  # a parent of the existing root
        assert result.registered == ["data"]
        assert result.absorbed_into == {"data": ["corpus"]}
        assert (result.overlapping, result.containing, result.outside_corpus) == ([], [], [])
        assert result.reached_corpus is True
        assert cfg.linked_roots == {"data": str((tmp_path / "data").resolve())}

    def test_rejects_root_that_is_ancestor_of_documents_dir(self, isolated_env, tmp_path):
        # documents_dir lives under tmp_path; registering tmp_path would re-index
        # every owned file a second time under the parent label.
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        result = register_sources([cfg.documents_dir.parent])
        assert result.registered == []
        assert result.containing == [cfg.documents_dir.parent.name]
        assert result.outside_corpus == [cfg.documents_dir.parent.name]
        assert cfg.linked_roots == {}

    def test_a_root_that_vanished_overlaps_nothing(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.core.config import cfg

        child = tmp_path / "data" / "corpus"
        child.mkdir(parents=True)
        register_sources([child])
        child.rmdir()
        result = register_sources([tmp_path / "data"])
        assert (result.registered, result.overlapping) == (["data"], [])
        assert cfg.linked_roots["data"] == str((tmp_path / "data").resolve())

    def test_a_path_given_twice_is_outside_the_corpus_once(self, isolated_env, tmp_path):
        from lilbee.app.ingest import register_sources

        logo = tmp_path / "logo.svg"
        logo.write_text("<svg/>", encoding="utf-8")
        result = register_sources([logo, logo])
        assert len(result.refused) == 2
        assert result.outside_corpus == ["logo.svg"]

    def test_names_outside_corpus_asks_registration_and_registers_nothing(
        self, isolated_env, tmp_path
    ):
        """Before registration the answer is what registering would add or leave out."""
        from lilbee.app.ingest import names_outside_corpus, register_sources
        from lilbee.core import settings
        from lilbee.core.config import cfg

        owned = cfg.documents_dir / "owned.txt"
        owned.parent.mkdir(parents=True, exist_ok=True)
        owned.write_text("owned", encoding="utf-8")
        holder = tmp_path / "holder" / "held.txt"
        taken = tmp_path / "other" / "held.txt"
        fresh = tmp_path / "other" / "fresh.txt"
        logo = tmp_path / "other" / "logo.svg"
        for path in (holder, taken, fresh, logo):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("text", encoding="utf-8")
        register_sources([holder])
        before = settings.load(cfg.data_root)["linked_roots"]

        names = names_outside_corpus([owned, holder, fresh, taken, logo, fresh, logo])

        assert names == ["held.txt", "logo.svg", "fresh.txt"]
        assert settings.load(cfg.data_root)["linked_roots"] == before
        assert cfg.linked_roots == before

    def test_names_outside_corpus_reads_no_registry_for_no_paths(self, isolated_env):
        """A stopped sync asks with no paths; an unreadable config.toml must not fail it."""
        from lilbee.app.ingest import names_outside_corpus

        with mock.patch("lilbee.app.ingest.settings.load", side_effect=OSError("unreadable")):
            assert names_outside_corpus([]) == []
            with pytest.raises(OSError, match="unreadable"):
                names_outside_corpus([Path("anything.txt")])

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
        assert result.name_taken == ["reports"]
        assert cfg.linked_roots == {}


def _indexed(services) -> set[str]:
    """Source keys currently in the index."""
    return {s["filename"] for s in services.store.get_sources()}


# Removing a source then explicitly adding it back must re-index it. A removal
# leaves a skip marker so discovery stops resurrecting the source; an ``add``
# naming the path is the user asking for it back and outranks that marker. Every
# surface that takes an explicit path gets the same round trip below.
class TestRegisterSourcesClearsMarkers:
    """The shared primitive every add surface funnels through."""

    async def test_file_under_a_directory_root_comes_back(self, isolated_env, mock_svc, tmp_path):
        from lilbee.app.ingest import register_sources, remove_documents_durably
        from lilbee.data.ingest import sync

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "gone.txt").write_text("removed", encoding="utf-8")
        (corpus / "stay.txt").write_text("kept", encoding="utf-8")
        register_sources([corpus])
        await sync(quiet=True)
        remove_documents_durably(["corpus/gone.txt"])

        register_sources([corpus])
        await sync(quiet=True)

        assert "corpus/gone.txt" in _indexed(mock_svc)

    async def test_owned_documents_file_comes_back(self, isolated_env, mock_svc):
        from lilbee.app.ingest import register_sources, remove_documents_durably
        from lilbee.data.ingest import sync

        doc = isolated_env / "notes.txt"
        doc.write_text("notes", encoding="utf-8")
        await sync(quiet=True)
        remove_documents_durably(["notes.txt"])

        register_sources([doc])
        await sync(quiet=True)

        assert "notes.txt" in _indexed(mock_svc)

    async def test_single_file_root_comes_back(self, isolated_env, mock_svc, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.discovery import file_hash
        from lilbee.data.ingest.skip_marker import write_skip_markers

        doc = tmp_path / "cv-manual.txt"
        doc.write_text("a manual", encoding="utf-8")
        register_sources([doc])
        await sync(quiet=True)
        # Mark it directly: a single-file root removed by its label un-registers
        # instead of marking, so this is the shape reached when the root is still
        # registered -- an extraction that yielded nothing, or a re-registered root.
        mock_svc.store.remove_documents(["cv-manual.txt"])
        write_skip_markers(cfg.data_root, {"cv-manual.txt": file_hash(doc)})

        register_sources([doc])
        await sync(quiet=True)

        assert "cv-manual.txt" in _indexed(mock_svc)

    async def test_markers_outside_the_named_paths_are_left_alone(
        self, isolated_env, mock_svc, tmp_path
    ):
        """An add clears the markers it covers, not the marker set."""
        from lilbee.app.ingest import register_sources, remove_documents_durably
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.skip_marker import load_skip_markers

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "gone.txt").write_text("removed", encoding="utf-8")
        other = isolated_env / "elsewhere.txt"
        other.write_text("also removed", encoding="utf-8")
        register_sources([corpus])
        await sync(quiet=True)
        remove_documents_durably(["corpus/gone.txt", "elsewhere.txt"])

        register_sources([corpus])

        markers = load_skip_markers(cfg.data_root)
        assert "corpus/gone.txt" not in markers
        assert "elsewhere.txt" in markers

    def test_re_adding_a_tracked_source_is_not_reported_as_a_collision(
        self, isolated_env, mock_svc, tmp_path
    ):
        """Re-adding the same path needs no --force, so it must not warn about one."""
        from lilbee.app.ingest import register_sources

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        register_sources([corpus])

        result = register_sources([corpus])

        assert result.tracked == ["corpus"]
        assert result.name_taken == []
        assert result.registered == []


class TestCliSurface:
    def test_add_paths_reindexes_a_removed_source(self, isolated_env, mock_svc, tmp_path):
        import asyncio

        from rich.console import Console

        from lilbee.app.ingest import register_sources, remove_documents_durably
        from lilbee.cli.helpers import add_paths
        from lilbee.data.ingest import sync

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "gone.txt").write_text("removed", encoding="utf-8")
        register_sources([corpus])
        asyncio.run(sync(quiet=True))
        remove_documents_durably(["corpus/gone.txt"])

        add_paths([corpus], Console(), run_sync=lambda _registration: asyncio.run(sync(quiet=True)))

        assert "corpus/gone.txt" in _indexed(mock_svc)

    def test_add_paths_skips_the_sync_when_nothing_reached_the_corpus(
        self, isolated_env, mock_svc, tmp_path
    ):
        """A refused path registers nothing, so no whole-vault sync runs and the line says so."""
        from rich.console import Console

        from lilbee.cli.helpers import add_paths

        drawing = tmp_path / "logo.svg"
        drawing.write_text("<svg/>", encoding="utf-8")
        runs: list[int] = []
        console = Console(record=True, width=120)

        add_paths([drawing], console, run_sync=lambda _registration: runs.append(1))

        assert runs == []
        assert "Registered 0 source(s)" in console.export_text()

    def test_registration_line_names_what_was_already_tracked(self):
        from lilbee.app.ingest import RegisterResult
        from lilbee.cli.helpers import describe_registration

        line = describe_registration(RegisterResult(registered=[], tracked=["cv-manual.pdf"]))
        assert "already tracked: cv-manual.pdf" in line
        assert line != "Registered 0 source(s)"

    def test_registration_line_names_what_overlaps_a_registered_source(self):
        from lilbee.app.ingest import RegisterResult
        from lilbee.cli.helpers import describe_registration

        line = describe_registration(RegisterResult(overlapping=["papers"]))
        assert "overlaps a registered source: papers" in line

    def test_registration_line_says_a_parent_of_a_source_was_not_added(self):
        from lilbee.app.ingest import RegisterResult
        from lilbee.cli.helpers import describe_registration

        result = RegisterResult(overlapping=["papers", "data"], containing=["data"])
        line = describe_registration(result)
        assert line == (
            "overlaps a registered source: papers, "
            "contains a source lilbee already indexes, not added: data"
        )

    def test_add_paths_skips_the_sync_when_the_name_is_taken(
        self, isolated_env, mock_svc, tmp_path
    ):
        """A label held by another source registers nothing, so no sync runs."""
        from rich.console import Console

        from lilbee.app.ingest import register_sources
        from lilbee.cli.helpers import add_paths

        one = tmp_path / "a" / "corpus"
        one.mkdir(parents=True)
        register_sources([one])
        two = tmp_path / "b" / "corpus"
        two.mkdir(parents=True)
        runs: list[int] = []
        console = Console(record=True, width=120)

        add_paths([two], console, run_sync=lambda _registration: runs.append(1))

        assert runs == []
        assert "is taken by another source" in console.export_text()


class TestPythonApiSurface:
    def test_lilbee_add_reindexes_a_removed_source(self, isolated_env, mock_svc, tmp_path):
        from lilbee.api import Lilbee

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "gone.txt").write_text("removed", encoding="utf-8")

        bee = Lilbee(config=cfg)
        bee._services = mock_svc
        bee.add([corpus])
        assert "corpus/gone.txt" in _indexed(mock_svc)

        bee.remove("corpus/gone.txt")
        assert "corpus/gone.txt" not in _indexed(mock_svc)

        bee.add([corpus])
        assert "corpus/gone.txt" in _indexed(mock_svc)

    def test_lilbee_add_skips_the_sync_when_nothing_reached_the_corpus(
        self, isolated_env, mock_svc, tmp_path
    ):
        from lilbee.api import Lilbee

        drawing = tmp_path / "logo.svg"
        drawing.write_text("<svg/>", encoding="utf-8")
        bee = Lilbee(config=cfg)
        bee._services = mock_svc

        with mock.patch("lilbee.data.ingest.sync", new_callable=mock.AsyncMock) as run_sync:
            result = bee.add([drawing])

        run_sync.assert_not_called()
        assert result.added == [] and result.failed == []


class TestMcpSurface:
    async def test_add_tool_reindexes_a_removed_source(self, isolated_env, mock_svc, tmp_path):
        from lilbee.app.ingest import register_sources, remove_documents_durably
        from lilbee.data.ingest import sync
        from lilbee.mcp_server import add as mcp_add

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "gone.txt").write_text("removed", encoding="utf-8")
        register_sources([corpus])
        await sync(quiet=True)
        remove_documents_durably(["corpus/gone.txt"])

        result = await mcp_add([str(corpus)])

        assert "corpus/gone.txt" in _indexed(mock_svc)
        assert result["tracked"] == ["corpus"]

    async def test_add_tool_skips_the_sync_when_nothing_reached_the_corpus(
        self, isolated_env, mock_svc, tmp_path
    ):
        from lilbee.mcp_server import add as mcp_add

        drawing = tmp_path / "logo.svg"
        drawing.write_text("<svg/>", encoding="utf-8")

        with mock.patch("lilbee.data.ingest.sync", new_callable=mock.AsyncMock) as run_sync:
            result = await mcp_add([str(drawing)])

        run_sync.assert_not_called()
        assert "indexed nothing" in result["error"]

    async def test_add_tool_reports_a_taken_name_without_a_sync(
        self, isolated_env, mock_svc, tmp_path
    ):
        from lilbee.app.ingest import register_sources
        from lilbee.mcp_server import add as mcp_add

        one = tmp_path / "a" / "corpus"
        one.mkdir(parents=True)
        register_sources([one])
        two = tmp_path / "b" / "corpus"
        two.mkdir(parents=True)

        with mock.patch("lilbee.data.ingest.sync", new_callable=mock.AsyncMock) as run_sync:
            result = await mcp_add([str(two)])

        run_sync.assert_not_called()
        assert result["name_taken"] == ["corpus"]
        assert result["overlapping"] == []
        assert result["sync"] is None


class TestHttpSurface:
    async def test_add_handler_reindexes_a_removed_source(self, isolated_env, mock_svc, tmp_path):
        from lilbee.app.ingest import register_sources, remove_documents_durably
        from lilbee.data.ingest import sync
        from lilbee.server.handlers import SseStream
        from lilbee.server.handlers.ingest import _run_add

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "gone.txt").write_text("removed", encoding="utf-8")
        register_sources([corpus])
        await sync(quiet=True)
        remove_documents_durably(["corpus/gone.txt"])

        summary = await _run_add(
            paths=[str(corpus)], force=False, ocr=None, ocr_timeout=None, sse=SseStream()
        )

        assert "corpus/gone.txt" in _indexed(mock_svc)
        assert summary.tracked == ["corpus"]

    async def test_add_handler_skips_the_sync_when_nothing_reached_the_corpus(
        self, isolated_env, mock_svc, tmp_path
    ):
        from lilbee.server.handlers import SseStream
        from lilbee.server.handlers.ingest import _run_add

        drawing = tmp_path / "logo.svg"
        drawing.write_text("<svg/>", encoding="utf-8")

        with mock.patch("lilbee.data.ingest.sync", new_callable=mock.AsyncMock) as run_sync:
            summary = await _run_add(
                paths=[str(drawing)],
                force=False,
                ocr=None,
                ocr_timeout=None,
                sse=SseStream(),
            )

        run_sync.assert_not_called()
        assert summary.sync is None
        assert summary.errors == ["logo.svg: vector graphic, not a document"]

    async def test_add_handler_reports_a_taken_name_without_a_sync(
        self, isolated_env, mock_svc, tmp_path
    ):
        from lilbee.app.ingest import register_sources
        from lilbee.server.handlers import SseStream
        from lilbee.server.handlers.ingest import _run_add

        one = tmp_path / "a" / "corpus"
        one.mkdir(parents=True)
        register_sources([one])
        two = tmp_path / "b" / "corpus"
        two.mkdir(parents=True)

        with mock.patch("lilbee.data.ingest.sync", new_callable=mock.AsyncMock) as run_sync:
            summary = await _run_add(
                paths=[str(two)], force=False, ocr=None, ocr_timeout=None, sse=SseStream()
            )

        run_sync.assert_not_called()
        assert summary.name_taken == ["corpus"]
        assert summary.overlapping == []
        assert summary.sync is None


class TestIngestPoolLifetime:
    def test_the_ingest_threads_a_sync_used_end_with_it(self, isolated_env, mock_svc):
        """A sync on a private loop leaves none of its ingest worker threads behind."""
        from lilbee.data import offload
        from lilbee.data.ingest import pipeline, sync

        (isolated_env / "note.txt").write_text("some words to index", encoding="utf-8")
        flush_threads: list[threading.Thread] = []
        # Holding the run's pool keeps it from being collected, so only an explicit
        # shutdown can end its idle workers.
        run_pools: list[object] = []
        real_flush = pipeline._flush_writes

        def _recording_flush(*args, **kwargs):
            flush_threads.append(threading.current_thread())
            run_pools.append(offload._run_pool.get())
            return real_flush(*args, **kwargs)

        with mock.patch.object(pipeline, "_flush_writes", _recording_flush):
            result = asyncio.run(sync(quiet=True))
        assert result.added == ["note.txt"]
        assert flush_threads
        assert all(t.name.startswith("lilbee-ingest") for t in flush_threads)
        assert None not in run_pools
        for thread in flush_threads:
            thread.join(timeout=2.0)
        assert [t.name for t in flush_threads if t.is_alive()] == []


class TestTuiSurface:
    def test_do_sync_cancelled_mid_extraction_holds_no_file_out(self, isolated_env, mock_svc):
        """A TUI sync cancelled while a file extracts ends cancelled, with no skip marker."""
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
        from lilbee.data.ingest.skip_marker import load_skip_markers
        from lilbee.runtime.cancellation import TaskCancelledError

        (isolated_env / "scan.pdf").write_bytes(b"%PDF-1.4 scanned")
        cancelled = threading.Event()
        reporter = MagicMock(spec=ProgressReporter)
        reporter.is_set.side_effect = cancelled.is_set

        async def _extract(*_args, config, **_kwargs):
            if config.disable_ocr:
                raise RuntimeError("probe")  # the page-count probe; its failure is ignored
            cancelled.set()
            raise RuntimeError("process_image failed: TaskCancelledError")

        screen = ChatScreen.__new__(ChatScreen)
        with (
            mock.patch("lilbee.runtime.asyncio_loop.run", new=asyncio.run),
            mock.patch("lilbee.data.extract.xberg.aextract_document", side_effect=_extract),
            pytest.raises(TaskCancelledError, match=msg.SYNC_CANCELLED_RESUME),
        ):
            screen._do_sync(reporter)
        assert cancelled.is_set()
        assert load_skip_markers(cfg.data_root) == {}

    async def test_do_add_reindexes_a_removed_source(self, isolated_env, mock_svc, tmp_path):
        """The TUI's /add worker body, with the real registration primitive."""
        import asyncio
        import threading

        from lilbee.app.ingest import register_sources, remove_documents_durably
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
        from lilbee.data.ingest import sync
        from tests._lilbee_app_test_host import await_chat, ready_services

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "gone.txt").write_text("removed", encoding="utf-8")
        register_sources([corpus])
        await sync(quiet=True)
        remove_documents_durably(["corpus/gone.txt"])

        with ready_services():
            # ready_services binds its own container to release the startup gate;
            # swap the stateful one back so the add actually ingests.
            import lilbee.app.services as svc_mod

            mock_svc.provider.role_ready.return_value = True
            svc_mod.set_services(mock_svc)
            app = LilbeeApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = await await_chat(app, pilot)
                assert screen is not None
                errors: list[BaseException] = []
                reporter = MagicMock(spec=ProgressReporter)
                reporter.is_set.return_value = False

                def _worker() -> None:
                    try:
                        # _do_add drives sync through the TUI's shared loop; run it
                        # on a private one so the test's loop is not re-entered.
                        with mock.patch(
                            "lilbee.runtime.asyncio_loop.run",
                            new=lambda coro: asyncio.run(coro),
                        ):
                            screen._do_add([corpus], reporter)
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)

                thread = threading.Thread(target=_worker, daemon=True)
                thread.start()
                while thread.is_alive():
                    await pilot.pause()
                thread.join(timeout=5)
                assert not errors, errors

        assert "corpus/gone.txt" in _indexed(mock_svc)

    async def test_do_add_skips_the_sync_when_nothing_reached_the_corpus(
        self, isolated_env, mock_svc, tmp_path
    ):
        import threading

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
        from tests._lilbee_app_test_host import await_chat, ready_services

        drawing = tmp_path / "logo.svg"
        drawing.write_text("<svg/>", encoding="utf-8")

        with ready_services():
            import lilbee.app.services as svc_mod

            mock_svc.provider.role_ready.return_value = True
            svc_mod.set_services(mock_svc)
            app = LilbeeApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = await await_chat(app, pilot)
                assert screen is not None
                errors: list[BaseException] = []

                def _worker() -> None:
                    try:
                        with mock.patch(
                            "lilbee.data.ingest.sync", new_callable=mock.AsyncMock
                        ) as run_sync:
                            screen._do_add([drawing], MagicMock(spec=ProgressReporter))
                        run_sync.assert_not_called()
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)

                thread = threading.Thread(target=_worker, daemon=True)
                thread.start()
                while thread.is_alive():
                    await pilot.pause()
                thread.join(timeout=5)
                assert not errors, errors

    def test_do_add_names_a_taken_label_and_skips_the_sync(self, isolated_env, tmp_path):
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter

        screen = ChatScreen.__new__(ChatScreen)
        notify = MagicMock()
        with (
            mock.patch("lilbee.cli.tui.screens.chat.call_from_thread", notify),
            mock.patch(
                "lilbee.app.ingest.register_sources",
                return_value=RegisterResult(name_taken=["corpus"]),
            ),
            mock.patch("lilbee.runtime.asyncio_loop.run") as run,
        ):
            screen._do_add([tmp_path / "corpus"], MagicMock(spec=ProgressReporter))

        run.assert_not_called()
        toasts = [call.args[2] for call in notify.call_args_list]
        assert msg.CMD_ADD_NAME_TAKEN.format(name="corpus") in toasts
        assert msg.CMD_ADD_NOTHING in toasts

    def test_do_add_names_an_overlapping_source_and_syncs(self, isolated_env, tmp_path):
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
        from lilbee.data.ingest import SyncResult

        screen = ChatScreen.__new__(ChatScreen)
        notify = MagicMock()
        with (
            mock.patch("lilbee.cli.tui.screens.chat.call_from_thread", notify),
            mock.patch(
                "lilbee.app.ingest.register_sources",
                return_value=RegisterResult(overlapping=["papers"]),
            ),
            mock.patch("lilbee.runtime.asyncio_loop.run", return_value=SyncResult()) as run,
        ):
            screen._do_add([tmp_path / "papers"], MagicMock(spec=ProgressReporter))

        run.assert_called_once()
        toasts = [call.args[2] for call in notify.call_args_list]
        assert msg.CMD_ADD_OVERLAPPING.format(names="papers") in toasts

    def test_do_add_says_a_parent_of_the_documents_directory_was_not_added(
        self, isolated_env, tmp_path
    ):
        from lilbee.app.ingest import register_sources
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
        from lilbee.data.ingest import SyncResult

        papers = tmp_path / "ext" / "lib" / "papers"
        papers.mkdir(parents=True)
        register_sources([papers.parent])
        around = isolated_env.parent
        screen = ChatScreen.__new__(ChatScreen)
        notify = MagicMock()
        with (
            mock.patch("lilbee.cli.tui.screens.chat.call_from_thread", notify),
            mock.patch("lilbee.runtime.asyncio_loop.run", return_value=SyncResult()),
        ):
            screen._do_add([papers, around], MagicMock(spec=ProgressReporter))

        assert set(cfg.linked_roots) == {"lib"}
        toasts = [call.args[2] for call in notify.call_args_list]
        assert msg.CMD_ADD_CONTAINING.format(names=around.name) in toasts
        assert msg.CMD_ADD_OVERLAPPING.format(names="papers") in toasts
        assert msg.CMD_ADD_OVERLAPPING.format(names=f"papers, {around.name}") not in toasts

    def test_do_add_names_a_rollback_error_with_the_cancel(self, isolated_env, tmp_path):
        """A cancelled TUI add whose rollback cannot read the registry still ends as a cancel."""
        import asyncio

        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
        from lilbee.runtime.cancellation import TaskCancelledError

        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock(spec=ProgressReporter)
        reporter.is_set.return_value = True
        reporter.cancelled_by_user.return_value = True
        stopped = f"{msg.SYNC_CANCELLED_RESUME} It also hit an error: unreadable."
        with (
            mock.patch("lilbee.cli.tui.screens.chat.call_from_thread"),
            mock.patch(
                "lilbee.app.ingest.register_sources",
                return_value=RegisterResult(tracked=["owned.txt"]),
            ),
            mock.patch("lilbee.runtime.asyncio_loop.run", side_effect=asyncio.CancelledError),
            mock.patch("lilbee.app.ingest.settings.load", side_effect=OSError("unreadable")),
            pytest.raises(TaskCancelledError) as raised,
        ):
            screen._do_add([tmp_path / "owned.txt"], reporter)
        assert str(raised.value) == stopped


def test_every_add_surface_names_each_registration_outcome():
    """The surfaces that render a registration name the taken-label and overlap outcomes."""
    import lilbee

    package = Path(lilbee.__file__).parent
    surfaces = [
        "mcp_server.py",
        "server/handlers/ingest.py",
        "cli/helpers.py",
        "cli/commands/ingest_sync.py",
        "cli/tui/screens/chat.py",
    ]
    for name in surfaces:
        text = (package / name).read_text(encoding="utf-8")
        assert "name_taken" in text and "overlapping" in text, name


def test_every_add_surface_asks_whether_anything_reached_the_corpus():
    """One predicate decides whether a sync follows registration on every surface."""
    import lilbee

    package = Path(lilbee.__file__).parent
    surfaces = [
        "api.py",
        "mcp_server.py",
        "server/handlers/ingest.py",
        "cli/helpers.py",
        "cli/commands/ingest_sync.py",
        "cli/tui/screens/chat.py",
    ]
    for name in surfaces:
        assert "reached_corpus" in (package / name).read_text(encoding="utf-8"), name


def test_every_add_surface_funnels_through_register_sources():
    """The marker rule lives in register_sources; a surface bypassing it would drift.

    Cheap guard against a sixth surface being added that registers roots its own
    way and silently reinstates the veto this file exists to prevent.
    """
    import lilbee

    package = Path(lilbee.__file__).parent
    surfaces = [
        "api.py",
        "mcp_server.py",
        "server/handlers/ingest.py",
        "cli/helpers.py",
        "cli/commands/ingest_sync.py",
        "cli/tui/screens/chat.py",
    ]
    for name in surfaces:
        assert "register_sources" in (package / name).read_text(encoding="utf-8"), name


class TestIngestArchive:
    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_members_are_named_under_the_archive_and_nested_archives_recurse(
        self, mock_kf, isolated_env, mock_svc
    ):
        from lilbee.data.extract.document import ingest_archive

        inner = _make_archive_result(
            [_member("notes.txt", "text/plain", _make_xberg_result(num_chunks=1))]
        )
        mock_kf.return_value = _make_archive_result(
            [
                _member("inner.zip", "application/zip", inner),
                _member(
                    "report.pdf",
                    "application/pdf",
                    _make_xberg_result(num_chunks=2, has_pages=True),
                ),
            ]
        )
        f = isolated_env / "docs.zip"
        f.write_bytes(b"PK\x03\x04")

        members = await ingest_archive(f, "docs.zip", "zip")

        assert [m.name for m in members] == ["docs.zip/inner.zip/notes.txt", "docs.zip/report.pdf"]
        assert [m.content_type for m in members] == ["txt", "pdf"]
        assert all(r["source"] == m.name for m in members for r in m.records)
        assert [len(m.records) for m in members] == [1, 2]
        assert {p["source"] for p in members[1].page_texts} == {"docs.zip/report.pdf"}
        mock_kf.assert_awaited_once()

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_member_over_the_cap_names_the_member(
        self, mock_kf, isolated_env, mock_svc, monkeypatch
    ):
        from lilbee.data.extract.chunk import ChunkLimitError
        from lilbee.data.extract.document import ingest_archive

        monkeypatch.setattr(cfg, "max_chunks_per_file", 1)
        mock_kf.return_value = _make_archive_result(
            [_member("big.txt", "text/plain", _make_xberg_result(num_chunks=3))]
        )
        f = isolated_env / "docs.zip"
        f.write_bytes(b"PK\x03\x04")

        with pytest.raises(ChunkLimitError, match=r"^docs.zip/big.txt: 3 chunks exceed"):
            await ingest_archive(f, "docs.zip", "zip")

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_an_archive_extraction_hands_xberg_no_progress_callback(
        self, mock_kf, isolated_env, mock_svc
    ):
        """xberg counts an archive's OCR'd pages across members, so no member reports them."""
        from lilbee.data.extract.document import ingest_archive

        cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
        mock_kf.return_value = _make_archive_result(
            [_member("a.pdf", "application/pdf", _make_xberg_result(num_chunks=1, has_pages=True))]
        )
        f = isolated_env / "docs.zip"
        f.write_bytes(b"PK\x03\x04")

        await ingest_archive(f, "docs.zip", "zip")

        mock_kf.assert_awaited_once()
        assert mock_kf.await_args.kwargs["on_progress"] is None

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_members_pass_the_same_extension_gate_as_files_on_disk(
        self, mock_kf, isolated_env, mock_svc, caplog
    ):
        """A member discovery would not pick up from disk is not chunked from an archive."""
        import logging

        from lilbee.data.extract.document import ingest_archive

        mock_kf.return_value = _make_archive_result(
            [
                _member("run.rerank.trec", "text/plain", _make_xberg_result(num_chunks=5)),
                _member("logo.svg", "image/svg+xml", _make_xberg_result(num_chunks=1)),
                _member("README", "text/plain", _make_xberg_result(num_chunks=1)),
                _member("tool.py", "text/x-python", _make_xberg_result(num_chunks=1)),
                _member("notes.txt", "text/plain", _make_xberg_result(num_chunks=1)),
            ]
        )
        f = isolated_env / "runs.gz"
        f.write_bytes(b"\x1f\x8b")

        with caplog.at_level(logging.INFO, logger="lilbee.data.extract.document"):
            members = await ingest_archive(f, "runs.gz", "gz")

        assert [(m.name, m.content_type) for m in members] == [
            ("runs.gz/tool.py", "code"),
            ("runs.gz/notes.txt", "txt"),
        ]
        skipped = [r.getMessage() for r in caplog.records if "runs.gz" in r.getMessage()]
        assert skipped == [
            "Skipped 3 member(s) of runs.gz, unsupported format: README, logo.svg, run.rerank.trec"
        ]

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_scanned_member_with_no_text_names_the_backend_that_ran(
        self, mock_kf, isolated_env, mock_svc, caplog
    ):
        """A scanned PDF member with no text gets the same backend-aware warning as a
        top-level scan: the archive's OCR choice threads into each member's warning,
        so OCR-off names the ocr setting instead of the Tesseract advice."""
        import logging

        from lilbee.data.extract.document import ingest_archive

        cfg.ocr = OcrMode.OFF
        cfg.vision_model = ""
        mock_kf.return_value = _make_archive_result(
            [_member("scan.pdf", "application/pdf", _make_xberg_result(num_chunks=0))]
        )
        f = isolated_env / "docs.zip"
        f.write_bytes(b"PK\x03\x04")

        with caplog.at_level(logging.WARNING, logger="lilbee.data.extract.document"):
            members = await ingest_archive(f, "docs.zip", "zip")

        assert [m.records for m in members] == [[]]
        assert "OCR is off (ocr = off)" in caplog.text
        assert "configure a vision model" not in caplog.text

    @mock.patch("lilbee.data.extract.xberg.aextract_document", new_callable=mock.AsyncMock)
    async def test_nested_archive_scanned_member_names_the_backend_that_ran(
        self, mock_kf, isolated_env, mock_svc, caplog
    ):
        """The archive's OCR choice threads through the nested-archive recursion too:
        a scanned PDF inside an archive inside an archive gets the same backend-aware
        warning as a member one level deep, so OCR-off still names the ocr setting
        instead of the Tesseract advice."""
        import logging

        from lilbee.data.extract.document import ingest_archive

        cfg.ocr = OcrMode.OFF
        cfg.vision_model = ""
        inner = _make_archive_result(
            [_member("scan.pdf", "application/pdf", _make_xberg_result(num_chunks=0))]
        )
        mock_kf.return_value = _make_archive_result(
            [_member("inner.zip", "application/zip", inner)]
        )
        f = isolated_env / "docs.zip"
        f.write_bytes(b"PK\x03\x04")

        with caplog.at_level(logging.WARNING, logger="lilbee.data.extract.document"):
            members = await ingest_archive(f, "docs.zip", "zip")

        assert [m.records for m in members] == [[]]
        assert "OCR is off (ocr = off)" in caplog.text
        assert "configure a vision model" not in caplog.text


class TestFlushArchiveMembers:
    def test_members_write_as_their_own_sources_and_stale_members_go(self, mock_svc):
        from lilbee.data.ingest import pipeline
        from lilbee.data.store import SourceMeta
        from lilbee.data.types import MemberRecords, _IngestResult

        store = mock_svc.store
        store.member_sources.return_value = ["docs.zip/old.pdf", "docs.zip/report.pdf"]
        member = MemberRecords(
            name="docs.zip/report.pdf",
            content_type="pdf",
            records=[{"source": "docs.zip/report.pdf", "text": "x"}],
            page_texts=[{"source": "docs.zip/report.pdf", "page": 1}],
            meta=SourceMeta(title="Report"),
        )
        result = _IngestResult(
            name="docs.zip",
            path=Path("docs.zip"),
            chunk_count=1,
            error=None,
            file_hash="h",
            records=[],
            needs_cleanup=True,
            members=[member],
            meta=SourceMeta(title="docs"),
        )

        pipeline._flush_batch([result])

        store.remove_documents.assert_called_once_with(["docs.zip/old.pdf"])
        items = store.write_chunks_batch.call_args[0][0]
        assert [it.source for it in items] == ["docs.zip", "docs.zip/report.pdf"]
        assert [it.file_hash for it in items] == ["h", "h"]
        assert items[1].needs_cleanup is True
        assert items[1].page_texts == member.page_texts
        assert items[1].meta == member.meta
        assert items[0].records == []


class TestArchiveResult:
    async def test_members_feed_concepts_entities_and_the_chunk_total(self, mock_svc, monkeypatch):
        from lilbee.data.ingest import pipeline
        from lilbee.data.store import ConceptRecords, SourceMeta
        from lilbee.data.types import FileToProcess, MemberRecords

        members = [
            MemberRecords(
                "docs.zip/a.txt", "txt", [{"source": "docs.zip/a.txt"}], [], SourceMeta()
            ),
            MemberRecords(
                "docs.zip/b.txt", "txt", [{"source": "docs.zip/b.txt"}] * 2, [], SourceMeta()
            ),
        ]
        monkeypatch.setattr(pipeline, "ingest_archive", mock.AsyncMock(return_value=members))
        concepts = ConceptRecords(nodes=[{"id": "n"}], edges=[], chunk_concepts=[])
        monkeypatch.setattr(
            pipeline, "build_concept_records", mock.AsyncMock(side_effect=[concepts, None])
        )
        monkeypatch.setattr(
            pipeline, "build_entity_records", mock.AsyncMock(side_effect=[[{"e": 1}], None])
        )
        entry = FileToProcess("docs.zip", Path("docs.zip"), "zip", "h", True)
        events: list = []
        pages_done = [0]

        result = await pipeline._archive_result(
            entry, lambda t, d: events.append((t, d)), pages_done, None
        )

        assert result.chunk_count == 3
        assert result.records == []
        assert result.members == members
        assert result.concept_records == concepts
        assert result.entity_rows == [{"e": 1}]
        assert result.meta.title == "docs"
        assert pages_done == [1]
        assert events[-1][1].chunks == 3


def _scan_result(ocr_backends: list[str | None]) -> MagicMock:
    """An xberg result with no text, one page per entry; a backend name marks an OCR'd page."""
    result = _make_xberg_result(num_chunks=0)
    result.pages = [
        mock.MagicMock(
            page_number=i + 1,
            content="",
            ocr_confidence=None if name is None else mock.MagicMock(backend=name),
        )
        for i, name in enumerate(ocr_backends)
    ]
    return result


async def _sync_scan(isolated_env, pages: list[str | None]):
    from lilbee.app.services import get_services
    from lilbee.data.ingest import sync

    get_services().provider.vision_slot_capacity.return_value = 1
    (isolated_env / "scan.pdf").write_bytes(b"%PDF-1.4 scanned")
    with mock.patch(
        "lilbee.data.extract.xberg.aextract_document",
        new_callable=mock.AsyncMock,
        return_value=_scan_result(pages),
    ) as extract:
        result = await sync(quiet=True)
    return result, extract.call_args.kwargs["config"]


class TestIngestSizingAsksTheOcrChooser:
    """Admission is sized to vision slots only when extraction would run vision OCR."""

    _VISION_MODEL = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"

    @pytest.mark.parametrize("vision_model", ["", _VISION_MODEL])
    async def test_ocr_off_requests_no_vision_slots(self, isolated_env, mock_svc, vision_model):
        from lilbee.app.services import get_services

        cfg.ocr = OcrMode.OFF
        cfg.vision_model = vision_model
        result, config = await _sync_scan(isolated_env, [None])
        assert result.skipped == ["scan.pdf"]  # the scan went through the ingest run
        assert config.ocr is None
        get_services().provider.vision_slot_capacity.assert_not_called()

    @pytest.mark.parametrize("ocr", [OcrMode.AUTO, OcrMode.ALL])
    async def test_ocr_on_with_a_vision_model_requests_vision_slots(
        self, isolated_env, mock_svc, ocr
    ):
        from lilbee.app.services import get_services

        cfg.ocr = ocr
        cfg.vision_model = self._VISION_MODEL
        await _sync_scan(isolated_env, ["lilbee-vision"])
        get_services().provider.vision_slot_capacity.assert_called_once_with()

    async def test_per_request_ocr_off_with_a_vision_model_requests_no_vision_slots(
        self, isolated_env, mock_svc
    ):
        from lilbee.app.ingest import temporary_ocr_config
        from lilbee.app.services import get_services

        cfg.ocr = OcrMode.AUTO
        cfg.vision_model = self._VISION_MODEL
        with temporary_ocr_config(ocr=OcrMode.OFF):
            _, config = await _sync_scan(isolated_env, [None])
        assert config.ocr is None
        get_services().provider.vision_slot_capacity.assert_not_called()

    async def test_per_request_auto_over_ocr_off_reads_with_the_vision_model(
        self, isolated_env, mock_svc
    ):
        from lilbee.app.ingest import temporary_ocr_config
        from lilbee.app.services import get_services

        cfg.ocr = OcrMode.OFF
        cfg.vision_model = self._VISION_MODEL
        with temporary_ocr_config(ocr=OcrMode.AUTO):
            _, config = await _sync_scan(isolated_env, ["lilbee-vision"])
        assert config.ocr.backend == OcrBackendName.LILBEE_VISION
        get_services().provider.vision_slot_capacity.assert_called_once_with()


class TestSkippedScanReportsTheOcrThatRan:
    """A scan that yields no text is reported with the OCR its extraction ran, not the config."""

    _VISION_MODEL = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"

    @pytest.mark.parametrize("vision_model", ["", _VISION_MODEL])
    async def test_ocr_off_reports_ocr_off_whatever_the_engine(
        self, isolated_env, mock_svc, caplog, vision_model
    ):
        from lilbee.cli.tui.log_routing import tui_log_path
        from lilbee.cli.tui.messages import sync_skipped_message
        from lilbee.data.types import OcrReport
        from lilbee.runtime.progress import OcrBackendUsed

        cfg.ocr = OcrMode.OFF
        cfg.vision_model = vision_model
        with caplog.at_level("WARNING", logger="lilbee.data.extract.document"):
            result, config = await _sync_scan(isolated_env, [None, None, None])

        assert config.ocr is None
        assert config.disable_ocr is True
        assert result.skipped == ["scan.pdf"]
        assert result.skipped_ocr == {"scan.pdf": OcrReport(backend=OcrBackendUsed.NONE)}
        message = sync_skipped_message(result, tui_log_path())
        assert "OCR is off" in message and "Set ocr to auto" in message
        assert "vision OCR returned no text" not in message
        assert "scan.pdf[/yellow]: ocr is off" in str(result)
        # the log line names the ocr setting, not the Tesseract advice
        assert "OCR is off (ocr = off)" in caplog.text
        assert "configure a vision model" not in caplog.text

    async def test_vision_that_returns_no_text_names_the_tui_log(
        self, isolated_env, mock_svc, caplog
    ):
        from lilbee.cli.tui.log_routing import tui_log_path
        from lilbee.cli.tui.messages import sync_skipped_message
        from lilbee.data.types import OcrReport
        from lilbee.runtime.progress import OcrBackendUsed

        cfg.ocr = OcrMode.AUTO
        cfg.vision_model = self._VISION_MODEL
        with caplog.at_level("WARNING", logger="lilbee.data.extract.document"):
            result, _ = await _sync_scan(isolated_env, ["lilbee-vision"] * 3)

        assert result.skipped_ocr == {"scan.pdf": OcrReport(backend=OcrBackendUsed.VISION, pages=3)}
        message = sync_skipped_message(result, tui_log_path())
        assert "vision OCR returned no text" in message
        # The TUI names its own log file, not the server's: this is the
        # process actually running the sync, so it is the file that has
        # the underlying error.
        assert str(tui_log_path()) in message
        assert "tui.log" in message
        assert "server.log" not in message
        assert "OCR is off" not in str(result)
        # the log line does not tell the user to configure a vision model that is already set
        assert "the vision model returned no usable text" in caplog.text
        assert "configure a vision model" not in caplog.text

    async def test_tesseract_that_returns_no_text_advises_a_vision_model(
        self, isolated_env, mock_svc, caplog
    ):
        from lilbee.cli.tui.log_routing import tui_log_path
        from lilbee.cli.tui.messages import sync_skipped_message
        from lilbee.data.types import OcrReport
        from lilbee.runtime.progress import OcrBackendUsed

        cfg.ocr = OcrMode.AUTO
        cfg.vision_model = ""
        with caplog.at_level("WARNING", logger="lilbee.data.extract.document"):
            result, _ = await _sync_scan(isolated_env, ["tesseract", None])

        assert result.skipped_ocr == {
            "scan.pdf": OcrReport(backend=OcrBackendUsed.TESSERACT, pages=1)
        }
        assert "Configure a vision_model" in sync_skipped_message(result, tui_log_path())
        # OCR already ran (Tesseract); the log line advises a vision model, not the ocr setting
        assert "configure a vision model via PUT /api/models/vision" in caplog.text
        assert "OCR is off" not in caplog.text

    async def test_trace_line_names_the_backend_and_ocr_pages(self, isolated_env, mock_svc, caplog):
        cfg.ocr = OcrMode.OFF
        cfg.vision_model = ""
        caplog.set_level("INFO", logger="lilbee.ingest.trace")
        await _sync_scan(isolated_env, [None, None])
        assert "ocr=none ocr_pages=0 vision=no" in caplog.text

    async def test_markdown_never_reaches_ocr_and_has_no_report(self, isolated_env, mock_svc):
        from lilbee.data.ingest import sync

        cfg.ocr = OcrMode.OFF
        (isolated_env / "empty.md").write_text("   ", encoding="utf-8")
        result = await sync(quiet=True)
        assert result.skipped == ["empty.md"]
        assert result.skipped_ocr == {}


class TestSyncLoadsThePersistedRegistry:
    """A sync indexes the roots ``config.toml`` holds, whatever this process loaded earlier."""

    @pytest.fixture(autouse=True)
    def _the_data_root_file_is_the_loaded_one(
        self, isolated_env, monkeypatch, overlay_reads_config_toml
    ):
        """The layout of the CLI, serve and the TUI: cfg was built from the data root's file."""
        from lilbee.core.config import CONFIG_FILE_NAME, model

        monkeypatch.setattr(model, "loaded_config_file", cfg.data_root / CONFIG_FILE_NAME)

    @staticmethod
    def _load_warnings(caplog):
        return [r.getMessage() for r in caplog.records if r.name.endswith("load_warnings")]

    @staticmethod
    def _root(tmp_path, label, filename):
        root = tmp_path / label
        root.mkdir()
        (root / filename).write_text(f"text of {filename}", encoding="utf-8")
        return root

    async def test_a_root_another_process_added_is_indexed(self, isolated_env, tmp_path):
        from lilbee.core import settings
        from lilbee.data.ingest import sync

        notes = self._root(tmp_path, "notes", "plan.txt")
        settings.set_value(cfg.data_root, "linked_roots", {"notes": str(notes)})
        cfg.linked_roots = {}  # what a server started before the add still holds

        result = await sync(quiet=True)

        assert result.added == ["notes/plan.txt"]
        assert cfg.linked_roots == {"notes": str(notes)}

    async def test_a_root_another_process_removed_is_not_walked(self, isolated_env, tmp_path):
        from lilbee.core import settings
        from lilbee.data.ingest import sync

        work = self._root(tmp_path, "work", "old.txt")
        (isolated_env / "owned.txt").write_text("owned text", encoding="utf-8")
        settings.set_value(cfg.data_root, "linked_roots", {})
        cfg.linked_roots = {"work": str(work)}

        result = await sync(quiet=True)

        assert result.added == ["owned.txt"]
        assert cfg.linked_roots == {}

    async def test_a_replaced_registry_wins_and_other_settings_stay(self, isolated_env, tmp_path):
        from lilbee.core import settings
        from lilbee.data.ingest import sync

        work = self._root(tmp_path, "work", "old.txt")
        notes = self._root(tmp_path, "notes", "plan.txt")
        settings.update_values(
            cfg.data_root, {"linked_roots": {"notes": str(notes)}, "chunk_size": 111}
        )
        cfg.linked_roots = {"work": str(work)}
        chunk_size = cfg.chunk_size

        result = await sync(quiet=True)

        assert result.added == ["notes/plan.txt"]
        assert chunk_size != 111
        assert cfg.chunk_size == chunk_size

    @pytest.mark.parametrize(
        ("config_text", "reported"),
        [
            ("linked_roots = [unclosed\n", "Failed to read {path}, ignoring"),
            ('linked_roots = "notes"\n', "{path}: linked_roots = 'notes' "),
            ('linked_roots = ["notes"]\n', "{path}: linked_roots = ['notes'] "),
            ("linked_roots = 3\n", "{path}: linked_roots = 3 "),
            ("[linked_roots]\nnotes = 3\n", "{path}: linked_roots = {{'notes': 3}} "),
            (
                '[linked_roots.notes]\nat = "/x"\n',
                "{path}: linked_roots = {{'notes': {{'at': '/x'}}}} ",
            ),
        ],
        ids=["unreadable", "string", "list", "integer", "non-string-value", "table-value"],
    )
    async def test_a_registry_a_sync_cannot_load_is_reported_once_and_the_loaded_one_stays(
        self, isolated_env, tmp_path, caplog, config_text, reported
    ):
        from lilbee.core.config import CONFIG_FILE_NAME
        from lilbee.data.ingest import sync

        work = self._root(tmp_path, "work", "old.txt")
        path = cfg.data_root / CONFIG_FILE_NAME
        path.write_text(config_text, encoding="utf-8")
        cfg.linked_roots = {"work": str(work)}

        with caplog.at_level("WARNING", logger="lilbee.core.config.load_warnings"):
            first = await sync(quiet=True)
            second = await sync(quiet=True)

        assert first.added == ["work/old.txt"]
        assert second.unchanged == 1
        assert cfg.linked_roots == {"work": str(work)}
        warnings = self._load_warnings(caplog)
        assert len(warnings) == 1
        assert warnings[0].startswith(reported.format(path=path))
        assert warnings[0].endswith(("ignoring", "; linked_roots keeps its value"))

    @pytest.mark.parametrize(
        "config_text",
        [None, "", "chunk_size = 111\n", 'linked_roots = ""\n'],
        ids=["no-file", "zero-bytes", "no-registry-key", "blank-string"],
    )
    async def test_a_file_that_sets_no_registry_keeps_the_loaded_one_in_silence(
        self, isolated_env, tmp_path, caplog, config_text
    ):
        from lilbee.core.config import CONFIG_FILE_NAME
        from lilbee.data.ingest import sync

        work = self._root(tmp_path, "work", "old.txt")
        if config_text is not None:
            (cfg.data_root / CONFIG_FILE_NAME).write_text(config_text, encoding="utf-8")
        cfg.linked_roots = {"work": str(work)}

        with caplog.at_level("WARNING", logger="lilbee.core.config.load_warnings"):
            result = await sync(quiet=True)

        assert result.added == ["work/old.txt"]
        assert cfg.linked_roots == {"work": str(work)}
        assert self._load_warnings(caplog) == []

    async def test_a_registry_that_goes_bad_another_way_is_reported_again(
        self, isolated_env, tmp_path, caplog
    ):
        from lilbee.core.config import CONFIG_FILE_NAME
        from lilbee.data.ingest import sync

        work = self._root(tmp_path, "work", "old.txt")
        path = cfg.data_root / CONFIG_FILE_NAME
        cfg.linked_roots = {"work": str(work)}

        with caplog.at_level("WARNING", logger="lilbee.core.config.load_warnings"):
            path.write_text('linked_roots = "notes"\n', encoding="utf-8")
            await sync(quiet=True)
            path.write_text("linked_roots = 3\n", encoding="utf-8")
            await sync(quiet=True)
            await sync(quiet=True)

        assert self._load_warnings(caplog) == [
            f"{path}: linked_roots = 'notes' is not a table of names and values; "
            "linked_roots keeps its value",
            f"{path}: linked_roots = 3 is not a table of names and values; "
            "linked_roots keeps its value",
        ]
        assert cfg.linked_roots == {"work": str(work)}

    async def test_a_refused_setting_beside_a_good_registry_is_not_the_syncs_to_report(
        self, isolated_env, tmp_path, caplog
    ):
        from lilbee.core.config import CONFIG_FILE_NAME
        from lilbee.data.ingest import sync

        notes = self._root(tmp_path, "notes", "plan.txt")
        (cfg.data_root / CONFIG_FILE_NAME).write_text(
            f'top_k = "many"\n[linked_roots]\nnotes = "{notes.as_posix()}"\n', encoding="utf-8"
        )
        cfg.linked_roots = {}

        with caplog.at_level("WARNING", logger="lilbee.core.config.load_warnings"):
            result = await sync(quiet=True)

        assert result.added == ["notes/plan.txt"]
        assert self._load_warnings(caplog) == []

    async def test_a_file_this_process_did_not_load_is_reported_the_same_way(
        self, isolated_env, tmp_path, caplog, monkeypatch
    ):
        from lilbee.core.config import CONFIG_FILE_NAME, model
        from lilbee.data.ingest import sync

        monkeypatch.setattr(model, "loaded_config_file", None)
        work = self._root(tmp_path, "work", "old.txt")
        path = cfg.data_root / CONFIG_FILE_NAME
        path.write_text('linked_roots = "notes"\n', encoding="utf-8")
        cfg.linked_roots = {"work": str(work)}

        with caplog.at_level("WARNING", logger="lilbee.core.config.load_warnings"):
            result = await sync(quiet=True)

        assert result.added == ["work/old.txt"]
        assert len(self._load_warnings(caplog)) == 1
        assert self._load_warnings(caplog)[0].startswith(f"{path}: linked_roots = 'notes' ")

    async def test_a_registry_the_load_reported_is_not_reported_again_by_a_sync(
        self, isolated_env, tmp_path, caplog, monkeypatch
    ):
        """The twin: the registry that goes bad another way after the load is reported."""
        from lilbee.core.config import CONFIG_FILE_NAME, model
        from lilbee.data.ingest import sync

        work = self._root(tmp_path, "work", "old.txt")
        path = cfg.data_root / CONFIG_FILE_NAME
        path.write_text("linked_roots = 3\n", encoding="utf-8")
        monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
        monkeypatch.setenv("LILBEE_DATA", str(cfg.data_root))
        _loaded, at_load = model._build_cfg()
        monkeypatch.setattr(model, "load_warnings", at_load)
        cfg.linked_roots = {"work": str(work)}

        with caplog.at_level("WARNING", logger="lilbee.core.config.load_warnings"):
            caplog.clear()
            first = await sync(quiet=True)
            after_the_same_file = self._load_warnings(caplog)
            path.write_text('linked_roots = "notes"\n', encoding="utf-8")
            await sync(quiet=True)

        assert at_load == (
            "config.toml: linked_roots = 3 is not a table of names and values; "
            "linked_roots uses its default",
        )
        assert first.added == ["work/old.txt"]
        assert after_the_same_file == []
        assert self._load_warnings(caplog) == [
            f"{path}: linked_roots = 'notes' is not a table of names and values; "
            "linked_roots keeps its value"
        ]

    async def test_an_unregister_during_the_load_waits_and_is_not_overwritten(
        self, isolated_env, tmp_path, monkeypatch
    ):
        from lilbee.app.ingest import register_sources, unregister_roots
        from lilbee.core import settings
        from lilbee.core.config import model
        from lilbee.data.ingest import sync

        register_sources([self._root(tmp_path, "work", "old.txt")])
        read = model._TomlSource.__call__
        unregister = threading.Thread(target=unregister_roots, args=(["work"],))
        done_before_the_load_set_the_registry = []

        def read_and_let_the_unregister_try(source):
            values = read(source)  # the load holds a registry with work in it
            if unregister.ident is None:
                unregister.start()
                unregister.join(timeout=1.0)
                done_before_the_load_set_the_registry.append(not unregister.is_alive())
            return values

        monkeypatch.setattr(model._TomlSource, "__call__", read_and_let_the_unregister_try)

        await sync(quiet=True)
        unregister.join(timeout=30)

        assert done_before_the_load_set_the_registry == [False]
        assert not unregister.is_alive()
        assert settings.load(cfg.data_root).get("linked_roots") == {}
        assert cfg.linked_roots == {}

    def test_an_unregister_waits_for_the_assignment_of_the_loaded_registry(
        self, isolated_env, tmp_path
    ):
        """The seam is the assignment: an un-register that got in there would be overwritten."""
        from lilbee.app.ingest import register_sources, unregister_roots
        from lilbee.core import settings

        register_sources([self._root(tmp_path, "work", "old.txt")])
        unregister = threading.Thread(target=unregister_roots, args=(["work"],))
        done_before_the_assignment = []

        class _ConfigWithASeam:
            data_root = cfg.data_root

            @property
            def linked_roots(self):
                return cfg.linked_roots

            @linked_roots.setter
            def linked_roots(self, value):
                unregister.start()
                unregister.join(timeout=1.0)
                done_before_the_assignment.append(not unregister.is_alive())
                cfg.linked_roots = value

        settings.overlay_persisted_roots(_ConfigWithASeam())
        unregister.join(timeout=30)

        assert done_before_the_assignment == [False]
        assert not unregister.is_alive()
        assert settings.load(cfg.data_root).get("linked_roots") == {}
        assert cfg.linked_roots == {}

    async def test_the_overlay_and_a_sync_report_one_bad_registry_once_between_them(
        self, isolated_env, tmp_path, caplog
    ):
        from lilbee.core import settings
        from lilbee.core.config import CONFIG_FILE_NAME
        from lilbee.data.ingest import sync

        work = self._root(tmp_path, "work", "old.txt")
        path = cfg.data_root / CONFIG_FILE_NAME
        path.write_text('linked_roots = "notes"\n', encoding="utf-8")
        cfg.linked_roots = {"work": str(work)}

        with caplog.at_level("WARNING", logger="lilbee.core.config.load_warnings"):
            settings.overlay_persisted_settings(cfg.data_root)
            after_the_overlay = self._load_warnings(caplog)
            result = await sync(quiet=True)
            after_the_sync = self._load_warnings(caplog)
            path.write_text("linked_roots = 3\n", encoding="utf-8")
            await sync(quiet=True)
            after_the_sync_of_the_changed_file = self._load_warnings(caplog)
            settings.overlay_persisted_settings(cfg.data_root)

        first = (
            f"{path}: linked_roots = 'notes' is not a table of names and values; "
            "linked_roots keeps its value"
        )
        second = (
            f"{path}: linked_roots = 3 is not a table of names and values; "
            "linked_roots keeps its value"
        )
        assert result.added == ["work/old.txt"]
        assert after_the_overlay == [first]
        assert after_the_sync == [first]
        assert after_the_sync_of_the_changed_file == [first, second]
        assert self._load_warnings(caplog) == [first, second]

    async def test_a_bad_registry_that_returns_after_a_repair_is_reported_again(
        self, isolated_env, tmp_path, caplog
    ):
        from lilbee.core.config import CONFIG_FILE_NAME
        from lilbee.data.ingest import sync

        work = self._root(tmp_path, "work", "old.txt")
        path = cfg.data_root / CONFIG_FILE_NAME
        cfg.linked_roots = {"work": str(work)}
        bad, good = 'linked_roots = "notes"\n', f'[linked_roots]\nwork = "{work.as_posix()}"\n'

        def _write(text: str, mtime_ns: int) -> None:
            path.write_text(text, encoding="utf-8")
            os.utime(path, ns=(mtime_ns, mtime_ns))

        with caplog.at_level("WARNING", logger="lilbee.core.config.load_warnings"):
            _write(bad, 1_000_000_000)
            await sync(quiet=True)
            await sync(quiet=True)
            _write(good, 2_000_000_000)
            await sync(quiet=True)
            _write(bad, 3_000_000_000)
            await sync(quiet=True)
            await sync(quiet=True)

        reported = (
            f"{path}: linked_roots = 'notes' is not a table of names and values; "
            "linked_roots keeps its value"
        )
        assert self._load_warnings(caplog) == [reported, reported]

    async def test_a_file_that_cannot_be_read_at_all_is_reported_once(
        self, isolated_env, tmp_path, caplog, monkeypatch
    ):
        from lilbee.core.config import CONFIG_FILE_NAME, model

        path = cfg.data_root / CONFIG_FILE_NAME
        path.write_text("linked_roots = 3\n", encoding="utf-8")
        real_stat = Path.stat

        def _no_stat(target, **kwargs):
            if target == path:
                raise OSError("gone")
            return real_stat(target, **kwargs)

        with caplog.at_level("WARNING", logger="lilbee.core.config.load_warnings"):
            monkeypatch.setattr(Path, "stat", _no_stat)
            assert model.toml_value(path, "linked_roots") is None
            assert model.toml_value(path, "linked_roots") is None

        assert len(self._load_warnings(caplog)) == 1

    async def test_the_switch_that_turns_config_toml_off_turns_the_load_off(
        self, isolated_env, tmp_path, monkeypatch
    ):
        from lilbee.core import settings
        from lilbee.data.ingest import sync

        work = self._root(tmp_path, "work", "old.txt")
        notes = self._root(tmp_path, "notes", "plan.txt")
        settings.set_value(cfg.data_root, "linked_roots", {"notes": str(notes)})
        cfg.linked_roots = {"work": str(work)}

        monkeypatch.setenv("LILBEE_SKIP_TOML_CONFIG", "1")
        switched_off = await sync(quiet=True)
        kept = dict(cfg.linked_roots)
        monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG")
        switched_on = await sync(quiet=True)

        assert switched_off.added == ["work/old.txt"]
        assert kept == {"work": str(work)}
        assert switched_on.added == ["notes/plan.txt"]
        assert cfg.linked_roots == {"notes": str(notes)}

    async def test_an_add_then_a_sync_reads_the_registry_the_add_wrote(
        self, isolated_env, tmp_path
    ):
        from lilbee.app.ingest import register_sources
        from lilbee.data.ingest import sync

        notes = self._root(tmp_path, "notes", "plan.txt")
        register_sources([notes])
        loaded = dict(cfg.linked_roots)

        first = await sync(quiet=True)
        second = await sync(quiet=True)

        assert loaded == {"notes": str(notes.resolve())}
        assert cfg.linked_roots == loaded
        assert first.added == ["notes/plan.txt"]
        assert second.unchanged == 1
        assert second.added == []

    async def test_a_worker_keeps_the_registry_its_parent_loaded(self, isolated_env, tmp_path):
        from lilbee.core import settings
        from lilbee.data.ingest import sync
        from lilbee.data.ingest.fanout import ShardId

        work = self._root(tmp_path, "work", "old.txt")
        notes = self._root(tmp_path, "notes", "plan.txt")
        settings.set_value(cfg.data_root, "linked_roots", {"notes": str(notes)})
        cfg.linked_roots = {"work": str(work)}

        result = await sync(quiet=True, shard=ShardId(index=0, count=1, records_root=cfg.data_root))

        assert result.added == ["work/old.txt"]
        assert cfg.linked_roots == {"work": str(work)}

    async def test_a_scoped_config_gets_the_registry_and_the_global_one_stays(
        self, isolated_env, tmp_path
    ):
        from lilbee.core import settings
        from lilbee.core.config import config_scope
        from lilbee.data.ingest import sync

        work = self._root(tmp_path, "work", "old.txt")
        notes = self._root(tmp_path, "notes", "plan.txt")
        settings.set_value(cfg.data_root, "linked_roots", {"notes": str(notes)})
        cfg.linked_roots = {"work": str(work)}
        scoped = cfg.model_copy(update={"linked_roots": {}})

        with config_scope(scoped):
            result = await sync(quiet=True)

        assert result.added == ["notes/plan.txt"]
        assert scoped.linked_roots == {"notes": str(notes)}
        assert cfg.linked_roots == {"work": str(work)}
