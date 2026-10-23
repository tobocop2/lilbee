"""Tests for LanceDB store operations: hybrid search + FTS index lifecycle."""

import errno
import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pytest

from lilbee.core.config import CHUNKS_TABLE, META_TABLE, PAGE_TEXTS_TABLE, SOURCES_TABLE, cfg
from lilbee.data.store import (
    ChunkType,
    ChunkWrite,
    CitationRecord,
    SearchChunk,
    SearchScope,
    SourceMeta,
    SourceMove,
    SourceRelocation,
    SourcesCheckpoint,
    SourceType,
    Store,
    cosine_sim,
    escape_sql_string,
    mmr_rerank,
    scope_to_chunk_type,
)
from lilbee.data.store.lance_helpers import (
    _has_fts_index,
    _has_scalar_index,
    _has_vector_index,
    _index_registry,
    open_committed,
    table_names,
)
from lilbee.runtime.lock import LockingUnsupportedError, write_lock
from tests._mock_effects import repeat_last


@pytest.fixture()
def test_config(tmp_path):
    """Build a Config pointing at a temp directory."""
    return cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb_test"})


@pytest.fixture()
def store(test_config):
    """A Store instance backed by the temp config."""
    return Store(test_config)


def _make_records(n=3, dim=None, chunk_type="raw"):
    if dim is None:
        dim = cfg.embedding_dim
    return [
        {
            "source": f"doc{i}.md",
            "content_type": "text",
            "chunk_type": chunk_type,
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": f"chunk number {i} with some text",
            "chunk_index": i,
            "vector": [float(i) / n] * dim,
        }
        for i in range(n)
    ]


_EARLY_SOURCES_SCHEMA = pa.schema(
    [
        pa.field("filename", pa.utf8()),
        pa.field("file_hash", pa.utf8()),
        pa.field("ingested_at", pa.utf8()),
        pa.field("chunk_count", pa.int32()),
        pa.field("source_type", pa.utf8()),
    ]
)


def _early_source(filename, file_hash="h", ingested_at="", chunk_count=1):
    """One ``_sources`` row as a lilbee from before the stat and metadata columns wrote it."""
    return {
        "filename": filename,
        "file_hash": file_hash,
        "ingested_at": ingested_at,
        "chunk_count": chunk_count,
        "source_type": "document",
    }


def _open_after(config, tables):
    """A Store over a library an older lilbee wrote: *tables* maps a name to its Arrow rows."""
    import lancedb

    db = lancedb.connect(str(config.lancedb_dir))
    for name, rows in tables.items():
        db.create_table(name, rows)
    return Store(config)


def _sources_of(store, table_name=CHUNKS_TABLE):
    """The path of the source of each row of *table_name*."""
    ids = store.open_table(table_name).to_arrow().column("source_id").to_pylist()
    paths = store.source_paths(ids)
    return [paths[source_id] for source_id in ids]


def _axis_record(source, axis, dim):
    """A chunk record whose vector is the float32 array the embedder now returns."""
    vector = np.zeros(dim, dtype=np.float32)
    vector[axis] = 1.0
    return {
        "source": source,
        "content_type": "text",
        "chunk_type": "raw",
        "page_start": 0,
        "page_end": 0,
        "line_start": 0,
        "line_end": 0,
        "chunk": f"{source} body",
        "chunk_index": 0,
        "vector": vector,
    }


class TestNumpyVectorRoundTrip:
    """Embedder vectors reach LanceDB as float32 arrays, never as Python float lists."""

    def test_stores_array_vectors_bit_identically(self, store):
        dim = cfg.embedding_dim
        vector = np.linspace(-1.0, 1.0, dim, dtype=np.float32)
        record = _axis_record("exact.md", 0, dim) | {"vector": vector}
        store.add_chunks([record])
        row = store.open_table(CHUNKS_TABLE).search().to_list()[0]
        assert np.array_equal(np.asarray(row["vector"], dtype=np.float32), vector)

    def test_array_query_vector_finds_its_own_row(self, store):
        dim = cfg.embedding_dim
        store.add_chunks([_axis_record("a.md", 0, dim), _axis_record("b.md", 1, dim)])
        query = np.zeros(dim, dtype=np.float32)
        query[1] = 1.0
        hits = store.search(query, top_k=1, max_distance=0, query_text=None)
        assert [h.source for h in hits] == ["b.md"]

    def test_array_dim_mismatch_is_rejected(self, store):
        record = _axis_record("wrong.md", 0, cfg.embedding_dim - 1)
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            store.add_chunks([record])


class TestWriteLockDir:
    def test_write_lock_keys_on_store_config_dir(self, store, test_config):
        """A per-instance store locks its own lancedb_dir, not the global cfg dir."""
        from lilbee.core.config import cfg as global_cfg

        assert test_config.lancedb_dir != global_cfg.lancedb_dir
        test_config.lancedb_dir.mkdir(parents=True, exist_ok=True)
        with store._write_lock(timeout=2):
            assert (test_config.lancedb_dir / ".lock").exists()


class _TableProxy:
    """Table proxy running *on_delete* in place of ``delete``; other calls pass through."""

    def __init__(self, table, on_delete):
        self._table = table
        self._on_delete = on_delete

    def delete(self, predicate):
        self._on_delete(predicate)

    def __getattr__(self, name):
        return getattr(self._table, name)


def _raising_delete(predicate):
    raise RuntimeError("delete failed")


class TestClearAndAdd:
    def test_replaces_rows_atomically(self, store):
        import pyarrow as pa

        schema = pa.schema([pa.field("concept", pa.utf8()), pa.field("n", pa.int64())])
        store.clear_and_add("t_demo", schema, [{"concept": "a", "n": 1}], "concept IS NOT NULL")
        store.clear_and_add("t_demo", schema, [{"concept": "b", "n": 2}], "concept IS NOT NULL")
        rows = store.open_table("t_demo").search().to_list()
        assert {r["concept"] for r in rows} == {"b"}  # old row replaced, not appended

    def test_holds_lock_across_delete_and_add(self, store, monkeypatch):
        import pyarrow as pa

        from lilbee.runtime.lock import _write_mutex

        schema = pa.schema([pa.field("concept", pa.utf8())])
        store.clear_and_add("t_lock", schema, [{"concept": "seed"}], "concept IS NOT NULL")

        locked_during: list[bool] = []
        import lilbee.data.store.core as core_mod

        real = core_mod.ensure_table

        def spying_table(db, name, table_schema):
            table = real(db, name, table_schema)

            def on_delete(predicate):
                locked_during.append(_write_mutex.locked())
                table.delete(predicate)

            return _TableProxy(table, on_delete)

        monkeypatch.setattr(core_mod, "ensure_table", spying_table)
        store.clear_and_add("t_lock", schema, [{"concept": "next"}], "concept IS NOT NULL")
        assert locked_during == [True]  # delete ran under the write lock

    def test_delete_failure_propagates_without_adding(self, store, monkeypatch):
        """Swallowing it would add rows over stale ones and report success."""
        import pyarrow as pa

        import lilbee.data.store.core as core_mod

        schema = pa.schema([pa.field("concept", pa.utf8())])
        store.clear_and_add("t_fail", schema, [{"concept": "old"}], "concept IS NOT NULL")
        real = core_mod.ensure_table
        monkeypatch.setattr(
            core_mod,
            "ensure_table",
            lambda db, name, s: _TableProxy(real(db, name, s), _raising_delete),
        )
        with pytest.raises(RuntimeError, match="delete failed"):
            store.clear_and_add("t_fail", schema, [{"concept": "new"}], "concept IS NOT NULL")
        rows = store.open_table("t_fail").search().to_list()
        assert {r["concept"] for r in rows} == {"old"}  # unchanged; new rows not added


def _page_records(source, n, chunk_type=ChunkType.WIKI):
    """*n* chunk records of the one source *source*."""
    return [{**record, "source": source} for record in _make_records(n, chunk_type=chunk_type)]


class TestReplaceChunks:
    def test_replaces_the_source_rows_of_the_type_in_place(self, store):
        store.add_chunks(_page_records("wiki/a.md", 2))
        store.add_chunks(_page_records("a.md", 1, chunk_type="raw"))
        store.add_chunks(_page_records("wiki/b.md", 1))
        replacement = _page_records("wiki/a.md", 1)
        replacement[0]["chunk"] = "the only wiki row left"

        added = store.replace_chunks("wiki/a.md", ChunkType.WIKI, replacement)

        assert added == 1
        rows = store.get_chunks_by_source("wiki/a.md")
        assert [(r.chunk_type, r.chunk) for r in rows] == [
            (ChunkType.WIKI, "the only wiki row left")
        ]
        assert len(store.get_chunks_by_source("wiki/b.md")) == 1
        assert len(store.get_chunks_by_source("a.md")) == 1

    def test_empty_replacement_just_clears(self, store):
        store.add_chunks(_page_records("wiki/a.md", 2))
        assert store.replace_chunks("wiki/a.md", ChunkType.WIKI, []) == 0
        assert store.open_table(CHUNKS_TABLE).search().to_list() == []

    def test_a_page_with_no_rows_takes_a_wiki_source_row_no_listing_shows(self, store):
        store.replace_chunks("wiki/a.md", ChunkType.WIKI, _page_records("wiki/a.md", 1))

        assert store.get_sources() == []
        assert [r.source for r in store.get_chunks_by_source("wiki/a.md")] == ["wiki/a.md"]
        types = store.open_table(SOURCES_TABLE).to_arrow().column("source_type").to_pylist()
        assert types == [SourceType.WIKI]

    def test_bad_dimension_is_rejected_before_the_delete(self, store):
        store.add_chunks(_page_records("wiki/a.md", 1))
        bad = _make_records(1, dim=cfg.embedding_dim - 1, chunk_type=ChunkType.WIKI)

        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            store.replace_chunks("wiki/a.md", ChunkType.WIKI, bad)

        assert len(store.open_table(CHUNKS_TABLE).search().to_list()) == 1

    def test_delete_failure_propagates_without_adding(self, store, monkeypatch):
        """The caller has to see it: a wiki page whose rows were not swapped is stale."""
        store.add_chunks(_page_records("wiki/a.md", 1))
        real = store._chunks_table
        monkeypatch.setattr(store, "_chunks_table", lambda: _TableProxy(real(), _raising_delete))

        with pytest.raises(RuntimeError, match="delete failed"):
            store.replace_chunks("wiki/a.md", ChunkType.WIKI, _page_records("wiki/a.md", 1))
        assert len(store.open_table(CHUNKS_TABLE).search().to_list()) == 1


class TestDeleteAllWikiRows:
    def test_removes_wiki_chunks_and_citations_but_not_documents(self, store):
        from tests.conftest import make_citation

        store.add_chunks(_make_records(2, chunk_type=ChunkType.WIKI))
        store.add_chunks(_make_records(1, chunk_type="raw"))
        store.add_citations([make_citation(wiki_source="wiki/concepts/a.md")])
        store.replace_wiki_mentions_for_source("doc0.md", [_mention("boeing", "doc0.md", 2, [0])])

        assert store.delete_all_wiki_rows() is True

        rows = store.open_table(CHUNKS_TABLE).search().to_list()
        assert [r["chunk_type"] for r in rows] == ["raw"]
        assert store.wiki_citation_sources() == set()
        assert store.wiki_mention_rows() == []

    def test_empty_store_succeeds(self, store):
        assert store.delete_all_wiki_rows() is True

    def test_reports_a_failed_delete(self, store, monkeypatch):
        """A caller must not report a wipe over a swallowed failure."""
        store.add_chunks(_make_records(1, chunk_type=ChunkType.WIKI))
        monkeypatch.setattr(store, "clear_table", lambda name, predicate: False)
        assert store.delete_all_wiki_rows() is False

    def test_every_delete_runs_even_when_the_first_fails(self, store, monkeypatch):
        """A short-circuit would leave citations or mentions behind whenever an
        earlier delete failed, so every delete runs and none is skipped."""
        cleared: list[str] = []

        def _clear(name: str, predicate: str) -> bool:
            cleared.append(name)
            return False

        monkeypatch.setattr(store, "clear_table", _clear)
        store.delete_all_wiki_rows()
        assert len(cleared) == 3


class TestWikiChunkSources:
    def test_returns_only_wiki_row_sources(self, store):
        """The raw row carries a source of its own. Reusing a wiki row's source
        would leave the filter nothing to remove, and an implementation
        returning every source in the table would answer identically. Prune
        feeds this set to delete_by_source for anything with no page on disk,
        so an unfiltered answer deletes the raw corpus."""
        store.add_chunks(_make_records(2, chunk_type=ChunkType.WIKI))
        store.add_chunks(_make_records(3, chunk_type="raw")[2:])
        assert store.wiki_chunk_sources() == {"doc0.md", "doc1.md"}

    def test_empty_store_returns_empty_set(self, store):
        assert store.wiki_chunk_sources() == set()


def _mention(slug, source, count, indices, kind="entity", type_hint="ORG"):
    return {
        "slug": slug,
        "label": slug.replace("-", " ").title(),
        "kind": kind,
        "type_hint": type_hint,
        "source": source,
        "mention_count": count,
        "chunk_indices": indices,
    }


class TestWikiPagesCiting:
    def test_it_names_the_pages_citing_a_source_at_or_below_each_path(self, store):
        from tests.conftest import make_citation

        cited = {
            "work/a.md": "wiki/entities/a.md",
            "work/deep/b.md": "wiki/entities/b.md",
            "docs.zip/m.md": "wiki/entities/m.md",
            "workshop/x.md": "wiki/entities/x.md",
            "plan.md": "wiki/concepts/plan.md",
        }
        store.add_citations(
            [make_citation(wiki_source=page, source_filename=src) for src, page in cited.items()]
        )

        assert store.wiki_pages_citing(["work", "docs.zip"]) == {
            "wiki/entities/a.md",
            "wiki/entities/b.md",
            "wiki/entities/m.md",
        }
        assert store.wiki_pages_citing(["plan.md"]) == {"wiki/concepts/plan.md"}
        assert store.wiki_pages_citing(["elsewhere"]) == set()
        assert store.wiki_pages_citing([]) == set()

    def test_a_store_with_no_citation_table_names_none(self, store):
        assert store.wiki_pages_citing(["work"]) == set()


class TestWikiCitationsByPage:
    def test_each_page_holds_its_citations_with_the_path_its_source_has_now(
        self, store, test_config, tmp_path
    ):
        from tests.conftest import make_citation

        notes = tmp_path / "notes"
        test_config.linked_roots = {"work": str(notes / "work")}
        store.add_citations(
            [
                make_citation(wiki_source="wiki/entities/a.md", source_filename="work/a.md"),
                make_citation(
                    wiki_source="wiki/entities/a.md", source_filename="plan.md", citation_key="src2"
                ),
                make_citation(wiki_source="wiki/concepts/b.md", source_filename="work/a.md"),
            ]
        )
        test_config.linked_roots = {"notes": str(notes)}

        by_page = store.wiki_citations_by_page()

        shown = {
            page: sorted((r["citation_key"], r["source_filename"]) for r in records)
            for page, records in by_page.items()
        }
        assert shown == {
            "wiki/entities/a.md": [("src1", "notes/work/a.md"), ("src2", "plan.md")],
            "wiki/concepts/b.md": [("src1", "notes/work/a.md")],
        }

    def test_a_store_with_no_citation_table_holds_none(self, store):
        assert store.wiki_citations_by_page() == {}


class TestWikiMentions:
    def test_a_row_names_its_source_by_id_and_by_path(self, store):
        store.replace_wiki_mentions_for_source("a.md", [_mention("boeing", "a.md", 2, [0, 1])])
        (row,) = store.wiki_mention_rows()
        assert row["source"] == "a.md"
        assert row["source_id"] == store.source_ids(["a.md"], SourceType.DOCUMENT)["a.md"]

    def test_round_trips_rows(self, store):
        store.replace_wiki_mentions_for_source("a.md", [_mention("boeing", "a.md", 2, [0, 1])])
        rows = store.wiki_mention_rows()
        assert len(rows) == 1
        assert rows[0]["slug"] == "boeing"
        assert rows[0]["mention_count"] == 2
        assert list(rows[0]["chunk_indices"]) == [0, 1]

    def test_replace_only_touches_the_named_source(self, store):
        store.replace_wiki_mentions_for_source("a.md", [_mention("boeing", "a.md", 2, [0, 1])])
        store.replace_wiki_mentions_for_source("b.md", [_mention("boeing", "b.md", 2, [0, 1])])
        # Re-syncing a.md must not disturb b.md's evidence.
        store.replace_wiki_mentions_for_source("a.md", [_mention("boeing", "a.md", 3, [0, 1, 2])])
        by_source = {
            (r["slug"], r["source"]): r["mention_count"] for r in store.wiki_mention_rows()
        }
        assert by_source == {("boeing", "a.md"): 3, ("boeing", "b.md"): 2}

    def test_aggregate_across_separately_synced_sources(self, store):
        """The corpus-wide total is the sum over per-source rows -- the property
        the stub floor is judged on."""
        store.replace_wiki_mentions_for_source("a.md", [_mention("boeing", "a.md", 2, [0, 1])])
        store.replace_wiki_mentions_for_source("b.md", [_mention("boeing", "b.md", 2, [0, 1])])
        total = sum(r["mention_count"] for r in store.wiki_mention_rows(slugs=["boeing"]))
        assert total == 4

    def test_rows_filter_by_slug(self, store):
        store.replace_wiki_mentions_for_source(
            "a.md",
            [_mention("boeing", "a.md", 2, [0]), _mention("airbus", "a.md", 1, [1])],
        )
        assert {r["slug"] for r in store.wiki_mention_rows(slugs=["boeing"])} == {"boeing"}
        assert store.wiki_mention_rows(slugs=[]) == []

    def test_clear_drops_every_row(self, store):
        store.replace_wiki_mentions_for_source("a.md", [_mention("boeing", "a.md", 2, [0])])
        assert store.clear_wiki_mentions() is True
        assert store.wiki_mention_rows() == []

    def test_rows_on_a_store_that_never_wrote_mentions(self, store):
        assert store.wiki_mention_rows() == []
        assert store.wiki_mention_rows(slugs=["boeing"]) == []

    def test_has_wiki_mentions_reflects_presence(self, store):
        assert store.has_wiki_mentions() is False
        store.replace_wiki_mentions_for_source("a.md", [_mention("boeing", "a.md", 2, [0])])
        assert store.has_wiki_mentions() is True
        store.clear_wiki_mentions()
        assert store.has_wiki_mentions() is False

    def test_removing_a_source_drops_its_mentions(self, store):
        """Mentions are per-source, so a source deletion takes its evidence with
        its chunks -- the refresh never has to subtract a removed source."""
        store.add_chunks(_records_for("doc.md", 1))
        store.replace_wiki_mentions_for_source("doc.md", [_mention("boeing", "doc.md", 2, [0])])
        store.replace_wiki_mentions_for_source("keep.md", [_mention("airbus", "keep.md", 2, [0])])
        store.delete_by_source("doc.md", SourceType.DOCUMENT)
        assert {r["source"] for r in store.wiki_mention_rows()} == {"keep.md"}


class TestWikiCitationSources:
    def test_returns_distinct_wiki_sources(self, store):
        from tests.conftest import make_citation

        store.add_citations(
            [
                make_citation(wiki_source="wiki/concepts/a.md", citation_key="src1"),
                make_citation(wiki_source="wiki/concepts/a.md", citation_key="src2"),
                make_citation(wiki_source="wiki/entities/b.md", citation_key="src1"),
            ]
        )
        assert store.wiki_citation_sources() == {"wiki/concepts/a.md", "wiki/entities/b.md"}

    def test_empty_store_returns_empty_set(self, store):
        assert store.wiki_citation_sources() == set()


class TestReplaceCitationsForWiki:
    def test_swaps_one_page_and_leaves_the_others(self, store):
        from tests.conftest import make_citation

        store.add_citations(
            [
                make_citation(wiki_source="wiki/concepts/a.md", citation_key="old"),
                make_citation(wiki_source="wiki/concepts/b.md", citation_key="other"),
            ]
        )

        store.replace_citations_for_wiki(
            "wiki/concepts/a.md",
            [make_citation(wiki_source="wiki/concepts/a.md", citation_key="new")],
        )

        assert [c["citation_key"] for c in store.get_citations_for_wiki("wiki/concepts/a.md")] == [
            "new"
        ]
        assert [c["citation_key"] for c in store.get_citations_for_wiki("wiki/concepts/b.md")] == [
            "other"
        ]


class TestEnsureFtsIndex:
    def test_noop_when_no_table(self, store):
        store.ensure_fts_index()
        assert not store._fts_ready

    def test_creates_index_after_add(self, store):
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        assert store._fts_ready

    def test_handles_exception_gracefully(self, store):
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        assert table is not None
        with mock.patch.object(
            type(table),
            "create_index",
            side_effect=RuntimeError("boom"),
        ):
            store.ensure_fts_index()
            assert not store._fts_ready

    def test_bm25_probe_returns_empty_when_index_unavailable(self, store):
        """A populated table whose FTS index won't build yields no probe hits."""
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        assert table is not None
        with mock.patch.object(
            type(table),
            "create_index",
            side_effect=RuntimeError("boom"),
        ):
            assert store.bm25_probe("anything") == []

    def test_second_call_optimizes_instead_of_rebuilding(self, store):
        """Incremental path: once the FTS index exists, ensure_fts_index
        calls table.optimize() rather than rebuilding from scratch. Prevents
        O(total_chunks) rebuild cost on every sync."""
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        assert store._fts_ready

        table = store.open_table("chunks")
        assert table is not None
        with (
            mock.patch.object(type(table), "create_index") as create_spy,
            mock.patch.object(type(table), "optimize") as optimize_spy,
        ):
            store.ensure_fts_index()

        create_spy.assert_not_called()
        optimize_spy.assert_called_once()

    def test_optimize_that_drops_index_files_repairs_the_registry(self, store, test_config):
        """An index whose files are gone right after optimize() is rebuilt in the same step.

        The prune is simulated; the producer of the missing files is not confirmed."""
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        table = store.open_table("chunks")
        assert table is not None

        with mock.patch.object(
            type(table), "optimize", side_effect=lambda: _remove_index_files(test_config)
        ):
            store.ensure_fts_index()

        assert _has_fts_index(_readable_registry(store.open_table("chunks")))
        assert store.bm25_probe("chunk number 1")

    def test_hybrid_failure_is_reported_as_a_health_warning(self, store):
        """A corpus-wide FTS breakage drops every query to vector-only recall.
        The log line alone never reached the user, so the store reports it."""
        from lilbee.core.health_warnings import WarningCode

        store.add_chunks(_make_records())
        store.ensure_fts_index()
        assert store.health_warnings() == []

        with mock.patch.object(
            type(store), "_hybrid_search", side_effect=RuntimeError("fts index gone")
        ):
            store.search([0.5] * cfg.embedding_dim, top_k=1, query_text="anything")

        codes = [w.code for w in store.health_warnings()]
        assert WarningCode.FTS_UNAVAILABLE in codes

    def test_an_index_that_never_builds_is_reported(self, store):
        """A create_index failure leaves keyword search off for every query. The
        hybrid arm never runs, so the failure has to be reported from readiness."""
        from lilbee.core.health_warnings import WarningCode

        store.add_chunks(_make_records())
        with mock.patch.object(type(store), "ensure_fts_index"):
            store.search([0.5] * cfg.embedding_dim, top_k=1, query_text="anything")

        codes = [w.code for w in store.health_warnings()]
        assert WarningCode.FTS_UNAVAILABLE in codes

    def test_an_emptied_corpus_reports_no_keyword_warning(self, store):
        """A store whose chunks are all gone keeps the table and cannot index it.
        There is nothing to degrade, so the warning must not fire."""
        store.add_chunks(_make_records())
        for record in _make_records():
            store.delete_by_source(record["source"], SourceType.DOCUMENT)
        with mock.patch.object(type(store), "ensure_fts_index"):
            store.search([0.5] * cfg.embedding_dim, top_k=1, query_text="anything")

        assert store.health_warnings() == []

    def test_hybrid_recovery_clears_the_health_warning(self, store):
        """The warning names a remedy, so running it must clear the warning.
        A latching flag would train the user to ignore the banner."""
        from lilbee.core.health_warnings import WarningCode

        store.add_chunks(_make_records())
        store.ensure_fts_index()
        with mock.patch.object(
            type(store), "_hybrid_search", side_effect=RuntimeError("fts index gone")
        ):
            store.search([0.5] * cfg.embedding_dim, top_k=1, query_text="anything")
        assert WarningCode.FTS_UNAVAILABLE in [w.code for w in store.health_warnings()]

        # Index healthy again: the next query reports the current state.
        store.search([0.5] * cfg.embedding_dim, top_k=1, query_text="anything")
        assert store.health_warnings() == []

    def test_optimize_failure_keeps_hybrid_ready(self, store):
        """An optimize() crash on an already-built index (a LanceDB encoding
        bug bites large corpora) must not disable hybrid search: the index
        still serves queries, so _fts_ready stays True instead of silently
        dropping every query to the vector-only fallback."""
        store.add_chunks(_make_records())
        store.ensure_fts_index()  # builds the index
        store._fts_ready = False  # a fresh process is unaware the index exists yet
        table = store.open_table("chunks")
        assert table is not None
        with mock.patch.object(
            type(table),
            "optimize",
            side_effect=RuntimeError("lance list offset overflow"),
        ):
            store.ensure_fts_index()
        assert store._fts_ready is True

    def test_positional_index_overflow_rebuilds_positionless(self, store):
        """A store whose FTS index was built with positions overflows on every
        optimize(); catching that specific error rebuilds the index positionless
        (replace=True) so index maintenance can complete instead of failing
        forever with no remediation short of a full re-ingest."""
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        table = store.open_table("chunks")
        assert table is not None
        overflow = RuntimeError("Max offset 1897296 exceeds length of values 1067891")
        with (
            mock.patch.object(type(table), "optimize", side_effect=overflow),
            mock.patch.object(type(table), "create_index") as rebuild,
        ):
            store.ensure_fts_index()
        assert any(
            c.args[:1] == ("chunk",)
            and c.kwargs.get("replace") is True
            and c.kwargs["config"].with_position is False
            for c in rebuild.call_args_list
        )
        assert store._fts_ready is True

    def test_generic_optimize_failure_does_not_rebuild(self, store, caplog):
        """An unrelated optimize() failure keeps the existing index and does NOT
        pay for a full positionless rebuild it cannot fix. It must also log a
        warning so an operator debugging a large corpus is not left in silence."""
        import logging

        store.add_chunks(_make_records())
        store.ensure_fts_index()
        table = store.open_table("chunks")
        assert table is not None
        with (
            mock.patch.object(type(table), "optimize", side_effect=RuntimeError("disk full")),
            mock.patch.object(type(table), "create_index") as rebuild,
            caplog.at_level(logging.WARNING),
        ):
            store.ensure_fts_index()
        rebuild.assert_not_called()
        assert any("optimize()" in r.message for r in caplog.records)

    def test_positionless_rebuild_covers_the_title_index_too(self, store, test_config):
        """The title index carries the same positional encoding that overflows, so
        a rebuild that skipped it would leave optimize() failing on half the store."""
        test_config.title_search = True
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        table = store.open_table("chunks")
        assert table is not None
        overflow = RuntimeError("Max offset 1897296 exceeds length of values 1067891")
        with (
            mock.patch.object(type(table), "optimize", side_effect=overflow),
            mock.patch.object(type(table), "create_index") as rebuild,
        ):
            store.ensure_fts_index()
        rebuilt = [c for c in rebuild.call_args_list if c.kwargs.get("replace") is True]
        assert [c.args[0] for c in rebuilt] == ["chunk", "title"]
        assert all(c.kwargs["config"].with_position is False for c in rebuilt)

    def test_positionless_rebuild_failure_leaves_the_old_index_serving(self, store, caplog):
        """The rebuild is remediation, not a dependency: the existing index still
        answers queries, so a failed rebuild warns rather than failing the build."""
        import logging

        store.add_chunks(_make_records())
        store.ensure_fts_index()
        table = store.open_table("chunks")
        assert table is not None
        overflow = RuntimeError("Max offset 1897296 exceeds length of values 1067891")
        with (
            mock.patch.object(type(table), "optimize", side_effect=overflow),
            mock.patch.object(type(table), "create_index", side_effect=RuntimeError("no space")),
            caplog.at_level(logging.WARNING),
        ):
            store.ensure_fts_index()  # must not raise
        assert any("Positionless FTS rebuild failed" in r.message for r in caplog.records)

    def test_first_call_creates_without_replace(self, store):
        """Fresh table goes through create_index(config=FTS()) with replace=False."""
        from lancedb.index import FTS

        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        assert table is not None

        with mock.patch.object(type(table), "create_index") as create_spy:
            store.ensure_fts_index()

        create_spy.assert_called_once()
        # Builds an FTS index on the chunk column, incrementally (replace was not
        # True, which would defeat the purpose of incremental optimize()).
        args, kwargs = create_spy.call_args
        assert args[0] == "chunk"
        assert isinstance(kwargs.get("config"), FTS)
        assert kwargs.get("config").with_position is False
        assert kwargs.get("replace") is False

    def test_first_call_creates_both_indexes_when_title_search_on(self, store, test_config):
        """Fresh table creates the chunk and title indexes, both positionless
        with replace=False, when the title arm is enabled."""
        test_config.title_search = True
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        assert table is not None

        with mock.patch.object(type(table), "create_index") as create_spy:
            store.ensure_fts_index()

        assert [c.args[0] for c in create_spy.call_args_list] == ["chunk", "title"]
        # Verify replace was NOT True (would defeat the purpose of incremental)
        assert all(c.kwargs.get("replace") is False for c in create_spy.call_args_list)
        # Both indexes are positionless: with_position=True overflows LanceDB's
        # list encoding on a large corpus, and no lilbee query needs exact-phrase
        # matching.
        assert all(c.kwargs["config"].with_position is False for c in create_spy.call_args_list)

    def test_fts_quoted_query_matches_terms_not_a_phrase(self, store):
        """A quoted query must return term matches, not raise on the positionless index.

        The chunk index carries no token positions, so a phrase query would
        error. FTS goes through ``MatchQuery``, which matches the quoted span's
        plain terms instead of parsing it as a phrase.
        """
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        results = store.bm25_probe('"some text"')
        assert results
        assert all("some text" in r.chunk for r in results)

    def test_bm25_probe_populates_bm25_score(self, store):
        """LanceDB FTS returns rows keyed on ``_score``; the probe must surface it as
        ``bm25_score`` so confidence-based expansion-skip sees a real signal. It must
        NOT land in the fusion-scale ``relevance_score`` (which stays None for FTS)."""
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        results = store.bm25_probe("text")
        assert results
        assert all(r.bm25_score is not None for r in results)
        assert results[0].bm25_score > 0
        assert all(r.relevance_score is None for r in results)

    def test_bm25_probe_filters_by_chunk_type(self, store):
        """An explicit scope on a ``term:`` query must be honoured by the probe."""
        store.add_chunks(_make_records(n=2, chunk_type="raw"))
        store.add_chunks(_make_records(n=2, chunk_type="wiki"))
        store.ensure_fts_index()
        results = store.bm25_probe("text", chunk_type=ChunkType.WIKI)
        assert results
        assert all(r.chunk_type == ChunkType.WIKI for r in results)

    def test_bm25_probe_survives_a_failing_search(self, store):
        """A probe whose LanceDB query raises degrades to no hits, not an error."""
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        table = store.open_table("chunks")
        assert table is not None
        with mock.patch.object(type(table), "search", side_effect=RuntimeError("boom")):
            assert store.bm25_probe("text") == []

    def test_a_registered_index_whose_files_are_gone_is_rebuilt(self, store, test_config):
        """LanceDB keeps the registration when the index directory vanishes, and
        every keyword query then fails on a missing file. The maintenance pass
        rebuilds such an index instead of optimizing it."""
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        _remove_index_files(test_config)

        store.ensure_fts_index()

        hits = store.search([0.5] * cfg.embedding_dim, top_k=1, query_text="chunk number 1")
        assert hits
        assert store.health_warnings() == []

    def test_the_search_path_rebuilds_a_dangling_index_instead_of_latching_ready(
        self, store, test_config
    ):
        """The read path marks an existing index ready before taking the lock; a
        dangling one must not count as existing, or every query tracebacks."""
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        _remove_index_files(test_config)
        fresh = Store(test_config)

        hits = fresh.search([0.5] * cfg.embedding_dim, top_k=1, query_text="chunk number 1")

        assert hits
        assert fresh.health_warnings() == []


def _remove_index_files(test_config, table_name=CHUNKS_TABLE, *, index_types=None):
    """Delete index directories under *table_name*, leaving the registrations.

    With *index_types*, only the indexes of those types lose their files.
    """
    import shutil

    from lancedb.db import LanceDBConnection

    table = LanceDBConnection(str(test_config.lancedb_dir)).open_table(table_name)
    indices_dir = test_config.lancedb_dir / f"{table_name}.lance" / "_indices"
    for idx in table.list_indices():
        if index_types is None or idx.index_type in index_types:
            shutil.rmtree(indices_dir / idx.index_uuid)


def _readable_registry(table):
    """The table's index registry, asserting that lancedb can list it."""
    indices = _index_registry(table, Path(table.uri).parent)
    assert indices is not None
    return indices


class TestIndexMismatch:
    """The one verdict search, the embedding setter, sync and health share."""

    def test_no_index_reports_no_mismatch(self, store):
        assert store.index_mismatch() is None

    def test_same_embedder_reports_no_mismatch(self, store):
        store.add_chunks(_make_records())
        assert store.index_mismatch() is None

    def test_another_model_name_is_a_mismatch(self, store, test_config):
        store.add_chunks(_make_records())
        built_with = test_config.embedding_model
        test_config.embedding_model = "acme/other-GGUF/other.gguf"

        mismatch = store.index_mismatch()

        assert mismatch is not None
        assert mismatch.persisted_model == built_with
        assert mismatch.current_model == "acme/other-GGUF/other.gguf"
        assert mismatch.dims_match

    def test_another_dimension_is_a_mismatch_even_with_the_same_name(self, store, test_config):
        store.add_chunks(_make_records())
        test_config.embedding_dim = test_config.embedding_dim // 2

        mismatch = store.index_mismatch()

        assert mismatch is not None
        assert not mismatch.dims_match

    def test_switching_back_to_the_building_model_clears_the_mismatch(self, store, test_config):
        store.add_chunks(_make_records())
        built_with = test_config.embedding_model
        test_config.embedding_model = "acme/other-GGUF/other.gguf"
        assert store.index_mismatch() is not None

        test_config.embedding_model = built_with

        assert store.index_mismatch() is None

    def test_an_index_built_with_another_model_is_a_health_warning(self, store, test_config):
        """Search refuses such an index with a 409, but a client that has not
        searched yet sees a green health; the warning carries both refs."""
        from lilbee.core.health_warnings import WarningCode

        store.add_chunks(_make_records())
        built_with = test_config.embedding_model
        test_config.embedding_model = "acme/other-GGUF/other.gguf"

        warnings = store.health_warnings()

        assert [w.code for w in warnings] == [WarningCode.INDEX_EMBEDDING_MISMATCH]
        assert built_with in warnings[0].message
        assert "acme/other-GGUF/other.gguf" in warnings[0].message
        assert warnings[0].remedy is not None
        assert built_with in warnings[0].remedy

    def test_the_warning_clears_once_the_embedder_agrees_again(self, store, test_config):
        store.add_chunks(_make_records())
        built_with = test_config.embedding_model
        test_config.embedding_model = "acme/other-GGUF/other.gguf"
        assert store.health_warnings()

        test_config.embedding_model = built_with

        assert store.health_warnings() == []


class TestSearchChunkScoreAlias:
    @staticmethod
    def _row(**extra):
        base = {
            "source": "a.md",
            "content_type": "text",
            "chunk_type": "raw",
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": "hi",
            "chunk_index": 0,
            "vector": [0.0] * cfg.embedding_dim,
        }
        base.update(extra)
        return base

    def test_fts_score_maps_to_bm25_score(self):
        """BM25/FTS ``_score`` populates the dedicated ``bm25_score`` field, kept
        separate from the fusion-scale ``relevance_score``."""
        chunk = SearchChunk(**self._row(_score=2.5))
        assert chunk.bm25_score == 2.5
        assert chunk.relevance_score is None

    def test_relevance_score_alias_still_works(self):
        chunk = SearchChunk(**self._row(_relevance_score=0.03))
        assert chunk.relevance_score == 0.03
        assert chunk.bm25_score is None

    def test_hybrid_row_keeps_scores_in_separate_fields(self):
        """A hybrid row carrying both keeps RRF in relevance_score and BM25 in bm25_score."""
        chunk = SearchChunk(**self._row(_relevance_score=0.03, _score=2.5))
        assert chunk.relevance_score == 0.03
        assert chunk.bm25_score == 2.5


def _make_indexable_records(n, dim):
    """Records with varied vectors so IVF_PQ has something to train on."""
    import math

    return [
        {
            "source": f"doc{i}.md",
            "content_type": "text",
            "chunk_type": "raw",
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": f"chunk number {i}",
            "chunk_index": i,
            "vector": [math.sin(i * 0.1 + j * 0.01) for j in range(dim)],
        }
        for i in range(n)
    ]


class TestEnsureScalarIndexes:
    """source and chunk_type get scalar indexes so their prefilters are lookups."""

    def test_creates_btree_on_source_and_bitmap_on_chunk_type(self, store):
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        assert table is not None
        with mock.patch.object(type(table), "create_scalar_index") as spy:
            store.ensure_scalar_indexes()
        assert [c.args[0] for c in spy.call_args_list] == ["source_id", "chunk_type"]
        kinds = {c.args[0]: c.kwargs.get("index_type") for c in spy.call_args_list}
        assert kinds == {"source_id": "BTREE", "chunk_type": "BITMAP"}
        assert all(c.kwargs.get("replace") is False for c in spy.call_args_list)

    def test_idempotent_once_the_indexes_exist(self, store):
        store.add_chunks(_make_records())
        store.ensure_scalar_indexes()  # builds them for real
        table = store.open_table("chunks")
        with mock.patch.object(type(table), "create_scalar_index") as spy:
            store.ensure_scalar_indexes()
        spy.assert_not_called()

    def test_handles_exception_gracefully(self, store):
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        assert table is not None
        with mock.patch.object(
            type(table), "create_scalar_index", side_effect=RuntimeError("boom")
        ):
            store.ensure_scalar_indexes()  # must not raise

    def test_noop_when_no_table(self, store):
        store.ensure_scalar_indexes()  # empty store, no chunks table yet

    def test_indexes_chunk_concepts_source_column(self, store):
        """The concept-boost path filters chunk_concepts by source_id per
        result, so that column gets its own BTree index too."""
        from lilbee.core.config import CHUNK_CONCEPTS_TABLE
        from lilbee.data.store import ensure_table
        from lilbee.retrieval.concepts.schema import _chunk_concepts_schema

        store.add_chunks(_make_records())
        cc = ensure_table(store.get_db(), CHUNK_CONCEPTS_TABLE, _chunk_concepts_schema())
        cc.add([{"source_id": "doc.md", "chunk_index": 0, "concept": "alpha"}])
        store.ensure_scalar_indexes()
        indices = _readable_registry(store.open_table(CHUNK_CONCEPTS_TABLE))
        assert _has_scalar_index(indices, "source_id")

    def test_one_column_failure_does_not_skip_the_other(self, store):
        """A BTree failure on 'source' must not skip the independent Bitmap on
        'chunk_type' -- each column gets its own try."""
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        attempted = []

        def _record(self, column, **kwargs):
            attempted.append(column)
            if column == "source_id":
                raise RuntimeError("boom")

        with mock.patch.object(type(table), "create_scalar_index", _record):
            store.ensure_scalar_indexes()
        assert attempted == ["source_id", "chunk_type"]

    def test_scalar_index_failure_on_populated_table_warns(self, store, caplog):
        """A create failure on a non-empty table warns (it silently loses the
        prefilter speedup), unlike the benign empty-table case."""
        import logging

        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        with (
            mock.patch.object(type(table), "create_scalar_index", side_effect=RuntimeError("boom")),
            caplog.at_level(logging.WARNING),
        ):
            store.ensure_scalar_indexes()
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Scalar index create failed" in r.message for r in warns)

    def test_scalar_build_skips_a_table_that_vanished_after_the_probe(self, store):
        """ensure_scalar_indexes probes for pending work, then builds under the
        lock; a table gone by then is skipped instead of crashing the build."""
        store.add_chunks(_make_records())
        assert store.open_table("no_such_table") is None
        store._ensure_scalar_index_on("no_such_table", (("source_id", "BTREE"),))  # must not raise

    def test_scalar_build_skips_a_column_the_table_does_not_have(self, store):
        """Columns are re-checked at build time, so one the schema lacks is
        skipped rather than attempted and logged as a failure."""
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        assert table is not None
        with mock.patch.object(type(table), "create_scalar_index") as spy:
            store._ensure_scalar_index_on("chunks", (("not_a_column", "BTREE"),))
        spy.assert_not_called()

    def test_search_builds_scalar_indexes_on_a_serve_only_store(self, store, test_config):
        """A store served without a fresh ingest never ran the ingest path that
        builds scalar indexes, so the first search builds them; later searches
        only probe (no rebuild). Readiness stays unlatched while chunk_concepts
        is missing so its index can build once the table appears."""
        store.add_chunks(_make_records())
        assert store._scalar_ready is False  # ingest path did not run here
        with mock.patch.object(
            store, "_ensure_scalar_index_on", wraps=store._ensure_scalar_index_on
        ) as spy:
            store.search([0.5] * test_config.embedding_dim, top_k=3)
            built = spy.call_count
            store.search([0.5] * test_config.embedding_dim, top_k=3)
        assert built >= 1
        assert spy.call_count == built  # second search probed, built nothing

        table = store.open_table("chunks")
        assert _has_scalar_index(_readable_registry(table), "source_id")
        assert _has_scalar_index(_readable_registry(table), "chunk_type")

    def test_fts_language_reaches_every_index_build(self, store, test_config):
        test_config.fts_language = "German"
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        with mock.patch.object(type(table), "create_index") as create:
            store.ensure_fts_index()
        assert create.call_args.kwargs["config"].language == "German"

    def test_pre_prefix_store_warns_once_for_a_doc_prefix_family(self, store, test_config, caplog):
        """A store built before its family's document prefixes existed warns
        (once) to rebuild instead of silently mixing embedding spaces."""
        import logging

        test_config.embedding_model = "nomic-ai/nomic-embed-text-v1.5-GGUF/n.gguf"
        store.add_chunks(_make_records())
        old_meta = {**store.get_meta(), "schema_version": 1}
        with (
            mock.patch.object(type(store), "get_meta", return_value=old_meta),
            caplog.at_level(logging.WARNING),
        ):
            store.search([0.5] * test_config.embedding_dim, top_k=1)
            store.search([0.5] * test_config.embedding_dim, top_k=1)
        warned = [r for r in caplog.records if "document prefixes" in r.message]
        assert len(warned) == 1

    def test_pre_prefix_store_reports_a_health_warning(self, store, test_config):
        """The rebuild advice was log-only, so a user whose retrieval was quietly
        degraded never saw it. The store reports it for the health endpoint."""
        from lilbee.core.health_warnings import WarningCode

        test_config.embedding_model = "nomic-ai/nomic-embed-text-v1.5-GGUF/n.gguf"
        store.add_chunks(_make_records())
        old_meta = {**store.get_meta(), "schema_version": 1}
        with mock.patch.object(type(store), "get_meta", return_value=old_meta):
            store.search([0.5] * test_config.embedding_dim, top_k=1)
            codes = [w.code for w in store.health_warnings()]

        assert WarningCode.EMBEDDING_PREFIX_MISMATCH in codes

    def test_rebuild_in_another_process_clears_the_prefix_warning(self, store, test_config):
        """`lilbee rebuild` is the remedy the warning names and it runs in its own
        process, so the server cannot report the verdict its query path cached."""
        from lilbee.core.health_warnings import WarningCode

        test_config.embedding_model = "nomic-ai/nomic-embed-text-v1.5-GGUF/n.gguf"
        store.add_chunks(_make_records())
        old_meta = {**store.get_meta(), "schema_version": 1}
        with mock.patch.object(type(store), "get_meta", return_value=old_meta):
            store.search([0.5] * test_config.embedding_dim, top_k=1)
            assert WarningCode.EMBEDDING_PREFIX_MISMATCH in [
                w.code for w in store.health_warnings()
            ]

        # Meta now reads at the current schema version, as after a rebuild.
        assert store.health_warnings() == []

    def test_blocking_index_builds_propagate_lock_timeouts(self, store, test_config):
        """Ingest-path callers keep the old contract: a lock timeout raises."""
        from lilbee.runtime.lock import LockTimeoutError

        test_config.title_search = True
        store.add_chunks(_make_records())
        with mock.patch.object(store, "_write_lock", side_effect=LockTimeoutError("held")):
            with pytest.raises(LockTimeoutError):
                store.ensure_fts_index()
            with pytest.raises(LockTimeoutError):
                store.ensure_scalar_indexes()
            with pytest.raises(LockTimeoutError):
                store.ensure_title_fts_index()

    def test_title_index_ensure_early_paths(self, store, test_config):
        """No chunks table is a no-op; an existing index latches without a build."""
        store.ensure_title_fts_index()
        assert store._title_fts_ready is False
        test_config.title_search = True
        store.add_chunks(_titled_records("a.pdf", 1, title="zebra manifesto"))
        store.ensure_fts_index()
        store._title_fts_ready = False
        store.ensure_title_fts_index()
        assert store._title_fts_ready is True

    def test_close_resets_index_latches(self, store):
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        store.close()
        assert store._fts_ready is False
        assert store._title_fts_ready is False
        assert store._scalar_ready is False

    def test_doc_prefix_warning_skips_a_store_without_meta(self, store):
        store._warn_stale_doc_prefix()
        assert store._doc_prefix_warned is True

    def test_search_survives_index_build_lock_contention(self, store, test_config):
        """Read-path index builds (scalar, FTS, title) skip when the write lock
        is held elsewhere; the query serves (vector-only if need be) instead of
        raising."""
        from lilbee.runtime.lock import LockTimeoutError

        test_config.title_search = True
        store.add_chunks(_make_records())
        with mock.patch.object(store, "_write_lock", side_effect=LockTimeoutError("held")):
            results = store.search(
                [0.5] * test_config.embedding_dim, top_k=3, query_text="chunk number"
            )
        assert results
        assert store._scalar_ready is False  # retried on a later search
        assert store._title_fts_ready is False

    @pytest.mark.skipif(sys.platform == "win32", reason="ENOLCK comes from POSIX flock")
    def test_search_survives_a_directory_that_cannot_lock(self, store, test_config):
        """Read-path index builds skip on a filesystem without locking; the query serves."""
        import fcntl  # POSIX only; skipped on Windows

        def _flock(_fd: int, _operation: int) -> None:
            raise OSError(errno.ENOLCK, "No locks available")

        test_config.title_search = True
        store.add_chunks(_make_records())
        with mock.patch.object(fcntl, "flock", _flock):
            results = store.search(
                [0.5] * test_config.embedding_dim, top_k=3, query_text="chunk number"
            )
            with pytest.raises(LockingUnsupportedError):
                store.ensure_scalar_indexes()
        assert results
        assert store._scalar_ready is False

    def test_fts_ensure_body_is_a_noop_without_a_chunks_table(self, store):
        store._ensure_fts_index_unlocked()
        assert store._fts_ready is False

    def test_scalar_index_builds_after_concepts_table_appears(self, store, test_config):
        """Serve ordering: chunk_concepts is created after the first search.
        The next search must still index it instead of latching ready early."""
        from lilbee.data.store.lance_helpers import ensure_table
        from lilbee.retrieval.concepts.schema import _chunk_concepts_schema

        store.add_chunks(_make_records())
        store.search([0.5] * test_config.embedding_dim, top_k=3)
        assert store._scalar_ready is False  # concepts table not there yet
        table = ensure_table(store.get_db(), "chunk_concepts", _chunk_concepts_schema())
        table.add([{"source_id": "doc0.md", "chunk_index": 0, "concept": "x"}])
        store.search([0.5] * test_config.embedding_dim, top_k=3)
        indices = _readable_registry(store.open_table("chunk_concepts"))
        assert _has_scalar_index(indices, "source_id")
        assert store._scalar_ready is True


def _dangling_scalar(store, table_name=CHUNKS_TABLE):
    """Columns of *table_name* whose scalar index is registered without its files."""
    from lilbee.data.store.lance_helpers import _scalar_index_dangling

    table = store.open_table(table_name)
    return _scalar_index_dangling(table, _readable_registry(table), store._config.lancedb_dir)


@contextmanager
def _spy_scalar_builds():
    """Record (column, replace) for every real create_scalar_index call."""
    from lancedb.table import LanceTable

    calls = []
    real = LanceTable.create_scalar_index

    def _record(self, column, **kwargs):
        calls.append((column, kwargs.get("replace")))
        return real(self, column, **kwargs)

    with mock.patch.object(LanceTable, "create_scalar_index", _record):
        yield calls


class TestScalarIndexDangling:
    """lancedb lists a scalar index whose directory is gone without error, so the
    store compares each registration with the ``_indices`` directory, reports the
    loss as SCALAR_INDEX_UNAVAILABLE, and replaces the index."""

    _SCALAR_TYPES = frozenset({"BTree", "Bitmap"})

    def _store_with_missing_scalar_files(self, store, test_config) -> Store:
        """A fresh Store over a corpus whose scalar index files were removed."""
        store.add_chunks(_make_records())
        store.ensure_scalar_indexes()
        store.ensure_fts_index()
        _remove_index_files(test_config, index_types=self._SCALAR_TYPES)
        return Store(test_config)

    def test_dangling_scalar_reports_scalar_warning_not_fts(self, store, test_config):
        from lilbee.core.health_warnings import WarningCode

        fresh = self._store_with_missing_scalar_files(store, test_config)

        assert sorted(_dangling_scalar(fresh)) == ["chunk_type", "source_id"]
        codes = [w.code for w in fresh.health_warnings()]
        assert WarningCode.SCALAR_INDEX_UNAVAILABLE in codes
        assert WarningCode.FTS_UNAVAILABLE not in codes

    def test_search_rebuilds_a_dangling_scalar_index_after_restart(self, store, test_config):
        """A fresh process verifies the scalar indexes on its first query and rebuilds."""
        fresh = self._store_with_missing_scalar_files(store, test_config)

        hits = fresh.search(
            [0.5] * cfg.embedding_dim,
            top_k=1,
            query_text="chunk number 1",
            chunk_type=ChunkType.RAW,
        )

        assert hits
        assert _dangling_scalar(fresh) == []
        assert fresh.health_warnings() == []

    def test_a_missing_bitmap_directory_is_replaced(self, store, test_config):
        store.add_chunks(_make_records())
        store.ensure_scalar_indexes()
        _remove_index_files(test_config, index_types={"Bitmap"})
        assert _dangling_scalar(store) == ["chunk_type"]

        with _spy_scalar_builds() as builds:
            store.ensure_scalar_indexes()

        assert builds == [("chunk_type", True)]
        assert _dangling_scalar(store) == []
        assert store.search([0.5] * cfg.embedding_dim, top_k=1, chunk_type=ChunkType.RAW)

    def test_one_dangling_index_among_healthy_ones_is_the_only_one_replaced(
        self, store, test_config
    ):
        """Only the BTree directory is gone; the FTS and Bitmap indexes keep serving."""
        store.add_chunks(_make_records())
        store.ensure_scalar_indexes()
        store.ensure_fts_index()
        _remove_index_files(test_config, index_types={"BTree"})
        assert _dangling_scalar(store) == ["source_id"]

        with _spy_scalar_builds() as builds:
            store.ensure_scalar_indexes()

        assert builds == [("source_id", True)]
        indices = _readable_registry(store.open_table("chunks"))
        assert _has_fts_index(indices)
        assert _dangling_scalar(store) == []
        assert store.bm25_probe("chunk number 1", chunk_type=ChunkType.RAW)
        hits = store.search([0.5] * cfg.embedding_dim, top_k=3, query_text="chunk number 1")
        assert hits

    def test_concepts_table_dangling_scalar_is_replaced(self, store, test_config):
        from lilbee.core.config import CHUNK_CONCEPTS_TABLE
        from lilbee.data.store.lance_helpers import ensure_table
        from lilbee.retrieval.concepts.schema import _chunk_concepts_schema

        store.add_chunks(_make_records())
        cc = ensure_table(store.get_db(), CHUNK_CONCEPTS_TABLE, _chunk_concepts_schema())
        cc.add([{"source_id": "doc0.md", "chunk_index": 0, "concept": "alpha"}])
        store.ensure_scalar_indexes()
        _remove_index_files(test_config, CHUNK_CONCEPTS_TABLE)
        assert _dangling_scalar(store, CHUNK_CONCEPTS_TABLE) == ["source_id"]

        store.ensure_scalar_indexes()

        assert _dangling_scalar(store, CHUNK_CONCEPTS_TABLE) == []

    def test_failed_hybrid_arm_drops_the_scalar_latch(self, store, test_config):
        """A hybrid failure re-verifies the scalar indexes on the next query, not on every one."""
        fresh = self._store_with_missing_scalar_files(store, test_config)
        fresh._scalar_ready = True
        fresh._fts_ready = True
        query = [0.5] * cfg.embedding_dim

        with mock.patch.object(Store, "_hybrid_search", side_effect=RuntimeError("io")):
            hits = fresh.search(query, top_k=1, query_text="chunk number 1")
        assert hits
        assert fresh._scalar_ready is False
        assert sorted(_dangling_scalar(fresh)) == ["chunk_type", "source_id"]

        fresh.search(query, top_k=1, query_text="chunk number 1")

        assert _dangling_scalar(fresh) == []

    def test_failed_vector_arm_drops_the_scalar_latch(self, store, test_config):
        """A vector-only failure surfaces and re-verifies the scalar indexes on the next query."""
        fresh = self._store_with_missing_scalar_files(store, test_config)
        fresh._scalar_ready = True

        with (
            mock.patch.object(Store, "_vector_arm", side_effect=RuntimeError("object not found")),
            pytest.raises(RuntimeError, match="object not found"),
        ):
            fresh.search([0.5] * cfg.embedding_dim, top_k=1)

        assert fresh._scalar_ready is False

    def test_dangling_probe_skips_index_without_uuid(self, store, test_config):
        """An index with no uuid has no directory to check, so it is not dangling."""
        from lilbee.data.store.lance_helpers import _dangling_indices

        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        assert table is not None
        no_uuid = mock.MagicMock(index_type="BTree", columns=["source_id"], index_uuid=None)
        assert _dangling_indices(table, [no_uuid], test_config.lancedb_dir) == []


def _index_uuids(store, table_name=CHUNKS_TABLE):
    """{index name: index_uuid} for every index on *table_name*."""
    return {idx.name: idx.index_uuid for idx in _readable_registry(store.open_table(table_name))}


class TestUnusableIndexRegistry:
    """A missing FTS directory makes the registry read as ``None``. The store
    rebuilds only the FTS indexes; scalar and vector indexes keep their files
    and are left alone."""

    def test_registry_is_empty_without_indexes_and_none_when_listing_raises(
        self, store, test_config
    ):
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        assert table is not None
        assert _index_registry(table, test_config.lancedb_dir) == []
        with mock.patch.object(type(table), "list_indices", side_effect=RuntimeError("boom")):
            assert _index_registry(table, test_config.lancedb_dir) is None

    def test_a_missing_fts_directory_rebuilds_only_the_fts_index(self, store, test_config):
        store.add_chunks(_make_records())
        store.ensure_scalar_indexes()
        store.ensure_fts_index()
        before = _index_uuids(store)
        _remove_index_files(test_config, index_types={"FTS"})
        assert _index_registry(store.open_table("chunks"), test_config.lancedb_dir) is None

        with _spy_scalar_builds() as builds:
            store.ensure_fts_index()

        assert builds == []
        after = _index_uuids(store)
        assert after["chunk_idx"] != before["chunk_idx"]
        assert after["source_id_idx"] == before["source_id_idx"]
        assert after["chunk_type_idx"] == before["chunk_type_idx"]
        assert store.bm25_probe("chunk number 1")

    def test_a_failed_title_rebuild_still_rebuilds_the_chunk_index(
        self, store, test_config, caplog
    ):
        import logging

        from lancedb.table import LanceTable

        test_config.title_search = True
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        _remove_index_files(test_config, index_types={"FTS"})
        fresh = Store(test_config)
        real = LanceTable.create_index

        def _fail_title(self, column, **kwargs):
            if column == "title":
                raise RuntimeError("no space")
            return real(self, column, **kwargs)

        with (
            mock.patch.object(LanceTable, "create_index", _fail_title),
            caplog.at_level(logging.WARNING),
        ):
            fresh.ensure_fts_index()

        assert "Could not rebuild the FTS index on 'chunks.title'" in caplog.text
        assert fresh._fts_ready is True
        assert fresh._title_fts_ready is False
        assert fresh.bm25_probe("plain body words")

    def test_a_listing_that_stays_unusable_warns_once_and_rebuilds_nothing_else(
        self, store, test_config, caplog
    ):
        """The FTS rebuild latches keyword readiness, so later queries do not
        retry it, and no scalar or vector index is rebuilt blind."""
        import logging

        from lancedb.table import LanceTable

        store.add_chunks(_make_records())
        store.ensure_scalar_indexes()
        store.ensure_fts_index()
        fresh = Store(test_config)
        real = LanceTable.create_index
        rebuilt = []

        def _record(self, column, **kwargs):
            rebuilt.append(column)
            return real(self, column, **kwargs)

        with (
            mock.patch.object(LanceTable, "list_indices", side_effect=RuntimeError("boom")),
            mock.patch.object(LanceTable, "create_index", _record),
            _spy_scalar_builds() as scalar_builds,
            caplog.at_level(logging.WARNING),
        ):
            fresh.ensure_fts_index()
            fresh.search([0.5] * cfg.embedding_dim, top_k=1, query_text="chunk number 1")
            fresh.search([0.5] * cfg.embedding_dim, top_k=1, query_text="chunk number 2")
            fresh.ensure_scalar_indexes()

        assert rebuilt == ["chunk"]
        assert scalar_builds == []
        assert caplog.text.count("still unusable after the FTS rebuild") == 1
        assert fresh._scalar_ready is False

    def test_scalar_build_skips_a_table_whose_listing_is_unusable(self, store):
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        with (
            mock.patch.object(type(table), "list_indices", side_effect=RuntimeError("boom")),
            _spy_scalar_builds() as builds,
        ):
            store._ensure_scalar_index_on("chunks", (("source_id", "BTREE"),))
        assert builds == []

    def test_title_index_build_repairs_an_unusable_registry(self, store, test_config):
        test_config.title_search = True
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        _remove_index_files(test_config)
        fresh = Store(test_config)

        fresh.ensure_title_fts_index()

        assert fresh._title_fts_ready is True
        indices = _readable_registry(fresh.open_table("chunks"))
        assert _has_fts_index(indices, "title")
        assert _has_fts_index(indices)


class TestEnsureVectorIndex:
    """Small vaults stay on exact flat search; large ones get an ANN index."""

    _INDEXABLE = 256  # enough rows for IVF_PQ to train under default params

    def test_noop_when_no_table(self, store):
        assert store.ensure_vector_index() is False

    def test_below_threshold_keeps_flat_search(self, store, test_config):

        store.add_chunks(_make_records())  # 3 rows, threshold defaults to 50_000
        assert store.ensure_vector_index() is False
        table = store.open_table("chunks")
        assert _has_vector_index(_readable_registry(table)) is False
        # Flat search still serves results without an ANN index.
        assert store.search([0.5] * test_config.embedding_dim, top_k=3)

    def test_threshold_zero_disables_build(self, store, test_config):

        test_config.ann_index_threshold = 0
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        assert store.ensure_vector_index() is False
        assert _has_vector_index(_readable_registry(store.open_table("chunks"))) is False

    def test_builds_index_above_threshold(self, store, test_config):
        import math

        test_config.ann_index_threshold = 50
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        assert store.ensure_vector_index() is True
        assert _has_vector_index(_readable_registry(store.open_table("chunks"))) is True
        # Search still finds the chunk whose vector matches the query (nprobes/refine).
        query = [math.sin(5 * 0.1 + j * 0.01) for j in range(test_config.embedding_dim)]
        results = store.search(query, top_k=3)
        assert results
        assert results[0].source == "doc5.md"

    def test_force_builds_below_threshold(self, store, test_config):

        test_config.ann_index_threshold = 1_000_000
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        assert store.ensure_vector_index() is False  # below threshold, no force
        assert store.ensure_vector_index(force=True) is True
        assert _has_vector_index(_readable_registry(store.open_table("chunks"))) is True

    def test_optimizes_when_index_exists(self, store, test_config):
        test_config.ann_index_threshold = 50
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        store.ensure_vector_index()
        table = store.open_table("chunks")
        with mock.patch.object(type(table), "optimize") as optimize_spy:
            assert store.ensure_vector_index() is True
        optimize_spy.assert_called_once()

    def test_optimize_that_drops_fts_files_repairs_fts_in_the_same_step(self, store, test_config):
        """The vector path optimizes the same chunks table every sync, so a prune
        that drops the FTS files there must repair them in the same step.

        The prune is simulated; the producer of the missing files is not confirmed."""
        test_config.ann_index_threshold = 50
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        store.ensure_fts_index()
        store.ensure_vector_index()
        table = store.open_table("chunks")
        assert table is not None

        with mock.patch.object(
            type(table),
            "optimize",
            side_effect=lambda: _remove_index_files(test_config, index_types={"FTS"}),
        ):
            assert store.ensure_vector_index() is True

        assert _has_fts_index(_readable_registry(store.open_table("chunks")))
        assert store.bm25_probe("chunk")

    def test_a_missing_fts_directory_leaves_the_vector_index_alone(self, store, test_config):
        """IVF_PQ retrains k-means on a rebuild, so only the FTS index is rebuilt."""
        import math

        test_config.ann_index_threshold = 50
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        store.ensure_fts_index()
        store.ensure_vector_index()
        vector_uuid = _index_uuids(store)["vector_idx"]
        _remove_index_files(test_config, index_types={"FTS"})
        fresh = Store(test_config)

        with mock.patch.object(Store, "_create_vector_index") as vector_build:
            assert fresh.ensure_vector_index() is True
            fresh.ensure_fts_index()

        vector_build.assert_not_called()
        assert _index_uuids(fresh)["vector_idx"] == vector_uuid
        assert fresh.bm25_probe("chunk")
        query = [math.sin(5 * 0.1 + j * 0.01) for j in range(test_config.embedding_dim)]
        assert fresh.search(query, top_k=3)[0].source == "doc5.md"

    def test_a_missing_vector_directory_above_threshold_is_rebuilt(self, store, test_config):
        """lancedb lists an IVF_PQ index whose directory is gone without error, so
        the directory check finds it and the index is rebuilt instead of optimized."""
        import math

        from lilbee.data.store.lance_helpers import _vector_index_dangling

        test_config.ann_index_threshold = 50
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        store.ensure_vector_index()
        _remove_index_files(test_config, index_types={"IvfPq"})
        table = store.open_table("chunks")
        assert _vector_index_dangling(table, _readable_registry(table), test_config.lancedb_dir)

        with mock.patch.object(type(table), "optimize") as optimize_spy:
            assert store.ensure_vector_index() is True

        optimize_spy.assert_not_called()
        table = store.open_table("chunks")
        assert not _vector_index_dangling(table, _readable_registry(table), test_config.lancedb_dir)
        query = [math.sin(5 * 0.1 + j * 0.01) for j in range(test_config.embedding_dim)]
        assert store.search(query, top_k=3)[0].source == "doc5.md"

    def test_a_failed_vector_rebuild_warns_and_returns_false(self, store, test_config, caplog):
        test_config.ann_index_threshold = 50
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        store.ensure_vector_index()
        _remove_index_files(test_config, index_types={"IvfPq"})
        table = store.open_table("chunks")

        with (
            mock.patch.object(type(table), "create_index", side_effect=RuntimeError("no space")),
            caplog.at_level("WARNING"),
        ):
            assert store.ensure_vector_index() is False

        assert "Could not rebuild the vector index" in caplog.text

    def test_unusable_registry_below_threshold_builds_no_vector_index(self, store, test_config):
        store.add_chunks(_make_records())  # 3 rows, threshold defaults to 50_000
        store.ensure_fts_index()
        _remove_index_files(test_config)

        assert store.ensure_vector_index() is False

        indices = _readable_registry(store.open_table("chunks"))
        assert _has_fts_index(indices)
        assert not _has_vector_index(indices)

    def test_forced_build_on_an_unusable_registry_repairs_then_builds(self, store, test_config):
        test_config.ann_index_threshold = 1_000_000
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        store.ensure_fts_index()
        _remove_index_files(test_config)

        assert store.ensure_vector_index(force=True) is True

        indices = _readable_registry(store.open_table("chunks"))
        assert _has_fts_index(indices)
        assert _has_vector_index(indices)

    def test_registry_that_stays_unusable_builds_nothing(self, store):
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        with (
            mock.patch.object(type(table), "list_indices", side_effect=RuntimeError("boom")),
            mock.patch.object(type(table), "create_index") as create,
        ):
            assert store.ensure_vector_index(force=True) is False
        assert [c.args[0] for c in create.call_args_list] == ["chunk"]

    def test_build_failure_warns_and_returns_false(self, store, test_config, caplog):
        """bb-con: a real ANN build failure at scale is surfaced as a warning with
        the flat-search impact, not swallowed at debug."""
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        with (
            mock.patch.object(type(table), "create_index", side_effect=RuntimeError("boom")),
            caplog.at_level("WARNING"),
        ):
            assert store.ensure_vector_index(force=True) is False
        assert "ANN index build failed" in caplog.text
        assert "flat scan" in caplog.text


class TestHasFtsIndex:
    def test_returns_false_on_fresh_table(self, store):
        store.add_chunks(_make_records())

        table = store.open_table("chunks")
        assert table is not None
        assert _has_fts_index(_readable_registry(table)) is False

    def test_returns_true_after_create(self, store):
        store.add_chunks(_make_records())
        store.ensure_fts_index()

        table = store.open_table("chunks")
        assert table is not None
        assert _has_fts_index(_readable_registry(table)) is True


class TestFtsIndexStaleFlag:
    def test_add_chunks_marks_stale(self, store):
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        assert store._fts_ready
        store.add_chunks(_make_records(1))
        assert not store._fts_ready

    def test_drop_all_marks_stale(self, store):
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        assert store._fts_ready
        store.drop_all()
        assert not store._fts_ready


def _records_for(source, n=2, dim=None):
    if dim is None:
        dim = cfg.embedding_dim
    return [
        {
            "source": source,
            "content_type": "text",
            "chunk_type": "raw",
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": f"{source} chunk {i}",
            "chunk_index": i,
            "vector": [float(i)] * dim,
        }
        for i in range(n)
    ]


class TestWriteChunksBatch:
    def test_writes_many_docs_and_upserts_sources_in_one_pass(self, store):
        from lilbee.data.store import ChunkWrite

        items = [
            ChunkWrite("a.md", "hash_a", _records_for("a.md", 2), needs_cleanup=False),
            ChunkWrite("b.md", "hash_b", _records_for("b.md", 3), needs_cleanup=False),
        ]
        added = store.write_chunks_batch(items)
        assert added == 5
        assert len(store.get_chunks_by_source("a.md")) == 2
        assert len(store.get_chunks_by_source("b.md")) == 3
        sources = {s["filename"]: s for s in store.get_sources()}
        assert sources["a.md"]["chunk_count"] == 2
        assert sources["b.md"]["file_hash"] == "hash_b"

    def test_cleanup_replaces_a_source_without_duplicating(self, store):
        from lilbee.data.store import ChunkWrite

        store.add_chunks(_records_for("a.md", 4))
        # Re-ingest a.md with fewer chunks; needs_cleanup must drop the old ones.
        store.write_chunks_batch(
            [ChunkWrite("a.md", "h2", _records_for("a.md", 2), needs_cleanup=True)]
        )
        assert len(store.get_chunks_by_source("a.md")) == 2

    def test_empty_batch_is_noop(self, store):
        assert store.write_chunks_batch([]) == 0
        assert store.get_sources() == []

    @staticmethod
    def _flush(store, name, chunks=2, page_texts=None):
        from lilbee.data.store import ChunkWrite

        item = ChunkWrite(
            name, f"hash of {name}", _records_for(name, chunks), True, page_texts=page_texts
        )
        store.write_chunks_batch([item])

    def test_a_flush_indexes_a_key_column_once_enough_rows_sit_past_the_index(
        self, store, monkeypatch
    ):
        monkeypatch.setattr("lilbee.data.store.core._KEY_INDEX_STALE_FLUSHES", 2)
        self._flush(store, "a.md")
        self._flush(store, "b.md", page_texts=_page_rows("b.md"))
        sources, chunks = store.open_table(SOURCES_TABLE), store.open_table(CHUNKS_TABLE)
        assert sources.index_stats("path_idx") is None
        assert chunks.index_stats("source_id_idx") is None

        self._flush(store, "c.md")

        assert store.open_table(SOURCES_TABLE).index_stats("path_idx").num_indexed_rows == 2
        assert store.open_table(CHUNKS_TABLE).index_stats("source_id_idx").num_indexed_rows == 4
        # One flush of page texts is under the bound of two.
        assert store.open_table(PAGE_TEXTS_TABLE).index_stats("source_id_idx") is None

    def test_an_index_is_rebuilt_only_after_enough_new_rows(self, store, monkeypatch):
        monkeypatch.setattr("lilbee.data.store.core._KEY_INDEX_STALE_FLUSHES", 2)
        for name in ("a.md", "b.md", "c.md", "d.md"):
            self._flush(store, name)
        # Built by the third flush over two rows; the fourth finds one row past it.
        assert store.open_table(SOURCES_TABLE).index_stats("path_idx").num_indexed_rows == 2

        self._flush(store, "e.md")

        assert store.open_table(SOURCES_TABLE).index_stats("path_idx").num_indexed_rows == 4

    def test_a_flush_replaces_rows_the_index_covers_and_rows_past_it(self, store, monkeypatch):
        monkeypatch.setattr("lilbee.data.store.core._KEY_INDEX_STALE_FLUSHES", 2)
        for name in ("a.md", "b.md", "c.md", "d.md"):
            self._flush(store, name, chunks=3)
        stats = store.open_table(CHUNKS_TABLE).index_stats("source_id_idx")
        assert (stats.num_indexed_rows, stats.num_unindexed_rows) == (6, 6)

        self._flush(store, "a.md", chunks=1)
        self._flush(store, "d.md", chunks=1)

        names = ("a.md", "b.md", "c.md", "d.md")
        assert [len(store.get_chunks_by_source(name)) for name in names] == [1, 3, 3, 1]
        rows = sorted((s["filename"], s["chunk_count"]) for s in store.get_sources())
        assert rows == [("a.md", 1), ("b.md", 3), ("c.md", 3), ("d.md", 1)]

    def test_a_failed_index_refresh_leaves_the_flush_correct(self, store, monkeypatch):
        import lancedb.table

        def _refuse(self, name):
            raise OSError("index files are gone")

        monkeypatch.setattr(lancedb.table.LanceTable, "index_stats", _refuse)
        self._flush(store, "a.md", chunks=3)
        self._flush(store, "a.md", chunks=1)

        assert len(store.get_chunks_by_source("a.md")) == 1
        assert [s["chunk_count"] for s in store.get_sources()] == [1]

    def test_a_refresh_that_keeps_failing_warns_once_for_each_table(
        self, store, monkeypatch, caplog
    ):
        import lancedb.table

        def _refuse(self, name):
            raise OSError("index files are gone")

        def _warned():
            return sorted(
                record.getMessage()
                for record in caplog.records
                if record.levelno == logging.WARNING and "Key index refresh" in record.getMessage()
            )

        self._flush(store, "a.md", chunks=2)
        monkeypatch.setattr(lancedb.table.LanceTable, "index_stats", _refuse)
        with caplog.at_level(logging.DEBUG, logger="lilbee.data.store.core"):
            self._flush(store, "b.md", chunks=2)
            once = _warned()
            self._flush(store, "c.md", chunks=2)
            self._flush(store, "d.md", chunks=2)
        scans = "each write scans the table until it succeeds"
        assert once == [
            f"Key index refresh failed on '_sources.path'; {scans}",
            f"Key index refresh failed on 'chunks.source_id'; {scans}",
        ]
        assert _warned() == once
        # The later failures are still logged, below the level a user sees.
        # Three key columns, three flushes; one warning a table, the rest at debug.
        assert sum("Key index refresh" in record.getMessage() for record in caplog.records) == 9

    def test_a_refresh_that_fails_on_an_empty_table_is_not_a_warning(
        self, store, monkeypatch, caplog
    ):
        import lancedb.table

        def _refuse(self, name):
            raise OSError("index files are gone")

        self._flush(store, "a.md", chunks=2)
        store.remove_documents(["a.md"])
        monkeypatch.setattr(lancedb.table.LanceTable, "index_stats", _refuse)
        with caplog.at_level(logging.DEBUG, logger="lilbee.data.store.core"):
            self._flush(store, "b.md", chunks=2)
        levels = {record.levelno for record in caplog.records if "Key index" in record.getMessage()}
        assert levels == {logging.DEBUG}

    def test_a_table_that_cannot_be_counted_is_taken_to_hold_rows(self):
        from lilbee.data.store.core import _holds_rows

        table = mock.MagicMock()
        table.count_rows.side_effect = OSError("cannot count")
        assert _holds_rows(table) is True
        table.count_rows.side_effect = None
        table.count_rows.return_value = 0
        assert _holds_rows(table) is False

    def test_batch_uses_the_patient_lock_timeout(self, store):
        """The flush lock waits BATCH_LOCK_TIMEOUT, not the interactive 30s.

        Failing the flush replans and re-embeds the whole batch, so the batch
        path outwaits a search-triggered FTS optimize instead of giving up.
        """
        from lilbee.data.store import ChunkWrite
        from lilbee.data.store.core import BATCH_LOCK_TIMEOUT

        seen: list[float] = []

        @contextmanager
        def _capture(_dir, timeout):
            seen.append(timeout)
            yield

        with mock.patch("lilbee.data.store.core.write_lock", _capture):
            store.write_chunks_batch(
                [ChunkWrite("a.md", "h", _records_for("a.md", 1), needs_cleanup=False)]
            )
        assert seen == [BATCH_LOCK_TIMEOUT]

    def test_replace_source_skips_add_on_swallowed_delete(self, store, monkeypatch):
        # A swallowed delete must not leave two _sources rows for one filename:
        # the replace skips the add and the file replans next sync.
        store.upsert_source("a.md", "h1", chunk_count=2)
        assert len([s for s in store.get_sources() if s["filename"] == "a.md"]) == 1
        monkeypatch.setattr("lilbee.data.store.core._safe_delete_unlocked", lambda *a, **k: False)
        store.upsert_source("a.md", "h2", chunk_count=9)
        rows = [s for s in store.get_sources() if s["filename"] == "a.md"]
        assert len(rows) == 1

    def test_zero_chunk_item_persists_page_texts_and_source_row(self, store):
        # A processed file with no chunkable text (whitespace-only OCR) keeps
        # its pages and source row so it stops replanning every sync.
        from lilbee.data.store import ChunkWrite

        page = {"source": "scan.pdf", "page": 1, "text": "  ", "content_type": "pdf"}
        items = [ChunkWrite("scan.pdf", "h", [], needs_cleanup=True, page_texts=[page])]
        assert store.write_chunks_batch(items) == 0
        sources = {s["filename"]: s for s in store.get_sources()}
        assert sources["scan.pdf"]["chunk_count"] == 0
        assert sources["scan.pdf"]["file_hash"] == "h"
        assert [row["page"] for row in store.get_page_texts("scan.pdf")] == [1]
        # Read paths stay healthy with a zero-chunk source present.
        assert store.get_chunks_by_source("scan.pdf") == []
        assert store.search([0.0] * cfg.embedding_dim) == []

    def test_cleanup_deletes_are_constant_per_flush(self, store):
        # N cleanup items go through one batched IN-delete pass, never one
        # delete set per item.
        import lilbee.data.store.core as core_mod
        from lilbee.data.store import ChunkWrite

        total = 4
        for i in range(total):
            store.add_chunks(_records_for(f"f{i}.md", 1))
        store.add_page_texts(
            [
                {"source": f"f{i}.md", "page": 1, "text": "old", "content_type": "pdf"}
                for i in range(total)
            ]
        )
        items = [
            ChunkWrite(
                f"f{i}.md",
                f"h{i}",
                _records_for(f"f{i}.md", 2),
                needs_cleanup=True,
                page_texts=[
                    {"source": f"f{i}.md", "page": 1, "text": "new", "content_type": "pdf"}
                ],
            )
            for i in range(total)
        ]
        with mock.patch.object(
            core_mod.Store,
            "_delete_rows_of_unlocked",
            autospec=True,
            side_effect=core_mod.Store._delete_rows_of_unlocked,
        ) as spy:
            store.write_chunks_batch(items)
        assert spy.call_count == 1
        ids = store.source_ids((f"f{i}.md" for i in range(total)), SourceType.DOCUMENT)
        assert sorted(spy.call_args.args[1]) == sorted(ids.values())
        # End state matches the per-item behavior: replaced rows, no orphans.
        for i in range(total):
            assert len(store.get_chunks_by_source(f"f{i}.md")) == 2
            assert [row["text"] for row in store.get_page_texts(f"f{i}.md")] == ["new"]

    def test_cleanup_delete_failure_propagates(self, store):
        # A swallowed delete would leave every flushed file silently stale, so
        # the flush must fail and let the files replan instead.
        import lilbee.data.store.core as core_mod
        from lilbee.data.store import ChunkWrite

        store.add_chunks(_records_for("a.md", 1))
        real_open_table = store.open_table
        broken = mock.MagicMock()
        broken.delete.side_effect = RuntimeError("commit conflict")

        def _broken_chunks_table(name):
            if name == core_mod.CHUNKS_TABLE:
                return broken
            return real_open_table(name)

        with (
            mock.patch.object(store, "open_table", side_effect=_broken_chunks_table),
            pytest.raises(RuntimeError, match="commit conflict"),
        ):
            store.write_chunks_batch(
                [ChunkWrite("a.md", "h2", _records_for("a.md", 1), needs_cleanup=True)]
            )

    def test_quoted_filename_survives_the_batched_cleanup(self, store):
        from lilbee.data.store import ChunkWrite

        name = "it's a note.md"
        store.add_chunks(_records_for(name, 1))
        store.write_chunks_batch(
            [ChunkWrite(name, "h2", _records_for(name, 2), needs_cleanup=True)]
        )
        assert len(store.get_chunks_by_source(name)) == 2

    def test_page_texts_land_in_the_batch_and_survive_cleanup(self, store):
        from lilbee.data.store import ChunkWrite

        page = {"source": "a.pdf", "page": 1, "text": "page one", "content_type": "pdf"}
        store.write_chunks_batch(
            [
                ChunkWrite(
                    "a.pdf", "h1", _records_for("a.pdf", 1), needs_cleanup=True, page_texts=[page]
                )
            ]
        )
        assert [row["text"] for row in store.get_page_texts("a.pdf")] == ["page one"]

        # Re-ingest: the same transaction's cleanup delete clears the old page
        # rows, then the fresh ones land, so nothing the batch wrote is wiped.
        edited = {**page, "text": "page one, edited"}
        store.write_chunks_batch(
            [
                ChunkWrite(
                    "a.pdf", "h2", _records_for("a.pdf", 1), needs_cleanup=True, page_texts=[edited]
                )
            ]
        )
        assert [row["text"] for row in store.get_page_texts("a.pdf")] == ["page one, edited"]

    def test_dimension_mismatch_rejects_whole_batch(self, store):
        from lilbee.data.store import ChunkWrite

        bad = _records_for("a.md", 1, dim=cfg.embedding_dim + 1)
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.write_chunks_batch([ChunkWrite("a.md", "h", bad, needs_cleanup=False)])
        assert store.get_sources() == []


class TestHybridSearch:
    def test_hybrid_search_with_fts_index(self, store, test_config):
        records = _make_records()
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=3, query_text="chunk number")
        assert len(results) > 0
        assert all(r.score is not None for r in results)
        scores = [r.score for r in results]
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_adaptive_fusion_feeds_a_derived_weight_to_fusion(self, store, test_config):
        """With adaptive_fusion on, the per-query factor from adaptive_weight_scale
        -- fed the configured margin -- scales the lexical weight reaching
        fuse_arms, not the fixed config value. Deleting the adaptive branch would
        fail this, unlike a smoke test on the score range."""
        test_config.adaptive_fusion = True
        test_config.adaptive_fusion_margin = 0.42
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        from lilbee.data.store import core as store_core

        with (
            mock.patch.object(store_core, "adaptive_weight_scale", return_value=0.5) as scale,
            mock.patch.object(store_core, "fuse_arms", wraps=store_core.fuse_arms) as fuse,
        ):
            results = store.search(query_vec, top_k=3, query_text="chunk number")
        scale.assert_called_once()
        # adaptive_weight_scale(vector_rows, margin): the margin is the config value.
        assert scale.call_args.args[1] == pytest.approx(0.42)
        assert fuse.call_args.kwargs["lexical_weight"] == pytest.approx(
            test_config.lexical_fusion_weight * 0.5
        )
        # fuse_arms normalizes by the effective arms itself; no fixed budget
        # is threaded through (a shared denominator capped confident queries).
        assert "weight_total" not in fuse.call_args.kwargs
        assert len(results) > 0

    def test_fixed_fusion_pins_the_config_weight(self, store, test_config):
        """Opting out of adaptive fusion skips adaptive_weight_scale and pins
        the fixed lexical_fusion_weight into fuse_arms; a non-default weight must
        reach fusion verbatim."""
        test_config.adaptive_fusion = False
        test_config.lexical_fusion_weight = 0.3
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        from lilbee.data.store import core as store_core

        with (
            mock.patch.object(store_core, "adaptive_weight_scale") as adapt,
            mock.patch.object(store_core, "fuse_arms", wraps=store_core.fuse_arms) as fuse,
        ):
            results = store.search(query_vec, top_k=3, query_text="chunk number")
        adapt.assert_not_called()
        assert fuse.call_args.kwargs["lexical_weight"] == pytest.approx(0.3)
        assert len(results) > 0

    def test_fallback_to_vector_when_no_query_text(self, store, test_config):
        records = _make_records()
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=3)
        assert len(results) > 0
        assert results[0].distance is not None

    def test_fallback_to_vector_when_no_fts_index(self, store, test_config):
        records = _make_records()
        store.add_chunks(records)
        with mock.patch.object(store, "ensure_fts_index"):
            query_vec = [0.5] * test_config.embedding_dim
            results = store.search(query_vec, top_k=3, query_text="chunk")
        assert len(results) > 0
        assert results[0].distance is not None

    def test_hybrid_fallback_on_exception(self, store, test_config, caplog):
        records = _make_records()
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        with (
            mock.patch.object(store, "_hybrid_search", side_effect=RuntimeError("boom")),
            caplog.at_level("WARNING", logger="lilbee.data.store.core"),
        ):
            results = store.search(query_vec, top_k=3, query_text="chunk")
        assert len(results) > 0
        assert results[0].distance is not None
        # The downgrade changes recall for the query; it must be visible.
        assert any("falling back to vector-only" in r.message for r in caplog.records)

    def test_vector_only_applies_mmr(self, store, test_config):
        """Vector-only path (no query_text) applies MMR when results > top_k."""
        records = _make_records(n=6)
        store.add_chunks(records)
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=2)
        assert len(results) == 2

    def test_vector_only_results_carry_canonical_score(self, store, test_config):
        records = _make_records()
        store.add_chunks(records)
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=3)
        assert all(r.score is not None for r in results)
        # Higher similarity (lower distance) must mean higher canonical score.
        pairs = [(r.distance, r.score) for r in results]
        assert sorted(pairs, key=lambda p: p[0]) == sorted(pairs, key=lambda p: p[1], reverse=True)

    def test_lexical_only_match_survives_hybrid(self, store, test_config):
        """A chunk only BM25 can find (its vector points away from the query)
        must appear in hybrid results: the identifier-query case."""
        records = _make_records(n=30)
        outlier_vec = [-1.0] * test_config.embedding_dim
        records.append(
            {
                "source": "parts_catalog_214.pdf",
                "content_type": "pdf",
                "chunk_type": "raw",
                "page_start": 1,
                "page_end": 1,
                "line_start": 0,
                "line_end": 0,
                "chunk": "part number PX4471 shipped from fresno",
                "chunk_index": 0,
                "vector": outlier_vec,
            }
        )
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=5, query_text="part number PX4471")
        assert any(r.source == "parts_catalog_214.pdf" for r in results)

    def test_hybrid_arms_fetch_exactly_top_k(self, store, test_config):
        """Deeper pools flood rank fusion with both-arm mediocre rows, so
        each arm is fetched exactly top_k deep."""
        records = _make_records(n=3)
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        with (
            mock.patch.object(store, "_vector_arm", return_value=[]) as vec_arm,
            mock.patch.object(store, "_fts_arm", return_value=[]) as fts_arm,
        ):
            store.search(query_vec, top_k=3, query_text="chunk")
        assert vec_arm.call_args[0][2] == 3
        assert fts_arm.call_args[0][2] == 3

    def test_hybrid_vector_arm_applies_ann_recovery(self, store, test_config):
        """With a vector index present, the hybrid vector arm probes extra
        partitions and refines against full vectors."""
        chain = mock.MagicMock()
        chain.to_list.return_value = []
        table = mock.MagicMock()
        table.search.return_value = chain
        table.count_rows.return_value = 1_000_000
        chain.distance_type.return_value = chain
        chain.limit.return_value = chain
        chain.nprobes.return_value = chain
        chain.refine_factor.return_value = chain
        with mock.patch("lilbee.data.store.core._has_vector_index", return_value=True):
            store._vector_arm(table, [0.5] * test_config.embedding_dim, 50, None)
        chain.nprobes.assert_called_once()
        chain.refine_factor.assert_called_once()

    def test_auto_ensures_fts_index_when_query_text(self, store, test_config):
        records = _make_records()
        store.add_chunks(records)
        assert not store._fts_ready
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=3, query_text="chunk number")
        assert store._fts_ready
        assert len(results) > 0


class TestChunkTypeFilter:
    def test_vector_search_filters_by_chunk_type(self, store, test_config):
        """Vector-only search with chunk_type filters results."""
        store.add_chunks(_make_records(n=2, chunk_type="raw"))
        store.add_chunks(
            [
                {
                    "source": "wiki/summaries/doc0.md",
                    "content_type": "text",
                    "chunk_type": "wiki",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "wiki summary text",
                    "chunk_index": 0,
                    "vector": [0.5] * test_config.embedding_dim,
                }
            ]
        )
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=5, chunk_type="wiki")
        assert all(r.chunk_type == "wiki" for r in results)
        assert len(results) == 1

    def test_vector_search_debug_logs_real_distance(self, store, test_config, caplog):
        """The debug log reads LanceDB's '_distance' column, not a missing 'distance'.

        With an orthogonal query the top distance is well above 0, so a wrong key
        (defaulting to 0) would be visible in the logged values.
        """
        dim = test_config.embedding_dim
        store.add_chunks(
            [
                {
                    "source": "doc0.md",
                    "content_type": "text",
                    "chunk_type": "raw",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "some text",
                    "chunk_index": 0,
                    "vector": [1.0] + [0.0] * (dim - 1),
                }
            ]
        )
        query_vec = [0.0] * (dim - 1) + [1.0]
        with caplog.at_level("DEBUG"):
            store.search(query_vec, top_k=5, max_distance=0)
        distance_logs = [
            r.getMessage() for r in caplog.records if "Top 5 distances" in r.getMessage()
        ]
        assert distance_logs
        assert "0.0" not in distance_logs[0].replace("Top 5 distances: ", "")

    def test_hybrid_search_filters_by_chunk_type(self, store, test_config):
        """Hybrid search with chunk_type filters results."""
        store.add_chunks(_make_records(n=2, chunk_type="raw"))
        store.add_chunks(
            [
                {
                    "source": "wiki/summaries/doc0.md",
                    "content_type": "text",
                    "chunk_type": "wiki",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "chunk number 5 with wiki text",
                    "chunk_index": 0,
                    "vector": [0.5] * test_config.embedding_dim,
                }
            ]
        )
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=5, query_text="chunk", chunk_type="wiki")
        assert all(r.chunk_type == "wiki" for r in results)


class TestMMRRerank:
    def test_selects_diverse_results(self):
        # Two results along x-axis (near-identical), one along y-axis (diverse but relevant)
        query = [0.8, 0.6]
        results = [
            SearchChunk(
                source="a.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="x-axis 1",
                chunk_index=0,
                vector=[1.0, 0.0],
                distance=0.2,
            ),
            SearchChunk(
                source="a.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="x-axis 2",
                chunk_index=1,
                vector=[1.0, 0.0],
                distance=0.2,
            ),
            SearchChunk(
                source="b.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="y-axis",
                chunk_index=0,
                vector=[0.0, 1.0],
                distance=0.4,
            ),
        ]
        selected = mmr_rerank(query, results, top_k=2, mmr_lambda=0.5)
        assert len(selected) == 2
        assert selected[0].chunk == "x-axis 1"
        assert selected[1].chunk == "y-axis"

    def test_returns_all_when_fewer_than_k(self):
        query = [1.0, 0.0]
        results = [
            SearchChunk(
                source="a.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="only one",
                chunk_index=0,
                vector=[0.9, 0.1],
                distance=0.1,
            ),
        ]
        selected = mmr_rerank(query, results, top_k=5)
        assert len(selected) == 1

    def test_wiki_paraphrase_near_duplicate_diversified_out(self):
        """Wiki paraphrase lands near its raw source in vector space; MMR
        drops one of them at ``top_k=2`` in favour of a clearly different
        candidate. Whichever side wins is acceptable (always a choice:
        neither is preferred by default) so diversity should simply
        do its job on near-duplicates regardless of ``chunk_type``.

        The query is at ``[1, 1]/√2`` so both the near-dup cluster on
        the x-axis and the orthogonal y-axis chunk are equally relevant;
        the diversity penalty is the only tiebreaker, which is what
        MMR is supposed to give.
        """
        query = [0.707, 0.707]
        wiki_chunk = SearchChunk(
            source="wiki/summaries/doc.md",
            content_type="text/markdown",
            chunk_type="wiki",
            page_start=1,
            page_end=1,
            line_start=1,
            line_end=1,
            chunk="Wiki paraphrase of doc content.",
            chunk_index=0,
            vector=[1.0, 0.0],
            distance=0.30,
        )
        raw_duplicate = SearchChunk(
            source="doc.md",
            content_type="text",
            chunk_type="raw",
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk="Raw doc content the wiki paraphrased.",
            chunk_index=0,
            vector=[1.0, 0.01],
            distance=0.30,
        )
        distinct = SearchChunk(
            source="other.md",
            content_type="text",
            chunk_type="raw",
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk="Orthogonal topic.",
            chunk_index=0,
            vector=[0.0, 1.0],
            distance=0.30,
        )
        selected = mmr_rerank(query, [wiki_chunk, raw_duplicate, distinct], top_k=2, mmr_lambda=0.5)
        sources = {r.source for r in selected}
        # The orthogonal chunk must appear. That's the diversity win.
        assert "other.md" in sources
        # Only one of the two near-duplicates survives.
        dup_count = sum(1 for r in selected if r.source in ("wiki/summaries/doc.md", "doc.md"))
        assert dup_count == 1

    def testcosine_sim_zero_vectors(self):
        assert cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0

    def testcosine_sim_identical(self):
        sim = cosine_sim([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6


class TestAdaptiveFilter:
    def test_returns_results_within_threshold(self, store):
        results = [
            SearchChunk(
                source="a.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="close",
                chunk_index=0,
                vector=[0.1],
                distance=0.2,
            ),
            SearchChunk(
                source="b.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="far",
                chunk_index=0,
                vector=[0.1],
                distance=0.8,
            ),
        ]
        filtered = store._adaptive_filter(results, top_k=1, initial_threshold=0.3)
        assert len(filtered) == 1
        assert filtered[0].chunk == "close"

    def test_widens_threshold_when_too_few(self, store):
        results = [
            SearchChunk(
                source="a.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="far",
                chunk_index=0,
                vector=[0.1],
                distance=0.6,
            ),
        ]
        filtered = store._adaptive_filter(results, top_k=1, initial_threshold=0.3)
        assert len(filtered) == 1

    def test_stops_at_max_threshold(self, store):
        results = [
            SearchChunk(
                source="a.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="very far",
                chunk_index=0,
                vector=[0.1],
                distance=1.5,
            ),
        ]
        filtered = store._adaptive_filter(results, top_k=1, initial_threshold=0.3)
        assert len(filtered) == 0


class TestRemoveDocuments:
    def test_removes_known_files(self, store):
        with (
            mock.patch.object(store, "get_sources", return_value=[{"filename": "a.md"}]),
            mock.patch.object(store, "_remove_many_unlocked") as mock_del,
        ):
            result = store.remove_documents(["a.md"])
            assert result.removed == ["a.md"]
            assert result.not_found == []
            mock_del.assert_called_once_with(["a.md"], None)

    def test_removes_an_archive_with_its_members(self, store):
        rows = [{"filename": n} for n in ("docs.zip", "docs.zip/a.pdf", "docs.zip.bak", "other.md")]
        with (
            mock.patch.object(store, "get_sources", return_value=rows),
            mock.patch.object(store, "_remove_many_unlocked") as mock_del,
        ):
            result = store.remove_documents(["docs.zip"])
            assert result.removed == ["docs.zip", "docs.zip/a.pdf"]
            assert result.not_found == []
            mock_del.assert_called_once_with(["docs.zip", "docs.zip/a.pdf"], None)

    def test_removing_many_names_reads_the_source_table_once(self, store):
        names = [f"n{index}.md" for index in range(500)]
        rows = [{"filename": n} for n in (*names, "n7.md/inside.txt", "a/b.zip", "a/b.zip/c/d.txt")]
        with (
            mock.patch.object(store, "get_sources", return_value=rows) as reads,
            mock.patch.object(store, "_remove_many_unlocked") as mock_del,
        ):
            result = store.remove_documents([*names, "a/b.zip"])
        assert reads.call_count == 1
        assert result.removed == [*names, "a/b.zip", "n7.md/inside.txt", "a/b.zip/c/d.txt"]
        mock_del.assert_called_once_with(result.removed, None)

    def test_not_found(self, store):
        with mock.patch.object(store, "get_sources", return_value=[]):
            result = store.remove_documents(["missing.md"])
            assert result.removed == []
            assert result.not_found == ["missing.md"]

    def test_never_deletes_physical_file(self, store, tmp_path):
        # Store removal is index-only; the source file on disk is never touched.
        with (
            mock.patch.object(store, "get_sources", return_value=[{"filename": "a.md"}]),
            mock.patch.object(store, "_remove_many_unlocked"),
        ):
            f = tmp_path / "a.md"
            f.write_text("content")
            result = store.remove_documents(["a.md"])
            assert result.removed == ["a.md"]
            assert f.exists()

    def test_nonexistent_file_still_removes_from_store(self, store):
        with (
            mock.patch.object(store, "get_sources", return_value=[{"filename": "gone.md"}]),
            mock.patch.object(store, "_remove_many_unlocked") as mock_del,
        ):
            result = store.remove_documents(["gone.md"])
            assert result.removed == ["gone.md"]
            mock_del.assert_called_once()

    def test_chunk_and_source_deleted_under_single_lock(self, store):
        """Both deletes for one document run inside one write_lock acquisition.

        Guards against the inconsistency window where chunks were visible
        after their source record had already been deleted.
        """
        acquisitions: list[str] = []

        @contextmanager
        def _tracking_lock(*args, **kwargs):
            acquisitions.append("acquire")
            yield

        with (
            mock.patch.object(
                store, "get_sources", return_value=[{"filename": "a.md"}, {"filename": "b.md"}]
            ),
            mock.patch.object(store, "source_ids", return_value={"a.md": "ida", "b.md": "idb"}),
            mock.patch.object(store, "_delete_rows_of_unlocked") as mock_chunks,
            mock.patch.object(store, "_drop_source_rows_unlocked") as mock_sources,
            mock.patch("lilbee.data.store.core.write_lock", _tracking_lock),
        ):
            result = store.remove_documents(["a.md", "b.md"])

        assert result.removed == ["a.md", "b.md"]
        mock_chunks.assert_called_once_with(["ida", "idb"])
        mock_sources.assert_called_once_with(["ida", "idb"])
        # Exactly one lock acquisition covers every delete for the whole set.
        assert acquisitions == ["acquire"]

    def test_removes_chunks_and_source_atomically(self, store):
        """End-to-end: a real removal clears both the chunk and source rows."""
        store.add_chunks(_make_records(n=2))
        store.upsert_source("doc0.md", "hash0", chunk_count=1)
        store.upsert_source("doc1.md", "hash1", chunk_count=1)

        result = store.remove_documents(["doc0.md"])

        assert result.removed == ["doc0.md"]
        assert {s["filename"] for s in store.get_sources()} == {"doc1.md"}
        assert store.get_chunks_by_source("doc0.md") == []
        assert len(store.get_chunks_by_source("doc1.md")) == 1


class TestBm25Probe:
    def test_returns_results_when_fts_ready(self, store):
        records = _make_records(n=3)
        store.add_chunks(records)
        store.ensure_fts_index()
        results = store.bm25_probe("chunk number", top_k=3)
        assert len(results) > 0

    def test_returns_empty_when_no_fts(self, store):
        records = _make_records(n=1)
        store.add_chunks(records)
        store._fts_ready = False
        results = store.bm25_probe("anything")
        assert isinstance(results, list)

    def test_returns_empty_when_no_table(self, store):
        results = store.bm25_probe("anything")
        assert results == []


class TestClearTable:
    def testclear_table_deletes_matching_rows(self, store):
        records = _make_records(n=1)
        store.add_chunks(records)
        assert store.clear_table("chunks", "chunk_index = 0") is True
        table = store.open_table("chunks")
        remaining = table.to_arrow()
        assert len(remaining) == 0

    def testclear_table_nonexistent_table_is_noop(self, store):
        assert store.clear_table("nonexistent", "source = 'doc0.md'") is True

    def testclear_table_reports_a_swallowed_delete_failure(self, store):
        store.add_chunks(_make_records(n=1))
        # An unparseable predicate makes the underlying delete raise; the
        # safe-delete wrapper swallows it and clear_table reports False.
        assert store.clear_table("chunks", "not a predicate ((") is False


class TestEscapeSqlString:
    def test_escapes_single_quotes(self):
        assert escape_sql_string("it's") == "it''s"

    def test_backslash_is_not_escaped(self):
        # LanceDB's Datafusion treats backslash literally inside a '...' literal,
        # so doubling it corrupts the value and a backslash-bearing name never
        # matches its predicate (bb-7jg1).
        assert escape_sql_string("path\\file") == "path\\file"

    def test_injection_payload(self):
        escaped = escape_sql_string("' OR 1=1 --")
        # The leading quote is doubled, so it becomes '' (escaped)
        assert escaped.startswith("''")
        # No lone single quote remains (all are doubled)
        stripped = escaped.replace("''", "")
        assert "'" not in stripped

    def test_delete_by_source_matches_backslash_source(self, store):
        """A source name with a backslash must match its predicate end-to-end; the
        prior backslash-doubling made it never match, leaking the source's chunks."""
        backslash_src = r"win\dir\doc.md"
        backslash = _make_records(n=1)
        backslash[0]["source"] = backslash_src
        plain = _make_records(n=1)
        plain[0]["source"] = "plain.md"
        plain[0]["chunk_index"] = 1
        store.add_chunks(backslash + plain)

        store.delete_by_source(backslash_src, SourceType.DOCUMENT)

        sources = set(_sources_of(store))
        assert backslash_src not in sources
        assert "plain.md" in sources


class TestChunkTypeField:
    def test_chunk_type_stored_and_retrieved(self, store):
        records = _make_records(n=1, chunk_type="wiki")
        store.add_chunks(records)
        results = store.get_chunks_by_source("doc0.md")
        assert len(results) == 1
        assert results[0].chunk_type == "wiki"

    def test_chunk_type_defaults_to_raw(self, store):
        records = _make_records(n=1)
        store.add_chunks(records)
        results = store.get_chunks_by_source("doc0.md")
        assert results[0].chunk_type == "raw"

    def test_get_chunks_by_source_filters_with_fts_index_built(self, store):
        """The filtered query still selects rows once the chunks table is FTS-indexed.

        Both chunk-fetch paths rely on the database doing the filtering, so an
        FTS-indexed table that rejected ``.where()`` would silently regress them
        into whole-table reads. Pin the behavior the fetch paths depend on.
        """
        store.add_chunks(_make_records(n=3))
        store.ensure_fts_index()
        results = store.get_chunks_by_source("doc1.md")
        assert [r.source for r in results] == ["doc1.md"]


def _one_source_records(source: str, n: int) -> list[dict]:
    """*n* sequential chunks all belonging to *source*."""
    records = _make_records(n=n)
    for record in records:
        record["source"] = source
    return records


class TestGetChunksByIndices:
    def test_returns_requested_indices_in_order(self, store):
        store.add_chunks(_one_source_records("a.md", 5))
        results = store.get_chunks_by_indices("a.md", [3, 1])
        assert [r.chunk_index for r in results] == [1, 3]
        assert all(r.source == "a.md" for r in results)

    def test_missing_indices_are_absent(self, store):
        store.add_chunks(_one_source_records("a.md", 2))
        results = store.get_chunks_by_indices("a.md", [1, 99])
        assert [r.chunk_index for r in results] == [1]

    def test_other_sources_are_excluded(self, store):
        store.add_chunks(_one_source_records("a.md", 2) + _one_source_records("b.md", 2))
        results = store.get_chunks_by_indices("a.md", [0, 1])
        assert {r.source for r in results} == {"a.md"}

    def test_empty_indices_returns_empty(self, store):
        store.add_chunks(_one_source_records("a.md", 1))
        assert store.get_chunks_by_indices("a.md", []) == []

    def test_no_table_returns_empty(self, store):
        assert store.get_chunks_by_indices("a.md", [0]) == []

    def test_filters_with_fts_index_built(self, store):
        """The compound source+index predicate survives an FTS-indexed table."""
        store.add_chunks(_one_source_records("a.md", 3) + _one_source_records("b.md", 3))
        store.ensure_fts_index()
        results = store.get_chunks_by_indices("a.md", [0, 2])
        assert [r.chunk_index for r in results] == [0, 2]
        assert {r.source for r in results} == {"a.md"}

    def test_search_chunk_default_is_raw(self):
        chunk = SearchChunk(
            source="a.md",
            content_type="text",
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk="text",
            chunk_index=0,
            vector=[0.1],
        )
        assert chunk.chunk_type == "raw"

    def test_search_chunk_none_chunk_type_coerced_to_raw(self):
        """LanceDB rows from before the chunk_type column return None."""
        chunk = SearchChunk(
            source="a.md",
            content_type="text",
            chunk_type=None,
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk="text",
            chunk_index=0,
            vector=[0.1],
        )
        assert chunk.chunk_type == "raw"


class TestSourceTypeField:
    def test_source_type_defaults_to_document(self, store):
        store.upsert_source("a.md", "hash123", 5)
        sources = store.get_sources()
        assert len(sources) == 1
        assert sources[0]["source_type"] == SourceType.DOCUMENT

    def test_source_type_imported(self, store):
        store.upsert_source("shared.pdf", "", 3, source_type=SourceType.IMPORTED)
        sources = store.get_sources()
        assert len(sources) == 1
        assert sources[0]["source_type"] == SourceType.IMPORTED


def _page_rows(source="doc.pdf", pages=(1, 2)):
    return [
        {"source": source, "page": p, "text": f"page {p} text", "content_type": "pdf"}
        for p in pages
    ]


class TestPageTexts:
    def test_add_and_get_all(self, store):
        assert store.add_page_texts(_page_rows()) == 2
        rows = store.get_page_texts()
        assert {r["page"] for r in rows} == {1, 2}
        assert rows[0]["content_type"] == "pdf"

    def test_add_empty_is_noop(self, store):
        assert store.add_page_texts([]) == 0
        assert store.get_page_texts() == []

    def test_get_by_source(self, store):
        store.add_page_texts(_page_rows("a.pdf", (1,)))
        store.add_page_texts(_page_rows("b.pdf", (1, 2)))
        assert {r["source"] for r in store.get_page_texts("b.pdf")} == {"b.pdf"}
        assert len(store.get_page_texts("b.pdf")) == 2

    def test_get_missing_table_returns_empty(self, store):
        assert store.get_page_texts() == []
        assert store.get_page_texts("x.pdf") == []

    def test_sources_arrow_keys_metadata_by_source(self, store):
        from lilbee.data.store import SourceMeta

        store.upsert_source("a.pdf", "h", 1, meta=SourceMeta("Alpha", "Ada", "2020-01-01"))
        store.upsert_source("b.pdf", "h", 1)
        arrow = store.sources_arrow()
        assert arrow.schema.names == ["source", "title", "authors", "created_at"]
        by_source = {r["source"]: r for r in arrow.to_pylist()}
        assert by_source["a.pdf"]["title"] == "Alpha"
        assert by_source["a.pdf"]["authors"] == "Ada"
        assert set(by_source) == {"a.pdf", "b.pdf"}

    def test_sources_arrow_missing_table(self, store):
        arrow = store.sources_arrow()
        assert arrow.num_rows == 0
        assert arrow.schema.names == ["source", "title", "authors", "created_at"]

    def test_sources_arrow_fills_columns_an_older_index_lacks(self, store):
        # A sources table written before the metadata columns existed must still
        # join, with nulls, rather than losing the column and breaking the export.
        early = pa.Table.from_pylist([_early_source("old.pdf")], schema=_EARLY_SOURCES_SCHEMA)
        arrow = _open_after(store._config, {"_sources": early}).sources_arrow()
        assert arrow.schema.names == ["source", "title", "authors", "created_at"]
        assert arrow.to_pylist() == [
            {"source": "old.pdf", "title": None, "authors": None, "created_at": None}
        ]

    def test_delete_by_source_removes_page_texts(self, store):
        store.add_chunks(_make_records(n=1))
        store.add_page_texts(_page_rows("doc0.md", (1,)))
        store.delete_by_source("doc0.md", SourceType.DOCUMENT)
        assert store.get_page_texts("doc0.md") == []


class TestGetSourcesPagination:
    """LanceDB-side limit/offset/search for /api/documents scalability."""

    def _seed(self, store, n: int = 10) -> None:
        for i in range(n):
            store.upsert_source(f"doc{i:02d}.md", f"hash{i}", i + 1)

    def test_no_args_returns_all(self, store):
        self._seed(store, 5)
        assert len(store.get_sources()) == 5

    def test_limit_caps_returned_rows(self, store):
        self._seed(store, 10)
        assert len(store.get_sources(limit=3)) == 3

    def test_offset_skips_leading_rows(self, store):
        self._seed(store, 10)
        filenames = {s["filename"] for s in store.get_sources(offset=5, limit=5)}
        # Offset runs in LanceDB, so exactly 5 of the 10 come back.
        assert len(filenames) == 5

    def test_search_filters_by_filename_case_insensitive(self, store):
        store.upsert_source("README.md", "h1", 1)
        store.upsert_source("setup.py", "h2", 1)
        store.upsert_source("readme_dev.md", "h3", 1)
        matches = {s["filename"] for s in store.get_sources(search="readme")}
        assert matches == {"README.md", "readme_dev.md"}

    def test_search_and_limit_compose(self, store):
        for i in range(20):
            store.upsert_source(f"readme_{i}.md", f"h{i}", 1)
        store.upsert_source("other.py", "h99", 1)
        result = store.get_sources(search="readme", limit=5)
        assert len(result) == 5

    def test_search_treats_underscore_literally(self, store):
        # Without escaping, '_' is a LIKE single-char wildcard, so 'a_b' would
        # also match 'axb'. The ESCAPE clause makes it match literally.
        store.upsert_source("a_b.md", "h1", 1)
        store.upsert_source("axb.md", "h2", 1)
        matches = {s["filename"] for s in store.get_sources(search="a_b")}
        assert matches == {"a_b.md"}

    def test_search_treats_percent_literally(self, store):
        # '%' is the LIKE any-length wildcard; a literal search must not match
        # an unrelated filename just because the pattern contains '%'.
        store.upsert_source("50%done.md", "h1", 1)
        store.upsert_source("50xdone.md", "h2", 1)
        matches = {s["filename"] for s in store.get_sources(search="50%done")}
        assert matches == {"50%done.md"}


class TestCountSources:
    def test_count_matches_row_count(self, store):
        for i in range(7):
            store.upsert_source(f"doc{i}.md", f"h{i}", 1)
        assert store.count_sources() == 7

    def test_count_with_search_filter(self, store):
        store.upsert_source("readme.md", "h1", 1)
        store.upsert_source("setup.py", "h2", 1)
        store.upsert_source("README_2.md", "h3", 1)
        assert store.count_sources(search="readme") == 2

    def test_count_empty_table_returns_zero(self, store):
        assert store.count_sources() == 0


class TestSourceIngestedAtMap:
    """Query-side temporal filter calls this per query; caching avoids
    a fresh get_sources() materialization for every question."""

    def test_returns_filename_to_ingested_at(self, store):
        store.upsert_source("a.md", "h1", 1)
        store.upsert_source("b.md", "h2", 1)
        result = store.source_ingested_at_map()
        assert set(result) == {"a.md", "b.md"}
        assert all(v for v in result.values())

    def test_cache_is_reused_until_mutation(self, store):
        store.upsert_source("a.md", "h1", 1)
        first = store.source_ingested_at_map()
        # Same object identity means the cache served the call without
        # a re-materialization pass over SOURCES.
        assert store.source_ingested_at_map() is first

    def test_upsert_invalidates_cache(self, store):
        store.upsert_source("a.md", "h1", 1)
        first = store.source_ingested_at_map()
        store.upsert_source("b.md", "h2", 1)
        second = store.source_ingested_at_map()
        assert first is not second
        assert "b.md" in second

    def test_delete_invalidates_cache(self, store):
        store.upsert_source("a.md", "h1", 1)
        store.source_ingested_at_map()
        store.delete_source("a.md")
        assert store.source_ingested_at_map() == {}

    def test_drop_all_invalidates_cache(self, store):
        store.upsert_source("a.md", "h1", 1)
        store.source_ingested_at_map()
        store.drop_all()
        assert store.source_ingested_at_map() == {}


def _make_citation(**overrides) -> CitationRecord:
    defaults: CitationRecord = {
        "wiki_source": "wiki/summaries/doc.md",
        "wiki_chunk_index": 0,
        "citation_key": "src1",
        "claim_type": "fact",
        "source_filename": "documents/source.pdf",
        "source_hash": "abc123",
        "page_start": 1,
        "page_end": 1,
        "line_start": 0,
        "line_end": 0,
        "excerpt": "Python supports gradual typing.",
        "created_at": "2026-04-04T00:00:00+00:00",
    }
    defaults.update(overrides)  # type: ignore[typeddict-item]
    return defaults


class TestCitationCrud:
    def test_add_and_retrieve_citations(self, store):
        citations = [_make_citation(), _make_citation(citation_key="src2", excerpt="PEP 695")]
        count = store.add_citations(citations)
        assert count == 2
        results = store.get_citations_for_wiki("wiki/summaries/doc.md")
        assert len(results) == 2

    def test_add_empty_list_returns_zero(self, store):
        assert store.add_citations([]) == 0

    def test_get_citations_for_nonexistent_wiki(self, store):
        assert store.get_citations_for_wiki("nonexistent.md") == []

    def test_get_citations_for_source_reverse_lookup(self, store):
        store.add_citations(
            [
                _make_citation(wiki_source="wiki/a.md", source_filename="docs/paper.pdf"),
                _make_citation(wiki_source="wiki/b.md", source_filename="docs/paper.pdf"),
                _make_citation(wiki_source="wiki/c.md", source_filename="docs/other.txt"),
            ]
        )
        results = store.get_citations_for_source("docs/paper.pdf")
        assert len(results) == 2
        wiki_sources = {r["wiki_source"] for r in results}
        assert wiki_sources == {"wiki/a.md", "wiki/b.md"}

    def test_get_citations_for_nonexistent_source(self, store):
        assert store.get_citations_for_source("nonexistent.pdf") == []

    def test_delete_citations_for_wiki(self, store):
        store.add_citations(
            [
                _make_citation(wiki_source="wiki/a.md"),
                _make_citation(wiki_source="wiki/b.md"),
            ]
        )
        store.delete_citations_for_wiki("wiki/a.md")
        assert store.get_citations_for_wiki("wiki/a.md") == []
        assert len(store.get_citations_for_wiki("wiki/b.md")) == 1

    def test_delete_citations_nonexistent_wiki_is_noop(self, store):
        store.delete_citations_for_wiki("nonexistent.md")

    def test_citation_claim_types(self, store):
        store.add_citations(
            [
                _make_citation(claim_type="fact", excerpt="Real excerpt"),
                _make_citation(citation_key="src2", claim_type="inference", excerpt=""),
            ]
        )
        results = store.get_citations_for_wiki("wiki/summaries/doc.md")
        facts = [r for r in results if r["claim_type"] == "fact"]
        inferences = [r for r in results if r["claim_type"] == "inference"]
        assert len(facts) == 1
        assert facts[0]["excerpt"] == "Real excerpt"
        assert len(inferences) == 1
        assert inferences[0]["excerpt"] == ""

    def test_drop_all_includes_citations(self, store):
        store.add_citations([_make_citation()])
        store.drop_all()
        assert store.get_citations_for_wiki("wiki/summaries/doc.md") == []


class TestHybridSearchDirect:
    def test_returns_search_chunks(self, store, test_config):
        """_hybrid_search returns SearchChunk instances from hybrid query."""
        records = _make_records(n=3)
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=3, query_text="chunk number")
        assert all(isinstance(r, SearchChunk) for r in results)


class TestAdaptiveFilterFinalPass:
    def test_final_pass_at_cap(self, store):
        """When widening exceeds cap, final pass at cap still filters correctly."""
        results = [
            SearchChunk(
                source="a.md",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="moderate",
                chunk_index=0,
                vector=[0.1],
                distance=0.95,
            ),
        ]
        # initial_threshold=0.3, step=0.2 -> 0.3, 0.5, 0.7, 0.9 then cap=1.0 final pass
        filtered = store._adaptive_filter(results, top_k=1, initial_threshold=0.3)
        assert len(filtered) == 1
        assert filtered[0].chunk == "moderate"


class TestTableNames:
    def test_follows_page_token_to_the_last_page(self):
        """Names from every page are returned, not only the first page."""
        from lance_namespace import ListTablesResponse

        from lilbee.data.store.lance_helpers import table_names

        db = mock.MagicMock()
        db.list_tables.side_effect = [
            ListTablesResponse(tables=["chunks", "sources"], page_token="sources.lance/"),
            ListTablesResponse(tables=["citations"], page_token=None),
        ]
        assert table_names(db) == ["chunks", "sources", "citations"]
        assert [c.kwargs["page_token"] for c in db.list_tables.call_args_list] == [
            None,
            "sources.lance/",
        ]

    def test_empty_page_token_ends_the_listing(self):
        """An empty page token marks the end of the listing, same as None."""
        from lance_namespace import ListTablesResponse

        from lilbee.data.store.lance_helpers import table_names

        db = mock.MagicMock()
        db.list_tables.return_value = ListTablesResponse(tables=["chunks"], page_token="")
        assert table_names(db) == ["chunks"]
        db.list_tables.assert_called_once()

    def test_lists_every_table_of_a_real_store_past_one_page(self, tmp_path):
        """A real connection with more tables than one small page lists them all."""
        import lancedb.db
        import pyarrow as pa

        from lilbee.data.store.lance_helpers import table_names

        db = lancedb.db.LanceDBConnection(str(tmp_path))
        names = [f"t{i:02d}" for i in range(12)]
        for name in names:
            db.create_table(name, schema=pa.schema([("a", pa.int64())]))
        real_list = db.list_tables
        with mock.patch.object(
            db, "list_tables", side_effect=lambda **kw: real_list(limit=5, **kw)
        ) as listing:
            assert sorted(table_names(db)) == names
        assert listing.call_count >= 3


class TestEnsureTableRealLance:
    """ensure_table against a real lancedb connection on disk."""

    @staticmethod
    def _schema():
        import pyarrow as pa

        return pa.schema([("a", pa.int64())])

    def test_create_race_opens_the_table_the_other_writer_made(self, tmp_path):
        """A table created after this connection's listing is opened, not recreated."""
        import lancedb.db

        from lilbee.data.store import lance_helpers

        ours = lancedb.db.LanceDBConnection(str(tmp_path))
        theirs = lancedb.db.LanceDBConnection(str(tmp_path))
        theirs.create_table("chunks", schema=self._schema()).add([{"a": 7}])
        with mock.patch.object(lance_helpers, "table_names", return_value=[]):
            table = lance_helpers.ensure_table(ours, "chunks", self._schema())
        assert table.to_arrow().column("a").to_pylist() == [7]

    def test_existing_table_is_opened_with_its_rows(self, tmp_path):
        import lancedb.db

        from lilbee.data.store.lance_helpers import ensure_table

        db = lancedb.db.LanceDBConnection(str(tmp_path))
        ensure_table(db, "chunks", self._schema()).add([{"a": 1}])
        assert ensure_table(db, "chunks", self._schema()).count_rows() == 1

    def test_corrupt_table_raises_instead_of_being_recreated(self, tmp_path):
        """An unreadable manifest surfaces as an error; no empty table replaces it."""
        import lancedb.db

        from lilbee.data.store.lance_helpers import ensure_table

        db = lancedb.db.LanceDBConnection(str(tmp_path))
        ensure_table(db, "chunks", self._schema()).add([{"a": 1}])
        for manifest in (tmp_path / "chunks.lance" / "_versions").iterdir():
            manifest.write_bytes(b"garbage")
        fresh = lancedb.db.LanceDBConnection(str(tmp_path))
        with pytest.raises(RuntimeError):
            ensure_table(fresh, "chunks", self._schema())


class TestSearchAdaptiveThresholdPath:
    def test_search_uses_adaptive_filter_when_enabled(self, test_config, store):
        """When adaptive_threshold is True, search calls _adaptive_filter."""
        test_config.adaptive_threshold = True
        records = _make_records(n=2)
        store.add_chunks(records)
        query_vec = [0.5] * test_config.embedding_dim
        with mock.patch.object(store, "_adaptive_filter", return_value=[]) as mock_af:
            store.search(query_vec, top_k=2)
            mock_af.assert_called_once()


class TestDeleteSourceNoneTable:
    def test_noop_when_no_table(self, store):
        """delete_source is a no-op when the sources table doesn't exist."""
        store.delete_source("nonexistent.md")  # Should not raise


class TestSuppressLancedbThreadError:
    """Tests for the opt-in threading.excepthook that silences lancedb shutdown noise."""

    def test_install_suppresses_lancedb_thread(self):
        """Errors from LanceDB background threads are silently dropped."""

        from lilbee.data.store import install_lancedb_thread_error_suppressor

        original = threading.excepthook
        try:
            install_lancedb_thread_error_suppressor()
            lance_thread = threading.Thread(target=lambda: None, name="LanceDBBackgroundEventLoop")
            args = threading.ExceptHookArgs(
                (RuntimeError, RuntimeError("shutdown"), None, lance_thread)
            )
            # Should return without calling original hook: no exception raised.
            threading.excepthook(args)
        finally:
            threading.excepthook = original

    def test_install_propagates_non_lancedb_thread(self):
        """Errors from other threads are forwarded to the original excepthook."""

        from lilbee.data.store import install_lancedb_thread_error_suppressor

        calls: list[threading.ExceptHookArgs] = []

        def fake_original(args: threading.ExceptHookArgs) -> None:
            calls.append(args)

        saved = threading.excepthook
        threading.excepthook = fake_original
        try:
            install_lancedb_thread_error_suppressor()
            other_thread = threading.Thread(target=lambda: None, name="SomeOtherThread")
            args = threading.ExceptHookArgs(
                (RuntimeError, RuntimeError("real error"), None, other_thread)
            )
            threading.excepthook(args)
        finally:
            threading.excepthook = saved

        assert len(calls) == 1
        assert calls[0] is args


class TestScopeResolution:
    """scope_to_chunk_type maps user-facing scope strings to store filter values."""

    def test_none_is_passthrough(self):
        assert scope_to_chunk_type(None) is None

    def test_both_disables_filter(self):
        assert scope_to_chunk_type("both") is None
        assert scope_to_chunk_type(SearchScope.BOTH) is None

    def test_raw_maps_to_raw(self):
        assert scope_to_chunk_type("raw") is ChunkType.RAW
        assert scope_to_chunk_type(SearchScope.RAW) is ChunkType.RAW

    def test_wiki_maps_to_wiki(self):
        assert scope_to_chunk_type("wiki") is ChunkType.WIKI
        assert scope_to_chunk_type(SearchScope.WIKI) is ChunkType.WIKI

    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError):
            scope_to_chunk_type("bogus")


class TestChunkTypeEnum:
    """ChunkType is a closed StrEnum whose members serialize as their values."""

    def test_members_are_str_values(self):
        assert ChunkType.RAW == "raw"
        assert ChunkType.WIKI == "wiki"
        assert f"{ChunkType.WIKI}" == "wiki"

    def test_decode_round_trip(self):
        assert ChunkType("raw") is ChunkType.RAW
        assert ChunkType("wiki") is ChunkType.WIKI

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError):
            ChunkType("bogus")


class TestHttpChunkTypeBoundary:
    """The HTTP request models decode chunk_type into ChunkType, rejecting junk."""

    def test_ask_request_decodes_wiki(self):
        from lilbee.server.models import AskRequest

        assert AskRequest(question="q", chunk_type="wiki").chunk_type is ChunkType.WIKI

    def test_ask_request_both_means_no_filter(self):
        from lilbee.server.models import AskRequest

        assert AskRequest(question="q", chunk_type="both").chunk_type is None

    def test_ask_request_rejects_unknown(self):
        from pydantic import ValidationError

        from lilbee.server.models import AskRequest

        with pytest.raises(ValidationError):
            AskRequest(question="q", chunk_type="bogus")


class TestChunkTypePredicate:
    """The SQL predicate for scope filtering tolerates NULL for raw."""

    def test_raw_matches_null_for_legacy_rows(self):
        from lilbee.data.store.lance_helpers import _chunk_type_predicate

        pred = _chunk_type_predicate("raw")
        assert "IS NULL" in pred
        assert "'raw'" in pred

    def test_wiki_does_not_match_null(self):
        from lilbee.data.store.lance_helpers import _chunk_type_predicate

        pred = _chunk_type_predicate("wiki")
        assert "IS NULL" not in pred
        assert pred == "chunk_type = 'wiki'"

    def test_raw_matches_extracted_tables(self):
        """``raw`` means document content. Scoping a search to the user's own
        documents must not drop the tables extracted from them."""
        from lilbee.data.store.lance_helpers import _chunk_type_predicate

        assert f"'{ChunkType.TABLE}'" in _chunk_type_predicate("raw")

    def test_raw_scoped_search_returns_table_rows(self, store):
        """The predicate is only half of it: prove a scoped query really serves
        table rows, since a wiki-disabled search narrows to raw by default."""
        store.add_chunks(_make_records(1, chunk_type=ChunkType.TABLE))
        store.add_chunks(_make_records(1, chunk_type=ChunkType.WIKI))

        rows = store.bm25_probe("text", chunk_type=ChunkType.RAW)

        assert [r.chunk_type for r in rows] == [ChunkType.TABLE]


class TestEmbeddingModelGate:
    """Refuse search/ingest when cfg.embedding_model drifts from the persisted _meta row."""

    def test_first_add_chunks_initializes_meta_from_cfg(self, store, test_config):
        """A fresh store has no _meta row; first ingest writes one from current cfg."""
        assert store.get_meta() is None
        store.add_chunks(_make_records())
        meta = store.get_meta()
        assert meta is not None
        assert meta["embedding_model"] == test_config.embedding_model
        assert meta["embedding_dim"] == test_config.embedding_dim
        from lilbee.data.store.types import META_SCHEMA_VERSION

        assert meta["schema_version"] == META_SCHEMA_VERSION
        assert meta["updated_at"]

    def test_add_chunks_raises_when_model_drifts_same_dim(self, store, test_config):
        """Same dim, different model = silent corruption today; the gate refuses now."""
        from lilbee.data.store import EmbeddingModelMismatchError

        store.add_chunks(_make_records())
        original_model = test_config.embedding_model
        test_config.embedding_model = "ollama/different-model:v1"

        with pytest.raises(EmbeddingModelMismatchError) as exc_info:
            store.add_chunks(_make_records())
        assert original_model in str(exc_info.value)
        assert "ollama/different-model:v1" in str(exc_info.value)
        # Same-dim drift is adoptable: the index can be used under its own
        # embedder, so the error names both models and reports dims_match.
        assert exc_info.value.dims_match is True
        assert exc_info.value.persisted_model == original_model

    def test_search_raises_when_model_drifts(self, store, test_config):
        """Search refuses to serve under a different embedding model than the persisted one."""
        from lilbee.data.store import EmbeddingModelMismatchError

        store.add_chunks(_make_records())
        test_config.embedding_model = "ollama/switched-model:latest"

        with pytest.raises(EmbeddingModelMismatchError):
            store.search([0.1] * test_config.embedding_dim)

    def test_search_raises_when_dim_drifts(self, store, test_config):
        """Switching to a different-dim model is rejected by the meta gate."""
        from lilbee.data.store import EmbeddingModelMismatchError

        store.add_chunks(_make_records())
        test_config.embedding_dim = test_config.embedding_dim + 16
        test_config.embedding_model = "ollama/wider-model:v1"

        with pytest.raises(EmbeddingModelMismatchError):
            store.search([0.1] * test_config.embedding_dim)

    def test_search_short_circuits_on_missing_chunks_table(self, store, test_config):
        """Empty stores (no chunks table) return [] before the gate runs."""
        assert store.search([0.1] * test_config.embedding_dim) == []

    def test_assert_embedding_compatible_passes_on_match(self, store):
        store.add_chunks(_make_records())
        store.assert_embedding_compatible()  # no raise under the same embedder

    def test_assert_embedding_compatible_raises_on_drift(self, store, test_config):
        from lilbee.data.store import EmbeddingModelMismatchError

        store.add_chunks(_make_records())
        test_config.embedding_model = "ollama/another-model:v1"
        with pytest.raises(EmbeddingModelMismatchError):
            store.assert_embedding_compatible()

    def test_assert_embedding_compatible_noop_on_empty_store(self, store):
        store.assert_embedding_compatible()  # no meta, no chunks: nothing to check

    def test_legacy_store_search_writes_meta_with_warning(self, store, test_config, caplog):
        """First search on a pre-upgrade store (chunks present, no _meta) lazy-inits meta."""
        import logging

        store.add_chunks(_make_records())
        meta_table = store.open_table(META_TABLE)
        assert meta_table is not None
        from lilbee.data.store.lance_helpers import _safe_delete_unlocked
        from lilbee.data.store.types import META_DELETE_ALL_PREDICATE

        _safe_delete_unlocked(meta_table, META_DELETE_ALL_PREDICATE)
        assert store.get_meta() is None

        with caplog.at_level(logging.WARNING, logger="lilbee.data.store"):
            results = store.search([0.1] * test_config.embedding_dim)

        assert isinstance(results, list)
        meta = store.get_meta()
        assert meta is not None
        assert meta["embedding_model"] == test_config.embedding_model
        assert any("Legacy store" in r.message for r in caplog.records)

    def test_drop_all_includes_meta(self, store):
        """drop_all wipes _meta along with the other tables."""
        store.add_chunks(_make_records())
        assert store.get_meta() is not None
        store.drop_all()
        assert store.get_meta() is None

    def test_meta_row_is_overwritten_not_appended(self, store, test_config):
        """Re-ingesting after drop_all writes a single fresh _meta row, not a second one."""
        store.add_chunks(_make_records())
        store.drop_all()
        store.add_chunks(_make_records())
        meta_table = store.open_table(META_TABLE)
        assert meta_table is not None
        assert meta_table.count_rows() == 1

    def test_initialize_meta_if_legacy_pins_old_identity(self, store, test_config):
        """Pinning legacy meta uses the cfg present at the time of the call.

        This is the bb-x1qa upgrade-window protection: ``set_embedding_model``
        calls this BEFORE mutating cfg, so the recorded model identity is the
        OLD model. The next search/ingest then detects drift instead of silently
        adopting the new model as if it had built the store.
        """
        store.add_chunks(_make_records())
        meta_table = store.open_table(META_TABLE)
        assert meta_table is not None
        from lilbee.data.store.lance_helpers import _safe_delete_unlocked
        from lilbee.data.store.types import META_DELETE_ALL_PREDICATE

        _safe_delete_unlocked(meta_table, META_DELETE_ALL_PREDICATE)
        original_model = test_config.embedding_model

        wrote = store.initialize_meta_if_legacy()
        assert wrote is True
        meta = store.get_meta()
        assert meta is not None
        assert meta["embedding_model"] == original_model

        # Second call is a no-op: meta already pinned.
        assert store.initialize_meta_if_legacy() is False

    def test_initialize_meta_if_legacy_noop_on_empty_store(self, store):
        """No chunks, no meta = nothing to pin."""
        assert store.initialize_meta_if_legacy() is False
        assert store.get_meta() is None

    def test_legacy_bare_repo_meta_is_compatible_with_full_ref(self, tmp_path):
        """A pre-canonical store meta (bare ``<org>/<repo>``) matches the new full ref.

        Same repo, same dim: gating treats them as the same model so search and
        ingest do not refuse, and the chat error stops showing the legacy name.
        """
        full_ref = "org/repo-GGUF/model.Q4_K_M.gguf"
        cfg_local = cfg.model_copy(
            update={"lancedb_dir": tmp_path / "lance_legacy", "embedding_model": full_ref}
        )
        local_store = Store(cfg_local)
        local_store.add_chunks(_make_records(dim=cfg_local.embedding_dim))
        with write_lock():
            local_store._write_meta_unlocked(
                embedding_model="org/repo-GGUF", embedding_dim=cfg_local.embedding_dim
            )
        # Search must NOT raise: legacy bare-repo + matching dim is compatible.
        local_store.search([0.1] * cfg_local.embedding_dim)

    def test_canonicalize_meta_if_legacy_rewrites_to_full_ref(self, tmp_path):
        """Legacy bare-repo meta is rewritten to the canonical full ref on first call.

        Once migrated, the legacy name never surfaces in error messages, the
        UI, or downstream inspections of the meta row.
        """
        full_ref = "org/repo-GGUF/model.Q4_K_M.gguf"
        cfg_local = cfg.model_copy(
            update={"lancedb_dir": tmp_path / "lance_migrate", "embedding_model": full_ref}
        )
        local_store = Store(cfg_local)
        local_store.add_chunks(_make_records(dim=cfg_local.embedding_dim))
        with write_lock():
            local_store._write_meta_unlocked(
                embedding_model="org/repo-GGUF", embedding_dim=cfg_local.embedding_dim
            )

        wrote = local_store.canonicalize_meta_if_legacy()
        assert wrote is True
        meta = local_store.get_meta()
        assert meta is not None
        assert meta["embedding_model"] == full_ref

        # Idempotent: a second call is a no-op.
        assert local_store.canonicalize_meta_if_legacy() is False

    def test_canonicalize_meta_if_legacy_skips_when_already_canonical(self, store):
        """No write when meta is already the canonical full ref."""
        store.add_chunks(_make_records())
        assert store.canonicalize_meta_if_legacy() is False

    def test_canonicalize_meta_if_legacy_skips_when_meta_changes_under_lock(
        self, tmp_path, monkeypatch
    ):
        """Race path: another writer canonicalized between outer and inner check.

        The outer (no-lock) check sees the rewrite is needed; the inner (under
        write_lock) re-read sees it no longer is. Method short-circuits so the
        racer's work isn't clobbered.
        """
        full_ref = "org/repo-GGUF/model.Q4_K_M.gguf"
        cfg_local = cfg.model_copy(
            update={"lancedb_dir": tmp_path / "lance_race", "embedding_model": full_ref}
        )
        local_store = Store(cfg_local)
        local_store.add_chunks(_make_records(dim=cfg_local.embedding_dim))
        with write_lock():
            local_store._write_meta_unlocked(
                embedding_model="org/repo-GGUF", embedding_dim=cfg_local.embedding_dim
            )

        # First check (outer) says yes; second check (inner, under lock) says no.
        call_count = {"n": 0}
        original = local_store._needs_canonical_meta_rewrite

        def flipping_check(meta, current_model, current_dim):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return original(meta, current_model, current_dim)
            return False

        monkeypatch.setattr(local_store, "_needs_canonical_meta_rewrite", flipping_check)
        assert local_store.canonicalize_meta_if_legacy() is False

    def test_has_chunks_public_predicate(self, store):
        """``has_chunks`` is True after add_chunks and False on an empty store."""
        assert store.has_chunks() is False
        store.add_chunks(_make_records())
        assert store.has_chunks() is True

    def test_refs_compatible_remote_refs_strict_equality(self):
        """Two non-native refs (no ``.gguf``) require strict raw equality."""
        from lilbee.data.store.lance_helpers import refs_compatible

        assert refs_compatible("ollama/embed:a", "ollama/embed:a", 768, 768) is True
        # Different non-native refs: not compatible even when dims match.
        assert refs_compatible("ollama/embed:a", "ollama/embed:b", 768, 768) is False

    def test_canonicalize_meta_if_legacy_skips_on_genuine_mismatch(self, tmp_path):
        """Different file in the same repo is NOT a legacy match; gate still refuses."""
        from lilbee.data.store import EmbeddingModelMismatchError

        full_ref = "org/repo-GGUF/model.Q4_K_M.gguf"
        cfg_local = cfg.model_copy(
            update={"lancedb_dir": tmp_path / "lance_mismatch", "embedding_model": full_ref}
        )
        local_store = Store(cfg_local)
        local_store.add_chunks(_make_records(dim=cfg_local.embedding_dim))
        other_full = "org/repo-GGUF/model.Q8_0.gguf"
        with write_lock():
            local_store._write_meta_unlocked(
                embedding_model=other_full, embedding_dim=cfg_local.embedding_dim
            )
        # canonicalize must not rewrite (refs are NOT compatible).
        assert local_store.canonicalize_meta_if_legacy() is False
        # And the gate still refuses.
        with pytest.raises(EmbeddingModelMismatchError):
            local_store.search([0.1] * cfg_local.embedding_dim)

    def test_initialize_meta_if_legacy_skips_when_another_caller_won_the_race(
        self, store, test_config
    ):
        """Defensive re-check inside the lock: if a concurrent caller already wrote
        meta between the unlocked pre-check and acquiring the lock, do not double-write."""
        store.add_chunks(_make_records())
        meta_table = store.open_table(META_TABLE)
        assert meta_table is not None
        from lilbee.data.store.lance_helpers import _safe_delete_unlocked
        from lilbee.data.store.types import META_DELETE_ALL_PREDICATE

        _safe_delete_unlocked(meta_table, META_DELETE_ALL_PREDICATE)

        winning_meta = {
            "embedding_model": "winner-model:v1",
            "embedding_dim": test_config.embedding_dim,
            "schema_version": 1,
            "updated_at": "2026-04-26T00:00:00+00:00",
        }
        with mock.patch.object(store, "get_meta", side_effect=repeat_last(None, winning_meta)):
            assert store.initialize_meta_if_legacy() is False


class TestSourceStatColumns:
    """Size/mtime travel with each source row; legacy tables migrate in place."""

    def test_upsert_with_stat_roundtrips(self, store):
        from lilbee.data.store import SourceStat, source_stat

        store.upsert_source("a.md", "h1", 2, stat=SourceStat(123, 456))
        record = store.get_sources()[0]
        assert source_stat(record) == SourceStat(123, 456)

    def test_upsert_without_stat_reads_as_unknown(self, store):
        from lilbee.data.store import source_stat

        store.upsert_source("a.md", "h1", 2)
        assert source_stat(store.get_sources()[0]) is None

    def test_null_stat_columns_read_as_unknown_not_crash(self):
        """A row whose nullable stat columns are NULL must read as unknown, not
        crash with int(None). Regression: adding a file when the store already
        held a null-stat row failed every add with 'int() argument ... NoneType'."""
        from lilbee.data.store import source_stat

        # NULL columns (present-but-None), as a nullable int64 column yields --
        # distinct from a missing key, which .get already handled.
        record = {"name": "a.pdf", "file_hash": "h", "size_bytes": None, "mtime_ns": None}
        assert source_stat(record) is None  # type: ignore[arg-type]

    def test_null_captured_with_valid_size_mtime_reads_stat(self):
        """A null stat_captured_ns alongside valid size/mtime must not crash; the
        capture time falls back to the unknown sentinel."""
        from lilbee.data.store import SOURCE_STAT_UNKNOWN, SourceStat, source_stat

        record = {
            "name": "a.pdf",
            "file_hash": "h",
            "size_bytes": 10,
            "mtime_ns": 20,
            "stat_captured_ns": None,
        }
        assert source_stat(record) == SourceStat(10, 20, SOURCE_STAT_UNKNOWN)  # type: ignore[arg-type]

    def test_write_chunks_batch_persists_stat(self, store):
        from lilbee.data.store import ChunkWrite, SourceStat, source_stat

        items = [
            ChunkWrite(
                "a.md", "h", _records_for("a.md", 1), needs_cleanup=False, stat=SourceStat(9, 8)
            )
        ]
        store.write_chunks_batch(items)
        assert source_stat(store.get_sources()[0]) == SourceStat(9, 8)

    def test_legacy_sources_table_gains_stat_columns(self, store):
        # Build a pre-stat table by hand (5 columns, one legacy row).
        import pyarrow as pa

        from lilbee.core.config import SOURCES_TABLE
        from lilbee.data.store import SourceStat, source_stat

        early = pa.Table.from_pylist([_early_source("old.md")], schema=_EARLY_SOURCES_SCHEMA)
        store = _open_after(store._config, {SOURCES_TABLE: early})

        # The open migrated the table; a write lands the stat.
        store.upsert_source("new.md", "h2", 1, stat=SourceStat(5, 6))
        records = {r["filename"]: r for r in store.get_sources()}
        assert source_stat(records["old.md"]) is None  # backfilled sentinel
        assert source_stat(records["new.md"]) == SourceStat(5, 6)

    def test_update_source_stats_batches_one_write(self, store):
        from lilbee.data.store import SourceStat, SourceStatBackfill, source_stat

        store.upsert_source("a.md", "ha", 1)
        store.upsert_source("b.md", "hb", 2)
        records = {r["filename"]: r for r in store.get_sources()}

        backfills = [
            SourceStatBackfill(records["a.md"], SourceStat(1, 2)),
            SourceStatBackfill(records["b.md"], SourceStat(3, 4)),
        ]
        with mock.patch.object(
            store, "_put_sources_unlocked", wraps=store._put_sources_unlocked
        ) as spy:
            store.update_source_stats(backfills)
        spy.assert_called_once()

        records = {r["filename"]: r for r in store.get_sources()}
        assert source_stat(records["a.md"]) == SourceStat(1, 2)
        assert source_stat(records["b.md"]) == SourceStat(3, 4)
        assert records["a.md"]["file_hash"] == "ha"

    def test_update_source_stats_chunks_huge_backfills(self, store):
        # A whole-corpus backfill must not join every filename into one delete
        # predicate: rows are replaced in slices, each its own locked write.
        import math

        import lilbee.data.store.core as core_mod
        from lilbee.data.store import SourceStat, SourceStatBackfill, source_stat

        total, batch_rows = 5, 2
        for i in range(total):
            store.upsert_source(f"f{i}.md", f"h{i}", 1)
        records = {r["filename"]: r for r in store.get_sources()}
        backfills = [
            SourceStatBackfill(records[f"f{i}.md"], SourceStat(i, i + 1)) for i in range(total)
        ]
        with (
            mock.patch.object(core_mod, "_SOURCE_STAT_BATCH_ROWS", batch_rows),
            mock.patch.object(
                store, "_put_sources_unlocked", wraps=store._put_sources_unlocked
            ) as spy,
        ):
            store.update_source_stats(backfills)
        assert spy.call_count == math.ceil(total / batch_rows)
        assert all(len(call.args[1]) <= batch_rows for call in spy.call_args_list)

        records = {r["filename"]: r for r in store.get_sources()}
        assert len(records) == total
        for i in range(total):
            assert source_stat(records[f"f{i}.md"]) == SourceStat(i, i + 1)
            assert records[f"f{i}.md"]["file_hash"] == f"h{i}"

    def test_update_source_stats_empty_is_noop(self, store):
        store.update_source_stats([])
        assert store.get_sources() == []


class TestBatchedSourceUpserts:
    """One flush of N files gives the new ones ids in one write and every row in one more."""

    def test_write_chunks_batch_puts_every_source_row_in_one_write(self, store):
        from lilbee.data.store import ChunkWrite

        items = [
            ChunkWrite(f"f{i}.md", f"h{i}", _records_for(f"f{i}.md", 1), needs_cleanup=False)
            for i in range(5)
        ]
        with mock.patch.object(
            store, "_put_sources_unlocked", wraps=store._put_sources_unlocked
        ) as spy:
            store.write_chunks_batch(items)
        assert spy.call_count == 2
        assert len(spy.call_args.args[1]) == 5
        assert len(store.get_sources()) == 5

    def test_optimize_sources_compacts(self, store):
        store.upsert_source("a.md", "h", 1)
        with mock.patch.object(store, "open_table", return_value=mock.MagicMock()) as opened:
            store.optimize_sources()
        opened.return_value.optimize.assert_called_once()

    def test_optimize_sources_noop_without_table(self, store):
        with mock.patch.object(store, "open_table", return_value=None) as opened:
            store.optimize_sources()
        opened.assert_called_once()

    def test_optimize_sources_survives_failure(self, store):
        failing = mock.MagicMock()
        failing.optimize.side_effect = RuntimeError("compaction failed")
        with mock.patch.object(store, "open_table", return_value=failing):
            store.optimize_sources()
        failing.optimize.assert_called_once()


class TestDeleteBySourceConceptRows:
    """Re-ingest cleanup removes the source's chunk-concept rows too."""

    def test_delete_by_source_clears_chunk_concepts(self, store):
        from lilbee.core.config import CHUNK_CONCEPTS_TABLE
        from lilbee.data.store import ensure_table
        from lilbee.retrieval.concepts.schema import _chunk_concepts_schema

        store.add_chunks(_records_for("doc.md", 2))
        doc = store.source_ids(["doc.md"], SourceType.DOCUMENT)["doc.md"]
        cc_table = ensure_table(store.get_db(), CHUNK_CONCEPTS_TABLE, _chunk_concepts_schema())
        cc_table.add(
            [
                {"source_id": doc, "chunk_index": 0, "concept": "alpha"},
                {"source_id": "other", "chunk_index": 0, "concept": "beta"},
            ]
        )

        store.delete_by_source("doc.md", SourceType.DOCUMENT)
        cc_table.checkout_latest()  # bypass the read-consistency interval
        remaining = cc_table.search().limit(None).to_list()
        assert [r["source_id"] for r in remaining] == ["other"]
        assert store.get_chunks_by_source("doc.md") == []


class TestAnnNprobesScaling:
    """nprobes follows the IVF partition count (~sqrt(N)) with a floor."""

    def test_small_corpus_keeps_floor(self):
        from lilbee.data.store.core import _ANN_NPROBES_FLOOR, _ann_nprobes

        assert _ann_nprobes(0) == _ANN_NPROBES_FLOOR
        # isqrt(10_000) = 100 partitions, ceil(100 * 0.15) = 15 < floor.
        assert _ann_nprobes(10_000) == _ANN_NPROBES_FLOOR

    def test_large_corpus_scales_past_floor(self):
        from lilbee.data.store.core import _ANN_NPROBES_FLOOR, _ann_nprobes

        # 50M rows: isqrt(50_000_000) = 7071 partitions, ceil(7071 * 0.15) = 1061 probes.
        fifty_million = 50_000_000
        assert _ann_nprobes(fifty_million) == 1061
        assert _ann_nprobes(fifty_million) > _ANN_NPROBES_FLOOR

    def test_negative_row_count_clamps_to_floor(self):
        from lilbee.data.store.core import _ANN_NPROBES_FLOOR, _ann_nprobes

        assert _ann_nprobes(-5) == _ANN_NPROBES_FLOOR


def _titled_records(source, n, *, title, base=0.1, dim=None):
    """Chunk records for one source with a document title on every row."""
    if dim is None:
        dim = cfg.embedding_dim
    return [
        {
            "source": source,
            "content_type": "text",
            "chunk_type": "raw",
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": f"plain body words {source} {i}",
            "chunk_index": i,
            "title": title,
            "vector": [base + i / 100] * dim,
        }
        for i in range(n)
    ]


def _pre_title_store(config, records):
    """A Store over a library whose chunks table an old lilbee wrote, before titles."""
    schema = pa.schema(
        [
            pa.field("source", pa.utf8()),
            pa.field("content_type", pa.utf8()),
            pa.field("chunk_type", pa.utf8()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("line_start", pa.int32()),
            pa.field("line_end", pa.int32()),
            pa.field("chunk", pa.utf8()),
            pa.field("chunk_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), cfg.embedding_dim)),
        ]
    )
    store = _open_after(config, {"chunks": pa.Table.from_pylist(records, schema=schema)})
    return store, store.open_table("chunks")


class TestTitleSearch:
    """The title lexical arm: BM25 over document titles fused into hybrid search."""

    def test_title_arm_surfaces_title_only_match(self, store, test_config):
        """A term that lives only in a document's title reaches hybrid results
        with lexical support (bm25_score) when title_search is on. The title arm
        surfaces one representative chunk per matched document, so the document
        appears with lexical support even though not every chunk carries it."""
        test_config.title_search = True
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto", base=0.9))
        store.add_chunks(_titled_records("b.pdf", 2, title="meeting notes", base=0.1))
        store.ensure_fts_index()
        query_vec = [0.1] * test_config.embedding_dim
        results = store.search(query_vec, top_k=4, max_distance=0, query_text="zebra")
        matched = [r for r in results if r.source == "a.pdf"]
        assert matched
        assert any(r.bm25_score is not None for r in matched)

    def test_title_arm_collapses_to_one_deterministic_row_per_document(self, store, test_config):
        """Every chunk of a document shares its title, so all tie on BM25. The
        arm must collapse each matched document to one deterministic row (its
        first chunk), not return an arbitrary tie-ordered subset of that doc."""
        test_config.title_search = True
        store.add_chunks(_titled_records("a.pdf", 5, title="zebra manifesto"))
        store.ensure_fts_index()
        table = store.open_table("chunks")
        rows = store._with_paths(store._title_arm(table, "zebra", 5, None))
        assert [(r.source, r.chunk_index) for r in rows] == [("a.pdf", 0)]
        # Deterministic across repeated calls (no implementation-defined tie order).
        again = store._with_paths(store._title_arm(table, "zebra", 5, None))
        assert [(r.source, r.chunk_index) for r in again] == [("a.pdf", 0)]

    def test_title_arm_one_row_per_matched_document(self, store, test_config):
        """Two matched documents surface one representative row each, not an
        arbitrary flood of one document's chunks."""
        test_config.title_search = True
        store.add_chunks(_titled_records("zebra.pdf", 4, title="zebra"))
        store.add_chunks(_titled_records("safari.pdf", 4, title="zebra safari"))
        store.ensure_fts_index()
        table = store.open_table("chunks")
        rows = store._with_paths(store._title_arm(table, "zebra", 5, None))
        assert sorted(r.source for r in rows) == ["safari.pdf", "zebra.pdf"]
        assert all(r.chunk_index == 0 for r in rows)

    def test_title_arm_long_document_does_not_starve_other_matches(
        self, store, test_config, monkeypatch
    ):
        """A long document's tied title rows saturating the fetch window must
        widen the fetch, not crowd every other title-matching document out."""
        from lilbee.data.store import core as store_core

        monkeypatch.setattr(store_core, "_TITLE_FETCH_FACTOR", 1)
        monkeypatch.setattr(store_core, "_TITLE_MIN_FETCH", 4)
        test_config.title_search = True
        # The exact-title tome outscores the diluted note title, so its six
        # tied rows fill the first window and force the fetch to widen.
        store.add_chunks(_titled_records("tome.pdf", 6, title="zebra"))
        store.add_chunks(_titled_records("note.pdf", 1, title="zebra safari park visit"))
        store.ensure_fts_index()
        table = store.open_table("chunks")
        rows = store._with_paths(store._title_arm(table, "zebra", 2, None))
        assert sorted(r.source for r in rows) == ["note.pdf", "tome.pdf"]

    def test_title_search_weight_reaches_fusion(self, store, test_config):
        """A non-default title_search_weight is threaded into fuse_arms, not
        hardcoded: the config value is what weights the title arm."""
        test_config.title_search = True
        test_config.title_search_weight = 0.2
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        query_vec = [0.1] * test_config.embedding_dim
        from lilbee.data.store import core as store_core

        with mock.patch.object(store_core, "fuse_arms", wraps=store_core.fuse_arms) as fuse:
            store.search(query_vec, top_k=4, max_distance=0, query_text="zebra")
        assert fuse.call_args.kwargs["title_weight"] == pytest.approx(0.2)
        # fuse_arms counts the title weight in its denominator only when title
        # rows exist, so titleless queries keep their undeflated scores.
        assert "weight_total" not in fuse.call_args.kwargs

    def test_adaptive_fusion_scales_the_title_arm_too(self, store, test_config):
        """Adaptive fusion downweights lexical; the title arm is also lexical, so
        it must be scaled by the same confidence, not left at full weight (which
        would re-admit the signal adaptive fusion just silenced)."""
        test_config.title_search = True
        test_config.title_search_weight = 0.5
        test_config.adaptive_fusion = True
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        query_vec = [0.1] * test_config.embedding_dim
        from lilbee.data.store import core as store_core

        with (
            mock.patch.object(store_core, "adaptive_weight_scale", return_value=0.25),
            mock.patch.object(store_core, "fuse_arms", wraps=store_core.fuse_arms) as fuse,
        ):
            store.search(query_vec, top_k=4, max_distance=0, query_text="zebra")
        assert fuse.call_args.kwargs["lexical_weight"] == pytest.approx(
            test_config.lexical_fusion_weight * 0.25
        )
        assert fuse.call_args.kwargs["title_weight"] == pytest.approx(0.5 * 0.25)

    def test_title_arm_off_by_default(self, store, test_config):
        """With title_search off, a title-only term earns no lexical support."""
        assert test_config.title_search is False
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        query_vec = [0.1] * test_config.embedding_dim
        results = store.search(query_vec, top_k=4, max_distance=0, query_text="zebra")
        assert all(r.bm25_score is None for r in results)

    def test_title_arm_respects_chunk_type_filter(self, store, test_config):

        test_config.title_search = True  # the title index is built only when enabled
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        table = store.open_table("chunks")
        assert _has_fts_index(_readable_registry(table), "title")
        assert store._title_arm(table, "zebra", 5, ChunkType.RAW)
        assert store._title_arm(table, "zebra", 5, ChunkType.WIKI) == []

    def test_title_arm_failure_degrades_to_empty(self, store, test_config):
        """A query-time title-arm failure returns no rows instead of raising,
        so a broken title index can't take down the healthy chunk-BM25 arm and
        collapse the whole hybrid search to vector-only."""
        test_config.title_search = True
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        table = store.open_table("chunks")
        from lilbee.data.store import core as store_core

        with mock.patch.object(store_core, "_lexical_rows", side_effect=RuntimeError("boom")):
            assert store._title_arm(table, "zebra", 5, None) == []

    def test_old_index_without_title_column_still_searches(self, store, test_config):
        """Feature detection: a pre-title index searches fine and the title arm
        silently contributes nothing."""
        test_config.title_search = True
        store, table = _pre_title_store(test_config, _make_records())
        store.ensure_fts_index()
        assert store._title_arm(table, "chunk", 5, None) == []
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=3, max_distance=0, query_text="chunk number")
        assert results
        assert all(r.title is None for r in results)

    def test_add_chunks_evolves_pre_title_table(self, store):
        """A write to an old index adds the title column in place."""
        store, table = _pre_title_store(store._config, _make_records())
        assert "title" not in table.schema.names
        store.add_chunks(_titled_records("new.pdf", 1, title="fresh document"))
        assert "title" in store.open_table("chunks").schema.names
        rows = store.get_chunks_by_source("new.pdf")
        assert [r.title for r in rows] == ["fresh document"]

    def test_title_search_enable_at_runtime_builds_index_on_next_query(self, store, test_config):
        """Enabling title_search after _fts_ready latched builds the title
        index on the next query instead of no-opping until restart."""

        test_config.title_search = False
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        assert not _has_fts_index(_readable_registry(store.open_table("chunks")), "title")
        test_config.title_search = True
        store.search([0.1] * test_config.embedding_dim, top_k=3, query_text="zebra")
        assert _has_fts_index(_readable_registry(store.open_table("chunks")), "title")

    def test_backfill_failure_keeps_nulls_and_warns(self, store, caplog):
        """A failing backfill degrades to the old NULL-title behavior."""
        import logging

        records = _make_records()
        records[0]["source"] = "project_falcon_notes.pdf"
        store, table = _pre_title_store(store._config, records)
        with (
            mock.patch.object(type(table), "update", side_effect=RuntimeError("boom")),
            caplog.at_level(logging.WARNING),
        ):
            store.add_chunks(_titled_records("new.pdf", 1, title="fresh document"))
        assert any("Title backfill failed" in r.message for r in caplog.records)

    def test_pre_title_migration_backfills_stem_titles(self, store, caplog):
        """Migrating an old store backfills filename-stem titles (junk stems
        stay NULL) so the title arm sees pre-upgrade documents; content-derived
        titles still need a rebuild and the log says so."""
        import logging

        records = _make_records()
        records[0]["source"] = "project_falcon_notes.pdf"
        store, _table = _pre_title_store(store._config, records)
        with caplog.at_level(logging.INFO):
            store.add_chunks(_titled_records("new.pdf", 1, title="fresh document"))
        assert any("lilbee rebuild" in r.message for r in caplog.records)
        batches = store.chunk_batches(["source", "title"], 100)
        titles = {row["source"]: row["title"] for rows in batches for row in rows}
        assert titles["project_falcon_notes.pdf"] == "project falcon notes"
        assert titles["doc1.md"] is None  # junk counter stem stays NULL

    def test_bm25_probe_stays_chunk_scoped(self, store):
        """The probe pins the chunk column: a title-only term is not a probe hit."""
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        assert store.bm25_probe("zebra") == []
        assert store.bm25_probe("plain body words")

    def test_title_index_failure_never_blocks_chunk_index(self, store, test_config, caplog):
        """A failing title index leaves chunk FTS ready; the arm degrades to
        empty. Because the arm is enabled, the failure warns (not debug) so an
        opted-in title arm that cannot build is not a silent no-op."""
        import logging

        test_config.title_search = True
        store.add_chunks(_titled_records("a.pdf", 1, title="zebra manifesto"))
        table = store.open_table("chunks")
        real_create = type(table).create_index

        def _fail_title(self, column, **kwargs):
            if column == "title":
                raise RuntimeError("boom")
            return real_create(self, column, **kwargs)

        with (
            mock.patch.object(type(table), "create_index", _fail_title),
            caplog.at_level(logging.WARNING),
        ):
            store.ensure_fts_index()
        assert store._fts_ready
        assert store._title_arm(store.open_table("chunks"), "zebra", 5, None) == []
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("title" in r.message.lower() for r in warnings)


class TestSourceMetadata:
    """Extraction-time document metadata persisted on the sources table."""

    def test_upsert_source_persists_meta(self, store):
        store.upsert_source(
            "a.pdf",
            "hash1",
            3,
            meta=SourceMeta(title="The Title", authors="Ada, Grace", created_at="2021-05-01"),
        )
        row = store.get_sources()[0]
        assert row["title"] == "The Title"
        assert row["authors"] == "Ada, Grace"
        assert row["created_at"] == "2021-05-01"

    def test_absent_meta_persists_null(self, store):
        store.upsert_source("a.pdf", "hash1", 3)
        row = store.get_sources()[0]
        assert row["title"] is None
        assert row["authors"] is None
        assert row["created_at"] is None

    def test_pre_meta_sources_table_evolves_in_place(self, store):
        """An old sources table gains the metadata columns on the next write."""
        import pyarrow as pa

        old = _early_source("old.pdf", "h0", "2020-01-01T00:00:00+00:00")
        early = pa.Table.from_pylist([old], schema=_EARLY_SOURCES_SCHEMA)
        store = _open_after(store._config, {"_sources": early})
        store.upsert_source("new.pdf", "h1", 2, meta=SourceMeta(title="New Doc"))
        rows = {r["filename"]: r for r in store.get_sources()}
        assert rows["old.pdf"]["title"] is None
        assert rows["new.pdf"]["title"] == "New Doc"

    def test_batched_write_persists_meta(self, store):
        from lilbee.data.store import ChunkWrite, SourceMeta

        records = _titled_records("doc.pdf", 1, title="Batched Title")
        store.write_chunks_batch(
            [
                ChunkWrite(
                    "doc.pdf",
                    "h",
                    records,
                    needs_cleanup=False,
                    meta=SourceMeta(title="Batched Title", authors="Ada"),
                )
            ]
        )
        row = store.get_sources()[0]
        assert row["title"] == "Batched Title"
        assert row["authors"] == "Ada"


class TestRelocateSources:
    def test_relocate_rekeys_chunks_and_source_preserving_vectors(self, store):
        from lilbee.data.store import SourceType
        from lilbee.data.store.types import SourceStat

        records = _make_records(n=2)
        for r in records:
            r["source"] = "old/a.md"
        store.add_chunks(records)
        store.upsert_source("old/a.md", "hash123", 2, SourceType.DOCUMENT)
        before = store.get_chunks_by_source("old/a.md")
        assert len(before) == 2

        store.relocate_sources([SourceMove(("old/a.md",), "new/a.md", SourceStat(10, 20, 30))])

        assert store.get_chunks_by_source("old/a.md") == []
        after = store.get_chunks_by_source("new/a.md")
        assert len(after) == 2  # chunks (and their vectors) carried over, not rebuilt
        names = {s["filename"] for s in store.get_sources()}
        assert "new/a.md" in names
        assert "old/a.md" not in names
        moved = next(s for s in store.get_sources() if s["filename"] == "new/a.md")
        assert moved["file_hash"] == "hash123"  # same content, hash unchanged

    def test_relocate_rekeys_every_source_table(self, store):
        # Guards _RELOCATABLE_TABLES: page_texts.source and citations.source_filename
        # must move too, so dropping a table from the list fails here.
        from lilbee.data.store import SourceType
        from lilbee.data.store.types import SourceStat

        records = _make_records(n=1)
        records[0]["source"] = "old/a.md"
        store.add_chunks(records)
        store.add_page_texts(
            [{"source": "old/a.md", "page": 1, "text": "t", "content_type": "text"}]
        )
        store.replace_wiki_mentions_for_source("old/a.md", [_mention("boeing", "old/a.md", 2, [0])])
        store.add_citations(
            [
                {
                    "wiki_source": "w.md",
                    "wiki_chunk_index": 0,
                    "citation_key": "k",
                    "claim_type": "support",
                    "source_filename": "old/a.md",
                    "source_hash": "h",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "excerpt": "e",
                    "created_at": "",
                }
            ]
        )
        store.upsert_source("old/a.md", "h", 1, SourceType.DOCUMENT)

        store.relocate_sources([SourceMove(("old/a.md",), "new/a.md", SourceStat(1, 2, 3))])

        assert len(store.get_chunks_by_source("new/a.md")) == 1
        assert store.get_chunks_by_source("old/a.md") == []
        assert [row["source"] for row in store.get_page_texts("new/a.md")] == ["new/a.md"]
        assert store.get_page_texts("old/a.md") == []
        assert len(store.get_citations_for_source("new/a.md")) == 1
        assert store.get_citations_for_source("old/a.md") == []
        assert [row["source"] for row in store.wiki_mention_rows()] == ["new/a.md"]
        assert _holders(store, "old/a.md") == set()

    def test_a_move_commits_the_source_table_alone(self, store):
        """No title changes: every other table names the source by id and stays as it was."""
        from lilbee.data.store.lance_helpers import table_names

        _seed_source(store, "old/a.md", title="a")
        db = store.get_db()
        before = {name: db.open_table(name).version for name in table_names(db)}

        store.relocate_sources([SourceMove(("old/a.md",), "new/a.md", None)])

        after = {name: db.open_table(name).version for name in table_names(db)}
        changed = [name for name in after if after[name] != before[name]]
        assert changed == [SOURCES_TABLE]
        assert after[SOURCES_TABLE] == before[SOURCES_TABLE] + 1

    def test_the_sources_inside_a_moved_archive_move_with_it(self, store):
        for key in ("old/bundle.zip", "old/bundle.zip/m0.md", "old/bundle.zip.md"):
            _seed_source(store, key)

        store.relocate_sources([SourceMove(("old/bundle.zip",), "new/bundle.zip", None)])

        assert sorted(row["filename"] for row in store.get_sources()) == [
            "new/bundle.zip",
            "new/bundle.zip/m0.md",
            "old/bundle.zip.md",
        ]
        assert len(store.get_chunks_by_source("new/bundle.zip/m0.md")) == 1

    def test_relocate_empty_is_noop(self, store):
        store.relocate_sources([])  # must not raise or acquire the lock


class TestRelocateTitles:
    """Relocation re-derives stem titles; extraction titles survive the move."""

    def test_stem_derived_title_follows_the_new_filename(self, store):
        from lilbee.data.store import SourceMeta, SourceType

        store.add_chunks(_titled_records("old_report.md", 2, title="old report"))
        store.upsert_source(
            "old_report.md", "h1", 2, SourceType.DOCUMENT, meta=SourceMeta(title="old report")
        )
        store.relocate_sources([SourceMove(("old_report.md",), "annual_summary.md", None)])
        row = next(s for s in store.get_sources() if s["filename"] == "annual_summary.md")
        assert row["title"] == "annual summary"
        chunks = store.get_chunks_by_source("annual_summary.md")
        assert all(c.title == "annual summary" for c in chunks)

    def test_a_folder_move_of_files_with_the_same_title_writes_no_chunk_row(self, store):
        from lilbee.data.store import SourceMeta, SourceType
        from lilbee.data.title import derive_title

        moves = []
        for number in range(3):
            old, new = f"old/report_{number}.md", f"new/report_{number}.md"
            title = derive_title(old)
            store.add_chunks(_titled_records(old, 2, title=title))
            store.upsert_source(
                old, f"h{number}", 2, SourceType.DOCUMENT, meta=SourceMeta(title=title)
            )
            moves.append(SourceMove((old,), new, None))
        store.ensure_fts_index()
        chunks = store.open_table(CHUNKS_TABLE)
        version = chunks.version
        index = next(i.name for i in chunks.list_indices() if i.columns == ["chunk"])

        store.relocate_sources(moves)

        assert sorted(row["filename"] for row in store.get_sources()) == [
            move.new for move in moves
        ]
        chunks = store.open_table(CHUNKS_TABLE)
        assert chunks.version == version
        assert chunks.index_stats(index).num_unindexed_rows == 0
        assert len(store.get_chunks_by_source("new/report_1.md")) == 2

    def test_extraction_title_survives_the_move(self, store):
        from lilbee.data.store import SourceMeta, SourceType

        store.add_chunks(_titled_records("notes-2024.md", 2, title="Frankenstein Analysis"))
        store.upsert_source(
            "notes-2024.md",
            "h1",
            2,
            SourceType.DOCUMENT,
            meta=SourceMeta(title="Frankenstein Analysis"),
        )
        store.relocate_sources([SourceMove(("notes-2024.md",), "renamed.md", None)])
        row = next(s for s in store.get_sources() if s["filename"] == "renamed.md")
        assert row["title"] == "Frankenstein Analysis"
        chunks = store.get_chunks_by_source("renamed.md")
        assert all(c.title == "Frankenstein Analysis" for c in chunks)

    def test_junk_new_stem_clears_the_stem_title(self, store):
        from lilbee.data.store import SourceMeta, SourceType

        store.add_chunks(_titled_records("real_notes.md", 1, title="real notes"))
        store.upsert_source(
            "real_notes.md", "h1", 1, SourceType.DOCUMENT, meta=SourceMeta(title="real notes")
        )
        store.relocate_sources([SourceMove(("real_notes.md",), "IMG_1234.md", None)])
        row = next(s for s in store.get_sources() if s["filename"] == "IMG_1234.md")
        assert row["title"] is None


# How long a blocked write gets to prove it is not blocked; an unlocked one takes milliseconds.
_LOCKED_WRITE_WAIT_S = 1.0

# The one column that holds a source's path; every other table names the source by id.
_KEY_COLUMNS = {("_sources", "path")}


def _seed_source(store, key, *, title=None):
    """Write one row for *key* into every table that holds a source key."""
    from lilbee.core.config import CHUNK_CONCEPTS_TABLE
    from lilbee.data.store import SourceMeta, SourceType, ensure_table
    from lilbee.retrieval.concepts.schema import _chunk_concepts_schema

    store.add_chunks(_titled_records(key, 1, title=title, dim=store._config.embedding_dim))
    store.add_page_texts([{"source": key, "page": 1, "text": "t", "content_type": "text"}])
    concepts = ensure_table(store.get_db(), CHUNK_CONCEPTS_TABLE, _chunk_concepts_schema())
    source_id = store.source_ids([key], SourceType.DOCUMENT)[key]
    concepts.add([{"source_id": source_id, "chunk_index": 0, "concept": "roadmap"}])
    store.add_entities(
        [
            {
                "entity": "Boeing",
                "type": "ORG",
                "normalized_value": "boeing",
                "source": key,
                "page": 1,
                "chunk_index": 0,
                "confidence": 1.0,
            }
        ]
    )
    store.replace_wiki_mentions_for_source(key, [_mention("boeing", key, 1, [0])])
    store.add_citations(
        [
            {
                "wiki_source": "wiki/page.md",
                "wiki_chunk_index": 0,
                "citation_key": "k",
                "claim_type": "support",
                "source_filename": key,
                "source_hash": "h",
                "page_start": 0,
                "page_end": 0,
                "line_start": 0,
                "line_end": 0,
                "excerpt": "e",
                "created_at": "",
            }
        ]
    )
    store.upsert_source(key, f"hash of {key}", 1, SourceType.DOCUMENT, meta=SourceMeta(title=title))


def _holders(store, key):
    """Every (table, column) in the database with a text value equal to *key*, read row by row."""
    import pyarrow as pa

    from lilbee.data.store.lance_helpers import table_names

    found = set()
    for name in table_names(store.get_db()):
        rows = store.get_db().open_table(name).to_arrow()
        for field in rows.schema:
            if pa.types.is_string(field.type) and key in rows.column(field.name).to_pylist():
                found.add((name, field.name))
    return found


def _dump(store):
    """Every row of every table, in an order that does not depend on how it was written."""
    from lilbee.data.store.lance_helpers import table_names

    db = store.get_db()
    paths = store._all_source_paths()

    def _shown(row):
        # A new source's id is random; two libraries of one history agree on its path.
        return {
            name: paths.get(value, value) if name == "source_id" else value
            for name, value in row.items()
        }

    return {
        name: sorted(repr(_shown(row)) for row in db.open_table(name).to_arrow().to_pylist())
        for name in table_names(db)
    }


class TestMemberSources:
    def test_only_the_names_under_the_archive_are_its_members(self, store):
        names = (
            "docs.zip",
            "docs.zip/a.pdf",
            "docs.zip/in/b.txt",
            "docs.zip.bak",
            "docs_zip/c.txt",
        )
        for name in names:
            store.upsert_source(name, "h", 1)
        assert sorted(store.member_sources("docs.zip")) == ["docs.zip/a.pdf", "docs.zip/in/b.txt"]

    def test_a_quote_or_a_wildcard_in_the_name_is_matched_as_text(self, store):
        for name in ("it's 100%.zip/a.txt", "it's 100x.zip/a.txt"):
            store.upsert_source(name, "h", 1)
        assert store.member_sources("it's 100%.zip") == ["it's 100%.zip/a.txt"]

    def test_no_source_table_means_no_members(self, store):
        assert store.member_sources("docs.zip") == []

    def test_the_whole_source_table_is_never_read(self, store):
        store.upsert_source("docs.zip/a.pdf", "h", 1)
        with mock.patch.object(store, "get_sources", side_effect=AssertionError("whole read")):
            assert store.member_sources("docs.zip") == ["docs.zip/a.pdf"]


class TestSourcesOfOneSlice:
    def test_no_source_table_reads_as_no_rows_and_no_version(self, store):
        assert store.get_sources_where(lambda name: True) == []
        assert store.sources_version() is None
        assert store.sources_by_hash(["h"], version=1) == {}

    def test_only_the_accepted_rows_are_returned_across_batches(self, store, monkeypatch):
        monkeypatch.setattr("lilbee.data.store.core._SOURCE_SCAN_BATCH_ROWS", 2)
        for index in range(5):
            store.upsert_source(f"f{index}.md", f"hash{index}", 1)
        kept = store.get_sources_where(lambda name: name in {"f0.md", "f3.md", "f4.md"})
        assert sorted(row["filename"] for row in kept) == ["f0.md", "f3.md", "f4.md"]
        assert {row["file_hash"] for row in kept} == {"hash0", "hash3", "hash4"}
        assert sorted(kept[0]) == sorted(store.get_sources()[0])

    def test_a_hash_lookup_reads_the_table_as_it_was_at_the_version(self, store, monkeypatch):
        monkeypatch.setattr("lilbee.data.store.core._SOURCE_STAT_BATCH_ROWS", 2)
        store.upsert_source("a.md", "same", 1)
        store.upsert_source("b.md", "same", 1)
        store.upsert_source("c.md", "other", 1)
        store.upsert_source("pack.md", "same", 1, SourceType.IMPORTED)
        version = store.sources_version()
        store.upsert_source("later.md", "same", 1)

        found = store.sources_by_hash(["missing", "other", "same"], version=version)

        assert {digest: sorted(names) for digest, names in found.items()} == {
            "other": ["c.md"],
            "same": ["a.md", "b.md"],
        }
        latest = store.sources_by_hash(["same"], version=store.sources_version())
        assert sorted(latest["same"]) == ["a.md", "b.md", "later.md"]


class TestMissingSources:
    def test_names_with_no_source_row_are_returned(self, store, monkeypatch):
        monkeypatch.setattr("lilbee.data.store.core._SOURCE_SCAN_BATCH_ROWS", 2)
        for name in ("a.md", "b.md", "c.md"):
            store.upsert_source(name, "h", 0)
        assert store.missing_sources(["a.md", "gone.md", "c.md", "lost.md"]) == {
            "gone.md",
            "lost.md",
        }
        assert store.missing_sources([]) == set()

    def test_with_no_source_table_every_name_is_missing(self, store):
        assert store.missing_sources(["a.md"]) == {"a.md"}


class TestRelocateGivesEachMoveOneCandidate:
    def test_a_move_whose_old_row_is_gone_is_skipped(self, store):
        store.add_chunks(_one_source_records("kept.md", 1) + _one_source_records("ghost.md", 1))
        store.upsert_source("kept.md", "h1", 1)

        step = store.relocate_sources(
            [SourceMove(("ghost.md",), "new.md", None), SourceMove(("kept.md",), "moved.md", None)]
        )

        assert step == SourceRelocation({"moved.md": "kept.md"}, settled=2)
        # The skipped move re-keys nothing: its chunks keep the old name.
        assert len(store.get_chunks_by_source("ghost.md")) == 1
        assert store.get_chunks_by_source("new.md") == []
        assert len(store.get_chunks_by_source("moved.md")) == 1
        assert [row["filename"] for row in store.get_sources()] == ["moved.md"]

    def test_moves_that_share_candidates_take_one_each_in_order(self, store):
        for name in ("a.md", "b.md", "c.md"):
            store.add_chunks(_one_source_records(name, 1))
            store.upsert_source(name, "same", 1)
        shared = ("a.md", "b.md", "c.md")

        taken = store.relocate_sources(
            [SourceMove(shared, "x.md", None), SourceMove(shared, "y.md", None)]
        ).taken

        assert taken == {"x.md": "a.md", "y.md": "b.md"}
        assert sorted(row["filename"] for row in store.get_sources()) == ["c.md", "x.md", "y.md"]
        assert [len(store.get_chunks_by_source(name)) for name in ("x.md", "y.md")] == [1, 1]

    def test_a_move_takes_the_next_candidate_when_another_writer_took_the_first(self, store):
        for name in ("a.md", "b.md"):
            store.add_chunks(_one_source_records(name, 1))
            store.upsert_source(name, "same", 1)
        shared = ("a.md", "b.md")
        assert store.relocate_sources([SourceMove(shared, "first.md", None)]).taken == {
            "first.md": "a.md"
        }

        assert store.relocate_sources(
            [SourceMove(shared, "second.md", None), SourceMove(shared, "third.md", None)]
        ) == SourceRelocation({"second.md": "b.md"}, settled=2)
        assert sorted(row["filename"] for row in store.get_sources()) == ["first.md", "second.md"]
        assert store.get_chunks_by_source("third.md") == []

    def test_more_candidates_than_one_filter_holds_are_all_asked(self, store, monkeypatch):
        monkeypatch.setattr("lilbee.data.store.core._SOURCE_STAT_BATCH_ROWS", 2)
        names = [f"n{number}.md" for number in range(5)]
        for name in names:
            store.upsert_source(name, "same", 0)

        taken = store.relocate_sources(
            [SourceMove(tuple(names), f"moved{number}.md", None) for number in range(5)]
        ).taken

        assert sorted(taken.values()) == names

    def test_with_no_source_table_every_move_is_skipped(self, store):
        assert store.relocate_sources([SourceMove(("a.md",), "b.md", None)]) == SourceRelocation(
            {}, settled=1
        )
        assert store.relocate_sources([]) == SourceRelocation({}, settled=0)


def _relocate_one_update_for_each_move(store, moves):
    """One update of the source row for each move, and its chunk titles: what a hold must leave."""
    from lilbee.data.store.core import _KEEP_TITLE, _title_after_move
    from lilbee.data.title import derive_title

    sources = store.open_table(SOURCES_TABLE)
    held = {row["filename"] for row in store.get_sources()}
    for move in moves:
        old = next((name for name in move.candidates if name in held), None)
        if old is None:
            continue
        held.discard(old)
        stored = sources.search().where(f"path = '{escape_sql_string(old)}'").to_list()[0]
        title = _title_after_move(stored, old, move.new, derive_title)
        row = {"path": move.new}
        if title is not _KEEP_TITLE:
            row["title"] = title
            source_id = store.source_ids([old], SourceType.DOCUMENT)[old]
            chunks = store.open_table(CHUNKS_TABLE)
            chunks.update(where=f"source_id = '{source_id}'", values={"title": title})
        if move.stat is not None:
            row["size_bytes"] = move.stat.size_bytes
            row["mtime_ns"] = move.stat.mtime_ns
            row["stat_captured_ns"] = move.stat.captured_ns
        sources.update(where=f"path = '{escape_sql_string(old)}'", values=row)


def _relocate_in_holds(store, moves):
    """Call the store until every move is settled; the names taken and the holds it took."""
    taken, settled, holds = {}, 0, 0
    while settled < len(moves):
        step = store.relocate_sources(moves[settled:])
        assert step.settled > 0
        taken |= step.taken
        settled += step.settled
        holds += 1
    return taken, holds


def _twin(store, tmp_path):
    """A second store over a copy of *store*'s files."""
    import shutil

    twin_dir = tmp_path / "twin"
    shutil.copytree(store._config.lancedb_dir, twin_dir)
    return Store(store._config.model_copy(update={"lancedb_dir": twin_dir}))


def _vector_bytes(store):
    """The bytes of every chunk vector, by chunk text."""
    rows = store.open_table(CHUNKS_TABLE).to_arrow()
    vectors = rows.column("vector").to_numpy(zero_copy_only=False)
    return {
        text: vector.tobytes()
        for text, vector in zip(rows.column("chunk").to_pylist(), vectors, strict=True)
    }


def _folder(count):
    """Seeds and moves for *count* files under ``bulk/`` renamed to ``moved/``."""
    from lilbee.data.store.types import SourceStat

    seeds = [(f"bulk/part_{number}.md", f"part {number}") for number in range(count)]
    moves = [
        SourceMove((key,), f"moved/{key.removeprefix('bulk/')}", SourceStat(number, 2, 3))
        for number, (key, _title) in enumerate(seeds)
    ]
    return seeds, moves


_SHARED = ("copy/a.md", "copy/b.md", "copy/c.md")
# (seeds as (key, title), moves): each history a batched re-key must leave as the per-move one does.
_RELOCATE_HISTORIES = {
    "a-folder": _folder(5),
    "a-folder-with-titles-of-every-kind": (
        [
            ("bulk/old_report.md", "old report"),
            ("bulk/deep/notes-2024.md", "Frankenstein Analysis"),
            ("bulk/IMG_1234.md", None),
            ("bulky/stays.md", "stays"),
        ],
        [
            SourceMove(("bulk/old_report.md",), "moved/old_report.md", None),
            SourceMove(("bulk/deep/notes-2024.md",), "moved/deep/notes-2024.md", None),
            SourceMove(("bulk/IMG_1234.md",), "moved/IMG_1234.md", None),
        ],
    ),
    "new-file-names": (
        [
            ("old_report.md", "old report"),
            ("notes-2024.md", "Frankenstein Analysis"),
            ("real_notes.md", "real notes"),
            ("IMG_1234.md", None),
        ],
        [
            SourceMove(("old_report.md",), "annual_summary.md", None),
            SourceMove(("notes-2024.md",), "renamed.md", None),
            SourceMove(("real_notes.md",), "IMG_9999.md", None),
            SourceMove(("IMG_1234.md",), "found_title.md", None),
            SourceMove(("gone.md",), "never.md", None),
        ],
    ),
    "new-file-names-in-one-folder": (
        [(f"bulk/ledger_{number}.md", f"ledger {number}") for number in range(5)],
        [
            SourceMove((f"bulk/ledger_{number}.md",), f"bulk/paper's_{number}.md", None)
            for number in range(5)
        ],
    ),
    "one-content": (
        [(key, "Shared Title") for key in _SHARED],
        [SourceMove(_SHARED, "kept/x.md", None), SourceMove(_SHARED, "kept/y.md", None)],
    ),
    "unrelated-names": (
        [
            ("ledger/q1.md", "First Quarter"),
            ("ledger/q2.md", "q2"),
            ("it's a draft.md", "it's a draft"),
            ("IMG_1234.md", None),
            ("scans/日本.md", "Frankenstein Analysis"),
        ],
        [
            SourceMove(("ledger/q1.md",), "shelf3/7 study of topic one.md", None),
            SourceMove(("ledger/q2.md",), "attic/second quarter's.txt", None),
            SourceMove(("it's a draft.md",), "final/IMG_0001.md", None),
            SourceMove(("IMG_1234.md",), "x/found 100%_title.md", None),
            SourceMove(("scans/日本.md",), "höhe/ü.md", None),
        ],
    ),
    "quotes-and-letters-wider-than-a-byte": (
        [("it's 100%_höhe/ü ber.md", "ü ber"), ("it's 100%_höhe/日本.md", "日本")],
        [
            SourceMove(("it's 100%_höhe/ü ber.md",), "tiefe's/ü ber.md", None),
            SourceMove(("it's 100%_höhe/日本.md",), "tiefe's/日本.md", None),
        ],
    ),
}


class TestRelocateInOnePass:
    @pytest.mark.parametrize("history", sorted(_RELOCATE_HISTORIES))
    @pytest.mark.parametrize("hold_moves", [2000, 2])
    def test_a_batch_leaves_each_table_as_one_update_for_each_move_does(
        self, store, tmp_path, monkeypatch, history, hold_moves
    ):
        monkeypatch.setattr("lilbee.data.store.core._RELOCATE_HOLD_MOVES", hold_moves)
        seeds, moves = _RELOCATE_HISTORIES[history]
        for key, title in seeds:
            _seed_source(store, key, title=title)
        twin = _twin(store, tmp_path)
        before, vectors = _dump(store), _vector_bytes(store)

        _relocate_one_update_for_each_move(twin, moves)
        taken, _holds = _relocate_in_holds(store, moves)

        assert _dump(store) == _dump(twin)
        assert _dump(store) != before
        assert _vector_bytes(store) == vectors
        assert len(vectors) == len(seeds)
        assert sorted(taken) == sorted(
            {row["filename"] for row in store.get_sources()} & {move.new for move in moves}
        )

    def test_a_hold_settles_no_more_moves_than_its_bound(self, store, monkeypatch):
        monkeypatch.setattr("lilbee.data.store.core._RELOCATE_HOLD_MOVES", 2)
        seeds, moves = _folder(5)
        for key, title in seeds:
            _seed_source(store, key, title=title)

        first = store.relocate_sources(moves)

        assert first == SourceRelocation({m.new: m.candidates[0] for m in moves[:2]}, settled=2)
        left = sorted(row["filename"] for row in store.get_sources() if "bulk/" in row["filename"])
        assert left == [m.candidates[0] for m in moves[2:]]
        assert _relocate_in_holds(store, moves[2:])[1] == 2

    def test_a_move_with_no_candidate_left_is_settled_without_an_update(self, store):
        _seed_source(store, "kept.md", title="kept")
        moves = [SourceMove(("ghost.md",), f"new{number}.md", None) for number in range(3)]
        before = _dump(store)

        assert store.relocate_sources(moves) == SourceRelocation({}, settled=3)
        assert _dump(store) == before

    def test_a_move_onto_a_name_that_holds_a_row_leaves_the_moved_row_there(self, store):
        store.add_chunks(_titled_records("old.md", 1, title="old"))
        store.upsert_source("old.md", "hash of the moved file", 1)
        store.add_chunks(_titled_records("new.md", 2, title="new"))
        store.upsert_source("new.md", "hash of another file", 2)

        store.relocate_sources([SourceMove(("old.md",), "new.md", None)])

        assert [(row["filename"], row["file_hash"]) for row in store.get_sources()] == [
            ("new.md", "hash of the moved file")
        ]
        assert len(store.get_chunks_by_source("new.md")) == 1
        assert store.open_table(CHUNKS_TABLE).count_rows() == 1

    def test_two_sources_that_swap_names_both_move(self, store):
        store.upsert_source("a.md", "hash of a", 1)
        store.upsert_source("b.md", "hash of b", 1)

        store.relocate_sources(
            [SourceMove(("a.md",), "b.md", None), SourceMove(("b.md",), "a.md", None)]
        )

        assert sorted((row["filename"], row["file_hash"]) for row in store.get_sources()) == [
            ("a.md", "hash of b"),
            ("b.md", "hash of a"),
        ]

    def test_files_of_one_content_share_one_pass_over_their_old_names(self):
        from lilbee.data.store.core import _claimed_names

        class Counted(tuple):
            reads = 0

            def __getitem__(self, index):
                type(self).reads += 1
                return super().__getitem__(index)

        names = Counted(f"old{number:03d}.md" for number in range(200))
        moves = [SourceMove(names, f"new{number}.md", None) for number in range(200)]

        claimed = _claimed_names(moves, set(names[100:]))

        assert claimed == [*names[100:], *([None] * 100)]
        assert Counted.reads <= 4 * len(names)


def _path_keyed_library(config):
    """A Store over a library the layout before source ids wrote: every table keys by path."""
    return _open_after(config, _path_keyed_tables(config))


def _path_keyed_tables(config):
    """The tables of a library that keys by path, by name.

    ``a.md`` and ``b.md`` are indexed; ``wiki/concepts/x.md`` has chunks and no source
    row; ``gone.md`` was removed and a citation still names it.
    """
    dim = config.embedding_dim
    sources = [
        {**_early_source("a.md", "ha"), "size_bytes": 1, "mtime_ns": 2, "stat_captured_ns": 3},
        {**_early_source("b.md", "hb"), "size_bytes": 4, "mtime_ns": 5, "stat_captured_ns": 6},
    ]
    sources = [{**row, "title": None, "authors": None, "created_at": None} for row in sources]
    chunks = [
        {**_axis_record(key, axis, dim), "chunk_type": chunk_type, "title": None}
        for axis, (key, chunk_type) in enumerate(
            [("a.md", "raw"), ("b.md", "raw"), ("wiki/concepts/x.md", ChunkType.WIKI)]
        )
    ]
    cited = {**_citation("gone.md"), "wiki_source": "wiki/concepts/x.md"}
    return {
        SOURCES_TABLE: pa.Table.from_pylist(sources),
        CHUNKS_TABLE: pa.Table.from_pylist(chunks),
        PAGE_TEXTS_TABLE: pa.Table.from_pylist(
            [{"source": "a.md", "page": 1, "text": "t", "content_type": "text"}]
        ),
        "_citations": pa.Table.from_pylist([_citation("a.md"), cited]),
        "_wiki_mentions": pa.Table.from_pylist([{**_mention("boeing", "b.md", 1, [0])}]),
        "chunk_concepts": pa.Table.from_pylist(
            [{"chunk_source": "a.md", "chunk_index": 0, "concept": "roadmap"}]
        ),
    }


def _meta_rows(config, stamped):
    """The ``_meta`` row of a library stamped schema version *stamped*."""
    from lilbee.data.store.schema import _meta_schema

    row = {
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "schema_version": stamped,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    return pa.Table.from_pylist([row], schema=_meta_schema())


def _untimed(dump):
    """*dump* without the time each source row was written."""
    return {
        name: sorted(re.sub(r"'ingested_at': '[^']*'", "", row) for row in rows)
        for name, rows in dump.items()
    }


def _md5(text):
    import hashlib

    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()


class TestChunkBatches:
    def test_an_empty_store_yields_no_batch(self, store):
        assert list(store.chunk_batches(["source", "chunk"], 10)) == []

    def test_a_chunk_no_source_row_names_is_left_out(self, store):
        store.add_chunks(_make_records(2))
        assert [len(rows) for rows in store.chunk_batches(["source"], 10)] == [2]
        store.get_db().drop_table(SOURCES_TABLE)

        assert list(store.chunk_batches(["source", "chunk_index"], 10)) == [[]]

    def test_each_row_names_its_source_by_path(self, store):
        store.add_chunks(_make_records(3))

        batches = list(store.chunk_batches(["source", "chunk_index"], 2))

        assert [len(rows) for rows in batches] == [2, 1]
        assert sorted((r["source"], r["chunk_index"]) for rows in batches for r in rows) == [
            ("doc0.md", 0),
            ("doc1.md", 1),
            ("doc2.md", 2),
        ]


class TestOpenTimeMigration:
    """The first open keys every table by ``source_id``; nothing is embedded again."""

    def test_every_table_names_its_source_by_id_and_no_path_column_is_left(self, test_config):
        store = _path_keyed_library(test_config)
        db = store.get_db()

        for name in ("chunks", "_page_texts", "_citations", "_wiki_mentions", "chunk_concepts"):
            names = db.open_table(name).schema.names
            assert "source_id" in names
            assert not {"source", "chunk_source", "source_filename"} & set(names), name
        names = db.open_table(SOURCES_TABLE).schema.names
        assert "path" in names and "filename" not in names and "held_out" in names

    def test_an_existing_source_takes_the_md5_of_its_path(self, test_config):
        store = _path_keyed_library(test_config)

        assert store.source_ids(["a.md", "b.md"], SourceType.DOCUMENT) == {
            "a.md": _md5("a.md"),
            "b.md": _md5("b.md"),
        }

    def test_what_a_reader_sees_is_unchanged(self, test_config):
        store = _path_keyed_library(test_config)
        dim = test_config.embedding_dim

        assert [(r["filename"], r["file_hash"]) for r in store.get_sources()] == [
            ("a.md", "ha"),
            ("b.md", "hb"),
        ]
        hits = store.search(_axis_record("q", 1, dim)["vector"], top_k=1, max_distance=0)
        assert [hit.source for hit in hits] == ["b.md"]
        assert [r["source"] for r in store.get_page_texts("a.md")] == ["a.md"]
        assert [r["source"] for r in store.wiki_mention_rows()] == ["b.md"]
        assert store.wiki_chunk_sources() == {"wiki/concepts/x.md"}
        cited = store.get_citations_for_wiki("wiki/concepts/x.md")
        assert [r["source_filename"] for r in cited] == ["gone.md"]
        assert len(store.get_citations_for_source("a.md")) == 1

    def test_a_path_no_source_row_held_keeps_a_row_no_listing_shows(self, test_config):
        store = _path_keyed_library(test_config)
        rows = {r["path"]: r for r in store.open_table(SOURCES_TABLE).to_arrow().to_pylist()}

        assert (
            rows["wiki/concepts/x.md"]["source_type"],
            rows["wiki/concepts/x.md"]["held_out"],
        ) == (
            SourceType.WIKI,
            None,
        )
        assert rows["gone.md"]["held_out"] == "removed"
        assert store.count_sources() == 2
        assert store.missing_sources(["gone.md", "a.md"]) == {"gone.md"}

    def test_the_key_comparison_makes_no_row_a_python_object(self, test_config):
        from lancedb.query import LanceQueryBuilder

        store = _path_keyed_library(test_config)

        real, converted = LanceQueryBuilder.to_list, []

        def _counted(self, *args, **kwargs):
            rows = real(self, *args, **kwargs)
            converted.extend(rows)
            return rows

        with mock.patch.object(LanceQueryBuilder, "to_list", _counted):
            store.get_db()

        assert converted == []

        rows = store.open_table(SOURCES_TABLE).to_arrow().to_pylist()
        assert sorted(r["path"] for r in rows) == ["a.md", "b.md", "gone.md", "wiki/concepts/x.md"]

    def test_a_table_gains_its_id_column_while_the_next_table_is_compared(
        self, test_config, monkeypatch
    ):
        from lancedb.table import LanceTable

        import lilbee.data.store.core as core_mod

        store = _path_keyed_library(test_config)
        compared, overlapped = threading.Event(), []
        real_add, real_missing = LanceTable.add_columns, core_mod._keys_missing_from

        def _add(self, *args, **kwargs):
            if self.name == CHUNKS_TABLE:
                overlapped.append(compared.wait(timeout=5))
            return real_add(self, *args, **kwargs)

        def _missing(sources, table, column):
            if table.name == PAGE_TEXTS_TABLE:
                compared.set()
            return real_missing(sources, table, column)

        monkeypatch.setattr(LanceTable, "add_columns", _add)
        monkeypatch.setattr(core_mod, "_keys_missing_from", _missing)

        store.get_db()

        assert overlapped == [True]
        assert "source_id" in store.open_table(CHUNKS_TABLE).schema.names

    def test_the_id_columns_are_added_while_the_source_table_is_upgraded(
        self, test_config, monkeypatch
    ):
        from lancedb.table import LanceTable

        import lilbee.data.store.core as core_mod

        store = _path_keyed_library(test_config)
        adding, overlapped = threading.Event(), []
        real_add, real_sources = LanceTable.add_columns, core_mod._migrate_sources_table

        def _add(self, *args, **kwargs):
            if self.name == CHUNKS_TABLE:
                adding.set()
            return real_add(self, *args, **kwargs)

        def _sources(table):
            overlapped.append(adding.wait(timeout=5))
            real_sources(table)

        monkeypatch.setattr(LanceTable, "add_columns", _add)
        monkeypatch.setattr(core_mod, "_migrate_sources_table", _sources)

        store.get_db()

        assert overlapped == [True]
        names = store.open_table(CHUNKS_TABLE).schema.names
        assert "source_id" in names and "source" not in names
        assert _sources_of(store) == ["a.md", "b.md", "wiki/concepts/x.md"]

    @pytest.mark.parametrize("key_type", [pa.string(), pa.large_string()])
    def test_the_missing_keys_are_the_keys_no_source_row_holds(self, tmp_path, key_type):
        import lancedb

        from lilbee.data.store.core import _keys_missing_from

        keys = ["a.md", "a.md", "", "it's.md", "日本/文書.md", "held.md", "wiki/p.md", "b.md"]
        keys += ["wiki/p.md", "wiki/p.md"]
        held = ["a.md", "held.md", "c.md", ""]
        db = lancedb.connect(str(tmp_path))
        table = db.create_table("t", pa.table({"source": pa.array(keys, key_type)}))
        rows = {"path": held, "source_type": [SourceType.DOCUMENT] * len(held)}
        sources = db.create_table("s", pa.table(rows))

        missing = _keys_missing_from(sources, table, "source")

        assert [key for key, _ in missing] == sorted(set(keys) - set(held))
        assert missing == [
            (key, SourceType.DOCUMENT) for key in ["b.md", "it's.md", "wiki/p.md", "日本/文書.md"]
        ]

    def test_a_wiki_chunk_misses_a_page_row_and_any_other_key_a_document_row(self, tmp_path):
        import lancedb

        from lilbee.data.store.core import _keys_missing_from

        chunks = {
            "source": ["wiki/p.md", "wiki/p.md", "wiki/q.md", "wiki/q.md", "d.md", "e.md", "f.md"],
            "chunk_type": ["wiki", "raw", "wiki", "raw", "raw", None, "raw"],
        }
        held = {
            "path": ["wiki/p.md", "wiki/q.md", "d.md", "f.md"],
            "source_type": [SourceType.DOCUMENT, SourceType.WIKI, SourceType.WIKI, None],
        }
        db = lancedb.connect(str(tmp_path))
        table = db.create_table(CHUNKS_TABLE, pa.table(chunks))
        sources = db.create_table("s", pa.table(held))

        assert _keys_missing_from(sources, table, "source") == [
            ("wiki/p.md", SourceType.WIKI),
            ("d.md", SourceType.DOCUMENT),
            ("e.md", SourceType.DOCUMENT),
            ("wiki/q.md", SourceType.DOCUMENT),
        ]

    def test_a_key_in_many_rows_of_several_tables_gets_one_row_for_each_type(self, test_config):
        tables = _path_keyed_tables(test_config)
        pages = [
            {"source": "wiki/concepts/x.md", "page": number, "text": "t", "content_type": "text"}
            for number in range(3)
        ]
        tables[PAGE_TEXTS_TABLE] = pa.Table.from_pylist(
            tables[PAGE_TEXTS_TABLE].to_pylist() + pages
        )
        wiki = tables[CHUNKS_TABLE].to_pylist()[-1]
        chunks = [{**wiki, "chunk_index": number} for number in range(1, 3)]
        tables[CHUNKS_TABLE] = pa.Table.from_pylist(
            tables[CHUNKS_TABLE].to_pylist() + chunks, schema=tables[CHUNKS_TABLE].schema
        )
        store = _open_after(test_config, tables)

        rows = store.open_table(SOURCES_TABLE).to_arrow().to_pylist()

        assert sorted((r["path"], r["source_type"]) for r in rows) == [
            ("a.md", SourceType.DOCUMENT),
            ("b.md", SourceType.DOCUMENT),
            ("gone.md", SourceType.DOCUMENT),
            ("wiki/concepts/x.md", SourceType.DOCUMENT),
            ("wiki/concepts/x.md", SourceType.WIKI),
        ]
        assert store.source_ids(["wiki/concepts/x.md"], SourceType.WIKI) == {
            "wiki/concepts/x.md": _md5("wiki/concepts/x.md\0wiki")
        }
        assert store.source_ids(["wiki/concepts/x.md"], SourceType.DOCUMENT) == {
            "wiki/concepts/x.md": _md5("wiki/concepts/x.md")
        }

    def test_a_second_open_changes_nothing(self, test_config):
        from lilbee.data.store.lance_helpers import table_names

        _path_keyed_library(test_config)
        db = Store(test_config).get_db()
        before = {name: db.open_table(name).version for name in table_names(db)}

        db = Store(test_config).get_db()

        assert {name: db.open_table(name).version for name in table_names(db)} == before

    @pytest.mark.parametrize(
        ("stop", "nth"),
        [
            *(("add_columns", nth) for nth in range(1, 8)),
            *(("drop_columns", nth) for nth in range(1, 6)),
            ("alter_columns", 1),
            *(("merge_insert", nth) for nth in (1, 2)),
        ],
    )
    def test_an_open_stopped_part_way_is_finished_by_the_next(
        self, test_config, tmp_path, stop, nth
    ):
        from lancedb.table import LanceTable

        clean = _dump(
            _path_keyed_library(test_config.model_copy(update={"lancedb_dir": tmp_path / "clean"}))
        )
        real, calls = getattr(LanceTable, stop), []

        def _stops(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == nth:
                raise KeyboardInterrupt
            return real(self, *args, **kwargs)

        with mock.patch.object(LanceTable, stop, _stops), pytest.raises(KeyboardInterrupt):
            _path_keyed_library(test_config).get_db()

        assert _untimed(_dump(Store(test_config))) == _untimed(clean)

    def test_two_stores_that_open_one_after_the_other_agree(self, test_config):
        _path_keyed_library(test_config)

        first, second = Store(test_config), Store(test_config)

        assert first.get_sources() == second.get_sources()
        assert _dump(first) == _dump(second)

    def test_the_open_takes_the_write_lock(self, test_config, monkeypatch):
        import lancedb

        import lilbee.data.store.core as core_mod

        db = lancedb.connect(str(test_config.lancedb_dir))
        db.create_table(SOURCES_TABLE, pa.Table.from_pylist([_early_source("a.md")]))
        taken = []
        real = core_mod.write_lock

        @contextmanager
        def _lock(*args, **kwargs):
            with real(*args, **kwargs):
                taken.append(Store(test_config)._config.lancedb_dir)
                yield

        monkeypatch.setattr(core_mod, "write_lock", _lock)

        Store(test_config).get_db()

        assert taken[:1] == [test_config.lancedb_dir]

    def test_a_key_index_moves_to_the_id_and_the_path(self, test_config):
        import lancedb
        from lancedb.index import BTree

        db = lancedb.connect(str(test_config.lancedb_dir))
        for name, rows in _path_keyed_tables(test_config).items():
            db.create_table(name, rows)
        for name, column in (
            ("chunks", "source"),
            ("_page_texts", "source"),
            (SOURCES_TABLE, "filename"),
        ):
            db.open_table(name).create_index(column, config=BTree())

        db = Store(test_config).get_db()

        def _indexes(name):
            return {(i.name, tuple(i.columns)) for i in db.open_table(name).list_indices()}

        assert ("source_id_idx", ("source_id",)) in _indexes("chunks")
        assert ("source_id_idx", ("source_id",)) in _indexes("_page_texts")
        assert _indexes(SOURCES_TABLE) == {
            ("path_idx", ("path",)),
            ("source_id_idx", ("source_id",)),
        }

    @pytest.mark.parametrize(("stamped", "after"), [(2, 3), (1, 1)])
    def test_a_migrated_library_reads_schema_version_three(self, test_config, stamped, after):
        tables = _path_keyed_tables(test_config)
        tables[META_TABLE] = _meta_rows(test_config, stamped)

        store = _open_after(test_config, tables)

        assert store.get_meta()["schema_version"] == after

    def test_an_open_stopped_before_its_schema_stamp_is_stamped_by_the_next(self, test_config):
        import lancedb
        from lancedb.table import LanceTable

        tables = _path_keyed_tables(test_config)
        tables[META_TABLE] = _meta_rows(test_config, 2)
        real = LanceTable.update

        def _stops(self, *args, **kwargs):
            if self.name == META_TABLE:
                raise KeyboardInterrupt
            return real(self, *args, **kwargs)

        with mock.patch.object(LanceTable, "update", _stops), pytest.raises(KeyboardInterrupt):
            _open_after(test_config, tables).get_db()
        db = lancedb.connect(str(test_config.lancedb_dir))
        left = {name: set(db.open_table(name).schema.names) for name in tables}
        assert not any({"source", "chunk_source", "source_filename"} & n for n in left.values())
        assert db.open_table(META_TABLE).to_arrow().column("schema_version").to_pylist() == [2]

        assert Store(test_config).get_meta()["schema_version"] == 3

    def test_the_stamped_schema_version_is_three(self):
        from lilbee.data.store.types import META_SCHEMA_VERSION

        assert META_SCHEMA_VERSION == 3


def _shared_key_tables(config):
    """A library keyed by path where a document and a wiki page share ``wiki/concepts/x.md``."""
    tables = _path_keyed_tables(config)
    shared = "wiki/concepts/x.md"
    sources = tables[SOURCES_TABLE].to_pylist()
    tables[SOURCES_TABLE] = pa.Table.from_pylist([*sources, {**sources[0], "filename": shared}])
    document = {**_axis_record(shared, 3, config.embedding_dim), "title": None}
    tables[CHUNKS_TABLE] = pa.Table.from_pylist(
        [*tables[CHUNKS_TABLE].to_pylist(), document], schema=tables[CHUNKS_TABLE].schema
    )
    return tables


def _bodies(store, source_type):
    """The text of each chunk of ``wiki/concepts/x.md`` as a source of *source_type*."""
    source_id = store.source_ids(["wiki/concepts/x.md"], source_type)["wiki/concepts/x.md"]
    rows = store.open_table(CHUNKS_TABLE).search().where(f"source_id = '{source_id}'")
    return sorted(row["chunk"] for row in rows.limit(None).to_list())


class TestAnUpgradeTypesWikiChunks:
    """A document and a wiki page that share a key keep two rows and two ids at an upgrade."""

    def test_the_page_and_the_document_each_keep_their_own_row_and_chunks(self, test_config):
        store = _open_after(test_config, _shared_key_tables(test_config))
        key = "wiki/concepts/x.md"

        page = store.source_ids([key], SourceType.WIKI)
        document = store.source_ids([key], SourceType.DOCUMENT)

        assert page == {key: _md5(f"{key}\0wiki")}
        assert document == {key: _md5(key)}
        assert _bodies(store, SourceType.WIKI) == [f"{key} body"]
        assert _bodies(store, SourceType.DOCUMENT) == [f"{key} body"]
        assert [r["filename"] for r in store.get_sources()] == ["a.md", "b.md", key]

    def test_a_page_written_again_leaves_no_old_body_and_the_document_keeps_its(self, test_config):
        store = _open_after(test_config, _shared_key_tables(test_config))
        key = "wiki/concepts/x.md"
        page = {**_axis_record(key, 4, test_config.embedding_dim), "chunk_type": ChunkType.WIKI}

        store.replace_chunks(key, ChunkType.WIKI, [{**page, "chunk": "new body"}])

        assert _bodies(store, SourceType.WIKI) == ["new body"]
        assert _bodies(store, SourceType.DOCUMENT) == [f"{key} body"]

    def test_an_upgrade_killed_after_the_id_column_is_finished_by_the_next(
        self, test_config, tmp_path
    ):
        from lancedb.table import LanceTable

        whole = _open_after(
            test_config.model_copy(update={"lancedb_dir": tmp_path / "whole"}),
            _shared_key_tables(test_config),
        )
        whole.get_db()
        killed = test_config.model_copy(update={"lancedb_dir": tmp_path / "killed"})
        real = LanceTable.update

        def _dies_typing_pages(self, *args, **kwargs):
            if kwargs.get("where") == "chunk_type = 'wiki'":
                raise KeyboardInterrupt
            return real(self, *args, **kwargs)

        with (
            mock.patch.object(LanceTable, "update", _dies_typing_pages),
            pytest.raises(KeyboardInterrupt),
        ):
            _open_after(killed, _shared_key_tables(test_config)).get_db()
        import lancedb

        half = lancedb.connect(str(killed.lancedb_dir)).open_table(CHUNKS_TABLE).schema.names
        assert {"source", "source_id"} <= set(half)

        store = Store(killed)

        assert _untimed(_dump(store)) == _untimed(_dump(whole))
        assert store.source_ids(["wiki/concepts/x.md"], SourceType.WIKI) == {
            "wiki/concepts/x.md": _md5("wiki/concepts/x.md\0wiki")
        }


class TestSourcesWithoutChunks:
    def test_a_drop_that_stops_part_way_leaves_no_source_row(self, store):
        """Rows of every other table can stay: a file with no source row is indexed again."""
        _seed_source(store, "a.md")
        database = store.get_db()
        real, dropped = type(database).drop_table, []

        def _stops_after_one(db, name, *args, **kwargs):
            if dropped:
                raise KeyboardInterrupt
            dropped.append(name)
            return real(db, name, *args, **kwargs)

        with (
            mock.patch.object(type(database), "drop_table", _stops_after_one),
            pytest.raises(KeyboardInterrupt),
        ):
            store.drop_all()

        assert dropped == [SOURCES_TABLE]
        assert store.open_table(CHUNKS_TABLE).count_rows() == 1
        assert store.get_sources() == []


class TestSourcesIngestedSince:
    def test_sources_ingested_since_a_time(self, store):
        assert store.sources_ingested_since("") == []
        store.upsert_source("early.md", "h", 1)
        between = datetime.now(UTC).isoformat()
        store.upsert_source("late.md", "h", 1)

        assert store.sources_ingested_since(between) == ["late.md"]
        assert sorted(store.sources_ingested_since("")) == ["early.md", "late.md"]

    def test_a_source_written_at_the_very_time_is_one_of_them(self, store):
        """A checkpoint and a row can carry one time, and the row may be the later of the two."""
        store.upsert_source("same.md", "h", 1)
        [row] = store.get_sources()

        assert store.sources_ingested_since(row["ingested_at"]) == ["same.md"]


def _watch_write_lock(store, monkeypatch) -> list[int]:
    """Wrap the write lock of *store*; the list it returns is non-empty while the lock is held."""
    held: list[int] = []
    real_lock = store._write_lock

    @contextmanager
    def _lock(*args, **kwargs):
        with real_lock(*args, **kwargs):
            held.append(1)
            try:
                yield
            finally:
                held.pop()

    monkeypatch.setattr(store, "_write_lock", _lock)
    return held


class TestSourcesCheckpoint:
    def test_a_row_written_after_a_checkpoint_is_at_or_after_its_time_and_off_its_version(
        self, store
    ):
        store.upsert_source("early.md", "h", 1)

        checkpoint = store.sources_checkpoint()
        store.upsert_source("late.md", "h", 1)

        assert checkpoint.version is not None
        assert store.sources_version() > checkpoint.version
        assert store.sources_ingested_since(checkpoint.time) == ["late.md"]

    def test_with_no_source_table_the_version_is_none(self, store):
        assert store.sources_checkpoint().version is None

    def test_a_checkpoint_is_read_under_the_write_lock(self, store, monkeypatch):
        """Read beside a flush, its time could pass the stamp of a row not yet committed."""
        held, read_under_lock = _watch_write_lock(store, monkeypatch), []
        real_version = store.sources_version

        def _version():
            read_under_lock.append(bool(held))
            return real_version()

        monkeypatch.setattr(store, "sources_version", _version)
        store.upsert_source("a.md", "h", 1)

        checkpoint = store.sources_checkpoint()

        assert read_under_lock == [True]
        assert checkpoint.version == real_version()

    @pytest.mark.parametrize("write", ["upsert_source", "write_chunks_batch"])
    def test_a_source_row_is_stamped_under_the_write_lock(self, store, monkeypatch, write):
        """A stamp taken before the lock could be older than a checkpoint its row commits after."""
        held, stamped_under_lock = _watch_write_lock(store, monkeypatch), []
        real_row = store._source_row

        def _row(*args, **kwargs):
            stamped_under_lock.append(bool(held))
            return real_row(*args, **kwargs)

        monkeypatch.setattr(store, "_source_row", _row)

        if write == "upsert_source":
            store.upsert_source("a.md", "h", 1)
        else:
            item = ChunkWrite(source="a.md", file_hash="h", records=[], needs_cleanup=False)
            store.write_chunks_batch([item])

        # A new source of a flush takes a held-out row first, then its listed one.
        assert stamped_under_lock == ([True] if write == "upsert_source" else [True, True])
        assert [row["filename"] for row in store.get_sources()] == ["a.md"]


class TestWikiRefreshed:
    def test_no_record_reads_as_none_and_logs_nothing(self, store, caplog):
        with caplog.at_level(logging.DEBUG):
            assert store.wiki_refreshed() is None
        assert caplog.records == []

    def test_a_write_that_fails_leaves_the_old_record_and_no_other_file(self, store, monkeypatch):
        beside_the_tables = store._config.lancedb_dir
        beside_the_tables.mkdir(parents=True, exist_ok=True)
        old = SourcesCheckpoint("2026-01-01T00:00:00+00:00", 7)
        store.record_wiki_refreshed(old)
        before = sorted(path.name for path in beside_the_tables.iterdir())
        assert ".wiki_refreshed" in before
        monkeypatch.setattr(os, "replace", mock.Mock(side_effect=OSError("the disk is full")))

        with pytest.raises(OSError, match="the disk is full"):
            store.record_wiki_refreshed(SourcesCheckpoint("2026-02-02T00:00:00+00:00", 9))

        assert sorted(path.name for path in beside_the_tables.iterdir()) == before
        assert store.wiki_refreshed() == old

    @pytest.mark.parametrize("version", [7, None])
    def test_a_recorded_checkpoint_is_read_back(self, store, version):
        beside_the_tables = store._config.lancedb_dir
        beside_the_tables.mkdir(parents=True, exist_ok=True)
        checkpoint = SourcesCheckpoint("2026-01-01T00:00:00+00:00", version)

        store.record_wiki_refreshed(checkpoint)

        assert store.wiki_refreshed() == checkpoint
        assert [path.name for path in beside_the_tables.glob(".wiki_refreshed*")] == [
            ".wiki_refreshed"
        ]

    @pytest.mark.parametrize(
        "text", ["", "{", "[]", '"text"', '{"time": "t"}', '{"time": "t", "version": "x"}']
    )
    def test_an_unreadable_record_is_reported_once_and_deleted(self, store, text, caplog):
        store._config.lancedb_dir.mkdir(parents=True, exist_ok=True)
        record = store._config.lancedb_dir / ".wiki_refreshed"
        record.write_text(text, encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            assert store.wiki_refreshed() is None
            assert not record.exists()
            assert store.wiki_refreshed() is None

        assert [(r.levelname, str(record) in r.getMessage()) for r in caplog.records] == [
            ("WARNING", True)
        ]

    def test_a_record_that_cannot_be_deleted_is_reported_at_each_read(self, store, caplog):
        record = store._config.lancedb_dir / ".wiki_refreshed"
        record.mkdir(parents=True)

        with caplog.at_level(logging.WARNING):
            assert store.wiki_refreshed() is None
            assert store.wiki_refreshed() is None

        assert record.is_dir()
        assert [r.levelname for r in caplog.records] == ["WARNING", "WARNING"]

    def test_a_forgotten_record_reads_as_none_and_forgetting_none_is_no_error(self, store):
        store.upsert_source("a.md", "h", 1)
        store.record_wiki_refreshed(store.sources_checkpoint())

        store.forget_wiki_refreshed()
        store.forget_wiki_refreshed()

        assert store.wiki_refreshed() is None
        assert store.sources_version() is not None

    def test_a_rebuild_drops_the_record_with_the_tables(self, store):
        store.upsert_source("a.md", "h", 1)
        store.record_wiki_refreshed(store.sources_checkpoint())
        assert store.wiki_refreshed() is not None

        store.drop_all()

        assert store.wiki_refreshed() is None
        assert store.sources_version() is None


class TestATableAnotherProcessIsCreating:
    """The listing names a table once its folder exists, before its first version does."""

    @pytest.fixture()
    def half_made(self, test_config):
        store = Store(test_config)
        store.get_db()

        def make(name):
            (test_config.lancedb_dir / f"{name}.lance").mkdir()
            assert name in table_names(store.get_db()), "the folder must be listed"
            with pytest.raises(ValueError, match="was not found"):
                store.get_db().open_table(name)
            return store

        return make

    def test_a_second_process_opens_the_library(self, half_made, test_config):
        half_made("_sources")

        assert Store(test_config).get_db() is not None

    def test_the_table_reads_as_absent(self, half_made):
        store = half_made("_sources")

        assert store.open_table("_sources") is None
        assert store.source_ids(["a.md"], SourceType.DOCUMENT) == {}

    def test_the_store_identity_reads_as_unset(self, half_made):
        assert half_made(META_TABLE).get_meta() is None


class TestATableThatCannotBeRead:
    """A table that exists but fails to open for a real reason is an error, not an absent table."""

    @staticmethod
    def _garble(store, config, name):
        import pyarrow as pa

        store.get_db().create_table(name, schema=pa.schema([("a", pa.string())]))
        versions = config.lancedb_dir / f"{name}.lance" / "_versions"
        manifests = [path for path in versions.iterdir() if path.suffix == ".manifest"]
        assert manifests, "the table must have a manifest to garble"
        for manifest in manifests:
            manifest.write_bytes(b"garbage")

    def test_the_helper_raises(self, store, test_config):
        self._garble(store, test_config, "scratch")

        with pytest.raises(RuntimeError, match="does not have sufficient data"):
            open_committed(Store(test_config).get_db(), "scratch")

    def test_the_store_raises(self, store, test_config):
        self._garble(store, test_config, "scratch")

        with pytest.raises(RuntimeError, match="does not have sufficient data"):
            Store(test_config).open_table("scratch")


def _citation(source, key="src1", **changes):
    return {
        "wiki_source": "wiki/entities/boeing.md",
        "wiki_chunk_index": 0,
        "citation_key": key,
        "claim_type": "fact",
        "source_filename": source,
        "source_hash": "h",
        "page_start": 0,
        "page_end": 0,
        "line_start": 1,
        "line_end": 2,
        "excerpt": "it's quoted",
        "created_at": "",
        **changes,
    }
