"""Tests for LanceDB store operations: hybrid search + FTS index lifecycle."""

from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest

from lilbee.core.config import CHUNKS_TABLE, META_TABLE, cfg
from lilbee.data.store import (
    ChunkType,
    CitationRecord,
    SearchChunk,
    SearchScope,
    SourceMeta,
    SourceType,
    Store,
    cosine_sim,
    escape_sql_string,
    mmr_rerank,
    scope_to_chunk_type,
)
from lilbee.runtime.lock import write_lock
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

        real = core_mod._safe_delete_unlocked

        def spy_delete(table, predicate):
            locked_during.append(_write_mutex.locked())
            return real(table, predicate)

        monkeypatch.setattr(core_mod, "_safe_delete_unlocked", spy_delete)
        store.clear_and_add("t_lock", schema, [{"concept": "next"}], "concept IS NOT NULL")
        assert locked_during == [True]  # delete ran under the write lock

    def test_skips_add_when_delete_fails(self, store, monkeypatch):
        import pyarrow as pa

        import lilbee.data.store.core as core_mod

        schema = pa.schema([pa.field("concept", pa.utf8())])
        store.clear_and_add("t_fail", schema, [{"concept": "old"}], "concept IS NOT NULL")
        # A failed delete must not add the new rows (would duplicate the stale ones).
        monkeypatch.setattr(core_mod, "_safe_delete_unlocked", lambda table, predicate: False)
        store.clear_and_add("t_fail", schema, [{"concept": "new"}], "concept IS NOT NULL")
        rows = store.open_table("t_fail").search().to_list()
        assert {r["concept"] for r in rows} == {"old"}  # unchanged; new rows not added


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
        assert [c.args[0] for c in spy.call_args_list] == ["source", "chunk_type"]
        kinds = {c.args[0]: c.kwargs.get("index_type") for c in spy.call_args_list}
        assert kinds == {"source": "BTREE", "chunk_type": "BITMAP"}
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

    def test_has_scalar_index_is_false_when_listing_raises(self, store):
        from lilbee.data.store.lance_helpers import _has_scalar_index

        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        assert table is not None
        with mock.patch.object(type(table), "list_indices", side_effect=RuntimeError("boom")):
            assert _has_scalar_index(table, "source") is False

    def test_indexes_chunk_concepts_source_column(self, store):
        """The concept-boost path filters chunk_concepts by chunk_source per
        result, so that column gets its own BTree index too."""
        from lilbee.core.config import CHUNK_CONCEPTS_TABLE
        from lilbee.data.store import ensure_table
        from lilbee.data.store.lance_helpers import _has_scalar_index
        from lilbee.retrieval.concepts.schema import _chunk_concepts_schema

        store.add_chunks(_make_records())
        cc = ensure_table(store.get_db(), CHUNK_CONCEPTS_TABLE, _chunk_concepts_schema())
        cc.add([{"chunk_source": "doc.md", "chunk_index": 0, "concept": "alpha"}])
        store.ensure_scalar_indexes()
        assert _has_scalar_index(store.open_table(CHUNK_CONCEPTS_TABLE), "chunk_source")

    def test_one_column_failure_does_not_skip_the_other(self, store):
        """A BTree failure on 'source' must not skip the independent Bitmap on
        'chunk_type' -- each column gets its own try."""
        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        attempted = []

        def _record(self, column, **kwargs):
            attempted.append(column)
            if column == "source":
                raise RuntimeError("boom")

        with mock.patch.object(type(table), "create_scalar_index", _record):
            store.ensure_scalar_indexes()
        assert attempted == ["source", "chunk_type"]

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
        store._ensure_scalar_index_on("no_such_table", (("source", "BTREE"),))  # must not raise

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
        from lilbee.data.store.lance_helpers import _has_scalar_index

        table = store.open_table("chunks")
        assert _has_scalar_index(table, "source")
        assert _has_scalar_index(table, "chunk_type")

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

    def test_fts_ensure_body_is_a_noop_without_a_chunks_table(self, store):
        store._ensure_fts_index_unlocked()
        assert store._fts_ready is False

    def test_scalar_index_builds_after_concepts_table_appears(self, store, test_config):
        """Serve ordering: chunk_concepts is created after the first search.
        The next search must still index it instead of latching ready early."""
        from lilbee.data.store.lance_helpers import _has_scalar_index, ensure_table
        from lilbee.retrieval.concepts.schema import _chunk_concepts_schema

        store.add_chunks(_make_records())
        store.search([0.5] * test_config.embedding_dim, top_k=3)
        assert store._scalar_ready is False  # concepts table not there yet
        table = ensure_table(store.get_db(), "chunk_concepts", _chunk_concepts_schema())
        table.add([{"chunk_source": "doc0.md", "chunk_index": 0, "concept": "x"}])
        store.search([0.5] * test_config.embedding_dim, top_k=3)
        assert _has_scalar_index(store.open_table("chunk_concepts"), "chunk_source")
        assert store._scalar_ready is True


class TestEnsureVectorIndex:
    """Small vaults stay on exact flat search; large ones get an ANN index."""

    _INDEXABLE = 256  # enough rows for IVF_PQ to train under default params

    def test_noop_when_no_table(self, store):
        assert store.ensure_vector_index() is False

    def test_below_threshold_keeps_flat_search(self, store, test_config):
        from lilbee.data.store.lance_helpers import _has_vector_index

        store.add_chunks(_make_records())  # 3 rows, threshold defaults to 50_000
        assert store.ensure_vector_index() is False
        table = store.open_table("chunks")
        assert _has_vector_index(table) is False
        # Flat search still serves results without an ANN index.
        assert store.search([0.5] * test_config.embedding_dim, top_k=3)

    def test_threshold_zero_disables_build(self, store, test_config):
        from lilbee.data.store.lance_helpers import _has_vector_index

        test_config.ann_index_threshold = 0
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        assert store.ensure_vector_index() is False
        assert _has_vector_index(store.open_table("chunks")) is False

    def test_builds_index_above_threshold(self, store, test_config):
        import math

        from lilbee.data.store.lance_helpers import _has_vector_index

        test_config.ann_index_threshold = 50
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        assert store.ensure_vector_index() is True
        assert _has_vector_index(store.open_table("chunks")) is True
        # Search still finds the chunk whose vector matches the query (nprobes/refine).
        query = [math.sin(5 * 0.1 + j * 0.01) for j in range(test_config.embedding_dim)]
        results = store.search(query, top_k=3)
        assert results
        assert results[0].source == "doc5.md"

    def test_force_builds_below_threshold(self, store, test_config):
        from lilbee.data.store.lance_helpers import _has_vector_index

        test_config.ann_index_threshold = 1_000_000
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        assert store.ensure_vector_index() is False  # below threshold, no force
        assert store.ensure_vector_index(force=True) is True
        assert _has_vector_index(store.open_table("chunks")) is True

    def test_optimizes_when_index_exists(self, store, test_config):
        test_config.ann_index_threshold = 50
        store.add_chunks(_make_indexable_records(self._INDEXABLE, test_config.embedding_dim))
        store.ensure_vector_index()
        table = store.open_table("chunks")
        with mock.patch.object(type(table), "optimize") as optimize_spy:
            assert store.ensure_vector_index() is True
        optimize_spy.assert_called_once()

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

    def test_has_vector_index_swallows_list_indices_errors(self, store):
        from lilbee.data.store.lance_helpers import _has_vector_index

        store.add_chunks(_make_records())
        table = store.open_table("chunks")
        with mock.patch.object(
            type(table), "list_indices", side_effect=RuntimeError("backend down")
        ):
            assert _has_vector_index(table) is False


class TestHasFtsIndex:
    def test_returns_false_on_fresh_table(self, store):
        store.add_chunks(_make_records())
        from lilbee.data.store.lance_helpers import _has_fts_index

        table = store.open_table("chunks")
        assert table is not None
        assert _has_fts_index(table) is False

    def test_returns_true_after_create(self, store):
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        from lilbee.data.store.lance_helpers import _has_fts_index

        table = store.open_table("chunks")
        assert table is not None
        assert _has_fts_index(table) is True

    def test_returns_false_on_list_indices_error(self, store):
        store.add_chunks(_make_records())
        from lilbee.data.store.lance_helpers import _has_fts_index

        table = store.open_table("chunks")
        assert table is not None
        with mock.patch.object(type(table), "list_indices", side_effect=RuntimeError("boom")):
            assert _has_fts_index(table) is False


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
            "_delete_by_sources_unlocked",
            autospec=True,
            side_effect=core_mod.Store._delete_by_sources_unlocked,
        ) as spy:
            store.write_chunks_batch(items)
        assert spy.call_count == 1
        assert sorted(spy.call_args.args[1]) == [f"f{i}.md" for i in range(total)]
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
        chain.metric.return_value = chain
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
            mock_del.assert_called_once_with(["a.md"])

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
            mock.patch.object(store, "_delete_by_sources_unlocked") as mock_chunks,
            mock.patch.object(store, "open_table", return_value=None),
            mock.patch("lilbee.data.store.core.write_lock", _tracking_lock),
        ):
            result = store.remove_documents(["a.md", "b.md"])

        assert result.removed == ["a.md", "b.md"]
        mock_chunks.assert_called_once_with(["a.md", "b.md"])
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
        store.clear_table("chunks", "source = 'doc0.md'")
        table = store.open_table("chunks")
        remaining = table.to_arrow()
        assert len(remaining) == 0

    def testclear_table_nonexistent_table_is_noop(self, store):
        store.clear_table("nonexistent", "source = 'doc0.md'")


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

        store.delete_by_source(backslash_src)

        rows = store.open_table("chunks").search().to_list()
        sources = {r["source"] for r in rows}
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
        import pyarrow as pa

        db = store.get_db()
        db.create_table(
            "_sources",
            pa.table({"filename": pa.array(["old.pdf"]), "file_hash": pa.array(["h"])}),
        )
        arrow = store.sources_arrow()
        assert arrow.schema.names == ["source", "title", "authors", "created_at"]
        assert arrow.to_pylist() == [
            {"source": "old.pdf", "title": None, "authors": None, "created_at": None}
        ]

    def test_delete_by_source_removes_page_texts(self, store):
        store.add_chunks(_make_records(n=1))
        store.add_page_texts(_page_rows("doc0.md", (1,)))
        store.delete_by_source("doc0.md")
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


class TestTableNamesAttributeError:
    def test_fallback_to_list_when_no_tables_attr(self, store):
        """table_names falls back to list() when result has no .tables attribute."""
        from lilbee.data.store.lance_helpers import table_names

        mock_db = mock.MagicMock()
        mock_db.list_tables.return_value = ["chunks", "sources"]
        result = table_names(mock_db)
        assert result == ["chunks", "sources"]


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
        import threading

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
        import threading

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
        from lilbee.data.store import SourceStat, ensure_table, source_stat

        legacy_schema = pa.schema(
            [
                pa.field("filename", pa.utf8()),
                pa.field("file_hash", pa.utf8()),
                pa.field("ingested_at", pa.utf8()),
                pa.field("chunk_count", pa.int32()),
                pa.field("source_type", pa.utf8()),
            ]
        )
        table = ensure_table(store.get_db(), SOURCES_TABLE, legacy_schema)
        table.add(
            [
                {
                    "filename": "old.md",
                    "file_hash": "h",
                    "ingested_at": "",
                    "chunk_count": 1,
                    "source_type": "document",
                }
            ]
        )

        # First write through the new path migrates the table and lands the stat.
        store.upsert_source("new.md", "h2", 1, stat=SourceStat(5, 6))
        records = {r["filename"]: r for r in store.get_sources()}
        assert source_stat(records["old.md"]) is None  # backfilled sentinel
        assert source_stat(records["new.md"]) == SourceStat(5, 6)

    def test_update_source_stats_batches_one_delete_one_add(self, store):
        from lilbee.data.store import SourceStat, SourceStatBackfill, source_stat

        store.upsert_source("a.md", "ha", 1)
        store.upsert_source("b.md", "hb", 2)
        records = {r["filename"]: r for r in store.get_sources()}

        backfills = [
            SourceStatBackfill(records["a.md"], SourceStat(1, 2)),
            SourceStatBackfill(records["b.md"], SourceStat(3, 4)),
        ]
        with mock.patch.object(
            store, "_replace_source_rows_unlocked", wraps=store._replace_source_rows_unlocked
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
                store, "_replace_source_rows_unlocked", wraps=store._replace_source_rows_unlocked
            ) as spy,
        ):
            store.update_source_stats(backfills)
        assert spy.call_count == math.ceil(total / batch_rows)
        assert all(len(call.args[0]) <= batch_rows for call in spy.call_args_list)

        records = {r["filename"]: r for r in store.get_sources()}
        assert len(records) == total
        for i in range(total):
            assert source_stat(records[f"f{i}.md"]) == SourceStat(i, i + 1)
            assert records[f"f{i}.md"]["file_hash"] == f"h{i}"

    def test_update_source_stats_empty_is_noop(self, store):
        store.update_source_stats([])
        assert store.get_sources() == []


class TestBatchedSourceUpserts:
    """One flush of N files produces one delete + one add on the sources table."""

    def test_write_chunks_batch_single_source_table_pass(self, store):
        from lilbee.data.store import ChunkWrite

        items = [
            ChunkWrite(f"f{i}.md", f"h{i}", _records_for(f"f{i}.md", 1), needs_cleanup=False)
            for i in range(5)
        ]
        with mock.patch.object(
            store, "_replace_source_rows_unlocked", wraps=store._replace_source_rows_unlocked
        ) as spy:
            store.write_chunks_batch(items)
        spy.assert_called_once()
        assert len(spy.call_args.args[0]) == 5
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
        cc_table = ensure_table(store.get_db(), CHUNK_CONCEPTS_TABLE, _chunk_concepts_schema())
        cc_table.add(
            [
                {"chunk_source": "doc.md", "chunk_index": 0, "concept": "alpha"},
                {"chunk_source": "other.md", "chunk_index": 0, "concept": "beta"},
            ]
        )

        store.delete_by_source("doc.md")
        cc_table.checkout_latest()  # bypass the read-consistency interval
        remaining = cc_table.search().limit(None).to_list()
        assert [r["chunk_source"] for r in remaining] == ["other.md"]
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


def _create_pre_title_chunks_table(store):
    """Create the chunks table with the pre-title schema, as an old index would have."""
    import pyarrow as pa

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
    return store.get_db().create_table("chunks", schema=schema)


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
        rows = store._title_arm(table, "zebra", 5, None)
        assert [(r.source, r.chunk_index) for r in rows] == [("a.pdf", 0)]
        # Deterministic across repeated calls (no implementation-defined tie order).
        again = store._title_arm(table, "zebra", 5, None)
        assert [(r.source, r.chunk_index) for r in again] == [("a.pdf", 0)]

    def test_title_arm_one_row_per_matched_document(self, store, test_config):
        """Two matched documents surface one representative row each, not an
        arbitrary flood of one document's chunks."""
        test_config.title_search = True
        store.add_chunks(_titled_records("zebra.pdf", 4, title="zebra"))
        store.add_chunks(_titled_records("safari.pdf", 4, title="zebra safari"))
        store.ensure_fts_index()
        table = store.open_table("chunks")
        rows = store._title_arm(table, "zebra", 5, None)
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
        rows = store._title_arm(table, "zebra", 2, None)
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
        from lilbee.data.store.lance_helpers import _has_fts_index

        test_config.title_search = True  # the title index is built only when enabled
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        table = store.open_table("chunks")
        assert _has_fts_index(table, "title")
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
        table = _create_pre_title_chunks_table(store)
        table.add(_make_records())
        store.ensure_fts_index()
        assert store._title_arm(table, "chunk", 5, None) == []
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=3, max_distance=0, query_text="chunk number")
        assert results
        assert all(r.title is None for r in results)

    def test_add_chunks_evolves_pre_title_table(self, store):
        """A write to an old index adds the title column in place."""
        table = _create_pre_title_chunks_table(store)
        table.add(_make_records())
        assert "title" not in table.schema.names
        store.add_chunks(_titled_records("new.pdf", 1, title="fresh document"))
        assert "title" in store.open_table("chunks").schema.names
        rows = store.get_chunks_by_source("new.pdf")
        assert [r.title for r in rows] == ["fresh document"]

    def test_title_search_enable_at_runtime_builds_index_on_next_query(self, store, test_config):
        """Enabling title_search after _fts_ready latched builds the title
        index on the next query instead of no-opping until restart."""
        from lilbee.data.store.lance_helpers import _has_fts_index

        test_config.title_search = False
        store.add_chunks(_titled_records("a.pdf", 2, title="zebra manifesto"))
        store.ensure_fts_index()
        assert not _has_fts_index(store.open_table("chunks"), "title")
        test_config.title_search = True
        store.search([0.1] * test_config.embedding_dim, top_k=3, query_text="zebra")
        assert _has_fts_index(store.open_table("chunks"), "title")

    def test_backfill_failure_keeps_nulls_and_warns(self, store, caplog):
        """A failing backfill degrades to the old NULL-title behavior."""
        import logging

        table = _create_pre_title_chunks_table(store)
        records = _make_records()
        records[0]["source"] = "project_falcon_notes.pdf"
        table.add(records)
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

        table = _create_pre_title_chunks_table(store)
        records = _make_records()
        records[0]["source"] = "project_falcon_notes.pdf"
        table.add(records)
        with caplog.at_level(logging.INFO):
            store.add_chunks(_titled_records("new.pdf", 1, title="fresh document"))
        assert any("lilbee rebuild" in r.message for r in caplog.records)
        rows = store.open_table("chunks").search().select(["source", "title"]).to_list()
        titles = {r["source"]: r["title"] for r in rows}
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

        old_schema = pa.schema(
            [
                pa.field("filename", pa.utf8()),
                pa.field("file_hash", pa.utf8()),
                pa.field("ingested_at", pa.utf8()),
                pa.field("chunk_count", pa.int32()),
                pa.field("source_type", pa.utf8()),
            ]
        )
        table = store.get_db().create_table("_sources", schema=old_schema)
        table.add(
            [
                {
                    "filename": "old.pdf",
                    "file_hash": "h0",
                    "ingested_at": "2020-01-01T00:00:00+00:00",
                    "chunk_count": 1,
                    "source_type": "document",
                }
            ]
        )
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

        store.relocate_sources([("old/a.md", "new/a.md", SourceStat(10, 20, 30))])

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
        from lilbee.core.config import CHUNKS_TABLE, CITATIONS_TABLE, PAGE_TEXTS_TABLE
        from lilbee.data.store import SourceType
        from lilbee.data.store.types import SourceStat

        records = _make_records(n=1)
        records[0]["source"] = "old/a.md"
        store.add_chunks(records)
        store.add_page_texts(
            [{"source": "old/a.md", "page": 1, "text": "t", "content_type": "text"}]
        )
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

        store.relocate_sources([("old/a.md", "new/a.md", SourceStat(1, 2, 3))])

        chunks = store.open_table(CHUNKS_TABLE)
        pages = store.open_table(PAGE_TEXTS_TABLE)
        cites = store.open_table(CITATIONS_TABLE)
        assert chunks.count_rows("source = 'new/a.md'") == 1
        assert chunks.count_rows("source = 'old/a.md'") == 0
        assert pages.count_rows("source = 'new/a.md'") == 1
        assert pages.count_rows("source = 'old/a.md'") == 0
        assert cites.count_rows("source_filename = 'new/a.md'") == 1
        assert cites.count_rows("source_filename = 'old/a.md'") == 0

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
        store.relocate_sources([("old_report.md", "annual_summary.md", None)])
        row = next(s for s in store.get_sources() if s["filename"] == "annual_summary.md")
        assert row["title"] == "annual summary"
        chunks = store.get_chunks_by_source("annual_summary.md")
        assert all(c.title == "annual summary" for c in chunks)

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
        store.relocate_sources([("notes-2024.md", "renamed.md", None)])
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
        store.relocate_sources([("real_notes.md", "IMG_1234.md", None)])
        row = next(s for s in store.get_sources() if s["filename"] == "IMG_1234.md")
        assert row["title"] is None

    def test_relocated_title_guard_branches(self, store):
        """No sources table, a failing read, and a missing row all keep the title."""
        from lilbee.data.store.core import _KEEP_TITLE
        from lilbee.data.title import derive_title

        assert store._relocated_title(None, "a.md", "b.md", derive_title) is _KEEP_TITLE
        broken = mock.MagicMock()
        broken.search.side_effect = RuntimeError("boom")
        assert store._relocated_title(broken, "a.md", "b.md", derive_title) is _KEEP_TITLE
        from lilbee.data.store import SourceType

        store.upsert_source("real.md", "h1", 1, SourceType.DOCUMENT)
        table = store.open_table("_sources")
        assert store._relocated_title(table, "absent.md", "b.md", derive_title) is _KEEP_TITLE
