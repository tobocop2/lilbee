"""Tests for LanceDB store operations — hybrid search + FTS index lifecycle."""

from unittest import mock

import pytest

from lilbee import store
from lilbee.config import cfg


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Point store at a temp directory and reset FTS flag."""
    original = cfg.lancedb_dir
    cfg.lancedb_dir = tmp_path / "lancedb_test"
    store._fts.ready = False
    yield
    cfg.lancedb_dir = original
    store._fts.ready = False


def _make_records(n=3):
    dim = cfg.embedding_dim
    return [
        {
            "source": f"doc{i}.md",
            "content_type": "text",
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


class TestEnsureFtsIndex:
    def test_noop_when_no_table(self):
        store.ensure_fts_index()
        assert not store._fts.ready

    def test_creates_index_after_add(self):
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        assert store._fts.ready

    def test_handles_exception_gracefully(self):
        store.add_chunks(_make_records())
        table = store._open_table(store.CHUNKS_TABLE)
        assert table is not None
        with mock.patch.object(
            type(table),
            "create_fts_index",
            side_effect=RuntimeError("boom"),
        ):
            store.ensure_fts_index()
            assert not store._fts.ready


class TestFtsIndexStaleFlag:
    def test_add_chunks_marks_stale(self):
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        assert store._fts.ready
        store.add_chunks(_make_records(1))
        assert not store._fts.ready

    def test_drop_all_marks_stale(self):
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        assert store._fts.ready
        store.drop_all()
        assert not store._fts.ready


class TestHybridSearch:
    def test_hybrid_search_with_fts_index(self):
        records = _make_records()
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * cfg.embedding_dim
        results = store.search(query_vec, top_k=3, query_text="chunk number")
        assert len(results) > 0
        assert results[0].relevance_score is not None

    def test_fallback_to_vector_when_no_query_text(self):
        records = _make_records()
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * cfg.embedding_dim
        results = store.search(query_vec, top_k=3)
        assert len(results) > 0
        assert results[0].distance is not None

    def test_fallback_to_vector_when_no_fts_index(self):
        records = _make_records()
        store.add_chunks(records)
        # Don't call ensure_fts_index, but patch it to not actually create
        with mock.patch("lilbee.store.ensure_fts_index"):
            query_vec = [0.5] * cfg.embedding_dim
            results = store.search(query_vec, top_k=3, query_text="chunk")
        assert len(results) > 0
        assert results[0].distance is not None

    def test_hybrid_fallback_on_exception(self):
        records = _make_records()
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * cfg.embedding_dim
        with mock.patch("lilbee.store._hybrid_search", side_effect=RuntimeError("boom")):
            results = store.search(query_vec, top_k=3, query_text="chunk")
        assert len(results) > 0
        assert results[0].distance is not None

    def test_auto_ensures_fts_index_when_query_text(self):
        records = _make_records()
        store.add_chunks(records)
        assert not store._fts.ready
        query_vec = [0.5] * cfg.embedding_dim
        results = store.search(query_vec, top_k=3, query_text="chunk number")
        assert store._fts.ready
        assert len(results) > 0
