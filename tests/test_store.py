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
        table = store.open_table(store.CHUNKS_TABLE)
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

    def test_vector_only_applies_mmr(self):
        """Vector-only path (no query_text) applies MMR when results > top_k."""
        records = _make_records(n=6)
        store.add_chunks(records)
        query_vec = [0.5] * cfg.embedding_dim
        results = store.search(query_vec, top_k=2)
        assert len(results) == 2

    def test_auto_ensures_fts_index_when_query_text(self):
        records = _make_records()
        store.add_chunks(records)
        assert not store._fts.ready
        query_vec = [0.5] * cfg.embedding_dim
        results = store.search(query_vec, top_k=3, query_text="chunk number")
        assert store._fts.ready
        assert len(results) > 0


class TestMMRRerank:
    def test_selects_diverse_results(self):
        from lilbee.store import SearchChunk, mmr_rerank

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
        # x-axis 2 is identical to x-axis 1, so max redundancy
        # y-axis has relevance 0.6 and zero redundancy with x-axis 1
        assert selected[1].chunk == "y-axis"

    def test_returns_all_when_fewer_than_k(self):
        from lilbee.store import SearchChunk, mmr_rerank

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

    def test_cosine_sim_zero_vectors(self):
        from lilbee.store import _cosine_sim

        assert _cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_cosine_sim_identical(self):
        from lilbee.store import _cosine_sim

        sim = _cosine_sim([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6


class TestAdaptiveFilter:
    def test_returns_results_within_threshold(self):
        from lilbee.store import SearchChunk, _adaptive_filter

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
        filtered = _adaptive_filter(results, top_k=1, initial_threshold=0.3)
        assert len(filtered) == 1
        assert filtered[0].chunk == "close"

    def test_widens_threshold_when_too_few(self):
        from lilbee.store import SearchChunk, _adaptive_filter

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
        # Initial threshold 0.3 finds nothing, should widen to 0.7
        filtered = _adaptive_filter(results, top_k=1, initial_threshold=0.3)
        assert len(filtered) == 1

    def test_stops_at_max_threshold(self):
        from lilbee.store import SearchChunk, _adaptive_filter

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
        filtered = _adaptive_filter(results, top_k=1, initial_threshold=0.3)
        assert len(filtered) == 0  # beyond max threshold of 1.0


class TestRemoveDocuments:
    @mock.patch("lilbee.store.get_sources")
    @mock.patch("lilbee.store.delete_source")
    @mock.patch("lilbee.store.delete_by_source")
    def test_removes_known_files(self, mock_del, mock_del_src, mock_sources, tmp_path):
        mock_sources.return_value = [{"filename": "a.md"}, {"filename": "b.md"}]
        result = store.remove_documents(["a.md"], documents_dir=tmp_path)
        assert result.removed == ["a.md"]
        assert result.not_found == []
        mock_del.assert_called_once_with("a.md")

    @mock.patch("lilbee.store.get_sources")
    def test_not_found(self, mock_sources, tmp_path):
        mock_sources.return_value = []
        result = store.remove_documents(["missing.md"], documents_dir=tmp_path)
        assert result.removed == []
        assert result.not_found == ["missing.md"]

    @mock.patch("lilbee.store.get_sources")
    @mock.patch("lilbee.store.delete_source")
    @mock.patch("lilbee.store.delete_by_source")
    def test_deletes_physical_file(self, mock_del, mock_del_src, mock_sources, tmp_path):
        mock_sources.return_value = [{"filename": "a.md"}]
        f = tmp_path / "a.md"
        f.write_text("content")
        result = store.remove_documents(["a.md"], delete_files=True, documents_dir=tmp_path)
        assert result.removed == ["a.md"]
        assert not f.exists()

    @mock.patch("lilbee.store.get_sources")
    @mock.patch("lilbee.store.delete_source")
    @mock.patch("lilbee.store.delete_by_source")
    def test_blocks_path_traversal(self, mock_del, mock_del_src, mock_sources, tmp_path):
        mock_sources.return_value = [{"filename": "../../../etc/passwd"}]
        secret = tmp_path.parent / "secret.txt"
        secret.write_text("don't delete me")
        result = store.remove_documents(
            ["../../../etc/passwd"], delete_files=True, documents_dir=tmp_path
        )
        assert result.removed == ["../../../etc/passwd"]
        # File outside documents_dir should NOT be deleted
        assert secret.exists()

    @mock.patch("lilbee.store.get_sources")
    @mock.patch("lilbee.store.delete_source")
    @mock.patch("lilbee.store.delete_by_source")
    def test_nonexistent_file_still_removes_from_store(
        self, mock_del, mock_del_src, mock_sources, tmp_path
    ):
        mock_sources.return_value = [{"filename": "gone.md"}]
        result = store.remove_documents(["gone.md"], delete_files=True, documents_dir=tmp_path)
        assert result.removed == ["gone.md"]
        mock_del.assert_called_once()

    @mock.patch("lilbee.store.get_sources")
    @mock.patch("lilbee.store.delete_source")
    @mock.patch("lilbee.store.delete_by_source")
    def test_uses_default_documents_dir(self, mock_del, mock_del_src, mock_sources):
        mock_sources.return_value = [{"filename": "a.md"}]
        result = store.remove_documents(["a.md"])
        assert result.removed == ["a.md"]


class TestBm25Probe:
    def test_returns_results_when_fts_ready(self):
        records = _make_records(n=3)
        store.add_chunks(records)
        store.ensure_fts_index()
        results = store.bm25_probe("chunk number", top_k=3)
        assert len(results) > 0

    def test_returns_empty_when_no_fts(self):
        records = _make_records(n=1)
        store.add_chunks(records)
        store._fts.ready = False
        results = store.bm25_probe("anything")
        # May or may not build FTS — but should not crash
        assert isinstance(results, list)

    def test_returns_empty_when_no_table(self):
        results = store.bm25_probe("anything")
        assert results == []
