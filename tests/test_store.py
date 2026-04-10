"""Tests for LanceDB store operations — hybrid search + FTS index lifecycle."""

from unittest import mock

import pytest

from lilbee.config import cfg
from lilbee.store import (
    CitationRecord,
    SearchChunk,
    Store,
    _cosine_sim,
    escape_sql_string,
    mmr_rerank,
)


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
            "create_fts_index",
            side_effect=RuntimeError("boom"),
        ):
            store.ensure_fts_index()
            assert not store._fts_ready


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


class TestHybridSearch:
    def test_hybrid_search_with_fts_index(self, store, test_config):
        records = _make_records()
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=3, query_text="chunk number")
        assert len(results) > 0
        assert results[0].relevance_score is not None

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

    def test_hybrid_fallback_on_exception(self, store, test_config):
        records = _make_records()
        store.add_chunks(records)
        store.ensure_fts_index()
        query_vec = [0.5] * test_config.embedding_dim
        with mock.patch("lilbee.store._hybrid_search", side_effect=RuntimeError("boom")):
            results = store.search(query_vec, top_k=3, query_text="chunk")
        assert len(results) > 0
        assert results[0].distance is not None

    def test_vector_only_applies_mmr(self, store, test_config):
        """Vector-only path (no query_text) applies MMR when results > top_k."""
        records = _make_records(n=6)
        store.add_chunks(records)
        query_vec = [0.5] * test_config.embedding_dim
        results = store.search(query_vec, top_k=2)
        assert len(results) == 2

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

    def test_cosine_sim_zero_vectors(self):
        assert _cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_cosine_sim_identical(self):
        sim = _cosine_sim([1.0, 0.0], [1.0, 0.0])
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
            mock.patch.object(store, "delete_by_source") as mock_del,
            mock.patch.object(store, "delete_source"),
        ):
            result = store.remove_documents(["a.md"])
            assert result.removed == ["a.md"]
            assert result.not_found == []
            mock_del.assert_called_once_with("a.md")

    def test_not_found(self, store):
        with mock.patch.object(store, "get_sources", return_value=[]):
            result = store.remove_documents(["missing.md"])
            assert result.removed == []
            assert result.not_found == ["missing.md"]

    def test_deletes_physical_file(self, store, tmp_path):
        with (
            mock.patch.object(store, "get_sources", return_value=[{"filename": "a.md"}]),
            mock.patch.object(store, "delete_by_source"),
            mock.patch.object(store, "delete_source"),
        ):
            f = tmp_path / "a.md"
            f.write_text("content")
            result = store.remove_documents(["a.md"], delete_files=True, documents_dir=tmp_path)
            assert result.removed == ["a.md"]
            assert not f.exists()

    def test_blocks_path_traversal(self, store, tmp_path):
        with (
            mock.patch.object(
                store, "get_sources", return_value=[{"filename": "../../../etc/passwd"}]
            ),
            mock.patch.object(store, "delete_by_source"),
            mock.patch.object(store, "delete_source"),
        ):
            secret = tmp_path.parent / "secret.txt"
            secret.write_text("don't delete me")
            result = store.remove_documents(
                ["../../../etc/passwd"], delete_files=True, documents_dir=tmp_path
            )
            assert result.removed == ["../../../etc/passwd"]
            assert secret.exists()

    def test_nonexistent_file_still_removes_from_store(self, store, tmp_path):
        with (
            mock.patch.object(store, "get_sources", return_value=[{"filename": "gone.md"}]),
            mock.patch.object(store, "delete_by_source") as mock_del,
            mock.patch.object(store, "delete_source"),
        ):
            result = store.remove_documents(["gone.md"], delete_files=True, documents_dir=tmp_path)
            assert result.removed == ["gone.md"]
            mock_del.assert_called_once()

    def test_uses_default_documents_dir(self, store):
        with (
            mock.patch.object(store, "get_sources", return_value=[{"filename": "a.md"}]),
            mock.patch.object(store, "delete_by_source"),
            mock.patch.object(store, "delete_source"),
        ):
            result = store.remove_documents(["a.md"])
            assert result.removed == ["a.md"]


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

    def test_escapes_backslashes(self):
        assert escape_sql_string("path\\file") == "path\\\\file"

    def test_injection_payload(self):
        escaped = escape_sql_string("' OR 1=1 --")
        # The leading quote is doubled, so it becomes '' (escaped)
        assert escaped.startswith("''")
        # No lone single quote remains (all are doubled)
        stripped = escaped.replace("''", "")
        assert "'" not in stripped


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

    def test_get_chunks_by_source_fallback(self, store):
        """Fallback path when table.search() raises (e.g. incompatible FTS builder)."""
        from unittest.mock import patch

        records = _make_records(n=2)
        store.add_chunks(records)

        # Make table.search() raise to trigger the Arrow fallback
        original_open = store.open_table

        def _broken_open(name):
            table = original_open(name)
            if table is None:
                return None

            def _raise_search(*args, **kwargs):
                raise AttributeError("LanceFtsQueryBuilder has no attribute 'metric'")

            table.search = _raise_search
            return table

        with patch.object(store, "open_table", side_effect=_broken_open):
            results = store.get_chunks_by_source("doc0.md")
        assert len(results) == 1
        assert results[0].source == "doc0.md"

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
        assert sources[0]["source_type"] == "document"

    def test_source_type_wiki(self, store):
        store.upsert_source("wiki/summary.md", "hash456", 3, source_type="wiki")
        sources = store.get_sources()
        assert len(sources) == 1
        assert sources[0]["source_type"] == "wiki"


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
        """_table_names falls back to list() when result has no .tables attribute."""
        from lilbee.store import _table_names

        mock_db = mock.MagicMock()
        mock_db.list_tables.return_value = ["chunks", "sources"]
        result = _table_names(mock_db)
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

        from lilbee.store import install_lancedb_thread_error_suppressor

        original = threading.excepthook
        try:
            install_lancedb_thread_error_suppressor()
            lance_thread = threading.Thread(target=lambda: None, name="LanceDBBackgroundEventLoop")
            args = threading.ExceptHookArgs(
                (RuntimeError, RuntimeError("shutdown"), None, lance_thread)
            )
            # Should return without calling original hook — no exception raised.
            threading.excepthook(args)
        finally:
            threading.excepthook = original

    def test_install_propagates_non_lancedb_thread(self):
        """Errors from other threads are forwarded to the original excepthook."""
        import threading

        from lilbee.store import install_lancedb_thread_error_suppressor

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
