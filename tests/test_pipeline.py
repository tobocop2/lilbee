"""Tests for embedder + store round-trip.

Requires a running Ollama instance with nomic-embed-text model.
"""

import pytest

from lilbee.config import cfg


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Point store at a temp directory, clean up after."""
    original = cfg.lancedb_dir
    cfg.lancedb_dir = tmp_path / "lancedb_test"
    yield
    cfg.lancedb_dir = original


def _embedding_model_available() -> bool:
    try:
        from lilbee.embedder import embed

        embed("test")
        return True
    except Exception:
        return False


requires_embedding = pytest.mark.skipif(
    not _embedding_model_available(),
    reason="Embedding model not available",
)


@requires_embedding
class TestEmbedder:
    def test_embed_returns_float_vector(self):
        from lilbee.embedder import embed

        vec = embed("test sentence")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch_returns_matching_count(self):
        from lilbee.embedder import embed_batch

        vecs = embed_batch(["hello", "world"])
        assert len(vecs) == 2

    def test_embed_batch_empty(self):
        from lilbee.embedder import embed_batch

        assert embed_batch([]) == []


@requires_embedding
class TestStoreRoundTrip:
    def test_add_and_search(self):
        from lilbee.embedder import embed
        from lilbee.store import add_chunks, search

        vec = embed("oil capacity is 5 quarts")
        add_chunks(
            [
                {
                    "source": "test.pdf",
                    "content_type": "pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "The oil capacity is 5 quarts with filter change.",
                    "chunk_index": 0,
                    "vector": vec,
                }
            ]
        )

        results = search(embed("how much oil does it need?"), top_k=1)
        assert len(results) == 1
        assert "5 quarts" in results[0]["chunk"]

    def test_delete_by_source_removes_chunks(self):
        from lilbee.embedder import embed
        from lilbee.store import add_chunks, delete_by_source, search

        vec = embed("tire pressure is 35 PSI")
        add_chunks(
            [
                {
                    "source": "delete_me.pdf",
                    "content_type": "pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "Tire pressure is 35 PSI front.",
                    "chunk_index": 0,
                    "vector": vec,
                }
            ]
        )

        delete_by_source("delete_me.pdf")
        for r in search(embed("tire pressure"), top_k=5):
            assert r["source"] != "delete_me.pdf"


class TestStoreOperations:
    """Cover store paths that don't need Ollama."""

    def test_add_chunks_and_search_empty_table(self):
        """add_chunks with data + search on that table."""
        from lilbee.store import add_chunks, search

        vec = [0.1] * 768
        count = add_chunks(
            [
                {
                    "source": "test.pdf",
                    "content_type": "pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "The oil capacity is 5 quarts.",
                    "chunk_index": 0,
                    "vector": vec,
                }
            ]
        )
        assert count == 1
        results = search(vec, top_k=1)
        assert len(results) == 1
        assert "5 quarts" in results[0]["chunk"]

    def test_add_chunks_empty_returns_zero(self):
        from lilbee.store import add_chunks

        assert add_chunks([]) == 0

    def test_search_empty_store(self):
        from lilbee.store import search

        assert search([0.1] * 768) == []

    def test_search_filters_by_max_distance(self):
        from lilbee.store import add_chunks, search

        vec = [0.1] * 768
        # Use a very different query vector to produce high distance
        far_vec = [-0.1] * 768
        add_chunks(
            [
                {
                    "source": "test.pdf",
                    "content_type": "pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "Relevant content.",
                    "chunk_index": 0,
                    "vector": vec,
                }
            ]
        )
        # Tight threshold filters out distant matches
        assert search(far_vec, max_distance=0.001) == []
        # Disabled filtering (0) returns everything
        assert len(search(far_vec, max_distance=0)) == 1
        # Generous threshold returns the match
        assert len(search(far_vec, max_distance=100.0)) == 1

    def test_delete_by_source(self):
        from lilbee.store import add_chunks, delete_by_source, search

        vec = [0.1] * 768
        add_chunks(
            [
                {
                    "source": "remove_me.txt",
                    "content_type": "text",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "Content to remove.",
                    "chunk_index": 0,
                    "vector": vec,
                }
            ]
        )
        delete_by_source("remove_me.txt")
        results = search(vec, top_k=5)
        assert all(r["source"] != "remove_me.txt" for r in results)

    def test_delete_by_source_with_single_quote(self):
        from lilbee.store import add_chunks, delete_by_source, search

        vec = [0.1] * 768
        add_chunks(
            [
                {
                    "source": "it's_a_file.txt",
                    "content_type": "text",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "Content with quote.",
                    "chunk_index": 0,
                    "vector": vec,
                }
            ]
        )
        delete_by_source("it's_a_file.txt")
        results = search(vec, top_k=5)
        assert all(r["source"] != "it's_a_file.txt" for r in results)

    def test_delete_by_source_no_table(self):
        from lilbee.store import delete_by_source

        # Should not raise on empty store
        delete_by_source("nonexistent.txt")

    def testsafe_delete_exception(self):
        """Cover safe_delete logging on failure."""
        from unittest.mock import MagicMock

        from lilbee.store import safe_delete

        mock_table = MagicMock()
        mock_table.delete.side_effect = RuntimeError("test error")
        # Should not raise
        safe_delete(mock_table, "bad predicate")

    def testensure_table_handles_already_exists(self):
        """ensure_table recovers when create_table raises ValueError."""
        from unittest import mock

        from lilbee.store import _chunks_schema, ensure_table, get_db

        db = get_db()
        mock_table = mock.MagicMock()

        with (
            mock.patch.object(db, "create_table", side_effect=ValueError("already exists")),
            mock.patch.object(db, "open_table", return_value=mock_table),
        ):
            result = ensure_table(db, "chunks", _chunks_schema())
            assert result is mock_table

    def test_add_chunks_wrong_dimension_raises(self):
        from lilbee.store import add_chunks

        wrong_dim_vec = [0.1] * 100  # Wrong dimension
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            add_chunks(
                [
                    {
                        "source": "test.pdf",
                        "content_type": "pdf",
                        "page_start": 1,
                        "page_end": 1,
                        "line_start": 0,
                        "line_end": 0,
                        "chunk": "test",
                        "chunk_index": 0,
                        "vector": wrong_dim_vec,
                    }
                ]
            )


class TestGetChunksBySource:
    def test_returns_chunks_for_source(self):
        from lilbee.store import add_chunks, get_chunks_by_source

        vec = [0.1] * 768
        add_chunks(
            [
                {
                    "source": "doc.txt",
                    "content_type": "text",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "Hello world",
                    "chunk_index": 0,
                    "vector": vec,
                },
            ]
        )
        chunks = get_chunks_by_source("doc.txt")
        assert len(chunks) == 1
        assert chunks[0]["chunk"] == "Hello world"

    def test_empty_store_returns_empty(self):
        from lilbee.store import get_chunks_by_source

        assert get_chunks_by_source("nope.txt") == []

    def test_filters_by_source(self):
        from lilbee.store import add_chunks, get_chunks_by_source

        vec = [0.1] * 768
        add_chunks(
            [
                {
                    "source": "a.txt",
                    "content_type": "text",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "From A",
                    "chunk_index": 0,
                    "vector": vec,
                },
                {
                    "source": "b.txt",
                    "content_type": "text",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "From B",
                    "chunk_index": 0,
                    "vector": vec,
                },
            ]
        )
        chunks = get_chunks_by_source("a.txt")
        assert len(chunks) == 1
        assert chunks[0]["source"] == "a.txt"


class TestSourceTracking:
    def test_upsert_and_retrieve(self):
        from lilbee.store import get_sources, upsert_source

        upsert_source("test.pdf", "abc123", 10)
        assert any(s["filename"] == "test.pdf" for s in get_sources())

    def test_delete_source(self):
        from lilbee.store import delete_source, get_sources, upsert_source

        upsert_source("to_delete.pdf", "xyz", 5)
        delete_source("to_delete.pdf")
        assert not any(s["filename"] == "to_delete.pdf" for s in get_sources())

    def test_upsert_source_with_single_quote(self):
        from lilbee.store import get_sources, upsert_source

        upsert_source("it's_a_file.pdf", "abc123", 10)
        sources = get_sources()
        assert any(s["filename"] == "it's_a_file.pdf" for s in sources)
        # Update should work too (tests the delete predicate in upsert)
        upsert_source("it's_a_file.pdf", "def456", 20)
        sources = get_sources()
        matching = [s for s in sources if s["filename"] == "it's_a_file.pdf"]
        assert len(matching) == 1
        assert matching[0]["chunk_count"] == 20

    def test_drop_all_clears_everything(self):
        from lilbee.store import drop_all, get_sources, upsert_source

        upsert_source("drop_test.pdf", "hash", 3)
        drop_all()
        assert get_sources() == []
