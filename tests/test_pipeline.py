"""Tests for embedder + store round-trip.

Requires a running embedding model backend with nomic-embed-text model.
"""

import pytest

from lilbee.config import cfg
from lilbee.store import Store


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Point store at a temp directory, clean up after."""
    from lilbee.services import reset_services

    original = cfg.lancedb_dir
    cfg.lancedb_dir = tmp_path / "lancedb_test"
    reset_services()
    yield
    reset_services()
    cfg.lancedb_dir = original


@pytest.fixture()
def store():
    """Create a Store instance bound to the isolated config."""
    return Store(cfg)


def _embedding_model_available() -> bool:
    """Check if a real embedding model is available (integration tests only)."""
    try:
        from lilbee.providers.llama_cpp_provider import _resolve_model_path

        _resolve_model_path(cfg.embedding_model)
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
        from lilbee.services import get_services

        vec = get_services().embedder.embed("test sentence")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch_returns_matching_count(self):
        from lilbee.services import get_services

        vecs = get_services().embedder.embed_batch(["hello", "world"])
        assert len(vecs) == 2

    def test_embed_batch_empty(self):
        from lilbee.services import get_services

        assert get_services().embedder.embed_batch([]) == []


@requires_embedding
class TestStoreRoundTrip:
    def test_add_and_search(self, store):
        from lilbee.services import get_services

        embedder = get_services().embedder
        vec = embedder.embed("oil capacity is 5 quarts")
        store.add_chunks(
            [
                {
                    "source": "test.pdf",
                    "content_type": "pdf",
                    "chunk_type": "raw",
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

        results = store.search(embedder.embed("how much oil does it need?"), top_k=1)
        assert len(results) == 1
        assert "5 quarts" in results[0].chunk

    def test_delete_by_source_removes_chunks(self, store):
        from lilbee.services import get_services

        embedder = get_services().embedder
        vec = embedder.embed("tire pressure is 35 PSI")
        store.add_chunks(
            [
                {
                    "source": "delete_me.pdf",
                    "content_type": "pdf",
                    "chunk_type": "raw",
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

        store.delete_by_source("delete_me.pdf")
        for r in store.search(embedder.embed("tire pressure"), top_k=5):
            assert r.source != "delete_me.pdf"


class TestStoreOperations:
    """Cover store paths that don't need a live backend."""

    def test_add_chunks_and_search_empty_table(self, store):
        """add_chunks with data + search on that table."""
        vec = [0.1] * 768
        count = store.add_chunks(
            [
                {
                    "source": "test.pdf",
                    "content_type": "pdf",
                    "chunk_type": "raw",
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
        results = store.search(vec, top_k=1)
        assert len(results) == 1
        assert "5 quarts" in results[0].chunk

    def test_add_chunks_empty_returns_zero(self, store):
        assert store.add_chunks([]) == 0

    def test_search_empty_store(self, store):
        assert store.search([0.1] * 768) == []

    def test_search_filters_by_max_distance(self, store):
        vec = [0.1] * 768
        # Use a very different query vector to produce high distance
        far_vec = [-0.1] * 768
        store.add_chunks(
            [
                {
                    "source": "test.pdf",
                    "content_type": "pdf",
                    "chunk_type": "raw",
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
        assert store.search(far_vec, max_distance=0.001) == []
        # Disabled filtering (0) returns everything
        assert len(store.search(far_vec, max_distance=0)) == 1
        # Generous threshold returns the match
        assert len(store.search(far_vec, max_distance=100.0)) == 1

    def test_delete_by_source(self, store):
        vec = [0.1] * 768
        store.add_chunks(
            [
                {
                    "source": "remove_me.txt",
                    "content_type": "text",
                    "chunk_type": "raw",
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
        store.delete_by_source("remove_me.txt")
        results = store.search(vec, top_k=5)
        assert all(r.source != "remove_me.txt" for r in results)

    def test_delete_by_source_with_single_quote(self, store):
        vec = [0.1] * 768
        store.add_chunks(
            [
                {
                    "source": "it's_a_file.txt",
                    "content_type": "text",
                    "chunk_type": "raw",
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
        store.delete_by_source("it's_a_file.txt")
        results = store.search(vec, top_k=5)
        assert all(r.source != "it's_a_file.txt" for r in results)

    def test_delete_by_source_no_table(self, store):
        # Should not raise on empty store
        store.delete_by_source("nonexistent.txt")

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

        from lilbee.store import ensure_table

        s = Store(cfg)
        db = s.get_db()
        schema = s._chunks_schema()
        mock_table = mock.MagicMock()

        with (
            mock.patch.object(db, "create_table", side_effect=ValueError("already exists")),
            mock.patch.object(db, "open_table", return_value=mock_table),
        ):
            result = ensure_table(db, "chunks", schema)
            assert result is mock_table

    def test_add_chunks_wrong_dimension_raises(self, store):
        wrong_dim_vec = [0.1] * 100  # Wrong dimension
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            store.add_chunks(
                [
                    {
                        "source": "test.pdf",
                        "content_type": "pdf",
                        "chunk_type": "raw",
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
    def test_returns_chunks_for_source(self, store):
        vec = [0.1] * 768
        store.add_chunks(
            [
                {
                    "source": "doc.txt",
                    "content_type": "text",
                    "chunk_type": "raw",
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
        chunks = store.get_chunks_by_source("doc.txt")
        assert len(chunks) == 1
        assert chunks[0].chunk == "Hello world"

    def test_empty_store_returns_empty(self, store):
        assert store.get_chunks_by_source("nope.txt") == []

    def test_filters_by_source(self, store):
        vec = [0.1] * 768
        store.add_chunks(
            [
                {
                    "source": "a.txt",
                    "content_type": "text",
                    "chunk_type": "raw",
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
                    "chunk_type": "raw",
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
        chunks = store.get_chunks_by_source("a.txt")
        assert len(chunks) == 1
        assert chunks[0].source == "a.txt"


class TestSourceTracking:
    def test_upsert_and_retrieve(self, store):
        store.upsert_source("test.pdf", "abc123", 10)
        assert any(s["filename"] == "test.pdf" for s in store.get_sources())

    def test_delete_source(self, store):
        store.upsert_source("to_delete.pdf", "xyz", 5)
        store.delete_source("to_delete.pdf")
        assert not any(s["filename"] == "to_delete.pdf" for s in store.get_sources())

    def test_upsert_source_with_single_quote(self, store):
        store.upsert_source("it's_a_file.pdf", "abc123", 10)
        sources = store.get_sources()
        assert any(s["filename"] == "it's_a_file.pdf" for s in sources)
        # Update should work too (tests the delete predicate in upsert)
        store.upsert_source("it's_a_file.pdf", "def456", 20)
        sources = store.get_sources()
        matching = [s for s in sources if s["filename"] == "it's_a_file.pdf"]
        assert len(matching) == 1
        assert matching[0]["chunk_count"] == 20

    def test_drop_all_clears_everything(self, store):
        store.upsert_source("drop_test.pdf", "hash", 3)
        store.drop_all()
        assert store.get_sources() == []
