"""Tests for embedder + store round-trip.

Requires a running Ollama instance with nomic-embed-text model.
"""

import pytest

import lilbee.config as cfg
import lilbee.store as store_mod


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Point store at a temp directory, clean up after."""
    test_dir = tmp_path / "lancedb_test"
    original = cfg.LANCEDB_DIR
    cfg.LANCEDB_DIR = test_dir
    store_mod.LANCEDB_DIR = test_dir
    yield
    cfg.LANCEDB_DIR = original
    store_mod.LANCEDB_DIR = original


def _embedding_model_available() -> bool:
    try:
        import ollama

        from lilbee.config import EMBEDDING_MODEL

        ollama.embed(model=EMBEDDING_MODEL, input="test")
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _embedding_model_available(),
    reason="Ollama not running or nomic-embed-text not pulled",
)


@requires_ollama
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


@requires_ollama
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

    def test_delete_by_source_no_table(self):
        from lilbee.store import delete_by_source

        # Should not raise on empty store
        delete_by_source("nonexistent.txt")

    def test_safe_delete_exception(self):
        """Cover _safe_delete logging on failure."""
        from unittest.mock import MagicMock

        from lilbee.store import _safe_delete

        mock_table = MagicMock()
        mock_table.delete.side_effect = RuntimeError("test error")
        # Should not raise
        _safe_delete(mock_table, "bad predicate")


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

    def test_drop_all_clears_everything(self):
        from lilbee.store import drop_all, get_sources, upsert_source

        upsert_source("drop_test.pdf", "hash", 3)
        drop_all()
        assert get_sources() == []
