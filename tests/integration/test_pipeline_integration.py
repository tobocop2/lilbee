"""Embedder + store round-trip integration tests.

Requires a real embedding model (nomic-embed-text) installed.
Moved from tests/test_pipeline.py — these need live model inference.

Run with:
    uv run pytest tests/integration/test_pipeline_integration.py -v
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
    """Check if a real embedding model is available."""
    try:
        from lilbee.providers.llama_cpp_provider import resolve_model_path

        resolve_model_path(cfg.embedding_model)
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
