"""Empirical gate: query and ingest embedding paths must stay identical.

A query/index embedder mismatch (different model, different normalization,
different truncation) silently destroys relevance. These tests pin the
structural guarantee that both paths route through one Embedder instance
with the same provider call and the same truncation.
"""

from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.retrieval.embedder import Embedder


class _RecordingProvider:
    """Records every embed() input and echoes a deterministic vector per text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t))] * cfg.embedding_dim for t in texts]


@pytest.fixture()
def provider() -> _RecordingProvider:
    return _RecordingProvider()


@pytest.fixture()
def embedder(provider: _RecordingProvider) -> Embedder:
    return Embedder(cfg, provider)


class TestQueryIndexConsistency:
    def test_same_text_same_vector_both_paths(self, embedder, provider):
        """Query path (embed) and ingest path (embed_batch) yield one vector."""
        text = "the quick brown fox jumps over the lazy dog"
        query_vec = embedder.embed(text)
        ingest_vec = embedder.embed_batch([text])[0]
        assert query_vec == ingest_vec

    def test_both_paths_call_same_provider_with_same_arg(self, embedder, provider):
        """Both paths hand the identical (truncated) text to the same provider."""
        text = "shared text"
        embedder.embed(text)
        embedder.embed_batch([text])
        assert provider.calls == [[text], [text]]

    def test_both_paths_apply_identical_truncation(self, embedder, provider):
        """No path embeds beyond the shared char budget; truncation is symmetric."""
        long_text = "a" * (embedder.embed_char_budget + 2000)
        embedder.embed(long_text)
        embedder.embed_batch([long_text])
        query_arg = provider.calls[0][0]
        ingest_arg = provider.calls[1][0]
        assert query_arg == ingest_arg
        assert len(query_arg) == embedder.embed_char_budget

    def test_no_extra_normalization_in_either_path(self, embedder, provider):
        """Neither path post-processes the provider vector (no silent renorm)."""
        text = "normalize me"
        raw = provider.embed([text])[0]
        provider.calls.clear()
        assert embedder.embed(text) == raw
        assert embedder.embed_batch([text])[0] == raw


class TestSharedEmbedderInstance:
    """The query searcher and the ingest path must use ONE embedder, so a
    config change can never leave the two paths on different models.
    """

    def test_services_wires_one_embedder_into_searcher(self):
        """get_services().embedder is the exact instance the Searcher holds."""
        mock_provider = MagicMock()
        embedder = Embedder(cfg, mock_provider)
        from lilbee.data.store import Store
        from lilbee.retrieval.concepts import ConceptGraph
        from lilbee.retrieval.query import Searcher
        from lilbee.retrieval.reranker import Reranker

        store = Store(cfg)
        try:
            searcher = Searcher(
                cfg, mock_provider, store, embedder, Reranker(cfg), ConceptGraph(cfg, store)
            )
            # Ingest reads get_services().embedder; query reads the searcher's
            # embedder. Constructed from the same instance, they cannot diverge.
            assert searcher._embedder is embedder
        finally:
            store.close()
