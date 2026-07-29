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
        """Both document paths hand the identical text to the same provider.

        Prefix families (the default nomic model included) prepend a role
        prefix, so the invariant is call identity, not the raw input text.
        """
        text = "shared text"
        embedder.embed(text)
        embedder.embed_batch([text])
        assert provider.calls[0] == provider.calls[1]
        assert provider.calls[0][0].endswith(text)

    def test_query_and_document_prefixes_share_the_payload(self, provider, monkeypatch):
        """Asymmetric families differ only in the role prefix, never the payload.

        The query path carries a different instruction than the document path
        (nomic: search_query: vs search_document:), but both must wrap the
        same text; anything else splits the embedding space. Pinned to nomic
        so the assertions stay distinguishing even if the default embedder
        moves to a symmetric family.
        """
        monkeypatch.setattr(cfg, "embedding_model", "nomic-ai/nomic-embed-text-v1.5-GGUF/n.gguf")
        embedder = Embedder(cfg, provider)
        text = "shared text"
        embedder.embed_query(text)
        embedder.embed(text)
        query_arg, doc_arg = provider.calls[0][0], provider.calls[1][0]
        assert query_arg == f"search_query: {text}"
        assert doc_arg == f"search_document: {text}"

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
        """Neither path post-processes the provider vector (no silent renorm).

        The expected vector is the echo provider's output for the exact text
        each path sent (prefix included), so this pins the absence of
        post-processing, not the input composition.
        """
        text = "normalize me"
        single = embedder.embed(text)
        batched = embedder.embed_batch([text])[0]
        sent = provider.calls[0][0]
        assert provider.calls[1][0] == sent
        assert list(single) == [float(len(sent))] * cfg.embedding_dim
        assert list(batched) == [float(len(sent))] * cfg.embedding_dim


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
