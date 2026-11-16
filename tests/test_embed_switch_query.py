"""A query after an embedding model switch names the drift before it embeds anything.

The switched-to model can answer with a width the config does not hold (a stale
``embedding_dim``). The query vector then fails the width check with a bare
``ValueError`` and the user sees a traceback, never the rebuild advice. The index
verdict has to come first.
"""

from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import EmbeddingModelMismatchError, Store
from lilbee.retrieval.embedder import Embedder
from lilbee.retrieval.query import Searcher

BUILT_WITH = "acme/built-GGUF/built.gguf"
SWITCHED_TO = "acme/switched-GGUF/switched.gguf"
STORED_WIDTH = 768
SERVED_WIDTH = 1024


class _WideProvider:
    """An engine whose model answers with SERVED_WIDTH, whatever the config says."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.5] * SERVED_WIDTH for _ in texts]


def _records(n: int = 3) -> list[dict]:
    return [
        {
            "source": f"doc{i}.md",
            "content_type": "text",
            "chunk_type": "raw",
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": f"chunk number {i} with some text",
            "chunk_index": i,
            "vector": [float(i) / n] * STORED_WIDTH,
        }
        for i in range(n)
    ]


@pytest.fixture()
def switched(tmp_path):
    """A store built under BUILT_WITH, then a config switched to a wider model."""
    config = cfg.model_copy(
        update={
            "lancedb_dir": tmp_path / "lancedb",
            "embedding_model": BUILT_WITH,
            "embedding_dim": STORED_WIDTH,
            "intent_routing": False,
            "query_expansion_count": 0,
            "concept_graph": False,
            "hyde": False,
            "memory_enabled": False,
        }
    )
    store = Store(config)
    store.add_chunks(_records())
    config.embedding_model = SWITCHED_TO
    provider = _WideProvider()
    embedder = Embedder(config, provider)
    searcher = Searcher(config, MagicMock(), store, embedder, MagicMock(), MagicMock())
    yield searcher, store, provider
    store.close()


def test_the_precondition_is_the_crash_the_user_sees(switched):
    """Without the index verdict first, the embedder's width check is what fires."""
    _, _, provider = switched
    embedder = Embedder(cfg.model_copy(update={"embedding_dim": STORED_WIDTH}), provider)

    with pytest.raises(ValueError, match=r"expected 768, got 1024.*lilbee rebuild"):
        embedder.embed_query("what do bees eat")


def test_search_names_the_mismatch_before_it_embeds(switched):
    searcher, _, provider = switched

    with pytest.raises(EmbeddingModelMismatchError, match=BUILT_WITH):
        searcher.search("what do bees eat")

    assert provider.calls == 0


def test_a_vec_prefixed_query_names_the_mismatch_before_it_embeds(switched):
    searcher, _, provider = switched

    with pytest.raises(EmbeddingModelMismatchError, match=BUILT_WITH):
        searcher.search("vec: what do bees eat")

    assert provider.calls == 0


def test_a_hyde_prefixed_query_names_the_mismatch_before_it_asks_the_model(switched):
    searcher, _, provider = switched

    with pytest.raises(EmbeddingModelMismatchError, match=BUILT_WITH):
        searcher.search("hyde: what do bees eat")

    searcher._provider.chat.assert_not_called()
    assert provider.calls == 0


def test_a_typed_retrieval_names_the_mismatch_before_it_embeds(switched):
    searcher, _, provider = switched

    with pytest.raises(EmbeddingModelMismatchError, match=BUILT_WITH):
        searcher._search_typed_arm("what do bees eat", 5, None)

    assert provider.calls == 0


def test_an_unswitched_index_searches_normally(switched):
    searcher, store, provider = switched
    store._config.embedding_model = BUILT_WITH
    store._config.embedding_dim = SERVED_WIDTH
    store.drop_all()
    records = _records()
    for record in records:
        record["vector"] = [0.5] * SERVED_WIDTH
    store.add_chunks(records)
    searcher._embedder._config.embedding_dim = SERVED_WIDTH

    results = searcher.search("what do bees eat")

    assert provider.calls > 0
    assert results


def test_an_empty_store_has_no_verdict_to_name(tmp_path):
    config = cfg.model_copy(
        update={
            "lancedb_dir": tmp_path / "lancedb",
            "embedding_dim": SERVED_WIDTH,
            "intent_routing": False,
            "query_expansion_count": 0,
            "concept_graph": False,
            "hyde": False,
        }
    )
    store = Store(config)
    try:
        searcher = Searcher(
            config, MagicMock(), store, Embedder(config, _WideProvider()), MagicMock(), MagicMock()
        )
        assert searcher.search("what do bees eat") == []
    finally:
        store.close()
