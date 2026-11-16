"""Each search path embeds the text after the index verdict and searches with that vector."""

from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import human_recall_predicate
from lilbee.retrieval.query.searcher import QueryMode, Searcher

QUERY_VECTOR = [0.25, 0.5]


def _wired(**changes):
    """A searcher over a mock store and embedder, with every optional retrieval step off."""
    config = cfg.model_copy(
        update={
            "intent_routing": False,
            "query_expansion_count": 0,
            "concept_graph": False,
            "hyde": False,
            "reranker_model": "",
            "expansion_skip_threshold": 0,
            **changes,
        }
    )
    store = MagicMock()
    store.search.return_value = []
    store.search_memories.return_value = []
    embedder = MagicMock()
    embedder.embed_query.return_value = QUERY_VECTOR
    embedder.embedding_available.return_value = True
    return Searcher(config, MagicMock(), store, embedder, MagicMock(), MagicMock()), store


def test_the_verdict_is_asked_before_the_query_is_embedded():
    searcher, store = _wired()
    order = MagicMock()
    order.attach_mock(store.ensure_servable, "verdict")
    order.attach_mock(searcher._embedder.embed_query, "embed")

    searcher._embed_query("bees")

    assert [call[0] for call in order.mock_calls] == ["verdict", "embed"]
    searcher._embedder.embed_query.assert_called_once_with("bees")


def test_a_vec_query_searches_with_its_own_vector_and_text():
    searcher, store = _wired()

    searcher._search_structured(QueryMode.VEC, "bees", 5)

    searcher._embedder.embed_query.assert_called_once_with("bees")
    store.search.assert_called_once()
    assert store.search.call_args.args == (QUERY_VECTOR,)
    assert store.search.call_args.kwargs["top_k"] == 5
    assert store.search.call_args.kwargs["query_text"] is None


def test_a_scoped_query_searches_with_its_vector_and_text():
    searcher, store = _wired()

    searcher._search_structured(QueryMode.RAW, "bees", 5)

    searcher._embedder.embed_query.assert_called_once_with("bees")
    assert store.search.call_args.args == (QUERY_VECTOR,)
    assert store.search.call_args.kwargs["top_k"] == 5
    assert store.search.call_args.kwargs["query_text"] == "bees"


def test_a_plain_query_searches_with_its_vector_and_text():
    searcher, store = _wired()

    searcher._candidates("bees", 5, None)

    searcher._embedder.embed_query.assert_called_once_with("bees")
    assert store.search.call_args.args == (QUERY_VECTOR,)
    assert store.search.call_args.kwargs["top_k"] == 5
    assert store.search.call_args.kwargs["query_text"] == "bees"


def test_the_typed_arm_searches_with_its_vector_and_text():
    searcher, store = _wired()

    searcher._search_typed_arm("bees", 5, None)

    searcher._embedder.embed_query.assert_called_once_with("bees")
    assert store.search.call_args.args == (QUERY_VECTOR,)
    assert store.search.call_args.kwargs["top_k"] == 5
    assert store.search.call_args.kwargs["query_text"] == "bees"


def test_memory_recall_searches_with_its_vector_and_the_configured_limits():
    searcher, store = _wired(memory_enabled=True, memory_top_k=3, memory_max_distance=0.4)

    searcher._memory_facts("bees")

    searcher._embedder.embed_query.assert_called_once_with("bees")
    store.ensure_servable.assert_called()
    assert store.search_memories.call_args.args == (QUERY_VECTOR,)
    assert store.search_memories.call_args.kwargs == {
        "owner_predicate": human_recall_predicate(),
        "top_k": 3,
        "max_distance": 0.4,
    }


@pytest.mark.parametrize(
    ("top_k", "available", "recalled"),
    [(0, True, False), (1, True, True), (3, False, False)],
)
def test_memory_recall_needs_a_positive_limit_and_an_embedder(top_k, available, recalled):
    searcher, store = _wired(memory_enabled=True, memory_top_k=top_k)
    searcher._embedder.embedding_available.return_value = available

    searcher._memory_facts("bees")

    assert store.search_memories.called is recalled


def test_a_search_warns_once_for_the_embed_inputs_cut_across_its_embeds(caplog):
    from lilbee.providers.truncation import note_truncated

    searcher, _ = _wired()

    def candidates(question, top_k, chunk_type):
        note_truncated(1, 1, 8)
        note_truncated(1, 1, 8)
        return []

    searcher._candidates = candidates
    with caplog.at_level("WARNING", logger="lilbee.providers.truncation"):
        searcher.search("bees")
        searcher.search("bees")

    assert len(caplog.records) == 2
    assert "2 of 2" in caplog.records[0].getMessage()
