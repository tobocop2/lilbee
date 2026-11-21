"""What a search hands to each of its steps: the question, ``top_k`` and the scope.

The store, the embedder, the provider and the reranker are fakes that record their
calls. Every test reads those calls after one ``search_report``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from lilbee.core.config import cfg
from lilbee.data.store import ChunkType
from lilbee.retrieval.query.searcher import Searcher
from tests.test_query import _make_result

QUESTION = "how do pods restart after a crash"
TOP_K = 5


@dataclass
class _Rig:
    searcher: Searcher
    store: mock.MagicMock
    provider: mock.MagicMock
    reranker: mock.MagicMock
    concepts: mock.MagicMock


def _build(monkeypatch: pytest.MonkeyPatch, **settings: object) -> _Rig:
    """A searcher over fakes, with every widening step off unless *settings* turn it on."""
    base: dict[str, object] = {
        "intent_routing": False,
        "wiki": True,
        "query_expansion_count": 0,
        "concept_graph": False,
        "hyde": False,
        "expansion_skip_threshold": 0.0,
        "temporal_filtering": False,
        "max_distance": 0.0,
        "min_relevance_score": 0.0,
        "diversity_max_per_source": 100,
        "reranker_model": "",
    }
    for name, value in {**base, **settings}.items():
        monkeypatch.setattr(cfg, name, value)
    store = mock.MagicMock()
    store.bm25_probe.return_value = []
    calls = iter(range(1000))
    store.search.side_effect = lambda *a, **k: [
        _make_result(source=f"hit{next(calls)}.md", distance=0.1, chunk="alpha beta gamma")
    ]
    embedder = mock.MagicMock()
    embedder.embed_query.return_value = np.full(8, 0.1, dtype=np.float32)
    embedder.embed_query_batch.side_effect = lambda texts, **kw: [
        np.full(8, 0.1, dtype=np.float32) for _ in texts
    ]
    provider = mock.MagicMock()
    provider.chat.return_value = SimpleNamespace(text="a hypothetical passage")
    reranker = mock.MagicMock()
    reranker.rerank.side_effect = lambda query, rows: rows
    concepts = mock.MagicMock()
    concepts.get_graph.return_value = True
    concepts.expand_query.return_value = ["variant phrasing"]
    concepts.extract_concepts.return_value = []
    searcher = Searcher(cfg, provider, store, embedder, reranker, concepts)
    return _Rig(searcher, store, provider, reranker, concepts)


def test_every_widening_search_gets_top_k_and_the_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, concept_graph=True, hyde=True, expansion_skip_threshold=0.5)

    rig.searcher.search_report(QUESTION, top_k=TOP_K, chunk_type=ChunkType.RAW)

    by_query = {call.kwargs["query_text"]: call.kwargs for call in rig.store.search.call_args_list}
    assert set(by_query) == {QUESTION, "variant phrasing", None}
    for kwargs in by_query.values():
        assert kwargs["top_k"] == TOP_K
        assert kwargs["chunk_type"] is ChunkType.RAW


def test_the_skip_check_probes_the_question_in_the_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, expansion_skip_threshold=0.5)

    rig.searcher.search_report(QUESTION, top_k=TOP_K, chunk_type=ChunkType.RAW)

    rig.store.bm25_probe.assert_called_once()
    assert rig.store.bm25_probe.call_args.args == (QUESTION,)
    assert rig.store.bm25_probe.call_args.kwargs["chunk_type"] is ChunkType.RAW


def test_the_concept_variants_are_made_from_the_question(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, concept_graph=True)

    rig.searcher.search_report(QUESTION, top_k=TOP_K)

    rig.concepts.expand_query.assert_called_once_with(QUESTION)


def test_the_hyde_passage_is_written_for_the_question(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, hyde=True)

    rig.searcher.search_report(QUESTION, top_k=TOP_K)

    prompt = rig.provider.chat.call_args.args[0][0]["content"]
    assert QUESTION in prompt


def test_the_reranker_receives_the_question_and_the_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, reranker_model="org/some-reranker.gguf")

    rig.searcher.search_report(QUESTION, top_k=TOP_K)

    rig.reranker.rerank.assert_called_once()
    assert rig.reranker.rerank.call_args.args[0] == QUESTION
    assert [row.source for row in rig.reranker.rerank.call_args.args[1]] == ["hit0.md"]


def test_a_prefixed_query_searches_its_clean_text_to_the_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = _build(monkeypatch)
    rig.store.bm25_probe.return_value = [_make_result(bm25_score=2.0)]

    rig.searcher.search_report("term: pods restart", top_k=4)

    rig.store.bm25_probe.assert_called_once()
    assert rig.store.bm25_probe.call_args.args == ("pods restart",)
    assert rig.store.bm25_probe.call_args.kwargs["top_k"] == 4


def test_the_candidate_pool_is_twice_top_k_before_the_per_source_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = _build(monkeypatch, diversity_max_per_source=1)
    # Five rows of one source rank first; the only other source ranks sixth, past 2 * top_k.
    rig.store.search.side_effect = lambda *a, **k: [
        *(
            _make_result(source="a.md", chunk_index=i, distance=0.10 + i / 100, chunk="alpha beta")
            for i in range(5)
        ),
        _make_result(source="b.md", distance=0.5, chunk="gamma delta"),
    ]

    report = rig.searcher.search_report(QUESTION, top_k=2)

    assert [row.source for row in report.rows] == ["a.md"]
