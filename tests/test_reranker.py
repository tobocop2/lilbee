"""Tests for cross-encoder reranking (provider-backed, mocked)."""

from unittest import mock

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import SearchChunk
from lilbee.retrieval.query.searcher import Searcher
from lilbee.retrieval.reranker import _BLEND_SCHEDULE, Reranker

_RERANKER_MODEL = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"


@pytest.fixture(autouse=True)
def _reset():
    """Snapshot + restore reranker config between tests."""
    snapshot = {
        "reranker_model": cfg.reranker_model,
        "rerank_candidates": cfg.rerank_candidates,
        "expansion_skip_threshold": cfg.expansion_skip_threshold,
    }
    yield
    for key, value in snapshot.items():
        setattr(cfg, key, value)


@pytest.fixture()
def reranker():
    """Create a fresh Reranker instance for each test."""
    return Reranker(cfg)


def _chunk(
    source: str, chunk: str, distance: float = 0.5, relevance: float | None = None
) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text",
        page_start=0,
        page_end=0,
        line_start=0,
        line_end=0,
        chunk=chunk,
        chunk_index=0,
        vector=[0.1],
        distance=distance,
        relevance_score=relevance,
    )


def _patch_provider(rerank_fn):
    """Patch get_services to return a provider whose rerank routes to *rerank_fn*."""
    provider = mock.MagicMock()
    provider.rerank.side_effect = rerank_fn
    services = mock.MagicMock(provider=provider)
    return mock.patch("lilbee.app.services.get_services", return_value=services)


class TestRerank:
    def test_returns_unchanged_when_no_model(self, reranker):
        cfg.reranker_model = ""
        results = [_chunk("a.md", "text")]
        assert reranker.rerank("query", results) == results

    def test_reranks_with_provider_scores(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        results = [
            _chunk("a.md", "chunk A", relevance=0.3),
            _chunk("b.md", "chunk B", relevance=0.8),
            _chunk("c.md", "chunk C", relevance=0.5),
        ]
        with _patch_provider(lambda query, cands: [0.9, 0.1, 0.5]):
            reranked = reranker.rerank("test query", results)
        assert [c.chunk for c in reranked] == ["chunk B", "chunk A", "chunk C"]

    def test_bm25_protection(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        cfg.expansion_skip_threshold = 0.8
        results = [
            _chunk("a.md", "exact match", distance=0.9, relevance=0.9),
            _chunk("b.md", "reranker favorite", relevance=0.95),
            _chunk("c.md", "mid", relevance=0.5),
        ]
        with _patch_provider(lambda query, cands: [0.0, 1.0, 0.5]):
            reranked = reranker.rerank("test", results)
        assert reranked[0].chunk == "exact match"

    def test_handles_remainder(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        cfg.rerank_candidates = 2
        results = [
            _chunk("a.md", "chunk A"),
            _chunk("b.md", "chunk B"),
            _chunk("c.md", "chunk C"),
        ]
        with _patch_provider(lambda query, cands: [0.5, 0.8]):
            reranked = reranker.rerank("test", results, candidates=2)
        assert len(reranked) == 3
        assert reranked[-1].chunk == "chunk C"

    def test_empty_results(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        assert reranker.rerank("query", []) == []

    def test_equal_scores(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        results = [_chunk("a.md", "A"), _chunk("b.md", "B")]
        with _patch_provider(lambda query, cands: [0.5, 0.5]):
            reranked = reranker.rerank("test", results)
        assert len(reranked) == 2
        chunks = {r.chunk for r in reranked}
        assert chunks == {"A", "B"}

    def test_provider_error_preserves_results(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        results = [_chunk("a.md", "A"), _chunk("b.md", "B")]

        def explode(query: str, cands: list[str]) -> list[float]:
            raise RuntimeError("backend down")

        with _patch_provider(explode):
            out = reranker.rerank("test", results)
        assert [c.chunk for c in out] == ["A", "B"]
        assert all(r.rerank_score is None for r in out)

    def test_stamps_rerank_score_on_candidates_only(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        results = [
            _chunk("a.md", "chunk A"),
            _chunk("b.md", "chunk B"),
            _chunk("c.md", "chunk C"),
        ]
        with _patch_provider(lambda query, cands: [0.9, 0.1]):
            reranked = reranker.rerank("test", results, candidates=2)
        scored = [r.chunk for r in reranked if r.rerank_score is not None]
        unscored = [r.chunk for r in reranked if r.rerank_score is None]
        assert sorted(scored) == ["chunk A", "chunk B"]
        assert unscored == ["chunk C"]
        assert all(r.rerank_score is None for r in results)  # inputs stay untouched

    def test_pinned_top_keeps_blended_rerank_score(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        cfg.expansion_skip_threshold = 0.8
        results = [
            _chunk("a.md", "exact match", relevance=0.9),
            _chunk("b.md", "reranker favorite", relevance=0.5),
        ]
        with _patch_provider(lambda query, cands: [0.0, 1.0]):
            reranked = reranker.rerank("test", results)
        assert [c.chunk for c in reranked] == ["exact match", "reranker favorite"]
        assert len(reranked) == 2  # the pin must not duplicate the top chunk
        # The pin sentinel orders the list; the stamped score stays the real blend.
        assert all(r.rerank_score is not None and r.rerank_score <= 1.0 for r in reranked)

    def test_sends_chunk_text_to_provider(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        results = [_chunk("a.md", "alpha"), _chunk("b.md", "beta")]
        captured: dict[str, list[str] | str] = {}

        def capture(query: str, cands: list[str]) -> list[float]:
            captured["query"] = query
            captured["cands"] = list(cands)
            return [0.1, 0.2]

        with _patch_provider(capture):
            reranker.rerank("q", results)
        assert captured["query"] == "q"
        assert captured["cands"] == ["alpha", "beta"]


class TestBlendSchedule:
    def test_schedule_weights_sum_to_one(self):
        for key, (fw, rw) in _BLEND_SCHEDULE.items():
            assert abs(fw + rw - 1.0) < 0.01, f"{key} weights don't sum to 1.0"


class TestRerankerBlendPositions:
    def test_mid_and_bottom_positions(self):
        cfg.reranker_model = _RERANKER_MODEL
        r = Reranker(cfg)
        scores = [0.9 - i * 0.05 for i in range(12)]

        results = [_chunk(f"s{i}.md", f"chunk {i}", relevance=0.5 - i * 0.02) for i in range(12)]
        with _patch_provider(lambda query, cands: scores):
            reranked = r.rerank("test", results, candidates=12)
        assert len(reranked) == 12

    def test_no_bm25_protection_when_below_threshold(self):
        cfg.reranker_model = _RERANKER_MODEL
        cfg.expansion_skip_threshold = 0.8
        r = Reranker(cfg)
        results = [
            _chunk("a.md", "low bm25", relevance=0.5),
            _chunk("b.md", "high rerank", relevance=0.3),
        ]
        with _patch_provider(lambda query, cands: [0.1, 0.9]):
            reranked = r.rerank("test", results)
        assert reranked[0].chunk == "high rerank"


class TestGoldenOrdering:
    """Empirical gate: controlled scores -> asserted output order.

    Higher rerank score sorts first; all-equal scores preserve input
    order; both-scores-None falls back to the 0.5 fusion default.
    """

    def test_higher_rerank_score_sorts_first(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        results = [
            _chunk("a.md", "least", relevance=0.5),
            _chunk("b.md", "most", relevance=0.5),
            _chunk("c.md", "middle", relevance=0.5),
        ]
        # Equal fusion isolates the reranker signal: 0.2 < 0.6 < 0.95.
        with _patch_provider(lambda query, cands: [0.2, 0.95, 0.6]):
            reranked = reranker.rerank("test", results)
        assert [c.chunk for c in reranked] == ["most", "middle", "least"]

    def test_all_equal_scores_preserve_input_order(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        cfg.expansion_skip_threshold = 0.6  # above chunk relevance: no BM25 pin
        results = [_chunk(f"{i}.md", f"chunk {i}", relevance=0.5) for i in range(5)]
        with _patch_provider(lambda query, cands: [0.5] * len(cands)):
            reranked = reranker.rerank("test", results)
        # Stable sort on equal blended scores keeps retrieval order intact.
        assert [c.chunk for c in reranked] == [f"chunk {i}" for i in range(5)]

    def test_both_scores_none_uses_fusion_default(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        cfg.expansion_skip_threshold = 0.6  # top relevance is None->0: no pin

        def _bare(source: str, chunk: str) -> SearchChunk:
            return SearchChunk(
                source=source,
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk=chunk,
                chunk_index=0,
                vector=[0.1],
            )

        results = [_bare("a.md", "A"), _bare("b.md", "B")]
        # relevance_score and distance both None -> fusion_score == 0.5
        # (reranker.py:57). Reranker scores then decide order: B over A.
        with _patch_provider(lambda query, cands: [0.1, 0.9]):
            reranked = reranker.rerank("test", results)
        assert [c.chunk for c in reranked] == ["B", "A"]


class TestRerankChangesContext:
    """Empirical gate: reranking changes WHICH chunks reach the model.

    The cross-encoder favorite shares no surface terms with the question,
    so term-coverage set cover alone can never select it.
    """

    _QUESTION = "radio resets and headlights dim at idle"

    def _candidates(self) -> list[SearchChunk]:
        return [
            _chunk("dock.md", "radio resets when the laptop dock is plugged", relevance=0.9),
            _chunk("lightbar.md", "headlights dim with the light bar load", relevance=0.8),
            _chunk("battery.md", "battery log shows the radio and headlights draw", relevance=0.7),
            _chunk("grounding.md", "grounding upgrade with one zero gauge cable", relevance=0.6),
        ]

    def test_reranking_changes_selected_context_set(self):
        cfg.reranker_model = _RERANKER_MODEL
        cfg.expansion_skip_threshold = 0.95  # top fusion hit stays below: no pin
        reranker = Reranker(cfg)
        searcher = Searcher(
            cfg,
            mock.MagicMock(),
            mock.MagicMock(),
            mock.MagicMock(),
            reranker,
            mock.MagicMock(),
        )

        baseline = searcher.select_context(self._candidates(), self._QUESTION, max_sources=3)
        assert "grounding.md" not in {r.source for r in baseline}

        with _patch_provider(lambda query, cands: [0.1, 0.2, 0.1, 1.0]):
            reranked = reranker.rerank(self._QUESTION, self._candidates())
        selected = searcher.select_context(reranked, self._QUESTION, max_sources=3)
        sources = {r.source for r in selected}
        assert "grounding.md" in sources
        assert sources != {r.source for r in baseline}


class TestMixedPoolBias:
    """A mixed wiki+raw pool shouldn't drop one side entirely when the
    reranker returns ambiguous scores. Regression guard against a future
    reranker model that happened to score markdown differently from
    plain text.
    """

    def _wiki_chunk(self, source: str, text: str, relevance: float) -> SearchChunk:
        return SearchChunk(
            source=source,
            content_type="text/markdown",
            chunk_type="wiki",
            page_start=1,
            page_end=1,
            line_start=1,
            line_end=1,
            chunk=text,
            chunk_index=0,
            vector=[0.1],
            relevance_score=relevance,
        )

    def test_ambiguous_scores_keep_both_sides(self, reranker):
        cfg.reranker_model = _RERANKER_MODEL
        results = [
            self._wiki_chunk("wiki/summaries/a.md", "wiki a", relevance=0.5),
            _chunk("a.md", "raw a", relevance=0.5),
            self._wiki_chunk("wiki/summaries/b.md", "wiki b", relevance=0.5),
            _chunk("b.md", "raw b", relevance=0.5),
        ]
        # Identical provider scores. Tie-break falls to blended fusion
        # weighting; no systematic wiki or raw bias should drop either.
        with _patch_provider(lambda query, cands: [0.5] * len(cands)):
            reranked = reranker.rerank("query", results)
        assert len(reranked) == 4
        chunk_types = {r.chunk_type for r in reranked}
        assert chunk_types == {"wiki", "raw"}
