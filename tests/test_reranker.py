"""Tests for cross-encoder reranking (provider-backed, mocked)."""

from unittest import mock

import pytest

from lilbee.core.config import cfg
from lilbee.reranker import _BLEND_SCHEDULE, Reranker
from lilbee.store import SearchChunk


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
    return mock.patch("lilbee.core.services.get_services", return_value=services)


class TestRerank:
    def test_returns_unchanged_when_no_model(self, reranker):
        cfg.reranker_model = ""
        results = [_chunk("a.md", "text")]
        assert reranker.rerank("query", results) == results

    def test_reranks_with_provider_scores(self, reranker):
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
        results = [
            _chunk("a.md", "chunk A", relevance=0.3),
            _chunk("b.md", "chunk B", relevance=0.8),
            _chunk("c.md", "chunk C", relevance=0.5),
        ]
        with _patch_provider(lambda query, cands: [0.9, 0.1, 0.5]):
            reranked = reranker.rerank("test query", results)
        assert [c.chunk for c in reranked] == ["chunk B", "chunk A", "chunk C"]

    def test_bm25_protection(self, reranker):
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
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
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
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
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
        assert reranker.rerank("query", []) == []

    def test_equal_scores(self, reranker):
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
        results = [_chunk("a.md", "A"), _chunk("b.md", "B")]
        with _patch_provider(lambda query, cands: [0.5, 0.5]):
            reranked = reranker.rerank("test", results)
        assert len(reranked) == 2
        chunks = {r.chunk for r in reranked}
        assert chunks == {"A", "B"}

    def test_provider_error_preserves_results(self, reranker):
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
        results = [_chunk("a.md", "A"), _chunk("b.md", "B")]

        def explode(query: str, cands: list[str]) -> list[float]:
            raise RuntimeError("backend down")

        with _patch_provider(explode):
            out = reranker.rerank("test", results)
        assert [c.chunk for c in out] == ["A", "B"]

    def test_sends_chunk_text_to_provider(self, reranker):
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
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
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
        r = Reranker(cfg)
        scores = [0.9 - i * 0.05 for i in range(12)]

        results = [_chunk(f"s{i}.md", f"chunk {i}", relevance=0.5 - i * 0.02) for i in range(12)]
        with _patch_provider(lambda query, cands: scores):
            reranked = r.rerank("test", results, candidates=12)
        assert len(reranked) == 12

    def test_no_bm25_protection_when_below_threshold(self):
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
        cfg.expansion_skip_threshold = 0.8
        r = Reranker(cfg)
        results = [
            _chunk("a.md", "low bm25", relevance=0.5),
            _chunk("b.md", "high rerank", relevance=0.3),
        ]
        with _patch_provider(lambda query, cands: [0.1, 0.9]):
            reranked = r.rerank("test", results)
        assert reranked[0].chunk == "high rerank"


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
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
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
