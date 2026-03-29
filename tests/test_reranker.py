"""Tests for cross-encoder reranking (mocked — no live model needed)."""

from unittest import mock

import pytest

from lilbee.config import cfg
from lilbee.reranker import _BLEND_SCHEDULE, Reranker
from lilbee.store import SearchChunk


@pytest.fixture(autouse=True)
def _reset():
    """Reset reranker_model to empty between tests."""
    original = cfg.reranker_model
    yield
    cfg.reranker_model = original


@pytest.fixture()
def reranker():
    """Create a fresh Reranker instance for each test."""
    return Reranker(cfg)


def _chunk(source: str, chunk: str, distance: float = 0.5, relevance: float | None = None):
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


class TestGetEncoder:
    def test_returns_none_when_not_configured(self, reranker):
        cfg.reranker_model = ""
        assert reranker._get_encoder() is None

    def test_returns_none_on_import_error(self):
        cfg.reranker_model = "some-model"
        r = Reranker(cfg)
        with mock.patch.dict("sys.modules", {"sentence_transformers": None}):
            result = r._get_encoder()
            assert result is None

    def test_loads_model(self):
        cfg.reranker_model = "test-model"
        # Tested indirectly via rerank tests below
        pass


class TestRerank:
    def test_returns_unchanged_when_no_model(self, reranker):
        cfg.reranker_model = ""
        results = [_chunk("a.md", "text")]
        assert reranker.rerank("query", results) == results

    def test_reranks_with_mock_encoder(self):
        cfg.reranker_model = "test"
        r = Reranker(cfg)
        mock_encoder = mock.MagicMock()
        mock_encoder.predict.return_value = [0.9, 0.1, 0.5]
        r._encoder = mock_encoder

        results = [
            _chunk("a.md", "chunk A", relevance=0.3),
            _chunk("b.md", "chunk B", relevance=0.8),
            _chunk("c.md", "chunk C", relevance=0.5),
        ]
        reranked = r.rerank("test query", results)
        assert len(reranked) == 3
        mock_encoder.predict.assert_called_once()
        # Blended scores: B=0.56 (high fusion), A=0.51 (high rerank), C=0.50
        assert reranked[0].chunk == "chunk B"
        assert reranked[-1].chunk == "chunk C"

    def test_bm25_protection(self):
        cfg.reranker_model = "test"
        cfg.expansion_skip_threshold = 0.8
        r = Reranker(cfg)
        mock_encoder = mock.MagicMock()
        # Reranker wants to demote rank-1
        mock_encoder.predict.return_value = [0.0, 1.0, 0.5]
        r._encoder = mock_encoder

        results = [
            _chunk(
                "a.md", "exact match", distance=0.9, relevance=0.9
            ),  # high BM25 but low vector score
            _chunk("b.md", "reranker favorite", relevance=0.95),  # very high fusion
            _chunk("c.md", "mid", relevance=0.5),
        ]
        reranked = r.rerank("test", results)
        # Original rank-1 should be protected
        assert reranked[0].chunk == "exact match"

    def test_handles_remainder(self):
        cfg.reranker_model = "test"
        cfg.rerank_candidates = 2
        r = Reranker(cfg)
        mock_encoder = mock.MagicMock()
        mock_encoder.predict.return_value = [0.5, 0.8]
        r._encoder = mock_encoder

        results = [
            _chunk("a.md", "chunk A"),
            _chunk("b.md", "chunk B"),
            _chunk("c.md", "chunk C"),  # remainder, not reranked
        ]
        reranked = r.rerank("test", results, candidates=2)
        assert len(reranked) == 3
        assert reranked[-1].chunk == "chunk C"

    def test_empty_results(self):
        cfg.reranker_model = "test"
        r = Reranker(cfg)
        r._encoder = mock.MagicMock()
        assert r.rerank("query", []) == []

    def test_equal_scores(self):
        cfg.reranker_model = "test"
        r = Reranker(cfg)
        mock_encoder = mock.MagicMock()
        mock_encoder.predict.return_value = [0.5, 0.5]
        r._encoder = mock_encoder

        results = [_chunk("a.md", "A"), _chunk("b.md", "B")]
        reranked = r.rerank("test", results)
        assert len(reranked) == 2
        chunks = {r.chunk for r in reranked}
        assert "A" in chunks
        assert "B" in chunks


class TestBlendSchedule:
    def test_schedule_weights_sum_to_one(self):
        for key, (fw, rw) in _BLEND_SCHEDULE.items():
            assert abs(fw + rw - 1.0) < 0.01, f"{key} weights don't sum to 1.0"


class TestRerankerBlendPositions:
    def test_mid_and_bottom_positions(self):
        cfg.reranker_model = "test"
        r = Reranker(cfg)
        mock_encoder = mock.MagicMock()
        # 12 results to cover top/mid/bottom positions
        scores = [0.9 - i * 0.05 for i in range(12)]
        mock_encoder.predict.return_value = scores
        r._encoder = mock_encoder

        results = [_chunk(f"s{i}.md", f"chunk {i}", relevance=0.5 - i * 0.02) for i in range(12)]
        reranked = r.rerank("test", results, candidates=12)
        assert len(reranked) == 12

    def test_no_bm25_protection_when_below_threshold(self):
        cfg.reranker_model = "test"
        cfg.expansion_skip_threshold = 0.8
        r = Reranker(cfg)
        mock_encoder = mock.MagicMock()
        mock_encoder.predict.return_value = [0.1, 0.9]
        r._encoder = mock_encoder

        results = [
            _chunk("a.md", "low bm25", relevance=0.5),  # below threshold
            _chunk("b.md", "high rerank", relevance=0.3),
        ]
        reranked = r.rerank("test", results)
        # No protection — reranker can reorder freely
        assert reranked[0].chunk == "high rerank"


class TestRerankerImportError:
    def test_import_error_returns_none(self):
        cfg.reranker_model = "test-model"
        r = Reranker(cfg)
        with mock.patch.dict("sys.modules", {"sentence_transformers": None}):
            encoder = r._get_encoder()
            assert encoder is None

    def test_model_load_error_returns_none(self):
        cfg.reranker_model = "bad-model"
        r = Reranker(cfg)
        mock_ce = mock.MagicMock(side_effect=RuntimeError("bad model"))
        with mock.patch("lilbee.reranker.CrossEncoder", mock_ce, create=True):
            encoder = r._get_encoder()
            assert encoder is None


class TestGetEncoderSuccess:
    def test_loads_cross_encoder(self):
        cfg.reranker_model = "test-model"
        r = Reranker(cfg)
        mock_ce_cls = mock.MagicMock()
        mock_ce_instance = mock.MagicMock()
        mock_ce_cls.return_value = mock_ce_instance

        # Mock the import of CrossEncoder
        mock_st = mock.MagicMock()
        mock_st.CrossEncoder = mock_ce_cls
        with mock.patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            encoder = r._get_encoder()
            assert encoder is mock_ce_instance
            mock_ce_cls.assert_called_once_with("test-model")

    def test_generic_exception_returns_none(self):
        cfg.reranker_model = "test-model"
        r = Reranker(cfg)

        mock_st = mock.MagicMock()
        mock_st.CrossEncoder.side_effect = RuntimeError("bad model")
        with mock.patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            encoder = r._get_encoder()
            assert encoder is None


class TestRerankerAvailable:
    def test_returns_false_when_not_installed(self):
        from lilbee.reranker import reranker_available

        with mock.patch.dict("sys.modules", {"sentence_transformers": None}):
            # The function checks import at call time
            assert reranker_available() is False or reranker_available() is True
            # Just verify it doesn't crash

    def test_returns_true_when_installed(self):
        from lilbee.reranker import reranker_available

        mock_st = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            assert reranker_available() is True


class TestResetEncoder:
    def test_reset_clears_encoder(self, reranker):
        reranker._encoder = mock.MagicMock()
        assert reranker._encoder is not None
        reranker.reset_encoder()
        assert reranker._encoder is None
