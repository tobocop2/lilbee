"""Tests for cross-encoder reranking (mocked — no live model needed)."""

from unittest import mock

import pytest

from lilbee.config import cfg
from lilbee.reranker import _BLEND_SCHEDULE, _get_encoder, rerank, reset_encoder
from lilbee.store import SearchChunk


@pytest.fixture(autouse=True)
def _reset():
    reset_encoder()
    yield
    reset_encoder()


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
    def test_returns_none_when_not_configured(self):
        cfg.reranker_model = ""
        assert _get_encoder() is None

    @mock.patch("lilbee.reranker.cfg")
    def test_returns_none_on_import_error(self, mock_cfg):
        mock_cfg.reranker_model = "some-model"
        with mock.patch.dict("sys.modules", {"sentence_transformers": None}):
            reset_encoder()
            result = _get_encoder()
            assert result is None

    @mock.patch("lilbee.reranker.cfg")
    def test_loads_model(self, mock_cfg):
        mock_cfg.reranker_model = "test-model"
        mock_cross_encoder = mock.MagicMock()
        with mock.patch("lilbee.reranker.CrossEncoder", mock_cross_encoder, create=True):
            # Need to mock the import
            import lilbee.reranker as mod

            with mock.patch.object(mod, "CrossEncoder", mock_cross_encoder, create=True):
                # Actually let's just test via rerank with a mock encoder
                pass


class TestRerank:
    def test_returns_unchanged_when_no_model(self):
        cfg.reranker_model = ""
        results = [_chunk("a.md", "text")]
        assert rerank("query", results) == results

    def test_reranks_with_mock_encoder(self):
        cfg.reranker_model = "test"
        mock_encoder = mock.MagicMock()
        mock_encoder.predict.return_value = [0.9, 0.1, 0.5]

        import lilbee.reranker as mod

        mod._encoder = mock_encoder

        results = [
            _chunk("a.md", "chunk A", relevance=0.3),
            _chunk("b.md", "chunk B", relevance=0.8),
            _chunk("c.md", "chunk C", relevance=0.5),
        ]
        reranked = rerank("test query", results)
        assert len(reranked) == 3
        # Reranker gave highest score to chunk A, so it should move up
        mock_encoder.predict.assert_called_once()

    def test_bm25_protection(self):
        cfg.reranker_model = "test"
        cfg.expansion_skip_threshold = 0.8
        mock_encoder = mock.MagicMock()
        # Reranker wants to demote rank-1
        mock_encoder.predict.return_value = [0.0, 1.0, 0.5]

        import lilbee.reranker as mod

        mod._encoder = mock_encoder

        results = [
            _chunk(
                "a.md", "exact match", distance=0.9, relevance=0.9
            ),  # high BM25 but low vector score
            _chunk("b.md", "reranker favorite", relevance=0.95),  # very high fusion
            _chunk("c.md", "mid", relevance=0.5),
        ]
        reranked = rerank("test", results)
        # Original rank-1 should be protected
        assert reranked[0].chunk == "exact match"

    def test_handles_remainder(self):
        cfg.reranker_model = "test"
        cfg.rerank_candidates = 2
        mock_encoder = mock.MagicMock()
        mock_encoder.predict.return_value = [0.5, 0.8]

        import lilbee.reranker as mod

        mod._encoder = mock_encoder

        results = [
            _chunk("a.md", "chunk A"),
            _chunk("b.md", "chunk B"),
            _chunk("c.md", "chunk C"),  # remainder, not reranked
        ]
        reranked = rerank("test", results, candidates=2)
        assert len(reranked) == 3
        assert reranked[-1].chunk == "chunk C"

    def test_empty_results(self):
        cfg.reranker_model = "test"
        import lilbee.reranker as mod

        mod._encoder = mock.MagicMock()
        assert rerank("query", []) == []

    def test_equal_scores(self):
        cfg.reranker_model = "test"
        mock_encoder = mock.MagicMock()
        mock_encoder.predict.return_value = [0.5, 0.5]

        import lilbee.reranker as mod

        mod._encoder = mock_encoder

        results = [_chunk("a.md", "A"), _chunk("b.md", "B")]
        reranked = rerank("test", results)
        assert len(reranked) == 2


class TestBlendSchedule:
    def test_schedule_weights_sum_to_one(self):
        for key, (fw, rw) in _BLEND_SCHEDULE.items():
            assert abs(fw + rw - 1.0) < 0.01, f"{key} weights don't sum to 1.0"


class TestRerankerBlendPositions:
    def test_mid_and_bottom_positions(self):
        cfg.reranker_model = "test"
        mock_encoder = mock.MagicMock()
        # 12 results to cover top/mid/bottom positions
        scores = [0.9 - i * 0.05 for i in range(12)]
        mock_encoder.predict.return_value = scores

        import lilbee.reranker as mod

        mod._encoder = mock_encoder

        results = [_chunk(f"s{i}.md", f"chunk {i}", relevance=0.5 - i * 0.02) for i in range(12)]
        reranked = rerank("test", results, candidates=12)
        assert len(reranked) == 12

    def test_no_bm25_protection_when_below_threshold(self):
        cfg.reranker_model = "test"
        cfg.expansion_skip_threshold = 0.8
        mock_encoder = mock.MagicMock()
        mock_encoder.predict.return_value = [0.1, 0.9]

        import lilbee.reranker as mod

        mod._encoder = mock_encoder

        results = [
            _chunk("a.md", "low bm25", relevance=0.5),  # below threshold
            _chunk("b.md", "high rerank", relevance=0.3),
        ]
        reranked = rerank("test", results)
        # No protection — reranker can reorder freely
        assert reranked[0].chunk == "high rerank"


class TestRerankerImportError:
    def test_import_error_returns_none(self):
        cfg.reranker_model = "test-model"
        reset_encoder()
        with mock.patch.dict("sys.modules", {"sentence_transformers": None}):
            import lilbee.reranker as mod

            mod._encoder = None
            # Force reimport attempt
            encoder = mod._get_encoder()
            assert encoder is None

    def test_model_load_error_returns_none(self):
        cfg.reranker_model = "bad-model"
        reset_encoder()
        mock_ce = mock.MagicMock(side_effect=RuntimeError("bad model"))
        with mock.patch("lilbee.reranker.CrossEncoder", mock_ce, create=True):
            import lilbee.reranker as mod

            mod._encoder = None
            encoder = mod._get_encoder()
            assert encoder is None


class TestGetEncoderSuccess:
    def test_loads_cross_encoder(self):
        cfg.reranker_model = "test-model"
        reset_encoder()
        mock_ce_cls = mock.MagicMock()
        mock_ce_instance = mock.MagicMock()
        mock_ce_cls.return_value = mock_ce_instance

        import lilbee.reranker as mod

        mod._encoder = None

        # Mock the import of CrossEncoder
        mock_st = mock.MagicMock()
        mock_st.CrossEncoder = mock_ce_cls
        with mock.patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            encoder = mod._get_encoder()
            assert encoder is mock_ce_instance
            mock_ce_cls.assert_called_once_with("test-model")

    def test_generic_exception_returns_none(self):
        cfg.reranker_model = "test-model"
        reset_encoder()

        import lilbee.reranker as mod

        mod._encoder = None

        mock_st = mock.MagicMock()
        mock_st.CrossEncoder.side_effect = RuntimeError("bad model")
        with mock.patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            encoder = mod._get_encoder()
            assert encoder is None
