"""Reranker integration tests with real cross-encoder model.

Requires sentence-transformers installed (via ``uv sync --all-extras``).
Downloads a small cross-encoder (~80 MB) on first run.
"""

from __future__ import annotations

import pytest

from lilbee.reranker import reranker_available

if not reranker_available():
    pytest.skip("sentence-transformers not installed", allow_module_level=True)

from lilbee.config import cfg
from lilbee.reranker import Reranker
from lilbee.store import SearchChunk

SMALL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.fixture(autouse=True)
def _isolate_cfg():
    snapshot = {name: getattr(cfg, name) for name in type(cfg).model_fields}
    yield
    for name, val in snapshot.items():
        setattr(cfg, name, val)


def _chunk(content: str, *, distance: float = 0.3, relevance: float = 0.7) -> SearchChunk:
    return SearchChunk(
        source="test.md",
        content_type="text",
        page_start=0,
        page_end=0,
        line_start=0,
        line_end=0,
        chunk=content,
        chunk_index=0,
        vector=[0.0],
        distance=distance,
        relevance_score=relevance,
    )


class TestRerankerIntegration:
    def test_rerank_with_real_model(self) -> None:
        """Real cross-encoder scores and reranks results."""
        cfg.reranker_model = SMALL_MODEL
        reranker = Reranker(cfg)

        chunks = [
            _chunk("The Eiffel Tower is located in Paris, France"),
            _chunk("Python is a popular programming language"),
            _chunk("Paris is the capital city of France"),
        ]

        reranked = reranker.rerank("Where is the Eiffel Tower?", chunks)

        assert len(reranked) == 3
        # The Eiffel Tower chunk should rank highly
        top_contents = [r.chunk for r in reranked[:2]]
        assert any("Eiffel" in c for c in top_contents)

    def test_encoder_lazy_loaded(self) -> None:
        """Encoder is not loaded until first rerank call."""
        cfg.reranker_model = SMALL_MODEL
        reranker = Reranker(cfg)

        assert reranker._encoder is None

        reranker.rerank("test", [_chunk("test content")])

        assert reranker._encoder is not None

    def test_no_model_returns_unchanged(self) -> None:
        """Without a model configured, results pass through unchanged."""
        cfg.reranker_model = ""
        reranker = Reranker(cfg)

        chunks = [_chunk("a"), _chunk("b")]
        result = reranker.rerank("query", chunks)

        assert result is chunks
