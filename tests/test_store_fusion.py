"""Tests for score-aware fusion of the vector and BM25 arms."""

import pytest

from lilbee.data.store import SearchChunk
from lilbee.data.store.fusion import fuse_arms, normalized_bm25, vector_similarity


def _chunk(source, idx, *, distance=None, bm25=None, dim=4):
    return SearchChunk(
        source=source,
        content_type="text",
        chunk_type="raw",
        page_start=0,
        page_end=0,
        line_start=0,
        line_end=0,
        chunk=f"{source}:{idx}",
        chunk_index=idx,
        vector=[0.1] * dim,
        distance=distance,
        bm25_score=bm25,
    )


class TestVectorSimilarity:
    def test_zero_distance_is_perfect(self):
        assert vector_similarity(0.0) == 1.0

    def test_opposite_vectors_clamp_to_zero(self):
        assert vector_similarity(2.0) == 0.0

    def test_midrange(self):
        assert vector_similarity(0.4) == pytest.approx(0.6)


class TestNormalizedBm25:
    def test_top_score_anchors_to_one(self):
        assert normalized_bm25([20.0, 10.0, 5.0]) == pytest.approx([1.0, 0.5, 0.25])

    def test_empty_list(self):
        assert normalized_bm25([]) == []

    def test_non_positive_scores_map_to_zero(self):
        assert normalized_bm25([0.0, -1.0]) == [0.0, 0.0]


class TestFuseArms:
    def test_both_arms_beat_either_alone(self):
        both = _chunk("a.md", 0, distance=0.3)
        vec_only = _chunk("b.md", 0, distance=0.3)
        lex = _chunk("a.md", 0, bm25=12.0)
        fused = fuse_arms([both, vec_only], [lex], alpha=0.6)
        scores = {(r.source, r.chunk_index): r.score for r in fused}
        assert scores[("a.md", 0)] > scores[("b.md", 0)]

    def test_lexical_only_row_survives_with_score(self):
        """The identifier case: a row invisible to the vector arm must carry
        real fused weight from its BM25 strength alone."""
        vec = [_chunk("noise.md", i, distance=0.5) for i in range(3)]
        lex = [_chunk("catalog_482.pdf", 0, bm25=35.0)]
        fused = fuse_arms(vec, lex, alpha=0.6)
        lexical_row = next(r for r in fused if r.source == "catalog_482.pdf")
        assert lexical_row.score == pytest.approx(0.4)

    def test_certain_lexical_hit_outranks_mediocre_dense_neighbors(self):
        """The pinpoint-document failure mode, inverted: top BM25 with weak
        vector support must beat vector rows at middling similarity."""
        vec = [_chunk("noise.md", i, distance=0.8) for i in range(5)]
        lex = [_chunk("target.pdf", 0, bm25=30.0), _chunk("noise.md", 0, bm25=3.0)]
        fused = fuse_arms(vec, lex, alpha=0.5)
        assert fused[0].source == "target.pdf"

    def test_alpha_one_is_pure_vector(self):
        fused = fuse_arms(
            [_chunk("v.md", 0, distance=0.2)], [_chunk("l.md", 0, bm25=50.0)], alpha=1.0
        )
        scores = {r.source: r.score for r in fused}
        assert scores["v.md"] == pytest.approx(0.8)
        assert scores["l.md"] == 0.0

    def test_alpha_zero_is_pure_lexical(self):
        fused = fuse_arms(
            [_chunk("v.md", 0, distance=0.2)], [_chunk("l.md", 0, bm25=50.0)], alpha=0.0
        )
        scores = {r.source: r.score for r in fused}
        assert scores["l.md"] == pytest.approx(1.0)
        assert scores["v.md"] == 0.0

    def test_dedup_keeps_both_provenance_fields(self):
        fused = fuse_arms(
            [_chunk("a.md", 0, distance=0.3)], [_chunk("a.md", 0, bm25=9.0)], alpha=0.5
        )
        assert len(fused) == 1
        assert fused[0].distance == pytest.approx(0.3)
        assert fused[0].bm25_score == pytest.approx(9.0)

    def test_sorted_descending_by_score(self):
        fused = fuse_arms(
            [_chunk("far.md", 0, distance=1.5), _chunk("near.md", 0, distance=0.1)],
            [],
            alpha=1.0,
        )
        assert [r.source for r in fused] == ["near.md", "far.md"]
        assert all(r.score is not None for r in fused)


class TestDropUnsupportedFarRows:
    def test_disabled_when_max_distance_zero(self):
        from lilbee.data.store.core import _drop_unsupported_far_rows

        rows = [_chunk("far.md", 0, distance=1.9)]
        assert _drop_unsupported_far_rows(rows, 0.0) == rows

    def test_drops_vector_only_far_row_keeps_supported(self):
        from lilbee.data.store.core import _drop_unsupported_far_rows

        far_unsupported = _chunk("drift.md", 0, distance=1.4)
        far_supported = _chunk("identifier.md", 0, distance=1.4, bm25=25.0)
        kept = _drop_unsupported_far_rows([far_unsupported, far_supported], 0.75)
        assert [r.source for r in kept] == ["identifier.md"]
