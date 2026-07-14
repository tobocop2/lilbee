"""Tests for reciprocal-rank fusion of the vector and BM25 arms."""

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
        fused = fuse_arms([both, vec_only], [lex])
        scores = {(r.source, r.chunk_index): r.score for r in fused}
        assert scores[("a.md", 0)] > scores[("b.md", 0)]

    def test_top_of_both_arms_scores_one(self):
        fused = fuse_arms([_chunk("a.md", 0, distance=0.3)], [_chunk("a.md", 0, bm25=12.0)])
        assert fused[0].score == pytest.approx(1.0)

    def test_lexical_only_row_survives_with_score(self):
        """The identifier case: a row invisible to the vector arm must carry
        real fused weight from its BM25-arm rank alone."""
        vec = [_chunk("noise.md", i, distance=0.5) for i in range(3)]
        lex = [_chunk("catalog_482.pdf", 0, bm25=35.0)]
        fused = fuse_arms(vec, lex)
        lexical_row = next(r for r in fused if r.source == "catalog_482.pdf")
        assert lexical_row.score == pytest.approx(0.5)

    def test_top_lexical_hit_outranks_all_but_the_top_dense_neighbor(self):
        """The pinpoint-document failure mode: an FTS-arm top hit unseen by
        the vector arm must rank above every vector row except at most the
        vector arm's own number one."""
        vec = [_chunk("noise.md", i, distance=0.1) for i in range(30)]
        lex = [_chunk("target.pdf", 0, bm25=30.0), _chunk("noise.md", 0, bm25=3.0)]
        fused = fuse_arms(vec, lex)
        rank = next(i for i, r in enumerate(fused) if r.source == "target.pdf")
        assert rank <= 1

    def test_rank_order_ignores_score_magnitude(self):
        """A wildly stronger BM25 score at rank 2 stays rank 2: fusion is
        scale-free by construction."""
        lex = [_chunk("first.md", 0, bm25=5.0), _chunk("second.md", 0, bm25=500.0)]
        fused = fuse_arms([], lex)
        assert [r.source for r in fused] == ["first.md", "second.md"]

    def test_dedup_keeps_both_provenance_fields(self):
        fused = fuse_arms([_chunk("a.md", 0, distance=0.3)], [_chunk("a.md", 0, bm25=9.0)])
        assert len(fused) == 1
        assert fused[0].distance == pytest.approx(0.3)
        assert fused[0].bm25_score == pytest.approx(9.0)

    def test_sorted_descending_by_score(self):
        fused = fuse_arms(
            [_chunk("near.md", 0, distance=0.1), _chunk("far.md", 0, distance=1.5)],
            [],
        )
        assert [r.source for r in fused] == ["near.md", "far.md"]
        assert all(r.score is not None for r in fused)


class TestRegressionMechanism:
    """Synthetic reproduction of the graded-A/B regression shape: a lexical
    query whose relevant passages are mutually similar (they all quote the
    same identifiers) competing against semantically-generic neighbors."""

    @staticmethod
    def _relevant(i, dim=8):
        # Relevant rows: strong BM25, decent query-sim, and nearly identical
        # to EACH OTHER (they all quote the same identifier table).
        v = [0.9] * dim
        v[0] += i * 0.001
        return SearchChunk(
            source=f"ledger_{i}.txt",
            content_type="text",
            chunk_type="raw",
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk=f"identifier table row {i}",
            chunk_index=i,
            vector=v,
            distance=0.45,
            bm25_score=30.0 - i,
        )

    @staticmethod
    def _generic(i, dim=8):
        # Generic rows: no lexical support, slightly better query-sim,
        # mutually diverse.
        v = [0.1] * dim
        v[i % dim] = 0.9
        return SearchChunk(
            source=f"essay_{i}.txt",
            content_type="text",
            chunk_type="raw",
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk=f"general discussion {i}",
            chunk_index=i,
            vector=v,
            distance=0.38,
            bm25_score=None,
        )

    def test_fusion_alone_keeps_the_lexical_cluster(self):
        """Rank fusion by itself keeps the both-arm relevant rows above
        vector-only generics: the fusion ordering is not the regression."""
        relevant = [self._relevant(i) for i in range(15)]
        generic = [self._generic(i) for i in range(35)]
        fused = fuse_arms(relevant + generic, [r.model_copy() for r in relevant])
        top12 = fused[:12]
        assert sum(1 for r in top12 if r.source.startswith("ledger")) >= 10

    def test_mmr_diversifies_away_the_relevant_cluster(self):
        """MMR at the default lambda over the fused pool demotes the mutually
        similar relevant rows: the regression mechanism, and why the hybrid
        path returns the fused ordering without diversity selection."""
        from lilbee.data.store import mmr_rerank

        relevant = [self._relevant(i) for i in range(15)]
        generic = [self._generic(i) for i in range(35)]
        fused = fuse_arms(relevant + generic, [r.model_copy() for r in relevant])
        query_vec = [0.5] * 8
        selected = mmr_rerank(query_vec, fused, 12, 0.5)
        kept_relevant = sum(1 for r in selected if r.source.startswith("ledger"))
        # The mutually-similar relevant set gets thinned hard in favor of
        # diverse generics; this documents the mechanism the graded A/B saw.
        assert kept_relevant < 10


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
