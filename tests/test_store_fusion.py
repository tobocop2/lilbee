"""Tests for reciprocal-rank fusion of the vector and BM25 arms."""

import pytest

from lilbee.data.store import SearchChunk
from lilbee.data.store.fusion import (
    adaptive_weight_scale,
    fuse_arms,
    normalized_bm25,
    vector_similarity,
)


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


class TestEffectiveWeightNormalization:
    """Scores normalize against the arms taking part in the call, so every
    call lands on one canonical [0, 1] scale. A fixed shared denominator was
    tried and capped confident sub-searches (lexical adaptively quieted) at
    the vector arm's share, demoting the original query's hits below
    paraphrase-variant hits when Searcher merged them."""

    def test_quieted_lexical_restores_canonical_scale(self):
        # Lexical arm quieted to the adaptive floor: the vector-only top hit
        # approaches full scale instead of being capped near the vector share.
        fused = fuse_arms([_chunk("a.md", 0, distance=0.3)], [], lexical_weight=0.05)
        assert fused[0].score == pytest.approx(1.0 / 1.05)

    def test_confident_call_is_not_demoted_below_flat_call(self):
        # The same vector-only top hit scores at least as high in a call whose
        # lexical arm was quieted as in one at full lexical weight: confidence
        # must never demote.
        quieted = fuse_arms([_chunk("a.md", 0, distance=0.3)], [], lexical_weight=0.05)
        flat = fuse_arms([_chunk("a.md", 0, distance=0.3)], [], lexical_weight=1.0)
        assert quieted[0].score > flat[0].score

    def test_normalization_is_a_uniform_rescale_within_a_call(self):
        # The denominator is one constant per call, so it never changes the
        # ranking inside that call.
        vec = [_chunk("near.md", 0, distance=0.1), _chunk("far.md", 1, distance=0.9)]
        lex = [_chunk("far.md", 1, bm25=30.0)]
        half = [r.source for r in fuse_arms(vec, lex, lexical_weight=0.5)]
        full = [r.source for r in fuse_arms(vec, lex, lexical_weight=1.0)]
        assert half == full

    def test_empty_title_arm_does_not_deflate_scores(self):
        # title_weight only enters the denominator when title rows exist, so
        # enabling title search does not silently rescale titleless queries.
        without = fuse_arms(
            [_chunk("a.md", 0, distance=0.3)], [_chunk("a.md", 0, bm25=9.0)], title_weight=0.5
        )
        with_empty = fuse_arms(
            [_chunk("a.md", 0, distance=0.3)],
            [_chunk("a.md", 0, bm25=9.0)],
            [],
            title_weight=0.5,
        )
        assert without[0].score == pytest.approx(1.0)
        assert with_empty[0].score == pytest.approx(without[0].score)


class TestAdaptiveWeightScale:
    """Per-query weight scale gated by the vector arm's confidence."""

    def test_peaked_dense_quiets_lexical_to_the_floor(self):
        # top similarity 1.0 (distance 0), field ~0.3: a wide margin bottoms
        # out at the floor, never exactly zero, so BM25 provenance (and the
        # distance-cut exemption it grants) survives full vector confidence.
        rows = [_chunk("a.md", 0, distance=0.0)] + [
            _chunk("b.md", i, distance=0.7) for i in range(1, 5)
        ]
        assert adaptive_weight_scale(rows, 0.3) == pytest.approx(0.05)

    def test_scale_is_independent_of_pool_depth(self):
        # The margin reads a fixed window of runners-up, so retrieving more
        # rows (a reranker deepening the pool) must not change the scale.
        top = [_chunk("a.md", 0, distance=0.4)] + [
            _chunk("b.md", i, distance=0.6) for i in range(1, 6)
        ]
        deep = top + [_chunk("tail.md", i, distance=1.4) for i in range(6, 60)]
        assert adaptive_weight_scale(deep, 0.3) == pytest.approx(adaptive_weight_scale(top, 0.3))

    def test_flat_dense_keeps_full_weight(self):
        # every row equally similar: zero margin => full scale.
        rows = [_chunk("a.md", i, distance=0.5) for i in range(5)]
        assert adaptive_weight_scale(rows, 0.3) == pytest.approx(1.0)

    def test_scales_linearly_between(self):
        # top sim 0.6, field 0.3, margin 0.3, scale 0.6 => confidence 0.5 => half.
        rows = [_chunk("a.md", 0, distance=0.4)] + [
            _chunk("b.md", i, distance=0.7) for i in range(1, 4)
        ]
        assert adaptive_weight_scale(rows, 0.6) == pytest.approx(0.5)

    def test_too_few_rows_returns_full(self):
        assert adaptive_weight_scale([_chunk("a.md", 0, distance=0.1)], 0.3) == 1.0
        assert adaptive_weight_scale([], 0.3) == 1.0

    def test_non_positive_margin_scale_disables(self):
        rows = [_chunk("a.md", 0, distance=0.0), _chunk("b.md", 1, distance=0.9)]
        assert adaptive_weight_scale(rows, 0.0) == 1.0

    def test_ignores_rows_without_distance(self):
        # lexical-only rows carry no distance; they must not enter the signal.
        rows = [
            _chunk("a.md", 0, distance=0.0),
            _chunk("b.md", 1, distance=0.6),
            _chunk("c.md", 2, bm25=9.0),
        ]
        both = adaptive_weight_scale(rows, 0.4)
        two = adaptive_weight_scale(rows[:2], 0.4)
        assert both == pytest.approx(two)


class TestLexicalFusionWeight:
    """The BM25 arm's fusion weight can be lowered so a strong dense arm dominates."""

    def _lex_row(self, weight: float):
        vec = [_chunk("noise.md", 0, distance=0.5)]
        lex = [_chunk("cat.pdf", 0, bm25=35.0)]
        return next(r for r in fuse_arms(vec, lex, lexical_weight=weight) if r.source == "cat.pdf")

    def test_default_weight_gives_the_arms_equal_voice(self):
        vec = [_chunk("noise.md", 0, distance=0.5)]
        lex = [_chunk("cat.pdf", 0, bm25=35.0)]
        default = {r.source: r.score for r in fuse_arms(vec, lex)}
        explicit = {r.source: r.score for r in fuse_arms(vec, lex, lexical_weight=1.0)}
        assert default == explicit

    def test_zero_weight_drops_the_lexical_only_arm(self):
        """A fully-silenced lexical arm contributes no rows at all, so a
        BM25-only hit never enters the pool carrying lexical provenance (and
        with it the downstream distance/structural exemptions)."""
        vec = [_chunk("noise.md", 0, distance=0.5)]
        lex = [_chunk("cat.pdf", 0, bm25=35.0)]
        fused = fuse_arms(vec, lex, lexical_weight=0.0)
        assert "cat.pdf" not in [r.source for r in fused]

    def test_lower_weight_shrinks_the_lexical_contribution(self):
        assert self._lex_row(0.5).score < self._lex_row(1.0).score


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


class TestTitleArmFusion:
    """The optional third arm: BM25 over document titles, weight-normalized."""

    def test_top_of_all_three_arms_scores_one(self):
        fused = fuse_arms(
            [_chunk("a.md", 0, distance=0.3)],
            [_chunk("a.md", 0, bm25=12.0)],
            [_chunk("a.md", 0, bm25=3.0)],
            title_weight=0.5,
        )
        assert fused[0].score == pytest.approx(1.0)

    def test_title_only_row_scores_its_weight_share(self):
        weight = 0.5
        fused = fuse_arms([], [], [_chunk("t.md", 0, bm25=3.0)], title_weight=weight)
        assert fused[0].score == pytest.approx(weight / (2.0 + weight))

    def test_empty_title_arm_matches_two_arm_scores(self):
        """No title rows = the classic two-arm fusion, share-for-share."""
        vector = [_chunk("a.md", 0, distance=0.3), _chunk("b.md", 0, distance=0.4)]
        fts = [_chunk("a.md", 0, bm25=12.0)]
        two_arm = fuse_arms(vector, fts)
        with_empty_title = fuse_arms(vector, fts, [], title_weight=0.5)
        assert [(r.source, r.score) for r in two_arm] == [
            (r.source, r.score) for r in with_empty_title
        ]

    def test_title_match_counts_as_lexical_support(self):
        """A row only the title arm matched carries bm25_score, so the
        distance exemption sees lexical support."""
        fused = fuse_arms(
            [_chunk("a.md", 0, distance=1.5)],
            [],
            [_chunk("a.md", 0, bm25=4.0)],
            title_weight=0.5,
        )
        assert fused[0].bm25_score == pytest.approx(4.0)
        assert fused[0].distance == pytest.approx(1.5)

    def test_chunk_arm_bm25_provenance_wins_over_title(self):
        """When both lexical arms match a row, the first-seen bm25_score is kept."""
        fused = fuse_arms(
            [],
            [_chunk("a.md", 0, bm25=12.0)],
            [_chunk("a.md", 0, bm25=3.0)],
            title_weight=0.5,
        )
        assert fused[0].bm25_score == pytest.approx(12.0)

    def test_title_weight_scales_contribution(self):
        low = fuse_arms([], [], [_chunk("t.md", 0, bm25=3.0)], title_weight=0.2)
        high = fuse_arms([], [], [_chunk("t.md", 0, bm25=3.0)], title_weight=1.0)
        assert low[0].score < high[0].score
        assert high[0].score == pytest.approx(1.0 / 3.0)
