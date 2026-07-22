"""Reciprocal-rank fusion of the vector, BM25, and optional title arms.

Each arm contributes ``(K + 1) / (K + rank)`` for the rows it retrieved
(rank is 1-based, K = 60, the standard RRF constant); a row's canonical
score is the weight-normalized sum of its arm contributions, so it lives
in [0, 1] and a row ranked first by every arm scores exactly 1. Rows seen
by only one arm score that arm's share of the total weight, which still
places an arm's top hit above every row deep in the other arms: the
property that keeps a lexical-only identifier match visible next to dense
neighbors. The vector arm weighs 1; the chunk-BM25 arm weighs
``lexical_weight`` (1.0 = equal voice, lower lets a strong dense arm
dominate); the optional title arm weighs ``title_weight``. Weights rescale
the shares without leaving the canonical range.

Scores normalize against the configured weight budget, not the per-query
adapted ``lexical_weight`` or whether a given query's title arm returned rows
(see ``fuse_arms``'s *weight_total*). That fixed denominator keeps scores on
one scale across queries and across the sub-searches a caller merges, so
``min_relevance_score`` stays a usable floor.

Rank fusion is deliberate. A convex combination of normalized raw scores
(``alpha * vector_similarity + (1 - alpha) * normalized_bm25``) was tried
here and measurably regressed graded precision: cosine similarities sit
in a high narrow band, giving every dense neighbor a floor that outranks
lexically-certain rows, and no blend weight fixes that asymmetry. Ranks
are scale-free, so neither arm's score distribution can crowd out the
other. Arm depth matters as much as the formula: rows both arms rank
mid-pool accumulate two contributions, so deep candidate pools crowd out
single-arm certainty; the hybrid path therefore feeds fusion arms of
exactly ``top_k`` rows.
"""

from __future__ import annotations

from statistics import fmean

from .types import SearchChunk

# Standard RRF smoothing constant (Cormack, Clarke & Buettcher 2009).
_RRF_K = 60

# Adaptive fusion needs a top hit plus at least one field row to measure a margin.
_MIN_ROWS_FOR_MARGIN = 2


def vector_similarity(distance: float) -> float:
    """Cosine distance to canonical [0, 1] similarity (distance spans [0, 2])."""
    return max(0.0, min(1.0, 1.0 - distance))


def adaptive_weight_scale(vector_rows: list[SearchChunk], margin_scale: float) -> float:
    """A [0, 1] factor to shrink the lexical arms by when the vector arm is
    confident about this query.

    A *peaked* vector ranking -- a top hit standing well clear of the field --
    means the dense embedder already located the answer and the lexical arms
    mostly add term-match noise. A *flat* ranking means dense is unsure and
    BM25's exact-term matching is worth trusting. The confidence signal is the
    margin between the top similarity and the mean of the rest, divided by
    *margin_scale*: at or above that margin the factor is 0 (arms silenced), at
    zero margin it is 1 (arms kept), scaling linearly between. Returns 1.0 when
    there is nothing to measure (fewer than two scored rows) or when
    *margin_scale* <= 0 (adaptation off).
    """
    if margin_scale <= 0:
        return 1.0
    sims = sorted(
        (vector_similarity(r.distance) for r in vector_rows if r.distance is not None),
        reverse=True,
    )
    if len(sims) < _MIN_ROWS_FOR_MARGIN:
        return 1.0
    margin = max(0.0, sims[0] - fmean(sims[1:]))
    confidence = min(1.0, margin / margin_scale)
    return 1.0 - confidence


def normalized_bm25(scores: list[float]) -> list[float]:
    """Scale raw BM25 scores against the list maximum, into (0, 1].

    BM25 has no absolute scale, so the top hit anchors the list; relative
    strength within one query's results is the meaningful quantity.
    Non-positive or absent maxima map everything to 0.
    """
    top = max(scores, default=0.0)
    if top <= 0.0:
        return [0.0] * len(scores)
    return [max(0.0, s) / top for s in scores]


def _key(chunk: SearchChunk) -> tuple[str, int]:
    return (chunk.source, chunk.chunk_index)


def _rank_weight(rank: int) -> float:
    """Reciprocal-rank contribution in (0, 1]; 1.0 at rank 1."""
    return (_RRF_K + 1) / (_RRF_K + rank)


def _merge_arm(
    merged: dict[tuple[str, int], SearchChunk],
    rows: list[SearchChunk],
    share: float,
) -> None:
    """Fold one arm's ranked rows into *merged*, each contributing *share* of its rank weight.

    A lexical row (title or chunk FTS) carries ``bm25_score``; when the row was
    already seen, that provenance is kept from whichever lexical arm set it
    first, so the lexical-support exemption applies either way.
    """
    for rank, row in enumerate(rows, start=1):
        key = _key(row)
        contribution = _rank_weight(rank) * share
        seen = merged.get(key)
        if seen is None:
            merged[key] = row.model_copy(update={"score": contribution})
        else:
            update: dict[str, object] = {"score": (seen.score or 0.0) + contribution}
            if seen.bm25_score is None and row.bm25_score is not None:
                update["bm25_score"] = row.bm25_score
            merged[key] = seen.model_copy(update=update)


def fuse_arms(
    vector_rows: list[SearchChunk],
    fts_rows: list[SearchChunk],
    title_rows: list[SearchChunk] | None = None,
    *,
    lexical_weight: float = 1.0,
    title_weight: float = 1.0,
    weight_total: float | None = None,
) -> list[SearchChunk]:
    """Merge the arms into one list scored by reciprocal rank.

    The vector arm weighs 1; the chunk-FTS (lexical) arm weighs *lexical_weight*
    relative to it (1.0 = equal voice, lower lets a strong dense arm dominate);
    a non-empty *title_rows* arm joins at *title_weight*. Rows found by several
    arms carry every provenance field (``distance`` from the vector arm,
    ``bm25_score`` from the FTS arms). The result is sorted by ``score``
    descending and deduplicated on ``(source, chunk_index)``.

    *weight_total* is the constant this call normalizes scores against. Pass the
    configured weight budget so scores from separate sub-searches -- whose
    adaptive *lexical_weight* and per-query title-arm presence differ -- stay on
    one comparable scale when a caller merges them. It is a uniform divisor, so
    the ranking within this call is unchanged. When omitted it defaults to the
    arms present in this call (the standalone behavior).
    """
    if weight_total is None:
        weight_total = 1.0 + lexical_weight + (title_weight if title_rows else 0.0)
    merged: dict[tuple[str, int], SearchChunk] = {}
    _merge_arm(merged, vector_rows, 1.0 / weight_total)
    # A zero-weight arm contributes nothing, so skip it rather than folding in
    # zero-score rows that would still carry lexical provenance (and its
    # downstream distance/structural exemptions) on no real support.
    if lexical_weight > 0:
        _merge_arm(merged, fts_rows, lexical_weight / weight_total)
    if title_rows and title_weight > 0:
        _merge_arm(merged, title_rows, title_weight / weight_total)
    return sorted(merged.values(), key=lambda r: r.score or 0.0, reverse=True)
