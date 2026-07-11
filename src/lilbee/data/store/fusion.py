"""Reciprocal-rank fusion of the vector and BM25 retrieval arms.

Each arm contributes ``(K + 1) / (K + rank)`` for the rows it retrieved
(rank is 1-based, K = 60, the standard RRF constant); a row's canonical
score is the mean of its two arm contributions, so it lives in [0, 1] and
a row ranked first by both arms scores exactly 1. Rows seen by only one
arm score half of that arm's contribution, which still places an arm's
top hit above every row deep in the other arm: the property that keeps a
lexical-only identifier match visible next to dense neighbors.

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

from .types import SearchChunk

# Standard RRF smoothing constant (Cormack, Clarke & Buettcher 2009).
_RRF_K = 60


def vector_similarity(distance: float) -> float:
    """Cosine distance to canonical [0, 1] similarity (distance spans [0, 2])."""
    return max(0.0, min(1.0, 1.0 - distance))


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


def fuse_arms(
    vector_rows: list[SearchChunk],
    fts_rows: list[SearchChunk],
) -> list[SearchChunk]:
    """Merge the two arms into one list scored by reciprocal rank.

    Both arms weigh equally. Rows found by both arms carry both provenance
    fields (``distance`` from the vector arm, ``bm25_score`` from the FTS
    arm). The result is sorted by ``score`` descending and deduplicated on
    ``(source, chunk_index)``.
    """
    merged: dict[tuple[str, int], SearchChunk] = {}
    for rank, row in enumerate(vector_rows, start=1):
        merged[_key(row)] = row.model_copy(update={"score": _rank_weight(rank) / 2})
    for rank, row in enumerate(fts_rows, start=1):
        key = _key(row)
        lexical = _rank_weight(rank) / 2
        seen = merged.get(key)
        if seen is None:
            merged[key] = row.model_copy(update={"score": lexical})
        else:
            merged[key] = seen.model_copy(
                update={"score": (seen.score or 0.0) + lexical, "bm25_score": row.bm25_score}
            )
    return sorted(merged.values(), key=lambda r: r.score or 0.0, reverse=True)
