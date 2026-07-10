"""Score-aware fusion of the vector and BM25 retrieval arms.

Rank-reciprocal fusion (the previous mechanism) discards score magnitude by
construction: it cannot tell an arm that was certain from an arm that was
guessing, so one strong lexical hit for an identifier query drowns under
mediocre dense neighbors. Fusing normalized scores keeps that information:

    score = alpha * vector_similarity + (1 - alpha) * normalized_bm25

Vector similarity is ``1 - cosine_distance`` clamped to [0, 1], an absolute
signal. BM25 is unbounded and corpus-dependent, so it is normalized against
the top score of the result list (rank-stable, in (0, 1]). A row seen by only
one arm scores zero on the other, which is itself signal: lexical-only rows
survive fusion on their BM25 strength instead of being invisible to a
rank-based scheme fed from a shallow pool.
"""

from __future__ import annotations

from .types import SearchChunk


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


def fuse_arms(
    vector_rows: list[SearchChunk],
    fts_rows: list[SearchChunk],
    alpha: float,
) -> list[SearchChunk]:
    """Merge the two arms into one list scored by convex combination.

    ``alpha`` weights the vector arm; ``1 - alpha`` the BM25 arm. Rows found
    by both arms carry both provenance fields. The result is sorted by
    ``score`` descending and deduplicated on ``(source, chunk_index)``.
    """
    bm25_norms = dict(
        zip(
            (_key(r) for r in fts_rows),
            normalized_bm25([r.bm25_score or 0.0 for r in fts_rows]),
            strict=True,
        )
    )
    merged: dict[tuple[str, int], SearchChunk] = {}
    for row in vector_rows:
        sim = vector_similarity(row.distance) if row.distance is not None else 0.0
        merged[_key(row)] = row.model_copy(update={"score": alpha * sim})
    for row in fts_rows:
        key = _key(row)
        lexical = (1.0 - alpha) * bm25_norms[key]
        seen = merged.get(key)
        if seen is None:
            merged[key] = row.model_copy(update={"score": lexical})
        else:
            merged[key] = seen.model_copy(
                update={"score": (seen.score or 0.0) + lexical, "bm25_score": row.bm25_score}
            )
    return sorted(merged.values(), key=lambda r: r.score or 0.0, reverse=True)
