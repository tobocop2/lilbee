"""Cross-encoder reranking for search results.

Optional precision pass that scores each (query, chunk) pair through the
active provider's ``rerank`` method. Only active when
``cfg.reranker_model`` is set.

Core technique: Nogueira & Cho 2019, "Passage Re-ranking with BERT"
(https://arxiv.org/abs/1901.04085).

Position-aware blending: derived from learning-to-rank literature
(Burges et al. 2005). Top positions trust hybrid fusion more, lower
positions trust the reranker more.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from lilbee.core.config import Config
from lilbee.data.store import SearchChunk

log = logging.getLogger(__name__)


class ScoredChunk(NamedTuple):
    """A search chunk paired with its blended score."""

    score: float
    chunk: SearchChunk


_TOP_POSITION_CUTOFF = 3
_MID_POSITION_CUTOFF = 10

_BLEND_SCHEDULE = {
    "top": (0.70, 0.30),
    "mid": (0.50, 0.50),
    "bottom": (0.30, 0.70),
}


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize raw cross-encoder scores to [0, 1]."""
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    if score_range > 0:
        return [(s - min_score) / score_range for s in scores]
    return [0.5] * len(scores)


def _fusion_signal(chunk: SearchChunk) -> float:
    """A chunk's retrieval confidence as a "higher = better" raw signal.

    Hybrid rows carry an RRF ``relevance_score`` (small positive magnitude);
    vector-only rows carry a cosine ``distance`` (0.0 = identical, lower = better).
    Both are mapped to higher-is-better; ``_blend_scores`` min-max normalizes the
    whole candidate set so their differing magnitudes become comparable.

    ``is None`` rather than truthiness is deliberate: a perfect vector match has
    ``distance == 0.0`` -- the strongest possible hit -- which falsy ``or`` would
    misread as the 0.5 default.
    """
    if chunk.relevance_score is not None:
        return chunk.relevance_score
    if chunk.distance is not None:
        return 1.0 - chunk.distance
    return 0.5


def _fusion_norms(to_rerank: list[SearchChunk]) -> list[float]:
    """Min-max normalize the fusion signal WITHIN each scoring family.

    Hybrid rows carry an RRF ``relevance_score`` (tiny magnitude); the rest
    (vector-only / HyDE recalls) carry a cosine ``distance``. The two scales are
    not comparable, so normalizing them together would let one family dominate
    purely as a scale artifact. Each family is scaled to [0, 1] independently;
    a row with neither signal sits in the non-RRF family at ``_fusion_signal``'s
    neutral 0.5.
    """
    rrf = [i for i, c in enumerate(to_rerank) if c.relevance_score is not None]
    non_rrf = [i for i, c in enumerate(to_rerank) if c.relevance_score is None]
    norms = [0.5] * len(to_rerank)
    for cohort in (rrf, non_rrf):
        if not cohort:
            continue
        scaled = _normalize_scores([_fusion_signal(to_rerank[i]) for i in cohort])
        for i, value in zip(cohort, scaled, strict=True):
            norms[i] = value
    return norms


def _blend_scores(
    to_rerank: list[SearchChunk], norm_scores: list[float], fusion_norms: list[float]
) -> list[ScoredChunk]:
    """Blend fusion scores with reranker scores using position-aware weights.

    Both inputs are already min-max normalized to [0, 1] across the candidate
    set, so a strong hybrid hit (whose raw RRF score is tiny in absolute terms)
    still earns real fusion weight. Each chunk is copied with ``rerank_score``
    set to its blended score; the input chunks are left untouched.
    """
    blended: list[ScoredChunk] = []
    for i, (chunk, rerank_score, fusion_norm) in enumerate(
        zip(to_rerank, norm_scores, fusion_norms, strict=True)
    ):
        if i < _TOP_POSITION_CUTOFF:
            fw, rw = _BLEND_SCHEDULE["top"]
        elif i < _MID_POSITION_CUTOFF:
            fw, rw = _BLEND_SCHEDULE["mid"]
        else:
            fw, rw = _BLEND_SCHEDULE["bottom"]

        final_score = fw * fusion_norm + rw * rerank_score
        scored = chunk.model_copy(update={"rerank_score": final_score})
        blended.append(ScoredChunk(final_score, scored))
    return blended


class Reranker:
    """Cross-encoder reranker with position-aware blending.

    Delegates scoring to the active provider's ``rerank``; blends the result with
    the normalized retrieval fusion signal so a confident hybrid hit keeps its
    standing against a reranker that favours a weaker chunk (Nogueira & Cho 2019,
    https://arxiv.org/abs/1901.04085).
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def rerank(
        self,
        query: str,
        results: list[SearchChunk],
        candidates: int | None = None,
    ) -> list[SearchChunk]:
        """Rerank search results through the provider's ``rerank`` method."""
        if not self._config.reranker_model:
            return results
        if candidates is None:
            candidates = self._config.rerank_candidates
        to_rerank = results[:candidates]
        remainder = results[candidates:]

        if not to_rerank:
            return results

        scores = _score_candidates(query, to_rerank)
        if scores is None:
            return results

        norm_scores = _normalize_scores(scores)
        fusion_norms = _fusion_norms(to_rerank)
        blended = _blend_scores(to_rerank, norm_scores, fusion_norms)
        blended_sorted = sorted(blended, key=lambda x: x.score, reverse=True)

        reranked = [chunk for _, chunk in blended_sorted]
        return reranked + remainder


def _score_candidates(query: str, to_rerank: list[SearchChunk]) -> list[float] | None:
    """Call the active provider's rerank; return None on error after logging."""
    # circular: services -> reranker via Searcher; deferred so test-time
    # monkeypatching of ``lilbee.services.get_services`` stays effective.
    from lilbee.app.services import get_services

    try:
        provider = get_services().provider
        return provider.rerank(query, [c.chunk for c in to_rerank])
    except Exception as exc:
        log.warning("Reranker failed; skipping rerank pass: %s", exc, exc_info=True)
        return None
