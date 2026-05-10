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


def _blend_scores(to_rerank: list[SearchChunk], norm_scores: list[float]) -> list[ScoredChunk]:
    """Blend fusion scores with reranker scores using position-aware weights."""
    blended: list[ScoredChunk] = []
    for i, (chunk, rerank_score) in enumerate(zip(to_rerank, norm_scores, strict=True)):
        fusion_score = chunk.relevance_score or (1.0 - (chunk.distance or 0.5))
        fusion_norm = max(0.0, min(1.0, fusion_score))

        if i < _TOP_POSITION_CUTOFF:
            fw, rw = _BLEND_SCHEDULE["top"]
        elif i < _MID_POSITION_CUTOFF:
            fw, rw = _BLEND_SCHEDULE["mid"]
        else:
            fw, rw = _BLEND_SCHEDULE["bottom"]

        final_score = fw * fusion_norm + rw * rerank_score
        blended.append(ScoredChunk(final_score, chunk))
    return blended


def _pin_original_top(
    blended: list[ScoredChunk],
    to_rerank: list[SearchChunk],
    skip_threshold: float,
) -> list[ScoredChunk]:
    """Pin the original top result if its relevance exceeds the skip threshold."""
    top_score = to_rerank[0].relevance_score or 0 if to_rerank else 0
    blended_sorted = sorted(blended, key=lambda x: x.score, reverse=True)
    if top_score >= skip_threshold:
        original_top = to_rerank[0]
        if blended_sorted[0].chunk is not original_top:
            blended_sorted = [ScoredChunk(999.0, original_top)] + [
                ScoredChunk(s, c) for s, c in blended_sorted if c is not original_top
            ]
    return blended_sorted


class Reranker:
    """Cross-encoder reranker with position-aware blending.

    Delegates scoring to the active provider's ``rerank``; handles result
    blending and the BM25-protection pin (Nogueira & Cho 2019,
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
        blended = _blend_scores(to_rerank, norm_scores)
        blended_sorted = _pin_original_top(
            blended, to_rerank, self._config.expansion_skip_threshold
        )

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
