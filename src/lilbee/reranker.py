"""Cross-encoder reranking for search results.

Optional precision pass that scores each (query, chunk) pair using a
cross-encoder model. Only active when ``cfg.reranker_model`` is set.

Core technique: Nogueira & Cho 2019, "Passage Re-ranking with BERT"
(https://arxiv.org/abs/1901.04085).

Position-aware blending: derived from learning-to-rank literature
(Burges et al. 2005). Top positions trust hybrid fusion more, lower
positions trust the reranker more.
"""

from __future__ import annotations

import logging
from typing import Any

from lilbee.config import Config
from lilbee.store import SearchChunk

log = logging.getLogger(__name__)

_TOP_POSITION_CUTOFF = 3
_MID_POSITION_CUTOFF = 10

_BLEND_SCHEDULE = {
    "top": (0.70, 0.30),
    "mid": (0.50, 0.50),
    "bottom": (0.30, 0.70),
}


def reranker_available() -> bool:
    """Check if sentence-transformers is installed."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize raw cross-encoder scores to [0, 1]."""
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    if score_range > 0:
        return [(s - min_score) / score_range for s in scores]
    return [0.5] * len(scores)


def _blend_scores(
    to_rerank: list[SearchChunk], norm_scores: list[float]
) -> list[tuple[float, SearchChunk]]:
    """Blend fusion scores with reranker scores using position-aware weights."""
    blended: list[tuple[float, SearchChunk]] = []
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
        blended.append((final_score, chunk))
    return blended


def _pin_original_top(
    blended: list[tuple[float, SearchChunk]],
    to_rerank: list[SearchChunk],
    skip_threshold: float,
) -> list[tuple[float, SearchChunk]]:
    """Pin the original top result if its relevance exceeds the skip threshold."""
    top_score = to_rerank[0].relevance_score or 0 if to_rerank else 0
    blended_sorted = sorted(blended, key=lambda x: x[0], reverse=True)
    if top_score >= skip_threshold:
        original_top = to_rerank[0]
        if blended_sorted[0][1] is not original_top:
            blended_sorted = [(999.0, original_top)] + [
                (s, c) for s, c in blended_sorted if c is not original_top
            ]
    return blended_sorted


class Reranker:
    """Cross-encoder reranker with position-aware blending.

    Core technique: Nogueira & Cho 2019, "Passage Re-ranking with BERT"
    (https://arxiv.org/abs/1901.04085).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._encoder: Any = None

    def _get_encoder(self) -> Any:
        """Lazy-load the cross-encoder model. Returns None if not configured."""
        if self._encoder is not None:
            return self._encoder
        model_name = self._config.reranker_model
        if not model_name:
            return None
        try:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(model_name)
            log.info("Loaded reranker model: %s", model_name)
            return self._encoder
        except ImportError:
            log.warning("sentence-transformers not installed -- reranking disabled")
            return None
        except Exception as exc:
            log.warning("Failed to load reranker model %s: %s", model_name, exc)
            return None

    def reset_encoder(self) -> None:
        """Clear the cached encoder. For testing only."""
        self._encoder = None

    def rerank(
        self,
        query: str,
        results: list[SearchChunk],
        candidates: int | None = None,
    ) -> list[SearchChunk]:
        """Rerank search results using a cross-encoder model."""
        encoder = self._get_encoder()
        if encoder is None:
            return results

        if candidates is None:
            candidates = self._config.rerank_candidates
        to_rerank = results[:candidates]
        remainder = results[candidates:]

        if not to_rerank:
            return results

        pairs = [(query, chunk.chunk) for chunk in to_rerank]
        scores = encoder.predict(pairs)

        norm_scores = _normalize_scores(scores)
        blended = _blend_scores(to_rerank, norm_scores)
        blended_sorted = _pin_original_top(
            blended, to_rerank, self._config.expansion_skip_threshold
        )

        reranked = [chunk for _, chunk in blended_sorted]
        return reranked + remainder
