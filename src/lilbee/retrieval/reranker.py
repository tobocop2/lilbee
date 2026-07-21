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
from lilbee.retrieval.query.dedup import fusion_norms as compute_fusion_norms
from lilbee.retrieval.query.dedup import normalize_scores

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


def _blend_scores(
    to_rerank: list[SearchChunk], norm_scores: list[float], fusion_norms: list[float]
) -> list[ScoredChunk]:
    """Blend fusion scores with reranker scores using position-aware weights.

    Both inputs are already min-max normalized to [0, 1] (``fusion_norms``
    across the pool's canonical scores, ``norm_scores`` across the reranker
    scores), so the blend weights compare like with like. Each chunk is
    copied with ``rerank_score`` set to its blended score; the input chunks
    are left untouched.
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

        norm_scores = normalize_scores(scores)
        if self._config.rerank_blend:
            fusion_norms = compute_fusion_norms(to_rerank)
            scored = _blend_scores(to_rerank, norm_scores, fusion_norms)
        else:
            # Pure cross-encoder ordering: no fusion blend, so the reranker's
            # effect is unattenuated (and measurable in isolation).
            scored = [
                ScoredChunk(s, c.model_copy(update={"rerank_score": s}))
                for s, c in zip(norm_scores, to_rerank, strict=True)
            ]
        scored_sorted = sorted(scored, key=lambda x: x.score, reverse=True)

        reranked = [chunk for _, chunk in scored_sorted]
        return reranked + remainder


def _score_candidates(query: str, to_rerank: list[SearchChunk]) -> list[float] | None:
    """Call the active provider's rerank; return None on error after logging.

    A provider that returns the wrong number of scores is contained here
    like any other failure: the scores cannot be paired with the candidates,
    so the pass is skipped and retrieval order stands.
    """
    # circular: services -> reranker via Searcher; deferred so test-time
    # monkeypatching of ``lilbee.app.services.get_services`` stays effective.
    from lilbee.app.services import get_services

    try:
        provider = get_services().provider
        scores = provider.rerank(query, [c.chunk for c in to_rerank])
    except Exception as exc:
        log.warning("Reranker failed; skipping rerank pass: %s", exc, exc_info=True)
        return None
    if len(scores) != len(to_rerank):
        log.warning(
            "Reranker returned %d scores for %d candidates; skipping rerank pass",
            len(scores),
            len(to_rerank),
        )
        return None
    return scores
