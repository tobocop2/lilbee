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
from lilbee.core.health_warnings import HealthWarning, WarningCode
from lilbee.data.store import SearchChunk
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.retrieval.query.dedup import fusion_norms as compute_fusion_norms
from lilbee.retrieval.query.dedup import normalize_scores

log = logging.getLogger(__name__)


_UNKNOWN = ProviderErrorKind.UNKNOWN

# The cause and remedy a user reads for each error kind. Error text never reaches a client:
# it can hold a server address with credentials, so the full error goes to the log only.
_FAILURE_TEXTS: dict[ProviderErrorKind, tuple[str, str]] = {
    ProviderErrorKind.AUTH: (
        "The reranker refused the credentials.",
        "Check the API key of the reranker preset.",
    ),
    ProviderErrorKind.RATE_LIMIT: ("The reranker is rate limited.", "Wait, then search again."),
    ProviderErrorKind.NOT_FOUND: (
        "The reranker model or preset was not found.",
        "Check reranker_model and the preset with lilbee status.",
    ),
    ProviderErrorKind.BAD_REQUEST: (
        "The reranker refused the request.",
        "Check reranker_model with lilbee status.",
    ),
    ProviderErrorKind.CONNECTION: (
        "The reranker did not answer.",
        "Check that the reranker server runs, then search again.",
    ),
    ProviderErrorKind.SERVER: (
        "The reranker reported a server error.",
        "Search again, or check the reranker server log.",
    ),
    ProviderErrorKind.CONTEXT_OVERFLOW: (
        "A candidate is longer than the reranker window.",
        "Use a reranker with a longer window.",
    ),
    ProviderErrorKind.CAPACITY: (
        "The reranker has no free capacity.",
        "Search again when other work ends.",
    ),
    ProviderErrorKind.STARTING: ("The reranker is still starting.", "Search again in a moment."),
    _UNKNOWN: (
        "The reranker raised an error.",
        "Search again, or clear reranker_model. Check the reranker with lilbee status.",
    ),
}


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
        *,
        degraded: list[HealthWarning],
    ) -> list[SearchChunk]:
        """Rerank search results through the provider's ``rerank`` method.

        A failed pass returns *results* unchanged and appends a warning to *degraded*.
        """
        if not self._config.reranker_model:
            return results
        if candidates is None:
            candidates = self._config.rerank_candidates
        to_rerank = results[:candidates]
        remainder = results[candidates:]

        if not to_rerank:
            return results

        scores = _score_candidates(query, to_rerank, degraded)
        if scores is None:
            return results

        floor = self._config.rerank_min_score
        if floor is not None:
            # Absolute floor on the raw provider score: min-max normalization
            # erases calibration, so without this a uniformly irrelevant pool
            # still yields a 1.0-scored "best" candidate.
            kept = [(s, c) for s, c in zip(scores, to_rerank, strict=True) if s >= floor]
            if len(kept) < len(to_rerank):
                log.info(
                    "Reranker dropped %d of %d candidates below rerank_min_score",
                    len(to_rerank) - len(kept),
                    len(to_rerank),
                )
            if not kept:
                return remainder
            scores = [s for s, _ in kept]
            to_rerank = [c for _, c in kept]

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


def _error_kind(exc: Exception) -> ProviderErrorKind:
    """The provider error kind of *exc*; a network or timeout error is a connection failure."""
    if isinstance(exc, ProviderError):
        return exc.kind
    if isinstance(exc, OSError):
        return ProviderErrorKind.CONNECTION
    return ProviderErrorKind.UNKNOWN


def _rerank_failed(degraded: list[HealthWarning], cause: str, remedy: str) -> None:
    """Record that the pass was skipped, with a fixed cause and remedy, once."""
    warning = HealthWarning(
        code=WarningCode.RERANK_FAILED,
        message=f"Reranking failed: results keep their retrieval order. {cause}",
        remedy=remedy,
    )
    if warning not in degraded:
        degraded.append(warning)


def _score_candidates(
    query: str, to_rerank: list[SearchChunk], degraded: list[HealthWarning]
) -> list[float] | None:
    """Call the active provider's rerank; return None on error after logging.

    A provider that returns the wrong number of scores is contained here
    like any other failure: the scores cannot be paired with the candidates,
    so the pass is skipped and retrieval order stands. The full error goes to
    the log; *degraded* gets a fixed text for the error kind.
    """
    # circular: services -> reranker via Searcher; deferred so test-time
    # monkeypatching of ``lilbee.app.services.get_services`` stays effective.
    from lilbee.app.services import get_services

    try:
        provider = get_services().provider
        # Title-prefixed passages: a cross-encoder cannot judge a chunk whose
        # relevance depends on its parent document from the bare text.
        scores = provider.rerank(
            query, [f"{c.title}\n{c.chunk}" if c.title else c.chunk for c in to_rerank]
        )
    except Exception as exc:
        log.warning("Reranker failed; skipping rerank pass: %s", exc, exc_info=True)
        cause, remedy = _FAILURE_TEXTS.get(_error_kind(exc), _FAILURE_TEXTS[_UNKNOWN])
        _rerank_failed(degraded, cause, remedy)
        return None
    if len(scores) != len(to_rerank):
        log.warning(
            "Reranker returned %d scores for %d candidates; skipping rerank pass",
            len(scores),
            len(to_rerank),
        )
        _rerank_failed(
            degraded,
            f"The reranker returned {len(scores)} scores for {len(to_rerank)} candidates.",
            _FAILURE_TEXTS[_UNKNOWN][1],
        )
        return None
    return scores
