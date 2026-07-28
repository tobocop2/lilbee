"""Result filtering, sorting, deduplication, and source-diversity helpers."""

from __future__ import annotations

import re

from lilbee.core.config import active_config
from lilbee.data.store import SearchChunk

_DEFAULT_RELEVANCE_WEIGHT = 0.5

_WHITESPACE_RE = re.compile(r"\s+")

# Neutral point of the [0, 1] fusion scale: used for a candidate with no usable
# score and for a cohort whose scores are all equal (no spread to rank on).
_NEUTRAL_SCORE = 0.5


def _relevance_weight(result: SearchChunk) -> float:
    """Return a [0, 1] relevance weight for distance-aware selection.

    Every store search path stamps the canonical ``score``; a scoreless row
    (constructed by hand, never by retrieval) weighs neutrally rather than
    resurrecting the pre-score per-arm arithmetic.
    """
    if result.score is not None:
        return min(1.0, max(0.0, result.score))
    return _DEFAULT_RELEVANCE_WEIGHT


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]; an all-equal set maps to the midpoint."""
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    if score_range > 0:
        return [(s - min_score) / score_range for s in scores]
    return [_NEUTRAL_SCORE] * len(scores)


def fusion_norms(results: list[SearchChunk]) -> list[float]:
    """Normalize each chunk's canonical score to [0, 1] across the result set.

    Every retrieval path scores in the same [0, 1] family, so one min-max
    pass suffices; a scoreless hand-built row sits at the neutral midpoint.
    """
    scored = [i for i, r in enumerate(results) if r.score is not None]
    norms = [_NEUTRAL_SCORE] * len(results)
    if scored:
        scaled = normalize_scores([results[i].score or 0.0 for i in scored])
        for i, value in zip(scored, scaled, strict=True):
            norms[i] = value
    return norms


def order_by_fusion(results: list[SearchChunk]) -> list[SearchChunk]:
    """Sort results best-first by canonical score (stable on ties)."""
    norms = fusion_norms(results)
    order = sorted(range(len(results)), key=lambda i: norms[i], reverse=True)
    return [results[i] for i in order]


def _greedy_cover(
    chunk_tokens: list[set[str]],
    question_terms: set[str],
    term_weights: dict[str, float],
    budget: int,
    relevance_weights: list[float] | None = None,
) -> list[int]:
    """Greedy weighted set cover: pick chunks that add the most uncovered weight.

    Standard (1 - 1/e) approximation for weighted set cover. Budget is
    always filled, falling back to retrieval order once no chunk can
    contribute any new weight. When *relevance_weights* is provided,
    each chunk's IDF gain is scaled by its relevance so that far-away
    chunks are penalised even when they share query terms.
    """
    selected: list[int] = []
    covered: set[str] = set()
    remaining = list(range(len(chunk_tokens)))
    while remaining and len(selected) < budget:
        best_pos = -1
        best_gain = 0.0
        for pos, idx in enumerate(remaining):
            new_terms = (chunk_tokens[idx] & question_terms) - covered
            gain = sum(term_weights[t] for t in new_terms)
            if relevance_weights is not None:
                gain *= relevance_weights[idx]
            if gain > best_gain:
                best_gain = gain
                best_pos = pos
        if best_pos < 0:
            break
        chosen = remaining.pop(best_pos)
        selected.append(chosen)
        covered |= chunk_tokens[chosen] & question_terms

    for idx in remaining:
        if len(selected) >= budget:
            break
        selected.append(idx)
    return selected


def filter_results(
    results: list[SearchChunk],
    max_distance: float,
    min_relevance_score: float = 0.0,
) -> list[SearchChunk]:
    """Drop results below min_relevance_score or above max_distance.

    ``min_relevance_score`` gates on the [0, 1] fused score, which normalizes
    against the arms in play, so the threshold means the same thing across
    queries. ``max_distance`` additionally drops rows whose only
    signal is a far vector match (a row with lexical support keeps its standing
    regardless of distance). Pass max_distance=0 to disable distance filtering.
    """
    if max_distance <= 0 and min_relevance_score <= 0:
        return results
    filtered: list[SearchChunk] = []
    for r in results:
        if min_relevance_score > 0 and r.score is not None and r.score < min_relevance_score:
            continue
        if (
            max_distance > 0
            and r.bm25_score is None
            and r.distance is not None
            and r.distance > max_distance
        ):
            continue
        filtered.append(r)
    return filtered


def _sort_key(r: SearchChunk) -> float:
    """Sort key: lower = more relevant; a scoreless hand-built row sorts last."""
    if r.score is not None:
        return -r.score
    return float("inf")


def sort_by_relevance(results: list[SearchChunk]) -> list[SearchChunk]:
    """Sort search results by relevance (works for both hybrid and vector results)."""
    return sorted(results, key=_sort_key)


def diversify_sources(
    results: list[SearchChunk], max_per_source: int | None = None
) -> list[SearchChunk]:
    """Cap results per source document to ensure diversity.
    Source diversity filtering: Zhai 2008, "Statistical Language Models for
    Information Retrieval" -- caps per-source representation to prevent
    any single document from dominating results.

    Callers holding a Config pass its ``diversity_max_per_source`` so the
    library API's scoped config is honored; the active-config default only
    covers direct ad-hoc calls.
    """
    if max_per_source is None:
        max_per_source = active_config().diversity_max_per_source
    counts: dict[str, int] = {}
    diverse: list[SearchChunk] = []
    for r in results:
        count = counts.get(r.source, 0)
        if count < max_per_source:
            diverse.append(r)
            counts[r.source] = count + 1
    return diverse


def dedup_near_identical(results: list[SearchChunk]) -> list[SearchChunk]:
    """Keep the first (best-ranked) copy of passages with identical normalized text.

    The per-source cap cannot catch the same file ingested under two paths or
    boilerplate repeated across documents; those copies add no information and
    crowd real passages out of the context.
    """
    seen: set[str] = set()
    kept: list[SearchChunk] = []
    for r in results:
        key = _WHITESPACE_RE.sub(" ", r.chunk).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept


def prepare_results(
    results: list[SearchChunk], max_per_source: int | None = None
) -> list[SearchChunk]:
    """Sort by relevance, drop near-identical copies, apply the source diversity cap."""
    return diversify_sources(dedup_near_identical(sort_by_relevance(results)), max_per_source)
