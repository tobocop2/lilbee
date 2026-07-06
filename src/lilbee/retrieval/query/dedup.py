"""Result filtering, sorting, deduplication, and source-diversity helpers."""

from __future__ import annotations

from lilbee.core.config import cfg
from lilbee.data.store import CitationRecord, SearchChunk
from lilbee.retrieval.query.formatting import format_source

_DEFAULT_RELEVANCE_WEIGHT = 0.5

# Neutral point of the [0, 1] fusion scale: used for a candidate with no usable
# score and for a cohort whose scores are all equal (no spread to rank on).
_NEUTRAL_SCORE = 0.5


def _relevance_weight(result: SearchChunk) -> float:
    """Return a [0, 1] relevance weight for distance-aware selection.

    Hybrid results (relevance_score set): use directly.
    Vector results (distance set): invert cosine distance.
    Neither: neutral default.
    """
    if result.relevance_score is not None:
        return min(1.0, max(0.0, result.relevance_score))
    if result.distance is not None:
        return max(0.0, 1.0 - result.distance)
    return _DEFAULT_RELEVANCE_WEIGHT


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]; an all-equal set maps to the midpoint."""
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    if score_range > 0:
        return [(s - min_score) / score_range for s in scores]
    return [_NEUTRAL_SCORE] * len(scores)


def _fusion_signal(result: SearchChunk) -> float:
    """A chunk's retrieval confidence as a "higher = better" raw signal.

    Hybrid rows carry an RRF ``relevance_score`` (small positive magnitude);
    vector-only rows carry a cosine ``distance`` (0.0 = identical, lower = better).
    ``is None`` rather than truthiness is deliberate: a perfect vector match has
    ``distance == 0.0`` -- the strongest possible hit -- which falsy ``or`` would
    misread as the neutral default.
    """
    if result.relevance_score is not None:
        return result.relevance_score
    if result.distance is not None:
        return 1.0 - result.distance
    return _NEUTRAL_SCORE


def fusion_norms(results: list[SearchChunk]) -> list[float]:
    """Normalize each chunk's fusion signal to [0, 1] WITHIN its scoring family.

    Hybrid rows carry an RRF ``relevance_score`` (tiny magnitude); the rest
    (vector-only / HyDE recalls) carry a cosine ``distance``. The two scales are
    not comparable, so normalizing them together would let one family dominate
    purely as a scale artifact. Each family is scaled independently; a row with
    neither signal sits in the non-RRF family at the neutral score.
    """
    rrf = [i for i, r in enumerate(results) if r.relevance_score is not None]
    non_rrf = [i for i, r in enumerate(results) if r.relevance_score is None]
    norms = [_NEUTRAL_SCORE] * len(results)
    for cohort in (rrf, non_rrf):
        if not cohort:
            continue
        scaled = normalize_scores([_fusion_signal(results[i]) for i in cohort])
        for i, value in zip(cohort, scaled, strict=True):
            norms[i] = value
    return norms


def order_by_fusion(results: list[SearchChunk]) -> list[SearchChunk]:
    """Sort results best-first by fusion signal, normalized within each scoring
    family so RRF (hybrid) and cosine-distance (vector/HyDE) rows are comparable
    and one scale can't dominate the order as an artifact of its magnitude.
    """
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
    """Drop results above max_distance or below min_relevance_score.

    Hybrid results (relevance_score set) are checked against min_relevance_score.
    Vector results (distance set) are checked against max_distance.
    Results with neither score pass through. When both scores are present,
    relevance_score takes priority (hybrid results use RRF scoring, not
    cosine distance). Pass max_distance=0 to disable distance filtering.
    """
    if max_distance <= 0 and min_relevance_score <= 0:
        return results
    filtered: list[SearchChunk] = []
    for r in results:
        # Hybrid results: check relevance_score (takes priority over distance)
        if r.relevance_score is not None:
            if min_relevance_score > 0 and r.relevance_score < min_relevance_score:
                continue
        elif r.distance is not None and max_distance > 0 and r.distance > max_distance:
            continue
        filtered.append(r)
    return filtered


def deduplicate_sources(
    results: list[SearchChunk],
    max_citations: int = 5,
    citations_map: dict[str, list[CitationRecord]] | None = None,
) -> list[str]:
    """Merge results from same source into deduplicated citation lines."""
    seen: set[str] = set()
    citation_lines: list[str] = []
    for r in results:
        cits = (citations_map or {}).get(r.source)
        line = format_source(r, citations=cits)
        if line not in seen:
            seen.add(line)
            citation_lines.append(line)
            if len(citation_lines) >= max_citations:
                break
    return citation_lines


def _sort_key(r: SearchChunk) -> float:
    """Sort key: lower = more relevant."""
    if r.relevance_score is not None:
        return -r.relevance_score
    if r.distance is not None:
        return r.distance
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
    """
    if max_per_source is None:
        max_per_source = cfg.diversity_max_per_source
    counts: dict[str, int] = {}
    diverse: list[SearchChunk] = []
    for r in results:
        count = counts.get(r.source, 0)
        if count < max_per_source:
            diverse.append(r)
            counts[r.source] = count + 1
    return diverse


def prepare_results(results: list[SearchChunk]) -> list[SearchChunk]:
    """Sort by relevance and apply source diversity cap."""
    return diversify_sources(sort_by_relevance(results))
