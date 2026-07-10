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

    The canonical ``score`` is authoritative when present. Legacy rows fall
    back to their arm-specific signal: RRF ``relevance_score`` (whose tiny
    magnitude made hybrid rows weigh ~0.03 against ~0.4 for vector rows,
    inverting the ranking inside greedy set cover) or inverted distance.
    """
    if result.score is not None:
        return min(1.0, max(0.0, result.score))
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

    The canonical ``score`` wins when present. Legacy rows: RRF
    ``relevance_score`` (small positive magnitude) or cosine ``distance``
    (0.0 = identical, lower = better). ``is None`` rather than truthiness is
    deliberate: a perfect vector match has ``distance == 0.0`` -- the
    strongest possible hit -- which falsy ``or`` would misread as neutral.
    """
    if result.score is not None:
        return result.score
    if result.relevance_score is not None:
        return result.relevance_score
    if result.distance is not None:
        return 1.0 - result.distance
    return _NEUTRAL_SCORE


def fusion_norms(results: list[SearchChunk]) -> list[float]:
    """Normalize each chunk's fusion signal to [0, 1] WITHIN its scoring family.

    Rows carrying the canonical ``score`` are one family (already mutually
    comparable). Legacy rows split as before: RRF ``relevance_score`` (tiny
    magnitude) versus cosine ``distance``; those scales are not comparable,
    so normalizing them together would let one family dominate purely as a
    scale artifact. A row with no signal at all sits at the neutral score.
    """
    canonical = [i for i, r in enumerate(results) if r.score is not None]
    rrf = [i for i, r in enumerate(results) if r.score is None and r.relevance_score is not None]
    legacy = [i for i, r in enumerate(results) if r.score is None and r.relevance_score is None]
    norms = [_NEUTRAL_SCORE] * len(results)
    for cohort in (canonical, rrf, legacy):
        if not cohort:
            continue
        scaled = normalize_scores([_fusion_signal(results[i]) for i in cohort])
        for i, value in zip(cohort, scaled, strict=True):
            norms[i] = value
    return norms


def order_by_fusion(results: list[SearchChunk]) -> list[SearchChunk]:
    """Sort results best-first by fusion signal, normalized within each scoring
    family so rows carrying different signals (canonical score, legacy RRF,
    cosine distance) stay comparable and one scale can't dominate the order as
    an artifact of its magnitude.
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
    """Drop results below min_relevance_score or above max_distance.

    Rows carrying the canonical ``score`` are gated on ``min_relevance_score``,
    which is what makes an abstention threshold possible: the score is [0, 1]
    with real meaning, unlike RRF magnitudes. ``max_distance`` additionally
    drops rows whose only signal is a far vector match (a row with lexical
    support keeps its standing regardless of distance). Legacy rows keep the
    old per-family checks. Pass max_distance=0 to disable distance filtering.
    """
    if max_distance <= 0 and min_relevance_score <= 0:
        return results
    filtered: list[SearchChunk] = []
    for r in results:
        if r.score is not None:
            if min_relevance_score > 0 and r.score < min_relevance_score:
                continue
            if (
                max_distance > 0
                and r.bm25_score is None
                and r.distance is not None
                and r.distance > max_distance
            ):
                continue
        elif r.relevance_score is not None:
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
    """Sort key: lower = more relevant.

    Canonical ``score`` first. The legacy branches are ordered so that
    comparing a leftover RRF row (key ~ -0.03) with a distance row (key >= 0)
    keeps the old bias only among rows the store did not score; scored rows
    never hit them.
    """
    if r.score is not None:
        return -r.score
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
