"""Faithfulness scoring and drift heuristics for wiki page bodies.

Holds the deterministic body-vs-source cosine score, the title/body
coherence pre-check that gates a page on a structurally valid heading,
and the unified-diff helpers used when an existing page's content
changed by more than the configured threshold (drift detection).
"""

from __future__ import annotations

import difflib
import logging

import numpy as np

from lilbee.core.config import Config
from lilbee.core.services import get_services
from lilbee.store import SearchChunk
from lilbee.wiki.citation import strip_citation_block
from lilbee.wiki.shared import clean_label_for_display, is_valid_label

log = logging.getLogger(__name__)

_MAX_DIFF_PREVIEW_LINES = 20  # lines of unified diff shown in drift warnings


def _content_change_ratio(old_text: str, new_text: str) -> float:
    """Fraction of lines that changed between two texts (0.0 = identical, 1.0 = total rewrite)."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    if not old_lines and not new_lines:
        return 0.0
    total = max(len(old_lines), len(new_lines))
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    changed = total - sum(block.size for block in matcher.get_matching_blocks())
    return changed / total


def _diff_summary(old_text: str, new_text: str) -> str:
    """Human-readable unified diff summary (first 20 diff lines)."""
    diff = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        lineterm="",
        fromfile="old",
        tofile="new",
    )
    lines = list(diff)
    if len(lines) > _MAX_DIFF_PREVIEW_LINES:
        extra = len(lines) - _MAX_DIFF_PREVIEW_LINES
        return "\n".join(lines[:_MAX_DIFF_PREVIEW_LINES]) + f"\n... ({extra} more lines)"
    return "\n".join(lines)


def _title_content_coherence(wiki_text: str, label: str) -> bool:
    """Deterministic pre-check: title and body must reference the concept.

    The LLM faithfulness score evaluates whether the prose reflects
    the source chunks but does not penalize structural noise in the
    title (bb-8b7s: ``| | designer`` passed at 0.90 because the body
    was coherent). This pre-check asserts three invariants:

    1. The first ``# `` heading must be a sanity-valid label per
       :func:`is_valid_label`. A heading like ``| | designer`` fails
       the structural-char gate even though it contains the cleaned
       display name as a substring.
    2. The cleaned display name must appear in the heading as a
       case-insensitive substring. Covers LLM drift where the
       heading names a different concept than requested.
    3. The body must mention the display name at least once outside
       the heading. Covers the "LLM talked about something adjacent
       but never named the concept" regression.

    Returns True when all three hold, False otherwise.
    """
    display = clean_label_for_display(label).lower()
    if not display:
        return False
    heading: str | None = None
    body_parts: list[str] = []
    for line in wiki_text.splitlines():
        if heading is None and line.startswith("# "):
            heading = line[2:].strip()
            continue
        body_parts.append(line)
    if heading is None:
        return False
    if not is_valid_label(heading):
        return False
    if display not in heading.lower():
        return False
    body = "\n".join(body_parts).lower()
    return display in body


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute the element-wise mean of a non-empty vector list.

    Empty input returns an empty list; callers must check before any
    downstream dot-product so we do not leak a shape mismatch.

    Routes through numpy so the inner loop runs in C: for the typical
    ``D=768``, ``N=10`` case this cuts per-call cost from ~8k Python
    ops to a single SIMD-backed reduction.
    """
    if not vectors:
        return []
    result: list[float] = np.asarray(vectors, dtype=np.float32).mean(axis=0).tolist()
    return result


def _embedding_faithfulness_score(
    body_vec: list[float],
    source_vectors: list[list[float]],
) -> float:
    """Cosine-similarity score between the body and the mean source vector.

    Assumes L2-normalized vectors (both the embedder and the store
    return normalized vectors); cosine reduces to a dot product.
    Falls through to :func:`cosine_sim` so a non-normalized vector
    does not silently produce an out-of-range value. Result is
    clamped at zero because a negative cosine means the body vector
    points the other way from the mean of the sources — treat that
    the same as uncorrelated for threshold purposes.

    Returns 0.0 on a dimension mismatch between the body vector and
    the source-vector mean. That is not expected in production (the
    embedder and the chunk vectors come from the same model), but a
    stub-driven test may hand in off-shape vectors and crashing the
    whole pipeline on the shape-check hides the real assertion.
    """
    from lilbee.store import cosine_sim

    mean_vec = _mean_vector(source_vectors)
    if not mean_vec or not body_vec:
        return 0.0
    if len(mean_vec) != len(body_vec):
        log.warning(
            "Body vector dim %d does not match source vector dim %d; scoring 0.0",
            len(body_vec),
            len(mean_vec),
        )
        return 0.0
    return max(0.0, cosine_sim(body_vec, mean_vec))


def _check_faithfulness(
    chunks: list[SearchChunk],
    wiki_text: str,
    label: str,
    config: Config | None = None,
) -> float:
    """Score the wiki body's similarity to its source chunks, 0.0 on failure.

    Phase D: replaces the LLM-based faithfulness call with a
    deterministic cosine-similarity score between the page body and
    the mean of its source chunk vectors. The B3 title/body coherence
    pre-check still runs first as a hard gate: a garbage H1 returns
    0.0 regardless of embedding similarity, so structurally broken
    pages route to drafts even when the prose happens to be coherent.

    ``chunks`` carries ``.vector`` populated by LanceDB (see
    ``SearchChunk`` in ``store.py``), so no extra embedder call is
    needed for the source side. The body is embedded once via the
    shared services embedder. Any exception in the embedder (model
    missing, network issue, invalid config) is caught and reported as
    0.0 so a single faulty page drops to drafts instead of aborting
    the whole build.
    """
    if not _title_content_coherence(wiki_text, label):
        log.info(
            "Faithfulness title/body coherence failed for %r; scoring 0.0",
            label,
        )
        return 0.0
    source_vectors = [c.vector for c in chunks if c.vector]
    if not source_vectors:
        log.warning("No source vectors for %s; scoring 0.0", label)
        return 0.0

    # Strip the frontmatter + citation block so we embed only the body
    # prose. render_citation_block may not have run yet when the score
    # is computed (it is appended later), but strip_citation_block is
    # idempotent on missing trailers.
    body_text = strip_citation_block(wiki_text).strip()
    if not body_text:
        log.warning("Empty body for %s; scoring 0.0", label)
        return 0.0

    try:
        body_vectors = get_services().embedder.embed_batch([body_text])
    except Exception as exc:
        log.warning("Body embedding failed for %s: %s", label, exc)
        return 0.0
    if not body_vectors:
        return 0.0
    return _embedding_faithfulness_score(body_vectors[0], source_vectors)
