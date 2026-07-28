"""Neighbor-window expansion: widen retrieved passages with adjacent chunks.

A hit that lands in the middle of an argument loses the sentences before and
after it. After context selection, each selected chunk pulls up to N adjacent
chunks per side from its own source and merges them into one contiguous
passage, deduplicating the overlap text adjacent chunks share from chunking.
The widened passage keeps the original chunk's score and citation identity;
only its text and page/line span change.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from lilbee.data.store import SearchChunk, Store

# Below this a suffix-prefix match is coincidence, not chunker overlap; deduping
# it would delete real text at an overlap-free seam.
_MIN_OVERLAP = 16

# kreuzberg markdown breadcrumb ("# Guide > ## Install\n\n"); heading-context
# content lines never start with "#", so only the breadcrumb matches.
_BREADCRUMB_RE = re.compile(r"\A#{1,6} [^\n]*\n\n")


def _overlap_chars(left: str, right: str) -> int:
    """Length of the longest suffix of *left* that is a prefix of *right*.

    Adjacent chunks share the chunker's overlap verbatim, so the longest match
    is the shared region. A fully contained text matches whole, which keeps a
    re-merge of an already-widened passage idempotent. Longest length first,
    so the first match wins.
    """
    for k in range(min(len(left), len(right)), 0, -1):
        if left.endswith(right[:k]):
            return k
    return 0


def _seam(left: str, right: str) -> tuple[int, str]:
    """(overlap length, effective right text) for one adjacent seam.

    A heading breadcrumb on the right hides the chunker overlap, so the
    stripped form is tried too and wins only when it reveals one. A match
    shorter than ``_MIN_OVERLAP`` counts only when one side is fully
    contained in the other.
    """
    k = _overlap_chars(left, right)
    if k >= _MIN_OVERLAP or k == len(right) or (k and k == len(left)):
        return k, right
    stripped = _BREADCRUMB_RE.sub("", right)
    if stripped != right:
        ks = _overlap_chars(left, stripped)
        if ks >= _MIN_OVERLAP or (ks and ks in (len(stripped), len(left))):
            return ks, stripped
    return 0, right


def _merge_span(
    span: list[int],
    texts: dict[int, str],
    seams: dict[tuple[int, int], tuple[int, str]],
) -> str:
    """Merge the span's texts in index order, deduplicating seam overlaps.

    *seams* caches per-pair seam scans so a caller re-merging shrinking spans
    (the budget shed loop) pays for each scan once.
    """
    merged = texts[span[0]]
    prev = span[0]
    for index in span[1:]:
        key = (prev, index)
        if key not in seams:
            seams[key] = _seam(texts[prev], texts[index])
        k, effective = seams[key]
        tail = effective[k:]
        if not tail:
            continue
        merged = merged + tail if k else f"{merged}\n{tail}"
        prev = index
    return merged


def merge_adjacent_texts(texts: list[str]) -> str:
    """Concatenate adjacent chunk texts, deduplicating their shared overlap.

    Adjacent texts with no real overlap (a chunk_overlap=0 build, or only a
    coincidental short match) join with a newline seam rather than gluing two
    words together or deleting text.
    """
    return _merge_span(list(range(len(texts))), dict(enumerate(texts)), {})


def expand_neighbors(
    results: list[SearchChunk],
    store: Store,
    radius: int,
    budget: int,
    cost: Callable[[str], int],
    *,
    exclude: Callable[[str], bool] | None = None,
) -> list[SearchChunk]:
    """Widen each result with up to *radius* adjacent same-source chunks.

    Results are processed in rank order and only spend *budget*, the tokens
    left over after the originals were fitted: a widened passage whose extra
    cost does not fit sheds its farthest neighbors first and falls back to
    the original text, so expansion is always trimmed before any original
    chunk. An index that is itself selected, or already claimed by a
    higher-ranked expansion, is never pulled again, so no passage text is
    duplicated (a document routed whole expands to nothing). A neighbor whose
    text matches *exclude* is treated as absent, ending the run at its side,
    so expansion cannot re-import text an upstream filter dropped.
    """
    if budget <= 0:
        # Nothing can be spent, so skip the per-source store fetches entirely:
        # on a tight window every widen attempt would fail against a zero
        # budget after paying for the reads.
        return results
    centers: dict[str, set[int]] = {}
    for r in results:
        centers.setdefault(r.source, set()).add(r.chunk_index)
    rows = _fetch_neighbor_rows(store, centers, radius)
    if exclude is not None:
        rows = {
            key: row
            for key, row in rows.items()
            if row.chunk_index in centers.get(row.source, ()) or not exclude(row.chunk)
        }
    if not rows:
        return results
    claimed = {source: set(indices) for source, indices in centers.items()}
    remaining = budget
    expanded: list[SearchChunk] = []
    for r in results:
        widened, spent = _widen(r, rows, claimed[r.source], radius, remaining, cost)
        remaining -= spent
        expanded.append(widened)
    return expanded


def _fetch_neighbor_rows(
    store: Store, centers: dict[str, set[int]], radius: int
) -> dict[tuple[str, int], SearchChunk]:
    """Every candidate neighbor row, fetched with one store call per source.

    The centers are fetched alongside their neighbors so the caller can tell
    whether the document was re-ingested since the search snapshot; indices past
    the end of a document are simply absent from the reply.
    """
    rows: dict[tuple[str, int], SearchChunk] = {}
    for source, owned in centers.items():
        wanted = sorted(
            {
                index
                for center in owned
                for index in range(center - radius, center + radius + 1)
                if index >= 0
            }
        )
        for row in store.get_chunks_by_indices(source, wanted):
            rows[(source, row.chunk_index)] = row
    return rows


def _neighbor_run(
    result: SearchChunk,
    rows: dict[tuple[str, int], SearchChunk],
    claimed: set[int],
    step: int,
    radius: int,
) -> list[int]:
    """Contiguous free neighbor indices on one side of the center, nearest first."""
    indices: list[int] = []
    for offset in range(1, radius + 1):
        index = result.chunk_index + step * offset
        if index in claimed or (result.source, index) not in rows:
            break
        indices.append(index)
    return indices


def _widen(
    result: SearchChunk,
    rows: dict[tuple[str, int], SearchChunk],
    claimed: set[int],
    radius: int,
    remaining: int,
    cost: Callable[[str], int],
) -> tuple[SearchChunk, int]:
    """One result widened within *remaining* tokens: (chunk, tokens spent).

    Neighbors extend from the center until a missing, selected, or already
    claimed index stops each side. While the widened text over-spends, the
    farthest neighbor is shed first (a tie sheds the trailing side, keeping
    the text that leads up to the hit); shedding everything keeps the
    original chunk untouched.
    """
    center = result.chunk_index
    current = rows.get((result.source, center))
    if current is not None and current.chunk != result.chunk:
        # Re-ingested since the search: the neighbor rows are a different
        # chunking, so splicing them would invent text and a page span.
        return result, 0
    left = _neighbor_run(result, rows, claimed, -1, radius)
    right = _neighbor_run(result, rows, claimed, +1, radius)
    texts = {center: result.chunk}
    for index in [*left, *right]:
        texts[index] = rows[(result.source, index)].chunk
    seams: dict[tuple[int, int], tuple[int, str]] = {}
    while left or right:
        span = sorted([*left, center, *right])
        merged = _merge_span(span, texts, seams)
        extra = cost(merged) - cost(result.chunk)
        if extra <= remaining:
            claimed.update(index for index in span if index != center)
            neighbors = [rows[(result.source, index)] for index in span if index != center]
            return _widened_copy(result, merged, neighbors), extra
        if right and (not left or right[-1] - center >= center - left[-1]):
            right.pop()
        else:
            left.pop()
    return result, 0


def _widened_copy(result: SearchChunk, merged: str, neighbors: list[SearchChunk]) -> SearchChunk:
    """The result with widened text and a truthfully recomputed page/line span."""
    spans = [result, *neighbors]
    return result.model_copy(
        update={
            "chunk": merged,
            "page_start": min(s.page_start for s in spans),
            "page_end": max(s.page_end for s in spans),
            "line_start": min(s.line_start for s in spans),
            "line_end": max(s.line_end for s in spans),
        }
    )
