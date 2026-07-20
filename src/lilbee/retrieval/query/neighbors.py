"""Neighbor-window expansion: widen retrieved passages with adjacent chunks.

A hit that lands in the middle of an argument loses the sentences before and
after it. After context selection, each selected chunk pulls up to N adjacent
chunks per side from its own source and merges them into one contiguous
passage, deduplicating the overlap text adjacent chunks share from chunking.
The widened passage keeps the original chunk's score and citation identity;
only its text and page/line span change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from lilbee.data.store import SearchChunk, Store


def _overlap_chars(left: str, right: str) -> int:
    """Length of the longest suffix of *left* that is a prefix of *right*.

    Adjacent chunks carry the chunker's overlap verbatim (both cuts come from
    the same document text), so the longest match IS the shared region. A
    fully contained text matches whole, which is what makes merging an
    already-widened passage idempotent instead of duplicating its neighbors.

    Computed with a prefix-function pass rather than trying every length: the
    no-overlap case (a ``chunk_overlap=0`` build) is common and would otherwise
    scan every length to exhaustion, quadratic in the chunk's own size.
    """
    if not left or not right:
        return 0
    # Prefix function of right alone. Concatenating the two around a sentinel
    # would be shorter to write but only correct for text that never contains
    # the sentinel, and extracted document text carries stray control bytes.
    failure = [0] * len(right)
    length = 0
    for i in range(1, len(right)):
        while length and right[i] != right[length]:
            length = failure[length - 1]
        if right[i] == right[length]:
            length += 1
        failure[i] = length
    # Only left's last len(right) chars can reach a prefix of right, so the scan
    # stays linear in the incoming chunk, not the accumulated passage. That
    # window is also what keeps ``right[length]`` in range: matching right in
    # full costs len(right) chars, so it can only complete on the last one.
    length = 0
    for char in left[-len(right) :]:
        while length and char != right[length]:
            length = failure[length - 1]
        if char == right[length]:
            length += 1
    return length


def merge_adjacent_texts(texts: list[str]) -> str:
    """Concatenate adjacent chunk texts, deduplicating their shared overlap.

    Adjacent texts with no detectable overlap (a chunk_overlap=0 build) join
    with a newline seam rather than gluing two words together.
    """
    merged = texts[0]
    for text in texts[1:]:
        k = _overlap_chars(merged, text)
        tail = text[k:]
        if not tail:
            continue
        merged = merged + tail if k else f"{merged}\n{tail}"
    return merged


def expand_neighbors(
    results: list[SearchChunk],
    store: Store,
    radius: int,
    budget: int,
    cost: Callable[[str], int],
) -> list[SearchChunk]:
    """Widen each result with up to *radius* adjacent same-source chunks.

    Results are processed in rank order and only spend *budget*, the tokens
    left over after the originals were fitted: a widened passage whose extra
    cost does not fit sheds its farthest neighbors first and falls back to
    the original text, so expansion is always trimmed before any original
    chunk. An index that is itself selected, or already claimed by a
    higher-ranked expansion, is never pulled again, so no passage text is
    duplicated (a document routed whole expands to nothing).
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
        # The document was re-ingested between the search and this fetch (reads
        # take no lock and the store's read-consistency window is seconds), so
        # the neighbor rows belong to a different chunking of the file. Splicing
        # them would invent text and a page span that existed in no version.
        return result, 0
    left = _neighbor_run(result, rows, claimed, -1, radius)
    right = _neighbor_run(result, rows, claimed, +1, radius)
    while left or right:
        span = sorted([*left, center, *right])
        texts = [
            result.chunk if index == center else rows[(result.source, index)].chunk
            for index in span
        ]
        merged = merge_adjacent_texts(texts)
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
