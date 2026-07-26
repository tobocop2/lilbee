"""Tests for neighbor-window expansion (pure logic, mocked store)."""

from unittest.mock import MagicMock

from lilbee.data.store import SearchChunk
from lilbee.retrieval.query.neighbors import expand_neighbors, merge_adjacent_texts


def _cost(text: str) -> int:
    """Mirror Searcher._budget_tokens (3 chars per token, floor 1)."""
    return max(1, len(text) // 3)


def _chunk(
    source="doc.pdf",
    index=0,
    text="text",
    content_type="pdf",
    page=1,
    line_start=0,
    line_end=0,
    score=0.9,
) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type=content_type,
        page_start=page,
        page_end=page,
        line_start=line_start,
        line_end=line_end,
        chunk=text,
        chunk_index=index,
        vector=[0.1],
        score=score,
    )


def _store_with(rows: list[SearchChunk]) -> MagicMock:
    """A store mock serving *rows* filtered by the requested source and indices."""
    store = MagicMock()
    store.get_chunks_by_indices.side_effect = lambda source, indices: [
        r for r in rows if r.source == source and r.chunk_index in set(indices)
    ]
    return store


class TestMergeAdjacentTexts:
    def test_dedups_the_shared_overlap(self):
        merged = merge_adjacent_texts(
            ["alpha beta shared overlap region", "shared overlap region delta"]
        )
        assert merged == "alpha beta shared overlap region delta"

    def test_short_coincidental_match_is_not_deleted(self):
        # A seam-boundary word match is not chunker overlap; deduping it would
        # silently delete the right side's opening text.
        merged = merge_adjacent_texts(["report ends with the ", "the next chunk"])
        assert "the next chunk" in merged

    def test_breadcrumb_on_the_right_does_not_hide_the_overlap(self):
        # kreuzberg heading-context chunks carry "# A > ## B\n\n" prefixes; the
        # overlap sits after the breadcrumb and the merge must still find it.
        overlap = "word219 word220 word221 word222"
        left = f"# Guide > ## Install\n\nword0 word1 {overlap}"
        right = f"# Guide > ## Install\n\n{overlap} word223 word224"
        merged = merge_adjacent_texts([left, right])
        assert merged == f"# Guide > ## Install\n\nword0 word1 {overlap} word223 word224"

    def test_no_overlap_joins_with_a_newline_seam(self):
        assert merge_adjacent_texts(["alpha", "delta"]) == "alpha\ndelta"

    def test_fully_contained_text_adds_nothing(self):
        assert merge_adjacent_texts(["alpha beta", "beta"]) == "alpha beta"

    def test_an_empty_side_has_no_overlap(self):
        # An empty chunk shares nothing, so the seam is a plain join rather than
        # a scan over a zero-length string.
        assert merge_adjacent_texts(["", "delta"]) == "\ndelta"
        assert merge_adjacent_texts(["alpha", ""]) == "alpha"

    def test_control_bytes_in_the_text_do_not_confuse_the_overlap(self):
        # Extracted document text can carry stray control bytes. A scan that
        # joined the two sides around a NUL sentinel would match past the
        # shared region and swallow right's tail.
        overlap = "\0a" * 8
        assert merge_adjacent_texts([f"X{overlap}", f"{overlap}Z"]) == f"X{overlap}Z"

    def test_merging_an_already_merged_passage_is_idempotent(self):
        a = "alpha overlap region one"
        b = "beta overlap region two"
        texts = [f"start {a}", f"{a} middle {b}", f"{b} end"]
        merged = merge_adjacent_texts(texts)
        assert merged == f"start {a} middle {b} end"
        # Re-merging with either neighbor changes nothing: the neighbor's text
        # is fully contained, so a second expansion pass cannot duplicate it.
        assert merge_adjacent_texts([f"start {a}", merged]) == merged
        assert merge_adjacent_texts([merged, f"{b} end"]) == merged


class TestExpandNeighbors:
    def test_widens_both_sides_and_recomputes_the_page_span(self):
        center = _chunk(index=2, text="gamma overlap padding delta overlap padding", page=3)
        rows = [
            _chunk(index=1, text="beta gamma overlap padding", page=2),
            _chunk(index=3, text="delta overlap padding epsilon", page=4),
        ]
        store = _store_with(rows)
        out = expand_neighbors([center], store, radius=1, budget=1000, cost=_cost)
        assert len(out) == 1
        assert out[0].chunk == "beta gamma overlap padding delta overlap padding epsilon"
        assert (out[0].page_start, out[0].page_end) == (2, 4)
        assert out[0].score == center.score
        assert out[0].chunk_index == center.chunk_index
        store.get_chunks_by_indices.assert_called_once_with("doc.pdf", [1, 2, 3])

    def test_recomputes_the_line_span_for_code(self):
        center = _chunk(content_type="code", index=1, text="mid", line_start=10, line_end=20)
        rows = [
            _chunk(content_type="code", index=0, text="top", line_start=1, line_end=9),
            _chunk(content_type="code", index=2, text="tail", line_start=21, line_end=30),
        ]
        out = expand_neighbors([center], _store_with(rows), radius=1, budget=1000, cost=_cost)
        assert (out[0].line_start, out[0].line_end) == (1, 30)

    def test_single_chunk_document_stays_untouched(self):
        center = _chunk(index=0, text="whole doc")
        store = _store_with([])
        out = expand_neighbors([center], store, radius=2, budget=1000, cost=_cost)
        assert out == [center]

    def test_selected_neighbors_are_never_pulled_twice(self):
        # Chunks 2 and 3 are both selected: each widens only away from the
        # other, so no passage text is duplicated.
        a = _chunk(index=2, text="two overlap padding three overlap padding")
        b = _chunk(index=3, text="three overlap padding four overlap padding")
        rows = [
            _chunk(index=1, text="one two overlap padding"),
            _chunk(index=4, text="four overlap padding five"),
        ]
        store = _store_with(rows)
        out = expand_neighbors([a, b], store, radius=1, budget=1000, cost=_cost)
        assert out[0].chunk == "one two overlap padding three overlap padding"
        assert out[1].chunk == "three overlap padding four overlap padding five"
        store.get_chunks_by_indices.assert_called_once_with("doc.pdf", [1, 2, 3, 4])

    def test_higher_ranked_expansion_claims_the_shared_neighbor(self):
        first = _chunk(index=2, text="bb")
        second = _chunk(index=4, text="dd")
        rows = [
            _chunk(index=1, text="aa"),
            _chunk(index=3, text="cc"),
            _chunk(index=5, text="ee"),
        ]
        out = expand_neighbors([first, second], _store_with(rows), 1, 1000, _cost)
        # Rank order wins: chunk 2 claims index 3, so chunk 4 widens right only.
        assert out[0].chunk == "aa\nbb\ncc"
        assert out[1].chunk == "dd\nee"

    def test_budget_sheds_farthest_neighbors_first_trailing_on_ties(self):
        center = _chunk(index=5, text="e" * 30)
        rows = [_chunk(index=i, text=c * 30) for i, c in ((3, "c"), (4, "d"), (6, "f"), (7, "g"))]
        # Full widening costs 41 extra tokens; 30 fits only after shedding the
        # trailing index 7 (the tie), then the leading index 3 (the farthest).
        out = expand_neighbors([center], _store_with(rows), radius=2, budget=30, cost=_cost)
        assert out[0].chunk == "\n".join(["d" * 30, "e" * 30, "f" * 30])

    def test_zero_budget_keeps_every_original(self):
        center = _chunk(index=2, text="core")
        rows = [_chunk(index=1, text="left"), _chunk(index=3, text="right")]
        store = _store_with(rows)
        out = expand_neighbors([center], store, radius=1, budget=0, cost=_cost)
        assert out == [center]
        # A zero budget can buy nothing, so it must not pay for the store reads.
        store.get_chunks_by_indices.assert_not_called()

    def test_reingested_document_is_not_spliced_across_versions(self):
        """If the file is re-ingested between the search and the neighbor fetch,
        the rows describe a different chunking. Splicing them would invent text
        and a page span that existed in no version, so widening is skipped."""
        center = _chunk(index=2, text="original center text")
        rows = [
            _chunk(index=1, text="new left"),
            # Same (source, chunk_index), different text: the file was re-chunked.
            _chunk(index=2, text="re-ingested center text"),
            _chunk(index=3, text="new right"),
        ]
        out = expand_neighbors([center], _store_with(rows), radius=1, budget=1000, cost=_cost)
        assert out == [center]

    def test_unchanged_center_still_widens(self):
        """The version guard must not block the normal path: a center whose
        stored row still matches widens as before."""
        center = _chunk(index=2, text="beta gamma")
        rows = [
            _chunk(index=1, text="alpha beta"),
            _chunk(index=2, text="beta gamma"),
            _chunk(index=3, text="gamma delta"),
        ]
        out = expand_neighbors([center], _store_with(rows), radius=1, budget=1000, cost=_cost)
        assert out[0].chunk != "beta gamma"

    def test_whole_document_selection_expands_nothing(self):
        # Every index is a selected passage already, so only the off-the-end
        # probe runs and nothing changes.
        selected = [_chunk(index=i, text=f"part {i}") for i in range(3)]
        store = _store_with([])
        out = expand_neighbors(selected, store, radius=1, budget=1000, cost=_cost)
        assert out == selected
        store.get_chunks_by_indices.assert_called_once_with("doc.pdf", [0, 1, 2, 3])

    def test_window_stops_at_a_gap_in_stored_indices(self):
        # Index 3 is missing from the store: index 2 must not leapfrog it,
        # or the merged passage would splice non-contiguous text.
        center = _chunk(index=4, text="dd")
        rows = [_chunk(index=2, text="bb"), _chunk(index=5, text="ee")]
        out = expand_neighbors([center], _store_with(rows), radius=2, budget=1000, cost=_cost)
        assert out[0].chunk == "dd\nee"


class TestExcludedNeighbors:
    def test_excluded_neighbor_text_is_never_imported(self):
        # The structural filter dropped a TOC page; expansion must treat it as
        # absent (ending the run) instead of splicing it back in.
        center = _chunk(index=2, text="body overlap padding text")
        rows = [
            _chunk(index=1, text="Contents ....... 1\nScope ....... 4\nIndex ....... 9"),
            _chunk(index=3, text="text following overlap padding"),
        ]
        out = expand_neighbors(
            [center],
            _store_with(rows),
            radius=1,
            budget=1000,
            cost=_cost,
            exclude=lambda t: "......." in t,
        )
        assert "Contents" not in out[0].chunk
        assert "following" in out[0].chunk

    def test_exclusion_never_applies_to_the_center(self):
        center = _chunk(index=2, text="looks excluded but is the original hit")
        rows = [_chunk(index=2, text="looks excluded but is the original hit")]
        out = expand_neighbors(
            [center],
            _store_with(rows),
            radius=1,
            budget=1000,
            cost=_cost,
            exclude=lambda t: "excluded" in t,
        )
        assert out == [center]
