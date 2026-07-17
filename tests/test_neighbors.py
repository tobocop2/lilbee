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
    chunk_type="raw",
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
        chunk_type=chunk_type,
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
        merged = merge_adjacent_texts(["alpha beta gamma", "gamma delta"])
        assert merged == "alpha beta gamma delta"

    def test_no_overlap_joins_with_a_newline_seam(self):
        assert merge_adjacent_texts(["alpha", "delta"]) == "alpha\ndelta"

    def test_fully_contained_text_adds_nothing(self):
        assert merge_adjacent_texts(["alpha beta", "beta"]) == "alpha beta"

    def test_merging_an_already_merged_passage_is_idempotent(self):
        texts = ["one two", "two three", "three four"]
        merged = merge_adjacent_texts(texts)
        assert merged == "one two three four"
        # Re-merging with either neighbor changes nothing: the neighbor's text
        # is fully contained, so a second expansion pass cannot duplicate it.
        assert merge_adjacent_texts(["one two", merged]) == merged
        assert merge_adjacent_texts([merged, "three four"]) == merged


class TestExpandNeighbors:
    def test_widens_both_sides_and_recomputes_the_page_span(self):
        center = _chunk(index=2, text="gamma delta", page=3)
        rows = [
            _chunk(index=1, text="beta gamma", page=2),
            _chunk(index=3, text="delta epsilon", page=4),
        ]
        store = _store_with(rows)
        out = expand_neighbors([center], store, radius=1, budget=1000, cost=_cost)
        assert len(out) == 1
        assert out[0].chunk == "beta gamma delta epsilon"
        assert (out[0].page_start, out[0].page_end) == (2, 4)
        assert out[0].score == center.score
        assert out[0].chunk_index == center.chunk_index
        store.get_chunks_by_indices.assert_called_once_with("doc.pdf", [1, 3])

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
        a = _chunk(index=2, text="two three")
        b = _chunk(index=3, text="three four")
        rows = [
            _chunk(index=1, text="one two"),
            _chunk(index=4, text="four five"),
        ]
        store = _store_with(rows)
        out = expand_neighbors([a, b], store, radius=1, budget=1000, cost=_cost)
        assert out[0].chunk == "one two three"
        assert out[1].chunk == "three four five"
        store.get_chunks_by_indices.assert_called_once_with("doc.pdf", [1, 4])

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
        out = expand_neighbors([center], _store_with(rows), radius=1, budget=0, cost=_cost)
        assert out == [center]

    def test_whole_document_selection_expands_nothing(self):
        # Every index is a selected passage already, so only the off-the-end
        # probe runs and nothing changes.
        selected = [_chunk(index=i, text=f"part {i}") for i in range(3)]
        store = _store_with([])
        out = expand_neighbors(selected, store, radius=1, budget=1000, cost=_cost)
        assert out == selected
        store.get_chunks_by_indices.assert_called_once_with("doc.pdf", [3])

    def test_window_stops_at_a_gap_in_stored_indices(self):
        # Index 3 is missing from the store: index 2 must not leapfrog it,
        # or the merged passage would splice non-contiguous text.
        center = _chunk(index=4, text="dd")
        rows = [_chunk(index=2, text="bb"), _chunk(index=5, text="ee")]
        out = expand_neighbors([center], _store_with(rows), radius=2, budget=1000, cost=_cost)
        assert out[0].chunk == "dd\nee"

    def test_table_chunk_is_never_widened(self):
        # Table chunks take synthetic indices appended after the content
        # chunks, so their stored neighbors are positionally unrelated text.
        center = _chunk(index=5, text="| h |\n| a |", chunk_type="table")
        rows = [
            _chunk(index=4, text="tail prose"),
            _chunk(index=6, text="| h2 |", chunk_type="table"),
        ]
        store = _store_with(rows)
        out = expand_neighbors([center], store, radius=1, budget=1000, cost=_cost)
        assert out == [center]
        store.get_chunks_by_indices.assert_not_called()

    def test_table_result_is_passed_through_beside_a_widened_prose_result(self):
        # A prose result with real neighbors keeps the fetch non-empty, so the
        # main loop runs and must pass the table result through untouched
        # rather than trying to widen it.
        prose = _chunk(source="a.pdf", index=2, text="body")
        table = _chunk(source="a.pdf", index=9, text="| h |\n| a |", chunk_type="table")
        rows = [
            _chunk(source="a.pdf", index=1, text="lead"),
            _chunk(source="a.pdf", index=3, text="tail"),
        ]
        out = expand_neighbors([prose, table], _store_with(rows), radius=1, budget=1000, cost=_cost)
        assert out[0].chunk == "lead\nbody\ntail"
        assert out[1] is table

    def test_table_rows_are_never_pulled_as_neighbors(self):
        # The last content chunk sits right before the appended table indices;
        # widening it must not splice table markdown onto its prose.
        center = _chunk(index=4, text="closing prose")
        rows = [
            _chunk(index=3, text="earlier prose"),
            _chunk(index=5, text="| h |\n| a |", chunk_type="table"),
        ]
        out = expand_neighbors([center], _store_with(rows), radius=1, budget=1000, cost=_cost)
        assert out[0].chunk == "earlier prose\nclosing prose"
