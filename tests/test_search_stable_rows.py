"""Equal scores give the same rows in the same order on every search.

LanceDB orders equal scores arbitrarily, and its limit cuts them at an arbitrary row.
The store tests write their rows against the order the store must return.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from lancedb.query import LanceQueryBuilder

from lilbee.core.config import CHUNKS_TABLE, cfg
from lilbee.data.store import Store, core
from lilbee.data.store.types import ChunkType

_TIED_ROWS = 30
_LOWEST_FIVE = [0, 1, 2, 3, 4]


def _record(index: int) -> dict[str, Any]:
    return {
        "source": "long_notes.md",
        "content_type": "text",
        "chunk_type": "raw",
        "page_start": 0,
        "page_end": 0,
        "line_start": 0,
        "line_end": 0,
        "chunk": "valve assembly notes",
        "chunk_index": index,
        "vector": [0.5] * cfg.embedding_dim,
    }


@pytest.fixture()
def store(tmp_path) -> Store:
    """A store whose rows all tie, written in the reverse of their chunk order."""
    built = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb"}))
    built.add_chunks([_record(i) for i in reversed(range(_TIED_ROWS))])
    built.ensure_fts_index()
    return built


def _indexes(rows: list[Any]) -> list[int]:
    return [row.chunk_index for row in rows]


@contextmanager
def _counted_reads() -> Iterator[list[tuple[int, bool]]]:
    """Each read on the real query path, as (rows returned, whole rows or not)."""
    reads: list[tuple[int, bool]] = []
    real = LanceQueryBuilder.to_list

    def counted(query: LanceQueryBuilder) -> list[dict[str, Any]]:
        rows = real(query)
        reads.append((len(rows), bool(rows) and "vector" in rows[0]))
        return rows

    with mock.patch.object(LanceQueryBuilder, "to_list", counted):
        yield reads


def _table_of(records: list[dict[str, Any]], tmp_path: Path) -> tuple[Store, Any]:
    built = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "tiers"}))
    built.add_chunks(records)
    built.ensure_fts_index()
    return built, built.open_table(CHUNKS_TABLE)


def _worded(index: int, words: int, source: str = "long_notes.md") -> dict[str, Any]:
    """A row whose chunk holds *words* words; rows of one length tie on BM25."""
    return {**_record(index), "source": source, "chunk": " ".join(["valve", *["pad"] * words])}


def test_tied_keyword_rows_keep_the_lowest_chunks_in_chunk_order(store: Store) -> None:
    assert _indexes(store.bm25_probe("valve", top_k=5)) == _LOWEST_FIVE


def test_tied_vector_rows_keep_the_lowest_chunks_in_chunk_order(store: Store) -> None:
    table = store.open_table(CHUNKS_TABLE)
    rows = store._vector_arm(table, [0.5] * cfg.embedding_dim, 5, None)
    assert _indexes(rows) == _LOWEST_FIVE


def test_a_hybrid_search_over_tied_arms_keeps_the_lowest_chunks(store: Store) -> None:
    rows = store.search([0.5] * cfg.embedding_dim, top_k=5, query_text="valve", max_distance=0)
    assert _indexes(rows) == _LOWEST_FIVE


def test_a_cut_between_two_scores_reads_whole_rows_once(tmp_path: Path) -> None:
    _built, table = _table_of([_worded(i, i * 10) for i in range(4)], tmp_path)
    with _counted_reads() as reads:
        rows = core._lexical_rows(table, "valve", 2, None)
    assert [row["chunk_index"] for row in rows] == [0, 1]
    assert reads == [(3, True)]


def test_a_wide_tie_reads_whole_rows_only_for_the_winners(store: Store) -> None:
    table = store.open_table(CHUNKS_TABLE)
    with _counted_reads() as reads:
        rows = core._lexical_rows(table, "valve", 5, None)
    assert [row["chunk_index"] for row in rows] == _LOWEST_FIVE
    assert reads == [(6, True), (_TIED_ROWS, False), (5, True)]


def test_a_tie_that_ends_past_the_cut_stops_widening_there(tmp_path: Path) -> None:
    _built, table = _table_of(
        [_worded(i, 1) for i in reversed(range(10))] + [_worded(i, 50) for i in range(10, 70)],
        tmp_path,
    )
    with _counted_reads() as reads:
        rows = core._lexical_rows(table, "valve", 5, None)
    assert [row["chunk_index"] for row in rows] == _LOWEST_FIVE
    assert reads == [(6, True), (6 * core._TIE_WIDEN_FACTOR, False), (5, True)]


def test_a_tie_past_the_ceiling_stops_widening_and_keeps_a_stable_order(store: Store) -> None:
    table = store.open_table(CHUNKS_TABLE)
    with mock.patch.object(core, "_TIE_FETCH_CEILING", 8), _counted_reads() as reads:
        rows = core._lexical_rows(table, "valve", 5, None)
    indexes = [row["chunk_index"] for row in rows]
    assert indexes == sorted(indexes)
    assert reads == [(6, True), (8, False), (5, True)]


def test_a_tied_search_prints_no_lance_warning(
    store: Store, capfd: pytest.CaptureFixture[str]
) -> None:
    table = store.open_table(CHUNKS_TABLE)
    core._lexical_rows(table, "valve", 5, None)
    store._vector_arm(table, [0.5] * cfg.embedding_dim, 5, None)
    captured = capfd.readouterr()
    assert "WARN" not in captured.err + captured.out


def test_a_widened_row_is_the_row_an_untied_fetch_gives(store: Store) -> None:
    table = store.open_table(CHUNKS_TABLE)
    whole = table.search().where("chunk_index = 0").to_list()[0]
    best = core._lexical_rows(table, "valve", 5, None)[0]
    score = best.pop(core._SCORE_COLUMN)
    assert best == whole
    assert score > 0


def test_a_winner_deleted_after_the_key_fetch_drops_out(store: Store) -> None:
    table = store.open_table(CHUNKS_TABLE)
    source = table.search().limit(1).to_list()[0][core._SOURCE_ID]
    keys = [
        {core._SOURCE_ID: source, core._CHUNK_INDEX: 0, core._SCORE_COLUMN: 2.0},
        {core._SOURCE_ID: source, core._CHUNK_INDEX: _TIED_ROWS, core._SCORE_COLUMN: 1.0},
    ]
    rows = core._whole_rows(table, keys, core._SCORE_COLUMN)
    assert [(row["chunk_index"], row[core._SCORE_COLUMN]) for row in rows] == [(0, 2.0)]


def test_no_keys_read_back_no_rows_and_make_no_read(store: Store) -> None:
    table = store.open_table(CHUNKS_TABLE)
    with _counted_reads() as reads:
        assert core._whole_rows(table, [], core._SCORE_COLUMN) == []
    assert reads == []


def test_a_tie_across_sources_ranks_by_source_then_chunk(tmp_path: Path) -> None:
    records = [_worded(i, 1, "one.md") for i in range(8)]
    records += [_worded(i, 1, "two's.md") for i in range(8)]
    _built, table = _table_of(records, tmp_path)
    rows = core._lexical_rows(table, "valve", 10, None)
    keys = [(row[core._SOURCE_ID], row["chunk_index"]) for row in rows]
    assert keys == sorted(keys)
    assert len(set(keys)) == 10
    assert len({source for source, _index in keys}) == 2


def test_a_chunk_type_scope_also_holds_on_the_wide_fetch(tmp_path: Path) -> None:
    """Wiki rows sort first by source, so a wide fetch with no filter returns them for RAW."""
    wiki = [{**_worded(i, 1, "a_wiki.md"), "chunk_type": "wiki"} for i in range(10)]
    raw = [_worded(i, 1, "b_raw.md") for i in range(10)]
    _built, table = _table_of(wiki + raw, tmp_path)
    for scope in (ChunkType.RAW, ChunkType.WIKI):
        with _counted_reads() as reads:
            rows = core._lexical_rows(table, "valve", 5, scope)
        assert reads == [(6, True), (10, False), (5, True)]
        assert {row["chunk_type"] for row in rows} == {scope.value}
        assert [row["chunk_index"] for row in rows] == _LOWEST_FIVE


def test_a_wide_vector_tie_reads_whole_rows_only_for_the_winners(store: Store) -> None:
    table = store.open_table(CHUNKS_TABLE)
    with _counted_reads() as reads:
        rows = store._vector_arm(table, [0.5] * cfg.embedding_dim, 5, None)
    assert _indexes(rows) == _LOWEST_FIVE
    assert all(row.distance is not None for row in rows)
    assert reads == [(6, True), (_TIED_ROWS, False), (5, True)]


def test_lower_distances_rank_first_on_the_vector_arm(tmp_path: Path) -> None:
    far = {**_record(0), "vector": [1.0] + [0.0] * (cfg.embedding_dim - 1)}
    near = {**_record(1), "vector": [0.0] + [1.0] * (cfg.embedding_dim - 1)}
    built, table = _table_of([far, near], tmp_path)
    rows = built._vector_arm(table, near["vector"], 2, None)
    assert _indexes(rows) == [1, 0]
