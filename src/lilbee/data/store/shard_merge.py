"""Fold the per-worker shard stores of a multi-GPU ingest into one index."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lilbee.core.config import CHUNKS_TABLE, INGEST_SOURCE_COLUMNS, META_TABLE, SOURCES_TABLE
from lilbee.data.store.lance_helpers import escape_sql_string, table_names

if TYPE_CHECKING:
    from pathlib import Path

    import lancedb

    from lilbee.data.store.core import Store

log = logging.getLogger(__name__)

# Rows read from a shard per append. A chunks row carries its vector, so a whole
# shard table does not fit in memory.
_MERGE_BATCH_ROWS = 10_000

# Source names per ``IN`` predicate when only part of a shard is merged. LanceDB
# parses the predicate as one SQL string, so the names are chunked rather than
# joined into a single clause per sync.
_NAMES_PER_PREDICATE = 500


def merge_shards(
    store: Store, shard_dirs: list[Path], *, sources: set[str] | None = None
) -> dict[str, int]:
    """Append every shard's rows into *store*, returning the rows merged per table.

    Shards own disjoint sources, so their union is an append with no dedup. A
    whole-shard copy (*sources* None) is the fresh-index case. Naming the touched
    sources is the re-sync case: the store's own rows for those keys are dropped
    first, so re-merging replaces them instead of doubling them.
    """
    import lancedb

    if sources is not None:
        store.remove_documents(sorted(sources))
    merged: dict[str, int] = {}
    adopted = _adopt_chunks(store, shard_dirs, sources)
    if adopted is not None:
        merged[CHUNKS_TABLE] = adopted
    for shard_dir in shard_dirs:
        database = lancedb.connect(str(shard_dir))
        for name in table_names(database):
            # The merged store writes its own meta row from the running config;
            # a shard's copy would land beside it as a second row.
            if name == META_TABLE:
                continue
            if name == CHUNKS_TABLE and adopted is not None:
                continue  # already taken over whole, without reading a row
            rows = _copy_table(database.open_table(name), store, name, sources)
            merged[name] = merged.get(name, 0) + rows
    log.info("Merged %d shard(s): %s", len(shard_dirs), merged)
    _reconcile_sources(store, shard_dirs)
    return merged


def _adopt_chunks(store: Store, shard_dirs: list[Path], sources: set[str] | None) -> int | None:
    """Take over every shard's chunk fragments whole; None when that cannot apply.

    The chunks table carries the vectors, so it is the whole cost of the merge:
    at 8.8M rows by 4096 dims the row copy rewrites about 144GB, and since the
    shard stores stay as resume state the corpus then sits on disk twice.
    Adopting the fragments is metadata only, and the hard links mean one physical
    copy with two names.

    Whole-fragment, so only a full merge qualifies: a scoped re-sync names its
    sources, and a fragment there holds touched and untouched rows together.
    Returns None when the caller should copy rows instead, which also covers a
    shard on another filesystem or a data file whose name is already taken; the
    merge is correct either way, only slower.
    """
    if sources is not None:
        return None
    tables = [shard_dir / f"{CHUNKS_TABLE}.lance" for shard_dir in shard_dirs]
    present = [table for table in tables if table.exists()]
    if not present:
        return None
    try:
        return store.adopt_fragments(CHUNKS_TABLE, present)
    except OSError as exc:
        log.warning("Adopting shard fragments failed (%s); copying rows instead", exc)
        return None


def _reconcile_sources(store: Store, shard_dirs: list[Path]) -> None:
    """Say so when the merged index tracks fewer sources than the workers hold.

    A scoped merge only takes what the run touched, so a source a worker holds and
    the index does not (an earlier merge that failed, a removal against the index
    alone) would otherwise stay missing with nothing to show for it.
    """
    import lancedb

    held = sum(_source_count(lancedb.connect(str(shard_dir))) for shard_dir in shard_dirs)
    merged = _source_count(store.get_db())
    if merged < held:
        log.warning(
            "The index tracks %d source(s) against %d across the ingest workers. "
            "Re-run with --force to fold every worker's shard back in.",
            merged,
            held,
        )


def _source_count(database: lancedb.DBConnection) -> int:
    """Rows in a store's source table, zero when it has none."""
    if SOURCES_TABLE not in table_names(database):
        return 0
    return int(database.open_table(SOURCES_TABLE).count_rows())


def _copy_table(
    table: lancedb.table.Table, store: Store, name: str, sources: set[str] | None
) -> int:
    """Append the rows of *table* that this merge wants into *store*."""
    return sum(
        _copy_rows(table, store, name, predicate) for predicate in _predicates(name, sources)
    )


def _predicates(name: str, sources: set[str] | None) -> list[str | None]:
    """The where-clauses selecting the rows to merge from table *name*.

    ``None`` is the whole table. A table with no source column holds corpus-level
    aggregates that the post-merge passes rebuild, so a scoped merge skips it.
    """
    if sources is None:
        return [None]
    column = INGEST_SOURCE_COLUMNS.get(name)
    if column is None:
        return []
    names = sorted(sources)
    return [
        _in_predicate(column, names[start : start + _NAMES_PER_PREDICATE])
        for start in range(0, len(names), _NAMES_PER_PREDICATE)
    ]


def _in_predicate(column: str, names: list[str]) -> str:
    """``column IN (...)`` over *names*."""
    quoted = ", ".join(f"'{escape_sql_string(name)}'" for name in names)
    return f"{column} IN ({quoted})"


def _copy_rows(table: lancedb.table.Table, store: Store, name: str, predicate: str | None) -> int:
    """Stream the rows *predicate* selects from *table* into *store*."""
    import pyarrow as pa

    query = table.search()
    if predicate is not None:
        query = query.where(predicate)
    reader = query.limit(0).to_batches(_MERGE_BATCH_ROWS)
    copied = 0
    for batch in reader:
        copied += store.absorb_rows(name, pa.Table.from_batches([batch], schema=reader.schema))
    return copied
