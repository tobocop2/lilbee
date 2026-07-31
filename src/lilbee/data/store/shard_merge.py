"""Fold the per-worker shard stores of a multi-GPU ingest into one index."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lilbee.core.config.defaults import INGEST_SOURCE_COLUMNS, META_TABLE
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
    for shard_dir in shard_dirs:
        database = lancedb.connect(str(shard_dir))
        for name in table_names(database):
            # The merged store writes its own meta row from the running config;
            # a shard's copy would land beside it as a second row.
            if name == META_TABLE:
                continue
            rows = _copy_table(database.open_table(name), store, name, sources)
            merged[name] = merged.get(name, 0) + rows
    log.info("Merged %d shard(s): %s", len(shard_dirs), merged)
    return merged


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
