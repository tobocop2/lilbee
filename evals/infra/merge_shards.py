#!/usr/bin/env python3
"""Merge N shard indexes into one lilbee index, identical to a single-host run.

Each shard is a complete lilbee LanceDB (its own chunks/_sources/_page_texts/...
tables) built from a disjoint slice of the corpus. Sources are keyed by content
hash and doc-id filename and chunking is per-document, so the union of the shard
tables is the table a single host would have produced. Only row order and the
embedder's inherent multi-slot numeric noise differ; a single multi-GPU host
carries the same noise against its own re-run.

Tables fall into three classes:

  concat     chunks/_sources/_page_texts/_citations/_memories/entities hold
             disjoint rows, so the merge is a plain concatenation with no dedup.
  singleton  _meta and _entity_schema hold one row by contract. Concatenating
             them yields N rows and a store whose identity depends on which row
             a reader picks, so the merge verifies the shards agree and keeps a
             single row.
  corpus     concept_nodes/concept_edges/chunk_concepts carry cluster ids that
             are only meaningful within the shard that assigned them. Gluing
             them produces a graph whose clusters were never computed over the
             whole corpus, so the merge refuses rather than shipping it.

Before writing anything the merge verifies, from each shard's manifest:
completeness (every shard of the declared set present exactly once), embedder
identity (same model, dim and schema version everywhere, agreeing with the
shard's own _meta row), and that each shard still holds the row counts it
recorded when its ingest finished. Any mismatch refuses the merge instead of
producing a wrong or partial index.

Rows stream shard-to-merged one batch at a time through LanceDB's own
RecordBatchReader path, so peak memory is one batch rather than the whole corpus
of vectors.

Usage:
  LILBEE_DATA=<merged_root> python merge_shards.py <shard_root_1> <shard_root_2> ...
where each <shard_root_i> contains data/lancedb, config.toml and
shard_manifest.json. The merged index lands at $LILBEE_DATA/data/lancedb.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import lancedb
import pyarrow as pa

# Table names come from lilbee rather than string literals here. A rename on
# lilbee's side would otherwise reclassify a table silently: a renamed concept
# table would fall through to the concat branch and glue shard-local clusters,
# which is the failure this tool exists to prevent. Importing the names is safe
# before LILBEE_DATA is set; only Config construction reads the environment.
from lilbee.core.config import (
    CHUNK_CONCEPTS_TABLE,
    CONCEPT_EDGES_TABLE,
    CONCEPT_NODES_TABLE,
    ENTITY_SCHEMA_TABLE,
    META_TABLE,
)

MANIFEST_NAME = "shard_manifest.json"

# One row by contract; see the store's get_meta/entity_schema_state readers.
SINGLETON_TABLES = (META_TABLE, ENTITY_SCHEMA_TABLE)

# Cluster ids are assigned per shard, so these cannot be concatenated.
CORPUS_WIDE_TABLES = (CONCEPT_NODES_TABLE, CONCEPT_EDGES_TABLE, CHUNK_CONCEPTS_TABLE)

# Rows per streamed batch. Bounds peak memory at batch_rows x row width.
DEFAULT_BATCH_ROWS = 4096

# Partitioning is the point; a one-shard "merge" is a copy and almost certainly
# a mistyped invocation.
MIN_SHARDS = 2


class MergeRefusedError(Exception):
    """A guardrail rejected the shard set; no merged index was written."""


@dataclass(frozen=True)
class ShardManifest:
    """What a shard records about itself when its ingest finishes."""

    root: str
    shard_index: int
    shard_count: int
    embedding_model: str
    embedding_dim: int
    schema_version: int
    table_rows: dict[str, int]


def _lancedb_dir(root: str) -> str:
    return os.path.join(root, "data", "lancedb")


def _table_names(db) -> list[str]:
    """Table names, unwrapping the ListTablesResponse newer LanceDB returns.

    Duplicates the store's private helper of the same name rather than
    importing across the package boundary for three lines.
    """
    result = db.list_tables()
    try:
        return list(result.tables)
    except AttributeError:
        return list(result)


def read_manifest(root: str) -> ShardManifest:
    """Load and shape-check one shard's manifest."""
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.exists(path):
        raise MergeRefusedError(
            f"{root}: no {MANIFEST_NAME}. A shard that does not state which slice of the "
            f"corpus it holds cannot be checked for completeness. Re-run its ingest with a "
            f"current ingest.sh."
        )
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise MergeRefusedError(f"{path}: unreadable manifest ({exc})") from exc

    required = (
        "shard_index",
        "shard_count",
        "embedding_model",
        "embedding_dim",
        "schema_version",
        "table_rows",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise MergeRefusedError(f"{path}: manifest missing {', '.join(missing)}")
    return ShardManifest(
        root=root,
        shard_index=int(raw["shard_index"]),
        shard_count=int(raw["shard_count"]),
        embedding_model=str(raw["embedding_model"]),
        embedding_dim=int(raw["embedding_dim"]),
        schema_version=int(raw["schema_version"]),
        table_rows={str(k): int(v) for k, v in raw["table_rows"].items()},
    )


def observed_counts(db) -> dict[str, int]:
    """Actual row count of every table in a shard."""
    return {name: db.open_table(name).count_rows() for name in _table_names(db)}


def meta_identity(db) -> tuple[str, int, int] | None:
    """(model, dim, schema_version) from a shard's own _meta row, newest wins."""
    if META_TABLE not in _table_names(db):
        return None
    rows = db.open_table(META_TABLE).search().limit(None).to_list()
    if not rows:
        return None
    row = max(rows, key=lambda r: r["updated_at"])
    return (
        str(row["embedding_model"]),
        int(row["embedding_dim"]),
        int(row["schema_version"]),
    )


def _verify_completeness(manifests: list[ShardManifest]) -> None:
    """Every shard of the declared set present exactly once."""
    declared = {m.shard_count for m in manifests}
    if len(declared) != 1:
        pairs = sorted((m.shard_index, m.shard_count) for m in manifests)
        raise MergeRefusedError(f"shards disagree on shard_count: {pairs}")
    count = declared.pop()
    if len(manifests) != count:
        missing = sorted(set(range(count)) - {m.shard_index for m in manifests})
        raise MergeRefusedError(
            f"incomplete shard set: {len(manifests)} roots supplied but the shards declare "
            f"shard_count={count}. Missing indices: {missing}"
        )
    seen: dict[int, str] = {}
    for m in manifests:
        if not 0 <= m.shard_index < count:
            raise MergeRefusedError(f"{m.root}: shard_index {m.shard_index} outside 0..{count - 1}")
        if m.shard_index in seen:
            raise MergeRefusedError(
                f"shard_index {m.shard_index} supplied twice: {seen[m.shard_index]} and {m.root}"
            )
        seen[m.shard_index] = m.root


def _verify_identity(
    manifests: list[ShardManifest], meta: dict[str, tuple[str, int, int] | None]
) -> None:
    """One embedder, one dim, one schema version, confirmed by each shard's _meta."""
    identities = {(m.embedding_model, m.embedding_dim, m.schema_version) for m in manifests}
    if len(identities) != 1:
        detail = ", ".join(
            f"shard {m.shard_index}={m.embedding_model}/{m.embedding_dim}d/v{m.schema_version}"
            for m in sorted(manifests, key=lambda m: m.shard_index)
        )
        raise MergeRefusedError(f"shards were embedded differently and cannot merge: {detail}")
    expected = identities.pop()
    for m in manifests:
        actual = meta.get(m.root)
        if actual is None:
            raise MergeRefusedError(
                f"{m.root}: no _meta row, so shard identity cannot be confirmed"
            )
        if actual != expected:
            raise MergeRefusedError(
                f"{m.root}: manifest says {expected[0]}/{expected[1]}d/v{expected[2]} but its "
                f"_meta row says {actual[0]}/{actual[1]}d/v{actual[2]}"
            )


def _verify_counts(manifests: list[ShardManifest], observed: dict[str, dict[str, int]]) -> None:
    """Each shard still holds what it recorded when its ingest finished."""
    for m in manifests:
        rows = observed.get(m.root, {})
        for name, expected in sorted(m.table_rows.items()):
            got = rows.get(name, 0)
            if got != expected:
                raise MergeRefusedError(
                    f"{m.root}: table {name} holds {got:,} rows but its manifest recorded "
                    f"{expected:,}. The shard is truncated or was written after its ingest."
                )


def _verify_no_glued_clusters(
    manifests: list[ShardManifest], observed: dict[str, dict[str, int]]
) -> None:
    """Concept-graph rows cannot be concatenated across shards."""
    glued = sorted(
        {
            name
            for m in manifests
            for name in CORPUS_WIDE_TABLES
            if observed.get(m.root, {}).get(name, 0) > 0
        }
    )
    if glued:
        raise MergeRefusedError(
            f"shards carry concept-graph rows ({', '.join(glued)}) whose cluster ids are "
            f"assigned per shard. Concatenating them yields clusters that were never computed "
            f"over the whole corpus, and corpus-wide re-clustering is not implemented. Disable "
            f"the concept wiki for partitioned runs, or re-cluster on one host after merging."
        )


def verify_shards(
    manifests: list[ShardManifest],
    observed: dict[str, dict[str, int]],
    meta: dict[str, tuple[str, int, int] | None],
) -> None:
    """Refuse any shard set that would merge into a wrong or partial index."""
    if len(manifests) < MIN_SHARDS:
        raise MergeRefusedError(f"merge needs at least {MIN_SHARDS} shards")
    _verify_completeness(manifests)
    _verify_identity(manifests, meta)
    _verify_counts(manifests, observed)
    _verify_no_glued_clusters(manifests, observed)


def _iter_batches(dbs: list, name: str, batch_rows: int):
    """Yield every row of `name` across shards, one batch at a time."""
    for db in dbs:
        if name not in _table_names(db):
            continue
        for batch in db.open_table(name).search().to_batches(batch_size=batch_rows):
            if batch.num_rows:
                yield batch


def _reference_schema(dbs: list, name: str) -> pa.Schema:
    """The schema `name` shares across shards, refusing if they disagree."""
    present = [(db, db.open_table(name).schema) for db in dbs if name in _table_names(db)]
    first_db, first = present[0]
    for db, schema in present[1:]:
        if not schema.equals(first):
            only_first = sorted(set(first.names) - set(schema.names)) or "none"
            only_other = sorted(set(schema.names) - set(first.names)) or "none"
            raise MergeRefusedError(
                f"table {name} has different schemas across shards ({first_db.uri} vs "
                f"{db.uri}); columns only in the first: {only_first}, only in the other: "
                f"{only_other}. The shards were built by different lilbee versions."
            )
    return first


def _copy_singleton(dbs: list, name: str, merged_db) -> int:
    """Keep one row of a single-row-by-contract table, newest wins."""
    newest: pa.Table | None = None
    newest_at = ""
    for db in dbs:
        if name not in _table_names(db):
            continue
        table = db.open_table(name).to_arrow()
        for i in range(table.num_rows):
            updated_at = str(table.column("updated_at")[i])
            if newest is None or updated_at > newest_at:
                newest, newest_at = table.slice(i, 1), updated_at
    if newest is None:
        return 0
    merged_db.create_table(name, newest)
    return 1


def merge(
    shard_roots: list[str], merged_root: str, batch_rows: int = DEFAULT_BATCH_ROWS
) -> dict[str, int]:
    """Verify the shard set, then stream it into one merged index."""
    manifests = [read_manifest(root) for root in shard_roots]
    dbs = [lancedb.connect(_lancedb_dir(root)) for root in shard_roots]
    observed = {root: observed_counts(db) for root, db in zip(shard_roots, dbs, strict=True)}
    meta = {root: meta_identity(db) for root, db in zip(shard_roots, dbs, strict=True)}
    verify_shards(manifests, observed, meta)

    merged_db_dir = _lancedb_dir(merged_root)
    Path(merged_db_dir).mkdir(parents=True, exist_ok=True)
    merged_db = lancedb.connect(merged_db_dir)

    # Shard 0's config is the reference; the identity check proved they agree.
    src_cfg = os.path.join(shard_roots[0], "config.toml")
    if os.path.exists(src_cfg):
        shutil.copyfile(src_cfg, os.path.join(merged_root, "config.toml"))

    table_names: list[str] = []
    for db in dbs:
        for name in _table_names(db):
            if name not in table_names:
                table_names.append(name)

    totals: dict[str, int] = {}
    for name in table_names:
        # Overwrite any prior partial merge.
        if name in _table_names(merged_db):
            merged_db.drop_table(name)

        if name in SINGLETON_TABLES:
            totals[name] = _copy_singleton(dbs, name, merged_db)
            print(f"  merged {name}: {totals[name]} row (single by contract)", flush=True)
            continue

        expected = sum(observed[root].get(name, 0) for root in shard_roots)
        schema = _reference_schema(dbs, name)
        if expected == 0:
            merged_db.create_table(name, schema=schema)
            totals[name] = 0
            print(f"  merged {name}: 0 rows", flush=True)
            continue

        reader = pa.RecordBatchReader.from_batches(schema, _iter_batches(dbs, name, batch_rows))
        merged_db.create_table(name, reader)
        landed = merged_db.open_table(name).count_rows()
        if landed != expected:
            raise MergeRefusedError(f"{name}: merged {landed:,} rows but shards hold {expected:,}")
        totals[name] = landed
        print(f"  merged {name}: {landed:,} rows", flush=True)

    return totals


def main() -> int:
    shards = [s.rstrip("/") for s in sys.argv[1:]]
    if len(shards) < MIN_SHARDS:
        print("usage: LILBEE_DATA=<merged_root> merge_shards.py <shard_root_1> <shard_root_2> ...")
        return 2
    merged_root = os.environ["LILBEE_DATA"]
    batch_rows = int(os.environ.get("MERGE_BATCH_ROWS", DEFAULT_BATCH_ROWS))

    try:
        totals = merge(shards, merged_root, batch_rows)
    except MergeRefusedError as exc:
        print(f"MERGE REFUSED: {exc}", file=sys.stderr, flush=True)
        return 1

    # Rebuild ANN and FTS with lilbee's own builders so construction, and the
    # full-corpus BM25 statistics, match a single-host sync exactly.
    os.environ["LILBEE_DATA"] = merged_root
    from lilbee.core.config import active_config
    from lilbee.data.store import Store

    store = Store(active_config())
    print("  rebuilding ANN vector index (IVF_PQ/cosine)...", flush=True)
    store.ensure_vector_index(force=True)
    print("  rebuilding FTS/BM25 index on 'chunk'...", flush=True)
    store.ensure_fts_index()
    store.close()

    print("=== MERGE RECONCILE ===", flush=True)
    for name, n in totals.items():
        print(f"  {name}: {n:,}")
    print("MERGE COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
