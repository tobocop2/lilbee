#!/usr/bin/env python3
"""Merge N shard indexes into one lilbee index, identical to a single-host run.

Each shard is a complete lilbee LanceDB (its own chunks/_sources/_page_texts/...
tables) built from a disjoint slice of the corpus. Because sources are keyed by
content hash + doc-id filename and chunking is per-document, the union of the
shard tables IS the table a single host would have produced -- only row order and
the embedder's inherent multi-slot noise differ (see bb-afdo4). This tool:

  1. Concatenates every table across the shards into a fresh merged LanceDB
     (disjoint doc-ids, so plain concat -- no dedup).
  2. Rebuilds the ANN (IVF_PQ/cosine) and FTS (BM25 on 'chunk') indexes on the
     merged chunks table by driving lilbee's OWN Store.ensure_*_index methods, so
     the indexes -- and the full-corpus BM25 statistics -- are constructed exactly
     as a single-host sync would build them, not reimplemented here.
  3. Reconciles: merged row counts == sum of shard row counts, per table.

Usage:
  LILBEE_DATA=<merged_root> python merge_shards.py <shard_root_1> <shard_root_2> ...
where each <shard_root_i> contains data/lancedb (and config.toml). The merged
index lands at $LILBEE_DATA/data/lancedb; config.toml is copied from shard 1.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import lancedb
import pyarrow as pa


def _lancedb_dir(root: str) -> str:
    return os.path.join(root, "data", "lancedb")


def main() -> int:
    shards = [s.rstrip("/") for s in sys.argv[1:]]
    if len(shards) < 2:
        print("usage: LILBEE_DATA=<merged_root> merge_shards.py <shard_root_1> <shard_root_2> ...")
        return 2
    merged_root = os.environ["LILBEE_DATA"]
    merged_db_dir = _lancedb_dir(merged_root)
    Path(merged_db_dir).mkdir(parents=True, exist_ok=True)

    # config.toml (embedder identity) from shard 1 -- all shards ran the same config.
    src_cfg = os.path.join(shards[0], "config.toml")
    if os.path.exists(src_cfg):
        shutil.copyfile(src_cfg, os.path.join(merged_root, "config.toml"))

    src_dbs = [lancedb.connect(_lancedb_dir(s)) for s in shards]
    merged_db = lancedb.connect(merged_db_dir)

    # Table set = the union of table names present in any shard (shard 0 is the
    # reference; a table absent from a shard just contributes no rows).
    table_names: list[str] = []
    seen = set()
    for db in src_dbs:
        for t in db.table_names():
            if t not in seen:
                seen.add(t)
                table_names.append(t)

    totals: dict[str, int] = {}
    for name in table_names:
        parts: list[pa.Table] = []
        shard_rows = 0
        for db in src_dbs:
            if name not in db.table_names():
                continue
            tbl = db.open_table(name).to_arrow()
            shard_rows += tbl.num_rows
            if tbl.num_rows:
                parts.append(tbl)
        if not parts:
            continue
        combined = parts[0] if len(parts) == 1 else pa.concat_tables(parts, promote_options="default")
        # Fresh table in the merged db (overwrite any prior partial merge).
        if name in merged_db.table_names():
            merged_db.drop_table(name)
        merged_db.create_table(name, combined)
        totals[name] = combined.num_rows
        assert combined.num_rows == shard_rows, f"{name}: concat {combined.num_rows} != sum {shard_rows}"
        print(f"  merged {name}: {combined.num_rows:,} rows", flush=True)

    # Rebuild ANN + FTS with lilbee's own builders so construction is identical to
    # a single-host sync (IVF_PQ/cosine on 'vector', BM25 FTS on 'chunk').
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
