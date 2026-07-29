#!/usr/bin/env python3
"""Write the manifest that merge_shards.py verifies before it will merge a shard.

The manifest is what lets the merge tell a complete shard set from a partial or
mismatched one: which slice this host holds, what embedded it, and the row counts
it finished with. Identity is read back from the index's own _meta row rather
than from the ingest's EMBED_MODEL, so a manifest cannot claim an embedder the
shard was not built with.

This lives beside the merge rather than in an ingest.sh heredoc so the writer and
merge_shards' read_manifest are exercised against each other by tests; a format
drift between them would otherwise only show up as a refused merge after the GPU
hours are already spent.
"""

from __future__ import annotations

import json
import os
import pathlib

import lancedb

from lilbee.core.config import META_TABLE


def _table_names(db) -> list[str]:
    result = db.list_tables()
    try:
        return list(result.tables)
    except AttributeError:
        return list(result)


def build_manifest(
    root: str | pathlib.Path,
    *,
    shard_index: int,
    shard_count: int,
    dataset_id: str = "",
    smoke_n: int = 0,
) -> dict:
    """Describe the shard rooted at `root` from the index it actually built."""
    root = pathlib.Path(root)
    db = lancedb.connect(str(root / "data" / "lancedb"))
    names = _table_names(db)
    # Absent and empty are the same story for the caller, and lancedb's
    # open_table error for the absent case names neither the shard nor the cause.
    rows = db.open_table(META_TABLE).search().limit(None).to_list() if META_TABLE in names else []
    if not rows:
        raise RuntimeError(
            f"{root}: {META_TABLE} holds no row, so the shard cannot state what embedded it. "
            f"The ingest did not complete."
        )
    identity = max(rows, key=lambda r: r["updated_at"])
    return {
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "dataset_id": dataset_id,
        "smoke_n": int(smoke_n),
        "embedding_model": identity["embedding_model"],
        "embedding_dim": int(identity["embedding_dim"]),
        "schema_version": int(identity["schema_version"]),
        "table_rows": {name: db.open_table(name).count_rows() for name in _table_names(db)},
    }


def write_manifest(
    root: str | pathlib.Path,
    *,
    shard_index: int,
    shard_count: int,
    dataset_id: str = "",
    smoke_n: int = 0,
) -> dict:
    """Build the manifest and persist it where the merge looks for it."""
    manifest = build_manifest(
        root,
        shard_index=shard_index,
        shard_count=shard_count,
        dataset_id=dataset_id,
        smoke_n=smoke_n,
    )
    pathlib.Path(root, "shard_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    manifest = write_manifest(
        os.environ["LILBEE_DATA"],
        shard_index=int(os.environ.get("SHARD_INDEX", "0")),
        shard_count=int(os.environ.get("SHARD_COUNT", "1")),
        dataset_id=os.environ.get("DATASET_ID", ""),
        smoke_n=int(os.environ.get("SMOKE_N", "0")),
    )
    print(
        f"manifest: shard {manifest['shard_index']} of {manifest['shard_count']}, "
        f"{manifest['embedding_model']} {manifest['embedding_dim']}d, "
        f"rows={manifest['table_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
