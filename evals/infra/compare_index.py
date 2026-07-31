#!/usr/bin/env python3
"""Compare a merged index against a single-host index built from the same corpus.

Byte identity is impossible and not the target: embeddings carry multi-slot
numeric noise, row order and fragment boundaries differ, and _meta carries a
timestamp. What must hold is that the merged index holds the same corpus.

  same tables, same row counts
  same source key set, same file hashes
  exactly one _meta row, same embedder identity
  same chunk text for every (source, chunk_index)
  vector drift no worse than a single host shows against its own re-run

That last line is why a floor root exists. Without it a drift number is
unfalsifiable: any value can be called small. With it the claim is a comparison.

Usage: compare_index.py <reference_root> <candidate_root> [floor_root]
"""

from __future__ import annotations

import sys
from pathlib import Path

import lancedb
import numpy as np

KEY_COLS = ("source", "chunk_index")
TEXT_COL = "chunk"
# Drift is noise, so the candidate is allowed to sit slightly above the floor
# rather than exactly at it; a real corruption is orders of magnitude out, not 20%.
FLOOR_ALLOWANCE = 1.2


def connect(root: str):
    return lancedb.connect(str(Path(root) / "data" / "lancedb"))


def table_names(db) -> set[str]:
    result = db.list_tables()
    return set(getattr(result, "tables", result))


def counts(db) -> dict[str, int]:
    return {name: db.open_table(name).count_rows() for name in sorted(table_names(db))}


def chunk_map(db) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], np.ndarray]]:
    """{key: text} and {key: vector} for every chunk row."""
    text: dict[tuple[str, int], str] = {}
    vecs: dict[tuple[str, int], np.ndarray] = {}
    # LanceDB's own batch reader, not to_lance(): pylance is a separate package
    # that the ingest venv does not carry.
    query = db.open_table("chunks").search().select([*KEY_COLS, TEXT_COL, "vector"])
    for batch in query.to_batches(batch_size=4096):
        if not batch.num_rows:
            continue
        rows = batch.to_pydict()
        for src, idx, body, vec in zip(
            rows["source"], rows["chunk_index"], rows[TEXT_COL], rows["vector"], strict=True
        ):
            key = (src, int(idx))
            text[key] = body
            vecs[key] = np.asarray(vec, dtype=np.float32)
    return text, vecs


def cosine_drift(a: dict, b: dict, keys) -> tuple[float, float]:
    """(mean, max) cosine distance over *keys* present in both."""
    worst = 0.0
    total = 0.0
    n = 0
    for key in keys:
        u, v = a[key], b[key]
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        d = 1.0 - (float(np.dot(u, v)) / denom if denom else 0.0)
        total += d
        worst = max(worst, d)
        n += 1
    return (total / n if n else 0.0, worst)


def sources(db) -> dict[str, str]:
    rows = db.open_table("_sources").search().limit(None).to_list()
    return {r["filename"]: r["file_hash"] for r in rows}


def identity(db) -> tuple[str, int, int]:
    rows = db.open_table("_meta").search().limit(None).to_list()
    if len(rows) != 1:
        raise AssertionError(f"_meta holds {len(rows)} rows, must hold exactly 1")
    row = rows[0]
    return (row["embedding_model"], int(row["embedding_dim"]), int(row["schema_version"]))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ref_root, cand_root = sys.argv[1], sys.argv[2]
    floor_root = sys.argv[3] if len(sys.argv) > 3 else None

    ref, cand = connect(ref_root), connect(cand_root)
    for label, root, db in (("reference", ref_root, ref), ("candidate", cand_root, cand)):
        if not table_names(db):
            print(f"COMPARE FAIL {label} index at {root} holds no tables", flush=True)
            return 1
    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{f': {detail}' if detail else ''}", flush=True)
        if not ok:
            failures.append(label)

    # Concept tables are refused by the merge, so their absence is expected, not
    # a difference worth failing on.
    ref_tables = table_names(ref) - {"concept_nodes", "concept_edges", "chunk_concepts"}
    cand_tables = table_names(cand) - {"concept_nodes", "concept_edges", "chunk_concepts"}
    check(ref_tables == cand_tables, "same tables", f"{sorted(ref_tables ^ cand_tables)}")

    ref_counts, cand_counts = counts(ref), counts(cand)
    for name in sorted(ref_tables & cand_tables):
        check(
            ref_counts[name] == cand_counts[name],
            f"row count {name}",
            f"reference {ref_counts[name]:,} candidate {cand_counts[name]:,}",
        )

    ref_src, cand_src = sources(ref), sources(cand)
    check(set(ref_src) == set(cand_src), "same source keys",
          f"{len(set(ref_src) ^ set(cand_src)):,} differ")
    check(ref_src == cand_src, "same file hashes")

    try:
        check(identity(ref) == identity(cand), "same embedder identity")
    except AssertionError as exc:
        check(False, "single _meta row", str(exc))

    ref_text, ref_vec = chunk_map(ref)
    cand_text, cand_vec = chunk_map(cand)
    shared = sorted(set(ref_text) & set(cand_text))
    check(len(shared) == len(ref_text) == len(cand_text), "same chunk keys",
          f"reference {len(ref_text):,} candidate {len(cand_text):,} shared {len(shared):,}")
    mismatched = sum(1 for k in shared if ref_text[k] != cand_text[k])
    check(mismatched == 0, "same chunk text", f"{mismatched:,} of {len(shared):,} differ")

    mean_d, max_d = cosine_drift(ref_vec, cand_vec, shared)
    print(f"  vector drift candidate: mean {mean_d:.3e} max {max_d:.3e}", flush=True)
    if floor_root:
        floor_text, floor_vec = chunk_map(connect(floor_root))
        floor_keys = sorted(set(ref_vec) & set(floor_vec))
        fmean, fmax = cosine_drift(ref_vec, floor_vec, floor_keys)
        print(f"  vector drift floor    : mean {fmean:.3e} max {fmax:.3e} "
              f"over {len(floor_keys):,} keys", flush=True)
        check(mean_d <= max(fmean * FLOOR_ALLOWANCE, 1e-9), "drift within the single-host floor",
              f"candidate {mean_d:.3e} floor {fmean:.3e}")
    else:
        print("  no floor root given, so drift is reported and not judged", flush=True)

    print(f"COMPARE {'PASS' if not failures else 'FAIL'} checks_failed={len(failures)} "
          f"{','.join(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
