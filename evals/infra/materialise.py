#!/usr/bin/env python3
"""Write one shard's passages into documents/ as one .txt per passage.

The partition rule: this host writes the passages whose GLOBAL index over the
non-empty passages satisfies ``i % shard_count == shard_index``. ir_datasets
iterates in a fixed order, so the split is deterministic and the union of every
shard is the passage set a single host would have written.

Paths are part of that equality. Both the bucket directory and the filename come
from global facts (the global index and the doc id), never from a shard-local
counter, so a passage lands at the same path however many shards ran. lilbee
stores that path as the source of every chunk, so a shard-local bucket would
give the merged index source paths a single host would never produce.

Run as a script it reads the corpus named by DATASET_ID through ir_datasets;
imported, ``materialise`` takes any iterable of objects with ``doc_id`` and
``text``.
"""

from __future__ import annotations

import os
import pathlib
import time
from collections.abc import Callable, Iterable
from typing import Protocol

# Passages per bucket directory. Keeps discovery's stat scan cheap at 8.8M files.
BUCKET_SIZE = 1000


class Passage(Protocol):
    doc_id: str
    text: str


def materialise(
    passages: Iterable[Passage],
    docs_dir: pathlib.Path,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    smoke: int = 0,
    bucket_size: int = BUCKET_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Write this shard's slice. Returns (written, global passages scanned)."""
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} outside 0..{shard_count - 1}")

    scanned = 0
    written = 0
    bucket: pathlib.Path | None = None
    for passage in passages:
        text = (getattr(passage, "text", "") or "").strip()
        if not text:
            continue
        # smoke caps the GLOBAL corpus first; this shard then takes its slice of
        # that same set, so a smoke run shards the same passages a full run does.
        if smoke and scanned >= smoke:
            break
        index = scanned
        scanned += 1
        if shard_count > 1 and index % shard_count != shard_index:
            continue

        # Bucket by the global index. It only grows, so buckets are visited in
        # order and one mkdir per bucket is enough.
        want = docs_dir / f"{index // bucket_size:05d}"
        if want != bucket:
            want.mkdir(parents=True, exist_ok=True)
            bucket = want
        # The stem is the doc_id the qrels join on; nothing else recovers it.
        (bucket / f"{passage.doc_id}.txt").write_text(text)
        written += 1
        if progress and written % 100000 == 0:
            progress(written, scanned)
    return written, scanned


def main() -> int:
    import ir_datasets

    docs_dir = pathlib.Path(os.environ["DOCS_DIR"])
    shard_index = int(os.environ.get("SHARD_INDEX", "0"))
    shard_count = int(os.environ.get("SHARD_COUNT", "1"))
    smoke = int(os.environ.get("SMOKE_N", "0"))

    dataset = ir_datasets.load(os.environ["DATASET_ID"])
    started = time.time()

    def report(written: int, scanned: int) -> None:
        rate = written / max(time.time() - started, 1e-9)
        print(
            f"  shard {shard_index}/{shard_count}: {written:,} written "
            f"({scanned:,} scanned) @ {rate:,.0f}/s",
            flush=True,
        )

    written, scanned = materialise(
        dataset.docs_iter(),
        docs_dir,
        shard_index=shard_index,
        shard_count=shard_count,
        smoke=smoke,
        progress=report,
    )
    print(
        f"  shard {shard_index}/{shard_count}: wrote {written:,} of {scanned:,} scanned "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
