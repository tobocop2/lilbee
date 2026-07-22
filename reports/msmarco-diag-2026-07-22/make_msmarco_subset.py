#!/usr/bin/env python3
"""Materialise a subset of the REAL MS MARCO passage corpus into a lilbee
documents dir -- so the fleet/ingest fix (bb-opgxa / bb-emkt0) can be tested
against real passages instead of a synthetic corpus.

This is the exact code the full 8,841,823-passage run used (see
evals/infra/ingest.sh), just parameterised by --n. ir_datasets owns the
download + cache, so no corpus files need to be shipped: the first run pulls
`msmarco-passage` (~3 GB) and caches it; subsequent runs are instant.

    pip install ir_datasets
    python make_msmarco_subset.py --n 50000 --out /root/kb/data/documents

The files land as one .txt per passage (stem = MS MARCO doc_id, which the qrels
join on), sharded 1000/dir. That directory IS lilbee's documents_dir, so:

    # write /root/kb/data/config.toml with:
    #   embedding_model = "Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf"
    #   embedding_dim = 4096
    #   enable_ocr = false
    LILBEE_DATA=/root/kb/data LILBEE_EMBED_REPLICAS=<#GPUs> lilbee sync

No `lilbee add` (that copies); `sync` indexes documents/ in place.

Reproducing the degradation (bb-opgxa): the embed fleet only idle-unloads once a
replica goes >300s without a request, so a tiny subset that finishes in seconds
will NOT show it. To trigger it deliberately on a small subset, either use enough
passages that the run lasts several minutes with more replicas than the
single-threaded dispatcher keeps busy, or temporarily lower the embed server ttl
so idle-unload fires in seconds. The fix (ttl=0 during ingest, and #590's
parallel dispatch) should keep every replica resident and busy for the whole run.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import ir_datasets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50000, help="passages to write (0 = full 8.8M)")
    ap.add_argument("--out", required=True, help="target documents dir (lilbee's documents_dir)")
    ap.add_argument("--dataset", default="msmarco-passage", help="ir_datasets id")
    args = ap.parse_args()

    docs = pathlib.Path(args.out)
    docs.mkdir(parents=True, exist_ok=True)
    ds = ir_datasets.load(args.dataset)
    started = time.time()
    n = 0
    for doc in ds.docs_iter():
        if args.n and n >= args.n:
            break
        text = (getattr(doc, "text", "") or "").strip()
        if not text:
            continue
        shard = docs / f"{n // 1000:05d}"
        if n % 1000 == 0:
            shard.mkdir(parents=True, exist_ok=True)
        (shard / f"{doc.doc_id}.txt").write_text(text)
        n += 1
        if n % 100000 == 0:
            print(f"  {n:,} @ {n / (time.time() - started):,.0f}/s", flush=True)
    print(f"wrote {n:,} passages to {docs} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
