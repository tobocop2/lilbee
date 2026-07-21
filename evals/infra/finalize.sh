#!/usr/bin/env bash
# Everything worth keeping from an ingest, staged into one directory to pull off
# the pod before it auto-downs. This pod is volume-free: its container disk is
# destroyed on --down, so an artifact that is not pulled is gone.
#
# What is preserved and why:
#   pages.parquet / pages.jsonl  the chunked text lilbee actually indexed, for
#                                scrutiny -- a reviewer reads what was searched
#                                without needing the 140GB of vectors.
#   ingest_trace.log             per-document extraction time; the extraction-vs
#                                -GPU split the whole run exists to measure.
#   provenance.jsonl             machines, timings, throughput, cost.
#   preflight.log / ingest.log   what ran and what it said.
#   commits.txt                  the two exact SHAs behind the run.
#   msmarco_index.tar            the full LanceDB index WITH vectors, so the run
#                                is reproducible rather than merely described.
set -euo pipefail
: "${LOCAL:=/root/msmarco}" ; : "${WORKSPACE:=/root/bench}"
: "${LILBEE_BIN:=/root/lilbee_venv/bin/lilbee}"
export LILBEE_DATA="$LOCAL/data"
ART=/root/artifacts
mkdir -p "$ART"
log() { printf '[finalize %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "exporting the indexed text as parquet and jsonl (vectors dropped)"
"$LILBEE_BIN" export "$ART/pages.parquet" --data-dir "$LILBEE_DATA"
"$LILBEE_BIN" export "$ART/pages.jsonl" --format jsonl --data-dir "$LILBEE_DATA"

log "staging logs, trace, provenance"
cp -f "$WORKSPACE/logs/ingest_trace.log" "$ART/" 2>/dev/null || true
cp -f "$WORKSPACE/logs/preflight.log" "$WORKSPACE/logs/ingest.log" "$ART/" 2>/dev/null || true
cp -f "$WORKSPACE/logs/commits.txt" "$ART/" 2>/dev/null || true
cp -f "$WORKSPACE/provenance.jsonl" "$ART/" 2>/dev/null || true
cp -f "$WORKSPACE/datasets/msmarco/qrels.trec" "$WORKSPACE/datasets/msmarco/queries.jsonl" "$ART/" 2>/dev/null || true
# The full index tar (built by ingest phase 5) stays where it is; it is large
# and pulled separately only if wanted.
[ -f "$WORKSPACE/msmarco_index.tar" ] && ln -sf "$WORKSPACE/msmarco_index.tar" "$ART/msmarco_index.tar"

log "artifact manifest"
{ echo "# msmarco ingest artifacts $(date -u +%FT%TZ)"; du -h "$ART"/* 2>/dev/null; } | tee "$ART/MANIFEST.txt"
log "DONE -- pull with: scp -r <cluster>:/root/artifacts ./"
