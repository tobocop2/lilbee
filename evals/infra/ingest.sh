#!/usr/bin/env bash
# Ingest MS MARCO passages into one lilbee index, adopting the proven
# lilbeekreuzbergstein pattern that has ingested millions of files:
#
#   1. config.toml in the data root sets the embedder (embedding_model +
#      embedding_dim). lilbee loads it from canonical_data_root/config.toml
#      (core/config/model.py). This is the mechanism the fleet actually reads --
#      NOT `use-embedder` and NOT env vars, both of which left the embed server
#      on the uninstalled default and silently embedded nothing.
#   2. Passages are materialised straight into documents/ and `lilbee sync`
#      indexes them IN PLACE. No copy. `add` = copy + sync; the copy duplicated
#      35G of already-on-disk data single-threaded before any embedding.
#   3. sync is hash-incremental, so a killed run resumes instead of restarting,
#      and a retry-to-zero loop clears transient per-file failures.
#
# SMOKE_N>0 materialises only the first N passages: prove the whole path and get
# a real docs/sec before committing the full 8.8M. SMOKE_N=0 is the full corpus.
set -uo pipefail

: "${LILBEE_DATA:=/root/msmarco/data}"
: "${DATASET_ID:=msmarco-passage}"
: "${EMBED_MODEL:?EMBED_MODEL must name the embedder}"
: "${SMOKE_N:=0}"
: "${LILBEE_BIN:=/root/lilbee_venv/bin/lilbee}"
: "${PYBIN:=/root/lilbee_venv/bin/python}"
: "${LOG_DIR:=/root/bench/logs}"

export DOCS_DIR="$LILBEE_DATA/documents"
mkdir -p "$DOCS_DIR" "$LOG_DIR"

# Fresh per-run log with a stable 'latest' symlink. Never tee -a into a shared
# file: a prior run's errors accumulate and every failure check false-fires on
# the stale lines (this exact bug tripped the monitor repeatedly).
RUN_LOG="$LOG_DIR/ingest.$(date -u +%Y%m%dT%H%M%SZ).log"
: > "$RUN_LOG"; ln -sf "$(basename "$RUN_LOG")" "$LOG_DIR/ingest.log"
exec > >(tee "$RUN_LOG") 2>&1

log() { printf '[ingest %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- config.toml
# The embedder mechanism. Written into the data root that lilbee loads.
#
# Fleet/saturation knobs (from the embed-fleet-during-ingest fix):
#   - ingest_max_inflight: files in their compute phase at once. 0 = AUTO, which
#     sizes to replicas x 8 (the per-replica in-flight that saturated one card in
#     the smoke). Left auto so it scales with whatever GPU count the ladder lands
#     (64 on 8 cards, 32 on 4) -- a hardcode would starve or overrun. Override via
#     INGEST_MAX_INFLIGHT only if validation shows GPUs under ~90% util.
#   - embed_batch_sequences: passages packed per embed request; 64 keeps the
#     engine's continuous-batching slots full (the smoke's saturating value).
#   The fleet is also held resident (llama-swap ttl 0) for the whole sync by
#   lilbee's keep_fleet_warm(), so idle replicas no longer unload and collapse.
: "${INGEST_MAX_INFLIGHT:=0}"
: "${EMBED_BATCH_SEQUENCES:=64}"
{
  echo "embedding_model = \"$EMBED_MODEL\""
  echo "embedding_dim = 4096"
  echo "enable_ocr = false"
  echo "embed_batch_sequences = $EMBED_BATCH_SEQUENCES"
  [ "${INGEST_MAX_INFLIGHT:-0}" -gt 0 ] 2>/dev/null && echo "ingest_max_inflight = $INGEST_MAX_INFLIGHT"
} > "$LILBEE_DATA/config.toml"
log "config.toml -> embedder=$EMBED_MODEL dim=4096 ocr=off batch_seq=$EMBED_BATCH_SEQUENCES inflight=${INGEST_MAX_INFLIGHT:-auto}"

# ---------------------------------------------------------------- materialise
# Stream passages straight from ir_datasets into documents/ as one .txt per
# passage, sharded 1000/dir so discovery's stat scan stays cheap on 8.8M files.
# No corpus.jsonl round-trip and no evals CLI: ir_datasets owns the download and
# cache, and sync reads documents/ in place (no copy). ir_datasets' own cache
# makes the docs_iter re-entrant, and the marker skips a completed materialise.
MARKER="$LILBEE_DATA/.materialised.${SMOKE_N}"
if [ -f "$MARKER" ]; then
  log "passages already materialised (SMOKE_N=$SMOKE_N), skipping"
else
  log "materialising $DATASET_ID into $DOCS_DIR (SMOKE_N=$SMOKE_N; 0=full corpus)"
  DATASET_ID="$DATASET_ID" "$PYBIN" - <<'PY'
import os, pathlib, time
import ir_datasets
docs = pathlib.Path(os.environ["DOCS_DIR"]); smoke = int(os.environ["SMOKE_N"])
ds = ir_datasets.load(os.environ["DATASET_ID"])
started = time.time(); n = 0
for d in ds.docs_iter():
    if smoke and n >= smoke:
        break
    text = (getattr(d, "text", "") or "").strip()
    if not text:
        continue
    shard = docs / f"{n // 1000:05d}"
    if n % 1000 == 0:
        shard.mkdir(parents=True, exist_ok=True)
    # The stem is the doc_id the qrels join on; nothing else recovers it.
    (shard / f"{d.doc_id}.txt").write_text(text)
    n += 1
    if n % 100000 == 0:
        print(f"  {n:,} passages @ {n / (time.time() - started):,.0f}/s", flush=True)
print(f"  wrote {n:,} passages in {time.time() - started:.0f}s", flush=True)
PY
  touch "$MARKER"
fi
DOCS=$(find "$DOCS_DIR" -type f -name '*.txt' | wc -l)
log "documents on disk: $DOCS"

# ---------------------------------------------------------------- sync
log "syncing in place (trace at ${LILBEE_INGEST_TRACE_FILE:-<unset>})"
T0=$(date +%s)
"$LILBEE_BIN" sync 2>&1 | tee "$LOG_DIR/sync.pass0.log" || log "pass 0 returned nonzero; retry loop follows"
SECS=$(( $(date +%s) - T0 ))

# ---------------------------------------------------------------- retry-to-zero
PASSLOG="$LOG_DIR/sync.pass0.log"
for round in 1 2 3 4 5; do
  N=$(grep -ac "Failed to ingest" "$PASSLOG" 2>/dev/null || true); N=${N:-0}
  log "failures after pass: $N"
  [ "$N" = "0" ] && break
  PASSLOG="$LOG_DIR/sync.retry${round}.log"
  log "retry round $round"
  "$LILBEE_BIN" sync --retry-skipped 2>&1 | tee "$PASSLOG" || true
done

# ---------------------------------------------------------------- reconcile
# Count landed rows straight from LanceDB, the way the proven harness verifies.
log "reconciliation"
LILBEE_DATA="$LILBEE_DATA" "$PYBIN" - <<'PY' | tee "$LOG_DIR/counts.txt"
import os, lancedb
db = lancedb.connect(os.path.join(os.environ["LILBEE_DATA"], "data/lancedb"))
def rows(t):
    try:
        return db.open_table(t).count_rows()
    except Exception as exc:
        return f"?({exc})"
print(f"SOURCES: {rows('_sources')}")
print(f"PAGES: {rows('_page_texts')}")
PY

SRC=$(grep -oE "SOURCES: [0-9]+" "$LOG_DIR/counts.txt" | awk '{print $2}')
PAGES=$(grep -oE "PAGES: [0-9]+" "$LOG_DIR/counts.txt" | awk '{print $2}')
DPS=$(python3 -c "print(f'{${SRC:-0} / max($SECS,1):.1f}')" 2>/dev/null || echo "?")
log "RESULT: input_docs=$DOCS landed_sources=${SRC:-0} pages=${PAGES:-0} | ${DPS} docs/sec first-pass (${SECS}s)"

# ---------------------------------------------------------------- finalize
# The lilbeekreuzbergstein pattern: snapshot the whole index to the network
# volume as ONE sequential tar (MooseFS handles large sequential I/O fine), so
# the 144GB index survives teardown for later grading. Then export _page_texts
# to parquet + jsonl.gz + SHA256SUMS -- the small text artifacts that get version
# controlled via Git LFS. Both land on the volume so nothing is lost at power-off.
WORKSPACE="${WORKSPACE:-/workspace}"
EXPORTS=/root/exports
mkdir -p "$EXPORTS"
if [ -d "$WORKSPACE" ]; then
  log "snapshotting index -> $WORKSPACE/msmarco-index.tar (this is large; sequential)"
  tar -C "$LILBEE_DATA" -cf "$WORKSPACE/msmarco-index.tar" data \
    && log "index tar: $(du -h "$WORKSPACE/msmarco-index.tar" | cut -f1)" \
    || log "WARN: index tar to volume failed"
else
  log "WARN: no $WORKSPACE volume mounted; index NOT snapshotted (would be lost at teardown)"
fi

log "exporting _page_texts -> parquet + jsonl.gz + SHA256SUMS (for Git LFS)"
EXPORTS="$EXPORTS" LILBEE_DATA="$LILBEE_DATA" DATASET_ID="$DATASET_ID" "$PYBIN" - <<'PY' || log "WARN: export failed"
import os, gzip, json, hashlib, pathlib, lancedb
import pyarrow.parquet as pq
out = pathlib.Path(os.environ["EXPORTS"]); out.mkdir(parents=True, exist_ok=True)
db = lancedb.connect(os.path.join(os.environ["LILBEE_DATA"], "data/lancedb"))
t = db.open_table("_page_texts").to_arrow()
t = t.select([c for c in ("source", "page", "text") if c in t.column_names])
stem = os.environ["DATASET_ID"].replace("/", "_")
written = []
pq.write_table(t, out / f"{stem}.parquet"); written.append(out / f"{stem}.parquet")
with gzip.open(out / f"{stem}.jsonl.gz", "wt") as fh:
    for row in t.to_pylist():
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
written.append(out / f"{stem}.jsonl.gz")
(out / "SHA256SUMS").write_text(
    "\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in written) + "\n"
)
print(f"exported {t.num_rows:,} pages -> {[p.name for p in written]}")
PY
# Copy the small text artifacts onto the volume too (retrievable even after teardown).
if [ -d "$WORKSPACE" ]; then cp -f "$EXPORTS"/* "$WORKSPACE/" 2>/dev/null && log "exports copied to volume"; fi
log "export sizes: $(du -sh "$EXPORTS" 2>/dev/null | cut -f1) at $EXPORTS"
log "DONE"
