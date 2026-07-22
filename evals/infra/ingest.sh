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
cat > "$LILBEE_DATA/config.toml" <<CFG
embedding_model = "$EMBED_MODEL"
embedding_dim = 4096
enable_ocr = false
CFG
log "config.toml -> embedding_model=$EMBED_MODEL embedding_dim=4096 ocr=off"

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
log "DONE"
