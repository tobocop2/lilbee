#!/usr/bin/env bash
# Ingest MS MARCO's 8,841,823 passages into one lilbee index, with per-document
# tracing, and leave the index on the network volume as a single tarball.
#
# Resumable by design. Secure pods get reclaimed mid-run, so every phase checks
# whether its output already exists and skips rather than redoing it. A run that
# loses its box at hour three should resume, not restart.
set -euo pipefail

: "${WORKSPACE:=/workspace}"          # the network volume: sequential files only
: "${LOCAL:=/root/msmarco}"           # container NVMe: the working set
: "${EMBED_MODEL:?EMBED_MODEL must name the embedder the manifest froze}"
: "${LILBEE_BIN:=/root/lilbee_venv/bin/lilbee}"
: "${PYBIN:=/root/lilbee_venv/bin/python}"

CORPUS_JSONL="$WORKSPACE/datasets/msmarco/corpus.jsonl"
DOCS_DIR="$LOCAL/documents"
LOG_DIR="$WORKSPACE/logs"
PROVENANCE="$WORKSPACE/provenance.jsonl"

# Tracing is the point of the run, not a nicety: without it the extraction-versus
# -GPU split is unrecoverable after the fact, and per-document failures are a
# count instead of a list.
export LILBEE_INGEST_TRACE=1
export LILBEE_INGEST_TRACE_FILE="$LOG_DIR/ingest_trace.log"
export LILBEE_LOG_LEVEL=DEBUG
export LILBEE_DATA="$LOCAL/data"
# Invoke the venv binaries directly. `uv run` in the source tree builds a SECOND
# venv that has lilbee_engine but no llama-server, so serve reports status ok,
# starts zero workers, and every request 503s with "No embed model server is
# running". It looks exactly like a broken lilbee and is not one.
export UV_PROJECT_ENVIRONMENT=/root/lilbee_venv

mkdir -p "$DOCS_DIR" "$LOG_DIR" "$LOCAL/data"
log() { printf '[ingest %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG_DIR/ingest.log"; }

# ---------------------------------------------------------------- phase 1
# One file per passage, because lilbee names a source by its path relative to the
# documents directory, and the retrieval scoring has to join back to
# passage-level qrels. Grouping passages into larger files would make the
# document id the group, and the run file could not be scored at all.
#
# Sharded 1000 files to a directory. A single directory of 8.8M entries makes
# every readdir linear; sharding keeps discovery's stat scan cheap. This is on
# local NVMe, never the volume: 8.8M small files on MooseFS is the exact shape
# that cost a previous run four hours with the GPUs idle.
if [ -f "$LOCAL/.passages_written" ]; then
  log "passages already materialised, skipping"
else
  log "materialising passages from $CORPUS_JSONL"
  "$PYBIN" - <<'PY'
import json, os, pathlib, sys, time
corpus = pathlib.Path(os.environ["CORPUS_JSONL"]); docs = pathlib.Path(os.environ["DOCS_DIR"])
started, written = time.time(), 0
with corpus.open() as handle:
    for line in handle:
        record = json.loads(line)
        # The stem is the doc_id the qrels use; nothing else recovers it.
        shard = docs / f"{written // 1000:05d}"
        if written % 1000 == 0:
            shard.mkdir(parents=True, exist_ok=True)
        (shard / f"{record['doc_id']}.txt").write_text(record["text"])
        written += 1
        if written % 500_000 == 0:
            rate = written / (time.time() - started)
            print(f"  {written:,} passages  {rate:,.0f}/s", flush=True)
print(f"  wrote {written:,} passages in {time.time() - started:.0f}s", flush=True)
PY
  touch "$LOCAL/.passages_written"
fi
log "documents on disk: $(find "$DOCS_DIR" -type f -name '*.txt' | wc -l)"

# ---------------------------------------------------------------- phase 2
log "starting lilbee and warming the embedder"
"$LILBEE_BIN" serve --port 8080 >"$LOG_DIR/serve.log" 2>&1 &
SERVE_PID=$!
# lilbee lazy-loads workers on the first request; gating on a readiness flag
# deadlocks because the flag only flips once a request arrives.
"$PYBIN" - <<'PY'
import httpx, os, sys, time
for attempt in range(90):
    try:
        httpx.post("http://127.0.0.1:8080/v1/embeddings",
                   json={"model": os.environ["EMBED_MODEL"], "input": "warmup"},
                   timeout=180).raise_for_status()
        print("  embedder warm"); break
    except Exception:
        time.sleep(10)
else:
    sys.exit("embedder never came up")
PY

# ---------------------------------------------------------------- phase 3
log "ingesting (this is the long one; trace at $LILBEE_INGEST_TRACE_FILE)"
"$PYBIN" - <<'PY'
import os, pathlib, subprocess, sys
sys.path.insert(0, "/opt/lilbee-src")
from evals.infra.provenance import start

run = start("ingest")
docs = pathlib.Path(os.environ["DOCS_DIR"])
total = sum(1 for _ in docs.rglob("*.txt"))
with run.stage("embed-and-index", documents=total,
               command=f"lilbee add {docs}") as stage:
    result = subprocess.run([os.environ["LILBEE_BIN"], "add", str(docs)],
                            capture_output=False)
    data = pathlib.Path(os.environ["LILBEE_DATA"])
    stage.bytes_out = sum(f.stat().st_size for f in data.rglob("*") if f.is_file())
    stage.notes = f"exit={result.returncode}"
run.write(pathlib.Path(os.environ["PROVENANCE"]))
print(f"  indexed; provenance appended to {os.environ['PROVENANCE']}")
sys.exit(result.returncode)
PY

# ---------------------------------------------------------------- phase 4
# Where the time actually went. This is the question the tracing exists to
# answer and the reason the run is worth doing at this scale.
log "extraction versus everything else"
"$PYBIN" - <<'PY'
import os, pathlib, re
trace = pathlib.Path(os.environ["LILBEE_INGEST_TRACE_FILE"])
elapsed = [int(m) for m in re.findall(r"elapsed_ms=(\d+)", trace.read_text())] if trace.exists() else []
ocr = len(re.findall(r"vision-ocr ", trace.read_text())) if trace.exists() else 0
if elapsed:
    total_s = sum(elapsed) / 1000
    print(f"  documents traced : {len(elapsed):,}")
    print(f"  extraction time  : {total_s / 3600:.2f} h  (sum of per-document elapsed_ms)")
    print(f"  median / p99     : {sorted(elapsed)[len(elapsed)//2]} ms / "
          f"{sorted(elapsed)[int(len(elapsed)*0.99)]} ms")
    print(f"  vision-OCR fired : {ocr:,} times")
else:
    print("  NO TRACE LINES - the extraction/GPU split cannot be reported")
PY

# grep -q, never `grep -c || echo 0`: grep -c prints "0" AND exits 1 on no match,
# so the `|| echo 0` appends a second line and the check false-fires. That
# powered off a healthy pod mid-run once.
if grep -qiE "traceback|failed to extract|extraction failed" "$LOG_DIR/serve.log"; then
  log "FAILURES present - see $LOG_DIR/serve.log"
  grep -icE "traceback|failed to extract" "$LOG_DIR/serve.log" | xargs -I{} log "  {} failure lines"
fi

# ---------------------------------------------------------------- phase 5
# One tarball, not thousands of loose files. Large sequential I/O is the only
# thing MooseFS does well; the index as loose files is what makes it crawl.
log "archiving the index to the volume"
kill "$SERVE_PID" 2>/dev/null || true
tar -C "$LOCAL" -cf "$WORKSPACE/msmarco_index.tar" data
log "index tar: $(du -h "$WORKSPACE/msmarco_index.tar" | cut -f1)"
cp "$LILBEE_INGEST_TRACE_FILE" "$WORKSPACE/logs/" 2>/dev/null || true

log "DONE - power the pod off with: sky down msmarco-ingest -y"
