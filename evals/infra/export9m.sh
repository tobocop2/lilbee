#!/usr/bin/env bash
# Export the merged index and push it to HuggingFace: parquet and jsonl of the
# passage text, plus the full index with its vectors and search indexes.
#
# The text side is lilbee's own exporter, once per format. `lilbee export` writes
# a {source, page, text} dataset and infers parquet or jsonl from the suffix, so
# there is no reason to hand-roll a writer; note it drops vectors by design,
# which is why the index goes up as a folder rather than as a wide parquet.
#
# Uploads go straight from where the data already is: upload_folder takes a
# path_in_repo, so dataset/ and index/ are chosen at upload time and nothing has
# to be copied or hard-linked into a staging tree. (upload_large_folder would
# need that staging, and hub 1.23 deprecates it in favour of this.)
#
# Sizes to expect at the full corpus, so nothing is a surprise at 3am:
#   parquet  ~1-2 GB     jsonl  ~3-4 GB     index  ~150 GB (8.8M x 4096 x 4B)
# The index dominates the upload; UPLOAD_INDEX=0 keeps the datasets and skips it.
set -uo pipefail
: "${HF_REPO:=beeberg/msmarco-ingest-checkpoint}"
: "${HF_PRIVATE:=1}"
: "${UPLOAD_INDEX:=1}"
: "${INDEX_ROOT:=/root/merged}"
: "${EXPORT_PREFIX:=msmarco}"
: "${EXPORT_DIR:=/root/export}"
VPY=$(cat /root/status/vpy 2>/dev/null || echo /root/venv/bin/python)
LB=$(dirname "$VPY")/lilbee
log() { printf '[export %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
fail() { log "FAILED: $*"; }

[ -d "$INDEX_ROOT/data/lancedb" ] || { fail "no merged index at $INDEX_ROOT"; exit 1; }
[ -n "${HF_TOKEN:-}" ] || { fail "no HF_TOKEN in the environment"; exit 1; }

rm -rf "$EXPORT_DIR"; mkdir -p "$EXPORT_DIR"
log "exporting parquet and jsonl from $INDEX_ROOT"
for fmt in parquet jsonl; do
  out="$EXPORT_DIR/${EXPORT_PREFIX}-passages.${fmt}"
  if "$LB" export "$out" --data-dir "$INDEX_ROOT" 2>&1 | tail -2; then
    log "  ${fmt}: $(du -h "$out" 2>/dev/null | cut -f1)"
  else
    fail "lilbee export ${fmt}"; exit 1
  fi
done

# A run summary travels with the data: an index with no provenance is not
# reusable six months later.
{
  echo "# MS MARCO passage index (lilbee, per-GPU ingest)"
  echo
  echo '```'
  cat /root/status/run.env 2>/dev/null
  grep -aE '^INGEST|^MERGE' /root/ingest.log 2>/dev/null
  echo '```'
  echo
  echo "Built $(date -u +%Y-%m-%dT%H:%M:%SZ) by evals/infra/ingest9m.sh."
  echo
  echo "- \`dataset/\` passage text as parquet and jsonl (no vectors)"
  echo "- \`index/\` a lilbee data root: point lilbee at it with"
  echo "  \`lilbee search --data-dir <dir>\` or \`LILBEE_DATA=<dir>\`"
} > "$EXPORT_DIR/README.md"

# Telemetry travels with the artifacts: the traces are the point of the run for
# anyone analysing extraction, and a report is not reproducible without the
# samplers behind it.
TELEMETRY="$EXPORT_DIR/../telemetry"
rm -rf "$TELEMETRY"; mkdir -p "$TELEMETRY"
log "collecting telemetry"
"$VPY" /root/prof/summarize.py /root/prof > /root/prof/SUMMARY.txt 2>&1 || true
for f in /root/prof/w*.trace.log.gz /root/prof/w*.trace.log /root/prof/*.folded \
         /root/prof/*.svg /root/prof/*.csv /root/prof/SUMMARY.txt /root/ingest.log; do
  [ -e "$f" ] || continue
  cp -a "$f" "$TELEMETRY/" 2>/dev/null
done
# Traces are the bulk; compress anything that escaped the run's own gzip.
for t in "$TELEMETRY"/w*.trace.log; do
  [ -e "$t" ] && gzip -f "$t" 2>/dev/null
done
cat > "$TELEMETRY/README.md" <<'TEOF'
# Ingest telemetry

Samplers and profiles from one full-corpus lilbee ingest across 8 GPUs.

| file | what |
|---|---|
| `SUMMARY.txt` | whole-run CPU/GPU load, power, throughput, extraction percentiles |
| `w<i>.trace.log.gz` | one line per extracted file: source, type, elapsed_ms, pages, chunks, ocr_pages |
| `w1.gil.folded` | py-spy samples holding the GIL (folded stacks) |
| `w2.wall.folded` | py-spy samples of all wall time (folded stacks) |
| `*.svg` | flame graphs rendered from the folded stacks |
| `sys.csv` | ts, cpu_pct, gpu_util_mean, gpu_mem_mb, gpu_watts, read_mb, write_mb (5s) |
| `gpu.csv` | ts + per-card GPU utilisation, all 8 cards (2s) |
| `rows.csv` | ts, cumulative rows across workers (20s) |
| `host.csv` | ts, load average, thread count, RSS MB (10s) |
| `ingest.log` | the run's own log |

The folded files load directly in https://speedscope.app and render with
flamegraph.pl. GIL-held fraction of sampled wall time is
`sum(w1.gil.folded) / sum(w2.wall.folded)`.
TEOF
log "  telemetry: $(du -sh "$TELEMETRY" 2>/dev/null | cut -f1) in $(ls "$TELEMETRY" | wc -l) files"

log "uploading to $HF_REPO (private=$HF_PRIVATE); the index is the long part"
HF_REPO="$HF_REPO" HF_PRIVATE="$HF_PRIVATE" EXPORT_DIR="$EXPORT_DIR" \
INDEX_ROOT="$INDEX_ROOT" UPLOAD_INDEX="$UPLOAD_INDEX" TELEMETRY="$TELEMETRY" "$VPY" - <<'PYEOF'
import os
from huggingface_hub import HfApi

repo = os.environ["HF_REPO"]
api = HfApi(token=os.environ["HF_TOKEN"])
# exist_ok so a rerun resumes into the same repo rather than failing, and private
# by default: publishing is something the caller opts into, not out of.
api.create_repo(repo_id=repo, repo_type="dataset",
                private=os.environ.get("HF_PRIVATE", "1") == "1", exist_ok=True)
print(f"[export] repo ready: {repo}", flush=True)

# Subdirectories, so whatever else lives in this repo (the source tarballs) is
# untouched: upload_folder adds paths, it does not mirror-delete.
api.upload_folder(repo_id=repo, repo_type="dataset",
                  folder_path=os.environ["EXPORT_DIR"], path_in_repo="dataset",
                  commit_message="passage dataset: parquet + jsonl")
print("[export] dataset/ uploaded", flush=True)

telemetry = os.environ.get("TELEMETRY", "")
if telemetry and os.path.isdir(telemetry):
    api.upload_folder(repo_id=repo, repo_type="dataset",
                      folder_path=telemetry, path_in_repo="telemetry",
                      commit_message="ingest telemetry: traces, profiles, samplers")
    print("[export] telemetry/ uploaded", flush=True)

if os.environ.get("UPLOAD_INDEX", "1") == "1":
    print("[export] uploading the index; hours at full corpus", flush=True)
    api.upload_folder(repo_id=repo, repo_type="dataset",
                      folder_path=os.environ["INDEX_ROOT"], path_in_repo="index",
                      commit_message="lilbee index: vectors, ANN and FTS")
    print("[export] index/ uploaded", flush=True)
else:
    print("[export] UPLOAD_INDEX=0: datasets only", flush=True)
PYEOF
rc=$?
[ "$rc" = "0" ] || { fail "upload rc=$rc"; exit 1; }
log "DONE: https://huggingface.co/datasets/$HF_REPO"
