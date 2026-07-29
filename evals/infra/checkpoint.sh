#!/usr/bin/env bash
# Milestone checkpoint of the growing lilbee index to a private HF dataset, so a
# pod-loss mid-run costs at most one milestone of GPU work instead of the whole
# ingest. We run volume-free (multi-region provisioning needs it -- a network
# volume pins the pod to one scarce datacenter), so object storage is the only
# durable target that does not reintroduce that region pin.
#
# Cadence is milestone-based, not time-based: the payload is ~100GB+ of fp32
# vectors, so an hourly push would eat a large share of the run's I/O for a
# low-probability event (an on-demand pod is not preempted; the real risk is
# hardware failure or an accidental stop). Pushing at 25/50/75/100% embedded
# keeps the safety net proportionate and off the GPU critical path.
#
# Restore (a fresh pod after a loss): download the latest checkpoint tar, extract
# into LILBEE_DATA, then `lilbee sync` resumes -- it is hash-incremental, so it
# re-embeds only the tail the checkpoint missed and skips everything already in
# the restored index.
set -uo pipefail

: "${LILBEE_DATA:=/root/msmarco/data}"
: "${CHECKPOINT_REPO:=beeberg/msmarco-ingest-checkpoint}"
: "${CHECKPOINT_TOTAL:=8841823}"        # GLOBAL corpus size (or SMOKE_N when smoking)
: "${PYBIN:=/root/lilbee_venv/bin/python}"
: "${LOG_DIR:=/root/bench/logs}"
: "${CHECKPOINT_POLL_S:=120}"           # how often to read the row count
mkdir -p "$LOG_DIR"

# CKPT_PATH (this shard's slot) and SHARD_TOTAL (what this host will ingest).
# Milestones are a fraction of THIS host's slice: keying them off the global
# corpus means a 2-shard host tops out near 50%, so the 75% and 100% milestones
# never fire, the final checkpoint is never pushed and the watcher never exits.
# shellcheck source=evals/infra/shard_env.sh
. "$(dirname "$0")/shard_env.sh" || exit 1
CKPT_LOG="$LOG_DIR/checkpoint.log"

log() { printf '[ckpt %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$CKPT_LOG"; }

rows() {
  LILBEE_DATA="$LILBEE_DATA" "$PYBIN" - <<'PY' 2>/dev/null || echo 0
import os, lancedb
try:
    db = lancedb.connect(os.path.join(os.environ["LILBEE_DATA"], "data/lancedb"))
    print(db.open_table("_page_texts").count_rows())
except Exception:
    print(0)
PY
}

push_checkpoint() {
  local tag="$1" n="$2"
  local tar="/root/ckpt-${tag}.tar"
  log "milestone ${tag}: rows=${n} -> taring index (best-effort, live)"
  # Lance commits the manifest last, so a live tar captures a consistent-or-stale
  # version; --ignore-failed-read tolerates a fragment rotating out mid-read.
  tar -C "$LILBEE_DATA" --ignore-failed-read -cf "$tar" data config.toml 2>/dev/null \
    || { log "WARN: tar failed for ${tag}"; return 1; }
  log "milestone ${tag}: tar=$(du -h "$tar" | cut -f1); uploading to ${CHECKPOINT_REPO}"
  CKPT_TAR="$tar" CKPT_TAG="$tag" CKPT_ROWS="$n" CHECKPOINT_REPO="$CHECKPOINT_REPO" \
    CKPT_PATH="$CKPT_PATH" \
    "$PYBIN" - <<'PY' 2>>"$CKPT_LOG" && log "milestone ${tag}: uploaded to ${CKPT_PATH}" || log "WARN: upload failed for ${tag}"
import os
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj=os.environ["CKPT_TAR"],
    # One rolling slot per shard: newest wins, storage stays bounded.
    path_in_repo=os.environ["CKPT_PATH"],
    repo_id=os.environ["CHECKPOINT_REPO"],
    repo_type="dataset",
    commit_message=f"checkpoint {os.environ['CKPT_TAG']} at {os.environ['CKPT_ROWS']} rows",
)
PY
  rm -f "$tar"
}

log "checkpoint watcher up: repo=${CHECKPOINT_REPO} shard=${SHARD_INDEX}/${SHARD_COUNT} \
target=${SHARD_TOTAL} of global ${CHECKPOINT_TOTAL} poll=${CHECKPOINT_POLL_S}s"
done_25=0 done_50=0 done_75=0 done_100=0
while :; do
  N=$(rows); N=${N:-0}
  PCT=$(( N * 100 / SHARD_TOTAL ))
  log "progress: ${N}/${SHARD_TOTAL} (${PCT}%)"
  if [ "$done_25" = 0 ] && [ "$PCT" -ge 25 ]; then push_checkpoint "25pct" "$N"; done_25=1; fi
  if [ "$done_50" = 0 ] && [ "$PCT" -ge 50 ]; then push_checkpoint "50pct" "$N"; done_50=1; fi
  if [ "$done_75" = 0 ] && [ "$PCT" -ge 75 ]; then push_checkpoint "75pct" "$N"; done_75=1; fi
  if [ "$done_100" = 0 ] && [ "$PCT" -ge 100 ]; then push_checkpoint "100pct" "$N"; done_100=1; log "final milestone reached; watcher exiting"; break; fi
  sleep "$CHECKPOINT_POLL_S"
done
