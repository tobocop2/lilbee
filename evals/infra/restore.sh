#!/usr/bin/env bash
# Resume a lost ingest on a fresh pod: pull the latest milestone checkpoint from
# the private HF dataset and extract it into LILBEE_DATA, so the following
# `lilbee sync` re-embeds only the tail the checkpoint missed instead of redoing
# the whole corpus. Run this BEFORE ingest.sh on a relaunch; it is a no-op (and
# exits 0) when no checkpoint exists yet, so it is safe to run unconditionally.
#
# Needs HF_TOKEN in the environment (the checkpoint repo is private).
set -uo pipefail

: "${LILBEE_DATA:=/root/msmarco/data}"
: "${CHECKPOINT_REPO:=beeberg/msmarco-ingest-checkpoint}"
: "${PYBIN:=/root/lilbee_venv/bin/python}"

log() { printf '[restore %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

if [ -d "$LILBEE_DATA/data/lancedb" ]; then
  log "index already present at $LILBEE_DATA/data/lancedb; not restoring over it"
  exit 0
fi

log "checking $CHECKPOINT_REPO for a checkpoint"
TAR=$(CHECKPOINT_REPO="$CHECKPOINT_REPO" "$PYBIN" - <<'PY'
import os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
repo = os.environ["CHECKPOINT_REPO"]
try:
    files = api.list_repo_files(repo, repo_type="dataset")
except Exception:
    files = []
if "checkpoint-latest.tar" not in files:
    print("")  # nothing to restore
else:
    print(hf_hub_download(repo, "checkpoint-latest.tar", repo_type="dataset"))
PY
)

if [ -z "$TAR" ]; then
  log "no checkpoint in repo yet; starting from an empty index"
  exit 0
fi

log "extracting $(du -h "$TAR" | cut -f1) checkpoint into $LILBEE_DATA"
mkdir -p "$LILBEE_DATA"
tar -C "$LILBEE_DATA" -xf "$TAR" && log "restore complete; sync will resume from here" || {
  log "WARN: extract failed; removing partial and starting fresh"
  rm -rf "$LILBEE_DATA/data"
}
