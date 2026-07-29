#!/usr/bin/env bash
# Shard arithmetic shared by the scripts that have to agree about it.
#
# checkpoint.sh writes a slot and restore.sh reads one. If the two ever computed
# the path differently a restore would return another shard's index and the merge
# would combine the wrong slices, so both source this instead of repeating it.
#
# Inputs (all optional, single-host defaults):
#   SHARD_INDEX, SHARD_COUNT   this host's slice of the corpus
#   CHECKPOINT_TOTAL           GLOBAL corpus size (or SMOKE_N when smoking)
# Outputs:
#   CKPT_PATH                  path in the checkpoint repo for this shard
#   SHARD_TOTAL                passages this host is expected to ingest

: "${SHARD_INDEX:=0}"
: "${SHARD_COUNT:=1}"

if [ "$SHARD_COUNT" -lt 1 ] || [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]; then
  echo "FATAL: SHARD_INDEX=$SHARD_INDEX must be 0..$((SHARD_COUNT - 1)) with SHARD_COUNT=$SHARD_COUNT" >&2
  return 1 2>/dev/null || exit 1
fi

# A single-host run keeps the original path so checkpoints written before
# sharding existed still load.
if [ "$SHARD_COUNT" -gt 1 ]; then
  CKPT_PATH="shard-${SHARD_INDEX}of${SHARD_COUNT}/checkpoint-latest.tar"
else
  CKPT_PATH="checkpoint-latest.tar"
fi

# The count ingest.sh materialises: global indices i with i % COUNT == INDEX.
if [ -n "${CHECKPOINT_TOTAL:-}" ]; then
  SHARD_TOTAL=$(( (CHECKPOINT_TOTAL + SHARD_COUNT - 1 - SHARD_INDEX) / SHARD_COUNT ))
  # Milestones divide by this; a zero would abort the watcher on the first poll.
  [ "$SHARD_TOTAL" -lt 1 ] && SHARD_TOTAL=1
fi
