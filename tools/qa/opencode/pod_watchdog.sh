#!/usr/bin/env bash
# Stop the pod when the QA run stalls: no new writes under any watched path and
# no GPU activity for IDLE_MIN minutes means something hung or died, and an idle
# pod bills for nothing. Linux-only; matrix.py arms it when RUNPOD_POD_ID is set.
# Usage: [IDLE_MIN=30] pod_watchdog.sh <watch-path> [<watch-path>...]
set -euo pipefail

[ "$#" -ge 1 ] || { echo "usage: [IDLE_MIN=30] pod_watchdog.sh <watch-path>..." >&2; exit 2; }
IDLE_MIN="${IDLE_MIN:-30}"
GPU_BUSY_PCT=5

# Survival rules learned the hard way: a watch path may not exist yet (HF_HOME
# before the first download), and `head -1` closing the pipe early SIGPIPEs the
# producers, which pipefail turns into a silent death on any large tree. Either
# way the watchdog dies and the billing protection it exists to provide is
# gone, so absent paths are tolerated and `sed -n 1p` (which drains its input)
# takes the place of head.
newest_mtime() {
  { find "$@" -type f -printf '%T@\n' 2>/dev/null || true; } | sort -nr | sed -n 1p | cut -d. -f1
}

echo "[watchdog] watching: $* (stop after ${IDLE_MIN}m of no writes + idle GPUs)"

last_seen="$(newest_mtime "$@")"
last_seen="${last_seen:-0}"
last_change="$(date +%s)"

while true; do
  mtime="$(newest_mtime "$@")"
  mtime="${mtime:-0}"
  gpu="$({ nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true; } | sort -nr | sed -n 1p)"
  gpu="${gpu:-0}"
  if [ "$mtime" != "$last_seen" ] || [ "$gpu" -gt "$GPU_BUSY_PCT" ]; then
    last_seen="$mtime"
    last_change="$(date +%s)"
  fi
  if [ $(($(date +%s) - last_change)) -ge $((IDLE_MIN * 60)) ]; then
    echo "[watchdog] no writes or GPU activity for ${IDLE_MIN}m; stopping pod" >&2
    if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
      runpodctl stop pod "$RUNPOD_POD_ID"
    fi
    exit 1
  fi
  sleep 60
done
