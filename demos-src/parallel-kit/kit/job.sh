#!/usr/bin/env bash
# Per-pod reel job. Args: <group-id> <reel-name>...
# Everything hot lives on local NVMe; the volume is read-only except this
# pod's own /workspace/reels-out/<reel>/ dirs. Self-terminates when done,
# on any crash (EXIT trap), or via the idle watchdog backstop.
set -uo pipefail
GROUP="$1"; shift
REELS=("$@")
export KIT=/root/kit
OUT_BASE=/workspace/reels-out
RUNPOD_POD_ID="${RUNPOD_POD_ID:?}"
API_KEY_FILE=/root/.runpod_key
TERMINATED=""

mkdir -p /root/takes "$OUT_BASE/_logs"
log() { echo "[job $(date -u +%H:%M:%S)] $*" | tee -a /root/job.log; }

terminate() {
  [ -n "${NO_TERMINATE:-}" ] && { log "NO_TERMINATE set — leaving pod alive (status=${1:-?})"; return 0; }
  [ -n "$TERMINATED" ] && return 0
  TERMINATED=1
  sync
  log "terminating pod $RUNPOD_POD_ID (status=${1:-crashed})"
  cp /root/job.log "$OUT_BASE/_logs/${GROUP}-job.log" 2>/dev/null || true
  curl -s -X POST "https://api.runpod.io/graphql?api_key=$(cat $API_KEY_FILE)" \
    -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)' \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" >/dev/null || true
  # if the API call fails, idle_watchdog and scatter watch() are the backstops
}
trap 'terminate crashed' EXIT

fail_reel() { # <reel> <reason> — upload failure bundle
  local reel="$1" reason="$2"
  log "FAIL $reel: $reason"
  mkdir -p "$OUT_BASE/$reel"
  { echo "$reason"; echo "--- gate.log:"; tail -120 /root/gate.log 2>/dev/null;
    echo "--- take.log:"; tail -120 /root/take.log 2>/dev/null; } > "$OUT_BASE/$reel/FAILED.txt"
  cp /root/takes/"$reel".mp4 "$OUT_BASE/$reel/" 2>/dev/null || true
  cp /root/qa-"$reel".json "$OUT_BASE/$reel/" 2>/dev/null || true
}

quiesce_render() {
  # kill only the RENDER pipeline (between take attempts) — leaves the warm
  # fleet (llama-swap/llama-server) running so the next take is not cold
  tmux kill-session -t take 2>/dev/null
  pkill -9 -x vhs 2>/dev/null; pkill -9 -x ttyd 2>/dev/null
  pkill -9 -f '[c]hrome.*--headless' 2>/dev/null
  sleep 2
}

quiesce() {
  # full reset at reel boundaries — also kills the fleet
  quiesce_render
  tmux kill-session -t fleet 2>/dev/null; tmux kill-session -t warm 2>/dev/null
  pkill -9 -x llama-swap 2>/dev/null; pkill -9 -x llama-server 2>/dev/null
  pkill -9 -f '[l]ilbee serve' 2>/dev/null
  sleep 3
}

# ---- boot ----
bash "$KIT/bootstrap.sh" || { fail_reel "${REELS[0]}" "bootstrap failed"; terminate boot_fail; exit 1; }
source "$KIT/env.sh"

# GPU health check BEFORE the expensive materialize: cloud pods occasionally
# ship a bad GPU (device enumerates but cudaSetDevice fails "busy or
# unavailable"). A single-GPU model would never touch it, but a multi-GPU
# take crashes at fleet boot after wasting the model pull. Test every visible
# device individually; fail fast + self-terminate so the group is re-launched
# on a fresh pod instead of burning materialize time on a dead card.
# real-compute health check (matmul per device) — a bad card enumerates fine
# but fails compute; catch it here before the 20-min materialize
if ! /usr/bin/python3 "$KIT/gpu_health.py" > /root/gpuhealth.log 2>&1; then
  cat /root/gpuhealth.log
  fail_reel "${REELS[0]}" "bad GPU on this pod: $(grep FAIL /root/gpuhealth.log | tr '\n' ' ')"
  terminate bad_gpu; exit 1
fi
log "GPU health: $(grep GPU_HEALTH_OK /root/gpuhealth.log)"

# derive a generous hard cap from this pod's total workload — heavy multi-reel
# groups legitimately run hours; the real idle guard is NO_JOB_GRACE (job gone)
export HARD_DEADLINE_S=$(python3 "$KIT/deadline.py" "$@")
nohup bash "$KIT/idle_watchdog.sh" > /root/watchdog.log 2>&1 &

POD_CLASSES=$(python3 -c "
import sys, yaml
m = yaml.safe_load(open('/root/kit/reels.yaml'))
print(','.join(sorted({m['reels'][r]['class'] for r in sys.argv[1:]})))" ${REELS[@]})
QUALGATE_LIGHT=1 QUALGATE_CLASSES="$POD_CLASSES" bash "$KIT/qualgate.sh" || { fail_reel "${REELS[0]}" "qualgate failed"; terminate qualgate_fail; exit 1; }

python3 "$KIT/materialize.py" "$KIT/reels.yaml" --group "$GROUP" \
  --models-dst /root/models 2>&1 | tee -a /root/job.log
[ "${PIPESTATUS[0]}" = 0 ] || { fail_reel "${REELS[0]}" "materialize failed"; terminate materialize_fail; exit 1; }

# warm page cache from LOCAL disk (no-op beyond RAM size; cheap)
find /root/models -name '*.gguf' -exec sh -c 'cat "$1" > /dev/null' _ {} \; 2>/dev/null

if [ "$GROUP" = "CANARY" ]; then
  bash "$KIT/canary_grade.sh" 2>&1 | tee -a /root/take.log \
    || { fail_reel "${REELS[0]}" "canary graded ask failed"; terminate canary_grade_fail; exit 1; }
fi

# ---- takes ----
overall=pass
for reel in "${REELS[@]}"; do
  passed=""
  take_budget=$(python3 - "$reel" <<'PY'
import sys, yaml
m = yaml.safe_load(open('/root/kit/reels.yaml'))
r = m['reels'][sys.argv[1]]
w = r.get('windows') or {}
d = r.get('duration_s') or {}
print(int(w.get('boot', 120)) + 3 * int(d.get('max', 300)) + 600)
PY
)
  [ -n "$take_budget" ] || take_budget=3600
  : > /root/gate.log
  : > /root/take.log

  # --- one-time reset + stage + WARM the fleet (gate, NOT a take attempt) ---
  quiesce
  if ! python3 "$KIT/stage.py" "$KIT/reels.yaml" "$reel" >> /root/gate.log 2>&1; then
    fail_reel "$reel" "stage failed"; overall=fail; continue
  fi
  # warm.sh runs the render/disk/font checks, boots the fleet, verifies
  # identity + a live 200, and LEAVES the fleet running. Retried up to 3x
  # because the FIRST cold boot of a heavy model races startup; a warm/gate
  # failure never consumes a take attempt. Once warm, every take is fast.
  warmed=""
  for w in 1 2 3; do
    if bash "$KIT/warm.sh" "$reel" >> /root/gate.log 2>&1; then warmed=1; break; fi
    log "warm attempt $w failed for $reel; retrying"
    quiesce
    python3 "$KIT/stage.py" "$KIT/reels.yaml" "$reel" >> /root/gate.log 2>&1 || true
  done
  if [ -z "$warmed" ]; then
    fail_reel "$reel" "fleet never warmed after 3 tries (see gate.log)"; overall=fail; continue
  fi

  # --- take attempts against the WARM fleet (fleet stays up between them) ---
  for attempt in 1 2 3; do
    log "reel $reel take attempt $attempt (budget ${take_budget}s)"
    quiesce_render
    rm -f /root/takes/"$reel".mp4 /root/takes/"$reel".png /root/TAKE_EXIT
    : > /root/take.log
    tmux new-session -d -s take "cd /root/takes && vhs $KIT/tapes/$reel.tape >> /root/take.log 2>&1; echo \$? > /root/TAKE_EXIT"
    waited=0
    while [ ! -f /root/TAKE_EXIT ] && [ "$waited" -lt "$take_budget" ]; do sleep 10; waited=$((waited+10)); done
    if [ ! -f /root/TAKE_EXIT ]; then
      log "take timed out after ${take_budget}s (attempt $attempt)"; tmux kill-session -t take 2>/dev/null; continue
    fi
    rc=$(cat /root/TAKE_EXIT); rm -f /root/TAKE_EXIT
    { [ "$rc" = 0 ] && [ -s /root/takes/"$reel".mp4 ]; } || { log "take rc=$rc (attempt $attempt)"; continue; }

    if python3 "$KIT/autoqa.py" "$KIT/reels.yaml" "$reel" /root/takes/"$reel".mp4 --report /root/qa-"$reel".json; then
      mkdir -p "$OUT_BASE/$reel"
      cp /root/takes/"$reel".mp4 /root/qa-"$reel".json "$OUT_BASE/$reel/"
      [ -f /root/takes/"$reel".png ] && cp /root/takes/"$reel".png "$OUT_BASE/$reel/"
      a=$(sha256sum /root/takes/"$reel".mp4 | cut -d' ' -f1)
      b=$(sha256sum "$OUT_BASE/$reel/$reel.mp4" | cut -d' ' -f1)
      [ "$a" = "$b" ] || { fail_reel "$reel" "volume read-back sha mismatch"; overall=fail; break; }
      log "PASS $reel"; passed=1; break
    fi
    log "autoqa fail $reel attempt $attempt"
  done
  quiesce   # kill the warm fleet before the next reel
  if [ -z "$passed" ]; then
    [ -f "$OUT_BASE/$reel/FAILED.txt" ] || fail_reel "$reel" "no passing take after 3 attempts"
    overall=fail
  fi
done

terminate "$overall"
trap - EXIT
