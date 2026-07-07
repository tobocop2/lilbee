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

quiesce() {
  tmux kill-session -t take 2>/dev/null; tmux kill-session -t fleet 2>/dev/null
  tmux kill-session -t dryrun 2>/dev/null
  pkill -9 -x llama-swap 2>/dev/null; pkill -9 -x llama-server 2>/dev/null
  pkill -9 -f '[l]ilbee serve' 2>/dev/null
  pkill -9 -x vhs 2>/dev/null; pkill -9 -x ttyd 2>/dev/null
  pkill -9 -f '[c]hrome.*--headless' 2>/dev/null
  sleep 3
}

# ---- boot ----
bash "$KIT/bootstrap.sh" || { fail_reel "${REELS[0]}" "bootstrap failed"; terminate boot_fail; exit 1; }
source "$KIT/env.sh"
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
  pre=$(python3 -c "import yaml; r=yaml.safe_load(open('/root/kit/reels.yaml'))['reels']['$reel']; print(r.get('pre_roll',''))")
  for attempt in 1 2; do
    log "reel $reel attempt $attempt (budget ${take_budget}s)"
    quiesce
    : > /root/gate.log
    : > /root/take.log
    python3 "$KIT/stage.py" "$KIT/reels.yaml" "$reel" >> /root/gate.log 2>&1 \
      || { log "stage failed (attempt $attempt)"; continue; }
    rm -f /root/takes/"$reel".mp4 /root/takes/"$reel".png /root/TAKE_EXIT

    if [ -n "$pre" ]; then
      # the dry-run boots and proves the fleet itself; no separate pretake boot
      bash "$KIT/${pre}.sh" >> /root/gate.log 2>&1 || { log "pre_roll $pre failed (attempt $attempt)"; continue; }
      quiesce
      python3 "$KIT/stage.py" "$KIT/reels.yaml" "$reel" >> /root/gate.log 2>&1 \
        || { log "post-dryrun stage failed (attempt $attempt)"; continue; }
    fi
    bash "$KIT/pretake.sh" "$reel" >> /root/gate.log 2>&1 \
      || { log "pretake gate failed (attempt $attempt)"; continue; }

    tmux new-session -d -s take "cd /root/takes && vhs $KIT/tapes/$reel.tape >> /root/take.log 2>&1; echo \$? > /root/TAKE_EXIT"
    waited=0
    while [ ! -f /root/TAKE_EXIT ] && [ "$waited" -lt "$take_budget" ]; do sleep 10; waited=$((waited+10)); done
    if [ ! -f /root/TAKE_EXIT ]; then
      log "take timed out after ${take_budget}s (attempt $attempt)"
      tmux kill-session -t take 2>/dev/null
      continue
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
      log "PASS $reel"
      passed=1
      break
    fi
    log "autoqa fail $reel attempt $attempt"
  done
  if [ -z "$passed" ]; then
    [ -f "$OUT_BASE/$reel/FAILED.txt" ] || fail_reel "$reel" "no passing take after 2 attempts (see take.log tail)"
    overall=fail
  fi
done

terminate "$overall"
trap - EXIT
