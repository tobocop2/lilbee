#!/usr/bin/env bash
# Autonomous pod-side demo matrix.
#
# Runs each model end-to-end via model_demo.sh, saves durable results to
# /workspace/results/<family>/ (the RunPod volume persists when the pod is
# stopped), gates the giants behind a cheap qwen3-coder canary that must render a
# real (multi-frame) gif, then POWERS THE POD OFF itself.
#
# Built to run unattended while the operator is offline:
#   cd /root/reel && git pull -q origin demo-reel/opencode-model-matrix
#   tmux new-session -d -s matrix "/root/reel/demos-src/opencode-model-reel/run_matrix.sh"
#
# Resumable: a model whose result dir has a `done` marker is skipped, so on a
# reconnect you can re-launch this and it continues. Per-model STATUS + the
# top-level STATE.md record exactly where it got and whether each demo passed.
set -uo pipefail
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1

RES=/workspace/results
DRIVER=/root/reel/demos-src/opencode-model-reel/model_demo.sh
MIN_FRAMES=10            # a real demo gif has hundreds of frames; >this rejects the empty-gif failure
mkdir -p "$RES"
ts(){ date -u +'%Y-%m-%d %H:%M:%S'; }
note(){ echo "[$(ts)] $*" | tee -a "$RES/matrix.log"; }

# Pod id for self-power-off. The tmux server may not carry RUNPOD_POD_ID, so fall
# back to PID 1's environment (the pod's init always has it).
POD_ID="${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p')}"

# family|full|repo|quant|qdir|multigpu   (only verified refs; giants needing ref
# checks are listed in STATE.md for a supervised pass)
MODELS=(
  "qwen3-coder|Qwen3-Coder-30B|unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF|*Q4_K_M*|/workspace/models/qwen3-coder|0"
  "minimax-m2|MiniMax-M2|unsloth/MiniMax-M2-GGUF|Q8_0/*|/workspace/models/minimax-q8|1"
)

install_runpodctl(){
  command -v runpodctl >/dev/null 2>&1 && return 0
  note "installing runpodctl"
  wget -qO /usr/local/bin/runpodctl \
    https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 2>/dev/null \
    && chmod +x /usr/local/bin/runpodctl
  command -v runpodctl >/dev/null 2>&1
}

poweroff_pod(){
  note "powering off pod ${POD_ID:-UNKNOWN}"
  sync
  if [ -z "$POD_ID" ]; then note "no pod id; cannot self-power-off (stop it manually)"; return; fi
  if install_runpodctl; then
    runpodctl stop pod "$POD_ID" || note "runpodctl stop failed; stop it manually"
  else
    note "runpodctl unavailable; stop the pod manually"
  fi
}

# --- stall watchdog -------------------------------------------------------------
# Powers the pod off if NOTHING makes progress for STALL_LIMIT: no log file is
# written, the download dir stops growing, AND the GPUs go idle. A slow-but-real
# giant download (bytes grow) or inference/load (GPU busy or giant-srv.log ticks)
# keeps resetting the timer, so only a genuine wedge trips it.
STALL_LIMIT="${STALL_LIMIT:-1800}"   # 30 min of total silence

gpu_busy(){  # echo 1 if any GPU is doing work, else 0
  local u
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END{print s+0}')
  [ "${u:-0}" -ge 5 ] && echo 1 || echo 0
}

progress_fp(){  # newest log mtime : total downloaded bytes
  local newest=0 t f
  for f in "$RES"/matrix.log "$RES"/*/driver.log /root/dl-*.log \
           /tmp/lilbee-serve.log /tmp/giant-srv.log; do
    [ -f "$f" ] || continue
    t=$(stat -c %Y "$f" 2>/dev/null) || t=0
    [ "$t" -gt "$newest" ] && newest=$t
  done
  local bytes; bytes=$(du -sb /workspace/models 2>/dev/null | cut -f1); bytes=${bytes:-0}
  echo "$newest:$bytes"
}

watchdog(){
  local prev="" last now
  last=$(date -u +%s)
  while true; do
    sleep 60
    now=$(date -u +%s)
    if [ "$(progress_fp)" != "$prev" ] || [ "$(gpu_busy)" = "1" ]; then
      prev=$(progress_fp); last=$now
    elif [ $((now - last)) -ge "$STALL_LIMIT" ]; then
      note "WATCHDOG: no progress for ${STALL_LIMIT}s (logs idle, download flat, GPUs idle) -> powering off"
      printf 'STALLED: no progress for %ss, powered off %s\nLast state was in %s\n' \
        "$STALL_LIMIT" "$(ts)" "$RES/STATE.md" > "$RES/STALLED.md"
      poweroff_pod
      return 0
    fi
  done
}

collect(){  # family full out
  local family=$1 full=$2 out=$3
  mkdir -p "$out/agent-output"
  cp -f "/root/demos/opencode-$full.gif" "$out/" 2>/dev/null
  cp -f "/root/demos/opencode-$full.mp4" "$out/" 2>/dev/null
  cp -f "/root/demos/opencode-$full.png" "$out/" 2>/dev/null
  cp -f "/root/audit-$family.png"        "$out/" 2>/dev/null
  cp -f "/root/SUMMARY-$family.txt"       "$out/" 2>/dev/null
  cp -f "/root/run-$family.log"           "$out/" 2>/dev/null
  cp -f /tmp/lilbee-serve.log             "$out/lilbee-serve.log" 2>/dev/null
  cp -f /tmp/giant-srv.log                "$out/giant-srv.log" 2>/dev/null
  cp -f /root/demo-proj/*.py              "$out/agent-output/" 2>/dev/null
}

frames_of(){  # gif -> integer frame count (0 if missing/non-numeric)
  local gif=$1 f=0
  [ -f "$gif" ] && f=$(ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of csv=p=0 "$gif" 2>/dev/null)
  case "$f" in (''|*[!0-9]*) f=0 ;; esac
  echo "$f"
}

write_status(){  # family full out -> writes STATUS; echoes frame count
  local family=$1 full=$2 out=$3
  local frames; frames=$(frames_of "$out/opencode-$full.gif")
  local grounded=NO
  grep -rqiE 'response_parser|providers/(worker|llama_cpp|families)|chat_completions_api|retrieval/|[a-z_]+\.py:?L?[0-9]' \
    "$out/agent-output" "$out/SUMMARY-$family.txt" 2>/dev/null && grounded=YES
  printf 'family=%s gif_frames=%s grounded=%s\n' "$family" "$frames" "$grounded" > "$out/STATUS"
  echo "$frames"
}

note "===== MATRIX START (pod=${POD_ID:-?}; models: qwen3-coder, minimax-m2) ====="
# Clean slate, but KEEP a warm lilbeeserve if one is up (re-warm is cheap but
# skipping it is cheaper). model_demo.sh relaunches the per-giant server itself.
# Use -x (exact process name) NOT -f: this script's path contains "opencode", so
# pkill -f opencode would kill the matrix runner itself.
tmux kill-session -t giantsrv 2>/dev/null || true
pkill -x vhs 2>/dev/null || true
pkill -x ttyd 2>/dev/null || true
pkill -x opencode 2>/dev/null || true

# Start the stall watchdog (powers off if everything goes idle for too long).
watchdog & WD_PID=$!

CANARY_OK=1
for i in "${!MODELS[@]}"; do
  IFS='|' read -r family full repo quant qdir mg <<< "${MODELS[$i]}"
  out="$RES/$family"
  if [ -f "$out/done" ]; then note "SKIP $family (already done)"; continue; fi
  if [ "$i" -gt 0 ] && [ "$CANARY_OK" != "1" ]; then
    note "SKIP $family (canary gif failed; not spending giant GPU hours)"; continue
  fi
  mkdir -p "$out"
  note "RUN $family ($repo $quant mg=$mg)"
  printf 'RUNNING %s since %s\n' "$family" "$(ts)" > "$RES/STATE.md"
  FAMILY="$family" FULL="$full" REPO="$repo" QUANT="$quant" QDIR="$qdir" MULTIGPU="$mg" \
    "$DRIVER" >> "$out/driver.log" 2>&1
  rc=$?
  collect "$family" "$full" "$out"
  frames=$(write_status "$family" "$full" "$out")
  touch "$out/done"
  note "RESULT $family rc=$rc gif_frames=$frames ($(cat "$out/STATUS" 2>/dev/null))"
  # Free the weights before the next model: two 240GB giants will not co-exist on
  # a 400GB volume. qwen is small but cleaning it keeps headroom for the giant.
  rm -rf "$qdir" 2>/dev/null && note "freed $qdir"
  # Canary gate is the PIPELINE mechanic (did a real gif render?), NOT model
  # grounding quality -- a weaker canary missing a citation says nothing about the
  # giant. Grounding is recorded per-model for human review, not gated on here.
  if [ "$family" = "qwen3-coder" ]; then
    if [ "$frames" -gt "$MIN_FRAMES" ]; then CANARY_OK=1; note "CANARY PASS (gif_frames=$frames)";
    else CANARY_OK=0; note "CANARY FAIL (gif_frames=$frames <= $MIN_FRAMES) -> giants skipped"; fi
  fi
done

{
  echo "# Matrix finished $(ts)"
  for s in "$RES"/*/STATUS; do [ -f "$s" ] && echo "- $(cat "$s")"; done
  [ "$CANARY_OK" = "1" ] || echo "CANARY FAILED: the gif pipeline is broken; debug from /workspace/results/qwen3-coder (driver.log, lilbee-serve.log, the gif)."
  echo "Pending (need HF-ref verification with operator reachable): Qwen3-235B, GLM-4.6, gpt-oss-120B."
  echo "Pod powering off now; bring it back up and re-launch run_matrix.sh to resume."
} > "$RES/STATE.md"
note "===== MATRIX DONE ====="
kill "$WD_PID" 2>/dev/null || true   # stop the watchdog; we power off cleanly below
poweroff_pod
