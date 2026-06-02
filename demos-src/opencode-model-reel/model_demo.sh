#!/usr/bin/env bash
# Self-contained, resumable end-to-end opencode demo for ONE model.
#
# Generalises run_giant_demo.sh (which was MiniMax-only) into a parameterised
# driver: it finishes the model download (resumable, retried through connection
# drops), serves the model, indexes lilbee's source, records the live opencode
# TUI on a grounding-forcing task, then audits the result into
# /root/SUMMARY-<family>.txt + /root/audit-<family>.png. Everything logs to
# /root/run-<family>.log.
#
# Used for both the cheap qwen3-coder-30B validation and each giant in the
# matrix; the only difference is the model spec passed in.
#
# Run it DETACHED so it survives an unstable SSH connection:
#   cd /root/reel && git pull -q origin demo-reel/opencode-model-matrix
#   FAMILY=qwen3-coder FULL=Qwen3-Coder-30B \
#     REPO=unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF QUANT='*Q4_K_M*' \
#     QDIR=/root/models/qwen3-coder MULTIGPU=0 \
#     tmux new-session -d -s work \
#       "/root/reel/demos-src/opencode-model-reel/model_demo.sh"
# Then reconnect any time and read /root/SUMMARY-<family>.txt and the log.
#
# HF auth: reads /root/.cache/huggingface/token if present. Never hard-coded.
set -uo pipefail
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
# Model weights (an 18GB+ GGUF, or a 240GB giant) MUST live on /workspace (the
# RunPod network volume) or the download fills the small root overlay and ENOSPCs;
# the volume also persists across pod restarts. The lilbee index stays on the local
# root disk (small, fast) via giant_demo.sh's default WS.
export LILBEE_MODELS_DIR="${LILBEE_MODELS_DIR:-/workspace/models}"

# --- model spec (env-driven; sensible defaults for the qwen3-coder validation) ---
FAMILY="${FAMILY:-qwen3-coder}"
FULL="${FULL:-Qwen3-Coder-30B}"
REPO="${REPO:-unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF}"
QUANT="${QUANT:-*Q4_K_M*}"
QDIR="${QDIR:-/workspace/models/$FAMILY}"
MULTIGPU="${MULTIGPU:-0}"
STREAM_SLEEP="${STREAM_SLEEP:-150}"   # giants stream slowly; fast-forward in post
                                      # (>~200s starves VHS's headless-chromium gif capture)
PROMPT="${PROMPT:-Using ONLY lilbee_search (the code is not on disk here), explain how lilbee turns an incoming chat request into a model call and how it handles tool calls. Name the REAL modules and functions for: (a) dispatching a chat request to the model, (b) parsing or normalizing tool-call arguments, and (c) translating tool calls to the OpenAI wire format. Cite each as path:Lstart-Lend. Then write lilbee_toolcall_walkthrough.py: a short annotated trace that references those exact lilbee functions (not a generic OpenAI tool_calls dict reader), with the path:Lstart-Lend citations inline. If lilbee_search does not surface a piece, say so rather than inventing it.}"

LOG=/root/run-$FAMILY.log
SUMMARY=/root/SUMMARY-$FAMILY.txt
ts(){ date -u +%H:%M:%S; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

echo "STARTED $(ts)" > "$SUMMARY"
say "====== MODEL_DEMO START ($FAMILY / $REPO $QUANT, multigpu=$MULTIGPU) ======"

# --- Phase 1: finish the download (resumable; retries through drops) ---
say "Phase 1: ensure $REPO $QUANT download complete -> $QDIR"
cd /root/lilbee
ok=0
for attempt in $(seq 1 30); do
  if .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO', allow_patterns='$QUANT', local_dir='$QDIR', max_workers=2)" >> /root/dl-$FAMILY.log 2>&1; then
    say "download complete on attempt $attempt"; ok=1; break
  fi
  say "download attempt $attempt dropped ($(du -sh "$QDIR" 2>/dev/null | cut -f1)); resuming"
  sleep 5
done
GGUF=$(find "$QDIR" -ipath '*00001-of-*.gguf' 2>/dev/null | head -1)
[ -z "$GGUF" ] && GGUF=$(find "$QDIR" -iname '*.gguf' ! -name '*.lock' 2>/dev/null | sort | head -1)
say "download ok=$ok GGUF=$GGUF size=$(du -sh "$QDIR" 2>/dev/null | cut -f1)"
if [ -z "$GGUF" ]; then say "FATAL: no GGUF after download"; echo "FAILED: no GGUF" >> "$SUMMARY"; exit 1; fi

# --- Phase 2: serve + index + record the live opencode TUI ---
say "Phase 2: build_reel (serve + index + record)"
rm -rf /root/demo-proj; rm -f /root/demos/_raw-$FAMILY.*
OUT_DIR=/root/demos MULTIGPU=$MULTIGPU /root/reel/demos-src/opencode-model-reel/build_reel.sh \
  "$FAMILY" "$FULL" "$GGUF" "$PROMPT" "$STREAM_SLEEP" "$MULTIGPU" >> "$LOG" 2>&1
say "build_reel exit=$?"

# --- Phase 3: audit into SUMMARY + montage ---
say "Phase 3: audit"
GIF=/root/demos/opencode-$FULL.gif
[ -f "$GIF" ] || GIF=/root/demos/_raw-$FAMILY.gif
mkdir -p /root/frames-$FAMILY; rm -f /root/frames-$FAMILY/*.png
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$GIF" 2>/dev/null)
for i in 0 1 2 3 4 5 6 7; do
  t=$(awk "BEGIN{printf \"%.2f\", ${DUR:-1}*($i+0.5)/8}")
  ffmpeg -nostdin -y -loglevel error -ss "$t" -i "$GIF" -frames:v 1 /root/frames-$FAMILY/f$i.png 2>/dev/null
done
ffmpeg -nostdin -y -loglevel error -i /root/frames-$FAMILY/f%d.png -vf "scale=640:-1,tile=2x4" /root/audit-$FAMILY.png 2>/dev/null
{
  echo "MODEL_DEMO SUMMARY ($FULL)  $(ts)"
  echo "gguf: $GGUF"
  echo "gif: $GIF  frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$GIF" 2>/dev/null)  dur=${DUR}s"
  echo "mcp CallToolRequest count: $(grep -ac CallToolRequest /tmp/lilbee-serve.log 2>/dev/null)"
  echo "agent wrote: $(ls /root/demo-proj/*.py 2>/dev/null || echo NONE)"
  echo "--- agent output (first 90 lines) ---"
  cat /root/demo-proj/*.py 2>/dev/null | head -90
} >> "$SUMMARY" 2>&1
say "====== MODEL_DEMO DONE -> $SUMMARY + /root/audit-$FAMILY.png ======"
