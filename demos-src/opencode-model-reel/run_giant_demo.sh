#!/usr/bin/env bash
# Self-contained, resumable end-to-end giant demo for the pod.
#
# Finishes the model download (resumable, retried through connection drops), serves
# the giant multi-GPU, indexes lilbee's source, records the live opencode TUI on a
# grounding-forcing task, then audits the result into /root/SUMMARY.txt +
# /root/audit_montage.png. Everything logs to /root/run.log.
#
# Designed to survive an unstable SSH connection: run it DETACHED and walk away.
#   cd /root/reel && git pull -q origin demo-reel/opencode-model-matrix
#   tmux new-session -d -s work "/root/reel/demos-src/opencode-model-reel/run_giant_demo.sh"
# Then reconnect any time and read /root/SUMMARY.txt and /root/run.log.
#
# HF auth: reads /root/.cache/huggingface/token if present (set once via
# `printf %s <token> > /root/.cache/huggingface/token`); falls back to
# unauthenticated + retries if absent. The token is NEVER hard-coded here.
set -uo pipefail
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export LILBEE_MODELS_DIR=/root/models
LOG=/root/run.log; SUMMARY=/root/SUMMARY.txt
ts(){ date -u +%H:%M:%S; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

REPO=unsloth/MiniMax-M2-GGUF
QDIR=/root/models/minimax-q8
QUANT='Q8_0/*'
FAMILY=minimax-m2
FULL=MiniMax-M2
STREAM_SLEEP=150          # giants generate slowly; fast-forward the slow spans in post
                          # (>~200s stresses VHS's headless-chromium capture -> empty gif;
                          #  the 128k context fix lets the giant finish inside 150s)
MULTIGPU=1               # MiniMax-M2 Q8_0 (~243GB) spans both H200s
PROMPT='Using ONLY lilbee_search (the code is not on disk here), find lilbee'\''s REAL response parser that extracts tool calls from a model'\''s raw text output. Name the actual module path, the class or function, and which model families it special-cases. Then write tool_call_example.py that mirrors lilbee'\''s real approach (not a generic OpenAI tool_calls dict reader) on a sample raw model-output string, citing the exact lilbee files as path:Lstart-Lend. If lilbee_search does not surface it, say so rather than inventing a generic parser.'

echo "STARTED $(ts)" > "$SUMMARY"
say "====== RUN_GIANT_DEMO START ($FAMILY) ======"

# --- Phase 1: finish the download (resumable; retries through drops) ---
say "Phase 1: ensure $REPO $QUANT download complete"
cd /root/lilbee
ok=0
for attempt in $(seq 1 30); do
  if .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO', allow_patterns='$QUANT', local_dir='$QDIR', max_workers=2)" >> /root/dl.log 2>&1; then
    say "download complete on attempt $attempt"; ok=1; break
  fi
  say "download attempt $attempt dropped ($(du -sh "$QDIR" 2>/dev/null | cut -f1)); resuming"
  sleep 5
done
GGUF=$(find "$QDIR" -ipath '*Q8_0*00001-of-*.gguf' 2>/dev/null | head -1)
[ -z "$GGUF" ] && GGUF=$(find "$QDIR" -ipath '*Q8_0*.gguf' ! -name '*.lock' 2>/dev/null | sort | head -1)
say "download ok=$ok GGUF=$GGUF size=$(du -sh "$QDIR" 2>/dev/null | cut -f1)"
if [ -z "$GGUF" ]; then say "FATAL: no GGUF after download"; echo "FAILED: no GGUF" >> "$SUMMARY"; exit 1; fi

# --- Phase 2: serve multi-GPU + index + record the live opencode TUI ---
say "Phase 2: build_reel (multi-GPU serve + index + record)"
rm -rf /root/demo-proj; rm -f /root/demos/_raw-$FAMILY.*
OUT_DIR=/root/demos MULTIGPU=$MULTIGPU /root/reel/demos-src/opencode-model-reel/build_reel.sh \
  "$FAMILY" "$FULL" "$GGUF" "$PROMPT" "$STREAM_SLEEP" "$MULTIGPU" >> "$LOG" 2>&1
say "build_reel exit=$?"

# --- Phase 3: audit into SUMMARY + montage ---
say "Phase 3: audit"
GIF=/root/demos/_raw-$FAMILY.gif
mkdir -p /root/frames; rm -f /root/frames/*.png
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$GIF" 2>/dev/null)
for i in 0 1 2 3 4 5 6 7; do
  t=$(awk "BEGIN{printf \"%.2f\", ${DUR:-1}*($i+0.5)/8}")
  ffmpeg -nostdin -y -loglevel error -ss "$t" -i "$GIF" -frames:v 1 /root/frames/f$i.png 2>/dev/null
done
ffmpeg -nostdin -y -loglevel error -i /root/frames/f%d.png -vf "scale=640:-1,tile=2x4" /root/audit_montage.png 2>/dev/null
{
  echo "RUN_GIANT_DEMO SUMMARY ($FULL)  $(ts)"
  echo "gguf: $GGUF"
  echo "gif: $GIF  frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$GIF" 2>/dev/null)  dur=${DUR}s"
  echo "mcp CallToolRequest count: $(grep -ac CallToolRequest /tmp/lilbee-serve.log 2>/dev/null)"
  echo "agent wrote: $(ls /root/demo-proj/*.py 2>/dev/null || echo NONE)"
  echo "--- agent output (first 90 lines) ---"
  cat /root/demo-proj/*.py 2>/dev/null | head -90
} >> "$SUMMARY" 2>&1
say "====== RUN_GIANT_DEMO DONE -> /root/SUMMARY.txt + /root/audit_montage.png ======"
