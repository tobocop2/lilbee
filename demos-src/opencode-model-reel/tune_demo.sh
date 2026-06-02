#!/usr/bin/env bash
# Record ONE fine-tuning reel: lilbee tuning retrieval to a chat model on the
# shared corpus. Also THE precompute step -- it persists the model's tuned knobs
# to /workspace/results/<family>/query-knobs.toml, which the opencode answer demo
# later merges into config.toml so it boots already-tuned.
#
# Needs ingest_corpus.sh to have built + verified the corpus first. Tuning loads
# no chat model (embed + rerank + scoring only), so every model's reel is
# recordable now, before the giant-serving question is solved.
#
# Usage: tune_demo.sh <family> <full-name> [n_ctx]
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.opencode/bin:/usr/local/bin:$PATH

FAMILY="$1"; FULL="$2"; N_CTX="${3:-131072}"
PACE="${PACE:-1.3}"

HERE="$(cd "$(dirname "$0")" && pwd)"
LM="${LM:-/root/lilbee}"
export LILBEE_DATA="${LILBEE_DATA:-/workspace/.lilbee}"
export LILBEE_MODELS_DIR="${LILBEE_MODELS_DIR:-/workspace/models}"
LILBEE="${LILBEE:-$LM/.venv/bin/lilbee}"
PORT="${PORT:-8080}"
OUT_DIR="${OUT_DIR:-/root/demos}"
RES="/workspace/results/$FAMILY"
DATA="$LILBEE_DATA/data"
mkdir -p "$OUT_DIR" "$RES"

say(){ echo "[tune $(date -u +%H:%M:%S)] $*"; }

# 1) Ensure serve is up and read the token (the index must already exist).
if ! curl -fsS -m2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  say "starting lilbee serve on :$PORT"
  tmux kill-session -t lilbeeserve 2>/dev/null || true
  tmux new-session -d -s lilbeeserve \
    "cd $LM && LILBEE_DATA=$LILBEE_DATA LILBEE_MODELS_DIR=$LILBEE_MODELS_DIR $LILBEE serve --port $PORT > /tmp/lilbee-serve.log 2>&1"
  for _ in $(seq 1 60); do curl -fsS -m2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 && break; sleep 2; done
fi
curl -fsS -m2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 \
  || { say "FATAL serve not healthy (see /tmp/lilbee-serve.log)"; exit 3; }
TOKEN=$(python3 -c "import json;print(json.load(open('$DATA/server.json'))['token'])")

# 2) Confirm the corpus is actually indexed; otherwise ingest_corpus.sh wasn't run.
if ! curl -fsS -m5 -H "Authorization: Bearer $TOKEN" \
     "http://127.0.0.1:$PORT/api/search?q=dispatch&top_k=1&chunk_type=raw" 2>/dev/null \
     | grep -qiE '"source"|\.py'; then
  say "FATAL corpus search returns nothing -- run ingest_corpus.sh first"; exit 5
fi

OUT="$RES/query-knobs.toml"
TOKFILE="/tmp/tune-tok-$FAMILY"; umask 077; printf '%s' "$TOKEN" > "$TOKFILE"

# 3) Precompute (authoritative): run the tuner once with no pacing to persist the
#    tuned knobs and to MEASURE the search wall-time, so the recording's wait is
#    sized to the real run instead of a guess.
say "precomputing tuned knobs for $FULL (n_ctx=$N_CTX) -> $OUT"
T0=$SECONDS
python3 "$HERE/tune_run.py" --model "$FULL" --n-ctx "$N_CTX" --probes "$HERE/probes.toml" \
  --out "$OUT" --base-url "http://127.0.0.1:$PORT" --token-file "$TOKFILE" --pace 0 \
  > "$RES/tune-precompute.log" 2>&1
PRC=$?
T_SEARCH=$((SECONDS - T0))
[ -f "$OUT" ] || { say "FATAL tuner did not write $OUT (see tune-precompute.log)"; exit 6; }
say "precompute rc=$PRC search_time=${T_SEARCH}s; artifact saved"

# 4) Size the reel: the recording re-runs the searches (~T_SEARCH) plus the narration
#    pacing, with a buffer so a slow pass never gets cut off mid-climb.
PACING=$(awk "BEGIN{printf \"%d\", 1.6 + 6*$PACE + 4}")
WAIT=$((T_SEARCH + PACING + 4))
say "reel wait=${WAIT}s (search ${T_SEARCH}s + pacing ${PACING}s)"

# 5) Launcher script keeps the token off the recorded command line.
LAUNCH="/tmp/tune-launch-$FAMILY.sh"
cat > "$LAUNCH" <<SH
#!/usr/bin/env bash
exec python3 "$HERE/tune_run.py" --model "$FULL" --n-ctx $N_CTX \\
  --probes "$HERE/probes.toml" --out "$OUT" \\
  --base-url "http://127.0.0.1:$PORT" --token-file "$TOKFILE" --pace $PACE
SH
chmod +x "$LAUNCH"

# 6) Render the reel.
RAW="_tune-$FAMILY"
STEM="$OUT_DIR/$RAW"
sed -e "s#__OUT__#$RAW#g" \
    -e "s#__CMD__#bash $LAUNCH#g" \
    -e "s#__WAIT__#${WAIT}s#g" \
    "$HERE/tune_demo.tape.tmpl" > "/tmp/tune-tape-$FAMILY.tape"
( cd "$OUT_DIR" && VHS_NO_SANDBOX=true vhs "/tmp/tune-tape-$FAMILY.tape" )

# 7) Transcode webm -> mp4 + high-quality gif (same recipe as the opencode reel).
FINAL_MP4="$OUT_DIR/tune-$FULL.mp4"
ffmpeg -nostdin -y -loglevel error -i "$STEM.webm" -c:v libx264 -pix_fmt yuv420p -r 60 \
  -vf scale=1600:1000 -movflags +faststart "$FINAL_MP4"
PAL="/tmp/tune-pal-$FAMILY.png"
ffmpeg -nostdin -y -loglevel error -i "$FINAL_MP4" -vf "fps=20,scale=1600:-1:flags=lanczos,palettegen" "$PAL"
ffmpeg -nostdin -y -loglevel error -i "$FINAL_MP4" -i "$PAL" \
  -lavfi "fps=20,scale=1600:-1:flags=lanczos[x];[x][1:v]paletteuse" "$OUT_DIR/tune-$FULL.gif"
cp -f "$STEM.png" "$OUT_DIR/tune-$FULL.png" 2>/dev/null || true

# 8) Timeline review gate + collect durable artifacts.
echo "[review] tune-$FULL"
python3 "$HERE/review_demo.py" "$FINAL_MP4" --frames 8 || \
  echo "[review] tune-$FULL flagged DEAD-SCREEN RISK -- inspect frames before publishing" >&2
cp -f "$OUT_DIR/tune-$FULL.gif" "$OUT_DIR/tune-$FULL.mp4" "$OUT_DIR/tune-$FULL.png" "$RES/" 2>/dev/null || true
say "DONE tune-$FULL -> $OUT_DIR + artifacts in $RES"
