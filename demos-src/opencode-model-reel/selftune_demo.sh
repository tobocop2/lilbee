#!/usr/bin/env bash
# Set 2 reel: the chat model tunes lilbee's retrieval ITSELF, recorded live.
#
# Real product path only: an EMPTY project + `lilbee launch opencode` (provider +
# MCP + shipped skill + warm serve). Retrieval starts deliberately NARROW so the
# model's first lilbee_search is genuinely thin and it has a real reason to widen
# via lilbee_settings_set (SKILL.md #5), search again, and answer with citations.
#
# Local Mac (Metal via the homebrew llama-server the fleet finds on PATH). Built
# to verify the HARNESS, not a pod. Produces gif + mp4 + webm + a capture of the
# knobs the MODEL chose (proof it tuned itself).
#
# Usage: selftune_demo.sh <chat_native_ref> <full_display_name> [boot_s] [think_s]
set -uo pipefail

CHAT_REF="$1"; FULL="$2"; BOOT="${3:-7}"; THINK="${4:-90}"
# Local Metal inference is slow, so we record the FULL turn (large THINK) and then
# fast-forward the body in post by SPEED, the way the giant pipeline compresses
# slow spans. SPEED=1 leaves it real-time (use on a fast GPU pod).
SPEED="${SPEED:-1}"
MODEL_PICK="lilbee/$CHAT_REF"

HERE="$(cd "$(dirname "$0")" && pwd)"
LM="${LM:-/Users/tobias/projects/lilbee-local-model-api}"
LILBEE="${LILBEE:-$LM/.venv/bin/lilbee}"
PORT="${PORT:-8080}"

ROOT="${ROOT:-$HOME/lilbee-selftune-demo}"
export LILBEE_DATA="$ROOT/.lilbee"          # local data dir (models stay global)
DATA="$LILBEE_DATA/data"
PROJ="$ROOT/scratch"                          # empty project: forces lilbee_search
OUT_DIR="${OUT_DIR:-$ROOT/out}"               # artifacts live OUTSIDE the repo
SERVE_TMUX="lilbeeserve-selftune"
EMBED_REF="nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
RERANK_REF="gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q4_K_M.gguf"
OPENCODE_CFG="$HOME/.config/opencode/opencode.json"

say(){ echo "[selftune $(date -u +%H:%M:%S)] $*"; }
mkdir -p "$LILBEE_DATA" "$PROJ" "$OUT_DIR"

# 1) Config: roles + a deliberately NARROW retrieval start so self-tuning matters.
#    num_ctx kept modest so a Q8 30B + KV cache fits 32 GB unified memory.
cat > "$LILBEE_DATA/config.toml" <<TOML
chat_model = "$CHAT_REF"
embedding_model = "$EMBED_REF"
reranker_model = "$RERANK_REF"
num_ctx = 16384
chunk_size = 320
concept_graph = true
top_k = 3
rerank_candidates = 2
diversity_max_per_source = 1
max_distance = 0.6
TOML
say "wrote config (narrow start) -> $LILBEE_DATA/config.toml"

# 2) Ensure the chat model is installed (embed + rerank already are).
if ! "$LILBEE" --json model list 2>/dev/null | grep -q "$CHAT_REF"; then
  say "chat model not installed: $CHAT_REF -- pull it first (see notes); aborting"
  exit 2
fi

# 3) Ingest the corpus subset that grounds the probes (server + providers).
if ! curl -fsS -m2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then :; fi
if [ ! -f "$LILBEE_DATA/.selftune_indexed" ]; then
  say "ingesting corpus subset (src/lilbee/server + providers)"
  LILBEE_DATA="$LILBEE_DATA" "$LILBEE" add "$LM/src/lilbee/server" "$LM/src/lilbee/providers"
  touch "$LILBEE_DATA/.selftune_indexed"
fi

# 4) Start serve so `lilbee launch opencode` reuses it AND we can read the
#    model-chosen config afterwards. Inline env (tmux server env trap).
tmux kill-session -t "$SERVE_TMUX" 2>/dev/null || true
if ! curl -fsS -m2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  say "starting lilbee serve on :$PORT (tmux $SERVE_TMUX)"
  tmux new-session -d -s "$SERVE_TMUX" \
    "cd $LM && LILBEE_DATA=$LILBEE_DATA $LILBEE serve --port $PORT > /tmp/selftune-serve.log 2>&1"
  for _ in $(seq 1 60); do curl -fsS -m2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 && break; sleep 2; done
fi
curl -fsS -m2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 \
  || { say "FATAL serve not healthy (see /tmp/selftune-serve.log)"; exit 3; }
TOKEN=$(python3 -c "import json;print(json.load(open('$DATA/server.json'))['token'])")

# 5) Warm embed + chat so the reel doesn't sit on a cold model.
say "warming embed + chat"
curl -fsS -m20 -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:$PORT/api/search?q=dispatch%20chat%20request&top_k=1&chunk_type=raw" >/dev/null 2>&1 || true
curl -fsS -m120 -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -d "{\"model\":\"$CHAT_REF\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" \
  >/tmp/selftune-warm.json 2>&1 || say "warn: chat warm call failed (see /tmp/selftune-warm.json)"

# 6) Pin the model + skip the first-run setup plan, so the reel is clean. The
#    setup marker makes `lilbee launch opencode` skip its plan print; the model
#    key in opencode.json selects qwen3-coder. Both are restored afterwards.
MARKER="$DATA/launchers/opencode-setup.json"
mkdir -p "$(dirname "$MARKER")"; printf '{"accepted":true}' > "$MARKER"
mkdir -p "$(dirname "$OPENCODE_CFG")"
BACKUP=""
if [ -f "$OPENCODE_CFG" ]; then BACKUP="$OPENCODE_CFG.selftune-bak"; cp -f "$OPENCODE_CFG" "$BACKUP"; fi
python3 - "$OPENCODE_CFG" "$MODEL_PICK" <<'PY'
import json, sys
from pathlib import Path
path, model = Path(sys.argv[1]), sys.argv[2]
cfg = {}
if path.exists():
    try: cfg = json.loads(path.read_text())
    except Exception: cfg = {}
if not isinstance(cfg, dict): cfg = {}
cfg.setdefault("$schema", "https://opencode.ai/config.json")
cfg["model"] = model
path.write_text(json.dumps(cfg, indent=2))
print("set opencode default model ->", model)
PY

restore_opencode(){
  if [ -n "$BACKUP" ]; then mv -f "$BACKUP" "$OPENCODE_CFG"; else rm -f "$OPENCODE_CFG"; fi
}
trap restore_opencode EXIT

# 7) Build the prompt (natural dev task that invites self-tuning) and render the tape.
PROMPT="I'm new to lilbee. Map the FULL path an incoming chat request takes: request dispatch, tool-call argument parsing, OpenAI wire-format translation, fleet routing to a backend, and the canonical request and response types. Cite each real file as path:line. Use the lilbee tools. If a search does not surface every piece, widen retrieval with lilbee_settings_set, then search again before you answer."
RAW="_selftune-$FULL"
STEM="$OUT_DIR/$RAW"
sed -e "s#__OUT__#$RAW#g" \
    -e "s#__PROJ__#$PROJ#g" \
    -e "s#__LMBIN__#$LM/.venv/bin#g" \
    -e "s#__PROMPT__#$PROMPT#g" \
    -e "s#__BOOT__#${BOOT}s#g" \
    -e "s#__THINK__#${THINK}s#g" \
    "$HERE/selftune.tape.tmpl" > "/tmp/selftune-$FULL.tape"
say "recording reel (boot=${BOOT}s think=${THINK}s)"
( cd "$OUT_DIR" && VHS_NO_SANDBOX=true vhs "/tmp/selftune-$FULL.tape" )
# VHS can return before the webm is fully flushed; wait until it stops growing so
# the transcode never reads a truncated file (the 48-byte mp4 failure mode).
for _ in $(seq 1 20); do
  sz1=$(stat -f %z "$STEM.webm" 2>/dev/null || echo 0); sleep 1
  sz2=$(stat -f %z "$STEM.webm" 2>/dev/null || echo 0)
  [ "$sz1" = "$sz2" ] && [ "$sz1" -gt 10000 ] && break
done

# 8) Capture the knobs the MODEL chose (proof it tuned itself), vs the narrow start.
curl -fsS -m5 -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:$PORT/api/config" \
  > "$OUT_DIR/selftune-$FULL-chosen-config.json" 2>/dev/null || true
say "saved model-chosen config -> $OUT_DIR/selftune-$FULL-chosen-config.json"

# 9) Transcode -> mp4 + gif.
FINAL_MP4="$OUT_DIR/selftune-$FULL.mp4"
ffmpeg -nostdin -y -loglevel error -i "$STEM.webm" -c:v libx264 -pix_fmt yuv420p -r 60 \
  -vf "setpts=PTS/${SPEED},scale=1600:1000" -an -movflags +faststart "$FINAL_MP4"
say "transcoded (speed=${SPEED}x) -> $FINAL_MP4"
PAL="/tmp/selftune-pal-$FULL.png"
ffmpeg -nostdin -y -loglevel error -i "$FINAL_MP4" -vf "fps=20,scale=1600:-1:flags=lanczos,palettegen" "$PAL"
ffmpeg -nostdin -y -loglevel error -i "$FINAL_MP4" -i "$PAL" \
  -lavfi "fps=20,scale=1600:-1:flags=lanczos[x];[x][1:v]paletteuse" "$OUT_DIR/selftune-$FULL.gif"
cp -f "$STEM.png" "$OUT_DIR/selftune-$FULL.png" 2>/dev/null || true

# 10) Timeline review gate.
echo "[review] selftune-$FULL"
python3 "$HERE/review_demo.py" "$FINAL_MP4" --frames 8 || \
  echo "[review] selftune-$FULL flagged DEAD-SCREEN RISK -- inspect frames" >&2
say "DONE -> $OUT_DIR/selftune-$FULL.{gif,mp4,webm,png} + chosen-config.json"
