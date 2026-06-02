#!/usr/bin/env bash
# Build one warm-first opencode demo end to end:
#   1. giant_demo.sh        -> warm the model, MEASURE the cold start, write sidecar
#   2. vhs <tape>           -> record opencode (warm, streaming, clean session name)
#   3. cold-start intro card -> a short labeled card built from the MEASURED time,
#                               concatenated in front (the honest "fast-forwarded
#                               cold start"); never real-time dead air
#   4. review_demo.py       -> timeline review gate (prompt + answer + quality)
#
# Runs on the pod (needs the GPUs + ffmpeg + vhs). The final demos/opencode-<full>.{gif,mp4}
# are produced under OUT_DIR for transfer to the reel.
#
# Usage: build_reel.sh <family> <full-name> <gguf_path> <prompt> [stream_sleep_s] [multigpu]
set -euo pipefail

FAMILY="$1"; FULL="$2"; GGUF="$3"; PROMPT="$4"
STREAM_SLEEP="${5:-45}"          # seconds to let the warm answer stream on screen
export MULTIGPU="${6:-0}"

HERE="$(cd "$(dirname "$0")" && pwd)"
WS=/root/demo-ws
OUT_DIR="${OUT_DIR:-/root/demos}"
INTRO_SECONDS="${INTRO_SECONDS:-2.6}"
mkdir -p "$OUT_DIR"

# Pick a monospace font for the intro card; fail loudly if none (don't guess).
FONT=""
for f in /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf \
         /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf; do
  [ -f "$f" ] && { FONT="$f"; break; }
done
[ -n "$FONT" ] || { echo "no usable font for intro card" >&2; exit 4; }

# 1) warm + measure (writes $WS/coldstart-$FAMILY.txt)
"$HERE/giant_demo.sh" "$FAMILY" "$GGUF"

# 2) render the streaming tape
# VHS's Output path MUST be relative (an absolute path trips its parser), so the
# tape gets a bare basename and VHS runs from OUT_DIR; STEM stays absolute for the
# ffmpeg steps below. VHS_NO_SANDBOX is required for headless chromium on the pod.
RAW="_raw-$FAMILY"
STEM="$OUT_DIR/$RAW"
sed -e "s#__OUT__#$RAW#g" \
    -e "s#__PROMPT__#$PROMPT#g" \
    -e "s#__SESSION__#$FULL#g" \
    -e "s#__STREAMSLEEP__#${STREAM_SLEEP}s#g" \
    "$HERE/giant_demo.tape.tmpl" > "/tmp/tape-$FAMILY.tape"
( cd "$OUT_DIR" && VHS_NO_SANDBOX=true vhs "/tmp/tape-$FAMILY.tape" )

# 3) cold-start intro card from the MEASURED numbers
# shellcheck disable=SC1090
. "$WS/coldstart-$FAMILY.txt"   # -> model, size_gb, cold_s, devices
INTRO="/tmp/intro-$FAMILY.mp4"
LABEL="Cold start: loading ${model} (${size_gb} GB) on ${devices}"
SUB="measured ${cold_s}s to first token, fast-forwarded   then: warm + streaming"
ffmpeg -nostdin -y -loglevel error -f lavfi -i "color=c=0x232136:s=1600x1000:d=${INTRO_SECONDS}:r=60" \
  -vf "drawtext=fontfile=${FONT}:text='${LABEL}':fontcolor=0xe0def4:fontsize=34:x=(w-text_w)/2:y=420,\
drawtext=fontfile=${FONT}:text='${SUB}':fontcolor=0x908caa:fontsize=22:x=(w-text_w)/2:y=480" \
  -c:v libx264 -pix_fmt yuv420p -r 60 "$INTRO"

# 4) concat intro + demo into the final mp4, then derive a high-quality gif
DEMO_MP4="/tmp/demo-$FAMILY.mp4"
ffmpeg -nostdin -y -loglevel error -i "$STEM.webm" -c:v libx264 -pix_fmt yuv420p -r 60 -vf scale=1600:1000 "$DEMO_MP4"
FINAL_MP4="$OUT_DIR/opencode-$FULL.mp4"
printf "file '%s'\nfile '%s'\n" "$INTRO" "$DEMO_MP4" > "/tmp/concat-$FAMILY.txt"
ffmpeg -nostdin -y -loglevel error -f concat -safe 0 -i "/tmp/concat-$FAMILY.txt" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$FINAL_MP4"
PAL="/tmp/pal-$FAMILY.png"
ffmpeg -nostdin -y -loglevel error -i "$FINAL_MP4" -vf "fps=20,scale=1600:-1:flags=lanczos,palettegen" "$PAL"
ffmpeg -nostdin -y -loglevel error -i "$FINAL_MP4" -i "$PAL" -lavfi "fps=20,scale=1600:-1:flags=lanczos[x];[x][1:v]paletteuse" "$OUT_DIR/opencode-$FULL.gif"
cp -f "$STEM.png" "$OUT_DIR/opencode-$FULL.png"

# 5) timeline review gate
echo "[review] $FULL"
python3 "$HERE/review_demo.py" "$FINAL_MP4" --frames 8 || {
  echo "[review] $FULL flagged DEAD-SCREEN RISK -- inspect frames before publishing" >&2
}
echo "DONE opencode-$FULL  (cold_s=${cold_s}, size=${size_gb}GB)"
