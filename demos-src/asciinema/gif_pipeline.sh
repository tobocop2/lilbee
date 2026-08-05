#!/usr/bin/env bash
# Local (Mac) gif generation from a 2x master mp4. Never runs on a pod.
# Mode is decided ONCE by the qualification A/B pixel audit and then applied
# to every reel: lanczos | sharp (lanczos + mild unsharp).
# Usage: gif_pipeline.sh <master.mp4> <out.gif> <display-width> <display-height> [mode]
set -euo pipefail
SRC="$1"; OUT="$2"; W="$3"; H="$4"; MODE="${5:-lanczos}"; DITHER="${6:-bayer:bayer_scale=4}"   # bayer:bayer_scale=4 | sierra2_4a | none
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

case "$MODE" in
  lanczos) VF="scale=${W}:${H}:flags=lanczos" ;;
  sharp)   VF="scale=${W}:${H}:flags=lanczos,unsharp=5:5:0.4:5:5:0.0" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

# hold the final frame ~2.5s so viewers can read the settled answer
HOLD="tpad=stop_mode=clone:stop_duration=2.5"
ffmpeg -v quiet -y -i "$SRC" -vf "$VF,$HOLD,fps=25,palettegen=stats_mode=diff" "$TMP/pal.png"
ffmpeg -v quiet -y -i "$SRC" -i "$TMP/pal.png" \
  -lavfi "$VF,$HOLD,fps=25 [x]; [x][1:v] paletteuse=dither=${DITHER}:diff_mode=rectangle" \
  "$TMP/raw.gif"
gifsicle -O2 --lossy=20 "$TMP/raw.gif" -o "$OUT"

SIZE=$(du -m "$OUT" | cut -f1)
echo "$OUT ${SIZE}MB $(ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$OUT")"
if [ "$SIZE" -gt 10 ]; then
  echo "WARNING: >10MB — GitHub README render cap at risk; consider trimming or --lossy=30"
fi
