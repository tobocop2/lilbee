#!/usr/bin/env bash
# GIF-first finish pipeline (2026-07-16, Wave 0 finding). The VHS-direct gif is
# theme-true; the VHS mp4 leg dims text ~40 points (untagged h264), so the mp4
# is used ONLY to find the settle time (same timeline as the gif). Ships:
#   <name>.gif   trimmed VHS gif + 2.5s hold, gifsicle-optimized
#   <name>.mp4   site player video, encoded FROM the trimmed gif, tagged
#   <name>.png   poster: last frame of the gif
# Usage: gif_finish.sh <vhs.gif> <vhs.mp4> <outdir> <name>
set -euo pipefail
GIF="$1"; MP4="$2"; OUTDIR="$3"; NAME="$4"; HOLD="${REEL_HOLD:-2.5}"
KIT="$(cd "$(dirname "$0")" && pwd)"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# settle time from the mp4 (identical timeline, motion analysis; bottom strip and
# gif-dither jitter are excluded inside trim_tail)
SETTLE=$(python3 "$KIT/trim_tail.py" --freeze 0.0 "$MP4" "$TMP/cut.mp4" | sed -n 's/.*settles at \([0-9.]*\)s.*/\1/p')
[ -n "$SETTLE" ] || { echo "no settle time found"; exit 1; }

# Decode the VHS gif to composited PNG frames, find the content-stabilization cut,
# then re-encode at a UNIFORM 25fps via ffmpeg. PIL's gif writer MERGES identical
# consecutive frames (summing delays), which collapsed static-heavy reels from
# 25fps to ~5fps and read as choppy; ffmpeg fps=25 keeps every frame like the
# original Mac reels. Palette from stats_mode=full + bayer dither preserves the
# near-white ExtraBold text (a full-frame settled answer has plenty of white).
FRD=$(mktemp -d); SEQ=$(mktemp -d); trap 'rm -rf "$FRD" "$SEQ"' EXIT
ffmpeg -v error -i "$GIF" "$FRD/f%05d.png"
# Select the frames to ship: cut at the content-stabilization point (the settled
# END STATE), then COMPRESS long static runs. A cold model load or any dead pause
# leaves the screen frozen mid-reel for 10s+; the tail-cut can't touch a mid-reel
# gap, so compress any still run >3s down to ~2s. Short intentional pauses (a UI
# walk pausing ~2s on each pane) are under the threshold and pass through, keeping
# real-time pace. Output is a clean renumbered sequence for the 25fps encode.
python3 - "$FRD" "$SEQ" <<'PY'
import sys, glob, shutil
import numpy as np
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"{src}/f*.png"))
def gray(p):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.int16)
    return a[: int(a.shape[0] * 0.8)]      # ignore the blinking-cursor input strip
grays = [gray(p) for p in fs]
end = grays[-1]
cut = len(fs) - 1
for i in range(len(fs) - 1, -1, -1):
    if (np.abs(grays[i] - end) > 30).mean() > 0.004:
        cut = min(i + 2, len(fs) - 1)
        break
FPS = 25
MAX_STILL = int(2.0 * FPS)   # a still run keeps at most 2s
GAP_TRIG = int(3.0 * FPS)    # ...but only if it ran longer than 3s
keep, still = [], 0
for i in range(cut + 1):
    moved = i == 0 or (np.abs(grays[i] - grays[i - 1]) > 30).mean() > 0.001
    still = 0 if moved else still + 1
    # drop frames deep inside a long static run (keep the first MAX_STILL of it)
    if still and still > MAX_STILL:
        continue
    keep.append(i)
for j, i in enumerate(keep, start=1):
    shutil.copy(fs[i], f"{dst}/c{j:05d}.png")
print(f"kept {len(keep)}/{cut + 1} frames (gaps compressed)")
PY
CUT=$(ls "$SEQ"/c*.png 2>/dev/null | wc -l | tr -d ' ')
ffmpeg -v error -y -framerate 25 -i "$SEQ/c%05d.png" -vf "palettegen=stats_mode=full" "$SEQ/pal.png"
ffmpeg -v error -y -framerate 25 -i "$SEQ/c%05d.png" -i "$SEQ/pal.png" \
  -lavfi "[0:v]tpad=stop_mode=clone:stop_duration=${HOLD}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
  "$OUTDIR/$NAME.gif"
# NO gifsicle -O: its optimizer MERGES identical consecutive frames (summing
# delays), which is exactly the 25fps->5fps regression. ffmpeg output ships as-is.
echo "stabilized at frame $CUT (uniform 25fps)"

# site mp4 FROM the gif: palette-true pixels, h264 tagged correctly
ffmpeg -v error -y -i "$OUTDIR/$NAME.gif" \
  -vf "format=yuv420p" -c:v libx264 -preset slow -crf 18 \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -movflags +faststart "$OUTDIR/$NAME.mp4"

# poster from the gif's settled frame
python3 - "$OUTDIR/$NAME.gif" "$OUTDIR/$NAME.png" <<'PY'
import sys
from PIL import Image, ImageSequence
im = Image.open(sys.argv[1])
frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
frames[-1].save(sys.argv[2])
PY

SIZE=$(du -m "$OUTDIR/$NAME.gif" | cut -f1)
echo "$OUTDIR/$NAME.{gif,mp4,png}  gif=${SIZE}MB settle=${SETTLE}s"
[ "$SIZE" -gt 10 ] && echo "WARNING: gif >10MB (GitHub README cap)"
exit 0
