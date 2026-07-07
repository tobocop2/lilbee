#!/usr/bin/env bash
# Render-qualification gate. Hard-blocks recording on any failure.
# Runs on: local Docker (dev loop), CPU prep pod (authoritative freeze),
# and EVERY GPU pod before its first take.
set -euo pipefail
KIT="${KIT:-/workspace/kit}"
WORK="${QUALGATE_WORK:-/root/qualgate}"
mkdir -p "$WORK" && cd "$WORK"

fail() { echo "QUALGATE_FAIL: $*" >&2; exit 1; }

# 1. Pinned tool versions
vhs --version 2>/dev/null | grep -q "0\.10\.0" || fail "vhs != 0.10.0 ($(vhs --version 2>&1))"
ttyd --version 2>/dev/null | grep -q "1\.7\.7" || fail "ttyd != 1.7.7"
google-chrome --version >/dev/null 2>&1 || fail "google-chrome missing"
command -v ffmpeg >/dev/null || fail "ffmpeg missing"
command -v tesseract >/dev/null || fail "tesseract missing"

# 2. Font stack: every family + weight the tapes rely on must resolve exactly
for spec in "JetBrains Mono" "JetBrains Mono:bold" "JetBrains Mono:italic"; do
  m=$(fc-match "$spec" family 2>/dev/null)
  [ "$m" = "JetBrains Mono" ] || fail "fc-match '$spec' -> '$m'"
done
fc-match "Symbols Nerd Font Mono" family | grep -q "Symbols Nerd Font Mono" || fail "Symbols Nerd Font Mono not installed"
fc-match "Noto Color Emoji" family | grep -q "Noto Color Emoji" || fail "Noto Color Emoji not installed"

# 3. Glyph/palette probe render + pixel checks
cp "$KIT/probes/probe-glyphs.tape" .
vhs probe-glyphs.tape >/dev/null 2>&1 || fail "probe-glyphs render failed"
[ -s probe-glyphs.png ] || fail "probe-glyphs.png missing"
CAL_ARGS=""
[ -f "$KIT/calibration.json" ] && CAL_ARGS="--calibration $KIT/calibration.json"
python3 "$KIT/check_probes.py" probe-glyphs.png --report probe-report.json $CAL_ARGS || fail "pixel checks failed (see $WORK/probe-report.json)"

# 4. Geometry. Full mode (prep pod): render 1x AND 2x per class, assert
# equality, freeze into geometry_cal.json. Light mode (job pods,
# QUALGATE_LIGHT=1): render only the 2x probe per class and assert stty
# matches the frozen calibration — one render per class, no re-derivation.
rm -f /tmp/geo-*.txt   # stale files from a prior run must never satisfy the check
python3 "$KIT/gen_geometry_probes.py" "$KIT/reels.yaml" geo/
if [ -n "${QUALGATE_LIGHT:-}" ]; then
  [ -f "$KIT/geometry_cal.json" ] || fail "light mode needs geometry_cal.json"
  for t in geo/geo-*-2x.tape; do
    cls=$(basename "$t" | sed 's/geo-\(.*\)-2x.tape/\1/')
    if [ -n "${QUALGATE_CLASSES:-}" ]; then
      case ",${QUALGATE_CLASSES}," in *",${cls},"*) ;; *) continue ;; esac
    fi
    (cd geo && vhs "geo-${cls}-2x.tape" >/dev/null 2>&1) || fail "geometry probe render failed for class $cls"
    got=$(cat "/tmp/geo-${cls}-2x.txt" 2>/dev/null)
    want=$(python3 -c "
import json
c = json.load(open('$KIT/geometry_cal.json'))['$cls']
print(f\"{c['rows']} {c['cols']}\")")
    [ -n "$got" ] && [ "$got" = "$want" ] || fail "class $cls geometry drift: got='$got' want='$want'"
    echo "GEOMETRY_OK $cls $got"
  done
else
  for t in geo/geo-*-1x.tape; do
    cls=$(basename "$t" | sed 's/geo-\(.*\)-1x.tape/\1/')
    (cd geo && vhs "geo-${cls}-1x.tape" >/dev/null 2>&1 && vhs "geo-${cls}-2x.tape" >/dev/null 2>&1) \
      || fail "geometry probe render failed for class $cls"
    a=$(cat "/tmp/geo-${cls}-1x.txt" 2>/dev/null) ; b=$(cat "/tmp/geo-${cls}-2x.txt" 2>/dev/null)
    [ -n "$a" ] && [ "$a" = "$b" ] || fail "class $cls cols/rows drift: 1x='$a' 2x='$b'"
    echo "GEOMETRY_OK $cls $a"
  done
fi

# 5. Render sanity: the probe mp4 must exist with real frames. (The specimen
# is static, so headless-Chrome screencast legitimately emits fewer frames
# than the idle sleep-sum — dropped-frame-under-load detection lives in
# autoqa on the actual ANIMATED takes, against tape-derived duration bounds.)
nf=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 probe-glyphs.mp4)
[ -n "$nf" ] && [ "$nf" -ge 25 ] || fail "probe render produced too few frames (nb_frames=$nf)"

echo "QUALGATE_PASS $(hostname) vhs=$(vhs --version 2>&1)"
