#!/usr/bin/env bash
# Re-encode a 1x-recorded GIF with per-section playback speeds via
# ffmpeg's split/trim/setpts/concat trick. VHS itself folds the final
# `Set PlaybackSpeed` into one global `setpts=PTS/n` filter applied to
# the whole recording, so per-section speed-ups have to happen here.
#
# Usage:
#   _postprocess_speed.sh INPUT.gif OUTPUT.gif SEGSPEC [SEGSPEC ...]
#
# SEGSPEC is "start_seconds-end_seconds@speed", e.g. "0-25@1.5" plays
# the first 25 seconds at 1.5x, "25-325@8" plays the next 5 minutes at
# 8x, etc. Segments must cover the full duration in order.
#
# Example (mcp-godot kickoff at 1.5x, indexing at 8x, answer at 1.5x):
#   _postprocess_speed.sh demos/_out/mcp-godot.gif \
#                         demos/_out/mcp-godot.gif \
#                         0-25@1.5 25-325@8 325-415@1.5

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 INPUT.gif OUTPUT.gif SEGSPEC [SEGSPEC ...]" >&2
    echo "  SEGSPEC: start_seconds-end_seconds@speed   (e.g. 0-25@1.5)" >&2
    exit 2
fi

INPUT="$1"
OUTPUT="$2"
shift 2

# Build the ffmpeg filter_complex: one trim+setpts per segment, then concat,
# then palettegen + paletteuse so the GIF is properly indexed (otherwise
# the output is 100x larger than necessary).
filter=""
labels=""
n=0
for spec in "$@"; do
    if [[ ! "$spec" =~ ^([0-9.]+)-([0-9.]+)@([0-9.]+)$ ]]; then
        echo "bad segment spec: $spec" >&2
        exit 2
    fi
    start="${BASH_REMATCH[1]}"
    end="${BASH_REMATCH[2]}"
    speed="${BASH_REMATCH[3]}"
    filter+="[0:v]trim=start=${start}:end=${end},setpts=(PTS-STARTPTS)/${speed}[s${n}];"
    labels+="[s${n}]"
    n=$((n + 1))
done
filter+="${labels}concat=n=${n}:v=1:a=0[concat];"
filter+="[concat]split[plt_a][plt_b];"
filter+="[plt_a]palettegen=max_colors=256[plt];"
filter+="[plt_b][plt]paletteuse[v]"

TMP=$(mktemp -t lilbee_pp_XXXXXX).gif

ffmpeg -y -loglevel warning -i "$INPUT" \
    -filter_complex "${filter}" \
    -map "[v]" \
    -f gif "$TMP"

mv "$TMP" "$OUTPUT"
echo "wrote $OUTPUT"
