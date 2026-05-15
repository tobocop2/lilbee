#!/usr/bin/env bash
# Export a small set of man pages as plain text, for the CLI demo's corpus.
# Usage: ./demos/make-manpages.sh <output-dir>

set -euo pipefail

OUT="${1:-./man-pages}"
mkdir -p "$OUT"

for p in find awk grep xargs sed; do
    man "$p" 2>/dev/null | col -bx > "$OUT/$p.txt"
done

echo "wrote $(ls "$OUT" | wc -l | tr -d ' ') man pages to $OUT"
