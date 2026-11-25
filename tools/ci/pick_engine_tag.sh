#!/usr/bin/env bash
# Pick the newest fork sync tag that engine-bad-tags.list does not list.
# Reads candidate tag names, one per line, on stdin. Prints the chosen tag.
# Each skipped bad tag gets a "skipping <tag>: <reason>" line on stderr.
# Exits 1 when no tag remains.
#
# Usage: pick_engine_tag.sh <denylist-file> < tags

set -euo pipefail

denylist="${1:?usage: pick_engine_tag.sh <denylist-file>}"

tags=$(grep -E '^memory-[0-9]{8}-[0-9]{6}$' | sort -ru || true)

while IFS= read -r tag; do
  [ -n "$tag" ] || continue
  reason=$(grep -E "^${tag}([[:space:]]|\$)" "$denylist" | head -n 1 || true)
  if [ -n "$reason" ]; then
    echo "skipping ${tag}: ${reason#"${tag}"}" >&2
    continue
  fi
  echo "$tag"
  exit 0
done <<< "$tags"

echo "no fork tag is left after the denylist" >&2
exit 1
