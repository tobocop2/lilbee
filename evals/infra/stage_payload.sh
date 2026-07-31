#!/usr/bin/env bash
# Stage what a pod run needs that is not on GitHub: the lilbee wheel under test,
# plus the merge and compare tools that live on feat/partitioned-ingest rather
# than main. Nothing here has to be pushed for a pod to measure it.
#
# Usage: stage_payload.sh <out_dir>   then   PAYLOAD=<out_dir> drive_pod.py ...
set -euo pipefail
OUT="${1:?usage: stage_payload.sh <out_dir>}"
REPO="${REPO:-$HOME/projects/lilbee}"
HERE="$(cd "$(dirname "$0")" && pwd)"

rm -rf "$OUT"; mkdir -p "$OUT"
( cd "$REPO" && uv build --wheel -o "$OUT" >/dev/null )
rm -f "$OUT/.gitignore"
for f in merge_shards.py shard_manifest.py; do
  git -C "$REPO" show "origin/feat/partitioned-ingest:evals/infra/$f" > "$OUT/$f"
done
cp "$HERE/compare_index.py" "$OUT/"
echo "staged from $(git -C "$REPO" rev-parse --short HEAD) on $(git -C "$REPO" rev-parse --abbrev-ref HEAD):"
ls -1 "$OUT"
