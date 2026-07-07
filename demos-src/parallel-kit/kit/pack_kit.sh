#!/usr/bin/env bash
# Assemble the flat runtime kit consumed by job.sh/qualgate ($KIT layout:
# scripts + reels.yaml at the root, tapes/ and probes/ subdirs) from the v2
# working tree. Gzip (not zstd): scatter extracts the kit on a STOCK pod
# before bootstrap installs zstd. Asserts completeness — a missing script
# here means every pod fails after billing boot.
# Usage: pack_kit.sh <v2-dir> <kit-out-dir> [golden-dir]
set -euo pipefail
V2="$1"; OUT="$2"; GOLDEN="${3:-}"
rm -rf "$OUT" && mkdir -p "$OUT/tapes" "$OUT/probes"
cp "$V2"/kit/*.py "$V2"/kit/*.sh "$OUT/"
cp "$V2"/kit/ground_truth.json "$OUT/"
cp "$V2"/kit/geometry_cal.json "$OUT/" 2>/dev/null || true
cp "$V2"/kit/calibration.json "$OUT/" 2>/dev/null || true
cp "$V2"/kit/vram_table.json "$OUT/" 2>/dev/null || true
cp "$V2"/kit/pull_refs.json "$OUT/" 2>/dev/null || true
rm -f "$OUT/pod_ledger.json"
cp "$V2"/reels.yaml "$OUT/"
cp "$V2"/tapes/generated/*.tape "$OUT/tapes/"
cp "$V2"/probes/*.tape "$OUT/probes/"

REQUIRED="bootstrap.sh qualgate.sh pretake.sh stage.py materialize.py autoqa.py check_probes.py env.sh job.sh idle_watchdog.sh canary_grade.sh reels.yaml geometry_cal.json"
# pull_refs.json is required for FAN-OUT packs (prep resolves it before final pack)
[ -f "$OUT/pull_refs.json" ] || echo "PACK_KIT_NOTE: pull_refs.json absent (ok only for the pre-resolve temp pack)"
for f in $REQUIRED; do
  [ -f "$OUT/$f" ] || { echo "PACK_KIT_FAIL: missing $f"; exit 1; }
done
NTAPES=$(ls "$OUT"/tapes/*.tape | wc -l | tr -d ' ')
NREELS=$(python3 -c "import yaml; print(len(yaml.safe_load(open('$V2/reels.yaml'))['reels']))")
[ "$NTAPES" = "$NREELS" ] || { echo "PACK_KIT_FAIL: $NTAPES tapes != $NREELS reels"; exit 1; }

if [ -n "$GOLDEN" ]; then
  ( cd "$(dirname "$OUT")" && tar -czf "$GOLDEN/kit.tar.gz" "$(basename "$OUT")" )
  echo "packed -> $GOLDEN/kit.tar.gz"
fi
echo "PACK_KIT_OK $OUT ($NTAPES tapes)"
