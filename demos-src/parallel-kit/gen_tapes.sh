#!/usr/bin/env bash
# Generate all 2x tapes from choreography sources + lint. Run locally and
# again on the prep pod after volume reconciliation (volume bodies win for
# reels marked choreography: volume:...).
set -euo pipefail
cd "$(dirname "$0")"
K5B=~/reels-provision/k5b
OUT=tapes/generated
mkdir -p "$OUT"

src_for() {
  # resolve choreography source; volume: paths fall back to local counterparts
  # until Phase 2 reconciliation replaces them (tracked in RECONCILE.txt)
  python3 - "$1" <<'PY'
import sys, yaml
m = yaml.safe_load(open('reels.yaml'))
print(m['reels'][sys.argv[1]].get('choreography', ''))
PY
}

: > "$OUT/RECONCILE.txt"
for reel in $(python3 -c "import yaml; print(' '.join(yaml.safe_load(open('reels.yaml'))['reels']))"); do
  cho=$(src_for "$reel")
  case "$cho" in
    local:*)    src="$K5B/${cho#local:}" ;;
    volume:*)   src="tapes/src/${reel}.body.tape"
                if [ ! -f "$src" ]; then
                  # stale local fallback until the prep pod copies the proven
                  # v8 body into tapes/src (strict lint applies after that)
                  src="$K5B/$(basename "${cho#volume:}")"
                  echo "$reel <- ${cho#volume:} (regenerate on prep pod)" >> "$OUT/RECONCILE.txt"
                fi ;;
    new:*)      src="tapes/src/${reel}.body.tape" ;;
    template:*) src="tapes/src/permodel.body.tape" ;;
    *)          echo "SKIP $reel (no choreography)"; continue ;;
  esac
  [ -f "$src" ] || { echo "MISSING SOURCE for $reel: $src"; exit 1; }
  python3 kit/retape.py reels.yaml "$reel" "$src" "$OUT/$reel.tape"
done

PENDING=$(cut -d' ' -f1 "$OUT/RECONCILE.txt" | grep -v '^$' | paste -sd, - || true)
python3 kit/tape_lint.py reels.yaml "$OUT" --pending "${PENDING:-none}"
echo "--- volume-reconciliation pending for:"
cat "$OUT/RECONCILE.txt"
