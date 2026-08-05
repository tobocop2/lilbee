#!/usr/bin/env bash
# Scripted dry-run of the manual-placement flow on main BEFORE the real take.
# Main lacks the reloading_placement guard (present on kreuzberg-5), so the
# apply->recovery window is measured, not assumed: apply the previewed plan
# as a manual spec, then prove a real `lilbee ask` roundtrip recovers (no
# port assumptions; lilbee serve binds port 0).
set -uo pipefail
source /root/kit/env.sh
# Recovery gate, not an on-camera settle measure: each CLI ask below
# cold-boots the fleet, so the budget scales with model size. The tape's
# hidden settle covers a WARM in-place reload; on-camera correctness is
# enforced by autoqa's 429/error OCR.
MODEL_GB=$(python3 -c "
import yaml
m = yaml.safe_load(open('/root/kit/reels.yaml'))
key = m['reels']['reel2-placement']['models'][0]
print(m['models'][key]['gb'])") || { echo "DRYRUN_FAIL manifest resolve"; exit 1; }
BUDGET=$(( 300 + 4 * MODEL_GB ))

rm -f /root/dryrun-smoke.txt
timeout "$BUDGET" lilbee ask 'Reply with just OK.' > /root/dryrun-smoke.txt 2>&1 \
  || { echo "DRYRUN_FAIL initial fleet smoke"; tail -3 /root/dryrun-smoke.txt; exit 1; }

lilbee --json placement preview > /root/dryrun-preview.json
SPEC=$(python3 -c "
import json
v = json.load(open('/root/dryrun-preview.json'))
spec = v.get('spec') or v.get('placement', {}).get('spec')
assert spec, f'no spec in preview payload: {list(v)}'
print(json.dumps(spec))
")
t0=$(date +%s)
echo "$SPEC" | lilbee placement set --spec - >/dev/null
timeout "$BUDGET" lilbee ask 'Reply with just OK.' > /root/dryrun-post.txt 2>&1
rc=$?
dt=$(( $(date +%s) - t0 ))
pkill -9 -x llama-swap 2>/dev/null; pkill -9 -x llama-server 2>/dev/null
pkill -9 -f '[l]ilbee serve' 2>/dev/null

if [ "$rc" = 0 ]; then
  echo "DRYRUN_PASS recovery=${dt}s budget=${BUDGET}s"
else
  echo "DRYRUN_FAIL recovery=${dt}s rc=$rc budget=${BUDGET}s"
  exit 1
fi
