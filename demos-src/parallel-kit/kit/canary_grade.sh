#!/usr/bin/env bash
# Canary-only content gate: the KBs were re-extracted with kreuzberg 4.9 on
# main, so the graded towing answer must be re-proven via real asks BEFORE
# any reel records. 3/3 asks must contain the trailer-limit and GCW facts.
set -uo pipefail
source /root/kit/env.sh
python3 /root/kit/stage.py /root/kit/reels.yaml pm-gptoss-20b

PASS=0
for i in 1 2 3; do
  out=$(timeout 900 lilbee ask "A customer wants to tow a 3,500 lb boat trailer through the mountains in August. Can this car pull that load per the manual, and what is the maximum trailer weight and gross combination weight?" 2>&1 || true)
  if echo "$out" | grep -qE '1[,.]?500' && echo "$out" | grep -qE '6[,.]?600'; then
    PASS=$((PASS+1)); echo "ASK_$i PASS"
  else
    echo "ASK_$i FAIL"; echo "$out" | tail -5
  fi
done
pkill -9 -x llama-swap 2>/dev/null; pkill -9 -x llama-server 2>/dev/null
pkill -9 -f '[l]ilbee serve' 2>/dev/null

[ "$PASS" = 3 ] && echo "CANARY_GRADE_PASS 3/3" || { echo "CANARY_GRADE_FAIL $PASS/3"; exit 1; }
