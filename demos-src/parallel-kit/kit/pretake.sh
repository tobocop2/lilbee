#!/usr/bin/env bash
# Hard gate before any VHS roll — fail here at minutes, not after a 4-minute
# take. Proves state, then (for chat reels) boots the fleet once via a real
# `lilbee ask` smoke: that materializes llama-swap-chat.json (identity) and
# proves end-to-end chat without assuming any port (lilbee serve binds port 0).
set -uo pipefail
REEL="$1"
source /root/kit/env.sh
FAIL=0
step(){ if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; FAIL=1; fi }

step "no stray llama-swap"   "! pgrep -x llama-swap >/dev/null"
step "no stray llama-server" "! pgrep -x llama-server >/dev/null"
step "lancedb dir exists (wizard guard)" "[ -d /root/demo-data/data/lancedb ]"
step "config exists"         "[ -f /root/demo-data/config.toml ]"
step "disk >30G free"        "[ \$(df --output=avail -BG /root | tail -1 | tr -dc 0-9) -gt 30 ]"
step "fonts resolved"        "fc-match 'JetBrains Mono' family | grep -q 'JetBrains Mono'"

if ! RESOLVED=$(python3 - "$REEL" <<'PYEOF'
import sys, yaml
m = yaml.safe_load(open('/root/kit/reels.yaml'))
r = m['reels'][sys.argv[1]]
key = r.get('model') or (r.get('models') or [None])[0]
if key:
    mm = m['models'][key]
    print(mm['ref'].replace('/', '--'), 300 + 4 * mm['gb'])
else:
    print('NONE 0')
PYEOF
); then
  echo "FAIL manifest resolve for $REEL"
  FAIL=1
  RESOLVED="NONE 0"
fi
read -r EXPECT SMOKE_BUDGET <<< "$RESOLVED"
[ "$EXPECT" = "NONE" ] && EXPECT=""
if [ -n "$EXPECT" ]; then
  step "chat model configured" "grep -q chat_model /root/demo-data/config.toml"
  rm -f /root/smoke.txt
  # lilbee is a RAG engine: a bare "ok" prompt returns a CITED answer, not the
  # literal word. The smoke proves the full fleet (chat + embed + retrieval)
  # answers end-to-end — pass on a clean, non-empty, error-free completion.
  timeout "$SMOKE_BUDGET" lilbee ask 'In one short sentence, what is this manual about?' > /root/smoke.txt 2>&1
  rc=$?
  tail -2 /root/smoke.txt
  if [ "$rc" = 0 ] && [ -s /root/smoke.txt ] \
     && ! grep -qiE "traceback|no .* server is running|error:|is busy" /root/smoke.txt; then
    echo "PASS smoke-ask"
  else
    echo "FAIL smoke-ask (rc=$rc, budget=${SMOKE_BUDGET}s)"; FAIL=1
  fi
  if grep -q "models--${EXPECT}" /root/demo-data/data/llama-swap-chat.json 2>/dev/null; then
    echo "PASS identity models--${EXPECT}"
  else
    echo "FAIL identity (llama-swap-chat.json does not reference models--${EXPECT})"
    FAIL=1
  fi
  pkill -9 -x llama-swap 2>/dev/null; pkill -9 -x llama-server 2>/dev/null
  pkill -9 -f '[l]ilbee serve' 2>/dev/null
  sleep 3
fi

[ $FAIL -eq 0 ] && echo "GATE_PASS" || echo "GATE_FAIL"
exit $FAIL
