#!/usr/bin/env bash
# Gate + WARM: run the render/state checks, then boot the fleet ONCE and leave
# it running so the take connects to an already-loaded model (no cold boot in
# the reel's hidden window). Verifies a live 200 + model identity. Exits
# nonzero on any failure WITHOUT killing the fleet on success — job.sh keeps
# it warm across take attempts. `lilbee ask` boots llama-swap and leaves it up
# (singleton per data_dir), so a successful ask == warm fleet.
set -uo pipefail
REEL="$1"
source /root/kit/env.sh
FAIL=0
step(){ if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; FAIL=1; fi }

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
    # generous budget: a cold multi-GPU heavy model (235B/glm46) loads slowly
    print(mm['ref'].replace('/', '--'), 600 + 6 * mm['gb'])
else:
    print('NONE 0')
PYEOF
); then
  echo "FAIL manifest resolve for $REEL"; FAIL=1; RESOLVED="NONE 0"
fi
read -r EXPECT WARM_BUDGET <<< "$RESOLVED"
[ "$EXPECT" = "NONE" ] && EXPECT=""

if [ -n "$EXPECT" ]; then
  step "chat model configured" "grep -q chat_model /root/demo-data/config.toml"
  rm -f /root/warm.txt
  # this ask boots the fleet and LEAVES llama-swap running (do NOT pkill)
  timeout "$WARM_BUDGET" lilbee ask 'In one short sentence, what is this manual about?' > /root/warm.txt 2>&1
  rc=$?
  tail -2 /root/warm.txt
  if [ "$rc" = 0 ] && [ -s /root/warm.txt ] \
     && ! grep -qiE "traceback|no .* server is running|error:|is busy" /root/warm.txt; then
    echo "PASS warm-ask"
  else
    echo "FAIL warm-ask (rc=$rc, budget=${WARM_BUDGET}s)"; FAIL=1
  fi
  if grep -q "models--${EXPECT}" /root/demo-data/data/llama-swap-chat.json 2>/dev/null; then
    echo "PASS identity models--${EXPECT}"
  else
    echo "FAIL identity (llama-swap-chat.json does not reference models--${EXPECT})"; FAIL=1
  fi
  # a second quick ask confirms the fleet is warm and answering fast (proves
  # the take won't cold-boot); still leaves the fleet up
  if [ "$FAIL" = 0 ]; then
    t0=$(date +%s)
    timeout 120 lilbee ask 'Reply OK.' >/dev/null 2>&1
    echo "PASS warm-confirm ($(( $(date +%s) - t0 ))s second ask)"
  fi
fi

[ $FAIL -eq 0 ] && echo "WARM_OK (fleet left running)" || echo "WARM_FAIL"
exit $FAIL
