#!/usr/bin/env bash
# Replay the EXACT request body opencode sent in a failed session and see
# if lilbee returns 404. If yes -> lilbee bug. If 200 -> something
# transport-level (connection reset, opencode-side retry) is the cause.

set -u

WORKSPACE=/root/lilbee/tools/qa/opencode/workspace/qwen3
PORT=6001

pkill -f "lilbee serve" 2>/dev/null || true
sleep 2

cd "$WORKSPACE"
LILBEE_LOG_LEVEL=DEBUG uv --project /root/lilbee run lilbee serve --port "$PORT" >/tmp/lilbee-trace.log 2>&1 &
SERVE_PID=$!
sleep 6

curl -fsS "http://localhost:${PORT}/api/health" >/dev/null || { echo "FAIL: serve never came up"; kill $SERVE_PID 2>/dev/null; exit 1; }
TOKEN=$(jq -r .token "${WORKSPACE}/.lilbee/data/server.json")
echo "==> serve up; replaying opencode request bodies"

# Find latest opencode log files containing requestBodyValues; extract the
# JSON body of each request as raw bytes.
LOGDIR=/root/.local/share/opencode/log
LATEST=$(ls -t "$LOGDIR" | head -10)

i=0
for f in $LATEST; do
  size=$(stat -c %s "$LOGDIR/$f")
  if [ "$size" -lt 5000 ]; then continue; fi
  python3 - <<PYEOF
import re, json, sys
data = open("$LOGDIR/$f").read()
pat = re.compile(r'"requestBodyValues":(\{)')
hits = []
pos = 0
while True:
    m = pat.search(data, pos)
    if not m: break
    start = m.end() - 1
    depth = 0; end = start
    for i, c in enumerate(data[start:], start=start):
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1; break
    hits.append(data[start:end])
    pos = end
if hits:
    last = hits[-1]
    print(f"# bytes={len(last)} from $f", file=sys.stderr)
    sys.stdout.write(last)
PYEOF
  i=$((i+1))
  if [ "$i" -ge 1 ]; then break; fi
done > /tmp/opencode-body.json 2>/tmp/opencode-body.meta

if [ ! -s /tmp/opencode-body.json ]; then
  echo "no opencode request body found"
  kill $SERVE_PID 2>/dev/null
  exit 1
fi

echo "==> body extracted: $(cat /tmp/opencode-body.meta)"
echo "==> first 200 chars: $(head -c 200 /tmp/opencode-body.json)..."
echo
echo "==> replay via curl"
HTTP_STATUS=$(curl -sS -o /tmp/replay-response.json -w "%{http_code}" \
  -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  --data-binary @/tmp/opencode-body.json)
echo "HTTP $HTTP_STATUS"
echo "Body (first 500 chars):"
head -c 500 /tmp/replay-response.json
echo
echo
echo "==> lilbee trace tail (40 lines)"
tail -40 /tmp/lilbee-trace.log

kill $SERVE_PID 2>/dev/null || true
sleep 2
