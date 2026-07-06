#!/usr/bin/env bash
# Repro: lilbee returns 404 model_not_found to opencode chat completion
# even when the model is installed. Stand up a launched-style serve in the
# qwen3 workspace cwd, fire three escalating requests, and dump the lilbee
# trace log so we can see which code path the 404 takes.

set -u

pkill -f "lilbee serve" 2>/dev/null || true
sleep 2

WORKSPACE=/root/lilbee/tools/qa/opencode/workspace/qwen3
PORT=6001
TRACE=/tmp/lilbee-trace.log
: >"$TRACE"

cd "$WORKSPACE"
LILBEE_LOG_LEVEL=DEBUG uv --project /root/lilbee run lilbee serve --port "$PORT" >"$TRACE" 2>&1 &
SERVE_PID=$!
sleep 6

if ! curl -fsS "http://localhost:${PORT}/api/health" >/dev/null; then
  echo "FAIL: serve never came up"
  kill $SERVE_PID 2>/dev/null
  exit 1
fi

TOKEN=$(jq -r .token "${WORKSPACE}/.lilbee/data/server.json")
echo "==> serve up on $PORT, token=${TOKEN:0:20}..."

echo
echo "==> R1: minimal chat (200 expected)"
curl -sS -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
  | head -c 300
echo

echo
echo "==> R2: opencode-shaped, no tools (max_tokens=32000)"
curl -sS -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf","max_tokens":32000,"top_p":1,"messages":[{"role":"user","content":"hi"}]}' \
  | head -c 600
echo

echo
echo "==> R3: opencode-shaped, with tools + stream"
curl -sN -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf","max_tokens":32000,"top_p":1,"messages":[{"role":"user","content":"hi"}],"stream":true,"tools":[{"type":"function","function":{"name":"x","description":"x","parameters":{"type":"object","properties":{}}}}],"tool_choice":"auto"}' \
  | head -c 800
echo

echo
echo "==> lilbee trace tail (50 lines)"
tail -50 "$TRACE"

kill $SERVE_PID 2>/dev/null || true
sleep 2
