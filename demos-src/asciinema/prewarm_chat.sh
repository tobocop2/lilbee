#!/bin/bash
# Load the LITE chat model into VRAM on data dir $1 and leave the server running
# so a subsequent lilbee TUI on the same dir connects to a warm engine.
set -u
DIR="${1:?data dir}"
rm -rf "$DIR"; mkdir -p "$DIR"
export LILBEE_MODELS_DIR=/root/models
export LILBEE_EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
export LILBEE_CHAT_MODEL="Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf"
export LILBEE_TEMPERATURE=0.0 LILBEE_SEED=42 HF_HUB_DISABLE_XET=1
setsid /root/venv/bin/lilbee serve -d "$DIR" -p 8090 -H 127.0.0.1 >/root/prewarm.log 2>&1 </dev/null &
TOK=""
for i in $(seq 1 120); do
  sleep 2
  [ -z "$TOK" ] && TOK=$(python3 -c "import json;print(json.load(open('$DIR/data/server.json'))['token'])" 2>/dev/null || true)
  [ -n "$TOK" ] || continue
  curl -s -m 15 http://127.0.0.1:8090/api/health -H "Authorization: Bearer $TOK" 2>/dev/null | grep -q '"chat_ready":true' && break
done
# force the chat model into VRAM
curl -s -m 120 -X POST http://127.0.0.1:8090/api/chat -H 'Content-Type: application/json' -H "Authorization: Bearer $TOK" -d '{"question":"hi"}' >/dev/null 2>&1
echo "prewarmed: server on $DIR port 8090, chat_ready"
