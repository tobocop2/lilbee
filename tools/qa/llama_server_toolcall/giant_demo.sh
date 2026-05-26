#!/usr/bin/env bash
# Stand up an opencode demo of a giant model doing RAG over lilbee's own source:
#   opencode  --(model)-->  llama-server --jinja  (the giant, native tool calls)
#             --(MCP)----->  lilbee serve /mcp  (lilbee_search over src/lilbee/)
#
# Idempotent prep (workspace + index + lilbee serve) runs once; the per-giant
# part launches llama-server for the requested giant and writes opencode.json.
#
# Usage: giant_demo.sh <family> <gguf_path> [chat_template_file]
set -euo pipefail

FAMILY="$1"
GGUF="$2"
TEMPLATE="${3:-}"

LC=/tmp/llama-build/llama-cpp-python-0.3.23/vendor/llama.cpp
LM=/root/lm
WS=/root/demo-ws
LS_PORT=8090                      # llama-server (the giant)
EMBED_REF="nomic-ai/nomic-embed-text-v1.5-GGUF"
TINY_CHAT="Qwen/Qwen3-4B-GGUF"    # only so lilbee serve starts; opencode never uses it
export PATH="$HOME/.local/bin:$HOME/.opencode/bin:$PATH"
export LD_LIBRARY_PATH="$LC/build/bin:$LC/build/src:${LD_LIBRARY_PATH:-}"
export LILBEE_DATA="$WS/.lilbee"

cd "$LM"
LILBEE=".venv/bin/lilbee"

# --- one-time prep: workspace + index + serve ---
if [ ! -f "$WS/.lilbee/.demo_indexed" ]; then
  echo "[prep] workspace + models + index"
  mkdir -p "$WS/.lilbee"
  cat > "$WS/.lilbee/config.toml" <<TOML
chat_model = "$TINY_CHAT"
embedding_model = "$EMBED_REF"
TOML
  "$LILBEE" model pull "$EMBED_REF"
  "$LILBEE" model pull "$TINY_CHAT"
  # Index lilbee's own source. A curated subset keeps CPU embedding quick and the
  # demo answers focused on the interesting machinery.
  for d in providers/worker/response_parser providers/llama_cpp providers/families \
           server/chat_completions_api retrieval; do
    [ -d "$LM/src/lilbee/$d" ] && "$LILBEE" add "$LM/src/lilbee/$d" || true
  done
  touch "$WS/.lilbee/.demo_indexed"
fi

# --- lilbee serve (background, for /mcp) ---
if ! curl -s "http://127.0.0.1:8080/api/health" >/dev/null 2>&1; then
  echo "[prep] starting lilbee serve"
  tmux kill-session -t lilbeeserve 2>/dev/null || true
  tmux new-session -d -s lilbeeserve \
    "cd $LM && LILBEE_DATA=$WS/.lilbee $LILBEE serve --port 8080 > /tmp/lilbee-serve.log 2>&1"
  for _ in $(seq 1 60); do
    curl -s "http://127.0.0.1:8080/api/health" >/dev/null 2>&1 && break; sleep 2
  done
fi
TOKEN=$(python3 -c "import json;print(json.load(open('$WS/.lilbee/data/server.json'))['token'])")
echo "[prep] lilbee serve up; token read"

# --- llama-server for the giant ---
echo "[giant] launching llama-server for $FAMILY"
tmux kill-session -t giantsrv 2>/dev/null || true
TMPL_ARG=""
[ -n "$TEMPLATE" ] && TMPL_ARG="--chat-template-file $TEMPLATE"
tmux new-session -d -s giantsrv \
  "LD_LIBRARY_PATH=$LC/build/bin:$LC/build/src $LC/build/bin/llama-server --jinja -m '$GGUF' -ngl 999 --host 127.0.0.1 --port $LS_PORT -c 32768 --no-webui $TMPL_ARG > /tmp/giant-srv.log 2>&1"
for _ in $(seq 1 300); do
  curl -s "http://127.0.0.1:$LS_PORT/health" >/dev/null 2>&1 && break; sleep 3
done
echo "[giant] $FAMILY served on :$LS_PORT"

# --- opencode.json: model from llama-server, lilbee_search from lilbee MCP ---
mkdir -p "$WS/.config/opencode"
cat > "$WS/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "giant/$FAMILY",
  "provider": {
    "giant": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "$FAMILY",
      "options": { "baseURL": "http://127.0.0.1:$LS_PORT/v1", "apiKey": "sk-noauth" },
      "models": { "$FAMILY": { "name": "$FAMILY" } }
    }
  },
  "mcp": {
    "lilbee": {
      "type": "remote",
      "url": "http://127.0.0.1:8080/mcp",
      "enabled": true,
      "headers": { "Authorization": "Bearer $TOKEN" }
    }
  }
}
JSON
echo "READY: opencode.json at $WS/opencode.json ; model=$FAMILY@:$LS_PORT ; mcp=lilbee@:8080"
