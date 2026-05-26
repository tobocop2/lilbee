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
# Real HF model name to show in opencode (not our internal family tag). Falls
# back to the family if unset so the script stays usable standalone.
DISPLAY_NAME="${3:-$1}"
TEMPLATE="${4:-}"

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
  # Index lilbee's retrieval stack -- the code the demo questions are about
  # (chunking, embedding, search, reranking, query expansion, concept graph).
  # This is the stable value-prop code, not the in-flux model-families parser.
  for d in retrieval data/ingest server/chat_completions_api; do
    [ -d "$LM/src/lilbee/$d" ] && "$LILBEE" add "$LM/src/lilbee/$d" || true
  done
  touch "$WS/.lilbee/.demo_indexed"
fi

# opencode runs in an EMPTY project dir so its file tools can't read the source
# directly -- the lilbee source lives ONLY in lilbee's index, forcing lilbee_search
# (the godot-demo dynamic). AGENTS.md carries the grounding directive so the prompt
# stays a natural dev task.
PROJ=/root/demo-proj
mkdir -p "$PROJ"
cat > "$PROJ/AGENTS.md" <<'AGENTS'
# Working on the lilbee codebase

lilbee is a local-first RAG engine with an OpenAI-compatible server and an
opencode/MCP integration. Your training data does not include lilbee's internals,
and the lilbee source is NOT in this directory.

- The ONLY way to see lilbee's code is the `lilbee_search` tool. Use it to look up
  lilbee's modules, classes, and conventions before writing code. Query the
  class/function/concept; do not guess APIs.
- Cite the files you rely on as `path:Lstart-Lend`.
- No clarifying questions: make reasonable assumptions and implement.
AGENTS

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
# Only the 200GB giants need both GPUs. Force-splitting a small model across both
# (-ngl 999 with 2 visible devices) trips llama.cpp's scheduler assert
# (GGML_SCHED_MAX_SPLIT_INPUTS) during the device-memory fit, e.g. gemma-4-E2B.
# Default to a single GPU; set MULTIGPU=1 for the giants that genuinely span both.
GPU_ENV="CUDA_VISIBLE_DEVICES=0"
[ "${MULTIGPU:-0}" = "1" ] && GPU_ENV=""
# --alias makes /v1/models advertise the real model name opencode is configured
# with, so the picker shows e.g. "Qwen3-4B" under the lilbee provider, not a
# duplicate from auto-discovery and not our internal family tag.
tmux new-session -d -s giantsrv \
  "$GPU_ENV LD_LIBRARY_PATH=$LC/build/bin:$LC/build/src $LC/build/bin/llama-server --jinja -m '$GGUF' --alias '$DISPLAY_NAME' -ngl 999 --host 127.0.0.1 --port $LS_PORT -c 32768 --no-webui $TMPL_ARG > /tmp/giant-srv.log 2>&1"
UP=0
for _ in $(seq 1 120); do
  curl -s "http://127.0.0.1:$LS_PORT/health" >/dev/null 2>&1 && { UP=1; break; }; sleep 3
done
if [ "$UP" != "1" ]; then
  echo "[giant] ERROR: $FAMILY llama-server did not come up (see /tmp/giant-srv.log)" >&2
  exit 3
fi
echo "[giant] $FAMILY served on :$LS_PORT"

# --- opencode.json: model from llama-server, lilbee_search from lilbee MCP ---
mkdir -p "$WS/.config/opencode"
cat > "$PROJ/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "lilbee/$DISPLAY_NAME",
  "provider": {
    "lilbee": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "lilbee",
      "options": { "baseURL": "http://127.0.0.1:$LS_PORT/v1", "apiKey": "sk-noauth" },
      "models": { "$DISPLAY_NAME": { "name": "$DISPLAY_NAME" } }
    }
  },
  "mcp": {
    "lilbee": {
      "type": "remote",
      "url": "http://127.0.0.1:8080/mcp",
      "enabled": true,
      "headers": { "Authorization": "Bearer $TOKEN" }
    }
  },
  "tools": {
    "write": false, "edit": false, "patch": false, "bash": false,
    "read": false, "glob": false, "grep": false, "list": false,
    "webfetch": false, "todowrite": false, "todoread": false, "task": false
  }
}
JSON

# opencode caches the provider's discovered model list in its sqlite db and does
# not refresh it when /v1/models changes between runs. Each model is served under
# a fresh --alias, so a stale cache makes `opencode -m lilbee/<name>` fail with
# "Model not found". Clear the cache, then warm discovery in a loop until opencode
# actually lists the just-launched alias (cold discovery against a fresh server is
# racy) so the TUI/headless run that follows resolves the model reliably.
rm -f "$HOME/.local/share/opencode/opencode.db" \
      "$HOME/.local/share/opencode/opencode.db-wal" \
      "$HOME/.local/share/opencode/opencode.db-shm" 2>/dev/null || true
export PATH="$HOME/.opencode/bin:$PATH"
for _ in $(seq 1 20); do
  opencode models 2>/dev/null | grep -qx "lilbee/$DISPLAY_NAME" && break
  sleep 2
done

echo "READY: opencode cwd=$PROJ ; provider=lilbee model=$DISPLAY_NAME@:$LS_PORT ; mcp=lilbee@:8080 ; tools=lilbee_search-only"
