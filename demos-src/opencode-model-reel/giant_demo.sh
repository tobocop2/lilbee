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

LC=/root/llama.cpp            # CUDA llama.cpp built by pod_bootstrap.sh
LM=/root/lilbee               # lilbee checkout (feat/local-model-api) from pod_bootstrap.sh
WS="${WS:-/root/demo-ws}"     # data dir (index + lilbee data); the model driver points
                              # this at /workspace so the index lives on the big volume
mkdir -p "$WS"
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
  # Index the modules a coding agent needs to answer questions about lilbee's chat
  # + tool-call handling. Paths are verified against the indexed branch's real
  # layout (the old providers/worker, providers/llama_cpp, providers/families dirs
  # were removed in the fleet/llama-server migration). A missing path is logged
  # LOUDLY, never silently skipped, so a stale layout can't quietly empty the corpus
  # and leave the demo asking for a file that was never indexed.
  for d in providers server/chat_dispatch server/chat_completions_api retrieval; do
    if [ -d "$LM/src/lilbee/$d" ]; then
      echo "[prep] indexing src/lilbee/$d"
      "$LILBEE" add "$LM/src/lilbee/$d"
    else
      echo "[prep] WARNING: index path missing on this branch, skipped: src/lilbee/$d" >&2
    fi
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
# tmux new-session inherits the tmux SERVER's start-time environment, not this
# script's, so LILBEE_MODELS_DIR (which the `lilbee model pull` above honoured to
# place nomic on /workspace) would be lost here -- serve would resolve models
# against the default global dir, report them "not installed", skip the embed
# sidecar, and 503 every search. Inline the var into the command (like LILBEE_DATA)
# so it reaches serve regardless of the server env.
MODELS_DIR_ENV="${LILBEE_MODELS_DIR:+LILBEE_MODELS_DIR=$LILBEE_MODELS_DIR}"
if ! curl -s "http://127.0.0.1:8080/api/health" >/dev/null 2>&1; then
  echo "[prep] starting lilbee serve"
  tmux kill-session -t lilbeeserve 2>/dev/null || true
  tmux new-session -d -s lilbeeserve \
    "cd $LM && LILBEE_DATA=$WS/.lilbee $MODELS_DIR_ENV $LILBEE serve --port 8080 > /tmp/lilbee-serve.log 2>&1"
  for _ in $(seq 1 60); do
    curl -s "http://127.0.0.1:8080/api/health" >/dev/null 2>&1 && break; sleep 2
  done
fi
TOKEN=$(python3 -c "import json;print(json.load(open('$WS/.lilbee/data/server.json'))['token'])")
echo "[prep] lilbee serve up; token read"

# Pre-warm the embed engine and verify a real search returns hits BEFORE recording.
# lilbee's fleet warms the embed role lazily and swallows warm-up failures, so the
# first lilbee_search can hit a cold engine and 503 -- which the agent misreads as
# "no embed model installed" and wastes the whole session bootstrapping search.
# Force the embed role warm via the exact path the MCP tool uses (/api/search), and
# abort rather than record a broken demo. (The proper fix is await-embed-warm in
# lilbee itself; this gate guarantees a clean demo regardless.)
echo "[prep] warming embed engine + verifying lilbee_search returns hits"
WARM_OK=0
for _ in $(seq 1 40); do
  if curl -s -H "Authorization: Bearer $TOKEN" \
       "http://127.0.0.1:8080/api/search?q=tool%20call%20parsing&top_k=3" 2>/dev/null \
       | grep -qiE '"source"|\.py'; then
    WARM_OK=1; break
  fi
  sleep 3
done
if [ "$WARM_OK" = "1" ]; then
  echo "[prep] embed engine warm; lilbee_search returns hits -> safe to record"
else
  echo "[giant] ERROR: lilbee_search never returned hits (embed engine did not warm); aborting" >&2
  exit 6
fi

# --- llama-server for the giant ---
echo "[giant] launching llama-server for $FAMILY"
tmux kill-session -t giantsrv 2>/dev/null || true
# Wait for any prior giant to release :LS_PORT before launching the new one, so the
# readiness gate below can't read a stale server's 200 and start recording against
# a model that has not finished loading.
for _ in $(seq 1 30); do
  curl -fsS -m1 "http://127.0.0.1:$LS_PORT/health" >/dev/null 2>&1 && sleep 1 || break
done
TMPL_ARG=""
[ -n "$TEMPLATE" ] && TMPL_ARG="--chat-template-file $TEMPLATE"
# Only the 200GB giants need both GPUs. Force-splitting a small model across both
# (-ngl 999 with 2 visible devices) trips llama.cpp's scheduler assert
# (GGML_SCHED_MAX_SPLIT_INPUTS) during the device-memory fit, e.g. gemma-4-E2B.
# Default to a single GPU; set MULTIGPU=1 for the giants that genuinely span both.
GPU_ENV="CUDA_VISIBLE_DEVICES=0"
[ "${MULTIGPU:-0}" = "1" ] && GPU_ENV=""
# --alias makes /v1/models advertise the same id opencode is configured with, so
# the picker shows one "lilbee" model, not a duplicate from auto-discovery.
tmux new-session -d -s giantsrv \
  "$GPU_ENV LD_LIBRARY_PATH=$LC/build/bin:$LC/build/src $LC/build/bin/llama-server --jinja -m '$GGUF' --alias '$FAMILY' -ngl 999 --host 127.0.0.1 --port $LS_PORT -c 131072 -fa on --no-webui $TMPL_ARG > /tmp/giant-srv.log 2>&1"
# Measure the cold start empirically: wall time from launch until /health reports
# the model loaded. This is the real number the demo's cold-start intro card shows
# (build_reel.sh reads the sidecar), so the "fast-forwarded cold start" is honest,
# never an invented duration. Use curl -fsS (NOT -s): while the model is still
# loading, /health returns 503 "Loading model", and a bare `curl -s` exits 0 on a
# 503 -- so the gate would pass instantly and record against a not-yet-loaded
# giant (empty demo). -fsS only exits 0 on a 2xx, i.e. weights resident. The 400x3s
# budget (20min) covers a 200GB+ giant loading across both GPUs from the volume.
COLD_START_TS=$SECONDS
UP=0
for _ in $(seq 1 400); do
  curl -fsS -m2 "http://127.0.0.1:$LS_PORT/health" >/dev/null 2>&1 && { UP=1; break; }; sleep 3
done
COLD_S=$((SECONDS - COLD_START_TS))
if [ "$UP" != "1" ]; then
  echo "[giant] ERROR: $FAMILY llama-server did not come up (see /tmp/giant-srv.log)" >&2
  exit 3
fi
# Sum every shard in the model dir (a sharded giant's first shard alone understates
# the real size the cold-start card reports).
GGUF_BYTES=$(find "$(dirname "$GGUF")" -iname '*.gguf' ! -name '*.lock' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
SIZE_GB=$(awk "BEGIN{printf \"%.0f\", $GGUF_BYTES / 1073741824}")
# Sidecar consumed by build_reel.sh to render the cold-start intro card.
printf 'model=%s\nsize_gb=%s\ncold_s=%s\ndevices="%s"\n' \
  "$FAMILY" "$SIZE_GB" "$COLD_S" "$([ "${MULTIGPU:-0}" = "1" ] && echo "2x H200" || echo "1x H200")" \
  > "$WS/coldstart-$FAMILY.txt"
echo "[giant] $FAMILY served on :$LS_PORT (cold start ${COLD_S}s, ${SIZE_GB}GB) -> warm; safe to record"

# --- opencode.json: model from llama-server, lilbee_search from lilbee MCP ---
mkdir -p "$WS/.config/opencode"
cat > "$PROJ/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "lilbee/$FAMILY",
  "provider": {
    "lilbee": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "lilbee",
      "options": { "baseURL": "http://127.0.0.1:$LS_PORT/v1", "apiKey": "sk-noauth" },
      "models": { "$FAMILY": { "name": "$FAMILY" } }
    }
  },
  "tools": { "bash": false, "grep": false, "glob": false, "list": false, "read": false, "webfetch": false, "task": false },
  "permission": { "edit": "allow", "external_directory": "deny" },
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
echo "READY: opencode cwd=$PROJ (empty, forces lilbee_search) ; provider=lilbee model=$FAMILY@:$LS_PORT ; mcp=lilbee@:8080"
