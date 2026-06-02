#!/usr/bin/env bash
# Build the SHARED, model-agnostic corpus index for the opencode demo using the
# real product path (lilbee add -> sync), then verify it grounds every probe
# BEFORE any model or GPU work. Run once per pod; every model's
# `lilbee launch opencode` reads this same data dir afterwards.
#
# This replaces the old inline indexing block in giant_demo.sh (hardcoded source
# dirs, no readiness check) that let a stale path quietly empty the corpus and
# leave the demo asking for a file that was never indexed.
#
#   cd /root/reel && git pull -q origin demo-reel/opencode-model-matrix
#   /root/reel/demos-src/opencode-model-reel/ingest_corpus.sh
set -uo pipefail
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1

HERE="$(cd "$(dirname "$0")" && pwd)"
LM="${LM:-/root/lilbee}"                                # lilbee checkout on the pod
# Data + models live on the network volume so they survive a pod stop/start and
# don't fill the small root overlay.
export LILBEE_DATA="${LILBEE_DATA:-/workspace/.lilbee}"
export LILBEE_MODELS_DIR="${LILBEE_MODELS_DIR:-/workspace/models}"
LILBEE="${LILBEE:-$LM/.venv/bin/lilbee}"
PORT="${PORT:-8080}"

# Featured retrieval models (real product defaults). The chat model is NOT set
# here: this index is model-agnostic, and the matrix sets chat_model per giant.
EMBED_REF="nomic-ai/nomic-embed-text-v1.5-GGUF"
RERANK_REF="gpustack/bge-reranker-v2-m3-GGUF"
CHUNK_SIZE=320

DATA="$LILBEE_DATA/data"
say(){ echo "[ingest $(date -u +%H:%M:%S)] $*"; }
say "data=$LILBEE_DATA models=$LILBEE_MODELS_DIR lilbee=$LILBEE"

# 1) Index-level retrieval profile, written BEFORE the sync so the corpus is
#    embedded + chunked + graphed under these settings (they define the index;
#    changing them later needs a force-rebuild, so they must be fixed up front).
mkdir -p "$LILBEE_DATA"
cat > "$LILBEE_DATA/config.toml" <<TOML
embedding_model = "$EMBED_REF"
reranker_model = "$RERANK_REF"
chunk_size = $CHUNK_SIZE
concept_graph = true
TOML
say "wrote index profile -> $LILBEE_DATA/config.toml"

# 2) Pull the retrieval models (embed is required to ingest; reranker is used at
#    query time and is pulled now so per-model tuning can lean on it).
"$LILBEE" model pull "$EMBED_REF"
"$LILBEE" model pull "$RERANK_REF"

# 3) Ingest the corpus (lilbee's own repo) via the real product path. lilbee add
#    copies each path under documents/ preserving its subtree, then syncs. A
#    missing path is logged LOUD, never silently skipped.
CORPUS=()
for p in "$LM/src/lilbee" "$LM/docs" "$LM/README.md"; do
  if [ -e "$p" ]; then CORPUS+=("$p"); else say "WARNING corpus path missing, skipped: $p"; fi
done
[ ${#CORPUS[@]} -gt 0 ] || { say "FATAL no corpus paths exist under $LM"; exit 2; }
say "lilbee add ${CORPUS[*]}"
"$LILBEE" add "${CORPUS[@]}"

# 4) Start serve (reuse a healthy one) and read the session token. LILBEE_DATA
#    and LILBEE_MODELS_DIR are inlined into the tmux command because a new tmux
#    session inherits the tmux SERVER's start-time env, not this script's.
if ! curl -fsS -m2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  say "starting lilbee serve on :$PORT"
  tmux kill-session -t lilbeeserve 2>/dev/null || true
  tmux new-session -d -s lilbeeserve \
    "cd $LM && LILBEE_DATA=$LILBEE_DATA LILBEE_MODELS_DIR=$LILBEE_MODELS_DIR $LILBEE serve --port $PORT > /tmp/lilbee-serve.log 2>&1"
  for _ in $(seq 1 60); do
    curl -fsS -m2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 && break; sleep 2
  done
fi
curl -fsS -m2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 \
  || { say "FATAL serve not healthy (see /tmp/lilbee-serve.log)"; exit 3; }
TOKEN=$(python3 -c "import json;print(json.load(open('$DATA/server.json'))['token'])")
say "serve healthy; token read"

# 5) Warm the embed engine. The fleet warms the embed role lazily and the first
#    /api/search can hit a cold engine and 503; poll a real search until it
#    returns hits so the gate measures a warm index, not a cold-start miss.
say "warming embed engine via a real search"
WARM=0
for _ in $(seq 1 40); do
  if curl -fsS -m5 -H "Authorization: Bearer $TOKEN" \
       "http://127.0.0.1:$PORT/api/search?q=dispatch%20chat%20request&top_k=1&chunk_type=raw" 2>/dev/null \
       | grep -qiE '"source"|\.py'; then WARM=1; break; fi
  sleep 3
done
[ "$WARM" = "1" ] || { say "FATAL embed engine never warmed; search returns no hits"; exit 5; }

# 6) Demo-readiness gate: every probe's expected file must surface, or abort
#    before spending GPU hours on a corpus that can't ground the demo.
say "running demo-readiness gate"
if python3 "$HERE/gate.py" --base-url "http://127.0.0.1:$PORT" --token "$TOKEN" --probes "$HERE/probes.toml"; then
  say "INGEST COMPLETE: corpus indexed + grounded; data dir ready for launch + tuning."
else
  say "GATE FAILED -> corpus not demo-ready; not proceeding"
  exit 4
fi
