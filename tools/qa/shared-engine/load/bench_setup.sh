#!/usr/bin/env bash
# Bench pod setup: engine seed + bootstrap + models + a serving lilbee.
# Run inside a pod tmux.
set -uo pipefail
BRANCH=fix/serve-data-dir-singleton
WHEEL_URL="https://github.com/tobocop2/lilbee/releases/download/v0.6.90b420.dev721/lilbee_engine-0.6.90b420.dev721-1.cu124-py3-none-manylinux_2_17_x86_64.whl"
ENGINE_CACHE=/workspace/engine-cache/cu124
CHAT="unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf"
EMBED="nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
log() { echo "[bench-setup $(date +%H:%M:%S)] $*"; }

if [ ! -x "$ENGINE_CACHE/llama-server" ]; then
  log "seeding engine cache"
  mkdir -p "$ENGINE_CACHE" /tmp/engwheel
  curl -fL --retry 3 -o /tmp/engwheel/engine.whl "$WHEEL_URL"
  python3 -c "import zipfile; zipfile.ZipFile('/tmp/engwheel/engine.whl').extractall('/tmp/engwheel/x')"
  cp -a /tmp/engwheel/x/lilbee_engine/bin/. "$ENGINE_CACHE/"
fi
chmod +x "$ENGINE_CACHE"/llama-* "$ENGINE_CACHE"/gguf-parser 2>/dev/null

log "bootstrap"
curl -fsSL "https://raw.githubusercontent.com/tobocop2/lilbee/$BRANCH/tools/qa/opencode/pod_bootstrap.sh" -o /tmp/pod_bootstrap.sh
BRANCH="$BRANCH" bash /tmp/pod_bootstrap.sh > /root/bootstrap.log 2>&1 || { log "BOOTSTRAP FAILED"; exit 1; }

source /root/lilbee_venv/bin/activate
export PATH="/root/lilbee_venv/bin:$PATH"
export LILBEE_ENGINE_DIR=/root/.cache/lilbee/engine LILBEE_MODELS_DIR=/root/models

log "pulling models"
mkdir -p /root/bench/kb /root/bench/.lilbee
cat > /root/bench/.lilbee/config.toml <<EOF
chat_model = "$CHAT"
embedding_model = "$EMBED"
keep_engine_warm = true
engine_idle_ttl_minutes = 0
chat_n_ctx_target = 65536
EOF
cd /root/bench
for i in 1 2 3; do lilbee model pull "$CHAT" && break; log "pull retry $i"; sleep 5; done
lilbee model pull "$EMBED"
cp /workspace/lilbee/docs/usage.md /workspace/lilbee/docs/architecture.md kb/ 2>/dev/null || cp /root/lilbee/docs/usage.md /root/lilbee/docs/architecture.md kb/ 2>/dev/null
lilbee add kb/ > /dev/null 2>&1

log "starting lilbee serve"
mkdir -p /root/results
nohup lilbee serve --data-dir /root/bench/.lilbee > /root/results/serve.log 2>&1 &
for _ in $(seq 1 90); do
  PORT=$(cat /root/bench/.lilbee/data/server.port 2>/dev/null)
  [ -n "$PORT" ] && curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 && break
  sleep 2
done
log "server on port $PORT"

log "warming chat"
timeout 420 lilbee ask "Reply with the single word: ready" > /root/warm.log 2>&1
echo "PORT=$PORT" > /root/bench.env
echo "TOKEN=$(python3 -c "import json;print(json.load(open('/root/bench/.lilbee/data/server.json'))['token'])")" >> /root/bench.env
log "installing llmperf"
python3 -m venv /root/perfvenv && /root/perfvenv/bin/pip install -q "ray[default]" 2>/dev/null
git clone -q --depth 1 https://github.com/ray-project/llmperf.git /root/llmperf && /root/perfvenv/bin/pip install -q -e /root/llmperf && echo LLMPERF-OK
nvidia-smi --query-gpu=memory.used --format=csv,noheader
log "SETUP-COMPLETE"
