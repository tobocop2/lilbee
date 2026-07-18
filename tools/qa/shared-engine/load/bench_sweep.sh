#!/usr/bin/env bash
# Capacity sweep against a running lilbee serve: llmperf at increasing
# concurrency, then the same sweep against the engine's llama-swap proxy
# directly, so lilbee's server overhead is a measured delta. Emits one
# summary JSON per cell under $RESULTS.
set -uo pipefail
source /root/bench.env   # PORT, TOKEN
RESULTS=${RESULTS:-/root/results/sweep}
MODEL="unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf"
CONCURRENCIES=${CONCURRENCIES:-"1 2 4 8 16"}
REQUESTS_PER_CELL=${REQUESTS_PER_CELL:-32}
mkdir -p "$RESULTS"
log() { echo "[sweep $(date +%H:%M:%S)] $*"; }

run_cell() { # run_cell <label> <base_url> <api_key> <model> <concurrency>
  local label=$1 base=$2 key=$3 model=$4 c=$5
  local dir="$RESULTS/$label-c$c"
  mkdir -p "$dir"
  log "cell $label c=$c"
  OPENAI_API_BASE="$base" OPENAI_API_KEY="$key" \
    /root/perfvenv/bin/python /root/llmperf/token_benchmark_ray.py \
    --model "$model" --llm-api openai \
    --mean-input-tokens 550 --stddev-input-tokens 150 \
    --mean-output-tokens 200 --stddev-output-tokens 20 \
    --max-num-completed-requests "$REQUESTS_PER_CELL" \
    --num-concurrent-requests "$c" \
    --timeout 1200 --results-dir "$dir" > "$dir/run.log" 2>&1
  echo "rc=$? for $label-c$c"
}

for c in $CONCURRENCIES; do
  run_cell lilbee "http://127.0.0.1:$PORT/v1" "$TOKEN" "$MODEL" "$c"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$RESULTS/gpu.log"
done

# Direct-to-engine comparison: llama-swap's chat proxy speaks /v1 with the
# model alias from its own config.
STATE=$(ls /root/.cache/lilbee/engine/llama-swap-chat.*.json 2>/dev/null | head -1)
PROXY_PORT=$(python3 -c "import json,sys,re; s=open('$STATE').read(); m=re.search(r'listen[^0-9]*(\d+)', s); print(m.group(1)) if m else sys.exit(1)" 2>/dev/null)
ALIAS=$(python3 -c "import json; d=json.load(open('$STATE')); print(next(iter(d.get('models',{}).keys())))" 2>/dev/null)
if [ -z "$PROXY_PORT" ]; then
  PROXY_PORT=$(python3 -c "import json; d=json.load(open('${STATE/llama-swap-chat/llama-swap.state.chat}')); print(d['proxy_port'])" 2>/dev/null)
fi
if [ -z "$ALIAS" ]; then
  ALIAS=$(python3 -c "import json; d=json.load(open('$STATE')); print(next(iter(d['models'])))" 2>/dev/null)
fi
log "engine proxy port=$PROXY_PORT alias=$ALIAS"
if [ -n "$PROXY_PORT" ] && [ -n "$ALIAS" ]; then
  for c in 1 4; do
    run_cell engine "http://127.0.0.1:$PROXY_PORT/v1" none "$ALIAS" "$c"
  done
fi
log "SWEEP-COMPLETE"
