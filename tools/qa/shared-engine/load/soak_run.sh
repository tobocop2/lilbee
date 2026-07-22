#!/usr/bin/env bash
# Endurance soak with chaos: rounds of concurrent chat load through lilbee's
# /v1 plus CLI engine-acquire cycles, per-round invariants, and a periodic
# kill of an engine process to prove self-healing. Metrics land as one CSV
# row per round; any invariant breach is recorded and the soak continues.
set -uo pipefail
source /root/bench.env   # PORT, TOKEN
ROUNDS=${ROUNDS:-40}
STREAMS=${STREAMS:-4}
CHAOS_EVERY=${CHAOS_EVERY:-5}
MODEL="unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf"
RESULTS=${RESULTS:-/root/results/soak}
mkdir -p "$RESULTS"
CSV="$RESULTS/rounds.csv"
echo "round,ok_streams,cli_ok,duration_s,vram_mb,swaps,servers,locks,chaos,failures" > "$CSV"
log() { echo "[soak $(date +%H:%M:%S)] $*"; }

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
# procps-ng pgrep -c prints the count even when it is 0 and still exits nonzero,
# so a bare `pgrep -fc ... || echo 0` yields "0\n0" and puts a newline mid-CSV-row.
count() { local n; n=$(pgrep -fc "$1" 2>/dev/null | head -1); echo "${n:-0}"; }
swaps() { count "bin/llama-swap"; }
servers() { count "bin/llama-server"; }
locks() { ls "$LILBEE_ENGINE_DIR/engine-users/" 2>/dev/null | wc -l | tr -d " "; }

one_stream() { # one_stream <round> <idx>: a multi-turn chat exchange
  local r=$1 idx=$2 out
  out=$(curl -sf -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"max_tokens\": 350, \"messages\": [{\"role\": \"user\", \"content\": \"Write a python function that parses RFC3339 timestamps and explain each regex group briefly. Round $r stream $idx.\"}]}" \
    | python3 -c "import json,sys; print(len(json.load(sys.stdin)['choices'][0]['message']['content']))" 2>/dev/null)
  [ -n "$out" ] && [ "$out" -gt 50 ]
}

cli_cycle() { # engine acquire/release through a one-shot CLI search
  (cd /root/bench && timeout 240 /root/lilbee_venv/bin/lilbee search "engine idle ttl" --json > /dev/null 2>&1)
}

chaos() { # kill one engine process; the design must self-heal
  local victim
  if [ $((RANDOM % 2)) -eq 0 ]; then
    victim=$(pgrep -f "bin/llama-server" | head -1)
    log "chaos: SIGKILL llama-server $victim"
  else
    victim=$(pgrep -f "bin/llama-swap" | head -1)
    log "chaos: SIGKILL llama-swap $victim"
  fi
  [ -n "$victim" ] && kill -9 "$victim" 2>/dev/null
}

export LILBEE_ENGINE_DIR=/root/.cache/lilbee/engine LILBEE_MODELS_DIR=/root/models
BASE_VRAM=$(vram)
BASE_SWAPS=$(swaps)
BASE_SERVERS=$(servers)
log "baseline vram=${BASE_VRAM}MB swaps=$BASE_SWAPS servers=$BASE_SERVERS rounds=$ROUNDS streams=$STREAMS chaos every $CHAOS_EVERY"

for r in $(seq 1 "$ROUNDS"); do
  t0=$SECONDS
  did_chaos=0
  failures=""
  if [ $((r % CHAOS_EVERY)) -eq 0 ]; then
    did_chaos=1
    (sleep $((RANDOM % 20 + 5)); chaos) &
  fi
  pids=""
  for s in $(seq 1 "$STREAMS"); do
    one_stream "$r" "$s" &
    pids="$pids $!"
  done
  ok=0
  for p in $pids; do
    wait "$p" && ok=$((ok + 1))
  done
  cli_ok=0
  cli_cycle && cli_ok=1
  wait   # chaos timer, if any
  dur=$((SECONDS - t0))
  v=$(vram); sw=$(swaps); sv=$(servers); lk=$(locks)
  # A forced kill may legitimately fail the streams it interrupted, but that is
  # recorded rather than dropped: exempting chaos rounds silently made "zero
  # invariant breaches" unfalsifiable on exactly the rounds under test. Chaos
  # degradations get their own markers and their own total.
  if [ "$did_chaos" -eq 0 ]; then
    [ "$ok" -lt "$STREAMS" ] && failures="${failures}streams;"
    [ "$cli_ok" -eq 0 ] && failures="${failures}cli;"
  else
    [ "$ok" -lt "$STREAMS" ] && failures="${failures}chaos-streams;"
    [ "$cli_ok" -eq 0 ] && failures="${failures}chaos-cli;"
  fi
  [ "$v" -gt $((BASE_VRAM + BASE_VRAM / 10)) ] && failures="${failures}vram-creep;"
  [ "$lk" -gt 3 ] && failures="${failures}lock-leak;"
  # Documented invariant that was never enforced: process counts return to
  # baseline. A kill that orphans a llama-server shows up here and nowhere else.
  [ "$sw" -gt "$BASE_SWAPS" ] && failures="${failures}swap-leak;"
  [ "$sv" -gt "$BASE_SERVERS" ] && failures="${failures}server-leak;"
  echo "$r,$ok,$cli_ok,$dur,$v,$sw,$sv,$lk,$did_chaos,$failures" >> "$CSV"
  log "round $r: ok=$ok/$STREAMS cli=$cli_ok ${dur}s vram=${v}MB swaps=$sw servers=$sv locks=$lk chaos=$did_chaos ${failures:+FAIL:$failures}"
done

# Anchored to the field so "chaos-streams;" is not counted as a hard "streams;".
BAD=$(grep -cE "(^|,)([^,]*;)*(streams|cli|vram-creep|lock-leak|swap-leak|server-leak);" "$CSV" || true)
CHAOS_DEGRADED=$(grep -c "chaos-streams;\|chaos-cli;" "$CSV" || true)
log "SOAK-COMPLETE rounds=$ROUNDS invariant-breaches=$BAD chaos-round-degradations=$CHAOS_DEGRADED"
[ "$BAD" -eq 0 ] || exit 1
