#!/usr/bin/env bash
# Shared-engine acceptance harness: four concurrent coding agents on one box
# must share exactly one engine. Run ON THE POD after pod_bootstrap.sh, inside
# tmux. Phases: fixtures -> baseline (1 agent) -> load (4 agents) -> assertions.
#
#   REPO_DIR=/workspace/lilbee bash tools/qa/shared-engine/harness.sh
#
# Each agent gets its own project root (per-project .lilbee KB) and its own
# HOME (opencode writes a global config), while LILBEE_ENGINE_DIR pins one
# shared machine slot for everyone: the property under test. Exit 0 means
# every load-bearing assertion held; $RESULTS/report.txt has the record.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/lilbee}"
VENV="${UV_PROJECT_ENVIRONMENT:-/root/lilbee_venv}"
AGENTS_ROOT="${AGENTS_ROOT:-/root/agents}"
RESULTS="${RESULTS:-/root/harness-results}"
CHAT_MODEL="${CHAT_MODEL:-unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"
EMBED_MODEL="${EMBED_MODEL:-nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf}"
AGENT_TIMEOUT_S="${AGENT_TIMEOUT_S:-900}"
SETTLE_S=25

export PATH="$VENV/bin:$HOME/.opencode/bin:$PATH"
export LILBEE_ENGINE_DIR="${LILBEE_ENGINE_DIR:-/root/.cache/lilbee/engine}"
export LILBEE_MODELS_DIR="${LILBEE_MODELS_DIR:-/workspace/models}"
TASKS_SRC="$REPO_DIR/tools/qa/shared-engine/tasks"
mkdir -p "$RESULTS"
REPORT="$RESULTS/report.txt"
: > "$REPORT"

PASS=0
FAIL=0
note() { echo "[harness] $*" | tee -a "$REPORT"; }
check() { # check <name> <0-for-pass>
  if [ "$2" -eq 0 ]; then note "PASS: $1"; PASS=$((PASS + 1)); else note "FAIL: $1"; FAIL=$((FAIL + 1)); fi
}

engine_llama_servers() { pgrep -fc "lilbee_engine/bin/llama-server"; }
engine_swaps() { pgrep -fc "lilbee_engine/bin/llama-swap"; }
vram_used_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
user_locks() { ls "$LILBEE_ENGINE_DIR/engine-users/" 2>/dev/null | wc -l; }

# ── Phase 0: fixtures ───────────────────────────────────────────────
note "phase 0: fixtures (4 project roots, per-project KBs, shared models+slot)"
for i in 1 2 3 4; do
  PROJ="$AGENTS_ROOT/proj$i"
  rm -rf "$PROJ/kb" "$PROJ/.lilbee/data" "$PROJ/home"
  mkdir -p "$PROJ/.lilbee" "$PROJ/kb" "$PROJ/home"
  cp "$REPO_DIR/docs/usage.md" "$REPO_DIR/docs/architecture.md" "$PROJ/kb/" 2>/dev/null
  # documents_dir stays default (.lilbee/documents): pointing it at kb/ would
  # make "lilbee add kb/" copy kb into itself.
  cat > "$PROJ/.lilbee/config.toml" <<EOF
chat_model = "$CHAT_MODEL"
embedding_model = "$EMBED_MODEL"
EOF
  case $i in
    1) cp "$TASKS_SRC/task1_slugify_test.py" "$PROJ/"; rm -f "$PROJ/slugify_impl.py"; cp "$TASKS_SRC/task1_prompt.txt" "$PROJ/prompt.txt";;
    2) cp "$TASKS_SRC/task2_window_impl.py" "$TASKS_SRC/task2_window_test.py" "$PROJ/"; cp "$TASKS_SRC/task2_prompt.txt" "$PROJ/prompt.txt";;
    3) rm -f "$PROJ/harness_answer.md"; cp "$TASKS_SRC/task3_prompt.txt" "$PROJ/prompt.txt";;
    4) cp "$TASKS_SRC/task4_dupes_impl.py" "$TASKS_SRC/task4_dupes_test.py" "$PROJ/"; cp "$TASKS_SRC/task4_prompt.txt" "$PROJ/prompt.txt";;
  esac
done

note "pulling models once into the shared models dir"
(cd "$AGENTS_ROOT/proj1" && lilbee model pull "$CHAT_MODEL" && lilbee model pull "$EMBED_MODEL") \
  >> "$RESULTS/pull.log" 2>&1
check "models pulled" $?

for i in 1 2 3 4; do
  note "ingesting proj$i KB"
  (cd "$AGENTS_ROOT/proj$i" && lilbee add kb/ >> "$RESULTS/ingest.log" 2>&1)
done
note "post-ingest engine census: servers=$(engine_llama_servers) swaps=$(engine_swaps)"

# engine stop covers the machine slot plus the CURRENT root's private dir,
# so run it once per project root to catch any private-overflow leftovers.
for i in 1 2 3 4; do
  (cd "$AGENTS_ROOT/proj$i" && lilbee engine stop) >> "$RESULTS/ingest.log" 2>&1 || true
done
sleep "$SETTLE_S"
note "post-ingest lingering check: swaps=$(engine_swaps) (0 expected: last CLI out stops the engine)"
note "idle VRAM: $(vram_used_mb)MB"

server_token() { # server_token <proj-index>
  python3 -c "import json;print(json.load(open('$AGENTS_ROOT/proj$1/.lilbee/data/server.json'))['token'])" 2>/dev/null
}

start_server() { # start_server <proj-index>
  local proj="$AGENTS_ROOT/proj$1"
  (cd "$proj" && nohup lilbee serve --data-dir "$proj/.lilbee" \
    > "$RESULTS/serve$1.log" 2>&1 & echo $! > "$RESULTS/serve$1.pid")
  for _ in $(seq 1 60); do
    [ -f "$proj/.lilbee/data/server.port" ] \
      && curl -sf "http://127.0.0.1:$(cat "$proj/.lilbee/data/server.port")/api/health" >/dev/null 2>&1 \
      && return 0
    sleep 2
  done
  return 1
}

wire_opencode() { # wire_opencode <proj-index>: launcher-equivalent config, per-agent HOME
  local proj="$AGENTS_ROOT/proj$1"
  local port token
  port=$(cat "$proj/.lilbee/data/server.port")
  token=$(server_token "$1")
  mkdir -p "$proj/home/.config/opencode"
  # chat_ctx caps the client's context accounting to what the engine actually
  # serves; without it opencode assumes the model's native window and requests
  # blow past the served slot size.
  (cd "$proj" && LILBEE_PORT="$port" LILBEE_TOKEN="$token" CHAT_REF="$CHAT_MODEL" \
    "$VENV/bin/python" - > "$proj/home/.config/opencode/opencode.json") <<'PY'
import json
import os

from lilbee.cli.agent_configs.opencode import opencode_config
from lilbee.cli.launchers.server import client_chat_ctx

port = int(os.environ["LILBEE_PORT"])
block = opencode_config(
    base_url=f"http://127.0.0.1:{port}",
    api_key=os.environ["LILBEE_TOKEN"],
    model_refs=[os.environ["CHAT_REF"]],
    chat_ctx=client_chat_ctx(port),
    default_ref=os.environ["CHAT_REF"],
    include_mcp=False,
)
block.setdefault("autoupdate", False)
print(json.dumps(block, indent=2))
PY
}

run_agent() { # run_agent <proj-index>
  local proj="$AGENTS_ROOT/proj$1"
  (
    cd "$proj"
    HOME="$proj/home" PATH="$PATH" timeout "$AGENT_TIMEOUT_S" \
      opencode run "$(cat prompt.txt)" > "$RESULTS/agent$1.log" 2>&1
    echo $? > "$RESULTS/agent$1.exit"
  )
}

stop_server() { # stop_server <proj-index>: the product's shutdown path, TERM fallback
  local proj="$AGENTS_ROOT/proj$1"
  local port token
  port=$(cat "$proj/.lilbee/data/server.port" 2>/dev/null)
  token=$(server_token "$1")
  if [ -n "$port" ] && [ -n "$token" ]; then
    curl -sf -X POST -H "Authorization: Bearer $token" \
      "http://127.0.0.1:$port/api/shutdown" >/dev/null 2>&1 && return
  fi
  kill -TERM "$(cat "$RESULTS/serve$1.pid" 2>/dev/null)" 2>/dev/null
}

verify_task() { # verify_task <proj-index>
  local proj="$AGENTS_ROOT/proj$1"
  case $1 in
    1) (cd "$proj" && "$VENV/bin/python" -m pytest -q task1_slugify_test.py >> "$RESULTS/verify.log" 2>&1);;
    2) (cd "$proj" && "$VENV/bin/python" -m pytest -q task2_window_test.py >> "$RESULTS/verify.log" 2>&1);;
    3) grep -q "engine_idle_ttl_minutes" "$proj/harness_answer.md" 2>/dev/null;;
    4) (cd "$proj" && "$VENV/bin/python" -m pytest -q task4_dupes_test.py >> "$RESULTS/verify.log" 2>&1);;
  esac
}

# ── Phase 1: baseline, one agent ────────────────────────────────────
note "phase 1: single-agent baseline (proj1)"
start_server 1; check "server 1 healthy" $?
wire_opencode 1
BASE_T0=$(date +%s)
run_agent 1
BASE_T1=$(date +%s)
BASELINE_VRAM=$(vram_used_mb)
note "baseline: $((BASE_T1 - BASE_T0))s, VRAM=${BASELINE_VRAM}MB, exit=$(cat "$RESULTS/agent1.exit" 2>/dev/null)"
verify_task 1; check "baseline task 1 verified" $?
rm -f "$AGENTS_ROOT/proj1/slugify_impl.py"

# ── Phase 2: 4-agent load (server 1 stays; 2-4 must BIND, not build) ─
note "phase 2: four agents concurrently"
for i in 2 3 4; do start_server "$i"; check "server $i healthy" $?; wire_opencode "$i"; done
POST_BIND_SERVERS=$(engine_llama_servers)
POST_BIND_SWAPS=$(engine_swaps)
note "after 4 servers up: llama-servers=$POST_BIND_SERVERS swaps=$POST_BIND_SWAPS locks=$(user_locks)"

LOAD_T0=$(date +%s)
for i in 1 2 3 4; do run_agent "$i" & done
sleep 90
MID_SERVERS=$(engine_llama_servers)
MID_SWAPS=$(engine_swaps)
MID_VRAM=$(vram_used_mb)
MID_LOCKS=$(user_locks)
note "mid-load: llama-servers=$MID_SERVERS swaps=$MID_SWAPS VRAM=${MID_VRAM}MB user-locks=$MID_LOCKS"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv >> "$RESULTS/midload-gpu.txt"
ls -la "$LILBEE_ENGINE_DIR" "$LILBEE_ENGINE_DIR/engine-users" >> "$RESULTS/midload-slot.txt" 2>&1
wait
LOAD_T1=$(date +%s)
note "load phase: $((LOAD_T1 - LOAD_T0))s total; exits: $(cat "$RESULTS"/agent{1,2,3,4}.exit 2>/dev/null | tr '\n' ' ')"

# ── Phase 3: assertions ─────────────────────────────────────────────
note "phase 3: assertions"
[ "$MID_SWAPS" -ge 1 ] && [ "$MID_SWAPS" -le 2 ]; check "one engine: llama-swap proxies 1..2 (got $MID_SWAPS)" $?
[ "$MID_SERVERS" -ge 1 ] && [ "$MID_SERVERS" -le 2 ]; check "one engine: llama-server count 1..2 (got $MID_SERVERS)" $?
[ "$POST_BIND_SWAPS" -le 2 ]; check "servers 2..4 bound instead of building (swaps stayed $POST_BIND_SWAPS)" $?
[ "$MID_LOCKS" -ge 4 ]; check "membership: >=4 user locks mid-load (got $MID_LOCKS)" $?
python3 - "$BASELINE_VRAM" "$MID_VRAM" <<'PY'
import sys

base, mid = float(sys.argv[1]), float(sys.argv[2])
sys.exit(0 if mid <= base * 1.15 else 1)
PY
check "VRAM flat: mid=${MID_VRAM}MB <= 1.15x baseline=${BASELINE_VRAM}MB" $?

for i in 1 2 3 4; do verify_task "$i"; check "task $i verified" $?; done

! grep -h "Server exited\|kept exiting\|Traceback" "$RESULTS"/serve*.log 2>/dev/null | grep -q .
check "no server crashes during the run" $?

# ── Phase 4: clean exit ─────────────────────────────────────────────
note "phase 4: teardown (SIGTERM all four servers; last out must stop the engine)"
for i in 1 2 3 4; do stop_server "$i"; done
sleep "$SETTLE_S"
END_SERVERS=$(engine_llama_servers || true)
END_LOCKS=$(user_locks)
[ "${END_SERVERS:-0}" -eq 0 ]; check "engine stopped after last exit (llama-servers=${END_SERVERS:-0})" $?
[ "${END_LOCKS:-0}" -eq 0 ]; check "engine-users empty after teardown (got $END_LOCKS)" $?

note "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
