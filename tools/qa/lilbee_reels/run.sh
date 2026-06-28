#!/usr/bin/env bash
# Record the agent-launcher reels on the LILBEE codebase, on the pod.
# Indexes the lilbee source, pulls one modern coder model, then records the
# opencode reel (search the lilbee source + make a real code change) with VHS.
# Writes /workspace/reels-out/ and a DONE marker so the driver can tear down.
set -uo pipefail

source /workspace/qa_env.sh 2>/dev/null || true
REPO=/workspace/lilbee
REEL_MODEL="${REEL_MODEL:?set REEL_MODEL}"
OUT=/workspace/reels-out
DATA=/workspace/.lilbee-reels
PORT=41750
mkdir -p "$OUT"
export LILBEE_DATA="$DATA"
export LILBEE_CHAT_MODEL="$REEL_MODEL"
# opencode's system prompt + MCP tool defs are ~28K tokens; the default served
# window (24576 on a big-RAM host) overflows. Qwen3-Coder-Next is 256K-capable and
# the 80GB card has KV headroom, so serve a generous 64K window.
export LILBEE_CHAT_N_CTX_TARGET=65536
export COLORTERM=truecolor TERM=xterm-256color
log(){ echo "[reels $(date +%H:%M:%S)] $*"; }

# 1. Pull the reel model (cached on the volume across retakes).
log "pulling reel model $REEL_MODEL"
uv run lilbee model pull "$REEL_MODEL" >>"$OUT/pull.log" 2>&1 || { log "model pull FAILED"; tail -5 "$OUT/pull.log"; }

# 2. Index the lilbee source so lilbee_search has real code to ground on.
log "indexing lilbee source"
( cd "$REPO" && uv run lilbee add src/lilbee AGENTS.md >>"$OUT/index.log" 2>&1 ) || log "index issues (see index.log)"

# 3. Pre-seed opencode's global config: edit:allow (no human to approve the write
#    in an automated reel) + no autoupdate. The launcher's non-destructive merge
#    preserves these and adds the lilbee provider on top.
mkdir -p "$HOME/.config/opencode"
cat > "$HOME/.config/opencode/opencode.json" <<'OC'
{ "$schema": "https://opencode.ai/config.json",
  "theme": "rose-pine",
  "permission": { "edit": "allow" },
  "tools": { "webfetch": false },
  "autoupdate": false }
OC
# Pre-accept the launcher's first-run consent at the EXACT cfg.data_dir (a shell
# guess at the path misses it, and the prompt then eats the typed reel prompt).
( cd "$REPO" && LILBEE_DATA="$DATA" uv run python -c "
from lilbee.core.config import cfg
import json
m = cfg.data_dir / 'launchers' / 'opencode-setup.json'
m.parent.mkdir(parents=True, exist_ok=True)
m.write_text(json.dumps({'accepted': True}))
print('setup marker ->', m)
" )

# 4. Pre-warm a serve so the launch is instant and the answer is the subject.
log "warming serve on $PORT"
pkill -f 'lilbee serve' 2>/dev/null; pkill -f 'llama-server' 2>/dev/null; pkill -f 'llama-swap' 2>/dev/null; sleep 2
tmux kill-session -t warmserve 2>/dev/null
tmux new-session -d -s warmserve "bash -c 'source /workspace/qa_env.sh; export LILBEE_DATA=$DATA LILBEE_CHAT_MODEL=\"$REEL_MODEL\" LILBEE_CHAT_N_CTX_TARGET=65536; cd $REPO; lilbee serve --port $PORT 2>&1 | tee $OUT/serve.log; sleep 7200'"
for _ in $(seq 1 240); do
  curl -s "http://127.0.0.1:$PORT/api/health" 2>/dev/null | grep -q '"chat_ready":true' && break
  sleep 10
done
curl -s "http://127.0.0.1:$PORT/api/health" | grep -q '"chat_ready":true' || { log "warm FAILED"; tail -8 "$OUT/serve.log"; echo "REELS_FAILED warm" > "$OUT/DONE"; exit 3; }
log "warm ready"

# 5. The reel tape: launch opencode (reuses the warm serve), ask one prompt that
#    forces a lilbee_search over the lilbee source AND a real code change.
PROMPT="Using my indexed lilbee source, briefly explain how lilbee registers itself into opencode's config (cite the files). Then add a 'lilbee launch --list' subcommand that prints the supported agent names, reading the real launcher module to match its style. Add a focused test for it under tests/cli/ and run ONLY that one test file with 'uv run pytest <thatfile> -q' to confirm it passes green. Do NOT run the full suite or 'make check'. Keep it minimal and do not ask me questions."
WS="$REPO"
cat > "$OUT/opencode.tape" <<TAPE
Output opencode.gif
Output opencode.mp4
Set Shell bash
Set Width 1600
Set Height 900
Set FontSize 14
Set Theme { "name": "rose-pine", "background": "#191724", "foreground": "#e0def4", "cursor": "#e0def4", "selection": "#403d52", "black": "#26233a", "red": "#eb6f92", "green": "#9ccfd8", "yellow": "#f6c177", "blue": "#31748f", "magenta": "#c4a7e7", "cyan": "#ebbcba", "white": "#e0def4", "brightBlack": "#6e6a86", "brightRed": "#eb6f92", "brightGreen": "#9ccfd8", "brightYellow": "#f6c177", "brightBlue": "#31748f", "brightMagenta": "#c4a7e7", "brightCyan": "#ebbcba", "brightWhite": "#e0def4" }
Env PATH "/root/lilbee_venv/bin:/root/.opencode/bin:/usr/local/go/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Env LILBEE_DATA "$DATA"
Env LILBEE_CHAT_MODEL "$REEL_MODEL"
Env COLORTERM "truecolor"
Sleep 2s
Type "cd $WS && lilbee launch opencode -y"
Sleep 500ms
Enter
Sleep 16s
Type "$PROMPT"
Sleep 1s
Enter
Sleep 175s
Screenshot opencode.png
TAPE

log "recording opencode reel"
# VHS must run from the tape's dir with RELATIVE Output paths (absolute trips its
# parser on the pod); the tape itself cd's into the repo for the commands.
( cd "$OUT" && VHS_NO_SANDBOX=true vhs opencode.tape > "$OUT/vhs-opencode.log" 2>&1 ) \
  && log "opencode reel recorded" \
  || { log "vhs FAILED"; tail -8 "$OUT/vhs-opencode.log"; echo "REELS_FAILED vhs" > "$OUT/DONE"; exit 4; }

# 6. Extract unique frames for review (frame-by-frame QA happens off-pod).
mkdir -p "$OUT/frames-opencode"
ffmpeg -y -loglevel error -i "$OUT/opencode.mp4" -vf fps=1 "$OUT/frames-opencode/f_%04d.png" 2>/dev/null || true

echo "REELS_DONE opencode $(date -u +%FT%TZ)" > "$OUT/DONE"
log "DONE (opencode). reels in $OUT"
