#!/usr/bin/env bash
# Record the HERMES agent-launcher reel on the lilbee codebase. Installs
# hermes-agent on the pod, reuses the cached model/engine/index on the volume,
# then records `lilbee launch hermes` doing real work on the lilbee source.
set -uo pipefail

source /workspace/qa_env.sh 2>/dev/null || true
REPO=/workspace/lilbee
REEL_MODEL="${REEL_MODEL:?set REEL_MODEL}"
OUT=/workspace/reels-out
DATA=/workspace/.lilbee-reels
HERMES_DIR=/workspace/hermes-agent
PORT=41751
mkdir -p "$OUT"
export LILBEE_DATA="$DATA" LILBEE_CHAT_MODEL="$REEL_MODEL" LILBEE_CHAT_N_CTX_TARGET=65536
export COLORTERM=truecolor TERM=xterm-256color
log(){ echo "[hermes-reel $(date +%H:%M:%S)] $*"; }

# 1. Install hermes-agent (clone + uv sync) and a `hermes` PATH shim so the
#    launcher's shutil.which("hermes") resolves it. ripgrep is a hermes dep.
log "installing hermes-agent"
command -v rg >/dev/null || (apt-get update -qq && apt-get install -y -qq ripgrep) >>"$OUT/hermes-install.log" 2>&1
[ -d "$HERMES_DIR/.git" ] || git clone --depth 1 https://github.com/NousResearch/hermes-agent "$HERMES_DIR" >>"$OUT/hermes-install.log" 2>&1
# CRITICAL: unset UV_PROJECT_ENVIRONMENT (qa_env points it at lilbee's venv) so
# hermes syncs into its OWN .venv instead of clobbering lilbee's.
( cd "$HERMES_DIR" && env -u UV_PROJECT_ENVIRONMENT uv sync >>"$OUT/hermes-install.log" 2>&1 ) || log "hermes uv sync issues (see hermes-install.log)"
cat > /usr/local/bin/hermes <<WRAP
#!/usr/bin/env bash
exec env -u UV_PROJECT_ENVIRONMENT uv run --project $HERMES_DIR hermes "\$@"
WRAP
chmod +x /usr/local/bin/hermes
command -v hermes >/dev/null && log "hermes shim ready: $(command -v hermes)" || { log "hermes NOT on PATH"; echo "REELS_FAILED hermes-install" > "$OUT/DONE-hermes"; exit 2; }

# 2. Pre-seed ~/.hermes so the launcher's non-destructive merge adds lilbee on top
#    and hermes opens straight to the TUI (no setup wizard). Mark onboarding seen.
mkdir -p "$HOME/.hermes"
cat > "$HOME/.hermes/config.yaml" <<'HCFG'
display:
  interface: tui
onboarding:
  seen:
    welcome: true
HCFG

# 3. Index is cached on the volume from the opencode run; ensure it exists.
( cd "$REPO" && uv run lilbee add src/lilbee AGENTS.md >>"$OUT/index.log" 2>&1 ) || log "index issues"

# 4. Warm a serve on the cached model.
log "warming serve on $PORT"
pkill -f 'lilbee serve' 2>/dev/null; pkill -f 'llama-server' 2>/dev/null; pkill -f 'llama-swap' 2>/dev/null; sleep 2
tmux kill-session -t warmh 2>/dev/null
tmux new-session -d -s warmh "bash -c 'source /workspace/qa_env.sh; export LILBEE_DATA=$DATA LILBEE_CHAT_MODEL=\"$REEL_MODEL\" LILBEE_CHAT_N_CTX_TARGET=65536; cd $REPO; lilbee serve --port $PORT 2>&1 | tee $OUT/serve-hermes.log; sleep 7200'"
for _ in $(seq 1 240); do
  curl -s "http://127.0.0.1:$PORT/api/health" 2>/dev/null | grep -q '"chat_ready":true' && break
  sleep 10
done
curl -s "http://127.0.0.1:$PORT/api/health" | grep -q '"chat_ready":true' || { log "warm FAILED"; tail -8 "$OUT/serve-hermes.log"; echo "REELS_FAILED warm" > "$OUT/DONE-hermes"; exit 3; }
log "warm ready"

# 5. Record the hermes reel: launch hermes (registers lilbee in ~/.hermes), ask a
#    lilbee-codebase question + a small code change.
PROMPT="Using my indexed lilbee source, explain how lilbee's fleet decides which GPU a model loads on, citing the files. Then add a small unit test under tests/ that verifies prune_lilbee removes the lilbee MCP entry while keeping sibling servers. Read the real code, keep it minimal, and do not ask me questions."
cat > "$OUT/hermes.tape" <<TAPE
Output hermes.gif
Output hermes.mp4
Set Shell bash
Set Width 1600
Set Height 900
Set FontSize 14
Set Theme { "name": "rose-pine", "background": "#191724", "foreground": "#e0def4", "cursor": "#e0def4", "selection": "#403d52", "black": "#26233a", "red": "#eb6f92", "green": "#9ccfd8", "yellow": "#f6c177", "blue": "#31748f", "magenta": "#c4a7e7", "cyan": "#ebbcba", "white": "#e0def4", "brightBlack": "#6e6a86", "brightRed": "#eb6f92", "brightGreen": "#9ccfd8", "brightYellow": "#f6c177", "brightBlue": "#31748f", "brightMagenta": "#c4a7e7", "brightCyan": "#ebbcba", "brightWhite": "#e0def4" }
Env PATH "/root/lilbee_venv/bin:/usr/local/bin:/root/.local/bin:/usr/local/go/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
Env LILBEE_DATA "$DATA"
Env LILBEE_CHAT_MODEL "$REEL_MODEL"
Env COLORTERM "truecolor"
Sleep 2s
Type "cd $REPO && lilbee launch hermes"
Sleep 500ms
Enter
Sleep 22s
Type "$PROMPT"
Sleep 1s
Enter
Sleep 140s
Screenshot hermes.png
TAPE

log "recording hermes reel"
( cd "$OUT" && VHS_NO_SANDBOX=true vhs hermes.tape > "$OUT/vhs-hermes.log" 2>&1 ) \
  && log "hermes reel recorded" \
  || { log "vhs FAILED"; tail -8 "$OUT/vhs-hermes.log"; echo "REELS_FAILED vhs" > "$OUT/DONE-hermes"; exit 4; }

mkdir -p "$OUT/frames-hermes"
ffmpeg -y -loglevel error -i "$OUT/hermes.mp4" -vf fps=1 "$OUT/frames-hermes/f_%04d.png" 2>/dev/null || true
echo "REELS_DONE hermes $(date -u +%FT%TZ)" > "$OUT/DONE-hermes"
log "DONE (hermes)."
