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
# --extra mcp: hermes ships HTTP MCP transport behind the optional `mcp` extra;
# without it `lilbee_search` shows "0 connected".
( cd "$HERMES_DIR" && env -u UV_PROJECT_ENVIRONMENT uv sync --extra mcp >>"$OUT/hermes-install.log" 2>&1 ) || log "hermes uv sync issues (see hermes-install.log)"
cat > /usr/local/bin/hermes <<WRAP
#!/usr/bin/env bash
exec env -u UV_PROJECT_ENVIRONMENT -u VIRTUAL_ENV uv run --project $HERMES_DIR hermes "\$@"
WRAP
chmod +x /usr/local/bin/hermes
command -v hermes >/dev/null && log "hermes shim ready: $(command -v hermes)" || { log "hermes NOT on PATH"; echo "REELS_FAILED hermes-install" > "$OUT/DONE-hermes"; exit 2; }

# hermes builds its TUI deps on the FIRST TUI launch ("Installing TUI dependencies").
# Trigger that now (cached on the volume) so the reel's `lilbee launch hermes` opens
# instantly instead of burning the launch window on the build.
log "pre-building hermes TUI deps"
timeout 300 hermes </dev/null >>"$OUT/hermes-install.log" 2>&1 || true
pkill -f "hermes_cli" 2>/dev/null; sleep 2
log "hermes TUI deps ready"

# 2. Pre-seed ~/.hermes so the launcher's non-destructive merge adds lilbee on top
#    and hermes opens straight to the TUI (no setup wizard). Mark onboarding seen.
mkdir -p "$HOME/.hermes"
# Custom rose-pine skin: the bordered response box + colored tool names give the
# same visual structure opencode has (mono is monochrome, so the structure blends
# into a flat, hard-to-follow stream). Foreground is rose-pine; the bg stays the
# rose-pine VHS terminal, so the whole TUI reads rose-pine and legible.
mkdir -p "$HOME/.hermes/skins"
cat > "$HOME/.hermes/skins/lilbee-rose-pine.yaml" <<'SKIN'
name: lilbee-rose-pine
description: lilbee rose-pine
colors:
  banner_border: "#c4a7e7"
  banner_title: "#f6c177"
  banner_accent: "#ebbcba"
  banner_dim: "#6e6a86"
  banner_text: "#e0def4"
  ui_accent: "#c4a7e7"
  ui_label: "#9ccfd8"
  ui_ok: "#9ccfd8"
  ui_error: "#eb6f92"
  ui_warn: "#f6c177"
  prompt: "#e0def4"
  input_rule: "#c4a7e7"
  response_border: "#c4a7e7"
  status_bar_bg: "#1f1d2e"
  status_bar_text: "#e0def4"
  status_bar_strong: "#c4a7e7"
  status_bar_dim: "#6e6a86"
  status_bar_good: "#9ccfd8"
  status_bar_warn: "#f6c177"
  status_bar_bad: "#eb6f92"
  status_bar_critical: "#eb6f92"
  session_label: "#9ccfd8"
  session_border: "#403d52"
branding:
  agent_name: "Hermes Agent"
  response_label: " ⚕ Hermes "
  prompt_symbol: "❯"
tool_prefix: "┊"
SKIN
cat > "$HOME/.hermes/config.yaml" <<'HCFG'
display:
  interface: tui
  skin: lilbee-rose-pine
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
PROMPT="Add a 'lilbee launch list' subcommand that prints the agents you can launch, reading the launcher registry in src/lilbee/cli/launchers/__init__.py. Run it to make sure it works, then add a focused test under tests/cli/ and run just that test."
# Same verified rose-pine recipe as the opencode tape (named theme + window chrome).
cat > "$OUT/hermes.tape" <<TAPE
Output hermes.gif
Output hermes.mp4
Set Shell bash
Set Width 1400
Set Height 900
Set FontSize 18
Set Padding 20
Set Theme "rose-pine"
Set Margin 30
Set MarginFill "#100f1a"
Set BorderRadius 10
Set WindowBar Colorful
Env PATH "/root/lilbee_venv/bin:/usr/local/bin:/root/.local/bin:/usr/local/go/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
Env LILBEE_DATA "$DATA"
Env LILBEE_CHAT_MODEL "$REEL_MODEL"
Env COLORTERM "truecolor"
Sleep 2s
Type "cd $REPO && lilbee launch hermes"
Sleep 500ms
Enter
Sleep 30s
Type "$PROMPT"
Sleep 1s
Enter
Sleep 1s
Enter
Sleep 270s
Screenshot hermes.png
TAPE

log "recording hermes reel"
( cd "$OUT" && VHS_NO_SANDBOX=true vhs hermes.tape > "$OUT/vhs-hermes.log" 2>&1 ) \
  && log "hermes reel recorded" \
  || { log "vhs FAILED"; tail -8 "$OUT/vhs-hermes.log"; echo "REELS_FAILED vhs" > "$OUT/DONE-hermes"; exit 4; }

mkdir -p "$OUT/frames-hermes"
ffmpeg -y -loglevel error -i "$OUT/hermes.mp4" -vf fps=1 "$OUT/frames-hermes/f_%04d.png" 2>/dev/null || true
BG=$(ffmpeg -v error -i "$OUT/hermes.png" -vf "crop=2:2:700:650,scale=1:1" -f rawvideo -pix_fmt rgb24 - 2>/dev/null | xxd -p | head -c6)
log "hermes reel bg hex => #$BG  (rose-pine base ~#1c1c2c)"
echo "REELS_DONE hermes bg=#$BG $(date -u +%FT%TZ)" > "$OUT/DONE-hermes"
log "DONE (hermes)."
