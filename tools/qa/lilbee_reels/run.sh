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
mkdir -p "$HOME/.config/opencode/themes"
# opencode 1.17.1's built-in "rose-pine" renders a near-black panel; ship an
# explicit rose-pine theme (purple #191724 base) so opencode matches the existing
# rose-pine agent demos. Verified to render bg ~#191724 (vs the built-in #0a0a0a).
cat > "$HOME/.config/opencode/themes/lilbee-rose-pine.json" <<'THEME'
{
  "$schema": "https://opencode.ai/theme.json",
  "defs": {
    "base": "#191724", "surface": "#1f1d2e", "overlay": "#26233a",
    "muted": "#6e6a86", "subtle": "#908caa", "text": "#e0def4",
    "love": "#eb6f92", "gold": "#f6c177", "rose": "#ebbcba",
    "pine": "#31748f", "foam": "#9ccfd8", "iris": "#c4a7e7", "highlightMed": "#403d52"
  },
  "theme": {
    "primary": "iris", "secondary": "foam", "accent": "rose",
    "error": "love", "warning": "gold", "success": "pine", "info": "foam",
    "text": "text", "textMuted": "muted",
    "background": "base", "backgroundPanel": "surface", "backgroundElement": "overlay",
    "border": "highlightMed", "borderActive": "iris", "borderSubtle": "overlay",
    "syntaxComment": "muted", "syntaxKeyword": "pine", "syntaxFunction": "rose",
    "syntaxVariable": "text", "syntaxString": "gold", "syntaxNumber": "gold",
    "syntaxType": "foam", "syntaxOperator": "subtle", "syntaxPunctuation": "subtle"
  }
}
THEME
cat > "$HOME/.config/opencode/opencode.json" <<'OC'
{ "$schema": "https://opencode.ai/config.json",
  "theme": "lilbee-rose-pine",
  "permission": { "edit": "allow" },
  "tools": { "webfetch": false },
  "autoupdate": false }
OC
# opencode persists the active theme in a state db that overrides the config file;
# clear it so the custom theme applies cleanly on this run.
rm -f "$HOME/.local/share/opencode/opencode.db"* 2>/dev/null || true
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
PROMPT="Add a 'lilbee launch list' subcommand that prints the agents you can launch, reading the launcher registry in src/lilbee/cli/launchers/__init__.py. Run it to make sure it works, then add a focused test under tests/cli/ and run just that test."
WS="$REPO"
# Verified rose-pine recipe: VHS's BUILT-IN named theme (an inline JSON theme
# silently falls back to gray), plus the macOS window chrome the existing agent
# demos use. Matches demos/mcp-code.png (bg ~#1c1c2c).
cat > "$OUT/opencode.tape" <<TAPE
Output opencode.gif
Output opencode.mp4
Set Shell bash
Set Width 1400
Set Height 900
Set FontSize 14
Set Padding 20
Set Theme "rose-pine"
Set Margin 30
Set MarginFill "#100f1a"
Set BorderRadius 10
Set WindowBar Colorful
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
Sleep 270s
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

# Sanity: log the reel's background hex (rose-pine base is ~#1c1c2c, purple-tinted;
# a neutral gray like #131313 means the theme didn't apply). PIL-free via ffmpeg.
BG=$(ffmpeg -v error -i "$OUT/opencode.png" -vf "crop=2:2:700:650,scale=1:1" -f rawvideo -pix_fmt rgb24 - 2>/dev/null | xxd -p | head -c6)
log "opencode reel bg hex => #$BG  (rose-pine base ~#1c1c2c)"
echo "REELS_DONE opencode bg=#$BG $(date -u +%FT%TZ)" > "$OUT/DONE"
log "DONE (opencode). reels in $OUT"
