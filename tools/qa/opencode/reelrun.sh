#!/usr/bin/env bash
# Reel factory: record one validated demo reel for <family> <ref> <tier>.
# Produces /workspace/reelfactory/<family>/ with gif+mp4+png+tape+frames/
# (unique frames only, timestamped) ready for frame-by-frame review.
set -eo pipefail

FAMILY="${1:?usage: reelrun.sh <family> <ref> <small|mid|coder|giant>}"
REF="${2:?ref}"
TIER="${3:?tier}"

source /workspace/qa_env.sh
QA=/workspace/lilbee/tools/qa/opencode
WS="$QA/workspace/$FAMILY"
OUT="/workspace/reelfactory/$FAMILY"
PORT=41700
mkdir -p "$OUT"

case "$TIER" in
  small) GEN_SLEEP=90;  PROMPT="Search the indexed Godot 4 class reference for the AStarGrid2D class and tell me, citing what the search returns, exactly what its get_id_path method returns." ;;
  mid)   GEN_SLEEP=120; PROMPT="In Godot 4 I am connecting signals between nodes. What is the exact signature of Object.connect, and what do the CONNECT_DEFERRED and CONNECT_ONE_SHOT flags do? Verify against my indexed reference and include their integer values." ;;
  coder) GEN_SLEEP=200; PROMPT="write level_generator.gd: a procedural level generator that places wall and floor tiles and scatters collectibles using pathfinding. Verify every Godot API you use against my indexed reference. Pick sensible defaults yourself and never ask me questions." ;;
  giant) GEN_SLEEP=280; PROMPT="write level_generator.gd: a procedural level generator that places wall and floor tiles and scatters collectibles using pathfinding. Verify every Godot API you use against my indexed reference. Pick sensible defaults yourself and never ask me questions." ;;
  *) echo "bad tier"; exit 2 ;;
esac

echo "[reelrun] ensuring $REF is registered"
lilbee model pull "$REF" >/dev/null 2>&1 || lilbee model pull "$REF"

echo "[reelrun] rebuilding workspace"
python3 - <<PYEOF
import sys
sys.path.insert(0, "$QA")
from workspace import write_per_cell_workspace
ws = write_per_cell_workspace("$FAMILY", "$REF")
import json
cfg = {"\$schema": "https://opencode.ai/config.json",
       "tools": {"webfetch": False, "bash": False, "question": False},
       "autoupdate": False}
(ws / "opencode.json").write_text(json.dumps(cfg, indent=2))
print("workspace ready:", ws)
PYEOF

echo "[reelrun] warming serve"
tmux kill-session -t warmserve 2>/dev/null || true
pkill -f 'lilbee serve' 2>/dev/null || true
pkill -f 'llama-server' 2>/dev/null || true
pkill -f 'llama-swap' 2>/dev/null || true
sleep 2
tmux new-session -d -s warmserve "bash -c 'source /workspace/qa_env.sh; export LILBEE_DATA=$WS/.lilbee; cd $WS; lilbee serve --port $PORT 2>&1 | tee $OUT/serve.log; sleep 7200'"
for _ in $(seq 1 240); do
  curl -s "http://127.0.0.1:$PORT/api/health" 2>/dev/null | grep -q '"chat_ready":true' && break
  sleep 10
done
curl -s "http://127.0.0.1:$PORT/api/health" | grep -q '"chat_ready":true' || { echo "[reelrun] warm FAILED"; exit 3; }
echo "[reelrun] warm ready"

echo "[reelrun] writing tape"
cat > "$WS/reel.tape" <<TAPE
Output reels-out/$FAMILY.gif
Output reels-out/$FAMILY.mp4

Set Shell bash
Set Width 1600
Set Height 900
Set FontSize 14
Set PlaybackSpeed 1.0

Env PATH "/root/lilbee_venv/bin:/root/.opencode/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Env LILBEE_DATA "$WS/.lilbee"
Env LILBEE_MODELS_DIR "/workspace/models"
Env HF_HOME "/workspace/hf"

Sleep 2s

Type "lilbee launch opencode"
Sleep 500ms
Enter
Sleep 25s

Type "$PROMPT"
Sleep 1s
Enter

Sleep ${GEN_SLEEP}s
Screenshot reels-out/$FAMILY.png
TAPE

echo "[reelrun] recording"
mkdir -p "$WS/reels-out"
( cd "$WS" && VHS_NO_SANDBOX=true vhs reel.tape > "$OUT/vhs.log" 2>&1 ) || { echo "[reelrun] vhs FAILED"; tail -5 "$OUT/vhs.log"; exit 4; }
cp "$WS/reels-out/$FAMILY".* "$OUT/" 2>/dev/null || true
cp "$WS/reel.tape" "$OUT/"

if [ "$TIER" = "coder" ] || [ "$TIER" = "giant" ]; then
  echo "[reelrun] verifying the agent wrote the file"
  GD="$(find "$WS" -name 'level_generator.gd' -not -path '*/reels-out/*' | head -1)"
  if [ -z "$GD" ]; then
    echo "[reelrun] ARTIFACT MISSING: level_generator.gd was never written"
    exit 5
  fi
  SIZE=$(stat -c %s "$GD")
  if [ "$SIZE" -lt 1000 ] || ! grep -qE 'extends|func ' "$GD"; then
    echo "[reelrun] ARTIFACT INVALID: $GD ($SIZE bytes) lacks real GDScript"
    exit 5
  fi
  cp "$GD" "$OUT/level_generator.gd"
  echo "[reelrun] artifact OK: $GD ($SIZE bytes); head:"
  head -15 "$GD"
fi

echo "[reelrun] extracting unique frames"
mkdir -p "$OUT/frames"
rm -f "$OUT/frames"/*.png
ffmpeg -y -loglevel error -i "$OUT/$FAMILY.mp4" -vf fps=1 "$OUT/frames/raw_%04d.png"
python3 - <<PYEOF
import hashlib, os, glob
prev = None
kept = 0
for f in sorted(glob.glob("$OUT/frames/raw_*.png")):
    h = hashlib.md5(open(f, "rb").read()).hexdigest()
    sec = int(f.split("raw_")[1].split(".")[0]) - 1
    if h == prev:
        os.unlink(f)
        continue
    prev = h
    kept += 1
    os.rename(f, f"$OUT/frames/t{sec:04d}.png")
print(f"unique frames kept: {kept}")
PYEOF

tar czf "/workspace/reelfactory/$FAMILY-frames.tgz" -C "$OUT" frames
echo "[reelrun] DONE $FAMILY -> $OUT"
