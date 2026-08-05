#!/bin/sh
# Drive ONE Wave 1 take end to end against the lingering session pod.
# Usage: w1_take.sh <reel> <host> <port>
# Reads kit/reel_bodies/<reel>.body.tape + pod timings.json, builds the tape
# (provenance header, ExtraBold class block, measured windows), records via
# ssh, white-gates on the pod, pulls, finishes gif locally, runs acceptance.
set -u
REEL="$1"; HOST="$2"; PORT="$3"; WSDIR="${4:-w1}"
KIT="$(cd "$(dirname "$0")" && pwd)"
OUT=$HOME/Desktop/lilbee-reels-review/wave1
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
mkdir -p "$OUT"
remote() {
  ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=15 -o ServerAliveInterval=10 "root@$HOST" "$1"
}
remote "cat /workspace/$WSDIR/timings.json" > "$OUT/timings.json" 2>/dev/null
[ -s "$OUT/timings.json" ] || { echo "NO TIMINGS (probe not done?)"; exit 3; }

python3 - "$REEL" "$KIT" "$OUT" <<'PY' > "$OUT/$REEL.tape" || exit 4
import json, sys, datetime, pathlib
reel, kit, out = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
t = json.load(open(f"{out}/timings.json"))
# per-reel: body file, gen slot in timings, visible lead seconds before hidden remainder
CFG = {
    "tui-chat": {"gen_slot": "gen_jump_s", "index_lead": 6,  "gen_lead": None},
    "tui-add":  {"gen_slot": "gen_oil_s",  "index_lead": 8,  "gen_lead": 4},
    "tui-tour": {"gen_slot": "gen_oil_s",  "index_lead": None, "gen_lead": 50},
    "tui-crawl": {"gen_slot": "gen_crawl_s", "index_lead": None, "gen_lead": 4, "crawl_lead": 3,
                  "config": "LITE: Qwen3-8B Q8 + nomic-embed-text-v1.5"},
    "what-is-lilbee": {"gen_slot": "gen_whatis_s", "index_lead": 6, "gen_lead": None,
                       "boot_floor": 40, "gen_floor": 40, "chat_pull": "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf", "config": "LITE: Qwen3-4B + nomic (matches the original)"},
    "tui-palette":  {"sweep": True, "boot_floor": 45, "config": "LITE: Qwen3-8B (UI sweep)"},
    "tui-settings": {"sweep": True, "boot_floor": 45, "config": "LITE: Qwen3-8B (UI sweep)"},
    "tui-unsupported": {"sweep": True, "boot_floor": 45, "search_rem": 15, "config": "LITE: HF search (UI sweep)"},
}
c = CFG[reel]
boot = max(t["chat_ready_s"] * 5 // 4 + 15, c.get("boot_floor", 0))
subs = {"__BOOT__": str(boot), "__REEL__": reel}
if c.get("sweep"):
    if c.get("search_rem"):
        subs["__SEARCHREM__"] = str(c["search_rem"])
    gen_total = 0
else:
    gen_total = t.get(c["gen_slot"]) or 0
if c.get("sweep"):
    pass
elif c.get("index_full"):
    subs["__INDEXREM__"] = str(t.get("index_s", 15) * 13 // 10 + 5)
elif c["index_lead"] is not None:
    subs["__INDEXREM__"] = str(max(t.get("index_s", 55) - c["index_lead"], 5) * 13 // 10 + 5)
if c.get("crawl_lead") is not None:
    subs["__CRAWLREM__"] = str(max(t.get("crawl_s", 20) - c["crawl_lead"], 3) * 13 // 10 + 5)
if not c.get("sweep"):
    if c["gen_lead"] is None:
        subs["__GEN__"] = str(max(gen_total * 2 + 15, c.get("gen_floor", 0)))  # generous; trim_tail cuts the static tail, so never cut a cold-load answer short
    else:
        subs["__GENREM__"] = str(max(gen_total - c["gen_lead"], 3) * 13 // 10 + 5)
body = (kit / "reel_bodies" / f"{reel}.body.tape").read_text()
for k, v in subs.items():
    body = body.replace(k, v)
assert "__" not in body, f"unfilled placeholder in {reel} body"
print(f'''# HW: {t["gpu"]} | Config {c.get("config", "MAX: Qwen3.6-35B-A3B Q8 + Qwen3-Embedding-8B Q8")} | ExtraBold, temp 0, seed 42
# Build: lilbee {t["build"]} cu124 | Recorded: {datetime.date.today()}
# Measured (probe this pod): boot {t["chat_ready_s"]}s {c.get("gen_slot","(sweep)")} {gen_total}s -- windows derived, see timings.json
Set Shell "bash"
Set FontFamily "JetBrains Mono ExtraBold, Symbols Nerd Font Mono, Noto Color Emoji"
Set FontSize 18
Set Width 1400
Set Height 900
Set Padding 20
Set Theme "rose-pine-moon"
Set WindowBar Colorful
Set WindowBarSize 30
Set TypingSpeed 35ms
Set Framerate 25
Set PlaybackSpeed 1
Env COLORTERM "truecolor"
Env TERM "xterm-256color"
Output {reel}.gif
Output {reel}.mp4
{body}''')
PY
grep -q 'Output' "$OUT/$REEL.tape" || { echo "TAPE BUILD FAILED"; exit 4; }

scp -q -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$OUT/$REEL.tape" "root@$HOST:/workspace/$WSDIR/" || { echo "SCP FAILED"; exit 5; }

# Some reels use a specific chat model (fidelity to the original / fast cold load).
# Pull it on the pod before recording; the body exports LILBEE_CHAT_MODEL to match.
case "$REEL" in
  what-is-lilbee)
    echo "=== pulling Qwen3-4B for $REEL ==="
    remote "LILBEE_MODELS_DIR=/root/models HF_HUB_DISABLE_XET=1 /root/venv/bin/lilbee model pull Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf >/root/pull4b.log 2>&1; echo pulled" ;;
esac
echo "=== recording $REEL on the pod ==="
remote "cd /workspace/$WSDIR && rm -f $REEL.gif $REEL.mp4 ${REEL}.png TAKE_${REEL}_DONE
( vhs $REEL.tape >vhs-$REEL.log 2>&1 || xvfb-run -a vhs $REEL.tape >vhs-$REEL.log 2>&1 )
python3 - <<'PYG'
import numpy as np
from PIL import Image, ImageSequence
im = Image.open('/workspace/$WSDIR/$REEL.gif')
fr = [f.convert('RGB') for f in ImageSequence.Iterator(im)]
a = np.asarray(fr[-1], dtype=np.uint8)
h, w, _ = a.shape
band = a[int(h*0.15):int(h*0.92), int(w*0.04):int(w*0.96)]
lum = 0.2126*band[...,0]+0.7152*band[...,1]+0.0722*band[...,2]
bright = band[lum > 180]
if len(bright):
    c, n = np.unique(bright.reshape(-1,3), axis=0, return_counts=True)
    print('WHITE GATE $REEL:', tuple(int(x) for x in c[n.argmax()]), 'n=', int(n.max()))
else:
    print('WHITE GATE $REEL: no bright pixels')
PYG
tar czf take-$REEL.tgz $REEL.gif $REEL.mp4 ${REEL}.png vhs-$REEL.log 2>/dev/null
touch TAKE_${REEL}_DONE; echo RECORD_DONE" || { echo "RECORD SSH FAILED"; exit 6; }

remote "cat /workspace/$WSDIR/take-$REEL.tgz" > "$OUT/take-$REEL.tgz" 2>/dev/null
tar xzf "$OUT/take-$REEL.tgz" -C "$OUT" || { echo "PULL FAILED"; exit 7; }
ls -la "$OUT/$REEL.gif" "$OUT/$REEL.mp4"

HOLDENV=""
case "$REEL" in what-is-lilbee|tui-tow-limits|tui-chat|tui-crawl|tui-crawl-site) HOLDENV="REEL_HOLD=4.0" ;; esac
env $HOLDENV bash "$KIT/gif_finish.sh" "$OUT/$REEL.gif" "$OUT/$REEL.mp4" "$OUT" "SHIP-$REEL" || exit 8
echo "=== COLOR ACCEPTANCE $REEL ==="
SWEEPFLAG=""
case "$REEL" in tui-palette|tui-settings|tui-unsupported|tui-tour|tui-setup|tui-catalog|first-start|later-start|cold-start) SWEEPFLAG="--sweep" ;; esac
python3 "$KIT/color_accept.py" "$OUT/SHIP-$REEL.gif" "$KIT/reference-approved.gif" $SWEEPFLAG
RC=$?
echo "acceptance exit: $RC (artifacts: $OUT/SHIP-$REEL.{gif,mp4,png})"
exit $RC
