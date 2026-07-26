#!/usr/bin/env bash
# Capture one engine load on whatever hardware this box has, and check lilbee's
# parser against it. Vendor-agnostic on purpose: the point is the backends nobody
# has run yet (AMD, Intel, and the CPU buffer types), and each needs the same
# three answers.
#
#   1. what the engine calls its devices  (CUDA0 / Vulkan0 / SYCL0 / ROCm0 ...)
#   2. what it calls its pinned-host allocator, which decides what is NOT VRAM
#   3. whether lilbee's parser agrees with both
#
# Usage, on the rented box:
#   ENGINE_INDEX=https://lilbee.sh/vulkan/ ./capture_engine_log.sh
#   ENGINE_INDEX=https://lilbee.sh/rocm/   ./capture_engine_log.sh
#
# Writes <out>/engine-load-<tag>.log, which is the fixture, plus a summary to
# paste back. Nothing here is destructive and nothing is installed system-wide.
set -uo pipefail

OUT=${OUT:-$HOME/lilbee-capture}
TAG=${TAG:-$(uname -s | tr '[:upper:]' '[:lower:]')-$(date +%s)}
ENGINE_INDEX=${ENGINE_INDEX:?set ENGINE_INDEX to the wheel index for this backend}
LILBEE_REF=${LILBEE_REF:-fix/ctx-slots-from-placed-device}
VENV=${VENV:-/tmp/lilbee-capture-venv}
mkdir -p "$OUT"

PY=$(for v in 3.13 3.12 3.11; do command -v "python$v" && break; done)
[ -n "$PY" ] || { echo "need python 3.11+"; exit 1; }
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q --pre lilbee-engine --extra-index-url "$ENGINE_INDEX"
"$VENV/bin/pip" install -q huggingface_hub
B=$("$VENV/bin/python" -c "import lilbee_engine;print(lilbee_engine.get_llama_server_path())")

echo "=== engine ==="
"$B" --version 2>&1 | head -2
echo
echo "=== devices this build sees ==="
"$B" --list-devices 2>&1 | head -10

# A vision model, because the projector is the one thing the report omits and
# the estimate has to be corrected for. Text-only loads miss that entirely.
read -r MODEL MMPROJ <<<"$("$VENV/bin/python" - <<'PYEOF'
from huggingface_hub import hf_hub_download
r = "ggml-org/SmolVLM-256M-Instruct-GGUF"
print(hf_hub_download(r, "SmolVLM-256M-Instruct-Q8_0.gguf"),
      hf_hub_download(r, "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"))
PYEOF
)"

LOG="$OUT/engine-load-$TAG.log"
rm -f "$LOG"
"$B" --model "$MODEL" --mmproj "$MMPROJ" --host 127.0.0.1 --port 39377 \
     --ctx-size 2048 --n-gpu-layers 999 --log-file "$LOG" --log-verbosity 4 >/dev/null 2>&1 &
PID=$!
for _ in $(seq 1 120); do
    grep -q "initializing" "$LOG" 2>/dev/null && break
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null

echo
echo "=== every buffer line (this is what the parser reads) ==="
grep -E "buffer size" "$LOG" | sed 's/^[0-9.]* [A-Z] //' || echo "NONE -- report this, it is the interesting case"

echo
echo "=== lilbee's parser against it ==="
"$VENV/bin/pip" install -q "git+https://github.com/tobocop2/lilbee@${LILBEE_REF}" 2>&1 | tail -1
"$VENV/bin/python" - "$LOG" <<'PYEOF'
import sys, pathlib
from lilbee.providers.fleet.readback import (
    device_footprint, engine_build, load_finished, parse_device_buffers,
)
t = pathlib.Path(sys.argv[1]).read_text(errors="replace")
print("  build         :", engine_build(t) or "(not found)")
print("  load finished :", load_finished(t))
print("  per device    :", {k: round(v / 1024**2, 2) for k, v in parse_device_buffers(t).items()})
print("  gpu footprint :", round(device_footprint(t) / 1024**2, 2), "MiB")
print()
print("  Check by eye: every GPU device above should be a real card, and any")
print("  host allocator (CPU*, *_Host) must be absent from the footprint.")
PYEOF

echo
echo "=== done. Send back: $LOG and the summary above ==="
