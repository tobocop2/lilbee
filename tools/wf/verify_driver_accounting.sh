#!/usr/bin/env bash
# Does the driver report per-process, per-device VRAM well enough to replace the
# engine-log parse?
#
# The log parse works but depends on three printf format strings in llama.cpp and
# on running the engine at a raised verbosity. If the driver already answers the
# same question through a stable query, that is a better source: an interface
# rather than a format, exact per-process attribution rather than a whole-device
# free-memory delta, and no engine cooperation at all.
#
# Loads a model, then asks the driver what that PID holds on each GPU, and prints
# the engine's own log figures beside it for comparison.
set -uo pipefail

OUT=${OUT:-/workspace/driver-out}
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

MODEL=${MODEL:?set MODEL}
LS=$(python3 -c "import lilbee_engine; print(lilbee_engine.get_llama_server_path())")

echo "=== the query surface this rests on ==="
nvidia-smi --help-query-compute-apps 2>&1 | head -30

echo
echo "=== start the engine across both cards ==="
rm -f "$OUT/engine.log"
"$LS" --model "$MODEL" --host 127.0.0.1 --port 39200 --ctx-size 8192 \
    --n-gpu-layers 999 --tensor-split 1,1 --split-mode layer \
    --log-file "$OUT/engine.log" --log-verbosity 4 >/dev/null 2>&1 &
PID=$!
for _ in $(seq 1 90); do
    grep -q "initializing slots" "$OUT/engine.log" 2>/dev/null && break
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done
echo "engine pid: $PID"

echo
echo "=== 1. what the DRIVER says this process holds, per device ==="
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
    --format=csv,noheader | tee "$OUT/compute-apps.csv"

echo
echo "=== 2. which device each uuid is ==="
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader | tee "$OUT/gpu-uuids.csv"

echo
echo "=== 3. what the ENGINE LOG says, for the same load ==="
grep -E "buffer size" "$OUT/engine.log" | sed 's/^[0-9.]* [A-Z] //'

echo
echo "=== 4. side by side ==="
python3 - "$OUT" <<'PY'
import csv, pathlib, sys
from lilbee.providers.fleet.readback import parse_device_buffers

out = pathlib.Path(sys.argv[1])
uuid_to_index = {}
for row in csv.reader((out / "gpu-uuids.csv").read_text().splitlines()):
    if len(row) >= 2:
        uuid_to_index[row[1].strip()] = row[0].strip()

print("driver, per process per device:")
for row in csv.reader((out / "compute-apps.csv").read_text().splitlines()):
    if len(row) >= 3:
        pid, uuid, used = (c.strip() for c in row[:3])
        print(f"  pid {pid}  CUDA{uuid_to_index.get(uuid, '?')}  {used}")

print("engine log, per device:")
for label, size in sorted(parse_device_buffers((out / "engine.log").read_text()).items()):
    print(f"  {label:12s} {size / 1024**2:8.1f} MiB")
PY

kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
echo
echo "=== 5. and after the process is gone ==="
sleep 2
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader || true
echo "(empty above = the driver stops reporting a dead process, which is what makes it a live signal)"
