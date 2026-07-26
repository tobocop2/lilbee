#!/usr/bin/env bash
# Verify on real NVIDIA hardware the things a laptop cannot answer:
#   1. the engine's buffer report exists on a CUDA build, at the verbosity lilbee asks for
#   2. its device labels are the CUDA0/CUDA1 names lilbee joins on
#   3. a tensor-split load reports per-device figures the per-device check can use
#   4. the cgroup memory files lilbee now reads are present and say what it expects
#
# Writes everything to /workspace/verify-out and prints a summary. Never compiles
# llama.cpp: the engine is installed prebuilt.
set -uo pipefail

OUT=${OUT:-/workspace/verify-out}
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

echo "=== 0. host ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo "NO nvidia-smi"
python3 -c "import torch; print('cuInit ok:', torch.cuda.is_available(), torch.cuda.device_count())" 2>/dev/null \
  || echo "torch absent; skipping the cuInit pre-flight"

echo
echo "=== 1. cgroup memory, which lilbee now caps its budgets by ==="
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current \
         /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.usage_in_bytes; do
    if [ -r "$f" ]; then echo "$f = $(cat "$f")"; else echo "$f = (absent)"; fi
done
echo "MemTotal from /proc/meminfo = $(awk '/MemTotal/{print $2*1024}' /proc/meminfo)"

echo
echo "=== 2. engine device list ==="
LS=$(python3 -c "import lilbee_engine; print(lilbee_engine.get_llama_server_path())")
echo "engine: $LS"
"$LS" --list-devices 2>&1 | tee "$OUT/list-devices.txt"

MODEL=${MODEL:?set MODEL to a gguf path}
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

run_case () {  # name, extra args...
    local name=$1; shift
    echo
    echo "=== $name ==="
    rm -f "$OUT/$name.log"
    "$LS" --model "$MODEL" --host 127.0.0.1 --port 39100 --ctx-size 2048 \
        --log-file "$OUT/$name.log" --log-verbosity 4 "$@" >/dev/null 2>&1 &
    local pid=$!
    for _ in $(seq 1 60); do
        grep -q "initializing slots" "$OUT/$name.log" 2>/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    echo "--- buffer lines ---"
    grep -E "buffer size" "$OUT/$name.log" | sed 's/^[0-9.]* [A-Z] //' || echo "NONE FOUND"
}

run_case single --n-gpu-layers 999
if [ "$NGPU" -ge 2 ]; then
    run_case split --n-gpu-layers 999 --tensor-split 1,1 --split-mode layer
else
    echo "only $NGPU GPU(s); skipping the tensor-split case"
fi

echo
echo "=== 3. lilbee's own parser against those logs ==="
python3 - "$OUT" <<'PY'
import sys, pathlib
from lilbee.providers.fleet.readback import (
    device_footprint, engine_build, load_finished, parse_device_buffers,
)

out = pathlib.Path(sys.argv[1])
for log in sorted(out.glob("*.log")):
    if log.name == "run.log":
        continue
    text = log.read_text(errors="replace")
    print(f"--- {log.name}")
    print("  build          :", engine_build(text) or "(not found)")
    print("  load finished  :", load_finished(text))
    print("  per device     :", {k: round(v / 1024**2, 1) for k, v in parse_device_buffers(text).items()})
    print("  gpu footprint  :", round(device_footprint(text) / 1024**3, 2), "GiB")
PY

echo
echo "=== 4. the planner's own decisions against these real devices ==="
python3 - <<'PYINNER'
import lilbee_engine
from lilbee.providers.fleet.placement import _split_ratio_candidates, _tight_device_group
from lilbee.providers.fleet.planning import (
    apply_ctx_downshift, record_ctx_downshift, resolve_devices,
)
from lilbee.providers.fleet.swap_manager import _ephemeral_range, _pick_free_ports
from lilbee.providers.roles import WorkerRole

print("--- ports")
rng = _ephemeral_range()
print("  kernel ephemeral range :", rng)
ports = _pick_free_ports(3)
print("  picked                 :", ports)
if rng:
    ok = all(p < rng[0] for p in ports)
    print("  verdict                :", "BELOW (cannot be auto-assigned)" if ok else "INSIDE THE RANGE")

print("--- tensor-split ratio ladder against this box")
devices = resolve_devices(lilbee_engine.get_llama_server_path())
remaining = {d.index: float(d.free_bytes) for d in devices}
print("  free per device        :", {k: round(v / 1024**3, 1) for k, v in remaining.items()})
if len(remaining) >= 2:
    idxs = sorted(remaining)
    print("  candidates             :", _split_ratio_candidates(idxs, remaining))
    print("  tight group for 1 TiB  :", _tight_device_group(1024**4, remaining))

print("--- context downshift ladder")
steps = 0
while record_ctx_downshift(WorkerRole.CHAT) and steps < 20:
    steps += 1
    print("  step", steps, "-> ", apply_ctx_downshift(WorkerRole.CHAT, 32768))
print("  terminates after", steps, "steps")
PYINNER

echo
echo "=== done; artifacts in $OUT ==="
