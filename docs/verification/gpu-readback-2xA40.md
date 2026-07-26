# GPU readback on 2x A40

Captured on a RunPod 2x NVIDIA A40 box, engine build 9665 (e3a74b299),
by `tools/wf/verify_gpu_readback.sh` and `verify_driver_accounting.sh` on
the `tools/gpu-verification-harness` branch. Verbatim run output.

```
=== 0. host ===
index, name, memory.total [MiB]
0, NVIDIA A40, 46068 MiB
1, NVIDIA A40, 46068 MiB
torch absent; skipping the cuInit pre-flight

=== 1. cgroup memory, which lilbee now caps its budgets by ===
/sys/fs/cgroup/memory.max = (absent)
/sys/fs/cgroup/memory.current = (absent)
/sys/fs/cgroup/memory/memory.limit_in_bytes = 110999998464
/sys/fs/cgroup/memory/memory.usage_in_bytes = 8965476352
MemTotal from /proc/meminfo = 5.40644e+11

=== 2. engine device list ===
engine: /workspace/venv/lib/python3.13/site-packages/lilbee_engine/bin/llama-server
Available devices:
  CUDA0: NVIDIA A40 (45498 MiB, 45231 MiB free)
  CUDA1: NVIDIA A40 (45498 MiB, 45231 MiB free)

=== single ===
--- buffer lines ---
load_tensors:   CPU_Mapped model buffer size =    28.69 MiB
load_tensors:        CUDA0 model buffer size =    37.86 MiB
load_tensors:        CUDA1 model buffer size =    61.08 MiB
llama_context:  CUDA_Host  output buffer size =     0.75 MiB
llama_kv_cache:      CUDA0 KV buffer size =    24.00 MiB
llama_kv_cache:      CUDA1 KV buffer size =    21.00 MiB
sched_reserve:      CUDA0 compute buffer size =    24.91 MiB
sched_reserve:      CUDA1 compute buffer size =    24.91 MiB
sched_reserve:  CUDA_Host compute buffer size =    10.30 MiB

=== split ===
--- buffer lines ---
load_tensors:   CPU_Mapped model buffer size =    28.69 MiB
load_tensors:        CUDA0 model buffer size =    37.86 MiB
load_tensors:        CUDA1 model buffer size =    61.08 MiB
llama_context:  CUDA_Host  output buffer size =     0.75 MiB
llama_kv_cache:      CUDA0 KV buffer size =    24.00 MiB
llama_kv_cache:      CUDA1 KV buffer size =    21.00 MiB
sched_reserve:      CUDA0 compute buffer size =    24.91 MiB
sched_reserve:      CUDA1 compute buffer size =    24.91 MiB
sched_reserve:  CUDA_Host compute buffer size =    10.30 MiB

=== 3. lilbee's own parser against those logs ===
--- single.log
  build          : 9665 (e3a74b299)
  load finished  : True
  per device     : {'CPU': 28.7, 'CUDA0': 86.8, 'CUDA1': 107.0, 'CUDA_Host': 11.0}
  gpu footprint  : 0.19 GiB
--- split.log
  build          : 9665 (e3a74b299)
  load finished  : True
  per device     : {'CPU': 28.7, 'CUDA0': 86.8, 'CUDA1': 107.0, 'CUDA_Host': 11.0}
  gpu footprint  : 0.19 GiB

=== 4. the planner's own decisions against these real devices ===
```
