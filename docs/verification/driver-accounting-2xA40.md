# Driver accounting vs engine log on 2x A40

Captured on a RunPod 2x NVIDIA A40 box, engine build 9665 (e3a74b299),
by `tools/wf/verify_gpu_readback.sh` and `verify_driver_accounting.sh` on
the `tools/gpu-verification-harness` branch. Verbatim run output.

```
=== the query surface this rests on ===
List of valid properties to query for the switch "--query-compute-apps":

Section about Active Compute Processes properties
List of processes having compute context on the device.

"timestamp"
The timestamp of when the query was made in format "YYYY/MM/DD HH:MM:SS.msec".

"gpu_name"
The official product name of the GPU. This is an alphanumeric string. For all products.

"gpu_bus_id"
PCI bus id as "domain:bus:device.function", in hex.

"gpu_serial"
This number matches the serial number physically printed on each board. It is a globally unique immutable alphanumeric value.

"gpu_uuid"
This value is the globally unique immutable alphanumeric identifier of the GPU. It does not correspond to any physical label on the board.

"pid"
Process ID of the compute application

"process_name" or "name"
Process Name

"used_gpu_memory" or "used_memory"
Amount memory used on the device by the context. Not available on Windows when running in WDDM mode because Windows KMD manages all the memory not NVIDIA driver.


=== start the engine across both cards ===
engine pid: 4339

=== 1. what the DRIVER says this process holds, per device ===
4339, GPU-c4dd6ed1-a6a9-9421-b6b4-4cdf657c919e, 480 MiB
4339, GPU-66b6a70f-3840-ce84-c78f-9366025c6124, 492 MiB

=== 2. which device each uuid is ===
0, GPU-c4dd6ed1-a6a9-9421-b6b4-4cdf657c919e, NVIDIA A40
1, GPU-66b6a70f-3840-ce84-c78f-9366025c6124, NVIDIA A40

=== 3. what the ENGINE LOG says, for the same load ===
load_tensors:   CPU_Mapped model buffer size =    28.69 MiB
load_tensors:        CUDA0 model buffer size =    37.86 MiB
load_tensors:        CUDA1 model buffer size =    61.08 MiB
llama_context:  CUDA_Host  output buffer size =     0.75 MiB
llama_kv_cache:      CUDA0 KV buffer size =    96.00 MiB
llama_kv_cache:      CUDA1 KV buffer size =    84.00 MiB
sched_reserve:      CUDA0 compute buffer size =    48.91 MiB
sched_reserve:      CUDA1 compute buffer size =    48.91 MiB
sched_reserve:  CUDA_Host compute buffer size =    34.30 MiB

=== 4. side by side ===
driver, per process per device:
  pid 4339  CUDA0  480 MiB
  pid 4339  CUDA1  492 MiB
engine log, per device:
  CPU              28.7 MiB
  CUDA0           182.8 MiB
  CUDA1           194.0 MiB
  CUDA_Host        35.0 MiB

=== 5. and after the process is gone ===
(empty above = the driver stops reporting a dead process, which is what makes it a live signal)
```
