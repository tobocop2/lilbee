# GPUs and backends tested

lilbee places models by reading what the engine reports: which devices exist, how much memory each has, and what it actually allocated after loading. Every backend words those answers differently, so each is verified on real silicon.

Untested does not mean broken. It means unverified.

## Verified on real hardware

| GPU | Backend | Engine build | What it confirmed |
|-----|---------|--------------|-------------------|
| 2x NVIDIA A40 (46 GB) | CUDA | 9665 `e3a74b299` | Per-device reporting on a tensor split, `CUDA_Host` excluded, cgroup limits honoured over `/proc/meminfo` |
| NVIDIA GTX 1070 Ti (8 GB) | Vulkan | 9665 `e3a74b299` | `Vulkan_Host` excluded, vision projector accounting, probe crash isolation |
| Intel UHD (CometLake) | Vulkan | 9665 `e3a74b299` | Vulkan labels on non-NVIDIA silicon, chat and vision loads |
| Intel UHD + GTX 1650 Ti | Vulkan, hybrid | 9665 `e3a74b299` | Integrated excluded from packing, discrete kept |
| Apple Silicon | Metal | 9310 `e2ef8fe42` | `MTL0` labels, unified-memory budgeting |
| Intel Xeon Platinum 8481C | CPU | 9665 `e3a74b299` | Host-only load, `CPU`/`CPU_Mapped` attribution |
| AMD Instinct MI300X (192 GB) | ROCm | 9665 `e3a74b299` | `ROCm_Host` excluded, dedicated-VRAM sizing, `HIP_VISIBLE_DEVICES` filtering |

Captured logs live on the `tools/gpu-verification-harness` branch with the script that produced them.

## Naming, per backend

The join between what lilbee plans and what the engine reports. Every label below was observed, not inferred.

| Backend | Device label | Host allocator | Observed on |
|---------|--------------|----------------|-------------|
| CUDA | `CUDA0`, `CUDA1` | `CUDA_Host` | 2x A40 |
| Vulkan | `Vulkan0`, `Vulkan1` | `Vulkan_Host` | 1070 Ti, Intel UHD |
| ROCm | `ROCm0` | `ROCm_Host` | MI300X (216.94 MiB on the card, host buffers out) |
| Metal | `MTL0` | *(unified)* | Apple Silicon |
| CPU | `CPU`, `CPU_Mapped` | *(host)* | Xeon 8481C |

Host allocators must be excluded from device memory. Charging one to a card reports a phantom overrun on every partially offloaded model.

## Findings

### Vision projector

A projector's weights appear in **no buffer line** on any backend. The engine reports them only as prose. Estimates must be corrected before comparison, or every correctly-sized vision load reports a shortfall that is not there.

Confirmed on Vulkan (1070 Ti, Intel UHD) and ROCm (MI300X, `CLIP using ROCm0 backend`).

### Driver and engine measure different things

On the A40 pair, for the same process:

| Source | GPU 0 | GPU 1 |
|--------|-------|-------|
| `nvidia-smi` | 480 MiB | 492 MiB |
| engine report | 182.8 MiB | 194.0 MiB |

The gap is CUDA context overhead the engine never sees. The driver says what is unavailable to others; the engine says what the model asked for.

### The report is vendor-independent

Identical loads report identical figures across vendors:

| Load | GTX 1070 Ti (Vulkan) | Intel UHD (Vulkan) |
|------|----------------------|--------------------|
| chat | 157.13 MiB | 157.13 MiB |
| vision | 215.73 MiB | 215.73 MiB |

The buffer report describes what the model asked for, not what the silicon is. No per-vendor table is needed.

### Integrated adapters advertise memory they do not own

The hybrid laptop lists both adapters:

```
Vulkan0: NVIDIA GeForce GTX 1650 Ti   (4342 MiB)
Vulkan1: Intel(R) UHD Graphics       (11748 MiB)
```

The integrated one advertises 2.7x the real card, because it reports system RAM it may borrow. Packing by size would leave a dedicated GPU idle. lilbee types them apart and drops the integrated one.

Loads behave identically across all three of the laptop's graphics modes.

### A datacenter card can report an APU's name

The MI300X calls itself `AMD Radeon Graphics` — the same string an AMD APU reports. The name cannot decide dedicated versus shared. The absence of an integrated adapter decides it, and a headless datacenter card has no Vulkan driver to ask.

On 192 GB, getting this backwards is the difference between planning onto VRAM and double-booking the host.

### One thing that did not hold

AMD visibility variables on ROCm 7.2:

| Variable | Filters? |
|----------|----------|
| `HIP_VISIBLE_DEVICES` | yes, empty value means no devices |
| `ROCR_VISIBLE_DEVICES` | yes, same |
| `GPU_DEVICE_ORDINAL` | **no effect at all** |

lilbee treats `GPU_DEVICE_ORDINAL` as second of three in precedence, so a host setting only that variable gets a pin the runtime ignores. Tracked as a defect.

### The published ROCm wheel could not produce its own capture

It carried no HIP backend, so installing it on an AMD card yielded a CPU load. The engine was built from source for that run. Fixed separately.

## Not yet tested

| Backend | Why not |
|---------|---------|
| Vulkan on AMD silicon | Needs an AMD GPU with a Vulkan ICD. The only AMD hardware any cloud rents is MI300X, and it is a headless datacenter card that ships no Vulkan driver. Waiting on an AMD desktop rather than on rental |
| ROCm on a consumer Radeon | CDNA verified; the RDNA targets the wheel builds for are not. No cloud provider rents consumer Radeon |
| Two same-rank backends on one host | CUDA, ROCm and HIP tie on rank; the tie breaks on dedicated memory. Needs NVIDIA and AMD in one machine |
| Mixed-vendor host, two discrete cards | Partly covered by the hybrid laptop. Two discrete cards from different vendors is not sold by any cloud |
| MIG-partitioned NVIDIA | Partitioning needs host root. Rented GPUs are containers without it |
| SYCL (Intel Arc, Max) | lilbee publishes no SYCL wheel. Upstream llama.cpp does ship SYCL binaries, so this is now blocked on Intel hardware as much as on the wheel |
| CANN (Huawei Ascend) | No hardware access |
| AMX-enabled CPU build | Not a hardware gap. The published CPU wheel is an AVX2 baseline with AMX compiled out, verified on a Xeon that has the instructions |

### What cloud rental can and cannot reach

Checked against RunPod's catalogue (48 GPU types):

| | |
|---|---|
| NVIDIA | 47 types, consumer through datacenter |
| AMD | 1 type, MI300X only |
| Intel | none |

So Vulkan-on-AMD is the only remaining row cloud rental can close. Everything else needs hardware nobody rents, or is not a hardware question.

### Deliberately not chased

**Intel discrete** (Arc, Max). Through Vulkan an Arc card reports the same labels as the integrated part already tested, from the same driver stack. Its only meaningful difference is typing as discrete rather than integrated, and both sides of that branch are covered by the hybrid laptop. A third instance of a covered path is not evidence.

What an Intel discrete card would genuinely add is SYCL.

### Known unreproduced case

A host whose vendor compute driver works but whose Vulkan ICD is absent. lilbee consults PCI presence as a second opinion there. That configuration has not been reproduced on hardware; the check is a safety net, not a fix for an observed failure.

## Reproducing any of these

`tools/wf/capture_engine_log.sh` on the `tools/gpu-verification-harness` branch takes one environment variable:

```
ENGINE_INDEX=https://lilbee.sh/vulkan/ TAG=my-card bash capture_engine_log.sh
```

It installs the backend's engine into a throwaway virtualenv, loads a vision model so the projector case is covered, prints every buffer line, and runs lilbee's parser against the result.

Wheel indexes: `/cpu`, `/vulkan`, `/rocm`, `/cu125`.

Captures from hardware not listed above are welcome. The untested rows are the useful ones.
