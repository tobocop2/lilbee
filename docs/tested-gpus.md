# Hardware lilbee has run on

lilbee places models by reading what the engine reports: which devices exist, how much memory each has, and what it actually allocated after loading. Every backend words those answers differently, so each is checked on real silicon.

## Machines

| Machine | Backend | What ran on it |
|---------|---------|----------------|
| 2x NVIDIA A40 (46 GB) | CUDA | Readback on a tensor split, cgroup memory and CPU limits honoured over the host's totals |
| NVIDIA A100 80GB PCIe | CUDA | Shared-engine load benchmark, concurrency sweep, 600-round endurance and chaos soaks |
| 8x NVIDIA A100 | CUDA | Ingest throughput at scale, 161 docs/sec |
| 3x NVIDIA A100 | CUDA | Auto-placement of a 235B chat model across three cards |
| 2x NVIDIA L40S | CUDA | Ingest with an even split across slower cards |
| NVIDIA H100 | CUDA | Validated |
| NVIDIA H200 | CUDA | Validated |
| NVIDIA RTX 5090 | CUDA | Validated |
| NVIDIA RTX 4090 | CUDA | Validated |
| NVIDIA RTX 3090 | CUDA | Validated |
| NVIDIA GTX 1070 Ti (8 GB) | Vulkan | Readback, vision projector accounting, probe crash isolation |
| Intel UHD (CometLake) | Vulkan | Readback on non-NVIDIA silicon, chat and vision loads |
| Intel UHD + GTX 1650 Ti | Vulkan, hybrid | Integrated excluded from packing, discrete kept |
| Apple Silicon | Metal | Unified-memory budgeting |
| Intel Xeon Platinum 8481C | CPU | Host-only load with no GPU present |
| AMD Instinct MI300X (192 GB) | ROCm | Readback, dedicated-VRAM sizing, `HIP_VISIBLE_DEVICES` filtering |
| AMD Radeon RX 9060 XT (16 GB) | ROCm, Vulkan | Full offload on RDNA4 (gfx1200), flatpak sandbox GPU access, power-based utilization capture |

The work is usually ingest and chat. Rows naming a specific finding have a captured engine log behind them, on the `tools/gpu-verification-harness` branch alongside the script that produced them; those captures are engine build 9665 `e3a74b299`, except Apple Silicon at 9310 `e2ef8fe42`.

## Backend naming

The join between what lilbee plans and what the engine reports. Every label was observed, not inferred.

| Backend | Device label | Host allocator | Observed on |
|---------|--------------|----------------|-------------|
| CUDA | `CUDA0`, `CUDA1` | `CUDA_Host` | 2x A40 |
| Vulkan | `Vulkan0`, `Vulkan1` | `Vulkan_Host` | 1070 Ti, Intel UHD |
| ROCm | `ROCm0` | `ROCm_Host` | MI300X (216.94 MiB on the card, host buffers out) |
| Metal | `MTL0` | *(unified)* | Apple Silicon |
| CPU | `CPU`, `CPU_Mapped` | *(host)* | Xeon 8481C |

Host allocators are excluded from device memory. Charging one to a card reports a phantom overrun on every partially offloaded model.

## Findings

### Vision projector

A projector's weights appear in **no buffer line** on any backend. The engine reports them only as prose. Estimates must be corrected before comparison, or every correctly-sized vision load reports a shortfall that is not there.

Confirmed on Vulkan (1070 Ti, Intel UHD) and ROCm (MI300X, `CLIP using ROCm0 backend`).

### AMD reports 100% busy whenever the engine is resident

On an RX 9060 XT (Bazzite, kernel 6.16), `rocm-smi` and `gpu_busy_percent` read a constant 100% from the moment a llama-server holds a compute context until it exits, idle or generating alike. The shader clock pins high (~44W against a 17W true idle), while the memory clock stays at its 456MHz floor. This is amdgpu's perf-level policy for resident compute contexts, not load.

Real activity shows in power draw and memory clock: during inference power steps to 50-103W and mclk to 1258MHz, and both fall back the moment the request completes while the busy flag still reads 100%. Read power and mclk, not the percent. From #677 on, lilbee's fleet panel derives AMD utilization from power draw over the board cap, so a pegged 100% in `rocm-smi` next to a moving lilbee gauge is expected, not a disagreement to debug.

### Driver and engine measure different things

On the A40 pair, same process:

| Source | GPU 0 | GPU 1 |
|--------|-------|-------|
| `nvidia-smi` | 480 MiB | 492 MiB |
| engine report | 182.8 MiB | 194.0 MiB |

The gap is CUDA context overhead the engine never sees. The driver says what is unavailable to others; the engine says what the model asked for.

### The report is vendor-independent

Identical loads, identical figures across vendors:

| Load | GTX 1070 Ti (Vulkan) | Intel UHD (Vulkan) |
|------|----------------------|--------------------|
| chat | 157.13 MiB | 157.13 MiB |
| vision | 215.73 MiB | 215.73 MiB |

The buffer report describes what the model asked for, not what the silicon is. No per-vendor table needed.

### Integrated adapters advertise memory they do not own

The hybrid laptop lists both:

```
Vulkan0: NVIDIA GeForce GTX 1650 Ti   (4342 MiB)
Vulkan1: Intel(R) UHD Graphics       (11748 MiB)
```

The integrated one advertises 2.7x the real card, because it reports system RAM it may borrow. Packing by size would leave a dedicated GPU idle. lilbee types them apart and drops the integrated one.

Loads behave identically across all three of the laptop's graphics modes.

### A datacenter card can report an APU's name

The MI300X calls itself `AMD Radeon Graphics` — the same string an AMD APU reports. The name cannot decide dedicated versus shared. The absence of an integrated adapter decides it, and a headless datacenter card has no Vulkan driver to ask.

On 192 GB, getting this backwards is the difference between planning onto VRAM and double-booking the host.

### Multi-card behaviour

| Machine | What it showed |
|---------|----------------|
| 2x A40 | Per-device accounting on a tensor split. A plan can be right in total and wrong on one card |
| 3x A100 | A 235B model auto-placed across three cards |
| 8x A100 | 161 docs/sec at ~78% util, cards ranging 10–98%, GPU-bound |
| 2x L40S | Slower cards, but an even split |

### AMD visibility variables

Measured on ROCm 7.2:

| Variable | Filters? |
|----------|----------|
| `HIP_VISIBLE_DEVICES` | yes, empty value means no devices |
| `ROCR_VISIBLE_DEVICES` | yes, same |
| `GPU_DEVICE_ORDINAL` | **no effect at all** |

lilbee treats `GPU_DEVICE_ORDINAL` as second of three in precedence, so a host setting only that variable gets a pin the runtime ignores. Tracked as a defect.

## Reproducing any of these

`tools/wf/capture_engine_log.sh` on the `tools/gpu-verification-harness` branch takes one environment variable:

```
ENGINE_INDEX=https://lilbee.sh/vulkan/ TAG=my-card bash capture_engine_log.sh
```

It installs the backend's engine into a throwaway virtualenv, loads a vision model so the projector case is covered, prints every buffer line, and runs lilbee's parser against the result.

Wheel indexes: `/cpu`, `/vulkan`, `/rocm`, `/cu125`.

Captures from hardware not listed above are welcome.
