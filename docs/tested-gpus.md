# GPUs and backends tested

lilbee decides where a model goes by reading what the engine says about your hardware: what devices exist, how much memory each has, and what the engine actually allocated once it loaded. Every one of those answers is worded differently per backend, so each backend is verified on real silicon rather than inferred from the last one.

This page records what has been run and what has not. A backend listed as untested is not known to be broken; it is unverified, which is a different and more honest thing to say.

## Verified on real hardware

| GPU | Backend | Engine build | What it confirmed |
|-----|---------|--------------|-------------------|
| 2x NVIDIA A40 (46 GB) | CUDA | 9665 `e3a74b299` | Per-device buffer reporting on a tensor split, `CUDA0`/`CUDA1` labels, `CUDA_Host` excluded from device memory, cgroup memory limits honoured over `/proc/meminfo` |
| NVIDIA GTX 1070 Ti (8 GB) | Vulkan | 9665 `e3a74b299` | `Vulkan0` labels, `Vulkan_Host` excluded from device memory, vision projector accounting, Vulkan device enumeration and its crash isolation |
| Apple Silicon | Metal | 9310 `e2ef8fe42` | `MTL0` labels, unified-memory budgeting |
| Intel UHD (CometLake) | Vulkan | 9665 `e3a74b299` | `Vulkan0` and `Vulkan_Host` on non-NVIDIA silicon, chat and vision loads, reported figures identical to the same loads on a discrete card |
| Intel UHD (CometLake) + GTX 1650 Ti | Vulkan, hybrid | 9665 `e3a74b299` | Two adapters of different types on one host: the integrated one classified as shared memory and excluded from packing, the discrete one kept, despite the integrated one advertising 2.7x more |
| Intel Xeon Platinum 8481C | CPU | 9665 `e3a74b299` | Host-only load with no GPU present, `CPU`/`CPU_Mapped` attribution |
| AMD Instinct MI300X (192 GB) | ROCm | 9665 `e3a74b299` | `ROCm0` labels, `ROCm_Host` excluded from device memory, a datacenter card sized as dedicated VRAM despite reporting an APU's name, `HIP_VISIBLE_DEVICES` filtering, vision projector accounting |

The captured logs behind these rows live on the `tools/gpu-verification-harness` branch, alongside the script that produced them.

### What the A40 pair settled

Two cards is where per-device accounting starts to matter: a plan can be right in total and wrong on one card, which is the failure that actually kills a load. The split reported `CUDA0` and `CUDA1` separately and the parser joined them to the planner's own per-device charges.

It also showed the driver and the engine disagree by design. `nvidia-smi` reported 480 and 492 MiB against the engine's 182.8 and 194.0 MiB for the same process. The gap is CUDA context overhead the engine never sees, so the two numbers answer different questions: the driver says what is unavailable to everyone else, the engine says what the model asked for.

### What the 1070 Ti settled

Vulkan is where every AMD and Intel GPU lands, so its wording matters well beyond NVIDIA. The engine names its pinned-host allocator `Vulkan_Host`, which lilbee must exclude from device memory or every Vulkan user sees a phantom overrun on every load.

A vision model on the same card settled a second question: a projector's weights appear in **no** buffer line at all. The engine reports them only as prose, so the estimate has to be corrected before it is compared, or every correctly-sized vision load reports a shortfall that is not there.

### What the Intel iGPU settled

Vulkan's wording had only ever been seen on an NVIDIA ICD, so `Vulkan0` and `Vulkan_Host` were verified for one vendor and assumed for the rest. An Intel CometLake iGPU produces the same labels, and the same two loads produce the same figures to the last two decimals: a chat model reports 157.13 MiB on the card, a vision model 215.73 MiB, with host allocators excluded from both.

That the numbers match a discrete NVIDIA card exactly is the useful part. The buffer report describes what the model asked for, not what the silicon is, so the readback does not need a per-vendor table.

### What the MI300X settled

ROCm was the last backend whose wording lilbee only knew from reading ggml's source, and it is the one where being wrong costs the most: an unrecognised host allocator gets charged to the card, so every partially offloaded model reports an overrun that is not there. The engine names its devices `ROCm0` and its pinned-host allocator `ROCm_Host`, so the existing `<backend>_Host` rule holds and ROCm needs no special case. The parser read the card's 216.94 MiB and left `ROCm_Host` and the CPU buffers out, matching the log's own arithmetic.

The card calls itself "AMD Radeon Graphics", which is the same string an AMD APU reports. So the name cannot decide whether memory is dedicated or a shared carveout of host RAM, and on a 192 GB card getting that backwards would be the difference between planning onto VRAM and double-booking the host. What decides it is the absence of an integrated adapter, and a headless datacenter card has no Vulkan driver to ask.

A vision model on the same card confirmed the projector behaves as it does on Vulkan: `CLIP using ROCm0 backend`, its memory reported only as prose, and no buffer line anywhere. The estimate has to be corrected for it before it is compared.

One thing did not hold. Of the three AMD visibility variables, `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` both filter as documented, and an empty value means no devices rather than no restriction. `GPU_DEVICE_ORDINAL` did nothing at all on ROCm 7.2: set to an index the host does not have, the card was still enumerated. lilbee treats it as the second of three in precedence order, so a host that sets only that variable would have a pin written somewhere the runtime ignores. That is tracked as a defect rather than papered over here.

**The published ROCm wheel could not produce this capture.** It carries no HIP backend at all, so installing it on an AMD card yields a CPU load. The engine was built from source at the pinned version for this run, and the wheel is fixed separately.

### What the hybrid laptop settled

With both adapters live the engine lists them together, and the numbers are the trap:

```
Vulkan0: NVIDIA GeForce GTX 1650 Ti   (4342 MiB)
Vulkan1: Intel(R) UHD Graphics        (11748 MiB)
```

The integrated adapter advertises 2.7 times the memory of the real card, because what it is reporting is system RAM it may borrow rather than memory it owns. A planner that packed by size would put models on the host's own memory and leave a dedicated GPU idle.

lilbee types them apart and drops the integrated one before packing, keeping the 4 GB card. That is the first time the packing filter has been exercised on hardware rather than on constructed device lists.

Loads then behave identically in all three of the laptop's graphics modes. A chat model reports 157.13 MiB on the card and a vision model 215.73 MiB, in integrated mode and in hybrid mode alike, and both match the figures from a discrete NVIDIA card on another machine.

The same machine with no NVIDIA driver installed enumerates the iGPU alone while the discrete card sits on the PCI bus. lilbee reads that as a host with no discrete GPU, which is harmless there because nothing can enumerate the card either. It would not be harmless on a host where the vendor's compute driver works but its Vulkan ICD is absent, so PCI presence is consulted as a second opinion. **That configuration has not been reproduced on hardware**; the check is a safety net rather than a fix for an observed failure.

## Not yet tested

| Backend | Status |
|---------|--------|
| Two backends of the same rank on one host | No hardware run. CUDA, ROCm and HIP tie at the same rank and the tie is broken on dedicated memory. Needs an NVIDIA and an AMD card in one machine, which is the mixed-vendor row below |
| ROCm on a consumer Radeon | No hardware run. CDNA is verified above; the RDNA targets the wheel builds for are not, and only cloud CDNA is rentable |
| Vulkan on AMD silicon | No hardware run. Vulkan is verified on NVIDIA and on an Intel iGPU; AMD is the one vendor untested |
| SYCL (Intel Arc, Max) | No engine wheel is published for SYCL, so this cannot be tested on any hardware today |
| CANN (Huawei Ascend) | No hardware run |
| Mixed-vendor host with two discrete cards | Partly covered. A hybrid Intel plus NVIDIA laptop is verified above; two discrete cards from different vendors in one machine is not, and cloud providers do not sell it |
| MIG-partitioned NVIDIA | No hardware run. Needs an A100 or H100 plus root to partition it |
| AMX-enabled CPU build | Not reachable. The published CPU wheel is an AVX2 baseline with AMX compiled out, verified on a Xeon that has the instructions |

### Deliberately not chased

**Intel discrete parts** (Arc, Max). Through Vulkan an Arc card reports the same labels as the integrated part already tested, from the same driver stack, and its only meaningful difference is typing as discrete rather than integrated. Both sides of that branch are already covered on the hybrid laptop, where an NVIDIA card and an Intel iGPU sit side by side. A third instance of a covered path is not evidence.

What an Intel discrete card would genuinely add is SYCL, which is blocked on the wheel rather than on hardware.

## Reproducing any of these

`tools/wf/capture_engine_log.sh` on the `tools/gpu-verification-harness` branch takes one environment variable and does the rest: installs the backend's engine into a throwaway virtualenv, loads a vision model so the projector case is covered, prints every buffer line, and runs lilbee's own parser against the result.

```
ENGINE_INDEX=https://lilbee.sh/vulkan/ TAG=my-card bash capture_engine_log.sh
```

Wheel indexes are `/cpu`, `/vulkan`, `/rocm` and `/cu125`. What comes back is the log plus the three answers that matter: what the engine calls its devices, what it calls its pinned-host allocator, and whether lilbee agrees with both.

Captures from hardware not listed above are welcome, and the untested rows are the useful ones.
