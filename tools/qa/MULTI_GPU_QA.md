# Multi-GPU fleet QA (real hardware)

The unit/integration tests mock subprocesses, sockets, and devices, so the
hardware-dependent behavior of the fleet has to be checked on a real multi-GPU
host. `multi_gpu_smoke.py` is that check.

## What it verifies

1. **Enumeration** — `llama-server --list-devices` finds the GPUs and the count
   matches `nvidia-smi` (catches the Vulkan-vs-CUDA index hazard).
2. **Placement** — prints which device-pinning env each server got, so you can
   confirm against `nvidia-smi` that models landed on the intended GPUs and that
   an unequal pair tensor-split by capacity.
3. **Concurrency** — fires N concurrent chat+embed requests; all must succeed
   (exercises the single-flight build and the atomic least-in-flight router).
4. **Restart** — kills a server's process group and asserts the monitor restarts
   it on a fresh pid and still serves.
5. **No orphans** — after shutdown, asserts there are no surviving
   `llama-server` processes and VRAM returned to the pre-test baseline.

## Running it on RunPod

A multi-GPU pod is required (e.g. 2x A100/A6000). Reuse the existing QA pod flow.

```bash
# On the pod:
pip install 'lilbee[multi-gpu]'            # brings the bundled llama-server
lilbee model pull <chat-gguf>              # placement reads real GGUFs on disk
lilbee model pull <embedding-gguf>
# point cfg at them (TUI settings, or env): chat_model / embedding_model

python tools/qa/multi_gpu_smoke.py --concurrency 16
```

Exit code 0 and a final `PASS` line means the fleet is correct on that hardware;
any check prints `FAIL: <reason>` and exits non-zero.

Notes:
- NVIDIA-focused (uses `nvidia-smi`/`pgrep`). On AMD, the enumeration/orphan
  checks degrade gracefully (no `nvidia-smi`), and placement is read from the
  printed `ROCR_VISIBLE_DEVICES` pinning instead.
- Run it twice back-to-back: the second run proves the orphan reaper cleans up
  anything a hard kill left behind.
