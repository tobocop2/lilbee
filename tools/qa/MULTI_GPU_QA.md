# Multi-GPU fleet QA (real hardware)

The unit/integration tests mock subprocesses, sockets, and devices, so the
hardware-dependent behavior of the fleet has to be checked on a real multi-GPU
host. `multi_gpu_smoke.py` is that check.

## What it verifies

1. **Enumeration** — `llama-server --list-devices` finds the GPUs and the count
   matches `nvidia-smi` (catches the Vulkan-vs-CUDA index hazard).
2. **Placement** — prints each planned server's device pinning (and which models
   tensor-split across cards), so you can confirm against `nvidia-smi` that the
   giant landed on the intended GPUs and an unequal split is by capacity.
3. **Concurrency** — fires N concurrent chat+embed requests; all must succeed
   (exercises the lazy fleet build and the atomic least-in-flight router).
4. **Restart** — hard-kills an upstream `llama-server`; llama-swap (which owns the
   upstream lifecycle) must respawn it on the next request and keep serving.
5. **No orphans** — after `provider.shutdown()`, asserts no surviving
   `llama-server` *or* `llama-swap` processes and VRAM returned to baseline.

## Running it on RunPod

A multi-GPU pod is required (e.g. 2x A100/A6000). Reuse the existing QA pod flow
(`tools/qa/cloud-setup.sh` bootstraps a fresh box).

```bash
# On the pod (lilbee installed from source; the bundled engine ships with it):
lilbee model pull <chat-gguf>              # placement reads real GGUFs on disk
lilbee model pull <embedding-gguf>
# point config at them: chat_model / embedding_model (config.toml or the TUI),
# plus optional embed_replicas / vision_replicas to fan a role across GPUs

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
