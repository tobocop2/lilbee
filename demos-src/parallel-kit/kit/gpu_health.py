#!/usr/bin/env python3
"""Real per-device GPU health: a matmul on each visible GPU. Enumeration
(llama-server --list-devices) passes on a bad card; only actual compute
catches "CUDA-capable device(s) is/are busy or unavailable" (RunPod ships
these on multi-GPU allocations). Exit 0 if ALL devices compute, 1 otherwise.
"""
import sys
try:
    import torch
except Exception as e:
    # a GPU job pod MUST have torch (runpod image); missing torch = broken env
    print(f"GPU_HEALTH_FAIL: torch import failed ({e})"); sys.exit(1)
n = torch.cuda.device_count()
bad = []
for i in range(n):
    try:
        torch.cuda.set_device(i)
        x = torch.ones(2048, 2048, device=f"cuda:{i}")
        _ = (x @ x).sum().item()
        torch.cuda.synchronize(i)
        print(f"device {i}: OK")
    except Exception as e:
        print(f"device {i}: FAIL {str(e)[:80]}")
        bad.append(i)
if bad:
    print(f"GPU_HEALTH_FAIL bad devices {bad} of {n}"); sys.exit(1)
print(f"GPU_HEALTH_OK {n} device(s)"); sys.exit(0)
