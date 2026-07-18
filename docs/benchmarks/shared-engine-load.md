# Shared Engine Load and Stability Benchmarks

Throughput, latency, and crash-recovery results for the shared inference
engine: one llama.cpp fleet serving every lilbee process on the machine.

## Test Setup

- **Date:** 2026-07-18
- **Hardware:** NVIDIA A100 80GB PCIe (driver 570.195.03), RunPod secure cloud
- **Software:** lilbee 0.6.9 (branch `fix/serve-data-dir-singleton`), engine pin
  `llama-cpp-0.3.30+swap-v223+gguf-v0.25.0`, Python 3.11.10
- **Chat model:** Qwen3.6-35B-A3B, Q8_0 GGUF (37 GB weights), 4 batching slots,
  65,536 tokens of context per slot (262,144 served), q8_0 KV cache, flash
  attention on
- **Embed model:** nomic-embed-text-v1.5, Q4_K_M

The harness lives in `tools/qa/shared-engine/load/`: a streaming asyncio driver
(`bench_driver.py`), the concurrency sweep (`bench_sweep.sh`), and the chaos
soak (`soak_run.sh`). Every number below is reproducible from those scripts.

## Throughput sweep

32 streamed chat completions per cell (16 at c=1), 200 generated tokens each,
against lilbee's OpenAI-compatible `/v1/chat/completions`. Token counts include
reasoning deltas (Qwen3.6 is a thinking model). TTFT is time to first streamed
token; per-stream throughput is each request's own decode rate.

| Concurrent streams | tok/s per stream (p50) | TTFT p50 | TTFT p95 | Aggregate tok/s |
|---|---|---|---|---|
| 1 | 130.9 | 0.11 s | 0.12 s | 120 |
| 2 | 108.8 | 0.25 s | 0.36 s | 188 |
| 4 | 70.2 | 0.57 s | 0.72 s | 231 |
| 8 | 70.2 | 3.97 s | 4.06 s | 235 |
| 16 | 70.3 | 10.90 s | 11.06 s | 230 |

Reading the shape: 1 to 4 streams share the server's 4 batching slots, so each
stream keeps a real interactive rate while aggregate throughput climbs. Past 4
streams the extra requests queue (TTFT grows linearly) while per-stream and
aggregate rates hold flat, which is the correct saturation behavior rather than
collapse.

A repeat of the c=1 cell at the end of the sweep (the drift control) measured
130.8 tok/s against 130.9 at the start, so the box did not degrade during the
run.

## Server overhead

The same driver pointed directly at the engine's llama-swap proxy, bypassing
lilbee's server entirely:

| Path | tok/s per stream (p50) | TTFT p50 |
|---|---|---|
| Bare engine (llama-swap direct), c=1 | 130.8 | 0.115 s |
| Through lilbee serve, c=1 | 130.9 | 0.114 s |
| Bare engine, c=4 | 69.9 | 0.567 s |
| Through lilbee serve, c=4 | 70.2 | 0.573 s |

lilbee's routing layer adds no measurable throughput cost and about a
millisecond of TTFT: the deltas at both concurrency levels are inside
run-to-run noise.

## Chaos soak

Rounds of 4 concurrent chat completions plus one CLI engine acquire/release
cycle. Every 5th round, at a random moment, one engine process (a llama-server
or the llama-swap proxy itself) is SIGKILLed. Per-round invariants: all streams
and the CLI succeed, VRAM stays within 10% of the round-1 baseline, process
counts return to baseline, and the engine-membership lock count never grows.

**40 rounds, 8 forced kills, zero invariant breaches.** Every round completed
4/4 streams green, including the chaos rounds themselves; a proxy kill costs
one slower round (about 35 s, the in-place rebuild) and full service resumes
the next round. VRAM stayed pinned at 39.4 GB across the entire run, and the
membership lock count never left 1.

The first execution of this soak, on a build without commit `4a43e0c9`, is why
that commit exists: killing the llama-swap proxy orphaned its VRAM, doubled
GPU memory by building a duplicate engine in the overflow slot, and left the
resident server returning errors until restart. The soak now passes the same
violence cleanly, which is the point of running it.

## What this does and does not show

These results show the shared-engine architecture holds up under concurrent
multi-client load and survives engine-process crashes without leaking VRAM,
locks, or processes, on one GPU with one model pairing. They do not cover
multi-GPU placement, other quantizations, consumer GPUs, or sustained
multi-hour operation (a longer endurance run is tracked separately). Absolute
throughput numbers are specific to this model, quantization, and card.
