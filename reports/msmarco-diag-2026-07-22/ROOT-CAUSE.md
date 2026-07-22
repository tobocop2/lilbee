# MS MARCO 8.8M ingest — throughput degradation root cause (2026-07-22)

## What happened

Full MS MARCO passage corpus (8,841,823 passages) ingest on **8× A100-80GB-SXM**
(RunPod US), lilbee `feat/kreuzberg-5` @ `6bbd2b47`, embedder
`Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf`, `config.toml` +
`lilbee sync` in place (no copy).

Materialize (8.8M `.txt` into `documents/`) and the single-threaded discovery
pass completed. Embedding then ran but **degraded over ~8h**: it reached
~2.3M/8.8M (26%) while throughput fell from **138 docs/sec to ~58** and the
active GPU count collapsed from 8 to ~3.

## Root cause: llama-swap 300s TTL idle-unload + uneven dispatch

Every embed server is launched by the fleet with **`"ttl": 300`** (llama-swap
idle-unload after 300s), one `--parallel 1` llama-server per GPU:

```
llama-server ... --parallel 1 --cont-batching --ctx-size 2056 --embeddings --pooling last --port <p>
"ttl": 300
```

The failure loop (from `logs/llama-swap-embed.log`):

1. The **single-threaded `sync` dispatcher (admission capped at 32)** does not
   spread requests evenly across all 8 replicas.
2. A replica that goes 300s without a request → llama-swap **unloads its model**
   (`<embed-7> Unloading model, TTL of 300s reached`). VRAM frees — the
   diagnostic snapshot shows cards 6–7 at **0 MiB**, cards 2–5 model-resident but
   **0% util** (starved), only cards 0–1 working.
3. When a request later routes to an unloaded server → **`proxy error: dial tcp
   …: connection refused`** (58 such errors) while it cold-reloads the 8 GB model
   (~30–60s), stalling that request.
4. Fewer live servers → load concentrates on them → the rest keep idling out and
   unloading → **positive-feedback collapse** to 1–2 active cards.

Not OOM (`dmesg-oom.txt` empty), not a crash — the servers stay *alive*, they
just get **unloaded for being idle** because the dispatcher never keeps them warm.

## The two real defects

1. **The embed fleet uses a 300s TTL during a bulk ingest.** For a batch job where
   every replica should stay hot end to end, idle-unload is wrong. It should be
   TTL=0 (never unload) for the duration of an ingest.
2. **Dispatch is single-threaded and uneven** (admission capped at `max_workers`
   = 32, ~4 per replica, and skewed), so even before the TTL bites, 8× A100 only
   hit ~54% util (138 docs/sec, ~4.6× the 1-card smoke, not 8×). **PR #590**
   (parallel discovery + dispatch) is the structural fix — it keeps all replicas
   fed, which both raises utilization and prevents the idle-unload collapse.

## Recommended fixes (for #590 / a follow-up)

- During ingest, launch embed servers with **`ttl: 0`** (or long enough to
  outlast the whole run) so idle replicas are never unloaded.
- Round-robin / least-loaded dispatch across all replicas to keep every card warm.
- Raise the admission ceiling above 32 on high-core boxes so 8 replicas can each
  run more than ~4 concurrent.

## State — resumable, nothing lost

- Pod `i7uvpqdjnhqtlp` (`mm-7a19a398-head`, 8× A100 SXM) **STOPPED via
  `runpodctl` (native stop — sky cannot stop RunPod, only terminate)**. GPU
  billing halted; disk (2.3M embeddings + 8.8M materialized docs) preserved.
  Disk-storage billing continues while stopped.
- `sync` is hash-incremental: on resume it re-scans `documents/`, skips the
  ~2.3M already embedded, and continues. Nothing embedded so far is lost.

### To resume (after fixing the TTL / merging #590)

```
runpodctl pod start i7uvpqdjnhqtlp          # bring the pod back (needs 8x A100 availability)
# ssh in, then re-run the run block:
LILBEE_DATA=/root/msmarco/data \
  bash /opt/evals/infra/ingest.sh            # sync resumes, skipping embedded docs
```

(Or terminate and re-ingest fresh once #590 lands — the corpus re-materializes in
~20 min; only the ~2.3M partial embeddings would be re-done.)

## Diagnostic bundle

`reports/msmarco-diag-2026-07-22/diag/` (96 MB tarball, extracted). Key files:

- `logs/llama-swap-embed.log` (235 MB) — the fleet log with the TTL unloads + proxy errors
- `llama-swap-embed.23314.json` — fleet config showing `ttl: 300` per server
- `llama-swap.state.embed.23314.json` — 8 member ports (the 8 replicas)
- `gpu-snapshot.csv`, `nvidia-smi-full.txt` — the 8→3 collapse
- `sync.pass0.log`, `ingest.log` (669 MB) — full DEBUG ingest log
- `ingest_trace.log.gz` — per-passage extraction trace
- `processes.txt`, `mem.txt`, `dmesg-oom.txt` (empty — no OOM)
