# Full-corpus MS MARCO ingest on the native per-GPU path

The 8.8M-passage ingest, run through PR #644's native fan-out: one bare
`lilbee sync` with no environment variables, no corpus dealing and no merge
script. Everything here is measurement, publishing and survival.

## What replaced what

The environment-variable harness (`ingest9m.sh`, `pod9m.sh`, `merge_shards.py`)
dealt the corpus into eight per-worker trees, launched eight `lilbee sync`
processes with four environment variables each, and folded the resulting
databases with a 403-line script. `lilbee sync` now does all of that itself, so
the harness is down to staging a corpus, sampling, and getting the result off the
box.

| old | now |
|---|---|
| deal 8842 buckets into 8 trees with `cp -al` | nothing: `ShardId.owns` filters during the walk |
| `CUDA_VISIBLE_DEVICES` / `LILBEE_DATA` / `LILBEE_ENGINE_DIR` / `LILBEE_ANN_INDEX_THRESHOLD` per worker | nothing |
| `merge_shards.py` after the run | nothing: `sync` folds the shards |
| eight `sync.log` files to read | one aggregate bar, plus `shards/wN/sync.log` for detail |

## Layout, and why it is split

```
/workspace              network volume, SURVIVES the pod
  kb/.lilbee/           data root: config.toml, data/lancedb, shards/w0..w7
  models/               the embedder, so a replacement pod does not re-pull 8GB
  prof/                 samplers, traces, folded stacks
  status/               expected, counts, phase, run.env
  export/               parquet + jsonl
/root/corpus/documents/ container disk: 8.8M files, re-downloadable in minutes
```

The index is on the volume because losing the pod must not lose the run. The
corpus is on the container disk because it is 8.8M files of ~325 bytes and a walk
of that over a network filesystem is the slowest thing in the pipeline; it is
also a 1.3GB tarball away from being rebuilt. `documents_dir` is a writable
config field, which is what lets the two live on different filesystems.

## Running it

```bash
./pod_native.sh up          # pick a datacenter with capacity, make the volume,
                            # provision, launch, start the dashboard + recording
./pod_native.sh attach      # the dashboard, locally, in your tmux
./pod_native.sh watch       # the same dashboard, served from the pod
./pod_native.sh status      # one-shot summary
./pod_native.sh publish     # swap the GPU pod for a cheap one on the SAME volume
./pod_native.sh resume      # replacement pod on the same volume, continue
./pod_native.sh down        # delete the pod, KEEP the index
./pod_native.sh nuke        # delete the pod AND the index
```

A trial is the same command with smaller knobs:

```bash
GPU_NAME="NVIDIA GeForce RTX 4090" GPUS=2 VOL_GB=60 DISK=60 HOURS=3 \
CORPUS_URL=".../msmarco-passage-80k.tar.gz" \
EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf" \
EMBED_DIM=1024 HF_REPO="beeberg/msmarco-ingest-trial" ./pod_native.sh up
```

## The two failures this design exists to prevent

**A guard that fired on a perfect run.** The previous full ingest completed and
was destroyed by its own harness: `EXPECTED` was computed as
buckets x files-per-bucket, which over-counts by 177 because the last bucket
holds 823 files rather than 1000. The guard fired, the merge and export were
skipped, and the watchdog stopped the pod. Two changes, and only the second one
matters: the count is now measured by walking the corpus, and **export is never
gated on it**. A mismatch annotates the summary and the published README. A
counting bug can cost accuracy in a report; it can no longer cost the artifacts.

**"Stop, don't delete" as a resume strategy.** Stopping a pod keeps its container
disk but does NOT reserve its GPUs. RunPod reallocated the cards, the pod could
not restart, and the index on its disk was unrecoverable. The index now lives on
a network volume, so the watchdog DELETES rather than stops (cheaper, and the
data is not on the thing being deleted), and `resume` is a new pod that is free
to land on different hardware. Verified: a pod deleted mid-campaign, a fresh pod
attached to the same volume, lilbee installed from scratch, index read, published.

The cost of the volume is a datacenter pin, because a volume cannot move. `up`
therefore picks the datacenter by live 8-pack availability at that moment rather
than from a preference, and capacity moves: CA-MTL-1 could serve an 8-pack one
hour and could not the next.

## Gotchas paid for on hardware

- **Never overwrite a running bash script.** bash reads a script incrementally by
  byte offset, so an `scp` over one mid-run makes it resume mid-token
  (`n_main,: command not found`). This once killed a completed ingest's
  post-merge steps. `launch_ingest` now runs a private snapshot under
  `/root/run/`, so re-uploading only affects the next launch.
- **rich renders nothing into a redirected file.** `lilbee sync > log` captures
  the startup warning and then silence. The sync runs under `script -qfec` for a
  pty. Its output is filtered rather than recorded verbatim, because the bar
  redraws ~10x/second: kept whole that is tens of gigabytes over a full run.
- **`lilbee.__version__` does not exist.** The lazy `__init__` raises
  AttributeError for it; use `importlib.metadata.version`.
- **`df` on the volume reports the whole cluster.** It is a MooseFS mount, so df
  reads 630T of 851T. Use `du`.
- **CPU pods are not always available** in the datacenter the volume is pinned
  to; `publish` falls back to one cheap GPU pod.
- **The venv is on the container disk**, so a replacement pod has no lilbee.
  `publish9m.sh` installs it (no engine wheel: export never embeds).
- **`pgrep -fc` prints 0 AND exits non-zero**, so a `|| echo 0` fallback emits two
  lines and every later sum becomes a syntax error. The watchdog then never
  fires, which is the failure that costs a night of GPU.
- **The watchdog cannot wait on `who`.** The dashboard holds five pty-allocating
  ssh sessions for the run's whole length, so `who` never empties. `GRACE_MIN` is
  the whole hold.

## Measured on the full corpus (8x H100 SXM, 160 vCPU, EUR-IS-3, 2026-08-01)

8,841,823 passages, Qwen3-Embedding-8B-Q8_0 at 4096 dims, index on a network volume.

| phase | wall time |
|---|---|
| stage corpus, count it, pull the 8GB embedder | 9 min 25 s |
| embed across 8 GPUs | 6 h 05 min |
| merge 8 shards (147GB) into one index | 59 min |
| corpus-wide IVF-PQ and BM25 build | 9 min |
| **total** | **7 h 17 min** |

`expected=8841823 landed=8841823 shard_sources=8841823 sync_rc=0`, no count
mismatch, every table at the corpus size and one `_meta` row.

Throughput 406 docs/s while embedding, 337 docs/s end to end. Shards finished
within 0.3% of each other and summed to exactly 8,841,823.

Per-card utilisation, sampled every 2s on all eight cards. Report both windows or
the number is misleading in one direction or the other:

| window | mean | p10 | p50 | at zero |
|---|---|---|---|---|
| while embedding (6.05 h) | 84.9% | 47% | 94% | 5.0% |
| whole run (7.29 h) | 70.7% | 0% | 94% | 20.8% |

Extraction over all 8,841,823 files: mean 3.91 ms, p50 1 ms, p99 100 ms, max
3454 ms, 94.7% at or under 3 ms.

**The merge cost five times the estimate.** 59 minutes, against ~12 extrapolated
from 800k rows on local NVMe. Folding 147GB over a network filesystem is a
different operation, and the extrapolation had no business being trusted across
11x the rows and a change of storage medium. It ran at ~2,990 rows/s, which is
faster per row than the 80k trial managed, so the batching amortises well; the
wall time is simply the bytes.

## What the full run cost that the trial could not have shown

- **A datacenter with GPU capacity may refuse network volumes.** AP-IN-1
  advertises 8xH100 and rejects volume creation outright, and nothing in the
  datacenter record says which datacenters support them. `pick_dc` returns every
  viable datacenter ranked and `up` walks them until a volume actually creates.
- **A datacenter pinned by a volume may have nothing cheap in it.** EUR-IS-3
  serves only H100 SXM: no CPU pods, no consumer cards. The publish path's
  hardcoded RTX 4090 fallback retried something that could never succeed there.
  It now asks what the datacenter actually offers and takes its cheapest card.
- **HuggingFace private storage is capped** (100GB on this account, reached at
  97.6GB). A 146GB index cannot fit under it no matter what is deleted. Public
  dataset storage is free and is what this run used in the end.
- **A failed upload was being recorded as a successful publish.** `push()` caught
  the exception, printed GAVE UP and returned None, so the script wrote
  PUBLISH_DONE over a partial index. Quota-shaped errors now fail terminally
  without retrying, and any failure is fatal to the publish.
- **`lilbee export` cannot export this corpus** without 64-bit string offsets.
  pyarrow's `string` caps a column at 2GB and the page text is ~3GB. Fixed on a
  branch and verified here: parquet 1.63GB and jsonl 4.15GB, both 8,841,823 rows.
  The export also holds the whole dataset in memory before writing: measured at
  ~77GB peak for the 4.15GB jsonl, so it needs a large machine until a streaming
  writer lands.
- **py-spy cannot profile the fan-out's workers.** `ptrace_scope=1` permits
  tracing descendants only, and the fan-out spawns its own workers, so the
  profiler is always a sibling. The old harness wrapped each worker at launch and
  was its parent. `/proc/sys` is read-only in the container, and `sysctl -w`
  prints the new value while ignoring the write, so read it back rather than
  trusting the echo. Checked up front now, with the reason logged.
