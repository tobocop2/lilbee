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

## Measured on the trial (2 x RTX 4090, 80k passages, Qwen3-Embedding-0.6B)

| | |
|---|---|
| throughput | 159 docs/s whole-run, 200 docs/s steady state |
| per-card utilisation | 92-96% on both cards |
| shard split | 40,041 / 39,959 of 80,000 |
| extraction | p50 1 ms, mean 2.32 ms, 95.2% at or under 3 ms |
| merge + ANN + FTS | under 45s at 80k |
| resume | 14s, `Unchanged: 80000`, merged index still 80,000 (not doubled) |
| index on disk | 408 files, 405 MB |
| volume vs local disk | no penalty: 165.3 docs/s was the container-disk reference |

The volume is not a throughput problem for the ingest phase. What the trial does
NOT establish is the merge and ANN build at 8.8M x 4096, which is 110x the rows
and 4x the dimensions; and the index folder at that scale is roughly 45,000 files
and ~158GB, which is an upload the trial only exercised at 409 files.
