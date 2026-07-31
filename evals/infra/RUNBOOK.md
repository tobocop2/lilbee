# Ingesting 8.8M MS MARCO passages on one 8xH100 box

How to run it, what each setting does and why, and what to watch. Written to be
followed by a human with no agent involved.

## TL;DR

```bash
cd ~/msmarco-ingest/evals/infra
./stage_payload.sh ~/msmarco-ingest/payload-kz5      # build the wheel under test
PAYLOAD=~/msmarco-ingest/payload-kz5 ./pod9m.sh up   # provision, run, attach the monitor
```

Detach with `ctrl-b d`; the ingest survives. `./pod9m.sh down` when finished.

## What the architecture is

One lilbee process per GPU, each with a private data root, all writing their own
LanceDB. A merge folds the shards into one index at the end. This exists because
**a single lilbee process saturates exactly one card**: measured 59-60 docs/s at
94% SM on one card, but 161.9 docs/s across four cards and *152* across eight, so
the eighth card made things worse than the fourth.

Per-GPU workers scale linearly instead: ~61.5 docs/s per card at every width from
2 to 8.

## The four settings that make a worker

Passed as environment variables, one set per worker. Nothing else differs.

| variable | value | why |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `i` | the card this worker owns |
| `LILBEE_DATA` | `/root/w<i>` | private data root: config, documents, lancedb |
| `LILBEE_ENGINE_DIR` | `/root/w<i>/engine` | private engine slot |
| `LILBEE_ANN_INDEX_THRESHOLD` | `0` | no per-shard ANN build |

**`LILBEE_ENGINE_DIR` is not optional.** Without it every worker scans the
machine-wide engine slot that `machine_engine_dir()` returns, finds worker 0's
live fleet and adopts it. The measured result was one card at 95% and seven at
0% — seven idle GPUs, no error message.

**`LILBEE_ANN_INDEX_THRESHOLD=0` is the throughput setting.** Default is 50,000
rows. Above it, every worker builds an IVF_PQ index over its own vectors, and the
merge then rebuilds one corpus-wide anyway (`ensure_vector_index(force=True)`), so
the per-shard build is pure waste. Measured: 10k rows/worker (under the default)
idled the GPUs 25% of the run; 100k rows/worker (over it) idled them 45%.

Pass it as an **environment variable, not in config.toml**. In at least one
environment the TOML source lost silently to the defaults, and a setting that is
quietly ignored here costs hours of GPU.

`LILBEE_INGEST_WORKERS` (the planning-pool share) is **not** in the list: an A/B
at width 8 measured 307.7 docs/s at the default against 312.5 with the pool
divided, which is inside noise. Do not bother setting it.

## The one lilbee code change: the port fix (PR #641, merged)

Before it, two lilbee processes starting at the same moment could hand their
fleets overlapping ports, so one worker's client drove another worker's engine.

The picker spread its search start by pid, but a fleet takes its ports
*contiguously*, so consecutive pids (which is what any launch loop produces)
overlapped on all but one port. The probe socket closes long before llama-server
binds, so the sibling that starts a moment later finds them free.

The fix partitions the sub-ephemeral window into 64-port blocks and lets the pid
select a whole block. Measured on 2xH100, changing only the launch stagger:

| launch | docs/s | per-card util |
|---|---|---|
| simultaneous, before | 53.9 | 8% / 87% |
| 15s stagger, before | 98.5 | 90% / 88% |
| simultaneous, after | 100.0 | 95% / 94% |

Verified at width 8 by reading the ports the engines actually bound: every one
landed on a 64-boundary, all eight blocks distinct, all eight cards holding their
own engine at ~9,868 MiB.

The launch stagger is no longer needed at any width.

## Corpus layout is load-bearing

The corpus must stay bucketed **1000 files per directory**. A validation run that
dealt 10,000 files into one directory measured 15.0 docs/s end-to-end; the same
run bucketed measured 49.9. The plan stream is pull-driven, so a slow discovery
stat scan over a huge directory starves it and leaves the GPU idle.

The launcher splits whole bucket directories round-robin across workers, keeping
each bucket's original name and hard-linking (`cp -al`) rather than copying. Same
names means a worker's source keys are the keys a single-host ingest produces,
which is what makes the shards comparable to a single-host index at all.

## Running it

```bash
cd ~/msmarco-ingest/evals/infra
./stage_payload.sh ~/msmarco-ingest/payload-kz5
PAYLOAD=~/msmarco-ingest/payload-kz5 ./pod9m.sh up
```

| command | does |
|---|---|
| `./pod9m.sh up` | provision, upload, launch, attach the monitor |
| `./pod9m.sh attach` | re-attach from any terminal |
| `./pod9m.sh status` | one-shot summary, no tmux |
| `./pod9m.sh fetch` | copy logs and numbers into `results/` |
| `./pod9m.sh resume` | restart a pod that stopped itself, to fetch results |
| `./pod9m.sh down` | delete the pod |

Knobs: `DISK` (default 500GB), `HOURS` (terminate-after backstop, 12),
`HF_REPO`, `UPLOAD_INDEX=0` to skip the ~150GB index push, `EXTRACT_GLOB` to
unpack a subset for a trial.

### The monitor

Five tmux windows, every pane titled in its border:

- **overview** — progress+eta, run log, plan rate, disk and run state
- **workers** — one pane per card, tailing that worker's sync log
- **gpu** — per-card util/vram/watts, plus `nvidia-smi dmon`
- **merge** — waits, then follows the merge and verification
- **shell** — scratch

### What to watch, in order

1. **Every card holds ~9,868 MiB.** A card at 0 MiB means its worker adopted a
   sibling's fleet: kill the run, the engine dir is not set.
2. **`busy_frac` in the progress pane.** This is the real health metric. Above
   ~0.85 means the GPUs are fed. Utilisation is *not* a substitute: it reads
   90%+ while the cards chew undersized batches.
3. **Rows tracking across workers.** They get equal slices, so they should stay
   within a few percent of each other.

## Cost and duration

At ~61.5 docs/s per card and 8 cards, 8,841,823 passages is roughly **5 hours**
of streaming, plus the merge. On RunPod 8xH100 at $23.92/hr that is about **$120
plus the merge tail**.

The merge is not the threat it first looked. Two measurements — 84s at 80k rows,
134s at 800k — fit `78s + 6.9e-5 s/row`, putting 8.8M at roughly **12 minutes**.
(An earlier single-point extrapolation said 2.6 hours; that was wrong.)

## Artifacts

At the end the run exports and pushes to HuggingFace
(`beeberg/msmarco-ingest-checkpoint` by default):

- `dataset/` passage text as parquet and jsonl, from `lilbee export`
- `index/` the full lilbee data root: vectors, ANN and FTS
- `README.md` the run's settings and measured numbers

Subdirectories, so anything already in the repo is untouched. The export runs
*before* the run's done-marker so the idle watchdog cannot stop the pod
underneath an upload in flight.

To use the index afterwards: `lilbee search --data-dir <dir>` or `LILBEE_DATA=<dir>`.

## Not paying for an idle box

A watchdog on the pod stops it via the RunPod API once the run finishes (after a
grace window) or after 30 minutes with no busy card, no worker, and no ingest
script alive. Stopping ends GPU billing while keeping the disk, so results
survive: `./pod9m.sh resume` brings it back, `down` deletes it. The pod's
`terminate-after` is the backstop behind both.

Two things that broke here and are worth knowing:

- The pod's `runpodctl` is 1.14, whose verb is `stop pod <id>`; newer CLIs use
  `pod stop <id>`. The watchdog tries both.
- `poweroff` inside a container returns success and does nothing, so it cannot be
  the middle of an `||` chain. Killing init is the terminal action.

## Verifying the result

`compare_index.py <reference> <candidate> [floor]` checks a merged index against
a single-host index built from the same corpus: same row counts, same source
keys, same file hashes, one `_meta` row, identical chunk text, and vector drift
no worse than a single host shows against its own re-run.

That last part is why a floor argument exists — embeddings are not deterministic,
so without a self-comparison any drift number is unfalsifiable.

Measured at 800k: all checks passed, drift mean 7.9e-07 against a floor of
6.6e-07.

## Known limits

- **The plan stream prefetches exactly one shard ahead** (`_plan_shards`), so
  planning and embedding are locked together and neither can buffer against the
  other. Deepening that queue is a lilbee change nobody has made.
- **Every worker still builds an FTS index** the merge discards. Only the ANN
  build can be switched off by configuration today.
