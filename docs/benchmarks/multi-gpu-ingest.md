# Multi-GPU ingest

What `lilbee sync` does on a machine with more than one GPU, measured end to end
with nothing configured.

## Test setup

- **Date:** 2026-07-31
- **Hardware:** RunPod, 2x NVIDIA GeForce RTX 4090 (24 GB each), 96 cores
- **Software:** lilbee at `feat/native-multi-gpu-ingest`, cu124 engine wheel, Python 3.12
- **Embedder:** `Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf`, 1024 dims
- **Corpus:** 20,000 MS MARCO passages, one file each, bucketed 1,000 per directory

## Method

Every run is `lilbee sync` from a directory holding a `.lilbee/` knowledge base.
No environment variables are set: not the device mask, not the data root, not the
engine slot, not the CPU quota. GPU utilisation is sampled every two seconds with
`nvidia-smi`; the row counts are read from the LanceDB tables afterwards.

## Results

### A first sync over the whole corpus

| | |
|---|---|
| Wall clock | 121 s |
| Throughput | 165 docs/sec |
| GPU utilisation, whole run | 68% / 67% |
| GPU utilisation, while embedding | 89% / 87% |
| Fraction of samples with the cards busy | 0.76 |

The two cards are within two points of each other at every stage. That symmetry is
the measurement: the failure this feature removes shows up as one card at 95% and
the rest at 0%, with the run completing and the index correct.

### One index at the end

| | Shards | Merged index | Input |
|---|---|---|---|
| chunks | 20,000 | 20,000 | 20,000 |
| sources | 20,000 | 20,000 | 20,000 |

Two shards, their union equal to the input, and the merged index holding exactly
that. A search against it returns hits from buckets belonging to different shards.

### A second sync with nothing changed

Finished in 8 seconds with the GPUs never waking (busy fraction 0.00), and left the
index at 20,000 chunks rather than 40,000. Both properties matter: a fan-out that
re-embedded, or one whose merge appended a second copy of everything, would pass a
row-count check on the first run and fail on the second.

### A kill mid-run

`kill -9` on the sync, 60 seconds in, with both workers embedding (86% / 79%):

| | |
|---|---|
| Worker processes before the kill | 2 |
| Engine processes before the kill | 2 |
| Orphaned workers after | 0 |
| Orphaned engine processes after | 0 |
| Rows kept in the shards | 8,000 of 20,000 |
| Rows in the merged index | 0 |

Nothing was left holding VRAM, the work already done stayed in the shards, and the
merge correctly did not run against a corpus the workers never finished.

### The sync that resumes after the kill

| | |
|---|---|
| Wall clock | 76 s |
| Throughput | 263 docs/sec |
| GPU utilisation, while embedding | 82% / 89% |
| Rows in the merged index afterwards | 20,000 of 20,000 |

The re-run planned only what the shards were missing, which is why it beat the
first sync's rate, and the index came out at exactly the corpus size. Nothing was
re-embedded and nothing was left behind.

## Reading the numbers

**Throughput is not the headline here.** Two consumer cards and a 0.6B embedder are
a small rig chosen to validate the mechanism cheaply. The throughput claim for this
work comes from 8xH100 with an 8B embedder: 152 docs/sec for one process across
eight cards against 415 for eight pinned workers.

**Utilisation is the headline.** Even, high utilisation across every card, from a
command with no configuration, is what the feature exists to produce. Mean
utilisation alone can hide starvation, so the busy-window figure (89/87) and the
share of samples where the cards are working (0.76) are reported beside it.
