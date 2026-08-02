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

## Is the merged index the index a single process would build?

Matching row counts are weak evidence: they match just as well if the merge
duplicated one source and dropped another. The oracle is a single-process ingest
of the same corpus, diffed against the merged one.

Three ingests of one shared documents dir on 2x RTX 4090, 20,000 passages: a
reference (`ingest_processes = 1`), a floor (the same again), and the candidate
(native fan-out). Embeddings are not deterministic, so the floor is what makes
the drift claim falsifiable: without it, any drift number can be called small.

| Check | Result |
|---|---|
| Same tables, same row count in each | chunks / sources / page texts 20,000, one meta row |
| Same source keys, same file hashes | 0 differ |
| Same embedder identity | matches |
| Same chunk text for every (source, chunk_index) | 0 of 20,000 differ |
| Vector drift against the reference | mean 2.455e-06 |
| Drift a single host shows against its own re-run | mean 2.098e-06 |

The candidate sits at the floor, so what is left is embedding noise rather than
anything the merge did.

Structural checks on the merged store, which the diff above does not cover: no
duplicate source keys, no duplicate `(source, chunk_index)`, no null vectors,
every vector the configured width, exactly one meta row recording the right
dimension, FTS and scalar indexes present on the merged chunks table, and a
search returning five hits drawn from four different buckets.

## Is it as fast as configuring it by hand?

The question that matters for anyone already running one lilbee per card: does
deriving the arrangement cost anything against typing it? Measured on 8x H100
80GB HBM3, 224 cores, 250,000 MS MARCO passages, Qwen3-Embedding-8B Q8 at 4096
dims. Three arms back to back on one box, same corpus, same wheel, no ANN build
in any of them.

| Arm | Mechanism | Ingest | docs/sec | Per-card while embedding | Busy |
|---|---|---|---|---|---|
| A1 | environment variables, 8 workers | 646 s | 387.0 | 91/93/91/90/90/90/90/90 | 0.90 |
| B | native, one bare `lilbee sync` | 647 s | 386.4 | 91/92/89/93/91/90/93/90 | 0.85 |
| A2 | environment variables again | 639 s | 391.2 | 90/93/92/91/90/90/91/90 | 0.91 |

A1 and A2 are the same mechanism run twice, and they disagree by 1.09%. Native
differs from their mean by 0.69%. The gap between mechanisms is smaller than the
gap between two identical runs, so dropping the four environment variables costs
nothing measurable.

Both arms run the same code. An environment-variable worker sees one card, so
the fan-out gate declines and it takes the in-process path; the A/B isolates the
mechanism rather than comparing two builds.

Native additionally pays a 28 s merge at this corpus size, taking it from 386.4
to 370.4 docs/sec end to end. The harness arm does not pay that inside the run
because the operator pays it afterwards by hand.

## Reading the numbers

**The two rigs answer different questions.** The 2x4090 runs above test behaviour
(one index, resume, no orphans) cheaply. The 8xH100 arms test throughput against
the hand-configured alternative.

**One number was not reproduced.** The 8xH100 box measured 387-391 docs/sec for
both mechanisms, where an earlier 800k run on this hardware recorded 415. Corpus
size does not explain it (a smaller run should be faster, since throughput falls
as a worker's own table grows) and core count matches at 224. The gap is tracked
rather than explained away; the parity claim above does not depend on it.

**Utilisation is the headline.** Even, high utilisation across every card, from a
command with no configuration, is what the feature exists to produce. Mean
utilisation alone can hide starvation, so the busy-window figure (89/87) and the
share of samples where the cards are working (0.76) are reported beside it.
