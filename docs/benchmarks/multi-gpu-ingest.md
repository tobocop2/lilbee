# Multi-GPU ingest

What `lilbee sync` does on a machine with more than one GPU, measured end to end
with nothing configured.

## Test setup

- **Date:** 2026-07-31
- **Hardware:** RunPod, 2x NVIDIA GeForce RTX 4090 (24 GB each)
- **Software:** lilbee at `feat/native-multi-gpu-ingest`, cu124 engine wheel, Python 3.12
- **Embedder:** `Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf`, 1024 dims
- **Corpus:** 20,000 MS MARCO passages as one file each, bucketed 1,000 per directory

## Method

Every run is `lilbee sync` from a directory holding a `.lilbee/` knowledge base.
No environment variables are set: not the device mask, not the data root, not the
engine slot, not the CPU quota. GPU utilisation is sampled every two seconds with
`nvidia-smi`, and the counters come from the LanceDB tables afterwards.

Four things are measured:

1. a first sync over the whole corpus,
2. a second sync with nothing changed,
3. `kill -9` on the sync mid-run, checking what it leaves behind,
4. the sync that resumes after that kill.

## Results

PENDING

## Reading the numbers

PENDING
