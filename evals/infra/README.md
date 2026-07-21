# Benchmark infrastructure

Every command the MS MARCO run executes, in order, with what each one produces.
This file is the run's method section: if a number in the report is questioned,
the answer should be a command on this page.

## The corpus, and why this one

`msmarco-passage`, **8,841,823 passages**.

MS MARCO passage ranking has two sub-tasks, and the one lilbee performs is *full
retrieval*: find the answer in the whole collection, not rerank a pre-selected
top-1000. Published baselines for that task on `dev/small` (6,980 queries,
MRR@10) are **0.165** for the official BM25 and **0.184** for Anserini's. Scoring
with `ir_measures` on its `trec_eval` backend puts lilbee's number in the same
units as those, with no methodology argument in between.

The larger `msmarco-qna` variant (9,048,606) was considered and rejected as the
ingest corpus. Its extra rows are not extra content: a QnA `doc_id` is literally
`{msmarco_passage_id}-{urlidx}`, so the surplus is the same passages appearing
under several URLs. Ingesting it would have made every number incomparable to
the leaderboard while adding no information, and building it requires a full
pass over `msmarco-passage` first anyway.

QnA is still used, for its **human-written answers**, joined back by
`doc_id.split('-')[0]`. That is what lets the generation tier be graded against
something a person wrote rather than something a model invented.

One caveat that belongs beside the headline number: `dev/small` carries 7,437
qrels over 6,980 queries, about **1.1 judged passages per query**. A system that
retrieves a genuinely relevant but unjudged passage is scored wrong. Every
system on that leaderboard has the same handicap so the comparison is fair, but
the metric is a relative signal, not an absolute quality measure.

## Architecture, and why the volume is not the working directory

RunPod network volumes are MooseFS. Measured on a previous run: per-file
operations are roughly **1000x slower** than local disk (open+read 12.9ms versus
0.01ms). Large sequential I/O is fine; many small files are pathological. A
LanceDB index is thousands of small files, and building one on the volume cost
a previous corpus run four hours with the GPUs idle.

So:

- **Network volume** holds big sequential files only: the corpus, and the
  finished index as a single tarball. It survives pod teardown.
- **Local NVMe** is the working set: the corpus is copied here, the index is
  built here, and only then tarred back.

The GPU count is worth a note. A previous run found multi-GPU tensor-splitting
all-reduce-bound on PCIe, but that applies to *generation*. Embedding is
data-parallel -- each card takes a different shard, no all-reduce -- so more
GPUs is the right call here for a different reason than the earlier warning
covers.

## Stage 0 - pick the region, then create the volume

A RunPod volume is pinned to a datacentre, so the datacentre has to be chosen
first, from live availability rather than the static catalogue (`sky show-gpus`
is only a price list).

```bash
# Live per-DC stock. The API sits behind Cloudflare and 403s without a browser
# User-Agent; the key is at ~/.runpod/config.toml under `apikey`.
python3 evals/infra/gpu_availability.py

sky volumes apply evals/infra/volume.sky.yaml    # created in the DC chosen above
```

## Stage 1 - hydrate, on a cheap CPU box

Downloading is I/O, not compute. It runs on the cheapest CPU pod available, not
on a GPU box billing by the hour to wait on a socket.

```bash
sky launch -c msmarco-hydrate evals/infra/hydrate.sky.yaml --retry-until-up -y
```

which runs, on the pod:

```bash
python -m evals.benchmark fetch \
  --manifest evals/benchmark/manifest.msmarco.yaml \
  --out /workspace/datasets
```

Produces on the volume: `corpus.jsonl` (8.8M passages), `queries.jsonl` (6,980),
`qrels.trec` (7,437), and `answers.json` (the QnA human answers, joined by
passage id). Provenance for the stage is appended to
`/workspace/provenance.jsonl`.

```bash
sky down msmarco-hydrate -y      # stop billing the moment the download ends
```

## Stage 2 - ingest, on the GPU box

```bash
sky launch -c msmarco-ingest evals/infra/ingest.sky.yaml --retry-until-up -y
```

which runs, on the pod:

```bash
bash evals/infra/preflight.sh          # fails BEFORE the hours start
bash evals/infra/ingest.sh
```

`preflight.sh` refuses to proceed unless CUDA actually initialises (nvidia-smi
alone does not prove it), the engine has kernels for this GPU's compute
capability (the prebuilt cu124 engine has no sm_90, so it dies on H100 with a
misleading "no CUDA-capable device"), the embedder returns a real vector, there
is disk headroom, and the trace file is writable.

`ingest.sh` copies the corpus to local NVMe, runs the ingest, and tars the index
back to the volume, with these set:

```bash
export LILBEE_INGEST_TRACE=1
export LILBEE_INGEST_TRACE_FILE=/workspace/logs/ingest_trace.log
export LILBEE_LOG_LEVEL=DEBUG
```

The trace emits one machine-parseable line per document on `lilbee.ingest.trace`:

```
extract source=... type=... elapsed_ms=... pages=... chunks=... ocr_pages=...
```

with a separate line on `lilbee.ingest.vision` whenever OCR fires. Summing
`elapsed_ms` against total wall time is what separates extraction cost from GPU
cost, and every failure is named rather than counted. The trace file is written
explicitly because the host front-end's root handler is WARNING+ and otherwise
swallows these lines.

## Stage 3 - retrieval, graded by humans

```bash
python -m evals.benchmark collect --system lilbee --run-tag lilbee \
  --queries /workspace/datasets/msmarco/queries.jsonl \
  --base-url http://127.0.0.1:8080 \
  --run /workspace/runs/lilbee.trec --checkpoint /workspace/runs/ck.jsonl

python -m evals.benchmark score-ir \
  --qrels /workspace/datasets/msmarco/qrels.trec \
  --run /workspace/runs/lilbee.trec \
  --dataset msmarco --run-tag lilbee --out /workspace/results/ir.jsonl
```

No model judges anything here. These numbers are reproducible by anyone holding
the run file and the qrels.

## Stage 4 - generation, graded against human answers

```bash
python -m evals.benchmark answer --arm lilbee --top-k 10 \
  --queries /workspace/datasets/msmarco/queries.sample.jsonl \
  --ground-truth /workspace/datasets/msmarco/answers.json \
  --base-url http://127.0.0.1:8080 --out /workspace/answers/lilbee.jsonl

python -m evals.benchmark score-ragas ...
python -m evals.benchmark score-ragchecker ...
```

A sample rather than all 101,093 queries, because that many generations per arm
is not a reasonable spend. The sample and its seed are recorded in the manifest
so the selection is reproducible rather than convenient.

## Stage 5 - grade the grader

```bash
python -m evals.benchmark calibrate \
  --work-dir /workspace/calibration --out /workspace/results/results.jsonl
```

Runs the same rubric over SummEval's expert-rated summaries and reports how
closely it tracks them, against the published expert-to-expert agreement as a
ceiling. This decides whether Stage 4's numbers carry any weight.

## Stage 6 - statistics, report, and stop billing

```bash
python -m evals.benchmark stats  --manifest ... --metrics-a ... --metrics-b ...
python -m evals.benchmark report --results /workspace/results/results.jsonl \
                                 --out /workspace/results/REPORT.md

sky down msmarco-ingest -y       # the pod stops the moment the work does
```
