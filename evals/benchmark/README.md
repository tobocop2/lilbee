# Benchmark harness: lilbee vs RAGFlow

A preregistered, reproducible A/B of retrieval quality. Two arms answer the
same public, human-labeled datasets with the same served model, so retrieval is
the only variable:

- **Arm A - lilbee**: the `test/retrieval-parity` branch with every parity
  feature enabled.
- **Arm B - RAGFlow**: the DeepDoc pipeline at its defaults.

One embedder and one generator are served once and pointed at by both arms, so
generation is identical and only the retrieval stack changes. Lives outside
`src/` on purpose: it never ships in the package.

## Two tiers

1. **Tier 1 - retrieval, no model opinion.** Each arm returns a ranked list per
   query, written as a TREC run file. `pytrec_eval` scores it against the
   dataset's published relevance labels (qrels): nDCG@10, Recall@20, MRR@10.
   Nothing is graded by a model, so the numbers are bit-for-bit reproducible by
   anyone with the same run files and qrels.
2. **Tier 2 - answer quality.** Both arms answer with the same model; RAGAS
   scores faithfulness, answer relevancy, and context precision/recall. The
   blind duplicate-arm judge from `evals/retrieval` is reused as a corroborating
   signal, with its noise floor measured by grading one arm twice.

Every cross-arm difference is paired per query and gets a bootstrap 95% CI and a
randomization-test p-value. A difference whose CI crosses zero is reported as
not significant, not as a win.

## Native vs derived labels

Passage sets (BEIR SciFact/FiQA/NFCorpus, HotpotQA, TREC-COVID) ship native TREC
qrels, used as published. The document-structured QA sets (TAT-DQA, OTT-QA) have
no retrieval labels of their own, so their qrels are **derived** from human
gold-evidence annotations by one documented pure function
(`datasets.derive_qrels_from_evidence`). Every derived dataset is marked
`label_kind: derived` in the manifest and flagged in the report.

## Pod workflow

One A100-80GB in a named tmux so a reclaimed pod reattaches. Preregister before
any data moves, then run each arm, score, and power off.

```bash
cd ~/lilbee  # the repo checkout; run everything from the repo root
RUN=/tmp/bench

# 0. Freeze the preregistration. Nothing can be cherry-picked after this.
uv run python -m evals.benchmark preregister \
  --manifest evals/benchmark/manifest.example.yaml --out "$RUN/manifest.frozen.json"

# 1. Serve the shared model with lilbee (all parity features on) and stand up
#    RAGFlow pointed at the same OpenAI-compatible endpoint.
export LILBEE_TITLE_SEARCH=true LILBEE_NEIGHBOR_EXPANSION=2 \
  LILBEE_TABLE_EXTRACTION=true LILBEE_LAYOUT_DETECTION=true LILBEE_INTENT_LLM=true
lilbee serve --port 8080 &
uv run python -m evals.benchmark.bootstrap_ragflow \
  --base-url http://127.0.0.1:9380 --api-key "$RAGFLOW_KEY" \
  --corpus-dir "$RUN/corpus" --llm-model qwen2.5-72b-instruct \
  --embedding-model qwen3-embedding

# 2. Ingest the same corpus into both systems (lilbee add; bootstrap uploaded
#    RAGFlow's copy above).

# 3. Collect a TREC run file per arm. Each query is checkpointed, so a killed
#    run resumes.
uv run python -m evals.benchmark collect --system lilbee \
  --queries "$RUN/queries.jsonl" --base-url http://127.0.0.1:8080 \
  --run-tag lilbee --run "$RUN/run-lilbee.trec" --checkpoint "$RUN/ck-lilbee.jsonl"
uv run python -m evals.benchmark collect --system ragflow \
  --queries "$RUN/queries.jsonl" --base-url http://127.0.0.1:9380 \
  --api-key "$RAGFLOW_KEY" --dataset-id "$RAGFLOW_DATASET" \
  --run-tag ragflow --run "$RUN/run-ragflow.trec" --checkpoint "$RUN/ck-ragflow.jsonl"

# 4. Tier 1: score each run against the qrels with pytrec_eval.
uv run python -m evals.benchmark score-ir --qrels "$RUN/qrels.json" \
  --run "$RUN/run-lilbee.trec" --dataset scifact --run-tag lilbee \
  --out "$RUN/ir-lilbee.jsonl"
uv run python -m evals.benchmark score-ir --qrels "$RUN/qrels.json" \
  --run "$RUN/run-ragflow.trec" --dataset scifact --run-tag ragflow \
  --out "$RUN/ir-ragflow.jsonl"

# 5. Tier 2: generate answers on each arm, then score with RAGAS.
uv run python -m evals.benchmark answer --queries "$RUN/queries.jsonl" \
  --ground-truth "$RUN/references.json" --base-url http://127.0.0.1:8080 \
  --arm lilbee --out "$RUN/answers-lilbee.jsonl"
uv run python -m evals.benchmark score-ragas \
  --samples "$RUN/answers-lilbee.jsonl" --out "$RUN/results.jsonl"

# 6. Paired statistics (CI + p) into the same results file.
uv run python -m evals.benchmark stats --manifest "$RUN/manifest.frozen.json" \
  --metrics-a "$RUN/ir-lilbee.jsonl" --metrics-b "$RUN/ir-ragflow.jsonl" \
  --arm-a-label lilbee --arm-b-label ragflow --out "$RUN/results.jsonl"

# 7. Render the report, then power the pod off.
uv run python -m evals.benchmark report \
  --results "$RUN/results.jsonl" --out "$RUN/report.md"
```

Keep the frozen manifest, run files, and qrels: with those, a third party can
re-score the whole Tier-1 comparison without the pod.

## Optional dependencies

The heavy scorers and dataset loaders are imported lazily, so the harness (and
its tests) load without them. They are kept out of the shipped lock on purpose
(installing them would otherwise drag core dependencies backward), so install
them from the standalone requirements file on the benchmark pod:

```bash
uv pip install -r evals/benchmark/requirements.txt   # pytrec_eval, ragas, beir, datasets
```

## Determinism and resume

- The frozen manifest pins the bootstrap seed, resamples, and alpha, so every
  CI is reproducible.
- `collect` and `answer` checkpoint per query. Kill and re-run freely; only
  unfinished queries repeat, and the run file is rebuilt from the full
  checkpoint each time.
