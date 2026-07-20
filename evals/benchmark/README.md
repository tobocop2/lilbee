# Benchmark harness: lilbee vs RAGFlow

A preregistered, reproducible A/B of retrieval quality. Two arms answer the
same public, human-labeled datasets with the same served model, so retrieval is
the only variable:

- **Arm A - lilbee**: the `test/retrieval-parity` branch with every parity
  feature enabled.
- **Arm B - RAGFlow**: the DeepDoc pipeline with chunking and retrieval settings
  pinned to RAGFlow's documented defaults.

One embedder and one generator are served once and pointed at by both arms, so
generation is identical and only the retrieval stack changes. Lives outside
`src/` on purpose: it never ships in the package.

Note what this framing does and does not claim. The lilbee arm runs with every
parity feature switched on while the RAGFlow arm runs at its defaults, so the
comparison is a tuned lilbee against an out-of-the-box RAGFlow, not two equally
tuned systems. The RAGFlow arm's knobs are pinned in `bootstrap_ragflow.py`
rather than left to the server, so they cannot drift with RAGFlow's version, but
they are still defaults and are reported as such.

Both arms are asked for the same *document* depth. lilbee returns results
grouped by source document; RAGFlow ranks chunks, so its collector pages until
it holds the same number of distinct parent documents. Without that, one arm is
scored on twenty documents and the other on however many its twenty chunks
happened to come from.

## Two tiers

1. **Tier 1 - retrieval, no model opinion.** Each arm returns a ranked list per
   query, written as a TREC run file. `pytrec_eval` scores it against the
   dataset's published relevance labels (qrels): nDCG@10, Recall@20, MRR@10.
   Nothing is graded by a model, so the numbers are reproducible by anyone with
   the same run files and qrels. MRR@10 truncates each run to depth 10 before
   scoring, and every metric is averaged over the qrels topic set, so a query an
   arm returned nothing for scores zero instead of leaving the denominator.
2. **Tier 2 - answer quality.** Both arms answer with the same model; RAGAS
   scores faithfulness, answer relevancy, and context precision/recall for
   *both* arms, using the judge model the manifest freezes. Each mean carries
   the number of answers that actually scored, since RAGAS cannot score every
   answer and the two arms need not fail equally often. The blind
   duplicate-arm judge from `evals/retrieval` is reused as a corroborating
   signal, with its noise floor measured by grading one arm twice under two
   equivalent phrasings of the grading prompt.

Every **Tier-1** cross-arm difference is paired per query and gets a bootstrap
95% CI and a randomization-test p-value. Because a study runs many such tests at
once, the p-values are Benjamini-Hochberg adjusted across the whole family and
significance is decided on the adjusted value; the CI is reported as the effect
size. Picking the best of several arms and quoting its raw p-value is not
evidence at that level.

**Tier 2 is not significance-tested.** RAGAS reports a mean per metric per arm
with the number of answers behind it, and no CI or p-value; a difference between
those means is not a tested result and must not be described as one. The
corroborating blind judge does run a paired per-question test on its own grading
dimensions, adjusted the same way, but that covers the judge's dimensions, not
the RAGAS metrics.

## Native vs derived labels

Passage sets (BEIR SciFact/FiQA/NFCorpus, HotpotQA, TREC-COVID) ship native TREC
qrels. They are used as published with one normalization: non-positive
judgments are dropped, which is how pytrec_eval treats them, and a topic whose
judgments are all non-positive has no relevant document to find and so is not
scorable. The document-structured QA sets (TAT-DQA, OTT-QA) have
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

# 0. Freeze the preregistration. The manifest must declare the arms and datasets
#    the run will actually compare: `stats` refuses to stamp this fingerprint on
#    any other pairing, so a comparison the manifest does not name cannot be
#    published under it.
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
#    RAGFlow's copy above). Both walk the corpus directory recursively, so a
#    nested layout reaches both indexes.

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
uv run python -m evals.benchmark answer --queries "$RUN/queries.jsonl" \
  --ground-truth "$RUN/references.json" --base-url http://127.0.0.1:9380 \
  --arm ragflow --out "$RUN/answers-ragflow.jsonl"
# Both arms are required: the tier scores both or emits nothing.
uv run python -m evals.benchmark score-ragas \
  --manifest "$RUN/manifest.frozen.json" \
  --samples-a "$RUN/answers-lilbee.jsonl" --samples-b "$RUN/answers-ragflow.jsonl" \
  --judge-base-url "$JUDGE_URL" --out "$RUN/results.jsonl"

# 6. Paired statistics (CI + p) into the same results file.
uv run python -m evals.benchmark stats --manifest "$RUN/manifest.frozen.json" \
  --metrics-a "$RUN/ir-lilbee.jsonl" --metrics-b "$RUN/ir-ragflow.jsonl" \
  --out "$RUN/results.jsonl"

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
