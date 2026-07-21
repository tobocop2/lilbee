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
   query, written as a TREC run file. `ir_measures` scores it against the
   dataset's published relevance labels (qrels): nDCG@10, Recall@20, MRR@10.
   Nothing is graded by a model, so the numbers are reproducible by anyone with
   the same run files and qrels. The cut depth is part of the measure
   (`RR@10`, not a truncated run), and every metric is averaged over the qrels
   topic set, so a query an arm returned nothing for scores zero instead of
   leaving the denominator.
2. **Tier 2 - answer quality.** Both arms answer with the same model; RAGAS
   scores faithfulness, answer relevancy, and context precision/recall for
   *both* arms, using the judge model the manifest freezes. Each mean carries
   the number of answers that actually scored, since RAGAS cannot score every
   answer and the two arms need not fail equally often. A blind duplicate-arm
   judge runs alongside as a corroborating signal: RAGAS' rubric metric grades
   each answer on faithfulness, relevance, and citation, and its noise floor is
   measured by grading one arm twice under two equivalent presentations of the
   rubric.

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
qrels and load through `ir_datasets`, whose id (`beir/fiqa/test`) names the
corpus, the split, and the published copy, and which checksums what it
downloads. They are used as published with one normalization: non-positive
judgments are dropped, which is how trec_eval treats them, and a topic whose
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

# 0b. Materialize the manifest's datasets: corpus, queries, and TREC qrels.
uv run python -m evals.benchmark fetch \
  --manifest evals/benchmark/manifest.example.yaml --out "$RUN/datasets"

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

# 4. Tier 1: score each run against the qrels with ir_measures.
uv run python -m evals.benchmark score-ir --qrels "$RUN/datasets/scifact/qrels.trec" \
  --run "$RUN/run-lilbee.trec" --dataset scifact --run-tag lilbee \
  --out "$RUN/ir-lilbee.jsonl"
uv run python -m evals.benchmark score-ir --qrels "$RUN/datasets/scifact/qrels.trec" \
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

Keep the frozen manifest, run files, and qrels. All three are standard formats,
so a third party can re-score the whole Tier-1 comparison without the pod and
without this harness.

## Optional dependencies

The harness is its own uv project (`evals/pyproject.toml`) with its own lock,
deliberately separate from lilbee's. A dependency group in the root would
resolve together with lilbee's dependencies, so ragchecker's torch, transformers
and spaCy would enter the resolution for the shipped package; and the root build
packages `src/lilbee` only, so the harness has no business in that lock either
way.

The lock is the reason this is not a requirements file. Pinning direct
dependencies while every transitive one floats is how the answer tier broke
before: ragas declares no upper bound on langchain-community, 0.4 removed a
module ragas imports unconditionally, and a fresh install of the *pinned* ragas
raised ImportError on every call. A resolved lock covers the whole graph.

```bash
uv sync --project evals                      # tier 1: retrieval scoring
uv sync --project evals --extra generation   # adds the answer tier (heavy: torch, spaCy)
uv sync --project evals --all-extras         # everything, including the audit statistics
```

Every scorer is imported lazily, so the harness and its tests still load without
the extras. Run from the repository root so both packages are importable:

```bash
PYTHONPATH=. uv run --project evals python -m evals.benchmark --help
```

The answer tier's RAGChecker needs its spaCy model as a separate step:

```bash
uv run --project evals python -m spacy download en_core_web_sm
```

## Determinism and resume

- The frozen manifest pins the bootstrap seed, resamples, and alpha, so every
  CI is reproducible.
- `collect` and `answer` checkpoint per query, so killing and re-running repeats
  only unfinished queries, and the run file is rebuilt from the full checkpoint
  each time. Each checkpoint records the arm and configuration that produced it
  and refuses to resume under a different one, so pointing the second arm at the
  first arm's checkpoint path fails loudly instead of emitting a complete run
  file built from the wrong arm's hits.
