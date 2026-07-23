# Benchmark harness: lilbee retrieval

A preregistered, reproducible A/B of lilbee's own retrieval quality. Two or more
arms answer the same public, human-labeled datasets with the same served model,
so one configuration feature is the only variable between them:

- **Arm A**: lilbee with the feature under test enabled (for example every
  retrieval feature on).
- **Arm B**: the same lilbee build and served model with that feature off (the
  baseline the feature is measured against).

One embedder and one generator are served once and pointed at by every arm, so
generation is identical and only the retrieval configuration changes. Lives
outside `src/` on purpose: it never ships in the package. See
`manifest.ablation.yaml` for a baseline-against-several-variants study.

Every arm is asked for the same *document* depth. lilbee returns results grouped
by source document, so one result is one document and the run is capped at the
same document depth for every arm; scoring one arm on twenty documents and
another on seven would put a pure depth artifact into the metric gap.

## Two tiers

1. **Tier 1 - retrieval, no model opinion.** Each arm returns a ranked list per
   query, written as a TREC run file. `ir_measures` scores it against the
   dataset's published relevance labels (qrels): nDCG@10, Recall@20, MRR@10.
   Nothing is graded by a model, so the numbers are reproducible by anyone with
   the same run files and qrels. The cut depth is part of the measure
   (`RR@10`, not a truncated run), and every metric is averaged over the qrels
   topic set, so a query an arm returned nothing for scores zero instead of
   leaving the denominator.
2. **Tier 2 - answer quality.** Every arm answers with the same model; RAGAS
   scores faithfulness, answer relevancy, and context precision/recall for each
   arm, using the judge model the manifest freezes. Each mean carries the number
   of answers that actually scored, since RAGAS cannot score every answer and
   two arms need not fail equally often. A blind duplicate-arm judge runs
   alongside as a corroborating signal: RAGAS' rubric metric grades each answer
   on faithfulness, relevance, and citation, and its noise floor is measured by
   grading one arm twice under two equivalent presentations of the rubric.

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
any data moves, then run each arm, score, and power off. The two arms differ only
in lilbee's configuration, so the same corpus is served twice with different
feature flags.

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

# 1. Serve arm A: lilbee with the feature under test on, over the shared model.
export LILBEE_TITLE_SEARCH=true LILBEE_NEIGHBOR_EXPANSION=2 \
  LILBEE_TABLE_EXTRACTION=true LILBEE_LAYOUT_DETECTION=true LILBEE_INTENT_LLM=true
lilbee serve --port 8080 &

# 2. Ingest the corpus (lilbee add). A nested layout is walked recursively.

# 3. Collect a TREC run file for arm A. Each query is checkpointed, so a killed
#    run resumes.
uv run python -m evals.benchmark collect \
  --queries "$RUN/queries.jsonl" --base-url http://127.0.0.1:8080 \
  --run-tag lilbee-full --run "$RUN/run-full.trec" --checkpoint "$RUN/ck-full.jsonl"

# 3b. Re-serve arm B with the feature off (baseline), then collect it the same way.
export LILBEE_TITLE_SEARCH=false LILBEE_NEIGHBOR_EXPANSION=0 \
  LILBEE_TABLE_EXTRACTION=false LILBEE_LAYOUT_DETECTION=false LILBEE_INTENT_LLM=false
lilbee serve --port 8081 &
uv run python -m evals.benchmark collect \
  --queries "$RUN/queries.jsonl" --base-url http://127.0.0.1:8081 \
  --run-tag lilbee-baseline --run "$RUN/run-baseline.trec" --checkpoint "$RUN/ck-baseline.jsonl"

# 4. Tier 1: score each run against the qrels with ir_measures.
uv run python -m evals.benchmark score-ir --qrels "$RUN/datasets/scifact/qrels.trec" \
  --run "$RUN/run-full.trec" --dataset scifact --run-tag lilbee-full \
  --out "$RUN/ir-full.jsonl"
uv run python -m evals.benchmark score-ir --qrels "$RUN/datasets/scifact/qrels.trec" \
  --run "$RUN/run-baseline.trec" --dataset scifact --run-tag lilbee-baseline \
  --out "$RUN/ir-baseline.jsonl"

# 5. Tier 2: generate answers on each arm, then score with RAGAS.
uv run python -m evals.benchmark answer --queries "$RUN/queries.jsonl" \
  --ground-truth "$RUN/references.json" --base-url http://127.0.0.1:8080 \
  --arm lilbee-full --out "$RUN/answers-full.jsonl"
uv run python -m evals.benchmark answer --queries "$RUN/queries.jsonl" \
  --ground-truth "$RUN/references.json" --base-url http://127.0.0.1:8081 \
  --arm lilbee-baseline --out "$RUN/answers-baseline.jsonl"
# Both arms are required: the tier scores both or emits nothing.
uv run python -m evals.benchmark score-ragas \
  --manifest "$RUN/manifest.frozen.json" \
  --samples-a "$RUN/answers-full.jsonl" --samples-b "$RUN/answers-baseline.jsonl" \
  --judge-base-url "$JUDGE_URL" --out "$RUN/results.jsonl"

# 6. Paired statistics (CI + p) into the same results file.
uv run python -m evals.benchmark stats --manifest "$RUN/manifest.frozen.json" \
  --metrics-a "$RUN/ir-full.jsonl" --metrics-b "$RUN/ir-baseline.jsonl" \
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
