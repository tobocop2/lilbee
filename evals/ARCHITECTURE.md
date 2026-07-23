# How the lilbee evaluation harness is built, and why you should believe it

This document exists because a benchmark that grades its own author is worth
exactly as much as the evidence it publishes about itself. It covers what the
harness measures, how each subsystem works, what has been done to make the
numbers trustworthy, and, in the last section, what it deliberately does not
claim. Read the last section before quoting a number from this harness anywhere.

The harness never ships. It lives outside `src/`, the wheel packages
`src/lilbee` only, and it is its own uv project with its own lock.

---

## 1. What it measures

It grades **lilbee against lilbee**: a baseline configuration against one or more
variants, on public datasets with human relevance labels, so that exactly one
thing differs between arms and any gap is attributable to that thing.

It answers "did this change help, and by how much, and is that a real effect or
noise?" It is not a cross-vendor bake-off; see §7.

Two tiers, with deliberately different epistemic status:

| | Tier 1: retrieval | Tier 2: generation |
|---|---|---|
| Scored by | `ir_measures` against published qrels | RAGAS + RAGChecker + a blind rubric judge |
| Model opinion involved | none | yes (an LLM judge) |
| Metrics | nDCG@10, Recall@20, MRR@10 | faithfulness, answer relevancy, context precision/recall, claim-level attribution |
| Significance tested | yes, paired + BH-adjusted | **no** (means with n only) |
| Reproducible by a third party | yes, from run files + qrels alone | only with the same judge model |

Tier 1 is the reproducible core and is what any conclusion should rest on.
Tier 2 corroborates; it is reported as means with the count behind each, never as
a tested result.

---

## 2. Subsystem map

Data flows in one direction: **fetch → collect → score → stats → report.**

```
manifest.py    preregistration: freeze arms/datasets/metrics/judge/seeds
    │
datasets.py    ir_datasets → corpus, queries, TREC qrels  (native or derived)
    │
collectors.py  query each arm's /api/search → chunk hits → checkpointed
runfile.py     collapse chunks to documents, cap at depth, write TREC run
    │
metrics.py     ir_measures → per-query + aggregate scores, judged@k coverage
    │
stats.py       paired bootstrap CI, sign-flip permutation p, Benjamini-Hochberg
experiment.py  PyTerrier multi-arm comparison table (optional cross-check)
    │
report.py      markdown: scores, effect sizes, adjusted p, coverage, provenance
```

### Preregistration (`manifest.py`)
A YAML manifest declares run id, arms and their configs, models (embedder,
generator, judge), datasets and label kind, metrics, bootstrap seed/resamples/
alpha, and the system under test. `preregister` freezes it and stamps a content
fingerprint.

What the frozen manifest actually enforces, mechanically:
- Pydantic models are `frozen=True, extra="forbid"`, so a misspelled key is a
  load error rather than a setting that silently does nothing.
- The stored fingerprint is re-verified against the content on load, and the
  fingerprint field is excluded from its own hash, so a post-hoc edit to a frozen
  manifest is detected instead of being made self-consistent.
- `judge != generator` fails validation. The judge cannot be the model that wrote
  the answers.
- `temperature` must be 0.
- Every arm's `system` must be `lilbee`.
- `stats` refuses to stamp the fingerprint on any arm pair or dataset the
  manifest does not declare, on a cross-dataset pairing, or on an arm compared
  with itself.
- `preregister` warns when the manifest does not record the build under test
  (`system.lilbee_commit`, `chunk_size`), i.e. when the run will be rescorable
  but not rebuildable.

### Datasets (`datasets.py`)
Loads through `ir_datasets` by pinned id (`beir/fiqa/test`), which names corpus,
split and published copy and checksums the download. Labels are **native** (the
dataset ships TREC qrels) or **derived** (QA sets with no retrieval labels, whose
qrels come from human gold-evidence annotations via one documented pure
function). Derived sets are marked in the manifest and flagged in the report, so
a reader always knows which kind of label produced a number.

### Collection (`collectors.py`, `runfile.py`, `checkpoint.py`)
Each arm is queried over HTTP. lilbee returns results grouped by source document,
so one result is one document; the run is capped at the same document depth
(default 20, tied to Recall@20) for every arm, so no arm is scored at a different
depth than another.

Ranking ties are broken by score descending then **doc_id descending**, which is
trec_eval's own rule. This matters because the scorer discards the rank column
and re-sorts; writing any other order would state one ranking while the scorer
used another.

Every query is checkpointed as it lands, and the checkpoint records a fingerprint
of the arm and configuration that produced it. Resuming under a different
configuration fails loudly rather than silently emitting a complete run file
built from two different arms' hits.

### Metrics (`metrics.py`)
Entirely delegated to `ir_measures` on its pytrec_eval backend. The harness
truncates nothing and breaks no ties itself: the cut depth lives inside the
measure string (`RR@10`), which is the whole depth contract, so there is no
second place a depth can drift out of step.

Two properties are load-bearing and are demonstrated in §4:
- **Denominator.** Metrics average over the *qrels* topic set, so a query an arm
  returned nothing for scores zero rather than leaving the mean. An arm cannot be
  rewarded for failing on its hard queries.
- **Coverage (`judged@k`).** The share of each arm's top 10 that carries any
  human judgment. BEIR qrels are pooled from systems that existed when the set
  was built, and anything outside the pool is scored non-relevant by convention.
  That convention is correct and is what keeps these numbers comparable to
  published baselines, but it penalises out-of-pool retrieval at a rate nobody
  measured. This reports the rate.

### Statistics (`stats.py`)
Every difference is paired per query and delegated to scipy:
- percentile bootstrap CI (`scipy.stats.bootstrap`)
- paired sign-flip permutation p (`scipy.stats.permutation_test`,
  `permutation_type="samples"`)
- Benjamini-Hochberg across the family (`scipy.stats.false_discovery_control`)

Nothing statistical is hand-rolled. The bootstrap and the permutation test use
distinct sub-seeds so the CI and the p-value are not the same draws twice, and
both seeds travel in the output row.

A p-value at the resampling floor is reported as a bound (`< 2.0e-04`), not as a
point estimate; the floor is two-sided, `2/(resamples+1)`.

Published significance is the **BH-adjusted p across every comparison in the
study**. The per-row bootstrap verdict is stored as `ci_excludes_zero`, named for
what it is so a consumer of the raw results file cannot mistake the uncorrected
single-test result for the study verdict.

### Answer tier (`ragas_tier.py`, `ragchecker_tier.py`, `judging.py`, `blinding.py`)
- **RAGAS** scores faithfulness, answer relevancy, context precision/recall
  through the manifest's pinned judge and temperature. NaN is counted, not
  dropped, and each mean carries the number of answers that actually scored.
- **RAGChecker** cross-checks it and splits the result into retriever-side and
  generator-side, which RAGAS cannot do (its faithfulness moves for either
  cause). Generator metrics mix polarities (more faithfulness is better, more
  hallucination is worse), so deltas are oriented before averaging: positive
  always means better.
- **Blind judge.** A rubric grader whose prompts receive only question, answer
  and ground passage. Arm, query id, replicate and source filename never enter
  the prompt, and rows are shuffled with a seeded RNG so file order carries no
  signal. Grade ids are content hashes of (qid, arm, replicate, answer), so a
  regenerated answer misses the checkpoint rather than inheriting a stale grade.
- **Noise floor.** One arm is graded twice under two genuinely different
  presentations of the same rubric, so byte-identical prompts at temperature 0
  cannot manufacture a floor of zero. A missing replicate raises instead of
  silently reporting 0.0.

### Judge calibration (`calibration.py`)
The obvious objection to Tier 2 is that one model grades another with no stated
error rate. The harness answers with a number rather than an assurance: the same
rubric is run over **SummEval** (100 articles × 16 summaries, each rated 1-5 by
three experts, years before this project existed), and the judge's grades are
correlated with the expert means. Spearman and Kendall are reported beside the
published inter-expert agreement.

Two honesty constraints are built in: citation is not calibrated (a summary cites
nothing, so no human label exists and inventing a mapping would be worse than
reporting the gap), and the ratio to expert agreement may exceed 100% because the
judge is correlated against the *mean* of three experts, which is more reliable
than any single expert. It is a reference point, not a hard ceiling.

### Report (`report.py`)
Renders scores, effect sizes, adjusted p, pool coverage, judge calibration,
scorer versions and coverage matrix. Tables are built by pandas from records
rather than assembled from f-strings, so a column count cannot silently
misalign.

Each row is labelled with **its own** arm pair. A results file accumulates
several comparisons (BH runs across all of them) and an ablation's comparisons do
not share one arm pair, so a single file-level label would print one comparison's
scores under another's arm names. A file holding several comparisons is titled
for the run and names each comparison in the table rather than claiming a single
pairing.

---

## 3. Dependency policy

The harness is its own uv project with a resolved lock covering the whole graph,
not a requirements file of direct pins. That distinction is not theoretical:
ragas declares no upper bound on langchain-community, 0.4 removed a module ragas
imports unconditionally, and a fresh install of the *pinned* ragas raised
ImportError on every call. A lock catches that class of break at lock time
instead of on the pod.

Scorers are pinned exactly (`ir_measures==0.4.3`, `pytrec_eval-terrier==0.5.10`,
`ir_datasets==0.6.3`, `python-terrier==1.0`, `scipy==1.18.0`, `ragas==0.4.3`,
`ragchecker==0.1.9`), because a scorer version moving without anyone deciding to
move it changes published numbers.

Heavy tiers are extras, so tier-1 scoring does not pull torch:

```bash
uv sync --project evals                      # tier 1
uv sync --project evals --extra generation   # + answer tier
uv sync --project evals --all-extras         # + judge calibration
```

---

## 4. Empirical evidence

Claims above are cheap. This section is what can be checked.

### The test suite runs, and now gates

**229 tests, all passing**, run under the harness's own environment:

```bash
PYTHONPATH=. uv run --project evals pytest tests/evals --confcutdir=tests/evals
# 229 passed
```

Distribution across the measurement-critical modules (the two that used to have
zero coverage, `metrics`/`ir_metrics` and `stats`, are now among the best
covered):

| area | tests | area | tests |
|---|---|---|---|
| manifest (preregistration) | 25 | provenance | 7 |
| stats | 22 | calibration | 7 |
| questions | 15 | blinding | 7 |
| judge noise | 14 | benchmark CLI | 7 |
| scoring | 13 | report (retrieval) | 6 |
| RAGAS tier | 13 | judging | 6 |
| RAGChecker tier | 12 | gid identity | 6 |
| datasets | 11 | checkpoint identity | 6 |
| experiment (PyTerrier) | 10 | report (benchmark) | 6 |
| store scan | 9 | answers | 6 |
| metrics | 7 | collectors | 5 |
| | | checkpoint | 5 |
| | | retrieval CLI | 4 |

They also **run in CI now** (`.github/workflows/evals.yml`), and the harness
source is covered by `make lint` under the repository's single ruff config. Both
were previously true of neither: the suite ran in no workflow and `evals/` was
never linted.

### The delegation is real, not claimed

Delegation that isn't verified is just a comment. Run this:

```python
from evals.benchmark.metrics import score_run, judged_at_k, METRIC_MEASURES
print(METRIC_MEASURES)
# {'nDCG@10': 'nDCG@10', 'Recall@20': 'R@20', 'MRR@10': 'RR@10'}
#  ^ the cut depth is inside the measure string; nothing here truncates a run

qrels = {"q1": {"d1": 1}, "q2": {"d9": 1}}   # two judged topics
run   = {"q1": {"d1": 10.0}}                  # the arm answered only one
out = score_run(qrels, run, ["nDCG@10"])
print(out["per_query"]["nDCG@10"])   # {'q1': 1.0, 'q2': 0.0}
print(out["aggregated"]["nDCG@10"])  # 0.5   <- unanswered topic scored 0, not dropped
```

Observed output confirms the denominator guarantee: the topic the run never
answered scores 0 and stays in the mean.

```python
from evals.benchmark.runfile import ChunkHit, collapse_hits
hits = [ChunkHit("q1","d1",1.0), ChunkHit("q1","d2",1.0), ChunkHit("q1","d3",1.0)]
print([e.doc_id for e in collapse_hits(hits, "arm")])   # ['d3', 'd2', 'd1']
```

Ties break on doc_id **descending**, matching what the scorer will do after it
re-sorts. Both properties are pinned by tests with exact expected values, not by
"does not throw" assertions.

### The failure modes it refuses, by construction

Each of these is a guard with a test behind it, and each exists because it is a
way a benchmark can look authoritative while being wrong:

| Failure mode | What stops it |
|---|---|
| Metric labelled `@10` but computed uncut | depth lives in the measure string; test asserts every metric carries its depth |
| Unanswered queries dropped from the mean | averaged over qrels topics; test asserts aggregate 0.5, not 1.0 |
| Run file states a ranking the scorer won't use | single tie rule shared with the PyTerrier path; test pins `['d3','d2','d1']` |
| Arms scored at different depths | one document-depth cap applied to every arm |
| A comparison's scores printed under another's arm names | per-row arm labels; test renders a 2-comparison file and asserts both are named |
| Reporting best-of-N raw p as a win | BH across the whole family; stored per-test flag renamed `ci_excludes_zero` |
| A p-value floor quoted as a measurement | rendered as `< 2.0e-04`; floor is two-sided |
| A model grading its own output | `judge != generator` fails manifest load; no fallback to the lilbee chat model |
| Zero judge noise floor | two different rubric presentations; missing replicate raises |
| Arm identity leaking to the judge | prompt carries question/answer/passage only; seeded shuffle; content-hash grade ids |
| A stale checkpoint reproducing the previous arm's run | checkpoints fingerprint arm + config and refuse a mismatched resume |
| Metric absent from a scored file rendering as a measured tie | `stats` raises instead of comparing two empty vectors |
| A frozen manifest edited after the numbers landed | fingerprint re-verified on load, excluded from its own hash |
| Comparing arms the study never declared | `require_declared_comparison` rejects undeclared arms/datasets/self-pairs |
| Generator improvement cancelled by a rise in hallucination | polarity-oriented deltas; test asserts a hallucination rise reads negative |
| Doc-id namespace mismatch read as "bad system" | `judged@k`; a zero triggers an explicit warning naming the real cause |

### A null control is preregistered

`manifest.ablation.yaml` declares an **A/A comparison** on fiqa: BEIR's fiqa
corpus carries no title field, so the `dense` and `dense-notitle` arms are the
same system there. Any significant difference between them is a defect in the
harness, not a finding about retrieval. A benchmark that ships a way to catch
itself lying is a different object from one that does not.

---

## 5. What was fixed, and what that says

This harness is a rewrite. Its predecessor failed a measurement-validity audit
with 63 findings, 20 of them blockers, where a blocker meant "reports a wrong
number, or states a methodological claim that is false." The classes that audit
found are the checklist §4 is organised around, and the current design closes
them structurally rather than by discipline.

A subsequent adversarial review of *this* harness (six code reviewers plus three
simulated hostile readers: a statistician, an IR researcher, a vendor-benchmark
skeptic) found further defects, since fixed:

- the report labelled every comparison with the first one's arm names
- `generator_delta` averaged metrics of opposite polarity into one signed number
- the report renderer crashed in the committed environment (`tabulate` was used
  but not declared or locked)
- `judge_backend()` could not record the judge's identity without importing the
  whole generation extra
- the report asserted a judge/generator distinction it did not derive, presented
  a per-question spread as an error bar on a difference of means, and claimed a
  correlation ceiling that the statistic can legitimately exceed
- `require_reproducible()` was documented as the reproducibility gate and called
  by nothing
- the test suite ran in no CI and the harness source was never linted

The point of listing them is not that the harness is now perfect. It is that the
defect class "looks authoritative while unsound" is the one being hunted, and
that hunting it is a repeated, adversarial activity rather than a one-time
review.

---

## 6. What this harness does **not** claim

Read this before quoting a number.

1. **No study has been run under the corrected harness yet.** Everything above
   describes an instrument, not results. There are no published lilbee numbers to
   cite from this branch.
2. **Tier 2 is not significance-tested.** RAGAS means carry an n and no CI or
   p-value. A difference between two RAGAS means is not a tested result and must
   not be described as one.
3. **The RAGAS tier's two arms are averaged over whatever each arm scored**, and
   the two need not fail on the same questions. The count is disclosed per arm,
   but the means are not paired the way the retrieval tier's are. Treat Tier 2 as
   corroboration, never as the finding.
4. **Preregistration binds content, not timing.** The fingerprint proves a frozen
   manifest was not edited afterwards; it does not prove it was written before
   the numbers were seen. Nothing here is an external timestamp. Committing the
   frozen manifest before collecting is an operator discipline, not an enforced
   one.
5. **Pool bias is measured, not corrected.** `judged@k` tells you how much of a
   run the labels cover. It does not fix the penalty an out-of-pool system pays.
6. **Derived qrels are inference, not ground truth.** TAT-DQA and OTT-QA labels
   are derived from gold-evidence annotations by this harness. They are marked
   everywhere they appear, and they are weaker evidence than native qrels.
7. **SummEval calibration is a proxy.** It says the judge grades faithfulness
   sensibly in general on summarization, not that it is calibrated on your
   corpus.
8. **No cross-system comparison.** See below.

---

## 7. Comparing lilbee against other systems

This harness deliberately does **not** do it. The RAGFlow arm was removed rather
than left in a state where it could produce a number, because a cross-system
comparison is a substantially harder measurement problem than an ablation, and a
broken one is worse than none.

What makes it hard, concretely, and what any future cross-system harness has to
solve first:

- **A document-id join.** Both systems must report results in the same namespace
  as the qrels. A system that returns its own internal UUIDs matches nothing in
  BEIR's qrels and scores a clean zero, which reads as "terrible system" rather
  than "namespace bug." `judged@k` at zero is the tripwire for this, and any such
  harness should fail closed on it.
- **Depth matching across different chunking.** Asking a chunk-ranked system for
  20 chunks and a document-grouped system for 20 documents compares different
  things. Collapse to parent documents and cap both at the same document depth.
- **Equal tuning, or an honest label.** A tuned system against another at stock
  defaults is a legitimate thing to measure, but the report must say so in the
  artifact people screenshot, not only in a README.
- **The same corpus, provably.** Both systems must index the same bytes, with
  ids that survive both ingestion paths on any operator's OS.
- **Configuration pinned on the route actually scored.** Pinning retrieval knobs
  on a chat endpoint while scoring a different retrieval endpoint attests to a
  configuration the scored arm never saw.

Until those are solved, the honest comparison against other tools is: publish
lilbee's Tier-1 numbers on public BEIR sets with the run files and qrels
attached, state the chunking and the coverage, and let anyone re-score them with
their own trec_eval against whatever else they like. That is why the artifacts
are standard formats.

---

## 8. Reproducing a run

```bash
uv run python -m evals.benchmark preregister --manifest evals/benchmark/manifest.example.yaml --out $RUN/manifest.frozen.json
uv run python -m evals.benchmark fetch       --manifest evals/benchmark/manifest.example.yaml --out $RUN/datasets
uv run python -m evals.benchmark collect     --queries $RUN/queries.jsonl --base-url http://127.0.0.1:8080 --run-tag arm-a --run $RUN/a.trec --checkpoint $RUN/ck-a.jsonl
uv run python -m evals.benchmark score-ir    --qrels $RUN/datasets/scifact/qrels.trec --run $RUN/a.trec --dataset scifact --run-tag arm-a --out $RUN/ir-a.jsonl
uv run python -m evals.benchmark stats       --manifest $RUN/manifest.frozen.json --metrics-a $RUN/ir-a.jsonl --metrics-b $RUN/ir-b.jsonl --out $RUN/results.jsonl
uv run python -m evals.benchmark report      --results $RUN/results.jsonl --out $RUN/report.md
```

Keep the frozen manifest, the run files and the qrels. All three are standard
formats, so a third party can re-score the entire Tier-1 comparison without this
harness and without the machine that produced it. That is the property that makes
the numbers checkable rather than merely published.
