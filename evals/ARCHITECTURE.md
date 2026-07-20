# The eval harness: what it is, how it works, why you can trust a number from it

Two harnesses share this directory. Both exist to answer "did that change make
retrieval better?" without the answer resting on anyone's judgement, including
the judgement of the model doing the retrieving.

- `evals/benchmark` scores retrieval against **human relevance labels** on public
  datasets. No model grades anything, so its numbers are reproducible from the
  committed artifacts alone.
- `evals/retrieval` runs a **blind A/B** of two servers over a private corpus,
  where no public labels exist and a judge model has to stand in for one.

Neither ships in the package. They live outside `src/` on purpose.

## Why the design looks like this

An eval harness is a measuring instrument, and the failure mode that matters is
not "the number is noisy", it is **"the number is wrong and looks fine"**. Every
structural decision below exists to make a specific silent failure loud instead.

The design leans on IR evaluation methodology rather than inventing its own.
Where a standard tool or statistic exists, the harness uses it and stays out of
the way. The parts that are ours are the plumbing that keeps the standard parts
honest: preregistration, blinding, depth matching, and refusing to emit a number
when the thing that produced it cannot be shown to be the thing claimed.

## Tier 1: retrieval against human labels

```
datasets.py    BEIR corpus/queries/qrels, normalized to one triple
   |
collectors.py  one arm's ranked results per query  -> chunk hits
   |
runfile.py     collapse chunks to parent documents -> TREC run file
   |
ir_metrics.py  pytrec_eval against the qrels       -> per-query + aggregate
   |
stats.py       paired bootstrap CI + randomization test, BH-adjusted
   |
report.py      markdown
```

**Datasets.** BEIR (Thakur et al., *BEIR: A Heterogenous Benchmark for
Zero-shot Evaluation of Information Retrieval Models*, NeurIPS 2021 Datasets and
Benchmarks). SciFact, FiQA and NFCorpus are used with their published TREC
qrels. Non-positive judgments are dropped, which is how `trec_eval` treats them.

**Run format.** The six-column TREC run format (`query_id Q0 doc_id rank score
run_tag`), so any TREC-compatible scorer can read the artifacts.

**Scoring.** `pytrec_eval` (Van Gysel and de Rijke, SIGIR 2018), a Python binding
over NIST's `trec_eval`. Using the reference implementation rather than
hand-rolled metrics is deliberate: nDCG has several published variants and a
local implementation is a place for a silent disagreement to live.

Two things the harness must get right *around* the scorer:

- **Depth.** `recip_rank` is uncut. It searches the whole run, so a first
  relevant document at rank 11 contributes 1/11 to something labelled MRR@10.
  Runs are collected deeper than 10, so metrics that do not cut internally
  declare an explicit depth and the run is truncated before scoring.
- **Tie order.** `pytrec_eval` ignores the run file's rank column and re-sorts by
  score, breaking ties on doc_id **descending** (`trec_eval`'s rule). Truncation
  and run-file ordering both use that rule, so the ten documents handed to the
  scorer are the ten it would have chosen. Rank fusion produces many exactly
  tied scores, so this is not hypothetical: 210 of FiQA's 648 queries have a tie
  straddling the rank-10 boundary at full lexical weight.
- **Denominator.** `pytrec_eval` returns only query ids present in the run, so a
  query an arm returned nothing for would vanish from the mean rather than
  scoring zero. Aggregation is over the qrels topic set instead, which is what
  stops an arm being rewarded for failing on hard queries.

**Depth matching.** Metrics are scored over documents. A document-level retriever
asked for 20 returns 20 documents; a chunk-level one asked for 20 chunks returns
however many parent documents those chunks belong to, usually far fewer.
Comparing those two lists puts a pure depth artifact in the gap, so chunk-level
collection pages until it holds the target number of distinct parents and every
arm's run is capped at the same document depth.

## Statistics

Every cross-arm difference is **paired per query**, because both arms answer the
same queries. Paired comparison is the standard in IR evaluation and it is much
more powerful than treating the two runs as independent samples.

| what | method | source |
|---|---|---|
| interval | percentile bootstrap over per-query differences, 10,000 resamples | Efron's bootstrap; percentile interval |
| test | two-sided sign-flip randomization test, 10,000 resamples | Fisher randomization; recommended for IR by Smucker, Allan and Carterette, *A Comparison of Statistical Significance Tests for IR Evaluation*, CIKM 2007 |
| multiplicity | Benjamini-Hochberg step-up FDR control | Benjamini and Hochberg, *Controlling the False Discovery Rate*, JRSS-B 1995 |

Three deliberate choices:

**The randomization test rather than a t-test.** Smucker et al. compared the
common options on IR data and found the randomization test the most appropriate;
the t-test's normality assumption is a poor fit for per-query metric differences,
which are bounded, discrete-ish and heavily zero-inflated.

**Multiplicity control is not optional.** A weight ablation over four arms, three
datasets and three metrics is 36 tests. Reporting the best of them at its raw
p-value is how an ablation manufactures a finding. Significance is decided on the
BH-adjusted p across the whole family; the confidence interval is reported as the
effect size, not as a second verdict that can disagree with the first.

**Both procedures are seeded, with distinct sub-seeds.** Driving the bootstrap
and the randomization test off one stream makes them the same draws twice rather
than corroborating evidence. Everything is reproducible bit-for-bit.

The BH implementation is ~10 lines and hand-rolled, because this module is
deliberately stdlib-only so scoring runs without the pod-only extras. It is
differential-tested against `scipy.stats.false_discovery_control(method="bh")`
over 2000 randomized families including exact ties; maximum absolute difference
2.2e-16.

## Preregistration

Borrowed straight from clinical trials, and from the replication-crisis response
in psychology: **declare the study before the data moves, then let the tooling
refuse anything else.**

`manifest.py` freezes the arms, datasets, metrics, held-constant models, seed,
resamples and alpha into a file plus a sha256 over its canonical JSON. The
`stats` step then refuses to stamp that fingerprint on a comparison whose arms or
dataset the manifest does not declare, and `Manifest.load` verifies a stored
fingerprint against the recomputed hash, so an edited frozen manifest is refused
rather than silently re-hashing to something self-consistent.

A fingerprint that attests to a study nobody ran is worse than no preregistration
at all, because it lends credibility rather than withholding it.

## Tier 2 and the blind judge

Where no human labels exist, a model has to grade, and the harness treats that as
a measurement problem rather than an oracle.

**RAGAS** scores faithfulness, answer relevancy and context precision/recall for
both arms, with the judge the manifest pins. Each mean carries the count of
answers that actually scored, and a metric covering under 90% of samples refuses
to publish, because RAGAS emits NaN when it cannot score and averaging past that
rewards whichever arm's answers fail more often. Tier 2 is **not**
significance-tested and the report says so.

**Blind judging** (`evals/retrieval`) is the more interesting piece:

- Every gradable (question, answer) pair gets an opaque id that is a sha256 of
  its content, and all rows shuffle together. The judge sees question, ground
  truth and one answer. Never an arm label, never a hint that a comparison is
  happening.
- The id is content-derived, not a positional draw, so a resumed run cannot
  attach a grade to a different answer than the one it was written for.
- **The judge must be a different model from the generator.** There is no
  fallback. Self-preference bias in LLM judges is well documented (Zheng et al.,
  *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, 2023), and letting
  the model that wrote the answers grade them is the single easiest way to
  manufacture a result.

**The noise floor** is the part most worth understanding. One arm is graded
twice, and the disagreement between those passes bounds how much of a reported
gap is judge instability. The trap: both chat backends decode greedily at
temperature 0, so re-sending an identical prompt returns an identical grade and
the measured "noise" is exactly zero by construction, which then marks every
difference as signal. The two replicates are therefore graded under two
**equivalent** presentations of the same rubric, differing only in the order the
material is laid out and the JSON fields are requested. That measures the judge's
sensitivity to presentation, which is real instability, and stays reproducible.

The floor is a per-question statement about judge steadiness. It is **not** a
threshold for a difference of means over many questions, which is a different
scale. Significance in that report comes from the same paired test as Tier 1,
BH-adjusted across the dimensions tested.

## Sampling

Question batteries are drawn from a streaming scan, so nothing materializes the
index and it scales to large private corpora. Passages use **reservoir sampling,
Algorithm R** (Vitter, *Random Sampling with a Reservoir*, ACM TOMS 1985), with
oversampling and a per-source dedupe.

One subtlety: Algorithm R leaves slots never hit by a replacement holding the
first candidates in stream order, and only about a quarter of the reservoir
survives the dedupe. Consuming it in slot order would bias the battery toward
whatever the table returned first, so it is shuffled before consumption. The
sample is uniform only once the order is discarded.

## Why you can trust a number from this

Not because it is careful, but because of what it refuses to do:

| refusal | the silent failure it replaces |
|---|---|
| Metrics aggregate over the qrels topic set | an arm rewarded for returning nothing on hard queries |
| Truncation uses the scorer's own tie rule | "MRR@10" that is not the scorer's MRR at depth 10 |
| Chunk arms page to a document target | a depth artifact read as a ranking difference |
| `stats` refuses undeclared arms or datasets | a fingerprint attesting to a study nobody ran |
| `Manifest.load` verifies the stored hash | a frozen preregistration edited after the numbers |
| Checkpoints carry an arm/config fingerprint | arm B's run file built from arm A's hits |
| A missing metric raises | a fabricated "measured tie" at p=1.0 |
| A missing noise replicate raises | a 0.0 floor marking every delta significant |
| No judge fallback to the generator | a model grading its own homework |
| Out-of-range judge output is rejected | a rubric violation coerced into a plausible score |
| Ungraded arm raises rather than scoring 0 | "never measured" indistinguishable from "worst possible" |
| Significance decided on adjusted p | best-of-N reported as a single test |

Every one of those was a real defect in this harness, found in a
measurement-validity audit of 63 findings and fixed. The audit is the reason to
trust the instrument more than the fact that it was written carefully; several of
the defects survived multiple careful readings and were only caught by running
the code against the committed data and against the reference implementations.

**What it still does not do.** It cannot tell you a corpus was indexed
identically by two systems, only that both were asked for the same depth. It has
no cross-encoder or human adjudication for the answer tier. And `Recall@20` on
run files collected before depth matching is recall at whatever depth the run
reached, comparable within a study and not across.

## Reading the code

| module | responsibility |
|---|---|
| `benchmark/datasets` | BEIR loading, native vs derived qrels |
| `benchmark/collectors` | per-system retrieval, depth-matched collection |
| `benchmark/runfile` | TREC format, chunk-to-document collapse |
| `benchmark/ir_metrics` | depth truncation, denominators, pytrec_eval seam |
| `benchmark/stats` | bootstrap, randomization test, Benjamini-Hochberg |
| `benchmark/manifest` | preregistration, fingerprint, comparison guard |
| `benchmark/ragas_tier` | RAGAS with coverage floor, corroborating judge |
| `retrieval/store_scan` | streaming reservoir sampling, exact term counts |
| `retrieval/questions` | battery generation, shortfall reporting |
| `retrieval/blinding` | content-derived ids, replicate construction |
| `retrieval/judging` | rubric, prompt variants, grade parsing |
| `retrieval/scoring` | paired means, noise floor, exact-truth checks |

The pure logic is separated from the I/O in every module so it can be tested
without a model, a server or the C extension. `ir_metrics` takes an evaluator
factory, `ragas_tier` takes an evaluate function, the collectors take an HTTP
client. That seam is why the suite runs in about a second.
