# Retrieval benchmark results: Qwen3-Embedding-0.6B

A significance-tested measurement of lilbee's hybrid retrieval against its own
vector-only baseline on three public, human-labeled BEIR datasets. The point of
this run is to answer two questions with evidence, not opinion: does lilbee's
rank fusion actually improve retrieval over the raw embedder, and is a single
fusion weight the right design.

Every number below was recomputed from the committed TREC run files after the
measurement bugs described under [Corrections](#corrections) were fixed. The
previous version of this document should not be cited: its MRR@10 was uncut, its
NFCorpus denominator dropped a query, and it reported a post-hoc best-of-four
arm's raw p-value as a significance verdict.

## Method

- **Embedder:** Qwen3-Embedding-0.6B (GGUF, Q8_0), served once; every arm points
  at the same model, so retrieval is the only variable.
- **Datasets:** BEIR SciFact, NFCorpus, FiQA, using their published TREC qrels.
  No labels were derived. Non-positive judgments are dropped, which is how
  `pytrec_eval` treats them.
- **Metrics:** nDCG@10, Recall@20, MRR@10, scored with `pytrec_eval`. MRR@10
  truncates each run to depth 10 before scoring. Every metric is averaged over
  the qrels topic set, so a query an arm returned nothing for scores zero rather
  than leaving the denominator.
- **Arms:** `dense` is vector-only (the lexical and title arms silenced);
  `w=X` is hybrid fusion with the BM25 arm at `lexical_fusion_weight=X` and the
  title arm on. This isolates what fusion adds over the embedder alone.
- **Significance:** every hybrid-vs-dense difference is paired per query and gets
  a bootstrap 95% CI and a randomization-test p-value. This study runs 36 tests
  (4 weights × 3 datasets × 3 metrics), so p-values are Benjamini-Hochberg
  adjusted across all 36 and significance is decided on the adjusted value. The
  CI is reported as the effect size, not as a second verdict.
- **Condition:** query expansion was off (no chat model served), so these are
  pure first-pass retrieval numbers with no LLM query rewriting on either arm.

## Results

nDCG@10 / Recall@20 / MRR@10 per arm. Bold marks the best arm per dataset on
nDCG@10.

| Dataset | metric | dense | w=0.25 | w=0.5 | w=0.75 | w=1.0 |
|---|---|---|---|---|---|---|
| SciFact (n=300) | nDCG@10 | 0.6981 | 0.7155 | 0.7167 | 0.7226 | **0.7280** |
| | Recall@20 | 0.8633 | 0.8633 | 0.8633 | 0.8633 | 0.8893 |
| | MRR@10 | 0.6625 | 0.6828 | 0.6836 | 0.6920 | 0.6928 |
| FiQA (n=648) | nDCG@10 | **0.4598** | 0.4363 | 0.4333 | 0.4289 | 0.4033 |
| | Recall@20 | 0.6186 | 0.6186 | 0.6186 | 0.6186 | 0.5802 |
| | MRR@10 | 0.5406 | 0.5045 | 0.5003 | 0.4937 | 0.4726 |
| NFCorpus (n=323) | nDCG@10 | 0.3655 | **0.3755** | 0.3744 | 0.3731 | 0.3722 |
| | Recall@20 | 0.2129 | 0.2129 | 0.2129 | 0.2129 | 0.2153 |
| | MRR@10 | 0.5863 | 0.5961 | 0.5856 | 0.5832 | 0.5838 |

Recall@20 is identical between dense and w=0.25/0.5/0.75 on all three datasets.
That is expected rather than suspicious: recall is set-based, and at those
weights fusion reorders the candidate set without changing which documents are
in it. Only w=1.0 changes the set, and it changes it for the worse on FiQA.

## Significance, adjusted across all 36 comparisons

Thirteen of the 36 tests survive correction. The full family is in
`results/*/stats-*.jsonl`; the survivors:

| Dataset | arm | metric | Δ vs dense | raw p | adj. p |
|---|---|---|---|---|---|
| FiQA | w=1.0 | MRR@10 | −0.0680 | ≤1e-4 | 0.0006 |
| FiQA | w=1.0 | nDCG@10 | −0.0564 | ≤1e-4 | 0.0006 |
| FiQA | w=0.75 | MRR@10 | −0.0469 | ≤1e-4 | 0.0006 |
| FiQA | w=0.5 | MRR@10 | −0.0403 | 0.0004 | 0.0016 |
| FiQA | w=1.0 | Recall@20 | −0.0384 | ≤1e-4 | 0.0006 |
| FiQA | w=0.25 | MRR@10 | −0.0361 | 0.0003 | 0.0013 |
| FiQA | w=0.75 | nDCG@10 | −0.0309 | ≤1e-4 | 0.0006 |
| FiQA | w=0.5 | nDCG@10 | −0.0264 | 0.0002 | 0.0010 |
| FiQA | w=0.25 | nDCG@10 | −0.0235 | ≤1e-4 | 0.0006 |
| SciFact | w=1.0 | nDCG@10 | +0.0299 | 0.0127 | 0.0381 |
| SciFact | w=0.75 | MRR@10 | +0.0295 | 0.0159 | 0.0440 |
| SciFact | w=0.75 | nDCG@10 | +0.0245 | 0.0122 | 0.0381 |
| NFCorpus | w=0.25 | nDCG@10 | +0.0100 | 0.0113 | 0.0381 |

The strongest and most consistent finding in this study is a regression, not a
win: on FiQA every fusion weight loses to dense on both nDCG@10 and MRR@10, all
nine of those tests survive correction, and the damage grows monotonically with
the weight. SciFact improves, significantly at the two higher weights. NFCorpus
improves marginally, significant only at w=0.25 on nDCG@10 and nowhere else:
its MRR@10 gain at the same weight (+0.0098) does not survive.

## What this establishes

1. **Fusion helps on SciFact, hurts on FiQA, and is marginal on NFCorpus.** The
   FiQA regression is the best-evidenced effect here by an order of magnitude in
   adjusted p. Any claim that hybrid fusion is a general improvement is not
   supported by this run.
2. **No single fixed fusion weight wins everywhere.** SciFact wants a strong
   lexical arm, NFCorpus wants a weak one and only barely, and FiQA wants none.
   That corpus-dependence is the empirical case for gating the lexical weight
   per query rather than fixing it.

Both conclusions survived the corrections. The previously headlined framing did
not: "significant win on two of three corpora" rested on selecting each
dataset's best weight after seeing the results and quoting its uncorrected
p-value.

## Adaptive fusion

A follow-up run measured `adaptive_fusion` (the BM25 weight scaled per query by
how peaked the vector ranking is) at three confidence margins. Only the m=0.15
arm's run file was kept, so only that arm can be rescored; the m=0.30 and m=0.50
numbers in `results-adaptive/` were produced by the uncorrected metric layer and
are not reproducible from committed artifacts. Because m=0.15 was itself
selected as the best of three margins and the other two runs are gone, that
selection cannot be corrected for from what is committed. Treat the adaptive
comparison as indicative and re-run it before citing it.

Rescored the same way, as its own family of 9 tests:

| Dataset | metric | dense | adaptive (m=0.15) | Δ | raw p | adj. p | verdict |
|---|---|---|---|---|---|---|---|
| SciFact | nDCG@10 | 0.6981 | 0.7186 | +0.0205 | 0.0109 | 0.0327 | significant |
| SciFact | MRR@10 | 0.6625 | 0.6855 | +0.0229 | 0.0214 | 0.0481 | significant |
| SciFact | Recall@20 | 0.8633 | 0.8667 | +0.0033 | 1.0000 | 1.0000 | n.s. |
| FiQA | nDCG@10 | 0.4598 | 0.4410 | −0.0188 | 0.0014 | 0.0067 | significant |
| FiQA | MRR@10 | 0.5406 | 0.5095 | −0.0311 | 0.0015 | 0.0067 | significant |
| FiQA | Recall@20 | 0.6186 | 0.6192 | +0.0006 | 1.0000 | 1.0000 | n.s. |
| NFCorpus | nDCG@10 | 0.3655 | 0.3739 | +0.0084 | 0.0317 | 0.0571 | n.s. |
| NFCorpus | MRR@10 | 0.5863 | 0.5978 | +0.0114 | 0.3807 | 0.4894 | n.s. |
| NFCorpus | Recall@20 | 0.2129 | 0.2134 | +0.0006 | 0.2504 | 0.3756 | n.s. |

Adaptive fusion keeps the SciFact win and gives the smallest FiQA regression of
any config measured (−0.0188 on nDCG@10, against −0.0235 at the best fixed
weight and −0.0564 at w=1.0), but it does not erase that regression: FiQA still
prefers pure dense, significantly. The NFCorpus gain does not survive correction
(adjusted p 0.0571), so the previous claim of a significant NFCorpus win for
adaptive fusion is withdrawn. The honest summary is that adaptive fusion is the
best policy found on two of three datasets and still loses on FiQA.

## Scope and what has NOT been re-run

This is the 0.6B-embedder study on lilbee's own arms. It is deliberately not a
"lilbee vs. everyone" number.

Two things are corrected here and two are not:

- **Corrected:** the Tier-1 metric layer and all paired statistics, recomputed
  from the committed run files.
- **Not re-collected:** the run files themselves predate the depth-matching fix.
  Collection now pages a chunk-level arm until it holds the target number of
  distinct parent documents and caps every arm at the same document depth. These
  runs were collected at the old depth, and most queries carry fewer than 20
  documents, so `Recall@20` here is really recall at the observed depth and is
  understated. It is comparable across these arms, which were all collected the
  same way, and is not comparable to any future run.
- **Not run at all:** the RAGFlow parity arm. No cross-system comparison in this
  repository has been performed with the corrected harness.
- **Not run at all:** Tier 2 (RAGAS answer quality). The previous harness scored
  one arm and wrote its value into both arms' columns, so no Tier-2 comparison
  has ever been measured.

## Corrections

The measurement bugs fixed before these numbers were regenerated, and what each
one moved:

- **MRR@10 was uncut reciprocal rank.** A first relevant document at ranks 11-20
  contributed 1/rank instead of 0. Every MRR@10 figure above is lower than the
  previously published one (SciFact dense 0.6625, was 0.6648).
- **Queries an arm returned nothing for were dropped from the denominator**
  rather than scored zero. NFCorpus was reported at n=322 against a 323-topic
  qrels; it is now n=323 and every NFCorpus figure moves slightly.
- **No multiple-comparison control.** 36 tests were run and the best arm per
  dataset was reported at its raw p. Significance is now BH-adjusted.
- **The CI and the p-value contradicted each other.** `significant` came from
  the bootstrap CI while the printed p came from an unrelated randomization
  test, and both were driven off one random stream. They now use distinct
  sub-seeds, and the adjusted p is the single decision rule.
- **The preregistration did not constrain the run.** The frozen manifest
  declares a lilbee-vs-RAGFlow study over seven datasets; these results stamped
  its fingerprint on a weight ablation over three. The stats step now refuses to
  stamp a comparison the manifest does not declare.
- **The committed w=1.0 metrics did not reproduce from the committed w=1.0 run
  files.** Fourteen of fifteen committed arms reproduce exactly as uncut
  `recip_rank`; all three w=1.0 arms match neither the cut nor the uncut
  recomputation, and 10-20% of their per-query values disagree with their own
  run file. Those stale artifacts have been discarded and every number here is
  computed from the run files.

## Reproducing

The frozen manifest, the TREC run files, and the qrels are kept under
`results/`. With those, anyone can re-score Tier 1 without a GPU:

```bash
gunzip -k results/scifact/run-w1.0.trec.gz
python -m evals.benchmark score-ir --qrels results/scifact/qrels.json \
  --run results/scifact/run-w1.0.trec --dataset scifact --run-tag w1.0 --out /tmp/ir.jsonl
```

## The bug this run caught

Before any of these numbers meant anything, every hybrid config scored
identically to the vector-only baseline. The cause was a production defect, not
a benchmark artifact: `ensure_fts_index()` runs `table.optimize()` on an
existing index, which on a real-sized corpus hits a LanceDB encoding bug and
raises; the failure was swallowed and left hybrid search disabled, so every
query silently fell back to vector-only. It only surfaced on real data, since
a tiny local corpus never tripped the LanceDB bug. Fixed by marking an existing,
queryable index ready before the best-effort optimize, so an optimize failure
degrades to a warning instead of disabling retrieval corpus-wide.
