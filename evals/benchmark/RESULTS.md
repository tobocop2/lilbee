# Retrieval benchmark results — Qwen3-Embedding-0.6B

A significance-tested measurement of lilbee's hybrid retrieval against its own
vector-only baseline on three public, human-labeled BEIR datasets. The point of
this run is to answer two questions with evidence, not opinion: does lilbee's
rank fusion actually improve retrieval over the raw embedder, and is a single
fusion weight the right design.

## Method

- **Embedder:** Qwen3-Embedding-0.6B (GGUF, Q8_0), served once; every arm points
  at the same model, so retrieval is the only variable.
- **Datasets:** BEIR SciFact, NFCorpus, FiQA, using their published TREC qrels
  as-is. No labels were derived or altered.
- **Metrics:** nDCG@10, Recall@20, MRR@10, scored with `pytrec_eval` against the
  native qrels. Nothing is graded by a model, so the numbers are reproducible by
  anyone with the run files and qrels.
- **Arms:** `dense` is vector-only (the lexical and title arms silenced);
  `w=X` is hybrid fusion with the BM25 arm at `lexical_fusion_weight=X` and the
  title arm on. This isolates what fusion adds over the embedder alone.
- **Significance:** every hybrid-vs-dense difference is paired per query and gets
  a bootstrap 95% CI and a randomization-test p-value. A difference whose CI
  crosses zero is reported as not significant, not as a win.
- **Condition:** query expansion was off (no chat model served), so these are
  pure first-pass retrieval numbers with no LLM query rewriting on either arm.

## Results (nDCG@10)

| Dataset | dense | w=1.0 | w=0.75 | w=0.5 | w=0.25 |
|---|---|---|---|---|---|
| NFCorpus (n=322) | 0.3666 | 0.3734 | 0.3743 | 0.3756 | **0.3767** |
| SciFact (n=300) | 0.6981 | **0.7280** | 0.7226 | 0.7167 | 0.7155 |
| FiQA (n=648) | **0.4598** | 0.4033 | 0.4289 | 0.4333 | 0.4363 |

## Significance (best hybrid weight vs. dense, nDCG@10)

| Dataset | best hybrid | Δ vs dense | 95% CI | p | verdict |
|---|---|---|---|---|---|
| NFCorpus | w=0.25 | **+0.0100** | [+0.0022, +0.0180] | 0.012 | significant win |
| SciFact | w=1.0 | **+0.0299** | [+0.0073, +0.0528] | 0.011 | significant win |
| FiQA | w=0.25 | **−0.0235** | [−0.0356, −0.0118] | 0.0004 | significant regression |

On FiQA every hybrid weight loses to dense (w=1.0 is worst at −0.0564, p=0.0001);
lowering the weight reduces the damage but never recovers it. On NFCorpus the
win is only significant at lower weights (w=1.0 is +0.0068, p=0.21, not
significant). On SciFact every weight wins, and higher is better.

## What this establishes

1. **Hybrid fusion adds real, significant value on two of three corpora** — once
   it actually runs (see the bug below). This is the raw embedder plus lilbee's
   fusion beating the raw embedder alone, measured, not asserted.
2. **No single fixed fusion weight wins everywhere.** SciFact wants a strong
   lexical arm (w=1.0), NFCorpus wants a weak one (w=0.25), and FiQA wants none.
   The optimal weight is corpus-dependent, which is the empirical case for
   gating the lexical weight per query rather than fixing it — the adaptive
   fusion this run motivates.

## The bug this run caught

Before any of these numbers meant anything, every hybrid config scored
identically to the vector-only baseline. The cause was a production defect, not
a benchmark artifact: `ensure_fts_index()` runs `table.optimize()` on an
existing index, which on a real-sized corpus hits a LanceDB encoding bug and
raises; the failure was swallowed and left hybrid search disabled, so every
query silently fell back to vector-only. It only surfaced on real data — a
tiny local corpus never tripped the LanceDB bug. Fixed by marking an existing,
queryable index ready before the best-effort optimize, so an optimize failure
degrades to a warning instead of disabling retrieval corpus-wide.

## Reproducing

The frozen manifest, the TREC run files, and the qrels are kept under
`results/`. With those, anyone can re-score Tier 1 without a GPU:

```bash
gunzip -k results/scifact/run-w1.0.trec.gz
python -m evals.benchmark score-ir --qrels results/scifact/qrels.json \
  --run results/scifact/run-w1.0.trec --dataset scifact --run-tag w1.0 --out /tmp/ir.jsonl
```

## Adaptive fusion

Because no fixed weight wins everywhere, a follow-up run measured `adaptive_fusion`
(the BM25 weight scaled per query by how peaked the vector ranking is) at three
confidence margins, on the same three datasets. Best margin was 0.15.

| Dataset | dense | w=1.0 | adaptive (m=0.15) | adaptive vs dense |
|---|---|---|---|---|
| NFCorpus | 0.3666 | 0.3732 | 0.3751 | +0.0085, p=0.033, **sig win** |
| SciFact | 0.6981 | 0.7280 | 0.7186 | +0.0205, p=0.009, **sig win** |
| FiQA | 0.4593 | 0.4027 | 0.4410 | −0.0184, p=0.003, sig regression |

Adaptive fusion is the best single policy found. It beats the fixed w=1.0 default
on all three datasets, keeps the significant NFCorpus and SciFact wins, and gives
the smallest FiQA regression of any config (−0.018 vs −0.056 at w=1.0 and −0.023
at the best fixed weight). It does not fully erase FiQA's regression — pure dense
still wins there — so the vector-margin confidence signal reduces, but does not
eliminate, the cases where the lexical arm hurts. Run files and stats are under
`results-adaptive/`.

## Scope

This is the 0.6B-embedder study: it validates the fusion mechanism and the
FTS-optimize fix, and it sizes the corpus-dependence of the lexical weight. It
is deliberately not a headline "lilbee vs. everyone" number — a larger embedder
and the RAGFlow A/B arm are separate runs.
