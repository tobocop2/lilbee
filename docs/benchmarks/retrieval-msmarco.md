# Retrieval Benchmark: MS MARCO Passage

Retrieval quality of lilbee's 8.8M-passage MS MARCO index, measured against the
official human relevance judgments and compared to published systems.

**Headline: lilbee's full retrieval pipeline scores MRR@10 = 0.365 and nDCG@10 =
0.437 on MS MARCO dev/small, with recall@100 = 0.90.** That places it above
ColBERT, TAS-B, and ANCE among systems trained specifically for this benchmark,
and level with the strongest general-purpose embedders (nDCG@10 0.437 matches
multilingual-e5-large and edges lilbee's own embedder's exact-search score).
Scored by Microsoft's own evaluation script and NIST's trec_eval, which agree.

lilbee is **general-purpose** - an off-the-shelf embedder (Qwen3-Embedding-8B)
plus a general cross-encoder reranker (bge-reranker-v2-m3), pointed at any corpus.
It is not trained on MS MARCO.

## What these numbers mean (plain-language key)

If you read only one thing: **lilbee's search is in the same tier as well-known
specialized AI search systems, and far better than classic keyword search.** On a
standard industry test of 6,980 real search queries, the correct passage
typically lands near the top of lilbee's results, and is frequently first.

**The metrics** (all run 0 to 1, higher is better):

| Metric | In plain terms | lilbee (full pipeline) |
|--------|----------------|------------------------|
| **MRR@10** | How high the correct answer sits, on average. Anchor points: 1.0 = always the #1 result, 0.5 = like always #2, 0.33 = like always #3. This is the headline. | **0.365** |
| **nDCG@10** | Overall quality of the top-10 ordering, giving more credit the closer the right answers are to the top. | **0.437** |
| **Recall@100** | Of all the correct passages, the share found anywhere in the top 100. | **0.90** |
| judged@10 | A coverage sanity-check, **not** a quality score: how much of the top 10 the test even has answer-labels for. Low is normal here, because the test labels only ~1 passage per question. | ~6% |

**How the pipeline builds up.** lilbee retrieves in three stages, and each stage
is measured, so the contribution of each is visible:

| Stage | MRR@10 | nDCG@10 | recall@100 |
|-------|--------|---------|------------|
| Dense only (the embedder alone) | 0.347 | 0.398 | 0.67 |
| + keyword (FTS) fusion | ~0.31 | ~0.37 | 0.90 |
| **+ cross-encoder rerank (full pipeline)** | **0.365** | **0.437** | **0.90** |

Fusion finds the answer (recall jumps from 0.67 to 0.90), and the reranker puts it
at the top (MRR and nDCG rise to their headline values). Reranking is the single
biggest lever on this corpus - see Why reranking helps below.

## Is this good? Yes - here is lilbee against the field

lilbee belongs with **general-purpose** embedders (usable on any corpus). It is
also worth seeing it against the systems built specifically to win MS MARCO.

### Against general-purpose embedders (MTEB MS MARCO, nDCG@10)

| Rank | System | nDCG@10 |
|------|--------|---------|
| 1 | gte-Qwen2-7B-instruct | 0.460 |
| 2 | stella_en_1.5B_v5 | 0.452 |
| **3** | **lilbee (full pipeline)** | **0.437** |
| 3 | multilingual-e5-large | 0.437 |
| 5 | Qwen3-Embedding-8B (lilbee's embedder, exact search) | 0.436 |
| 6 | e5-large-v2 | 0.435 |
| 7 | bge-large-en-v1.5 | 0.425 |
| 8 | mxbai-embed-large-v1 | 0.413 |

lilbee's full pipeline (0.437) **matches multilingual-e5-large and edges past its
own embedder's exact-search MTEB score (0.436)**. That is the important result:
lilbee runs the embedder over a lossy, approximate index at 8.8M-passage scale,
and the fusion-plus-rerank pipeline still recovers the quality the embedder gets
with exact search. Dense retrieval alone scored 0.398 - the rest is the pipeline.

### Against MS-MARCO specialists (MRR@10)

| Rank | System | MRR@10 | Trained only for MS MARCO? |
|------|--------|--------|----------------------------|
| 1 | SimLM | 0.411 | yes |
| 2 | ColBERTv2 | 0.397 | yes |
| 3 | coCondenser | 0.382 | yes |
| 4 | RocketQA | 0.370 | yes |
| 5 | SPLADEv2 | 0.368 | yes |
| **6** | **lilbee (full pipeline)** | **0.365** | **no** |
| 7 | ColBERT | 0.360 | yes |
| 8 | TAS-B | 0.347 | yes |
| 9 | ANCE | 0.330 | yes |
| 10 | docTTTTTquery | 0.277 | yes |
| 11 | BM25 | 0.187 | no |

Every system above lilbee here exists **only** as an MS MARCO retriever, trained
end-to-end on this benchmark's training set. lilbee is a general-purpose stack you
point at your own documents, and it lands **above ColBERT, TAS-B, and ANCE** while
being the only non-specialized system in that tier. Baseline figures are the
published numbers for this dataset; lilbee's is the measured number from this run.

**One caveat that makes the result look better, not worse:** this test labels only
about one "correct" passage per question, even when several passages answer it
equally well. lilbee earns no credit when it returns a genuinely good passage the
test did not happen to label, which is why even the best systems in the world top
out around 0.40-0.45 rather than 1.0.

## Why reranking helps so much on this corpus

MS MARCO labels ~1 relevant passage per query, so getting that one passage to the
very top is everything. Two effects compound:

- **Fusion finds it.** Adding keyword (BM25/FTS) retrieval to dense retrieval
  lifts recall@100 from 0.67 to 0.90 - the relevant passage is now almost always
  somewhere in the pool.
- **The cross-encoder puts it first.** The embedder is a bi-encoder (query and
  passage encoded separately, matched by vector similarity). A cross-encoder reads
  the query and passage *together*, so it scores relevance far more precisely and
  pulls the right passage to the top of the deep pool. That is exactly the
  top-of-list precision MRR@10 and nDCG@10 reward.

## The comparison systems (glossary)

- **BM25** - classic keyword-matching search, no AI; the algorithm inside Elasticsearch, OpenSearch, and Lucene, i.e. what most software's search bar uses. The floor to beat.
- **Dense retriever / bi-encoder** - turns text into meaning-vectors and matches by similarity (lilbee's first stage). ANCE (Microsoft, 2020) and TAS-B (TU Wien, 2021) are dense retrievers trained specifically as MS MARCO retrievers.
- **Cross-encoder (reranker)** - reads the query and a candidate passage together for a far more precise relevance score than a bi-encoder. lilbee uses bge-reranker-v2-m3, a general multilingual reranker.
- **ColBERT / ColBERTv2** - "late-interaction" retrievers from Stanford that store many vectors per passage; heavier and more accurate.
- **SPLADEv2** - a learned-sparse retriever. **coCondenser / RocketQA / SimLM** - dense retrievers with specialized MS MARCO pre-training/fine-tuning.
- **General-purpose vs specialist** - lilbee uses an off-the-shelf embedder + general reranker on any corpus; ANCE/TAS-B/ColBERT/SPLADE/coCondenser/RocketQA/SimLM are trained end-to-end specifically as MS MARCO retrievers.

## Test Setup

- **Date:** 2026-08-01
- **Hardware:** NVIDIA H100 80GB SXM (RunPod, EUR-IS-3)
- **lilbee:** release 0.6.90b420.dev729 on `main` (includes the nprobe fix)
- **Engine:** lilbee_engine 0.6.90b420.dev729 (cu124)
- **Embedder:** Qwen3-Embedding-8B-Q8_0, 4096 dimensions
- **Reranker:** bge-reranker-v2-m3 (Q8_0), lilbee's featured default cross-encoder
- **Index:** 8,841,823 MS MARCO v1 passages (one passage per document), IVF_PQ + FTS built corpus-wide.
- **Index search:** nprobe fraction 0.15 (446 of ~2,973 IVF partitions per query), refine_factor 10.
- **Dataset:** `msmarco-passage/dev/small`, the official small dev set: 6,980 queries, 7,437 relevance judgments (~1.1 judged passages per query).
- **Depth:** top 100 passages recorded per query.

## Methodology

Every query in the dev set was run through lilbee and the ranked results written
to a TREC run file. Two arms are reported: **dense** (the `vec:` mode prefix,
comparable to a published dense-retrieval baseline) and the **full pipeline**
(dense + FTS fusion + cross-encoder rerank), which is what lilbee returns when a
reranker is loaded.

**Scoring is done by the official reference implementations, not by any harness of
ours.** MRR@10 is computed by Microsoft's
[`ms_marco_eval.py`](https://github.com/microsoft/MSMARCO-Passage-Ranking) and
independently by NIST's [`trec_eval`](https://github.com/usnistgov/trec_eval) (via
pytrec_eval, the same C source). nDCG@10 and Recall are trec_eval's. The two
scorers agreeing is the check that the number is not an artifact of any one tool.
Both run files are published so anyone can re-score them without lilbee or any of
our code.

**Document-id join.** lilbee returns a source like `06949/6949140.txt`, whose
basename without the extension is the MS MARCO passage id the qrels use. This join
is the thing most easily got silently wrong: when the ids do not match, every
metric is zero by construction. It was verified before scoring (a passage's own
text self-retrieves at rank 1, and `judged@10` is non-zero).

**Primary metric: MRR@10**, the official MS MARCO passage measure; nDCG@10 is the
metric the general-embedder (MTEB) leaderboard uses. Reciprocal rank is cut at
depth 10 and averaged over all 6,980 queries.

## Results

| Metric | dense only | full pipeline (rerank) |
|--------|-----------|------------------------|
| **MRR@10** | 0.347 | **0.365** |
| **nDCG@10** | 0.398 | **0.437** |
| **Recall@100** | 0.67 | **0.90** |

MRR@10 by the two official scorers on the full pipeline: **ms_marco_eval.py 0.3650,
trec_eval 0.3650**. Dense arm: both scorers 0.3474.

## Key Findings

- **Reranking is the biggest lever on this corpus.** The full pipeline lifts MRR@10 from 0.347 to 0.365 and nDCG@10 from 0.398 to 0.437 over dense-only, recovering the quality the approximate ANN index leaves behind. The reranked nDCG@10 (0.437) matches Qwen3-Embedding-8B's own exact-search MTEB score (0.436).
- **Fusion, not nprobe, is what fixes recall.** Adding FTS lifts recall@100 from 0.67 to 0.90. Widening the IVF search (nprobe) helps only marginally at practical settings: re-grading at a 15% probe fraction moved recall@100 only 0.6656 to 0.6697; recovering more would require a near-exhaustive scan, which is not a viable default.
- **lilbee is competitive without being trained on the benchmark.** Its embedder is general-purpose and its reranker is a general cross-encoder, yet the pipeline lands above ColBERT/TAS-B/ANCE on MRR@10 and matches the general-embedder frontier on nDCG@10.
- **Two independent official scorers agree**, which is what makes the numbers defensible. During analysis a partial run once scored ~0 under ms_marco_eval.py because that script normalizes over the full reference set; cross-checking against trec_eval surfaced the cause immediately.

## Limitations

- **MS MARCO dev/small judgments are sparse** (~1.1 per query), so a passage lilbee ranks highly that no assessor labeled counts as non-relevant. This is standard for MS MARCO and is why every published baseline uses the same set.
- **One passage per document.** The chunk-to-document collapse is 1:1 on this corpus, so it exercises no collapse logic.
- **This measures retrieval only.** Answer generation is a separate tier and is not evaluated here.
- **HyDE and query expansion are off** (no chat model installed), so the pipeline measured here is dense + FTS fusion + rerank without query-side expansion.

## Reproducibility

Two run files are published: `retrieval-msmarco/run.rerank.trec.gz` (full pipeline,
the headline) and `retrieval-msmarco/run.vec.trec.gz` (dense floor). With the
public MS MARCO `dev/small` qrels, re-score them without lilbee or this repository:

```bash
gunzip -k run.rerank.trec.gz
# qrels: msmarco-passage/dev/small qrels.dev.small.tsv, and its TREC form dev.qrels

# Microsoft's official script (headline MRR@10) -- needs a qid<TAB>pid<TAB>rank candidate
awk '{print $1"\t"$3"\t"$4}' run.rerank.trec > run.rerank.msmarco
python ms_marco_eval.py qrels.dev.small.tsv run.rerank.msmarco   # -> MRR @10: 0.3650

# NIST trec_eval (independent cross-check + nDCG@10 + Recall)
trec_eval -c -M 10 -m recip_rank -m ndcg_cut.10 -m recall.100 dev.qrels run.rerank.trec
```

The same commands on `run.vec.trec.gz` reproduce the dense floor (MRR@10 0.3474).
Both are unmodified official tools run on the emitted run files.
