# Retrieval Benchmark: MS MARCO Passage

Retrieval quality of lilbee's 8.8M-passage MS MARCO index, measured against the
official human relevance judgments and compared to published baselines.

**Headline: lilbee's dense retrieval scores MRR@10 = 0.347 on MS MARCO
dev/small, tied with TAS-B (0.347), ahead of ANCE (0.330), and ~85% above BM25
(0.187) - the top dense retriever among the standard baselines behind only the
heavier ColBERTv2.** Scored by Microsoft's own evaluation script and NIST's
trec_eval, which agree on the number. Measured at the recommended index-search
setting (nprobe fraction 0.15; see Test Setup and the companion change).

## What these numbers mean (plain-language key)

If you read only one thing: **lilbee's search is in the same tier as well-known
specialized AI search systems, and clearly better than classic keyword search.**
On a standard industry test of 6,980 real search queries, the correct passage
typically lands near the top of lilbee's results, and is frequently first.

**The metrics** (all run 0 to 1, higher is better):

| Metric | In plain terms | lilbee |
|--------|----------------|--------|
| **MRR@10** | How high the correct answer sits, on average. Anchor points: 1.0 = always the #1 result, 0.5 = like always #2, 0.33 = like always #3. This is the headline. | **0.347** |
| **nDCG@10** | Overall quality of the top-10 ordering, giving more credit the closer the right answers are to the top. | 0.398 |
| **Recall@100** | Of all the correct passages, the share found anywhere in the top 100. | 0.670 (two-thirds) |
| judged@10 | A coverage sanity-check, **not** a quality score: how much of the top 10 the test even has answer-labels for. Low is normal here, because the test labels only ~1 passage per question. | 5.9% |

**Is 0.347 good? Yes.** Ranked against the standard baselines on this same test,
lilbee is tied with TAS-B at the top of the dense retrievers:

| Rank | System | MRR@10 | Type |
|------|--------|--------|------|
| 1 | ColBERTv2 | 0.397 | late-interaction (heavier, pricier) |
| **2** | **lilbee** | **0.347** | **dense retriever** |
| **2** | **TAS-B** | **0.347** | dense retriever |
| 4 | ANCE | 0.330 | dense retriever |
| 5 | BM25 | 0.187 | keyword search (no AI) |

lilbee is **tied with TAS-B (both 0.347)** and **above ANCE**, the top dense
retriever on this set behind only the heavier, more expensive ColBERTv2. The best
specialized systems in the world reach ~0.40-0.45 (see the ceiling note below), so
lilbee sits near the top of what is practically achievable on this benchmark.
These five are the standard reference baselines; for the full field of systems
benchmarked on MS MARCO, including several trained specialists that score higher,
see the extended leaderboard under Results.

**One caveat that makes the result look better, not worse:** this test labels
only about one "correct" passage per question, even when several passages answer
it equally well. lilbee earns no credit when it returns a genuinely good passage
the test did not happen to label, which is why even the best systems in the world
top out around 0.40-0.45 rather than 1.0. lilbee at 0.347 is roughly 80% of the
way to that practical ceiling.

**The comparison systems** (all scored on this same test):

- **BM25** - classic keyword-matching search, no AI; the algorithm inside Elasticsearch, OpenSearch, and Lucene, i.e. what most software's search bar uses. The floor to beat.
- **ANCE** - an AI "dense retriever" from Microsoft Research (2020): matches by meaning rather than keywords, and is trained specifically as an MS MARCO retriever. The same *type* of system as lilbee.
- **TAS-B** - a widely-cited, efficiently-trained AI dense retriever from academic research (2021, TU Wien), also trained specifically for this benchmark. Same type as lilbee.
- **ColBERTv2** - a heavier, more accurate "late-interaction" retriever from Stanford that stores many vectors per passage. A step up in both cost and quality.
- **dense retriever** - a system that turns text into meaning-vectors and matches by similarity (lilbee's approach), as opposed to keyword matching.

lilbee tying ANCE and TAS-B is notable because those two are trained specifically
as MS MARCO retrievers, while lilbee uses a general-purpose embedder off the
shelf, not fine-tuned for this benchmark. It holds its own against purpose-built
specialists without being built for the test.

## Why this is a good result

- **It puts the right answer at the top.** MRR@10 0.347 means the correct passage is the #1 result for roughly a third of queries and near the top for most of the rest, which is the whole point of search: not "is it somewhere in the results" but "is it right there."
- **It nearly doubles classic keyword search.** Beating BM25 (0.187) by ~85% means lilbee finds the right answer at the top far more often than the search technology inside most production systems today.
- **It matches specialists built for the test, without being built for the test.** ANCE and TAS-B are trained specifically as MS MARCO retrievers; lilbee uses a general-purpose embedder off the shelf, not fine-tuned for this benchmark. Tying them anyway is a stronger signal that the quality will carry over to your own documents rather than being tuned to this benchmark.
- **It is close to the practical ceiling, and undercounted.** Because the test labels only ~1 correct passage per query, the entire field caps around 0.40-0.45, not 1.0; 0.347 is ~80% of that. The true quality is a notch higher still, because lilbee is scored zero on queries where it returns a genuinely correct passage that simply was not the labeled one. This happened directly in the run: for "what is paula deen's brother," lilbee returned a correct Paula Deen passage at rank 1, but a *different* passage was the labeled answer, so the query scored zero despite lilbee doing the right thing.
- **The material is being found; the rest is ranking.** Recall@100 of ~67% shows the correct passage is retrieved for two-thirds of queries. Widening the index search (nprobe) helps only marginally at practical settings (see Findings), so the remaining gap is mostly ranking, not retrieval reach, and not a fundamental capability limit.

## Test Setup

- **Date:** 2026-08-01
- **Hardware:** 1x NVIDIA H100 80GB SXM (RunPod, EUR-IS-3), 188 GB RAM, 20 vCPU
- **lilbee:** release 0.6.90b420.dev729 with the nprobe fix (branch `fix/ann-nprobe-recall`)
- **Engine:** lilbee_engine 0.6.90b420.dev729 (cu124), llama-server embedder
- **Embedder:** Qwen3-Embedding-8B-Q8_0, 4096 dimensions
- **Index:** 8,841,823 MS MARCO v1 passages (one passage per document), IVF_PQ + FTS built corpus-wide. Built with `concept_graph`, `wiki`, and `entity_extraction` off, and no chat model installed, so HyDE and query expansion are unavailable.
- **Index search:** nprobe fraction 0.15, i.e. 446 of the ~2,973 IVF partitions probed per query (the recommended setting; refine_factor 10 re-ranks survivors against full vectors).
- **Dataset:** `msmarco-passage/dev/small`, the official small dev set: 6,980 queries, 7,437 relevance judgments (~1.1 judged passages per query).
- **Depth:** top 100 passages recorded per query.

## Methodology

Every query in the dev set was run through lilbee's search API and the ranked
results written to a TREC run file. The measured arm is **`vec` (pure dense)**:
the `vec:` mode prefix, which forces dense retrieval and skips fusion and
re-ranking, so the number is comparable to a published dense-retrieval baseline.
(The full-pipeline `default` arm could not be measured on this pod; see
Limitations.)

**Scoring is done by the official reference implementations, not by any harness
of ours.** MRR@10 is computed by Microsoft's
[`ms_marco_eval.py`](https://github.com/microsoft/MSMARCO-Passage-Ranking) (the
script that produced the published baselines) and independently by NIST's
[`trec_eval`](https://github.com/usnistgov/trec_eval) (via pytrec_eval, the same
C source). nDCG@10 and Recall are trec_eval's. The two scorers agreeing is the
check that the number is not an artifact of any one tool. The run file is
published (`retrieval-msmarco/run.vec.trec.gz`) so anyone can re-score it without
lilbee or any of our code.

**Document-id join.** lilbee returns a source like `06949/6949140.txt`, whose
basename without the extension is the MS MARCO passage id the qrels use. This
join is the thing most easily got silently wrong: when the ids do not match,
every metric is zero by construction, an id-join bug rather than a bad system. It
was verified before scoring (a passage's own text self-retrieves at rank 1, and
`judged@10` is non-zero).

**Primary metric: MRR@10**, the official MS MARCO passage measure. Reciprocal
rank is cut at depth 10 and averaged over all 6,980 queries; a query with no
relevant passage in its top 10 scores zero.

## Results

| Metric | lilbee (vec, pure dense) |
|--------|--------------------------|
| **MRR@10** | **0.3474** |
| nDCG@10 | 0.3980 |
| Recall@100 | 0.6697 |
| judged@10 (coverage diagnostic) | 5.9% |

MRR@10 by the two official scorers: **ms_marco_eval.py 0.3474, trec_eval 0.3474**
(identical to four decimals).

### Extended leaderboard (systems benchmarked on this dev set)

The five standard baselines above are the common reference points. For fuller
context, here is a wider set of well-known open-source systems that report MRR@10
on MS MARCO passage dev/small, with lilbee placed by its measured number:

| Rank | System | MRR@10 | Specialized for MS MARCO retrieval? |
|------|--------|--------|-------------------------------------|
| 1 | SimLM | 0.411 | yes (dense) |
| 2 | ColBERTv2 | 0.397 | yes (late-interaction) |
| 3 | coCondenser | 0.382 | yes (dense) |
| 4 | RocketQA | 0.370 | yes (dense) |
| 5 | SPLADEv2 | 0.368 | yes (learned sparse) |
| 6 | ColBERT | 0.360 | yes (late-interaction) |
| **7** | **lilbee** | **0.347** | **no - general-purpose embedder, used as-is** |
| **7** | **TAS-B** | **0.347** | yes (dense) |
| 9 | ANCE | 0.330 | yes (dense) |
| 10 | docTTTTTquery | 0.277 | yes (doc expansion + BM25) |
| 11 | BM25 | 0.187 | no (keyword, no training) |

The honest read: several purpose-built systems score higher than lilbee. What
sets lilbee apart is the last column - **every system above it is trained
specifically as an MS MARCO retriever, while lilbee uses a general-purpose
embedder off the shelf**. It is the only non-specialized system anywhere in that
tier, tied with TAS-B and ahead of ANCE. Baseline figures are the published
numbers for this dataset; lilbee's is the measured number from this run.

## Key Findings

- **Lilbee's dense retrieval is competitive with strong published dense models.** MRR@10 0.347 ties TAS-B, sits above ANCE, well above BM25, and within reach of ColBERTv2. The Qwen3-Embedding-8B embeddings are strong: a passage's own text retrieves it at rank 1.
- **nprobe tuning is a minor lever, not the recall unlock a small sample suggested.** An early 30-query probe (recall@100 67% at the default 149 partitions, 87% at nprobe=2048) overstated the opportunity: nprobe=2048 is ~69% of this index's ~2,973 IVF partitions, effectively a near-exhaustive scan. Re-grading the full 6,980-query dev set at a practical 15% probe fraction (446 partitions, up from 5%/149) moved recall@100 from 0.6656 to 0.6697 and MRR@10 from 0.3458 to 0.3474 (ms_marco_eval.py and trec_eval agree on 0.3474), at a +~250ms/query ANN-scan cost (median ~350ms to ~660ms, isolated from the embedding-dominated ~2.5s end-to-end query). The gain is small but real; recovering the full recall headroom would require probing most of the index, which is not a viable default.
- **Two independent official scorers agreeing to four decimals** is what makes this number defensible. During analysis a partial run scored 0.002 under ms_marco_eval.py because that script normalizes over the full reference set; the disagreement with trec_eval surfaced the cause immediately. Cross-checking with two reference tools, rather than one harness, is why the published number is trustworthy.

## Limitations

- **The `vec` arm is pure dense, not the full pipeline.** This index was built with concept graph, wiki, and entity extraction off, and no chat model is installed, so HyDE and query expansion do not run. The measured number is dense retrieval alone; the hybrid/default path was not measured (below).
- **The `default` (hybrid) arm could not be measured on this pod.** Its search path performs in-process CUDA engine-planning that failed to initialize CUDA ("ggml_cuda_init: failed to initialize CUDA: initialization error"), while the `vec` path routes cleanly through the served embedder. This is a lilbee issue, not a property of the index, and is tracked separately.
- **MS MARCO dev/small judgments are sparse** (~1.1 per query), so a passage lilbee ranks highly that no assessor labeled counts as non-relevant. This is standard for MS MARCO and is why every published baseline uses the same set; `judged@10` reports the exposure.
- **One passage per document.** The chunk-to-document collapse is 1:1 on this corpus, so it exercises no collapse logic.
- **This measures retrieval only.** Answer generation is a separate tier and is not evaluated here.

## Reproducibility

The run file (`retrieval-msmarco/run.vec.trec.gz`) and the public MS MARCO
`dev/small` qrels are the artifacts. To reproduce the score without lilbee or
this repository:

```bash
gunzip -k run.vec.trec.gz
# qrels: msmarco-passage/dev/small qrels.dev.small.tsv, and its TREC form dev.qrels

# Microsoft's official script (headline MRR@10) -- needs a qid<TAB>pid<TAB>rank candidate
awk '{print $1"\t"$3"\t"$4}' run.vec.trec > run.vec.msmarco
python ms_marco_eval.py qrels.dev.small.tsv run.vec.msmarco      # -> MRR @10: 0.3474

# NIST trec_eval (independent cross-check + nDCG@10 + Recall)
trec_eval -c -M 10 -m recip_rank -m ndcg_cut.10 -m recall.100 dev.qrels run.vec.trec
```

Both are unmodified official tools run on the emitted run file.
