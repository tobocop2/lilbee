# We audited our own retrieval benchmark and it had been lying to us

We added hybrid retrieval to a local document search tool: BM25 fused with vector
search, weighted by a knob we called `lexical_fusion_weight`. We benchmarked it on
three public BEIR datasets, got a result we liked, and wrote it up.

Then we audited the benchmark itself. Sixty-three problems, twenty of which meant
it was reporting numbers that did not mean what their labels said. Fixing them
changed the headline. The short version is that our "significant win on two of
three corpora" was mostly an artifact of how we were measuring, and the strongest
real effect in the data is a regression we had been underselling.

Here is what broke, what it cost, and what the numbers actually say.

## Five ways an IR benchmark quietly reports the wrong number

**1. Your MRR@10 is probably not cut at 10.** We used `pytrec_eval`'s
`recip_rank`, which is the standard reciprocal rank measure. It is uncut: it
searches the entire run, not the top 10. We collected 20 documents per query, so
a first relevant document at rank 11 through 20 contributed `1/rank` to a number
we published as MRR@10. On our runs that inflated it by up to 0.004, which is the
same order as the deltas we were reporting as findings.

There is a second, nastier layer. Once you truncate, you have to truncate the way
the scorer would. `trec_eval` breaks score ties by document ID *descending*. We
cut using ascending. That sounds pedantic until you look at fused runs, which
produce enormous numbers of exactly-tied scores: 210 of FiQA's 648 queries had a
tie straddling the rank-10 boundary. The two tie rules gave 0.4726 and 0.4729 for
the same run. Whichever you prefer, only one of them is "pytrec_eval's MRR@10",
and if you are going to put that label on it you have to hand the scorer the same
ten documents it would have picked itself.

**2. Queries your system whiffs on vanish from the denominator.** `pytrec_eval`
returns scores only for query IDs present in the run file. If your retriever
returns nothing for a hard query, that query does not score zero. It disappears.
Your mean is now conditioned on your system having produced output, which rewards
failure and makes two systems non-comparable whenever they fail on different
queries. We caught this because NFCorpus qrels have 323 topics and every one of
our run files had 322. We had been publishing "n=322".

The fix is one line and it is the qrels topic set, not the run, that defines your
denominator.

**3. Depth matching is not optional when the systems have different output
units.** Our own search returns results grouped by source document, so asking for
20 gets you 20 documents. A chunk-level system asked for 20 chunks gives you 20
chunks, which collapse to however many parent documents they happen to belong to,
usually far fewer because a relevant document contributes several chunks. If you
score a 20-document list against a 7-document list, the gap between them contains
a pure depth artifact that has nothing to do with ranking quality. Chunk-level
retrieval has to over-fetch until it holds the target number of distinct parents.

**4. If you run 36 tests, you have to say so.** We compared four fusion weights
across three datasets on three metrics. That is 36 paired tests. We then picked
the best weight per dataset and quoted its raw p-value as the verdict. Selecting
the maximum of several correlated arms and testing it on the same data is exactly
the setup multiple-comparison correction exists for. Every number in this writeup
is Benjamini-Hochberg adjusted across all 36.

We also had two contradicting verdicts on every row: `significant` came from
whether the bootstrap CI crossed zero, while the printed p came from a separate
randomization test, and both were driven off the same seeded random stream. Pick
one decision rule, report the interval as an effect size, and give the two
procedures independent seeds if you want them to corroborate each other.

**5. An LLM judge graded at temperature 0 has a noise floor of exactly zero.** We
had a "judge noise floor" measured by grading one arm's answers twice under fresh
opaque IDs and taking the disagreement. Both replicates were built from the same
answer into byte-identical prompts, and both chat backends decode greedily. Same
prompt plus greedy decode equals the same grade, so the measured noise was zero by
construction, and the report then flagged every nonzero difference as outside the
noise. What we had actually measured was decoder determinism.

Worse, the judge defaulted to the same local model that had written the questions
and generated both arms' answers. It was grading its own homework against ground
truth it had paraphrased.

## What the numbers actually say

Three BEIR passage sets, Qwen3-Embedding-0.6B served once so retrieval is the only
variable, no query rewriting on either side. `dense` is vector-only. `w=X` is
hybrid fusion with the BM25 arm at that weight. All p-values adjusted across the
full 36-test family.

Thirteen of the 36 comparisons survive correction. Best nDCG@10 arm per dataset:

| dataset | n | dense | best arm | delta |
|---|---|---|---|---|
| SciFact | 300 | 0.6981 | w=1.0 → 0.7280 | +0.0299 |
| NFCorpus | 323 | 0.3655 | w=0.25 → 0.3755 | +0.0100 |
| FiQA | 648 | 0.4598 | w=0.25 → 0.4363 | **−0.0235** |

The single best-evidenced effect in the study is the FiQA regression. Every
fusion weight loses to pure dense on both nDCG@10 and MRR@10, all nine of those
tests survive correction with adjusted p at or below 0.0016, and the damage grows
monotonically with the weight. At w=1.0 it is −0.0564 nDCG@10 and −0.0677 MRR@10.
Nothing else in the study is anywhere near that well supported.

SciFact goes the other way and likes a strong lexical arm. NFCorpus improves
slightly and only at the lowest weight; its MRR@10 gain at that same weight does
not survive correction.

We also tried scaling the lexical weight per query by how peaked the vector
ranking is. It gives the smallest FiQA regression of anything we measured
(−0.0188) and keeps the SciFact win, but it does not erase the regression: FiQA
still prefers pure dense, significantly. Its NFCorpus win does not survive
correction, so we withdrew that claim.

## The conclusion we kept, and the one we lost

What survived: **no single fixed fusion weight wins everywhere.** SciFact wants a
strong lexical arm, NFCorpus wants a weak one and barely, FiQA wants none at all.
That corpus-dependence is real and it is the reason to gate the weight per query
rather than ship a constant.

What did not survive: "hybrid fusion is a significant win on two of three
corpora." That framing came from picking each dataset's best weight after seeing
the results and quoting its uncorrected p-value. Under correction the honest
statement is that fusion helps on one of three, is marginal on one, and clearly
hurts on one.

Our intuition for why FiQA behaves differently is that it is financial questions
against forum answers, where lexical overlap between a question and a good answer
is weak and BM25 mostly injects noise. SciFact is scientific claims against
abstracts, where exact terminology matters a lot. We have not tested that story,
so treat it as a hypothesis rather than a finding, which is roughly the discipline
this whole exercise was about.

## Things we are still not claiming

The run files predate the depth-matching fix, so `Recall@20` here is really recall
at whatever depth each run reached, and it is comparable within this study and not
to anything else. The cross-system comparison and the answer-quality tier have
never run under the corrected harness at all. And one of our arms had committed
metrics that do not reproduce from its own committed run file, which we found
while verifying something else and which is its own small horror story.

The harness, run files, and qrels are in the repo. Tier 1 needs no GPU: with the
run files and the qrels you can re-score the whole thing yourself and check
whether we got it right this time.
