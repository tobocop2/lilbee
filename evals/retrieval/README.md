# Retrieval eval harness

Blind, noise-calibrated A/B evaluation of lilbee's end-to-end retrieval
quality. Two lilbee servers (arm A and arm B: old vs new version, or two
configs) answer the same question battery over the SAME index; a judge model
grades answers blind; exact-truth questions are checked mechanically against
the store. Lives outside `src/` on purpose: it never ships in the package.

## How it works

1. **questions** generates the battery from an existing indexed library:
   - *topical*: the configured chat model writes one question from each
     sampled stored passage, so the passage that must support the answer is
     known ground truth. Passages are reservoir-sampled from a streaming scan
     (one per source); nothing materializes the index, so it scales to very
     large libraries.
   - *known_item*: asks what a sampled document is about; ground truth is the
     document's head chunks, captured in the same scan.
   - *count*: asks how many documents mention a mid-frequency term, and the
     check verifies exactly that one number. Ground truth is an exact streaming
     scan of the LanceDB store, counting word-level mentions rather than substrings so the oracle answers the question that was asked; no judge involved. The scan also records the
     chunk total as provenance, but the answer is not required to volunteer a
     figure it was never asked for.
2. **answer** runs the battery against one server (`/api/ask`), one run per
   arm. It waits for the server's health route, retries each question three
   times, and checkpoints every row to JSONL, so a killed pod run resumes
   without redoing completed questions.
3. **judge** shuffles every gradable answer under an opaque id and grades them
   one at a time through RAGAS' rubric metric, which owns the prompt, the
   structured output, and the retry when a response does not validate. The judge
   sees only question + ground truth + one answer, never arm labels, and never
   knows a comparison is happening. Arm B's answers are judged twice under
   different ids and under two equivalent presentations of the rubric; the
   disagreement between those two passes is the judge's noise floor. The two
   presentations describe the same five levels and differ only in wording and
   layout, which is what makes the second pass a measurement: both backends
   decode greedily at temperature 0, so an identical rubric would produce an
   identical prompt, return an identical grade, and report a noise floor of
   exactly zero. Grades are checkpointed too.
4. **score** unblinds mechanically: per-dimension means (faithfulness,
   relevance, citation, each on RAGAS' 1-5 rubric scale), a paired test of each
   dimension across the two arms, and exact pass/fail for count and known-item
   questions. Writes machine-readable `results.jsonl`.
5. **report** renders `results.jsonl` as markdown. Significance comes from the
   paired per-question test, Benjamini-Hochberg adjusted across the dimensions
   tested. The noise floor is reported as what it is, a per-question statement
   about how steady the judge is on one answer; it is not a threshold for a
   difference of means over many questions, which is a different scale.

## Running on a pod

Typical detached run against a large private corpus: one shared index, two
installed lilbee versions serving on different ports.

```bash
# One venv per arm, both pointed at the same data root.
OLD_PORT=8081 NEW_PORT=8082
"$OLD_VENV/bin/lilbee" --data-dir "$DATA_ROOT" serve --port "$OLD_PORT" &
"$NEW_VENV/bin/lilbee" --data-dir "$DATA_ROOT" serve --port "$NEW_PORT" &

cd ~/lilbee  # the repo checkout; run everything from the repo root

# 1. Questions (uses the configured chat model; scans the index directly).
uv run python -m evals.retrieval questions \
  --data-root "$DATA_ROOT" --out /tmp/eval/questions.jsonl

# 2. Answers, one run per arm. Interrupted runs resume from the checkpoint.
uv run python -m evals.retrieval answer \
  --questions /tmp/eval/questions.jsonl --base-url "http://127.0.0.1:$OLD_PORT" \
  --arm old --out /tmp/eval/answers-old.jsonl
uv run python -m evals.retrieval answer \
  --questions /tmp/eval/questions.jsonl --base-url "http://127.0.0.1:$NEW_PORT" \
  --arm new --out /tmp/eval/answers-new.jsonl

# 3. Blind judging. --answers-b is the arm judged twice for the noise floor.
#    Grades are checkpointed; re-running the same command resumes.
uv run python -m evals.retrieval judge \
  --questions /tmp/eval/questions.jsonl \
  --answers-a /tmp/eval/answers-old.jsonl --answers-b /tmp/eval/answers-new.jsonl \
  --work-dir /tmp/eval/judge

# 4 + 5. Score and render.
uv run python -m evals.retrieval score \
  --questions /tmp/eval/questions.jsonl \
  --answers-a /tmp/eval/answers-old.jsonl --answers-b /tmp/eval/answers-new.jsonl \
  --work-dir /tmp/eval/judge --out /tmp/eval/results.jsonl
uv run python -m evals.retrieval report \
  --results /tmp/eval/results.jsonl --out /tmp/eval/report.md
```

Question generation talks to the configured lilbee chat model (set
`LILBEE_CHAT_MODEL` before step 1). **Judging requires a separate endpoint and
model, and there is no fallback.** That model wrote the questions and generated
both arms' answers, so letting it judge means one model grades its own output
against ground truth it paraphrased. `judge` refuses to run until both variables
are set, and records which model produced the grades in `results.jsonl` and the
report:

```bash
export LILBEE_EVAL_JUDGE_BASE_URL="https://api.example.com/v1"
export LILBEE_EVAL_JUDGE_MODEL="judge-model-name"   # required, not optional
export LILBEE_EVAL_JUDGE_API_KEY="..."              # optional
```

## Determinism and resume

- `--seed` (questions, judge) makes sampling and shuffles reproducible. Blind
  ids are not seeded: each is a hash of its (question, arm, replicate, answer),
  so re-running `judge` after regenerating questions or re-running an answer
  misses the checkpoint for exactly the rows that changed instead of inheriting
  a grade that belongs to a different answer.
- `answer` and `judge` checkpoint per row, so killing and re-running repeats
  only unfinished work. A checkpoint is bound to the run that created it: the
  answers file records its arm, endpoint, depth and a digest of the question
  set, and resuming it under a different one is refused rather than silently
  producing a file that mixes two configurations.
- Answers that hard-fail after all retries are recorded as failures and score at
  the rubric's bottom level everywhere (prefailed: they never reach the judge).
  The bottom level already describes a missing answer, so they stay on the same
  scale as every other number in the report rather than sitting below it.

## Files in a judge work dir

| file | contents |
| --- | --- |
| `blind_rows.jsonl` | what judges see: gid, question, ground truth, answer |
| `gid_map.json` | secret gid to (qid, arm, replicate) mapping |
| `prefailed.json` | gids scored zero without judging |
| `grades.jsonl` | checkpointed judge output, one row per gid |
| `judge_meta.json` | which arm was graded twice, and the judge model that graded |
