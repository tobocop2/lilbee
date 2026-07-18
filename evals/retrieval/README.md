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
   - *count*: asks how many chunks and documents mention a mid-frequency
     term. Ground truth is an exact streaming scan of the LanceDB store; no
     judge involved.
2. **answer** runs the battery against one server (`/api/ask`), one run per
   arm. It waits for the server's health route, retries each question three
   times, and checkpoints every row to JSONL, so a killed pod run resumes
   without redoing completed questions.
3. **judge** shuffles every gradable answer under an opaque id and grades
   them one at a time: the judge sees only question + ground truth + one
   answer, never arm labels, and never knows a comparison is happening. Arm
   B's answers are judged twice under different ids; the disagreement between
   those two passes is the judge's noise floor. Grades are checkpointed too.
4. **score** unblinds mechanically: per-dimension means (faithfulness,
   relevance, citation, each 0-2) with the noise floor as the error bar, plus
   exact pass/fail for count and known-item questions. Writes machine-readable
   `results.jsonl`.
5. **report** renders `results.jsonl` as markdown; any cross-arm delta at or
   below the noise floor is labeled within noise.

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

Question generation and judging talk to the configured lilbee chat model by
default (set `LILBEE_CHAT_MODEL` before step 1, and switch it to a stronger
judge model before step 3 if both run locally). The judge can instead be any
OpenAI-compatible endpoint:

```bash
export LILBEE_EVAL_JUDGE_BASE_URL="https://api.example.com/v1"
export LILBEE_EVAL_JUDGE_MODEL="judge-model-name"
export LILBEE_EVAL_JUDGE_API_KEY="..."   # optional
```

## Determinism and resume

- `--seed` (questions, judge) makes sampling, blinding ids, and shuffles
  reproducible. Re-running `judge` with the same seed and inputs regenerates
  the identical blind set, so the grades checkpoint stays valid.
- `answer` and `judge` both checkpoint per row. Kill and re-run freely; only
  unfinished work repeats.
- Answers that hard-fail after all retries are recorded as failures and score
  zero everywhere (prefailed: they never reach the judge).

## Files in a judge work dir

| file | contents |
| --- | --- |
| `blind_rows.jsonl` | what judges see: gid, question, ground truth, answer |
| `gid_map.json` | secret gid to (qid, arm, replicate) mapping |
| `prefailed.json` | gids scored zero without judging |
| `grades.jsonl` | checkpointed judge output, one row per gid |
