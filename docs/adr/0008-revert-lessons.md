# 8. Lessons from reverts

## Status
Accepted

## Context
Several features were reverted during development. Each revert taught something about assumptions that seemed reasonable at the time.

## Findings

**Speculative xdist fixes (reverted)**: Multiple guesses about the xdist hang (serial tests, reduced parallelism, asyncio executor shutdown) were reverted because they correlated with timing changes but didn't address root causes. Lesson: correlation-based fixes are cargo cult. Root cause analysis (thread state inspection) beats trial-and-error.

**Frontmatter stripping (reverted)**: Frontmatter was stripped before chunking because it seemed "noisy." But frontmatter contains searchable metadata (author, date, tags). The real issue was a test expecting empty chunks for frontmatter-only files. Lesson: don't strip data you might search later. Fix the test expectation instead.

**Duplicate citation blocks (reverted)**: The LLM was generating its own "Key sources:" block, and code was appending a second "Sources:" block. The fix wasn't to strip LLM citations via post-processing but to change the system prompt to use inline [N] references. Lesson: instrumentation problems (two citation blocks) often point to prompt design, not code fixes.

## Consequences
- Prefer root cause analysis over correlation-based fixes
- Preserve data at ingestion time, filter at query time
- When output format is wrong, check the prompt before adding post-processing
